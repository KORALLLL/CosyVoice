"""CUDA-primary CosyVoice3 speech tokens, listening pilots, and cache-worker gate."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import tarfile
import tempfile
from typing import Any, Callable, Mapping, Sequence
import wave

import numpy as np

from .artifacts import StageRequirementError, StageStore, atomic_write_json, sha256_file
from .cache import (
    TOKEN_MAX,
    TOKEN_MIN,
    AudioInput,
    CacheIntegrityError,
    CacheManifest,
    CacheShardRequest,
    CacheShardResult,
    TarAudioSample,
    _load_mapping,
    _verify_phase_metadata,
    build_cache_shard,
    verify_cache,
)
from .config import RunPaths
from .sources import SourceIntegrityError, inventory_sources, iter_split_rows


class TokenizerError(RuntimeError):
    """Raised for any unsafe or invalid CUDA speech-token extraction."""


class PilotApprovalError(RuntimeError):
    """Raised when a cache run lacks approval for the current pilot manifest."""


class PilotReviewRequired(RuntimeError):
    """Intentional stop after publishing a listening bundle for a human review."""

    def __init__(self, manifest: "PilotManifest") -> None:
        self.manifest = manifest
        super().__init__(f"pilot review required; approve manifest sha256 {manifest.manifest_sha256}")


@dataclass(frozen=True)
class PilotManifest:
    """Published listening bundle identity, including its StageStore checksum."""

    root: Path
    index_path: Path
    stage_path: Path
    manifest_sha256: str
    clips: tuple[Mapping[str, object], ...]


FeatureBuilder = Callable[[AudioInput], np.ndarray]


class OnnxSpeechTokenizer:
    """A persistent, single-CUDA-device ONNX Runtime speech-token session."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        session: Any | None = None,
        max_batch_size: int = 8,
        local_rank: int | None = None,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")
        self.max_batch_size = max_batch_size
        self._feature_builder = feature_builder or _whisper_features
        if session is None:
            if model_path is None or not model_path.is_file():
                raise TokenizerError(f"missing batch speech-tokenizer model: {model_path}")
            session = _create_cuda_session(model_path, self.local_rank)
        self.session = session
        _require_cuda_provider(session)

    @classmethod
    def for_paths(cls, paths: RunPaths, *, local_rank: int | None = None, max_batch_size: int = 8) -> "OnnxSpeechTokenizer":
        return cls(
            paths.base_model_dir / "speech_tokenizer_v3.batch.onnx",
            local_rank=local_rank,
            max_batch_size=max_batch_size,
        )

    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        """Extract validated tokens, bisecting only the exact CUDA-OOM batch."""

        if not audio:
            return []
        for item in audio:
            _validate_audio(item)
        extracted: list[list[int]] = []
        for offset in range(0, len(audio), self.max_batch_size):
            extracted.extend(self._extract_with_oom_bisection(audio[offset : offset + self.max_batch_size]))
        return extracted

    def feature_lengths(self, audio: Sequence[AudioInput]) -> list[int]:
        """Expose CPU-side feature lengths for qualification records."""

        return [int(_feature_array(self._feature_builder(item)).shape[1]) for item in audio]

    def _extract_with_oom_bisection(self, audio: list[AudioInput]) -> list[list[int]]:
        try:
            return self._run_batch(audio)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                if isinstance(exc, TokenizerError):
                    raise
                raise TokenizerError("CUDA ONNX speech-token inference failed") from exc
            if len(audio) == 1:
                raise TokenizerError(f"CUDA OOM for one audio item: {audio[0].source_relative_path}") from exc
            middle = len(audio) // 2
            return self._extract_with_oom_bisection(audio[:middle]) + self._extract_with_oom_bisection(audio[middle:])

    def _run_batch(self, audio: list[AudioInput]) -> list[list[int]]:
        features = [_feature_array(self._feature_builder(item)) for item in audio]
        lengths = np.asarray([feature.shape[1] for feature in features], dtype=np.int32)
        max_frames = int(lengths.max())
        batched = np.zeros((len(features), 128, max_frames), dtype=np.float32)
        for index, feature in enumerate(features):
            batched[index, :, : feature.shape[1]] = feature
        inputs = self.session.get_inputs()
        if len(inputs) < 2:
            raise TokenizerError("speech-tokenizer ONNX model must expose features and lengths inputs")
        outputs = self.session.run(None, {inputs[0].name: batched, inputs[1].name: lengths})
        return _decode_onnx_outputs(outputs, len(audio), lengths)


def run_tokenizer_qualification(paths: RunPaths) -> dict[str, object]:
    """Qualify deterministic CUDA token extraction independently on all eight GPUs."""

    _require_eight_devices(paths)
    return publish_tokenizer_qualification(
        paths,
        [qualify_tokenizer_device(paths, device) for device in paths.visible_devices],
    )


def qualify_tokenizer_device(paths: RunPaths, device: int) -> dict[str, object]:
    """Run the real deterministic speech-token checks on exactly one local GPU."""

    if device not in paths.visible_devices:
        raise TokenizerError(f"tokenizer qualification device is not visible: {device}")
    fixed = AudioInput("qualification/silence.wav", np.zeros(24_000, dtype=np.float32), 24_000, 24_000)
    torch = import_module("torch")
    original_device = int(torch.cuda.current_device())
    try:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = OnnxSpeechTokenizer.for_paths(paths, local_rank=device)
        first = tokenizer.extract([fixed])[0]
        second = tokenizer.extract([fixed])[0]
        if first != second:
            raise TokenizerError(f"non-deterministic speech tokens on CUDA device {device}")
        _validate_tokens(first, fixed.source_relative_path)
        rate = len(first) / (fixed.frames / fixed.sample_rate)
        if not 20.0 <= rate <= 30.0:
            raise TokenizerError(f"speech token rate is not approximately 25 Hz on CUDA device {device}: {rate:.3f}")
        feature_length = tokenizer.feature_lengths([fixed])[0]
        expected_tokens = max(1, (feature_length + 3) // 4)
        if len(first) != expected_tokens:
            raise TokenizerError(
                f"speech-token length does not match CPU feature length on CUDA device {device}: "
                f"{len(first)} != {expected_tokens}"
            )
        return {
            "device": device,
            "providers": _session_providers(tokenizer.session),
            "feature_frames": feature_length,
            "token_count": len(first),
            "token_rate_hz": rate,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    finally:
        torch.cuda.set_device(original_device)


def publish_tokenizer_qualification(paths: RunPaths, devices: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate and publish one gathered qualification record per visible GPU."""

    _require_eight_devices(paths)
    records = [dict(record) for record in devices]
    observed = [record.get("device") for record in records]
    if (
        len(records) != len(paths.visible_devices)
        or any(type(device) is not int for device in observed)
        or sorted(observed) != sorted(paths.visible_devices)
    ):
        raise TokenizerError(f"tokenizer qualification did not cover each visible device exactly once: {observed}")
    records.sort(key=lambda record: int(record["device"]))
    payload: dict[str, object] = {
        "model": str(paths.base_model_dir / "speech_tokenizer_v3.batch.onnx"),
        "model_sha256": sha256_file(paths.base_model_dir / "speech_tokenizer_v3.batch.onnx"),
        "devices": records,
    }
    StageStore(paths.stages_dir).publish("tokenizer_qualification", payload)
    return payload


def build_pilot(paths: RunPaths) -> PilotManifest:
    """Build a three-clip, checksum-audited A/B reconstruction listening bundle."""

    qualification = StageStore(paths.stages_dir).require("tokenizer_qualification")
    clips = _select_duration_stratified_clips(paths)
    pilot_root = paths.run_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    tokenizer = OnnxSpeechTokenizer.for_paths(paths, local_rank=0)
    reconstructor = _load_frozen_reconstructor(paths)
    records: list[dict[str, object]] = []
    labels = ("short", "median", "long")
    for label, audio in zip(labels, clips, strict=True):
        original = pilot_root / f"{label}.original.wav"
        tokens_path = pilot_root / f"{label}.tokens.npy"
        reconstructed = pilot_root / f"{label}.reconstructed.wav"
        _write_wav(audio, original)
        tokens = tokenizer.extract([audio])[0]
        np.save(tokens_path, np.asarray(tokens, dtype=np.int32), allow_pickle=False)
        _reconstruct_to_wav(reconstructor, original, reconstructed)
        duration = audio.frames / audio.sample_rate
        records.append(
            {
                "label": label,
                "source_relative_path": audio.source_relative_path,
                "duration_seconds": duration,
                "token_count": len(tokens),
                "token_rate_hz": len(tokens) / duration,
                "token_min": min(tokens),
                "token_max": max(tokens),
                "original": original.name,
                "original_sha256": sha256_file(original),
                "tokens": tokens_path.name,
                "tokens_sha256": sha256_file(tokens_path),
                "reconstructed": reconstructed.name,
                "reconstructed_sha256": sha256_file(reconstructed),
            }
        )
    index = pilot_root / "index.md"
    _atomic_write_text(index, _pilot_index(records))
    payload = {
        "qualification_manifest_sha256": qualification.manifest_sha256,
        "base_model": str(paths.base_model_dir),
        "base_model_tokenizer_sha256": sha256_file(paths.base_model_dir / "speech_tokenizer_v3.batch.onnx"),
        "index": index.name,
        "index_sha256": sha256_file(index),
        "clips": records,
        "artifacts": _pilot_artifacts(index, records),
    }
    record = StageStore(paths.stages_dir).publish("pilot", payload)
    manifest = PilotManifest(pilot_root, index, record.path, record.manifest_sha256, tuple(records))
    raise PilotReviewRequired(manifest)


def approve_pilot(paths: RunPaths, checksum: str) -> Path:
    """Atomically record a human approval only for the current pilot checksum."""

    if not isinstance(checksum, str) or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise PilotApprovalError("pilot approval checksum must be a lowercase 64-hex SHA-256")
    try:
        pilot = StageStore(paths.stages_dir).require("pilot")
    except StageRequirementError as exc:
        raise PilotApprovalError("current pilot manifest is unavailable") from exc
    if checksum != pilot.manifest_sha256:
        raise PilotApprovalError("provided approval checksum does not match the current pilot manifest")
    _verify_pilot_artifacts(paths, pilot.payload)
    approval = StageStore(paths.stages_dir).publish(
        "pilot_approval", {"pilot_manifest_sha256": pilot.manifest_sha256}
    )
    return approval.path


def run_cache_workers(paths: RunPaths) -> CacheManifest:
    """Run approved cache work through exactly eight persistent spawned CUDA workers."""

    require_pilot_approval(paths)
    _require_eight_devices(paths)
    inventory = inventory_sources(paths)
    cache_root = paths.run_root / "cache"
    plan_dir = paths.run_root / "split_plan"
    expected_shards = {int(archive.stem.split("_")[1]) for archive in inventory.source_archives}
    existing: set[int] = set()
    manifests = list((cache_root / "shard_manifests").glob("shard_*.json"))
    if manifests:
        existing = set(verify_cache(cache_root).shards)
        unexpected = existing - expected_shards
        if unexpected:
            raise TokenizerError(f"verified cache contains shards outside canonical inventory: {sorted(unexpected)}")
    requests = [
        CacheShardRequest(
            source_tar=archive,
            plan_dir=plan_dir,
            cache_root=cache_root,
            shard=int(archive.stem.split("_")[1]),
            expected_source_sha256=inventory.source_archive_sha256[archive.name],
            expected_plan_sha256=sha256_file(plan_dir / f"{archive.stem}.jsonl"),
        )
        for archive in inventory.source_archives
        if int(archive.stem.split("_")[1]) in expected_shards - existing
    ]
    if not requests:
        return _verify_requested_cache_set(cache_root, expected_shards)
    context = mp.get_context("spawn")
    work: mp.Queue[Mapping[str, object] | None] = context.Queue()
    results: mp.Queue[Mapping[str, object]] = context.Queue()
    for request in requests:
        work.put(request.to_dict())
    for _ in paths.visible_devices:
        work.put(None)
    workers = [
        context.Process(
            target=_cache_worker,
            args=(device, str(paths.base_model_dir / "speech_tokenizer_v3.batch.onnx"), work, results),
            daemon=False,
        )
        for device in paths.visible_devices
    ]
    for worker in workers:
        worker.start()
    remaining_shards = {request.shard for request in requests}
    try:
        while remaining_shards:
            try:
                message = results.get(timeout=1)
            except queue.Empty:
                failed = next((worker for worker in workers if worker.exitcode not in (None, 0)), None)
                if failed is not None:
                    raise TokenizerError(f"cache worker exited unsuccessfully: {failed.pid} ({failed.exitcode})")
                continue
            if message.get("error") is not None:
                raise TokenizerError(f"cache worker failed: {message['error']}")
            result = CacheShardResult.from_dict(_mapping(message.get("result"), "cache worker result"))
            if result.shard not in remaining_shards:
                raise TokenizerError(f"cache worker returned a duplicate or unleased shard: {result.shard}")
            request = next(request for request in requests if request.shard == result.shard)
            _validate_worker_result(request, result)
            remaining_shards.remove(result.shard)
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise TokenizerError(f"cache worker exited unsuccessfully: {worker.pid} ({worker.exitcode})")
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join()
        raise
    return _verify_requested_cache_set(cache_root, expected_shards)


def _create_cuda_session(model_path: Path, local_rank: int) -> Any:
    try:
        onnxruntime = import_module("onnxruntime")
        available = list(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise TokenizerError("ONNX Runtime CUDAExecutionProvider is unavailable")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        return onnxruntime.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": local_rank})],
        )
    except TokenizerError:
        raise
    except Exception as exc:
        raise TokenizerError(f"failed to create CUDA ONNX speech-tokenizer session on device {local_rank}") from exc


def _require_cuda_provider(session: Any) -> None:
    providers = _session_providers(session)
    if providers[0] != "CUDAExecutionProvider":
        raise TokenizerError(f"speech-tokenizer session did not bind CUDAExecutionProvider as its primary provider: {providers}")


def _session_providers(session: Any) -> list[str]:
    getter = getattr(session, "get_providers", None)
    if not callable(getter):
        raise TokenizerError("cannot verify CUDA ONNX session provider identity")
    try:
        providers = [str(provider) for provider in getter()]
    except Exception as exc:
        raise TokenizerError("cannot verify CUDA ONNX session providers") from exc
    if not providers:
        raise TokenizerError("CUDA ONNX session reports no execution providers")
    return providers


def _validate_audio(audio: AudioInput) -> None:
    if audio.sample_rate != 24_000 or audio.frames < 1:
        raise TokenizerError(f"tokenizer requires non-empty 24 kHz audio: {audio.source_relative_path}")


def _whisper_features(audio: AudioInput) -> np.ndarray:
    _validate_audio(audio)
    try:
        torch = import_module("torch")
        torchaudio = import_module("torchaudio")
        whisper = import_module("whisper")
        samples = torch.as_tensor(audio.samples, dtype=torch.float32).squeeze()
        if samples.ndim != 1 or samples.numel() != audio.frames or not bool(torch.isfinite(samples).all()):
            raise TokenizerError(f"invalid mono samples for {audio.source_relative_path}")
        resampled = torchaudio.functional.resample(samples.unsqueeze(0), 24_000, 16_000).squeeze(0)
        return _feature_array(whisper.log_mel_spectrogram(resampled, n_mels=128).detach().cpu().numpy())
    except TokenizerError:
        raise
    except Exception as exc:
        raise TokenizerError(f"cannot build Whisper features for {audio.source_relative_path}") from exc


def _feature_array(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] != 128 or feature.shape[1] < 1 or not np.isfinite(feature).all():
        raise TokenizerError("Whisper feature builder must return finite [128, frames] features")
    return feature


def _decode_onnx_outputs(outputs: Any, batch_size: int, feature_lengths: np.ndarray) -> list[list[int]]:
    if not isinstance(outputs, Sequence) or not outputs:
        raise TokenizerError("speech-tokenizer ONNX session returned no outputs")
    values = np.asarray(outputs[0])
    if values.ndim == 1:
        if batch_size != 1:
            raise TokenizerError("speech-tokenizer ONNX output omitted batch dimension")
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != batch_size:
        raise TokenizerError("speech-tokenizer ONNX output has invalid batch shape")
    lengths = np.full(batch_size, -1, dtype=np.int64)
    if len(outputs) > 1:
        candidate = np.asarray(outputs[1]).reshape(-1)
        if candidate.size == batch_size:
            lengths = candidate.astype(np.int64, copy=False)
    result: list[list[int]] = []
    for index, row in enumerate(values):
        length = int(lengths[index]) if lengths[index] >= 0 else max(1, (int(feature_lengths[index]) + 3) // 4)
        if length < 1 or length > row.size:
            raise TokenizerError("speech-tokenizer ONNX output has invalid token length")
        tokens = [int(token) for token in row[:length]]
        _validate_tokens(tokens, f"batch item {index}")
        result.append(tokens)
    return result


def _validate_tokens(tokens: Sequence[int], identifier: str) -> None:
    if not tokens:
        raise TokenizerError(f"speech-tokenizer returned no tokens for {identifier}")
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)) or token < TOKEN_MIN or token > TOKEN_MAX:
            raise TokenizerError(f"speech token outside [0, 6560] for {identifier}: {token!r}")


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error 2" in message or "cuda malloc" in message


def _require_eight_devices(paths: RunPaths) -> None:
    if tuple(paths.visible_devices) != tuple(range(8)):
        raise TokenizerError("tokenizer qualification/cache requires exactly CUDA devices 0 through 7")


def _select_duration_stratified_clips(paths: RunPaths) -> list[AudioInput]:
    """Choose short/median/long without rereading every source audio payload."""

    selected, source_checksums = _select_pilot_source_paths(paths)
    samples = _read_selected_pilot_audio(paths, selected, source_checksums)
    from .cache import _decode_audio

    decoded = [_decode_audio(sample) for sample in samples]
    decoded.sort(key=lambda item: (item.frames, item.source_relative_path))
    return [decoded[0], decoded[len(decoded) // 2], decoded[-1]]


def _select_pilot_source_paths(paths: RunPaths) -> tuple[tuple[str, ...], Mapping[str, str]]:
    """Authenticate the split plan and retain its 33 lowest seeded source IDs."""

    plan_dir = paths.run_root / "split_plan"
    try:
        manifest = _load_mapping(plan_dir / "manifest.json")
    except CacheIntegrityError as exc:
        raise TokenizerError("pilot cannot load the authenticated split-plan manifest") from exc
    if manifest.get("seed") != paths.seed:
        raise TokenizerError("pilot split-plan seed does not match the configured seed")
    plan_shards = manifest.get("plan_shards")
    source_inventory = manifest.get("source_inventory")
    source_checksums = source_inventory.get("source_archives") if isinstance(source_inventory, Mapping) else None
    if not isinstance(plan_shards, Mapping) or not plan_shards or not isinstance(source_checksums, Mapping):
        raise TokenizerError("pilot split-plan manifest has no authenticated shard inventory")
    actual_shards = {path.name: path for path in plan_dir.glob("shard_*.jsonl") if path.is_file()}
    if set(actual_shards) != set(plan_shards):
        raise TokenizerError("pilot split-plan shard set does not match its manifest")

    candidates: list[tuple[bytes, str]] = []
    observed_rows = 0
    for name in sorted(plan_shards):
        expected_sha256 = plan_shards[name]
        match = re.fullmatch(r"shard_(\d{6})\.jsonl", name)
        if (
            match is None
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or sha256_file(actual_shards[name]) != expected_sha256
        ):
            raise TokenizerError(f"pilot split-plan shard checksum changed: {name}")
        shard = int(match.group(1))
        try:
            rows = iter_split_rows(plan_dir, shard)
            for row in rows:
                if not row.source_relative_path.startswith(f"{shard:06d}/"):
                    raise TokenizerError(f"pilot split-plan row is stored under the wrong shard: {row.source_relative_path}")
                candidate = (
                    hashlib.sha256(f"{paths.seed}\0{row.source_relative_path}".encode("utf-8")).digest(),
                    row.source_relative_path,
                )
                if len(candidates) < 33:
                    bisect.insort(candidates, candidate)
                elif candidate < candidates[-1]:
                    bisect.insort(candidates, candidate)
                    candidates.pop()
                observed_rows += 1
        except (OSError, SourceIntegrityError, ValueError) as exc:
            raise TokenizerError(f"pilot cannot read split-plan shard: {name}") from exc
    total = manifest.get("total")
    if isinstance(total, bool) or not isinstance(total, int) or observed_rows != total:
        raise TokenizerError(f"pilot split-plan row count changed: expected {total!r}, got {observed_rows}")
    if len(candidates) < 3:
        raise TokenizerError("pilot needs at least three deterministic source clips")

    checksums: dict[str, str] = {}
    for name, value in source_checksums.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"shard_\d{6}\.tar", name) is None
            or not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise TokenizerError("pilot source archive inventory is malformed")
        checksums[name] = value
    return tuple(source_relative_path for _, source_relative_path in candidates), checksums


def _read_selected_pilot_audio(
    paths: RunPaths,
    selected: Sequence[str],
    source_checksums: Mapping[str, str],
) -> list[TarAudioSample]:
    """Read only selected MP3 members after checking each touched source tar."""

    grouped: dict[str, list[tuple[str, str]]] = {}
    for source_relative_path in selected:
        match = re.fullmatch(r"(\d{6})/([^/]+\.mp3)", source_relative_path)
        if match is None:
            raise TokenizerError(f"invalid pilot source path: {source_relative_path}")
        archive_name = f"shard_{match.group(1)}.tar"
        grouped.setdefault(archive_name, []).append((source_relative_path, match.group(2)))

    loaded: dict[str, TarAudioSample] = {}
    for archive_name, requested in grouped.items():
        expected_sha256 = source_checksums.get(archive_name)
        archive_path = paths.dataset_root / "train" / archive_name
        if expected_sha256 is None or not archive_path.is_file() or sha256_file(archive_path) != expected_sha256:
            raise TokenizerError(f"pilot source archive checksum changed: {archive_name}")
        wanted = {member_name for _, member_name in requested}
        found: dict[str, list[tarfile.TarInfo]] = {name: [] for name in wanted}
        try:
            with tarfile.open(archive_path, "r") as archive:
                for member in archive:
                    if member.name in found:
                        found[member.name].append(member)
                for source_relative_path, member_name in requested:
                    members = found[member_name]
                    if len(members) != 1 or not members[0].isfile():
                        raise TokenizerError(f"pilot source member is missing, duplicated, or not a file: {source_relative_path}")
                    extracted = archive.extractfile(members[0])
                    if extracted is None:
                        raise TokenizerError(f"pilot source member cannot be extracted: {source_relative_path}")
                    payload = extracted.read()
                    if len(payload) != members[0].size:
                        raise TokenizerError(f"pilot source member is truncated: {source_relative_path}")
                    loaded[source_relative_path] = TarAudioSample(source_relative_path, payload)
        except (OSError, tarfile.TarError) as exc:
            raise TokenizerError(f"pilot cannot read selected source archive: {archive_name}") from exc
    if set(loaded) != set(selected):
        raise TokenizerError("pilot selected source set was not extracted exactly once")
    return [loaded[source_relative_path] for source_relative_path in selected]


def _load_frozen_reconstructor(paths: RunPaths) -> Any:
    if not paths.base_model_dir.is_dir():
        raise TokenizerError(f"missing frozen base CosyVoice3 model: {paths.base_model_dir}")
    cosyvoice = import_module("cosyvoice.cli.cosyvoice")
    return cosyvoice.CosyVoice3(str(paths.base_model_dir), load_trt=False, load_vllm=False, fp16=False)


def _write_wav(audio: AudioInput, path: Path) -> None:
    samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    if samples.size != audio.frames:
        raise TokenizerError(f"audio frame count disagrees for {audio.source_relative_path}")
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype("<i2", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm.tobytes())


def _reconstruct_to_wav(reconstructor: Any, original: Path, destination: Path) -> None:
    try:
        output = next(reconstructor.inference_vc(str(original), str(original), stream=False))
        samples = np.asarray(output["tts_speech"].squeeze().detach().cpu(), dtype=np.float32)
        _write_wav(AudioInput(original.name, samples, 24_000, int(samples.size)), destination)
    except Exception as exc:
        raise TokenizerError(f"frozen flow/HiFT reconstruction failed for {original}") from exc


def _pilot_index(records: Sequence[Mapping[str, object]]) -> str:
    lines = ["# Speech-token reconstruction pilot", "", "Listen to each original/reconstruction pair before approving this exact manifest.", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['label']}: `{record['source_relative_path']}`",
                "",
                f"- Original: [{record['original']}]({record['original']})",
                f"- Reconstruction: [{record['reconstructed']}]({record['reconstructed']})",
                f"- Tokens: [{record['tokens']}]({record['tokens']}); count={record['token_count']}, rate={record['token_rate_hz']:.2f} Hz, IDs={record['token_min']}..{record['token_max']}",
                "",
                "Listening decision: [ ] intelligible  [ ] speaker/prompt conditioning acceptable  [ ] approve",
                "",
            ]
        )
    return "\n".join(lines)


def _pilot_artifacts(index: Path, records: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Return the complete, relative file-to-checksum inventory a reviewer heard."""

    artifacts = {index.name: sha256_file(index)}
    for record in records:
        for name_key, checksum_key in (
            ("original", "original_sha256"),
            ("tokens", "tokens_sha256"),
            ("reconstructed", "reconstructed_sha256"),
        ):
            name = record.get(name_key)
            checksum = record.get(checksum_key)
            if not isinstance(name, str) or not isinstance(checksum, str):
                raise TokenizerError(f"pilot record is missing {name_key} checksum")
            artifacts[name] = checksum
    return artifacts


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def require_pilot_approval(paths: RunPaths) -> None:
    """Require approval and rehash every listening artifact before cache work."""

    store = StageStore(paths.stages_dir)
    try:
        pilot = store.require("pilot")
        approval = store.require("pilot_approval")
    except StageRequirementError as exc:
        raise PilotApprovalError("cache tokenization requires a current checksum-bound pilot approval") from exc
    if approval.payload.get("pilot_manifest_sha256") != pilot.manifest_sha256:
        raise PilotApprovalError("pilot approval does not match the current pilot manifest")
    _verify_pilot_artifacts(paths, pilot.payload)


def _require_current_pilot_approval(paths: RunPaths) -> None:
    """Backward-compatible internal spelling for the checksum-bound approval gate."""

    require_pilot_approval(paths)


def _verify_pilot_artifacts(paths: RunPaths, payload: Mapping[str, object]) -> None:
    """Fail closed unless every recorded listening artifact remains under pilot/ intact."""

    root = (paths.run_root / "pilot").resolve()
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise PilotApprovalError("pilot artifact inventory is missing or invalid")
    checked: dict[str, str] = {}
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise PilotApprovalError("pilot artifact inventory is invalid")
        path = _pilot_artifact_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise PilotApprovalError(f"pilot artifact is missing or checksum-mismatched: {relative}")
        checked[relative] = expected
    index = payload.get("index")
    index_sha256 = payload.get("index_sha256")
    if not isinstance(index, str) or not isinstance(index_sha256, str) or checked.get(index) != index_sha256:
        raise PilotApprovalError("pilot listening index is absent from the artifact inventory")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise PilotApprovalError("pilot clips are missing from the artifact inventory")
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise PilotApprovalError("pilot clip record is invalid")
        for name_key, checksum_key in (
            ("original", "original_sha256"),
            ("tokens", "tokens_sha256"),
            ("reconstructed", "reconstructed_sha256"),
        ):
            name = clip.get(name_key)
            expected = clip.get(checksum_key)
            if not isinstance(name, str) or not isinstance(expected, str) or checked.get(name) != expected:
                raise PilotApprovalError(f"pilot {name_key} is absent from the artifact inventory")


def _pilot_artifact_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts:
        raise PilotApprovalError(f"pilot artifact path is unsafe: {relative!r}")
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise PilotApprovalError(f"pilot artifact path escapes pilot directory: {relative!r}")
    return resolved


def _validate_worker_result(request: CacheShardRequest, result: CacheShardResult) -> None:
    """Check a worker's claim against its lease before considering that lease complete."""

    if result.shard != request.shard:
        raise TokenizerError(f"cache worker returned wrong shard: {result.shard} != {request.shard}")
    expected_paths = {
        "phase1": request.cache_root / "phase1" / f"shard_{request.shard:06d}.parquet",
        "phase2": request.cache_root / "phase2" / f"shard_{request.shard:06d}.parquet",
        "prompts": request.cache_root / "eval_prompts.parquet",
        "manifest": request.cache_root / "shard_manifests" / f"shard_{request.shard:06d}.json",
    }
    for label, actual, expected in (
        ("phase1", result.phase1_path, expected_paths["phase1"]),
        ("phase2", result.phase2_path, expected_paths["phase2"]),
        ("prompt", result.eval_prompts_path, expected_paths["prompts"]),
        ("shard manifest", result.shard_manifest_path, expected_paths["manifest"]),
    ):
        if actual != expected:
            raise TokenizerError(f"cache worker returned unexpected {label} path")
    source_sha256 = request.expected_source_sha256 or sha256_file(request.source_tar)
    plan_path = request.plan_dir / f"shard_{request.shard:06d}.jsonl"
    plan_sha256 = request.expected_plan_sha256 or sha256_file(plan_path)
    if result.source_sha256 != source_sha256 or result.plan_sha256 != plan_sha256:
        raise TokenizerError("cache worker result checksum identity does not match its lease")
    _verify_published_shard(request, result, source_sha256, plan_path, plan_sha256)


def _verify_published_shard(
    request: CacheShardRequest,
    result: CacheShardResult,
    source_sha256: str,
    plan_path: Path,
    plan_sha256: str,
) -> None:
    if not result.shard_manifest_path.is_file():
        raise TokenizerError("cache worker did not publish its shard manifest")
    try:
        manifest = _load_mapping(result.shard_manifest_path)
        if manifest.get("shard") != request.shard:
            raise TokenizerError("published shard manifest identity does not match its lease")
        if manifest.get("source_tar") != str(request.source_tar) or manifest.get("source_sha256") != source_sha256:
            raise TokenizerError("published shard manifest source identity does not match its lease")
        if manifest.get("split_plan") != str(plan_path) or manifest.get("split_plan_sha256") != plan_sha256:
            raise TokenizerError("published shard manifest plan identity does not match its lease")
        phase = manifest.get("phase")
        if not isinstance(phase, Mapping):
            raise TokenizerError("published shard manifest phase metadata is invalid")
        verified_rows: dict[int, int] = {}
        for number, output_path in ((1, result.phase1_path), (2, result.phase2_path)):
            metadata = phase.get(str(number))
            if not isinstance(metadata, Mapping) or metadata.get("path") != str(output_path.relative_to(request.cache_root)):
                raise TokenizerError("published shard manifest phase path does not match its lease")
            verified_rows[number] = _verify_phase_metadata(request.cache_root, metadata, number)
        if dict(result.phase_rows) != verified_rows:
            raise TokenizerError("published shard manifest phase rows do not match worker result")
        if not result.eval_prompts_path.is_file():
            raise TokenizerError("cache worker did not publish eval prompt metadata")
    except TokenizerError:
        raise
    except (CacheIntegrityError, OSError, ValueError, TypeError, KeyError) as exc:
        raise TokenizerError("published cache shard verification failed") from exc


def _verify_requested_cache_set(cache_root: Path, expected_shards: set[int]) -> CacheManifest:
    manifest = verify_cache(cache_root)
    if set(manifest.shards) != expected_shards:
        raise TokenizerError(
            f"verified cache shard set does not match requested shard set: "
            f"{sorted(manifest.shards)} != {sorted(expected_shards)}"
        )
    return manifest


def _cache_worker(device: int, model_path: str, work: Any, results: Any) -> None:
    os.environ["LOCAL_RANK"] = str(device)
    try:
        tokenizer = OnnxSpeechTokenizer(Path(model_path), local_rank=device)
        while True:
            value = work.get()
            if value is None:
                return
            request = CacheShardRequest.from_dict(_mapping(value, "cache worker request"))
            result = build_cache_shard(request, tokenizer)
            results.put({"result": result.to_dict()})
    except BaseException as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TokenizerError(f"invalid {name}")
    return value

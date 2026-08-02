"""Streaming, auditable metadata-only cache construction for Balalaika tars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import io
import json
import os
from pathlib import Path
import re
import struct
import tarfile
from typing import Any, Iterator, Mapping, Protocol, Sequence
import wave

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import sha256_file
from .sources import JoinedRow, SourceIntegrityError, iter_split_rows


TOKEN_MIN = 0
TOKEN_MAX = 6560
PROMPT_RESERVATION_COUNT = 20
_TAR_NAME = re.compile(r"shard_(\d{6})\.tar")
_PROMPT_WAV = re.compile(r"eval_prompts/voice_(\d{2})\.wav")
_PHASE_COLUMNS = ["source_relative_path", "text", "instruct", "agreement", "speech_token", "speech_token_len"]
_PROMPT_COLUMNS = [
    "source_relative_path",
    "text",
    "instruct",
    "agreement",
    "reservation_score",
    "audio_path",
    "sample_rate",
    "frames",
    "duration_seconds",
    "wav_sha256",
]


class CacheIntegrityError(RuntimeError):
    """Raised when immutable cache inputs or published cache outputs disagree."""


class SpeechTokenizer(Protocol):
    """The bounded, injectable speech-token extraction interface used by workers."""

    def extract(self, audio: list["AudioInput"]) -> list[list[int]]: ...


@dataclass(frozen=True)
class TarAudioSample:
    """One validated tar JSON/MP3 pair; compressed bytes never touch disk."""

    source_relative_path: str
    audio_bytes: bytes


@dataclass(frozen=True)
class AudioInput:
    """A decoded, 24 kHz mono input handed to the injected tokenizer."""

    source_relative_path: str
    samples: Any
    sample_rate: int
    frames: int


@dataclass(frozen=True)
class CacheShardRequest:
    """Serializable work item for one source tar and its split-plan shard."""

    source_tar: Path
    plan_dir: Path
    cache_root: Path
    shard: int
    batch_size: int = 8
    expected_source_sha256: str | None = None
    expected_plan_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in ("source_tar", "plan_dir", "cache_root"):
            value[key] = str(value[key])
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CacheShardRequest":
        return cls(
            source_tar=Path(_required_string(value, "source_tar")),
            plan_dir=Path(_required_string(value, "plan_dir")),
            cache_root=Path(_required_string(value, "cache_root")),
            shard=_required_int(value, "shard"),
            batch_size=_optional_int(value, "batch_size", 8),
            expected_source_sha256=_optional_string(value, "expected_source_sha256"),
            expected_plan_sha256=_optional_string(value, "expected_plan_sha256"),
        )


@dataclass(frozen=True)
class CacheShardResult:
    """Serializable completed cache-shard identity for persistent workers."""

    shard: int
    phase1_path: Path
    phase2_path: Path
    eval_prompts_path: Path
    shard_manifest_path: Path
    source_sha256: str
    plan_sha256: str
    phase_rows: Mapping[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "shard": self.shard,
            "phase1_path": str(self.phase1_path),
            "phase2_path": str(self.phase2_path),
            "eval_prompts_path": str(self.eval_prompts_path),
            "shard_manifest_path": str(self.shard_manifest_path),
            "source_sha256": self.source_sha256,
            "plan_sha256": self.plan_sha256,
            "phase_rows": {str(key): value for key, value in self.phase_rows.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CacheShardResult":
        rows = value.get("phase_rows")
        if not isinstance(rows, Mapping):
            raise ValueError("phase_rows must be a mapping")
        return cls(
            shard=_required_int(value, "shard"),
            phase1_path=Path(_required_string(value, "phase1_path")),
            phase2_path=Path(_required_string(value, "phase2_path")),
            eval_prompts_path=Path(_required_string(value, "eval_prompts_path")),
            shard_manifest_path=Path(_required_string(value, "shard_manifest_path")),
            source_sha256=_required_string(value, "source_sha256"),
            plan_sha256=_required_string(value, "plan_sha256"),
            phase_rows={int(key): _value_int(item, f"phase_rows.{key}") for key, item in rows.items()},
        )


@dataclass(frozen=True)
class CacheManifest:
    """The verified aggregate of independently published cache shards."""

    root: Path
    shards: Mapping[int, Path]
    phase_rows: Mapping[int, int]
    prompt_count: int


@dataclass(frozen=True)
class _StagedArtifact:
    """A verified same-directory temporary artifact awaiting transactional rename."""

    target: Path
    partial: Path


def iter_tar_audio(path: Path) -> Iterator[TarAudioSample]:
    """Stream exactly adjacent JSON/MP3 pairs, retaining at most one unpaired key."""

    match = _TAR_NAME.fullmatch(path.name)
    if match is None:
        raise CacheIntegrityError(f"source archive must be named shard_NNNNNN.tar: {path}")
    shard = match.group(1)
    pending_key: str | None = None
    pending_json: Mapping[str, object] | None = None
    pending_audio: bytes | None = None
    seen: set[str] = set()
    try:
        archive = tarfile.open(path, "r")
    except (OSError, tarfile.TarError) as exc:
        raise CacheIntegrityError(f"cannot read source tar: {path}") from exc
    with archive:
        for member in archive:
            if not member.isfile():
                raise CacheIntegrityError(f"unexpected non-file tar member: {member.name}")
            member_path = Path(member.name)
            if member_path.parent != Path("."):
                raise CacheIntegrityError(f"unexpected tar member path: {member.name}")
            suffix = member_path.suffix
            if suffix not in {".json", ".mp3"}:
                raise CacheIntegrityError(f"unexpected tar member suffix: {member.name}")
            key = member_path.stem
            if not key:
                raise CacheIntegrityError(f"empty tar member stem: {member.name}")
            if pending_key is not None and key != pending_key:
                missing = "JSON" if pending_json is None else "MP3"
                raise CacheIntegrityError(f"missing {missing} partner for tar member {pending_key}")
            if pending_key is None:
                if key in seen:
                    raise CacheIntegrityError(f"duplicate tar pair: {key}")
                pending_key = key
            extracted = archive.extractfile(member)
            if extracted is None:
                raise CacheIntegrityError(f"cannot extract tar member: {member.name}")
            payload = extracted.read()
            if suffix == ".json":
                if pending_json is not None:
                    raise CacheIntegrityError(f"duplicate JSON member for tar pair: {key}")
                pending_json = _tar_json(payload, member.name)
            else:
                if pending_audio is not None:
                    raise CacheIntegrityError(f"duplicate MP3 member for tar pair: {key}")
                pending_audio = payload
            if pending_json is not None and pending_audio is not None:
                source_relative_path = _required_string(pending_json, "source_relative_path")
                expected = f"{shard}/{pending_key}.mp3"
                if source_relative_path != expected:
                    raise CacheIntegrityError(
                        f"tar JSON source_relative_path mismatch: expected {expected}, got {source_relative_path}"
                    )
                yield TarAudioSample(source_relative_path, pending_audio)
                seen.add(pending_key)
                pending_key = None
                pending_json = None
                pending_audio = None
    if pending_key is not None:
        missing = "JSON" if pending_json is None else "MP3"
        raise CacheIntegrityError(f"missing {missing} partner for tar member {pending_key}")


def build_cache_shard(request: CacheShardRequest, tokenizer: SpeechTokenizer) -> CacheShardResult:
    """Build, reopen-verify, and atomically publish one metadata-only cache shard."""

    _validate_request(request)
    plan_path = request.plan_dir / f"shard_{request.shard:06d}.jsonl"
    source_sha256 = sha256_file(request.source_tar)
    plan_sha256 = sha256_file(plan_path)
    if request.expected_source_sha256 is not None and source_sha256 != request.expected_source_sha256:
        raise CacheIntegrityError("source tar checksum does not match request")
    if request.expected_plan_sha256 is not None and plan_sha256 != request.expected_plan_sha256:
        raise CacheIntegrityError("split-plan checksum does not match request")
    rows = _split_rows(request.plan_dir, request.shard)
    prompt_indices = _prompt_indices(request.plan_dir, rows)
    phase_rows: dict[int, list[dict[str, object]]] = {1: [], 2: []}
    pending_batch: list[tuple[JoinedRow, AudioInput]] = []
    prompt_audio: list[tuple[JoinedRow, AudioInput, int]] = []
    remaining = dict(rows)

    def extract_batch() -> None:
        if not pending_batch:
            return
        inputs = [audio for _, audio in pending_batch]
        try:
            tokens_by_audio = tokenizer.extract(inputs)
        except Exception as exc:
            raise CacheIntegrityError("speech-token extraction failed") from exc
        if len(tokens_by_audio) != len(pending_batch):
            raise CacheIntegrityError("speech tokenizer returned a different number of token sequences")
        for (row, _), tokens in zip(pending_batch, tokens_by_audio, strict=True):
            phase = row.phase
            assert phase in {1, 2}
            token_list = _validated_tokens(tokens, row.source_relative_path)
            phase_rows[phase].append(
                {
                    "source_relative_path": row.source_relative_path,
                    "text": row.text,
                    "instruct": row.instruct,
                    "agreement": row.agreement,
                    "speech_token": token_list,
                    "speech_token_len": len(token_list),
                }
            )
        pending_batch.clear()

    for sample in iter_tar_audio(request.source_tar):
        row = remaining.pop(sample.source_relative_path, None)
        if row is None:
            raise CacheIntegrityError(f"tar member has no split-plan row: {sample.source_relative_path}")
        _validate_row_route(row)
        if row.phase is None and not row.reserved:
            continue
        try:
            decoded = _decode_audio(sample)
        except CacheIntegrityError:
            raise
        except Exception as exc:
            raise CacheIntegrityError(f"decode failed for {sample.source_relative_path}") from exc
        if decoded.source_relative_path != sample.source_relative_path or decoded.sample_rate != 24_000 or decoded.frames < 1:
            raise CacheIntegrityError(f"decoder returned invalid 24 kHz mono audio for {sample.source_relative_path}")
        if row.reserved:
            prompt_audio.append((row, decoded, prompt_indices[row.source_relative_path]))
        else:
            pending_batch.append((row, decoded))
            if len(pending_batch) >= request.batch_size:
                extract_batch()
    extract_batch()
    if remaining:
        raise CacheIntegrityError(f"split-plan rows missing from source tar: {sorted(remaining)[0]}")

    phase1_path, phase2_path, shard_manifest_path = _publish_shard_transaction(
        request.cache_root,
        request.shard,
        request.source_tar,
        source_sha256,
        plan_path,
        plan_sha256,
        len(rows),
        phase_rows,
        prompt_audio,
    )
    return CacheShardResult(
        shard=request.shard,
        phase1_path=phase1_path,
        phase2_path=phase2_path,
        eval_prompts_path=request.cache_root / "eval_prompts.parquet",
        shard_manifest_path=shard_manifest_path,
        source_sha256=source_sha256,
        plan_sha256=plan_sha256,
        phase_rows={1: len(phase_rows[1]), 2: len(phase_rows[2])},
    )


def verify_cache(root: Path) -> CacheManifest:
    """Reopen and reconcile every published source-shard cache artifact."""

    manifest_paths = sorted((root / "shard_manifests").glob("shard_*.json"))
    if not manifest_paths:
        raise CacheIntegrityError(f"no cache shard manifests found in {root}")
    phase_rows = {1: 0, 2: 0}
    prompts: list[dict[str, object]] = []
    shards: dict[int, Path] = {}
    for manifest_path in manifest_paths:
        data = _load_mapping(manifest_path)
        shard = _required_int(data, "shard")
        if shard in shards:
            raise CacheIntegrityError(f"duplicate cache shard manifest: {shard}")
        shards[shard] = manifest_path
        _verify_input_checksum(data, "source_tar", "source_sha256")
        _verify_input_checksum(data, "split_plan", "split_plan_sha256")
        plan_rows = _count_plan_rows(Path(_required_string(data, "split_plan")))
        if plan_rows != _required_int(data, "plan_row_count"):
            raise CacheIntegrityError(f"split-plan row count changed for shard {shard}")
        phase = data.get("phase")
        if not isinstance(phase, Mapping):
            raise CacheIntegrityError(f"invalid phase metadata for shard {shard}")
        for number in (1, 2):
            metadata = phase.get(str(number))
            if not isinstance(metadata, Mapping):
                raise CacheIntegrityError(f"missing phase {number} metadata for shard {shard}")
            phase_rows[number] += _verify_phase_metadata(root, metadata, number)
        stored_prompts = data.get("prompts")
        if not isinstance(stored_prompts, list):
            raise CacheIntegrityError(f"invalid prompt metadata for shard {shard}")
        for prompt in stored_prompts:
            if not isinstance(prompt, dict):
                raise CacheIntegrityError(f"invalid prompt record for shard {shard}")
            wav = root / _required_string(prompt, "audio_path")
            if not wav.is_file() or sha256_file(wav) != _required_string(prompt, "wav_sha256"):
                raise CacheIntegrityError(f"prompt WAV checksum changed: {wav}")
            prompts.append(prompt)
    _verify_prompt_set(root, prompts)
    _verify_prompt_parquet(root, prompts)
    aggregate_path = root / "manifest.json"
    if aggregate_path.is_file():
        aggregate = _load_mapping(aggregate_path)
        for manifest_path in manifest_paths:
            listed = aggregate.get("shard_manifests", {})
            if not isinstance(listed, Mapping) or listed.get(manifest_path.name) != sha256_file(manifest_path):
                raise CacheIntegrityError(f"aggregate manifest does not reconcile {manifest_path.name}")
    return CacheManifest(root=root, shards=shards, phase_rows=phase_rows, prompt_count=len(prompts))


def _decode_audio(sample: TarAudioSample) -> AudioInput:
    """Decode an MP3 directly from its tar bytes and resample it to 24 kHz mono."""

    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(io.BytesIO(sample.audio_bytes))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 24_000:
        waveform = torchaudio.transforms.Resample(sample_rate, 24_000)(waveform)
    return AudioInput(sample.source_relative_path, waveform, 24_000, int(waveform.shape[-1]))


def _validate_request(request: CacheShardRequest) -> None:
    if request.shard < 0 or request.shard > 999_999 or request.batch_size < 1:
        raise ValueError("shard must be non-negative and batch_size must be positive")
    match = _TAR_NAME.fullmatch(request.source_tar.name)
    if match is None or int(match.group(1)) != request.shard:
        raise CacheIntegrityError("source tar name must match request shard")
    if not request.source_tar.is_file():
        raise CacheIntegrityError(f"missing source tar: {request.source_tar}")
    if not (request.plan_dir / f"shard_{request.shard:06d}.jsonl").is_file():
        raise CacheIntegrityError(f"missing split-plan shard: {request.shard}")


def _split_rows(plan_dir: Path, shard: int) -> dict[str, JoinedRow]:
    try:
        rows = list(iter_split_rows(plan_dir, shard))
    except (OSError, SourceIntegrityError, ValueError) as exc:
        raise CacheIntegrityError(f"cannot read split-plan shard {shard}") from exc
    by_id: dict[str, JoinedRow] = {}
    for row in rows:
        if row.source_relative_path in by_id:
            raise CacheIntegrityError(f"duplicate split-plan row: {row.source_relative_path}")
        by_id[row.source_relative_path] = row
    return by_id


def _prompt_indices(plan_dir: Path, rows: Mapping[str, JoinedRow]) -> dict[str, int]:
    manifest = _load_mapping(plan_dir / "manifest.json")
    entries = manifest.get("reserved_prompts")
    if not isinstance(entries, list):
        raise CacheIntegrityError("split-plan manifest is missing reserved_prompts")
    identifiers: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CacheIntegrityError("invalid reserved prompt entry")
        identifiers.append(_required_string(entry, "source_relative_path"))
    if len(identifiers) != PROMPT_RESERVATION_COUNT:
        raise CacheIntegrityError(f"split-plan manifest must contain exactly {PROMPT_RESERVATION_COUNT} reserved prompt identities")
    if len(set(identifiers)) != PROMPT_RESERVATION_COUNT:
        raise CacheIntegrityError("split-plan manifest reserved prompt identities must be unique")
    indices = {identifier: index for index, identifier in enumerate(identifiers)}
    for row in rows.values():
        if not row.reserved:
            continue
        if row.source_relative_path not in indices:
            raise CacheIntegrityError(f"reserved row missing from split-plan manifest: {row.source_relative_path}")
    return indices


def _validate_row_route(row: JoinedRow) -> None:
    if row.reserved and row.phase is not None:
        raise CacheIntegrityError(f"reserved prompt retains a training phase: {row.source_relative_path}")
    if not row.reserved and row.phase not in {None, 1, 2}:
        raise CacheIntegrityError(f"invalid split-plan phase for {row.source_relative_path}")
    if row.phase in {1, 2} and row.agreement is None:
        raise CacheIntegrityError(f"training row lacks agreement: {row.source_relative_path}")


def _validated_tokens(tokens: object, source_relative_path: str) -> list[int]:
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or not tokens:
        raise CacheIntegrityError(f"speech tokenizer returned empty/non-list tokens for {source_relative_path}")
    result: list[int] = []
    for token in tokens:
        if not isinstance(token, int) or isinstance(token, bool) or token < TOKEN_MIN or token > TOKEN_MAX:
            raise CacheIntegrityError(f"speech token outside [0, 6560] for {source_relative_path}: {token!r}")
        result.append(token)
    return result


def _phase_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.table(
        {
            "source_relative_path": pa.array([row["source_relative_path"] for row in rows], type=pa.string()),
            "text": pa.array([row["text"] for row in rows], type=pa.string()),
            "instruct": pa.array([row["instruct"] for row in rows], type=pa.string()),
            "agreement": pa.array([row["agreement"] for row in rows], type=pa.float64()),
            "speech_token": pa.array([row["speech_token"] for row in rows], type=pa.list_(pa.int32())),
            "speech_token_len": pa.array([row["speech_token_len"] for row in rows], type=pa.int32()),
        }
    )


def _phase_metadata(root: Path, phase: int, shard: int, rows: list[dict[str, object]], staged: _StagedArtifact) -> dict[str, object]:
    path = root / f"phase{phase}" / f"shard_{shard:06d}.parquet"
    tokens = [token for row in rows for token in row["speech_token"]]
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(staged.partial),
        "row_count": len(rows),
        "token_min": min(tokens) if tokens else None,
        "token_max": max(tokens) if tokens else None,
    }


def _publish_shard_transaction(
    root: Path,
    shard: int,
    source_tar: Path,
    source_sha256: str,
    plan_path: Path,
    plan_sha256: str,
    plan_row_count: int,
    phase_rows: Mapping[int, list[dict[str, object]]],
    prompt_audio: list[tuple[JoinedRow, AudioInput, int]],
) -> tuple[Path, Path, Path]:
    """Stage every shard output before any final path is replaced, then commit as one unit."""

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".cache.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _stage_and_commit_shard(
                root, shard, source_tar, source_sha256, plan_path, plan_sha256, plan_row_count, phase_rows, prompt_audio
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _stage_and_commit_shard(
    root: Path,
    shard: int,
    source_tar: Path,
    source_sha256: str,
    plan_path: Path,
    plan_sha256: str,
    plan_row_count: int,
    phase_rows: Mapping[int, list[dict[str, object]]],
    prompt_audio: list[tuple[JoinedRow, AudioInput, int]],
) -> tuple[Path, Path, Path]:
    phase_paths = {phase: root / f"phase{phase}" / f"shard_{shard:06d}.parquet" for phase in (1, 2)}
    shard_manifest_path = root / "shard_manifests" / f"shard_{shard:06d}.json"
    staged: list[_StagedArtifact] = []
    try:
        phase_artifacts = {
            phase: _stage_parquet(phase_paths[phase], _phase_table(phase_rows[phase]), lambda loaded, rows=phase_rows[phase]: _verify_phase_table(loaded, rows))
            for phase in (1, 2)
        }
        staged.extend(phase_artifacts.values())
        prompt_records, prompt_artifacts = _stage_prompts(root, prompt_audio)
        staged.extend(prompt_artifacts)
        shard_manifest = {
            "shard": shard,
            "source_tar": str(source_tar),
            "source_sha256": source_sha256,
            "split_plan": str(plan_path),
            "split_plan_sha256": plan_sha256,
            "plan_row_count": plan_row_count,
            "phase": {
                str(phase): _phase_metadata(root, phase, shard, phase_rows[phase], phase_artifacts[phase]) for phase in (1, 2)
            },
            "prompts": prompt_records,
        }
        shard_manifest_artifact = _stage_json(shard_manifest_path, shard_manifest)
        staged.append(shard_manifest_artifact)
        old_manifests = [
            _load_mapping(path)
            for path in sorted((root / "shard_manifests").glob("shard_*.json"))
            if path != shard_manifest_path
        ]
        manifests = [*old_manifests, shard_manifest]
        prompts = sorted(
            [prompt for manifest in manifests for prompt in manifest.get("prompts", [])],
            key=lambda prompt: _required_string(prompt, "audio_path"),
        )
        if not all(isinstance(prompt, dict) for prompt in prompts):
            raise CacheIntegrityError("invalid prompt metadata while staging aggregate")
        eval_artifact = _stage_prompt_parquet(root / "eval_prompts.parquet", prompts)
        staged.append(eval_artifact)
        root_manifest = {
            "schema_version": "balalaika-cache-v1",
            "shard_manifests": {
                **{path.name: sha256_file(path) for path in sorted((root / "shard_manifests").glob("shard_*.json")) if path != shard_manifest_path},
                shard_manifest_path.name: sha256_file(shard_manifest_artifact.partial),
            },
            "phase_rows": {
                str(phase): sum(_required_int(manifest["phase"][str(phase)], "row_count") for manifest in manifests)
                for phase in (1, 2)
            },
            "prompt_count": len(prompts),
        }
        root_manifest_artifact = _stage_json(root / "manifest.json", root_manifest)
        staged.append(root_manifest_artifact)
        _transactional_publish([
            phase_artifacts[1],
            phase_artifacts[2],
            *prompt_artifacts,
            eval_artifact,
            shard_manifest_artifact,
            root_manifest_artifact,
        ])
        return phase_paths[1], phase_paths[2], shard_manifest_path
    except Exception:
        for artifact in staged:
            _remove_partial(artifact.partial)
        raise


def _stage_prompts(root: Path, prompts: list[tuple[JoinedRow, AudioInput, int]]) -> tuple[list[dict[str, object]], list[_StagedArtifact]]:
    records: list[dict[str, object]] = []
    artifacts: list[_StagedArtifact] = []
    for row, audio, index in prompts:
        relative = f"eval_prompts/voice_{index:02d}.wav"
        artifact = _stage_bytes(root / relative, _pcm_wav(audio))
        artifacts.append(artifact)
        records.append(
            {
                "source_relative_path": row.source_relative_path,
                "text": row.text,
                "instruct": row.instruct,
                "agreement": row.agreement,
                "reservation_score": row.reservation_score,
                "audio_path": relative,
                "sample_rate": audio.sample_rate,
                "frames": audio.frames,
                "duration_seconds": audio.frames / audio.sample_rate,
                "wav_sha256": sha256_file(artifact.partial),
            }
        )
    return records, artifacts


def _stage_prompt_parquet(path: Path, prompts: list[dict[str, object]]) -> _StagedArtifact:
    table = pa.table({column: pa.array([prompt.get(column) for prompt in prompts], type=_prompt_type(column)) for column in _PROMPT_COLUMNS})
    return _stage_parquet(path, table, _verify_prompt_table)


def _prompt_type(column: str) -> pa.DataType:
    if column in {"sample_rate", "frames"}:
        return pa.int32()
    if column in {"agreement", "duration_seconds"}:
        return pa.float64()
    return pa.string()


def _stage_parquet(path: Path, table: pa.Table, verifier) -> _StagedArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    _recover_backup(path)
    _remove_partial(partial)
    try:
        pq.write_table(table, partial, compression="zstd")
        _fsync_file(partial)
        verifier(pq.read_table(partial))
        return _StagedArtifact(path, partial)
    except Exception:
        _remove_partial(partial)
        raise


def _stage_bytes(path: Path, payload: bytes) -> _StagedArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    _recover_backup(path)
    _remove_partial(partial)
    try:
        with partial.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not partial.is_file() or not payload:
            raise CacheIntegrityError(f"failed to write prompt WAV: {path}")
        return _StagedArtifact(path, partial)
    except Exception:
        _remove_partial(partial)
        raise


def _stage_json(path: Path, value: Mapping[str, object]) -> _StagedArtifact:
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise CacheIntegrityError(f"cannot serialize cache manifest: {path}") from exc
    artifact = _stage_bytes(path, payload)
    _load_mapping(artifact.partial)
    return artifact


def _transactional_publish(artifacts: list[_StagedArtifact]) -> None:
    """Replace staged finals in order, restoring all prior finals if any replacement fails."""

    published: list[tuple[_StagedArtifact, Path | None]] = []
    try:
        for artifact in artifacts:
            backup = _backup_path(artifact.target)
            _recover_backup(artifact.target)
            if artifact.target.exists():
                os.replace(artifact.target, backup)
                _fsync_directory(artifact.target.parent)
                published.append((artifact, backup))
            else:
                published.append((artifact, None))
            _replace_staged(artifact.partial, artifact.target)
            _fsync_directory(artifact.target.parent)
        for _, backup in published:
            if backup is not None and backup.exists():
                backup.unlink()
                _fsync_directory(backup.parent)
    except Exception:
        for artifact, backup in reversed(published):
            if artifact.target.exists():
                artifact.target.unlink()
            if backup is not None and backup.exists():
                os.replace(backup, artifact.target)
            _fsync_directory(artifact.target.parent)
        raise
    finally:
        for artifact in artifacts:
            _remove_partial(artifact.partial)


def _replace_staged(source: Path, destination: Path) -> None:
    """Indirection keeps replacement failure handling focused and injectable in tests."""

    os.replace(source, destination)


def _backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.rollback")


def _recover_backup(path: Path) -> None:
    """Recover a pre-rename backup left by an interrupted prior transaction."""

    backup = _backup_path(path)
    if not backup.exists():
        return
    if path.exists():
        backup.unlink()
    else:
        os.replace(backup, path)
    _fsync_directory(path.parent)


def _verify_phase_table(table: pa.Table, expected_rows: list[dict[str, object]] | None) -> None:
    if table.column_names != _PHASE_COLUMNS or (expected_rows is not None and table.num_rows != len(expected_rows)):
        raise CacheIntegrityError("published phase Parquet schema or row count is invalid")
    tokens = table.column("speech_token").to_pylist()
    lengths = table.column("speech_token_len").to_pylist()
    for values, length in zip(tokens, lengths, strict=True):
        if not isinstance(values, list) or len(values) != length:
            raise CacheIntegrityError("published phase Parquet token length is invalid")
        _validated_tokens(values, "published Parquet")


def _verify_prompt_table(table: pa.Table) -> None:
    if table.column_names != _PROMPT_COLUMNS:
        raise CacheIntegrityError("published prompt Parquet schema is invalid")
    names = table.column("audio_path").to_pylist()
    if len(names) != len(set(names)):
        raise CacheIntegrityError("published prompt Parquet has duplicate WAV identities")


def _verify_phase_metadata(root: Path, metadata: Mapping[str, object], phase: int) -> int:
    relative = Path(_required_string(metadata, "path"))
    if relative.is_absolute() or relative.parts[:1] != (f"phase{phase}",):
        raise CacheIntegrityError("phase Parquet path is invalid")
    path = root / relative
    if not path.is_file() or sha256_file(path) != _required_string(metadata, "sha256"):
        raise CacheIntegrityError(f"phase Parquet checksum changed: {path}")
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise CacheIntegrityError(f"cannot reopen phase Parquet: {path}") from exc
    _verify_phase_table(table, None)
    row_count = _required_int(metadata, "row_count")
    if table.num_rows != row_count:
        raise CacheIntegrityError(f"phase Parquet row count changed: {path}")
    tokens = [token for values in table.column("speech_token").to_pylist() for token in values]
    token_min = min(tokens) if tokens else None
    token_max = max(tokens) if tokens else None
    if metadata.get("token_min") != token_min or metadata.get("token_max") != token_max:
        raise CacheIntegrityError(f"phase Parquet token range changed: {path}")
    return row_count


def _verify_prompt_parquet(root: Path, prompts: list[dict[str, object]]) -> None:
    path = root / "eval_prompts.parquet"
    if not path.is_file():
        raise CacheIntegrityError("missing eval_prompts.parquet")
    table = pq.read_table(path)
    _verify_prompt_table(table)
    if table.num_rows != len(prompts):
        raise CacheIntegrityError("eval prompt row count changed")
    actual = table.to_pylist()
    expected = sorted(prompts, key=lambda prompt: _required_string(prompt, "audio_path"))
    if actual != expected:
        raise CacheIntegrityError("eval prompt metadata changed")


def _verify_prompt_set(root: Path, prompts: list[dict[str, object]]) -> None:
    if len(prompts) != PROMPT_RESERVATION_COUNT:
        raise CacheIntegrityError(f"cache must contain exactly {PROMPT_RESERVATION_COUNT} aggregate prompt records")
    paths = [_required_string(prompt, "audio_path") for prompt in prompts]
    indices: list[int] = []
    for path in paths:
        match = _PROMPT_WAV.fullmatch(path)
        if match is None:
            raise CacheIntegrityError(f"invalid prompt WAV path: {path}")
        indices.append(int(match.group(1)))
    if len(set(paths)) != PROMPT_RESERVATION_COUNT or len(set(indices)) != PROMPT_RESERVATION_COUNT:
        raise CacheIntegrityError("cache prompt records must have unique indices and WAV paths")
    expected_paths = {f"eval_prompts/voice_{index:02d}.wav" for index in range(PROMPT_RESERVATION_COUNT)}
    if set(paths) != expected_paths:
        raise CacheIntegrityError(f"cache must contain exactly {PROMPT_RESERVATION_COUNT} prompt WAVs with indices 00-19")
    wav_paths = {
        f"eval_prompts/{path.name}"
        for path in (root / "eval_prompts").glob("*.wav")
        if path.is_file()
    }
    if wav_paths != expected_paths:
        raise CacheIntegrityError(f"cache must contain exactly {PROMPT_RESERVATION_COUNT} prompt WAV files")


def _verify_input_checksum(data: Mapping[str, object], path_key: str, checksum_key: str) -> None:
    path = Path(_required_string(data, path_key))
    if not path.is_file() or sha256_file(path) != _required_string(data, checksum_key):
        raise CacheIntegrityError(f"{path_key} checksum changed: {path}")


def _count_plan_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise CacheIntegrityError(f"cannot read split plan: {path}") from exc


def _pcm_wav(audio: AudioInput) -> bytes:
    values = _sample_values(audio.samples)
    if len(values) != audio.frames:
        raise CacheIntegrityError("decoded audio frame count does not match samples")
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, round(float(value) * 32767)))) for value in values)
    destination = io.BytesIO()
    with wave.open(destination, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)
    return destination.getvalue()


def _sample_values(samples: Any) -> list[float]:
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().reshape(-1).tolist()
    elif hasattr(samples, "reshape") and hasattr(samples, "tolist"):
        samples = samples.reshape(-1).tolist()
    if not isinstance(samples, list):
        raise CacheIntegrityError("decoded audio samples are not a sequence")
    try:
        return [float(sample) for sample in samples]
    except (TypeError, ValueError) as exc:
        raise CacheIntegrityError("decoded audio samples are not numeric") from exc


def _tar_json(payload: bytes, member_name: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"invalid tar JSON: {member_name}") from exc
    if not isinstance(value, dict):
        raise CacheIntegrityError(f"tar JSON must be an object: {member_name}")
    return value


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise CacheIntegrityError(f"JSON artifact must be an object: {path}")
    return value


def _required_string(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise CacheIntegrityError(f"missing/invalid {field}")
    return item


def _optional_string(value: Mapping[str, object], field: str) -> str | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ValueError(f"invalid {field}")
    return item


def _required_int(value: Mapping[str, object], field: str) -> int:
    return _value_int(value.get(field), field)


def _optional_int(value: Mapping[str, object], field: str, default: int) -> int:
    item = value.get(field, default)
    return _value_int(item, field)


def _value_int(item: object, field: str) -> int:
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"invalid {field}")
    return item


def _remove_partial(path: Path) -> None:
    if path.exists():
        path.unlink()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

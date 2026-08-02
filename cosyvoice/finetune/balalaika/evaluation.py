"""Distributed hard-number synthesis and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import os
import re
import shutil
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
import wave

import numpy as np

from .artifacts import atomic_write_json, sha256_file
from .memorization import require_memorization_gate
from .metrics import (
    NumberSpan,
    _levenshtein_alignment,
    aggregate_scores,
    extract_number_span,
    normalize_asr_text,
    score_row,
)


_VOICE_ID = re.compile(r"voice_(\d{2})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_ROWS = 2_000
_EXPECTED_VOICES = 20
_EXPECTED_WORLD_SIZE = 8
_EXPECTED_LOCAL_ROWS = 250


class EvaluationIntegrityError(RuntimeError):
    """Raised when evaluation inputs or outputs are incomplete or changed."""


class GigaAmError(RuntimeError):
    """Raised when GigaAM cannot provide a qualified CUDA transcription."""


class WandbSyncError(RuntimeError):
    """Raised when validation evidence exists locally but W&B is incomplete."""


class Synthesizer(Protocol):
    def synthesize(self, item: "EvaluationItem", destination: Path) -> None: ...


@dataclass(frozen=True)
class EvaluationItem:
    """One immutable benchmark row paired with one fixed prompt voice."""

    benchmark_id: int
    stressed: str
    normalized_gold: str
    hard_number: str
    category: str
    number_span: NumberSpan
    voice_id: str
    prompt_wav: Path
    prompt_text: str
    prompt_sha256: str


@dataclass(frozen=True)
class EvaluationProvenance:
    checkpoint_sha256: str
    model_state_sha256: str
    adapter_sha256: str
    base_checkpoint_sha256: str
    benchmark_snapshot_sha256: str
    benchmark_revision: str
    asr_config: Mapping[str, object]
    synthesis_config: Mapping[str, object]
    code_version: str
    config_version: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "model_state_sha256",
            "adapter_sha256",
            "base_checkpoint_sha256",
            "benchmark_snapshot_sha256",
        ):
            if not isinstance(value := getattr(self, name), str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("benchmark_revision", "code_version", "config_version"):
            if not isinstance(value := getattr(self, name), str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("asr_config", "synthesis_config"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a nonempty mapping")
            object.__setattr__(self, name, _freeze_json(value))


@dataclass(frozen=True)
class EvaluationIdentity:
    payload: Mapping[str, object]
    sha256: str


def build_evaluation_identity(
    validation_index: int,
    items: Sequence[EvaluationItem],
    provenance: EvaluationProvenance,
    *,
    asr_batch_size: int = 8,
) -> EvaluationIdentity:
    """Bind one checkpoint/index to every benchmark, voice, ASR, and inference input."""

    if isinstance(validation_index, bool) or validation_index not in range(41):
        raise ValueError("validation_index must be in 0 through 40")
    if not isinstance(provenance, EvaluationProvenance):
        raise TypeError("provenance must be EvaluationProvenance")
    if isinstance(asr_batch_size, bool) or not isinstance(asr_batch_size, int) or asr_batch_size < 1:
        raise ValueError("asr_batch_size must be a positive integer")
    semantic = _semantic_assignment(items)
    payload = {
        "format_version": 1,
        "validation_index": validation_index,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "model_state_sha256": provenance.model_state_sha256,
        "adapter_sha256": provenance.adapter_sha256,
        "base_checkpoint_sha256": provenance.base_checkpoint_sha256,
        "benchmark_snapshot_sha256": provenance.benchmark_snapshot_sha256,
        "benchmark_revision": provenance.benchmark_revision,
        "semantic_assignment_sha256": _canonical_sha256(semantic),
        "asr_batch_size": asr_batch_size,
        "asr_config": _thaw_json(provenance.asr_config),
        "synthesis_config": _thaw_json(provenance.synthesis_config),
        "code_version": provenance.code_version,
        "config_version": provenance.config_version,
    }
    return EvaluationIdentity(_freeze_json(payload), _canonical_sha256(payload))


@dataclass(frozen=True)
class EvaluationRequest:
    """Complete dependencies and durable destinations for one validation point."""

    rows: Sequence[object]
    prompts: Sequence[Mapping[str, object]]
    accelerator: Any
    synthesizer: Synthesizer
    recognizer: GigaAmRecognizer
    provenance: EvaluationProvenance
    validation_index: int
    output_jsonl: Path
    summary_json: Path
    panel_dir: Path
    temporary_audio_dir: Path
    assignment_manifest: Path
    memorization_path: Path
    memorization_expected_provenance: Mapping[str, object] | None = None
    wandb_logger: Any | None = None
    asr_batch_size: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.validation_index, bool) or self.validation_index not in range(41):
            raise ValueError("validation_index must be in 0 through 40")
        if isinstance(self.asr_batch_size, bool) or self.asr_batch_size < 1:
            raise ValueError("asr_batch_size must be positive")
        if not isinstance(self.provenance, EvaluationProvenance):
            raise TypeError("provenance must be EvaluationProvenance")
        paths = (
            self.output_jsonl,
            self.summary_json,
            self.panel_dir,
            self.temporary_audio_dir,
            self.assignment_manifest,
            self.memorization_path,
        )
        if any(not isinstance(path, Path) for path in paths):
            raise TypeError("evaluation artifact destinations must be pathlib.Path values")


@dataclass(frozen=True)
class EvaluationReport:
    """Published local audit source of truth for one validation index."""

    validation_index: int
    output_jsonl: Path
    summary_json: Path
    panel_dir: Path
    assignment_checksum: str
    identity_checksum: str
    artifact_checksums: Mapping[str, str]
    metrics: Mapping[str, object]
    row_count: int
    worst_errors: tuple[Mapping[str, object], ...]
    panel_audio: Mapping[str, Path]
    generation_latency_seconds: float
    asr_latency_seconds: float


@dataclass(frozen=True)
class FinalValidationEvidence:
    """Validated Task 10 publication identity for final-export consumers."""

    identity_sha256: str
    artifact_checksums: Mapping[str, str]
    checkpoint_sha256: str
    model_state_sha256: str
    wandb: "WandbCommitEvidence"


@dataclass(frozen=True)
class WandbCommitEvidence:
    run_id: str
    context: Mapping[str, object]
    marker: str
    run_manifest_sha256: str
    ledger_path: Path
    ledger_sha256: str
    remote_markers: Mapping[str, object]


def verify_final_validation_evidence(
    request: EvaluationRequest,
    *,
    expected_checkpoint_sha256: str,
    expected_model_state_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_adapter_sha256: str,
    wandb_logger: "WandbValidationLogger",
) -> FinalValidationEvidence:
    """Fail closed on any changed Task 10 validation-40 publication or ledger."""

    if not isinstance(request, EvaluationRequest) or request.validation_index != 40:
        raise EvaluationIntegrityError("final export requires Task 10 validation index 40")
    items = build_voice_assignment(request.rows, request.prompts)
    identity = build_evaluation_identity(request.validation_index, items, request.provenance, asr_batch_size=request.asr_batch_size)
    expected = {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "model_state_sha256": expected_model_state_sha256,
        "base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "adapter_sha256": expected_adapter_sha256,
    }
    if any(identity.payload.get(name) != value for name, value in expected.items()):
        raise EvaluationIntegrityError("Task 10 evaluation identity differs from final export lineage")
    report = _require_published_report(request, items, identity, _canonical_sha256(_semantic_assignment(items)))
    if not isinstance(wandb_logger, WandbValidationLogger):
        raise TypeError("final validation evidence requires WandbValidationLogger")
    wandb = wandb_logger.verify_committed(report, 40)
    return FinalValidationEvidence(identity.sha256, report.artifact_checksums, expected_checkpoint_sha256, expected_model_state_sha256, wandb)


class WandbValidationLogger:
    """Idempotently commit validation evidence to Task 8's existing W&B run."""

    def __init__(
        self,
        accelerator: Any,
        run_manifest: Path,
        *,
        table_factory: Callable[[Sequence[Mapping[str, object]]], object] | None = None,
        audio_factory: Callable[[Path], object] | None = None,
        ledger_writer: Callable[[Path, Mapping[str, object]], None] = atomic_write_json,
        remote_history_reader: Callable[[Any, Sequence[str]], Any] | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.run_manifest = Path(run_manifest)
        self.commit_dir = self.run_manifest.parent / "wandb-validation-commits"
        self._table_factory = table_factory or _wandb_table
        self._audio_factory = audio_factory or _wandb_audio
        self._ledger_writer = ledger_writer
        self._remote_history_reader = remote_history_reader or _wandb_api_history
        self._owned_run_id: str | None = None

    @staticmethod
    def resume_init_kwargs(run_manifest: Path) -> dict[str, dict[str, str]]:
        """Return the only approved later-launch Accelerate tracker init kwargs."""

        manifest = _read_wandb_manifest(Path(run_manifest))
        return {"wandb": {"id": manifest["run_id"], "resume": "must"}}

    def log(self, report: EvaluationReport, validation_index: int) -> None:
        """Log media then scalars once, recording each successful durable phase."""

        run = self.preflight(validation_index)
        if getattr(report, "validation_index", None) != validation_index:
            raise WandbSyncError("W&B validation index disagrees with the evaluation report")
        run_id = run.id
        output = getattr(report, "output_jsonl", None)
        if not isinstance(output, Path) or not output.is_file():
            raise WandbSyncError("local validation JSONL must exist before W&B logging")
        context = _wandb_commit_context(report)
        marker = _canonical_sha256(context)
        scalar_marker = f"validation/commit/{validation_index:02d}/scalars"
        media_marker = f"validation/commit/{validation_index:02d}/media"
        commit_path = self.commit_dir / f"validation-{validation_index:02d}.json"
        state = _read_wandb_commit(commit_path, validation_index, run_id, context, marker)
        sync_dir = getattr(run, "dir", self.run_manifest.parent)
        try:
            remote = _wandb_remote_markers(
                run, scalar_marker, media_marker, self._remote_history_reader
            )
            if any(value is not None and value != marker for value in remote.values()):
                raise WandbSyncError("W&B remote validation marker changed")
            if (remote[scalar_marker] is None) != (remote[media_marker] is None):
                raise WandbSyncError("W&B remote validation markers are incomplete")
            if remote[scalar_marker] == marker and remote[media_marker] == marker:
                if not state["committed"]:
                    state.update({"media_logged": True, "scalars_logged": True, "committed": True})
                    self._ledger_writer(commit_path, state)
                return
            if state["committed"]:
                raise WandbSyncError("local W&B commit lacks its remote marker")
            if remote[media_marker] is None:
                media = {"worst-errors": self._table_factory(report.worst_errors)}
                media.update(
                    {
                        f"listening-panel/{voice_id}": self._audio_factory(path)
                        for voice_id, path in sorted(report.panel_audio.items())
                    }
                )
                media[media_marker] = marker
                run.log(media, step=validation_index, commit=False)
            before_scalars = _wandb_remote_markers(
                run, scalar_marker, media_marker, self._remote_history_reader
            )
            if before_scalars != {scalar_marker: None, media_marker: None}:
                raise WandbSyncError("W&B remote markers changed before scalar logging")
            scalars = _wandb_scalars(report)
            scalars[scalar_marker] = marker
            self.accelerator.log(
                scalars,
                step=validation_index,
                log_kwargs={"wandb": {"commit": True}},
            )
            remote = _wandb_remote_markers(
                run, scalar_marker, media_marker, self._remote_history_reader
            )
            if remote != {scalar_marker: marker, media_marker: marker}:
                raise WandbSyncError("W&B remote validation commit could not be verified")
            state.update({"media_logged": True, "scalars_logged": True, "committed": True})
            self._ledger_writer(commit_path, state)
        except Exception as exc:
            raise WandbSyncError(
                f"W&B validation {validation_index} is incomplete; syncable local run preserved at {sync_dir}"
            ) from exc

    def verify_committed(self, report: EvaluationReport, validation_index: int) -> WandbCommitEvidence:
        """Re-authenticate local and remote W&B completion evidence."""

        run = self.preflight(validation_index)
        if getattr(report, "validation_index", None) != validation_index:
            raise WandbSyncError("W&B validation index disagrees with the evaluation report")
        context = _wandb_commit_context(report)
        marker = _canonical_sha256(context)
        manifest = _read_wandb_manifest(self.run_manifest)
        run_id = getattr(run, "id", None)
        if run_id != manifest["run_id"]:
            raise WandbSyncError("W&B active run differs from the persisted run manifest")
        ledger_path = self.commit_dir / f"validation-{validation_index:02d}.json"
        ledger = _read_wandb_commit(ledger_path, validation_index, run_id, context, marker)
        if (ledger["media_logged"], ledger["scalars_logged"], ledger["committed"]) != (True, True, True):
            raise WandbSyncError("W&B validation ledger is incomplete")
        scalar = f"validation/commit/{validation_index:02d}/scalars"
        media = f"validation/commit/{validation_index:02d}/media"
        remote = _wandb_remote_markers(run, scalar, media, self._remote_history_reader)
        if remote != {scalar: marker, media: marker}:
            raise WandbSyncError("W&B remote validation markers are missing or changed")
        return WandbCommitEvidence(run_id, context, marker, sha256_file(self.run_manifest), ledger_path, sha256_file(ledger_path), remote)

    def preflight(self, validation_index: int) -> Any:
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise WandbSyncError("WANDB_API_KEY is required in the environment")
        if isinstance(validation_index, bool) or validation_index not in range(41):
            raise WandbSyncError("W&B validation index must be in 0 through 40")
        try:
            run = self.accelerator.get_tracker("wandb", unwrap=True)
        except Exception as exc:
            raise WandbSyncError("Accelerate has no initialized W&B tracker") from exc
        run_id = getattr(run, "id", None)
        if not isinstance(run_id, str) or not run_id:
            raise WandbSyncError("Accelerate W&B tracker has no run ID")
        self._bind_existing_run(run, run_id, validation_index)
        return run

    def _bind_existing_run(self, run: Any, run_id: str, validation_index: int) -> None:
        if self.run_manifest.is_file():
            manifest = _read_wandb_manifest(self.run_manifest)
            if manifest["run_id"] != run_id:
                raise WandbSyncError("Accelerate W&B run ID changed")
            later_launch = self._owned_run_id is None and validation_index > 0
            settings = getattr(run, "settings", None)
            resume_mode = getattr(settings, "resume", None)
            if later_launch and getattr(run, "resumed", False) is not True and resume_mode != "must":
                raise WandbSyncError("later W&B launches require resume='must'")
        else:
            if validation_index != 0:
                raise WandbSyncError("W&B baseline run manifest is missing")
            atomic_write_json(self.run_manifest, {"format_version": 1, "run_id": run_id})
        self._owned_run_id = run_id


class GigaAmRecognizer:
    """Bounded GigaAM v3 RNN-T recognition on exactly one local CUDA rank."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        local_rank: int | None = None,
        max_batch_size: int = 8,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("local_rank must be a non-negative integer")
        if model is None:
            if model_loader is None:
                from onnx_asr import load_model

                model_loader = load_model
            try:
                model = model_loader(
                    "gigaam-v3-rnnt",
                    providers=[("CUDAExecutionProvider", {"device_id": rank})],
                )
            except Exception as exc:
                raise GigaAmError("unable to load gigaam-v3-rnnt on CUDA") from exc
        _require_gigaam_cuda(model, rank)
        self.model = model
        self.local_rank = rank
        self.max_batch_size = max_batch_size

    def transcribe(self, paths: Sequence[Path]) -> list[str]:
        """Recognize paths in bounded batches, bisecting only CUDA-OOM batches."""

        results: list[str] = []
        values = [Path(path) for path in paths]
        for offset in range(0, len(values), self.max_batch_size):
            results.extend(self._transcribe_batch(values[offset : offset + self.max_batch_size]))
        return results

    def provenance(self) -> Mapping[str, object]:
        return {
            "model": "gigaam-v3-rnnt",
            "provider": "CUDAExecutionProvider",
            "device_id": self.local_rank,
            "max_batch_size": self.max_batch_size,
        }

    def _transcribe_batch(self, paths: list[Path]) -> list[str]:
        try:
            values = self.model.recognize(paths)
            if not isinstance(values, list) or len(values) != len(paths) or any(not isinstance(value, str) for value in values):
                raise GigaAmError("GigaAM returned an invalid transcription batch")
            return values
        except Exception as exc:
            if not _is_cuda_oom(exc):
                if isinstance(exc, GigaAmError):
                    raise
                raise GigaAmError("GigaAM v3 RNN-T transcription failed") from exc
            if len(paths) == 1:
                raise GigaAmError(f"GigaAM CUDA OOM for one item: {paths[0]}") from exc
            middle = len(paths) // 2
            return self._transcribe_batch(paths[:middle]) + self._transcribe_batch(paths[middle:])


class CosyVoiceSynthesizer:
    """Voice-clone with the current LoRA LLM and the base frozen flow/HiFT."""

    def __init__(self, pipeline: Any, current_llm: Any, accelerator: Any) -> None:
        if getattr(pipeline, "sample_rate", None) != 24_000:
            raise EvaluationIntegrityError("CosyVoice evaluation pipeline must synthesize at 24 kHz")
        model = getattr(pipeline, "model", None)
        if model is None:
            raise EvaluationIntegrityError("CosyVoice evaluation pipeline has no full model")
        for name in ("flow", "hift"):
            component = getattr(model, name, None)
            parameters = getattr(component, "parameters", None)
            if component is None or not callable(parameters):
                raise EvaluationIntegrityError(f"CosyVoice evaluation pipeline has no frozen {name}")
            if any(getattr(parameter, "requires_grad", False) for parameter in parameters()):
                raise EvaluationIntegrityError(f"CosyVoice evaluation {name} must remain frozen")
        unwrap = getattr(accelerator, "unwrap_model", None)
        if not callable(unwrap):
            raise EvaluationIntegrityError("Accelerator cannot unwrap the current LoRA LLM")
        self.current_llm = unwrap(current_llm)
        model.llm = self.current_llm
        self.pipeline = pipeline
        if not callable(getattr(pipeline, "add_zero_shot_spk", None)):
            raise EvaluationIntegrityError("CosyVoice pipeline cannot register fixed zero-shot voices")
        self._registered_voices: set[str] = set()

    def provenance(self) -> Mapping[str, object]:
        return {"method": "inference_zero_shot", "sample_rate": 24_000, "stream": False}

    def synthesize(self, item: EvaluationItem, destination: Path) -> None:
        """Generate one non-streaming zero-shot utterance without changing train mode."""

        was_training = getattr(self.current_llm, "training", None)
        evaluator = getattr(self.current_llm, "eval", None)
        trainer = getattr(self.current_llm, "train", None)
        if callable(evaluator):
            evaluator()
        chunks: list[np.ndarray] = []
        try:
            speaker_id = f"validation:{item.voice_id}"
            if speaker_id not in self._registered_voices:
                registered = self.pipeline.add_zero_shot_spk(
                    item.prompt_text,
                    str(item.prompt_wav),
                    speaker_id,
                )
                if registered is not True:
                    raise EvaluationIntegrityError(f"CosyVoice could not register {item.voice_id}")
                self._registered_voices.add(speaker_id)
            outputs = self.pipeline.inference_zero_shot(
                item.stressed,
                item.prompt_text,
                str(item.prompt_wav),
                zero_shot_spk_id=speaker_id,
                stream=False,
            )
            for output in outputs:
                speech = output.get("tts_speech") if isinstance(output, Mapping) else None
                if speech is None:
                    raise EvaluationIntegrityError("CosyVoice synthesis output lacks tts_speech")
                if hasattr(speech, "detach"):
                    speech = speech.detach().cpu().numpy()
                samples = np.asarray(speech, dtype=np.float32).reshape(-1)
                if samples.size < 1 or not np.isfinite(samples).all():
                    raise EvaluationIntegrityError("CosyVoice synthesis emitted invalid samples")
                chunks.append(samples)
        finally:
            if isinstance(was_training, bool) and callable(trainer):
                trainer(was_training)
        if not chunks:
            raise EvaluationIntegrityError("CosyVoice synthesis emitted no audio")
        _write_pcm_24khz_mono(destination, np.concatenate(chunks))


def build_voice_assignment(
    rows: Sequence[object], prompts: Sequence[Mapping[str, object]]
) -> list[EvaluationItem]:
    """Build the stable benchmark-to-prompt assignment."""

    benchmark = sorted((_benchmark_fields(row) for row in rows), key=lambda row: row["id"])
    voices = sorted((_prompt_fields(prompt) for prompt in prompts), key=lambda prompt: prompt["voice_id"])
    if len(benchmark) != 2_000:
        raise EvaluationIntegrityError("evaluation requires exactly 2000 benchmark rows")
    if {row["id"] for row in benchmark} != set(range(1, 2_001)):
        raise EvaluationIntegrityError("benchmark IDs must be unique and cover 1 through 2000")
    if len(voices) != 20:
        raise EvaluationIntegrityError("evaluation requires exactly 20 prompt voices")
    expected_voices = {f"voice_{index:02d}" for index in range(20)}
    if {prompt["voice_id"] for prompt in voices} != expected_voices:
        raise EvaluationIntegrityError("prompt voice IDs must be unique and cover voice_00 through voice_19")
    return [
        EvaluationItem(
            benchmark_id=row["id"],
            stressed=row["stressed"],
            normalized_gold=row["normalized_gold"],
            hard_number=row["hard_number"],
            category=row["category"],
            number_span=row["number_span"],
            voice_id=prompt["voice_id"],
            prompt_wav=prompt["audio_path"],
            prompt_text=prompt["text"],
            prompt_sha256=prompt["wav_sha256"],
        )
        for index, row in enumerate(benchmark)
        for prompt in (voices[index % 20],)
    ]


def evaluate_checkpoint(request: EvaluationRequest) -> EvaluationReport:
    """Generate, transcribe, publish, and log one restart-safe validation point."""

    require_memorization_gate(
        request.memorization_path,
        expected_provenance=request.memorization_expected_provenance,
    )
    accelerator = request.accelerator
    if getattr(accelerator, "num_processes", None) != _EXPECTED_WORLD_SIZE:
        raise EvaluationIntegrityError("production evaluation requires exactly eight Accelerate ranks")
    _collective_wandb_preflight(request)
    local_process_index = getattr(accelerator, "local_process_index", None)
    if getattr(request.recognizer, "local_rank", None) != local_process_index:
        raise EvaluationIntegrityError("GigaAM local rank must match Accelerator local_process_index")
    _require_component_provenance(request)
    items = build_voice_assignment(request.rows, request.prompts)
    assignment_checksum = _ensure_assignment_manifest(request, items)
    identity = build_evaluation_identity(
        request.validation_index,
        items,
        request.provenance,
        asr_batch_size=request.asr_batch_size,
    )
    accelerator.wait_for_everyone()

    report = _load_published_report(request, items, identity, assignment_checksum)
    if report is None:
        report = _generate_and_publish(request, items, identity, assignment_checksum)
    accelerator.wait_for_everyone()
    if report is None:
        report = _require_published_report(request, items, identity, assignment_checksum)

    logging_error: str | None = None
    if getattr(accelerator, "is_main_process", False):
        try:
            if not isinstance(request.wandb_logger, WandbValidationLogger):
                raise WandbSyncError("main process lost its W&B validation logger")
            request.wandb_logger.log(report, request.validation_index)
        except Exception as exc:
            logging_error = str(exc)
    logging_status = _broadcast_main_status(accelerator, "wandb-log", logging_error)
    if not logging_status["ok"]:
        raise WandbSyncError(logging_status["error"])
    return report


def _generate_and_publish(
    request: EvaluationRequest,
    items: Sequence[EvaluationItem],
    identity: EvaluationIdentity,
    assignment_checksum: str,
) -> EvaluationReport | None:
    accelerator = request.accelerator
    with accelerator.split_between_processes(list(items)) as shard:
        local_items = list(shard)
    if len(local_items) != _EXPECTED_LOCAL_ROWS:
        raise EvaluationIntegrityError(
            f"world size eight requires exactly 250 local evaluation items, got {len(local_items)}"
        )
    expected_local_ids = {item.benchmark_id for item in local_items}
    if len(expected_local_ids) != _EXPECTED_LOCAL_ROWS:
        raise EvaluationIntegrityError("local evaluation shard contains duplicate benchmark IDs")

    rank = getattr(accelerator, "process_index", None)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(_EXPECTED_WORLD_SIZE):
        raise EvaluationIntegrityError("Accelerate process_index must identify one of eight ranks")
    rank_root = request.temporary_audio_dir / f"rank-{rank:02d}"
    journal_path = rank_root / "records.jsonl"
    existing = _load_rank_journal(journal_path, local_items, identity)
    pending: list[tuple[EvaluationItem, Path, float]] = []
    for item in local_items:
        if item.benchmark_id in existing:
            continue
        destination = rank_root / f"benchmark-{item.benchmark_id:04d}.wav"
        generation_latency = 0.0
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        started = time.perf_counter()
        try:
            request.synthesizer.synthesize(item, destination)
        except Exception as exc:
            raise EvaluationIntegrityError(f"synthesis failed for benchmark {item.benchmark_id}") from exc
        generation_latency = time.perf_counter() - started
        if not _is_pcm_24khz_mono(destination):
            raise EvaluationIntegrityError(
                f"synthesizer did not write 24 kHz mono PCM WAV for benchmark {item.benchmark_id}"
            )
        pending.append((item, destination, generation_latency))

    for offset in range(0, len(pending), request.asr_batch_size):
        batch = pending[offset : offset + request.asr_batch_size]
        started = time.perf_counter()
        hypotheses = request.recognizer.transcribe([path for _, path, _ in batch])
        elapsed = time.perf_counter() - started
        if len(hypotheses) != len(batch):
            raise EvaluationIntegrityError("ASR batch did not return one hypothesis per generated item")
        latency = elapsed / len(batch)
        for (item, path, generation_latency), hypothesis in zip(batch, hypotheses, strict=True):
            if not isinstance(hypothesis, str):
                raise EvaluationIntegrityError("ASR hypotheses must be strings")
            existing[item.benchmark_id] = {
                "benchmark_id": item.benchmark_id,
                "evaluation_identity_sha256": identity.sha256,
                "item_sha256": _item_sha256(item),
                "hypothesis": hypothesis,
                "audio_path": str(path),
                "audio_sha256": sha256_file(path),
                "generation_latency_seconds": generation_latency,
                "asr_latency_seconds": latency,
            }
            existing[item.benchmark_id]["record_sha256"] = _journal_record_sha256(existing[item.benchmark_id])
        _atomic_write_jsonl(journal_path, [existing[key] for key in sorted(existing)])

    if set(existing) != expected_local_ids:
        raise EvaluationIntegrityError("local evaluation journal is missing benchmark IDs")
    gathered = accelerator.gather_object([existing[key] for key in sorted(existing)])
    gathered_records = _flatten_gathered_records(gathered)
    if not getattr(accelerator, "is_main_process", False):
        return None
    report = _publish_report(request, items, gathered_records, identity, assignment_checksum)
    if os.environ.get("KEEP_EVAL_AUDIO") != "1":
        _remove_evaluation_audio(request.temporary_audio_dir)
    return report


def _publish_report(
    request: EvaluationRequest,
    items: Sequence[EvaluationItem],
    gathered: Sequence[Mapping[str, object]],
    identity: EvaluationIdentity,
    assignment_checksum: str,
) -> EvaluationReport:
    by_id: dict[int, Mapping[str, object]] = {}
    item_by_id = {item.benchmark_id: item for item in items}
    for record in gathered:
        identifier = record.get("benchmark_id")
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier in by_id:
            raise EvaluationIntegrityError("gathered evaluation records contain missing or duplicate IDs")
        item = item_by_id.get(identifier)
        if item is None or not _valid_gathered_record(record, item, identity):
            raise EvaluationIntegrityError(f"gathered evaluation record/audio is invalid for benchmark {identifier}")
        by_id[identifier] = record
    if set(by_id) != set(range(1, _EXPECTED_ROWS + 1)):
        raise EvaluationIntegrityError("gathered evaluation records must cover exactly IDs 1 through 2000")

    published: list[dict[str, object]] = []
    scored = []
    for identifier in range(1, _EXPECTED_ROWS + 1):
        item = item_by_id[identifier]
        raw = by_id[identifier]
        hypothesis = raw.get("hypothesis")
        audio_path = raw.get("audio_path")
        if not isinstance(hypothesis, str) or not isinstance(audio_path, str):
            raise EvaluationIntegrityError(f"gathered record {identifier} is incomplete")
        scores = score_row(item.normalized_gold, hypothesis, item.number_span)
        scored.append(scores)
        published.append(_published_row(item, raw, hypothesis, scores, identity))

    summary = aggregate_scores(scored)
    metrics = summary.as_dict()
    generation_latency = sum(float(record["generation_latency_seconds"]) for record in published)
    asr_latency = sum(float(record["asr_latency_seconds"]) for record in published)
    generation_latency /= _EXPECTED_ROWS
    asr_latency /= _EXPECTED_ROWS
    worst = tuple(
        sorted(
            published,
            key=lambda record: (
                -int(record["edit_counts"]["utterance"]["word"]["distance"]),
                -int(record["edit_counts"]["number"]["word"]["distance"]),
                int(record["benchmark_id"]),
            ),
        )[:20]
    )
    _atomic_write_jsonl(request.output_jsonl, published)
    if request.panel_dir.exists():
        shutil.rmtree(request.panel_dir)
    panel_audio = _publish_listening_panel(request.panel_dir, items, by_id, identity)
    panel_manifest = request.panel_dir / "manifest.json"
    summary_payload = {
        "format_version": 2,
        "validation_index": request.validation_index,
        "row_count": _EXPECTED_ROWS,
        "assignment_sha256": assignment_checksum,
        "evaluation_identity": _thaw_json(identity.payload),
        "evaluation_identity_sha256": identity.sha256,
        "results": str(request.output_jsonl),
        "results_sha256": sha256_file(request.output_jsonl),
        "panel_manifest": str(panel_manifest),
        "panel_manifest_sha256": sha256_file(panel_manifest),
        "metrics": metrics,
        "diagnostics": {
            "generation_latency_seconds": generation_latency,
            "asr_latency_seconds": asr_latency,
        },
        "worst_benchmark_ids": [record["benchmark_id"] for record in worst],
        "listening_panel": {voice: str(path) for voice, path in panel_audio.items()},
    }
    atomic_write_json(request.summary_json, summary_payload)
    seal_path = _validation_seal_path(request)
    atomic_write_json(
        seal_path,
        {
            "format_version": 1,
            "evaluation_identity_sha256": identity.sha256,
            "artifacts": {
                "results_jsonl": sha256_file(request.output_jsonl),
                "summary_json": sha256_file(request.summary_json),
                "panel_manifest": sha256_file(panel_manifest),
            },
        },
    )
    artifact_checksums = {
        "results_jsonl": sha256_file(request.output_jsonl),
        "summary_json": sha256_file(request.summary_json),
        "panel_manifest": sha256_file(panel_manifest),
        "validation_seal": sha256_file(seal_path),
    }
    return EvaluationReport(
        validation_index=request.validation_index,
        output_jsonl=request.output_jsonl,
        summary_json=request.summary_json,
        panel_dir=request.panel_dir,
        assignment_checksum=assignment_checksum,
        identity_checksum=identity.sha256,
        artifact_checksums=artifact_checksums,
        metrics=metrics,
        row_count=_EXPECTED_ROWS,
        worst_errors=worst,
        panel_audio=panel_audio,
        generation_latency_seconds=generation_latency,
        asr_latency_seconds=asr_latency,
    )


def _published_row(
    item: EvaluationItem,
    raw: Mapping[str, object],
    hypothesis: str,
    scores: Any,
    identity: EvaluationIdentity,
) -> dict[str, object]:
    reference_tokens = normalize_asr_text(item.normalized_gold).split()
    hypothesis_tokens = normalize_asr_text(hypothesis).split()
    alignment = _levenshtein_alignment(reference_tokens, hypothesis_tokens)
    indices = [
        hypothesis_index
        for reference_index, hypothesis_index, _ in alignment
        if reference_index is not None
        and item.number_span.reference_start <= reference_index < item.number_span.reference_end
        and hypothesis_index is not None
    ]
    hypothesis_start = min(indices) if indices else None
    hypothesis_end = max(indices) + 1 if indices else None
    hypothesis_number = "" if not indices else " ".join(hypothesis_tokens[hypothesis_start:hypothesis_end])
    row = {
        "benchmark_id": item.benchmark_id,
        "evaluation_identity_sha256": identity.sha256,
        "voice_id": item.voice_id,
        "prompt_wav": str(item.prompt_wav),
        "prompt_text": item.prompt_text,
        "prompt_sha256": item.prompt_sha256,
        "stressed": item.stressed,
        "normalized_gold": item.normalized_gold,
        "hard_number": item.hard_number,
        "hypothesis": hypothesis,
        "category": item.category,
        "number_span": {
            "reference_start": item.number_span.reference_start,
            "reference_end": item.number_span.reference_end,
            "reference_text": " ".join(
                reference_tokens[item.number_span.reference_start : item.number_span.reference_end]
            ),
            "hypothesis_start": hypothesis_start,
            "hypothesis_end": hypothesis_end,
            "hypothesis_text": hypothesis_number,
        },
        "edit_counts": _scores_payload(scores),
        "audio_path": raw["audio_path"],
        "audio_sha256": raw.get("audio_sha256"),
        "generation_latency_seconds": float(raw.get("generation_latency_seconds", 0.0)),
        "asr_latency_seconds": float(raw.get("asr_latency_seconds", 0.0)),
    }
    row["row_sha256"] = _canonical_sha256(row)
    return row


def _scores_payload(scores: Any) -> dict[str, object]:
    def region(value: Any) -> dict[str, object]:
        return {
            "word": {**asdict(value.word), "distance": value.word.distance},
            "character": {**asdict(value.character), "distance": value.character.distance},
        }

    return {"utterance": region(scores.utterance), "number": region(scores.number)}


def _ensure_assignment_manifest(request: EvaluationRequest, items: Sequence[EvaluationItem]) -> str:
    entries = _semantic_assignment(items)
    checksum = _canonical_sha256(entries)
    path = request.assignment_manifest
    if getattr(request.accelerator, "is_main_process", False):
        if path.is_file():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise EvaluationIntegrityError("voice assignment manifest is invalid") from exc
            if not isinstance(manifest, Mapping) or manifest != {
                "format_version": 1,
                "assignment_sha256": checksum,
                "rows": _EXPECTED_ROWS,
                "voices": _EXPECTED_VOICES,
                "assignment": entries,
            }:
                raise EvaluationIntegrityError("voice assignment changed after baseline")
        else:
            if request.validation_index != 0:
                raise EvaluationIntegrityError("voice assignment baseline manifest is missing")
            atomic_write_json(
                path,
                {
                    "format_version": 1,
                    "assignment_sha256": checksum,
                    "rows": _EXPECTED_ROWS,
                    "voices": _EXPECTED_VOICES,
                    "assignment": entries,
                },
            )
    return checksum


def _load_rank_journal(
    path: Path,
    expected_items: Sequence[EvaluationItem],
    identity: EvaluationIdentity,
) -> dict[int, dict[str, object]]:
    expected = {item.benchmark_id: item for item in expected_items}
    if not path.is_file():
        return {}
    records: dict[int, dict[str, object]] = {}
    seen_ids: set[int] = set()
    duplicate_ids: set[int] = set()
    invalid = False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            value = json.loads(line)
            identifier = value.get("benchmark_id") if isinstance(value, Mapping) else None
            item = expected.get(identifier) if isinstance(identifier, int) and not isinstance(identifier, bool) else None
            destination = path.parent / f"benchmark-{identifier:04d}.wav" if item is not None else None
            if item is not None and identifier in seen_ids:
                invalid = True
                duplicate_ids.add(identifier)
                records.pop(identifier, None)
                if destination is not None and (destination.exists() or destination.is_symlink()):
                    destination.unlink()
                continue
            if item is not None:
                seen_ids.add(identifier)
            if (
                item is None
                or not _valid_journal_record(value, item, identity, destination)
            ):
                invalid = True
                if destination is not None and (destination.exists() or destination.is_symlink()):
                    destination.unlink()
                continue
            records[identifier] = dict(value)
    except (OSError, json.JSONDecodeError) as exc:
        invalid = True
        records.clear()
        for item in expected_items:
            destination = path.parent / f"benchmark-{item.benchmark_id:04d}.wav"
            if destination.exists() or destination.is_symlink():
                destination.unlink()
    for identifier, record in list(records.items()):
        item = expected[identifier]
        destination = path.parent / f"benchmark-{identifier:04d}.wav"
        if identifier in duplicate_ids or not _valid_journal_record(record, item, identity, destination):
            invalid = True
            records.pop(identifier)
            if destination.exists() or destination.is_symlink():
                destination.unlink()
    if invalid:
        if records:
            _atomic_write_jsonl(path, [records[key] for key in sorted(records)])
        elif path.exists():
            path.unlink()
    return records


def _flatten_gathered_records(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records: list[Mapping[str, object]] = []
        for item in value:
            records.extend(_flatten_gathered_records(item))
        return records
    raise EvaluationIntegrityError("Accelerate gather_object returned invalid evaluation records")


def _publish_listening_panel(
    panel_dir: Path,
    items: Sequence[EvaluationItem],
    records: Mapping[int, Mapping[str, object]],
    identity: EvaluationIdentity,
) -> dict[str, Path]:
    selected: dict[str, EvaluationItem] = {}
    for item in items:
        selected.setdefault(item.voice_id, item)
    if set(selected) != {f"voice_{index:02d}" for index in range(_EXPECTED_VOICES)}:
        raise EvaluationIntegrityError("listening panel cannot select exactly one item per voice")
    if panel_dir.exists():
        raise EvaluationIntegrityError(f"listening panel target already exists: {panel_dir}")
    panel_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=panel_dir.parent, prefix=f".{panel_dir.name}."))
    try:
        manifest_rows = []
        for voice_id, item in sorted(selected.items()):
            source_value = records[item.benchmark_id].get("audio_path")
            source = Path(source_value) if isinstance(source_value, str) else None
            if source is None or not _is_pcm_24khz_mono(source):
                raise EvaluationIntegrityError(f"listening panel audio is missing for {voice_id}")
            destination = temporary / f"{voice_id}.wav"
            shutil.copyfile(source, destination)
            checksum = sha256_file(destination)
            if checksum != records[item.benchmark_id].get("audio_sha256"):
                raise EvaluationIntegrityError(f"listening panel copy checksum changed for {voice_id}")
            manifest_rows.append(
                {
                    "voice_id": voice_id,
                    "benchmark_id": item.benchmark_id,
                    "file": f"{voice_id}.wav",
                    "wav_sha256": checksum,
                }
            )
        index = temporary / "README.md"
        index.write_text(
            "# Deterministic validation listening panel\n\n"
            + "\n".join(
                f"- [{voice_id}]({voice_id}.wav): benchmark {selected[voice_id].benchmark_id}"
                for voice_id in sorted(selected)
            )
            + "\n",
            encoding="utf-8",
        )
        atomic_write_json(
            temporary / "manifest.json",
            {
                "format_version": 1,
                "evaluation_identity_sha256": identity.sha256,
                "items": manifest_rows,
                "readme_sha256": sha256_file(index),
            },
        )
        os.replace(temporary, panel_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {voice: panel_dir / f"{voice}.wav" for voice in sorted(selected)}


def _load_published_report(
    request: EvaluationRequest,
    items: Sequence[EvaluationItem],
    identity: EvaluationIdentity,
    assignment_checksum: str,
) -> EvaluationReport | None:
    seal = _validation_seal_path(request)
    artifacts_exist = any(
        path.exists() for path in (request.output_jsonl, request.summary_json, request.panel_dir, seal)
    )
    if not artifacts_exist or not seal.is_file():
        return None
    return _require_published_report(request, items, identity, assignment_checksum)


def _require_published_report(
    request: EvaluationRequest,
    items: Sequence[EvaluationItem],
    identity: EvaluationIdentity,
    assignment_checksum: str,
) -> EvaluationReport:
    panel_manifest_path = request.panel_dir / "manifest.json"
    seal_path = _validation_seal_path(request)
    required = (request.output_jsonl, request.summary_json, panel_manifest_path, seal_path)
    if any(not path.is_file() for path in required):
        raise EvaluationIntegrityError("committed validation publication is incomplete")
    try:
        summary = json.loads(request.summary_json.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in request.output_jsonl.read_text(encoding="utf-8").splitlines()]
        panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError("published validation results are invalid") from exc
    expected_seal = {
        "format_version": 1,
        "evaluation_identity_sha256": identity.sha256,
        "artifacts": {
            "results_jsonl": sha256_file(request.output_jsonl),
            "summary_json": sha256_file(request.summary_json),
            "panel_manifest": sha256_file(panel_manifest_path),
        },
    }
    if seal != expected_seal:
        raise EvaluationIntegrityError("validation success seal or artifact checksum changed")
    scored, validated_rows = _validate_published_rows(rows, items, identity)
    metrics = aggregate_scores(scored).as_dict()
    generation_latency = sum(row["generation_latency_seconds"] for row in validated_rows) / _EXPECTED_ROWS
    asr_latency = sum(row["asr_latency_seconds"] for row in validated_rows) / _EXPECTED_ROWS
    worst = tuple(
        sorted(
            validated_rows,
            key=lambda record: (
                -int(record["edit_counts"]["utterance"]["word"]["distance"]),
                -int(record["edit_counts"]["number"]["word"]["distance"]),
                int(record["benchmark_id"]),
            ),
        )[:20]
    )
    panel_audio = _validate_panel(request.panel_dir, panel_manifest, items, validated_rows, identity)
    expected_summary = {
        "format_version": 2,
        "validation_index": request.validation_index,
        "row_count": _EXPECTED_ROWS,
        "assignment_sha256": assignment_checksum,
        "evaluation_identity": _thaw_json(identity.payload),
        "evaluation_identity_sha256": identity.sha256,
        "results": str(request.output_jsonl),
        "results_sha256": sha256_file(request.output_jsonl),
        "panel_manifest": str(panel_manifest_path),
        "panel_manifest_sha256": sha256_file(panel_manifest_path),
        "metrics": metrics,
        "diagnostics": {
            "generation_latency_seconds": generation_latency,
            "asr_latency_seconds": asr_latency,
        },
        "worst_benchmark_ids": [record["benchmark_id"] for record in worst],
        "listening_panel": {voice: str(path) for voice, path in panel_audio.items()},
    }
    if summary != expected_summary:
        raise EvaluationIntegrityError("published summary does not match recomputed row metrics and artifacts")
    artifact_checksums = {
        "results_jsonl": sha256_file(request.output_jsonl),
        "summary_json": sha256_file(request.summary_json),
        "panel_manifest": sha256_file(panel_manifest_path),
        "validation_seal": sha256_file(seal_path),
    }
    return EvaluationReport(
        validation_index=request.validation_index,
        output_jsonl=request.output_jsonl,
        summary_json=request.summary_json,
        panel_dir=request.panel_dir,
        assignment_checksum=assignment_checksum,
        identity_checksum=identity.sha256,
        artifact_checksums=artifact_checksums,
        metrics=metrics,
        row_count=_EXPECTED_ROWS,
        worst_errors=worst,
        panel_audio=panel_audio,
        generation_latency_seconds=generation_latency,
        asr_latency_seconds=asr_latency,
    )


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            temporary_name = output.name
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validation_seal_path(request: EvaluationRequest) -> Path:
    return request.summary_json.with_name("validation-success.json")


def _validate_published_rows(
    rows: object,
    items: Sequence[EvaluationItem],
    identity: EvaluationIdentity,
) -> tuple[list[Any], list[dict[str, object]]]:
    if not isinstance(rows, list) or len(rows) != _EXPECTED_ROWS:
        raise EvaluationIntegrityError("published JSONL must contain exactly 2000 rows")
    scored: list[Any] = []
    validated: list[dict[str, object]] = []
    for item, row in zip(items, rows, strict=True):
        if not isinstance(row, dict) or row.get("benchmark_id") != item.benchmark_id:
            raise EvaluationIntegrityError("published JSONL IDs must be exact, ordered, and unique")
        hypothesis = row.get("hypothesis")
        if not isinstance(hypothesis, str):
            raise EvaluationIntegrityError(f"published hypothesis is invalid for benchmark {item.benchmark_id}")
        if not _finite_nonnegative(row.get("generation_latency_seconds")) or not _finite_nonnegative(
            row.get("asr_latency_seconds")
        ):
            raise EvaluationIntegrityError(f"published latency is invalid for benchmark {item.benchmark_id}")
        scores = score_row(item.normalized_gold, hypothesis, item.number_span)
        expected = _published_row(item, row, hypothesis, scores, identity)
        if row != expected:
            raise EvaluationIntegrityError(
                f"published row semantics/edit counts/checksum changed for benchmark {item.benchmark_id}"
            )
        scored.append(scores)
        validated.append(row)
    return scored, validated


def _validate_panel(
    panel_dir: Path,
    manifest: object,
    items: Sequence[EvaluationItem],
    rows: Sequence[Mapping[str, object]],
    identity: EvaluationIdentity,
) -> dict[str, Path]:
    if not isinstance(manifest, dict):
        raise EvaluationIntegrityError("listening panel manifest is invalid")
    selected: dict[str, EvaluationItem] = {}
    for item in items:
        selected.setdefault(item.voice_id, item)
    rows_by_id = {row["benchmark_id"]: row for row in rows}
    readme = panel_dir / "README.md"
    expected_items = [
        {
            "voice_id": voice,
            "benchmark_id": item.benchmark_id,
            "file": f"{voice}.wav",
            "wav_sha256": rows_by_id[item.benchmark_id]["audio_sha256"],
        }
        for voice, item in sorted(selected.items())
    ]
    expected_manifest = {
        "format_version": 1,
        "evaluation_identity_sha256": identity.sha256,
        "items": expected_items,
        "readme_sha256": sha256_file(readme) if readme.is_file() else None,
    }
    if manifest != expected_manifest:
        raise EvaluationIntegrityError("listening panel mapping/identity changed")
    expected_names = {"README.md", "manifest.json"} | {entry["file"] for entry in expected_items}
    actual_names = {path.name for path in panel_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise EvaluationIntegrityError("listening panel files changed")
    panel_audio: dict[str, Path] = {}
    for entry in expected_items:
        path = panel_dir / entry["file"]
        if not _is_pcm_24khz_mono(path) or sha256_file(path) != entry["wav_sha256"]:
            raise EvaluationIntegrityError(f"listening panel WAV changed for {entry['voice_id']}")
        panel_audio[entry["voice_id"]] = path
    return panel_audio


def _is_pcm_24khz_mono(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as source:
            return (
                source.getnchannels() == 1
                and source.getframerate() == 24_000
                and source.getsampwidth() in (1, 2, 3, 4)
                and source.getnframes() > 0
                and source.getcomptype() == "NONE"
            )
    except (OSError, wave.Error):
        return False


def _write_pcm_24khz_mono(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * np.iinfo(np.int16).max).astype(
        "<i2", copy=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24_000)
            output.writeframes(pcm.tobytes())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_evaluation_audio(path: Path) -> None:
    if not path.exists():
        return
    if path.name != "audio" or path.is_symlink():
        raise EvaluationIntegrityError(f"refusing to remove unexpected evaluation audio path: {path}")
    shutil.rmtree(path)


def _collective_wandb_preflight(request: EvaluationRequest) -> None:
    accelerator = request.accelerator
    error: str | None = None
    if getattr(accelerator, "is_main_process", False):
        try:
            if not isinstance(request.wandb_logger, WandbValidationLogger):
                raise WandbSyncError("WandbValidationLogger is mandatory on the main process")
            if request.wandb_logger.accelerator is not accelerator:
                raise WandbSyncError("W&B logger must use the evaluation Accelerator")
            request.wandb_logger.preflight(request.validation_index)
        except Exception as exc:
            error = str(exc)
    status = _broadcast_main_status(accelerator, "wandb-preflight", error)
    if not status["ok"]:
        raise WandbSyncError(status["error"])


def _broadcast_main_status(
    accelerator: Any,
    phase: str,
    error: str | None,
) -> dict[str, object]:
    payload: dict[str, object] | None = None
    if getattr(accelerator, "is_main_process", False):
        payload = {
            "format_version": 1,
            "phase": phase,
            "ok": error is None,
            "error": error,
        }
    values = [payload]
    broadcaster = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcaster):
        broadcaster(values, from_process=0)
    else:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(values, from_process=0)
    result = values[0]
    if (
        not isinstance(result, dict)
        or set(result) != {"format_version", "phase", "ok", "error"}
        or result.get("format_version") != 1
        or result.get("phase") != phase
        or type(result.get("ok")) is not bool
        or not (
            (result["ok"] is True and result.get("error") is None)
            or (result["ok"] is False and isinstance(result.get("error"), str) and bool(result["error"]))
        )
    ):
        raise WandbSyncError(f"collective {phase} status is invalid")
    return result


def _read_wandb_manifest(path: Path) -> dict[str, str | int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WandbSyncError("W&B run manifest is missing or invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("format_version") != 1
        or set(value) != {"format_version", "run_id"}
        or not isinstance(value.get("run_id"), str)
        or not value["run_id"]
    ):
        raise WandbSyncError("W&B run manifest is invalid")
    return value


def _read_wandb_commit(
    path: Path,
    validation_index: int,
    run_id: str,
    context: Mapping[str, object],
    marker: str,
) -> dict[str, object]:
    expected = {
        "format_version": 2,
        "validation_index": validation_index,
        "run_id": run_id,
        "evaluation_identity_sha256": context["evaluation_identity_sha256"],
        "assignment_sha256": context["assignment_sha256"],
        "artifact_checksums": context["artifact_checksums"],
        "metrics_sha256": context["metrics_sha256"],
        "remote_marker_sha256": marker,
        "media_logged": False,
        "scalars_logged": False,
        "committed": False,
    }
    if not path.is_file():
        return expected
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WandbSyncError("W&B validation commit ledger is invalid") from exc
    if not isinstance(value, dict) or set(value) != set(expected):
        raise WandbSyncError("W&B validation commit ledger fields are invalid")
    for name in (
        "format_version",
        "validation_index",
        "run_id",
        "evaluation_identity_sha256",
        "assignment_sha256",
        "artifact_checksums",
        "metrics_sha256",
        "remote_marker_sha256",
    ):
        if value.get(name) != expected[name]:
            raise WandbSyncError("W&B validation commit ledger changed")
    flags = tuple(value.get(name) for name in ("media_logged", "scalars_logged", "committed"))
    if any(type(flag) is not bool for flag in flags) or flags[2] and flags[:2] != (True, True):
        raise WandbSyncError("W&B validation commit state is invalid")
    return value


def _wandb_commit_context(report: EvaluationReport) -> dict[str, object]:
    if _SHA256.fullmatch(report.identity_checksum) is None or _SHA256.fullmatch(report.assignment_checksum) is None:
        raise WandbSyncError("evaluation identity checksums are invalid")
    expected_paths = {
        "results_jsonl": report.output_jsonl,
        "summary_json": report.summary_json,
        "panel_manifest": report.panel_dir / "manifest.json",
        "validation_seal": report.summary_json.with_name("validation-success.json"),
    }
    actual: dict[str, str] = {}
    for name, path in expected_paths.items():
        if not path.is_file():
            raise WandbSyncError(f"local W&B artifact is missing: {name}")
        actual[name] = sha256_file(path)
    if dict(report.artifact_checksums) != actual:
        raise WandbSyncError("local W&B artifact checksums changed")
    return {
        "evaluation_identity_sha256": report.identity_checksum,
        "assignment_sha256": report.assignment_checksum,
        "artifact_checksums": actual,
        "metrics_sha256": _canonical_sha256(report.metrics),
    }


def _wandb_api_history(run: Any, keys: Sequence[str]) -> Any:
    """Query durable history through the public API, not the active logging run."""

    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    run_id = getattr(run, "id", None)
    if any(not isinstance(value, str) or not value for value in (entity, project, run_id)):
        raise WandbSyncError("W&B run lacks entity/project/id for remote history queries")
    try:
        import wandb

        api_run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        return api_run.scan_history(keys=list(keys))
    except WandbSyncError:
        raise
    except Exception as exc:
        raise WandbSyncError("W&B API history query failed") from exc


def _wandb_remote_markers(
    run: Any,
    scalar_marker: str,
    media_marker: str,
    history_reader: Callable[[Any, Sequence[str]], Any],
) -> dict[str, object]:
    result = {scalar_marker: None, media_marker: None}
    try:
        for row in history_reader(run, [scalar_marker, media_marker]):
            if not isinstance(row, Mapping):
                raise WandbSyncError("W&B remote history returned an invalid marker row")
            for name in result:
                if row.get(name) is not None:
                    result[name] = row[name]
    except WandbSyncError:
        raise
    except Exception as exc:
        raise WandbSyncError("W&B remote commit marker query failed") from exc
    return result


def _wandb_scalars(report: EvaluationReport) -> dict[str, float | int]:
    values: dict[str, float | int] = {"validation-row-count": report.row_count}
    for name, value in report.metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = value
    categories = report.metrics.get("by_category")
    if isinstance(categories, Mapping):
        for category, category_metrics in sorted(categories.items()):
            if not isinstance(category, str) or not isinstance(category_metrics, Mapping):
                raise WandbSyncError("category metrics are invalid")
            for name, value in sorted(category_metrics.items()):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise WandbSyncError("category metric values are invalid")
                values[f"category/{category}/{name}"] = value
    values["generation-latency-seconds"] = report.generation_latency_seconds
    values["asr-latency-seconds"] = report.asr_latency_seconds
    return values


def _wandb_table(rows: Sequence[Mapping[str, object]]) -> object:
    import wandb

    columns = sorted({name for row in rows for name in row})
    return wandb.Table(
        columns=columns,
        data=[[json.dumps(row.get(name), ensure_ascii=False, sort_keys=True) for name in columns] for row in rows],
    )


def _wandb_audio(path: Path) -> object:
    import wandb

    return wandb.Audio(str(path), sample_rate=24_000)


def _semantic_assignment(items: Sequence[EvaluationItem]) -> list[dict[str, object]]:
    return [
        {
            "benchmark_id": item.benchmark_id,
            "stressed": item.stressed,
            "normalized_gold": item.normalized_gold,
            "hard_number": item.hard_number,
            "category": item.category,
            "number_span": {
                "reference_start": item.number_span.reference_start,
                "reference_end": item.number_span.reference_end,
                "category": item.number_span.category,
            },
            "voice_id": item.voice_id,
            "prompt_text": item.prompt_text,
            "prompt_sha256": item.prompt_sha256,
        }
        for item in items
    ]


def _item_sha256(item: EvaluationItem) -> str:
    return _canonical_sha256(_semantic_assignment([item])[0])


def _journal_record_sha256(record: Mapping[str, object]) -> str:
    return _canonical_sha256({name: value for name, value in record.items() if name != "record_sha256"})


def _valid_journal_record(
    value: object,
    item: EvaluationItem,
    identity: EvaluationIdentity,
    destination: Path | None,
) -> bool:
    expected_fields = {
        "benchmark_id",
        "evaluation_identity_sha256",
        "item_sha256",
        "hypothesis",
        "audio_path",
        "audio_sha256",
        "generation_latency_seconds",
        "asr_latency_seconds",
        "record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields or destination is None:
        return False
    if (
        value.get("benchmark_id") != item.benchmark_id
        or value.get("evaluation_identity_sha256") != identity.sha256
        or value.get("item_sha256") != _item_sha256(item)
        or value.get("audio_path") != str(destination)
        or not isinstance(value.get("hypothesis"), str)
        or not isinstance(value.get("audio_sha256"), str)
        or _SHA256.fullmatch(value["audio_sha256"]) is None
        or not _finite_nonnegative(value.get("generation_latency_seconds"))
        or not _finite_nonnegative(value.get("asr_latency_seconds"))
        or value.get("record_sha256") != _journal_record_sha256(value)
        or not _is_pcm_24khz_mono(destination)
        or sha256_file(destination) != value["audio_sha256"]
    ):
        return False
    return True


def _valid_gathered_record(value: Mapping[str, object], item: EvaluationItem, identity: EvaluationIdentity) -> bool:
    expected_fields = {
        "benchmark_id",
        "evaluation_identity_sha256",
        "item_sha256",
        "hypothesis",
        "audio_path",
        "audio_sha256",
        "generation_latency_seconds",
        "asr_latency_seconds",
        "record_sha256",
    }
    path_value = value.get("audio_path")
    path = Path(path_value) if isinstance(path_value, str) else None
    return bool(
        set(value) == expected_fields
        and value.get("benchmark_id") == item.benchmark_id
        and value.get("evaluation_identity_sha256") == identity.sha256
        and value.get("item_sha256") == _item_sha256(item)
        and isinstance(value.get("hypothesis"), str)
        and isinstance(value.get("audio_sha256"), str)
        and _SHA256.fullmatch(value["audio_sha256"]) is not None
        and _finite_nonnegative(value.get("generation_latency_seconds"))
        and _finite_nonnegative(value.get("asr_latency_seconds"))
        and value.get("record_sha256") == _journal_record_sha256(value)
        and path is not None
        and _is_pcm_24khz_mono(path)
        and sha256_file(path) == value["audio_sha256"]
    )


def _finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0


def _require_component_provenance(request: EvaluationRequest) -> None:
    recognizer_provenance = getattr(request.recognizer, "provenance", None)
    synthesizer_provenance = getattr(request.synthesizer, "provenance", None)
    if not callable(recognizer_provenance) or _thaw_json(recognizer_provenance()) != _thaw_json(request.provenance.asr_config):
        raise EvaluationIntegrityError("GigaAM runtime config does not match evaluation provenance")
    if not callable(synthesizer_provenance) or _thaw_json(synthesizer_provenance()) != _thaw_json(request.provenance.synthesis_config):
        raise EvaluationIntegrityError("synthesis runtime config does not match evaluation provenance")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(_thaw_json(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        json.dumps(value, allow_nan=False)
        return value
    raise ValueError(f"identity config is not JSON-serializable: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_thaw_json(item) for item in value]
    return value


def _benchmark_fields(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        value = dict(row)
    else:
        raw = getattr(row, "raw", None)
        value = dict(raw) if isinstance(raw, Mapping) else {}
        for name in ("id", "stressed", "normalized_gold", "hard_number", "category"):
            value[name] = getattr(row, name, value.get(name))
    identifier = value.get("id")
    if isinstance(identifier, bool) or not isinstance(identifier, int):
        raise EvaluationIntegrityError("benchmark IDs must be integers")
    strings = {name: value.get(name) for name in ("stressed", "normalized_gold", "hard_number", "category")}
    if any(not isinstance(item, str) or not item.strip() for item in strings.values()):
        raise EvaluationIntegrityError("benchmark text fields must be nonempty strings")
    span_value = value.get("number_span")
    if isinstance(span_value, NumberSpan):
        span = span_value
    elif isinstance(span_value, Mapping):
        start, end = span_value.get("reference_start"), span_value.get("reference_end")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
            raise EvaluationIntegrityError("benchmark number span is invalid")
        span = NumberSpan(start, end, strings["category"])
    else:
        span = extract_number_span(value)
    return {"id": identifier, **strings, "number_span": span}


def _prompt_fields(prompt: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(prompt, Mapping):
        raise EvaluationIntegrityError("prompt metadata must be mappings")
    path_value = prompt.get("audio_path")
    path = Path(path_value) if isinstance(path_value, (str, Path)) else None
    voice_value = prompt.get("voice_id")
    if voice_value is None and path is not None:
        voice_value = path.stem
    if not isinstance(voice_value, str) or _VOICE_ID.fullmatch(voice_value) is None:
        raise EvaluationIntegrityError("prompt voice_id must match voice_NN")
    text = prompt.get("text")
    checksum = prompt.get("wav_sha256")
    if path is None or not path.is_file() or not isinstance(text, str) or not text.strip():
        raise EvaluationIntegrityError(f"prompt metadata is incomplete for {voice_value}")
    if not isinstance(checksum, str) or len(checksum) != 64 or sha256_file(path) != checksum:
        raise EvaluationIntegrityError(f"prompt WAV checksum changed for {voice_value}")
    return {"voice_id": voice_value, "audio_path": path, "text": text, "wav_sha256": checksum}


def _require_gigaam_cuda(model: Any, local_rank: int) -> None:
    asr = getattr(model, "asr", None)
    options = getattr(asr, "runtime_config", None)
    if not isinstance(options, Mapping):
        raise GigaAmError("GigaAM model cannot prove CUDAExecutionProvider placement")
    providers = options.get("providers")
    provider_options = options.get("provider_options")
    if not isinstance(providers, Sequence) or isinstance(providers, (str, bytes)) or not providers:
        raise GigaAmError("GigaAM model cannot prove CUDAExecutionProvider placement")
    first = providers[0]
    name = first[0] if isinstance(first, tuple) and first else first
    if name != "CUDAExecutionProvider":
        raise GigaAmError("GigaAM must use CUDAExecutionProvider without CPU fallback")
    cuda_options: object = first[1] if isinstance(first, tuple) and len(first) == 2 else None
    if cuda_options is None and isinstance(provider_options, Sequence) and provider_options:
        cuda_options = provider_options[0]
    if not isinstance(cuda_options, Mapping):
        raise GigaAmError("GigaAM CUDAExecutionProvider device cannot be verified")
    try:
        device_id = int(cuda_options.get("device_id", -1))
    except (TypeError, ValueError) as exc:
        raise GigaAmError("GigaAM CUDAExecutionProvider device cannot be verified") from exc
    if device_id != local_rank:
        raise GigaAmError(f"GigaAM CUDA device {device_id} does not match local rank {local_rank}")
    sessions = tuple(getattr(asr, name, None) for name in ("_encoder", "_decoder", "_joiner"))
    if any(session is None for session in sessions):
        raise GigaAmError("GigaAM v3 RNN-T CUDA sessions cannot be verified")
    for session in sessions:
        try:
            active = session.get_providers()
            active_options = session.get_provider_options()
        except Exception as exc:
            raise GigaAmError("GigaAM v3 RNN-T CUDA sessions cannot be verified") from exc
        if not active or active[0] != "CUDAExecutionProvider":
            raise GigaAmError("GigaAM RNN-T session did not activate CUDAExecutionProvider")
        try:
            active_device = int(active_options["CUDAExecutionProvider"].get("device_id", -1))
        except (KeyError, TypeError, ValueError) as exc:
            raise GigaAmError("GigaAM CUDAExecutionProvider device cannot be verified") from exc
        if active_device != local_rank:
            raise GigaAmError(
                f"GigaAM active CUDA device {active_device} does not match local rank {local_rank}"
            )


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and ("out of memory" in message or "oom" in message)

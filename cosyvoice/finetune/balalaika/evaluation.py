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
class EvaluationRequest:
    """Complete dependencies and durable destinations for one validation point."""

    rows: Sequence[object]
    prompts: Sequence[Mapping[str, object]]
    accelerator: Any
    synthesizer: Synthesizer
    recognizer: GigaAmRecognizer
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
    metrics: Mapping[str, object]
    row_count: int
    worst_errors: tuple[Mapping[str, object], ...]
    panel_audio: Mapping[str, Path]
    generation_latency_seconds: float
    asr_latency_seconds: float


class WandbValidationLogger:
    """Idempotently commit validation evidence to Task 8's existing W&B run."""

    def __init__(
        self,
        accelerator: Any,
        run_manifest: Path,
        *,
        table_factory: Callable[[Sequence[Mapping[str, object]]], object] | None = None,
        audio_factory: Callable[[Path], object] | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.run_manifest = Path(run_manifest)
        self.commit_dir = self.run_manifest.parent / "wandb-validation-commits"
        self._table_factory = table_factory or _wandb_table
        self._audio_factory = audio_factory or _wandb_audio
        self._owned_run_id: str | None = None

    @staticmethod
    def resume_init_kwargs(run_manifest: Path) -> dict[str, dict[str, str]]:
        """Return the only approved later-launch Accelerate tracker init kwargs."""

        manifest = _read_wandb_manifest(Path(run_manifest))
        return {"wandb": {"id": manifest["run_id"], "resume": "must"}}

    def log(self, report: EvaluationReport, validation_index: int) -> None:
        """Log media then scalars once, recording each successful durable phase."""

        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise WandbSyncError("WANDB_API_KEY is required in the environment")
        if isinstance(validation_index, bool) or validation_index not in range(41):
            raise WandbSyncError("W&B validation index must be in 0 through 40")
        if getattr(report, "validation_index", None) != validation_index:
            raise WandbSyncError("W&B validation index disagrees with the evaluation report")
        try:
            run = self.accelerator.get_tracker("wandb", unwrap=True)
        except Exception as exc:
            raise WandbSyncError("Accelerate has no initialized W&B tracker") from exc
        run_id = getattr(run, "id", None)
        if not isinstance(run_id, str) or not run_id:
            raise WandbSyncError("Accelerate W&B tracker has no run ID")
        self._bind_existing_run(run, run_id, validation_index)
        output = getattr(report, "output_jsonl", None)
        if not isinstance(output, Path) or not output.is_file():
            raise WandbSyncError("local validation JSONL must exist before W&B logging")
        results_checksum = sha256_file(output)
        commit_path = self.commit_dir / f"validation-{validation_index:02d}.json"
        state = _read_wandb_commit(commit_path, validation_index, run_id, results_checksum)
        if state["committed"]:
            return
        sync_dir = getattr(run, "dir", self.run_manifest.parent)
        try:
            if not state["media_logged"]:
                media = {"worst-errors": self._table_factory(report.worst_errors)}
                media.update(
                    {
                        f"listening-panel/{voice_id}": self._audio_factory(path)
                        for voice_id, path in sorted(report.panel_audio.items())
                    }
                )
                run.log(media, step=validation_index, commit=False)
                state["media_logged"] = True
                atomic_write_json(commit_path, state)
            if not state["scalars_logged"]:
                self.accelerator.log(_wandb_scalars(report), step=validation_index)
                state["scalars_logged"] = True
                atomic_write_json(commit_path, state)
            state["committed"] = True
            atomic_write_json(commit_path, state)
        except Exception as exc:
            raise WandbSyncError(
                f"W&B validation {validation_index} is incomplete; syncable local run preserved at {sync_dir}"
            ) from exc

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
    items = build_voice_assignment(request.rows, request.prompts)
    assignment_checksum = _ensure_assignment_manifest(request, items)
    accelerator.wait_for_everyone()

    report = _load_published_report(request, assignment_checksum)
    if report is None:
        report = _generate_and_publish(request, items, assignment_checksum)
    accelerator.wait_for_everyone()
    if report is None:
        report = _require_published_report(request, assignment_checksum)

    logging_error: str | None = None
    if getattr(accelerator, "is_main_process", False) and request.wandb_logger is not None:
        try:
            request.wandb_logger.log(report, request.validation_index)
        except Exception as exc:
            logging_error = str(exc)
    logging_error = _broadcast_main_value(accelerator, logging_error)
    if logging_error is not None:
        raise WandbSyncError(logging_error)
    return report


def _generate_and_publish(
    request: EvaluationRequest,
    items: Sequence[EvaluationItem],
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
    existing = _load_rank_journal(journal_path, expected_local_ids)
    pending: list[tuple[EvaluationItem, Path, float]] = []
    for item in local_items:
        if item.benchmark_id in existing:
            continue
        destination = rank_root / f"benchmark-{item.benchmark_id:04d}.wav"
        generation_latency = 0.0
        if not _is_pcm_24khz_mono(destination):
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
                "hypothesis": hypothesis,
                "audio_path": str(path),
                "audio_sha256": sha256_file(path),
                "generation_latency_seconds": generation_latency,
                "asr_latency_seconds": latency,
            }
        _atomic_write_jsonl(journal_path, [existing[key] for key in sorted(existing)])

    if set(existing) != expected_local_ids:
        raise EvaluationIntegrityError("local evaluation journal is missing benchmark IDs")
    gathered = accelerator.gather_object([existing[key] for key in sorted(existing)])
    gathered_records = _flatten_gathered_records(gathered)
    if not getattr(accelerator, "is_main_process", False):
        return None
    report = _publish_report(request, items, gathered_records, assignment_checksum)
    if os.environ.get("KEEP_EVAL_AUDIO") != "1":
        _remove_evaluation_audio(request.temporary_audio_dir)
    return report


def _publish_report(
    request: EvaluationRequest,
    items: Sequence[EvaluationItem],
    gathered: Sequence[Mapping[str, object]],
    assignment_checksum: str,
) -> EvaluationReport:
    by_id: dict[int, Mapping[str, object]] = {}
    for record in gathered:
        identifier = record.get("benchmark_id")
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier in by_id:
            raise EvaluationIntegrityError("gathered evaluation records contain missing or duplicate IDs")
        by_id[identifier] = record
    if set(by_id) != set(range(1, _EXPECTED_ROWS + 1)):
        raise EvaluationIntegrityError("gathered evaluation records must cover exactly IDs 1 through 2000")

    published: list[dict[str, object]] = []
    scored = []
    item_by_id = {item.benchmark_id: item for item in items}
    for identifier in range(1, _EXPECTED_ROWS + 1):
        item = item_by_id[identifier]
        raw = by_id[identifier]
        hypothesis = raw.get("hypothesis")
        audio_path = raw.get("audio_path")
        if not isinstance(hypothesis, str) or not isinstance(audio_path, str):
            raise EvaluationIntegrityError(f"gathered record {identifier} is incomplete")
        scores = score_row(item.normalized_gold, hypothesis, item.number_span)
        scored.append(scores)
        published.append(_published_row(item, raw, hypothesis, scores))

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
    panel_audio = _publish_listening_panel(request.panel_dir, items, by_id)
    summary_payload = {
        "format_version": 1,
        "validation_index": request.validation_index,
        "row_count": _EXPECTED_ROWS,
        "assignment_sha256": assignment_checksum,
        "results": str(request.output_jsonl),
        "results_sha256": sha256_file(request.output_jsonl),
        "metrics": metrics,
        "diagnostics": {
            "generation_latency_seconds": generation_latency,
            "asr_latency_seconds": asr_latency,
        },
        "worst_benchmark_ids": [record["benchmark_id"] for record in worst],
        "listening_panel": {voice: str(path) for voice, path in panel_audio.items()},
    }
    atomic_write_json(request.summary_json, summary_payload)
    return EvaluationReport(
        validation_index=request.validation_index,
        output_jsonl=request.output_jsonl,
        summary_json=request.summary_json,
        panel_dir=request.panel_dir,
        assignment_checksum=assignment_checksum,
        metrics=metrics,
        row_count=_EXPECTED_ROWS,
        worst_errors=worst,
        panel_audio=panel_audio,
        generation_latency_seconds=generation_latency,
        asr_latency_seconds=asr_latency,
    )


def _published_row(item: EvaluationItem, raw: Mapping[str, object], hypothesis: str, scores: Any) -> dict[str, object]:
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
    return {
        "benchmark_id": item.benchmark_id,
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


def _scores_payload(scores: Any) -> dict[str, object]:
    def region(value: Any) -> dict[str, object]:
        return {
            "word": {**asdict(value.word), "distance": value.word.distance},
            "character": {**asdict(value.character), "distance": value.character.distance},
        }

    return {"utterance": region(scores.utterance), "number": region(scores.number)}


def _ensure_assignment_manifest(request: EvaluationRequest, items: Sequence[EvaluationItem]) -> str:
    entries = [
        {
            "benchmark_id": item.benchmark_id,
            "voice_id": item.voice_id,
            "prompt_sha256": item.prompt_sha256,
            "stressed_sha256": hashlib.sha256(item.stressed.encode("utf-8")).hexdigest(),
            "normalized_gold_sha256": hashlib.sha256(item.normalized_gold.encode("utf-8")).hexdigest(),
        }
        for item in items
    ]
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(canonical).hexdigest()
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


def _load_rank_journal(path: Path, expected_ids: set[int]) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, object]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            value = json.loads(line)
            identifier = value.get("benchmark_id") if isinstance(value, Mapping) else None
            audio_path = value.get("audio_path") if isinstance(value, Mapping) else None
            audio_checksum = value.get("audio_sha256") if isinstance(value, Mapping) else None
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or identifier not in expected_ids
                or identifier in records
                or not isinstance(value.get("hypothesis"), str)
                or not isinstance(audio_path, str)
                or not isinstance(audio_checksum, str)
                or not _is_pcm_24khz_mono(Path(audio_path))
                or sha256_file(Path(audio_path)) != audio_checksum
            ):
                raise EvaluationIntegrityError("rank evaluation journal is incomplete or changed")
            records[identifier] = dict(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError("rank evaluation journal is invalid") from exc
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
        for voice_id, item in sorted(selected.items()):
            source_value = records[item.benchmark_id].get("audio_path")
            source = Path(source_value) if isinstance(source_value, str) else None
            if source is None or not _is_pcm_24khz_mono(source):
                raise EvaluationIntegrityError(f"listening panel audio is missing for {voice_id}")
            shutil.copyfile(source, temporary / f"{voice_id}.wav")
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
        os.replace(temporary, panel_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {voice: panel_dir / f"{voice}.wav" for voice in sorted(selected)}


def _load_published_report(request: EvaluationRequest, assignment_checksum: str) -> EvaluationReport | None:
    artifacts = (request.output_jsonl.is_file(), request.summary_json.is_file(), request.panel_dir.is_dir())
    if not any(artifacts):
        return None
    if not request.summary_json.is_file():
        return None
    if not all(artifacts):
        raise EvaluationIntegrityError("committed validation publication is incomplete")
    return _require_published_report(request, assignment_checksum)


def _require_published_report(request: EvaluationRequest, assignment_checksum: str) -> EvaluationReport:
    try:
        summary = json.loads(request.summary_json.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in request.output_jsonl.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError("published validation results are invalid") from exc
    if (
        not isinstance(summary, Mapping)
        or summary.get("format_version") != 1
        or summary.get("validation_index") != request.validation_index
        or summary.get("row_count") != _EXPECTED_ROWS
        or summary.get("assignment_sha256") != assignment_checksum
        or summary.get("results_sha256") != sha256_file(request.output_jsonl)
        or len(rows) != _EXPECTED_ROWS
        or [row.get("benchmark_id") for row in rows if isinstance(row, Mapping)] != list(range(1, _EXPECTED_ROWS + 1))
    ):
        raise EvaluationIntegrityError("published validation results changed or are incomplete")
    metrics = summary.get("metrics")
    diagnostics = summary.get("diagnostics")
    panel = summary.get("listening_panel")
    worst_ids = summary.get("worst_benchmark_ids")
    if not isinstance(metrics, Mapping) or not isinstance(diagnostics, Mapping) or not isinstance(panel, Mapping) or not isinstance(worst_ids, list):
        raise EvaluationIntegrityError("published validation summary fields are invalid")
    panel_audio = {voice: Path(path) for voice, path in panel.items() if isinstance(voice, str) and isinstance(path, str)}
    if (
        set(panel_audio) != {f"voice_{index:02d}" for index in range(_EXPECTED_VOICES)}
        or any(not _is_pcm_24khz_mono(path) for path in panel_audio.values())
    ):
        raise EvaluationIntegrityError("published listening panel changed or is incomplete")
    by_id = {row["benchmark_id"]: row for row in rows}
    worst = tuple(by_id[identifier] for identifier in worst_ids)
    return EvaluationReport(
        validation_index=request.validation_index,
        output_jsonl=request.output_jsonl,
        summary_json=request.summary_json,
        panel_dir=request.panel_dir,
        assignment_checksum=assignment_checksum,
        metrics=dict(metrics),
        row_count=_EXPECTED_ROWS,
        worst_errors=worst,
        panel_audio=panel_audio,
        generation_latency_seconds=float(diagnostics["generation_latency_seconds"]),
        asr_latency_seconds=float(diagnostics["asr_latency_seconds"]),
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


def _broadcast_main_value(accelerator: Any, value: str | None) -> str | None:
    values = [value]
    broadcaster = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcaster):
        broadcaster(values, from_process=0)
    else:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(values, from_process=0)
    return values[0]


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


def _read_wandb_commit(path: Path, validation_index: int, run_id: str, results_checksum: str) -> dict[str, object]:
    expected = {
        "format_version": 1,
        "validation_index": validation_index,
        "run_id": run_id,
        "results_sha256": results_checksum,
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
    for name in ("format_version", "validation_index", "run_id", "results_sha256"):
        if value.get(name) != expected[name]:
            raise WandbSyncError("W&B validation commit ledger changed")
    flags = tuple(value.get(name) for name in ("media_logged", "scalars_logged", "committed"))
    if any(type(flag) is not bool for flag in flags) or flags[2] and flags[:2] != (True, True):
        raise WandbSyncError("W&B validation commit state is invalid")
    return value


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

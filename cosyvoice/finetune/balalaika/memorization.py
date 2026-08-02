"""Mandatory, isolated teacher-forced four-sample memorization proof."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from itertools import cycle
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import torch
from torch import nn

from .artifacts import atomic_write_json, sha256_file
from .cache import CacheManifest, TOKEN_MAX, TOKEN_MIN
from .config import DEFAULT_SEED
from .data import CachedRow
from .model import LoraSettings, audit_trainable_parameters, inject_lora, load_base_llm
from .training import SchedulerSpec, _build_scheduler


@dataclass(frozen=True)
class SampleAccuracy:
    """Teacher-forced correct/target speech-token counts for one cached row."""

    source_relative_path: str
    correct_tokens: int
    target_tokens: int

    def __post_init__(self) -> None:
        if not self.source_relative_path:
            raise ValueError("sample identity must be nonempty")
        if self.correct_tokens < 0 or self.target_tokens < 1 or self.correct_tokens > self.target_tokens:
            raise ValueError("sample accuracy counts are invalid")

    @property
    def exact(self) -> bool:
        return self.correct_tokens == self.target_tokens


class MemorizationGateError(RuntimeError):
    """The mandatory token-level trainability proof did not pass."""


@dataclass(frozen=True)
class MemorizationCheck:
    """One complete no-dropout evaluation of the four selected samples."""

    check_index: int
    optimizer_step: int
    samples: tuple[SampleAccuracy, ...]


@dataclass(frozen=True)
class MemorizationReport:
    """Success-only, checksum-bound evidence for the memorization proof."""

    path: Path
    steps: int
    rows: tuple[CachedRow, CachedRow, CachedRow, CachedRow]
    checks: tuple[MemorizationCheck, ...]
    cache_checksum: str
    base_checkpoint_checksum: str


@dataclass(frozen=True)
class VerifiedMemorizationEvidence:
    """Tamper-checked success evidence suitable for a downstream stage gate."""

    path: Path
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class MemorizationRequest:
    """Explicit dependencies for one fresh, isolated memorization attempt."""

    split_plan: Path
    cache: CacheManifest
    output_root: Path
    base_model_dir: Path
    seed: int = DEFAULT_SEED
    max_steps: int = 2_000
    check_every: int = 10
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    batch_size: int = 1
    world_size: int = 8
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    required_consecutive_checks: int = 3
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_foreach: bool | None = None
    optimizer_capturable: bool = False
    optimizer_differentiable: bool = False
    optimizer_fused: bool | None = None
    scheduler_spec: SchedulerSpec = SchedulerSpec()
    dataloader_identity: str = "cosyvoice.balalaika.memorization-loader:v1"
    tokenizer_identity: str = "cosyvoice.qwen-tokenizer:v3"
    tokenizer: Any | None = None
    model_loader: Callable[[Path], nn.Module] = load_base_llm
    adapter_injector: Callable[[nn.Module, LoraSettings], nn.Module] = inject_lora
    audit_fn: Callable[[nn.Module], Any] = audit_trainable_parameters
    dataloader_factory: Callable[..., Iterable[Mapping[str, Any]]] | None = None
    accelerator_factory: Callable[..., Any] | None = None
    wav_writer: Callable[[nn.Module, tuple[CachedRow, CachedRow, CachedRow, CachedRow], Path], Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.max_steps < 1 or self.max_steps > 2_000:
            raise ValueError("max_steps must be in [1, 2000]")
        if self.check_every < 1 or self.check_every > self.max_steps:
            raise ValueError("check_every must be in [1, max_steps]")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.batch_size < 1:
            raise ValueError("learning_rate, max_grad_norm, and batch_size must be positive")
        if self.world_size != 8 or self.mixed_precision != "bf16" or self.gradient_accumulation_steps != 1:
            raise ValueError("memorization gate requires world_size=8, bf16, and one accumulation step")
        if self.required_consecutive_checks != 3:
            raise ValueError("memorization gate requires exactly three consecutive checks")
        if self.optimizer_betas != (0.9, 0.999) or self.optimizer_eps != 1e-8 or self.optimizer_weight_decay != 0.01:
            raise ValueError("memorization gate requires the fixed production AdamW hyperparameters")
        if (
            self.optimizer_foreach is not None
            or self.optimizer_capturable is not False
            or self.optimizer_differentiable is not False
            or self.optimizer_fused is not None
        ):
            raise ValueError("memorization gate requires the fixed production AdamW execution settings")
        if type(self.scheduler_spec) is not SchedulerSpec:
            raise ValueError("scheduler_spec must be a SchedulerSpec")
        if not isinstance(self.dataloader_identity, str) or not self.dataloader_identity:
            raise ValueError("dataloader_identity must be a nonempty string")
        if not isinstance(self.tokenizer_identity, str) or not self.tokenizer_identity:
            raise ValueError("tokenizer_identity must be a nonempty string")


def run_memorization_gate(request: MemorizationRequest) -> MemorizationReport:
    """Run the four-row proof through the production LoRA/Accelerate loss path.

    No manifest is published for a failed attempt.  A successful proof owns an
    immutable ``memorization/`` directory and cannot be reused as a stale gate.
    """

    rows = select_memorization_rows(request.split_plan, request.cache, request.seed)
    cache_checksum = _cache_checksum(request.cache)
    accelerator = _memorization_accelerator(request)
    if getattr(accelerator, "num_processes", None) != request.world_size:
        raise MemorizationGateError("memorization gate requires exactly eight Accelerate ranks")
    target = Path(request.output_root) / "memorization"
    temporary = Path(request.output_root) / ".memorization.incomplete"
    if accelerator.is_main_process:
        if target.exists() or temporary.exists():
            raise MemorizationGateError("memorization evidence already exists; refuse stale or incomplete gate")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
    accelerator.wait_for_everyone()

    try:
        model = request.model_loader(Path(request.base_model_dir))
        adapted = request.adapter_injector(model, LoraSettings())
        audit = request.audit_fn(adapted)
        trainable = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
        if not trainable:
            raise MemorizationGateError("fresh memorization adapter has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=request.learning_rate,
            betas=request.optimizer_betas,
            eps=request.optimizer_eps,
            weight_decay=request.optimizer_weight_decay,
            amsgrad=False,
            maximize=False,
            foreach=request.optimizer_foreach,
            capturable=request.optimizer_capturable,
            differentiable=request.optimizer_differentiable,
            fused=request.optimizer_fused,
        )
        scheduler = _build_scheduler(optimizer, request.scheduler_spec)
        train_loader = _memorization_loader(request, rows)
        prepared = accelerator.prepare(adapted, optimizer, scheduler, train_loader)
        adapted, optimizer, scheduler, train_loader = prepared
        train_batches = tuple(train_loader)
        if not train_batches:
            raise MemorizationGateError("memorization dataloader is empty")
        evaluation_batches = tuple(_memorization_loader(request, rows))
        if not evaluation_batches:
            raise MemorizationGateError("memorization evaluation dataloader is empty")

        losses: list[float] = []
        checks: list[MemorizationCheck] = []
        latest_tokens: Mapping[str, Mapping[str, list[int]]] = {}
        consecutive = 0
        for step, batch in zip(range(1, request.max_steps + 1), cycle(train_batches)):
            adapted.train()
            optimizer.zero_grad(set_to_none=True)
            with accelerator.accumulate(adapted):
                result = adapted(batch, accelerator.device)
                loss = _loss(result)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(trainable, request.max_grad_norm)
                optimizer.step()
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % request.check_every:
                continue
            samples, latest_tokens = _evaluate_all_rows(adapted, evaluation_batches, accelerator, rows)
            check = MemorizationCheck(len(checks) + 1, step, samples)
            checks.append(check)
            exact = all(sample.exact for sample in samples)
            consecutive = consecutive + 1 if exact else 0
            if consecutive == request.required_consecutive_checks:
                report: MemorizationReport | None = None
                if accelerator.is_main_process:
                    report = _publish_success(
                        temporary,
                        request,
                        rows,
                        checks,
                        latest_tokens,
                        losses,
                        audit,
                        cache_checksum,
                        accelerator.unwrap_model(adapted),
                        step,
                    )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.replace(temporary, target)
                accelerator.wait_for_everyone()
                base_checksum = getattr(accelerator.unwrap_model(adapted), "_balalaika_base_checkpoint_sha256", None)
                if not isinstance(base_checksum, str):
                    raise MemorizationGateError("adapted model lacks a base checkpoint checksum")
                if report is not None:
                    return replace(report, path=target)
                return MemorizationReport(target, step, rows, tuple(checks), cache_checksum, base_checksum)
        raise MemorizationGateError(
            f"memorization gate did not reach three consecutive exact per-sample checks in {request.max_steps} steps"
        )
    except Exception:
        if accelerator.is_main_process and temporary.exists():
            shutil.rmtree(temporary)
        accelerator.wait_for_everyone()
        raise


def _memorization_accelerator(request: MemorizationRequest) -> Any:
    factory = request.accelerator_factory
    if factory is None:
        from accelerate import Accelerator

        factory = Accelerator
    return factory(
        mixed_precision=request.mixed_precision,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        log_with="wandb",
    )


def _memorization_loader(
    request: MemorizationRequest,
    rows: tuple[CachedRow, CachedRow, CachedRow, CachedRow],
) -> Iterable[Mapping[str, Any]]:
    if request.dataloader_factory is not None:
        return request.dataloader_factory(rows, tokenizer=request.tokenizer, batch_size=request.batch_size)
    from .data import build_memorization_dataloader

    return build_memorization_dataloader(rows, tokenizer=request.tokenizer, batch_size=request.batch_size)


def _loss(result: Any) -> torch.Tensor:
    if not isinstance(result, Mapping) or not isinstance(result.get("loss"), torch.Tensor):
        raise MemorizationGateError("adapted model must return a tensor loss")
    loss = result["loss"]
    if loss.numel() != 1 or not bool(torch.isfinite(loss).all().item()):
        raise MemorizationGateError("loss must be finite scalar before optimization or evaluation")
    return loss


def _evaluate_all_rows(
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]],
    accelerator: Any,
    rows: tuple[CachedRow, CachedRow, CachedRow, CachedRow],
) -> tuple[tuple[SampleAccuracy, ...], Mapping[str, Mapping[str, list[int]]]]:
    expected = tuple(row.source_relative_path for row in rows)
    measured: dict[str, SampleAccuracy] = {}
    teacher_forced_tokens: dict[str, Mapping[str, list[int]]] = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            sources = batch.get("utts")
            if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
                raise MemorizationGateError("memorization batch must contain utterance identities")
            result = model(batch, accelerator.device)
            if not isinstance(result, Mapping):
                raise MemorizationGateError("adapted model returned an invalid evaluation result")
            _loss(result)
            correct = result.get("correct_tokens_per_sample")
            total = result.get("target_tokens_per_sample")
            if not isinstance(correct, torch.Tensor) or not isinstance(total, torch.Tensor):
                raise MemorizationGateError("model lacks per-sample speech-token counts")
            if correct.dtype != torch.int64 or total.dtype != torch.int64:
                raise MemorizationGateError("per-sample speech-token counts must be int64")
            if correct.numel() != len(sources) or total.numel() != len(sources):
                raise MemorizationGateError("per-sample speech-token counts do not match batch identities")
            predictions = result.get("teacher_forced_predictions")
            targets = result.get("teacher_forced_targets")
            if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
                raise MemorizationGateError("model lacks teacher-forced predictions and targets")
            if predictions.dtype != torch.int64 or targets.dtype != torch.int64 or predictions.shape != targets.shape:
                raise MemorizationGateError("teacher-forced predictions and targets are invalid")
            if predictions.ndim < 1 or predictions.shape[0] != len(sources):
                raise MemorizationGateError("teacher-forced predictions do not match batch identities")
            prediction_rows = predictions.detach().cpu().tolist()
            target_rows = targets.detach().cpu().tolist()
            for source, right, count, prediction, target in zip(
                sources,
                correct.detach().cpu().tolist(),
                total.detach().cpu().tolist(),
                prediction_rows,
                target_rows,
            ):
                if not isinstance(source, str) or source in measured:
                    raise MemorizationGateError("evaluation emitted duplicate or invalid sample identity")
                measured[source] = SampleAccuracy(source, int(right), int(count))
                teacher_forced_tokens[source] = {"predictions": prediction, "targets": target}
    model.train(was_training)
    if tuple(measured) != expected:
        raise MemorizationGateError("evaluation did not produce exactly the four selected samples")
    return tuple(measured[source] for source in expected), teacher_forced_tokens


def _publish_success(
    root: Path,
    request: MemorizationRequest,
    rows: tuple[CachedRow, CachedRow, CachedRow, CachedRow],
    checks: Sequence[MemorizationCheck],
    teacher_forced_tokens: Mapping[str, Mapping[str, list[int]]],
    losses: Sequence[float],
    audit: Any,
    cache_checksum: str,
    model: nn.Module,
    steps: int,
) -> MemorizationReport:
    generated: tuple[str, ...] = ()
    if request.wav_writer is not None:
        wav_root = root / "generated_wavs"
        wav_root.mkdir()
        request.wav_writer(model, rows, wav_root)
        generated = tuple(sorted(str(path.relative_to(root)) for path in wav_root.rglob("*") if path.is_file()))
    base_checksum = getattr(model, "_balalaika_base_checkpoint_sha256", None)
    if not isinstance(base_checksum, str) or len(base_checksum) != 64:
        raise MemorizationGateError("adapted model lacks a base checkpoint checksum")
    audit_payload = asdict(audit) if hasattr(audit, "__dataclass_fields__") else audit
    if not isinstance(audit_payload, Mapping):
        raise MemorizationGateError("adapter audit is not serializable")
    atomic_write_json(root / "cross_entropy.json", {"per_optimizer_step": list(losses)})
    atomic_write_json(root / "teacher_forced_tokens.json", dict(teacher_forced_tokens))
    evidence = _evidence_checksums(root)
    rows_payload = [
        {
            "source_relative_path": row.source_relative_path,
            "speech_token_len": row.speech_token_len,
            "speech_token_sha256": hashlib.sha256(json.dumps(list(row.speech_token)).encode("utf-8")).hexdigest(),
        }
        for row in rows
    ]
    atomic_write_json(
        root / "memorization_manifest.json",
        {
            "format_version": 2,
            "provenance": {
                "cache_checksum": cache_checksum,
                "base_checkpoint_sha256": base_checksum,
                "rows": rows_payload,
                "run_config": _run_config(request),
                "adapter": {"settings": asdict(LoraSettings()), "trainable_inventory": dict(audit_payload)},
            },
            "adapter": {"settings": asdict(LoraSettings()), "trainable_inventory": dict(audit_payload)},
            "rows": rows_payload,
            "checks": [
                {
                    "check_index": check.check_index,
                    "optimizer_step": check.optimizer_step,
                    "samples": [
                        {"source_relative_path": sample.source_relative_path, "correct_tokens": sample.correct_tokens, "target_tokens": sample.target_tokens}
                        for sample in check.samples
                    ],
                }
                for check in checks
            ],
            "generated_wavs": list(generated),
            "evidence": evidence,
        },
    )


def _run_config(request: MemorizationRequest) -> dict[str, object]:
    return {
        "max_steps": request.max_steps,
        "learning_rate": request.learning_rate,
        "max_grad_norm": request.max_grad_norm,
        "optimizer": {
            "class": "torch.optim.adamw.AdamW",
            "betas": list(request.optimizer_betas),
            "eps": request.optimizer_eps,
            "weight_decay": request.optimizer_weight_decay,
            "amsgrad": False,
            "maximize": False,
            "foreach": request.optimizer_foreach,
            "capturable": request.optimizer_capturable,
            "differentiable": request.optimizer_differentiable,
            "fused": request.optimizer_fused,
        },
        "scheduler": {"kind": request.scheduler_spec.kind.value},
        "mixed_precision": request.mixed_precision,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "world_size": request.world_size,
        "local_batch_size": request.batch_size,
        "effective_batch_size": request.batch_size * request.gradient_accumulation_steps * request.world_size,
        "check_every": request.check_every,
        "required_consecutive_checks": request.required_consecutive_checks,
        "seed": request.seed,
        "dataloader_identity": request.dataloader_identity,
        "tokenizer_identity": request.tokenizer_identity,
    }


def _evidence_checksums(root: Path) -> dict[str, dict[str, str]]:
    return {
        str(path.relative_to(root)): {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "memorization_manifest.json"
    }


def require_memorization_gate(
    path: Path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> VerifiedMemorizationEvidence:
    """Reject a missing, altered, or provenance-mismatched gate before use."""

    root = Path(path)
    manifest_path = root / "memorization_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("memorization success manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 2:
        raise ValueError("memorization success manifest format is invalid")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("memorization success provenance is invalid")
    if expected_provenance is not None and provenance != dict(expected_provenance):
        raise ValueError("memorization success provenance changed")
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or evidence != _evidence_checksums(root):
        raise ValueError("memorization evidence checksum changed")
    rows = provenance.get("rows")
    checks = manifest.get("checks")
    config = provenance.get("run_config")
    if not isinstance(rows, list) or len(rows) != 4 or not isinstance(checks, list) or not isinstance(config, dict):
        raise ValueError("memorization success provenance is incomplete")
    required_checks = config.get("required_consecutive_checks")
    if required_checks != 3 or len(checks) < required_checks:
        raise ValueError("memorization success checks are incomplete")
    identities = [row.get("source_relative_path") for row in rows if isinstance(row, dict)]
    if len(identities) != 4 or len(set(identities)) != 4 or any(not isinstance(item, str) for item in identities):
        raise ValueError("memorization row provenance is invalid")
    for check in checks[-required_checks:]:
        if not isinstance(check, dict) or not isinstance(check.get("samples"), list):
            raise ValueError("memorization check evidence is invalid")
        samples = check["samples"]
        if len(samples) != 4 or {sample.get("source_relative_path") for sample in samples if isinstance(sample, dict)} != set(identities):
            raise ValueError("memorization check sample identities changed")
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("correct_tokens") != sample.get("target_tokens") or not isinstance(sample.get("target_tokens"), int) or sample["target_tokens"] < 1:
                raise ValueError("memorization check is not exact per sample")
    return VerifiedMemorizationEvidence(root, manifest)
    return MemorizationReport(
        root,
        steps,
        rows,
        tuple(checks),
        cache_checksum,
        base_checksum,
    )


def _cache_checksum(cache: CacheManifest) -> str:
    digest = hashlib.sha256()
    for shard in sorted(cache.shards):
        manifest = Path(cache.shards[shard])
        if not manifest.is_file():
            raise MemorizationGateError(f"cache shard manifest is missing: {manifest}")
        digest.update(str(shard).encode("ascii"))
        digest.update(sha256_file(manifest).encode("ascii"))
        for phase in (1, 2):
            path = cache.root / f"phase{phase}" / f"shard_{shard:06d}.parquet"
            if not path.is_file():
                raise MemorizationGateError(f"cache phase artifact is missing: {path}")
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def memorization_passed(checks: Sequence[Sequence[SampleAccuracy]]) -> bool:
    """Return whether a history contains three consecutive all-sample exact checks."""

    consecutive = 0
    for check in checks:
        exact = bool(check) and all(sample.exact for sample in check)
        consecutive = consecutive + 1 if exact else 0
        if consecutive >= 3:
            return True
    return False


def select_memorization_rows(
    split_plan: Path,
    cache: CacheManifest,
    seed: int = 1986,
) -> tuple[CachedRow, CachedRow, CachedRow, CachedRow]:
    """Select hash-ordered shortest/longest cache rows from both agreement phases.

    This scans cache shards independently, reconciles each cached row with its
    split-plan row, and retains only four metadata/token rows in memory.
    """

    plan_root = Path(split_plan)
    if not plan_root.is_dir():
        raise ValueError("split plan must be a directory")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if set(cache.phase_rows) != {1, 2} or not cache.shards:
        raise ValueError("cache manifest must contain both production phases and shards")

    extremes: dict[int, dict[str, CachedRow | None]] = {
        1: {"short": None, "long": None},
        2: {"short": None, "long": None},
    }
    lengths: dict[int, dict[str, int | None]] = {
        1: {"short": None, "long": None},
        2: {"short": None, "long": None},
    }
    totals = {1: 0, 2: 0}
    for shard in sorted(cache.shards):
        plan_rows = _phase_plan_rows(plan_root / f"shard_{shard:06d}.jsonl")
        for phase in (1, 2):
            cached = _cached_phase_rows(cache.root / f"phase{phase}" / f"shard_{shard:06d}.parquet")
            expected = plan_rows[phase]
            if set(cached) != set(expected):
                raise ValueError(f"cached phase {phase} rows do not match split plan for shard {shard}")
            totals[phase] += len(cached)
            for source, row in cached.items():
                plan = expected[source]
                if row.text != plan["text"] or row.instruct != plan["instruct"]:
                    raise ValueError(f"cached row {source} no longer matches split plan")
                row = CachedRow(
                    row.source_relative_path,
                    row.text,
                    row.instruct,
                    row.speech_token,
                    row.speech_token_len,
                    int(plan["text_token_count"]),
                )
                _consider_extreme(extremes[phase], lengths[phase], row, phase, seed)
    if totals != dict(cache.phase_rows):
        raise ValueError("cache phase row counts do not match the cache manifest")

    selected: list[CachedRow] = []
    for phase in (1, 2):
        short = extremes[phase]["short"]
        long = extremes[phase]["long"]
        if short is None or long is None or short.speech_token_len == long.speech_token_len:
            raise ValueError(f"phase {phase} lacks distinct short and long cached rows")
        selected.extend((short, long))
    if len({row.source_relative_path for row in selected}) != 4:
        raise RuntimeError("memorization selection contains duplicate row identities")
    return tuple(selected)  # type: ignore[return-value]


def _phase_plan_rows(path: Path) -> dict[int, dict[str, Mapping[str, object]]]:
    if not path.is_file():
        raise ValueError(f"split plan shard is missing: {path}")
    selected: dict[int, dict[str, Mapping[str, object]]] = {1: {}, 2: {}}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid split plan JSON: {path}:{number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"invalid split plan row: {path}:{number}")
            phase = value.get("phase")
            if phase not in (1, 2):
                continue
            source = value.get("source_relative_path")
            text = value.get("text")
            instruct = value.get("instruct")
            text_length = value.get("text_token_count")
            agreement = value.get("agreement")
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(text, str)
                or not text
                or not isinstance(instruct, str)
                or not instruct
                or isinstance(text_length, bool)
                or not isinstance(text_length, int)
                or text_length < 1
                or isinstance(agreement, bool)
                or not isinstance(agreement, (int, float))
                or (phase == 1 and not agreement < 0.95)
                or (phase == 2 and not agreement >= 0.95)
                or value.get("reserved") is not False
                or value.get("model_limit_exclusion") is not None
            ):
                raise ValueError(f"invalid eligible split plan row: {path}:{number}")
            if source in selected[phase]:
                raise ValueError(f"duplicate split plan source: {source}")
            selected[phase][source] = value
    return selected


def _cached_phase_rows(path: Path) -> dict[str, CachedRow]:
    if not path.is_file():
        raise ValueError(f"cached phase Parquet is missing: {path}")
    try:
        records = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise ValueError(f"cannot read cached phase Parquet: {path}") from exc
    rows: dict[str, CachedRow] = {}
    for record in records:
        source = record.get("source_relative_path")
        text = record.get("text")
        instruct = record.get("instruct")
        tokens = record.get("speech_token")
        length = record.get("speech_token_len")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(text, str)
            or not text
            or not isinstance(instruct, str)
            or not instruct
            or not isinstance(tokens, list)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
            or len(tokens) != length
            or any(isinstance(token, bool) or not isinstance(token, int) or token < TOKEN_MIN or token > TOKEN_MAX for token in tokens)
        ):
            raise ValueError(f"invalid cached row in {path}")
        if source in rows:
            raise ValueError(f"duplicate cached source: {source}")
        # The split plan's preflight token length is assigned during reconciliation.
        rows[source] = CachedRow(source, text, instruct, tuple(tokens), length, 0)
    return rows


def _consider_extreme(
    selected: dict[str, CachedRow | None],
    lengths: dict[str, int | None],
    row: CachedRow,
    phase: int,
    seed: int,
) -> None:
    for name, prefer_short in (("short", True), ("long", False)):
        current_length = lengths[name]
        better_length = current_length is None or (row.speech_token_len < current_length if prefer_short else row.speech_token_len > current_length)
        same_length = current_length == row.speech_token_len
        current = selected[name]
        if better_length or (
            same_length
            and current is not None
            and _selection_score(row.source_relative_path, phase, seed) < _selection_score(current.source_relative_path, phase, seed)
        ):
            selected[name] = row
            lengths[name] = row.speech_token_len


def _selection_score(source_relative_path: str, phase: int, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{phase}\0{source_relative_path}".encode("utf-8")).digest()

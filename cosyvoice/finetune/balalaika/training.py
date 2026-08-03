"""Exact global progress and restart-safe Accelerate phase training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import torch
from torch import nn

from .artifacts import atomic_write_json, sha256_file
from .config import DEFAULT_SEED, PhaseSpec
from .data import DEFAULT_LENGTH_WINDOW
from .model import LoraSettings, audit_trainable_parameters, save_adapter
from cosyvoice.utils.scheduler import ConstantLR


CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
_FRACTIONS_PER_EPOCH = 8


class TrainingCapacityError(RuntimeError):
    """Raised when a fixed production batch limit exceeds available memory."""


class SchedulerKind(str, Enum):
    """Closed scheduler algorithms supported by the Balalaika recipe."""

    CONSTANT = "constant-v1"


@dataclass(frozen=True)
class SchedulerSpec:
    """Immutable, JSON-serializable scheduler configuration."""

    kind: SchedulerKind | str = SchedulerKind.CONSTANT

    def __post_init__(self) -> None:
        try:
            kind = SchedulerKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown scheduler kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class FractionBoundary:
    """One exact one-eighth epoch boundary and its completed-sample threshold."""

    ordinal: int
    numerator: int
    denominator: int
    sample_target: int

    @classmethod
    def for_epoch(cls, eligible_samples: int) -> tuple["FractionBoundary", ...]:
        if isinstance(eligible_samples, bool) or not isinstance(eligible_samples, int) or eligible_samples < 1:
            raise ValueError("eligible_samples must be a positive integer")
        return tuple(
            cls(
                ordinal=ordinal,
                numerator=ordinal,
                denominator=_FRACTIONS_PER_EPOCH,
                sample_target=(eligible_samples * ordinal + _FRACTIONS_PER_EPOCH - 1) // _FRACTIONS_PER_EPOCH,
            )
            for ordinal in range(1, _FRACTIONS_PER_EPOCH + 1)
        )


@dataclass
class ProgressState:
    """Checkpointable phase cursor based only on completed real samples."""

    phase: int
    epoch: int = 0
    batch_offset: int = 0
    epoch_samples: int = 0
    global_samples: int = 0
    next_boundary_ordinal: int = 1
    validation_index: int = 0
    optimizer_steps: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = tuple(asdict(self))
        if set(state) != set(expected):
            raise ValueError("checkpoint progress fields are invalid")
        values: dict[str, int] = {}
        for name in expected:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"checkpoint progress field {name} is invalid")
            values[name] = value
        if values["phase"] not in (1, 2) or values["next_boundary_ordinal"] not in range(1, 10):
            raise ValueError("checkpoint progress cursor is invalid")
        for name, value in values.items():
            setattr(self, name, value)


@dataclass(frozen=True)
class ValidationEvent:
    """Stable callback payload for one successfully reached fraction."""

    phase: int
    epoch: int
    boundary: FractionBoundary
    validation_index: int
    global_samples: int
    checkpoint: Path
    model: nn.Module
    accelerator: Any


@dataclass(frozen=True)
class TrainingCallbacks:
    """External validation boundary owned by the later evaluation task."""

    validate: Callable[[ValidationEvent], bool | None]
    after_resume_loaded: Callable[[], None] | None = None


@dataclass(frozen=True)
class TrainRequest:
    """All immutable identity and injectable data boundaries for one phase."""

    model: nn.Module
    phase: PhaseSpec
    eligible_samples: int
    cache_checksum: str
    checkpoint_root: Path
    token_limit: int
    accumulation_steps: int
    dataloader_factory: Callable[[int, Any], Iterable[Mapping[str, Any]]]
    resume_from: Path | None = None
    accelerator_factory: Callable[..., Any] | None = None
    scheduler_spec: SchedulerSpec = SchedulerSpec()
    max_grad_norm: float = 1.0
    sampler_seed: int = DEFAULT_SEED
    sampler_window_size: int = DEFAULT_LENGTH_WINDOW
    dataloader_identity: str = "cosyvoice.balalaika.cached-rank-loader:v1"
    validation_index_base: int | None = None
    initial_adapter_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            approved_phase = PhaseSpec.for_phase(self.phase.number)
        except ValueError as exc:
            raise ValueError("phase must be an exact approved phase schedule") from exc
        if self.phase != approved_phase:
            raise ValueError("phase must be an exact approved phase schedule")
        approved_validation_base = 0 if self.phase.number == 1 else 16
        if self.validation_index_base is None:
            object.__setattr__(self, "validation_index_base", approved_validation_base)
        elif (
            isinstance(self.validation_index_base, bool)
            or not isinstance(self.validation_index_base, int)
            or self.validation_index_base != approved_validation_base
        ):
            raise ValueError(
                f"phase {self.phase.number} validation_index_base must be {approved_validation_base}"
            )
        if self.phase.number == 1:
            if self.initial_adapter_sha256 is not None:
                raise ValueError("phase 1 must start from a fresh no-op adapter")
        elif (
            not isinstance(self.initial_adapter_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.initial_adapter_sha256) is None
        ):
            raise ValueError("phase 2 requires the authenticated phase-1 adapter SHA-256")
        if type(self.scheduler_spec) is not SchedulerSpec:
            raise ValueError("scheduler_spec must be a SchedulerSpec")
        if isinstance(self.eligible_samples, bool) or self.eligible_samples < 1:
            raise ValueError("eligible_samples must be positive")
        if len(self.cache_checksum) != 64:
            raise ValueError("cache_checksum must be a SHA-256 digest")
        if self.token_limit < 1 or self.accumulation_steps < 1:
            raise ValueError("token_limit and accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if isinstance(self.sampler_seed, bool) or not isinstance(self.sampler_seed, int):
            raise ValueError("sampler_seed must be an integer")
        if isinstance(self.sampler_window_size, bool) or self.sampler_window_size < 1:
            raise ValueError("sampler_window_size must be positive")
        if not isinstance(self.dataloader_identity, str) or not self.dataloader_identity:
            raise ValueError("dataloader_identity must be a nonempty string")


@dataclass(frozen=True)
class PhaseResult:
    """Terminal or deliberately interrupted state of one phase invocation."""

    progress: ProgressState
    checkpoint: Path | None
    validation_boundaries: tuple[ValidationEvent, ...]
    completed: bool


def train_phase(request: TrainRequest, callbacks: TrainingCallbacks) -> PhaseResult:
    """Train one adapter phase through Accelerate and exact global boundaries."""

    accelerator = _accelerator(request)
    if getattr(accelerator, "num_processes", None) != 8:
        raise ValueError("Balalaika production training requires exactly eight Accelerate ranks")

    audit = audit_trainable_parameters(request.model)
    named_parameters = dict(request.model.named_parameters())
    audited = [named_parameters[name] for name in audit.trainable_parameters]
    trainable = [parameter for parameter in request.model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in trainable} != {id(parameter) for parameter in audited}:
        raise RuntimeError("optimizer trainables do not equal the audited adapter parameter set")
    optimizer = torch.optim.AdamW(trainable, lr=request.phase.learning_rate)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in audited}:
        raise RuntimeError("optimizer parameter IDs do not equal audited adapter parameter IDs")
    scheduler = _build_scheduler(optimizer, request.scheduler_spec)
    run_identity = _identity(request, accelerator)

    progress = ProgressState(
        phase=request.phase.number,
        validation_index=cast(int, request.validation_index_base),
    )
    resume_manifest: Mapping[str, object] | None = None
    if request.resume_from is not None:
        resume_manifest = _require_resume_manifest(Path(request.resume_from), run_identity, request.phase.number)
        progress.load_state_dict(_mapping(resume_manifest, "progress"))

    accelerator.register_for_checkpointing(progress)
    model, optimizer, scheduler = accelerator.prepare(request.model, optimizer, scheduler)
    if request.resume_from is not None:
        accelerator.load_state(str(request.resume_from))
        if callbacks.after_resume_loaded is not None:
            callbacks.after_resume_loaded()

    optimizer.zero_grad(set_to_none=True)
    reached: list[ValidationEvent] = []
    latest_checkpoint = Path(request.resume_from) if request.resume_from is not None else None
    boundaries = FractionBoundary.for_epoch(request.eligible_samples)

    def release_due_boundaries() -> bool:
        nonlocal latest_checkpoint
        while progress.next_boundary_ordinal <= _FRACTIONS_PER_EPOCH:
            boundary = boundaries[progress.next_boundary_ordinal - 1]
            if progress.epoch_samples < boundary.sample_target:
                break
            progress.next_boundary_ordinal += 1
            progress.validation_index += 1
            target = _checkpoint_path(request, progress.validation_index)
            event = ValidationEvent(
                phase=request.phase.number,
                epoch=progress.epoch,
                boundary=boundary,
                validation_index=progress.validation_index,
                global_samples=progress.global_samples,
                checkpoint=target,
                model=model,
                accelerator=accelerator,
            )
            should_continue = _save_validate_publish(
                request,
                accelerator,
                model,
                optimizer,
                scheduler,
                progress,
                event,
                callbacks,
                run_identity,
            )
            latest_checkpoint = target
            reached.append(event)
            if not should_continue:
                return False
        return True

    if resume_manifest is not None and resume_manifest.get("validation_status") == "pending":
        fraction = _mapping(resume_manifest, "fraction")
        ordinal = fraction.get("numerator")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in range(1, 9):
            raise ValueError("pending checkpoint fraction is invalid")
        boundary = boundaries[ordinal - 1]
        if fraction != {
            "numerator": boundary.numerator,
            "denominator": boundary.denominator,
            "sample_target": boundary.sample_target,
        }:
            raise ValueError("pending checkpoint fraction changed")
        event = ValidationEvent(
            phase=request.phase.number,
            epoch=progress.epoch,
            boundary=boundary,
            validation_index=progress.validation_index,
            global_samples=progress.global_samples,
            checkpoint=latest_checkpoint,
            model=model,
            accelerator=accelerator,
        )
        callback_result = callbacks.validate(event)
        accelerator.wait_for_everyone()
        _mark_validation_succeeded(accelerator, latest_checkpoint)
        accelerator.wait_for_everyone()
        reached.append(event)
        if callback_result is False:
            return PhaseResult(progress, latest_checkpoint, tuple(reached), False)
    if not release_due_boundaries():
        return PhaseResult(progress, latest_checkpoint, tuple(reached), False)

    while progress.epoch < request.phase.epochs:
        dataloader = _prepare_rank_dataloader(
            request.dataloader_factory(progress.epoch, accelerator), accelerator
        )
        if progress.batch_offset:
            dataloader = accelerator.skip_first_batches(dataloader, progress.batch_offset)
        pending_samples = 0
        pending_batches = 0
        for batch in dataloader:
            local_samples = _real_sample_count(batch)
            global_samples = _global_sample_count(accelerator, local_samples)
            pending_batches += 1
            if global_samples == 0:
                continue
            rank_samples = _rank_sample_counts(accelerator, local_samples, global_samples)
            forward_batch, zero_contribution = _collective_forward_batch(
                accelerator, batch, local_samples, rank_samples
            )
            try:
                with accelerator.accumulate(model):
                    loss = _batch_loss(model, forward_batch, accelerator, zero_contribution)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable, request.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as exc:
                raise TrainingCapacityError(
                    f"production token limit {request.token_limit} exhausted GPU memory; rerun qualification"
                ) from exc
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                raise TrainingCapacityError(
                    f"production token limit {request.token_limit} exhausted GPU memory; rerun qualification"
                ) from exc

            pending_samples += global_samples
            if not accelerator.sync_gradients:
                continue
            progress.batch_offset += pending_batches
            progress.epoch_samples += pending_samples
            progress.global_samples += pending_samples
            progress.optimizer_steps += 1
            pending_samples = 0
            pending_batches = 0
            if not release_due_boundaries():
                return PhaseResult(progress, latest_checkpoint, tuple(reached), False)

        if progress.epoch_samples != request.eligible_samples:
            raise RuntimeError(
                f"epoch {progress.epoch} completed {progress.epoch_samples} real samples; "
                f"expected {request.eligible_samples}"
            )
        progress.epoch += 1
        progress.batch_offset = 0
        progress.epoch_samples = 0
        progress.next_boundary_ordinal = 1

    return PhaseResult(progress, latest_checkpoint, tuple(reached), True)


def _accelerator(request: TrainRequest) -> Any:
    factory = request.accelerator_factory
    if factory is None:
        from accelerate import Accelerator

        factory = Accelerator
    return factory(
        mixed_precision="bf16",
        gradient_accumulation_steps=request.accumulation_steps,
        log_with="wandb",
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: SchedulerSpec,
) -> torch.optim.lr_scheduler.LRScheduler:
    if type(spec) is not SchedulerSpec:
        raise ValueError("scheduler spec must be a SchedulerSpec")
    if spec.kind is SchedulerKind.CONSTANT:
        return ConstantLR(optimizer)
    raise AssertionError(f"unhandled scheduler kind: {spec.kind!r}")


def _real_sample_count(batch: Mapping[str, Any]) -> int:
    utterances = batch.get("utts")
    if not isinstance(utterances, Sequence) or isinstance(utterances, (str, bytes)):
        raise ValueError("training batch must expose an utts sequence")
    return len(utterances)


def _global_sample_count(accelerator: Any, local_samples: int) -> int:
    value = torch.tensor(local_samples, dtype=torch.int64, device=accelerator.device)
    reduced = accelerator.reduce(value, reduction="sum")
    result = int(reduced.item())
    if result < 0:
        raise RuntimeError("global real-sample reduction was negative")
    return result


def _rank_sample_counts(accelerator: Any, local_samples: int, global_samples: int) -> tuple[int, ...]:
    fixture_gather = getattr(accelerator, "gather_sample_counts", None)
    if fixture_gather is not None:
        gathered = tuple(int(value) for value in fixture_gather(local_samples))
    else:
        value = torch.tensor([local_samples], dtype=torch.int64, device=accelerator.device)
        gathered = tuple(int(item) for item in accelerator.gather(value).reshape(-1).cpu().tolist())
    if len(gathered) != accelerator.num_processes or any(value < 0 for value in gathered):
        raise RuntimeError("real-sample counts were not gathered from every rank")
    if sum(gathered) != global_samples:
        raise RuntimeError("gathered rank sample counts disagree with the global reduction")
    return gathered


def _collective_forward_batch(
    accelerator: Any,
    batch: Mapping[str, Any],
    local_samples: int,
    rank_samples: Sequence[int],
) -> tuple[Mapping[str, Any], bool]:
    if all(value > 0 for value in rank_samples):
        return batch, False
    template = _one_sample_template(batch) if local_samples else None
    fixture_gather = getattr(accelerator, "gather_object", None)
    if fixture_gather is not None:
        gathered = fixture_gather([template])
    else:
        from accelerate.utils import gather_object

        gathered = gather_object([template])
    shared = next((value for value in gathered if value is not None), None)
    if not isinstance(shared, Mapping):
        raise RuntimeError("no real-shaped batch was available for an empty rank")
    if local_samples:
        return batch, False
    dummy = dict(shared)
    dummy["utts"] = []
    if "text" in dummy:
        dummy["text"] = []
    return dummy, True


def _one_sample_template(batch: Mapping[str, Any]) -> dict[str, Any]:
    template: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            template[name] = value.detach().cpu() if value.ndim == 0 else value[:1].detach().cpu()
        elif isinstance(value, list):
            template[name] = value[:1]
        elif isinstance(value, tuple):
            template[name] = value[:1]
        else:
            template[name] = value
    return template


def _batch_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    accelerator: Any,
    zero_contribution: bool,
) -> torch.Tensor:
    result = model(batch, accelerator.device)
    if not isinstance(result, Mapping) or not isinstance(result.get("loss"), torch.Tensor):
        raise RuntimeError("adapted model must return a tensor loss")
    return result["loss"] * 0 if zero_contribution else result["loss"]


def _prepare_rank_dataloader(dataloader: Iterable[Mapping[str, Any]], accelerator: Any) -> Iterable[Mapping[str, Any]]:
    fixture_prepare = getattr(accelerator, "prepare_rank_dataloader", None)
    if fixture_prepare is not None:
        return fixture_prepare(dataloader)

    from torch.utils.data import DataLoader

    if not isinstance(dataloader, DataLoader):
        raise TypeError("production dataloader_factory must return a torch DataLoader")
    from accelerate.data_loader import prepare_data_loader

    return prepare_data_loader(
        dataloader,
        accelerator.device,
        num_processes=1,
        process_index=0,
        split_batches=False,
        put_on_device=True,
        even_batches=False,
    )


def _checkpoint_path(request: TrainRequest, validation_index: int) -> Path:
    return Path(request.checkpoint_root) / f"phase-{request.phase.number}-validation-{validation_index:02d}"


def _save_validate_publish(
    request: TrainRequest,
    accelerator: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: ProgressState,
    event: ValidationEvent,
    callbacks: TrainingCallbacks,
    run_identity: Mapping[str, object],
) -> bool:
    target = event.checkpoint
    temporary = target.with_name(f".{target.name}.incomplete")
    if accelerator.is_main_process:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"checkpoint publication target already exists: {target}")
        if temporary.exists() or temporary.is_symlink():
            _discard_stale_staging(temporary, target)
        temporary.mkdir()
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temporary))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_adapter(accelerator.unwrap_model(model), temporary / "adapter")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        payload = {
            "format_version": 1,
            "validation_status": "pending",
            "identity": dict(run_identity),
            "progress": progress.state_dict(),
            "fraction": {
                "numerator": event.boundary.numerator,
                "denominator": event.boundary.denominator,
                "sample_target": event.boundary.sample_target,
            },
            "optimizer": optimizer.__class__.__qualname__,
            "scheduler": scheduler.__class__.__qualname__,
            "state_files": _state_checksums(temporary),
            "rng_files": sorted(
                str(path.relative_to(temporary))
                for path in temporary.rglob("random_states*")
                if path.is_file()
            ),
        }
        atomic_write_json(temporary / CHECKPOINT_MANIFEST, payload)
        os.replace(temporary, target)
    accelerator.wait_for_everyone()

    callback_result = callbacks.validate(event)
    accelerator.wait_for_everyone()
    _mark_validation_succeeded(accelerator, target)
    accelerator.wait_for_everyone()
    return callback_result is not False


def _discard_stale_staging(temporary: Path, target: Path) -> None:
    expected_name = f".{target.name}.incomplete"
    if temporary.parent.resolve() != target.parent.resolve() or temporary.name != expected_name:
        raise RuntimeError(f"refusing to remove unexpected checkpoint staging path: {temporary}")
    if temporary.is_symlink() or temporary.is_file():
        temporary.unlink()
    elif temporary.is_dir():
        shutil.rmtree(temporary)
    else:
        raise RuntimeError(f"unsupported checkpoint staging entry: {temporary}")


def _mark_validation_succeeded(accelerator: Any, checkpoint: Path) -> None:
    if not accelerator.is_main_process:
        return
    manifest_path = checkpoint / CHECKPOINT_MANIFEST
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pending checkpoint: {checkpoint}") from exc
    if not isinstance(payload, dict) or payload.get("validation_status") != "pending":
        raise ValueError(f"checkpoint is not pending validation: {checkpoint}")
    payload["validation_status"] = "succeeded"
    atomic_write_json(manifest_path, payload)


def _identity(request: TrainRequest, accelerator: Any) -> dict[str, object]:
    unwrapped = accelerator.unwrap_model(request.model)
    settings = getattr(unwrapped, "_balalaika_lora_settings", None)
    base_checksum = getattr(
        unwrapped, "_balalaika_base_checkpoint_sha256", None
    )
    if not isinstance(settings, LoraSettings) or not isinstance(base_checksum, str) or len(base_checksum) != 64:
        raise ValueError("adapted model lacks complete base-checksum/LoRA provenance")
    return {
        "cache_manifest_sha256": request.cache_checksum,
        "phase": request.phase.number,
        "phase_spec": asdict(request.phase),
        "eligible_samples": request.eligible_samples,
        "initial_adapter_sha256": request.initial_adapter_sha256,
        "base_checkpoint_sha256": base_checksum,
        "lora": asdict(settings),
        "token_limit": request.token_limit,
        "accumulation_steps": request.accumulation_steps,
        "max_grad_norm": request.max_grad_norm,
        "sampler_seed": request.sampler_seed,
        "sampler_window_size": request.sampler_window_size,
        "dataloader_identity": request.dataloader_identity,
        "scheduler": asdict(request.scheduler_spec),
        "validation_index_base": request.validation_index_base,
        "world_size": accelerator.num_processes,
    }


def _state_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != CHECKPOINT_MANIFEST
    }


def _require_resume_manifest(
    path: Path,
    expected_identity: Mapping[str, object],
    phase_number: int,
) -> Mapping[str, object]:
    manifest_path = path / CHECKPOINT_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resume checkpoint: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError(f"invalid resume checkpoint: {path}")
    if manifest.get("validation_status") not in ("pending", "succeeded"):
        raise ValueError("resume checkpoint validation status is invalid")
    if manifest.get("identity") != expected_identity:
        raise ValueError("resume checkpoint identity changed")
    checksums = _mapping(manifest, "state_files")
    actual = _state_checksums(path)
    if checksums != actual:
        raise ValueError("resume checkpoint state checksum changed")
    progress = _mapping(manifest, "progress")
    if progress.get("phase") != phase_number:
        raise ValueError("resume checkpoint phase changed")
    return manifest


def _mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    result = value.get(field)
    if not isinstance(result, dict):
        raise ValueError(f"checkpoint {field} is invalid")
    return result


@dataclass(frozen=True)
class QualificationRequest:
    """Representative long-batch probes executed on every fixed worker."""

    accelerator: Any
    run_candidate: Callable[[int], tuple[int, int] | None]
    candidates: tuple[int, ...] = (2000, 3000, 4000, 5000, 6000)
    required_headroom_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if self.candidates != (2000, 3000, 4000, 5000, 6000):
            raise ValueError("qualification candidates must be exactly 2000 through 6000 in 1000-token increments")
        if self.required_headroom_bytes != 2 * 1024**3:
            raise ValueError("qualification requires exactly two GiB of free headroom")


@dataclass(frozen=True)
class CandidateMemory:
    """Observed peak and physical capacity for one candidate on every rank."""

    token_limit: int
    peak_bytes_by_rank: tuple[int, ...]
    total_bytes_by_rank: tuple[int, ...]


@dataclass(frozen=True)
class BatchQualification:
    """Largest safe token limit plus the complete bounded sweep evidence."""

    token_limit: int
    candidates: tuple[CandidateMemory, ...] = field(default_factory=tuple)


def qualify_token_limit(request: QualificationRequest) -> BatchQualification:
    """Select the largest fixed candidate retaining two GiB on every rank."""

    accelerator = request.accelerator
    if getattr(accelerator, "num_processes", None) != 8:
        raise ValueError("Balalaika qualification requires exactly eight Accelerate ranks")
    records: list[CandidateMemory] = []
    selected: int | None = None
    for candidate in request.candidates:
        observed: tuple[int, int] | None = None
        local_oom = 0
        try:
            observed = request.run_candidate(candidate)
        except torch.cuda.OutOfMemoryError:
            local_oom = 1
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            local_oom = 1
        oom = torch.tensor([local_oom], dtype=torch.int64, device=accelerator.device)
        rank_oom = accelerator.gather(oom).reshape(-1).cpu().tolist()
        if len(rank_oom) != 8:
            raise RuntimeError("memory qualification did not gather exactly eight rank statuses")
        if any(rank_oom):
            break
        peak, total = observed if observed is not None else _cuda_memory_observation(accelerator.device)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (peak, total)):
            raise ValueError("memory probe must return positive integer peak and total byte counts")
        pair = torch.tensor([peak, total], dtype=torch.int64, device=accelerator.device)
        gathered = accelerator.gather(pair).reshape(-1, 2).cpu().tolist()
        if len(gathered) != 8:
            raise RuntimeError("memory qualification did not gather exactly eight rank observations")
        record = CandidateMemory(
            candidate,
            tuple(int(item[0]) for item in gathered),
            tuple(int(item[1]) for item in gathered),
        )
        records.append(record)
        if all(total_bytes - peak_bytes >= request.required_headroom_bytes for peak_bytes, total_bytes in gathered):
            selected = candidate
    if selected is None:
        raise TrainingCapacityError("no qualified token limit retains two GiB of headroom on every rank")
    return BatchQualification(selected, tuple(records))


def _cuda_memory_observation(device: torch.device) -> tuple[int, int]:
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    return peak, total

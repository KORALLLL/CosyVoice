"""Isolated three-rank LoRA smoke run for the Balalaika recipe."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import secrets
import shutil
from itertools import cycle
from typing import Any, Callable, Mapping, Sequence

import torch

from .artifacts import atomic_write_json, sha256_file
from .cache import CacheManifest
from .data import build_memorization_dataloader
from .memorization import select_memorization_rows
from .model import LoraSettings, audit_trainable_parameters, inject_lora, load_base_llm, validate_trainable_audit_payload


class ThreeGpuSmokeError(RuntimeError):
    """The isolated three-rank smoke run could not complete safely."""


class _CollectiveFailure(ThreeGpuSmokeError):
    """A distributed operation failed after its process group was torn down."""


_ATTEMPT_MARKER = ".attempt_id"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


@dataclass(frozen=True)
class ThreeGpuSmokeRequest:
    """Immutable inputs for one fresh three-rank smoke attempt."""

    cache: CacheManifest
    output_root: Path
    base_model_dir: Path
    split_plan: Path | None = None
    steps: int = 2
    world_size: int = 3
    mixed_precision: str = "bf16"
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    accelerator_factory: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache, CacheManifest):
            raise TypeError("cache must be a CacheManifest")
        if not isinstance(self.output_root, Path) or not isinstance(self.base_model_dir, Path):
            raise TypeError("output_root and base_model_dir must be pathlib.Path values")
        if self.world_size != 3:
            raise ValueError("three-GPU smoke requires world_size=3")
        if self.mixed_precision != "bf16":
            raise ValueError("three-GPU smoke requires mixed_precision='bf16'")
        if self.steps != 2:
            raise ValueError("three-GPU smoke requires steps=2")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.output_root.name != "three_gpu_smoke":
            raise ValueError("three-GPU smoke output directory must be named three_gpu_smoke")
        if self.split_plan is None:
            object.__setattr__(self, "split_plan", self.cache.root / "split_plan")
        else:
            raise ValueError("three-GPU smoke forbids a split_plan override")

    @property
    def temporary_root(self) -> Path:
        return self.output_root.parent / ".three_gpu_smoke.incomplete"


def run_three_gpu_smoke(request: ThreeGpuSmokeRequest) -> dict[str, object]:
    """Run exactly two isolated LoRA optimization steps across three ranks."""

    if not isinstance(request, ThreeGpuSmokeRequest):
        raise TypeError("request must be a ThreeGpuSmokeRequest")
    accelerator = _create_accelerator(request)
    if getattr(accelerator, "num_processes", None) != request.world_size:
        raise ThreeGpuSmokeError("three-GPU smoke requires exactly three Accelerate ranks")

    attempt_id = _create_temporary_root(accelerator, request)

    setup_error: Mapping[str, object] | None = None
    model = None
    optimizer = None
    train_loader = None
    audit_payload: dict[str, object] | None = None
    cache_checksum: str | None = None
    base_checksum: str | None = None
    try:
        from .workflow import _load_memorization_cache

        cache_checksum = _cache_manifest_checksum(request.cache)
        cache = _load_memorization_cache(request.cache.root)
        rows = select_memorization_rows(cache.root / "split_plan", cache)
        train_loader = build_memorization_dataloader(rows, batch_size=1)
        batches = tuple(train_loader)
        if not batches:
            raise ThreeGpuSmokeError("three-GPU smoke dataloader is empty")
        base_checkpoint = request.base_model_dir / "llm.pt"
        if not base_checkpoint.is_file():
            raise FileNotFoundError(f"missing CosyVoice3 LLM checkpoint: {base_checkpoint}")
        base_checksum = sha256_file(base_checkpoint)
        model = inject_lora(load_base_llm(request.base_model_dir), LoraSettings())
        audit_payload = _validated_audit_payload(audit_trainable_parameters(model))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ThreeGpuSmokeError("fresh smoke adapter has no trainable parameters")
        optimizer = torch.optim.AdamW(trainable, lr=request.learning_rate)
    except Exception as exc:
        setup_error = _failure_status(accelerator, exc)
    _raise_if_any_rank_failed(accelerator, setup_error, request.temporary_root, attempt_id)

    assert model is not None and optimizer is not None and train_loader is not None
    assert audit_payload is not None and cache_checksum is not None and base_checksum is not None
    model, optimizer, train_loader = _run_collective(
        accelerator,
        "prepare",
        request.temporary_root,
        attempt_id,
        lambda: accelerator.prepare(model, optimizer, train_loader),
    )

    batches_error: Mapping[str, object] | None = None
    batches: tuple[object, ...] = ()
    try:
        batches = tuple(train_loader)
        if not batches:
            raise ThreeGpuSmokeError("three-GPU smoke prepared dataloader is empty")
    except Exception as exc:
        batches_error = _failure_status(accelerator, exc)
    _raise_if_any_rank_failed(accelerator, batches_error, request.temporary_root, attempt_id)

    losses_by_step: list[list[float]] = []
    local_batches = cycle(batches)
    for _step in range(request.steps):
        step_error: Mapping[str, object] | None = None
        loss: torch.Tensor | None = None
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = next(local_batches)
            result = _run_collective(
                accelerator,
                "forward",
                request.temporary_root,
                attempt_id,
                lambda: model(batch, accelerator.device),
            )
            loss = _finite_scalar_loss(result)
        except _CollectiveFailure:
            raise
        except Exception as exc:
            step_error = _failure_status(accelerator, exc)
        _raise_if_any_rank_failed(accelerator, step_error, request.temporary_root, attempt_id)

        assert loss is not None
        optimization_error: Mapping[str, object] | None = None
        try:
            _run_collective(
                accelerator,
                "backward",
                request.temporary_root,
                attempt_id,
                lambda: accelerator.backward(loss),
            )
            accelerator.clip_grad_norm_(model.parameters(), request.max_grad_norm)
            optimizer.step()
        except _CollectiveFailure:
            raise
        except Exception as exc:
            optimization_error = _failure_status(accelerator, exc)
        _raise_if_any_rank_failed(accelerator, optimization_error, request.temporary_root, attempt_id)

        gather_error: Mapping[str, object] | None = None
        gathered = _run_collective(
            accelerator,
            "gather",
            request.temporary_root,
            attempt_id,
            lambda: accelerator.gather(loss.detach().reshape(1)),
        )
        try:
            loss_values = [float(value) for value in gathered.detach().cpu().reshape(-1).tolist()]
            if len(loss_values) != request.world_size or not all(torch.isfinite(torch.tensor(value)).item() for value in loss_values):
                raise ThreeGpuSmokeError("three-GPU smoke must gather one finite loss per rank")
            losses_by_step.append(loss_values)
        except Exception as exc:
            gather_error = _failure_status(accelerator, exc)
        _raise_if_any_rank_failed(accelerator, gather_error, request.temporary_root, attempt_id)

    manifest: dict[str, object] = {
        "attempt_id": attempt_id,
        "world_size": request.world_size,
        "steps": request.steps,
        "cache_manifest_sha256": cache_checksum,
        "base_llm_sha256": base_checksum,
        "trainable_audit": audit_payload,
        "losses_by_step": losses_by_step,
    }
    publish_error: Mapping[str, object] | None = None
    if getattr(accelerator, "is_main_process", False):
        try:
            atomic_write_json(request.temporary_root / "manifest.json", manifest)
        except Exception as exc:
            publish_error = _failure_status(accelerator, exc)
    _raise_if_any_rank_failed(accelerator, publish_error, request.temporary_root, attempt_id)

    rename_error: Mapping[str, object] | None = None
    published_output: Path | None = None
    if getattr(accelerator, "is_main_process", False):
        try:
            _rename_directory_noreplace(request.temporary_root, request.output_root)
            published_output = request.output_root
            (request.output_root / _ATTEMPT_MARKER).unlink()
        except Exception as exc:
            rename_error = _failure_status(accelerator, exc)
    _raise_if_any_rank_failed(
        accelerator,
        rename_error,
        request.temporary_root,
        attempt_id,
        published_output=published_output,
    )
    return manifest


def _create_accelerator(request: ThreeGpuSmokeRequest) -> Any:
    factory = request.accelerator_factory
    if factory is None:
        from accelerate import Accelerator

        factory = Accelerator
    return factory(mixed_precision=request.mixed_precision)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a same-filesystem directory without replacing a target."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise ThreeGpuSmokeError("atomic no-replace directory publication is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def _create_temporary_root(accelerator: Any, request: ThreeGpuSmokeRequest) -> str:
    """Share an ownership token before rank zero creates the temporary root."""

    attempt_id = _run_collective(
        accelerator,
        "attempt-id broadcast",
        request.temporary_root,
        None,
        lambda: _broadcast_attempt_id(accelerator),
    )
    created = False
    status: Mapping[str, object] | None = None
    if getattr(accelerator, "is_main_process", False):
        try:
            if request.output_root.exists() or request.temporary_root.exists():
                raise ThreeGpuSmokeError("three_gpu_smoke target or incomplete directory already exists")
            request.temporary_root.mkdir(parents=True)
            created = True
            (request.temporary_root / _ATTEMPT_MARKER).write_text(attempt_id, encoding="ascii")
        except Exception as exc:
            if created:
                shutil.rmtree(request.temporary_root, ignore_errors=True)
            status = _failure_status(accelerator, exc)
    shared = _run_collective(
        accelerator,
        "preflight broadcast",
        request.temporary_root,
        attempt_id,
        lambda: _broadcast_main_status(accelerator, status),
    )
    _run_collective(
        accelerator,
        "preflight barrier",
        request.temporary_root,
        attempt_id,
        accelerator.wait_for_everyone,
    )
    error = shared.get("error")
    if error is None:
        return attempt_id
    _cleanup_smoke_outputs(request.temporary_root, attempt_id)
    _run_collective(
        accelerator,
        "preflight cleanup barrier",
        request.temporary_root,
        attempt_id,
        accelerator.wait_for_everyone,
    )
    raise ThreeGpuSmokeError(f"three-GPU smoke failed on rank {shared['rank']}: {error}")


def _run_collective(
    accelerator: Any,
    name: str,
    temporary: Path,
    attempt_id: str | None,
    operation: Callable[[], Any],
    *,
    published_output: Path | None = None,
) -> Any:
    """Run one collective, aborting and cleaning before any later collective."""

    try:
        return operation()
    except Exception as exc:
        _teardown_collectives(accelerator)
        _cleanup_smoke_outputs(temporary, attempt_id, published_output)
        raise _CollectiveFailure(f"{name} collective failed after process-group teardown: {exc}") from exc


def _cleanup_smoke_outputs(
    temporary: Path,
    attempt_id: str | None,
    published_output: Path | None = None,
) -> None:
    """Best-effort removal of outputs owned by the current smoke attempt."""

    if _smoke_output_is_owned(temporary, attempt_id):
        shutil.rmtree(temporary, ignore_errors=True)
    if published_output is not None and _smoke_output_is_owned(published_output, attempt_id):
        shutil.rmtree(published_output, ignore_errors=True)


def _smoke_output_is_owned(path: Path, attempt_id: str | None) -> bool:
    if attempt_id is None or not path.is_dir() or path.is_symlink():
        return False
    marker = path / _ATTEMPT_MARKER
    try:
        if not marker.is_symlink() and marker.read_text(encoding="ascii") == attempt_id:
            return True
    except (OSError, UnicodeError):
        pass
    manifest = path / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("attempt_id") == attempt_id


def _teardown_collectives(accelerator: Any) -> None:
    """Best-effort abort/teardown that lets blocked peers leave a failed collective."""

    state = getattr(accelerator, "state", None)
    candidates = (
        getattr(accelerator, "process_group", None),
        getattr(state, "process_group", None),
        getattr(torch.distributed.group, "WORLD", None),
    )
    for process_group in candidates:
        abort = getattr(process_group, "abort", None)
        if callable(abort):
            try:
                abort()
                return
            except Exception:
                pass
    end_training = getattr(accelerator, "end_training", None)
    if callable(end_training):
        try:
            end_training()
            return
        except Exception:
            pass
    destroy = getattr(state, "destroy_process_group", None)
    if callable(destroy):
        try:
            destroy()
            return
        except Exception:
            pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


def _cache_manifest_checksum(cache: CacheManifest) -> str:
    manifest = cache.root / "manifest.json"
    if not manifest.is_file():
        raise ThreeGpuSmokeError(f"cache manifest is missing: {manifest}")
    return sha256_file(manifest)


def _validated_audit_payload(audit: Any) -> dict[str, object]:
    payload = asdict(audit) if hasattr(audit, "__dataclass_fields__") else audit
    if not isinstance(payload, Mapping):
        raise ThreeGpuSmokeError("adapter audit is not serializable")
    result = dict(payload)
    validate_trainable_audit_payload(result)
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in result.items()
    }


def _finite_scalar_loss(result: Any) -> torch.Tensor:
    if not isinstance(result, Mapping) or not isinstance(result.get("loss"), torch.Tensor):
        raise ThreeGpuSmokeError("adapted model must return a tensor loss")
    loss = result["loss"]
    if loss.numel() != 1 or not bool(torch.isfinite(loss).all().item()):
        raise ThreeGpuSmokeError("loss must be a finite scalar")
    return loss


def _failure_status(accelerator: Any, exc: Exception) -> Mapping[str, object]:
    return {
        "rank": int(getattr(accelerator, "process_index", 0)),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _broadcast_attempt_id(accelerator: Any) -> str:
    values: list[object] = [secrets.token_hex(16)] if getattr(accelerator, "is_main_process", False) else [None]
    broadcaster = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcaster):
        broadcaster(values, from_process=0)
    else:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(values, from_process=0)
    shared = values[0]
    if (
        not isinstance(shared, str)
        or len(shared) != 32
        or shared != shared.lower()
        or any(character not in "0123456789abcdef" for character in shared)
    ):
        raise ThreeGpuSmokeError("three-GPU smoke attempt-id broadcast is invalid")
    return shared


def _broadcast_main_status(accelerator: Any, status: Mapping[str, object] | None) -> Mapping[str, object]:
    values: list[object] = [
        dict(status) if status is not None else {"rank": 0, "error": None}
    ] if getattr(accelerator, "is_main_process", False) else [None]
    broadcaster = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcaster):
        broadcaster(values, from_process=0)
    else:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(values, from_process=0)
    shared = values[0]
    if not isinstance(shared, Mapping) or not isinstance(shared.get("rank"), int) or "error" not in shared:
        raise ThreeGpuSmokeError("three-GPU smoke preflight broadcast is invalid")
    if shared["error"] is not None and not isinstance(shared["error"], str):
        raise ThreeGpuSmokeError("three-GPU smoke preflight broadcast has an invalid error")
    return shared


def _raise_if_any_rank_failed(
    accelerator: Any,
    local_failure: Mapping[str, object] | None,
    temporary: Path,
    attempt_id: str,
    *,
    published_output: Path | None = None,
) -> None:
    status = local_failure or {"rank": int(getattr(accelerator, "process_index", 0)), "error": None}
    statuses = _run_collective(
        accelerator,
        "status gather",
        temporary,
        attempt_id,
        lambda: _gather_statuses(accelerator, status),
        published_output=published_output,
    )
    _run_collective(
        accelerator,
        "status barrier",
        temporary,
        attempt_id,
        accelerator.wait_for_everyone,
        published_output=published_output,
    )
    failures = sorted(
        (
            item for item in statuses
            if isinstance(item.get("rank"), int) and isinstance(item.get("error"), str) and item["error"]
        ),
        key=lambda item: int(item["rank"]),
    )
    if not failures:
        return
    _cleanup_smoke_outputs(temporary, attempt_id, published_output)
    _run_collective(
        accelerator,
        "cleanup barrier",
        temporary,
        attempt_id,
        accelerator.wait_for_everyone,
        published_output=published_output,
    )
    first = failures[0]
    raise ThreeGpuSmokeError(f"three-GPU smoke failed on rank {first['rank']}: {first['error']}")


def _gather_statuses(accelerator: Any, status: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    gather_object = getattr(accelerator, "gather_object", None)
    if callable(gather_object):
        gathered = gather_object([dict(status)])
    else:
        from accelerate.utils import gather_object as accelerate_gather_object

        gathered = accelerate_gather_object([dict(status)])
    if not isinstance(gathered, Sequence) or isinstance(gathered, (str, bytes)):
        raise ThreeGpuSmokeError("three-GPU smoke failure status gather is invalid")
    if len(gathered) != 3 or any(not isinstance(item, Mapping) for item in gathered):
        raise ThreeGpuSmokeError("three-GPU smoke must gather one failure status per rank")
    return gathered

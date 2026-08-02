"""Fail-closed two-phase orchestration for the Balalaika CosyVoice3 recipe.

This module is the only Python entry point used by the two operator launchers.
It deliberately contains no publication/upload operation.  Expensive task
implementations are hidden behind a typed backend so the state machine can be
tested without CUDA while the production backend calls Tasks 1--11 directly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import IntEnum
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar, cast

from .artifacts import StageRequirementError, StageStore, sha256_file
from .config import (
    DEFAULT_BASE_MODEL_DIR,
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPOSITORY_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SEED,
    PhaseSpec,
    RunPaths,
)


WORKFLOW_VERSION = "cosyvoice3-balalaika-two-phase-v1"
VALIDATION_GENERATIONS = 2_000
PHASE1_VALIDATION_INDICES = tuple(range(1, 17))
PHASE2_VALIDATION_INDICES = tuple(range(17, 41))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SECRET_WORDS = ("TOKEN", "KEY", "SECRET")


class ExitCode(IntEnum):
    """Stable process results used by shell automation."""

    SUCCESS = 0
    PILOT_REVIEW_REQUIRED = 20


@dataclass(frozen=True)
class WorkflowOptions:
    """Non-secret operator configuration shared by both phases."""

    paths: RunPaths
    approve_pilot_sha256: str | None = None
    token_limit: int = 6_000
    batch_limit: int = 8
    accumulation_steps: int = 1
    wandb_project: str = "cosyvoice3-balalaika-lora"
    wandb_name: str = "cosyvoice3-balalaika"
    keep_eval_audio: bool = False
    keep_checkpoints: int = 3
    resume_checkpoint: Path | None = None
    allow_test_export: bool = False

    def __post_init__(self) -> None:
        if self.approve_pilot_sha256 is not None and _SHA256.fullmatch(self.approve_pilot_sha256) is None:
            raise ValueError("approve_pilot_sha256 must be a lowercase SHA-256 digest")
        if self.token_limit < 1 or self.batch_limit < 1 or self.accumulation_steps < 1:
            raise ValueError("token_limit, batch_limit, and accumulation_steps must be positive")
        if self.keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be positive")
        if type(self.allow_test_export) is not bool:
            raise ValueError("allow_test_export must be boolean")
        if not self.wandb_project.strip() or not self.wandb_name.strip():
            raise ValueError("W&B project and run name must be nonempty")
        if len(self.paths.visible_devices) != 8 or len(set(self.paths.visible_devices)) != 8:
            raise ValueError("Balalaika workflow requires exactly eight unique visible device IDs")


class Coordinator(Protocol):
    """Small, serializable collective boundary backed by one Accelerator."""

    is_main_process: bool
    process_index: int
    num_processes: int

    def broadcast(self, value: object) -> object: ...
    def gather(self, value: object) -> Sequence[object]: ...
    def barrier(self) -> None: ...


class WorkflowBackend(Protocol):
    """Typed Tasks 1--11 service surface consumed by the state machine."""

    coordinator: Coordinator

    def authenticate_stage(self, name: str, payload: dict[str, object]) -> None: ...
    def preflight(self, options: WorkflowOptions) -> dict[str, object]: ...
    def qualify_tokenizer(self, options: WorkflowOptions) -> dict[str, object]: ...
    def ensure_pilot(self, options: WorkflowOptions) -> dict[str, object]: ...
    def approve_pilot(self, options: WorkflowOptions, checksum: str) -> dict[str, object]: ...
    def prepare_memorization(self, options: WorkflowOptions) -> dict[str, object]: ...
    def memorize(self, options: WorkflowOptions) -> dict[str, object]: ...
    def build_cache(self, options: WorkflowOptions) -> dict[str, object]: ...
    def capacity_smoke(self, options: WorkflowOptions) -> dict[str, object]: ...
    def ensure_logging(self, options: WorkflowOptions) -> dict[str, object]: ...
    def evaluate(self, options: WorkflowOptions, validation_index: int, generations: int) -> dict[str, object]: ...
    def train(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        *,
        initial_adapter_sha256: str | None,
        fresh_optimizer: bool,
    ) -> dict[str, object]: ...
    def export(self, options: WorkflowOptions, phase2: dict[str, object]) -> dict[str, object]: ...


class _AccelerateCoordinator:
    def __init__(self, accelerator: Any) -> None:
        self.accelerator = accelerator
        self.is_main_process = bool(accelerator.is_main_process)
        self.process_index = int(accelerator.process_index)
        self.num_processes = int(accelerator.num_processes)

    def broadcast(self, value: object) -> object:
        values = [value if self.is_main_process else None]
        broadcaster = getattr(self.accelerator, "broadcast_object_list", None)
        if callable(broadcaster):
            broadcaster(values, from_process=0)
        else:
            from accelerate.utils import broadcast_object_list

            broadcast_object_list(values, from_process=0)
        return values[0]

    def gather(self, value: object) -> Sequence[object]:
        gatherer = getattr(self.accelerator, "gather_object", None)
        if callable(gatherer):
            gathered = gatherer(value)
        else:
            from accelerate.utils import gather_object

            gathered = gather_object(value)
        if isinstance(gathered, Sequence) and not isinstance(gathered, (str, bytes)):
            return list(gathered)
        return [gathered]

    def barrier(self) -> None:
        self.accelerator.wait_for_everyone()


class ProductionBackend:
    """Production bindings to the concrete task implementations.

    Request manifests are created by preceding tasks and loaded here; missing
    or incomplete artifacts raise :class:`StageRequirementError` rather than
    substituting a successful no-op.
    """

    def __init__(self, options: WorkflowOptions) -> None:
        if options.allow_test_export:
            raise StageRequirementError("ProductionBackend cannot enable test export")
        from accelerate import Accelerator

        self.accelerator = Accelerator(
            gradient_accumulation_steps=options.accumulation_steps,
            mixed_precision="bf16",
            log_with="wandb",
        )
        self.coordinator = _AccelerateCoordinator(self.accelerator)
        self._evaluation_model: Any | None = None
        self._evaluation_checkpoint: Path | None = None
        self._tracker_initialized = False
        self._validation_logger: Any | None = None
        if self.coordinator.num_processes != 8:
            raise StageRequirementError(
                f"workflow must run under Accelerate with exactly eight ranks, got {self.coordinator.num_processes}"
            )

    def authenticate_stage(self, name: str, payload: dict[str, object]) -> None:
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping) or not evidence:
            raise StageRequirementError(f"workflow stage has no authenticated evidence: {name}")
        _verify_evidence_files(evidence, name)
        if name == "pilot_approved":
            from .tokenizer import require_pilot_approval

            require_pilot_approval(self._paths(payload))
        elif name == "cache_complete":
            from .cache import verify_cache

            verify_cache(Path(cast(str, evidence["cache_root"])))
        elif name == "memorization_cache_ready":
            from .memorization import select_memorization_rows

            cache = _load_memorization_cache(Path(cast(str, evidence["cache_root"])))
            select_memorization_rows(cache.root / "split_plan", cache, self._paths(payload).seed)
        elif name == "memorization_complete":
            from .memorization import require_memorization_gate

            manifest = evidence.get("manifest")
            if not isinstance(manifest, str):
                raise StageRequirementError("memorization stage has no sealed manifest")
            require_memorization_gate(Path(manifest).parent)
        elif name in {"phase1_training", "phase1_complete", "phase2_training"}:
            options = _options_from_workflow_payload(payload)
            checkpoint_value = evidence.get("checkpoint")
            if not isinstance(checkpoint_value, str):
                raise StageRequirementError(f"{name} has no checkpoint lineage")
            checkpoint = Path(checkpoint_value)
            checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
            if (
                    not checkpoint_manifest.is_file()
                    or sha256_file(checkpoint_manifest) != evidence.get("checkpoint_manifest_sha256")
            ):
                raise StageRequirementError(f"{name} checkpoint manifest changed")
            checkpoint_payload = _read_mapping(checkpoint_manifest)
            expected_phase = 2 if name == "phase2_training" else 1
            progress = _required_mapping(checkpoint_payload, "progress")
            if (
                    checkpoint_payload.get("validation_status") != "succeeded"
                    or progress.get("phase") != expected_phase
                    or progress.get("validation_index") != (40 if expected_phase == 2 else 16)
            ):
                raise StageRequirementError(f"{name} checkpoint is not validation-complete")
            adapter_dir = _required_path(evidence, "adapter_dir")
            adapter_manifest = _read_mapping(adapter_dir / "adapter_manifest.json")
            weights_name = adapter_manifest.get("weights")
            if not isinstance(weights_name, str):
                raise StageRequirementError(f"{name} adapter manifest has no weights path")
            weights = adapter_dir / weights_name
            if (
                    not weights.is_file()
                    or sha256_file(weights) != evidence.get("adapter_weights_sha256")
                    or adapter_manifest.get("weights_sha256") != evidence.get("adapter_weights_sha256")
            ):
                raise StageRequirementError(f"{name} adapter changed")
            summary = _required_path(evidence, "validation_summary")
            summary_payload = _read_mapping(summary)
            if summary_payload.get("validation_index") != (40 if expected_phase == 2 else 16):
                raise StageRequirementError(f"{name} final validation summary changed")
            expected_indices = PHASE2_VALIDATION_INDICES if expected_phase == 2 else PHASE1_VALIDATION_INDICES
            self._verify_required_validations(options, expected_indices, evidence.get("validations"))
            if expected_phase == 2:
                phase1 = _stage_store(options).require("phase1_complete")
                phase1_evidence = _required_mapping(phase1.payload, "evidence")
                identity = _required_mapping(evidence, "training_identity")
                if identity.get("initial_adapter_sha256") != phase1_evidence.get("adapter_weights_sha256"):
                    raise StageRequirementError("phase-2 checkpoint is not initialized from sealed phase-1 adapter")
        elif name == "validation_00":
            options = _options_from_workflow_payload(payload)
            self._verify_required_validations(options, (0,), [evidence])
        elif name in {"final_export", "complete"}:
            from .model import require_committed_final

            require_committed_final(
                Path(cast(str, evidence["output_dir"])),
                self._paths(payload).base_model_dir,
                expected_mode="production",
            )

    def preflight(self, options: WorkflowOptions) -> dict[str, object]:
        from .config import collect_environment
        from .sources import build_split_plan, inventory_sources
        from .validation_data import MANIFEST_NAME, fetch_validation_rows

        environment = collect_environment()
        inventory = inventory_sources(options.paths)
        split = build_split_plan(options.paths, options.paths.seed)
        validation_dir = options.paths.run_root / "validation_data"
        rows = fetch_validation_rows(validation_dir)
        validation_manifest = validation_dir / MANIFEST_NAME
        return {
            "environment": environment,
            "source_archives": len(inventory.source_archives),
            "split_manifest": str(split.plan_dir / "manifest.json"),
            "split_manifest_sha256": split.manifest_sha256,
            "validation_rows": len(rows),
            "validation_manifest": str(validation_manifest),
            "validation_manifest_sha256": sha256_file(validation_manifest),
        }

    def qualify_tokenizer(self, options: WorkflowOptions) -> dict[str, object]:
        from .tokenizer import run_tokenizer_qualification

        return cast(dict[str, object], run_tokenizer_qualification(options.paths))

    def ensure_pilot(self, options: WorkflowOptions) -> dict[str, object]:
        from .artifacts import StageStore
        from .tokenizer import PilotReviewRequired, build_pilot

        store = StageStore(options.paths.stages_dir)
        try:
            record = store.require("pilot")
        except StageRequirementError as exc:
            if (options.paths.stage("pilot")).exists():
                raise
            try:
                build_pilot(options.paths)
            except PilotReviewRequired as review:
                record = store.require("pilot")
                return {
                    "pilot_manifest": str(record.path),
                    "pilot_manifest_sha256": record.manifest_sha256,
                    "listening_index": str(review.manifest.index_path),
                }
            raise exc
        return {
            "pilot_manifest": str(record.path),
            "pilot_manifest_sha256": record.manifest_sha256,
            "listening_index": str(options.paths.run_root / "pilot/index.md"),
        }

    def approve_pilot(self, options: WorkflowOptions, checksum: str) -> dict[str, object]:
        from .tokenizer import approve_pilot

        path = approve_pilot(options.paths, checksum)
        return {
            "pilot_manifest_sha256": checksum,
            "approval": str(path),
            "approval_sha256": sha256_file(path),
        }

    def memorize(self, options: WorkflowOptions) -> dict[str, object]:
        from .memorization import MemorizationRequest, run_memorization_gate

        cache_root = options.paths.run_root / "memorization_cache"
        cache = _load_memorization_cache(cache_root)
        with _accelerator_scope(self.accelerator):
            report = run_memorization_gate(MemorizationRequest(
                split_plan=cache_root / "split_plan",
                cache=cache,
                output_root=options.paths.run_root,
                base_model_dir=options.paths.base_model_dir,
                seed=options.paths.seed,
                accelerator_factory=lambda **_: self.accelerator,
            ))
        manifest = report.path / "memorization_manifest.json"
        return {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "steps": report.steps}

    def prepare_memorization(self, options: WorkflowOptions) -> dict[str, object]:
        cache = _build_memorization_cache(options)
        manifest = cache.root / "manifest.json"
        return {
            "cache_root": str(cache.root),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "phase1_rows": cache.phase_rows[1],
            "phase2_rows": cache.phase_rows[2],
        }

    def build_cache(self, options: WorkflowOptions) -> dict[str, object]:
        from .tokenizer import run_cache_workers

        cache = run_cache_workers(options.paths)
        return {
            "cache_root": str(cache.root),
            "shards": len(cache.shards),
            "phase1_rows": cache.phase_rows[1],
            "phase2_rows": cache.phase_rows[2],
            "prompt_count": cache.prompt_count,
        }

    def capacity_smoke(self, options: WorkflowOptions) -> dict[str, object]:
        with _accelerator_scope(self.accelerator):
            return self._capacity_smoke_scoped(options)

    def _capacity_smoke_scoped(self, options: WorkflowOptions) -> dict[str, object]:
        # The production-path smoke uses the same request builder as training,
        # but is bounded to one optimizer accumulation and never publishes a
        # phase checkpoint.  It remains collective on all eight ranks.
        from .cache import verify_cache
        from .data import build_phase_dataloader
        from .model import LoraSettings, inject_lora, load_base_llm
        from .training import QualificationRequest, qualify_token_limit
        import torch

        cache = verify_cache(options.paths.run_root / "cache")
        model = inject_lora(load_base_llm(options.paths.base_model_dir), LoraSettings())
        prepared = self.accelerator.prepare(model)

        def probe(candidate: int) -> tuple[int, int]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
            loader = build_phase_dataloader(
                cache, PhaseSpec.for_phase(1), self.accelerator, 0,
                max_tokens_per_gpu=candidate,
            )
            batch = next(iter(loader), None)
            if batch is None:
                raise StageRequirementError("capacity smoke has no eligible phase-1 batch")
            with self.accelerator.autocast():
                result = prepared(batch, self.accelerator.device)
                loss = result[0] if isinstance(result, tuple) else result.get("loss")
            if loss is None or not bool(loss.detach().isfinite().item()):
                raise StageRequirementError("capacity smoke produced a non-finite loss")
            self.accelerator.backward(loss)
            prepared.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.accelerator.device)
            properties = torch.cuda.get_device_properties(self.accelerator.device)
            return int(torch.cuda.max_memory_allocated(self.accelerator.device)), int(properties.total_memory)

        qualification = qualify_token_limit(QualificationRequest(self.accelerator, probe))
        if options.token_limit > qualification.token_limit:
            raise StageRequirementError(
                f"requested token limit {options.token_limit} exceeds qualified {qualification.token_limit}"
            )
        return {
            "requested_token_limit": options.token_limit,
            "qualified_token_limit": qualification.token_limit,
            "candidates": [asdict(item) for item in qualification.candidates],
        }

    def evaluate(self, options: WorkflowOptions, validation_index: int, generations: int) -> dict[str, object]:
        if generations != VALIDATION_GENERATIONS:
            raise ValueError("every validation point must contain exactly 2,000 generations")
        request = self._load_evaluation_request(options, validation_index)
        from .evaluation import evaluate_checkpoint

        if options.keep_eval_audio:
            os.environ["KEEP_EVAL_AUDIO"] = "1"
        report = evaluate_checkpoint(request)
        return {
            "validation_index": report.validation_index,
            "generations": report.row_count,
            "summary": str(report.summary_json),
            "summary_sha256": sha256_file(report.summary_json),
            "identity_sha256": report.identity_checksum,
            "artifact_checksums": dict(report.artifact_checksums),
        }

    def ensure_logging(self, options: WorkflowOptions) -> dict[str, object]:
        self._ensure_tracker(options)
        logger = self._wandb_logger(options)
        return {"logger_class": logger.__class__.__qualname__, "run_manifest": str(options.paths.run_root / "wandb-run.json")}

    def train(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        *,
        initial_adapter_sha256: str | None,
        fresh_optimizer: bool,
    ) -> dict[str, object]:
        if not fresh_optimizer:
            raise ValueError("both phase starts require a freshly constructed optimizer/scheduler")
        return self._run_training(options, phase, validation_indices, initial_adapter_sha256)

    def export(self, options: WorkflowOptions, phase2: dict[str, object]) -> dict[str, object]:
        from .model import ExportRequest, export_final_llm

        checkpoint = _required_path(phase2, "checkpoint")
        validation = _required_path(phase2, "validation_summary")
        request = ExportRequest(
            base_model_dir=options.paths.base_model_dir,
            phase2_checkpoint=checkpoint,
            validation_summary=validation,
            output_dir=options.paths.run_root / "final",
            validation_request=self._load_committed_evaluation_request(options, 40),
            wandb_logger=self._wandb_logger(options),
            expected_training_identity=_required_mapping(phase2, "training_identity"),
        )
        final = export_final_llm(request)
        return {
            "output_dir": str(final.path.parent),
            "manifest": str(final.path),
            "manifest_sha256": sha256_file(final.path),
            "mode": final.mode,
            "production_ready": final.production_ready,
        }

    def _run_training(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        initial_adapter_sha256: str | None,
    ) -> dict[str, object]:
        with _accelerator_scope(self.accelerator):
            try:
                return self._run_training_scoped(
                    options,
                    phase,
                    validation_indices,
                    initial_adapter_sha256,
                )
            finally:
                self._evaluation_model = None
                self._evaluation_checkpoint = None

    def _run_training_scoped(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        initial_adapter_sha256: str | None,
    ) -> dict[str, object]:
        from .cache import verify_cache
        from .data import build_phase_dataloader
        from .model import LoraSettings, inject_lora, load_base_llm
        from .training import TrainRequest, TrainingCallbacks, train_phase

        if phase.number == 2 and initial_adapter_sha256 is None:
            raise StageRequirementError("phase 2 requires the authenticated phase-1 adapter checksum")
        cache = verify_cache(options.paths.run_root / "cache")
        model = inject_lora(load_base_llm(options.paths.base_model_dir), LoraSettings())
        if phase.number == 2:
            self._load_phase1_adapter(model, options, initial_adapter_sha256)
        cache_manifest = options.paths.run_root / "cache/manifest.json"
        if not cache_manifest.is_file():
            raise StageRequirementError("verified cache aggregate manifest is missing")
        validation_evidence: list[dict[str, object]] = []

        def load_resume_evidence() -> None:
            validation_evidence.extend(self._resume_validation_evidence(
                options,
                phase,
                validation_indices,
                options.resume_checkpoint,
            ))

        def validate(event: Any) -> bool:
            expected = validation_indices[len(validation_evidence)]
            if event.validation_index != expected:
                raise StageRequirementError(f"trainer emitted validation {event.validation_index}, expected {expected}")
            self._evaluation_model = event.model
            self._evaluation_checkpoint = Path(event.checkpoint)
            evidence = self.evaluate(options, event.validation_index, VALIDATION_GENERATIONS)
            validation_evidence.append(evidence)
            return True

        request = TrainRequest(
            model=model,
            phase=phase,
            eligible_samples=cache.phase_rows[phase.number],
            cache_checksum=sha256_file(cache_manifest),
            checkpoint_root=options.paths.run_root / f"phase{phase.number}/checkpoints",
            token_limit=options.token_limit,
            accumulation_steps=options.accumulation_steps,
            dataloader_factory=lambda epoch, accelerator: build_phase_dataloader(
                cache, phase, accelerator, epoch, max_tokens_per_gpu=options.token_limit,
            ),
            resume_from=options.resume_checkpoint,
            accelerator_factory=lambda **_: self.accelerator,
            validation_index_base=0 if phase.number == 1 else 16,
            initial_adapter_sha256=initial_adapter_sha256,
        )
        result = train_phase(request, TrainingCallbacks(
            validate=validate,
            after_resume_loaded=load_resume_evidence,
        ))
        if (
            not result.completed
            or [item.get("validation_index") for item in validation_evidence] != list(validation_indices)
            or result.checkpoint is None
        ):
            raise StageRequirementError("phase training ended without its exact validation schedule")
        manifest = result.checkpoint / "checkpoint_manifest.json"
        raw = _read_mapping(manifest)
        adapter_dir = result.checkpoint / "adapter"
        adapter_manifest = _read_mapping(adapter_dir / "adapter_manifest.json")
        if self.coordinator.is_main_process:
            _retain_checkpoints(request.checkpoint_root, result.checkpoint, options.keep_checkpoints)
        self.coordinator.barrier()
        return {
            "completed": True,
            "checkpoint": str(result.checkpoint),
            "checkpoint_manifest_sha256": sha256_file(manifest),
            "adapter_dir": str(adapter_dir),
            "adapter_weights_sha256": adapter_manifest["weights_sha256"],
            "validation_summary": str(options.paths.run_root / f"evaluation/validation-{validation_indices[-1]:02d}/summary.json"),
            "final_validation_index": validation_indices[-1],
            "fresh_optimizer": True,
            "training_identity": raw["identity"],
            "validations": validation_evidence,
        }

    def _resume_validation_evidence(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        resume_checkpoint: Path | None,
    ) -> list[dict[str, object]]:
        if resume_checkpoint is None:
            return []
        received = _main_call(
            self.coordinator,
            "resume validation evidence",
            lambda: self._resume_validation_evidence_main(
                options,
                phase,
                validation_indices,
                resume_checkpoint,
            ),
        )
        if not isinstance(received, list) or any(not isinstance(item, Mapping) for item in received):
            raise StageRequirementError("resume validation evidence broadcast is invalid")
        return [dict(item) for item in received]

    def _resume_validation_evidence_main(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        resume_checkpoint: Path,
    ) -> list[dict[str, object]]:
        manifest = _read_mapping(Path(resume_checkpoint) / "checkpoint_manifest.json")
        status = manifest.get("validation_status")
        if status not in {"pending", "succeeded"}:
            raise StageRequirementError("resume checkpoint validation status is invalid")
        progress = _required_mapping(manifest, "progress")
        current = progress.get("validation_index")
        if (
            progress.get("phase") != phase.number
            or isinstance(current, bool)
            or not isinstance(current, int)
        ):
            raise StageRequirementError("resume checkpoint phase/validation cursor is invalid")
        if current not in validation_indices:
            raise StageRequirementError("resume checkpoint validation cursor is outside the phase schedule")
        last_committed = current if status == "succeeded" else current - 1
        prior = tuple(index for index in validation_indices if index <= last_committed)
        expected = tuple(range(validation_indices[0], last_committed + 1))
        if prior != expected:
            raise StageRequirementError("resume checkpoint skips required validation indices")
        return [self._committed_validation_evidence(options, index) for index in prior]

    def _load_phase1_adapter(self, model: Any, options: WorkflowOptions, expected: str | None) -> None:
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        stage = _stage_store(options).require("phase1_complete")
        evidence = _required_mapping(stage.payload, "evidence")
        adapter_dir = _required_path(evidence, "adapter_dir")
        weights = adapter_dir / "adapter_model.safetensors"
        if expected is None or sha256_file(weights) != expected:
            raise StageRequirementError("phase-1 adapter checksum changed")
        result = set_peft_model_state_dict(model, load_file(str(weights), device="cpu"), adapter_name="default")
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise StageRequirementError("phase-1 adapter tensors do not match the fresh phase-2 model")

    def _load_evaluation_request(self, options: WorkflowOptions, validation_index: int) -> Any:
        import pyarrow.parquet as pq

        from cosyvoice.cli.cosyvoice import CosyVoice3
        from .evaluation import (
            CosyVoiceSynthesizer,
            EvaluationProvenance,
            EvaluationRequest,
            GigaAmRecognizer,
        )
        from .model import checkpoint_identity_sha256, model_state_identity_sha256
        from .validation_data import MANIFEST_NAME, PARQUET_NAME

        validation_root = options.paths.run_root / "validation_data"
        benchmark_parquet = validation_root / PARQUET_NAME
        benchmark_manifest_path = validation_root / MANIFEST_NAME
        cache_root = options.paths.run_root / "cache"
        prompt_parquet = cache_root / "eval_prompts.parquet"
        for path in (benchmark_parquet, benchmark_manifest_path, prompt_parquet):
            if not path.is_file():
                raise StageRequirementError(f"evaluation prerequisite is missing: {path}")
        rows = pq.read_table(benchmark_parquet).to_pylist()
        prompts = pq.read_table(prompt_parquet).to_pylist()
        for prompt in prompts:
            relative = prompt.get("audio_path")
            if not isinstance(relative, str):
                raise StageRequirementError("reserved prompt has no audio path")
            prompt["audio_path"] = cache_root / relative
        benchmark_manifest = _read_mapping(benchmark_manifest_path)
        revision = benchmark_manifest.get("revision")
        if not isinstance(revision, str) or not revision:
            raise StageRequirementError("validation snapshot lacks an immutable revision")

        pipeline = CosyVoice3(str(options.paths.base_model_dir), load_trt=False, load_vllm=False, fp16=False)
        current_llm = self._evaluation_model
        checkpoint_sha256 = sha256_file(options.paths.base_model_dir / "llm.pt")
        model_state_sha256 = checkpoint_sha256
        adapter_sha256 = checkpoint_sha256
        if current_llm is None:
            current_llm = pipeline.model.llm
        if self._evaluation_checkpoint is not None:
            checkpoint_manifest_path = self._evaluation_checkpoint / "checkpoint_manifest.json"
            checkpoint_manifest = _read_mapping(checkpoint_manifest_path)
            state_files = _required_mapping(checkpoint_manifest, "state_files")
            checkpoint_sha256 = checkpoint_identity_sha256(checkpoint_manifest)
            model_state_sha256 = model_state_identity_sha256(state_files)
            adapter_manifest = _read_mapping(self._evaluation_checkpoint / "adapter/adapter_manifest.json")
            adapter_value = adapter_manifest.get("weights_sha256")
            if not isinstance(adapter_value, str) or _SHA256.fullmatch(adapter_value) is None:
                raise StageRequirementError("validation checkpoint adapter checksum is invalid")
            adapter_sha256 = adapter_value

        synthesizer = CosyVoiceSynthesizer(pipeline, current_llm, self.accelerator)
        recognizer = GigaAmRecognizer(
            local_rank=int(self.accelerator.local_process_index),
            max_batch_size=options.batch_limit,
        )
        base_sha256 = sha256_file(options.paths.base_model_dir / "llm.pt")
        provenance = EvaluationProvenance(
            checkpoint_sha256=checkpoint_sha256,
            model_state_sha256=model_state_sha256,
            adapter_sha256=adapter_sha256,
            base_checkpoint_sha256=base_sha256,
            benchmark_snapshot_sha256=sha256_file(benchmark_manifest_path),
            benchmark_revision=revision,
            asr_config=recognizer.provenance(),
            synthesis_config=synthesizer.provenance(),
            code_version=WORKFLOW_VERSION,
            config_version=WORKFLOW_VERSION,
        )
        self._ensure_tracker(options)
        output = options.paths.run_root / f"evaluation/validation-{validation_index:02d}"
        return EvaluationRequest(
            rows=rows,
            prompts=prompts,
            accelerator=self.accelerator,
            synthesizer=synthesizer,
            recognizer=recognizer,
            provenance=provenance,
            validation_index=validation_index,
            output_jsonl=output / "results.jsonl",
            summary_json=output / "summary.json",
            panel_dir=output / "listening-panel",
            temporary_audio_dir=output / "audio",
            assignment_manifest=options.paths.run_root / "evaluation/voice-assignment.json",
            memorization_path=options.paths.run_root / "memorization",
            wandb_logger=self._wandb_logger(options),
            asr_batch_size=options.batch_limit,
        )

    def _load_committed_evaluation_request(self, options: WorkflowOptions, validation_index: int) -> Any:
        import pyarrow.parquet as pq

        from .evaluation import EvaluationProvenance, EvaluationRequest
        from .validation_data import PARQUET_NAME

        validation_root = options.paths.run_root / "validation_data"
        cache_root = options.paths.run_root / "cache"
        rows = pq.read_table(validation_root / PARQUET_NAME).to_pylist()
        prompts = pq.read_table(cache_root / "eval_prompts.parquet").to_pylist()
        for prompt in prompts:
            relative = prompt.get("audio_path")
            if not isinstance(relative, str):
                raise StageRequirementError("reserved prompt has no audio path")
            prompt["audio_path"] = cache_root / relative
        output = options.paths.run_root / f"evaluation/validation-{validation_index:02d}"
        summary = _read_mapping(output / "summary.json")
        identity = _required_mapping(summary, "evaluation_identity")
        if identity.get("validation_index") != validation_index:
            raise StageRequirementError(f"validation-{validation_index:02d} identity index changed")
        provenance = EvaluationProvenance(
            checkpoint_sha256=cast(str, identity.get("checkpoint_sha256")),
            model_state_sha256=cast(str, identity.get("model_state_sha256")),
            adapter_sha256=cast(str, identity.get("adapter_sha256")),
            base_checkpoint_sha256=cast(str, identity.get("base_checkpoint_sha256")),
            benchmark_snapshot_sha256=cast(str, identity.get("benchmark_snapshot_sha256")),
            benchmark_revision=cast(str, identity.get("benchmark_revision")),
            asr_config=_required_mapping(identity, "asr_config"),
            synthesis_config=_required_mapping(identity, "synthesis_config"),
            code_version=cast(str, identity.get("code_version")),
            config_version=cast(str, identity.get("config_version")),
        )
        if identity.get("asr_batch_size") != options.batch_limit:
            raise StageRequirementError(f"validation-{validation_index:02d} ASR batch identity changed")
        return EvaluationRequest(
            rows=rows,
            prompts=prompts,
            accelerator=self.accelerator,
            synthesizer=cast(Any, object()),
            recognizer=cast(Any, object()),
            provenance=provenance,
            validation_index=validation_index,
            output_jsonl=output / "results.jsonl",
            summary_json=output / "summary.json",
            panel_dir=output / "listening-panel",
            temporary_audio_dir=output / "audio",
            assignment_manifest=options.paths.run_root / "evaluation/voice-assignment.json",
            memorization_path=options.paths.run_root / "memorization",
            wandb_logger=self._wandb_logger(options),
            asr_batch_size=options.batch_limit,
        )

    def _verify_required_validations(
        self,
        options: WorkflowOptions,
        indices: Sequence[int],
        recorded: object,
    ) -> None:
        if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
            raise StageRequirementError("phase validation evidence list is missing")
        by_index: dict[int, Mapping[str, object]] = {}
        for item in recorded:
            if not isinstance(item, Mapping):
                raise StageRequirementError("phase validation evidence entry is invalid")
            index = item.get("validation_index")
            if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
                raise StageRequirementError("phase validation evidence indices are invalid")
            by_index[index] = item
        if set(by_index) != set(indices):
            raise StageRequirementError("phase validation evidence does not cover the exact required indices")
        for index in indices:
            committed = self._committed_validation_evidence(options, index)
            expected = by_index[index]
            if (
                committed.get("identity_sha256") != expected.get("identity_sha256")
                or committed.get("artifact_checksums") != expected.get("artifact_checksums")
                or committed.get("generations") != VALIDATION_GENERATIONS
                or expected.get("generations") != VALIDATION_GENERATIONS
            ):
                raise StageRequirementError(f"validation-{index:02d} evidence changed")

    def _committed_validation_evidence(
        self,
        options: WorkflowOptions,
        index: int,
    ) -> dict[str, object]:
        from .evaluation import verify_committed_evaluation

        request = self._load_committed_evaluation_request(options, index)
        committed = verify_committed_evaluation(
            request,
            wandb_logger=self._wandb_logger(options),
        )
        report = committed.report
        return {
            "validation_index": report.validation_index,
            "generations": report.row_count,
            "summary": str(report.summary_json),
            "summary_sha256": sha256_file(report.summary_json),
            "identity_sha256": report.identity_checksum,
            "artifact_checksums": dict(report.artifact_checksums),
        }

    def _ensure_tracker(self, options: WorkflowOptions) -> None:
        if self._tracker_initialized:
            return
        from .artifacts import atomic_write_json
        from .evaluation import WandbValidationLogger

        run_manifest = options.paths.run_root / "wandb-run.json"
        init_kwargs: dict[str, dict[str, str]] = {"wandb": {"name": options.wandb_name}}
        if run_manifest.is_file():
            init_kwargs = WandbValidationLogger.resume_init_kwargs(run_manifest)
            init_kwargs["wandb"]["name"] = options.wandb_name
        self.accelerator.init_trackers(
            options.wandb_project,
            config=cast(dict[str, object], redact_secrets(_workflow_identity(options))),
            init_kwargs=init_kwargs,
        )
        if self.coordinator.is_main_process:
            run = self.accelerator.get_tracker("wandb", unwrap=True)
            run_id = getattr(run, "id", None)
            if not isinstance(run_id, str) or not run_id:
                raise StageRequirementError("W&B tracker has no stable run ID")
            if run_manifest.is_file():
                existing = _read_mapping(run_manifest)
                if existing != {"format_version": 1, "run_id": run_id}:
                    raise StageRequirementError("W&B tracker resumed a different run ID")
            else:
                atomic_write_json(run_manifest, {"format_version": 1, "run_id": run_id})
        self.coordinator.barrier()
        self._tracker_initialized = True

    def _wandb_logger(self, options: WorkflowOptions) -> Any:
        from .evaluation import WandbValidationLogger

        if self._validation_logger is None:
            self._validation_logger = WandbValidationLogger(
                self.accelerator,
                options.paths.run_root / "wandb-run.json",
            )
        return self._validation_logger

    def _paths(self, payload: Mapping[str, object]) -> RunPaths:
        config = _required_mapping(payload, "workflow")
        return RunPaths(
            dataset_root=Path(cast(str, config["dataset_root"])),
            repository_root=Path(cast(str, config["repository_root"])),
            run_root=Path(cast(str, config["run_root"])),
            base_model_dir=Path(cast(str, config["base_model_dir"])),
            visible_devices=tuple(cast(list[int], config["visible_devices"])),
            seed=int(cast(int, config["seed"])),
        )


@contextmanager
def _accelerator_scope(accelerator: Any) -> Any:
    """Give one workflow stage exclusive ownership of prepared Accelerate state."""

    def clear() -> None:
        free_memory = getattr(accelerator, "free_memory", None)
        custom_objects = getattr(accelerator, "_custom_objects", None)
        if not callable(free_memory) or not isinstance(custom_objects, list):
            raise StageRequirementError("Accelerator does not expose the required lifecycle state")
        free_memory()
        # Accelerate 1.12 clears prepared models, optimizers, schedulers, and
        # dataloaders but leaves registered checkpoint objects behind.
        custom_objects.clear()

    clear()
    try:
        yield accelerator
    finally:
        clear()


def _build_memorization_cache(options: WorkflowOptions) -> Any:
    """Tokenize only four deterministic, duration-varied rows before full cache work."""

    import hashlib
    import heapq
    import shutil
    import tempfile

    import pyarrow.parquet as pq

    from .artifacts import atomic_write_json
    from .cache import CacheManifest, _decode_audio, _phase_table, iter_tar_audio
    from .tokenizer import OnnxSpeechTokenizer

    target = options.paths.run_root / "memorization_cache"
    if target.exists() or target.is_symlink():
        return _load_memorization_cache(target)
    plan_root = options.paths.run_root / "split_plan"
    if not plan_root.is_dir():
        raise StageRequirementError("split plan is missing before memorization subset selection")

    # Retaining a small hash sample avoids holding the four-million-row plan in
    # memory while still giving each phase enough deterministic duration
    # candidates to select distinct short and long examples.
    candidates: dict[int, list[tuple[int, str, dict[str, object]]]] = {1: [], 2: []}
    for plan_path in sorted(plan_root.glob("shard_*.jsonl")):
        with plan_path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StageRequirementError(f"invalid split row: {plan_path}:{number}") from exc
                if not isinstance(row, dict) or row.get("phase") not in (1, 2) or row.get("reserved") is not False:
                    continue
                if row.get("model_limit_exclusion") is not None:
                    continue
                source = row.get("source_relative_path")
                if not isinstance(source, str) or not source:
                    raise StageRequirementError(f"eligible split row has no identity: {plan_path}:{number}")
                phase = int(cast(int, row["phase"]))
                score = int.from_bytes(hashlib.sha256(f"{options.paths.seed}\0{source}".encode()).digest(), "big")
                entry = (-score, source, row)
                heap = candidates[phase]
                if len(heap) < 32:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    if any(len(values) < 2 for values in candidates.values()):
        raise StageRequirementError("both agreement phases need at least two memorization candidates")

    candidate_rows = {
        source: row
        for values in candidates.values()
        for _, source, row in values
    }
    by_shard: dict[int, set[str]] = {}
    for source in candidate_rows:
        try:
            shard = int(source.split("/", 1)[0])
        except (ValueError, IndexError) as exc:
            raise StageRequirementError(f"memorization source has invalid shard identity: {source}") from exc
        by_shard.setdefault(shard, set()).add(source)
    decoded: dict[str, Any] = {}
    for shard, wanted in sorted(by_shard.items()):
        archive = options.paths.dataset_root / "train" / f"shard_{shard:06d}.tar"
        if not archive.is_file():
            raise StageRequirementError(f"memorization source archive is missing: {archive}")
        for sample in iter_tar_audio(archive):
            if sample.source_relative_path in wanted:
                decoded[sample.source_relative_path] = _decode_audio(sample)
                if wanted <= decoded.keys():
                    break
    if set(decoded) != set(candidate_rows):
        missing = sorted(set(candidate_rows) - set(decoded))
        raise StageRequirementError(f"memorization candidates are missing from source archives: {missing[:4]}")

    selected: list[dict[str, object]] = []
    for phase in (1, 2):
        values = [
            (decoded[source].frames / decoded[source].sample_rate, source, row)
            for _, source, row in candidates[phase]
        ]
        values.sort(key=lambda item: (item[0], item[1]))
        short, long = values[0], values[-1]
        if short[0] == long[0]:
            raise StageRequirementError(f"phase {phase} memorization candidates lack varied durations")
        selected.extend((short[2], long[2]))

    tokenizer = OnnxSpeechTokenizer.for_paths(options.paths, local_rank=0)
    rows_by_phase: dict[int, list[dict[str, object]]] = {1: [], 2: []}
    for row in selected:
        source = cast(str, row["source_relative_path"])
        tokens = tokenizer.extract([decoded[source]])[0]
        phase = int(cast(int, row["phase"]))
        rows_by_phase[phase].append({
            "source_relative_path": source,
            "text": row["text"],
            "instruct": row["instruct"],
            "agreement": row["agreement"],
            "speech_token": tokens,
            "speech_token_len": len(tokens),
        })
    if any(len(rows_by_phase[phase]) != 2 for phase in (1, 2)):
        raise StageRequirementError("memorization subset must contain exactly two rows per phase")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".memorization_cache.", dir=target.parent))
    try:
        (temporary / "phase1").mkdir()
        (temporary / "phase2").mkdir()
        (temporary / "split_plan").mkdir()
        for phase in (1, 2):
            pq.write_table(_phase_table(rows_by_phase[phase]), temporary / f"phase{phase}/shard_000000.parquet", compression="zstd")
        plan_lines = "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in selected
        )
        (temporary / "split_plan/shard_000000.jsonl").write_text(plan_lines, encoding="utf-8")
        shard_manifest = temporary / "shard_manifest.json"
        atomic_write_json(shard_manifest, {
            "format_version": 1,
            "rows": 4,
            "phase1_sha256": sha256_file(temporary / "phase1/shard_000000.parquet"),
            "phase2_sha256": sha256_file(temporary / "phase2/shard_000000.parquet"),
            "plan_sha256": sha256_file(temporary / "split_plan/shard_000000.jsonl"),
        })
        atomic_write_json(temporary / "manifest.json", {
            "format_version": 1,
            "shard_manifest": "shard_manifest.json",
            "shard_manifest_sha256": sha256_file(shard_manifest),
            "phase_rows": {"1": 2, "2": 2},
            "seed": options.paths.seed,
        })
        cache = CacheManifest(temporary, {0: shard_manifest}, {1: 2, 2: 2}, 0)
        from .memorization import select_memorization_rows

        select_memorization_rows(temporary / "split_plan", cache, options.paths.seed)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load_memorization_cache(target)


def _load_memorization_cache(root: Path) -> Any:
    from .cache import CacheManifest

    manifest = _read_mapping(root / "manifest.json")
    if manifest.get("format_version") != 1 or manifest.get("phase_rows") != {"1": 2, "2": 2}:
        raise StageRequirementError("memorization cache manifest is invalid")
    shard = root / cast(str, manifest.get("shard_manifest"))
    digest = manifest.get("shard_manifest_sha256")
    if not isinstance(digest, str) or not shard.is_file() or sha256_file(shard) != digest:
        raise StageRequirementError("memorization cache shard manifest changed")
    for phase in (1, 2):
        path = root / f"phase{phase}/shard_000000.parquet"
        if not path.is_file():
            raise StageRequirementError(f"memorization phase-{phase} cache is missing")
    if not (root / "split_plan/shard_000000.jsonl").is_file():
        raise StageRequirementError("memorization split plan is missing")
    return CacheManifest(root, {0: shard}, {1: 2, 2: 2}, 0)


def _retain_checkpoints(root: Path, current: Path, keep: int) -> None:
    """Retain the newest authenticated checkpoints, always including current."""

    import shutil

    if keep < 1 or not root.is_dir():
        return
    completed: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        manifest_path = path / "checkpoint_manifest.json"
        try:
            manifest = _read_mapping(manifest_path)
            progress = _required_mapping(manifest, "progress")
            index = progress.get("validation_index")
        except StageRequirementError:
            continue
        if manifest.get("validation_status") == "succeeded" and isinstance(index, int) and not isinstance(index, bool):
            completed.append((index, path))
    retained = {path.resolve() for _, path in sorted(completed, reverse=True)[:keep]}
    retained.add(current.resolve())
    for _, path in completed:
        if path.resolve() not in retained:
            shutil.rmtree(path)


T = TypeVar("T")


def _main_call(coordinator: Coordinator, label: str, operation: Callable[[], T]) -> T:
    status: dict[str, object] | None = None
    if coordinator.is_main_process:
        try:
            status = {"ok": True, "value": operation()}
        except Exception as exc:
            status = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
    received = coordinator.broadcast(status)
    coordinator.barrier()
    return _unwrap_status(label, received)


def _collective_call(coordinator: Coordinator, label: str, operation: Callable[[], T]) -> T:
    try:
        local: dict[str, object] = {"ok": True, "value": operation()}
    except Exception as exc:
        local = {"ok": False, "rank": coordinator.process_index, "type": type(exc).__name__, "error": str(exc)}
    gathered = coordinator.gather(local)
    failures = [value for value in gathered if isinstance(value, Mapping) and value.get("ok") is not True]
    coordinator.barrier()
    if failures:
        first = cast(Mapping[str, object], failures[0])
        _raise_remote(label, str(first.get("type", "RuntimeError")), str(first.get("error", "collective failure")))
    return cast(T, local["value"])


def _unwrap_status(label: str, value: object) -> Any:
    if not isinstance(value, Mapping) or value.get("ok") not in (True, False):
        raise RuntimeError(f"{label}: invalid main-rank synchronization payload")
    if value["ok"] is not True:
        _raise_remote(label, str(value.get("type", "RuntimeError")), str(value.get("error", "operation failed")))
    return value.get("value")


def _raise_remote(label: str, error_type: str, error: str) -> None:
    message = f"{label}: {error}"
    if error_type == "StageRequirementError":
        raise StageRequirementError(message)
    raise RuntimeError(message)


def _stage_store(options: WorkflowOptions) -> StageStore:
    return StageStore(
        options.paths.run_root / "workflow_stages",
        dependency_lock={"workflow_version": WORKFLOW_VERSION},
        input_provenance=_workflow_identity(options),
    )


def _workflow_identity(options: WorkflowOptions) -> dict[str, object]:
    paths = options.paths
    return {
        "dataset_root": str(paths.dataset_root),
        "repository_root": str(paths.repository_root),
        "run_root": str(paths.run_root),
        "base_model_dir": str(paths.base_model_dir),
        "visible_devices": list(paths.visible_devices),
        "seed": paths.seed,
        "token_limit": options.token_limit,
        "batch_limit": options.batch_limit,
        "accumulation_steps": options.accumulation_steps,
        "wandb_project": options.wandb_project,
        "wandb_name": options.wandb_name,
        "keep_eval_audio": options.keep_eval_audio,
        "keep_checkpoints": options.keep_checkpoints,
        "allow_test_export": options.allow_test_export,
    }


def _options_from_workflow_payload(payload: Mapping[str, object]) -> WorkflowOptions:
    value = _required_mapping(payload, "workflow")
    paths = RunPaths(
        dataset_root=Path(cast(str, value.get("dataset_root"))),
        repository_root=Path(cast(str, value.get("repository_root"))),
        run_root=Path(cast(str, value.get("run_root"))),
        base_model_dir=Path(cast(str, value.get("base_model_dir"))),
        visible_devices=tuple(cast(Sequence[int], value.get("visible_devices"))),
        seed=cast(int, value.get("seed")),
    )
    return WorkflowOptions(
        paths=paths,
        token_limit=cast(int, value.get("token_limit")),
        batch_limit=cast(int, value.get("batch_limit")),
        accumulation_steps=cast(int, value.get("accumulation_steps")),
        wandb_project=cast(str, value.get("wandb_project")),
        wandb_name=cast(str, value.get("wandb_name")),
        keep_eval_audio=cast(bool, value.get("keep_eval_audio")),
        keep_checkpoints=cast(int, value.get("keep_checkpoints")),
        allow_test_export=cast(bool, value.get("allow_test_export", False)),
    )


def _stage_payload(options: WorkflowOptions, evidence: Mapping[str, object]) -> dict[str, object]:
    return {"workflow": _workflow_identity(options), "evidence": dict(evidence)}


def _ensure_stage(
    options: WorkflowOptions,
    backend: WorkflowBackend,
    name: str,
    *,
    collective: bool,
    operation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    store = _stage_store(options)

    def existing() -> dict[str, object] | None:
        path = store.root / f"{name}.json"
        if not path.exists():
            return None
        record = store.require(name)
        payload = dict(record.payload)
        backend.authenticate_stage(name, payload)
        return dict(_required_mapping(payload, "evidence"))

    prior = _main_call(backend.coordinator, f"require {name}", existing)
    if prior is not None:
        return cast(dict[str, object], prior)

    runner = _collective_call if collective else _main_call
    evidence = runner(backend.coordinator, name, operation)
    if not isinstance(evidence, Mapping) or not evidence:
        raise StageRequirementError(f"{name}: operation returned no evidence")

    def publish() -> dict[str, object]:
        payload = _stage_payload(options, evidence)
        backend.authenticate_stage(name, payload)
        store.publish(name, payload)
        return dict(evidence)

    return cast(dict[str, object], _main_call(backend.coordinator, f"publish {name}", publish))


def run_phase1(args: argparse.Namespace) -> int:
    """Run or resume phase 1, stopping at the checksum-bound listening gate."""

    options, backend = _resolve(args)
    _ensure_stage(options, backend, "preflight_complete", collective=False, operation=lambda: backend.preflight(options))
    _ensure_stage(
        options, backend, "tokenizer_qualified", collective=False,
        operation=lambda: backend.qualify_tokenizer(options),
    )
    pilot = _ensure_stage(options, backend, "pilot_ready", collective=False, operation=lambda: backend.ensure_pilot(options))
    pilot_sha256 = pilot.get("pilot_manifest_sha256")
    if not isinstance(pilot_sha256, str) or _SHA256.fullmatch(pilot_sha256) is None:
        raise StageRequirementError("pilot evidence has no valid manifest checksum")
    if options.approve_pilot_sha256 is None:
        if backend.coordinator.is_main_process:
            print(json.dumps({
                "status": "pilot_review_required",
                "pilot_manifest_sha256": pilot_sha256,
                "listening_index": pilot.get("listening_index"),
            }, sort_keys=True))
        return int(ExitCode.PILOT_REVIEW_REQUIRED)
    if options.approve_pilot_sha256 != pilot_sha256:
        raise StageRequirementError("provided pilot approval checksum does not match the current pilot")
    _ensure_stage(
        options, backend, "pilot_approved", collective=False,
        operation=lambda: backend.approve_pilot(options, options.approve_pilot_sha256 or ""),
    )
    _ensure_stage(
        options, backend, "memorization_cache_ready", collective=False,
        operation=lambda: backend.prepare_memorization(options),
    )
    _ensure_stage(options, backend, "memorization_complete", collective=True, operation=lambda: backend.memorize(options))
    _ensure_stage(options, backend, "cache_complete", collective=False, operation=lambda: backend.build_cache(options))
    _ensure_stage(options, backend, "capacity_smoke_complete", collective=True, operation=lambda: backend.capacity_smoke(options))
    _collective_call(backend.coordinator, "initialize W&B logging", lambda: backend.ensure_logging(options))
    _ensure_stage(
        options, backend, "validation_00", collective=True,
        operation=lambda: backend.evaluate(options, 0, VALIDATION_GENERATIONS),
    )
    trained = _ensure_stage(
        options, backend, "phase1_training", collective=True,
        operation=lambda: backend.train(
            options, PhaseSpec.for_phase(1), PHASE1_VALIDATION_INDICES,
            initial_adapter_sha256=None, fresh_optimizer=True,
        ),
    )
    _validate_phase_result(trained, 1, 16)
    _ensure_stage(options, backend, "phase1_complete", collective=False, operation=lambda: dict(trained))
    return int(ExitCode.SUCCESS)


def run_phase2(args: argparse.Namespace) -> int:
    """Run phase 2 from only the sealed phase-1 adapter, then export locally."""

    options, backend = _resolve(args)
    _collective_call(backend.coordinator, "resume W&B logging", lambda: backend.ensure_logging(options))
    phase1 = _require_stage(options, backend, "phase1_complete")
    _require_stage(options, backend, "validation_00")
    _validate_phase_result(phase1, 1, 16)
    adapter_sha256 = phase1.get("adapter_weights_sha256")
    if not isinstance(adapter_sha256, str) or _SHA256.fullmatch(adapter_sha256) is None:
        raise StageRequirementError("phase1_complete lacks a valid adapter checksum")
    trained = _ensure_stage(
        options, backend, "phase2_training", collective=True,
        operation=lambda: backend.train(
            options, PhaseSpec.for_phase(2), PHASE2_VALIDATION_INDICES,
            initial_adapter_sha256=adapter_sha256, fresh_optimizer=True,
        ),
    )
    _validate_phase_result(trained, 2, 40)
    phase2_identity = _required_mapping(trained, "training_identity")
    if phase2_identity.get("initial_adapter_sha256") != adapter_sha256:
        raise StageRequirementError("phase 2 training identity differs from sealed phase-1 adapter")
    exported = _ensure_stage(
        options, backend, "final_export", collective=False,
        operation=lambda: backend.export(options, trained),
    )
    production_export = exported.get("production_ready") is True and exported.get("mode") == "production"
    authenticated_test_export = (
        options.allow_test_export
        and exported.get("production_ready") is False
        and exported.get("mode") == "test"
    )
    if not production_export and not authenticated_test_export:
        raise StageRequirementError("Task 11 final export is neither production-ready nor an authorized test export")
    _ensure_stage(options, backend, "complete", collective=False, operation=lambda: dict(exported))
    return int(ExitCode.SUCCESS)


def _require_stage(options: WorkflowOptions, backend: WorkflowBackend, name: str) -> dict[str, object]:
    store = _stage_store(options)

    def require() -> dict[str, object]:
        record = store.require(name)
        payload = dict(record.payload)
        backend.authenticate_stage(name, payload)
        return dict(_required_mapping(payload, "evidence"))

    return cast(dict[str, object], _main_call(backend.coordinator, f"require {name}", require))


def _validate_phase_result(result: Mapping[str, object], phase: int, final_index: int) -> None:
    if result.get("completed") is not True or result.get("fresh_optimizer") is not True:
        raise StageRequirementError(f"phase {phase} did not complete with a fresh optimizer contract")
    if result.get("final_validation_index") != final_index:
        raise StageRequirementError(f"phase {phase} lacks sealed validation index {final_index}")
    for name in ("checkpoint_manifest_sha256", "adapter_weights_sha256"):
        if not isinstance(value := result.get(name), str) or _SHA256.fullmatch(value) is None:
            raise StageRequirementError(f"phase {phase} lacks valid {name} evidence")
    expected_indices = PHASE1_VALIDATION_INDICES if phase == 1 else PHASE2_VALIDATION_INDICES
    validations = result.get("validations")
    if (
        not isinstance(validations, Sequence)
        or isinstance(validations, (str, bytes))
        or [item.get("validation_index") if isinstance(item, Mapping) else None for item in validations]
        != list(expected_indices)
    ):
        raise StageRequirementError(f"phase {phase} lacks its exact validation evidence sequence")
    identity = result.get("training_identity")
    if not isinstance(identity, Mapping) or identity.get("phase") != phase:
        raise StageRequirementError(f"phase {phase} lacks immutable training identity")
    if phase == 1 and identity.get("initial_adapter_sha256") is not None:
        raise StageRequirementError("phase 1 training identity is not a fresh adapter")


def _resolve(args: argparse.Namespace) -> tuple[WorkflowOptions, WorkflowBackend]:
    options = getattr(args, "options", None)
    if not isinstance(options, WorkflowOptions):
        options = _options_from_namespace(args)
    backend = getattr(args, "backend", None)
    if backend is None:
        backend = ProductionBackend(options)
    return options, cast(WorkflowBackend, backend)


def redact_secrets(value: object, key: str = "") -> object:
    """Recursively replace string or structured values under semantic secret keys."""

    if _has_secret_identifier_segment(key) and _is_secret_value(value):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): redact_secrets(item, str(name)) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [redact_secrets(item) for item in value]
    return value


def _has_secret_identifier_segment(key: str) -> bool:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    segments = re.findall(r"[A-Za-z0-9]+", camel_split.upper())
    return any(segment in _SECRET_WORDS for segment in segments)


def _is_secret_value(value: object) -> bool:
    return (
        isinstance(value, (str, bytes, Mapping))
        or isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    )


def _status(args: argparse.Namespace) -> int:
    options = _options_from_namespace(args)
    stages: dict[str, object] = {}
    root = options.paths.run_root / "workflow_stages"
    store = _stage_store(options)
    for name in (
        "preflight_complete", "tokenizer_qualified", "pilot_ready", "pilot_approved",
        "memorization_cache_ready", "memorization_complete", "cache_complete", "capacity_smoke_complete", "validation_00",
        "phase1_training", "phase1_complete", "phase2_training", "final_export", "complete",
    ):
        path = root / f"{name}.json"
        state: dict[str, object] = {"present": path.is_file(), "path": str(path), "envelope_valid": False}
        if path.is_file():
            try:
                record = store.require(name)
                state["envelope_valid"] = True
                state["manifest_sha256"] = record.manifest_sha256
            except StageRequirementError as exc:
                state["error"] = str(exc)
        stages[name] = state
    output = {"workflow": _workflow_identity(options), "stages": stages}
    print(json.dumps(redact_secrets(output), ensure_ascii=False, sort_keys=True, indent=2))
    return int(ExitCode.SUCCESS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-phase CosyVoice3 Balalaika LoRA workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("phase1", "phase2", "status"):
        item = subparsers.add_parser(command)
        _common_arguments(item)
        if command == "phase1":
            item.add_argument("--approve-pilot-sha256", type=_sha256_argument)
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("BALALAIKA_DATASET_ROOT", DEFAULT_DATASET_ROOT)))
    parser.add_argument("--repository-root", type=Path, default=Path(os.environ.get("BALALAIKA_REPOSITORY_ROOT", DEFAULT_REPOSITORY_ROOT)))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("BALALAIKA_RUN_ROOT", DEFAULT_RUN_ROOT)))
    parser.add_argument("--base-model-dir", type=Path, default=Path(os.environ.get("BALALAIKA_BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)))
    parser.add_argument("--visible-devices", default=os.environ.get("BALALAIKA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--token-limit", type=_positive_int, default=6_000)
    parser.add_argument("--batch-limit", type=_positive_int, default=8)
    parser.add_argument("--accumulation-steps", type=_positive_int, default=1)
    parser.add_argument("--wandb-project", default="cosyvoice3-balalaika-lora")
    parser.add_argument("--wandb-name", default="cosyvoice3-balalaika")
    parser.add_argument("--keep-eval-audio", action="store_true")
    parser.add_argument("--keep-checkpoints", type=_positive_int, default=3)
    parser.add_argument("--resume-checkpoint", type=Path)


def _options_from_namespace(args: argparse.Namespace) -> WorkflowOptions:
    devices = _visible_devices(args.visible_devices)
    base_model = Path(args.base_model_dir)
    if any("_RL" in component.upper() for component in base_model.parts):
        raise ValueError("base model must select the non-RL checkpoint")
    paths = RunPaths(
        dataset_root=Path(args.dataset_root), repository_root=Path(args.repository_root),
        run_root=Path(args.run_root), base_model_dir=base_model,
        visible_devices=devices, seed=int(args.seed),
    )
    return WorkflowOptions(
        paths=paths,
        approve_pilot_sha256=getattr(args, "approve_pilot_sha256", None),
        token_limit=args.token_limit, batch_limit=args.batch_limit,
        accumulation_steps=args.accumulation_steps,
        wandb_project=args.wandb_project, wandb_name=args.wandb_name,
        keep_eval_audio=args.keep_eval_audio, keep_checkpoints=args.keep_checkpoints,
        resume_checkpoint=args.resume_checkpoint,
        allow_test_export=False,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return value


def _visible_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("visible devices must be comma-separated integer IDs") from exc
    if len(devices) != 8 or len(set(devices)) != 8 or any(item < 0 for item in devices):
        raise ValueError("visible devices must contain exactly eight unique non-negative IDs")
    return devices


def _verify_evidence_files(evidence: Mapping[str, object], name: str) -> None:
    for key, value in evidence.items():
        if not key.endswith("_sha256") or not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            continue
        path_key = key.removesuffix("_sha256")
        path_value = evidence.get(path_key)
        if isinstance(path_value, str):
            path = Path(path_value)
            if not path.is_file() or sha256_file(path) != value:
                raise StageRequirementError(f"{name} evidence changed: {path_key}")


def _required_mapping(value: Mapping[str, object], name: str) -> dict[str, object]:
    item = value.get(name)
    if not isinstance(item, Mapping):
        raise StageRequirementError(f"missing mapping: {name}")
    return dict(item)


def _required_path(value: Mapping[str, object], name: str) -> Path:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise StageRequirementError(f"missing path: {name}")
    return Path(item)


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageRequirementError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise StageRequirementError(f"manifest is not a mapping: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one internal command and return a stable exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            return _status(args)
        if args.command == "phase1":
            return run_phase1(args)
        if args.command == "phase2":
            return run_phase2(args)
        parser.error(f"unknown command: {args.command}")
    except (StageRequirementError, ValueError) as exc:
        parser.error(str(exc))
    return int(ExitCode.SUCCESS)


if __name__ == "__main__":
    raise SystemExit(main())

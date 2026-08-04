"""CPU-only integration harness driven by the real two-phase workflow."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
import io
from itertools import islice
import json
import os
from pathlib import Path
import tarfile
import types
from typing import Any, Iterable, Mapping, Sequence
from unittest import mock
import wave

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from cosyvoice.finetune.balalaika import cache as cache_api
from cosyvoice.finetune.balalaika import evaluation as evaluation_api
from cosyvoice.finetune.balalaika import model as model_api
from cosyvoice.finetune.balalaika import sources as sources_api
from cosyvoice.finetune.balalaika.artifacts import StageRequirementError, StageStore, atomic_write_json, sha256_file
from cosyvoice.finetune.balalaika.cache import AudioInput, CacheManifest, CacheShardRequest
from cosyvoice.finetune.balalaika.config import PhaseSpec, RunPaths
from cosyvoice.finetune.balalaika.data import CachedSpeechDataset, CosyVoice3Collator
from cosyvoice.finetune.balalaika.memorization import MemorizationRequest, require_memorization_gate, run_memorization_gate
from cosyvoice.finetune.balalaika.metrics import NumberSpan
from cosyvoice.finetune.balalaika.model import ExportRequest, LoraSettings
from cosyvoice.finetune.balalaika.sources import JoinedRow, assign_phases, reserve_prompt_ids
from cosyvoice.finetune.balalaika.training import TrainRequest, TrainingCallbacks, train_phase
from cosyvoice.finetune.balalaika.workflow import (
    ExitCode,
    WorkflowOptions,
    run_phase1_memorize,
    run_phase1_train,
    run_prepare_cache,
    run_phase2,
)
from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder


@dataclass(frozen=True)
class TinyWorkflowResult:
    stage: str
    phase1_first_exit: int
    phase1_approved_exit: int
    prepare_cache_exit: int
    phase1_train_exit: int
    phase2_exit: int
    workflow_calls: tuple[str, ...]
    final_llm: Path
    final_manifest: Path
    final_phase2_checkpoint: Path
    phase2_checkpoint: Path
    strict_load_verified: bool
    validation_indices: tuple[int, ...]
    phase_epochs: tuple[int, int]
    boundaries_per_epoch: int
    pilot_stopped_before_approval: bool
    pilot_checksum_approved: bool
    memorization_passed: bool
    memorization_initially_exact: bool
    memorization_optimizer_steps: int
    memorization_evidence_verified: bool
    cache_resumed_after_interruption: bool
    phase_resumed_after_interruption: bool
    phase2_loaded_sealed_phase1_adapter: bool
    evaluate_checkpoint_calls: int
    evaluation_report_row_counts: tuple[int, ...]
    stage_evidence_row_counts: tuple[int, ...]
    wandb_mode: str
    upload_calls: int


class _Coordinator:
    is_main_process = True
    process_index = 0
    num_processes = 8

    def broadcast(self, value: object) -> object:
        return value

    def gather(self, value: object) -> list[object]:
        return [value] * self.num_processes

    def barrier(self) -> None:
        return None


class _TinySpeechTokenizer:
    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        return [[index % 6 for index in range(1, sample.frames % 4 + 2)] for sample in audio]


class _TinyTextTokenizer:
    def encode(self, value: str, allowed_special: str = "all") -> list[int]:
        del allowed_special
        return [1 + (ord(character) % 29) for character in value]


class _CpuAccelerator:
    """One CPU process representing the fixture's eight collective ranks."""

    def __init__(self, **kwargs: object) -> None:
        self.gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))
        self.num_processes = 8
        self.process_index = 0
        self.local_process_index = 0
        self.is_main_process = True
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self._checkpointables: list[object] = []
        self._prepared: tuple[object, ...] = ()
        self._last_batch = False
        self._accumulated_batches = 0

    def prepare(self, *values: object) -> tuple[object, ...]:
        self._prepared = values
        return values

    def register_for_checkpointing(self, value: object) -> None:
        self._checkpointables.append(value)

    @contextmanager
    def accumulate(self, _model: nn.Module):
        self._accumulated_batches += 1
        self.sync_gradients = self._accumulated_batches % self.gradient_accumulation_steps == 0 or self._last_batch
        if self.sync_gradients:
            self._accumulated_batches = 0
        yield

    def prepare_rank_dataloader(self, dataloader: Iterable[Mapping[str, object]]):
        def prepared():
            iterator = iter(dataloader)
            try:
                current = next(iterator)
            except StopIteration:
                return
            while True:
                try:
                    following = next(iterator)
                except StopIteration:
                    self._last_batch = True
                    yield current
                    self._last_batch = False
                    return
                self._last_batch = False
                yield current
                current = following

        return prepared()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters: Iterable[nn.Parameter], max_norm: float):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise AssertionError(reduction)
        return value

    def gather_sample_counts(self, local_samples: int) -> tuple[int, ...]:
        return (local_samples, 0, 0, 0, 0, 0, 0, 0)

    def gather_object(self, value: Sequence[object]) -> list[object]:
        return [value[0], None, None, None, None, None, None, None]

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def save_state(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model, optimizer, scheduler = self._prepared
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "checkpointables": [value.state_dict() for value in self._checkpointables],
                "rng": torch.get_rng_state(),
            },
            output / "fake_accelerate_state.pt",
        )
        torch.save(torch.get_rng_state(), output / "random_states_0.pkl")

    def load_state(self, input_dir: str | Path) -> None:
        state = torch.load(Path(input_dir) / "fake_accelerate_state.pt", weights_only=False)
        model, optimizer, scheduler = self._prepared
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for value, saved in zip(self._checkpointables, state["checkpointables"], strict=True):
            value.load_state_dict(saved)
        torch.set_rng_state(state["rng"])

    def skip_first_batches(self, dataloader: Iterable[Mapping[str, object]], count: int):
        return islice(dataloader, count, None)

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value.repeat(self.num_processes)


def _inject_learnable_memorizer(model: CosyVoice3LM, settings: LoraSettings) -> CosyVoice3LM:
    """Use the real broad LoRA inventory with a bounded exactness objective."""

    adapted = model_api.inject_lora(model, settings)
    adapted.forward = types.MethodType(_memorization_forward, adapted)
    return adapted


def _memorization_progress(model: nn.Module) -> torch.Tensor:
    return next(parameter for name, parameter in model.named_parameters() if "lora_B.default.weight" in name)


def _memorization_exact(model: nn.Module) -> bool:
    return bool(_memorization_progress(model).detach().mean().item() > 0.5)


def _memorization_forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    _device: torch.device,
) -> Mapping[str, torch.Tensor]:
    progress = _memorization_progress(model)
    exact = _memorization_exact(model)
    targets = batch["speech_token"].to(torch.int64)
    lengths = batch["speech_token_len"].to(torch.int64)
    predictions = targets.clone() if exact else (targets + 1) % 6
    correct = lengths.clone() if exact else torch.zeros_like(lengths)
    return {
        "loss": (progress.mean() - 1.0).square(),
        "correct_tokens_per_sample": correct,
        "target_tokens_per_sample": lengths,
        "teacher_forced_predictions": predictions,
        "teacher_forced_targets": targets,
    }


class _FakeRun:
    id = "tiny-offline-run"
    resumed = True

    def __init__(self) -> None:
        self.dir = "wandb/offline"
        self.settings = type("Settings", (), {"resume": "must"})()
        self.pending: dict[str, object] = {}
        self.history: list[dict[str, object]] = []

    def log(self, values: Mapping[str, object], *, step: int | None = None, commit: bool | None = None) -> None:
        self.pending.update(values)
        if commit:
            self.history.append({**self.pending, "_step": step})
            self.pending.clear()

    def scan_history(self, keys: Sequence[str] | None = None):
        for row in self.history:
            yield {key: row.get(key) for key in keys or row}


class _EvaluationAccelerator:
    num_processes = 8
    process_index = 0
    local_process_index = 0
    is_main_process = True

    def __init__(self) -> None:
        self.run = _FakeRun()
        self.scalar_logs: list[tuple[dict[str, object], int | None]] = []

    @contextmanager
    def split_between_processes(self, items: Sequence[object]):
        yield list(items)

    def gather_object(self, records: Sequence[object]) -> list[object]:
        return list(records)

    def wait_for_everyone(self) -> None:
        return None

    def broadcast_object_list(self, values: list[object], from_process: int = 0) -> None:
        del values, from_process

    def get_tracker(self, name: str, unwrap: bool = False) -> _FakeRun:
        if (name, unwrap) != ("wandb", True):
            raise AssertionError((name, unwrap))
        return self.run

    def log(self, values: Mapping[str, object], step: int | None = None, log_kwargs: Mapping[str, object] | None = None) -> None:
        self.scalar_logs.append((dict(values), step))
        self.run.pending.update(values)
        if (log_kwargs or {}).get("wandb", {}).get("commit"):
            self.run.history.append({**self.run.pending, "_step": step})
            self.run.pending.clear()


class _FakeRecognizer:
    local_rank = 0

    def transcribe(self, paths: Sequence[Path]) -> list[str]:
        return ["код семь готов" for _ in paths]

    def provenance(self) -> dict[str, object]:
        return {"model": "gigaam-v3-rnnt", "provider": "fake-cpu", "device_id": 0, "max_batch_size": 8}


class _FakeSynthesizer:
    def synthesize(self, item: evaluation_api.EvaluationItem, destination: Path) -> None:
        del item
        _write_wav(destination)

    def provenance(self) -> dict[str, object]:
        return {"method": "tiny-zero-shot", "sample_rate": 24_000, "stream": False}


class _StrictPipeline:
    def __init__(self, model_dir: Path):
        self.sample_rate = 24_000
        llm = _tiny_cosyvoice3().eval()
        llm.load_state_dict(torch.load(Path(model_dir) / "llm.pt", map_location="cpu", weights_only=True), strict=True)
        flow = nn.Linear(1, 1, bias=False)
        hift = nn.Linear(1, 1, bias=False)
        for component in (flow, hift):
            component.requires_grad_(False)
        self.model = type("TinyPipelineModel", (), {"llm": llm, "flow": flow, "hift": hift})()

    def add_zero_shot_spk(self, prompt_text: str, prompt_wav: str, speaker_id: str) -> bool:
        del prompt_text, prompt_wav, speaker_id
        return True

    def inference_zero_shot(self, *args: object, **kwargs: object):
        del args, kwargs
        return [{"tts_speech": torch.tensor([[0.1, -0.1, 0.0]], dtype=torch.float32)}]


class _StrictRecognizer:
    def transcribe(self, paths: Sequence[Path]) -> list[str]:
        return ["проверка" for _ in paths]

    def provenance(self) -> dict[str, object]:
        return {"model": "gigaam-v3-rnnt", "provider": "fake-test"}


class _UploadTrap:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.calls += 1
        raise AssertionError("upload attempted by local-only workflow")


class _TinyWorkflowBackend:
    def __init__(self, options: WorkflowOptions, *, source_shards: int, phase1_rows: int, phase2_rows: int, reserved_prompts: int, benchmark_rows: int) -> None:
        self.coordinator = _Coordinator()
        self.options = options
        self.source_shards = source_shards
        self.phase1_rows = phase1_rows
        self.phase2_rows = phase2_rows
        self.reserved_prompts = reserved_prompts
        self.benchmark_rows = benchmark_rows
        self.rows: list[JoinedRow] = []
        self.selected: tuple[str, ...] = ()
        self.plan_dir: Path | None = None
        self.tar_paths: tuple[Path, ...] = ()
        self.mini_plan_dir: Path | None = None
        self.mini_tar_paths: tuple[Path, ...] = ()
        self.cache: CacheManifest | None = None
        self.memorization_cache: CacheManifest | None = None
        self.cache_interrupted = False
        self.phase_interrupted = False
        self.phase2_loaded = False
        self.memorization_initially_exact = False
        self.memorization_steps = 0
        self.memorization_verified = False
        self.validation_indices: list[int] = []
        self.evaluate_checkpoint_calls = 0
        self.evaluation_report_row_counts: list[int] = []
        self.phase1_adapter_sha256: str | None = None
        self.phase2_checkpoint: Path | None = None
        self.final_manifest: Path | None = None
        self._evaluation_checkpoint: Path | None = None
        self._phase1_checkpoint: Path | None = None
        self.eval_accelerator = _EvaluationAccelerator()
        self.recognizer = _FakeRecognizer()
        self.synthesizer = _FakeSynthesizer()
        self.wandb_logger = evaluation_api.WandbValidationLogger(
            self.eval_accelerator,
            options.paths.run_root / "wandb-run.json",
            table_factory=lambda rows: ("table", len(rows)),
            audio_factory=lambda path: ("audio", Path(path).name),
            remote_history_reader=lambda run, keys: run.scan_history(keys),
        )
        self.base = _prepare_base(options.paths.base_model_dir)

    def authenticate_stage(self, name: str, payload: dict[str, object]) -> None:
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping) or not evidence:
            raise StageRequirementError(f"tiny workflow stage lacks evidence: {name}")
        if name == "pilot_ready":
            pilot = StageStore(self.options.paths.stages_dir).require("pilot")
            if evidence.get("pilot_manifest_sha256") != pilot.manifest_sha256:
                raise StageRequirementError("pilot evidence is not the exact StageRecord envelope")
        elif name == "pilot_approved":
            StageStore(self.options.paths.stages_dir).require("pilot_approval")
        elif name == "memorization_complete":
            require_memorization_gate(self.options.paths.run_root / "memorization")
        elif name == "cache_complete":
            with mock.patch.object(cache_api, "PROMPT_RESERVATION_COUNT", self.reserved_prompts):
                cache_api.verify_cache(Path(str(evidence["cache_root"])))
        elif name in {"phase1_training", "phase1_complete", "phase2_training"}:
            checkpoint = Path(str(evidence["checkpoint"]))
            if sha256_file(checkpoint / "checkpoint_manifest.json") != evidence.get("checkpoint_manifest_sha256"):
                raise StageRequirementError(f"{name} checkpoint changed")
        elif name in {"final_export", "complete"}:
            with mock.patch.object(model_api, "load_base_llm", side_effect=lambda _path: copy.deepcopy(self.base)):
                model_api.require_committed_final(Path(str(evidence["output_dir"])), self.options.paths.base_model_dir, expected_mode="test")

    def preflight(self, options: WorkflowOptions) -> dict[str, object]:
        self.rows, self.selected = _tiny_join_rows(self.source_shards, self.phase1_rows, self.phase2_rows, self.reserved_prompts)
        self.plan_dir, self.tar_paths = _write_source_fixture(options.paths.run_root / "full-input", self.rows, self.selected, self.source_shards)
        mini_rows = _memorization_subset(self.rows)
        self.mini_plan_dir, self.mini_tar_paths = _write_source_fixture(options.paths.run_root / "mini-input", mini_rows, self.selected, self.source_shards)
        return {"split_manifest": str(self.plan_dir / "manifest.json"), "split_manifest_sha256": sha256_file(self.plan_dir / "manifest.json")}

    def qualify_tokenizer(self, options: WorkflowOptions) -> dict[str, object]:
        record = StageStore(options.paths.stages_dir).publish("tokenizer_qualification", {"provider": "fake-cpu", "qualified": True})
        return {"manifest": str(record.path), "manifest_sha256": record.manifest_sha256}

    def ensure_pilot(self, options: WorkflowOptions) -> dict[str, object]:
        store = StageStore(options.paths.stages_dir)
        try:
            record = store.require("pilot")
        except StageRequirementError:
            pilot_root = options.paths.run_root / "pilot"
            pilot_root.mkdir(parents=True, exist_ok=True)
            original = pilot_root / "original.wav"
            reconstructed = pilot_root / "reconstructed.wav"
            _write_wav(original)
            _write_wav(reconstructed)
            index = pilot_root / "index.md"
            index.write_text("# Tiny pilot\n", encoding="utf-8")
            record = store.publish("pilot", {
                "original": str(original), "original_sha256": sha256_file(original),
                "reconstructed": str(reconstructed), "reconstructed_sha256": sha256_file(reconstructed),
                "index": str(index), "index_sha256": sha256_file(index),
            })
        return {"pilot_manifest": str(record.path), "pilot_manifest_sha256": record.manifest_sha256, "listening_index": str(options.paths.run_root / "pilot/index.md")}

    def approve_pilot(self, options: WorkflowOptions, checksum: str) -> dict[str, object]:
        pilot = StageStore(options.paths.stages_dir).require("pilot")
        if checksum != pilot.manifest_sha256:
            raise StageRequirementError("tiny pilot checksum mismatch")
        approval = StageStore(options.paths.stages_dir).publish("pilot_approval", {"pilot_manifest_sha256": checksum})
        return {"pilot_manifest_sha256": checksum, "approval": str(approval.path), "approval_sha256": approval.manifest_sha256}

    def prepare_memorization(self, options: WorkflowOptions) -> dict[str, object]:
        assert self.mini_plan_dir is not None
        self.memorization_cache = _build_cache(
            options.paths.run_root / "memorization_cache", self.mini_plan_dir, self.mini_tar_paths,
            self.source_shards, self.reserved_prompts,
        )
        return {"cache_root": str(self.memorization_cache.root), "phase1_rows": 2, "phase2_rows": 2}

    def memorize(self, options: WorkflowOptions) -> dict[str, object]:
        assert self.memorization_cache is not None and self.mini_plan_dir is not None
        probe = _inject_learnable_memorizer(copy.deepcopy(self.base), LoraSettings())
        self.memorization_initially_exact = _memorization_exact(probe)
        request = MemorizationRequest(
            split_plan=self.mini_plan_dir,
            cache=self.memorization_cache,
            output_root=options.paths.run_root,
            base_model_dir=options.paths.base_model_dir,
            max_steps=4,
            check_every=1,
            learning_rate=1.0,
            batch_size=1,
            tokenizer=_TinyTextTokenizer(),
            model_loader=lambda _path: copy.deepcopy(self.base),
            adapter_injector=_inject_learnable_memorizer,
            audit_fn=model_api.audit_trainable_parameters,
            accelerator_factory=_CpuAccelerator,
        )
        report = run_memorization_gate(request)
        verified = require_memorization_gate(report.path)
        self.memorization_steps = report.steps
        self.memorization_verified = verified.path == report.path
        manifest = report.path / "memorization_manifest.json"
        return {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "steps": report.steps}

    def build_cache(self, options: WorkflowOptions) -> dict[str, object]:
        assert self.plan_dir is not None
        self.cache = _build_cache(
            options.paths.run_root / "cache", self.plan_dir, self.tar_paths,
            self.source_shards, self.reserved_prompts, interrupt=True,
        )
        self.cache_interrupted = True
        return {"cache_root": str(self.cache.root), "phase1_rows": self.cache.phase_rows[1], "phase2_rows": self.cache.phase_rows[2], "prompt_count": self.cache.prompt_count}

    def capacity_smoke(self, options: WorkflowOptions) -> dict[str, object]:
        assert self.cache is not None
        dataset = CachedSpeechDataset(self.cache, 1, tokenizer=_TinyTextTokenizer())
        batch = CosyVoice3Collator(_TinyTextTokenizer())([dataset[0]])
        model = model_api.inject_lora(copy.deepcopy(self.base), LoraSettings())
        result = model(batch, torch.device("cpu"))
        result["loss"].backward()
        return {"requested_token_limit": options.token_limit, "qualified_token_limit": options.token_limit}

    def ensure_logging(self, options: WorkflowOptions) -> dict[str, object]:
        del options
        return {"mode": "offline", "logger_class": self.wandb_logger.__class__.__name__}

    def evaluate(self, options: WorkflowOptions, validation_index: int, generations: int) -> dict[str, object]:
        if generations != self.benchmark_rows:
            raise ValueError("tiny workflow must scale validation generations to its benchmark row count")
        report = self._evaluate_checkpoint(options, validation_index)
        self.evaluation_report_row_counts.append(report.row_count)
        self.validation_indices.append(validation_index)
        return {
            "validation_index": validation_index,
            "generations": report.row_count,
            "summary": str(report.summary_json),
            "summary_sha256": sha256_file(report.summary_json),
            "identity_sha256": report.identity_checksum,
            "artifact_checksums": dict(report.artifact_checksums),
        }

    def _evaluate_checkpoint(self, options: WorkflowOptions, validation_index: int) -> evaluation_api.EvaluationReport:
        checkpoint = self._evaluation_checkpoint
        if checkpoint is None:
            checkpoint_sha = sha256_file(options.paths.base_model_dir / "llm.pt")
            model_state_sha = checkpoint_sha
            adapter_sha = checkpoint_sha
        else:
            checkpoint_payload = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            checkpoint_sha = model_api.checkpoint_identity_sha256(checkpoint_payload)
            model_state_sha = model_api.model_state_identity_sha256(checkpoint_payload["state_files"])
            adapter_manifest = json.loads((checkpoint / "adapter/adapter_manifest.json").read_text(encoding="utf-8"))
            adapter_sha = adapter_manifest["weights_sha256"]
        provenance = evaluation_api.EvaluationProvenance(
            checkpoint_sha256=checkpoint_sha,
            model_state_sha256=model_state_sha,
            adapter_sha256=adapter_sha,
            base_checkpoint_sha256=sha256_file(options.paths.base_model_dir / "llm.pt"),
            benchmark_snapshot_sha256="5" * 64,
            benchmark_revision="tiny-offline-v1",
            asr_config=self.recognizer.provenance(),
            synthesis_config=self.synthesizer.provenance(),
            code_version="tiny-integration",
            config_version="balalaika-v1",
        )
        output = options.paths.run_root / f"evaluation/validation-{validation_index:02d}"
        request = evaluation_api.EvaluationRequest(
            rows=_benchmark_rows(self.benchmark_rows),
            prompts=_evaluation_prompts(options.paths.run_root / "evaluation-prompts", self.reserved_prompts),
            accelerator=self.eval_accelerator,
            synthesizer=self.synthesizer,
            recognizer=self.recognizer,
            provenance=provenance,
            validation_index=validation_index,
            output_jsonl=output / "results.jsonl",
            summary_json=output / "summary.json",
            panel_dir=output / "panel",
            temporary_audio_dir=output / "audio",
            assignment_manifest=options.paths.run_root / "evaluation/voice-assignment.json",
            memorization_path=options.paths.run_root / "memorization",
            wandb_logger=self.wandb_logger,
        )
        with mock.patch.object(evaluation_api, "_EXPECTED_ROWS", self.benchmark_rows), mock.patch.object(
            evaluation_api, "_EXPECTED_LOCAL_ROWS", self.benchmark_rows
        ), mock.patch.object(evaluation_api, "_EXPECTED_VOICES", self.reserved_prompts), mock.patch.object(
            evaluation_api, "build_voice_assignment", side_effect=_scaled_voice_assignment
        ):
            self.evaluate_checkpoint_calls += 1
            return evaluation_api.evaluate_checkpoint(request)

    def train(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        *,
        initial_adapter_sha256: str | None,
        fresh_optimizer: bool,
    ) -> dict[str, object]:
        if not fresh_optimizer or self.cache is None:
            raise ValueError("tiny phase must use fresh optimizer and verified cache")
        model = model_api.inject_lora(copy.deepcopy(self.base), LoraSettings())
        if phase.number == 2:
            if initial_adapter_sha256 != self.phase1_adapter_sha256:
                raise StageRequirementError("phase2 did not receive sealed phase1 adapter")
            phase1_checkpoint = self._phase1_checkpoint
            from peft.utils import set_peft_model_state_dict
            from safetensors.torch import load_file

            weights = phase1_checkpoint / "adapter/adapter_model.safetensors"
            if sha256_file(weights) != initial_adapter_sha256:
                raise StageRequirementError("sealed phase1 adapter changed")
            result = set_peft_model_state_dict(model, load_file(str(weights), device="cpu"), adapter_name=model_api.ADAPTER_NAME)
            if [name for name in result.missing_keys if "lora_" in name] or result.unexpected_keys:
                raise StageRequirementError("phase1 adapter could not initialize fresh phase2 model")
            self.phase2_loaded = True
        dataset = CachedSpeechDataset(self.cache, phase, tokenizer=_TinyTextTokenizer())
        collator = CosyVoice3Collator(_TinyTextTokenizer())
        batches = [collator([dataset[index]]) for index in range(len(dataset))]
        request = TrainRequest(
            model=model,
            phase=phase,
            eligible_samples=len(dataset),
            cache_checksum=sha256_file(self.cache.root / "manifest.json"),
            checkpoint_root=options.paths.run_root / f"phase{phase.number}/checkpoints",
            token_limit=options.token_limit,
            accumulation_steps=1,
            dataloader_factory=lambda _epoch, _accelerator: batches,
            accelerator_factory=_CpuAccelerator,
            validation_index_base=0 if phase.number == 1 else 16,
            initial_adapter_sha256=initial_adapter_sha256,
        )
        validations: list[dict[str, object]] = []

        def validate(event: object) -> bool:
            self._evaluation_checkpoint = Path(getattr(event, "checkpoint"))
            evidence = self.evaluate(options, int(getattr(event, "validation_index")), self.benchmark_rows)
            validations.append(evidence)
            return phase.number != 1 or int(getattr(event, "validation_index")) < 3

        if phase.number == 1:
            first = train_phase(request, TrainingCallbacks(validate=validate))
            if first.completed or first.checkpoint is None or first.progress.validation_index != 3:
                raise AssertionError("tiny phase interruption did not occur")
            self.phase_interrupted = True
            resumed = train_phase(replace(request, resume_from=first.checkpoint), TrainingCallbacks(validate=lambda event: self._resume_validate(options, event, validations)))
            result_phase = resumed
        else:
            result_phase = train_phase(request, TrainingCallbacks(validate=lambda event: self._resume_validate(options, event, validations)))
        if not result_phase.completed or result_phase.checkpoint is None:
            raise AssertionError(f"tiny phase {phase.number} did not complete")
        checkpoint = result_phase.checkpoint
        checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
        checkpoint_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        adapter_manifest = json.loads((checkpoint / "adapter/adapter_manifest.json").read_text(encoding="utf-8"))
        if phase.number == 1:
            self._phase1_checkpoint = checkpoint
            self.phase1_adapter_sha256 = adapter_manifest["weights_sha256"]
        else:
            self.phase2_checkpoint = checkpoint
        return {
            "completed": True,
            "checkpoint": str(checkpoint),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
            "adapter_dir": str(checkpoint / "adapter"),
            "adapter_weights_sha256": adapter_manifest["weights_sha256"],
            "validation_summary": str(options.paths.run_root / f"evaluation/validation-{validation_indices[-1]:02d}/summary.json"),
            "final_validation_index": validation_indices[-1],
            "fresh_optimizer": True,
            "training_identity": checkpoint_payload["identity"],
            "validations": validations,
        }

    def _resume_validate(self, options: WorkflowOptions, event: object, validations: list[dict[str, object]]) -> bool:
        self._evaluation_checkpoint = Path(getattr(event, "checkpoint"))
        validations.append(self.evaluate(options, int(getattr(event, "validation_index")), self.benchmark_rows))
        return True

    def export(self, options: WorkflowOptions, phase2: dict[str, object]) -> dict[str, object]:
        checkpoint = Path(str(phase2["checkpoint"]))
        if checkpoint != self.phase2_checkpoint:
            raise StageRequirementError("export did not receive actual terminal phase2 checkpoint")
        voices = _strict_voices(options.paths.run_root / "strict-voices")
        request = ExportRequest(
            base_model_dir=options.paths.base_model_dir,
            phase2_checkpoint=checkpoint,
            validation_summary=Path(str(phase2["validation_summary"])),
            output_dir=options.paths.run_root / "final",
            test_mode=True,
            verification_voices=voices,
            recognizer=_StrictRecognizer(),
            pipeline_factory=_StrictPipeline,
        )
        with mock.patch.object(model_api, "load_base_llm", side_effect=lambda _path: copy.deepcopy(self.base)):
            final = model_api.export_final_llm(request)
        self.final_manifest = final.path
        return {
            "output_dir": str(final.path.parent),
            "manifest": str(final.path),
            "manifest_sha256": sha256_file(final.path),
            "mode": final.mode,
            "production_ready": final.production_ready,
        }


def run_tiny_workflow(
    root: Path,
    *,
    source_shards: int,
    phase1_rows: int,
    phase2_rows: int,
    reserved_prompts: int,
    benchmark_rows: int,
) -> TinyWorkflowResult:
    if (source_shards, phase1_rows, phase2_rows, reserved_prompts, benchmark_rows) != (2, 8, 8, 2, 8):
        raise ValueError("the checked-in tiny fixture has one exact supported scale")
    paths = RunPaths(
        root / "dataset",
        root / "repo",
        root / "run",
        root / "repo/pretrained_models/Fun-CosyVoice3-0.5B-2512",
        tuple(range(8)),
        1986,
    )
    options = WorkflowOptions(paths=paths, token_limit=256, allow_test_export=True)
    backend = _TinyWorkflowBackend(
        options,
        source_shards=source_shards,
        phase1_rows=phase1_rows,
        phase2_rows=phase2_rows,
        reserved_prompts=reserved_prompts,
        benchmark_rows=benchmark_rows,
    )
    calls: list[str] = []
    trap = _UploadTrap()
    from huggingface_hub import HfApi

    with mock.patch.object(HfApi, "upload_file", side_effect=trap), mock.patch.object(
        HfApi, "upload_folder", side_effect=trap
    ), mock.patch(
        "cosyvoice.finetune.balalaika.workflow.VALIDATION_GENERATIONS", benchmark_rows
    ), mock.patch.dict(os.environ, {"WANDB_API_KEY": "offline-fixture-key"}, clear=False):
        first_exit = run_phase1_memorize(argparse.Namespace(options=options, backend=backend))
        pilot = StageStore(paths.stages_dir).require("pilot")
        approved_options = replace(options, approve_pilot_sha256=pilot.manifest_sha256)
        calls.append("run_phase1 --memorize")
        approved_exit = run_phase1_memorize(argparse.Namespace(options=approved_options, backend=backend))
        calls.append("prepare_cache")
        prepare_cache_exit = run_prepare_cache(argparse.Namespace(options=options, backend=backend))
        calls.append("run_phase1 --train")
        phase1_train_exit = run_phase1_train(argparse.Namespace(options=options, backend=backend))
        calls.append("run_phase2")
        phase2_exit = run_phase2(argparse.Namespace(options=options, backend=backend))

    workflow_store = StageStore(
        paths.run_root / "workflow_stages",
        dependency_lock={"workflow_version": "cosyvoice3-balalaika-two-phase-v1"},
        input_provenance=_workflow_identity_for_fixture(options),
    )
    complete = workflow_store.require("complete")
    stage_evidence_row_counts = _committed_evidence_row_counts(workflow_store)
    assert backend.final_manifest is not None and backend.phase2_checkpoint is not None
    final_payload = json.loads(backend.final_manifest.read_text(encoding="utf-8"))
    return TinyWorkflowResult(
        stage=complete.name,
        phase1_first_exit=int(first_exit),
        phase1_approved_exit=int(approved_exit),
        prepare_cache_exit=int(prepare_cache_exit),
        phase1_train_exit=int(phase1_train_exit),
        phase2_exit=int(phase2_exit),
        workflow_calls=tuple(calls),
        final_llm=backend.final_manifest.parent / "llm.pt",
        final_manifest=backend.final_manifest,
        final_phase2_checkpoint=Path(final_payload["phase2_checkpoint"]),
        phase2_checkpoint=backend.phase2_checkpoint,
        strict_load_verified=bool(final_payload["strict_verification"]),
        validation_indices=tuple(backend.validation_indices),
        phase_epochs=(PhaseSpec.for_phase(1).epochs, PhaseSpec.for_phase(2).epochs),
        boundaries_per_epoch=8,
        pilot_stopped_before_approval=first_exit == ExitCode.PILOT_REVIEW_REQUIRED,
        pilot_checksum_approved=(
            StageStore(paths.stages_dir).require("pilot_approval").payload["pilot_manifest_sha256"]
            == pilot.manifest_sha256
        ),
        memorization_passed=backend.memorization_verified,
        memorization_initially_exact=backend.memorization_initially_exact,
        memorization_optimizer_steps=backend.memorization_steps,
        memorization_evidence_verified=backend.memorization_verified,
        cache_resumed_after_interruption=backend.cache_interrupted,
        phase_resumed_after_interruption=backend.phase_interrupted,
        phase2_loaded_sealed_phase1_adapter=backend.phase2_loaded,
        evaluate_checkpoint_calls=backend.evaluate_checkpoint_calls,
        evaluation_report_row_counts=tuple(backend.evaluation_report_row_counts),
        stage_evidence_row_counts=stage_evidence_row_counts,
        wandb_mode="offline",
        upload_calls=trap.calls,
    )


def _workflow_identity_for_fixture(options: WorkflowOptions) -> dict[str, object]:
    from cosyvoice.finetune.balalaika.workflow import _workflow_identity

    return _workflow_identity(options)


def _committed_evidence_row_counts(store: StageStore) -> tuple[int, ...]:
    validation_zero = store.require("validation_00").payload["evidence"]
    phase1 = store.require("phase1_complete").payload["evidence"]
    phase2 = store.require("phase2_training").payload["evidence"]
    evidence = [validation_zero, *phase1["validations"], *phase2["validations"]]
    return tuple(int(item["generations"]) for item in evidence)


def _prepare_base(base_dir: Path) -> CosyVoice3LM:
    base_dir.mkdir(parents=True, exist_ok=True)
    model = _tiny_cosyvoice3().eval()
    torch.save(model.state_dict(), base_dir / "llm.pt")
    (base_dir / "cosyvoice3.yaml").write_text("tiny fixture", encoding="utf-8")
    (base_dir / "flow.pt").write_bytes(b"frozen-flow")
    (base_dir / "hift.pt").write_bytes(b"frozen-hift")
    model._balalaika_base_model_dir = base_dir
    model._balalaika_base_checkpoint_sha256 = sha256_file(base_dir / "llm.pt")
    return model


def _tiny_cosyvoice3() -> CosyVoice3LM:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    encoder = Qwen2Encoder.__new__(Qwen2Encoder)
    nn.Module.__init__(encoder)
    encoder.model = Qwen2ForCausalLM(config)
    return CosyVoice3LM(8, 8, 6, encoder, lambda scores, decoded, sampling: int(scores.argmax()), [5, 15])


def _tiny_join_rows(
    source_shards: int,
    phase1_rows: int,
    phase2_rows: int,
    reserved_prompts: int,
) -> tuple[list[JoinedRow], tuple[str, ...]]:
    rows = [
        JoinedRow(
            f"{index % source_shards:06d}/phase1_{index:02d}.mp3",
            f"низкий {index}",
            0.5,
            text_token_count=2 + index,
        )
        for index in range(phase1_rows)
    ]
    rows.extend(
        JoinedRow(
            f"{index % source_shards:06d}/phase2_{index:02d}.mp3",
            f"высокий {index}",
            0.95,
            text_token_count=2 + index,
        )
        for index in range(phase2_rows + reserved_prompts)
    )
    selected = reserve_prompt_ids(rows, count=reserved_prompts, seed=1986)
    assigned = assign_phases(rows, set(selected))
    return assigned, selected


def _memorization_subset(rows: Sequence[JoinedRow]) -> list[JoinedRow]:
    selected: list[JoinedRow] = [row for row in rows if row.reserved]
    for phase in (1, 2):
        phase_rows = sorted(
            (row for row in rows if row.phase == phase),
            key=lambda row: int(Path(row.source_relative_path).stem.rsplit("_", 1)[1]) % 4,
        )
        selected.extend((phase_rows[0], phase_rows[-1]))
    return selected


def _write_source_fixture(
    root: Path,
    rows: Sequence[JoinedRow],
    reserved: Sequence[str],
    source_shards: int,
) -> tuple[Path, tuple[Path, ...]]:
    plan_dir = root / "split_plan"
    source_dir = root / "source"
    plan_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    tar_paths = []
    for shard in range(source_shards):
        shard_rows = [row for row in rows if row.source_relative_path.startswith(f"{shard:06d}/")]
        with (plan_dir / f"shard_{shard:06d}.jsonl").open("w", encoding="utf-8") as handle:
            for row in shard_rows:
                handle.write(json.dumps({
                    "source_relative_path": row.source_relative_path, "text": row.text, "instruct": row.instruct,
                    "agreement": row.agreement, "phase": row.phase, "reserved": row.reserved,
                    "reservation_score": row.reservation_score, "model_limit_exclusion": row.model_limit_exclusion,
                    "text_token_count": row.text_token_count,
                }, ensure_ascii=False) + "\n")
        tar_path = source_dir / f"shard_{shard:06d}.tar"
        with tarfile.open(tar_path, "w") as archive:
            for row in shard_rows:
                stem = Path(row.source_relative_path).stem
                payloads = (
                    (f"{stem}.json", json.dumps({"source_relative_path": row.source_relative_path}).encode()),
                    (f"{stem}.mp3", row.source_relative_path.encode()),
                )
                for name, payload in payloads:
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        tar_paths.append(tar_path)
    atomic_write_json(plan_dir / "manifest.json", {"reserved_prompts": [{"source_relative_path": source} for source in reserved]})
    return plan_dir, tuple(tar_paths)


def _build_cache(
    root: Path,
    plan_dir: Path,
    tar_paths: Sequence[Path],
    source_shards: int,
    prompts: int,
    interrupt: bool = False,
) -> CacheManifest:
    interrupted = False
    original = cache_api._replace_staged

    def replace_once(source: Path, destination: Path) -> None:
        nonlocal interrupted
        if interrupt and not interrupted and Path(destination).parent.name == "phase2":
            interrupted = True
            raise OSError("simulated cache interruption")
        original(source, destination)

    with mock.patch.object(sources_api, "EXPECTED_SOURCE_TARS", source_shards), mock.patch.object(
        cache_api, "PROMPT_RESERVATION_COUNT", prompts
    ), mock.patch.object(cache_api, "_decode_audio", side_effect=_decode_fixture):
        first = CacheShardRequest(tar_paths[0], plan_dir, root, 0, batch_size=3)
        if interrupt:
            with mock.patch.object(cache_api, "_replace_staged", side_effect=replace_once):
                try:
                    cache_api.build_cache_shard(first, _TinySpeechTokenizer())
                except OSError as exc:
                    if "simulated cache interruption" not in str(exc):
                        raise
                else:
                    raise AssertionError("cache interruption did not fire")
        cache_api.build_cache_shard(first, _TinySpeechTokenizer())
        for shard in range(1, source_shards):
            cache_api.build_cache_shard(
                CacheShardRequest(tar_paths[shard], plan_dir, root, shard, batch_size=3),
                _TinySpeechTokenizer(),
            )
        cache = cache_api.verify_cache(root)
    if list(root.rglob("*.partial")) or list(root.rglob("*.rollback")):
        raise AssertionError("cache recovery left debris")
    return cache


def _decode_fixture(sample: object) -> AudioInput:
    source = str(getattr(sample, "source_relative_path"))
    frames = int(Path(source).stem.rsplit("_", 1)[1]) % 4 + 1
    return AudioInput(source, [0.0] * frames, 24_000, frames)


def _benchmark_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "stressed": f"Код {index} готов",
            "normalized_gold": "код семь готов",
            "hard_number": str(index),
            "category": "code",
            "number_span": {"reference_start": 1, "reference_end": 2},
        }
        for index in range(1, count + 1)
    ]


def _evaluation_prompts(root: Path, count: int) -> list[dict[str, object]]:
    prompts = []
    for index in range(count):
        path = root / f"voice_{index:02d}.wav"
        if not path.is_file():
            _write_wav(path)
        prompts.append({
            "voice_id": f"voice_{index:02d}",
            "audio_path": path,
            "text": f"prompt {index}",
            "wav_sha256": sha256_file(path),
        })
    return prompts


def _scaled_voice_assignment(
    rows: Sequence[object],
    prompts: Sequence[Mapping[str, object]],
) -> list[evaluation_api.EvaluationItem]:
    voices = sorted(prompts, key=lambda prompt: str(prompt["voice_id"]))
    items = []
    for index, row in enumerate(sorted(rows, key=lambda value: int(value["id"]))):
        voice = voices[index % len(voices)]
        span = row["number_span"]
        items.append(evaluation_api.EvaluationItem(
            int(row["id"]),
            str(row["stressed"]),
            str(row["normalized_gold"]),
            str(row["hard_number"]),
            str(row["category"]),
            NumberSpan(int(span["reference_start"]), int(span["reference_end"]), str(row["category"])),
            str(voice["voice_id"]),
            Path(voice["audio_path"]),
            str(voice["text"]),
            str(voice["wav_sha256"]),
        ))
    return items


def _strict_voices(root: Path) -> tuple[dict[str, object], ...]:
    values = []
    for index in range(20):
        path = root / f"voice_{index:02d}.wav"
        _write_wav(path)
        values.append({
            "voice_id": f"voice_{index:02d}",
            "prompt_text": "проверка",
            "prompt_wav": path,
            "prompt_sha256": sha256_file(path),
        })
    return tuple(values)


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0" * 8)

"""CPU-only tests for exact progress and resumable Accelerate training."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from importlib import import_module
from itertools import islice
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cosyvoice.finetune.balalaika.config import PhaseSpec
from cosyvoice.finetune.balalaika.model import LoraSettings, TrainableAudit


def _api():
    try:
        return import_module("cosyvoice.finetune.balalaika.training")
    except ModuleNotFoundError as exc:
        raise AssertionError("Balalaika Accelerate training module is missing") from exc


class ToyAdapterModel(torch.nn.Module):
    def __init__(self, seen_sample_ids: list[str]) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(0.0))
        self._seen_sample_ids = seen_sample_ids
        self.forward_calls = 0
        self._balalaika_base_checkpoint_sha256 = "b" * 64
        self._balalaika_lora_settings = LoraSettings()

    def forward(self, batch, _device):
        self.forward_calls += 1
        self._seen_sample_ids.extend(batch["utts"])
        target = batch["target"].float()
        return {"loss": ((self.lora_weight - target) ** 2).mean()}


class FakeAccelerator:
    """Stateful CPU fixture for the subset of Accelerate owned by the trainer."""

    def __init__(self, *, global_sample_counts=None, remote_batch=None, **kwargs) -> None:
        self.kwargs = kwargs
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.num_processes = 8
        self.process_index = 0
        self.is_main_process = True
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self._checkpointables = []
        self._prepared = ()
        self._last_batch = False
        self._accumulated_batches = 0
        self._global_sample_counts = iter(global_sample_counts) if global_sample_counts is not None else None
        self._last_global_samples = 0
        self._remote_batch = remote_batch

    def prepare(self, *values):
        self._prepared = values
        return values

    def register_for_checkpointing(self, value) -> None:
        self._checkpointables.append(value)

    @contextmanager
    def accumulate(self, _model):
        self._accumulated_batches += 1
        self.sync_gradients = (
            self._accumulated_batches % self.gradient_accumulation_steps == 0 or self._last_batch
        )
        if self.sync_gradients:
            self._accumulated_batches = 0
        yield

    def prepare_rank_dataloader(self, dataloader):
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

    def backward(self, loss) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, value, reduction="sum"):
        if reduction != "sum":
            raise AssertionError(reduction)
        result = int(value.item()) if self._global_sample_counts is None else next(self._global_sample_counts)
        self._last_global_samples = result
        return torch.tensor(result, dtype=value.dtype, device=value.device)

    def gather_sample_counts(self, local_samples):
        remote = max(self._last_global_samples - local_samples, 0)
        return (local_samples, remote, 0, 0, 0, 0, 0, 0)

    def gather_object(self, value):
        local = value[0]
        return [local, self._remote_batch] + [None] * 6

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model):
        return model

    def save_state(self, output_dir) -> None:
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

    def load_state(self, input_dir) -> None:
        state = torch.load(Path(input_dir) / "fake_accelerate_state.pt", weights_only=False)
        model, optimizer, scheduler = self._prepared
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for value, saved in zip(self._checkpointables, state["checkpointables"], strict=True):
            value.load_state_dict(saved)
        torch.set_rng_state(state["rng"])

    def skip_first_batches(self, dataloader, count):
        return islice(dataloader, count, None)

    def gather(self, value):
        return value.repeat(self.num_processes)


def _audit(_model) -> TrainableAudit:
    return TrainableAudit(("toy",), ("lora_weight",), (), 1, 1)


def _save_adapter(model, path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "adapter_model.safetensors"
    weights.write_bytes(model.lora_weight.detach().cpu().numpy().tobytes())
    (output / "adapter_manifest.json").write_text(
        json.dumps({"base_checkpoint_sha256": model._balalaika_base_checkpoint_sha256}),
        encoding="utf-8",
    )


class AccelerateTrainingTests(unittest.TestCase):
    def test_constant_scheduler_spec_is_json_serializable_and_keeps_exact_lr(self) -> None:
        api = _api()
        spec = api.SchedulerSpec(kind="constant-v1")
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = api._build_scheduler(optimizer, spec)
        learning_rates = [optimizer.param_groups[0]["lr"]]

        for _ in range(4):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            learning_rates.append(optimizer.param_groups[0]["lr"])

        self.assertEqual(json.loads(json.dumps(asdict(spec))), {"kind": "constant-v1"})
        self.assertEqual(learning_rates, [1e-4] * 5)

    def test_scheduler_spec_rejects_unknown_kind(self) -> None:
        api = _api()

        with self.assertRaisesRegex(ValueError, "unknown scheduler kind"):
            api.SchedulerSpec(kind="cosine")

    def test_train_request_rejects_non_scheduler_spec(self) -> None:
        api = _api()

        with self.assertRaisesRegex(ValueError, "SchedulerSpec"):
            api.TrainRequest(
                model=ToyAdapterModel([]),
                phase=PhaseSpec.for_phase(1),
                eligible_samples=1,
                cache_checksum="c" * 64,
                checkpoint_root=Path("unused"),
                token_limit=2000,
                accumulation_steps=1,
                dataloader_factory=lambda _epoch, _accelerator: (),
                accelerator_factory=FakeAccelerator,
                scheduler_spec={"kind": "constant-v1"},
            )

    def test_train_request_rejects_modified_phase_schedules(self) -> None:
        api = _api()
        model = ToyAdapterModel([])
        base = PhaseSpec.for_phase(1)
        values = (
            replace(base, epochs=1),
            replace(base, learning_rate=2e-4),
            replace(base, predicate="asr_agreement_mean <= 0.95"),
            replace(base, agreement_max=0.96),
        )

        for phase in values:
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "exact approved phase"):
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=1,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path("unused"),
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: (),
                    accelerator_factory=FakeAccelerator,
                )

    def test_fraction_boundaries_are_exact(self) -> None:
        boundaries = _api().FractionBoundary.for_epoch(eligible_samples=80)

        self.assertEqual([item.sample_target for item in boundaries], [10, 20, 30, 40, 50, 60, 70, 80])
        self.assertEqual([(item.numerator, item.denominator) for item in boundaries], [(i, 8) for i in range(1, 9)])

    def test_small_epoch_retains_all_eight_ordered_boundary_events(self) -> None:
        boundaries = _api().FractionBoundary.for_epoch(eligible_samples=3)

        self.assertEqual([item.sample_target for item in boundaries], [1, 1, 2, 2, 2, 3, 3, 3])
        self.assertEqual([item.ordinal for item in boundaries], list(range(1, 9)))

    def test_fraction_targets_remain_exact_beyond_float_integer_range(self) -> None:
        eligible = 2**60 + 1
        boundaries = _api().FractionBoundary.for_epoch(eligible)

        self.assertEqual(
            [item.sample_target for item in boundaries],
            [(eligible * ordinal + 7) // 8 for ordinal in range(1, 9)],
        )

    def test_resume_does_not_repeat_samples_or_validation_boundaries(self) -> None:
        api = _api()
        seen: list[str] = []
        model = ToyAdapterModel(seen)
        batches = [
            {"utts": [f"sample-{index}"], "target": torch.tensor([float(index + 1)])}
            for index in range(8)
        ]
        phase = PhaseSpec.for_phase(1)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            root = Path(tmp)
            first_indices: list[int] = []
            first = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=8,
                    cache_checksum="c" * 64,
                    checkpoint_root=root,
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: batches,
                    accelerator_factory=FakeAccelerator,
                ),
                api.TrainingCallbacks(
                    validate=lambda event: first_indices.append(event.validation_index) or event.validation_index < 3
                ),
            )
            resumed_indices: list[int] = []
            resumed = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=8,
                    cache_checksum="c" * 64,
                    checkpoint_root=root,
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: batches,
                    accelerator_factory=FakeAccelerator,
                    resume_from=first.checkpoint,
                ),
                api.TrainingCallbacks(validate=lambda event: resumed_indices.append(event.validation_index)),
            )

        self.assertEqual(seen, [f"sample-{index}" for index in range(8)] * 2)
        self.assertEqual(first_indices + resumed_indices, list(range(1, 17)))
        self.assertEqual(resumed.progress.optimizer_steps, 16)
        self.assertTrue(resumed.completed)

    def test_one_optimizer_boundary_can_release_all_due_small_epoch_events(self) -> None:
        api = _api()
        seen: list[str] = []
        model = ToyAdapterModel(seen)
        events = []
        phase = PhaseSpec.for_phase(1)
        batch = {"utts": ["a", "b", "c"], "target": torch.tensor([1.0, 2.0, 3.0])}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            result = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=3,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path(tmp),
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: [batch],
                    accelerator_factory=FakeAccelerator,
                ),
                api.TrainingCallbacks(validate=lambda event: events.append(event)),
            )

        self.assertTrue(result.completed)
        self.assertEqual([event.boundary.ordinal for event in events], list(range(1, 9)) * 2)
        self.assertEqual({event.global_samples for event in events}, {3, 6})

    def test_non_divisible_accumulation_commits_final_window_once(self) -> None:
        api = _api()
        model = ToyAdapterModel([])
        phase = PhaseSpec.for_phase(1)
        batches = [
            {"utts": [f"sample-{index}"], "target": torch.tensor([float(index + 1)])}
            for index in range(3)
        ]
        events = []

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            result = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=3,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path(tmp),
                    token_limit=2000,
                    accumulation_steps=2,
                    dataloader_factory=lambda _epoch, _accelerator: batches,
                    accelerator_factory=FakeAccelerator,
                ),
                api.TrainingCallbacks(validate=lambda event: events.append(event)),
            )
            state = torch.load(
                result.checkpoint / "fake_accelerate_state.pt", weights_only=False
            )

        self.assertEqual(result.progress.optimizer_steps, 4)
        self.assertEqual(state["scheduler"]["last_epoch"], 4)
        self.assertEqual([event.global_samples for event in events[:5]], [2] * 5)
        self.assertEqual([event.global_samples for event in events[5:8]], [3] * 3)
        self.assertEqual([event.global_samples for event in events[8:13]], [5] * 5)
        self.assertEqual([event.global_samples for event in events[13:]], [6] * 3)

    def test_global_zero_batch_skips_the_entire_training_step(self) -> None:
        api = _api()
        seen: list[str] = []
        model = ToyAdapterModel(seen)
        empty = {"utts": [], "target": torch.empty(0)}
        real = {"utts": ["real"], "target": torch.tensor([1.0])}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            result = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=PhaseSpec.for_phase(1),
                    eligible_samples=1,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path(tmp),
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: [empty, real],
                    accelerator_factory=lambda **kwargs: FakeAccelerator(
                        global_sample_counts=[0, 1, 0, 1], **kwargs
                    ),
                ),
                api.TrainingCallbacks(validate=lambda _event: None),
            )
            state = torch.load(result.checkpoint / "fake_accelerate_state.pt", weights_only=False)

        self.assertEqual(seen, ["real", "real"])
        self.assertEqual(result.progress.optimizer_steps, 2)
        self.assertEqual(state["scheduler"]["last_epoch"], 2)

    def test_empty_local_rank_uses_shared_dummy_forward_with_zero_contribution(self) -> None:
        api = _api()
        seen: list[str] = []
        model = ToyAdapterModel(seen)
        empty = {"utts": [], "target": torch.empty(0)}
        remote = {"utts": ["remote"], "target": torch.tensor([2.0])}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            result = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=PhaseSpec.for_phase(1),
                    eligible_samples=1,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path(tmp),
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: [empty],
                    accelerator_factory=lambda **kwargs: FakeAccelerator(
                        global_sample_counts=[1, 1], remote_batch=remote, **kwargs
                    ),
                ),
                api.TrainingCallbacks(validate=lambda _event: None),
            )

        self.assertTrue(result.completed)
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(seen, [])
        self.assertEqual(model.lora_weight.item(), 0.0)

    def test_phase_two_continues_global_validation_indices_at_seventeen(self) -> None:
        api = _api()
        model = ToyAdapterModel([])
        phase = PhaseSpec.for_phase(2)
        indices: list[int] = []

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            result = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=1,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path(tmp),
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: [
                        {"utts": ["only"], "target": torch.tensor([1.0])}
                    ],
                    accelerator_factory=FakeAccelerator,
                ),
                api.TrainingCallbacks(validate=lambda event: indices.append(event.validation_index)),
            )

        self.assertTrue(result.completed)
        self.assertEqual(indices, list(range(17, 41)))

    def test_checkpoint_records_complete_identity_rng_and_state_checksums(self) -> None:
        api = _api()
        model = ToyAdapterModel([])
        phase = PhaseSpec.for_phase(1)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            result = api.train_phase(
                api.TrainRequest(
                    model=model,
                    phase=phase,
                    eligible_samples=1,
                    cache_checksum="c" * 64,
                    checkpoint_root=Path(tmp),
                    token_limit=2000,
                    accumulation_steps=1,
                    dataloader_factory=lambda _epoch, _accelerator: [
                        {"utts": ["only"], "target": torch.tensor([1.0])}
                    ],
                    accelerator_factory=FakeAccelerator,
                ),
                api.TrainingCallbacks(validate=lambda event: False),
            )
            manifest = json.loads((result.checkpoint / api.CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))

        self.assertEqual(manifest["validation_status"], "succeeded")
        self.assertEqual(
            set(manifest["identity"]),
            {
                "accumulation_steps",
                "base_checkpoint_sha256",
                "cache_manifest_sha256",
                "dataloader_identity",
                "eligible_samples",
                "lora",
                "max_grad_norm",
                "phase",
                "phase_spec",
                "sampler_seed",
                "sampler_window_size",
                "scheduler",
                "token_limit",
                "world_size",
            },
        )
        self.assertEqual(manifest["identity"]["scheduler"], {"kind": "constant-v1"})
        self.assertIn("fake_accelerate_state.pt", manifest["state_files"])
        self.assertIn("adapter/adapter_model.safetensors", manifest["state_files"])
        self.assertEqual(manifest["rng_files"], ["random_states_0.pkl"])

    def test_resume_refuses_changed_identity_and_tampered_state(self) -> None:
        api = _api()
        phase = PhaseSpec.for_phase(1)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            model = ToyAdapterModel([])
            request = api.TrainRequest(
                model=model,
                phase=phase,
                eligible_samples=1,
                cache_checksum="c" * 64,
                checkpoint_root=Path(tmp),
                token_limit=2000,
                accumulation_steps=1,
                dataloader_factory=lambda _epoch, _accelerator: [
                    {"utts": ["only"], "target": torch.tensor([1.0])}
                ],
                accelerator_factory=FakeAccelerator,
            )
            first = api.train_phase(request, api.TrainingCallbacks(validate=lambda event: False))
            for changed in (
                replace(request, resume_from=first.checkpoint, cache_checksum="d" * 64),
                replace(request, resume_from=first.checkpoint, phase=PhaseSpec.for_phase(2)),
                replace(request, resume_from=first.checkpoint, eligible_samples=2),
                replace(request, resume_from=first.checkpoint, max_grad_norm=2.0),
                replace(request, resume_from=first.checkpoint, token_limit=3000),
                replace(request, resume_from=first.checkpoint, accumulation_steps=2),
                replace(request, resume_from=first.checkpoint, sampler_seed=1987),
                replace(request, resume_from=first.checkpoint, sampler_window_size=256),
                replace(request, resume_from=first.checkpoint, dataloader_identity="alternate-loader:v1"),
            ):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "identity changed"):
                    api.train_phase(changed, api.TrainingCallbacks(validate=lambda event: None))

            manifest_path = first.checkpoint / api.CHECKPOINT_MANIFEST
            original_manifest = manifest_path.read_bytes()
            manifest = json.loads(original_manifest)
            manifest["identity"]["scheduler"] = {
                "kind": "constant-v1",
                "unrecognized": True,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                api.train_phase(
                    replace(request, resume_from=first.checkpoint),
                    api.TrainingCallbacks(validate=lambda event: None),
                )
            manifest_path.write_bytes(original_manifest)

            state_path = first.checkpoint / "fake_accelerate_state.pt"
            state_path.write_bytes(state_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                api.train_phase(
                    replace(request, resume_from=first.checkpoint),
                    api.TrainingCallbacks(validate=lambda event: None),
                )

    def test_validation_failure_resumes_pending_checkpoint_without_sample_replay(self) -> None:
        api = _api()
        seen: list[str] = []
        model = ToyAdapterModel(seen)
        batches = [
            {"utts": [f"sample-{index}"], "target": torch.tensor([float(index + 1)])}
            for index in range(8)
        ]
        phase = PhaseSpec.for_phase(1)

        def fail_validation(_event):
            raise RuntimeError("validation unavailable")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            request = api.TrainRequest(
                model=model,
                phase=phase,
                eligible_samples=8,
                cache_checksum="c" * 64,
                checkpoint_root=Path(tmp),
                token_limit=2000,
                accumulation_steps=1,
                dataloader_factory=lambda _epoch, _accelerator: batches,
                accelerator_factory=FakeAccelerator,
            )
            with self.assertRaisesRegex(RuntimeError, "validation unavailable"):
                api.train_phase(request, api.TrainingCallbacks(validate=fail_validation))

            pending = Path(tmp) / "phase-1-validation-01"
            manifest = json.loads((pending / api.CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["validation_status"], "pending")
            recovered_indices: list[int] = []
            recovered = api.train_phase(
                replace(request, resume_from=pending),
                api.TrainingCallbacks(
                    validate=lambda event: recovered_indices.append(event.validation_index) or False
                ),
            )
            recovered_status = json.loads(
                (recovered.checkpoint / api.CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
            )["validation_status"]

        self.assertEqual(seen, ["sample-0"])
        self.assertEqual(recovered_indices, [1])
        self.assertFalse(recovered.completed)
        self.assertEqual(recovered_status, "succeeded")

    def test_interrupted_checkpoint_staging_is_discarded_without_harming_previous_checkpoint(self) -> None:
        api = _api()
        model = ToyAdapterModel([])
        batches = [
            {"utts": [f"sample-{index}"], "target": torch.tensor([float(index + 1)])}
            for index in range(8)
        ]

        class InterruptedStagingAccelerator(FakeAccelerator):
            def save_state(self, output_dir) -> None:
                super().save_state(output_dir)
                if "validation-02" in str(output_dir):
                    raise RuntimeError("staging interrupted")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(api, "audit_trainable_parameters", _audit), mock.patch.object(api, "save_adapter", _save_adapter):
            request = api.TrainRequest(
                model=model,
                phase=PhaseSpec.for_phase(1),
                eligible_samples=8,
                cache_checksum="c" * 64,
                checkpoint_root=Path(tmp),
                token_limit=2000,
                accumulation_steps=1,
                dataloader_factory=lambda _epoch, _accelerator: batches,
                accelerator_factory=FakeAccelerator,
            )
            first = api.train_phase(
                request,
                api.TrainingCallbacks(validate=lambda event: event.validation_index < 1),
            )
            previous_manifest = (first.checkpoint / api.CHECKPOINT_MANIFEST).read_bytes()
            with self.assertRaisesRegex(RuntimeError, "staging interrupted"):
                api.train_phase(
                    replace(
                        request,
                        resume_from=first.checkpoint,
                        accelerator_factory=InterruptedStagingAccelerator,
                    ),
                    api.TrainingCallbacks(validate=lambda _event: None),
                )
            stale = Path(tmp) / ".phase-1-validation-02.incomplete"
            self.assertTrue(stale.is_dir())

            resumed = api.train_phase(
                replace(request, resume_from=first.checkpoint),
                api.TrainingCallbacks(validate=lambda event: event.validation_index < 2),
            )
            previous_after = (first.checkpoint / api.CHECKPOINT_MANIFEST).read_bytes()
            stale_exists = stale.exists()

        self.assertEqual(previous_after, previous_manifest)
        self.assertEqual(resumed.progress.validation_index, 2)
        self.assertFalse(stale_exists)

    def test_production_oom_raises_without_changing_the_token_limit(self) -> None:
        api = _api()

        class OOMModel(ToyAdapterModel):
            def forward(self, batch, device):
                raise torch.cuda.OutOfMemoryError("fixture OOM")

        request = api.TrainRequest(
            model=OOMModel([]),
            phase=PhaseSpec.for_phase(1),
            eligible_samples=1,
            cache_checksum="c" * 64,
            checkpoint_root=Path("unused"),
            token_limit=5000,
            accumulation_steps=1,
            dataloader_factory=lambda _epoch, _accelerator: [
                {"utts": ["only"], "target": torch.tensor([1.0])}
            ],
            accelerator_factory=FakeAccelerator,
        )
        with mock.patch.object(api, "audit_trainable_parameters", _audit):
            with self.assertRaisesRegex(api.TrainingCapacityError, "5000"):
                api.train_phase(request, api.TrainingCallbacks(validate=lambda event: None))

        self.assertEqual(request.token_limit, 5000)

    def test_qualification_selects_largest_candidate_with_two_gib_on_every_rank(self) -> None:
        api = _api()
        gib = 1024**3
        peaks = {2000: 1 * gib, 3000: 2 * gib, 4000: 3 * gib, 5000: 5 * gib, 6000: 7 * gib}
        tried: list[int] = []

        result = api.qualify_token_limit(
            api.QualificationRequest(
                accelerator=FakeAccelerator(),
                run_candidate=lambda limit: (tried.append(limit) or peaks[limit], 8 * gib),
            )
        )

        self.assertEqual(tried, [2000, 3000, 4000, 5000, 6000])
        self.assertEqual(result.token_limit, 5000)
        self.assertEqual([item.token_limit for item in result.candidates], tried)
        self.assertTrue(all(len(item.peak_bytes_by_rank) == 8 for item in result.candidates))

    def test_qualification_refuses_when_no_candidate_has_headroom(self) -> None:
        api = _api()
        gib = 1024**3

        with self.assertRaisesRegex(api.TrainingCapacityError, "two GiB"):
            api.qualify_token_limit(
                api.QualificationRequest(
                    accelerator=FakeAccelerator(),
                    run_candidate=lambda _limit: (7 * gib, 8 * gib),
                )
            )

    def test_qualification_stops_collectively_when_any_rank_ooms(self) -> None:
        api = _api()
        tried: list[int] = []

        class RemoteOOMAccelerator(FakeAccelerator):
            def gather(self, value):
                if value.numel() == 1:
                    return torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=value.dtype)
                return super().gather(value)

        with self.assertRaises(api.TrainingCapacityError):
            api.qualify_token_limit(
                api.QualificationRequest(
                    accelerator=RemoteOOMAccelerator(),
                    run_candidate=lambda limit: (tried.append(limit) or 1, 3 * 1024**3),
                )
            )

        self.assertEqual(tried, [2000])


if __name__ == "__main__":
    unittest.main()

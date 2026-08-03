"""Tests for the isolated three-rank LoRA smoke runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cosyvoice.finetune.balalaika.artifacts import sha256_file
from cosyvoice.finetune.balalaika.cache import CacheManifest
from cosyvoice.finetune.balalaika.model import TrainableAudit
from cosyvoice.finetune.balalaika.smoke import ThreeGpuSmokeError, ThreeGpuSmokeRequest, run_three_gpu_smoke


class ThreeGpuSmokeTests(unittest.TestCase):
    def test_request_accepts_only_three_bf16_ranks_and_distinct_smoke_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = CacheManifest(root / "cache", {0: root / "cache" / "shard.json"}, {1: 2, 2: 2}, 0)
            base = root / "Fun-CosyVoice3-0.5B-2512"

            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
            )

            self.assertEqual(request.world_size, 3)
            self.assertEqual(request.steps, 2)
            self.assertEqual(request.split_plan, cache.root / "split_plan")
            with self.assertRaisesRegex(ValueError, "world_size=3"):
                ThreeGpuSmokeRequest(
                    cache=cache,
                    output_root=root / "three_gpu_smoke",
                    base_model_dir=base,
                    world_size=8,
                )
            with self.assertRaisesRegex(ValueError, "three_gpu_smoke"):
                ThreeGpuSmokeRequest(
                    cache=cache,
                    output_root=root / "memorization",
                    base_model_dir=base,
                )

    def test_runner_gathers_two_finite_losses_and_publishes_only_smoke_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            base = root / "Fun-CosyVoice3-0.5B-2512"
            base.mkdir()
            (base / "llm.pt").write_bytes(b"base-llm")
            accelerators: list[_ThreeRankAccelerator] = []

            def accelerator_factory(**kwargs):
                accelerator = _ThreeRankAccelerator(**kwargs)
                accelerators.append(accelerator)
                return accelerator

            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
                accelerator_factory=accelerator_factory,
            )
            model = _FiniteLossModel()
            with (
                mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=model),
                mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
            ):
                report = run_three_gpu_smoke(request)

            manifest = json.loads((root / "three_gpu_smoke" / "manifest.json").read_text())
            self.assertEqual(report, manifest)
            self.assertEqual(manifest["world_size"], 3)
            self.assertEqual(manifest["steps"], 2)
            self.assertEqual(
                manifest["cache_manifest_sha256"],
                sha256_file(cache.root / "manifest.json"),
            )
            self.assertEqual(len(manifest["losses_by_step"]), 2)
            self.assertTrue(all(len(values) == 3 for values in manifest["losses_by_step"]))
            self.assertFalse((root / "workflow_stages").exists())
            self.assertFalse((root / "memorization").exists())
            self.assertEqual(
                sorted(path.name for path in (root / "three_gpu_smoke").iterdir()),
                ["manifest.json"],
            )
            self.assertEqual(accelerators[0].kwargs, {"mixed_precision": "bf16"})
            self.assertEqual(accelerators[0].broadcast_calls, 2)

    def test_nonfinite_loss_removes_temporary_output_and_never_publishes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            base = root / "Fun-CosyVoice3-0.5B-2512"
            base.mkdir()
            (base / "llm.pt").write_bytes(b"base-llm")
            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
                accelerator_factory=lambda **kwargs: _ThreeRankAccelerator(**kwargs),
            )
            with (
                mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_NonfiniteLossModel()),
                mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
            ):
                with self.assertRaisesRegex(ThreeGpuSmokeError, "finite"):
                    run_three_gpu_smoke(request)

            self.assertFalse((root / "three_gpu_smoke").exists())
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_missing_outer_cache_manifest_is_rejected_before_data_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            (cache.root / "manifest.json").unlink()
            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=root / "Fun-CosyVoice3-0.5B-2512",
                accelerator_factory=lambda **kwargs: _ThreeRankAccelerator(**kwargs),
            )

            with mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows") as select:
                with self.assertRaisesRegex(ThreeGpuSmokeError, "manifest.*missing"):
                    run_three_gpu_smoke(request)

            select.assert_not_called()
            self.assertFalse((root / "three_gpu_smoke").exists())
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_rank_zero_preflight_rejects_existing_target_before_loader_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            target = root / "three_gpu_smoke"
            target.mkdir()
            accelerators: list[_ThreeRankAccelerator] = []

            def accelerator_factory(**kwargs):
                accelerator = _ThreeRankAccelerator(**kwargs)
                accelerators.append(accelerator)
                return accelerator

            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=target,
                base_model_dir=root / "Fun-CosyVoice3-0.5B-2512",
                accelerator_factory=accelerator_factory,
            )

            with self.assertRaisesRegex(ThreeGpuSmokeError, "already exists"):
                run_three_gpu_smoke(request)

            self.assertTrue(target.exists())
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())
            self.assertEqual(accelerators[0].broadcast_calls, 2)

    def test_collective_failures_teardown_without_a_following_status_gather(self) -> None:
        for operation in ("prepare", "backward", "gather"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = _fixture_cache(root)
                base = root / "Fun-CosyVoice3-0.5B-2512"
                base.mkdir()
                (base / "llm.pt").write_bytes(b"base-llm")
                accelerators: list[_FailingCollectiveAccelerator] = []

                def accelerator_factory(**kwargs):
                    accelerator = _FailingCollectiveAccelerator(operation, **kwargs)
                    accelerators.append(accelerator)
                    return accelerator

                request = ThreeGpuSmokeRequest(
                    cache=cache,
                    output_root=root / "three_gpu_smoke",
                    base_model_dir=base,
                    accelerator_factory=accelerator_factory,
                )
                with (
                    mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_FiniteLossModel()),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
                ):
                    with self.assertRaisesRegex(ThreeGpuSmokeError, f"{operation} collective"):
                        run_three_gpu_smoke(request)

                accelerator = accelerators[0]
                self.assertEqual(accelerator.teardown_calls, 1)
                self.assertEqual(accelerator.status_gathers_after_failure, 0)
                self.assertFalse((root / "three_gpu_smoke").exists())
                self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_control_collective_failures_abort_cleanup_and_stop_collectives(self) -> None:
        for operation in ("broadcast", "status_gather", "barrier"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = _fixture_cache(root)
                base = root / "Fun-CosyVoice3-0.5B-2512"
                base.mkdir()
                (base / "llm.pt").write_bytes(b"base-llm")
                accelerators: list[_FailingControlCollectiveAccelerator] = []

                def accelerator_factory(**kwargs):
                    accelerator = _FailingControlCollectiveAccelerator(operation, **kwargs)
                    accelerators.append(accelerator)
                    return accelerator

                request = ThreeGpuSmokeRequest(
                    cache=cache,
                    output_root=root / "three_gpu_smoke",
                    base_model_dir=base,
                    accelerator_factory=accelerator_factory,
                )
                with (
                    mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_FiniteLossModel()),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                    mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
                ):
                    with self.assertRaisesRegex(ThreeGpuSmokeError, "collective failed"):
                        run_three_gpu_smoke(request)

                accelerator = accelerators[0]
                self.assertEqual(accelerator.teardown_calls, 1)
                self.assertEqual(accelerator.collectives_after_failure, 0)
                self.assertFalse((root / "three_gpu_smoke").exists())
                self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_cleanup_barrier_failure_aborts_without_another_collective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            base = root / "Fun-CosyVoice3-0.5B-2512"
            base.mkdir()
            (base / "llm.pt").write_bytes(b"base-llm")
            accelerator = _FailingCleanupBarrierAccelerator()
            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
                accelerator_factory=lambda **_kwargs: accelerator,
            )
            with (
                mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_NonfiniteLossModel()),
                mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
            ):
                with self.assertRaisesRegex(ThreeGpuSmokeError, "cleanup barrier collective failed"):
                    run_three_gpu_smoke(request)

            self.assertEqual(accelerator.teardown_calls, 1)
            self.assertEqual(accelerator.collectives_after_failure, 0)
            self.assertFalse((root / "three_gpu_smoke").exists())
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_non_main_collective_failure_removes_shared_temporary_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            base = root / "Fun-CosyVoice3-0.5B-2512"
            base.mkdir()
            (base / "llm.pt").write_bytes(b"base-llm")
            accelerator = _NonMainPrepareFailureAccelerator(root / ".three_gpu_smoke.incomplete")
            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
                accelerator_factory=lambda **_kwargs: accelerator,
            )
            with (
                mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_FiniteLossModel()),
                mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
            ):
                with self.assertRaisesRegex(ThreeGpuSmokeError, "prepare collective failed"):
                    run_three_gpu_smoke(request)

            self.assertEqual(accelerator.teardown_calls, 1)
            self.assertEqual(accelerator.status_gathers_after_failure, 0)
            self.assertFalse((root / "three_gpu_smoke").exists())
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_non_main_preflight_collective_failures_remove_owned_temporary_output(self) -> None:
        for operation in ("broadcast", "barrier"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                accelerator = _NonMainPreflightFailureAccelerator(
                    operation,
                    root / ".three_gpu_smoke.incomplete",
                )
                request = ThreeGpuSmokeRequest(
                    cache=_fixture_cache(root),
                    output_root=root / "three_gpu_smoke",
                    base_model_dir=root / "Fun-CosyVoice3-0.5B-2512",
                    accelerator_factory=lambda **_kwargs: accelerator,
                )

                with self.assertRaisesRegex(ThreeGpuSmokeError, f"preflight {operation} collective failed"):
                    run_three_gpu_smoke(request)

                self.assertEqual(accelerator.broadcast_calls, 2)
                self.assertEqual(accelerator.teardown_calls, 1)
                self.assertEqual(accelerator.collectives_after_failure, 0)
                self.assertFalse((root / "three_gpu_smoke").exists())
                self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_rename_collision_preserves_foreign_target_and_removes_owned_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            base = root / "Fun-CosyVoice3-0.5B-2512"
            base.mkdir()
            (base / "llm.pt").write_bytes(b"base-llm")
            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
                accelerator_factory=lambda **kwargs: _ThreeRankAccelerator(**kwargs),
            )
            replace = Path.replace

            def collide(source: Path, target: Path) -> Path:
                if source == request.temporary_root:
                    target.mkdir()
                    (target / "foreign.txt").write_text("do not delete", encoding="utf-8")
                return replace(source, target)

            with (
                mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_FiniteLossModel()),
                mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
                mock.patch.object(Path, "replace", autospec=True, side_effect=collide),
            ):
                with self.assertRaisesRegex(ThreeGpuSmokeError, "FileExistsError"):
                    run_three_gpu_smoke(request)

            self.assertEqual((root / "three_gpu_smoke/foreign.txt").read_text(encoding="utf-8"), "do not delete")
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())

    def test_status_gather_failure_after_publish_removes_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = _fixture_cache(root)
            base = root / "Fun-CosyVoice3-0.5B-2512"
            base.mkdir()
            (base / "llm.pt").write_bytes(b"base-llm")
            accelerator = _PostPublishStatusFailureAccelerator(root / "three_gpu_smoke")
            request = ThreeGpuSmokeRequest(
                cache=cache,
                output_root=root / "three_gpu_smoke",
                base_model_dir=base,
                accelerator_factory=lambda **_kwargs: accelerator,
            )
            with (
                mock.patch("cosyvoice.finetune.balalaika.smoke.select_memorization_rows", return_value=(object(),)),
                mock.patch("cosyvoice.finetune.balalaika.smoke.build_memorization_dataloader", return_value=[{"target": torch.tensor(1.0)}]),
                mock.patch("cosyvoice.finetune.balalaika.smoke.load_base_llm", return_value=_FiniteLossModel()),
                mock.patch("cosyvoice.finetune.balalaika.smoke.inject_lora", side_effect=lambda value, _: value),
                mock.patch("cosyvoice.finetune.balalaika.smoke.audit_trainable_parameters", return_value=_audit()),
            ):
                with self.assertRaisesRegex(ThreeGpuSmokeError, "status gather collective failed"):
                    run_three_gpu_smoke(request)

            self.assertEqual(accelerator.teardown_calls, 1)
            self.assertEqual(accelerator.collectives_after_failure, 0)
            self.assertFalse((root / "three_gpu_smoke").exists())
            self.assertFalse((root / ".three_gpu_smoke.incomplete").exists())


def _fixture_cache(root: Path) -> CacheManifest:
    cache_root = root / "cache"
    (cache_root / "phase1").mkdir(parents=True)
    (cache_root / "phase2").mkdir()
    (cache_root / "split_plan").mkdir()
    phase1 = cache_root / "phase1/shard_000000.parquet"
    phase2 = cache_root / "phase2/shard_000000.parquet"
    split_plan = cache_root / "split_plan/shard_000000.jsonl"
    phase1.write_bytes(b"phase-one")
    phase2.write_bytes(b"phase-two")
    split_plan.write_text('{"phase":1}\n{"phase":2}\n', encoding="utf-8")
    shard = cache_root / "shard_manifest.json"
    shard.write_text(
        json.dumps({
            "format_version": 1,
            "rows": 4,
            "phase1_sha256": sha256_file(phase1),
            "phase2_sha256": sha256_file(phase2),
            "plan_sha256": sha256_file(split_plan),
        }),
        encoding="utf-8",
    )
    (cache_root / "manifest.json").write_text(
        json.dumps({
            "format_version": 1,
            "shard_manifest": "shard_manifest.json",
            "shard_manifest_sha256": sha256_file(shard),
            "phase_rows": {"1": 2, "2": 2},
            "seed": 1986,
        }),
        encoding="utf-8",
    )
    return CacheManifest(cache_root, {0: shard}, {1: 2, 2: 2}, 0)


def _audit() -> TrainableAudit:
    return TrainableAudit(
        target_modules=("speech_embedding", "llm_decoder", "llm.model.model.embed_tokens"),
        trainable_parameters=(
            "speech_embedding.lora_embedding_A.default",
            "speech_embedding.lora_embedding_B.default",
            "llm_decoder.lora_A.default.weight",
            "llm_decoder.lora_B.default.weight",
            "llm.model.model.embed_tokens.lora_embedding_A.default",
            "llm.model.model.embed_tokens.lora_embedding_B.default",
        ),
        unexpected_dense_parameters=(),
        trainable_parameter_count=12,
        total_parameter_count=120,
    )


class _FiniteLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, _device):
        return {"loss": (self.weight - batch["target"]) ** 2}


class _NonfiniteLossModel(_FiniteLossModel):
    def forward(self, batch, _device):
        return {"loss": self.weight * float("nan")}


class _ThreeRankAccelerator:
    device = torch.device("cpu")
    num_processes = 3
    process_index = 0
    is_main_process = True

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.broadcast_calls = 0

    def prepare(self, *values):
        return values

    def backward(self, loss) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def gather(self, value):
        return value.reshape(-1).repeat(3)

    def gather_object(self, value):
        return [item for _rank in range(self.num_processes) for item in value]

    def broadcast_object_list(self, values, from_process):
        if from_process != 0:
            raise AssertionError(from_process)
        self.broadcast_calls += 1

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self) -> None:
        return None


class _FailingCollectiveAccelerator(_ThreeRankAccelerator):
    def __init__(self, failing_operation: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failing_operation = failing_operation
        self.collective_failed = False
        self.status_gathers_after_failure = 0
        self.teardown_calls = 0

    def prepare(self, *values):
        if self.failing_operation == "prepare":
            self.collective_failed = True
            raise RuntimeError("simulated prepare collective failure")
        return super().prepare(*values)

    def backward(self, loss) -> None:
        if self.failing_operation == "backward":
            self.collective_failed = True
            raise RuntimeError("simulated backward collective failure")
        super().backward(loss)

    def gather(self, value):
        if self.failing_operation == "gather":
            self.collective_failed = True
            raise RuntimeError("simulated gather collective failure")
        return super().gather(value)

    def gather_object(self, value):
        if self.collective_failed:
            self.status_gathers_after_failure += 1
            raise AssertionError("status gather after failed collective would deadlock")
        return super().gather_object(value)

    def end_training(self) -> None:
        self.teardown_calls += 1


class _FailingControlCollectiveAccelerator(_ThreeRankAccelerator):
    def __init__(self, failing_operation: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failing_operation = failing_operation
        self.collective_failed = False
        self.collectives_after_failure = 0
        self.teardown_calls = 0

    def _fail_or_guard(self, operation: str) -> None:
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("collective invoked after process-group failure")
        if self.failing_operation == operation:
            self.collective_failed = True
            raise RuntimeError(f"simulated {operation} collective failure")

    def broadcast_object_list(self, values, from_process):
        self._fail_or_guard("broadcast")
        super().broadcast_object_list(values, from_process)

    def gather_object(self, value):
        self._fail_or_guard("status_gather")
        return super().gather_object(value)

    def wait_for_everyone(self) -> None:
        self._fail_or_guard("barrier")

    def end_training(self) -> None:
        self.teardown_calls += 1


class _FailingCleanupBarrierAccelerator(_ThreeRankAccelerator):
    def __init__(self) -> None:
        super().__init__()
        self.failure_status_seen = False
        self.failure_barriers = 0
        self.collective_failed = False
        self.collectives_after_failure = 0
        self.teardown_calls = 0

    def gather_object(self, value):
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("status gather after process-group failure")
        status = value[0]
        if isinstance(status, dict) and status.get("error"):
            self.failure_status_seen = True
        return super().gather_object(value)

    def wait_for_everyone(self) -> None:
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("barrier after process-group failure")
        if self.failure_status_seen:
            self.failure_barriers += 1
            if self.failure_barriers == 2:
                self.collective_failed = True
                raise RuntimeError("simulated cleanup barrier collective failure")

    def end_training(self) -> None:
        self.teardown_calls += 1


class _NonMainPrepareFailureAccelerator(_FailingCollectiveAccelerator):
    process_index = 1
    is_main_process = False

    def __init__(self, temporary_root: Path) -> None:
        super().__init__("prepare")
        self.temporary_root = temporary_root

    def broadcast_object_list(self, values, from_process):
        super().broadcast_object_list(values, from_process)
        if self.broadcast_calls == 1:
            values[0] = "b" * 32
            return
        values[0] = {"rank": 0, "error": None}
        self.temporary_root.mkdir(parents=True)
        (self.temporary_root / ".attempt_id").write_text("b" * 32, encoding="ascii")


class _NonMainPreflightFailureAccelerator(_ThreeRankAccelerator):
    process_index = 1
    is_main_process = False

    def __init__(self, failing_operation: str, temporary_root: Path) -> None:
        super().__init__()
        self.failing_operation = failing_operation
        self.temporary_root = temporary_root
        self.attempt_id = "a" * 32
        self.collective_failed = False
        self.collectives_after_failure = 0
        self.teardown_calls = 0

    def broadcast_object_list(self, values, from_process):
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("broadcast after process-group failure")
        self.broadcast_calls += 1
        if self.broadcast_calls == 1:
            values[0] = self.attempt_id
            return
        self.temporary_root.mkdir(parents=True)
        (self.temporary_root / ".attempt_id").write_text(self.attempt_id, encoding="ascii")
        if self.failing_operation == "broadcast":
            self.collective_failed = True
            raise RuntimeError("simulated preflight broadcast collective failure")
        values[0] = {"rank": 0, "error": None}

    def wait_for_everyone(self) -> None:
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("barrier after process-group failure")
        if self.failing_operation == "barrier" and self.broadcast_calls == 2:
            self.collective_failed = True
            raise RuntimeError("simulated preflight barrier collective failure")

    def end_training(self) -> None:
        self.teardown_calls += 1


class _PostPublishStatusFailureAccelerator(_ThreeRankAccelerator):
    def __init__(self, output_root: Path) -> None:
        super().__init__()
        self.output_root = output_root
        self.collective_failed = False
        self.collectives_after_failure = 0
        self.teardown_calls = 0

    def gather_object(self, value):
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("status gather after process-group failure")
        if self.output_root.exists():
            self.collective_failed = True
            raise RuntimeError("simulated post-publish status gather collective failure")
        return super().gather_object(value)

    def wait_for_everyone(self) -> None:
        if self.collective_failed:
            self.collectives_after_failure += 1
            raise AssertionError("barrier after process-group failure")

    def end_training(self) -> None:
        self.teardown_calls += 1


if __name__ == "__main__":
    unittest.main()

"""Tests for the isolated three-rank LoRA smoke runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

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
            self.assertEqual(len(manifest["losses_by_step"]), 2)
            self.assertTrue(all(len(values) == 3 for values in manifest["losses_by_step"]))
            self.assertFalse((root / "workflow_stages").exists())
            self.assertFalse((root / "memorization").exists())
            self.assertEqual(accelerators[0].kwargs, {"mixed_precision": "bf16"})
            self.assertEqual(accelerators[0].broadcast_calls, 1)

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
            self.assertEqual(accelerators[0].broadcast_calls, 1)

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


def _fixture_cache(root: Path) -> CacheManifest:
    cache_root = root / "cache"
    shard = cache_root / "shard_manifests" / "shard_000000.json"
    shard.parent.mkdir(parents=True)
    shard.write_text("{}", encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()

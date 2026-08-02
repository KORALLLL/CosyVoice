from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from cosyvoice.finetune.balalaika.artifacts import StageRequirementError, sha256_file
from cosyvoice.finetune.balalaika.config import PhaseSpec, RunPaths
from cosyvoice.finetune.balalaika.workflow import (
    ExitCode,
    ProductionBackend,
    WorkflowOptions,
    _collective_call,
    _stage_payload,
    main,
    redact_secrets,
    run_phase1,
    run_phase2,
)


class FakeCoordinator:
    is_main_process = True
    process_index = 0
    num_processes = 8

    def __init__(self) -> None:
        self.broadcasts: list[object] = []
        self.gathers: list[object] = []
        self.barriers = 0

    def broadcast(self, value: object) -> object:
        self.broadcasts.append(value)
        return value

    def gather(self, value: object) -> list[object]:
        self.gathers.append(value)
        return [value] * 8

    def barrier(self) -> None:
        self.barriers += 1


class FakeBackend:
    def __init__(self) -> None:
        self.coordinator = FakeCoordinator()
        self.calls: list[object] = []
        self.validation_indices: list[int] = []
        self.generations: list[int] = []
        self.invalid_stages: set[str] = set()
        self.fail_at: str | None = None
        self.phase1_adapter_sha256 = "1" * 64

    def _call(self, name: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        self.calls.append(name if payload is None else (name, payload))
        if self.fail_at == name:
            raise RuntimeError(f"failed:{name}")
        return {"operation": name, "sha256": (str(len(self.calls))[-1] * 64)}

    def authenticate_stage(self, name: str, payload: dict[str, object]) -> None:
        self.calls.append(("authenticate", name))
        if name in self.invalid_stages:
            raise StageRequirementError(f"invalid evidence: {name}")

    def preflight(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("preflight")

    def qualify_tokenizer(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("qualify_tokenizer")

    def ensure_pilot(self, options: WorkflowOptions) -> dict[str, object]:
        result = self._call("ensure_pilot")
        result["pilot_manifest_sha256"] = "a" * 64
        return result

    def approve_pilot(self, options: WorkflowOptions, checksum: str) -> dict[str, object]:
        self.calls.append(("approve_pilot", checksum))
        if checksum != "a" * 64:
            raise StageRequirementError("pilot approval checksum mismatch")
        return {"pilot_manifest_sha256": checksum}

    def memorize(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("memorize")

    def prepare_memorization(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("prepare_memorization")

    def build_cache(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("build_cache")

    def capacity_smoke(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("capacity_smoke")

    def ensure_logging(self, options: WorkflowOptions) -> dict[str, object]:
        return self._call("ensure_logging")

    def evaluate(self, options: WorkflowOptions, validation_index: int, generations: int) -> dict[str, object]:
        self.validation_indices.append(validation_index)
        self.generations.append(generations)
        result = self._call(f"evaluate_{validation_index}")
        result.update({
            "validation_index": validation_index,
            "generations": generations,
            "identity_sha256": f"{validation_index % 10}" * 64,
            "artifact_checksums": {"summary_json": f"{validation_index % 10}" * 64},
        })
        return result

    def train(
        self,
        options: WorkflowOptions,
        phase: PhaseSpec,
        validation_indices: tuple[int, ...],
        *,
        initial_adapter_sha256: str | None,
        fresh_optimizer: bool,
    ) -> dict[str, object]:
        self.calls.append(("train", phase, validation_indices, initial_adapter_sha256, fresh_optimizer))
        validations = []
        for index in validation_indices:
            validations.append(self.evaluate(options, index, 2_000))
        return {
            "completed": True,
            "checkpoint_manifest_sha256": "c" * 64,
            "adapter_weights_sha256": self.phase1_adapter_sha256 if phase.number == 1 else "2" * 64,
            "final_validation_index": validation_indices[-1],
            "fresh_optimizer": fresh_optimizer,
            "training_identity": {
                "phase": phase.number,
                "initial_adapter_sha256": initial_adapter_sha256,
                "validation_index_base": 0 if phase.number == 1 else 16,
            },
            "validations": validations,
        }

    def export(self, options: WorkflowOptions, phase2: dict[str, object]) -> dict[str, object]:
        result = self._call("export")
        result.update({"production_ready": True, "final_manifest_sha256": "f" * 64})
        return result


def _args(root: Path, backend: FakeBackend, approval: str | None = None) -> argparse.Namespace:
    paths = RunPaths(
        dataset_root=root / "dataset",
        repository_root=root / "repo",
        run_root=root / "run",
        base_model_dir=root / "repo/pretrained_models/Fun-CosyVoice3-0.5B-2512",
        visible_devices=tuple(range(8)),
        seed=1986,
    )
    return argparse.Namespace(
        options=WorkflowOptions(paths=paths, approve_pilot_sha256=approval),
        backend=backend,
    )


class WorkflowTests(unittest.TestCase):
    def test_phase1_stops_for_manual_pilot_review_before_later_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend()
            result = run_phase1(_args(Path(directory), backend))
            self.assertEqual(result, ExitCode.PILOT_REVIEW_REQUIRED)
            names = [
                item if isinstance(item, str) else item[0]
                for item in backend.calls
                if not (isinstance(item, tuple) and item[0] == "authenticate")
            ]
            self.assertEqual(names[:3], ["preflight", "qualify_tokenizer", "ensure_pilot"])
            self.assertNotIn("memorize", names)
            self.assertNotIn("build_cache", names)
            self.assertNotIn("capacity_smoke", names)
            self.assertFalse((_args(Path(directory), backend).options.paths.run_root / "workflow_stages/phase1_complete.json").exists())

    def test_pilot_approval_is_checksum_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend()
            with self.assertRaises(StageRequirementError):
                run_phase1(_args(Path(directory), backend, "b" * 64))
            self.assertNotIn("memorize", backend.calls)

    def test_exact_gate_order_phase_specs_and_41_validations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeBackend()
            self.assertEqual(run_phase1(_args(root, backend, "a" * 64)), ExitCode.SUCCESS)
            operations = [item for item in backend.calls if isinstance(item, str)]
            self.assertEqual(
                operations[:9],
                [
                    "preflight", "qualify_tokenizer", "ensure_pilot", "prepare_memorization",
                    "memorize", "build_cache", "capacity_smoke", "ensure_logging", "evaluate_0",
                ],
            )
            self.assertEqual(run_phase2(_args(root, backend)), ExitCode.SUCCESS)
            self.assertEqual(backend.validation_indices, list(range(41)))
            self.assertEqual(backend.generations, [2_000] * 41)
            trains = [item for item in backend.calls if isinstance(item, tuple) and item[0] == "train"]
            self.assertEqual([item[1] for item in trains], [PhaseSpec.for_phase(1), PhaseSpec.for_phase(2)])
            self.assertEqual(trains[0][2], tuple(range(1, 17)))
            self.assertIsNone(trains[0][3])
            self.assertTrue(trains[0][4])
            self.assertEqual(trains[1][2], tuple(range(17, 41)))
            self.assertEqual(trains[1][3], "1" * 64)
            self.assertTrue(trains[1][4])

    def test_phase2_requires_authenticated_phase1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend()
            with self.assertRaises(StageRequirementError):
                run_phase2(_args(Path(directory), backend))

    def test_failed_operation_never_publishes_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeBackend()
            backend.fail_at = "capacity_smoke"
            with self.assertRaises(RuntimeError):
                run_phase1(_args(root, backend, "a" * 64))
            self.assertFalse((root / "run/workflow_stages/phase1_complete.json").exists())

    def test_resume_skips_only_authenticated_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = FakeBackend()
            self.assertEqual(run_phase1(_args(root, first, "a" * 64)), ExitCode.SUCCESS)
            resumed = FakeBackend()
            self.assertEqual(run_phase1(_args(root, resumed, "a" * 64)), ExitCode.SUCCESS)
            self.assertFalse(any(item == "preflight" for item in resumed.calls))
            self.assertIn(("authenticate", "phase1_complete"), resumed.calls)
            corrupt = FakeBackend()
            corrupt.invalid_stages.add("phase1_complete")
            with self.assertRaises(StageRequirementError):
                run_phase1(_args(root, corrupt, "a" * 64))

    def test_collective_and_main_calls_synchronize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend()
            run_phase1(_args(Path(directory), backend, "a" * 64))
            self.assertGreater(backend.coordinator.barriers, 0)
            self.assertGreater(len(backend.coordinator.broadcasts), 0)
            self.assertGreater(len(backend.coordinator.gathers), 0)

    def test_collective_remote_error_is_propagated_on_every_rank(self) -> None:
        coordinator = FakeCoordinator()

        def gather_with_remote_error(value: object) -> list[object]:
            return [value, {"ok": False, "rank": 1, "type": "StageRequirementError", "error": "remote gate"}]

        coordinator.gather = gather_with_remote_error  # type: ignore[method-assign]
        with self.assertRaisesRegex(StageRequirementError, "remote gate"):
            _collective_call(coordinator, "qualification", lambda: {"ok": True})
        self.assertEqual(coordinator.barriers, 1)

    def test_recursive_secret_redaction(self) -> None:
        value = {
            "HF_TOKEN": "hugging-face-secret",
            "nested": {"WANDB_API_KEY": "wandb-secret", "safe": "ok"},
            "items": [{"client_secret": "oauth-secret"}],
        }
        rendered = json.dumps(redact_secrets(value))
        self.assertNotIn("hugging-face-secret", rendered)
        self.assertNotIn("wandb-secret", rendered)
        self.assertNotIn("oauth-secret", rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_production_backend_reuses_one_wandb_validation_logger_per_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ProductionBackend.__new__(ProductionBackend)
            backend.accelerator = object()
            backend._validation_logger = None
            options = _args(Path(directory), FakeBackend()).options
            first = backend._wandb_logger(options)
            first._owned_run_id = "same-active-run"
            second = backend._wandb_logger(options)
            self.assertIs(first, second)
            self.assertEqual(second._owned_run_id, "same-active-run")

    def test_production_phase_authentication_requires_all_committed_validations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = _args(root, FakeBackend()).options
            checkpoint = root / "phase1-checkpoint"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"adapter")
            (adapter / "adapter_manifest.json").write_text(json.dumps({
                "weights": weights.name,
                "weights_sha256": sha256_file(weights),
            }), encoding="utf-8")
            checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
            checkpoint_manifest.write_text(json.dumps({
                "validation_status": "succeeded",
                "progress": {"phase": 1, "validation_index": 16},
            }), encoding="utf-8")
            summary = root / "validation-16-summary.json"
            summary.write_text(json.dumps({"validation_index": 16}), encoding="utf-8")
            validations = [{"validation_index": index} for index in range(1, 17)]
            evidence = {
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
                "adapter_dir": str(adapter),
                "adapter_weights_sha256": sha256_file(weights),
                "validation_summary": str(summary),
                "validations": validations,
                "training_identity": {"phase": 1, "initial_adapter_sha256": None},
            }
            backend = ProductionBackend.__new__(ProductionBackend)
            with mock.patch.object(backend, "_verify_required_validations") as verify:
                backend.authenticate_stage("phase1_complete", _stage_payload(options, evidence))
            verify.assert_called_once_with(options, tuple(range(1, 17)), validations)

    def test_cli_rejects_secret_flags_and_bad_values(self) -> None:
        with self.assertRaises(SystemExit):
            main(["phase1", "--hf-token", "secret"])
        with self.assertRaises(SystemExit):
            main(["phase1", "--token-limit", "0"])
        with self.assertRaises(SystemExit):
            main(["phase1", "--approve-pilot-sha256", "not-a-digest"])

    def test_status_is_read_only_and_secret_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "never-created"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), mock.patch.dict(
                os.environ, {"HF_TOKEN": "should-not-leak"}, clear=False
            ):
                code = main(["status", "--run-root", str(root)])
            self.assertEqual(code, ExitCode.SUCCESS)
            self.assertFalse(root.exists())
            self.assertNotIn("should-not-leak", stdout.getvalue())

    def test_recipe_has_exactly_two_accelerate_launchers(self) -> None:
        recipe = Path(__file__).parents[3] / "examples/balalaika/cosyvoice3_lora"
        launchers = sorted(recipe.glob("*.sh"))
        self.assertEqual([path.name for path in launchers], ["run_phase1.sh", "run_phase2.sh"])
        for path, phase in zip(launchers, ("phase1", "phase2"), strict=True):
            text = path.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", text)
            self.assertIn("accelerate launch", text)
            self.assertIn("--num_processes 8", text)
            self.assertIn("--mixed_precision bf16", text)
            self.assertIn("examples/balalaika/cosyvoice3_lora/conf/accelerate.yaml", text)
            self.assertIn(f"cosyvoice.finetune.balalaika.workflow {phase}", text)
            self.assertNotIn("torchrun", text)
            self.assertNotIn("huggingface_hub", text)
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_workflow_has_no_upload_surface(self) -> None:
        source = (Path(__file__).parents[3] / "cosyvoice/finetune/balalaika/workflow.py").read_text(encoding="utf-8")
        for forbidden in ("upload_file", "upload_folder", "HfApi", "create_repo", "push_to_hub"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

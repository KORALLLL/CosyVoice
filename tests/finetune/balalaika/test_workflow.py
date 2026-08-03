from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from datetime import timedelta
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
    _AccelerateCoordinator,
    _collective_call,
    _accelerator_scope,
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
        result.update({"mode": "production", "production_ready": True, "final_manifest_sha256": "f" * 64})
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
    def test_accelerate_coordinator_wraps_object_before_flattening_gather(self) -> None:
        class Accelerator:
            is_main_process = True
            process_index = 0
            num_processes = 2

            def __init__(self) -> None:
                self.seen: object | None = None

            def gather_object(self, value: object) -> list[object]:
                self.seen = value
                return [item for rank_value in (value, value) for item in rank_value]

        accelerator = Accelerator()
        coordinator = _AccelerateCoordinator(accelerator)
        payload = {"ok": False, "error": "device failed"}

        gathered = coordinator.gather(payload)

        self.assertEqual(accelerator.seen, [payload])
        self.assertEqual(gathered, [payload, payload])

    def test_production_tokenizer_qualification_gathers_one_local_device_per_rank(self) -> None:
        backend = ProductionBackend.__new__(ProductionBackend)
        backend.coordinator = FakeCoordinator()
        options = _args(Path("/tmp/qualification-fixture"), FakeBackend()).options
        records = [{"device": device, "token_count": 25} for device in range(8)]
        statuses = [{"ok": True, "record": record} for record in records]
        backend.coordinator.gather = lambda value: statuses  # type: ignore[method-assign]
        payload = {"model_sha256": "a" * 64, "devices": records}

        with (
            mock.patch(
                "cosyvoice.finetune.balalaika.tokenizer.qualify_tokenizer_device",
                return_value=records[0],
            ) as qualify,
            mock.patch(
                "cosyvoice.finetune.balalaika.tokenizer.publish_tokenizer_qualification",
                return_value=payload,
            ) as publish,
        ):
            result = backend.qualify_tokenizer(options)

        qualify.assert_called_once_with(options.paths, 0)
        publish.assert_called_once_with(options.paths, records)
        self.assertEqual(result, payload)

    def test_production_tokenizer_qualification_gathers_local_error_before_raising(self) -> None:
        backend = ProductionBackend.__new__(ProductionBackend)
        backend.coordinator = FakeCoordinator()
        options = _args(Path("/tmp/qualification-error-fixture"), FakeBackend()).options
        remote_error = {
            "ok": False,
            "rank": 3,
            "type": "TokenizerError",
            "error": "device 3 failed",
        }
        backend.coordinator.gather = lambda value: [value, remote_error]  # type: ignore[method-assign]

        with (
            mock.patch(
                "cosyvoice.finetune.balalaika.tokenizer.qualify_tokenizer_device",
                return_value={"device": 0},
            ),
            mock.patch(
                "cosyvoice.finetune.balalaika.tokenizer.publish_tokenizer_qualification"
            ) as publish,
        ):
            with self.assertRaisesRegex(RuntimeError, "device 3 failed"):
                backend.qualify_tokenizer(options)

        publish.assert_not_called()

    def test_production_resume_reconstructs_only_prior_committed_validation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = ProductionBackend.__new__(ProductionBackend)
            backend._validation_logger = None
            backend.coordinator = FakeCoordinator()
            options = _args(root, FakeBackend()).options
            checkpoint = root / "resume"
            checkpoint.mkdir()
            manifest = checkpoint / "checkpoint_manifest.json"
            calls: list[int] = []

            def committed(_options: WorkflowOptions, index: int) -> dict[str, object]:
                calls.append(index)
                return {"validation_index": index, "generations": 2_000}

            backend._committed_validation_evidence = committed  # type: ignore[method-assign]
            for status, current, expected in (
                ("succeeded", 4, [1, 2, 3, 4]),
                ("pending", 4, [1, 2, 3]),
                ("succeeded", 16, list(range(1, 17))),
            ):
                with self.subTest(status=status, current=current):
                    manifest.write_text(json.dumps({
                        "validation_status": status,
                        "progress": {"phase": 1, "validation_index": current},
                    }), encoding="utf-8")
                    resumed = backend._resume_validation_evidence(
                        options,
                        PhaseSpec.for_phase(1),
                        tuple(range(1, 17)),
                        checkpoint,
                    )
                    self.assertEqual([item["validation_index"] for item in resumed], expected)
                    self.assertEqual(calls, expected)
                    calls.clear()

            for phase_number, current, indices in (
                (1, 0, tuple(range(1, 17))),
                (2, 16, tuple(range(17, 41))),
            ):
                with self.subTest(phase=phase_number, synthetic_base=current):
                    manifest.write_text(json.dumps({
                        "validation_status": "succeeded",
                        "progress": {"phase": phase_number, "validation_index": current},
                    }), encoding="utf-8")
                    with self.assertRaisesRegex(StageRequirementError, "outside the phase schedule"):
                        backend._resume_validation_evidence(
                            options,
                            PhaseSpec.for_phase(phase_number),
                            indices,
                            checkpoint,
                        )
                    self.assertEqual(calls, [])

    def test_production_resume_evidence_is_verified_only_on_main_and_broadcast_to_peers(self) -> None:
        class Channel:
            value: object = None

        class RankCoordinator(FakeCoordinator):
            def __init__(self, *, main: bool, channel: Channel) -> None:
                super().__init__()
                self.is_main_process = main
                self.process_index = 0 if main else 1
                self.channel = channel

            def broadcast(self, value: object) -> object:
                self.broadcasts.append(value)
                if self.is_main_process:
                    self.channel.value = value
                return self.channel.value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = _args(root, FakeBackend()).options
            checkpoint = root / "resume"
            checkpoint.mkdir()
            (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
                "validation_status": "succeeded",
                "progress": {"phase": 1, "validation_index": 2},
            }), encoding="utf-8")
            channel = Channel()
            main = ProductionBackend.__new__(ProductionBackend)
            peer = ProductionBackend.__new__(ProductionBackend)
            main.coordinator = RankCoordinator(main=True, channel=channel)
            peer.coordinator = RankCoordinator(main=False, channel=channel)
            main_calls: list[int] = []
            peer_calls: list[int] = []

            def committed(_options: WorkflowOptions, index: int) -> dict[str, object]:
                main_calls.append(index)
                return {
                    "validation_index": index,
                    "generations": 2_000,
                    "summary": f"summary-{index}.json",
                    "summary_sha256": str(index) * 64,
                    "identity_sha256": str(index) * 64,
                    "artifact_checksums": {"results_jsonl": str(index) * 64},
                }

            main._committed_validation_evidence = committed  # type: ignore[method-assign]
            peer._committed_validation_evidence = lambda _options, index: peer_calls.append(index) or {}  # type: ignore[method-assign]
            expected = main._resume_validation_evidence(
                options, PhaseSpec.for_phase(1), tuple(range(1, 17)), checkpoint,
            )
            received = peer._resume_validation_evidence(
                options, PhaseSpec.for_phase(1), tuple(range(1, 17)), checkpoint,
            )

            self.assertEqual(received, expected)
            self.assertEqual([item["validation_index"] for item in received], [1, 2])
            self.assertEqual(main_calls, [1, 2])
            self.assertEqual(peer_calls, [])
            self.assertEqual(main.coordinator.barriers, 1)
            self.assertEqual(peer.coordinator.barriers, 1)

    def test_production_resume_evidence_broadcasts_main_verification_error_to_peers(self) -> None:
        class Channel:
            value: object = None

        class RankCoordinator(FakeCoordinator):
            def __init__(self, *, main: bool, channel: Channel) -> None:
                super().__init__()
                self.is_main_process = main
                self.process_index = 0 if main else 1
                self.channel = channel

            def broadcast(self, value: object) -> object:
                if self.is_main_process:
                    self.channel.value = value
                return self.channel.value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = _args(root, FakeBackend()).options
            checkpoint = root / "resume"
            checkpoint.mkdir()
            (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
                "validation_status": "succeeded",
                "progress": {"phase": 1, "validation_index": 1},
            }), encoding="utf-8")
            channel = Channel()
            main = ProductionBackend.__new__(ProductionBackend)
            peer = ProductionBackend.__new__(ProductionBackend)
            main.coordinator = RankCoordinator(main=True, channel=channel)
            peer.coordinator = RankCoordinator(main=False, channel=channel)
            main._committed_validation_evidence = mock.Mock(side_effect=StageRequirementError("W&B marker missing"))  # type: ignore[method-assign]
            peer._committed_validation_evidence = mock.Mock(side_effect=AssertionError("peer touched W&B"))  # type: ignore[method-assign]

            for backend in (main, peer):
                with self.assertRaisesRegex(StageRequirementError, "W&B marker missing"):
                    backend._resume_validation_evidence(
                        options, PhaseSpec.for_phase(1), tuple(range(1, 17)), checkpoint,
                    )
            peer._committed_validation_evidence.assert_not_called()

    def test_accelerator_scope_clears_prepared_and_new_checkpoint_state_on_success_and_error(self) -> None:
        class Accelerator:
            def __init__(self) -> None:
                self._models = ["old-model"]
                self._optimizers = ["old-optimizer"]
                self._schedulers = ["old-scheduler"]
                self._dataloaders = ["old-loader"]
                self._custom_objects = [object()]
                self.free_calls = 0

            def free_memory(self) -> None:
                self.free_calls += 1
                self._models.clear()
                self._optimizers.clear()
                self._schedulers.clear()
                self._dataloaders.clear()

        for fails in (False, True):
            with self.subTest(fails=fails):
                accelerator = Accelerator()
                try:
                    with _accelerator_scope(accelerator):
                        accelerator._models.append("new-model")
                        accelerator._optimizers.append("new-optimizer")
                        accelerator._schedulers.append("new-scheduler")
                        accelerator._dataloaders.append("new-loader")
                        accelerator._custom_objects.append(object())
                        if fails:
                            raise RuntimeError("training failed")
                except RuntimeError:
                    if not fails:
                        raise
                self.assertEqual(accelerator.free_calls, 2)
                self.assertEqual(accelerator._models, [])
                self.assertEqual(accelerator._optimizers, [])
                self.assertEqual(accelerator._schedulers, [])
                self.assertEqual(accelerator._dataloaders, [])
                self.assertEqual(accelerator._custom_objects, [])

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
            self.assertEqual(len(backend.coordinator.gathers), 1)
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
            "access_token": {"raw": "structured-secret"},
            "token_limit": 6_000,
            "TOKENIZER": {"qualified": True},
        }
        redacted = redact_secrets(value)
        rendered = json.dumps(redacted)
        self.assertNotIn("hugging-face-secret", rendered)
        self.assertNotIn("wandb-secret", rendered)
        self.assertNotIn("oauth-secret", rendered)
        self.assertNotIn("structured-secret", rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertEqual(redacted["HF_TOKEN"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["WANDB_API_KEY"], "[REDACTED]")
        self.assertEqual(redacted["items"][0]["client_secret"], "[REDACTED]")
        self.assertEqual(redacted["access_token"], "[REDACTED]")
        self.assertEqual(redacted["token_limit"], 6_000)
        self.assertEqual(redacted["TOKENIZER"], {"qualified": True})

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
                code = main(["status", "--run-root", str(root), "--token-limit", "6000"])
            self.assertEqual(code, ExitCode.SUCCESS)
            self.assertFalse(root.exists())
            self.assertNotIn("should-not-leak", stdout.getvalue())
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["workflow"]["token_limit"], 6_000)
            self.assertIsInstance(status["stages"]["tokenizer_qualified"], dict)

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

    def test_test_export_capability_is_not_cli_or_production_accessible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = _args(Path(directory), FakeBackend()).options
            self.assertFalse(options.allow_test_export)
            test_options = replace(options, allow_test_export=True)
            with mock.patch("accelerate.Accelerator") as accelerator:
                with self.assertRaisesRegex(StageRequirementError, "test export"):
                    ProductionBackend(test_options)
                accelerator.assert_not_called()

        parser_output = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(parser_output):
            main(["phase2", "--allow-test-export"])

    def test_production_backend_uses_24_hour_process_group_timeout(self) -> None:
        from accelerate.utils import InitProcessGroupKwargs

        with tempfile.TemporaryDirectory() as directory, mock.patch("accelerate.Accelerator") as accelerator:
            accelerator.return_value.num_processes = 8
            ProductionBackend(_args(Path(directory), FakeBackend()).options)

        handlers = accelerator.call_args.kwargs["kwargs_handlers"]
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0], InitProcessGroupKwargs)
        self.assertEqual(handlers[0].timeout, timedelta(hours=24))


if __name__ == "__main__":
    unittest.main()

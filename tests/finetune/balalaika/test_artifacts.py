"""Behavioral tests for Balalaika runtime configuration and stage artifacts."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from cosyvoice.finetune.balalaika.artifacts import StageRequirementError, StageStore, atomic_write_json, sha256_file
from cosyvoice.finetune.balalaika.config import PhaseSpec, RunPaths, collect_environment


class ArtifactTests(unittest.TestCase):
    def test_phase_boundary_is_literal(self):
        self.assertEqual(PhaseSpec.for_phase(1).predicate, "asr_agreement_mean < 0.95")
        self.assertEqual(PhaseSpec.for_phase(1).epochs, 2)
        self.assertEqual(PhaseSpec.for_phase(2).predicate, "asr_agreement_mean >= 0.95")
        self.assertEqual(PhaseSpec.for_phase(2).epochs, 3)

    def test_unknown_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "phase must be 1 or 2, got 3"):
            PhaseSpec.for_phase(3)

    def test_run_paths_use_documented_defaults_and_environment_overrides(self):
        defaults = RunPaths.from_env({})
        self.assertEqual(defaults.dataset_root, Path("/workspace/balalaika_proprietary_v2"))
        self.assertEqual(defaults.repository_root, Path("/workspace/CosyVoice"))
        self.assertEqual(defaults.run_root, Path("/workspace/cosyvoice3-balalaika-lora"))
        self.assertEqual(defaults.base_model_dir, Path("/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"))
        self.assertEqual(defaults.visible_devices, (0, 1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(defaults.seed, 1986)

        paths = RunPaths.from_env(
            {
                "BALALAIKA_DATASET_ROOT": "/data/balalaika",
                "BALALAIKA_REPOSITORY_ROOT": "/repo",
                "BALALAIKA_RUN_ROOT": "/runs/current",
                "BALALAIKA_BASE_MODEL_DIR": "/models/base",
                "BALALAIKA_VISIBLE_DEVICES": "2,3",
                "BALALAIKA_SEED": "7",
            }
        )
        self.assertEqual(paths.dataset_root, Path("/data/balalaika"))
        self.assertEqual(paths.repository_root, Path("/repo"))
        self.assertEqual(paths.run_root, Path("/runs/current"))
        self.assertEqual(paths.base_model_dir, Path("/models/base"))
        self.assertEqual(paths.visible_devices, (2, 3))
        self.assertEqual(paths.seed, 7)

    def test_run_paths_rejects_rl_base_model_override(self):
        with self.assertRaisesRegex(ValueError, "base/non-RL"):
            RunPaths.from_env({"BALALAIKA_BASE_MODEL_DIR": "/models/Fun-CosyVoice3-0.5B-2512_RL"})

    def test_run_paths_rejects_checkpoint_nested_under_rl_model(self):
        with self.assertRaisesRegex(ValueError, "base/non-RL"):
            RunPaths.from_env({"BALALAIKA_BASE_MODEL_DIR": "/models/Fun-CosyVoice3_RL/checkpoint"})

    def test_run_paths_allows_checkpoint_nested_under_non_rl_model(self):
        paths = RunPaths.from_env({"BALALAIKA_BASE_MODEL_DIR": "/models/Fun-CosyVoice3/checkpoint"})
        self.assertEqual(paths.base_model_dir, Path("/models/Fun-CosyVoice3/checkpoint"))

    def test_atomic_write_replaces_complete_json_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            atomic_write_json(path, {"rows": 4, "values": ["a", "b"]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"rows": 4, "values": ["a", "b"]})
            self.assertFalse(list(path.parent.glob(".manifest.json.*")))

    def test_stage_payload_is_checksum_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StageStore(Path(tmp))
            published = store.publish("pilot", {"rows": 4, "seed": 1986})
            loaded = store.require("pilot")
            self.assertEqual(loaded.payload["rows"], 4)
            self.assertEqual(loaded.manifest_sha256, sha256_file(published.path))

    def test_stage_rejects_tampered_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StageStore(Path(tmp))
            record = store.publish("pilot", {"rows": 4})
            manifest = json.loads(record.path.read_text(encoding="utf-8"))
            manifest["payload"]["rows"] = 5
            record.path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(StageRequirementError, "payload checksum"):
                store.require("pilot")

    def test_stage_rejects_changed_dependency_lock_or_input_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StageStore(
                Path(tmp),
                dependency_lock={"torch": "2.8.0+cu128"},
                input_provenance={"shards": "abc123"},
            )
            record = store.publish("pilot", {"rows": 4})
            manifest = json.loads(record.path.read_text(encoding="utf-8"))
            manifest["dependency_lock"]["torch"] = "2.9.0"
            record.path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(StageRequirementError, "dependency lock"):
                store.require("pilot")

            store.publish("pilot", {"rows": 4})
            manifest = json.loads(record.path.read_text(encoding="utf-8"))
            manifest["input_provenance"]["shards"] = "changed"
            record.path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(StageRequirementError, "input provenance"):
                store.require("pilot")

    def test_stage_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(StageRequirementError, "missing required stage"):
                StageStore(Path(tmp)).require("pilot")

    def test_collect_environment_records_qualified_blackwell_stack(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5090",
            get_device_capability=lambda index: (12, 0),
            is_bf16_supported=lambda: True,
        )
        torch = types.SimpleNamespace(__version__="2.8.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)
        onnxruntime = types.SimpleNamespace(__version__="1.26.0", get_available_providers=lambda: ["CUDAExecutionProvider"])
        versions = {
            "accelerate": "1.12.0",
            "peft": "0.20.0",
            "transformers": "4.57.3",
            "pyarrow": "25.0.0",
            "wandb": "0.28.1",
            "onnx-asr": "0.12.0",
            "openai-whisper": "20250625",
        }

        with (
            patch("cosyvoice.finetune.balalaika.config.import_module", side_effect=lambda name: {"torch": torch, "onnxruntime": onnxruntime}[name]),
            patch("cosyvoice.finetune.balalaika.config._distribution_version", side_effect=lambda name: versions[name]),
        ):
            environment = collect_environment()

        self.assertEqual(environment["python"], ".".join(map(str, __import__("sys").version_info[:3])))
        self.assertEqual(environment["torch"], "2.8.0+cu128")
        self.assertEqual(environment["cuda_runtime"], "12.8")
        self.assertEqual(environment["gpus"][0], {"name": "NVIDIA GeForce RTX 5090", "capability": [12, 0]})
        self.assertEqual(environment["onnxruntime"], "1.26.0")
        self.assertEqual(environment["onnx_asr"], "0.12.0")
        self.assertEqual(environment["openai_whisper"], "20250625")

    def test_blackwell_requirements_override_whisper_for_python312_and_triton3(self):
        requirements = (
            Path(__file__).parents[3]
            / "examples/balalaika/cosyvoice3_lora/requirements-cu128.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("openai-whisper==20250625", requirements.splitlines())
        self.assertNotIn("openai-whisper==20231117", requirements.splitlines())

    def test_collect_environment_rejects_missing_openai_whisper(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5090",
            get_device_capability=lambda index: (12, 0),
            is_bf16_supported=lambda: True,
        )
        torch = types.SimpleNamespace(__version__="2.8.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)
        onnxruntime = types.SimpleNamespace(__version__="1.26.0", get_available_providers=lambda: ["CUDAExecutionProvider"])

        def distribution_version(name: str) -> str:
            if name == "openai-whisper":
                raise RuntimeError("missing required dependency: openai-whisper")
            return "1.0.0"

        with (
            patch("cosyvoice.finetune.balalaika.config.import_module", side_effect=lambda name: {"torch": torch, "onnxruntime": onnxruntime}[name]),
            patch("cosyvoice.finetune.balalaika.config._distribution_version", side_effect=distribution_version),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing required dependency: openai-whisper"):
                collect_environment()

    def test_collect_environment_rejects_unsupported_onnx_runtime(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5090",
            get_device_capability=lambda index: (12, 0),
            is_bf16_supported=lambda: True,
        )
        torch = types.SimpleNamespace(__version__="2.8.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)
        onnxruntime = types.SimpleNamespace(__version__="1.27.0", get_available_providers=lambda: ["CUDAExecutionProvider"])

        with (
            patch("cosyvoice.finetune.balalaika.config.import_module", side_effect=lambda name: {"torch": torch, "onnxruntime": onnxruntime}[name]),
            patch("cosyvoice.finetune.balalaika.config._distribution_version", return_value="1.0.0"),
        ):
            with self.assertRaisesRegex(RuntimeError, "onnxruntime-gpu must be below 1.27"):
                collect_environment()

    def test_collect_environment_rejects_more_than_eight_gpus(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 9,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5090",
            get_device_capability=lambda index: (12, 0),
            is_bf16_supported=lambda: True,
        )
        torch = types.SimpleNamespace(__version__="2.8.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda)
        onnxruntime = types.SimpleNamespace(__version__="1.26.0", get_available_providers=lambda: ["CUDAExecutionProvider"])

        with (
            patch("cosyvoice.finetune.balalaika.config.import_module", side_effect=lambda name: {"torch": torch, "onnxruntime": onnxruntime}[name]),
            patch("cosyvoice.finetune.balalaika.config._distribution_version", return_value="1.0.0"),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly eight GPUs"):
                collect_environment()


if __name__ == "__main__":
    unittest.main()

"""Tests for final standalone CosyVoice3 LLM export and verification."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import wave

import torch

from cosyvoice.finetune.balalaika.artifacts import sha256_file
from cosyvoice.finetune.balalaika import evaluation as evaluation_api
from cosyvoice.finetune.balalaika import model as api
from tests.finetune.balalaika import test_evaluation as evaluation_fixtures
from tests.finetune.balalaika.test_model import _logits, tiny_cosyvoice3_llm


class _Recognizer:
    def transcribe(self, paths):
        return ["проверка" for _ in paths]

    def provenance(self):
        return {"model": "gigaam-v3-rnnt", "provider": "fake"}


class _ProductionRecognizer:
    local_rank = 0
    max_batch_size = 8

    def transcribe(self, paths):
        return ["проверка" for _ in paths]

    def provenance(self):
        return {
            "model": "gigaam-v3-rnnt",
            "provider": "CUDAExecutionProvider",
            "device_id": 0,
            "max_batch_size": 8,
        }


_ProductionRecognizer.__module__ = "cosyvoice.finetune.balalaika.evaluation"
_ProductionRecognizer.__qualname__ = "GigaAmRecognizer"


class _FailingRecognizer(_Recognizer):
    def transcribe(self, paths):
        raise RuntimeError("ASR failed")


class _Pipeline:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.sample_rate = 24_000
        self.model = types.SimpleNamespace(
            llm=tiny_cosyvoice3_llm().eval(),
            flow=torch.nn.Linear(1, 1, bias=False),
            hift=torch.nn.Linear(1, 1, bias=False),
        )
        for component in (self.model.flow, self.model.hift):
            for parameter in component.parameters():
                parameter.requires_grad_(False)
        self.model.llm.load_state_dict(
            torch.load(self.model_dir / "llm.pt", map_location="cpu", weights_only=True),
            strict=True,
        )

    def add_zero_shot_spk(self, prompt_text, prompt_wav, speaker_id):
        return True

    def inference_zero_shot(self, *args, **kwargs):
        return [{"tts_speech": torch.tensor([[0.1, -0.1, 0.0]], dtype=torch.float32)}]


class _FailingPipeline(_Pipeline):
    def inference_zero_shot(self, *args, **kwargs):
        raise RuntimeError("synthesis failed")


class FinalMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base_dir = self.root / "Fun-CosyVoice3-0.5B-2512"
        self.base_dir.mkdir()
        (self.base_dir / "cosyvoice3.yaml").write_text("fixture", encoding="utf-8")
        (self.base_dir / "flow.pt").write_bytes(b"frozen-flow")
        (self.base_dir / "hift.pt").write_bytes(b"frozen-hift")
        self.base = tiny_cosyvoice3_llm().eval()
        self.base_state = copy.deepcopy(self.base.state_dict())
        torch.save(self.base_state, self.base_dir / "llm.pt")
        self.base._balalaika_base_model_dir = self.base_dir
        self.base._balalaika_base_checkpoint_sha256 = sha256_file(self.base_dir / "llm.pt")
        self.fresh_base = copy.deepcopy(self.base)
        self.phase2 = self.root / "phase-2-validation-40"
        self.adapter_dir = self.phase2 / "adapter"
        adapted = api.inject_lora(self.base, api.LoraSettings()).eval()
        for name, parameter in adapted.named_parameters():
            if "lora_" in name:
                parameter.data.fill_(0.05)
        self.adapter_active_logits = _logits(adapted).detach().clone()
        self.adapter = api.save_adapter(adapted, self.adapter_dir)
        self._write_phase2_manifest()
        self.validation_summary = self.root / "validation-40-summary.json"
        self.validation_summary.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "validation_index": 40,
                    "evaluation_identity": {
                        "base_checkpoint_sha256": self.adapter.base_checkpoint_sha256,
                        "adapter_sha256": self.adapter.weights_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        validation = json.loads(self.validation_summary.read_text(encoding="utf-8"))
        validation["evaluation_identity_sha256"] = __import__("hashlib").sha256(
            json.dumps(validation["evaluation_identity"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.validation_summary.write_text(json.dumps(validation), encoding="utf-8")
        (self.root / "validation-success.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "evaluation_identity_sha256": validation["evaluation_identity_sha256"],
                    "artifacts": {"summary_json": sha256_file(self.validation_summary)},
                }
            ),
            encoding="utf-8",
        )
        self.voices = []
        for index in range(20):
            path = self.root / f"voice_{index:02d}.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24_000); stream.writeframes(b"\0\0" * 4)
            self.voices.append({"voice_id": f"voice_{index:02d}", "prompt_text": "проверка", "prompt_wav": path, "prompt_sha256": sha256_file(path)})
        self.export_request = api.ExportRequest(
            base_model_dir=self.base_dir,
            phase2_checkpoint=self.phase2,
            validation_summary=self.validation_summary,
            output_dir=self.root / "final",
            test_mode=True,
            verification_voices=tuple(self.voices),
            recognizer=_Recognizer(),
            pipeline_factory=_Pipeline,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_phase2_manifest(self) -> None:
        adapter_files = {
            str(path.relative_to(self.phase2)): sha256_file(path)
            for path in sorted(self.adapter_dir.rglob("*"))
            if path.is_file()
        }
        (self.phase2 / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "validation_status": "succeeded",
                    "identity": {
                        "phase": 2,
                        "base_checkpoint_sha256": self.adapter.base_checkpoint_sha256,
                        "lora": {"r": 64, "alpha": 128, "dropout": 0.05, "bias": "none"},
                    },
                    "progress": {"phase": 2, "validation_index": 40},
                    "state_files": adapter_files,
                }
            ),
            encoding="utf-8",
        )

    def _export(self):
        with mock.patch.object(api, "load_base_llm", side_effect=lambda _: copy.deepcopy(self.fresh_base)):
            return api.export_final_llm(self.export_request)

    def _export_changed(self, request):
        with mock.patch.object(api, "load_base_llm", side_effect=lambda _: copy.deepcopy(self.fresh_base)):
            return api.export_final_llm(request)

    def _production_export_request(self, name="production-final"):
        root = self.root / name
        rows = evaluation_fixtures._rows()
        prompts = evaluation_fixtures._prompts(root / "prompts")
        checkpoint = json.loads((self.phase2 / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        provenance = evaluation_api.EvaluationProvenance(
            checkpoint_sha256=api.checkpoint_identity_sha256(checkpoint),
            model_state_sha256=api.model_state_identity_sha256(checkpoint["state_files"]),
            adapter_sha256=self.adapter.weights_sha256,
            base_checkpoint_sha256=self.adapter.base_checkpoint_sha256,
            benchmark_snapshot_sha256="5" * 64,
            benchmark_revision="private-revision-1",
            asr_config=evaluation_fixtures._FakeRecognizer().provenance(),
            synthesis_config=evaluation_fixtures._FakeSynthesizer().provenance(),
            code_version="commit-a",
            config_version="balalaika-v1",
        )
        remote = evaluation_fixtures._seal_remote_records(
            evaluation_api,
            evaluation_fixtures._remote_records(root),
            rows,
            prompts,
            provenance,
            validation_index=40,
        )
        accelerator = evaluation_fixtures._FakeAccelerator(remote)
        accelerator.run.resumed = True
        evaluation_api.atomic_write_json(root / "wandb-run.json", {"format_version": 1, "run_id": "run-123"})
        logger = evaluation_fixtures._logger(evaluation_api, accelerator, root)
        request = evaluation_api.EvaluationRequest(
            rows=rows,
            prompts=prompts,
            accelerator=accelerator,
            synthesizer=evaluation_fixtures._FakeSynthesizer(),
            recognizer=evaluation_fixtures._FakeRecognizer(),
            provenance=provenance,
            validation_index=40,
            output_jsonl=root / "validation-40/results.jsonl",
            summary_json=root / "validation-40/summary.json",
            panel_dir=root / "validation-40/listening-panel",
            temporary_audio_dir=root / "validation-40/audio",
            assignment_manifest=root / "voice-assignment.json",
            memorization_path=root / "memorization",
            wandb_logger=logger,
        )
        items = evaluation_api.build_voice_assignment(rows, prompts)
        assignment = evaluation_api._semantic_assignment(items)
        evaluation_api.atomic_write_json(
            request.assignment_manifest,
            {
                "format_version": 1,
                "assignment_sha256": evaluation_api._canonical_sha256(assignment),
                "rows": 2_000,
                "voices": 20,
                "assignment": assignment,
            },
        )
        with mock.patch.object(evaluation_api, "require_memorization_gate", return_value=object()):
            evaluation_api.evaluate_checkpoint(request)
        return api.ExportRequest(
            base_model_dir=self.base_dir,
            phase2_checkpoint=self.phase2,
            validation_summary=request.summary_json,
            output_dir=self.root / f"{name}-export",
            validation_request=request,
            wandb_run_manifest=root / "wandb-run.json",
            wandb_logger=logger,
            expected_training_identity=checkpoint["identity"],
        )

    def test_merged_checkpoint_has_original_keys_only(self):
        manifest = self._export()

        state = torch.load(manifest.llm_path, map_location="cpu", weights_only=True)
        strict = tiny_cosyvoice3_llm().eval()
        strict.load_state_dict(state, strict=True)

        self.assertEqual(set(state), set(self.base_state))
        self.assertFalse(any("lora_" in key for key in state))
        torch.testing.assert_close(_logits(strict), self.adapter_active_logits, rtol=2e-2, atol=2e-2)
        self.assertTrue((self.adapter_dir / "adapter_model.safetensors").is_file())
        final = json.loads(manifest.path.read_text(encoding="utf-8"))
        self.assertFalse(final["production_ready"])
        self.assertEqual(final["mode"], "test")
        self.assertTrue((manifest.path.parent / "final-success.json").is_file())
        self.assertTrue((manifest.path.parent / "strict-verification/strict-verification.json").is_file())

    def test_export_binds_structured_logit_evidence(self):
        manifest = self._export()
        payload = json.loads(manifest.path.read_text(encoding="utf-8"))
        evidence = payload["logit_verification"]

        expected_probe = {"token_ids": [[0, 0, 0]], "input_shape": [1, 3]}
        expected_hash = hashlib.sha256(
            json.dumps(expected_probe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            set(evidence),
            {
                "format_version", "probe", "probe_sha256", "atol", "rtol",
                "max_abs_error", "max_relative_error", "finite", "pass",
            },
        )
        self.assertEqual(evidence["probe"], expected_probe)
        self.assertEqual(evidence["probe_sha256"], expected_hash)
        self.assertEqual((evidence["atol"], evidence["rtol"]), (0.02, 0.02))
        self.assertTrue(evidence["finite"])
        self.assertTrue(evidence["pass"])
        self.assertGreaterEqual(evidence["max_abs_error"], 0.0)
        self.assertGreaterEqual(evidence["max_relative_error"], 0.0)
        seal = json.loads((manifest.path.parent / "final-success.json").read_text(encoding="utf-8"))
        self.assertEqual(seal["logit_verification_sha256"], api._canonical_mapping_sha256(evidence))

    def test_logit_drift_aborts_without_publishing_output(self):
        original = api._fixed_probe_logits
        calls = 0

        def drifted(model):
            nonlocal calls
            calls += 1
            value = original(model)
            return value if calls == 1 else value + 1.0

        request = replace(self.export_request, output_dir=self.root / "drift-final")
        with mock.patch.object(api, "_fixed_probe_logits", side_effect=drifted):
            with self.assertRaisesRegex(RuntimeError, "logits"):
                self._export_changed(request)
        self.assertFalse(request.output_dir.exists())

    def test_adapter_directory_must_be_exact_regular_inventory(self):
        cases = ("extra", "symlink")
        for case in cases:
            with self.subTest(case=case):
                extra = self.adapter_dir / ("extra.bin" if case == "extra" else "extra-link")
                if case == "extra":
                    extra.write_bytes(b"extra")
                else:
                    extra.symlink_to(self.adapter_dir / "adapter_model.safetensors")
                self._write_phase2_manifest()
                request = replace(self.export_request, output_dir=self.root / f"adapter-{case}")
                with self.assertRaisesRegex(ValueError, "adapter.*inventory|symlink"):
                    self._export_changed(request)
                extra.unlink()
                self._write_phase2_manifest()

    def test_adapter_weights_and_fixed_settings_are_reauthenticated(self):
        original_weights = (self.adapter_dir / "adapter_model.safetensors").read_bytes()
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(original_weights + b"corrupt")
        self._write_phase2_manifest()
        with self.assertRaisesRegex(ValueError, "weights checksum"):
            self._export_changed(replace(self.export_request, output_dir=self.root / "weights-corrupt"))
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(original_weights)

        manifest_path = self.adapter_dir / "adapter_manifest.json"
        adapter = json.loads(manifest_path.read_text(encoding="utf-8"))
        adapter["settings"]["r"] = 32
        evaluation_api.atomic_write_json(manifest_path, adapter)
        self._write_phase2_manifest()
        with self.assertRaisesRegex(ValueError, "settings"):
            self._export_changed(replace(self.export_request, output_dir=self.root / "settings-corrupt"))

    def test_committed_final_is_idempotent_and_exactly_verified(self):
        first = self._export()
        before = {path.relative_to(first.path.parent): path.read_bytes() for path in first.path.parent.rglob("*") if path.is_file()}

        second = self._export()

        after = {path.relative_to(second.path.parent): path.read_bytes() for path in second.path.parent.rglob("*") if path.is_file()}
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(api.require_committed_final(first.path.parent, expected_mode="test"), first)
        with self.assertRaisesRegex(ValueError, "mode"):
            api.require_committed_final(first.path.parent, expected_mode="production")

    def test_committed_final_rejects_manifest_seal_wav_adapter_and_extra_corruption(self):
        def corrupt_seal(root):
            path = root / "final-success.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["manifest_sha256"] = "f" * 64
            evaluation_api.atomic_write_json(path, value)

        mutations = {
            "manifest": lambda root: (root / "final_model_manifest.json").write_bytes((root / "final_model_manifest.json").read_bytes() + b" "),
            "seal": corrupt_seal,
            "wav": lambda root: (root / "strict-verification/audio/smoke-01.wav").write_bytes((root / "strict-verification/audio/smoke-01.wav").read_bytes() + b"corrupt"),
            "adapter": lambda root: (root / "adapter/adapter_model.safetensors").write_bytes((root / "adapter/adapter_model.safetensors").read_bytes() + b"corrupt"),
            "extra": lambda root: (root / "extra.bin").write_bytes(b"extra"),
            "extra-directory": lambda root: (root / "extra-dir").mkdir(),
            "symlink": lambda root: (root / "extra-link").symlink_to(root / "llm.pt"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                request = replace(self.export_request, output_dir=self.root / f"corrupt-{name}")
                manifest = self._export_changed(request)
                mutate(manifest.path.parent)
                with self.assertRaises(ValueError):
                    api.require_committed_final(manifest.path.parent, expected_mode="test")
                with self.assertRaises((ValueError, FileExistsError)):
                    self._export_changed(request)

    def test_self_resealed_strict_report_semantic_corruption_is_rejected(self):
        manifest = self._export()
        root = manifest.path.parent
        report_path = root / "strict-verification/strict-verification.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["audio"][0]["asr"] = ""
        evaluation_api.atomic_write_json(report_path, report)
        payload = json.loads(manifest.path.read_text(encoding="utf-8"))
        payload["strict_verification"]["report_sha256"] = sha256_file(report_path)
        evaluation_api.atomic_write_json(manifest.path, payload)
        seal_path = root / "final-success.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["manifest_sha256"] = sha256_file(manifest.path)
        seal["artifacts"] = api._regular_file_inventory(root, excluded={"final-success.json"})
        evaluation_api.atomic_write_json(seal_path, seal)

        with self.assertRaisesRegex(ValueError, "ASR"):
            api.require_committed_final(root, expected_mode="test")

    def test_strict_report_uses_safe_relative_audio_paths(self):
        manifest = self._export()
        report = json.loads((manifest.path.parent / "strict-verification/strict-verification.json").read_text(encoding="utf-8"))
        self.assertNotIn("final_model_manifest_sha256", report)
        self.assertEqual(
            [row["path"] for row in report["audio"]],
            [f"strict-verification/audio/smoke-{index:02d}.wav" for index in range(1, 5)],
        )
        self.assertTrue(all((manifest.path.parent / row["path"]).is_file() for row in report["audio"]))

    def test_synthesis_and_asr_failure_cleanup_allows_retry(self):
        failures = {
            "synthesis": {"pipeline_factory": _FailingPipeline},
            "asr": {"recognizer": _FailingRecognizer()},
        }
        for name, changes in failures.items():
            with self.subTest(name=name):
                output = self.root / f"retry-{name}"
                failing = replace(self.export_request, output_dir=output, **changes)
                with self.assertRaises(RuntimeError):
                    self._export_changed(failing)
                self.assertFalse(output.exists())
                self.assertEqual(list(output.parent.glob(f".{output.name}.*")), [])
                recovered = self._export_changed(replace(self.export_request, output_dir=output))
                self.assertEqual(api.require_committed_final(output, expected_mode="test"), recovered)

    def test_post_rename_failure_recovers_idempotently(self):
        request = replace(self.export_request, output_dir=self.root / "post-rename")
        publish = api._publish_directory

        def publish_then_fail(stage, destination):
            publish(stage, destination)
            if destination == request.output_dir:
                raise OSError("parent fsync failed")

        with mock.patch.object(api, "_publish_directory", side_effect=publish_then_fail):
            with self.assertRaisesRegex(OSError, "fsync"):
                self._export_changed(request)
        self.assertTrue(request.output_dir.is_dir())
        recovered = self._export_changed(request)
        self.assertEqual(recovered, api.require_committed_final(request.output_dir, expected_mode="test"))

    def test_existing_final_with_different_prompt_lineage_is_refused(self):
        manifest = self._export()
        changed_voices = [dict(voice) for voice in self.voices]
        changed_voices[0]["prompt_text"] = "другая проверка"
        changed = replace(self.export_request, verification_voices=changed_voices)

        with self.assertRaisesRegex(ValueError, "lineage"):
            self._export_changed(changed)
        self.assertEqual(api.require_committed_final(manifest.path.parent, expected_mode="test"), manifest)

    def test_code_identity_is_structured_and_bound(self):
        identity = {"head": "abc123", "dirty": True, "diff_sha256": "d" * 64}
        request = replace(self.export_request, output_dir=self.root / "code-identity")
        with mock.patch.object(api, "_code_identity", return_value=identity):
            manifest = self._export_changed(request)
            payload = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["code_identity"], identity)
            self.assertEqual(self._export_changed(request), manifest)

        changed_identity = {**identity, "diff_sha256": "e" * 64}
        with mock.patch.object(api, "_code_identity", return_value=changed_identity):
            with self.assertRaisesRegex(ValueError, "lineage"):
                self._export_changed(request)

    def test_existing_final_reauthenticates_current_base_inventory(self):
        manifest = self._export()
        (self.base_dir / "flow.pt").write_bytes(b"changed-flow")

        with self.assertRaises(ValueError):
            self._export()
        self.assertEqual(api.require_committed_final(manifest.path.parent, expected_mode="test"), manifest)

    def test_base_asset_inventory_rejects_the_10001st_file(self):
        original = Path.rglob
        repeated = [self.base_dir / "llm.pt"] * 10_001

        def bounded(path, pattern):
            if path == self.base_dir:
                return iter(repeated)
            return original(path, pattern)

        with mock.patch.object(Path, "rglob", bounded):
            with self.assertRaisesRegex(ValueError, "safe bound"):
                api.build_base_asset_manifest(self.base_dir)

    def test_export_rejects_fabricated_validation_seal(self):
        seal = self.root / "validation-success.json"
        payload = json.loads(seal.read_text(encoding="utf-8"))
        payload["evaluation_identity_sha256"] = "0" * 64
        seal.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "validation"):
            self._export()

    def test_production_export_consumes_authenticated_task10_prompt_schema(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "unit-test-key"}):
            request = self._production_export_request()
            with (
                mock.patch.object(api, "load_base_llm", side_effect=lambda _: copy.deepcopy(self.fresh_base)),
                mock.patch.object(evaluation_api, "GigaAmRecognizer", _ProductionRecognizer),
                mock.patch.object(api, "_normal_cosyvoice3_pipeline", side_effect=lambda view, factory: _Pipeline(view)),
            ):
                manifest = api.export_final_llm(request)

        payload = json.loads(manifest.path.read_text(encoding="utf-8"))
        strict = json.loads((manifest.path.parent / "strict-verification/strict-verification.json").read_text(encoding="utf-8"))
        normalized, expected_inventory_sha256 = evaluation_api.authenticated_prompt_inventory(request.validation_request.prompts)
        self.assertTrue(manifest.production_ready)
        self.assertTrue(payload["production_ready"])
        self.assertEqual(payload["prompt_inventory_sha256"], expected_inventory_sha256)
        self.assertEqual(strict["prompt_source"], "task10_validation_request")
        self.assertEqual(strict["selected_voices"], [value["voice_id"] for value in normalized[:4]])

    def test_production_audit_rejects_mode_confusion_and_arbitrary_recognizer(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "unit-test-key"}):
            request = self._production_export_request("production-audit")
            with (
                mock.patch.object(api, "load_base_llm", side_effect=lambda _: copy.deepcopy(self.fresh_base)),
                mock.patch.object(evaluation_api, "GigaAmRecognizer", _ProductionRecognizer),
                mock.patch.object(api, "_normal_cosyvoice3_pipeline", side_effect=lambda view, factory: _Pipeline(view)),
            ):
                manifest = api.export_final_llm(request)

        self.assertEqual(api.require_committed_final(manifest.path.parent, expected_mode="production"), manifest)
        with self.assertRaisesRegex(ValueError, "mode"):
            api.require_committed_final(manifest.path.parent, expected_mode="test")
        voices, _ = evaluation_api.authenticated_prompt_inventory(request.validation_request.prompts)
        verify = api.VerifyRequest(
            base_model_dir=self.base_dir,
            final_manifest=manifest,
            recognizer=_Recognizer(),
            voices=voices,
            output_dir=self.root / "arbitrary-recognizer",
        )
        with self.assertRaisesRegex(TypeError, "exact Task 10"):
            api.strict_verify_final_model(verify)

        with mock.patch.object(evaluation_api, "GigaAmRecognizer", _ProductionRecognizer):
            wrong_rank = replace(verify, recognizer=_ProductionRecognizer(), expected_local_rank=1)
            with self.assertRaisesRegex(ValueError, "local rank"):
                api.strict_verify_final_model(wrong_rank)

    def test_production_export_rejects_different_validation_summary_path(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "unit-test-key"}):
            request = self._production_export_request("summary-path")
            copied_summary = self.root / "copied-task10/summary.json"
            copied_summary.parent.mkdir(parents=True)
            copied_summary.write_bytes(request.validation_summary.read_bytes())
            copied_seal = copied_summary.with_name("validation-success.json")
            seal = json.loads(request.validation_summary.with_name("validation-success.json").read_text(encoding="utf-8"))
            seal["artifacts"]["summary_json"] = sha256_file(copied_summary)
            evaluation_api.atomic_write_json(copied_seal, seal)
            changed = replace(request, validation_summary=copied_summary)

            with self.assertRaisesRegex(ValueError, "exactly Task 10"):
                api.export_final_llm(changed)

    def test_production_export_rejects_task10_checkpoint_or_model_state_mismatch(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "unit-test-key"}):
            request = self._production_export_request("provenance-mismatch")
            for field in ("checkpoint_sha256", "model_state_sha256"):
                with self.subTest(field=field):
                    provenance = replace(request.validation_request.provenance, **{field: "f" * 64})
                    changed = replace(
                        request,
                        validation_request=replace(request.validation_request, provenance=provenance),
                    )
                    with self.assertRaisesRegex(evaluation_api.EvaluationIntegrityError, "lineage"):
                        api.export_final_llm(changed)

    def test_strict_loader_accepts_merged_checkpoint(self):
        manifest = self._export()
        voice_paths = []
        for index in range(20):
            voice = self.root / f"voice_{index:02d}.wav"
            with wave.open(str(voice), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24_000)
                stream.writeframes(b"\0\0" * 4)
            voice_paths.append(voice)
        request = api.VerifyRequest(
            base_model_dir=self.base_dir,
            final_manifest=manifest,
            recognizer=_Recognizer(),
            voices=tuple(
                {"voice_id": f"voice_{index:02d}", "prompt_text": "проверка", "prompt_wav": path, "prompt_sha256": sha256_file(path)}
                for index, path in enumerate(voice_paths)
            ),
            output_dir=self.root / "verification",
            pipeline_factory=_Pipeline,
            test_mode=True,
        )

        report = api.strict_verify_final_model(request)

        self.assertTrue(report.strict_load)
        self.assertEqual(report.smoke_utterances, 4)
        self.assertEqual(len(report.audio_paths), 4)
        self.assertTrue(all(path.is_file() for path in report.audio_paths))


if __name__ == "__main__":
    unittest.main()

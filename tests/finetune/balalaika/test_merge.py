"""Tests for final standalone CosyVoice3 LLM export and verification."""

from __future__ import annotations

import copy
from dataclasses import replace
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
                mock.patch.object(evaluation_api, "GigaAmRecognizer", return_value=evaluation_fixtures._FakeRecognizer()),
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

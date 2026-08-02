"""Tests for final standalone CosyVoice3 LLM export and verification."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import wave

import torch

from cosyvoice.finetune.balalaika.artifacts import sha256_file
from cosyvoice.finetune.balalaika import model as api
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

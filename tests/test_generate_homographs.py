from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_homographs.py"


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
    ):
        self._chunks = chunks or []
        self.status_code = status_code
        self.headers = headers or {
            "X-Sample-Rate": "24000",
            "X-Audio-Format": "int16",
            "X-Channels": "1",
            "Content-Type": "application/octet-stream",
        }
        self.text = text

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield from self._chunks


def load_module(env: dict[str, str] | None = None):
    module_name = "generate_homographs_test_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    with mock.patch.dict(os.environ, env or {}, clear=False):
        spec.loader.exec_module(module)
    return module


class GenerateHomographsTests(unittest.TestCase):
    def test_compute_audio_metrics_counts_silence_and_dbfs(self):
        module = load_module({"SILENCE_DBFS": "-50"})
        pcm = (
            b"\x00\x00" * 2
            + (16384).to_bytes(2, "little", signed=True) * 4
            + b"\x00\x00" * 2
        )

        metrics = module.compute_audio_metrics(pcm, sample_rate=8)

        self.assertEqual(metrics["duration_sec"], 1.0)
        self.assertEqual(metrics["silence_ratio"], 0.5)
        self.assertEqual(metrics["leading_silence_sec"], 0.25)
        self.assertEqual(metrics["trailing_silence_sec"], 0.25)
        self.assertAlmostEqual(metrics["peak_dbfs"], -6.0206, places=3)
        self.assertAlmostEqual(metrics["rms_dbfs"], -9.0309, places=3)

    def test_main_writes_success_and_error_manifest_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "audio")
            manifest_path = os.path.join(tmpdir, "manifest.jsonl")
            module = load_module(
                {
                    "OUTPUT_DIR": output_dir,
                    "MANIFEST_PATH": manifest_path,
                    "MIN_DURATION_SEC": "0",
                    "TTS_URL": "http://tts.example/tts-stream",
                    "COSYVOICE_COMPAT_MODE": "cross_lingual",
                    "COSYVOICE_DEFAULT_REF_AUDIO": "/refs/qwen.wav",
                }
            )
            module.sentences = ["ok text", "bad text"]
            pcm = (1024).to_bytes(2, "little", signed=True) * 24
            responses = [
                FakeResponse([pcm[:20], pcm[20:]]),
                FakeResponse(status_code=503, text="unavailable"),
            ]

            health = FakeResponse(status_code=200)
            health.json = lambda: {"backend": "trtllm-serve"}

            with mock.patch.object(module.requests, "get", return_value=health):
                post_mock = mock.patch.object(module.requests, "post", side_effect=responses)
                with post_mock:
                    module.main()

            with open(manifest_path, encoding="utf-8") as manifest_file:
                records = [json.loads(line) for line in manifest_file]

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["status"], "ok")
            self.assertEqual(records[0]["text"], "ok text")
            self.assertEqual(records[0]["url"], "http://tts.example/tts-stream")
            self.assertEqual(records[0]["backend"], "trtllm-serve")
            self.assertEqual(records[0]["mode"], "cross_lingual")
            self.assertEqual(records[0]["prompt_audio_path"], "/refs/qwen.wav")
            self.assertIsNone(records[0]["token_count"])
            self.assertEqual(records[0]["sample_rate"], 24000)
            self.assertIsNone(records[0]["error"])
            self.assertIsNone(records[0]["asr_transcript"])
            self.assertTrue(os.path.exists(records[0]["output_path"]))
            with wave.open(records[0]["output_path"], "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 24000)

            self.assertEqual(records[1]["status"], "error")
            self.assertEqual(records[1]["text"], "bad text")
            self.assertIn("HTTP 503", records[1]["error"])
            self.assertEqual(
                records[1]["output_path"],
                os.path.join(output_dir, "homograph_02.wav"),
            )
            self.assertIsNone(records[1]["duration_sec"])


if __name__ == "__main__":
    unittest.main()

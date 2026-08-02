"""Fixture-only tests for the CUDA tokenizer and pilot approval boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cosyvoice.finetune.balalaika.artifacts import StageStore
from cosyvoice.finetune.balalaika.cache import AudioInput
from cosyvoice.finetune.balalaika.config import RunPaths
from cosyvoice.finetune.balalaika.tokenizer import (
    OnnxSpeechTokenizer,
    PilotApprovalError,
    TokenizerError,
    approve_pilot,
)


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class OomOnceSession:
    """A narrow ORT fake: first batch OOMs, retries report their item identities."""

    def __init__(self) -> None:
        self.item_history: list[list[int]] = []
        self._runs = 0

    def get_inputs(self):
        return [_Input("features"), _Input("lengths")]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, _, inputs):
        ids = [int(item) for item in inputs["features"][:, 0, 0]]
        self.item_history.append(ids)
        self._runs += 1
        if self._runs == 1:
            raise RuntimeError("CUDA out of memory")
        return [np.asarray([[item] for item in ids], dtype=np.int32), np.ones(len(ids), dtype=np.int32)]


class CpuFallbackSession(OomOnceSession):
    def get_providers(self):
        return ["CPUExecutionProvider"]


class OutOfRangeSession(OomOnceSession):
    def run(self, _, inputs):
        ids = [int(item) for item in inputs["features"][:, 0, 0]]
        self.item_history.append(ids)
        return [np.asarray([[6561] for _ in ids], dtype=np.int32), np.ones(len(ids), dtype=np.int32)]


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = RunPaths(root / "dataset", root / "repository", root / "run", root / "model", tuple(range(8)), 1986)
        self.ids = list(range(10, 18))
        self.audio = [AudioInput(f"clip-{item}", np.asarray([item], dtype=np.float32), 24_000, 1) for item in self.ids]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_inference_batch_retries_same_items(self) -> None:
        # Returning a partial/reordered retry would attach speech tokens to the wrong cache rows.
        session = OomOnceSession()
        backend = OnnxSpeechTokenizer(session=session, max_batch_size=8, feature_builder=self._features)

        result = backend.extract(self.audio)

        self.assertEqual(session.item_history, [self.ids, self.ids[:4], self.ids[4:]])
        self.assertEqual(result, [[item] for item in self.ids])

    def test_session_rejects_cpu_provider_fallback(self) -> None:
        # A seemingly successful CPU session invalidates the per-GPU qualification/cache contract.
        with self.assertRaisesRegex(TokenizerError, "CUDAExecutionProvider"):
            OnnxSpeechTokenizer(session=CpuFallbackSession(), feature_builder=self._features)

    def test_token_ids_outside_model_vocabulary_are_fatal(self) -> None:
        # Persisting 6561 would make training fail after expensive corpus extraction.
        backend = OnnxSpeechTokenizer(session=OutOfRangeSession(), feature_builder=self._features)
        with self.assertRaisesRegex(TokenizerError, r"outside \[0, 6560\]"):
            backend.extract(self.audio[:1])

    def test_approval_is_bound_to_exact_pilot(self) -> None:
        # A stale approval must not authorize tokenization for a changed listening bundle.
        manifest = StageStore(self.paths.stages_dir).publish("pilot", {"clips": ["short", "median", "long"]})
        with self.assertRaises(PilotApprovalError):
            approve_pilot(self.paths, "0" * 64)

        approval = approve_pilot(self.paths, manifest.manifest_sha256)

        self.assertTrue(approval.exists())
        replacement = StageStore(self.paths.stages_dir).publish("pilot", {"clips": ["changed"]})
        with self.assertRaisesRegex(PilotApprovalError, "does not match"):
            approve_pilot(self.paths, manifest.manifest_sha256)
        self.assertNotEqual(manifest.manifest_sha256, replacement.manifest_sha256)

    @staticmethod
    def _features(audio: AudioInput) -> np.ndarray:
        return np.full((128, 1), int(audio.samples[0]), dtype=np.float32)


if __name__ == "__main__":
    unittest.main()

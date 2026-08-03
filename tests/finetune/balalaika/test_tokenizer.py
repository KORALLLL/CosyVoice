"""Fixture-only tests for the CUDA tokenizer and pilot approval boundary."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cosyvoice.finetune.balalaika.artifacts import StageStore
from cosyvoice.finetune.balalaika.cache import AudioInput, CacheManifest, CacheShardRequest, CacheShardResult
from cosyvoice.finetune.balalaika.config import RunPaths
from cosyvoice.finetune.balalaika.tokenizer import (
    OnnxSpeechTokenizer,
    PilotApprovalError,
    TokenizerError,
    _validate_worker_result,
    _verify_requested_cache_set,
    approve_pilot,
    require_pilot_approval,
    run_cache_workers,
    run_tokenizer_qualification,
)


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class OomOnceSession:
    """A narrow ORT fake: first batch OOMs, retries report their item identities."""

    def __init__(self) -> None:
        self.item_history: list[list[int]] = []
        self.shape_history: list[tuple[int, ...]] = []
        self._runs = 0

    def get_inputs(self):
        return [_Input("features"), _Input("lengths")]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, _, inputs):
        ids = [int(item) for item in inputs["features"][:, 0, 0]]
        self.item_history.append(ids)
        self.shape_history.append(tuple(inputs["features"].shape))
        self._runs += 1
        if self._runs == 1:
            raise RuntimeError("CUDA out of memory")
        return [np.asarray([[item] for item in ids], dtype=np.int32), np.ones(len(ids), dtype=np.int32)]


class CpuFallbackSession(OomOnceSession):
    def get_providers(self):
        return ["CPUExecutionProvider"]


class CudaPrimaryWithCpuFallbackSession(OomOnceSession):
    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class CpuPrimarySession(OomOnceSession):
    def get_providers(self):
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]


class UnverifiableSession(OomOnceSession):
    get_providers = None


class ProviderErrorSession(OomOnceSession):
    def get_providers(self):
        raise RuntimeError("provider query failed")


class EmptyProviderSession(OomOnceSession):
    def get_providers(self):
        return []


class OutOfRangeSession(OomOnceSession):
    def run(self, _, inputs):
        ids = [int(item) for item in inputs["features"][:, 0, 0]]
        self.item_history.append(ids)
        return [np.asarray([[6561] for _ in ids], dtype=np.int32), np.ones(len(ids), dtype=np.int32)]


class TokenizerTests(unittest.TestCase):
    def test_qualification_restores_rank_cuda_device_on_success_and_failure(self) -> None:
        class Cuda:
            def __init__(self) -> None:
                self.current = 3
                self.set_history: list[int] = []

            def current_device(self) -> int:
                return self.current

            def set_device(self, device: int) -> None:
                self.current = device
                self.set_history.append(device)

            def reset_peak_memory_stats(self, _device: int) -> None:
                return None

            def max_memory_allocated(self, _device: int) -> int:
                return 1

        class QualifiedTokenizer:
            session = OomOnceSession()

            def __init__(self, fail: bool) -> None:
                self.fail = fail

            def extract(self, _audio):
                if self.fail:
                    raise TokenizerError("qualification failure")
                return [list(range(25))]

            def feature_lengths(self, _audio):
                return [100]

        model = self.paths.base_model_dir / "speech_tokenizer_v3.batch.onnx"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"onnx")
        for fails in (False, True):
            with self.subTest(fails=fails):
                cuda = Cuda()
                torch = types.SimpleNamespace(cuda=cuda)
                tokenizer = QualifiedTokenizer(fails)
                patches = (
                    patch("cosyvoice.finetune.balalaika.tokenizer._require_eight_devices"),
                    patch("cosyvoice.finetune.balalaika.tokenizer.import_module", return_value=torch),
                    patch.object(OnnxSpeechTokenizer, "for_paths", return_value=tokenizer),
                )
                with patches[0], patches[1], patches[2]:
                    if fails:
                        with self.assertRaisesRegex(TokenizerError, "qualification failure"):
                            run_tokenizer_qualification(self.paths)
                    else:
                        run_tokenizer_qualification(self.paths)
                self.assertEqual(cuda.current, 3)
                self.assertEqual(cuda.set_history[-1], 3)

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
        self.assertEqual(session.shape_history, [(8, 128, 1), (4, 128, 1), (4, 128, 1)])
        self.assertEqual(result, [[item] for item in self.ids])

    def test_session_accepts_cuda_primary_with_standard_cpu_fallback(self) -> None:
        session = CudaPrimaryWithCpuFallbackSession()
        session._runs = 1
        backend = OnnxSpeechTokenizer(session=session, feature_builder=self._features)

        self.assertEqual(backend.extract(self.audio[:1]), [[self.ids[0]]])

    def test_session_rejects_cpu_only_or_cpu_primary_provider(self) -> None:
        # A seemingly successful CPU session invalidates the per-GPU qualification/cache contract.
        for session in (CpuFallbackSession(), CpuPrimarySession()):
            with self.subTest(session=type(session).__name__):
                with self.assertRaisesRegex(TokenizerError, "CUDAExecutionProvider"):
                    OnnxSpeechTokenizer(session=session, feature_builder=self._features)

    def test_session_rejects_unverifiable_or_empty_provider_identity(self) -> None:
        # A session that cannot prove CUDA placement must never get an inference call.
        for session in (UnverifiableSession(), ProviderErrorSession(), EmptyProviderSession()):
            with self.subTest(session=type(session).__name__):
                with self.assertRaisesRegex(TokenizerError, "provider"):
                    OnnxSpeechTokenizer(session=session, feature_builder=self._features)
                self.assertEqual(session.item_history, [])

    def test_token_ids_outside_model_vocabulary_are_fatal(self) -> None:
        # Persisting 6561 would make training fail after expensive corpus extraction.
        backend = OnnxSpeechTokenizer(session=OutOfRangeSession(), feature_builder=self._features)
        with self.assertRaisesRegex(TokenizerError, r"outside \[0, 6560\]"):
            backend.extract(self.audio[:1])

    def test_approval_is_bound_to_exact_pilot(self) -> None:
        # A stale approval must not authorize tokenization for a changed listening bundle.
        manifest, _ = self._publish_fake_pilot()
        with self.assertRaises(PilotApprovalError):
            approve_pilot(self.paths, "0" * 64)

        approval = approve_pilot(self.paths, manifest.manifest_sha256)

        self.assertTrue(approval.exists())
        replacement, _ = self._publish_fake_pilot(index_contents="changed listening sheet")
        with self.assertRaisesRegex(PilotApprovalError, "does not match"):
            approve_pilot(self.paths, manifest.manifest_sha256)
        self.assertNotEqual(manifest.manifest_sha256, replacement.manifest_sha256)

    def test_approved_pilot_rejects_missing_edited_and_path_traversal_artifacts(self) -> None:
        # Approval is for the exact audio bundle, not merely the stage JSON that names it.
        cases = (
            ("missing", lambda paths: paths["original"].unlink()),
            ("edited", lambda paths: paths["tokens"].write_bytes(b"changed")),
        )
        for name, tamper in cases:
            with self.subTest(name=name):
                manifest, artifact_paths = self._publish_fake_pilot()
                approve_pilot(self.paths, manifest.manifest_sha256)
                tamper(artifact_paths)
                with self.assertRaisesRegex(PilotApprovalError, "artifact|pilot"):
                    require_pilot_approval(self.paths)

    def test_path_traversal_artifact_cannot_be_approved(self) -> None:
        # A manifest may not redirect a reviewer-approved hash to a file outside pilot/.
        manifest, _ = self._publish_fake_pilot(artifact_name="../outside.wav")
        with self.assertRaisesRegex(PilotApprovalError, "artifact"):
            approve_pilot(self.paths, manifest.manifest_sha256)

    def test_worker_result_rejects_fabricated_output_before_leasing_is_complete(self) -> None:
        # A worker may not claim a shard while pointing at arbitrary cache paths.
        request = CacheShardRequest(
            self.paths.dataset_root / "shard_000007.tar",
            self.paths.run_root / "split_plan",
            self.paths.run_root / "cache",
            7,
            expected_source_sha256="a" * 64,
            expected_plan_sha256="b" * 64,
        )
        fabricated = CacheShardResult(
            7,
            self.paths.run_root / "other.parquet",
            self.paths.run_root / "cache/phase2/shard_000007.parquet",
            self.paths.run_root / "cache/eval_prompts.parquet",
            self.paths.run_root / "cache/shard_manifests/shard_000007.json",
            "a" * 64,
            "b" * 64,
            {1: 1, 2: 1},
        )
        with self.assertRaisesRegex(TokenizerError, "phase1 path"):
            _validate_worker_result(request, fabricated)

    def test_worker_result_rejects_fabricated_checksum_identity(self) -> None:
        # Matching filenames are insufficient when a worker attributes another source/plan to its lease.
        request = CacheShardRequest(
            self.paths.dataset_root / "shard_000007.tar",
            self.paths.run_root / "split_plan",
            self.paths.run_root / "cache",
            7,
            expected_source_sha256="a" * 64,
            expected_plan_sha256="b" * 64,
        )
        root = request.cache_root
        fabricated = CacheShardResult(
            7,
            root / "phase1/shard_000007.parquet",
            root / "phase2/shard_000007.parquet",
            root / "eval_prompts.parquet",
            root / "shard_manifests/shard_000007.json",
            "c" * 64,
            "b" * 64,
            {1: 1, 2: 1},
        )
        with self.assertRaisesRegex(TokenizerError, "checksum identity"):
            _validate_worker_result(request, fabricated)

    def test_aggregate_cache_set_must_equal_leased_shards(self) -> None:
        # A missing worker result cannot be hidden by a valid subset cache manifest.
        cache_root = self.paths.run_root / "cache"
        verified = CacheManifest(cache_root, {0: cache_root / "shard_manifests/shard_000000.json"}, {1: 0, 2: 0}, 20)
        with patch("cosyvoice.finetune.balalaika.tokenizer.verify_cache", return_value=verified):
            with self.assertRaisesRegex(TokenizerError, "shard set"):
                _verify_requested_cache_set(cache_root, {0, 1})

    def test_cache_gate_rejects_extra_preexisting_shard_outside_inventory(self) -> None:
        # A valid cache manifest cannot smuggle shard 7 into the canonical 0-only fixture inventory.
        cache_root = self.paths.run_root / "cache"
        (cache_root / "shard_manifests").mkdir(parents=True)
        (cache_root / "shard_manifests" / "shard_000000.json").write_text("{}", encoding="utf-8")
        inventory = types.SimpleNamespace(
            source_archives=(self.paths.dataset_root / "train/shard_000000.tar",),
            source_archive_sha256={"shard_000000.tar": "a" * 64},
        )
        verified = CacheManifest(
            cache_root,
            {0: cache_root / "shard_manifests/shard_000000.json", 7: cache_root / "shard_manifests/shard_000007.json"},
            {1: 0, 2: 0},
            20,
        )
        with (
            patch("cosyvoice.finetune.balalaika.tokenizer.require_pilot_approval"),
            patch("cosyvoice.finetune.balalaika.tokenizer.inventory_sources", return_value=inventory),
            patch("cosyvoice.finetune.balalaika.tokenizer.verify_cache", return_value=verified),
        ):
            with self.assertRaisesRegex(TokenizerError, "outside canonical inventory"):
                run_cache_workers(self.paths)

    def test_cache_gate_noop_rechecks_exact_canonical_shard_set(self) -> None:
        # No missing leases does not excuse a final cache verification that lost shard 1.
        cache_root = self.paths.run_root / "cache"
        (cache_root / "shard_manifests").mkdir(parents=True)
        (cache_root / "shard_manifests" / "shard_000000.json").write_text("{}", encoding="utf-8")
        inventory = types.SimpleNamespace(
            source_archives=(
                self.paths.dataset_root / "train/shard_000000.tar",
                self.paths.dataset_root / "train/shard_000001.tar",
            ),
            source_archive_sha256={"shard_000000.tar": "a" * 64, "shard_000001.tar": "b" * 64},
        )
        complete = CacheManifest(cache_root, {0: cache_root / "shard_manifests/shard_000000.json", 1: cache_root / "shard_manifests/shard_000001.json"}, {1: 0, 2: 0}, 20)
        missing = CacheManifest(cache_root, {0: cache_root / "shard_manifests/shard_000000.json"}, {1: 0, 2: 0}, 20)
        with (
            patch("cosyvoice.finetune.balalaika.tokenizer.require_pilot_approval"),
            patch("cosyvoice.finetune.balalaika.tokenizer.inventory_sources", return_value=inventory),
            patch("cosyvoice.finetune.balalaika.tokenizer.verify_cache", side_effect=(complete, missing)),
        ):
            with self.assertRaisesRegex(TokenizerError, "shard set"):
                run_cache_workers(self.paths)

    @staticmethod
    def _features(audio: AudioInput) -> np.ndarray:
        return np.full((128, 1), int(audio.samples[0]), dtype=np.float32)

    def _publish_fake_pilot(self, *, index_contents: str = "listening sheet", artifact_name: str | None = None):
        root = self.paths.run_root / "pilot"
        root.mkdir(parents=True, exist_ok=True)
        paths = {
            "index": root / "index.md",
            "original": root / "short.original.wav",
            "tokens": root / "short.tokens.npy",
            "reconstructed": root / "short.reconstructed.wav",
        }
        for path, contents in ((paths["index"], index_contents.encode()), (paths["original"], b"original"), (paths["tokens"], b"tokens"), (paths["reconstructed"], b"reconstructed")):
            path.write_bytes(contents)
        artifacts = {path.name: __import__("hashlib").sha256(path.read_bytes()).hexdigest() for path in paths.values()}
        if artifact_name is not None:
            artifacts = {artifact_name: artifacts["short.original.wav"]}
        payload = {
            "index": "index.md",
            "index_sha256": artifacts.get("index.md", "0" * 64),
            "clips": [{"original": "short.original.wav", "original_sha256": artifacts.get("short.original.wav", "0" * 64), "tokens": "short.tokens.npy", "tokens_sha256": artifacts.get("short.tokens.npy", "0" * 64), "reconstructed": "short.reconstructed.wav", "reconstructed_sha256": artifacts.get("short.reconstructed.wav", "0" * 64)}],
            "artifacts": artifacts,
        }
        return StageStore(self.paths.stages_dir).publish("pilot", payload), paths


if __name__ == "__main__":
    unittest.main()

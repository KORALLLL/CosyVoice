"""Behavioral tests for the compact, audio-free Balalaika cache."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from cosyvoice.finetune.balalaika.cache import (
    AudioInput,
    CacheIntegrityError,
    CacheShardRequest,
    build_cache_shard,
    iter_tar_audio,
    verify_cache,
)


FIXTURES = Path(__file__).with_name("fixtures")


class FakeTokenizer:
    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        return [[sample.frames % 6561, 7, 9] for sample in audio]


class BadTokenizer:
    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        return [[6561] for _ in audio]


class ChangedTokenizer:
    def extract(self, audio: list[AudioInput]) -> list[list[int]]:
        return [[11, 7, 9] for _ in audio]


class CacheTests(unittest.TestCase):
    """Each test catches an unsafe cache publication or tar reconciliation."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tar_path = self.root / "shard_000000.tar"
        self.plan_dir = self.root / "split_plan"
        self.cache_root = self.root / "cache"
        self.plan_dir.mkdir()
        shutil.copyfile(FIXTURES / "shard_000000.tar", self.tar_path)
        self.rows = [
            {"source_relative_path": "000000/a.mp3", "text": "one", "instruct": "inst", "agreement": 0.5, "phase": 1, "reserved": False},
            {"source_relative_path": "000000/b.mp3", "text": "two", "instruct": "inst", "agreement": 0.95, "phase": 2, "reserved": False},
            {"source_relative_path": "000000/c.mp3", "text": "three", "instruct": "inst", "agreement": 0.99, "phase": None, "reserved": True, "reservation_score": "00"},
        ]
        self._write_plan(self.rows, reserved=self._global_reserved("000000/c.mp3"))
        self.request = CacheShardRequest(self.tar_path, self.plan_dir, self.cache_root, 0, batch_size=2)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cache_has_no_audio_column(self) -> None:
        # Adding source bytes or features to training rows would defeat compact caching.
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            result = build_cache_shard(self.request, FakeTokenizer())
        phase1 = pq.read_table(result.phase1_path)
        self.assertEqual(
            phase1.column_names,
            ["source_relative_path", "text", "instruct", "agreement", "speech_token", "speech_token_len"],
        )
        self.assertNotIn("audio_data", phase1.column_names)
        self.assertEqual(phase1.column("speech_token").type.value_type.bit_width, 32)

    def test_tar_pairs_json_and_mp3_in_either_member_order(self) -> None:
        # Treating member order as schema would drop valid archive content.
        samples = list(iter_tar_audio(self.tar_path))
        self.assertEqual([sample.source_relative_path for sample in samples], ["000000/a.mp3", "000000/b.mp3", "000000/c.mp3"])
        self.assertEqual([sample.audio_bytes for sample in samples], [b"a", b"b", b"c"])

    def test_tar_rejects_missing_partner_unexpected_suffix_and_directory(self) -> None:
        # A partial pair, extra payload, or directory makes the immutable archive non-canonical.
        self._write_tar([("a.mp3", b"a")])
        with self.assertRaisesRegex(CacheIntegrityError, "missing JSON"):
            list(iter_tar_audio(self.tar_path))
        self._write_tar([("a.txt", b"bad")])
        with self.assertRaisesRegex(CacheIntegrityError, "unexpected tar member suffix"):
            list(iter_tar_audio(self.tar_path))
        self._write_tar([("nested", None)])
        with self.assertRaisesRegex(CacheIntegrityError, "unexpected non-file tar member"):
            list(iter_tar_audio(self.tar_path))

    def test_decode_and_token_range_errors_are_fatal(self) -> None:
        # Silently filtering either condition would make cache counts lie.
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=RuntimeError("bad mp3")):
            with self.assertRaisesRegex(CacheIntegrityError, "decode failed"):
                build_cache_shard(self.request, FakeTokenizer())
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            with self.assertRaisesRegex(CacheIntegrityError, r"outside \[0, 6560\]"):
                build_cache_shard(self.request, BadTokenizer())

    def test_stale_partial_is_recovered_and_phase_reserved_rows_route_correctly(self) -> None:
        # Restarting must discard incomplete output and never train a reserved prompt.
        partial = self.cache_root / "phase1" / ".shard_000000.parquet.partial"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"incomplete")
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            result = build_cache_shard(self.request, FakeTokenizer())
        self.assertFalse(partial.exists())
        self.assertEqual(pq.read_table(result.phase1_path).column("source_relative_path").to_pylist(), ["000000/a.mp3"])
        self.assertEqual(pq.read_table(result.phase2_path).column("source_relative_path").to_pylist(), ["000000/b.mp3"])
        prompts = pq.read_table(result.eval_prompts_path)
        self.assertEqual(prompts.column("source_relative_path").to_pylist(), ["000000/c.mp3"])
        self.assertTrue((self.cache_root / "eval_prompts/voice_00.wav").is_file())
        with self.assertRaisesRegex(CacheIntegrityError, "exactly 20 aggregate prompt"):
            verify_cache(self.cache_root)

    def test_split_manifest_requires_exactly_twenty_unique_prompt_identities(self) -> None:
        # A global prompt selection with 19 or a duplicate cannot name the fixed evaluation set.
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            self._write_plan(self.rows, reserved=self._global_reserved("000000/c.mp3")[:19])
            with self.assertRaisesRegex(CacheIntegrityError, "exactly 20"):
                build_cache_shard(self.request, FakeTokenizer())
            duplicate = self._global_reserved("000000/c.mp3")
            duplicate[-1] = duplicate[-2]
            self._write_plan(self.rows, reserved=duplicate)
            with self.assertRaisesRegex(CacheIntegrityError, "unique"):
                build_cache_shard(self.request, FakeTokenizer())

    def test_verify_requires_twenty_unique_prompt_records_and_wavs(self) -> None:
        # A completed cache must expose the exact globally reserved prompt set, not a shard subset.
        self._configure_twenty_prompt_cache()
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            build_cache_shard(self.request, FakeTokenizer())
        self.assertEqual(verify_cache(self.cache_root).prompt_count, 20)
        manifest_path = self.cache_root / "shard_manifests/shard_000000.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extra = dict(manifest["prompts"][0])
        shutil.copyfile(self.cache_root / extra["audio_path"], self.cache_root / "eval_prompts/voice_20.wav")
        extra["source_relative_path"] = "000000/extra.mp3"
        extra["audio_path"] = "eval_prompts/voice_20.wav"
        manifest["prompts"].append(extra)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(CacheIntegrityError, "exactly 20 aggregate prompt"):
            verify_cache(self.cache_root)

    def test_verify_rejects_an_unlisted_prompt_wav(self) -> None:
        # Counting only voice_NN names would leave an untracked audio exception in the cache.
        self._configure_twenty_prompt_cache()
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            build_cache_shard(self.request, FakeTokenizer())
        (self.cache_root / "eval_prompts/unlisted.wav").write_bytes(b"unexpected")
        with self.assertRaisesRegex(CacheIntegrityError, "exactly 20 prompt WAV files"):
            verify_cache(self.cache_root)

    def test_mid_publication_failure_restores_prior_cache_and_leaves_no_partials(self) -> None:
        # A failed replacement after phase 1 must not leave a new phase paired with an old manifest.
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            result = build_cache_shard(self.request, FakeTokenizer())
        original_phase1 = result.phase1_path.read_bytes()
        original_manifest = result.shard_manifest_path.read_bytes()
        original_replace = __import__("os").replace

        def fail_phase2(source, destination):
            if Path(destination) == result.phase2_path and Path(source).name.endswith(".partial"):
                raise OSError("injected phase2 publish failure")
            return original_replace(source, destination)

        with patch("cosyvoice.finetune.balalaika.cache.os.replace", side_effect=fail_phase2):
            with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
                with self.assertRaisesRegex(OSError, "injected phase2"):
                    build_cache_shard(self.request, ChangedTokenizer())
        self.assertEqual(result.phase1_path.read_bytes(), original_phase1)
        self.assertEqual(result.shard_manifest_path.read_bytes(), original_manifest)
        self.assertFalse(list(self.cache_root.rglob("*.partial")))
        self.assertFalse(list(self.cache_root.rglob("*.rollback")))

    def test_verify_rejects_phase_parquet_checksum_or_token_audit_changes(self) -> None:
        # Trusting a manifest without reopening the Parquet permits undetected token corruption.
        with patch("cosyvoice.finetune.balalaika.cache._decode_audio", side_effect=self._decoded):
            result = build_cache_shard(self.request, FakeTokenizer())
        table = pq.read_table(result.phase1_path)
        pq.write_table(table, result.phase1_path, compression="zstd")
        with self.assertRaisesRegex(CacheIntegrityError, "phase Parquet checksum changed"):
            verify_cache(self.cache_root)

    def _decoded(self, sample) -> AudioInput:
        return AudioInput(sample.source_relative_path, samples=[0.0] * 13, sample_rate=24_000, frames=13)

    def _write_plan(self, rows: list[dict[str, object]], *, reserved: list[str]) -> None:
        (self.plan_dir / "shard_000000.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        (self.plan_dir / "manifest.json").write_text(
            json.dumps({"reserved_prompts": [{"source_relative_path": key} for key in reserved]}), encoding="utf-8"
        )

    def _write_tar(self, members: list[tuple[str, object]]) -> None:
        with tarfile.open(self.tar_path, "w") as archive:
            for name, contents in members:
                if contents is None:
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                    continue
                payload = json.dumps(contents).encode("utf-8") if isinstance(contents, dict) else contents
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    def _global_reserved(self, first: str) -> list[str]:
        return [first, *[f"{shard:06d}/prompt_{shard:02d}.mp3" for shard in range(1, 20)]]

    def _configure_twenty_prompt_cache(self) -> None:
        prompts = [f"000000/prompt_{index:02d}.mp3" for index in range(20)]
        rows = [self.rows[0], self.rows[1]]
        members: list[tuple[str, object]] = [
            ("a.json", {"source_relative_path": "000000/a.mp3"}),
            ("a.mp3", b"a"),
            ("b.json", {"source_relative_path": "000000/b.mp3"}),
            ("b.mp3", b"b"),
        ]
        for index, source_relative_path in enumerate(prompts):
            stem = f"prompt_{index:02d}"
            rows.append({"source_relative_path": source_relative_path, "text": stem, "instruct": "inst", "agreement": 0.99, "phase": None, "reserved": True, "reservation_score": f"{index:02d}"})
            members.extend([(f"{stem}.json", {"source_relative_path": source_relative_path}), (f"{stem}.mp3", stem.encode("utf-8"))])
        self._write_tar(members)
        self._write_plan(rows, reserved=prompts)


if __name__ == "__main__":
    unittest.main()

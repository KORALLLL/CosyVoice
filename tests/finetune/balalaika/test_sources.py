"""Behavioral tests for the deterministic Balalaika source split plan."""

from __future__ import annotations

import json
import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cosyvoice.finetune.balalaika.config import RunPaths
from cosyvoice.finetune.balalaika.sources import (
    INSTRUCT,
    JoinedRow,
    SourceIntegrityError,
    assign_phases,
    build_split_plan,
    inventory_sources,
    iter_split_rows,
    reserve_prompt_ids,
)


FIXTURES = Path(__file__).with_name("fixtures")


def joined(source_relative_path: str, agreement: float | None) -> JoinedRow:
    return JoinedRow(source_relative_path=source_relative_path, text="Текст", agreement=agreement)


class SourceTests(unittest.TestCase):
    """Each assertion catches a bad split, non-canonical join, or bad audit."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        train = self.dataset_root / "train"
        train.mkdir(parents=True)
        for shard in range(519):
            (train / f"shard_{shard:06d}.tar").touch()
        combined = self.dataset_root / "combined_sidecars/rover-punctuation-stress-v1"
        combined.mkdir(parents=True)
        shutil.copyfile(FIXTURES / "combined.jsonl", combined / "rover-punctuation-stress.jsonl")
        rover = self.dataset_root / "punctuation_artifacts/20260729T135419Z"
        rover.mkdir(parents=True)
        shutil.copyfile(FIXTURES / "rover.jsonl", rover / "rover.jsonl")
        self.paths = RunPaths(
            dataset_root=self.dataset_root,
            repository_root=self.root / "repository",
            run_root=self.root / "run",
            base_model_dir=self.root / "model",
            visible_devices=tuple(range(8)),
            seed=1986,
        )
        self.rows = [joined(f"000000/{index:02d}.mp3", 0.95) for index in range(25)]
        self.by_id = {row.source_relative_path: row for row in self.rows}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_split_boundary_and_null(self) -> None:
        # A <= change here would wrongly put the exact boundary in phase 1.
        rows = [
            joined("000000/a.mp3", 0.949999),
            joined("000000/b.mp3", 0.95),
            joined("000000/c.mp3", None),
        ]
        assigned = assign_phases(rows, reserved=set())
        self.assertEqual([row.phase for row in assigned], [1, 2, None])

    def test_reservation_is_stable_and_high_agreement_only(self) -> None:
        # Selection must not depend on sidecar traversal order or include phase 1.
        selected_a = reserve_prompt_ids(self.rows, count=20, seed=1986)
        selected_b = reserve_prompt_ids(reversed(self.rows), count=20, seed=1986)
        self.assertEqual(selected_a, selected_b)
        self.assertTrue(all(self.by_id[key].agreement >= 0.95 for key in selected_a))

    def test_model_limit_exclusion_has_no_training_phase(self) -> None:
        # A preflighted row must not regain phase 2 through the generic assignment helper.
        limited = replace(joined("000000/limited.mp3", 0.95), model_limit_exclusion="text_token_length>200")
        assigned = assign_phases([limited], reserved=set())
        self.assertEqual((assigned[0].phase, assigned[0].reserved), (None, False))

    def test_duplicate_combined_id_is_rejected(self) -> None:
        # Silently keeping either duplicate would make the plan non-canonical.
        combined = self._combined_path()
        with combined.open("a", encoding="utf-8") as handle:
            handle.write(combined.read_text(encoding="utf-8").splitlines()[0] + "\n")
        with self.assertRaisesRegex(SourceIntegrityError, "duplicate combined"):
            self._build_fixture_plan()

    def test_duplicate_rover_id_is_rejected(self) -> None:
        # A duplicated score must fail rather than select an arbitrary agreement.
        rover = self._rover_path()
        with rover.open("a", encoding="utf-8") as handle:
            handle.write(rover.read_text(encoding="utf-8").splitlines()[0] + "\n")
        with self.assertRaisesRegex(SourceIntegrityError, "duplicate ROVER"):
            self._build_fixture_plan()

    def test_unequal_join_keys_are_rejected(self) -> None:
        # Missing transcript/agreement data must never be silently dropped.
        rover = self._rover_path()
        rows = [json.loads(line) for line in rover.read_text(encoding="utf-8").splitlines()]
        rows[-1]["source_relative_path"] = "000000/missing.mp3"
        rover.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "join keys differ"):
            self._build_fixture_plan()

    def test_inventory_rejects_518_or_520_archives(self) -> None:
        # A missing or extra archive changes the immutable corpus identity.
        (self.dataset_root / "train/shard_000518.tar").unlink()
        with self.assertRaisesRegex(SourceIntegrityError, "519 source tar archives"):
            inventory_sources(self.paths)

    def test_split_plan_reuses_prequalified_inventory_without_rehashing(self) -> None:
        with patch("cosyvoice.finetune.balalaika.sources.ROVER_ARCHIVE_RELATIVE", "punctuation_artifacts/20260729T135419Z/rover.jsonl"):
            inventory = inventory_sources(self.paths)
            with (
                patch(
                    "cosyvoice.finetune.balalaika.sources.inventory_sources",
                    side_effect=AssertionError("source inventory must not be rebuilt"),
                ),
                patch("cosyvoice.finetune.balalaika.sources.EXPECTED_SOURCE_ROWS", 3),
                patch("cosyvoice.finetune.balalaika.sources.EXPECTED_NULL_ROWS", 1),
                patch("cosyvoice.finetune.balalaika.sources.PROMPT_RESERVATION_COUNT", 0),
            ):
                counts = build_split_plan(
                    self.paths,
                    source_inventory=inventory,
                    _count_text_tokens=lambda text: 1,
                )

        self.assertEqual(counts.total, 3)
        (self.dataset_root / "train/shard_000518.tar").touch()
        (self.dataset_root / "train/shard_000519.tar").touch()
        with self.assertRaisesRegex(SourceIntegrityError, "519 source tar archives"):
            inventory_sources(self.paths)

    def test_join_rejects_sidecar_paths_outside_the_canonical_tar_range(self) -> None:
        # A six-digit but non-existent shard must not create an unreachable plan row.
        combined = self._combined_path()
        rows = [json.loads(line) for line in combined.read_text(encoding="utf-8").splitlines()]
        rows[-1]["source_relative_path"] = "000519/c.mp3"
        combined.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "outside canonical tar range"):
            self._build_fixture_plan()

    def test_split_manifest_reconciles_phase_null_and_reserved_counts(self) -> None:
        # A reserved prompt must be removed from phase 2 yet remain in total audit.
        counts = self._build_fixture_plan(expected_nulls=1, reserve_count=1)
        self.assertEqual((counts.phase1, counts.phase2, counts.null, counts.reserved), (1, 0, 1, 1))
        self.assertEqual(counts.total, 3)
        manifest = json.loads((counts.plan_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["phase1"] + manifest["phase2"] + manifest["null"] + manifest["reserved"] + manifest["model_limit_exclusions"], 3)
        self.assertEqual(
            manifest["source_schema_versions"],
            {"combined_sidecar": 1, "rover_archive": 1},
        )
        rows = list(iter_split_rows(counts.plan_dir, 0))
        self.assertEqual([row.instruct for row in rows], [INSTRUCT, INSTRUCT, INSTRUCT])
        self.assertEqual(sum(row.reserved for row in rows), 1)

    def test_non_finite_agreement_and_wrong_schema_are_rejected(self) -> None:
        # NaN and unrecognized data revisions invalidate agreement ordering.
        rover = self._rover_path()
        rows = [json.loads(line) for line in rover.read_text(encoding="utf-8").splitlines()]
        rows[0]["asr_agreement_mean"] = float("nan")
        rover.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "non-finite agreement"):
            self._build_fixture_plan()

        shutil.copyfile(FIXTURES / "rover.jsonl", rover)
        rows = [json.loads(line) for line in rover.read_text(encoding="utf-8").splitlines()]
        rows[0]["schema_version"] = "unknown-v2"
        rover.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "schema version"):
            self._build_fixture_plan()

    def test_combined_and_rover_each_require_canonical_integer_schema(self) -> None:
        combined = self._combined_path()
        combined_rows = [json.loads(line) for line in combined.read_text(encoding="utf-8").splitlines()]
        combined_rows[0]["schema_version"] = "rover-punctuation-stress-v1"
        combined.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SourceIntegrityError, "combined schema version"):
            self._build_fixture_plan()

        shutil.copyfile(FIXTURES / "combined.jsonl", combined)
        rover = self._rover_path()
        rover_rows = [json.loads(line) for line in rover.read_text(encoding="utf-8").splitlines()]
        rover_rows[0]["schema_version"] = "rover-punctuation-stress-v1"
        rover.write_text("".join(json.dumps(row) + "\n" for row in rover_rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "ROVER schema version"):
            self._build_fixture_plan()

    def test_rover_archive_is_streamed_and_requires_agreement_field(self) -> None:
        # Treating an absent score as a null would silently contaminate the null audit.
        rover_jsonl = self._rover_path()
        archive = rover_jsonl.with_name("rover.tar.zst")
        self._write_zstd_tar(archive, rover_jsonl.read_bytes())
        with patch("cosyvoice.finetune.balalaika.sources.ROVER_ARCHIVE_RELATIVE", "punctuation_artifacts/20260729T135419Z/rover.tar.zst"):
            with (
                patch("cosyvoice.finetune.balalaika.sources.EXPECTED_SOURCE_ROWS", 3),
                patch("cosyvoice.finetune.balalaika.sources.EXPECTED_NULL_ROWS", 1),
                patch("cosyvoice.finetune.balalaika.sources.PROMPT_RESERVATION_COUNT", 0),
            ):
                counts = build_split_plan(self.paths, _count_text_tokens=lambda text: 1)
        self.assertEqual(counts.total, 3)

        rows = [json.loads(line) for line in rover_jsonl.read_text(encoding="utf-8").splitlines()]
        del rows[0]["asr_agreement_mean"]
        rover_jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "missing asr_agreement_mean"):
            self._build_fixture_plan()

    def test_rover_archive_reverse_member_order_joins_ascending_combined_sidecar(self) -> None:
        combined_rows = [
            {"schema_version": 1, "source_relative_path": "000000/a.mp3", "rover_punctuated_accented": "Первый"},
            {"schema_version": 1, "source_relative_path": "000001/b.mp3", "rover_punctuated_accented": "Второй"},
        ]
        self._combined_path().write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows),
            encoding="utf-8",
        )
        rover = self._rover_path().with_name("reverse-rover.tar.zst")
        self._write_zstd_tar_members(
            rover,
            [
                ("results/train/shard_000001.jsonl", {"schema_version": 1, "source_relative_path": "000001/b.mp3", "asr_agreement_mean": 0.96}),
                ("results/train/shard_000000.jsonl", {"schema_version": 1, "source_relative_path": "000000/a.mp3", "asr_agreement_mean": 0.90}),
            ],
        )

        with (
            patch("cosyvoice.finetune.balalaika.sources.ROVER_ARCHIVE_RELATIVE", "punctuation_artifacts/20260729T135419Z/reverse-rover.tar.zst"),
            patch("cosyvoice.finetune.balalaika.sources.EXPECTED_SOURCE_ROWS", 2),
            patch("cosyvoice.finetune.balalaika.sources.EXPECTED_NULL_ROWS", 0),
            patch("cosyvoice.finetune.balalaika.sources.PROMPT_RESERVATION_COUNT", 0),
        ):
            counts = build_split_plan(self.paths, _count_text_tokens=lambda text: 1)

        self.assertEqual((counts.phase1, counts.phase2, counts.total), (1, 1, 2))

    def test_over_limit_phase2_text_is_excluded_before_reservation(self) -> None:
        # A character-count proxy could reserve this row despite 201 real tokens.
        combined = self._combined_path()
        combined_rows = [json.loads(line) for line in combined.read_text(encoding="utf-8").splitlines()]
        combined_rows[-1]["rover_punctuated_accented"] = "Слишком длинный текст"
        combined.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows), encoding="utf-8")
        rover = self._rover_path()
        rover_rows = [json.loads(line) for line in rover.read_text(encoding="utf-8").splitlines()]
        rover_rows[-1]["asr_agreement_mean"] = 0.96
        rover.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rover_rows), encoding="utf-8")

        counts = self._build_fixture_plan(
            expected_nulls=0,
            reserve_count=1,
            count_tokens=lambda text: 201 if text == "Слишком длинный текст" else 1,
        )

        self.assertEqual((counts.phase1, counts.phase2, counts.null, counts.reserved, counts.model_limit_exclusions), (1, 0, 0, 1, 1))
        rows = {row.source_relative_path: row for row in iter_split_rows(counts.plan_dir, 0)}
        self.assertFalse(rows["000000/c.mp3"].reserved)
        self.assertEqual(rows["000000/c.mp3"].model_limit_exclusion, "text_token_length>200")
        self.assertEqual(rows["000000/c.mp3"].text_token_count, 201)

    def test_canonical_rover_rejects_legacy_agreement_field(self) -> None:
        # Legacy agreement must not be mistaken for canonical ROVER evidence.
        rover = self._rover_path()
        rows = [json.loads(line) for line in rover.read_text(encoding="utf-8").splitlines()]
        rows[0]["agreement"] = rows[0].pop("asr_agreement_mean")
        rover.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SourceIntegrityError, "missing asr_agreement_mean"):
            self._build_fixture_plan()

    def _combined_path(self) -> Path:
        return self.dataset_root / "combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl"

    def _rover_path(self) -> Path:
        return self.dataset_root / "punctuation_artifacts/20260729T135419Z/rover.jsonl"

    def _build_fixture_plan(self, *, expected_nulls: int = 1, reserve_count: int = 0, count_tokens=None):
        with (
            patch("cosyvoice.finetune.balalaika.sources.ROVER_ARCHIVE_RELATIVE", "punctuation_artifacts/20260729T135419Z/rover.jsonl"),
            patch("cosyvoice.finetune.balalaika.sources.EXPECTED_SOURCE_ROWS", 3),
            patch("cosyvoice.finetune.balalaika.sources.EXPECTED_NULL_ROWS", expected_nulls),
            patch("cosyvoice.finetune.balalaika.sources.PROMPT_RESERVATION_COUNT", reserve_count),
        ):
            return build_split_plan(self.paths, seed=1986, _count_text_tokens=count_tokens or (lambda text: 1))

    def _write_zstd_tar(self, destination: Path, rover_jsonl: bytes) -> None:
        tar_path = destination.with_suffix("")
        with tarfile.open(tar_path, "w") as archive:
            info = tarfile.TarInfo("rover.jsonl")
            info.size = len(rover_jsonl)
            archive.addfile(info, io.BytesIO(rover_jsonl))
        subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(destination)], check=True)

    def _write_zstd_tar_members(self, destination: Path, rows: list[tuple[str, dict[str, object]]]) -> None:
        tar_path = destination.with_suffix("")
        with tarfile.open(tar_path, "w") as archive:
            for name, row in rows:
                payload = (json.dumps(row) + "\n").encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(destination)], check=True)


if __name__ == "__main__":
    unittest.main()

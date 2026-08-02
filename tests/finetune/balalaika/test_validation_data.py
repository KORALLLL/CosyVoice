"""Local-fixture tests for the private hard-number validation snapshot."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cosyvoice.finetune.balalaika.validation_data import ValidationDataError, fetch_validation_rows


def rows(count: int = 2000) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "stressed": f"Товар {index} готов",
            "normalized_gold": f"Товар номер {index} готов",
            "hard_number": str(index),
            "category": f"category-{(index - 1) % 12}",
            "source": "fixture",
            "language": "ru",
            "split": "train",
            "revision": "fixture-revision",
            "notes": "local-only",
        }
        for index in range(1, count + 1)
    ]


class ValidationDataTests(unittest.TestCase):
    def test_fetch_validates_all_rows_and_publishes_a_checksum_bound_manifest(self):
        fixture_rows = rows()
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                patch("cosyvoice.finetune.balalaika.validation_data._resolve_parquet", return_value=("https://example.invalid/snapshot.parquet", "fixture-revision")),
                patch("cosyvoice.finetune.balalaika.validation_data._download_parquet", side_effect=lambda url, target, token: target.write_bytes(b"fixture parquet")),
                patch("cosyvoice.finetune.balalaika.validation_data._load_parquet_rows", return_value=(fixture_rows, tuple(fixture_rows[0]))),
            ):
                loaded = fetch_validation_rows(cache_dir, "secret-token")

            self.assertEqual(len(loaded), 2000)
            manifest = json.loads((cache_dir / "hard-number-validation.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "bitmanagerai/hard_number_eval_for_tts")
            self.assertEqual(manifest["revision"], "fixture-revision")
            self.assertEqual(len(manifest["number_spans"]), 2000)
            self.assertEqual(sum(manifest["category_counts"].values()), 2000)
            self.assertNotIn("secret-token", (cache_dir / "hard-number-validation.manifest.json").read_text(encoding="utf-8"))

    def test_fetch_refuses_publication_when_any_row_has_an_ambiguous_span(self):
        fixture_rows = rows()
        fixture_rows[145] = {**fixture_rows[145], "stressed": "Код 25 готов", "hard_number": "25", "normalized_gold": "Код двадцать пять готов код двадцать пять готов"}
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                patch("cosyvoice.finetune.balalaika.validation_data._resolve_parquet", return_value=("https://example.invalid/snapshot.parquet", "fixture-revision")),
                patch("cosyvoice.finetune.balalaika.validation_data._download_parquet", side_effect=lambda url, target, token: target.write_bytes(b"fixture parquet")),
                patch("cosyvoice.finetune.balalaika.validation_data._load_parquet_rows", return_value=(fixture_rows, tuple(fixture_rows[0]))),
            ):
                with self.assertRaisesRegex(ValidationDataError, "number-span preflight failed"):
                    fetch_validation_rows(cache_dir, "secret-token")
            self.assertFalse((cache_dir / "hard-number-validation.manifest.json").exists())

    def test_fetch_rejects_wrong_shape_without_echoing_the_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                patch("cosyvoice.finetune.balalaika.validation_data._resolve_parquet", return_value=("https://example.invalid/snapshot.parquet", "fixture-revision")),
                patch("cosyvoice.finetune.balalaika.validation_data._download_parquet", side_effect=lambda url, target, token: target.write_bytes(b"fixture parquet")),
                patch("cosyvoice.finetune.balalaika.validation_data._load_parquet_rows", return_value=(rows(1999), tuple(rows()[0]))),
            ):
                with self.assertRaises(ValidationDataError) as raised:
                    fetch_validation_rows(cache_dir, "secret-token")
            self.assertNotIn("secret-token", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

"""Local-fixture tests for the private hard-number validation snapshot."""

import json
import os
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


def schema() -> tuple[tuple[str, str], ...]:
    return tuple((name, "int64" if name == "id" else "string") for name in rows(1)[0])


class ValidationDataTests(unittest.TestCase):
    def test_fetch_validates_all_rows_and_publishes_a_checksum_bound_manifest(self):
        fixture_rows = rows()
        downloaded_tokens: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                patch("cosyvoice.finetune.balalaika.validation_data._resolve_parquet", return_value=("https://example.invalid/snapshot.parquet", "fixture-revision")),
                patch("cosyvoice.finetune.balalaika.validation_data._download_parquet", side_effect=lambda url, target, token: (downloaded_tokens.append(token), target.write_bytes(b"fixture parquet"))[1]),
                patch("cosyvoice.finetune.balalaika.validation_data._load_parquet_rows", return_value=(fixture_rows, schema())),
                patch.dict(os.environ, {"HF_TOKEN": "secret-token"}, clear=False),
            ):
                loaded = fetch_validation_rows(cache_dir)

            self.assertEqual(len(loaded), 2000)
            manifest = json.loads((cache_dir / "hard-number-validation.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "bitmanagerai/hard_number_eval_for_tts")
            self.assertEqual(manifest["revision"], "fixture-revision")
            self.assertEqual(len(manifest["number_spans"]), 2000)
            self.assertEqual(sum(manifest["category_counts"].values()), 2000)
            self.assertEqual(manifest["schema"]["fields"][0], {"name": "id", "type": "int64"})
            self.assertTrue((cache_dir / "hard-number-validation.parquet").is_file())
            self.assertEqual(downloaded_tokens, ["secret-token"])
            self.assertNotIn("secret-token", (cache_dir / "hard-number-validation.manifest.json").read_text(encoding="utf-8"))

    def test_fetch_refuses_publication_when_any_row_has_an_ambiguous_span(self):
        fixture_rows = rows()
        fixture_rows[145] = {**fixture_rows[145], "stressed": "Код 25 готов", "hard_number": "25", "normalized_gold": "Код двадцать пять готов код двадцать пять готов"}
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                patch("cosyvoice.finetune.balalaika.validation_data._resolve_parquet", return_value=("https://example.invalid/snapshot.parquet", "fixture-revision")),
                patch("cosyvoice.finetune.balalaika.validation_data._download_parquet", side_effect=lambda url, target, token: target.write_bytes(b"fixture parquet")),
                patch("cosyvoice.finetune.balalaika.validation_data._load_parquet_rows", return_value=(fixture_rows, schema())),
                patch.dict(os.environ, {"HF_TOKEN": "secret-token"}, clear=False),
            ):
                with self.assertRaisesRegex(ValidationDataError, "number-span preflight failed"):
                    fetch_validation_rows(cache_dir)
            self.assertFalse((cache_dir / "hard-number-validation.manifest.json").exists())
            self.assertFalse((cache_dir / "hard-number-validation.parquet").exists())
            self.assertFalse(list(cache_dir.glob(".hard-number-validation.parquet.*")))

    def test_fetch_rejects_wrong_shape_without_echoing_the_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            with (
                patch("cosyvoice.finetune.balalaika.validation_data._resolve_parquet", return_value=("https://example.invalid/snapshot.parquet", "fixture-revision")),
                patch("cosyvoice.finetune.balalaika.validation_data._download_parquet", side_effect=lambda url, target, token: target.write_bytes(b"fixture parquet")),
                patch("cosyvoice.finetune.balalaika.validation_data._load_parquet_rows", return_value=(rows(1999), schema())),
                patch.dict(os.environ, {"HF_TOKEN": "secret-token"}, clear=False),
            ):
                with self.assertRaises(ValidationDataError) as raised:
                    fetch_validation_rows(cache_dir)
            self.assertNotIn("secret-token", str(raised.exception))

    def test_fetch_requires_hf_token_from_environment(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValidationDataError, "HF_TOKEN is required"):
                fetch_validation_rows(Path(temporary))


if __name__ == "__main__":
    unittest.main()

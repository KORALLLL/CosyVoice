"""Authenticated, checksum-bound private hard-number validation snapshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .artifacts import atomic_write_json, sha256_file
from .metrics import NumberSpanError, NumberSpan, extract_number_span


DATASET = "bitmanagerai/hard_number_eval_for_tts"
CONFIG = "default"
SPLIT = "train"
EXPECTED_ROWS = 2_000
EXPECTED_COLUMNS = 10
EXPECTED_CATEGORIES = 12
PARQUET_NAME = "hard-number-validation.parquet"
MANIFEST_NAME = "hard-number-validation.manifest.json"


class ValidationDataError(RuntimeError):
    """Raised when the private benchmark cannot be safely published."""


@dataclass(frozen=True)
class ValidationRow:
    id: int
    stressed: str
    normalized_gold: str
    hard_number: str
    category: str
    raw: Mapping[str, object]


def fetch_validation_rows(cache_dir: Path, token: str) -> list[ValidationRow]:
    """Fetch, validate, preflight, and atomically publish the exact benchmark."""

    if not isinstance(token, str) or not token.strip():
        raise ValidationDataError("HF_TOKEN is required to fetch the private validation dataset")
    try:
        parquet_url, revision_or_etag = _resolve_parquet(token)
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / PARQUET_NAME
        _download_parquet(parquet_url, parquet_path, token)
        raw_rows, schema = _load_parquet_rows(parquet_path)
        validation_rows = _validate_rows(raw_rows, schema)
    except ValidationDataError:
        raise
    except Exception as exc:
        raise ValidationDataError("unable to fetch or load private validation dataset") from exc

    spans, failures = _preflight_spans(validation_rows)
    if failures:
        raise ValidationDataError(f"number-span preflight failed for {len(failures)} rows")

    category_counts = Counter(row.category for row in validation_rows)
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "rows": EXPECTED_ROWS,
        "revision": revision_or_etag,
        "etag": revision_or_etag,
        "parquet": {"name": parquet_path.name, "sha256": sha256_file(parquet_path)},
        "schema": {"columns": list(schema), "column_count": len(schema)},
        "category_counts": dict(sorted(category_counts.items())),
        "number_spans": [
            {"id": row.id, "reference_start": span.reference_start, "reference_end": span.reference_end}
            for row, span in zip(validation_rows, spans, strict=True)
        ],
    }
    atomic_write_json(cache_dir / MANIFEST_NAME, manifest)
    return validation_rows


def _resolve_parquet(token: str) -> tuple[str, str]:
    query = urlencode({"dataset": DATASET})
    request = Request(f"https://datasets-server.huggingface.co/parquet?{query}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
            etag = response.headers.get("ETag", "")
    except Exception as exc:
        raise ValidationDataError("dataset viewer request failed") from exc
    files = payload.get("parquet_files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValidationDataError("dataset viewer returned no Parquet listing")
    matching = [entry for entry in files if isinstance(entry, dict) and entry.get("config") == CONFIG and entry.get("split") == SPLIT]
    if len(matching) != 1 or not isinstance(matching[0].get("url"), str):
        raise ValidationDataError("dataset viewer did not resolve the required default/train Parquet")
    revision = matching[0].get("revision")
    identity = revision if isinstance(revision, str) and revision else etag
    if not identity:
        raise ValidationDataError("dataset snapshot has no revision or ETag")
    return matching[0]["url"], identity


def _download_parquet(url: str, target: Path, token: str) -> None:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    temporary_name: str | None = None
    try:
        with urlopen(request) as response, tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
            temporary_name = handle.name
            while block := response.read(1024 * 1024):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception as exc:
        raise ValidationDataError("private Parquet download failed") from exc
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_parquet_rows(path: Path) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    try:
        from pyarrow.parquet import read_table

        table = read_table(path)
    except Exception as exc:
        raise ValidationDataError("private Parquet cannot be read with PyArrow") from exc
    return list(table.to_pylist()), tuple(table.column_names)


def _validate_rows(raw_rows: list[dict[str, object]], schema: tuple[str, ...]) -> list[ValidationRow]:
    if len(schema) != EXPECTED_COLUMNS:
        raise ValidationDataError(f"validation schema must contain exactly {EXPECTED_COLUMNS} columns")
    if len(raw_rows) != EXPECTED_ROWS:
        raise ValidationDataError(f"validation dataset must contain exactly {EXPECTED_ROWS} rows")
    required = ("id", "stressed", "normalized_gold", "hard_number", "category")
    if any(name not in schema for name in required):
        raise ValidationDataError("validation schema is missing required columns")

    rows: list[ValidationRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValidationDataError("validation row is not a mapping")
        identifier = raw.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise ValidationDataError("validation IDs must be integers")
        values = {name: raw.get(name) for name in required[1:]}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValidationDataError("validation required strings must be nonempty")
        rows.append(
            ValidationRow(
                id=identifier,
                stressed=values["stressed"],
                normalized_gold=values["normalized_gold"],
                hard_number=values["hard_number"],
                category=values["category"],
                raw=dict(raw),
            )
        )
    if {row.id for row in rows} != set(range(1, EXPECTED_ROWS + 1)):
        raise ValidationDataError("validation IDs must be unique and cover 1 through 2000")
    if len({row.category for row in rows}) != EXPECTED_CATEGORIES:
        raise ValidationDataError(f"validation dataset must contain exactly {EXPECTED_CATEGORIES} categories")
    return rows


def _preflight_spans(rows: list[ValidationRow]) -> tuple[list[NumberSpan], list[int]]:
    spans: list[NumberSpan] = []
    failures: list[int] = []
    for row in rows:
        try:
            spans.append(extract_number_span(row.raw))
        except NumberSpanError:
            failures.append(row.id)
    return spans, failures

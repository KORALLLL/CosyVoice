"""Authenticated, checksum-bound private hard-number validation snapshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_url

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
PARQUET_REVISION = "refs/convert/parquet"


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


def fetch_validation_rows(cache_dir: Path) -> list[ValidationRow]:
    """Fetch, validate, preflight, and atomically publish the exact benchmark."""

    token = os.environ.get("HF_TOKEN")
    if not isinstance(token, str) or not token.strip():
        raise ValidationDataError("HF_TOKEN is required to fetch the private validation dataset")
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged_parquet = _temporary_path(cache_dir, PARQUET_NAME)
    staged_manifest = _temporary_path(cache_dir, MANIFEST_NAME)
    try:
        parquet_url, revision_or_etag = _resolve_parquet(token)
        _download_parquet(parquet_url, staged_parquet, token)
        raw_rows, schema = _load_parquet_rows(staged_parquet)
        validation_rows = _validate_rows(raw_rows, schema)
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
            "parquet": {"name": PARQUET_NAME, "sha256": sha256_file(staged_parquet)},
            "schema": {
                "fields": [{"name": name, "type": field_type} for name, field_type in schema],
                "columns": [name for name, _ in schema],
                "column_count": len(schema),
            },
            "category_counts": dict(sorted(category_counts.items())),
            "number_spans": [
                {"id": row.id, "reference_start": span.reference_start, "reference_end": span.reference_end}
                for row, span in zip(validation_rows, spans, strict=True)
            ],
        }
        atomic_write_json(staged_manifest, manifest)
        _publish_snapshot(staged_parquet, staged_manifest, cache_dir / PARQUET_NAME, cache_dir / MANIFEST_NAME)
    except ValidationDataError:
        raise
    except Exception as exc:
        raise ValidationDataError("unable to fetch or load private validation dataset") from exc
    finally:
        _unlink_if_exists(staged_parquet)
        _unlink_if_exists(staged_manifest)
    return validation_rows


def _resolve_parquet(token: str) -> tuple[str, str]:
    query = urlencode({"dataset": DATASET})
    request = Request(f"https://datasets-server.huggingface.co/parquet?{query}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise ValidationDataError("dataset viewer request failed") from exc
    files = payload.get("parquet_files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValidationDataError("dataset viewer returned no Parquet listing")
    matching = [entry for entry in files if isinstance(entry, dict) and entry.get("config") == CONFIG and entry.get("split") == SPLIT]
    if len(matching) != 1 or not isinstance(matching[0].get("url"), str):
        raise ValidationDataError("dataset viewer did not resolve the required default/train Parquet")
    filename = matching[0].get("filename")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValidationDataError("dataset viewer returned an invalid Parquet filename")
    try:
        revision = HfApi(token=token).dataset_info(DATASET, revision=PARQUET_REVISION).sha
    except Exception as exc:
        raise ValidationDataError("dataset conversion revision lookup failed") from exc
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValidationDataError("dataset conversion snapshot has no immutable revision")
    immutable_url = hf_hub_url(
        DATASET,
        f"{CONFIG}/{SPLIT}/{filename}",
        repo_type="dataset",
        revision=revision,
    )
    return immutable_url, revision


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


def _load_parquet_rows(path: Path) -> tuple[list[dict[str, object]], tuple[tuple[str, str], ...]]:
    try:
        from pyarrow.parquet import read_table

        table = read_table(path)
    except Exception as exc:
        raise ValidationDataError("private Parquet cannot be read with PyArrow") from exc
    return list(table.to_pylist()), tuple((field.name, str(field.type)) for field in table.schema)


def _validate_rows(raw_rows: list[dict[str, object]], schema: tuple[tuple[str, str], ...]) -> list[ValidationRow]:
    if len(schema) != EXPECTED_COLUMNS:
        raise ValidationDataError(f"validation schema must contain exactly {EXPECTED_COLUMNS} columns")
    if len(raw_rows) != EXPECTED_ROWS:
        raise ValidationDataError(f"validation dataset must contain exactly {EXPECTED_ROWS} rows")
    required = ("id", "stressed", "normalized_gold", "hard_number", "category")
    if any(not isinstance(name, str) or not name or not isinstance(field_type, str) or not field_type for name, field_type in schema):
        raise ValidationDataError("validation schema fields must have nonempty names and PyArrow types")
    field_names = tuple(name for name, _ in schema)
    if len(set(field_names)) != len(field_names) or any(name not in field_names for name in required):
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


def _temporary_path(directory: Path, filename: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=f".{filename}.")
    os.close(descriptor)
    return Path(name)


def _publish_snapshot(staged_parquet: Path, staged_manifest: Path, parquet_path: Path, manifest_path: Path) -> None:
    """Replace the two finalized files only after a complete staged preflight."""

    targets = (parquet_path, manifest_path)
    backups = {target: _temporary_path(target.parent, target.name) for target in targets if target.is_file()}
    published: list[Path] = []
    try:
        for target, backup in backups.items():
            os.replace(target, backup)
        os.replace(staged_parquet, parquet_path)
        published.append(parquet_path)
        os.replace(staged_manifest, manifest_path)
        published.append(manifest_path)
    except Exception:
        for target in published:
            _unlink_if_exists(target)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            _unlink_if_exists(backup)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

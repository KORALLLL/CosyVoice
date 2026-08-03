"""Read-only canonical Balalaika sidecar join and deterministic split plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Callable, Iterable, Iterator, Mapping, TextIO

from .artifacts import atomic_write_json, sha256_file
from .config import RunPaths


COMBINED_SIDECAR_RELATIVE = Path("combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl")
ROVER_ARCHIVE_RELATIVE = Path("punctuation_artifacts/20260729T135419Z/balalaika-rover-results-20260729T135419Z.tar.zst")
COMBINED_SCHEMA_VERSION = 1
ROVER_SCHEMA_VERSION = 1
SPLIT_PLAN_SCHEMA_VERSION = "rover-punctuation-stress-v1"
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
EXPECTED_SOURCE_TARS = 519
EXPECTED_SOURCE_ROWS = 4_075_032
EXPECTED_NULL_ROWS = 141
EXPECTED_EMPTY_TEXT_ROWS = 168
EXPECTED_OVER_LIMIT_ROWS = 26_729
PROMPT_RESERVATION_COUNT = 20
MAX_ROWS_PER_SHARD = 8_000
TEXT_TOKEN_MAX_LENGTH = 200

_SOURCE_PATH = re.compile(r"(?P<shard>\d{6})/(?P<name>[^/]+\.mp3)")


class SourceIntegrityError(RuntimeError):
    """Raised when a supposedly canonical corpus input cannot be reconciled."""


@dataclass(frozen=True)
class JoinedRow:
    """The auditable text/agreement record for one source MP3."""

    source_relative_path: str
    text: str
    agreement: float | None
    instruct: str = INSTRUCT
    phase: int | None = None
    reserved: bool = False
    reservation_score: str | None = None
    model_limit_exclusion: str | None = None
    text_token_count: int | None = None


@dataclass(frozen=True)
class SourceInventory:
    """Checksummed immutable inputs required to construct one split plan."""

    source_archives: tuple[Path, ...]
    source_archive_sha256: Mapping[str, str]
    combined_sidecar: Path
    combined_sha256: str
    rover_archive: Path
    rover_sha256: str


@dataclass(frozen=True)
class SplitCounts:
    """Reconciled split totals and the published plan location."""

    phase1: int
    phase2: int
    null: int
    reserved: int
    model_limit_exclusions: int
    total: int
    plan_dir: Path
    manifest_sha256: str


@dataclass
class _JoinAudit:
    combined_rows: int = 0
    rover_rows: int = 0


def inventory_sources(paths: RunPaths) -> SourceInventory:
    """Validate the exact immutable archive set and checksum every input file."""

    train_dir = paths.dataset_root / "train"
    archives = tuple(sorted(train_dir.glob("shard_*.tar")))
    expected_names = {f"shard_{shard:06d}.tar" for shard in range(EXPECTED_SOURCE_TARS)}
    if {archive.name for archive in archives} != expected_names:
        raise SourceIntegrityError(f"expected exactly {EXPECTED_SOURCE_TARS} source tar archives named shard_000000.tar through shard_000518.tar")

    combined = paths.dataset_root / COMBINED_SIDECAR_RELATIVE
    rover = paths.dataset_root / ROVER_ARCHIVE_RELATIVE
    if not combined.is_file():
        raise SourceIntegrityError(f"missing combined sidecar: {combined}")
    if not rover.is_file():
        raise SourceIntegrityError(f"missing ROVER archive: {rover}")

    return SourceInventory(
        source_archives=archives,
        source_archive_sha256={archive.name: sha256_file(archive) for archive in archives},
        combined_sidecar=combined,
        combined_sha256=sha256_file(combined),
        rover_archive=rover,
        rover_sha256=sha256_file(rover),
    )


def reservation_score(source_relative_path: str, seed: int) -> bytes:
    """Return the order-independent selection key prescribed for prompt clips."""

    material = f"{seed}\0{source_relative_path}".encode("utf-8")
    return hashlib.sha256(material).digest()


def reserve_prompt_ids(rows: Iterable[JoinedRow], count: int = PROMPT_RESERVATION_COUNT, seed: int = 1986) -> tuple[str, ...]:
    """Select the ``count`` lexicographically lowest deterministic prompt scores."""

    if count < 0:
        raise ValueError("prompt reservation count must be non-negative")
    selected: list[tuple[bytes, str]] = []
    for row in rows:
        if row.agreement is None or row.agreement < 0.95 or _model_limit_excluded(row):
            continue
        candidate = (reservation_score(row.source_relative_path, seed), row.source_relative_path)
        if len(selected) < count:
            selected.append(candidate)
            selected.sort(reverse=True)
        elif count and candidate < selected[0]:
            selected[0] = candidate
            selected.sort(reverse=True)
    return tuple(source_relative_path for _, source_relative_path in sorted(selected))


def assign_phases(rows: Iterable[JoinedRow], reserved: set[str]) -> list[JoinedRow]:
    """Assign only non-null, non-reserved rows to their literal agreement phase."""

    assigned: list[JoinedRow] = []
    for row in rows:
        if _model_limit_excluded(row):
            assigned.append(replace(row, phase=None, reserved=False, reservation_score=None))
        elif row.agreement is None:
            assigned.append(replace(row, phase=None, reserved=False, reservation_score=None))
        elif row.source_relative_path in reserved:
            assigned.append(replace(row, phase=None, reserved=True))
        elif row.agreement < 0.95:
            assigned.append(replace(row, phase=1, reserved=False, reservation_score=None))
        else:
            assigned.append(replace(row, phase=2, reserved=False, reservation_score=None))
    return assigned


def build_split_plan(
    paths: RunPaths,
    seed: int = 1986,
    *,
    source_inventory: SourceInventory | None = None,
    _count_text_tokens: Callable[[str], int] | None = None,
) -> SplitCounts:
    """Publish a per-source-shard plan only after full canonical reconciliation."""

    inventory = source_inventory if source_inventory is not None else inventory_sources(paths)
    reused = _reuse_published_split_plan(paths, inventory, seed)
    if reused is not None:
        return reused
    count_text_tokens = _count_text_tokens or _load_cosyvoice3_text_token_counter(paths)
    first_audit = _JoinAudit()
    selected_ids = reserve_prompt_ids(
        _preflight_text_limits(_iter_joined_rows(inventory, first_audit), count_text_tokens),
        PROMPT_RESERVATION_COUNT,
        seed,
    )
    if first_audit.combined_rows != EXPECTED_SOURCE_ROWS or first_audit.rover_rows != EXPECTED_SOURCE_ROWS:
        raise SourceIntegrityError(
            "canonical sidecar row counts must both equal "
            f"{EXPECTED_SOURCE_ROWS}, got combined={first_audit.combined_rows}, ROVER={first_audit.rover_rows}"
        )
    if len(selected_ids) != PROMPT_RESERVATION_COUNT:
        raise SourceIntegrityError(f"expected exactly {PROMPT_RESERVATION_COUNT} reservable phase-2 prompt rows, got {len(selected_ids)}")

    plan_dir = paths.run_root / "split_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    counts, shard_sha256, selected_scores, text_exclusions = _write_split_rows(
        inventory, plan_dir, set(selected_ids), seed, count_text_tokens
    )
    if counts.null != EXPECTED_NULL_ROWS:
        raise SourceIntegrityError(f"expected exactly {EXPECTED_NULL_ROWS} null agreement rows, got {counts.null}")
    if counts.reserved != PROMPT_RESERVATION_COUNT:
        raise SourceIntegrityError(f"expected exactly {PROMPT_RESERVATION_COUNT} reserved rows, got {counts.reserved}")
    empty_text_rows = text_exclusions["text_token_length=0"]
    if empty_text_rows != EXPECTED_EMPTY_TEXT_ROWS:
        raise SourceIntegrityError(f"expected exactly {EXPECTED_EMPTY_TEXT_ROWS} empty-text rows, got {empty_text_rows}")
    over_limit_key = f"text_token_length>{TEXT_TOKEN_MAX_LENGTH}"
    over_limit_rows = text_exclusions[over_limit_key]
    if over_limit_rows != EXPECTED_OVER_LIMIT_ROWS:
        raise SourceIntegrityError(
            f"expected exactly {EXPECTED_OVER_LIMIT_ROWS} over-limit text rows, got {over_limit_rows}"
        )
    if counts.model_limit_exclusions != sum(text_exclusions.values()):
        raise SourceIntegrityError("model-limit exclusion count does not reconcile to the audited text-exclusion reasons")
    if counts.total != EXPECTED_SOURCE_ROWS:
        raise SourceIntegrityError(f"split total must equal {EXPECTED_SOURCE_ROWS}, got {counts.total}")
    if counts.phase1 + counts.phase2 + counts.null + counts.reserved + counts.model_limit_exclusions != EXPECTED_SOURCE_ROWS:
        raise SourceIntegrityError("phase/null/reserved/model-limit counts do not reconcile to the canonical source total")

    manifest_path = plan_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": SPLIT_PLAN_SCHEMA_VERSION,
            "source_schema_versions": {
                "combined_sidecar": COMBINED_SCHEMA_VERSION,
                "rover_archive": ROVER_SCHEMA_VERSION,
            },
            "seed": seed,
            "phase1": counts.phase1,
            "phase2": counts.phase2,
            "null": counts.null,
            "reserved": counts.reserved,
            "model_limit_exclusions": counts.model_limit_exclusions,
            "text_exclusions": text_exclusions,
            "total": counts.total,
            "source_inventory": {
                "source_archives": dict(inventory.source_archive_sha256),
                "combined_sidecar": str(inventory.combined_sidecar),
                "combined_sha256": inventory.combined_sha256,
                "rover_archive": str(inventory.rover_archive),
                "rover_sha256": inventory.rover_sha256,
            },
            "plan_shards": shard_sha256,
            "reserved_prompts": [
                {"source_relative_path": source_relative_path, "reservation_score": selected_scores[source_relative_path]}
                for source_relative_path in selected_ids
            ],
        },
    )
    return replace(counts, manifest_sha256=sha256_file(manifest_path))


def _reuse_published_split_plan(paths: RunPaths, inventory: SourceInventory, seed: int) -> SplitCounts | None:
    """Reuse a complete plan only after revalidating all provenance and shard hashes."""

    plan_dir = paths.run_root / "split_plan"
    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.is_file() or any(plan_dir.glob(".*.partial")) or any(plan_dir.glob("*.partial")):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    expected_inventory = {
        "source_archives": dict(inventory.source_archive_sha256),
        "combined_sidecar": str(inventory.combined_sidecar),
        "combined_sha256": inventory.combined_sha256,
        "rover_archive": str(inventory.rover_archive),
        "rover_sha256": inventory.rover_sha256,
    }
    if (
        manifest.get("schema_version") != SPLIT_PLAN_SCHEMA_VERSION
        or manifest.get("source_schema_versions")
        != {"combined_sidecar": COMBINED_SCHEMA_VERSION, "rover_archive": ROVER_SCHEMA_VERSION}
        or manifest.get("seed") != seed
        or manifest.get("source_inventory") != expected_inventory
    ):
        return None

    names = ("phase1", "phase2", "null", "reserved", "model_limit_exclusions", "total")
    values = {name: manifest.get(name) for name in names}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        return None
    text_exclusions = manifest.get("text_exclusions")
    expected_text_exclusions = {
        "text_token_length=0": EXPECTED_EMPTY_TEXT_ROWS,
        f"text_token_length>{TEXT_TOKEN_MAX_LENGTH}": EXPECTED_OVER_LIMIT_ROWS,
    }
    if text_exclusions != expected_text_exclusions:
        return None
    if (
        values["null"] != EXPECTED_NULL_ROWS
        or values["reserved"] != PROMPT_RESERVATION_COUNT
        or values["model_limit_exclusions"] != sum(expected_text_exclusions.values())
        or values["total"] != EXPECTED_SOURCE_ROWS
        or values["phase1"]
        + values["phase2"]
        + values["null"]
        + values["reserved"]
        + values["model_limit_exclusions"]
        != EXPECTED_SOURCE_ROWS
    ):
        return None

    reserved_prompts = manifest.get("reserved_prompts")
    if not isinstance(reserved_prompts, list) or len(reserved_prompts) != PROMPT_RESERVATION_COUNT:
        return None
    plan_shards = manifest.get("plan_shards")
    if not isinstance(plan_shards, dict) or not plan_shards:
        return None
    shard_paths = {path.name: path for path in plan_dir.glob("shard_*.jsonl") if path.is_file()}
    if set(shard_paths) != set(plan_shards):
        return None
    for name, expected_sha256 in plan_shards.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"shard_\d{6}\.jsonl", name) is None
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or sha256_file(shard_paths[name]) != expected_sha256
        ):
            return None
    return SplitCounts(
        phase1=values["phase1"],
        phase2=values["phase2"],
        null=values["null"],
        reserved=values["reserved"],
        model_limit_exclusions=values["model_limit_exclusions"],
        total=values["total"],
        plan_dir=plan_dir,
        manifest_sha256=sha256_file(manifest_path),
    )


def iter_split_rows(plan_dir: Path, shard: int) -> Iterator[JoinedRow]:
    """Yield one shard's published plan rows without loading other shards."""

    if shard < 0 or shard >= EXPECTED_SOURCE_TARS:
        raise ValueError(f"invalid split-plan shard: {shard}")
    path = plan_dir / f"shard_{shard:06d}.jsonl"
    if not path.is_file():
        raise SourceIntegrityError(f"missing split-plan shard: {path}")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            data = _json_mapping(line, path, number)
            yield JoinedRow(
                source_relative_path=_source_relative_path(data, path, number),
                text=_text_field(data, "text", path, number),
                instruct=_string_field(data, "instruct", path, number),
                agreement=_agreement(data, path, number),
                phase=_phase(data, path, number),
                reserved=_bool_field(data, "reserved", path, number),
                reservation_score=_optional_string_field(data, "reservation_score", path, number),
                model_limit_exclusion=_optional_string_field(data, "model_limit_exclusion", path, number),
                text_token_count=_optional_non_negative_int(data, "text_token_count", path, number),
            )


def _write_split_rows(
    inventory: SourceInventory,
    plan_dir: Path,
    reserved_ids: set[str],
    seed: int,
    count_text_tokens: Callable[[str], int],
) -> tuple[SplitCounts, dict[str, str], dict[str, str], dict[str, int]]:
    counts = SplitCounts(0, 0, 0, 0, 0, 0, plan_dir, "")
    shard_sha256: dict[str, str] = {}
    scores: dict[str, str] = {}
    text_exclusions = {
        "text_token_length=0": 0,
        f"text_token_length>{TEXT_TOKEN_MAX_LENGTH}": 0,
    }
    current_shard: int | None = None
    handle: TextIO | None = None
    partial: Path | None = None

    def close_current() -> None:
        nonlocal handle, partial
        if handle is None or partial is None or current_shard is None:
            return
        handle.flush()
        handle.close()
        completed = plan_dir / f"shard_{current_shard:06d}.jsonl"
        partial.replace(completed)
        shard_sha256[completed.name] = sha256_file(completed)
        handle = None
        partial = None

    try:
        for row in _preflight_text_limits(_iter_joined_rows(inventory, _JoinAudit()), count_text_tokens):
            shard = _shard_number(row.source_relative_path)
            if current_shard != shard:
                close_current()
                current_shard = shard
                partial = plan_dir / f".shard_{shard:06d}.jsonl.partial"
                if partial.exists():
                    partial.unlink()
                handle = partial.open("w", encoding="utf-8")
            is_model_excluded = _model_limit_excluded(row)
            score = reservation_score(row.source_relative_path, seed).hex() if row.source_relative_path in reserved_ids else None
            assigned = assign_phases([row], reserved_ids)[0]
            assigned = replace(assigned, reservation_score=score)
            if is_model_excluded:
                reason = assigned.model_limit_exclusion
                if reason not in text_exclusions:
                    raise SourceIntegrityError(f"unexpected model-limit exclusion reason: {reason}")
                text_exclusions[reason] += 1
                assigned = replace(assigned, phase=None, reserved=False, reservation_score=None)
                counts = replace(counts, model_limit_exclusions=counts.model_limit_exclusions + 1)
            elif assigned.reserved:
                scores[assigned.source_relative_path] = score or ""
                counts = replace(counts, reserved=counts.reserved + 1)
            elif assigned.phase == 1:
                counts = replace(counts, phase1=counts.phase1 + 1)
            elif assigned.phase == 2:
                counts = replace(counts, phase2=counts.phase2 + 1)
            else:
                counts = replace(counts, null=counts.null + 1)
            counts = replace(counts, total=counts.total + 1)
            assert handle is not None
            handle.write(json.dumps(_row_json(assigned), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        close_current()
    except Exception:
        if handle is not None:
            handle.close()
        if partial is not None and partial.exists():
            partial.unlink()
        raise
    return counts, shard_sha256, scores, text_exclusions


def _iter_joined_rows(inventory: SourceInventory, audit: _JoinAudit) -> Iterator[JoinedRow]:
    rover_rows = iter(_iter_rover_rows(inventory.rover_archive))
    rover = next(rover_rows, None)
    first_rover_shard = (
        _shard_number(_source_relative_path(rover, inventory.rover_archive, 0))
        if rover is not None
        else 0
    )
    reverse = first_rover_shard > 0
    for shard, combined in _iter_combined_groups(inventory.combined_sidecar, audit, reverse=reverse):
        if rover is not None:
            rover_shard = _shard_number(_source_relative_path(rover, inventory.rover_archive, 0))
            if (reverse and rover_shard > shard) or (not reverse and rover_shard < shard):
                raise SourceIntegrityError("combined and ROVER join keys differ")
        seen_rover: set[str] = set()
        while rover is not None and _shard_number(_source_relative_path(rover, inventory.rover_archive, 0)) == shard:
            audit.rover_rows += 1
            source_relative_path = _source_relative_path(rover, inventory.rover_archive, audit.rover_rows)
            if source_relative_path in seen_rover:
                raise SourceIntegrityError(f"duplicate ROVER source_relative_path: {source_relative_path}")
            seen_rover.add(source_relative_path)
            combined_row = combined.pop(source_relative_path, None)
            if combined_row is None:
                raise SourceIntegrityError("combined and ROVER join keys differ")
            text = _text_field(combined_row, "rover_punctuated_accented", inventory.combined_sidecar, audit.combined_rows)
            agreement = _agreement(rover, inventory.rover_archive, audit.rover_rows, canonical_rover=True)
            yield JoinedRow(source_relative_path=source_relative_path, text=text, agreement=agreement)
            rover = next(rover_rows, None)
        if combined:
            raise SourceIntegrityError("combined and ROVER join keys differ")
    if rover is not None:
        raise SourceIntegrityError("combined and ROVER join keys differ")


def _iter_combined_groups(
    path: Path,
    audit: _JoinAudit,
    *,
    reverse: bool = False,
) -> Iterator[tuple[int, dict[str, Mapping[str, object]]]]:
    current_shard: int | None = None
    rows: dict[str, Mapping[str, object]] = {}
    process: subprocess.Popen[str] | None = None
    handle: TextIO | None = None
    completed = False
    try:
        if reverse:
            process = subprocess.Popen(
                ["tac", "--", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
            if process.stdout is None:
                raise SourceIntegrityError("tac did not provide combined-sidecar stdout")
            handle = process.stdout
        else:
            handle = path.open(encoding="utf-8")
        for number, line in enumerate(handle, start=1):
            row = _json_mapping(line, path, number)
            _combined_schema(row, path, number)
            source_relative_path = _source_relative_path(row, path, number)
            shard = _shard_number(source_relative_path)
            if current_shard is not None and (
                (not reverse and shard < current_shard)
                or (reverse and shard > current_shard)
            ):
                direction = "descending" if reverse else "ascending"
                raise SourceIntegrityError(f"combined sidecar shards are not in {direction} order")
            if current_shard is not None and shard != current_shard:
                yield current_shard, rows
                rows = {}
            current_shard = shard
            audit.combined_rows += 1
            if source_relative_path in rows:
                raise SourceIntegrityError(f"duplicate combined source_relative_path: {source_relative_path}")
            rows[source_relative_path] = row
            if len(rows) > MAX_ROWS_PER_SHARD:
                raise SourceIntegrityError(f"combined shard {shard:06d} exceeds {MAX_ROWS_PER_SHARD} rows")
        if current_shard is not None:
            yield current_shard, rows
        completed = True
        if process is not None:
            handle.close()
            return_code = process.wait()
            if return_code != 0:
                raise SourceIntegrityError(f"tac failed while reading combined sidecar with exit code {return_code}")
    except (OSError, UnicodeDecodeError) as exc:
        raise SourceIntegrityError(f"cannot stream combined sidecar: {path}") from exc
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if process is not None and completed and process.returncode not in {0, None}:
            raise SourceIntegrityError(f"tac failed while reading combined sidecar with exit code {process.returncode}")


def _iter_rover_rows(path: Path) -> Iterator[Mapping[str, object]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                row = _json_mapping(line, path, number)
                _rover_schema(row, path, number)
                yield row
        return

    process: subprocess.Popen[bytes] | None = None
    archive: tarfile.TarFile | None = None
    try:
        process = subprocess.Popen(["unzstd", "-c", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if process.stdout is None:
            raise SourceIntegrityError("unzstd did not provide stdout")
        archive = tarfile.open(fileobj=process.stdout, mode="r|")
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SourceIntegrityError(f"cannot read ROVER archive member: {member.name}")
            for number, raw in enumerate(extracted, start=1):
                row = _json_mapping(raw.decode("utf-8"), path, number)
                _rover_schema(row, path, number)
                yield row
        archive.close()
        archive = None
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise SourceIntegrityError(f"unzstd failed while reading ROVER archive with exit code {return_code}")
    except (OSError, UnicodeDecodeError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise SourceIntegrityError(f"cannot stream ROVER archive: {path}") from exc
    finally:
        if archive is not None:
            archive.close()
        if process is not None:
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


def _combined_schema(row: Mapping[str, object], path: Path, number: int) -> None:
    if row.get("schema_version") != COMBINED_SCHEMA_VERSION:
        raise SourceIntegrityError(f"unexpected combined schema version in {path}:{number}")


def _rover_schema(row: Mapping[str, object], path: Path, number: int) -> None:
    if row.get("schema_version") != ROVER_SCHEMA_VERSION:
        raise SourceIntegrityError(f"unexpected ROVER schema version in {path}:{number}")


def _json_mapping(line: str, path: Path, number: int) -> Mapping[str, object]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SourceIntegrityError(f"invalid JSON in {path}:{number}") from exc
    if not isinstance(value, dict):
        raise SourceIntegrityError(f"expected JSON object in {path}:{number}")
    return value


def _source_relative_path(row: Mapping[str, object], path: Path, number: int) -> str:
    value = row.get("source_relative_path")
    match = _SOURCE_PATH.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise SourceIntegrityError(f"invalid source_relative_path in {path}:{number}")
    if int(match.group("shard")) >= EXPECTED_SOURCE_TARS:
        raise SourceIntegrityError(f"source_relative_path outside canonical tar range in {path}:{number}")
    return value


def _string_field(row: Mapping[str, object], field: str, path: Path, number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise SourceIntegrityError(f"missing non-empty {field} in {path}:{number}")
    return value


def _text_field(row: Mapping[str, object], field: str, path: Path, number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise SourceIntegrityError(f"missing string {field} in {path}:{number}")
    return value


def _optional_string_field(row: Mapping[str, object], field: str, path: Path, number: int) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SourceIntegrityError(f"invalid {field} in {path}:{number}")
    return value


def _agreement(row: Mapping[str, object], path: Path, number: int, *, canonical_rover: bool = False) -> float | None:
    if canonical_rover:
        if "asr_agreement_mean" not in row:
            raise SourceIntegrityError(f"missing asr_agreement_mean in {path}:{number}")
        value = row["asr_agreement_mean"]
    else:
        value = row.get("agreement")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise SourceIntegrityError(f"non-finite agreement in {path}:{number}")
    return float(value)


def _phase(row: Mapping[str, object], path: Path, number: int) -> int | None:
    value = row.get("phase")
    if value is None:
        return None
    if value not in (1, 2):
        raise SourceIntegrityError(f"invalid phase in {path}:{number}")
    return int(value)


def _bool_field(row: Mapping[str, object], field: str, path: Path, number: int) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise SourceIntegrityError(f"invalid {field} in {path}:{number}")
    return value


def _optional_non_negative_int(row: Mapping[str, object], field: str, path: Path, number: int) -> int | None:
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceIntegrityError(f"invalid {field} in {path}:{number}")
    return value


def _shard_number(source_relative_path: str) -> int:
    match = _SOURCE_PATH.fullmatch(source_relative_path)
    assert match is not None
    return int(match.group("shard"))


def _load_cosyvoice3_text_token_counter(paths: RunPaths) -> Callable[[str], int]:
    """Lazily load the authoritative CosyVoice3 Qwen tokenizer at plan build time."""

    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

    tokenizer = get_qwen_tokenizer(
        token_path=str(paths.base_model_dir / "CosyVoice-BlankEN"),
        skip_special_tokens=True,
        version="cosyvoice3",
    )

    def count_text_tokens(text: str) -> int:
        return len(tokenizer.encode(text, allowed_special="all"))

    return count_text_tokens


def _preflight_text_limits(rows: Iterable[JoinedRow], count_text_tokens: Callable[[str], int]) -> Iterator[JoinedRow]:
    """Annotate canonical rows excluded by the same text-token limit as CosyVoice3."""

    for row in rows:
        token_count = count_text_tokens(row.text)
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise SourceIntegrityError("CosyVoice3 text token counter returned an invalid count")
        exclusion = None
        if token_count == 0:
            exclusion = "text_token_length=0"
        elif token_count > TEXT_TOKEN_MAX_LENGTH:
            exclusion = f"text_token_length>{TEXT_TOKEN_MAX_LENGTH}"
        yield replace(row, text_token_count=token_count, model_limit_exclusion=exclusion)


def _model_limit_excluded(row: JoinedRow) -> bool:
    """Return whether tokenizer preflight intentionally removed this text row."""

    return row.model_limit_exclusion is not None


def _row_json(row: JoinedRow) -> dict[str, object]:
    return {
        "source_relative_path": row.source_relative_path,
        "text": row.text,
        "instruct": row.instruct,
        "agreement": row.agreement,
        "phase": row.phase,
        "reserved": row.reserved,
        "reservation_score": row.reservation_score,
        "model_limit_exclusion": row.model_limit_exclusion,
        "text_token_count": row.text_token_count,
    }

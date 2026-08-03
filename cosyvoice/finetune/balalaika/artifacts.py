"""Checksum-bound atomic artifacts for restart-safe Balalaika stages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping


class StageRequirementError(RuntimeError):
    """Raised when a required stage is absent or no longer provenance-safe."""


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum without loading a potentially large file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _checksum(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Durably publish JSON using a same-directory temporary file and rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(_canonical_bytes(value))
            temporary.write(b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class StageRecord:
    """A checked stage manifest and its computed completed-file checksum."""

    name: str
    path: Path
    payload: Mapping[str, object]
    dependency_lock: Mapping[str, object]
    input_provenance: Mapping[str, object]
    manifest_sha256: str


class StageStore:
    """Publish and require stage manifests bound to code and input identity."""

    def __init__(
        self,
        root: Path,
        dependency_lock: Mapping[str, object] | None = None,
        input_provenance: Mapping[str, object] | None = None,
    ) -> None:
        self.root = root
        self.dependency_lock = _immutable_mapping(dependency_lock or {})
        self.input_provenance = _immutable_mapping(input_provenance or {})

    def publish(self, name: str, payload: Mapping[str, object]) -> StageRecord:
        """Atomically publish a stage bound to the current locks and provenance."""

        path = self._path(name)
        payload_copy = _immutable_mapping(payload)
        manifest = {
            "name": name,
            "payload": payload_copy,
            "payload_sha256": _checksum(payload_copy),
            "dependency_lock": self.dependency_lock,
            "dependency_lock_sha256": _checksum(self.dependency_lock),
            "input_provenance": self.input_provenance,
            "input_provenance_sha256": _checksum(self.input_provenance),
        }
        atomic_write_json(path, manifest)
        return self.require(name)

    def require(self, name: str) -> StageRecord:
        """Load a completed stage only if its payload and provenance still match."""

        path = self._path(name)
        if not path.is_file():
            raise StageRequirementError(f"missing required stage: {name}")
        try:
            with path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise StageRequirementError(f"invalid stage manifest: {name}") from exc
        if not isinstance(manifest, dict) or manifest.get("name") != name:
            raise StageRequirementError(f"invalid stage manifest: {name}")
        payload = _mapping_field(manifest, "payload", name)
        dependency_lock = _mapping_field(manifest, "dependency_lock", name)
        input_provenance = _mapping_field(manifest, "input_provenance", name)
        if manifest.get("payload_sha256") != _checksum(payload):
            raise StageRequirementError(f"stage payload checksum changed: {name}")
        if manifest.get("dependency_lock_sha256") != _checksum(dependency_lock) or dependency_lock != self.dependency_lock:
            raise StageRequirementError(f"stage dependency lock changed: {name}")
        if manifest.get("input_provenance_sha256") != _checksum(input_provenance) or input_provenance != self.input_provenance:
            raise StageRequirementError(f"stage input provenance changed: {name}")
        return StageRecord(
            name=name,
            path=path,
            payload=payload,
            dependency_lock=dependency_lock,
            input_provenance=input_provenance,
            manifest_sha256=sha256_file(path),
        )

    def _path(self, name: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError(f"invalid stage name: {name!r}")
        return self.root / f"{name}.json"


def _immutable_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _mapping_field(manifest: Mapping[str, object], field: str, name: str) -> dict[str, object]:
    value = manifest.get(field)
    if not isinstance(value, dict):
        raise StageRequirementError(f"invalid {field} in stage: {name}")
    return value

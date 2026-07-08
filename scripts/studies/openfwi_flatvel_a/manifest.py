"""Shard manifest helpers for the OpenFWI FlatVel-A smoke gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_SHARDS = ("data1.npy", "model1.npy", "data49.npy", "model49.npy")


class OpenFWIManifestBlocker(RuntimeError):
    """Controlled blocker for data-access and manifest failures."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        missing: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.missing = missing or []
        self.details = details or {}

    def to_payload(self, *, run_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reason": self.reason,
            "message": str(self),
            "missing": list(self.missing),
            "details": self.details,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def file_identity(path: Path, *, sha256: bool = True) -> dict[str, Any]:
    """Return stable file identity fields without loading array contents."""
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload["sha256"] = digest.hexdigest()
    return payload


def resolve_required_shards(data_root: Path) -> dict[str, Path]:
    """Resolve the four pinned FlatVel-A smoke shards."""
    data_root = Path(data_root).expanduser().resolve()
    missing = [name for name in REQUIRED_SHARDS if not (data_root / name).is_file()]
    if missing:
        raise OpenFWIManifestBlocker(
            "missing_required_shards",
            f"missing required OpenFWI FlatVel-A smoke shards: {missing}",
            missing=missing,
            details={"data_root": str(data_root), "required_shards": list(REQUIRED_SHARDS)},
        )
    return {name: data_root / name for name in REQUIRED_SHARDS}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_data_root_policy(data_root: Path, *, repo_root: Path) -> None:
    """Require data to be outside git unless under the ignored artifact root."""
    data_root = Path(data_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    if not _is_relative_to(data_root, repo_root):
        return
    relative = data_root.relative_to(repo_root)
    if relative.parts and relative.parts[0] == ".artifacts":
        return
    raise OpenFWIManifestBlocker(
        "data_root_inside_repo",
        f"OpenFWI data root must be outside git or under ignored .artifacts/: {data_root}",
        details={"data_root": str(data_root), "repo_root": str(repo_root)},
    )


def _split_role(filename: str) -> str:
    return "train" if filename in {"data1.npy", "model1.npy"} else "validation_test"


def build_data_manifest(
    *,
    data_root: Path,
    shards: dict[str, Path],
    source_url: str,
    license_note: str,
    access_note: str,
    run_id: str,
    redistribution_policy: str = "referenced_only_not_redistributed",
) -> dict[str, Any]:
    """Build a manifest for the exact FlatVel-A smoke shard files."""
    data_root = Path(data_root).expanduser().resolve()
    shard_payload = []
    for name in REQUIRED_SHARDS:
        identity = file_identity(shards[name])
        identity["split_role"] = _split_role(name)
        identity["array_role"] = "seismic_data" if name.startswith("data") else "velocity_model"
        shard_payload.append(identity)
    return {
        "schema_version": "openfwi_flatvel_a_data_manifest_v1",
        "run_id": run_id,
        "dataset_family": "OpenFWI",
        "dataset_variant": "FlatVel-A",
        "source_url": source_url,
        "license_note": license_note,
        "access_note": access_note,
        "local_data_root": str(data_root),
        "redistribution_policy": redistribution_policy,
        "required_shards": list(REQUIRED_SHARDS),
        "shards": shard_payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

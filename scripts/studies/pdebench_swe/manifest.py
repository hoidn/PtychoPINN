"""HDF5 and source manifest helpers for the PDEBench SWE smoke gate."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class ManifestBlocker(RuntimeError):
    """Controlled blocker for data-contract failures before training."""

    def __init__(self, reason: str, message: str, *, candidate_datasets: list[str] | None = None):
        super().__init__(message)
        self.reason = reason
        self.candidate_datasets = candidate_datasets or []

    def to_payload(self, *, run_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reason": self.reason,
            "message": str(self),
            "candidate_datasets": self.candidate_datasets,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def file_identity(path: Path, *, sha256: bool = True) -> dict[str, Any]:
    """Return file identity fields used in dataset manifests."""
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


def inspect_hdf5(path: Path) -> dict[str, Any]:
    """Recursively inspect HDF5 groups and datasets without loading full arrays."""
    path = Path(path).expanduser().resolve()
    datasets: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            attrs = {str(key): _jsonable(value) for key, value in obj.attrs.items()}
            if isinstance(obj, h5py.Dataset):
                datasets.append(
                    {
                        "path": name,
                        "shape": [int(dim) for dim in obj.shape],
                        "ndim": int(obj.ndim),
                        "dtype": str(obj.dtype),
                        "attrs": attrs,
                    }
                )
            elif isinstance(obj, h5py.Group):
                groups.append({"path": name, "attrs": attrs})

        handle.visititems(visitor)

    return {
        "file": str(path),
        "groups": groups,
        "datasets": datasets,
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _is_numeric_dataset(record: dict[str, Any]) -> bool:
    try:
        dtype = np.dtype(record["dtype"])
    except TypeError:
        return False
    return bool(np.issubdtype(dtype, np.number))


def _candidate_score(record: dict[str, Any]) -> int:
    name = Path(str(record["path"])).name.lower()
    score = 0
    if name == "data":
        score += 10
    if "state" in name:
        score += 5
    if "u" == name:
        score += 4
    if int(record.get("ndim", 0)) == 5:
        score += 2
    return score


def infer_axis_order(shape: list[int]) -> str | None:
    """Infer common PDEBench state layouts conservatively."""
    if len(shape) == 5:
        if shape[-1] <= 16:
            return "NTHWC"
        if shape[2] <= 16:
            return "NTCHW"
    if len(shape) == 4:
        return "NTHW"
    return None


def _infer_trajectory_axis_order(shape: list[int]) -> str | None:
    if len(shape) == 4 and shape[-1] <= 16:
        return "THWC"
    if len(shape) == 3:
        return "THW"
    return infer_axis_order(shape)


def _select_repeated_trajectory_dataset(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for record in candidates:
        match = re.fullmatch(r"(\d+)/(.+)", str(record["path"]))
        if not match:
            continue
        trajectory_id = int(match.group(1))
        suffix = match.group(2)
        grouped.setdefault(suffix, []).append((trajectory_id, record))
    for suffix, records in grouped.items():
        if suffix != "data" or len(records) < 2:
            continue
        records = sorted(records, key=lambda item: item[0])
        ids = [item[0] for item in records]
        if ids != list(range(ids[0], ids[0] + len(ids))):
            continue
        shapes = {tuple(item[1]["shape"]) for item in records}
        dtypes = {str(item[1]["dtype"]) for item in records}
        if len(shapes) != 1 or len(dtypes) != 1:
            continue
        child_shape = [int(dim) for dim in records[0][1]["shape"]]
        child_axis_order = _infer_trajectory_axis_order(child_shape)
        if child_axis_order is None:
            continue
        return {
            "path": f"*/{suffix}",
            "path_pattern": f"{{trajectory_id:04d}}/{suffix}",
            "trajectory_count": len(records),
            "trajectory_ids": ids,
            "trajectory_shape": child_shape,
            "shape": [len(records), *child_shape],
            "ndim": len(child_shape) + 1,
            "dtype": str(records[0][1]["dtype"]),
            "axis_order": f"N{child_axis_order}",
            "attrs": {},
        }
    return None


def select_state_dataset(metadata: dict[str, Any], requested: str | None = None) -> dict[str, Any]:
    """Select the state tensor dataset or raise a controlled ambiguity blocker."""
    datasets = list(metadata.get("datasets", []))
    candidates = [
        record
        for record in datasets
        if int(record.get("ndim", 0)) >= 4 and _is_numeric_dataset(record)
    ]
    if requested:
        normalized = requested.strip("/")
        if normalized in {"*/data", "* /data"}:
            repeated = _select_repeated_trajectory_dataset(candidates)
            if repeated is not None:
                return repeated
        for record in datasets:
            if str(record["path"]).strip("/") == normalized:
                selected = dict(record)
                selected["axis_order"] = infer_axis_order(selected["shape"])
                return selected
        raise ManifestBlocker(
            "missing_state_dataset",
            f"requested state dataset not found: {requested}",
            candidate_datasets=[str(item["path"]) for item in datasets],
        )

    if not candidates:
        raise ManifestBlocker(
            "missing_state_dataset",
            "no numeric HDF5 dataset with four or more dimensions was found",
            candidate_datasets=[str(item["path"]) for item in datasets],
        )
    repeated = _select_repeated_trajectory_dataset(candidates)
    if repeated is not None:
        return repeated

    scored = sorted(candidates, key=_candidate_score, reverse=True)
    top_score = _candidate_score(scored[0])
    top = [record for record in scored if _candidate_score(record) == top_score]
    if len(top) != 1:
        raise ManifestBlocker(
            "ambiguous_state_dataset",
            "multiple candidate state datasets matched; pass --state-dataset",
            candidate_datasets=[str(item["path"]) for item in candidates],
        )
    selected = dict(top[0])
    selected["axis_order"] = infer_axis_order(selected["shape"])
    if selected["axis_order"] is None:
        raise ManifestBlocker(
            "ambiguous_axis_order",
            "selected state dataset needs an explicit --axis-order",
            candidate_datasets=[str(selected["path"])],
        )
    return selected


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_dataset_manifests(
    *,
    data_file: Path,
    output_root: Path,
    dataset_source: str = "",
    dataset_source_url: str = "",
    dataset_darus_id: str | int | None = None,
    license_note: str = "",
    license_note_file: Path | None = None,
    state_dataset: str | None = None,
    axis_order: str | None = None,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Write root dataset and HDF5 metadata manifests."""
    output_root = Path(output_root)
    identity = file_identity(data_file)
    metadata = inspect_hdf5(data_file)
    selected = select_state_dataset(metadata, requested=state_dataset)
    if axis_order:
        selected["axis_order"] = axis_order
    license_note_file_str = str(license_note_file) if license_note_file is not None else None
    license_note_excerpt = None
    if license_note_file is not None and Path(license_note_file).exists():
        license_text = Path(license_note_file).read_text(encoding="utf-8")
        license_note_excerpt = license_text.strip().splitlines()[0] if license_text.strip() else ""

    common = {"run_id": run_id} if run_id else {}
    dataset_manifest = {
        **common,
        "schema_version": "pdebench_swe_dataset_manifest_v1",
        "source": {
            "name": dataset_source,
            "url": dataset_source_url,
            "darus_id": str(dataset_darus_id) if dataset_darus_id is not None else None,
            "license_note": license_note,
            "license_note_file": license_note_file_str,
            "license_note_excerpt": license_note_excerpt,
        },
        "file_identity": identity,
        "selected_state_dataset": selected,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    hdf5_payload = {
        **common,
        "schema_version": "pdebench_swe_hdf5_metadata_v1",
        **metadata,
        "selected_state_dataset": selected,
    }
    return (
        write_json(output_root / "dataset_manifest.json", dataset_manifest),
        write_json(output_root / "hdf5_metadata.json", hdf5_payload),
    )

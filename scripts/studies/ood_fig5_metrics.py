"""Compute Fig. 5 ID/OOD metrics from pinned reconstruction artifacts.

This is a bounded revision-study script. It intentionally treats the Fig. 5
NPZ files as study-local inputs rather than introducing a new general data
contract.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptycho.image.cropping import align_for_evaluation
from ptycho.image.registration import apply_shift_and_crop, find_translation_offset
import ptycho.params as legacy_params

APPROVED_DESIGN_PATH = (
    ".artifacts/revision_studies/"
    "fig5_ood_metrics_low_frequency_phase_20260413T071529Z/approved_design.md"
)
PLAN_PATH = (
    ".artifacts/revision_studies/"
    "fig5_ood_metrics_low_frequency_phase_20260413T071529Z/implementation_plan.md"
)
DEFAULT_PAPER_ROOT = "/home/ollie/Documents/ptychopinnpaper2"
DEFAULT_COORDINATE_SOURCE = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
DEFAULT_FULL_REFERENCE = "experiment_outputs/ground_truth/run1084_ground_truth.npz"
REFERENCE_MAX_ABS_RTOL = 1e-10
REFERENCE_RELATIVE_L2_TOL = 1e-12
MIN_POST_EVAL_TRIM_DIM = 56
EVALUATION_REGION_ALL_SCAN = "all_scan"
EVALUATION_REGION_HELDOUT = "heldout_test_half"
HELDOUT_SPLIT_FROM_INDICES = "from_indices"
HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y = "bottom_half_by_sorted_y"
HELDOUT_SPLIT_TOP_BY_SORTED_Y = "top_half_by_sorted_y"
HELDOUT_SPLIT_PAPER_TOP_TRAIN_BOTTOM_TEST_BY_HIGH_Y = "paper_top_train_bottom_test_by_high_y"
SPLIT_NONOVERLAP_COORDINATE_INDICES = "coordinate_indices"
SPLIT_NONOVERLAP_OBJECT_FOOTPRINT = "object_footprint"
PAPER_RUN1084_HISTORICAL_TRAIN_COUNT = 512

PAPER_FIG5_PNG_SOURCES = {
    "pinn_fly64trained_run1084test_amplitude.png": "experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/pinn_fly64trained_run1084test_amplitude.png",
    "pinn_fly64trained_run1084test_phase.png": "experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/pinn_fly64trained_run1084test_phase.png",
    "pinn_fly64trained_run1084test_composite.png": "experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/pinn_fly64trained_run1084test_composite.png",
    "baseline_fly64trained_run1084test_amplitude.png": "experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_fly64trained_run1084test_amplitude.png",
    "baseline_fly64trained_run1084test_phase.png": "experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_fly64trained_run1084test_phase.png",
    "baseline_fly64trained_run1084test_composite.png": "experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_fly64trained_run1084test_composite.png",
    "pinn_run1084_amplitude.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/pinn_run1084_amplitude.png",
    "pinn_run1084_phase.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/pinn_run1084_phase.png",
    "pinn_run1084_composite.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/pinn_run1084_composite.png",
    "baseline_run1084_amplitude.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_run1084_amplitude.png",
    "baseline_run1084_phase.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_run1084_phase.png",
    "baseline_run1084_composite.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_run1084_composite.png",
    "ground_truth_run1084_amplitude.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/ground_truth_run1084_amplitude.png",
    "ground_truth_run1084_phase.png": "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/ground_truth_run1084_phase.png",
    "fly64_probe_amplitude.png": "experiment_outputs/probe_visualizations/fly64_probe_amplitude.png",
    "fly64_probe_phase.png": "experiment_outputs/probe_visualizations/fly64_probe_phase.png",
    "run1084_probe_amplitude.png": "experiment_outputs/probe_visualizations/run1084_probe_amplitude.png",
    "run1084_probe_phase.png": "experiment_outputs/probe_visualizations/run1084_probe_phase.png",
}


@dataclass(frozen=True)
class StudyRow:
    row_id: str
    condition: str
    model: str
    reconstruction_path: Path
    panel_reference_path: Path | None
    notes: str = ""


class OutputLockError(RuntimeError):
    """Raised when a study output root is already locked."""


class StopCondition(RuntimeError):
    """Raised when a scientific stop/pivot gate blocks paper-facing claims."""


@dataclass
class OutputLock:
    output_root: Path
    path: Path
    pid: int
    replaced_stale_lock: bool = False
    stale_lock_content: str | None = None
    _released: bool = False

    def __enter__(self) -> "OutputLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        release_output_lock(self)


def build_default_rows(repo_root: Path) -> list[StudyRow]:
    root = Path(repo_root)
    return [
        StudyRow(
            row_id="id_ptychopinn",
            condition="ID",
            model="PtychoPINN",
            reconstruction_path=root
            / "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/reconstruction.npz",
            panel_reference_path=root
            / "experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/ground_truth_run1084.npz",
        ),
        StudyRow(
            row_id="id_supervised_baseline",
            condition="ID",
            model="Supervised baseline",
            reconstruction_path=root
            / "experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_reconstruction.npz",
            panel_reference_path=None,
            notes="No baseline-specific ID panel reference NPZ found; validate against ID PtychoPINN panel reference where possible.",
        ),
        StudyRow(
            row_id="ood_ptychopinn",
            condition="OOD",
            model="PtychoPINN",
            reconstruction_path=root
            / "experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/reconstruction.npz",
            panel_reference_path=root
            / "experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/ground_truth_run1084_for_fly64trained.npz",
        ),
        StudyRow(
            row_id="ood_supervised_baseline",
            condition="OOD",
            model="Supervised baseline",
            reconstruction_path=root
            / "experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_reconstruction.npz",
            panel_reference_path=root
            / "experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/ground_truth_run1084_for_fly64trained.npz",
        ),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_npz(path: Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    arrays: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        for key in keys:
            arr = data[key]
            arrays[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "ndim": int(arr.ndim),
            }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
        "keys": keys,
        "arrays": arrays,
    }


def _require_finite_nonempty(array: np.ndarray, *, source: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        raise ValueError(f"{source} is zero-size")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{source} contains non-finite values")
    return arr


def squeeze_to_2d_complex(array: np.ndarray, source: str) -> np.ndarray:
    arr = _require_finite_nonempty(np.asarray(array), source=source)
    if arr.ndim == 2:
        return arr.astype(np.complex64 if arr.dtype == np.complex64 else np.complex128, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        squeezed = arr[:, :, 0]
        return squeezed.astype(
            np.complex64 if squeezed.dtype == np.complex64 else np.complex128,
            copy=False,
        )
    raise ValueError(f"{source} must reduce to one 2D complex reconstruction, got shape {arr.shape}")


def load_complex_array(
    path: Path,
    *,
    complex_keys: tuple[str, ...],
    amp_key: str,
    phase_key: str,
) -> np.ndarray:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        for key in complex_keys:
            if key in data:
                return squeeze_to_2d_complex(data[key], f"{path}:{key}")
        if amp_key not in data or phase_key not in data:
            missing = [key for key in (*complex_keys, amp_key, phase_key) if key not in data]
            raise KeyError(f"{path} is missing required complex or amplitude/phase keys: {missing}")
        amp = _require_finite_nonempty(np.asarray(data[amp_key]), source=f"{path}:{amp_key}")
        phase = _require_finite_nonempty(np.asarray(data[phase_key]), source=f"{path}:{phase_key}")
        if amp.shape != phase.shape:
            raise ValueError(f"{path} amplitude/phase shapes differ: {amp.shape} vs {phase.shape}")
        return squeeze_to_2d_complex(amp * np.exp(1j * phase), f"{path}:{amp_key}+{phase_key}")


def load_scan_coords_yx(coordinate_source: Path) -> np.ndarray:
    coordinate_source = Path(coordinate_source)
    with np.load(coordinate_source, allow_pickle=True) as data:
        if "xcoords" not in data or "ycoords" not in data:
            raise KeyError(f"{coordinate_source} must contain xcoords and ycoords")
        xcoords = _require_finite_nonempty(np.asarray(data["xcoords"]), source=f"{coordinate_source}:xcoords")
        ycoords = _require_finite_nonempty(np.asarray(data["ycoords"]), source=f"{coordinate_source}:ycoords")
    if xcoords.shape != ycoords.shape:
        raise ValueError(f"xcoords and ycoords must have matching shapes, got {xcoords.shape} vs {ycoords.shape}")
    return np.stack([ycoords.reshape(-1), xcoords.reshape(-1)], axis=1)


def _load_reference_from_path(path: Path, *, allow_object_guess: bool) -> tuple[np.ndarray, dict[str, Any]]:
    complex_keys = ("ground_truth_complex",)
    if allow_object_guess:
        complex_keys = (*complex_keys, "objectGuess")
    reference = load_complex_array(
        path,
        complex_keys=complex_keys,
        amp_key="ground_truth_amplitude",
        phase_key="ground_truth_phase",
    )
    source_key = next(_first_existing_key(path, complex_keys, "ground_truth_amplitude"))
    return reference, {"source_path": str(path), "source_key": source_key}


def _first_existing_key(path: Path, keys: Iterable[str], fallback_key: str) -> Iterable[str]:
    with np.load(path, allow_pickle=True) as data:
        for key in keys:
            if key in data:
                yield key
                return
        if fallback_key in data:
            yield fallback_key
            return
    yield fallback_key


def load_full_reference(primary_reference: Path, coordinate_source: Path) -> tuple[np.ndarray, dict[str, Any]]:
    primary_reference = Path(primary_reference)
    coordinate_source = Path(coordinate_source)
    if primary_reference.exists():
        return _load_reference_from_path(primary_reference, allow_object_guess=True)
    return _load_reference_from_path(coordinate_source, allow_object_guess=True)


def load_panel_reference(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    return load_complex_array(
        Path(path),
        complex_keys=("ground_truth_complex",),
        amp_key="ground_truth_amplitude",
        phase_key="ground_truth_phase",
    )


def _shape(array: np.ndarray) -> list[int]:
    return [int(dim) for dim in np.asarray(array).shape]


def _coordinate_ranges(scan_coords_yx: np.ndarray) -> dict[str, list[float]]:
    coords = np.asarray(scan_coords_yx, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"scan_coords_yx must have shape (n_positions, 2), got {coords.shape}")
    return {
        "y": [float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))],
        "x": [float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))],
    }


def coordinate_bbox_rows_cols(
    scan_coords_yx: np.ndarray,
    stitch_patch_size: int,
    object_shape: tuple[int, int] | list[int],
) -> list[int]:
    coords = np.asarray(scan_coords_yx, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"scan_coords_yx must have non-empty shape (n_positions, 2), got {coords.shape}")
    effective_radius = int(stitch_patch_size) // 2
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    object_h, object_w = int(object_shape[0]), int(object_shape[1])
    return [
        max(0, int(min_y) - effective_radius),
        min(object_h, int(max_y) + effective_radius),
        max(0, int(min_x) - effective_radius),
        min(object_w, int(max_x) + effective_radius),
    ]


def _aligned_gt_bbox_rows_cols(
    *,
    all_scan_bbox_rows_cols: list[int],
    aligned_gt_shape: tuple[int, int] | list[int],
) -> list[int]:
    start_row, end_row, start_col, end_col = [int(value) for value in all_scan_bbox_rows_cols]
    bbox_h = end_row - start_row
    bbox_w = end_col - start_col
    aligned_h, aligned_w = int(aligned_gt_shape[0]), int(aligned_gt_shape[1])
    if aligned_h <= 0 or aligned_w <= 0:
        raise ValueError(f"aligned ground truth shape must be positive, got {aligned_gt_shape}")
    if aligned_h > bbox_h or aligned_w > bbox_w:
        raise ValueError(
            f"aligned ground truth shape {aligned_gt_shape} exceeds coordinate bbox shape {(bbox_h, bbox_w)}"
        )
    row_offset = (bbox_h - aligned_h) // 2
    col_offset = (bbox_w - aligned_w) // 2
    return [
        start_row + row_offset,
        start_row + row_offset + aligned_h,
        start_col + col_offset,
        start_col + col_offset + aligned_w,
    ]


def _intersect_bbox_rows_cols(a: list[int], b: list[int]) -> list[int]:
    row0 = max(int(a[0]), int(b[0]))
    row1 = min(int(a[1]), int(b[1]))
    col0 = max(int(a[2]), int(b[2]))
    col1 = min(int(a[3]), int(b[3]))
    if row1 <= row0 or col1 <= col0:
        raise ValueError(f"held-out bbox {b} does not intersect aligned full bbox {a}")
    return [row0, row1, col0, col1]


def _relative_slice_rows_cols(*, child_bbox_rows_cols: list[int], parent_bbox_rows_cols: list[int]) -> list[int]:
    return [
        int(child_bbox_rows_cols[0]) - int(parent_bbox_rows_cols[0]),
        int(child_bbox_rows_cols[1]) - int(parent_bbox_rows_cols[0]),
        int(child_bbox_rows_cols[2]) - int(parent_bbox_rows_cols[2]),
        int(child_bbox_rows_cols[3]) - int(parent_bbox_rows_cols[2]),
    ]


def _crop_relative(array: np.ndarray, relative_slice_rows_cols: list[int]) -> np.ndarray:
    row0, row1, col0, col1 = [int(value) for value in relative_slice_rows_cols]
    arr = np.asarray(array)
    if row0 < 0 or col0 < 0 or row1 > arr.shape[0] or col1 > arr.shape[1] or row1 <= row0 or col1 <= col0:
        raise ValueError(f"relative crop {relative_slice_rows_cols} is invalid for shape {arr.shape}")
    return arr[row0:row1, col0:col1]


def _load_holdout_indices_from_source(split_source: Path) -> dict[str, list[int]]:
    source = Path(split_source)
    if not source.exists():
        raise FileNotFoundError(f"held-out split source does not exist: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text())
        eval_indices = payload.get("eval_indices", payload.get("test_indices"))
        train_indices = payload.get("train_indices")
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=True) as data:
            eval_key = "eval_indices" if "eval_indices" in data else "test_indices"
            if eval_key not in data:
                raise KeyError(f"{source} must contain eval_indices or test_indices")
            eval_indices = data[eval_key]
            train_indices = data["train_indices"] if "train_indices" in data else None
    else:
        raise ValueError(f"Unsupported held-out split source suffix for {source}")
    if eval_indices is None:
        raise KeyError(f"{source} must contain eval_indices or test_indices")
    return {
        "eval_indices": [int(value) for value in np.asarray(eval_indices).reshape(-1)],
        "train_indices": [int(value) for value in np.asarray(train_indices).reshape(-1)] if train_indices is not None else [],
    }


def select_spatial_holdout_split(
    scan_coords_yx: np.ndarray,
    *,
    policy: str,
    split_source: Path | None,
) -> dict[str, Any]:
    coords = np.asarray(scan_coords_yx, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"scan_coords_yx must have non-empty shape (n_positions, 2), got {coords.shape}")
    n_coords = int(coords.shape[0])

    if policy == HELDOUT_SPLIT_FROM_INDICES:
        if split_source is None:
            raise ValueError("heldout split policy from_indices requires --heldout-split-source")
        loaded = _load_holdout_indices_from_source(split_source)
        eval_indices = loaded["eval_indices"]
        train_indices = loaded["train_indices"] or [idx for idx in range(n_coords) if idx not in set(eval_indices)]
        train_half = "source_train_indices"
        eval_half = "source_eval_indices"
        axis_direction_evidence = f"explicit split source {split_source}"
    elif policy in {
        HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y,
        HELDOUT_SPLIT_TOP_BY_SORTED_Y,
        HELDOUT_SPLIT_PAPER_TOP_TRAIN_BOTTOM_TEST_BY_HIGH_Y,
    }:
        order = [int(value) for value in np.argsort(coords[:, 0], kind="mergesort")]
        midpoint = n_coords // 2
        top_indices = order[:midpoint]
        bottom_indices = order[midpoint:]
        if policy == HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y:
            train_indices = top_indices
            eval_indices = bottom_indices
            train_half = "top"
            eval_half = "bottom"
            axis_direction_evidence = "paper states top half training and bottom half testing; sorted y coordinate fallback"
        elif policy == HELDOUT_SPLIT_TOP_BY_SORTED_Y:
            train_indices = bottom_indices
            eval_indices = top_indices
            train_half = "bottom"
            eval_half = "top"
            axis_direction_evidence = "explicit top-half evaluation fallback by sorted y coordinate"
        else:
            train_indices = bottom_indices
            eval_indices = top_indices
            train_half = "paper_top_high_y"
            eval_half = "paper_bottom_low_y"
            axis_direction_evidence = (
                "paper states top half training and bottom half testing; local split scripts define "
                "top-half training as ycoords >= threshold, so held-out bottom/test uses the lower-y sorted half"
            )
    else:
        raise ValueError(f"Unsupported heldout split policy: {policy}")

    for idx in eval_indices + train_indices:
        if idx < 0 or idx >= n_coords:
            raise ValueError(f"held-out split index {idx} outside coordinate range 0..{n_coords - 1}")
    eval_coords = coords[np.asarray(eval_indices, dtype=int)]
    train_coords = coords[np.asarray(train_indices, dtype=int)] if train_indices else np.empty((0, 2), dtype=float)
    return {
        "heldout_split_policy": policy,
        "heldout_split_source": str(split_source) if split_source is not None else None,
        "axis": "y",
        "axis_direction_evidence": axis_direction_evidence,
        "train_half": train_half,
        "eval_half": eval_half,
        "train_indices": [int(value) for value in train_indices],
        "eval_indices": [int(value) for value in eval_indices],
        "total_coordinate_count": n_coords,
        "train_coordinate_count": int(len(train_indices)),
        "eval_coordinate_count": int(len(eval_indices)),
        "train_coordinate_ranges_yx": _coordinate_ranges(train_coords) if len(train_coords) else None,
        "eval_coordinate_ranges_yx": _coordinate_ranges(eval_coords),
    }


def _normalize_indices(indices: Iterable[int], *, n_coords: int, label: str) -> np.ndarray:
    values = np.asarray([int(value) for value in indices], dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise StopCondition(f"{label} split must contain at least one coordinate index")
    if np.any(values < 0) or np.any(values >= int(n_coords)):
        raise StopCondition(f"{label} split contains coordinate indices outside 0..{int(n_coords) - 1}")
    if np.unique(values).size != values.size:
        raise StopCondition(f"{label} split contains duplicate coordinate indices")
    return values


def _filter_eval_indices_by_train_footprint(
    scan_coords_yx: np.ndarray,
    *,
    train_indices: Iterable[int],
    eval_indices: Iterable[int],
    stitch_patch_size: int,
    full_reference_shape: tuple[int, int] | list[int],
    heldout_guard_band_pixels: int,
) -> tuple[list[int], dict[str, Any]]:
    coords = np.asarray(scan_coords_yx, dtype=float)
    train = _normalize_indices(train_indices, n_coords=coords.shape[0], label="train")
    eval_ = _normalize_indices(eval_indices, n_coords=coords.shape[0], label="eval")
    guard = max(0, int(heldout_guard_band_pixels))
    train_exclusion_mask = coordinate_footprint_mask(
        coords[train],
        stitch_patch_size=int(stitch_patch_size) + 2 * guard,
        full_reference_shape=full_reference_shape,
    )
    kept: list[int] = []
    dropped: list[int] = []
    for idx in eval_:
        eval_mask = coordinate_footprint_mask(
            coords[[int(idx)]],
            stitch_patch_size=stitch_patch_size,
            full_reference_shape=train_exclusion_mask.shape,
        )
        if bool(np.logical_and(train_exclusion_mask, eval_mask).any()):
            dropped.append(int(idx))
        else:
            kept.append(int(idx))
    if not kept:
        raise StopCondition("split contract failed: guard band dropped all held-out evaluation coordinates")
    return kept, {
        "guard_band_eval_original_coordinate_count": int(eval_.size),
        "guard_band_eval_coordinate_count_after_drop": int(len(kept)),
        "guard_band_eval_dropped_count": int(len(dropped)),
        "guard_band_eval_dropped_indices": dropped,
        "guard_band_exclusion_stitch_patch_size": int(stitch_patch_size) + 2 * guard,
    }


def apply_eval_guard_band_to_split(
    heldout_split: dict[str, Any],
    *,
    scan_coords_yx: np.ndarray,
    train_indices: Iterable[int],
    stitch_patch_size: int,
    full_reference_shape: tuple[int, int] | list[int],
    heldout_guard_band_pixels: int,
) -> dict[str, Any]:
    kept, metadata = _filter_eval_indices_by_train_footprint(
        scan_coords_yx,
        train_indices=train_indices,
        eval_indices=heldout_split["eval_indices"],
        stitch_patch_size=stitch_patch_size,
        full_reference_shape=full_reference_shape,
        heldout_guard_band_pixels=heldout_guard_band_pixels,
    )
    coords = np.asarray(scan_coords_yx, dtype=float)
    filtered = dict(heldout_split)
    filtered["eval_indices"] = kept
    filtered["eval_coordinate_count"] = int(len(kept))
    filtered["eval_coordinate_ranges_yx"] = _coordinate_ranges(coords[np.asarray(kept, dtype=int)])
    filtered.update(metadata)
    return filtered


def _footprint_shape(
    scan_coords_yx: np.ndarray,
    *,
    stitch_patch_size: int,
    full_reference_shape: tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    if full_reference_shape is not None:
        return int(full_reference_shape[0]), int(full_reference_shape[1])
    coords = np.asarray(scan_coords_yx, dtype=float)
    radius = max(1, int(stitch_patch_size) // 2)
    return (
        max(1, int(np.max(coords[:, 0])) + radius + 1),
        max(1, int(np.max(coords[:, 1])) + radius + 1),
    )


def coordinate_footprint_mask(
    scan_coords_yx: np.ndarray,
    *,
    stitch_patch_size: int,
    full_reference_shape: tuple[int, int] | list[int] | None = None,
) -> np.ndarray:
    coords = np.asarray(scan_coords_yx, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"scan_coords_yx must have non-empty shape (n_positions, 2), got {coords.shape}")
    h, w = _footprint_shape(
        coords,
        stitch_patch_size=stitch_patch_size,
        full_reference_shape=full_reference_shape,
    )
    mask = np.zeros((h, w), dtype=bool)
    radius = max(0, int(stitch_patch_size) // 2)
    for y_value, x_value in coords:
        center_y = int(y_value)
        center_x = int(x_value)
        row0 = max(0, center_y - radius)
        row1 = min(h, center_y + radius)
        col0 = max(0, center_x - radius)
        col1 = min(w, center_x + radius)
        if row1 <= row0:
            row1 = min(h, row0 + 1)
        if col1 <= col0:
            col1 = min(w, col0 + 1)
        mask[row0:row1, col0:col1] = True
    return mask


def count_connected_components(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"component mask must be 2D, got {values.shape}")
    seen = np.zeros(values.shape, dtype=bool)
    components = 0
    for start_y, start_x in zip(*np.nonzero(values)):
        if seen[start_y, start_x]:
            continue
        components += 1
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy = y + dy
                xx = x + dx
                if (
                    0 <= yy < values.shape[0]
                    and 0 <= xx < values.shape[1]
                    and values[yy, xx]
                    and not seen[yy, xx]
                ):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
    return components


def audit_split_contract(
    scan_coords_yx: np.ndarray,
    *,
    train_indices: Iterable[int],
    eval_indices: Iterable[int],
    stitch_patch_size: int,
    nonoverlap_level: str = SPLIT_NONOVERLAP_COORDINATE_INDICES,
    full_reference_shape: tuple[int, int] | list[int] | None = None,
    heldout_guard_band_pixels: int = 0,
    metric_region_matches_displayed_panel: bool | None = None,
    allow_panel_metric_region_mismatch: bool = False,
    policy: str | None = None,
    train_index_source: str | None = None,
) -> dict[str, Any]:
    coords = np.asarray(scan_coords_yx, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"scan_coords_yx must have non-empty shape (n_positions, 2), got {coords.shape}")
    if nonoverlap_level not in {SPLIT_NONOVERLAP_COORDINATE_INDICES, SPLIT_NONOVERLAP_OBJECT_FOOTPRINT}:
        raise ValueError(f"Unsupported split non-overlap level: {nonoverlap_level}")
    train = _normalize_indices(train_indices, n_coords=coords.shape[0], label="train")
    eval_ = _normalize_indices(eval_indices, n_coords=coords.shape[0], label="eval")
    index_overlap = np.intersect1d(train, eval_)
    if index_overlap.size:
        raise StopCondition(
            "split contract failed: train/eval coordinate indices overlap "
            f"({int(index_overlap.size)} shared indices)"
        )

    train_mask = coordinate_footprint_mask(
        coords[train],
        stitch_patch_size=stitch_patch_size,
        full_reference_shape=full_reference_shape,
    )
    eval_mask = coordinate_footprint_mask(
        coords[eval_],
        stitch_patch_size=stitch_patch_size,
        full_reference_shape=train_mask.shape,
    )
    train_components = count_connected_components(train_mask)
    eval_components = count_connected_components(eval_mask)
    if train_components != 1:
        raise StopCondition(f"split contract failed: train split is not spatially contiguous ({train_components} components)")
    if eval_components != 1:
        raise StopCondition(f"split contract failed: eval split is not spatially contiguous ({eval_components} components)")

    footprint_overlap = int(np.logical_and(train_mask, eval_mask).sum())
    if nonoverlap_level == SPLIT_NONOVERLAP_OBJECT_FOOTPRINT and footprint_overlap > 0:
        raise StopCondition(
            "split contract failed: object footprint overlap remains "
            f"({footprint_overlap} pixels); guard-band rerun required"
        )
    if nonoverlap_level == SPLIT_NONOVERLAP_OBJECT_FOOTPRINT:
        footprint_policy = "enforced_no_overlap"
    elif footprint_overlap > 0:
        footprint_policy = "recorded_not_enforced_coordinate_split_only"
    else:
        footprint_policy = "not_enforced_no_overlap_observed"

    if metric_region_matches_displayed_panel is None:
        mismatch_policy = "not_evaluated"
    elif bool(metric_region_matches_displayed_panel):
        mismatch_policy = "matches_displayed_panel"
    elif allow_panel_metric_region_mismatch:
        mismatch_policy = "internal_provenance_only"
    else:
        mismatch_policy = "recorded_internal_provenance_only"

    return {
        "status": "ok",
        "policy": policy,
        "nonoverlap_level": nonoverlap_level,
        "train_index_source": train_index_source,
        "train_coordinate_count": int(train.size),
        "eval_coordinate_count": int(eval_.size),
        "train_contiguous": True,
        "eval_contiguous": True,
        "train_connected_component_count": int(train_components),
        "eval_connected_component_count": int(eval_components),
        "coordinate_index_overlap_count": 0,
        "object_footprint_overlap_pixel_count": footprint_overlap,
        "object_footprint_overlap_policy": footprint_policy,
        "heldout_guard_band_pixels": int(heldout_guard_band_pixels),
        "train_footprint_pixel_count": int(train_mask.sum()),
        "eval_footprint_pixel_count": int(eval_mask.sum()),
        "train_coordinate_ranges_yx": _coordinate_ranges(coords[train]),
        "eval_coordinate_ranges_yx": _coordinate_ranges(coords[eval_]),
        "metric_region_matches_displayed_panel": metric_region_matches_displayed_panel,
        "allow_panel_metric_region_mismatch": bool(allow_panel_metric_region_mismatch),
        "panel_metric_region_mismatch_policy": mismatch_policy,
    }


def _split_contract_train_indices(heldout_split: dict[str, Any]) -> tuple[list[int], str]:
    if heldout_split.get("heldout_split_policy") == HELDOUT_SPLIT_PAPER_TOP_TRAIN_BOTTOM_TEST_BY_HIGH_Y:
        total = int(heldout_split.get("total_coordinate_count", 0))
        count = min(PAPER_RUN1084_HISTORICAL_TRAIN_COUNT, total)
        return list(range(count)), "inferred_from_historical_loader_first512_no_persisted_split_manifest"
    return [int(value) for value in heldout_split.get("train_indices", [])], "heldout_split_train_indices"


def apply_heldout_crop_after_alignment(
    *,
    aligned_recon: np.ndarray,
    aligned_gt: np.ndarray,
    scan_coords_yx: np.ndarray,
    full_reference_shape: tuple[int, int] | list[int],
    stitch_patch_size: int,
    heldout_split: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_bbox = coordinate_bbox_rows_cols(scan_coords_yx, stitch_patch_size, full_reference_shape)
    aligned_bbox = _aligned_gt_bbox_rows_cols(
        all_scan_bbox_rows_cols=all_bbox,
        aligned_gt_shape=np.asarray(aligned_gt).shape[:2],
    )
    eval_indices = np.asarray(heldout_split["eval_indices"], dtype=int)
    eval_coords = np.asarray(scan_coords_yx, dtype=float)[eval_indices]
    heldout_bbox = coordinate_bbox_rows_cols(eval_coords, stitch_patch_size, full_reference_shape)
    intersected_bbox = _intersect_bbox_rows_cols(aligned_bbox, heldout_bbox)
    relative_slice = _relative_slice_rows_cols(
        child_bbox_rows_cols=intersected_bbox,
        parent_bbox_rows_cols=aligned_bbox,
    )
    cropped_recon = _crop_relative(aligned_recon, relative_slice)
    cropped_gt = _crop_relative(aligned_gt, relative_slice)
    manifest = {
        **heldout_split,
        "evaluation_region": EVALUATION_REGION_HELDOUT,
        "heldout_only": True,
        "all_scan_bbox_rows_cols": all_bbox,
        "aligned_full_bbox_rows_cols": aligned_bbox,
        "heldout_bbox_rows_cols": heldout_bbox,
        "heldout_intersected_bbox_rows_cols": intersected_bbox,
        "heldout_relative_slice_rows_cols": relative_slice,
        "heldout_crop_shape": _shape(cropped_recon),
    }
    return cropped_recon, cropped_gt, manifest


def canonical_complex_digest(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.complex128))
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": _shape(arr), "dtype": "complex128"}, sort_keys=True).encode("utf-8"))
    digest.update(arr.tobytes(order="C"))
    return {
        "shape": _shape(arr),
        "dtype": "complex128",
        "sha256": digest.hexdigest(),
    }


def validate_panel_reference(
    *,
    row: StudyRow,
    aligned_gt: np.ndarray,
    panel_reference: np.ndarray | None,
    panel_reference_inventory: dict[str, Any] | None,
    reference_validation_scope: str = "row_specific_panel_reference",
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "panel_reference_path": str(row.panel_reference_path) if row.panel_reference_path else None,
        "reference_validation_scope": reference_validation_scope,
        "reference_validation_status": "failed",
        "reference_validation_mode": "missing_panel_reference",
        "panel_validation_status": "missing_panel_reference",
        "aligned_ground_truth_shape": _shape(aligned_gt),
        "panel_reference_shape": None,
        "panel_reference_sha256": None,
        "panel_reference_keys": [],
        "max_abs_diff": None,
        "relative_l2_diff": None,
        "max_abs_diff_tolerance": None,
        "relative_l2_diff_tolerance": REFERENCE_RELATIVE_L2_TOL,
        "canonical_cropped_reference_digest": canonical_complex_digest(aligned_gt),
        "panel_reference_digest": None,
    }
    if panel_reference_inventory:
        validation["panel_reference_sha256"] = panel_reference_inventory.get("sha256")
        validation["panel_reference_keys"] = list(panel_reference_inventory.get("keys", []))
    if panel_reference is None:
        return validation

    validation["panel_reference_shape"] = _shape(panel_reference)
    validation["panel_reference_digest"] = canonical_complex_digest(panel_reference)
    if tuple(panel_reference.shape) != tuple(aligned_gt.shape):
        validation["reference_validation_mode"] = "shape_mismatch"
        validation["panel_validation_status"] = "shape_mismatch"
        return validation

    aligned_c128 = np.ascontiguousarray(np.asarray(aligned_gt, dtype=np.complex128))
    panel_c128 = np.ascontiguousarray(np.asarray(panel_reference, dtype=np.complex128))
    if validation["canonical_cropped_reference_digest"]["sha256"] == validation["panel_reference_digest"]["sha256"]:
        validation["max_abs_diff"] = 0.0
        validation["relative_l2_diff"] = 0.0
        validation["max_abs_diff_tolerance"] = REFERENCE_MAX_ABS_RTOL * max(1.0, float(np.max(np.abs(panel_c128))))
        validation["reference_validation_status"] = "passed"
        validation["reference_validation_mode"] = "exact_digest_match"
        validation["panel_validation_status"] = "matched"
        return validation

    diff = panel_c128 - aligned_c128
    max_abs_diff = float(np.max(np.abs(diff)))
    relative_l2_diff = float(np.linalg.norm(diff.ravel()) / max(1.0, np.linalg.norm(panel_c128.ravel())))
    max_abs_tolerance = REFERENCE_MAX_ABS_RTOL * max(1.0, float(np.max(np.abs(panel_c128))))
    validation["max_abs_diff"] = max_abs_diff
    validation["relative_l2_diff"] = relative_l2_diff
    validation["max_abs_diff_tolerance"] = max_abs_tolerance
    if max_abs_diff <= max_abs_tolerance and relative_l2_diff <= REFERENCE_RELATIVE_L2_TOL:
        validation["reference_validation_status"] = "passed"
        validation["reference_validation_mode"] = "numeric_identity_tolerance"
        validation["panel_validation_status"] = "matched"
    else:
        validation["reference_validation_mode"] = "numeric_mismatch"
        validation["panel_validation_status"] = "failed"
    return validation


def _center_crop_2d(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(array)
    h, w = arr.shape
    target_h, target_w = target_shape
    if target_h > h or target_w > w:
        raise ValueError(f"Cannot crop shape {arr.shape} to larger target {target_shape}")
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return arr[y0 : y0 + target_h, x0 : x0 + target_w]


def _panel_artifact_exception_alignment(
    *,
    reconstruction: np.ndarray,
    panel_reference: np.ndarray | None,
    panel_exception_reason: str,
    alignment_failure: Exception,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not panel_exception_reason.strip():
        raise ValueError("--panel-exception-reason is required when using --allow-panel-artifact-exception")
    if panel_reference is None:
        raise ValueError("panel artifact exception requires a panel reference")
    target_shape = (
        min(int(reconstruction.shape[0]), int(panel_reference.shape[0])),
        min(int(reconstruction.shape[1]), int(panel_reference.shape[1])),
    )
    aligned_recon = _center_crop_2d(reconstruction, target_shape)
    aligned_gt = _center_crop_2d(panel_reference, target_shape)
    return aligned_recon, aligned_gt, {
        "alignment_mode": "panel_artifact_exception",
        "panel_exception_reason": panel_exception_reason,
        "coordinate_alignment_failure": repr(alignment_failure),
        "panel_exception_crop_policy": "center_crop_to_common_shape_without_row_specific_tuning",
    }


def align_row_for_metrics(
    *,
    row: StudyRow,
    reconstruction: np.ndarray,
    full_reference: np.ndarray,
    scan_coords_yx: np.ndarray,
    stitch_patch_size: int,
    panel_reference: np.ndarray | None,
    panel_reference_inventory: dict[str, Any] | None,
    allow_panel_artifact_exception: bool,
    panel_exception_reason: str,
    reference_validation_scope: str = "row_specific_panel_reference",
    evaluation_region: str = EVALUATION_REGION_ALL_SCAN,
    heldout_split_policy: str = HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y,
    heldout_split_source: Path | None = None,
    split_nonoverlap_level: str = SPLIT_NONOVERLAP_COORDINATE_INDICES,
    heldout_guard_band_pixels: int = 0,
    allow_panel_metric_region_mismatch: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if evaluation_region not in {EVALUATION_REGION_ALL_SCAN, EVALUATION_REGION_HELDOUT}:
        raise ValueError(f"Unsupported evaluation region: {evaluation_region}")
    manifest: dict[str, Any] = {
        "row_id": row.row_id,
        "condition": row.condition,
        "model": row.model,
        "input_reconstruction_shape": _shape(reconstruction),
        "full_reference_shape": _shape(full_reference),
        "coordinate_count": int(np.asarray(scan_coords_yx).shape[0]),
        "total_coordinate_count": int(np.asarray(scan_coords_yx).shape[0]),
        "eval_coordinate_count": int(np.asarray(scan_coords_yx).shape[0]),
        "coordinate_ranges_yx": _coordinate_ranges(scan_coords_yx),
        "stitch_patch_size": int(stitch_patch_size),
        "panel_exception_available": bool(allow_panel_artifact_exception),
        "primary_registration_mode": "none",
        "evaluation_region": evaluation_region,
        "heldout_only": False,
    }
    try:
        aligned_recon, aligned_gt = align_for_evaluation(
            reconstruction,
            full_reference,
            scan_coords_yx,
            stitch_patch_size,
        )
        manifest.update(
            {
                "alignment_mode": "coordinate_align_for_evaluation",
                "coordinate_aligned_reconstruction_shape": _shape(aligned_recon),
                "coordinate_aligned_ground_truth_shape": _shape(aligned_gt),
            }
        )
    except Exception as exc:
        if not allow_panel_artifact_exception:
            raise
        aligned_recon, aligned_gt, exception_manifest = _panel_artifact_exception_alignment(
            reconstruction=reconstruction,
            panel_reference=panel_reference,
            panel_exception_reason=panel_exception_reason,
            alignment_failure=exc,
        )
        manifest.update(exception_manifest)
        manifest.update(
            {
                "coordinate_aligned_reconstruction_shape": _shape(aligned_recon),
                "coordinate_aligned_ground_truth_shape": _shape(aligned_gt),
            }
        )

    validation_aligned_gt = aligned_gt
    manifest.update(
        validate_panel_reference(
            row=row,
            aligned_gt=validation_aligned_gt,
            panel_reference=panel_reference,
            panel_reference_inventory=panel_reference_inventory,
            reference_validation_scope=reference_validation_scope,
        )
    )
    if evaluation_region == EVALUATION_REGION_HELDOUT:
        if manifest.get("alignment_mode") != "coordinate_align_for_evaluation":
            raise StopCondition("held-out evaluation requires coordinate alignment, not panel artifact exception")
        heldout_split = select_spatial_holdout_split(
            scan_coords_yx,
            policy=heldout_split_policy,
            split_source=heldout_split_source,
        )
        contract_train_indices, train_index_source = _split_contract_train_indices(heldout_split)
        if split_nonoverlap_level == SPLIT_NONOVERLAP_OBJECT_FOOTPRINT or heldout_guard_band_pixels:
            heldout_split = apply_eval_guard_band_to_split(
                heldout_split,
                scan_coords_yx=scan_coords_yx,
                train_indices=contract_train_indices,
                stitch_patch_size=stitch_patch_size,
                full_reference_shape=np.asarray(full_reference).shape[:2],
                heldout_guard_band_pixels=heldout_guard_band_pixels,
            )
        aligned_recon, aligned_gt, heldout_manifest = apply_heldout_crop_after_alignment(
            aligned_recon=aligned_recon,
            aligned_gt=aligned_gt,
            scan_coords_yx=scan_coords_yx,
            full_reference_shape=np.asarray(full_reference).shape[:2],
            stitch_patch_size=stitch_patch_size,
            heldout_split=heldout_split,
        )
        manifest.update(heldout_manifest)
        manifest["split_contract"] = audit_split_contract(
            scan_coords_yx,
            train_indices=contract_train_indices,
            eval_indices=heldout_split["eval_indices"],
            stitch_patch_size=stitch_patch_size,
            nonoverlap_level=split_nonoverlap_level,
            full_reference_shape=np.asarray(full_reference).shape[:2],
            heldout_guard_band_pixels=heldout_guard_band_pixels,
            metric_region_matches_displayed_panel=False,
            allow_panel_metric_region_mismatch=allow_panel_metric_region_mismatch,
            policy=heldout_split.get("heldout_split_policy"),
            train_index_source=train_index_source,
        )
    manifest["reference_validation_completed_before_metrics"] = False
    return aligned_recon, aligned_gt, manifest


def register_aligned_row(
    aligned_recon: np.ndarray,
    aligned_gt: np.ndarray,
    *,
    upsample_factor: int,
    border_crop: int,
    fail_on_error: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    try:
        offset = find_translation_offset(
            aligned_recon,
            aligned_gt,
            upsample_factor=upsample_factor,
        )
        registered_recon, registered_gt = apply_shift_and_crop(
            aligned_recon,
            aligned_gt,
            offset,
            border_crop=border_crop,
        )
    except Exception as exc:
        manifest = {
            "registration_status": "failed",
            "registration_error": repr(exc),
            "registration_offset_yx": None,
            "registration_offset_norm": None,
            "upsample_factor": int(upsample_factor),
            "border_crop": int(border_crop),
        }
        if fail_on_error:
            raise
        return None, None, manifest

    offset_yx = [float(offset[0]), float(offset[1])]
    return registered_recon, registered_gt, {
        "registration_status": "ok",
        "registration_offset_yx": offset_yx,
        "registration_offset_norm": float(np.linalg.norm(offset_yx)),
        "upsample_factor": int(upsample_factor),
        "border_crop": int(border_crop),
        "registered_reconstruction_shape": _shape(registered_recon),
        "registered_ground_truth_shape": _shape(registered_gt),
    }


def validate_eval_offset(eval_offset: int, registered_shape: tuple[int, int]) -> int:
    offset = int(eval_offset)
    if offset <= 0 or offset % 2:
        raise ValueError(f"eval_offset must be a positive even integer, got {eval_offset}")
    if offset >= min(int(registered_shape[0]), int(registered_shape[1])):
        raise ValueError(f"eval_offset must be smaller than registered shape, got {offset} for {registered_shape}")
    return offset // 2


def describe_frc_field_of_view(post_eval_metric_shape: tuple[int, int] | list[int]) -> dict[str, Any]:
    h, w = [int(v) for v in post_eval_metric_shape]
    if h == w:
        return {
            "frc_input_shape": [h, w],
            "frc_square_crop_applied": False,
            "frc_shape": [h, w],
            "frc_crop_slices_yx": {"y": [0, h], "x": [0, w]},
            "frc_field_of_view_matches_core_metrics": True,
            "frc_field_of_view_policy": "post_eval_metric_shape",
        }
    min_dim = min(h, w)
    y0 = (h - min_dim) // 2
    x0 = (w - min_dim) // 2
    return {
        "frc_input_shape": [h, w],
        "frc_square_crop_applied": True,
        "frc_shape": [min_dim, min_dim],
        "frc_crop_slices_yx": {"y": [y0, y0 + min_dim], "x": [x0, x0 + min_dim]},
        "frc_field_of_view_matches_core_metrics": False,
        "frc_field_of_view_policy": "post_eval_metric_shape -> frc_cutoffs_square_center_crop",
    }


def eval_reconstruction(*args, **kwargs):
    from ptycho.evaluation import eval_reconstruction as _eval_reconstruction

    return _eval_reconstruction(*args, **kwargs)


def _trimmed_view(array: np.ndarray, trim_per_edge: int) -> np.ndarray:
    return np.asarray(array)[trim_per_edge:-trim_per_edge, trim_per_edge:-trim_per_edge]


def _amplitude_psnr(target: np.ndarray, pred: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mse_value = float(np.mean((target - pred) ** 2))
    if not math.isfinite(mse_value):
        return None
    if mse_value == 0.0:
        return float("inf")
    data_range = float(np.max(target) - np.min(target))
    if data_range <= 0.0 or not math.isfinite(data_range):
        data_range = max(1.0, float(np.max(np.abs(target))))
    return float(20.0 * math.log10(data_range / math.sqrt(mse_value)))


def _metric_pair(metrics: dict[str, Any], key: str) -> tuple[float, float]:
    value = metrics.get(key, (np.nan, np.nan))
    if value is None:
        return np.nan, np.nan
    return float(value[0]), float(value[1])


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pair_is_finite(pair: tuple[float, float]) -> bool:
    return all(math.isfinite(float(value)) for value in pair)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compute_row_metrics(
    *,
    row: StudyRow,
    recon_registered: np.ndarray,
    gt_registered: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if recon_registered.shape != gt_registered.shape:
        raise ValueError(f"Metric shapes differ for {row.row_id}: {recon_registered.shape} vs {gt_registered.shape}")
    if recon_registered.ndim != 2:
        raise ValueError(f"Metric reconstruction for {row.row_id} must be 2D, got {recon_registered.shape}")
    if not np.all(np.isfinite(recon_registered)) or not np.all(np.isfinite(gt_registered)):
        raise ValueError(f"Metric arrays for {row.row_id} contain non-finite values")
    trim_per_edge = validate_eval_offset(args.eval_offset, tuple(recon_registered.shape))
    post_eval_amp_gt = np.abs(_trimmed_view(gt_registered, trim_per_edge))
    post_eval_amp_recon = np.abs(_trimmed_view(recon_registered, trim_per_edge))
    min_post_eval_trim_dim = int(getattr(args, "min_post_eval_trim_dim", MIN_POST_EVAL_TRIM_DIM))
    if min_post_eval_trim_dim <= 0:
        raise ValueError(f"min_post_eval_trim_dim must be positive, got {min_post_eval_trim_dim}")
    if min(post_eval_amp_gt.shape) < min_post_eval_trim_dim:
        raise ValueError(
            f"post-eval-trim shape for {row.row_id} must be at least "
            f"{min_post_eval_trim_dim} by {min_post_eval_trim_dim}, got {post_eval_amp_gt.shape}"
        )
    gt_mean = float(np.mean(post_eval_amp_gt))
    recon_mean = float(np.mean(post_eval_amp_recon))
    if not math.isfinite(gt_mean) or not math.isfinite(recon_mean) or recon_mean == 0.0:
        raise ValueError(f"Cannot compute finite amplitude scale factor for {row.row_id}")
    amplitude_scale_factor = gt_mean / recon_mean
    amplitude_mean_ratio_unscaled = recon_mean / gt_mean if gt_mean != 0.0 else None
    amplitude_mse_unscaled = float(np.mean((post_eval_amp_gt - post_eval_amp_recon) ** 2))
    amplitude_mae_unscaled = float(np.mean(np.abs(post_eval_amp_gt - post_eval_amp_recon)))
    amplitude_psnr_unscaled = _amplitude_psnr(post_eval_amp_gt, post_eval_amp_recon)
    post_eval_metric_shape = _shape(post_eval_amp_gt)
    fov = describe_frc_field_of_view(post_eval_metric_shape)
    stitched_obj_eval = recon_registered[None, :, :, None]
    ground_truth_obj_eval = gt_registered[:, :, None]

    previous_offset = legacy_params.cfg.get("offset")
    legacy_params.cfg["offset"] = int(args.eval_offset)
    try:
        metrics = eval_reconstruction(
            stitched_obj_eval,
            ground_truth_obj_eval,
            label=row.row_id,
            phase_align_method=args.phase_align_method,
            frc_sigma=args.frc_sigma,
            ms_ssim_sigma=args.ms_ssim_sigma,
        )
    finally:
        if previous_offset is None:
            legacy_params.cfg.pop("offset", None)
        else:
            legacy_params.cfg["offset"] = previous_offset

    mae_amp, mae_phase = _metric_pair(metrics, "mae")
    mse_amp, mse_phase = _metric_pair(metrics, "mse")
    psnr_amp, psnr_phase = _metric_pair(metrics, "psnr")
    ssim_amp, ssim_phase = _metric_pair(metrics, "ssim")
    ms_ssim_amp, ms_ssim_phase = _metric_pair(metrics, "ms_ssim")
    frc50_amp, frc50_phase = _metric_pair(metrics, "frc50")
    frc1over7_amp, frc1over7_phase = _metric_pair(metrics, "frc1over7")

    frc_available = _pair_is_finite((frc50_amp, frc50_phase)) and _pair_is_finite(
        (frc1over7_amp, frc1over7_phase)
    )
    if not frc_available:
        frc_paper_table_status = "omitted_due_to_frc_failure"
    elif not fov["frc_field_of_view_matches_core_metrics"]:
        frc_paper_table_status = "artifact_only_due_to_frc_square_crop"
    else:
        frc_paper_table_status = "paper_candidate_same_fov"

    common = {
        "row_id": row.row_id,
        "condition": row.condition,
        "model": row.model,
        "metric_status": "ok",
        "metric_contract": "coordinate_align_for_evaluation_only_eval_reconstruction_legacy_trim_mean_scaled",
        "primary_registration_mode": "none",
        "eval_offset": int(args.eval_offset),
        "eval_trim_per_edge": int(trim_per_edge),
        "metric_trim_offset": int(args.eval_offset),
        "metric_trim_pixels_per_edge": int(trim_per_edge),
        "min_post_eval_trim_dim": min_post_eval_trim_dim,
        "pre_eval_shape": _shape(recon_registered),
        "post_eval_trim_shape": post_eval_metric_shape,
        "pre_eval_metric_shape": _shape(recon_registered),
        "post_eval_metric_shape": post_eval_metric_shape,
        "pre_eval_adapter_reconstruction_shape": _shape(recon_registered),
        "pre_eval_adapter_reference_shape": _shape(gt_registered),
        "eval_stitched_obj_shape": _shape(stitched_obj_eval),
        "eval_ground_truth_obj_shape": _shape(ground_truth_obj_eval),
        "eval_shape_adapter_no_semantic_change": True,
        "amplitude_scale_factor": float(amplitude_scale_factor),
        "amplitude_mean_reference": gt_mean,
        "amplitude_mean_reconstruction": recon_mean,
        "amplitude_mean_ratio_unscaled": amplitude_mean_ratio_unscaled,
        "amplitude_mse_unscaled": amplitude_mse_unscaled,
        "amplitude_mae_unscaled": amplitude_mae_unscaled,
        "amplitude_psnr_unscaled": amplitude_psnr_unscaled,
        "amplitude_scaled_unscaled_conclusion_conflict": False,
        "amplitude_scale_factor_source": "post_eval_trim",
        "metric_field_of_view_policy": "coordinate_align_for_evaluation -> eval_reconstruction_legacy_trim",
        "phase_align_method": args.phase_align_method,
        "phase_policy": "phase plane alignment; absolute global phase and removed linear phase ramps are not scored",
        "background_policy": "no_support_or_background_mask",
        "frc_paper_table_status": frc_paper_table_status,
        **fov,
    }
    payload = {
        **common,
        "amplitude_mae": _number_or_none(mae_amp),
        "phase_mae": _number_or_none(mae_phase),
        "amplitude_mse": _number_or_none(mse_amp),
        "phase_mse": _number_or_none(mse_phase),
        "amplitude_psnr": _number_or_none(psnr_amp),
        "phase_psnr": _number_or_none(psnr_phase),
        "amplitude_ssim": _number_or_none(ssim_amp),
        "phase_ssim": _number_or_none(ssim_phase),
        "amplitude_ms_ssim": _number_or_none(ms_ssim_amp),
        "phase_ms_ssim": _number_or_none(ms_ssim_phase),
        "amplitude_frc50": _number_or_none(frc50_amp),
        "phase_frc50": _number_or_none(frc50_phase),
        "amplitude_frc1over7": _number_or_none(frc1over7_amp),
        "phase_frc1over7": _number_or_none(frc1over7_phase),
    }
    return payload, dict(payload)


def compute_fine_registration_sensitivity(
    *,
    row: StudyRow,
    aligned_recon: np.ndarray,
    aligned_gt: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        offset_primary = find_translation_offset(
            aligned_recon,
            aligned_gt,
            upsample_factor=int(args.upsample_factor),
        )
        offset_stability = find_translation_offset(
            aligned_recon,
            aligned_gt,
            upsample_factor=int(args.registration_stability_factor),
        )
    except Exception as exc:
        return {
            "registration_diagnostic_status": "failed",
            "registration_diagnostic_error": repr(exc),
            "registration_offset_yx_upsample50": None,
            "registration_offset_yx_upsample10": None,
            "fine_registration_sensitivity_status": "not_scored_registration_diagnostic_failed",
        }

    offset50 = [float(offset_primary[0]), float(offset_primary[1])]
    offset10 = [float(offset_stability[0]), float(offset_stability[1])]
    max_abs50 = float(max(abs(offset50[0]), abs(offset50[1])))
    max_abs10 = float(max(abs(offset10[0]), abs(offset10[1])))
    delta_max = float(max(abs(offset50[0] - offset10[0]), abs(offset50[1] - offset10[1])))
    gate = float(args.border_crop)
    crosses_gate = (max_abs50 <= gate) != (max_abs10 <= gate)
    manifest: dict[str, Any] = {
        "registration_diagnostic_status": "ok",
        "registration_offset_yx_upsample50": offset50,
        "registration_offset_yx_upsample10": offset10,
        "registration_offset_norm_upsample50": float(np.linalg.norm(offset50)),
        "registration_offset_norm_upsample10": float(np.linalg.norm(offset10)),
        "registration_offset_max_abs_component_upsample50": max_abs50,
        "registration_offset_max_abs_component_upsample10": max_abs10,
        "registration_offset_delta_max_abs_component_50_vs_10": delta_max,
        "border_crop": int(args.border_crop),
        "diagnostic_crop_safety_status": "crop_safe" if max_abs50 <= gate else "offset_exceeds_border_crop",
        "offset_stability_status": "stable" if delta_max <= 0.5 and not crosses_gate else "unstable",
        "fine_registered_primary_fields_promoted": False,
    }
    if max_abs50 > gate:
        manifest["fine_registration_sensitivity_status"] = "not_scored_offset_exceeds_border_crop"
        return manifest
    if delta_max > 0.5 or crosses_gate:
        manifest["fine_registration_sensitivity_status"] = "not_scored_offset_instability"
        return manifest

    shifted_recon, shifted_gt = apply_shift_and_crop(
        aligned_recon,
        aligned_gt,
        tuple(offset_primary),
        border_crop=int(args.border_crop),
    )
    sensitivity_payload, _ = compute_row_metrics(
        row=row,
        recon_registered=shifted_recon,
        gt_registered=shifted_gt,
        args=args,
    )
    for key, value in sensitivity_payload.items():
        if key in {"row_id", "condition", "model"}:
            continue
        manifest[f"fine_registered_{key}"] = value
    manifest["fine_registration_sensitivity_status"] = "scored_crop_safe_sensitivity_only"
    return manifest


CSV_FIELDNAMES = [
    "row_id",
    "condition",
    "model",
    "amplitude_mae",
    "amplitude_mse",
    "amplitude_psnr",
    "amplitude_ssim",
    "amplitude_ms_ssim",
    "phase_mae",
    "phase_mse",
    "phase_psnr",
    "phase_ssim",
    "phase_ms_ssim",
    "amplitude_frc50",
    "phase_frc50",
    "amplitude_frc1over7",
    "phase_frc1over7",
    "pre_eval_metric_shape",
    "post_eval_metric_shape",
    "eval_offset",
    "eval_trim_per_edge",
    "metric_trim_offset",
    "metric_trim_pixels_per_edge",
    "min_post_eval_trim_dim",
    "pre_eval_adapter_reconstruction_shape",
    "pre_eval_adapter_reference_shape",
    "eval_stitched_obj_shape",
    "eval_ground_truth_obj_shape",
    "primary_registration_mode",
    "alignment_mode",
    "evaluation_region",
    "heldout_only",
    "heldout_split_policy",
    "heldout_split_source",
    "total_coordinate_count",
    "eval_coordinate_count",
    "heldout_crop_shape",
    "split_contract",
    "reference_validation_status",
    "amplitude_scale_factor",
    "amplitude_mean_reference",
    "amplitude_mean_reconstruction",
    "amplitude_mean_ratio_unscaled",
    "amplitude_mse_unscaled",
    "amplitude_mae_unscaled",
    "amplitude_psnr_unscaled",
    "amplitude_scaled_unscaled_conclusion_conflict",
    "fine_registration_sensitivity_status",
    "registration_offset_yx_upsample50",
    "registration_offset_yx_upsample10",
    "registration_offset_max_abs_component_upsample50",
    "frc_input_shape",
    "frc_square_crop_applied",
    "frc_shape",
    "frc_crop_slices_yx",
    "frc_field_of_view_matches_core_metrics",
    "frc_field_of_view_policy",
    "frc_paper_table_status",
]


def _format_metric(value: Any) -> str:
    if value is None:
        return "--"
    number = float(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.4g}"


def render_metrics_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Condition & Model & Amp. MSE & Amp. PSNR & Amp. SSIM & Phase MSE & Phase PSNR & Phase SSIM \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["condition"]),
                    str(row["model"]).replace("_", r"\_"),
                    _format_metric(row.get("amplitude_mse")),
                    _format_metric(row.get("amplitude_psnr")),
                    _format_metric(row.get("amplitude_ssim")),
                    _format_metric(row.get("phase_mse")),
                    _format_metric(row.get("phase_psnr")),
                    _format_metric(row.get("phase_ssim")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def annotate_scaled_unscaled_conflicts(rows: list[dict[str, Any]]) -> bool:
    conflict = False
    for condition in sorted({str(row.get("condition")) for row in rows}):
        condition_rows = [row for row in rows if str(row.get("condition")) == condition]
        if len(condition_rows) < 2:
            continue
        scaled = sorted(condition_rows, key=lambda row: float(row.get("amplitude_mse", math.inf)))
        unscaled = sorted(condition_rows, key=lambda row: float(row.get("amplitude_mse_unscaled", math.inf)))
        if scaled[0].get("model") != unscaled[0].get("model"):
            conflict = True
    for row in rows:
        row["amplitude_scaled_unscaled_conclusion_conflict"] = conflict
    return conflict


def write_metrics_artifacts(
    *,
    output_root: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    metric_policy: dict[str, Any],
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conflict = annotate_scaled_unscaled_conflicts(rows)
    manifest["amplitude_scaled_unscaled_conclusion_conflict"] = conflict
    payload = {"metric_policy": metric_policy, "rows": rows}
    _write_json(output_root / "fig5_ood_metrics.json", payload)
    with (output_root / "fig5_ood_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(row.get(key))) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in CSV_FIELDNAMES})
    table = render_metrics_table(rows)
    (output_root / "fig5_ood_metrics_table.tex").write_text(table)
    summary_lines = [
        "# Fig. 5 OOD Metrics Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- Alignment: {metric_policy.get('alignment', 'coordinate_align_for_evaluation')}",
        f"- Background policy: {metric_policy.get('background_policy', 'not recorded')}",
        f"- Phase policy: {metric_policy.get('phase_policy', 'not recorded')}",
        "",
    ]
    for row in rows:
        summary_lines.append(
            f"- {row['condition']} {row['model']}: amp SSIM {_format_metric(row.get('amplitude_ssim'))}, "
            f"phase SSIM {_format_metric(row.get('phase_ssim'))}, FRC table status {row.get('frc_paper_table_status')}"
        )
    (output_root / "fig5_ood_metrics_summary.md").write_text("\n".join(summary_lines) + "\n")
    refresh_output_artifact_status(manifest, output_root=output_root, accepted_metrics=True)
    _write_json(output_root / "fig5_ood_metrics_manifest.json", manifest)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, 0)
        return True
    return False


def _write_lock_file(lock_path: Path, pid: int) -> None:
    with lock_path.open("x") as handle:
        handle.write(f"{pid}\n")


def acquire_output_lock(output_root: Path, force_stale_lock: bool) -> OutputLock:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "run.lock"
    pid = os.getpid()
    try:
        _write_lock_file(lock_path, pid)
    except FileExistsError as exc:
        content = lock_path.read_text().strip()
        try:
            existing_pid = int(content)
        except ValueError:
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise OutputLockError(f"{lock_path} is active for PID {existing_pid}") from exc
        if not force_stale_lock:
            raise OutputLockError(f"{lock_path} is stale; inspect it and rerun with --force-stale-lock if safe") from exc
        lock_path.unlink()
        _write_lock_file(lock_path, pid)
        return OutputLock(
            output_root=output_root,
            path=lock_path,
            pid=pid,
            replaced_stale_lock=True,
            stale_lock_content=content,
        )
    return OutputLock(output_root=output_root, path=lock_path, pid=pid)


def release_output_lock(lock: OutputLock) -> None:
    if lock._released:
        return
    if lock.path.exists() and lock.path.read_text().strip() == str(lock.pid):
        lock.path.unlink()
    lock._released = True


def collect_paper_fig5_png_checksums(*, repo_root: Path, paper_root: Path) -> dict[str, Any]:
    rows = []
    for paper_name, source_rel in PAPER_FIG5_PNG_SOURCES.items():
        paper_path = Path(paper_root) / "figures" / "out-dist-fig" / paper_name
        source_path = Path(repo_root) / source_rel
        paper_exists = paper_path.exists()
        source_exists = source_path.exists()
        paper_sha = sha256_file(paper_path) if paper_exists else None
        source_sha = sha256_file(source_path) if source_exists else None
        rows.append(
            {
                "paper_path": str(paper_path),
                "source_path": str(source_path),
                "paper_exists": paper_exists,
                "source_exists": source_exists,
                "paper_sha256": paper_sha,
                "source_sha256": source_sha,
                "byte_identical": bool(paper_exists and source_exists and paper_sha == source_sha),
            }
        )
    return {
        "entries": rows,
        "all_required_byte_identical": all(row["byte_identical"] for row in rows),
    }


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    coordinate_source = (repo_root / args.coordinate_source).resolve()
    full_reference = (repo_root / args.full_reference).resolve()
    rows = build_default_rows(repo_root)
    row_entries = []
    for row in rows:
        entry = {
            "row_id": row.row_id,
            "condition": row.condition,
            "model": row.model,
            "reconstruction_path": str(row.reconstruction_path),
            "panel_reference_path": str(row.panel_reference_path) if row.panel_reference_path else None,
            "notes": row.notes,
            "reconstruction_inventory": inventory_npz(row.reconstruction_path),
            "panel_reference_inventory": inventory_npz(row.panel_reference_path)
            if row.panel_reference_path and row.panel_reference_path.exists()
            else None,
        }
        row_entries.append(entry)
    excluded = [
        {
            "path": "experiment_outputs/fly64_trained_models/comparison_metrics.csv",
            "reason": "Fly64-to-Fly64 comparison output, not APS-trained-to-Run1084 evidence",
            "used_as_metric_source": False,
        },
        {
            "path": "experiment_outputs/fly64_trained_models/reconstructions_aligned.npz",
            "reason": "Prohibited provenance-only input without APS-trained-to-Run1084 proof",
            "used_as_metric_source": False,
        },
        {
            "path": "run1084_trained_models_fixed/",
            "reason": "Not active Fig. 5 provenance without paper source-image approval",
            "used_as_metric_source": False,
        },
    ]
    return {
        "repo_root": str(repo_root),
        "paper_root": str(Path(args.paper_root).resolve()),
        "coordinate_source": str(coordinate_source),
        "full_reference": str(full_reference),
        "evaluation_region": getattr(args, "evaluation_region", EVALUATION_REGION_ALL_SCAN),
        "heldout_split_policy": getattr(args, "heldout_split_policy", None),
        "heldout_split_source": str(args.heldout_split_source) if getattr(args, "heldout_split_source", None) else None,
        "require_heldout_eval": bool(getattr(args, "require_heldout_eval", False)),
        "min_post_eval_trim_dim": int(getattr(args, "min_post_eval_trim_dim", MIN_POST_EVAL_TRIM_DIM)),
        "split_nonoverlap_level": getattr(args, "split_nonoverlap_level", SPLIT_NONOVERLAP_COORDINATE_INDICES),
        "heldout_guard_band_pixels": int(getattr(args, "heldout_guard_band_pixels", 0)),
        "allow_panel_metric_region_mismatch": bool(getattr(args, "allow_panel_metric_region_mismatch", False)),
        "coordinate_source_inventory": inventory_npz(coordinate_source),
        "full_reference_inventory": inventory_npz(full_reference) if full_reference.exists() else None,
        "rows": row_entries,
        "metric_source_paths": {
            "reconstruction_npzs": [str(row.reconstruction_path) for row in rows],
            "coordinate_source": str(coordinate_source),
            "full_reference": str(full_reference),
        },
        "excluded_metric_sources": [entry["path"] for entry in excluded],
        "explicit_excluded_sources": excluded,
        "paper_fig5_png_checksum_comparison": collect_paper_fig5_png_checksums(
            repo_root=repo_root,
            paper_root=Path(args.paper_root).resolve(),
        ),
    }


def _run_text(command: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _git_summary(path: Path) -> dict[str, Any]:
    repo_path = Path(path).resolve()
    code, commit, stderr = _run_text(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
    status_code, status, status_stderr = _run_text(["git", "-C", str(repo_path), "status", "--short"])
    return {
        "path": str(repo_path),
        "commit": commit if code == 0 else None,
        "commit_status": "ok" if code == 0 else "failed",
        "commit_error": stderr if code != 0 else None,
        "dirty_status": status.splitlines() if status_code == 0 and status else [],
        "dirty_status_command_status": "ok" if status_code == 0 else "failed",
        "dirty_status_error": status_stderr if status_code != 0 else None,
    }


def _package_version(distribution_name: str) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _environment_summary() -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "skimage": _package_version("scikit-image"),
            "scipy": _package_version("scipy"),
            "tensorflow": _package_version("tensorflow"),
        },
    }


def _output_artifact_paths(output_root: Path) -> dict[str, str]:
    root = Path(output_root)
    return {
        "output_root": str(root),
        "invocation_json": str(root / "invocation.json"),
        "invocation_sh": str(root / "invocation.sh"),
        "manifest": str(root / "fig5_ood_metrics_manifest.json"),
        "source_inventory": str(root / "fig5_source_inventory.json"),
        "reference_validation": str(root / "fig5_reference_validation.json"),
        "metrics_json": str(root / "fig5_ood_metrics.json"),
        "metrics_csv": str(root / "fig5_ood_metrics.csv"),
        "metrics_table_tex": str(root / "fig5_ood_metrics_table.tex"),
        "metrics_summary_md": str(root / "fig5_ood_metrics_summary.md"),
        "pivot_summary_md": str(root / "pivot_summary.md"),
        "paper_asset_emission_invocation_json": str(root / "paper_asset_emission_invocation" / "invocation.json"),
        "paper_asset_emission_invocation_sh": str(root / "paper_asset_emission_invocation" / "invocation.sh"),
    }


METRIC_CLAIM_ARTIFACT_KEYS = (
    "metrics_json",
    "metrics_csv",
    "metrics_table_tex",
    "metrics_summary_md",
)


def _output_artifact_status(
    output_root: Path,
    *,
    accepted_metrics: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for key, value in _output_artifact_paths(output_root).items():
        path = Path(value)
        exists = path.exists()
        record = {
            "path": str(path),
            "exists": exists,
            "size_bytes": int(path.stat().st_size) if exists else None,
        }
        if key in METRIC_CLAIM_ARTIFACT_KEYS:
            record["accepted_for_paper_claims"] = bool(accepted_metrics and exists)
        elif key == "pivot_summary_md":
            record["accepted_for_paper_claims"] = False
        status[key] = record

    accepted_metric_artifacts = bool(
        accepted_metrics and all(status[key]["exists"] for key in METRIC_CLAIM_ARTIFACT_KEYS)
    )
    return status, {
        "accepted_metrics_artifacts": accepted_metric_artifacts,
        "paper_claims_allowed": accepted_metric_artifacts,
        "required_metric_artifacts": list(METRIC_CLAIM_ARTIFACT_KEYS),
    }


def refresh_output_artifact_status(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    accepted_metrics: bool,
) -> None:
    status, acceptance = _output_artifact_status(output_root, accepted_metrics=accepted_metrics)
    manifest["output_artifact_status"] = status
    manifest["artifact_acceptance"] = acceptance


def attach_run_provenance(
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    raw_argv: list[str] | None,
    invocation_json_path: Path,
    invocation_sh_path: Path,
    invocation_artifact_kind: str,
    include_paper_git: bool,
) -> dict[str, Any]:
    from scripts.studies.invocation_logging import build_command_line

    argv = [str(arg) for arg in (raw_argv or [])]
    script_path = "scripts/studies/ood_fig5_metrics.py"
    output_root = Path(args.output_root)
    manifest["command_invocation"] = {
        "script": script_path,
        "argv": argv,
        "command": build_command_line(script_path, argv),
        "cwd": str(Path.cwd()),
    }
    manifest["invocation_artifacts"] = {
        "kind": invocation_artifact_kind,
        "invocation_json": str(Path(invocation_json_path)),
        "invocation_sh": str(Path(invocation_sh_path)),
    }
    manifest["source_repo_git"] = _git_summary(Path(args.repo_root))
    if include_paper_git:
        manifest["paper_repo_git"] = _git_summary(Path(args.paper_root))
    manifest["environment"] = _environment_summary()
    manifest["output_artifacts"] = _output_artifact_paths(output_root)
    refresh_output_artifact_status(manifest, output_root=output_root, accepted_metrics=False)
    manifest["metric_policy"] = metric_policy(args)
    return manifest


def validate_heldout_evaluation_requirement(args: argparse.Namespace) -> None:
    if not getattr(args, "require_heldout_eval", False):
        return
    if getattr(args, "evaluation_region", None) != EVALUATION_REGION_HELDOUT:
        raise StopCondition(
            "held-out evaluation is required for this paper-facing run; "
            f"got evaluation_region={getattr(args, 'evaluation_region', None)!r}"
        )


def validate_paper_asset_manifest_is_heldout(manifest: dict[str, Any], metrics_payload: dict[str, Any]) -> None:
    metric_policy = metrics_payload.get("metric_policy", {})
    if manifest.get("evaluation_region") != EVALUATION_REGION_HELDOUT:
        raise StopCondition("paper asset emission refused non held-out metric manifest")
    if metric_policy.get("evaluation_region") != EVALUATION_REGION_HELDOUT:
        raise StopCondition("paper asset emission refused non held-out metric payload")
    rows = list(manifest.get("rows", []))
    payload_rows = list(metrics_payload.get("rows", []))
    if not rows or not payload_rows:
        raise StopCondition("paper asset emission requires held-out metric rows")
    bad_rows = [
        row.get("row_id")
        for row in rows
        if row.get("evaluation_region") != EVALUATION_REGION_HELDOUT or row.get("heldout_only") is not True
    ]
    bad_payload_rows = [
        row.get("row_id")
        for row in payload_rows
        if row.get("evaluation_region") != EVALUATION_REGION_HELDOUT or row.get("heldout_only") is not True
    ]
    if bad_rows or bad_payload_rows:
        raise StopCondition(
            "paper asset emission refused non held-out rows: "
            f"manifest={bad_rows}, payload={bad_payload_rows}"
        )
    bad_contract_rows = [
        row.get("row_id")
        for row in rows
        if row.get("split_contract", {}).get("status") != "ok"
    ]
    bad_payload_contract_rows = [
        row.get("row_id")
        for row in payload_rows
        if row.get("split_contract", {}).get("status") != "ok"
    ]
    if bad_contract_rows or bad_payload_contract_rows:
        raise StopCondition(
            "paper asset emission refused rows without ok split contract: "
            f"manifest={bad_contract_rows}, payload={bad_payload_contract_rows}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path(DEFAULT_PAPER_ROOT))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--coordinate-source", type=Path, default=Path(DEFAULT_COORDINATE_SOURCE))
    parser.add_argument("--full-reference", type=Path, default=Path(DEFAULT_FULL_REFERENCE))
    parser.add_argument("--stitch-patch-size", type=int, default=20)
    parser.add_argument("--upsample-factor", type=int, default=50)
    parser.add_argument("--registration-stability-factor", type=int, default=10)
    parser.add_argument("--border-crop", type=int, default=2)
    parser.add_argument("--eval-offset", type=int, default=4)
    parser.add_argument("--phase-align-method", choices=("plane", "mean"), default="plane")
    parser.add_argument("--frc-sigma", type=float, default=0.0)
    parser.add_argument("--ms-ssim-sigma", type=float, default=1.0)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--validate-references-only", action="store_true")
    parser.add_argument("--emit-paper-assets", action="store_true")
    parser.add_argument("--allow-panel-artifact-exception", action="store_true")
    parser.add_argument("--panel-exception-reason", default="")
    parser.add_argument("--force-stale-lock", action="store_true")
    parser.add_argument(
        "--evaluation-region",
        choices=(EVALUATION_REGION_ALL_SCAN, EVALUATION_REGION_HELDOUT),
        default=EVALUATION_REGION_ALL_SCAN,
    )
    parser.add_argument(
        "--heldout-split-policy",
        choices=(
            HELDOUT_SPLIT_FROM_INDICES,
            HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y,
            HELDOUT_SPLIT_TOP_BY_SORTED_Y,
            HELDOUT_SPLIT_PAPER_TOP_TRAIN_BOTTOM_TEST_BY_HIGH_Y,
        ),
        default=HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y,
    )
    parser.add_argument("--heldout-split-source", type=Path, default=None)
    parser.add_argument("--require-heldout-eval", action="store_true")
    parser.add_argument("--min-post-eval-trim-dim", type=int, default=MIN_POST_EVAL_TRIM_DIM)
    parser.add_argument(
        "--split-nonoverlap-level",
        choices=(SPLIT_NONOVERLAP_COORDINATE_INDICES, SPLIT_NONOVERLAP_OBJECT_FOOTPRINT),
        default=SPLIT_NONOVERLAP_COORDINATE_INDICES,
    )
    parser.add_argument("--heldout-guard-band-pixels", type=int, default=0)
    parser.add_argument("--allow-panel-metric-region-mismatch", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def preserve_reference_validation_only_gate(output_root: Path) -> dict[str, Any] | None:
    output_root = Path(output_root)
    manifest_path = output_root / "fig5_ood_metrics_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    if manifest.get("status") != "reference_validation_only":
        return None
    if manifest.get("metric_evaluation_started") is not False:
        return None

    gate_dir = output_root / "reference_validation_only_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": "fig5_ood_metrics_manifest.json",
        "reference_validation": "fig5_reference_validation.json",
        "source_inventory": "fig5_source_inventory.json",
        "invocation_json": "invocation.json",
        "invocation_sh": "invocation.sh",
    }
    archived: dict[str, Any] = {
        "preserved_before_metric_invocation": True,
        "source_status": manifest.get("status"),
        "source_metric_evaluation_started": manifest.get("metric_evaluation_started"),
        "directory": str(gate_dir),
        "files": {},
    }
    for key, name in files.items():
        source = output_root / name
        if not source.exists():
            continue
        destination = gate_dir / name
        destination.write_bytes(source.read_bytes())
        archived[key] = str(destination)
        archived["files"][key] = str(destination)
    _write_json(gate_dir / "archive_manifest.json", archived)
    return archived


def run_inventory_only(args: argparse.Namespace, lock: OutputLock) -> None:
    validate_heldout_evaluation_requirement(args)
    provenance = collect_provenance(args)
    provenance["status"] = "inventory_only"
    provenance["reference_validation_completed_before_metrics"] = False
    provenance["metric_evaluation_started"] = False
    provenance["lock"] = {
        "pid": lock.pid,
        "replaced_stale_lock": lock.replaced_stale_lock,
        "stale_lock_content": lock.stale_lock_content,
    }
    attach_run_provenance(
        provenance,
        args=args,
        raw_argv=getattr(args, "_raw_argv", None),
        invocation_json_path=getattr(args, "_invocation_json_path", args.output_root / "invocation.json"),
        invocation_sh_path=getattr(args, "_invocation_sh_path", args.output_root / "invocation.sh"),
        invocation_artifact_kind=getattr(args, "_invocation_artifact_kind", "metric_run"),
        include_paper_git=False,
    )
    _write_json(args.output_root / "fig5_source_inventory.json", provenance)
    _write_json(args.output_root / "fig5_ood_metrics_manifest.json", provenance)


def _load_alignment_records(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_heldout_evaluation_requirement(args)
    provenance = collect_provenance(args)
    repo_root = Path(args.repo_root).resolve()
    coordinate_source = (repo_root / args.coordinate_source).resolve()
    full_reference_path = (repo_root / args.full_reference).resolve()
    scan_coords_yx = load_scan_coords_yx(coordinate_source)
    full_reference, full_reference_provenance = load_full_reference(full_reference_path, coordinate_source)
    provenance["full_reference_provenance"] = full_reference_provenance

    rows = build_default_rows(repo_root)
    id_panel_reference_path = next(
        (row.panel_reference_path for row in rows if row.row_id == "id_ptychopinn"),
        None,
    )
    row_entries = {entry["row_id"]: entry for entry in provenance["rows"]}
    records: list[dict[str, Any]] = []
    for row in rows:
        reconstruction = load_complex_array(
            row.reconstruction_path,
            complex_keys=("reconstructed_object",),
            amp_key="reconstructed_amplitude",
            phase_key="reconstructed_phase",
        )
        panel_path = row.panel_reference_path
        validation_row = row
        if panel_path is None and row.row_id == "id_supervised_baseline":
            panel_path = id_panel_reference_path
            reference_validation_scope = "shared_run1084_panel_reference"
            validation_row = StudyRow(
                row_id=row.row_id,
                condition=row.condition,
                model=row.model,
                reconstruction_path=row.reconstruction_path,
                panel_reference_path=panel_path,
                notes=row.notes,
            )
        else:
            reference_validation_scope = "row_specific_panel_reference"
        panel_reference = load_panel_reference(panel_path) if panel_path else None
        panel_reference_inventory = (
            inventory_npz(panel_path) if panel_path is not None and Path(panel_path).exists() else None
        )
        aligned_recon, aligned_gt, alignment_manifest = align_row_for_metrics(
            row=validation_row,
            reconstruction=reconstruction,
            full_reference=full_reference,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=args.stitch_patch_size,
            panel_reference=panel_reference,
            panel_reference_inventory=panel_reference_inventory,
            allow_panel_artifact_exception=args.allow_panel_artifact_exception,
            panel_exception_reason=args.panel_exception_reason,
            reference_validation_scope=reference_validation_scope,
            evaluation_region=getattr(args, "evaluation_region", EVALUATION_REGION_ALL_SCAN),
            heldout_split_policy=getattr(args, "heldout_split_policy", HELDOUT_SPLIT_BOTTOM_BY_SORTED_Y),
            heldout_split_source=getattr(args, "heldout_split_source", None),
            split_nonoverlap_level=getattr(args, "split_nonoverlap_level", SPLIT_NONOVERLAP_COORDINATE_INDICES),
            heldout_guard_band_pixels=getattr(args, "heldout_guard_band_pixels", 0),
            allow_panel_metric_region_mismatch=getattr(args, "allow_panel_metric_region_mismatch", False),
        )
        if row.panel_reference_path is None and panel_path is not None:
            alignment_manifest["panel_reference_path"] = str(panel_path)
            alignment_manifest["panel_reference_note"] = "ID baseline validated against ID PtychoPINN panel reference because no row-specific ID baseline reference NPZ was found."
        alignment_manifest["reference_validation_completed_before_metrics"] = True
        row_entries[row.row_id].update(alignment_manifest)
        records.append(
            {
                "row": row,
                "aligned_recon": aligned_recon,
                "aligned_gt": aligned_gt,
                "manifest_entry": row_entries[row.row_id],
            }
        )
    provenance["reference_validation_completed_before_metrics"] = True
    provenance["metric_evaluation_started"] = False
    provenance["reference_validation_summary"] = {
        "row_count": len(records),
        "passed": all(
            record["manifest_entry"].get("reference_validation_status") in {"passed", "panel_artifact_exception_approved"}
            for record in records
        ),
        "failed_rows": [
            record["row"].row_id
            for record in records
            if record["manifest_entry"].get("reference_validation_status") not in {"passed", "panel_artifact_exception_approved"}
        ],
    }
    return provenance, records


def _reference_validation_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = list(manifest.get("rows", []))
    return {
        "status": manifest.get("status"),
        "reference_validation_completed_before_metrics": manifest.get("reference_validation_completed_before_metrics"),
        "metric_evaluation_started": manifest.get("metric_evaluation_started"),
        "rows": [
            {
                key: row.get(key)
                for key in [
                    "row_id",
                    "condition",
                    "model",
                    "alignment_mode",
                    "evaluation_region",
                    "heldout_only",
                    "heldout_split_policy",
                    "total_coordinate_count",
                    "eval_coordinate_count",
                    "heldout_crop_shape",
                    "split_contract",
                    "primary_registration_mode",
                    "coordinate_aligned_reconstruction_shape",
                    "coordinate_aligned_ground_truth_shape",
                    "panel_reference_path",
                    "reference_validation_scope",
                    "reference_validation_status",
                    "reference_validation_mode",
                    "canonical_cropped_reference_digest",
                    "panel_reference_digest",
                    "max_abs_diff",
                    "max_abs_diff_tolerance",
                    "relative_l2_diff",
                    "relative_l2_diff_tolerance",
                ]
            }
            for row in rows
        ],
    }


def _all_references_passed(manifest: dict[str, Any]) -> bool:
    return all(
        row.get("reference_validation_status") in {"passed", "panel_artifact_exception_approved"}
        for row in manifest.get("rows", [])
    )


def run_validate_references_only(args: argparse.Namespace, lock: OutputLock) -> int:
    manifest, _records = _load_alignment_records(args)
    manifest["status"] = "reference_validation_only" if _all_references_passed(manifest) else "reference_validation_failed"
    manifest["metric_evaluation_started"] = False
    manifest["lock"] = {
        "pid": lock.pid,
        "replaced_stale_lock": lock.replaced_stale_lock,
        "stale_lock_content": lock.stale_lock_content,
    }
    attach_run_provenance(
        manifest,
        args=args,
        raw_argv=getattr(args, "_raw_argv", None),
        invocation_json_path=getattr(args, "_invocation_json_path", args.output_root / "invocation.json"),
        invocation_sh_path=getattr(args, "_invocation_sh_path", args.output_root / "invocation.sh"),
        invocation_artifact_kind=getattr(args, "_invocation_artifact_kind", "metric_run"),
        include_paper_git=False,
    )
    _write_json(args.output_root / "fig5_source_inventory.json", manifest)
    _write_json(args.output_root / "fig5_reference_validation.json", _reference_validation_payload(manifest))
    refresh_output_artifact_status(manifest, output_root=args.output_root, accepted_metrics=False)
    _write_json(args.output_root / "fig5_ood_metrics_manifest.json", manifest)
    return 0 if manifest["status"] == "reference_validation_only" else 1


def metric_policy(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_region = getattr(args, "evaluation_region", EVALUATION_REGION_ALL_SCAN)
    heldout_note = ""
    if evaluation_region == EVALUATION_REGION_HELDOUT:
        heldout_note = (
            f"; held-out test-half spatial crop after coordinate alignment "
            f"using policy={getattr(args, 'heldout_split_policy', None)}"
        )
    return {
        "alignment": f"coordinate crop with align_for_evaluation using Run1084 scan coordinates and stitch_patch_size={args.stitch_patch_size}{heldout_note}",
        "alignment_mode": "coordinate_align_for_evaluation",
        "evaluation_region": evaluation_region,
        "heldout_split_policy": getattr(args, "heldout_split_policy", None),
        "heldout_split_source": str(args.heldout_split_source) if getattr(args, "heldout_split_source", None) else None,
        "split_nonoverlap_level": getattr(args, "split_nonoverlap_level", SPLIT_NONOVERLAP_COORDINATE_INDICES),
        "heldout_guard_band_pixels": int(getattr(args, "heldout_guard_band_pixels", 0)),
        "allow_panel_metric_region_mismatch": bool(getattr(args, "allow_panel_metric_region_mismatch", False)),
        "primary_registration_mode": "none",
        "registration": "no fine registration before primary metrics; magnitude registration is diagnostic sensitivity only",
        "fine_registration_diagnostic": f"upsample_factor={args.upsample_factor}, stability_upsample_factor={args.registration_stability_factor}, border_crop={args.border_crop}",
        "eval_offset": int(args.eval_offset),
        "eval_trim_per_edge": int(args.eval_offset) // 2,
        "min_post_eval_trim_dim": int(getattr(args, "min_post_eval_trim_dim", MIN_POST_EVAL_TRIM_DIM)),
        "phase_policy": f"phase_align_method={args.phase_align_method}; absolute global phase and removed linear phase ramps are not scored",
        "background_policy": "no support/background mask; background pixels included after coordinate crop and eval offset trim",
        "amplitude_policy": "mean scale factor computed on post-eval-trim amplitude arrays; PSNR is derived from MSE on the post-trim arrays",
        "frc_policy": "FRC values are artifact-only unless reference-based FRC succeeds for all rows with the same field of view as core metrics",
    }


def _write_pivot_summary(output_root: Path, reason: str, manifest: dict[str, Any]) -> None:
    rows = list(manifest.get("rows", []))
    evaluated_rows = [row for row in rows if row.get("metric_status") == "ok"]
    unevaluated_rows = [row for row in rows if row.get("metric_status") != "ok"]
    lines = [
        "# Fig. 5 OOD Metrics Pivot Summary",
        "",
        f"Stop condition: {reason}",
        "",
        f"Rows inventoried/aligned: {len(rows)}",
        f"Rows metric-evaluated: {len(evaluated_rows)}",
        "",
        "## Row Status",
        "",
    ]
    for row in rows:
        offset = row.get("registration_offset_norm")
        offset_text = "not run" if offset is None else f"{float(offset):.3f}"
        lines.append(
            f"- {row.get('row_id')}: metric_status={row.get('metric_status', 'not_evaluated')}, "
            f"registration_status={row.get('registration_status', 'not_run')}, "
            f"registration_offset_norm={offset_text}"
        )
    if unevaluated_rows:
        lines.extend(
            [
                "",
                "## Paper-Update Recommendation",
                "",
                "Do not update the manuscript with Fig. 5 metric claims from this stopped run.",
                "Treat the run as exploratory/provenance evidence only until the large OOD PtychoPINN registration offset is reviewed.",
                "Escalate for a human decision before using panel-artifact exception mode, changing the registration policy, changing Fig. 5 source images, or adding a metric table.",
                "A text-only response should narrow the Fig. 5 claim rather than add numeric metrics if the large offset remains scientifically unsafe.",
                "",
            ]
        )
    (Path(output_root) / "pivot_summary.md").write_text("\n".join(lines))


def run_metrics(args: argparse.Namespace, lock: OutputLock) -> None:
    manifest, records = _load_alignment_records(args)
    manifest["lock"] = {
        "pid": lock.pid,
        "replaced_stale_lock": lock.replaced_stale_lock,
        "stale_lock_content": lock.stale_lock_content,
    }
    attach_run_provenance(
        manifest,
        args=args,
        raw_argv=getattr(args, "_raw_argv", None),
        invocation_json_path=getattr(args, "_invocation_json_path", args.output_root / "invocation.json"),
        invocation_sh_path=getattr(args, "_invocation_sh_path", args.output_root / "invocation.sh"),
        invocation_artifact_kind=getattr(args, "_invocation_artifact_kind", "metric_run"),
        include_paper_git=False,
    )
    reference_gate = getattr(args, "_reference_validation_only_gate", None)
    if reference_gate is not None:
        manifest["reference_validation_only_gate"] = reference_gate
    if not _all_references_passed(manifest):
        manifest["status"] = "reference_validation_failed"
        manifest["metric_evaluation_started"] = False
        _write_json(args.output_root / "fig5_source_inventory.json", manifest)
        _write_json(args.output_root / "fig5_reference_validation.json", _reference_validation_payload(manifest))
        refresh_output_artifact_status(manifest, output_root=args.output_root, accepted_metrics=False)
        _write_json(args.output_root / "fig5_ood_metrics_manifest.json", manifest)
        raise StopCondition("reference validation failed before metric evaluation")

    manifest["status"] = "metrics_running"
    manifest["reference_validation_completed_before_metrics"] = True
    manifest["metric_evaluation_started"] = True
    _write_json(args.output_root / "fig5_source_inventory.json", manifest)
    _write_json(args.output_root / "fig5_reference_validation.json", _reference_validation_payload(manifest))
    rows_payload: list[dict[str, Any]] = []
    for record in records:
        row = record["row"]
        payload, metrics_manifest = compute_row_metrics(
            row=row,
            recon_registered=record["aligned_recon"],
            gt_registered=record["aligned_gt"],
            args=args,
        )
        record["manifest_entry"].update(metrics_manifest)
        payload.update(
            {
                "alignment_mode": record["manifest_entry"].get("alignment_mode"),
                "evaluation_region": record["manifest_entry"].get("evaluation_region"),
                "heldout_only": record["manifest_entry"].get("heldout_only"),
                "heldout_split_policy": record["manifest_entry"].get("heldout_split_policy"),
                "heldout_split_source": record["manifest_entry"].get("heldout_split_source"),
                "total_coordinate_count": record["manifest_entry"].get("total_coordinate_count"),
                "eval_coordinate_count": record["manifest_entry"].get("eval_coordinate_count"),
                "heldout_crop_shape": record["manifest_entry"].get("heldout_crop_shape"),
                "split_contract": record["manifest_entry"].get("split_contract"),
                "reference_validation_status": record["manifest_entry"].get("reference_validation_status"),
                "reference_validation_mode": record["manifest_entry"].get("reference_validation_mode"),
                "reference_validation_scope": record["manifest_entry"].get("reference_validation_scope"),
                "reference_validation_completed_before_metrics": True,
                "metric_source_paths": {
                    "reconstruction_path": record["manifest_entry"].get("reconstruction_path"),
                    "coordinate_source": manifest.get("coordinate_source"),
                    "full_reference": manifest.get("full_reference"),
                },
            }
        )
        rows_payload.append(payload)

    for record, payload in zip(records, rows_payload):
        diagnostics = compute_fine_registration_sensitivity(
            row=record["row"],
            aligned_recon=record["aligned_recon"],
            aligned_gt=record["aligned_gt"],
            args=args,
        )
        record["manifest_entry"].update(diagnostics)
        payload.update(diagnostics)

    manifest["status"] = "metrics_complete"
    manifest["metric_policy"] = metric_policy(args)
    write_metrics_artifacts(
        output_root=args.output_root,
        rows=rows_payload,
        manifest=manifest,
        metric_policy=manifest["metric_policy"],
    )


def emit_paper_assets(args: argparse.Namespace, lock: OutputLock) -> None:
    output_root = Path(args.output_root)
    manifest_path = output_root / "fig5_ood_metrics_manifest.json"
    metrics_path = output_root / "fig5_ood_metrics.json"
    table_path = output_root / "fig5_ood_metrics_table.tex"
    if not manifest_path.exists() or not metrics_path.exists() or not table_path.exists():
        raise StopCondition("paper asset emission requires an accepted metric run")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("artifact_acceptance", {}).get("accepted_metrics_artifacts"):
        raise StopCondition("paper asset emission refused unaccepted or partial metrics")
    metrics_payload = json.loads(metrics_path.read_text())
    validate_paper_asset_manifest_is_heldout(manifest, metrics_payload)
    paper_root = Path(args.paper_root)
    data_dir = paper_root / "data"
    tables_dir = paper_root / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_metrics_path = data_dir / "fig5_ood_metrics.json"
    paper_table_path = tables_dir / "fig5_ood_metrics.tex"
    paper_payload = {
        "source_run_root": str(output_root),
        "source_metrics_json": str(metrics_path),
        "source_metrics_csv": str(output_root / "fig5_ood_metrics.csv"),
        "source_manifest": str(manifest_path),
        "metric_policy": metrics_payload.get("metric_policy", {}),
        "rows": metrics_payload.get("rows", []),
    }
    _write_json(paper_metrics_path, paper_payload)
    paper_table_path.write_text(table_path.read_text())

    manifest["paper_asset_emission"] = {
        "status": "complete",
        "paper_metrics_json": str(paper_metrics_path),
        "paper_table_tex": str(paper_table_path),
        "paper_asset_emission_invocation_json": str(getattr(args, "_invocation_json_path", output_root / "paper_asset_emission_invocation" / "invocation.json")),
        "paper_asset_emission_invocation_sh": str(getattr(args, "_invocation_sh_path", output_root / "paper_asset_emission_invocation" / "invocation.sh")),
        "lock_pid": lock.pid,
    }
    manifest["paper_repo_git"] = _git_summary(paper_root)
    _write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    validate_heldout_evaluation_requirement(args)
    args._raw_argv = raw_argv
    invocation_output_dir = (
        args.output_root / "paper_asset_emission_invocation" if args.emit_paper_assets else args.output_root
    )
    args._invocation_artifact_kind = "paper_asset_emission" if args.emit_paper_assets else "metric_run"
    args._reference_validation_only_gate = None
    with acquire_output_lock(args.output_root, force_stale_lock=args.force_stale_lock) as lock:
        from scripts.studies.invocation_logging import write_invocation_artifacts

        if not args.inventory_only and not args.validate_references_only and not args.emit_paper_assets:
            args._reference_validation_only_gate = preserve_reference_validation_only_gate(args.output_root)
        invocation_json_path, invocation_sh_path = write_invocation_artifacts(
            output_dir=invocation_output_dir,
            script_path="scripts/studies/ood_fig5_metrics.py",
            argv=raw_argv,
            parsed_args=vars(args),
            extra={
                "approved_design_path": APPROVED_DESIGN_PATH,
                "plan_path": PLAN_PATH,
                "acquired_lock_pid": lock.pid,
                "replaced_stale_lock": lock.replaced_stale_lock,
                "stale_lock_content": lock.stale_lock_content,
            },
        )
        args._invocation_json_path = invocation_json_path
        args._invocation_sh_path = invocation_sh_path
        if args.inventory_only:
            run_inventory_only(args, lock)
            return 0
        if args.validate_references_only:
            return run_validate_references_only(args, lock)
        if args.emit_paper_assets:
            emit_paper_assets(args, lock)
            return 0
        run_metrics(args, lock)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

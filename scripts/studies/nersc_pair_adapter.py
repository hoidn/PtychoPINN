#!/usr/bin/env python3
"""Adapters for NERSC-style paired HDF5 datasets used in external-raw studies."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import h5py
import numpy as np

PROBE_MODE_POLICY_CHOICES = ("incoherent_aggregate", "first_mode")


def _select_object_guess(obj: np.ndarray) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.ndim == 2:
        return arr.astype(np.complex64)
    if arr.ndim == 3:
        return arr[0].astype(np.complex64)
    raise ValueError(f"Unsupported object dataset shape: {arr.shape}")


def _select_probe_modes(probe: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(probe)
    if arr.ndim == 2:
        return arr[np.newaxis, ...].astype(np.complex64), tuple(arr.shape)
    if arr.ndim == 3:
        return arr.astype(np.complex64), tuple(arr.shape)
    if arr.ndim == 4:
        return arr[0].astype(np.complex64), tuple(arr.shape)
    raise ValueError(f"Unsupported probe dataset shape: {arr.shape}")


def _collapse_probe_guess(
    probe: np.ndarray, *, probe_mode_policy: str = "incoherent_aggregate"
) -> tuple[np.ndarray, dict[str, Any]]:
    if probe_mode_policy not in PROBE_MODE_POLICY_CHOICES:
        raise ValueError(
            f"Unsupported probe_mode_policy='{probe_mode_policy}', "
            f"expected one of {PROBE_MODE_POLICY_CHOICES}."
        )
    probe_modes, source_shape = _select_probe_modes(probe)
    mode_power = np.sum(np.abs(probe_modes) ** 2, axis=(1, 2), dtype=np.float64)
    total_power = float(np.sum(mode_power))
    if total_power > 0.0:
        mode_weights = mode_power / total_power
    else:
        mode_weights = np.zeros_like(mode_power, dtype=np.float64)

    if probe_mode_policy == "first_mode":
        collapsed = probe_modes[0]
    else:
        incoherent_amp = np.sqrt(np.sum(np.abs(probe_modes) ** 2, axis=0, dtype=np.float64))
        phase_mode0 = np.angle(probe_modes[0])
        collapsed = incoherent_amp * np.exp(1j * phase_mode0)

    metadata = {
        "probe_mode_policy": probe_mode_policy,
        "probe_source_shape": [int(x) for x in source_shape],
        "probe_mode_power_weights": mode_weights.astype(np.float64).tolist(),
    }
    return np.asarray(collapsed, dtype=np.complex64), metadata


def _resolve_pixel_size_from_object(handle: h5py.File) -> tuple[float, float]:
    if "object" not in handle:
        raise KeyError("para HDF5 missing required dataset: object")
    object_ds = handle["object"]
    if "pixel_width_m" not in object_ds.attrs or "pixel_height_m" not in object_ds.attrs:
        raise KeyError("object dataset missing required pixel attrs: pixel_width_m/pixel_height_m")
    return float(object_ds.attrs["pixel_width_m"]), float(object_ds.attrs["pixel_height_m"])


def _resolve_object_center_from_object(handle: h5py.File) -> tuple[float, float]:
    if "object" not in handle:
        raise KeyError("para HDF5 missing required dataset: object")
    object_ds = handle["object"]
    if "center_x_m" not in object_ds.attrs or "center_y_m" not in object_ds.attrs:
        raise KeyError("object dataset missing required center attrs: center_x_m/center_y_m")
    return float(object_ds.attrs["center_x_m"]), float(object_ds.attrs["center_y_m"])


def ensure_required_para_attrs(para_h5: Path, *, default_pixel_m: float | None = None) -> None:
    """Ensure para HDF5 probe dataset provides pixel attrs required by downstream consumers."""
    para_h5 = Path(para_h5)
    with h5py.File(para_h5, "r+") as handle:
        if "probe" not in handle:
            raise KeyError("para HDF5 missing required dataset: probe")
        probe_ds = handle["probe"]

        has_w = "pixel_width_m" in probe_ds.attrs
        has_h = "pixel_height_m" in probe_ds.attrs
        if has_w and has_h:
            return

        if default_pixel_m is not None:
            pixel_w = float(default_pixel_m)
            pixel_h = float(default_pixel_m)
        else:
            pixel_w, pixel_h = _resolve_pixel_size_from_object(handle)

        if not has_w:
            probe_ds.attrs["pixel_width_m"] = float(pixel_w)
        if not has_h:
            probe_ds.attrs["pixel_height_m"] = float(pixel_h)


def materialize_pair_working_copy(dp_h5: Path, para_h5: Path, out_dir: Path) -> tuple[Path, Path]:
    """Copy a paired HDF5 bundle into a writable workspace and patch required attrs."""
    dp_h5 = Path(dp_h5)
    para_h5 = Path(para_h5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_copy = out_dir / dp_h5.name
    para_copy = out_dir / para_h5.name
    shutil.copy2(dp_h5, dp_copy)
    shutil.copy2(para_h5, para_copy)
    ensure_required_para_attrs(para_copy)
    return dp_copy, para_copy


def pair_to_external_npz(
    dp_h5: Path,
    para_h5: Path,
    out_npz: Path,
    *,
    probe_mode_policy: str = "incoherent_aggregate",
    metadata_out: dict[str, Any] | None = None,
) -> Path:
    """Convert paired HDF5 inputs into canonical external-raw NPZ keys."""
    dp_h5 = Path(dp_h5)
    para_h5 = Path(para_h5)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dp_h5, "r") as dp_file:
        if "dp" not in dp_file:
            raise KeyError(f"{dp_h5} missing required dataset: dp")
        dp = np.asarray(dp_file["dp"], dtype=np.float32)

    if dp.ndim == 4 and dp.shape[-1] == 1:
        dp = np.squeeze(dp, axis=-1)
    if dp.ndim != 3:
        raise ValueError(f"Expected dp rank-3 [N,H,W], got {dp.shape}")
    diff3d = np.sqrt(np.clip(dp, a_min=0.0, a_max=None)).astype(np.float32)

    with h5py.File(para_h5, "r") as para_file:
        object_guess = _select_object_guess(np.asarray(para_file["object"]))
        probe_guess, probe_meta = _collapse_probe_guess(
            np.asarray(para_file["probe"]),
            probe_mode_policy=probe_mode_policy,
        )
        x_m = np.asarray(para_file["probe_position_x_m"], dtype=np.float64)
        y_m = np.asarray(para_file["probe_position_y_m"], dtype=np.float64)
        pixel_w, pixel_h = _resolve_pixel_size_from_object(para_file)
        center_x_m, center_y_m = _resolve_object_center_from_object(para_file)

    if metadata_out is not None:
        metadata_out.update(probe_meta)

    if x_m.shape != y_m.shape:
        raise ValueError(
            f"Probe position vectors must have same shape, got x={x_m.shape} y={y_m.shape}"
        )
    if x_m.shape[0] != diff3d.shape[0]:
        raise ValueError(
            f"Scan count mismatch: dp has {diff3d.shape[0]} scans but positions have {x_m.shape[0]}"
        )

    xcoords = ((x_m - center_x_m) / float(pixel_w)).astype(np.float64)
    ycoords = ((y_m - center_y_m) / float(pixel_h)).astype(np.float64)
    scan_index = np.arange(diff3d.shape[0], dtype=np.int64)

    np.savez_compressed(
        out_npz,
        xcoords=xcoords,
        ycoords=ycoords,
        xcoords_start=xcoords.copy(),
        ycoords_start=ycoords.copy(),
        diff3d=diff3d,
        probeGuess=probe_guess,
        objectGuess=object_guess,
        scan_index=scan_index,
    )
    return out_npz

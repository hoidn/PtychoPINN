#!/usr/bin/env python3
"""Validate a Ptychodus HDF5 product file against the data contract.

Checks:
- Required root attributes exist
- Required datasets exist
- Probe/object datasets have complex dtype and valid pixel geometry attributes
- Position datasets lengths match; units inferred as meters
- Object layer spacing shape is L-1
- Optional: cross-check coordinates against a source NPZ file using object pixel size

Exit codes:
- 0: PASS (no errors)
- 2: FAIL (one or more errors)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import numpy as np


@dataclass
class Result:
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


REQUIRED_ROOT_ATTRS = [
    "name",
    "comments",
    "detector_object_distance_m",
    "probe_energy_eV",
    "exposure_time_s",
]

REQUIRED_DATASETS = [
    "probe",
    "object",
    "probe_position_indexes",
    "probe_position_x_m",
    "probe_position_y_m",
    "object_layer_spacing_m",
    "loss_values",  # required; can be empty
]


def validate_file(path: Path, source_npz: Path | None = None) -> Result:
    errs: List[str] = []
    warns: List[str] = []

    try:
        f = h5py.File(path, "r")
    except Exception as e:
        return Result([f"cannot open file: {e}"], [])

    with f:
        # Root attributes
        for k in REQUIRED_ROOT_ATTRS:
            if k not in f.attrs:
                errs.append(f"missing root attribute: {k}")

        # Datasets presence
        for k in REQUIRED_DATASETS:
            if k not in f:
                errs.append(f"missing dataset: {k}")

        # Probe checks
        if "probe" in f:
            arr = f["probe"]
            data = arr[()]
            if not np.iscomplexobj(data):
                errs.append("probe must be complex array (complex64/complex128)")
            if "pixel_width_m" not in arr.attrs or "pixel_height_m" not in arr.attrs:
                errs.append("probe missing pixel geometry attrs (pixel_width_m/height_m)")
            else:
                pw = float(arr.attrs["pixel_width_m"]) or 0.0
                ph = float(arr.attrs["pixel_height_m"]) or 0.0
                if pw <= 0 or ph <= 0:
                    errs.append("probe pixel sizes must be > 0")

        # Object checks
        if "object" in f:
            arr = f["object"]
            data = arr[()]
            if not np.iscomplexobj(data):
                errs.append("object must be complex array (complex64/complex128)")
            for ak in ["pixel_width_m", "pixel_height_m", "center_x_m", "center_y_m"]:
                if ak not in arr.attrs:
                    errs.append(f"object missing attribute: {ak}")
            if "object_layer_spacing_m" in f:
                L = data.shape[0] if data.ndim == 3 else 1
                spacing = np.asarray(f["object_layer_spacing_m"], dtype=float)
                if spacing.shape[0] != max(L - 1, 0):
                    errs.append(
                        f"object_layer_spacing_m length {spacing.shape[0]} != L-1 ({L-1})"
                    )

        # Position arrays: shapes and consistency
        for k in ["probe_position_indexes", "probe_position_x_m", "probe_position_y_m"]:
            if k in f:
                if f[k].ndim != 1:
                    errs.append(f"{k} must be 1D array; got {f[k].shape}")
        if all(k in f for k in ["probe_position_indexes", "probe_position_x_m", "probe_position_y_m"]):
            n = f["probe_position_indexes"].shape[0]
            if f["probe_position_x_m"].shape[0] != n or f["probe_position_y_m"].shape[0] != n:
                errs.append("position arrays lengths mismatch")

        # Optional raw_data bundle checks
        if "raw_data" in f:
            grp = f["raw_data"]
            # Check diffraction present when declared as bundle
            if "diffraction" not in grp:
                # Not fatal: may be missing in some bundles
                pass
            else:
                d = grp["diffraction"]
                if d.ndim != 3:
                    errs.append(f"raw_data/diffraction expected 3D, got {d.shape}")
                # Enforce canonical NHW: first dim must match xcoords length
                if "xcoords" in grp and d.ndim == 3:
                    n = grp["xcoords"].shape[0]
                    if d.shape[0] != n:
                        errs.append(
                            f"raw_data/diffraction first axis ({d.shape[0]}) != len(xcoords) ({n})"
                        )
            for k in ["xcoords", "ycoords", "scan_index"]:
                if k not in grp:
                    errs.append(f"raw_data missing {k}")
            # probeGuess/objectGuess may be hard links to root datasets; ok if absent

        # Cross-check against source NPZ (optional)
        if source_npz is not None and "object" in f:
            try:
                z = np.load(source_npz, allow_pickle=True)
            except Exception as e:
                warns.append(f"cannot open npz for cross-check: {e}")
            else:
                try:
                    x_px = z["xcoords"]
                    y_px = z["ycoords"]
                    ox = np.asarray(f["probe_position_x_m"])  # meters
                    oy = np.asarray(f["probe_position_y_m"])  # meters
                    opw = float(f["object"].attrs["pixel_width_m"])
                    oph = float(f["object"].attrs["pixel_height_m"])
                    if not (np.allclose(ox, x_px * opw, rtol=1e-4, atol=1e-9) and np.allclose(oy, y_px * oph, rtol=1e-4, atol=1e-9)):
                        warns.append("coordinate mismatch vs source NPZ (pixel->meter conversion)")
                except KeyError as e:
                    warns.append(f"npz missing key: {e}")

    return Result(errs, warns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a Ptychodus HDF5 product file")
    p.add_argument("product_h5", type=Path, help="Path to .h5/.hdf5 product file")
    p.add_argument(
        "--source-npz",
        type=Path,
        default=None,
        help="Optional NPZ to cross-check coordinates (pixel->meter)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    res = validate_file(args.product_h5, args.source_npz)
    print(f"Validated: {args.product_h5}")
    if args.source_npz:
        print(f"Cross-checked against: {args.source_npz}")
    for w in res.warnings:
        print(f"WARN: {w}")
    for e in res.errors:
        print(f"ERROR: {e}")
    print(f"Summary: {len(res.errors)} errors, {len(res.warnings)} warnings")
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

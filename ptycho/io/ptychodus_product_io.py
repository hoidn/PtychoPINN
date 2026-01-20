from __future__ import annotations

"""
Scaffold for exporting/importing the Ptychodus product format.

This module provides function signatures and a metadata container for writing
and reading Ptychodus "product" files (HDF5) derived from RawData. The full
implementation will adhere to specs/spec-ptycho-interfaces.md and interop with
ptychodus/src/ptychodus/plugins/h5_product_file.py.

Notes
-----
- Coordinates in NPZ/RawData are treated as pixels (relative to object pixels).
- HDF5 product stores coordinates in meters (object/world frame).
- Loss history is intentionally not handled in this initial scaffold.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Avoid runtime tensorflow dependency during import
    from ptycho.raw_data import RawData


@dataclass
class ExportMeta:
    """Ptychodus product metadata for export.

    Fields mirror the HDF5 root attributes and pixel geometry attributes defined
    in specs/data_contracts.md. When values are unavailable from input datasets,
    callers may supply reasonable defaults.
    """

    # Root attributes
    name: str = ""
    comments: str = ""
    detector_distance_m: float = 0.0
    probe_energy_eV: float = 0.0
    exposure_time_s: float = 0.0
    probe_photon_count: float = 0.0
    mass_attenuation_m2_kg: float = 0.0
    tomography_angle_deg: float = 0.0

    # Object pixel geometry (meters)
    object_pixel_width_m: float = 5.0e-8
    object_pixel_height_m: float = 5.0e-8

    # Probe pixel geometry (meters)
    probe_pixel_width_m: float = 1.25e-7
    probe_pixel_height_m: float = 1.25e-7

    # Object center (meters)
    object_center_x_m: float = 0.0
    object_center_y_m: float = 0.0


def export_product_from_rawdata(
    raw: "RawData",
    out_path: Path,
    meta: Optional[ExportMeta] = None,
    include_raw: bool = True,
) -> None:
    """Export a RawData instance to a Ptychodus HDF5 product file.

    Contract
    --------
    - Writes an HDF5 file adhering to specs/data_contracts.md
    - Converts RawData pixel coordinates to meters using object pixel size
    - Writes probe/object arrays and required attributes
    - Does not include losses

    Parameters
    ----------
    raw:
        Source RawData.
    out_path:
        Destination HDF5 path (".h5" or ".hdf5").
    meta:
        Export metadata and pixel geometry. When None, defaults are used.

    Raises
    ------
    NotImplementedError
        This is a scaffold; implementation will be provided in a follow-up step.
    """
    m = meta or ExportMeta()

    # Pull required arrays from the raw-like object
    x_px = np.asarray(getattr(raw, "xcoords"))
    y_px = np.asarray(getattr(raw, "ycoords"))
    scan_index = getattr(raw, "scan_index", None)
    if scan_index is None:
        scan_index = np.zeros_like(x_px, dtype=np.int64)
    else:
        scan_index = np.asarray(scan_index)

    probe = np.asarray(getattr(raw, "probeGuess"))
    obj = np.asarray(getattr(raw, "objectGuess"))

    # Convert pixels → meters using object pixel size
    x_m = x_px * float(m.object_pixel_width_m)
    y_m = y_px * float(m.object_pixel_height_m)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        # Root attributes
        f.attrs["name"] = m.name
        f.attrs["comments"] = m.comments
        f.attrs["detector_object_distance_m"] = float(m.detector_distance_m)
        f.attrs["probe_energy_eV"] = float(m.probe_energy_eV)
        f.attrs["probe_photon_count"] = float(m.probe_photon_count)
        f.attrs["exposure_time_s"] = float(m.exposure_time_s)
        f.attrs["mass_attenuation_m2_kg"] = float(m.mass_attenuation_m2_kg)
        f.attrs["tomography_angle_deg"] = float(m.tomography_angle_deg)

        # Positions
        f.create_dataset("probe_position_indexes", data=scan_index)
        f.create_dataset("probe_position_x_m", data=x_m)
        f.create_dataset("probe_position_y_m", data=y_m)

        # Probe
        d_probe = f.create_dataset("probe", data=probe.astype(np.complex64, copy=False))
        d_probe.attrs["pixel_width_m"] = float(m.probe_pixel_width_m)
        d_probe.attrs["pixel_height_m"] = float(m.probe_pixel_height_m)

        # Object
        d_obj = f.create_dataset("object", data=obj.astype(np.complex64, copy=False))
        d_obj.attrs["center_x_m"] = float(m.object_center_x_m)
        d_obj.attrs["center_y_m"] = float(m.object_center_y_m)
        d_obj.attrs["pixel_width_m"] = float(m.object_pixel_width_m)
        d_obj.attrs["pixel_height_m"] = float(m.object_pixel_height_m)
        f.create_dataset("object_layer_spacing_m", data=np.array([], dtype=np.float64))

        # Losses (write empty datasets for reader compatibility)
        f.create_dataset("loss_values", data=np.array([], dtype=np.float64))
        f.create_dataset("loss_epochs", data=np.array([], dtype=np.int64))

        # Optional raw-data bundle (non-normative extension)
        if include_raw:
            f.attrs["bundle_version"] = "1.0"
            f.attrs["contains_raw_data"] = 1
            g = f.require_group("raw_data")

            # Diffraction as-is if available on raw stub
            diff = getattr(raw, "diff3d", None)
            if diff is None:
                # Some NPZs use 'diffraction' key naming; in CLI we map to diff3d
                diff = getattr(raw, "diffraction", None)
            if diff is not None:
                diff = np.asarray(diff)
                # Canonicalize axis order to (N, H, W): NHW
                original_order = None
                if diff.ndim == 3:
                    n_images = x_px.shape[0]
                    # Find axis whose length matches number of points
                    try:
                        axis = [i for i, d in enumerate(diff.shape) if d == n_images][0]
                    except IndexError:
                        axis = 0  # fallback, leave as-is
                    if axis == 0:
                        original_order = "NHW"
                        diff_c = diff
                    elif axis == 1:
                        original_order = "HNW"
                        diff_c = np.moveaxis(diff, 1, 0)
                    else:  # axis == 2 or fallback
                        original_order = "HWN"
                        diff_c = np.moveaxis(diff, 2, 0)
                else:
                    diff_c = diff
                dset = g.create_dataset(
                    "diffraction",
                    data=diff_c,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                dset.attrs["axis_canonical"] = "NHW"
                if original_order is not None:
                    dset.attrs["original_axis_order"] = original_order

            # Coordinates and indices (store as pixels for fidelity with source NPZ)
            g.create_dataset("xcoords", data=np.asarray(x_px, dtype=np.float64))
            g.create_dataset("ycoords", data=np.asarray(y_px, dtype=np.float64))
            g.create_dataset("scan_index", data=scan_index)

            # Link guesses to root datasets to avoid duplication
            if "probe" in f:
                g["probeGuess"] = f["probe"]
            if "object" in f:
                g["objectGuess"] = f["object"]


def import_product_to_rawdata(in_path: Path) -> "RawData":
    """Import a Ptychodus HDF5 product file into a RawData instance.

    Contract
    --------
    - Reads HDF5 product per specs/data_contracts.md
    - Converts stored meter coordinates back to pixels using object pixel sizes
    - Populates probeGuess/objectGuess; diffraction remains None

    Parameters
    ----------
    in_path:
        Source HDF5 path.

    Returns
    -------
    RawData
        A RawData instance with positions and guesses populated.

    Raises
    ------
    NotImplementedError
        This is a scaffold; implementation will be provided in a follow-up step.
    """
    in_path = Path(in_path)

    with h5py.File(in_path, "r") as f:
        # Read positions
        scan_index = np.asarray(f["probe_position_indexes"], dtype=np.int64)
        x_m = np.asarray(f["probe_position_x_m"], dtype=np.float64)
        y_m = np.asarray(f["probe_position_y_m"], dtype=np.float64)

        # Read probe + its pixel geometry
        probe = np.asarray(f["probe"]).astype(np.complex64, copy=False)
        probe_px_w = float(f["probe"].attrs["pixel_width_m"])
        probe_px_h = float(f["probe"].attrs["pixel_height_m"])
        _ = (probe_px_w, probe_px_h)  # reserved; not needed to build RawData

        # Read object + geometry
        obj = np.asarray(f["object"]).astype(np.complex64, copy=False)
        obj_px_w = float(f["object"].attrs["pixel_width_m"])
        obj_px_h = float(f["object"].attrs["pixel_height_m"])

        # Convert meters → pixels
        x_px = x_m / obj_px_w if obj_px_w != 0 else x_m
        y_px = y_m / obj_px_h if obj_px_h != 0 else y_m

    # Attempt to construct a RawData instance; if unavailable, return a simple shim
    try:
        from ptycho.raw_data import RawData  # local import (tensorflow may be missing)

        # diff3d is not part of product; set to None
        raw = RawData(
            xcoords=x_px,
            ycoords=y_px,
            xcoords_start=x_px,
            ycoords_start=y_px,
            diff3d=None,
            probeGuess=probe,
            scan_index=scan_index,
            objectGuess=obj,
        )
        return raw
    except Exception:
        # Create a lightweight shim with expected attributes
        class _Shim:
            pass

        shim = _Shim()
        shim.xcoords = x_px
        shim.ycoords = y_px
        shim.xcoords_start = x_px
        shim.ycoords_start = y_px
        shim.diff3d = None
        shim.probeGuess = probe
        shim.scan_index = scan_index
        shim.objectGuess = obj
        shim.Y = None
        shim.norm_Y_I = None
        shim.metadata = None
        return shim

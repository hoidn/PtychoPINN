from __future__ import annotations

"""
Scaffold for exporting/importing the Ptychodus product format.

This module provides function signatures and a metadata container for writing
and reading Ptychodus "product" files (HDF5) derived from RawData. The full
implementation will adhere to specs/data_contracts.md and interop with
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
    raw: RawData,
    out_path: Path,
    meta: Optional[ExportMeta] = None,
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
    raise NotImplementedError("export_product_from_rawdata is not implemented yet")


def import_product_to_rawdata(in_path: Path) -> RawData:
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
    raise NotImplementedError("import_product_to_rawdata is not implemented yet")


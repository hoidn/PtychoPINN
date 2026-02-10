from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .contracts import REQUIRED_DP_KEYS, REQUIRED_PARA_KEYS


def validate_hdf5_pair(dp_hdf5: Path, para_hdf5: Path) -> None:
    """Validate a PtychoViT paired HDF5 bundle against required keys and shapes."""
    dp_hdf5 = Path(dp_hdf5)
    para_hdf5 = Path(para_hdf5)

    with h5py.File(dp_hdf5, "r") as dp_file, h5py.File(para_hdf5, "r") as para_file:
        for key in REQUIRED_DP_KEYS:
            if key not in dp_file:
                raise ValueError(f"Missing required dataset '{key}'")
        for key in REQUIRED_PARA_KEYS:
            if key not in para_file:
                raise ValueError(f"Missing required dataset '{key}'")

        dp_arr = np.asarray(dp_file["dp"])
        if dp_arr.ndim != 3:
            raise ValueError("dp must be rank-3 [N,H,W]")
        if not np.issubdtype(dp_arr.dtype, np.floating):
            raise ValueError("dp must be float dtype")

        x = np.asarray(para_file["probe_position_x_m"])
        y = np.asarray(para_file["probe_position_y_m"])
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("probe_position vectors must be rank-1")
        if x.shape[0] != y.shape[0]:
            raise ValueError("probe_position_x_m and probe_position_y_m must have same length")
        if dp_arr.shape[0] != x.shape[0]:
            raise ValueError("dp scan count must match probe position vector length")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("probe positions must be finite")

        for ds_name in ("object", "probe"):
            dset = para_file[ds_name]
            if "pixel_height_m" not in dset.attrs or "pixel_width_m" not in dset.attrs:
                raise ValueError(f"{ds_name} attrs missing pixel size")


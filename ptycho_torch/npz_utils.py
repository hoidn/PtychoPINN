"""NPZ helpers shared across PyTorch workflows."""

from __future__ import annotations

import numpy as np


def read_npy_shape(npy_file) -> tuple[int, ...]:
    """Read the shape from an .npy file handle without loading the array."""
    version = np.lib.format.read_magic(npy_file)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(npy_file)
    elif version == (2, 0):
        shape, _, _ = np.lib.format.read_array_header_2_0(npy_file)
    elif version == (3, 0):
        shape, _, _ = np.lib.format.read_array_header_3_0(npy_file)
    else:
        raise ValueError(f"Unsupported .npy version: {version}")
    return shape

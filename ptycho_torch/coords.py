import numpy as np


def coords_relative_from_nominal(coords: np.ndarray) -> np.ndarray:
    """Convert nominal coords to TF-style relative offsets.

    Expected input shape: (B, 1, 2, C) where C = gridsize**2.
    Uses local_offset_sign = -1 and centers per-group.
    """
    coords_np = np.asarray(coords, dtype=np.float32)
    if coords_np.ndim != 4 or coords_np.shape[1:3] != (1, 2):
        raise ValueError(f"coords must have shape (B, 1, 2, C); got {coords_np.shape}")
    mean = coords_np.mean(axis=3, keepdims=True)
    return -(coords_np - mean)

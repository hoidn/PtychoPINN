"""Coordinate-layout helper for grouped Torch batches."""

import numpy as np


def get_relative_coords(coords_nn, local_offset_sign=-1):
    """Return group centroids and TF-sign relative coordinates."""

    assert np.ndim(coords_nn) == 4
    coords_offsets = np.mean(coords_nn, axis=1)[:, None, :, :]
    return coords_offsets, local_offset_sign * (coords_nn - coords_offsets)

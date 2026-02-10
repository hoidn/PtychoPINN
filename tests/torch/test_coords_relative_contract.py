import numpy as np
import pytest

from ptycho_torch import coords as coords_mod


def test_coords_relative_from_nominal_c1_is_zero():
    coords = np.array([[[[10.0]], [[-5.0]]]], dtype=np.float32)
    coords = coords.reshape(1, 1, 2, 1)
    rel = coords_mod.coords_relative_from_nominal(coords)
    assert np.allclose(rel, 0.0)


def test_coords_relative_from_nominal_c4_matches_tf_sign():
    coords = np.array([
        [[[1.0, 3.0, 5.0, 7.0], [2.0, 0.0, -2.0, 4.0]]]
    ], dtype=np.float32)  # (1, 1, 2, 4)
    mean = coords.mean(axis=3, keepdims=True)
    expected = -(coords - mean)
    rel = coords_mod.coords_relative_from_nominal(coords)
    assert np.allclose(rel, expected)

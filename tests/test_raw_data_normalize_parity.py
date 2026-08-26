"""Parity fixture for the ``np.sum`` swap in ``ptycho.raw_data.normalize_data``.

``normalize_data`` previously reduced the squared diffraction array with
``tf.reduce_sum``; it now uses ``np.sum``.  Both are float64 reductions, so the
only permitted delta is float64 summation order (last-ulp).  This fixture pins
the result against (a) the hand-computed float64 normalization and (b) the
``tf.reduce_sum`` result, with explicit relative tolerances — never exact ``==``.
"""
import numpy as np
import pytest

from ptycho.raw_data import normalize_data


def _reference_normalize(diffraction, N, reducer):
    """Reference normalization using ``reducer(arr, axis=(1, 2))`` for the sum."""
    X_full_norm = np.float32(
        np.sqrt(
            ((N / 2) ** 2)
            / np.mean(reducer(np.square(diffraction.astype(np.float64)), axis=(1, 2)))
        )
    )
    return X_full_norm * diffraction.astype(np.float32, copy=False)


def test_matches_hand_computed_normalization():
    N = 16
    rng = np.random.default_rng(0)
    diffraction = rng.integers(0, 300, size=(4, N, N), dtype=np.uint16)
    result = normalize_data({"diffraction": diffraction}, N)
    expected = _reference_normalize(diffraction, N, np.sum)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_matches_tf_reduce_sum_reference():
    tf = pytest.importorskip("tensorflow")
    N = 16
    rng = np.random.default_rng(1)
    diffraction = rng.uniform(0.0, 300.0, size=(4, N, N)).astype(np.float64)

    result = normalize_data({"diffraction": diffraction}, N)

    def tf_sum(arr, axis):
        return tf.reduce_sum(arr, axis=axis).numpy()

    expected = _reference_normalize(diffraction, N, tf_sum)
    # tf.reduce_sum and np.sum differ only by float64 summation order.
    np.testing.assert_allclose(result, expected, rtol=1e-9)


def test_returns_float32():
    N = 8
    diffraction = np.full((2, N, N), 300, dtype=np.float32)
    result = normalize_data({"diffraction": diffraction}, N)
    assert result.dtype == np.float32

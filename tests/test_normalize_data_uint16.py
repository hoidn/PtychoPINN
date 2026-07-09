"""Regression tests for uint16 overflow in ptycho.raw_data.normalize_data.

normalize_data squares the diffraction array in its own dtype before
computing the mean; for uint16 count data (values up to ~884 in our
counts NPZs) squaring wraps mod 65536, producing a wrong normalization
constant. See docs/findings.md#NORMALIZE-DATA-UINT16-001.
"""
import numpy as np
import pytest

from ptycho.raw_data import normalize_data


def _reference_norm_const(diffraction: np.ndarray, N: int) -> np.ndarray:
    """Reference normalization constant computed in float64 (no overflow)."""
    diffraction64 = diffraction.astype(np.float64)
    return np.sqrt(
        ((N / 2) ** 2) / np.mean(np.sum(diffraction64 ** 2, axis=(1, 2)))
    )


class TestNormalizeDataUint16Overflow:
    def test_uint16_regression_matches_float64_reference(self):
        """A uint16 diffraction array must normalize identically to its
        float64 equivalent; on the buggy code path the in-place uint16
        square wraps mod 65536, producing a mismatched constant."""
        N = 8
        diffraction = np.full((2, 8, 8), 300, dtype=np.uint16)
        dset = {'diffraction': diffraction}

        result = normalize_data(dset, N)

        expected_const = _reference_norm_const(diffraction, N)
        expected = expected_const * diffraction.astype(np.float64)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_uint16_and_float64_inputs_are_invariant(self):
        """Identical values presented as uint16 vs float64 must give
        identical (allclose, tight rtol) normalized outputs."""
        N = 8
        values = np.full((2, 8, 8), 300, dtype=np.uint16)
        dset_uint16 = {'diffraction': values}
        dset_float64 = {'diffraction': values.astype(np.float64)}

        result_uint16 = normalize_data(dset_uint16, N)
        result_float64 = normalize_data(dset_float64, N)

        np.testing.assert_allclose(result_uint16, result_float64, rtol=1e-10)

    def test_float32_input_returns_float32(self):
        """A float32 diffraction array must normalize to float32 output;
        under numpy>=2 (NEP 50) a float64 scalar would otherwise promote
        the product to float64."""
        N = 8
        diffraction = np.full((2, 8, 8), 300, dtype=np.float32)
        dset = {'diffraction': diffraction}

        result = normalize_data(dset, N)

        assert result.dtype == np.float32

    def test_uint16_input_returns_float32(self):
        """A uint16 diffraction array must also normalize to float32
        output, not float64."""
        N = 8
        diffraction = np.full((2, 8, 8), 300, dtype=np.uint16)
        dset = {'diffraction': diffraction}

        result = normalize_data(dset, N)

        assert result.dtype == np.float32

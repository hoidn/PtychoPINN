"""Global gauge quotient: one scale, one phase constant, one linear phase ramp."""
from __future__ import annotations

import math

import numpy as np
import pytest


def _object(shape=(48, 40)) -> np.ndarray:
    rng = np.random.default_rng(20260904)
    y, x = np.indices(shape, dtype=np.float64)
    amplitude = 0.6 + 0.4 * rng.random(shape) + 0.1 * np.sin(x / 3.0)
    phase = 0.9 * np.cos(y / 5.0) * np.sin(x / 7.0) + 0.3 * rng.random(shape)
    return amplitude * np.exp(1j * phase)


def _gauged(truth: np.ndarray, scale: float, offset: float, ramp_x: float, ramp_y: float):
    y, x = np.indices(truth.shape, dtype=np.float64)
    return truth * scale * np.exp(1j * (offset + ramp_x * x + ramp_y * y))


def _partial_mask(shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[4:-6, 3:-5] = True
    mask[10:14, 8:20] = False
    return mask


@pytest.mark.parametrize(
    "scale, offset, ramp_x, ramp_y",
    [
        (0.37, 1.1, 0.021, -0.013),
        (2.4, -2.9, -0.004, 0.009),
        (1.0, 0.0, 0.15, 0.11),  # more than one FFT bin across the array
    ],
)
def test_fit_recovers_a_known_gauge_on_a_partial_mask(scale, offset, ramp_x, ramp_y):
    from ptycho_torch.reconstruction_gauge import fit_global_gauge

    truth = _object()
    reconstruction = _gauged(truth, scale, offset, ramp_x, ramp_y)
    mask = _partial_mask(truth.shape)
    gauge = fit_global_gauge(reconstruction, truth, mask)
    assert gauge.scale == pytest.approx(1.0 / scale, rel=1e-6)
    assert gauge.ramp_x == pytest.approx(-ramp_x, abs=1e-6)
    assert gauge.ramp_y == pytest.approx(-ramp_y, abs=1e-6)
    residual = np.angle(np.exp(1j * (gauge.phase_offset + offset)))
    assert abs(residual) < 1e-6
    restored = gauge.apply(reconstruction)
    assert np.max(np.abs(restored - truth)[mask]) < 1e-6


def test_identity_gauge_for_zero_energy_reconstruction():
    from ptycho_torch.reconstruction_gauge import IDENTITY_GAUGE, fit_global_gauge

    truth = _object()
    gauge = fit_global_gauge(np.zeros_like(truth), truth, np.ones(truth.shape, dtype=bool))
    assert gauge == IDENTITY_GAUGE
    assert np.array_equal(IDENTITY_GAUGE.apply(truth), truth)


def test_jsonable_record_names_the_method_and_ramp_span():
    from ptycho_torch.reconstruction_gauge import fit_global_gauge

    truth = _object()
    reconstruction = _gauged(truth, 1.0, 0.0, 0.02, 0.0)
    gauge = fit_global_gauge(reconstruction, truth, np.ones(truth.shape, dtype=bool))
    record = gauge.to_jsonable(truth.shape)
    assert record["method"] == "global_scale_affine_phase_v1"
    assert set(record) == {
        "method",
        "scale",
        "phase_offset",
        "ramp_x_rad_per_px",
        "ramp_y_rad_per_px",
        "ramp_span_rad",
    }
    assert record["ramp_span_rad"] == pytest.approx(0.02 * (truth.shape[1] - 1), rel=1e-6)
    assert all(math.isfinite(value) for key, value in record.items() if key != "method")


def test_rejects_shape_and_mask_mismatch():
    from ptycho_torch.reconstruction_gauge import GaugeError, fit_global_gauge

    truth = _object()
    with pytest.raises(GaugeError):
        fit_global_gauge(truth[:-1], truth, np.ones(truth.shape, dtype=bool))
    with pytest.raises(GaugeError):
        fit_global_gauge(truth, truth, np.ones(truth.shape, dtype=np.uint8))

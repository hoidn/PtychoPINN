"""One scoring door for the reconstruction metric contract."""
from __future__ import annotations

import json

import numpy as np
import pytest


def _truth(shape=(40, 40)) -> np.ndarray:
    rng = np.random.default_rng(4)
    y, x = np.indices(shape, dtype=np.float64)
    amplitude = 0.7 + 0.3 * rng.random(shape) + 0.05 * np.sin(x / 2.0)
    phase = 0.8 * np.sin(y / 4.0) * np.cos(x / 6.0) + 0.2 * rng.random(shape)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def _anchor(shape):
    return {
        "scan_com": [shape[1] / 2.0, shape[0] / 2.0],
        "canvas_shape": list(shape),
        "canvas_origin_offset": [0.0, 0.0],
        "truth_origin": [0, 0],
    }


def _inputs():
    truth = _truth()
    rows, cols = np.indices(truth.shape, dtype=np.float64)
    canvas = 2.5 * truth * np.exp(1j * (0.4 + 0.02 * cols - 0.01 * rows))
    weights = np.full(truth.shape, 4.0, dtype=np.float32)
    weights[:3, :] = 0.05  # below 10% of the single-frame probe peak
    probe = np.ones((4, 4), dtype=np.complex64)
    return canvas, weights, _anchor(truth.shape), truth, probe


def test_score_canvas_returns_the_contract_bundle_in_the_gauge_quotient():
    from ptycho_torch.reconstruction_scoring import SCORE_KEYS, score_canvas

    canvas, weights, anchor, truth, probe = _inputs()
    scored = score_canvas(canvas, weights, anchor, truth, probe)
    assert set(scored) == set(SCORE_KEYS)
    assert scored["amp_mae"] == pytest.approx(0.0, abs=1e-6)
    assert scored["phase_wrapped_mae"] == pytest.approx(0.0, abs=1e-6)
    assert scored["amplitude_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert scored["phase_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert scored["absolute_amp_mae"] == pytest.approx(
        float(np.mean(1.5 * np.abs(truth[3:])))
    )
    assert scored["amp_mean_ratio"] == pytest.approx(2.5, rel=1e-5)
    assert scored["gauge"]["scale"] == pytest.approx(0.4, rel=1e-5)
    assert scored["gauge"]["ramp_x_rad_per_px"] == pytest.approx(-0.02, abs=1e-6)
    assert scored["valid_pixel_count"] == 37 * 40
    assert scored["amplitude_frc50_frequency"] == pytest.approx(0.495)
    assert len(scored["frc_curves"]["frequency"]) == 50
    json.dumps(scored, allow_nan=False)


def test_score_canvas_honours_an_explicit_metric_support():
    from ptycho_torch.reconstruction_scoring import score_canvas

    canvas, weights, anchor, truth, probe = _inputs()
    support = np.zeros(truth.shape, dtype=bool)
    support[10:30, 10:30] = True
    scored = score_canvas(canvas, weights, anchor, truth, probe, metric_support=support)
    assert scored["valid_pixel_count"] == 400
    assert scored["amp_mae"] == pytest.approx(0.0, abs=1e-6)


def test_score_canvas_rejects_invalid_weights():
    from ptycho_torch.reconstruction_scoring import score_canvas

    canvas, weights, anchor, truth, probe = _inputs()
    weights = weights.copy()
    weights[5, 5] = -1.0
    with pytest.raises(ValueError):
        score_canvas(canvas, weights, anchor, truth, probe)

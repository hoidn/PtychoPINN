"""Validation-harness tests for the BRDT operator.

These tests exercise the small-grid validation helpers that produce the
``operator_validation.json`` artifact. The full driver run is exercised in
its entirety so that any regression in the helpers, oracles, or JSON
schema is caught here without needing to run the longer module entry
point.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.studies.born_rytov_dt.validate_operator import (
    DEFAULT_JSON_PATH,
    check_analytic_phantom,
    check_cpu_dtype_reproducibility,
    check_cuda_reproducibility,
    check_direct_born_integral,
    check_gradcheck,
    check_numpy_consistency,
    check_odtbrain_inverse_consistency,
    direct_born_integral_2d,
    free_space_propagate,
    gaussian_phantom,
    numpy_reimplementation,
    run_all,
)


# ----------------------------------------------------------------------
# Helper unit tests
# ----------------------------------------------------------------------
def test_gaussian_phantom_shape_and_amplitude():
    q = gaussian_phantom(16, (8.0, 8.0), 2.0, 0.5)
    assert q.shape == (16, 16)
    assert q[8, 8] == pytest.approx(0.5)
    assert q.min() >= 0.0


def test_direct_born_integral_finite_and_zero_for_zero_object():
    q = np.zeros((16, 16))
    out = direct_born_integral_2d(q, np.array([0.0]), 16, 4.0, z_detector=20.0)
    assert out.shape == (1, 16)
    assert np.allclose(out, 0)


def test_free_space_propagator_zero_distance_is_identity():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((4, 16)) + 1j * rng.standard_normal((4, 16))
    out = free_space_propagate(u, k_m=0.5, z_target=0.0)
    # propagator at z=0 is the propagating-band projector; not strictly
    # identity because evanescent components are zeroed. For broadband
    # input we expect agreement within the band.
    band = np.fft.fft(u, axis=-1)
    kx = 2.0 * math.pi * np.fft.fftfreq(16, d=1.0)
    band_filtered = band * (np.abs(kx) < 0.5)
    expected = np.fft.ifft(band_filtered, axis=-1)
    assert np.allclose(out, expected, atol=1e-12)


def test_numpy_reimplementation_matches_torch_within_tight_tol():
    N = 16
    D = 16
    angles = np.array([0.0, 0.5])
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((N, N)) * 0.05).astype(np.float64)
    np_out = numpy_reimplementation(q, angles, D, wavelength_px=4.0, normalize="odtbrain_compatible")
    from ptycho_torch.physics import BornRytovForward2D

    op = BornRytovForward2D(
        grid_size=N,
        detector_size=D,
        angles=torch.tensor(angles, dtype=torch.float64),
        wavelength_px=4.0,
        medium_ri=1.333,
        normalize="odtbrain_compatible",
    )
    out = op(torch.from_numpy(q).double().unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
    op_complex = out[..., 0] + 1j * out[..., 1]
    rel = np.linalg.norm(op_complex - np_out) / (np.linalg.norm(np_out) + 1e-30)
    assert rel < 1e-6


# ----------------------------------------------------------------------
# Individual check tests
# ----------------------------------------------------------------------
def test_check_numpy_consistency_passes():
    rng = np.random.default_rng(0)
    result = check_numpy_consistency(rng)
    assert result.status == "pass"
    assert result.metric is not None
    assert result.metric < (result.tolerance or 0)
    assert result.sample_count >= 1


def test_check_direct_born_integral_passes_within_loose_tolerance():
    rng = np.random.default_rng(0)
    result = check_direct_born_integral(rng)
    assert result.status == "pass"
    assert result.metric is not None
    assert result.metric < (result.tolerance or 0)
    # explicit tolerance must be recorded
    assert result.tolerance is not None and result.tolerance >= 0.5


def test_check_analytic_phantom_finite_and_nontrivial():
    result = check_analytic_phantom()
    assert result.status == "pass"
    assert result.details["all_finite"] is True
    assert result.details["nontrivial_output"] is True


def test_check_gradcheck_succeeds():
    result = check_gradcheck()
    assert result.status == "pass", result.details


def test_check_cpu_dtype_reproducibility_passes():
    rng = np.random.default_rng(0)
    result = check_cpu_dtype_reproducibility(rng)
    assert result.status == "pass"
    assert result.metric is not None
    assert result.metric < (result.tolerance or 0)


def test_check_cuda_reproducibility_recorded_either_way():
    rng = np.random.default_rng(0)
    result = check_cuda_reproducibility(rng)
    assert result.status in ("pass", "skipped")
    if result.status == "skipped":
        assert result.details.get("reason") == "cuda_unavailable"
    else:
        assert result.metric is not None
        assert result.metric < (result.tolerance or 0)


def test_check_odtbrain_inverse_consistency_records_skip_reason():
    result = check_odtbrain_inverse_consistency()
    assert result.status == "skipped"
    assert result.details.get("reason") == "dependency_unavailable"


def test_check_odtbrain_inverse_consistency_uses_installed_package(monkeypatch: pytest.MonkeyPatch):
    fake_module = types.ModuleType("odtbrain")
    fake_module.__version__ = "test-double"

    def fake_backpropagate_2d(
        uSin,
        angles,
        res,
        nm,
        lD=0,
        coords=None,
        weight_angles=True,
        onlyreal=False,
        padding=True,
        padval=0,
        count=None,
        max_count=None,
        verbose=0,
    ):
        assert uSin.shape[0] == 16
        return np.zeros((uSin.shape[-1], uSin.shape[-1]), dtype=np.complex128)

    fake_module.backpropagate_2d = fake_backpropagate_2d
    monkeypatch.setitem(sys.modules, "odtbrain", fake_module)

    result = check_odtbrain_inverse_consistency()
    assert result.status in ("pass", "fail")
    assert result.sample_count == 16
    assert result.details["odtbrain_version"] == "test-double"
    assert result.details["used_api"] == "backpropagate_2d"


# ----------------------------------------------------------------------
# Driver tests
# ----------------------------------------------------------------------
def test_run_all_produces_expected_schema(tmp_path: Path):
    log_path = tmp_path / "validate.log"
    payload = run_all(seed=0, log_path=log_path)

    # Top-level fields
    for key in (
        "schema_version",
        "operator",
        "operator_identity",
        "environment",
        "checks",
        "verdict",
        "downstream_authorization",
        "known_limits",
    ):
        assert key in payload, f"missing top-level key {key}"

    assert payload["schema_version"] == "1.0"
    assert payload["verdict"] in ("pass", "pass_with_documented_limits", "fail")

    # Operator contract fields the report and downstream items rely on
    contract = payload["operator"]
    for key in (
        "module",
        "class",
        "mode",
        "normalize",
        "grid_size",
        "detector_size",
        "wavelength_px",
        "medium_ri",
        "k_m",
        "angle_count",
        "coordinate_convention",
        "detector_frequency_convention",
        "ewald_sampling",
        "fft_normalization",
        "output_layout",
    ):
        assert key in contract, f"missing operator contract key {key}"

    # All required validation families present
    names = {c["name"] for c in payload["checks"]}
    assert {
        "numpy_consistency",
        "direct_born_integral",
        "analytic_phantom",
        "gradcheck",
        "cpu_dtype_reproducibility",
        "cuda_reproducibility",
        "odtbrain_inverse_consistency",
    } == names

    # Each check has the expected keys
    for c in payload["checks"]:
        assert set(c.keys()) >= {"name", "status", "sample_count", "tolerance", "metric", "details"}
        assert c["status"] in ("pass", "fail", "skipped")

    # Identity provenance fields
    identity = payload["operator_identity"]
    for key in (
        "git_sha",
        "git_dirty",
        "execution_command",
        "started_utc",
        "finished_utc",
        "elapsed_seconds",
        "seed",
    ):
        assert key in identity

    # Downstream authorization carries the next-item and may_proceed flag
    auth = payload["downstream_authorization"]
    assert auth["next_item"] == "2026-04-29-brdt-dataset-preflight"
    assert auth["may_proceed"] is True
    direct_tol = next(c["tolerance"] for c in payload["checks"] if c["name"] == "direct_born_integral")
    assert f"{direct_tol:.1f}" in payload["known_limits"][0]
    assert "0.5" not in payload["known_limits"][0]


def test_run_all_writes_log_lines(tmp_path: Path):
    log_path = tmp_path / "log.txt"
    run_all(seed=0, log_path=log_path)
    text = log_path.read_text()
    assert "running numpy_consistency" in text
    assert "running direct_born_integral" in text


def test_default_json_path_under_artifact_root():
    # Sanity: the default path lives under the canonical artifact root.
    p = DEFAULT_JSON_PATH
    parts = p.parts
    assert ".artifacts" in parts
    assert "2026-04-29-brdt-operator-validation" in parts
    assert p.name == "operator_validation.json"


def test_artifact_json_is_present_and_valid():
    """The committed artifact must be syntactically valid and pass-verdicted.

    This guards against accidental drift between the harness output and
    the stored artifact when the harness is re-run.
    """
    if not DEFAULT_JSON_PATH.exists():
        pytest.skip("operator_validation.json not yet generated for this checkout")
    payload = json.loads(DEFAULT_JSON_PATH.read_text())
    assert payload["verdict"] in ("pass", "pass_with_documented_limits")
    assert payload["downstream_authorization"]["may_proceed"] is True
    direct_tol = next(c["tolerance"] for c in payload["checks"] if c["name"] == "direct_born_integral")
    assert f"{direct_tol:.1f}" in payload["known_limits"][0]
    assert "0.5" not in payload["known_limits"][0]

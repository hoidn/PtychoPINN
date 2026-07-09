import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/studies"))

import make_dose_ladder_datasets as D  # noqa: E402
import make_synthetic_truth_datasets as M  # noqa: E402


def test_to_counts_default_matches_previous_output():
    """to_counts(amp) with no target_mean_count kwarg must reproduce the exact
    pre-fix arithmetic (round(amp**2 * 108/mean(amp**2)) -> uint16) on a fixed
    deterministic array -- the default-108 path is byte-for-byte unchanged."""
    amp = np.array([[0.5, 1.0], [0.25, 0.75]], dtype=np.float64)
    intensity = amp ** 2
    expected_S = 108.0 / intensity.mean()
    expected = np.round(intensity * expected_S).astype(np.uint16)

    result = M.to_counts(amp)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint16


def test_to_counts_accepts_explicit_target_mean_count():
    amp = np.array([[0.5, 1.0], [0.25, 0.75]], dtype=np.float64)
    intensity = amp ** 2
    target = 432.0
    expected = np.round(intensity * (target / intensity.mean())).astype(np.uint16)

    result = M.to_counts(amp, target_mean_count=target)

    np.testing.assert_array_equal(result, expected)


def test_to_counts_raises_valueerror_on_saturation():
    """A single dominant pixel among many near-zero pixels drives the scaled
    max far past 65535 -- must raise before casting to uint16 (which would
    otherwise silently wrap), not assert or overflow."""
    amp = np.full((1, 10, 10), 1e-6, dtype=np.float64)
    amp[0, 0, 0] = 1.0  # single dominant pixel

    with pytest.raises(ValueError, match="65535"):
        M.to_counts(amp, target_mean_count=1000.0)


def test_to_counts_saturation_message_names_max_and_target():
    amp = np.full((1, 10, 10), 1e-6, dtype=np.float64)
    amp[0, 0, 0] = 1.0

    with pytest.raises(ValueError) as excinfo:
        M.to_counts(amp, target_mean_count=1000.0)

    msg = str(excinfo.value)
    assert "1000" in msg
    assert "65535" in msg


def test_extract_coords_carries_through_source_values(tmp_path):
    xcoords = np.array([1.5, 2.5, 3.5])
    ycoords = np.array([4.5, 5.5, 6.5])
    npz_path = tmp_path / "fake_source.npz"
    np.savez(npz_path, xcoords=xcoords, ycoords=ycoords)

    xc, yc = D.extract_coords(npz_path)

    np.testing.assert_array_equal(xc, xcoords)
    np.testing.assert_array_equal(yc, ycoords)


def test_build_split_measured_dose_arithmetic_matches_counts_sum(tmp_path, monkeypatch):
    """Tiny end-to-end build: photons/image reported in the returned info must
    equal the actual per-image sum of the saved counts array (measured dose,
    not an assumed constant)."""
    tiny_n = 8
    monkeypatch.setattr(D, "N", tiny_n)
    monkeypatch.setattr(M, "DS_DIR", tmp_path)

    obj_res = 64
    rng = np.random.default_rng(0)
    amp = 0.3 + 0.7 * rng.random((obj_res, obj_res))
    phase = 0.5 * (2.0 * rng.random((obj_res, obj_res)) - 1.0)
    obj = (amp * np.exp(1j * phase)).astype(np.complex64)
    probe = np.ones((tiny_n, tiny_n), dtype=np.complex64)

    xcoords = np.array([16.0, 20.0, 24.0, 16.0, 20.0, 24.0, 16.0, 20.0, 24.0])
    ycoords = np.array([16.0, 16.0, 16.0, 20.0, 20.0, 20.0, 24.0, 24.0, 24.0])
    source_npz = tmp_path / "fake_source_train.npz"
    np.savez(source_npz, xcoords=xcoords, ycoords=ycoords)

    info = D.build_split("train", source_npz, 432.0, obj, probe)

    out_path = Path(info["path"])
    assert out_path.exists()
    with np.load(out_path) as d:
        counts = d["diff3d"].astype(np.float64)
    photons = counts.sum(axis=(1, 2))

    assert info["photons_per_image"]["mean"] == pytest.approx(photons.mean(), abs=0.05)
    assert info["photons_per_image"]["min"] == pytest.approx(photons.min(), abs=0.05)
    assert info["photons_per_image"]["max"] == pytest.approx(photons.max(), abs=0.05)
    assert info["counts_mean"] == pytest.approx(counts.mean(), abs=0.005)

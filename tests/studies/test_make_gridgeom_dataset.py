import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/studies"))

import make_gridgeom_dataset as G  # noqa: E402
import make_synthetic_truth_datasets as M  # noqa: E402


def _write_fake_gs2_npz(path: Path, centers, member_pitch=4.0):
    """Tiny synthetic gs2-grouped npz: coords_nominal (relative, +-member_pitch/2
    pattern) and coords_offsets (absolute group centers), matching the frozen
    gate file's grouped format (n_groups, 1, 2, 4)."""
    n_groups = len(centers)
    half = member_pitch / 2.0
    coords_nominal = np.zeros((n_groups, 1, 2, 4), dtype=np.float32)
    coords_nominal[:, 0, 0, :] = [half, -half, half, -half]
    coords_nominal[:, 0, 1, :] = [half, half, -half, -half]
    coords_offsets = np.zeros((n_groups, 1, 2, 4), dtype=np.float32)
    for i, (cx, cy) in enumerate(centers):
        coords_offsets[i, 0, 0, :] = cx
        coords_offsets[i, 0, 1, :] = cy
    np.savez(path, coords_nominal=coords_nominal, coords_offsets=coords_offsets)


def test_extract_unique_scan_positions_recovers_dense_grid(tmp_path):
    centers = [(10, 10), (10, 18), (18, 10), (18, 18)]
    npz_path = tmp_path / "fake_gate.npz"
    _write_fake_gs2_npz(npz_path, centers, member_pitch=4.0)

    xc, yc = G.extract_unique_scan_positions(npz_path)

    assert sorted(np.unique(xc).tolist()) == [8.0, 12.0, 16.0, 20.0]
    assert sorted(np.unique(yc).tolist()) == [8.0, 12.0, 16.0, 20.0]


def test_verify_grid_pitch_passes_at_4px():
    xc = np.array([8.0, 12.0, 16.0, 20.0])
    assert G.verify_grid_pitch(xc, 4.0) == 4.0


def test_verify_grid_pitch_raises_on_mismatch():
    xc = np.array([8.0, 11.0, 14.0, 17.0])
    with pytest.raises(AssertionError):
        G.verify_grid_pitch(xc, 4.0)


def test_measure_photons_per_image_reports_mean_min_max():
    diff3d = np.array(
        [np.full((2, 2), 10, dtype=np.uint16), np.full((2, 2), 20, dtype=np.uint16)]
    )
    dose = G.measure_photons_per_image(diff3d)
    assert dose == {"mean": 60.0, "min": 40.0, "max": 80.0}


def test_build_split_emits_flat_npz_matching_lines_contract(tmp_path, monkeypatch):
    """Tiny (N=8, 3x3 grid) end-to-end invocation, entirely synthetic -- no real
    artifacts, no GPU. Verifies key/dtype/shape contract mirrors
    lines_N128_*.npz and that grid pitch is asserted correctly."""
    # Note: the >=1e6 photons/image dose floor is enforced in main() against
    # info["photons_per_image"], not inside build_split -- a tiny synthetic
    # probe/object at this scale cannot clear that production dose floor
    # without saturating uint16 (the diffraction peak-to-mean ratio is much
    # higher at N=8 than at the real N=128 production scale). This test
    # exercises the key/dtype/shape contract and pitch check only.
    tiny_n = 8
    monkeypatch.setattr(G, "N", tiny_n)
    monkeypatch.setattr(M, "DS_DIR", tmp_path)

    centers = [(x, y) for x in (16.0, 20.0, 24.0) for y in (16.0, 20.0, 24.0)]
    gate_npz = tmp_path / "fake_gate_train.npz"
    _write_fake_gs2_npz(gate_npz, centers, member_pitch=4.0)

    obj_res = 64
    rng = np.random.default_rng(0)
    amp = 0.3 + 0.7 * rng.random((obj_res, obj_res))
    phase = 0.5 * (2.0 * rng.random((obj_res, obj_res)) - 1.0)
    obj = (amp * np.exp(1j * phase)).astype(np.complex64)
    probe = np.ones((tiny_n, tiny_n), dtype=np.complex64)

    info = G.build_split("train", gate_npz, obj, probe)

    assert info["grid_pitch_px"] == 4.0
    out_path = Path(info["path"])
    assert out_path.exists()

    with np.load(out_path) as d:
        assert set(d.keys()) >= {
            "xcoords", "ycoords", "xcoords_start", "ycoords_start",
            "diff3d", "probeGuess", "objectGuess", "scan_index",
            "ground_truth_patches",
        }
        n = d["xcoords"].shape[0]
        assert d["ycoords"].shape == (n,)
        assert d["diff3d"].shape == (n, tiny_n, tiny_n)
        assert d["diff3d"].dtype == np.uint16
        assert d["probeGuess"].shape == (tiny_n, tiny_n)
        assert d["objectGuess"].shape == (obj_res, obj_res)
        assert d["scan_index"].shape == (n,)
        assert d["ground_truth_patches"].shape == (n, tiny_n, tiny_n, 1)

    assert info["photons_per_image"]["mean"] >= 1.0

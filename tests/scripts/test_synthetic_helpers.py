import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ptycho import params  # noqa: E402
from ptycho.raw_data import RawData  # noqa: E402
from scripts.simulation import synthetic_helpers as helpers  # noqa: E402


def test_make_lines_object_seeded_restores_data_source():
    original_source = params.get("data_source")
    obj_a = helpers.make_lines_object(16, seed=123)
    obj_b = helpers.make_lines_object(16, seed=123)

    assert obj_a.shape == (16, 16)
    assert obj_a.dtype == np.complex64
    assert np.array_equal(obj_a, obj_b)
    assert params.get("data_source") == original_source


def test_make_probe_custom_loads_probe(tmp_path):
    N = 8
    probe = (np.ones((N, N)) + 1j).astype(np.complex64)
    probe_path = tmp_path / "probe.npz"
    np.savez(probe_path, probeGuess=probe)

    loaded = helpers.make_probe(N, mode="custom", path=probe_path)
    assert loaded.shape == (N, N)
    assert loaded.dtype == np.complex64


def test_make_probe_custom_requires_key(tmp_path):
    probe_path = tmp_path / "probe.npz"
    np.savez(probe_path, not_probe=np.ones((4, 4)))

    with pytest.raises(KeyError):
        helpers.make_probe(4, mode="custom", path=probe_path)


def test_simulate_nongrid_seeded():
    N = 16
    object_guess = np.ones((32, 32), dtype=np.complex64)
    probe_guess = np.ones((N, N), dtype=np.complex64)

    raw_a = helpers.simulate_nongrid_raw_data(
        object_guess,
        probe_guess,
        N=N,
        n_images=4,
        nphotons=1e3,
        seed=11,
        buffer=5.0,
    )
    raw_b = helpers.simulate_nongrid_raw_data(
        object_guess,
        probe_guess,
        N=N,
        n_images=4,
        nphotons=1e3,
        seed=11,
        buffer=5.0,
    )

    assert isinstance(raw_a, RawData)
    assert raw_a.diff3d.shape == (4, N, N)
    assert np.array_equal(raw_a.xcoords, raw_b.xcoords)
    assert np.array_equal(raw_a.ycoords, raw_b.ycoords)

    with pytest.raises(ValueError):
        helpers.simulate_nongrid_raw_data(
            object_guess,
            probe_guess,
            N=N,
            n_images=2,
            nphotons=1e3,
            seed=1,
            sim_gridsize=2,
        )


def test_split_raw_data_by_axis():
    n_images = 6
    N = 4
    diff3d = np.zeros((n_images, N, N), dtype=np.float32)
    probe = np.ones((N, N), dtype=np.complex64)
    xcoords = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    ycoords = np.array([10, 9, 8, 7, 6, 5], dtype=float)
    raw_data = RawData.from_coords_without_pc(
        xcoords,
        ycoords,
        diff3d,
        probe,
        scan_index=np.zeros(n_images, dtype=int),
    )

    train_raw, test_raw = helpers.split_raw_data_by_axis(
        raw_data,
        split_fraction=0.5,
        axis="y",
    )

    assert train_raw.diff3d.shape[0] == 3
    assert test_raw.diff3d.shape[0] == 3
    assert test_raw.ycoords.min() >= train_raw.ycoords.max()

    with pytest.raises(ValueError):
        helpers.split_raw_data_by_axis(raw_data, split_fraction=0.0, axis="y")
    with pytest.raises(ValueError):
        helpers.split_raw_data_by_axis(raw_data, split_fraction=0.5, axis="z")

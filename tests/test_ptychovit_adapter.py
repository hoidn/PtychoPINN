from pathlib import Path

import h5py
import numpy as np


def _write_grid_lines_like_npz(path: Path) -> None:
    diffraction = np.abs(np.random.default_rng(0).normal(size=(5, 64, 64))).astype(np.float32)
    yy_ground_truth = (np.ones((1, 392, 392)) + 1j * np.ones((1, 392, 392))).astype(np.complex64)
    probe_guess = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    coords_nominal = np.stack(
        [np.linspace(-2, 2, 5, dtype=np.float32), np.linspace(-1, 1, 5, dtype=np.float32)],
        axis=1,
    )
    coords_offsets = coords_nominal[:, None, :, None]
    np.savez(
        path,
        diffraction=diffraction,
        YY_ground_truth=yy_ground_truth,
        probeGuess=probe_guess,
        coords_nominal=coords_nominal,
        coords_offsets=coords_offsets,
    )


def test_contract_defines_required_hdf5_keys():
    from ptycho.interop.ptychovit.contracts import REQUIRED_DP_KEYS, REQUIRED_PARA_KEYS

    assert "dp" in REQUIRED_DP_KEYS
    assert "object" in REQUIRED_PARA_KEYS
    assert "probe" in REQUIRED_PARA_KEYS
    assert "probe_position_x_m" in REQUIRED_PARA_KEYS
    assert "probe_position_y_m" in REQUIRED_PARA_KEYS


def test_convert_grid_lines_npz_to_ptychovit_hdf5(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "test.npz"
    _write_grid_lines_like_npz(npz_path)

    out = convert_npz_split_to_hdf5_pair(
        npz_path=npz_path,
        out_dir=tmp_path,
        object_name="object_a",
        pixel_size_m=1.0e-9,
    )
    assert out.dp_hdf5.exists()
    assert out.para_hdf5.exists()

    with h5py.File(out.dp_hdf5, "r") as dp_file:
        assert "dp" in dp_file
    with h5py.File(out.para_hdf5, "r") as para_file:
        assert "object" in para_file
        assert "probe" in para_file
        assert "probe_position_x_m" in para_file
        assert "probe_position_y_m" in para_file


def test_convert_accepts_coords_nominal_n_1_2_1_layout(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "layout.npz"
    diffraction = np.abs(np.random.default_rng(0).normal(size=(5, 64, 64))).astype(np.float32)
    yy_ground_truth = (np.ones((1, 392, 392)) + 1j * np.ones((1, 392, 392))).astype(np.complex64)
    probe_guess = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    coords = np.zeros((5, 1, 2, 1), dtype=np.float32)
    coords[:, 0, 0, 0] = np.linspace(-2, 2, 5, dtype=np.float32)  # y
    coords[:, 0, 1, 0] = np.linspace(-1, 1, 5, dtype=np.float32)  # x
    np.savez(
        npz_path,
        diffraction=diffraction,
        YY_ground_truth=yy_ground_truth,
        probeGuess=probe_guess,
        coords_nominal=coords,
        coords_offsets=coords,
    )

    out = convert_npz_split_to_hdf5_pair(
        npz_path=npz_path,
        out_dir=tmp_path,
        object_name="layout",
        pixel_size_m=1.0e-9,
    )
    with h5py.File(out.para_hdf5, "r") as para_file:
        x = np.asarray(para_file["probe_position_x_m"])
        y = np.asarray(para_file["probe_position_y_m"])
    assert x.shape == (5,)
    assert y.shape == (5,)


def test_convert_rejects_local_only_coords_without_offsets(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "missing_offsets.npz"
    diffraction = np.abs(np.random.default_rng(0).normal(size=(5, 64, 64))).astype(np.float32)
    yy_ground_truth = (np.ones((1, 392, 392)) + 1j * np.ones((1, 392, 392))).astype(np.complex64)
    probe_guess = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    coords_nominal = np.zeros((5, 1, 2, 1), dtype=np.float32)
    np.savez(
        npz_path,
        diffraction=diffraction,
        YY_ground_truth=yy_ground_truth,
        probeGuess=probe_guess,
        coords_nominal=coords_nominal,
    )

    with np.testing.assert_raises_regex(KeyError, "coords_offsets"):
        convert_npz_split_to_hdf5_pair(
            npz_path=npz_path,
            out_dir=tmp_path,
            object_name="missing_offsets",
            pixel_size_m=1.0e-9,
        )


def test_convert_prefers_coords_offsets_for_probe_positions(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "offset_priority.npz"
    diffraction = np.abs(np.random.default_rng(0).normal(size=(5, 64, 64))).astype(np.float32)
    yy_ground_truth = (np.ones((1, 392, 392)) + 1j * np.ones((1, 392, 392))).astype(np.complex64)
    probe_guess = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    coords_nominal = np.zeros((5, 1, 2, 1), dtype=np.float32)
    coords_offsets = np.zeros((5, 1, 2, 1), dtype=np.float32)
    coords_offsets[:, 0, 0, 0] = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    coords_offsets[:, 0, 1, 0] = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    np.savez(
        npz_path,
        diffraction=diffraction,
        YY_ground_truth=yy_ground_truth,
        probeGuess=probe_guess,
        coords_nominal=coords_nominal,
        coords_offsets=coords_offsets,
    )

    out = convert_npz_split_to_hdf5_pair(
        npz_path=npz_path,
        out_dir=tmp_path,
        object_name="offset_priority",
        pixel_size_m=1.0e-9,
    )
    with h5py.File(out.para_hdf5, "r") as para_file:
        x = np.asarray(para_file["probe_position_x_m"])
        y = np.asarray(para_file["probe_position_y_m"])
    assert np.unique(x).size > 1
    assert np.unique(y).size > 1


def test_convert_selects_first_object_from_ground_truth_stack(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "stacked_object.npz"
    diffraction = np.abs(np.random.default_rng(0).normal(size=(5, 64, 64))).astype(np.float32)
    yy_ground_truth = np.stack(
        [
            np.full((32, 32), 1.0 + 1.0j, dtype=np.complex64),
            np.full((32, 32), 2.0 + 2.0j, dtype=np.complex64),
            np.full((32, 32), 3.0 + 3.0j, dtype=np.complex64),
        ],
        axis=0,
    )
    probe_guess = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    coords_offsets = np.zeros((5, 1, 2, 1), dtype=np.float32)
    np.savez(
        npz_path,
        diffraction=diffraction,
        YY_ground_truth=yy_ground_truth,
        probeGuess=probe_guess,
        coords_offsets=coords_offsets,
    )

    out = convert_npz_split_to_hdf5_pair(
        npz_path=npz_path,
        out_dir=tmp_path,
        object_name="stacked_object",
        pixel_size_m=1.0e-9,
    )
    with h5py.File(out.para_hdf5, "r") as para_file:
        obj = np.asarray(para_file["object"])

    assert obj.shape == (1, 32, 32)
    np.testing.assert_allclose(obj[0], yy_ground_truth[0])


def test_convert_prefers_yy_full_when_both_object_keys_exist(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "yy_full_priority.npz"
    diffraction = np.abs(np.random.default_rng(0).normal(size=(5, 64, 64))).astype(np.float32)
    yy_full = np.full((1, 48, 48, 1), 7.0 + 7.0j, dtype=np.complex64)
    yy_ground_truth = np.full((24, 24, 1), 3.0 + 3.0j, dtype=np.complex64)
    probe_guess = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    coords_offsets = np.zeros((5, 1, 2, 1), dtype=np.float32)
    np.savez(
        npz_path,
        diffraction=diffraction,
        YY_full=yy_full,
        YY_ground_truth=yy_ground_truth,
        probeGuess=probe_guess,
        coords_offsets=coords_offsets,
    )

    out = convert_npz_split_to_hdf5_pair(
        npz_path=npz_path,
        out_dir=tmp_path,
        object_name="yy_full_priority",
        pixel_size_m=1.0e-9,
    )
    with h5py.File(out.para_hdf5, "r") as para_file:
        obj = np.asarray(para_file["object"])

    assert obj.shape == (1, 48, 48)
    np.testing.assert_allclose(obj[0], yy_full[0, :, :, 0])


def test_convert_recenters_absolute_positions_for_ptychovit_frame(tmp_path: Path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair

    npz_path = tmp_path / "absolute_positions.npz"
    diffraction = np.abs(np.random.default_rng(0).normal(size=(2, 8, 8))).astype(np.float32)
    yy_full = np.full((1, 16, 16, 1), 1.0 + 1.0j, dtype=np.complex64)
    probe_guess = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)

    # Absolute pixel positions in top-left-origin frame.
    coords_offsets = np.zeros((2, 1, 2, 1), dtype=np.float32)
    coords_offsets[0, 0, 0, 0] = 4.5  # y
    coords_offsets[0, 0, 1, 0] = 4.5  # x
    coords_offsets[1, 0, 0, 0] = 10.5
    coords_offsets[1, 0, 1, 0] = 10.5
    np.savez(
        npz_path,
        diffraction=diffraction,
        YY_full=yy_full,
        probeGuess=probe_guess,
        coords_offsets=coords_offsets,
    )

    out = convert_npz_split_to_hdf5_pair(
        npz_path=npz_path,
        out_dir=tmp_path,
        object_name="absolute_positions",
        pixel_size_m=1.0,
    )
    with h5py.File(out.para_hdf5, "r") as para_file:
        x = np.asarray(para_file["probe_position_x_m"])
        y = np.asarray(para_file["probe_position_y_m"])

    # Object origin used by PtychoViT loader is round(H/2)+0.5 = 8.5 for H=W=16.
    np.testing.assert_allclose(y, np.array([-4.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(x, np.array([-4.0, 2.0], dtype=np.float64))


def _write_minimal_hdf5_pair(dp_path: Path, para_path: Path, *, n_scans: int = 5, n_pos: int = 5) -> None:
    with h5py.File(dp_path, "w") as dp_file:
        dp_file.create_dataset("dp", data=np.ones((n_scans, 8, 8), dtype=np.float32))

    with h5py.File(para_path, "w") as para_file:
        obj = para_file.create_dataset("object", data=np.ones((1, 16, 16), dtype=np.complex64))
        obj.attrs["pixel_height_m"] = 1.0e-9
        obj.attrs["pixel_width_m"] = 1.0e-9
        probe = para_file.create_dataset("probe", data=np.ones((1, 1, 8, 8), dtype=np.complex64))
        probe.attrs["pixel_height_m"] = 1.0e-9
        probe.attrs["pixel_width_m"] = 1.0e-9
        para_file.create_dataset("probe_position_x_m", data=np.arange(n_pos, dtype=np.float64))
        para_file.create_dataset("probe_position_y_m", data=np.arange(n_pos, dtype=np.float64))


def test_validate_hdf5_pair_rejects_missing_probe_positions(tmp_path: Path):
    from ptycho.interop.ptychovit.validate import validate_hdf5_pair

    dp_path = tmp_path / "a_dp.hdf5"
    para_path = tmp_path / "a_para.hdf5"
    _write_minimal_hdf5_pair(dp_path, para_path, n_scans=5, n_pos=5)
    with h5py.File(para_path, "a") as para_file:
        del para_file["probe_position_y_m"]

    with np.testing.assert_raises_regex(ValueError, "probe_position"):
        validate_hdf5_pair(dp_path, para_path)


def test_validate_hdf5_pair_rejects_mismatched_position_lengths(tmp_path: Path):
    from ptycho.interop.ptychovit.validate import validate_hdf5_pair

    dp_path = tmp_path / "b_dp.hdf5"
    para_path = tmp_path / "b_para.hdf5"
    _write_minimal_hdf5_pair(dp_path, para_path, n_scans=5, n_pos=5)
    with h5py.File(para_path, "a") as para_file:
        del para_file["probe_position_y_m"]
        para_file.create_dataset("probe_position_y_m", data=np.arange(4, dtype=np.float64))

    with np.testing.assert_raises_regex(ValueError, "same length"):
        validate_hdf5_pair(dp_path, para_path)


def test_validate_hdf5_pair_rejects_scan_count_mismatch(tmp_path: Path):
    from ptycho.interop.ptychovit.validate import validate_hdf5_pair

    dp_path = tmp_path / "c_dp.hdf5"
    para_path = tmp_path / "c_para.hdf5"
    _write_minimal_hdf5_pair(dp_path, para_path, n_scans=6, n_pos=5)

    with np.testing.assert_raises_regex(ValueError, "dp scan count"):
        validate_hdf5_pair(dp_path, para_path)

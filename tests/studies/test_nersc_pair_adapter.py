from pathlib import Path

import h5py
import numpy as np


def _write_pair(
    dp_h5: Path,
    para_h5: Path,
    *,
    include_probe_attrs: bool,
    probe_guess: np.ndarray | None = None,
) -> None:
    dp = np.array(
        [
            [[4.0, 9.0], [16.0, -1.0]],
            [[25.0, 36.0], [49.0, 64.0]],
        ],
        dtype=np.float32,
    )
    with h5py.File(dp_h5, "w") as handle:
        handle.create_dataset("dp", data=dp)

    object_guess = np.ones((1, 8, 8), dtype=np.complex64) * (2.0 + 1.0j)
    if probe_guess is None:
        probe_guess = np.ones((1, 1, 2, 2), dtype=np.complex64) * (0.5 + 0.25j)
    probe_guess = np.asarray(probe_guess, dtype=np.complex64)
    with h5py.File(para_h5, "w") as handle:
        object_ds = handle.create_dataset("object", data=object_guess)
        object_ds.attrs["pixel_width_m"] = 1e-6
        object_ds.attrs["pixel_height_m"] = 2e-6
        object_ds.attrs["center_x_m"] = 0.0
        object_ds.attrs["center_y_m"] = 0.0

        probe_ds = handle.create_dataset("probe", data=probe_guess)
        if include_probe_attrs:
            probe_ds.attrs["pixel_width_m"] = 1e-6
            probe_ds.attrs["pixel_height_m"] = 2e-6

        handle.create_dataset("probe_position_x_m", data=np.array([0.0, 2e-6], dtype=np.float64))
        handle.create_dataset("probe_position_y_m", data=np.array([1e-6, 3e-6], dtype=np.float64))


def test_pair_preflight_adds_missing_probe_pixel_attrs_to_working_copy(tmp_path):
    from scripts.studies.nersc_pair_adapter import materialize_pair_working_copy

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    _write_pair(src_dp, src_para, include_probe_attrs=False)

    working_dp, working_para = materialize_pair_working_copy(src_dp, src_para, tmp_path / "work")
    assert working_dp.exists()
    assert working_para.exists()

    with h5py.File(src_para, "r") as src:
        assert "pixel_width_m" not in src["probe"].attrs
        assert "pixel_height_m" not in src["probe"].attrs

    with h5py.File(working_para, "r") as patched:
        assert patched["probe"].attrs["pixel_width_m"] == patched["object"].attrs["pixel_width_m"]
        assert patched["probe"].attrs["pixel_height_m"] == patched["object"].attrs["pixel_height_m"]


def test_pair_to_npz_converts_dp_intensity_to_amplitude_and_positions_to_pixels(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    _write_pair(src_dp, src_para, include_probe_attrs=True)

    out_npz = pair_to_external_npz(src_dp, src_para, tmp_path / "converted.npz")

    with np.load(out_npz, allow_pickle=True) as data:
        expected_first = np.sqrt(np.maximum(np.array([[4.0, 9.0], [16.0, -1.0]]), 0.0))
        assert data["diff3d"].shape == (2, 2, 2)
        assert np.allclose(data["diff3d"][0], expected_first)
        assert np.allclose(data["xcoords"], np.array([0.0, 2.0], dtype=np.float64))
        assert np.allclose(data["ycoords"], np.array([0.5, 1.5], dtype=np.float64))


def test_pair_to_npz_emits_external_raw_required_keys(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    _write_pair(src_dp, src_para, include_probe_attrs=True)

    out_npz = pair_to_external_npz(src_dp, src_para, tmp_path / "converted.npz")

    with np.load(out_npz, allow_pickle=True) as data:
        required = {
            "xcoords",
            "ycoords",
            "xcoords_start",
            "ycoords_start",
            "diff3d",
            "probeGuess",
            "objectGuess",
            "scan_index",
        }
        assert required.issubset(set(data.files))
        assert np.iscomplexobj(data["probeGuess"])
        assert np.iscomplexobj(data["objectGuess"])


def test_pair_to_npz_uses_object_center_for_centered_pixel_coordinates(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    _write_pair(src_dp, src_para, include_probe_attrs=True)
    with h5py.File(src_para, "r+") as handle:
        handle["object"].attrs["center_x_m"] = 1e-6
        handle["object"].attrs["center_y_m"] = 2e-6

    out_npz = pair_to_external_npz(src_dp, src_para, tmp_path / "converted.npz")
    with np.load(out_npz, allow_pickle=True) as data:
        assert np.allclose(data["xcoords"], np.array([-1.0, 1.0], dtype=np.float64))
        assert np.allclose(data["ycoords"], np.array([-0.5, 0.5], dtype=np.float64))


def test_pair_to_npz_defaults_to_incoherent_aggregate_for_multimode_probe(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    probe_modes = np.zeros((1, 3, 2, 2), dtype=np.complex64)
    probe_modes[0, 0, :, :] = (1.0 + 1.0j)
    probe_modes[0, 1, :, :] = (2.0 - 1.0j)
    probe_modes[0, 2, :, :] = (0.5 + 0.0j)
    _write_pair(src_dp, src_para, include_probe_attrs=True, probe_guess=probe_modes)

    out_npz = pair_to_external_npz(src_dp, src_para, tmp_path / "converted.npz")
    with np.load(out_npz, allow_pickle=True) as data:
        expected_amp = np.sqrt(np.sum(np.abs(probe_modes[0]) ** 2, axis=0))
        expected = expected_amp * np.exp(1j * np.angle(probe_modes[0, 0]))
        assert np.allclose(data["probeGuess"], expected.astype(np.complex64))


def test_pair_to_npz_supports_first_mode_policy_override(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    probe_modes = np.zeros((1, 3, 2, 2), dtype=np.complex64)
    probe_modes[0, 0, :, :] = (1.0 + 0.5j)
    probe_modes[0, 1, :, :] = (5.0 + 2.0j)
    probe_modes[0, 2, :, :] = (-3.0 + 1.0j)
    _write_pair(src_dp, src_para, include_probe_attrs=True, probe_guess=probe_modes)

    out_npz = pair_to_external_npz(
        src_dp,
        src_para,
        tmp_path / "converted.npz",
        probe_mode_policy="first_mode",
    )
    with np.load(out_npz, allow_pickle=True) as data:
        assert np.allclose(data["probeGuess"], probe_modes[0, 0].astype(np.complex64))


def test_pair_to_npz_legacy_call_still_returns_path(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    _write_pair(src_dp, src_para, include_probe_attrs=True)

    out_npz = pair_to_external_npz(src_dp, src_para, tmp_path / "converted.npz")
    assert isinstance(out_npz, Path)
    assert out_npz.exists()


def test_pair_to_npz_populates_probe_metadata_sink_for_multimode_probe(tmp_path):
    from scripts.studies.nersc_pair_adapter import pair_to_external_npz

    src_dp = tmp_path / "scan_dp.hdf5"
    src_para = tmp_path / "scan_para.hdf5"
    probe_modes = np.zeros((1, 3, 2, 2), dtype=np.complex64)
    probe_modes[0, 0, :, :] = (1.0 + 0.0j)
    probe_modes[0, 1, :, :] = (2.0 + 0.0j)
    probe_modes[0, 2, :, :] = (3.0 + 0.0j)
    _write_pair(src_dp, src_para, include_probe_attrs=True, probe_guess=probe_modes)

    metadata: dict[str, object] = {}
    out_npz = pair_to_external_npz(
        src_dp,
        src_para,
        tmp_path / "converted.npz",
        metadata_out=metadata,
    )

    assert out_npz.exists()
    assert metadata["probe_mode_policy"] == "incoherent_aggregate"
    assert metadata["probe_source_shape"] == [1, 3, 2, 2]
    weights = np.asarray(metadata["probe_mode_power_weights"], dtype=np.float64)
    assert weights.shape == (3,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)

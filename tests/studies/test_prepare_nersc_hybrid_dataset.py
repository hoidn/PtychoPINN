from pathlib import Path
import json

import h5py
import numpy as np


def _write_cameraman_pair(dp_h5: Path, para_h5: Path) -> None:
    rng = np.random.default_rng(7)
    dp = np.abs(rng.normal(size=(4, 256, 256))).astype(np.float32)
    with h5py.File(dp_h5, "w") as handle:
        handle.create_dataset("dp", data=dp)

    object_guess = np.ones((1, 256, 256), dtype=np.complex64) * (1.0 + 0.5j)
    probe_guess = np.ones((1, 1, 256, 256), dtype=np.complex64) * (0.1 + 0.2j)
    with h5py.File(para_h5, "w") as handle:
        object_ds = handle.create_dataset("object", data=object_guess)
        object_ds.attrs["pixel_width_m"] = 1e-6
        object_ds.attrs["pixel_height_m"] = 1e-6
        object_ds.attrs["center_x_m"] = 0.0
        object_ds.attrs["center_y_m"] = 0.0

        probe_ds = handle.create_dataset("probe", data=probe_guess)
        probe_ds.attrs["pixel_width_m"] = 1e-6
        probe_ds.attrs["pixel_height_m"] = 1e-6

        handle.create_dataset(
            "probe_position_x_m",
            data=np.array([0.0, 1e-6, 2e-6, 3e-6], dtype=np.float64),
        )
        handle.create_dataset(
            "probe_position_y_m",
            data=np.array([-2e-6, -1e-6, 1e-6, 2e-6], dtype=np.float64),
        )


def test_prepare_hybrid_dataset_writes_train_half_and_full_test_npz(tmp_path):
    from scripts.studies.prepare_nersc_hybrid_dataset import prepare_hybrid_dataset

    dp_h5 = tmp_path / "cameraman_dp.hdf5"
    para_h5 = tmp_path / "cameraman_para.hdf5"
    _write_cameraman_pair(dp_h5, para_h5)

    result = prepare_hybrid_dataset(
        dp_h5=dp_h5,
        para_h5=para_h5,
        output_dir=tmp_path / "prepared",
        half="top",
    )

    with np.load(result["train_npz"], allow_pickle=True) as train:
        assert train["diffraction"].shape == (2, 128, 128)
        assert train["objectGuess"].shape == (128, 128)
        assert train["probeGuess"].shape == (128, 128)
        assert np.all(train["ycoords"] >= float(result["split_threshold"]))

    with np.load(result["test_npz"], allow_pickle=True) as test:
        assert test["diffraction"].shape == (4, 128, 128)


def test_prepare_hybrid_dataset_supports_top_and_bottom_half(tmp_path):
    from scripts.studies.prepare_nersc_hybrid_dataset import prepare_hybrid_dataset

    dp_h5 = tmp_path / "cameraman_dp.hdf5"
    para_h5 = tmp_path / "cameraman_para.hdf5"
    _write_cameraman_pair(dp_h5, para_h5)

    top = prepare_hybrid_dataset(
        dp_h5=dp_h5,
        para_h5=para_h5,
        output_dir=tmp_path / "top",
        half="top",
    )
    bottom = prepare_hybrid_dataset(
        dp_h5=dp_h5,
        para_h5=para_h5,
        output_dir=tmp_path / "bottom",
        half="bottom",
    )

    with np.load(top["train_npz"], allow_pickle=True) as top_train, np.load(
        bottom["train_npz"], allow_pickle=True
    ) as bottom_train:
        top_idx = set(np.asarray(top_train["scan_index"]).tolist())
        bottom_idx = set(np.asarray(bottom_train["scan_index"]).tolist())
        assert top_idx
        assert bottom_idx
        assert top_idx.isdisjoint(bottom_idx)


def test_prepare_hybrid_dataset_records_manifest_and_counts(tmp_path):
    from scripts.studies.prepare_nersc_hybrid_dataset import prepare_hybrid_dataset

    dp_h5 = tmp_path / "cameraman_dp.hdf5"
    para_h5 = tmp_path / "cameraman_para.hdf5"
    _write_cameraman_pair(dp_h5, para_h5)

    result = prepare_hybrid_dataset(
        dp_h5=dp_h5,
        para_h5=para_h5,
        output_dir=tmp_path / "prepared",
        half="top",
    )

    manifest = json.loads(Path(result["manifest_json"]).read_text())
    assert manifest["source_dp"] == str(dp_h5)
    assert manifest["source_para"] == str(para_h5)
    assert manifest["half"] == "top"
    assert manifest["n_total"] == 4
    assert manifest["n_train"] == 2
    assert manifest["n_test"] == 4
    assert "source_dp_sha256" in manifest
    assert "source_para_sha256" in manifest


def test_downsample_external_payload_handles_odd_object_shape_by_center_crop():
    from scripts.studies.prepare_nersc_hybrid_dataset import _downsample_external_payload

    payload = {
        "diffraction": np.ones((2, 6, 6), dtype=np.float32),
        "objectGuess": np.ones((11, 11), dtype=np.complex64),
        "probeGuess": np.ones((6, 6), dtype=np.complex64),
        "xcoords": np.array([0.0, 2.0], dtype=np.float64),
        "ycoords": np.array([1.0, 3.0], dtype=np.float64),
        "xcoords_start": np.array([0.0, 2.0], dtype=np.float64),
        "ycoords_start": np.array([1.0, 3.0], dtype=np.float64),
    }

    out = _downsample_external_payload(payload, target_n=3)
    assert out["diffraction"].shape == (2, 3, 3)
    assert out["objectGuess"].shape == (5, 5)
    assert out["probeGuess"].shape == (3, 3)


def test_downsample_external_payload_bins_diffraction_blocks():
    from scripts.studies.prepare_nersc_hybrid_dataset import _downsample_external_payload

    diffraction = np.arange(1, 37, dtype=np.float32).reshape(1, 6, 6)
    payload = {
        "diffraction": diffraction,
        "objectGuess": np.ones((6, 6), dtype=np.complex64),
        "probeGuess": np.ones((6, 6), dtype=np.complex64),
        "xcoords": np.array([0.0], dtype=np.float64),
        "ycoords": np.array([0.0], dtype=np.float64),
    }

    out = _downsample_external_payload(payload, target_n=3)
    expected = np.array(
        [[[4.5, 6.5, 8.5], [16.5, 18.5, 20.5], [28.5, 30.5, 32.5]]],
        dtype=np.float32,
    )
    assert np.allclose(out["diffraction"], expected)


def test_downsample_external_payload_center_crops_real_space_not_bin():
    from scripts.studies.prepare_nersc_hybrid_dataset import _downsample_external_payload

    object_guess = np.arange(1, 122, dtype=np.float32).reshape(11, 11).astype(np.complex64)
    probe_guess = np.arange(1, 37, dtype=np.float32).reshape(6, 6).astype(np.complex64)
    payload = {
        "diffraction": np.ones((1, 6, 6), dtype=np.float32),
        "objectGuess": object_guess,
        "probeGuess": probe_guess,
        "xcoords": np.array([0.0], dtype=np.float64),
        "ycoords": np.array([0.0], dtype=np.float64),
    }

    out = _downsample_external_payload(payload, target_n=3)
    assert np.array_equal(out["objectGuess"], object_guess[3:8, 3:8])
    assert np.array_equal(out["probeGuess"], probe_guess[1:4, 1:4])


def test_downsample_external_payload_preserves_coords_under_real_space_crop():
    from scripts.studies.prepare_nersc_hybrid_dataset import _downsample_external_payload

    xcoords = np.array([0.0, 2.0], dtype=np.float64)
    ycoords = np.array([1.0, 3.0], dtype=np.float64)
    payload = {
        "diffraction": np.ones((2, 6, 6), dtype=np.float32),
        "objectGuess": np.ones((6, 6), dtype=np.complex64),
        "probeGuess": np.ones((6, 6), dtype=np.complex64),
        "xcoords": xcoords,
        "ycoords": ycoords,
        "xcoords_start": xcoords.copy(),
        "ycoords_start": ycoords.copy(),
    }

    out = _downsample_external_payload(payload, target_n=3)
    assert np.array_equal(out["xcoords"], xcoords)
    assert np.array_equal(out["ycoords"], ycoords)
    assert np.array_equal(out["xcoords_start"], xcoords)
    assert np.array_equal(out["ycoords_start"], ycoords)


def test_prepare_cli_writes_invocation_artifacts(tmp_path, monkeypatch):
    from scripts.studies.prepare_nersc_hybrid_dataset import main

    dp_h5 = tmp_path / "cameraman_dp.hdf5"
    para_h5 = tmp_path / "cameraman_para.hdf5"
    _write_cameraman_pair(dp_h5, para_h5)
    output_dir = tmp_path / "prepared_cli"

    argv = [
        "prepare_nersc_hybrid_dataset.py",
        "--dp-h5",
        str(dp_h5),
        "--para-h5",
        str(para_h5),
        "--output-dir",
        str(output_dir),
        "--half",
        "top",
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()

    assert (output_dir / "invocation.json").exists()
    assert (output_dir / "invocation.sh").exists()

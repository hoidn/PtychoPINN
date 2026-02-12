from pathlib import Path

import numpy as np


def _write_external_raw_npz(path: Path, *, include_object: bool = True) -> None:
    n = 96
    N = 64
    data = {
        "diffraction": np.random.rand(n, N, N).astype(np.float32),
        "probeGuess": np.ones((N, N), dtype=np.complex64),
        "xcoords": np.linspace(0.0, 11.0, n).astype(np.float32),
        "ycoords": np.linspace(1.0, 12.0, n).astype(np.float32),
        "xcoords_start": np.linspace(0.0, 11.0, n).astype(np.float32),
        "ycoords_start": np.linspace(1.0, 12.0, n).astype(np.float32),
    }
    if include_object:
        data["objectGuess"] = np.ones((128, 128), dtype=np.complex64)
    np.savez(path, **data)


def _grid_cfg(tmp_path: Path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig

    return GridLinesConfig(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
        nimgs_train=1,
        nimgs_test=1,
    )


def test_build_synthetic_delegates_to_grid_lines_builder(monkeypatch, tmp_path):
    from scripts.studies.grid_study_dataset_builder import build_datasets

    expected = {
        64: {
            "train_npz": str(tmp_path / "datasets" / "N64" / "gs1" / "train.npz"),
            "test_npz": str(tmp_path / "datasets" / "N64" / "gs1" / "test.npz"),
            "gt_recon": str(tmp_path / "recons" / "gt" / "recon.npz"),
            "tag": "N64",
        }
    }
    called = {"ok": False}

    def fake_build(base_cfg, required_ns):
        called["ok"] = True
        assert base_cfg.N == 64
        assert sorted(required_ns) == [64]
        return expected

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n",
        fake_build,
    )
    out = build_datasets(
        dataset_source="synthetic_lines",
        cfg=_grid_cfg(tmp_path),
        required_ns=[64],
    )
    assert called["ok"] is True
    assert out == expected


def test_build_external_raw_generates_grouped_train_test_npz(tmp_path):
    from scripts.studies.grid_study_dataset_builder import build_datasets

    train_raw = tmp_path / "fly_train_raw.npz"
    test_raw = tmp_path / "fly_test_raw.npz"
    _write_external_raw_npz(train_raw, include_object=True)
    _write_external_raw_npz(test_raw, include_object=True)

    out = build_datasets(
        dataset_source="external_raw_npz",
        cfg=_grid_cfg(tmp_path),
        required_ns=[64],
        train_data=train_raw,
        test_data=test_raw,
        n_groups=4,
        n_subsample=8,
        neighbor_count=3,
        subsample_seed=7,
    )
    bundle = out[64]
    train_npz = Path(bundle["train_npz"])
    test_npz = Path(bundle["test_npz"])
    assert train_npz.exists()
    assert test_npz.exists()

    with np.load(train_npz, allow_pickle=True) as train_data:
        for key in (
            "diffraction",
            "Y_I",
            "Y_phi",
            "coords_nominal",
            "coords_true",
            "coords_offsets",
            "YY_full",
        ):
            assert key in train_data.files


def test_build_external_raw_fails_without_object_ground_truth(tmp_path):
    import pytest
    from scripts.studies.grid_study_dataset_builder import build_datasets

    train_raw = tmp_path / "fly_train_raw_missing_object.npz"
    test_raw = tmp_path / "fly_test_raw_missing_object.npz"
    _write_external_raw_npz(train_raw, include_object=False)
    _write_external_raw_npz(test_raw, include_object=False)

    with pytest.raises(ValueError, match="objectGuess"):
        build_datasets(
            dataset_source="external_raw_npz",
            cfg=_grid_cfg(tmp_path),
            required_ns=[64],
            train_data=train_raw,
            test_data=test_raw,
            n_groups=4,
            n_subsample=8,
            neighbor_count=3,
            subsample_seed=7,
        )

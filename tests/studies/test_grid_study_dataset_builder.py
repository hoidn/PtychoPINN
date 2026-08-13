from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _write_external_raw_npz(path: Path, *, include_object: bool = True, n: int = 96) -> None:
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


def test_build_external_raw_with_none_n_groups_uses_full_split_sizes(tmp_path):
    from scripts.studies.grid_study_dataset_builder import build_datasets

    train_raw = tmp_path / "fly_train_raw.npz"
    test_raw = tmp_path / "fly_test_raw.npz"
    _write_external_raw_npz(train_raw, include_object=True, n=80)
    _write_external_raw_npz(test_raw, include_object=True, n=120)

    out = build_datasets(
        dataset_source="external_raw_npz",
        cfg=_grid_cfg(tmp_path),
        required_ns=[64],
        train_data=train_raw,
        test_data=test_raw,
        n_groups=None,
        n_subsample=None,
        neighbor_count=3,
        subsample_seed=7,
    )
    bundle = out[64]
    train_npz = Path(bundle["train_npz"])
    test_npz = Path(bundle["test_npz"])

    with np.load(train_npz, allow_pickle=True) as train_data:
        assert train_data["diffraction"].shape[0] == 80
    with np.load(test_npz, allow_pickle=True) as test_data:
        assert test_data["diffraction"].shape[0] == 120


def test_external_grouping_uses_raw_data_without_mutating_ambient_params(
    monkeypatch,
    tmp_path,
):
    from ptycho import params
    from scripts.studies import grid_study_dataset_builder as builder

    ambient = {
        "N": "poison-N",
        "gridsize": "poison-gridsize",
        "use_xla_translate": "poison-xla",
        "unrelated": "preserve-me",
    }
    monkeypatch.setattr(params, "cfg", ambient)
    ambient_before = dict(ambient)

    object_guess = (
        np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        + 1j * np.ones((128, 128), dtype=np.float32)
    ).astype(np.complex64)
    probe_guess = np.full((64, 64), 2.0 + 3.0j, dtype=np.complex64)

    grouped_by_split = {}
    grouping_calls = []

    def grouped_payload(split_value):
        values = np.full((2, 2, 2, 1), split_value, dtype=np.float32)
        phase = np.float32(split_value / 10.0)
        grouped = {
            "X_full": values,
            "Y": np.exp(1j * phase).astype(np.complex64) * values,
            "coords_relative": np.full(
                (2, 1, 2, 1), split_value + 4.0, dtype=np.float32
            ),
            "coords_offsets": np.full(
                (2, 1, 2, 1), split_value + 8.0, dtype=np.float32
            ),
        }
        grouped_by_split[split_value] = grouped
        return grouped

    def raw_split(split_value):
        coords = np.arange(4, dtype=np.float32)

        def generate_grouped_data(**kwargs):
            grouping_calls.append((split_value, kwargs))
            return grouped_payload(split_value)

        return SimpleNamespace(
            xcoords=coords,
            ycoords=coords + 10,
            xcoords_start=coords + 20,
            ycoords_start=coords + 30,
            diff3d=np.full((4, 64, 64), split_value, dtype=np.float32),
            probeGuess=probe_guess,
            scan_index=np.arange(4, dtype=np.int32),
            objectGuess=object_guess,
            Y=None,
            norm_Y_I=None,
            metadata={"split": split_value},
            sample_indices=np.arange(4, dtype=np.int32),
            subsample_seed=7,
            generate_grouped_data=generate_grouped_data,
        )

    raw_splits = iter((raw_split(1.0), raw_split(2.0)))
    monkeypatch.setattr(
        builder.wf_components,
        "load_data",
        lambda *_args, **_kwargs: next(raw_splits),
    )

    train_raw_path = tmp_path / "fly_train_raw.npz"
    test_raw_path = tmp_path / "fly_test_raw.npz"
    result = builder.build_datasets(
        dataset_source="external_raw_npz",
        cfg=_grid_cfg(tmp_path),
        required_ns=[64],
        train_data=train_raw_path,
        test_data=test_raw_path,
        n_groups=2,
        n_subsample=None,
        neighbor_count=3,
        subsample_seed=7,
    )

    assert params.cfg is ambient
    assert params.cfg == ambient_before
    assert grouping_calls == [
        (
            1.0,
            {
                "N": 64,
                "K": 3,
                "nsamples": 2,
                "dataset_path": str(train_raw_path),
                "seed": 7,
                "gridsize": 1,
            },
        ),
        (
            2.0,
            {
                "N": 64,
                "K": 3,
                "nsamples": 2,
                "dataset_path": str(test_raw_path),
                "seed": 7,
                "gridsize": 1,
            },
        ),
    ]

    for split_value, key in ((1.0, "train_npz"), (2.0, "test_npz")):
        expected = grouped_by_split[split_value]
        with np.load(result[64][key], allow_pickle=True) as saved:
            np.testing.assert_array_equal(saved["diffraction"], expected["X_full"])
            np.testing.assert_array_equal(saved["Y_I"], np.abs(expected["Y"]))
            np.testing.assert_array_equal(saved["Y_phi"], np.angle(expected["Y"]))
            np.testing.assert_array_equal(
                saved["coords_relative"], expected["coords_relative"]
            )
            np.testing.assert_array_equal(
                saved["coords_offsets"], expected["coords_offsets"]
            )

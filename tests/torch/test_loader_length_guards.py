"""Regression tests for the two memory-map sizing guards in ``PtychoDataset``.

Both guards existed on mainline and were lost when the fno-stable loader was
overlaid onto the branch:

- ``537aa175`` truncated scan positions to the diffraction stack length. Some
  datasets carry trailing coordinate entries with no matching pattern; without
  the guard those indices run off the end of the diffraction stack and
  ``memory_map_data`` dies with ``IndexError``.
- ``687c50fb`` reconciled the memory-map allocation with the true group count.
  ``group_coords`` discards '4_quadrant' centers whose quadrants are not all
  populated, so ``n_valid_points * n_subsample`` overcounts and the slice
  assignment dies with ``RuntimeError``.

The current loader computes the grouping once in ``calculate_length`` and
reuses it, so the allocation, ``cum_length``, and the written tensors agree by
construction. These tests pin that, plus the coordinate-alignment contract.
"""
import json

import numpy as np
import pytest
import torch

from ptycho_torch import reassembly
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import (
    PtychoDataset,
    _get_diffraction_stack,
    npz_headers,
)


N_PIX = 32


def _write_npz(path, n_diff, xcoords, ycoords, *, pattern_size=N_PIX,
               legacy_hwn=False):
    rng = np.random.default_rng(0)
    raw = rng.random((n_diff, pattern_size, pattern_size)).astype(np.float32)
    diff3d = raw / np.sqrt((raw ** 2).sum(axis=(-2, -1), keepdims=True))
    if legacy_hwn:
        diff3d = np.transpose(diff3d, (1, 2, 0))
    probe = (rng.random((pattern_size, pattern_size)) +
             1j * rng.random((pattern_size, pattern_size))).astype(np.complex128)
    obj = (rng.random((pattern_size, pattern_size)) +
           1j * rng.random((pattern_size, pattern_size))).astype(np.complex128)
    np.savez(path, xcoords=xcoords, ycoords=ycoords, diff3d=diff3d,
             probeGuess=probe, objectGuess=obj)


def _build(tmp_path, data_config, model_config):
    return PtychoDataset(
        ptycho_dir=str(tmp_path / "npz"), model_config=model_config,
        data_config=data_config, training_config=TrainingConfig(batch_size=8),
        data_dir=str(tmp_path / "mm"), remake_map=True,
    )


def _line_scan(n):
    g = np.linspace(0.0, 10.0, n).astype(np.float64)
    return g, g.copy()


def _raster(side, spacing=1.5):
    g = np.arange(side) * spacing
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)


def test_memory_map_survives_extra_coordinates(tmp_path):
    """25 positions, 20 patterns: previously IndexError deep in the write loop."""
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(25)
    _write_npz(tmp_path / "npz" / "a.npz", 20, x, y)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=1,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1)
    with pytest.warns(RuntimeWarning, match="dropping the trailing 5 positions"):
        dataset = _build(tmp_path, data_config, model_config)

    # Truncation happens before bounds filtering, so exactly the 20 positions
    # backed by a pattern survive.
    assert len(dataset) == 20
    assert dataset.mmap_ptycho["images"].shape == (20, 1, N_PIX, N_PIX)
    assert int(dataset.mmap_ptycho["nn_indices"].max()) < 20


def test_dataset_rejects_fewer_positions_with_file_context(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(15)
    _write_npz(tmp_path / "npz" / "a.npz", 20, x, y)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(ValueError, match=r"a\.npz.*15 scan positions.*20 diffraction"):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_non_1d_coordinates_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    xcoords = np.zeros((20, 2), dtype=np.float64)
    ycoords = np.ones((20, 2), dtype=np.float64)
    _write_npz(tmp_path / "npz" / "bad_coords.npz", 20, xcoords, ycoords)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(
        ValueError,
        match=r"bad_coords\.npz.*xcoords shape \(20, 2\).*ycoords shape \(20, 2\).*one-dimensional",
    ):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_non_3d_diffraction_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(N_PIX)
    np.savez(
        tmp_path / "npz" / "flat.npz",
        xcoords=x,
        ycoords=y,
        diff3d=np.ones((N_PIX, N_PIX), dtype=np.float32),
        probeGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
        objectGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
    )

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(
        ValueError, match=r"flat\.npz.*3D.*\(M, H, W\).*got \(32, 32\)"
    ):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_missing_probe_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    np.savez(
        tmp_path / "npz" / "missing_probe.npz",
        xcoords=x,
        ycoords=y,
        diff3d=np.ones((20, N_PIX, N_PIX), dtype=np.float32),
        objectGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
    )

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(ValueError, match=r"missing_probe\.npz.*probeGuess"):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_incompatible_probe_shape_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    np.savez(
        tmp_path / "npz" / "bad_probe.npz",
        xcoords=x,
        ycoords=y,
        diff3d=np.ones((20, N_PIX, N_PIX), dtype=np.float32),
        probeGuess=np.ones((16, 16), dtype=np.complex64),
        objectGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
    )

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(ValueError, match=r"bad_probe\.npz.*probeGuess.*shape"):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_supervised_dataset_rejects_missing_label_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "missing_label.npz", 20, x, y)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, mode="Supervised",
                               object_big=False)

    with pytest.raises(ValueError, match=r"missing_label\.npz.*label"):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_supervised_dataset_rejects_malformed_label_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    np.savez(
        tmp_path / "npz" / "bad_label.npz",
        xcoords=x,
        ycoords=y,
        diff3d=np.ones((20, N_PIX, N_PIX), dtype=np.float32),
        probeGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
        objectGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
        label=np.ones((20, 16, 16), dtype=np.complex64),
    )

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, mode="Supervised",
                               object_big=False)

    with pytest.raises(
        ValueError, match=r"bad_label\.npz.*label.*Expected \(20, 32, 32\), got \(20, 16, 16\)"
    ):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_non_2d_object_guess_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    np.savez(
        tmp_path / "npz" / "bad_object.npz",
        xcoords=x,
        ycoords=y,
        diff3d=np.ones((20, N_PIX, N_PIX), dtype=np.float32),
        probeGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
        objectGuess=np.ones((2, N_PIX, N_PIX), dtype=np.complex64),
    )

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(
        ValueError, match=r"bad_object\.npz.*objectGuess.*2D.*shape \(2, 32, 32\)"
    ):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_cross_file_image_shape_mismatch(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "a.npz", 20, x, y)
    _write_npz(tmp_path / "npz" / "b.npz", 20, x, y, pattern_size=16)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(
        ValueError, match=r"b\.npz.*Expected \(32, 32\), got \(16, 16\)"
    ):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_nonzero_rank_rejects_invalid_headers_before_barrier(tmp_path, monkeypatch):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(15)
    _write_npz(tmp_path / "npz" / "rank1_invalid.npz", 20, x, y)

    def barrier_must_not_run():
        pytest.fail("barrier must not run after calculate_length validation fails")

    monkeypatch.setattr("ptycho_torch.dataloader.get_current_rank", lambda: 1)
    monkeypatch.setattr(
        "ptycho_torch.dataloader.is_ddp_initialized_and_active", lambda: True
    )
    monkeypatch.setattr("ptycho_torch.dataloader.dist.barrier", barrier_must_not_run)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(
        ValueError, match=r"rank1_invalid\.npz.*15 scan positions.*20 diffraction"
    ):
        _build(tmp_path, data_config, model_config)


def test_nonzero_rank_rejects_zero_length_before_barrier(tmp_path, monkeypatch):
    (tmp_path / "npz").mkdir()

    def barrier_must_not_run():
        pytest.fail("barrier must not run after zero-length validation fails")

    monkeypatch.setattr(
        PtychoDataset,
        "calculate_length",
        lambda self: (0, (N_PIX, N_PIX), [0], [], []),
    )
    monkeypatch.setattr("ptycho_torch.dataloader.get_current_rank", lambda: 1)
    monkeypatch.setattr(
        "ptycho_torch.dataloader.is_ddp_initialized_and_active", lambda: True
    )
    monkeypatch.setattr("ptycho_torch.dataloader.dist.barrier", barrier_must_not_run)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    with pytest.raises(ValueError, match=r"calculate_length\(\) resulted in 0 items"):
        _build(tmp_path, data_config, model_config)


def test_legacy_five_value_length_result_recovers_all_source_scan_ids(
    tmp_path, monkeypatch
):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "legacy_length_result.npz", 20, x, y)
    calculate_length = PtychoDataset.calculate_length

    def legacy_calculate_length(self):
        length, shape, cumulative, valid, _source, grouping = calculate_length(self)
        return length, shape, cumulative, valid, grouping

    monkeypatch.setattr(PtychoDataset, "calculate_length", legacy_calculate_length)
    data_config = DataConfig(
        N=N_PIX,
        grid_size=(1, 1),
        C=1,
        K=4,
        n_subsample=1,
        x_bounds=(0.25, 0.75),
        y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)

    dataset = _build(tmp_path, data_config, model_config)

    assert len(dataset.valid_indices_per_file[0]) < 20
    assert dataset.source_indices_per_file[0].tolist() == list(range(20))


def test_legacy_grouped_five_value_result_marks_center_ids_unavailable(
    tmp_path, monkeypatch
):
    (tmp_path / "npz").mkdir()
    x, y = _raster(7)
    _write_npz(tmp_path / "npz" / "legacy_grouped.npz", len(x), x, y)
    calculate_length = PtychoDataset.calculate_length

    def legacy_calculate_length(self):
        length, shape, cumulative, valid, _source, grouping = calculate_length(self)
        legacy_grouping = [
            None if record is None else (record[0], record[1])
            for record in grouping
        ]
        return length, shape, cumulative, valid, legacy_grouping

    monkeypatch.setattr(PtychoDataset, "calculate_length", legacy_calculate_length)
    data_config = DataConfig(
        N=N_PIX,
        C=4,
        grid_size=(2, 2),
        neighbor_function="4_quadrant",
        K_quadrant=30,
        min_neighbor_distance=0.0,
        max_neighbor_distance=4.0,
        x_bounds=(0.15, 0.85),
        y_bounds=(0.15, 0.85),
    )
    model_config = ModelConfig(
        C_model=4, C_forward=4, object_big=True, probe_big=False
    )

    dataset = _build(tmp_path, data_config, model_config)

    centers = dataset.mmap_ptycho["center_scan_id"].cpu().numpy()
    available = dataset.mmap_ptycho["center_scan_id_available"].cpu().numpy()
    assert centers.shape == (len(dataset),)
    assert np.all(centers == -1)
    assert not available.any()
    assert all(
        record is None or len(record) == 4
        for record in dataset.grouping_per_file
    )


def test_asymmetric_legacy_group_does_not_fabricate_centroid_center(
    tmp_path, monkeypatch
):
    (tmp_path / "npz").mkdir()
    x = np.asarray([0.0, 10.0, 20.0])
    y = np.zeros(3)
    _write_npz(tmp_path / "npz" / "asymmetric.npz", 3, x, y)
    nn_indices = np.asarray([[0, 2]], dtype=np.int64)
    coords_nn = np.asarray([[[[0.0, 0.0]], [[20.0, 0.0]]]])

    monkeypatch.setattr(
        PtychoDataset,
        "calculate_length",
        lambda self: (
            1,
            (N_PIX, N_PIX),
            [0, 1],
            [np.asarray([0, 1, 2])],
            [(nn_indices, coords_nn)],
        ),
    )
    data_config = DataConfig(N=N_PIX, C=2, grid_size=(1, 2))
    model_config = ModelConfig(
        C_model=2, C_forward=2, object_big=True, probe_big=False
    )

    dataset = _build(tmp_path, data_config, model_config)

    assert dataset.mmap_ptycho["center_scan_id"].tolist() == [-1]
    assert dataset.mmap_ptycho["center_scan_id_available"].tolist() == [False]
    participating, centers, available, filtered, source = (
        reassembly._scan_identity_evidence(dataset, dataset, 0)
    )
    assert participating == (0, 2)
    assert centers == ()
    assert available is False
    assert filtered == (0, 1, 2)
    assert source == (0, 1, 2)


def test_stale_v1_mmap_without_center_scan_id_requires_rebuild(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "stale.npz", 20, x, y)
    data_config = DataConfig(
        N=N_PIX, C=1, grid_size=(1, 1), x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0)
    )
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    _build(tmp_path, data_config, model_config)
    manifest_path = tmp_path / "mmap_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest["required_fields"].remove("center_scan_id")
    manifest["required_fields"].remove("center_scan_id_available")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RuntimeError,
        match=r"schema_version=1.*expected 4.*Rebuild it with remake_map=True",
    ):
        PtychoDataset.from_existing_map(
            tmp_path / "mm", model_config, data_config
        )


def test_memory_map_loads_legacy_hwn_layout(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(40)
    _write_npz(tmp_path / "npz" / "legacy_hwn.npz", 40, x, y,
               pattern_size=32, legacy_hwn=True)

    data_config = DataConfig(N=32, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    dataset = _build(tmp_path, data_config, model_config)

    assert len(dataset) == 40
    assert dataset.mmap_ptycho["images"].shape == (40, 1, 32, 32)
    assert int(dataset.mmap_ptycho["nn_indices"].max()) < 40


def test_memory_map_loads_legacy_hwn_layout_when_n_is_not_largest_axis(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "legacy_hwn.npz", 20, x, y,
               pattern_size=32, legacy_hwn=True)

    data_config = DataConfig(N=32, grid_size=(1, 1), C=1, K=4,
                             n_subsample=1, x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    dataset = _build(tmp_path, data_config, model_config)

    assert len(dataset) == 20
    assert dataset.im_shape == (32, 32)
    assert dataset.mmap_ptycho["images"].shape == (20, 1, 32, 32)


@pytest.mark.parametrize(
    ("exact_key", "decoy_key"),
    [("diffraction", "diffraction_backup"),
     ("diff3d", "diff3d_backup")],
)
def test_header_and_loader_ignore_prefixed_diffraction_decoys(
        tmp_path, exact_key, decoy_key):
    path = tmp_path / f"{exact_key}_with_decoy.npz"
    x, y = _line_scan(20)
    exact = np.arange(20 * 32 * 32, dtype=np.float32).reshape(20, 32, 32)
    decoy = np.zeros((7, 16, 16), dtype=np.float32)
    np.savez(path, **{
        decoy_key: decoy,
        exact_key: exact,
        "xcoords": x,
        "ycoords": y,
        "probeGuess": np.ones((32, 32), dtype=np.complex64),
    })

    shape, xa, ya = npz_headers(path)
    loaded = _get_diffraction_stack(path)

    assert shape == exact.shape
    assert len(xa) == len(ya) == len(exact)
    np.testing.assert_array_equal(loaded, exact)


def test_square_plane_keeps_canonical_stack_despite_trailing_coordinate_collision(tmp_path):
    path = tmp_path / "canonical_trailing_collision.npz"
    x, y = _line_scan(32)
    canonical = np.arange(20 * 32 * 32, dtype=np.float32).reshape(20, 32, 32)
    np.savez(
        path,
        xcoords=x,
        ycoords=y,
        diff3d=canonical,
        probeGuess=np.ones((32, 32), dtype=np.complex64),
    )

    with pytest.warns(RuntimeWarning, match="dropping the trailing 12 positions"):
        shape, xa, ya = npz_headers(path)
    with pytest.warns(RuntimeWarning, match="dropping the trailing 12 positions"):
        loaded = _get_diffraction_stack(path)

    assert shape == (20, 32, 32)
    assert len(xa) == len(ya) == 20
    np.testing.assert_array_equal(loaded, canonical)


def test_square_plane_transposes_legacy_stack_despite_coordinate_collision(tmp_path):
    path = tmp_path / "legacy_trailing_collision.npz"
    x, y = _line_scan(32)
    canonical = np.arange(20 * 32 * 32, dtype=np.float32).reshape(20, 32, 32)
    legacy = np.transpose(canonical, (1, 2, 0))
    np.savez(
        path,
        xcoords=x,
        ycoords=y,
        diff3d=legacy,
        probeGuess=np.ones((32, 32), dtype=np.complex64),
    )

    with pytest.warns(RuntimeWarning, match="dropping the trailing 12 positions"):
        shape, xa, ya = npz_headers(path)
    with pytest.warns(RuntimeWarning, match="dropping the trailing 12 positions"):
        loaded = _get_diffraction_stack(path)

    assert shape == (20, 32, 32)
    assert len(xa) == len(ya) == 20
    np.testing.assert_array_equal(loaded, canonical)


# ---------------------------------------------------------------------------
# Group-count / allocation consistency
# ---------------------------------------------------------------------------

def _quadrant_configs(n_subsample):
    data_config = DataConfig(N=N_PIX, grid_size=(2, 2), C=4, K=6,
                             n_subsample=n_subsample,
                             neighbor_function="4_quadrant",
                             scan_pattern="Isotropic",
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=4, C_forward=4)
    return data_config, model_config


@pytest.mark.parametrize("n_subsample", [1, 3])
def test_quadrant_grouping_allocates_true_group_count(tmp_path, n_subsample):
    """8x8 raster: only the 6x6 interior forms complete quadrant groups."""
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "b.npz", len(x), x, y)

    dataset = _build(tmp_path, *_quadrant_configs(n_subsample))

    expected_groups = 36 * n_subsample  # 6x6 interior centers, times subsampling
    assert len(dataset) == expected_groups
    assert dataset.cum_length == [0, expected_groups]
    for key, shape in (("images", (expected_groups, 4, N_PIX, N_PIX)),
                       ("coords_relative", (expected_groups, 4, 1, 2)),
                       ("coords_center", (expected_groups, 1, 1, 2)),
                       ("nn_indices", (expected_groups, 4))):
        assert dataset.mmap_ptycho[key].shape == shape, key


def test_quadrant_grouping_writes_every_allocated_row(tmp_path):
    """No unwritten tail: every allocated row carries a real coordinate group."""
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "b.npz", len(x), x, y)

    dataset = _build(tmp_path, *_quadrant_configs(1))

    # nn_indices are global scan indices; an unwritten MemoryMappedTensor row
    # would be an all-zero group, which a real quadrant group never is.
    nn = dataset.mmap_ptycho["nn_indices"]
    assert int(nn.max()) < len(x)
    assert not bool((nn == 0).all(dim=1).any())
    assert float(dataset.mmap_ptycho["rms_scaling_constant"].min()) > 0


def test_mmap_grouping_matches_owner_plan_across_object_banks(tmp_path):
    from ptycho.grouping import plan_scan_centered

    source_dir = tmp_path / "npz"
    source_dir.mkdir()
    base_x, base_y = _raster(3, spacing=1.0)
    xcoords = np.concatenate([base_x, base_x])
    ycoords = np.concatenate([base_y, base_y])
    object_index = np.repeat(np.arange(2, dtype=np.int64), len(base_x))
    path = source_dir / "object_banks.npz"
    _write_npz(path, len(xcoords), xcoords, ycoords)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez(path, **arrays, object_index=object_index)

    data_config = DataConfig(
        N=N_PIX,
        grid_size=(2, 2),
        C=4,
        K=4,
        n_subsample=1,
        subsample_seed=5,
        neighbor_function="Nearest",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    dataset = _build(
        tmp_path,
        data_config,
        ModelConfig(C_model=4, C_forward=4),
    )
    expected = plan_scan_centered(
        xcoords,
        ycoords,
        eligible_indices=np.arange(len(xcoords)),
        object_index=object_index,
        experiment_id=0,
        policy="Nearest",
        group_size=4,
        neighbor_count=4,
        repeats=1,
        seed=5,
    )

    rows = dataset.mmap_ptycho["nn_indices"].cpu().numpy()
    centers = dataset.mmap_ptycho["center_scan_id"].cpu().numpy()
    np.testing.assert_array_equal(rows, expected.neighbor_indices)
    np.testing.assert_array_equal(centers, expected.center_indices)
    assert len(dataset) == len(expected.neighbor_indices)
    assert all(np.unique(object_index[row]).size == 1 for row in rows)


def test_mmap_coords_relative_uses_tf_sign(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "sign.npz", len(x), x, y)

    np.random.seed(123)
    dataset = _build(tmp_path, *_quadrant_configs(1))

    coords_global = dataset.mmap_ptycho["coords_global"]
    coords_center = dataset.mmap_ptycho["coords_center"]
    coords_relative = dataset.mmap_ptycho["coords_relative"]
    expected = -(coords_global - coords_center)

    torch.testing.assert_close(coords_relative, expected, rtol=0, atol=1e-6)
    assert coords_relative.abs().max() > 0


def test_quadrant_grouping_is_not_redrawn_on_write(tmp_path):
    """coords_center must match the grouping used to size the map."""
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "b.npz", len(x), x, y)

    dataset = _build(tmp_path, *_quadrant_configs(1))

    # Recompute the group centroid from the stored global coords and compare
    # against the stored center; a regrouped write pass would desync these.
    coords_global = dataset.mmap_ptycho["coords_global"]  # (M, C, 1, 2)
    centroid = coords_global.mean(dim=1, keepdim=True)
    torch.testing.assert_close(centroid, dataset.mmap_ptycho["coords_center"],
                               rtol=1e-4, atol=1e-4)


def test_nearest_gs1_length_unchanged(tmp_path):
    """Default 'Nearest' gs1 path keeps n_valid * n_subsample, as before."""
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(40)
    _write_npz(tmp_path / "npz" / "c.npz", 40, x, y)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=7,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    dataset = _build(tmp_path, data_config, ModelConfig(C_model=1, C_forward=1))

    assert len(dataset) == 40 * 7


def test_supervised_object_big_sizes_without_subsampling(tmp_path):
    """Supervised takes the ungrouped write branch, so it must be sized for it.

    calculate_length previously keyed the n_subsample multiplier on object_big
    alone while memory_map_data grouped only for Unsupervised, so this config
    allocated n_subsample times too many rows.
    """
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(30)
    rng = np.random.default_rng(1)
    raw = rng.random((30, N_PIX, N_PIX)).astype(np.float32)
    diff3d = raw / np.sqrt((raw ** 2).sum(axis=(-2, -1), keepdims=True))
    label = (rng.random((30, N_PIX, N_PIX)) + 1j * rng.random((30, N_PIX, N_PIX)))
    np.savez(tmp_path / "npz" / "d.npz", xcoords=x, ycoords=y, diff3d=diff3d,
             probeGuess=(rng.random((N_PIX, N_PIX)) + 1j * rng.random((N_PIX, N_PIX))),
             objectGuess=(rng.random((N_PIX, N_PIX)) + 1j * rng.random((N_PIX, N_PIX))),
             label=label)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=7,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, mode='Supervised')
    dataset = _build(tmp_path, data_config, model_config)

    assert len(dataset) == 30
    assert dataset.mmap_ptycho["label_amp"].shape[0] == 30


# ---------------------------------------------------------------------------
# Local grouping RNG ownership
# ---------------------------------------------------------------------------

def _assert_legacy_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_group_coords_uses_dataconfig_seed_without_ambient_numpy_state():
    from ptycho_torch.patch_generator import get_neighbor_indices, group_coords

    xcoords, ycoords = _raster(5, spacing=1.0)
    valid = np.arange(len(xcoords), dtype=np.int64)
    config = DataConfig(
        N=N_PIX,
        C=4,
        K=8,
        n_subsample=3,
        subsample_seed=1447,
        grid_size=(2, 2),
        neighbor_function="Nearest",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    np.random.seed(6621)
    ambient_before = np.random.get_state()

    first_indices, first_coords = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        get_neighbor_indices,
        valid,
        config,
    )
    second_indices, second_coords = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        get_neighbor_indices,
        valid,
        config,
    )

    _assert_legacy_rng_state_equal(ambient_before, np.random.get_state())
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_coords, second_coords)


@pytest.mark.parametrize("policy", ["Nearest", "Min_dist", "4_quadrant"])
def test_group_coords_matches_scan_centered_owner_with_object_identity(policy):
    from ptycho.grouping import plan_scan_centered
    from ptycho_torch.patch_generator import group_coords

    base_x, base_y = _raster(3, spacing=1.0)
    xcoords = np.concatenate([base_x, base_x])
    ycoords = np.concatenate([base_y, base_y])
    object_index = np.repeat(np.arange(2, dtype=np.int64), len(base_x))
    experiment_id = np.arange(len(xcoords), dtype=np.int64) + 40
    valid = np.arange(len(xcoords), dtype=np.int64)
    config = DataConfig(
        N=N_PIX,
        C=4,
        K=8,
        K_quadrant=20,
        n_subsample=2,
        subsample_seed=23,
        grid_size=(2, 2),
        neighbor_function=policy,
        min_neighbor_distance=0.0,
        max_neighbor_distance=3.0,
        scan_pattern="Isotropic",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )

    rows, coords, centers = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        None,
        valid,
        config,
        return_center_indices=True,
        object_index=object_index,
        experiment_id=experiment_id,
    )
    expected = plan_scan_centered(
        xcoords,
        ycoords,
        eligible_indices=valid,
        object_index=object_index,
        experiment_id=experiment_id,
        policy=policy,
        group_size=4,
        neighbor_count=8,
        repeats=2,
        seed=23,
        min_neighbor_distance=0.0,
        max_neighbor_distance=3.0,
        quadrant_neighbor_count=20,
        scan_pattern="Isotropic",
    )

    np.testing.assert_array_equal(rows, expected.neighbor_indices)
    np.testing.assert_array_equal(centers, expected.center_indices)
    assert rows.flags.writeable
    assert centers.flags.writeable
    np.testing.assert_array_equal(
        coords,
        np.stack([xcoords[rows], ycoords[rows]], axis=2)[:, :, None, :],
    )


def test_nearest_group_coords_can_repair_complete_participant_coverage():
    """One reconstruction group per center must cover every eligible scan."""
    from ptycho_torch.patch_generator import get_neighbor_indices, group_coords

    coordinate_rng = np.random.default_rng(118549108)
    xcoords = coordinate_rng.uniform(64.0, 328.0, size=1024)
    ycoords = coordinate_rng.uniform(64.0, 328.0, size=1024)
    valid = np.arange(len(xcoords), dtype=np.int64)
    config = DataConfig(
        N=N_PIX,
        C=4,
        K=4,
        n_subsample=1,
        subsample_seed=523213049,
        grid_size=(2, 2),
        neighbor_function="Nearest",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )

    np.random.seed(34521)
    ambient_before = np.random.get_state()
    original_indices, _original_coords, _original_centers = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        get_neighbor_indices,
        valid,
        config,
        return_center_indices=True,
    )
    first_indices, first_coords, first_centers = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        get_neighbor_indices,
        valid,
        config,
        return_center_indices=True,
        ensure_complete_coverage=True,
    )
    second_indices, second_coords, second_centers = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        get_neighbor_indices,
        valid,
        config,
        return_center_indices=True,
        ensure_complete_coverage=True,
    )

    _assert_legacy_rng_state_equal(ambient_before, np.random.get_state())
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_coords, second_coords)
    np.testing.assert_array_equal(first_centers, second_centers)
    assert first_indices.shape == (len(valid), 4)
    assert first_centers.tolist() == valid.tolist()
    assert all(len(set(row.tolist())) == 4 for row in first_indices)
    missing_before = set(valid.tolist()) - set(
        original_indices.reshape(-1).tolist()
    )
    assert missing_before == {695, 720, 920}
    changed_rows = np.any(first_indices != original_indices, axis=1)
    assert int(np.count_nonzero(changed_rows)) == len(missing_before)
    assert set(first_indices.reshape(-1).tolist()) == set(valid.tolist())


def test_complete_coverage_accepts_boolean_mask_and_small_k_equals_c():
    """Boolean masks and K=C=N must not create -1/sentinel participants."""
    from ptycho_torch.patch_generator import get_neighbor_indices, group_coords

    xcoords, ycoords = _raster(2, spacing=1.0)
    valid_mask = np.ones(len(xcoords), dtype=np.bool_)
    config = DataConfig(
        N=N_PIX,
        C=4,
        K=4,
        n_subsample=1,
        subsample_seed=9,
        grid_size=(2, 2),
        neighbor_function="Nearest",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )

    indices, _coords, centers = group_coords(
        xcoords,
        ycoords,
        xcoords,
        ycoords,
        get_neighbor_indices,
        valid_mask,
        config,
        return_center_indices=True,
        ensure_complete_coverage=True,
    )

    assert centers.tolist() == [0, 1, 2, 3]
    assert indices.shape == (4, 4)
    assert np.all(indices >= 0)
    assert all(len(set(row.tolist())) == 4 for row in indices)
    assert set(indices.reshape(-1).tolist()) == {0, 1, 2, 3}


def test_real_mmap_persists_and_identifies_complete_group_coverage(tmp_path):
    source_dir = tmp_path / "npz"
    source_dir.mkdir()
    xcoords, ycoords = _raster(8, spacing=1.0)
    _write_npz(
        source_dir / "scan.npz",
        len(xcoords),
        xcoords,
        ycoords,
    )
    data_config = DataConfig(
        N=N_PIX,
        C=4,
        K=4,
        n_subsample=1,
        subsample_seed=523213049,
        grid_size=(2, 2),
        neighbor_function="Nearest",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig(C_model=4, C_forward=4)
    map_dir = tmp_path / "mmap" / "memmap"

    anchored = PtychoDataset(
        ptycho_dir=str(source_dir),
        model_config=model_config,
        data_config=data_config,
        training_config=TrainingConfig(batch_size=8),
        data_dir=str(map_dir),
        remake_map=True,
        require_complete_group_coverage=True,
    )
    rows = torch.as_tensor(anchored.mmap_ptycho["nn_indices"]).cpu().numpy()
    centers = (
        torch.as_tensor(anchored.mmap_ptycho["center_scan_id"])
        .cpu()
        .numpy()
        .reshape(-1)
    )

    assert rows.shape == (len(xcoords), 4)
    assert centers.tolist() == list(range(len(xcoords)))
    assert set(rows.reshape(-1).tolist()) == set(centers.tolist())
    assert all(len(set(row.tolist())) == 4 for row in rows)

    with pytest.raises(ValueError, match="require_complete_group_coverage"):
        PtychoDataset(
            ptycho_dir=str(source_dir),
            model_config=model_config,
            data_config=data_config,
            training_config=TrainingConfig(batch_size=8),
            data_dir=str(map_dir),
            remake_map=False,
        )

    reused = PtychoDataset(
        ptycho_dir=str(source_dir),
        model_config=model_config,
        data_config=data_config,
        training_config=TrainingConfig(batch_size=8),
        data_dir=str(map_dir),
        remake_map=False,
        require_complete_group_coverage=True,
    )
    torch.testing.assert_close(
        reused.mmap_ptycho["nn_indices"],
        anchored.mmap_ptycho["nn_indices"],
        rtol=0,
        atol=0,
    )


def test_fresh_mmap_builds_replay_grouping_without_ambient_numpy_state(
    tmp_path,
):
    """The dataset build owns one reproducible grouping stream per mmap."""
    source_dir = tmp_path / "npz"
    source_dir.mkdir()
    xcoords, ycoords = _raster(8)
    _write_npz(source_dir / "scan.npz", len(xcoords), xcoords, ycoords)
    data_config, model_config = _quadrant_configs(n_subsample=3)

    def build(map_name):
        return PtychoDataset(
            ptycho_dir=str(source_dir),
            model_config=model_config,
            data_config=data_config,
            training_config=TrainingConfig(batch_size=8),
            data_dir=str(tmp_path / map_name),
            remake_map=True,
        )

    np.random.seed(8347)
    ambient_before = np.random.get_state()
    first = build("mmap-first")
    second = build("mmap-second")

    _assert_legacy_rng_state_equal(ambient_before, np.random.get_state())
    torch.testing.assert_close(
        first.mmap_ptycho["nn_indices"],
        second.mmap_ptycho["nn_indices"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.mmap_ptycho["coords_global"],
        second.mmap_ptycho["coords_global"],
        rtol=0,
        atol=0,
    )


def test_quadrant_grouping_uses_global_center_identity_and_local_rng(monkeypatch):
    from ptycho_torch.patch_generator import group_coords

    xcoords, ycoords = _raster(8, spacing=1.5)
    # Deliberately make bounded row ids differ from their global scan ids.
    valid = np.array([18, 19, 20, 26, 27, 28, 34, 35, 36], dtype=np.int64)
    config = DataConfig(
        N=N_PIX,
        C=4,
        K=8,
        K_quadrant=30,
        n_subsample=2,
        subsample_seed=909,
        grid_size=(2, 2),
        neighbor_function="4_quadrant",
        min_neighbor_distance=0.0,
        max_neighbor_distance=3.0,
        scan_pattern="Isotropic",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    monkeypatch.setattr(
        np.random,
        "choice",
        lambda *_args, **_kwargs: pytest.fail(
            "grouping read ambient np.random.choice"
        ),
    )

    rows, _coords, centers = group_coords(
        xcoords,
        ycoords,
        xcoords[valid],
        ycoords[valid],
        None,
        valid,
        config,
        return_center_indices=True,
    )

    assert len(rows) > 0
    assert set(centers).issubset(set(valid))
    assert all(len(set(row.tolist())) == 4 for row in rows)

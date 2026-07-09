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
import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import (
    PtychoDataset,
    _align_coords_to_diffraction,
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


# ---------------------------------------------------------------------------
# Coordinate / diffraction alignment
# ---------------------------------------------------------------------------

def test_align_coords_drops_trailing_positions_with_warning():
    x, y = _line_scan(25)
    with pytest.warns(RuntimeWarning, match="dropping the trailing 5 positions"):
        xa, ya = _align_coords_to_diffraction(x, y, 20, "fixture.npz")
    assert len(xa) == len(ya) == 20
    np.testing.assert_array_equal(xa, x[:20])


def test_align_coords_rejects_missing_positions():
    x, y = _line_scan(15)
    with pytest.raises(ValueError, match="Every pattern needs a position"):
        _align_coords_to_diffraction(x, y, 20, "fixture.npz")


def test_align_coords_rejects_unequal_xy_lengths():
    x, _ = _line_scan(20)
    _, y = _line_scan(19)

    with pytest.raises(ValueError) as excinfo:
        _align_coords_to_diffraction(x, y, 20, "fixture.npz")

    message = str(excinfo.value)
    assert "fixture.npz" in message
    assert "xcoords=20" in message
    assert "ycoords=19" in message


def test_align_coords_passthrough_when_matched():
    x, y = _line_scan(20)
    xa, ya = _align_coords_to_diffraction(x, y, 20, "fixture.npz")
    assert xa is x and ya is y


def test_memory_map_survives_extra_coordinates(tmp_path):
    """25 positions, 20 patterns: previously IndexError deep in the write loop."""
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(25)
    _write_npz(tmp_path / "npz" / "a.npz", 20, x, y)

    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=1,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1)
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
        ValueError, match=r"flat\.npz.*3D.*\(N, H, W\).*shape \(32, 32\)"
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


def test_coordinate_count_keeps_canonical_stack_when_size_heuristic_disagrees(tmp_path):
    path = tmp_path / "canonical.npz"
    x, y = _line_scan(20)
    diff3d = np.arange(20 * 16 * 32, dtype=np.float32).reshape(20, 16, 32)
    np.savez(path, xcoords=x, ycoords=y, diff3d=diff3d)

    shape, _, _ = npz_headers(path)
    loaded = _get_diffraction_stack(path)

    assert shape == (20, 16, 32)
    assert loaded.shape == (20, 16, 32)
    np.testing.assert_array_equal(loaded, diff3d)


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
    np.savez(path, xcoords=x, ycoords=y, diff3d=canonical)

    with pytest.warns(RuntimeWarning, match="dropping the trailing 12 positions"):
        shape, xa, ya = npz_headers(path)
    loaded = _get_diffraction_stack(path)

    assert shape == (20, 32, 32)
    assert len(xa) == len(ya) == 20
    np.testing.assert_array_equal(loaded, canonical)


def test_square_plane_transposes_legacy_stack_despite_coordinate_collision(tmp_path):
    path = tmp_path / "legacy_trailing_collision.npz"
    x, y = _line_scan(32)
    canonical = np.arange(20 * 32 * 32, dtype=np.float32).reshape(20, 32, 32)
    legacy = np.transpose(canonical, (1, 2, 0))
    np.savez(path, xcoords=x, ycoords=y, diff3d=legacy)

    with pytest.warns(RuntimeWarning, match="dropping the trailing 12 positions"):
        shape, xa, ya = npz_headers(path)
    loaded = _get_diffraction_stack(path)

    assert shape == (20, 32, 32)
    assert len(xa) == len(ya) == 20
    np.testing.assert_array_equal(loaded, canonical)


def test_ambiguous_layout_retains_existing_size_heuristic(tmp_path):
    path = tmp_path / "ambiguous.npz"
    x, y = _line_scan(25)
    legacy = np.arange(12 * 8 * 20, dtype=np.float32).reshape(12, 8, 20)
    np.savez(path, xcoords=x, ycoords=y, diff3d=legacy)

    with pytest.warns(RuntimeWarning, match="dropping the trailing 5 positions"):
        shape, xa, ya = npz_headers(path)
    loaded = _get_diffraction_stack(path)

    assert shape == (20, 12, 8)
    assert len(xa) == len(ya) == 20
    np.testing.assert_array_equal(loaded, np.transpose(legacy, (2, 0, 1)))


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

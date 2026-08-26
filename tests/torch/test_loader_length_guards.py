"""Regression tests for the memory-map sizing and grouping guards in
``PtychoDataset``.

- ``537aa175`` truncated scan positions to the diffraction stack length. Some
  datasets carry trailing coordinate entries with no matching pattern; without
  the guard those indices run off the end of the diffraction stack and
  ``memory_map_data`` dies with ``IndexError``.

The loader now plans every bounded row through ``plan_nearest_groups`` once in
``calculate_length`` (centered-nearest-v1 contract) and reuses that cached
``GroupingPlan`` during the mmap write, so the allocation, ``cum_length``, and
the written tensors agree by construction. These tests pin that the cached
plan is reused without regrouping, that its row count and the final write
cursor must match the allocation fail-loud, that persisted center ids are
mapped through ``plan.source_indices``, and that the mmap schema advances to
version 5 with the centered contract and no availability field.
"""
import json

import numpy as np
import pytest
import torch

from ptycho.grouping import plan_nearest_groups
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


def _build(tmp_path, data_config, model_config, groups_per_center=1):
    return PtychoDataset(
        ptycho_dir=str(tmp_path / "npz"), model_config=model_config,
        data_config=data_config, training_config=TrainingConfig(batch_size=8),
        data_dir=str(tmp_path / "mm"), remake_map=True,
        groups_per_center=groups_per_center,
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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig()
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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

    with pytest.raises(ValueError, match=r"a\.npz.*15 scan positions.*20 diffraction"):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_dataset_rejects_non_1d_coordinates_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    xcoords = np.zeros((20, 2), dtype=np.float64)
    ycoords = np.ones((20, 2), dtype=np.float64)
    _write_npz(tmp_path / "npz" / "bad_coords.npz", 20, xcoords, ycoords)

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

    with pytest.raises(ValueError, match=r"bad_probe\.npz.*probeGuess.*shape"):
        _build(tmp_path, data_config, model_config)

    mm_path = tmp_path / "mm"
    assert not mm_path.exists() or not any(mm_path.iterdir())


def test_supervised_dataset_rejects_missing_label_before_allocation(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "missing_label.npz", 20, x, y)

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(mode="Supervised",
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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(mode="Supervised",
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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

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
        lambda self: (0, (N_PIX, N_PIX), [0], [], [], []),
    )
    monkeypatch.setattr("ptycho_torch.dataloader.get_current_rank", lambda: 1)
    monkeypatch.setattr(
        "ptycho_torch.dataloader.is_ddp_initialized_and_active", lambda: True
    )
    monkeypatch.setattr("ptycho_torch.dataloader.dist.barrier", barrier_must_not_run)

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)

    with pytest.raises(ValueError, match=r"calculate_length\(\) resulted in 0 items"):
        _build(tmp_path, data_config, model_config)


def test_stale_v1_mmap_without_center_scan_id_requires_rebuild(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "stale.npz", 20, x, y)
    data_config = DataConfig(
        N=N_PIX, gridsize=1, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0)
    )
    model_config = ModelConfig(object_big=False)
    _build(tmp_path, data_config, model_config)
    manifest_path = tmp_path / "mmap_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest["required_fields"].remove("center_scan_id")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RuntimeError,
        match=r"schema_version=1.*expected 5.*Rebuild it with remake_map=True",
    ):
        PtychoDataset.from_existing_map(
            tmp_path / "mm", model_config, data_config
        )


def test_v4_mmap_requires_remake_map(tmp_path):
    """Schema-4 stores (with the availability field) must fail with the
    established remake_map=True instruction; no old-row upgrader exists."""
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "v4.npz", len(x), x, y)
    data_config = DataConfig(
        N=N_PIX, gridsize=1, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0)
    )
    model_config = ModelConfig(object_big=False)
    _build(tmp_path, data_config, model_config)
    manifest_path = tmp_path / "mmap_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 4
    manifest["required_fields"] = sorted(
        set(manifest["required_fields"]) | {"center_scan_id_available"}
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RuntimeError,
        match=r"schema_version=4.*expected 5.*Rebuild it with remake_map=True",
    ):
        PtychoDataset.from_existing_map(
            tmp_path / "mm", model_config, data_config
        )


def test_mmap_schema_v5_requires_centered_contract_without_availability_field(
    tmp_path,
):
    """Fresh maps are schema 5, record the centered contract, and never carry
    the retired center availability field."""
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "v5.npz", len(x), x, y)
    data_config = DataConfig(
        N=N_PIX, gridsize=1, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0)
    )
    model_config = ModelConfig(object_big=False)
    dataset = _build(tmp_path, data_config, model_config)
    manifest = json.loads(dataset.manifest_path.read_text())

    assert manifest["schema_version"] == 5
    assert manifest["grouping_contract"] == "centered-nearest-v1"
    assert "center_scan_id" in manifest["required_fields"]
    assert "center_scan_id_available" not in manifest["required_fields"]
    assert "center_scan_id_available" not in dataset.mmap_ptycho.keys()

    # A v5 store reloads cleanly.
    reused = PtychoDataset.from_existing_map(
        tmp_path / "mm", model_config, data_config
    )
    assert len(reused) == len(x)

    # A manifest without the centered contract is rejected with the rebuild
    # instruction.
    manifest.pop("grouping_contract")
    dataset.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        RuntimeError, match=r"grouping_contract.*remake_map=True"
    ):
        PtychoDataset.from_existing_map(
            tmp_path / "mm", model_config, data_config
        )


def test_memory_map_loads_legacy_hwn_layout(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(40)
    _write_npz(tmp_path / "npz" / "legacy_hwn.npz", 40, x, y,
               pattern_size=32, legacy_hwn=True)

    data_config = DataConfig(N=32, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)
    dataset = _build(tmp_path, data_config, model_config)

    assert len(dataset) == 40
    assert dataset.mmap_ptycho["images"].shape == (40, 1, 32, 32)
    assert int(dataset.mmap_ptycho["nn_indices"].max()) < 40


def test_memory_map_loads_legacy_hwn_layout_when_n_is_not_largest_axis(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "legacy_hwn.npz", 20, x, y,
               pattern_size=32, legacy_hwn=True)

    data_config = DataConfig(N=32, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0),
                             y_bounds=(0.0, 1.0))
    model_config = ModelConfig(object_big=False)
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
# Centered-nearest mmap plan/cache contract
# ---------------------------------------------------------------------------

def _assert_legacy_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_bounded_rows_are_centers_and_candidates_with_contiguous_repeats(
    tmp_path,
):
    """Every bounded row is both a center and a candidate; repeats are
    contiguous so each center appears in column zero of consecutive rows."""
    (tmp_path / "npz").mkdir()
    xcoords, ycoords = _raster(8)
    _write_npz(tmp_path / "npz" / "bounded.npz", len(xcoords), xcoords, ycoords)

    xmin, xmax = xcoords.min(), xcoords.max()
    ymin, ymax = ycoords.min(), ycoords.max()
    x_lower = xmin + 0.15 * (xmax - xmin)
    x_upper = xmin + 0.85 * (xmax - xmin)
    y_lower = ymin + 0.15 * (ymax - ymin)
    y_upper = ymin + 0.85 * (ymax - ymin)
    mask = (
        (xcoords >= x_lower)
        & (xcoords <= x_upper)
        & (ycoords >= y_lower)
        & (ycoords <= y_upper)
    )
    valid = np.where(mask)[0]

    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=6,
        subsample_seed=0,
        x_bounds=(0.15, 0.85),
        y_bounds=(0.15, 0.85),
    )
    dataset = _build(tmp_path, data_config, ModelConfig(), groups_per_center=2)

    centers = dataset.mmap_ptycho["center_scan_id"].cpu().numpy().reshape(-1)
    nn = dataset.mmap_ptycho["nn_indices"].cpu().numpy()
    assert nn.shape == (len(dataset), 4)
    np.testing.assert_array_equal(centers, np.repeat(valid, 2))
    np.testing.assert_array_equal(nn[:, 0], centers)
    assert set(nn.reshape(-1).tolist()) <= set(valid.tolist())
    assert len(dataset) == len(valid) * 2


def test_length_and_write_reuse_one_cached_grouping_plan(tmp_path, monkeypatch):
    """The length pass plans once per file; the write pass reuses that plan
    instead of regrouping (a regroup would draw a second RNG stream)."""
    from dataclasses import replace

    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "reuse.npz", len(x), x, y)

    calls = []
    real_plan = plan_nearest_groups

    def spied_plan(*args, **kwargs):
        calls.append(kwargs["center_indices"])
        return real_plan(*args, **kwargs)

    monkeypatch.setattr("ptycho_torch.dataloader.plan_nearest_groups", spied_plan)

    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=6,
        subsample_seed=7,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    dataset = _build(tmp_path, data_config, ModelConfig(), groups_per_center=2)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.arange(len(x)))
    assert dataset.cum_length == [0, len(x) * 2]
    # The cached plan was released once written, and the written rows agree
    # with the single planned group count.
    assert all(record is None for record in dataset.grouping_per_file)
    assert dataset.mmap_ptycho["nn_indices"].shape[0] == len(x) * 2

    # The write pass also materialized coordinates from that same plan: the
    # stored group centroid equals the centroid of the stored global coords.
    coords_global = dataset.mmap_ptycho["coords_global"]
    centroid = coords_global.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        centroid,
        dataset.mmap_ptycho["coords_center"],
        rtol=1e-4,
        atol=1e-4,
    )


def test_mmap_write_count_must_match_cached_plan(tmp_path, monkeypatch):
    """A cached plan whose row count disagrees with the allocation fails
    loudly instead of exposing a partial memory map."""
    from dataclasses import replace

    from ptycho.grouping import plan_nearest_groups as real_plan

    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "count.npz", len(x), x, y)
    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=6,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )

    # Guard 1: the length pass refuses a plan whose row count contradicts the
    # bounded rows x groups_per_center allocation.
    def short_plan(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        return replace(
            plan,
            neighbor_indices=plan.neighbor_indices[:5],
            center_indices=plan.center_indices[:5],
            object_index=plan.object_index[:5],
            experiment_id=plan.experiment_id[:5],
        )

    monkeypatch.setattr("ptycho_torch.dataloader.plan_nearest_groups", short_plan)
    with pytest.raises(ValueError, match="cached grouping plan"):
        _build(tmp_path, data_config, ModelConfig())

    # Guard 2: an allocation slice that disagrees with the cached plan fails
    # before any partial map is exposed.
    monkeypatch.setattr(
        "ptycho_torch.dataloader.plan_nearest_groups",
        real_plan,
    )
    calculate_length = PtychoDataset.calculate_length

    def tampered_calculate_length(self):
        length, shape, cumulative, valid, source, grouping = calculate_length(self)
        return length - 1, shape, [0, length - 1], valid, source, grouping

    monkeypatch.setattr(
        PtychoDataset, "calculate_length", tampered_calculate_length
    )
    with pytest.raises(RuntimeError, match="allocated mmap slice"):
        _build(tmp_path, data_config, ModelConfig())


def test_mmap_center_ids_are_mapped_from_plan_source_indices(tmp_path):
    """Persisted center ids are the plan-mapped bounded source rows, not local
    rows inside the bounded subset."""
    (tmp_path / "npz").mkdir()
    xcoords, ycoords = _raster(8)
    _write_npz(
        tmp_path / "npz" / "mapped.npz", len(xcoords), xcoords, ycoords
    )

    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=6,
        x_bounds=(0.25, 0.75),
        y_bounds=(0.25, 0.75),
    )
    dataset = _build(tmp_path, data_config, ModelConfig())

    valid = dataset.valid_indices_per_file[0]
    assert len(valid) < len(xcoords)
    centers = dataset.mmap_ptycho["center_scan_id"].cpu().numpy().reshape(-1)
    nn = dataset.mmap_ptycho["nn_indices"].cpu().numpy()
    # centers are the global bounded source rows (mapped through
    # plan.source_indices), and every group carries its center in column zero.
    np.testing.assert_array_equal(centers, valid)
    np.testing.assert_array_equal(nn[:, 0], centers)
    assert set(nn.reshape(-1).tolist()) <= set(valid.tolist())


def test_object_banks_never_cross_partitions(tmp_path):
    """A group never mixes independent object canvases; the persisted
    object_index matches the mapped center's partition."""
    (tmp_path / "npz").mkdir()
    base_x, base_y = _raster(3, spacing=1.0)
    xcoords = np.concatenate([base_x, base_x])
    ycoords = np.concatenate([base_y, base_y])
    object_index = np.repeat(np.arange(2, dtype=np.int64), len(base_x))
    path = tmp_path / "npz" / "object_banks.npz"
    _write_npz(path, len(xcoords), xcoords, ycoords)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez(path, **arrays, object_index=object_index)

    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=4,
        subsample_seed=5,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    dataset = _build(tmp_path, data_config, ModelConfig())

    rows = dataset.mmap_ptycho["nn_indices"].cpu().numpy()
    centers = dataset.mmap_ptycho["center_scan_id"].cpu().numpy().reshape(-1)
    stored_objects = (
        dataset.mmap_ptycho["object_index"].cpu().numpy().reshape(-1)
    )
    assert len(dataset) == len(xcoords)
    assert all(np.unique(object_index[row]).size == 1 for row in rows)
    np.testing.assert_array_equal(centers, np.arange(len(xcoords)))
    np.testing.assert_array_equal(stored_objects, object_index[centers])


def test_c1_mmap_grouping_is_centered_identity(tmp_path):
    """C=1 plans each ordered bounded row as its own centered group."""
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "c1.npz", len(x), x, y)

    data_config = DataConfig(
        N=N_PIX,
        gridsize=1,
        neighbor_count=1,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    dataset = _build(tmp_path, data_config, ModelConfig(), groups_per_center=3)

    nn = dataset.mmap_ptycho["nn_indices"].cpu().numpy()
    centers = dataset.mmap_ptycho["center_scan_id"].cpu().numpy().reshape(-1)
    expected = np.repeat(np.arange(len(x)), 3)
    np.testing.assert_array_equal(centers, expected)
    np.testing.assert_array_equal(nn, expected[:, None])
    assert len(dataset) == len(x) * 3


def test_mmap_coords_relative_uses_tf_sign(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "sign.npz", len(x), x, y)

    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=6,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    dataset = _build(tmp_path, data_config, ModelConfig())

    coords_global = dataset.mmap_ptycho["coords_global"]
    coords_center = dataset.mmap_ptycho["coords_center"]
    coords_relative = dataset.mmap_ptycho["coords_relative"]
    expected = -(coords_global - coords_center)

    torch.testing.assert_close(coords_relative, expected, rtol=0, atol=1e-6)
    assert coords_relative.abs().max() > 0


def test_nearest_gs1_length_unchanged(tmp_path):
    """Default 'Nearest' gs1 path keeps n_valid * n_subsample, as before."""
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(40)
    _write_npz(tmp_path / "npz" / "c.npz", 40, x, y)

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    dataset = _build(tmp_path, data_config, ModelConfig(), groups_per_center=7)

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

    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4,
                             x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig(mode='Supervised')
    dataset = _build(tmp_path, data_config, model_config)

    assert len(dataset) == 30
    assert dataset.mmap_ptycho["label_amp"].shape[0] == 30


# ---------------------------------------------------------------------------
# Local grouping RNG ownership
# ---------------------------------------------------------------------------

def test_fresh_mmap_builds_replay_grouping_without_ambient_numpy_state(
    tmp_path,
):
    """The dataset build owns one reproducible grouping stream per mmap."""
    source_dir = tmp_path / "npz"
    source_dir.mkdir()
    xcoords, ycoords = _raster(8)
    _write_npz(source_dir / "scan.npz", len(xcoords), xcoords, ycoords)
    data_config = DataConfig(
        N=N_PIX,
        gridsize=2,
        neighbor_count=6,
        subsample_seed=11,
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig()
    groups_per_center = 3

    def build(map_name):
        return PtychoDataset(
            ptycho_dir=str(source_dir),
            model_config=model_config,
            data_config=data_config,
            training_config=TrainingConfig(batch_size=8),
            data_dir=str(tmp_path / map_name),
            remake_map=True,
            groups_per_center=groups_per_center,
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

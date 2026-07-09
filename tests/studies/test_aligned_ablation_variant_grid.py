"""CPU-cheap tests for the methodology-critical pieces of the aligned
inference-variant grid harness (Task 4): coords bridge orientation, border
crop, and norm_Y_I gating (docs/findings.md REASSEMBLY-BRIDGE-001). Synthetic
arrays only -- no checkpoints, no GPU, no torch import required by the
module under test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.studies.aligned_ablation_variant_grid import (
    apply_norm_y_i_bridge,
    bridge_coords,
    build_bridge_arrays,
    crop_border,
    squeeze_singleton_dims,
    write_bridge_npz,
)


def test_bridge_coords_swaps_row_col_to_x_y():
    # coords_offsets column 0 = iy (row/vertical), column 1 = ix (col/horizontal)
    # per grid_lines_workflow._build_scan_positions -- xcoords must come from
    # column 1, ycoords from column 0, not passed through in column order.
    iy = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    ix = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    coords_offsets = np.zeros((3, 1, 2, 1), dtype=np.float32)
    coords_offsets[:, 0, 0, 0] = iy
    coords_offsets[:, 0, 1, 0] = ix

    xcoords, ycoords = bridge_coords(coords_offsets)

    np.testing.assert_array_equal(xcoords, ix)
    np.testing.assert_array_equal(ycoords, iy)


def test_bridge_coords_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        bridge_coords(np.zeros((3, 1, 3, 1), dtype=np.float32))


def test_squeeze_singleton_dims_drops_trailing_and_leading_axes():
    arr = np.zeros((1, 5, 5, 1), dtype=np.complex64)
    arr[0, 2, 3, 0] = 1 + 2j

    squeezed = squeeze_singleton_dims(arr)

    assert squeezed.shape == (5, 5)
    assert squeezed[2, 3] == 1 + 2j


def test_build_bridge_arrays_squeezes_diffraction_and_swaps_coords():
    n, h, w = 4, 6, 6
    iy = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ix = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    coords_offsets = np.zeros((n, 1, 2, 1), dtype=np.float32)
    coords_offsets[:, 0, 0, 0] = iy
    coords_offsets[:, 0, 1, 0] = ix

    test_data = {
        "coords_offsets": coords_offsets,
        "diffraction": np.ones((n, h, w, 1), dtype=np.float32) * 7.0,
        "probeGuess": np.ones((h, w), dtype=np.complex64),
        "YY_ground_truth": np.ones((h, w, 1), dtype=np.complex64) * 3.0,
    }

    bridged = build_bridge_arrays(test_data)

    assert bridged["diffraction"].shape == (n, h, w)
    np.testing.assert_array_equal(bridged["diffraction"], 7.0)
    np.testing.assert_array_equal(bridged["xcoords"], ix)
    np.testing.assert_array_equal(bridged["ycoords"], iy)
    assert bridged["objectGuess"].shape == (h, w)
    assert bridged["probeGuess"].shape == (h, w)


def test_write_bridge_npz_round_trips_expected_keys(tmp_path: Path):
    n, h, w = 2, 4, 4
    coords_offsets = np.zeros((n, 1, 2, 1), dtype=np.float32)
    coords_offsets[:, 0, 1, 0] = [1.0, 2.0]  # ix
    coords_offsets[:, 0, 0, 0] = [3.0, 4.0]  # iy
    source = tmp_path / "test.npz"
    np.savez(
        source,
        coords_offsets=coords_offsets,
        diffraction=np.zeros((n, h, w, 1), dtype=np.float32),
        probeGuess=np.ones((h, w), dtype=np.complex64),
        YY_ground_truth=np.ones((h, w, 1), dtype=np.complex64),
        norm_Y_I=np.float64(4.6975),
    )

    out_path = write_bridge_npz(source, tmp_path / "bridged.npz")

    with np.load(out_path) as bridged:
        assert set(bridged.files) == {"xcoords", "ycoords", "diffraction", "probeGuess", "objectGuess"}
        np.testing.assert_array_equal(bridged["xcoords"], [1.0, 2.0])
        np.testing.assert_array_equal(bridged["ycoords"], [3.0, 4.0])


def test_crop_border_removes_exact_margin_each_edge():
    canvas = np.arange(100).reshape(10, 10).astype(np.complex64)

    cropped = crop_border(canvas, border_px=2)

    assert cropped.shape == (6, 6)
    np.testing.assert_array_equal(cropped, canvas[2:8, 2:8])


def test_crop_border_noop_when_zero():
    canvas = np.arange(16).reshape(4, 4)
    assert crop_border(canvas, border_px=0) is canvas


def test_crop_border_rejects_oversized_trim():
    canvas = np.zeros((10, 10))
    with pytest.raises(ValueError):
        crop_border(canvas, border_px=5)


def test_apply_norm_y_i_bridge_applies_only_when_not_varpro_scaling():
    canvas = np.ones((3, 3), dtype=np.complex64) * 2.0
    norm_y_i = 4.6975

    novarpro_canvas, novarpro_applied = apply_norm_y_i_bridge(canvas, norm_y_i, varpro_scaling=False)
    varpro_canvas, varpro_applied = apply_norm_y_i_bridge(canvas, norm_y_i, varpro_scaling=True)

    assert novarpro_applied is True
    np.testing.assert_allclose(novarpro_canvas, canvas * norm_y_i)
    assert varpro_applied is False
    np.testing.assert_array_equal(varpro_canvas, canvas)

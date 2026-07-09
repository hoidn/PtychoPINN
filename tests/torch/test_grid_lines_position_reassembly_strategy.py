from pathlib import Path

import numpy as np
import pytest

from scripts.studies.grid_lines_torch_runner import (
    TorchRunnerConfig,
    _normalize_position_inputs,
    _reassemble_with_coords_offsets,
)


def test_large_external_n128_auto_prefers_shift_sum(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    def fake_shift_sum(patches, offsets_b112, M):
        _ = (patches, offsets_b112)
        return np.full((M, M), 2.0 + 0.0j, dtype=np.complex64)

    def fail_batched(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("batched backend should not be selected by auto")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fake_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        fail_batched,
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="auto",
        batch_size=16,
        position_crop_border=0,
    )

    assert out.shape == (128, 128)
    assert np.allclose(out, np.full((128, 128), 2.0 + 0.0j, dtype=np.complex64))


def test_explicit_batched_backend_still_uses_batched(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    def fail_shift_sum(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("shift_sum should not be used when backend='batched'")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fail_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        lambda patches, offsets_b12c, M, batch_size: np.full((M, M), 3.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="batched",
        batch_size=16,
        position_crop_border=0,
    )

    assert out.shape == (128, 128)
    assert np.allclose(out, np.full((128, 128), 3.0 + 0.0j, dtype=np.complex64))


def test_auto_default_applies_nonzero_crop(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        lambda patches, offsets_b112, M: np.full((M, M), 4.0 + 0.0j, dtype=np.complex64),
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
        lambda patches, offsets_b12c, M, batch_size: np.full((M, M), 8.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="auto",
        batch_size=16,
        position_crop_border=None,
    )

    assert out.shape == (64, 64)


def test_explicit_zero_crop_preserves_legacy_patch_size(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        lambda patches, offsets_b112, M: np.full((M, M), 2.0 + 0.0j, dtype=np.complex64),
    )

    out = _reassemble_with_coords_offsets(
        pred,
        test_data,
        M=128,
        backend="shift_sum",
        batch_size=16,
        position_crop_border=0,
    )

    assert out.shape == (128, 128)


def test_gridsize2_grid_lines_default_reroutes_to_position_path(monkeypatch):
    """gridsize>1 runs left at the default reassembly_mode='grid_lines' must
    reconstruct through the position path, not the flat-tiled
    grid_lines/stitch_predictions product (docs/findings.md#TORCH-GS2-STITCH-001).
    """
    from scripts.studies.grid_lines_torch_runner import _reassemble_predictions_for_metrics

    N = 16
    gridsize = 2
    channels = gridsize ** 2
    n_groups = 4
    # Channels-first per-patch predictions, matching forward_predict's real
    # inference output layout (B, C, H, W).
    pred_chw = np.ones((n_groups, channels, N, N), dtype=np.complex64)
    test_data = {
        # Real gs2 coords_offsets layout: B groups x C channels, which drives the
        # fixed per-patch mismatch branch of _normalize_position_inputs (not the
        # already-flat matching path).
        "coords_offsets": np.zeros((n_groups, 1, 2, channels), dtype=np.float32),
        "norm_Y_I": np.array(1.0, dtype=np.float32),
    }
    metadata = {"additional_parameters": {"nimgs_test": 1, "outer_offset_test": 16}}
    cfg = TorchRunnerConfig(
        train_npz=Path("train.npz"),
        test_npz=Path("test.npz"),
        output_dir=Path("out"),
        architecture="cnn",
        gridsize=gridsize,
        N=N,
        position_crop_border=0,
    )  # reassembly_mode left at its "grid_lines" default
    ground_truth = np.zeros((N, N), dtype=np.complex64)

    position_called = {"ok": False}
    grid_lines_called = {"ok": False}

    def fake_shift_sum(patches, offsets_b112, M):
        position_called["ok"] = True
        return np.full((M, M), 1.0 + 0.0j, dtype=np.complex64)

    def fake_stitch(pred_complex, cfg_obj, metadata_obj, norm_Y_I):
        grid_lines_called["ok"] = True
        # Stand-in for the flat-tiled stitch_predictions defect product: a
        # differently-shaped, differently-valued array than the position path.
        return np.full((2, 2), 9.0 + 0.0j, dtype=np.complex64)

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fake_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._stitch_for_metrics",
        fake_stitch,
    )

    output, effective_mode, runtime_contract = _reassemble_predictions_for_metrics(
        pred_chw, ground_truth, test_data, metadata, cfg
    )

    assert effective_mode == "position"
    assert position_called["ok"] is True
    assert grid_lines_called["ok"] is False
    assert np.squeeze(output).shape == (N, N)
    assert runtime_contract  # position runtime contract populated, not empty


def test_gridsize1_default_keeps_grid_lines_path(monkeypatch):
    """Pin: a gridsize=1 config routed through the same dispatcher the runner
    uses (_reassemble_predictions_for_metrics) must stay on the
    grid_lines/stitch_predictions path — the gridsize>1 auto-route must not
    fire (docs/findings.md#TORCH-GS2-STITCH-001)."""
    from scripts.studies.grid_lines_torch_runner import _reassemble_predictions_for_metrics

    N = 16
    n_groups = 4
    pred_chw = np.ones((n_groups, 1, N, N), dtype=np.complex64)
    test_data = {
        "coords_offsets": np.zeros((n_groups, 1, 2, 1), dtype=np.float32),
        "norm_Y_I": np.array(1.0, dtype=np.float32),
    }
    metadata = {"additional_parameters": {"nimgs_test": 1, "outer_offset_test": 16}}
    cfg = TorchRunnerConfig(
        train_npz=Path("train.npz"),
        test_npz=Path("test.npz"),
        output_dir=Path("out"),
        architecture="fno",
        gridsize=1,
        N=N,
    )
    assert cfg.reassembly_mode == "grid_lines"
    ground_truth = np.zeros((N, N), dtype=np.complex64)

    position_called = {"ok": False}
    grid_lines_called = {"ok": False}

    def fake_shift_sum(patches, offsets_b112, M):
        position_called["ok"] = True
        return np.full((M, M), 1.0 + 0.0j, dtype=np.complex64)

    def fake_stitch(pred_complex, cfg_obj, metadata_obj, norm_Y_I):
        grid_lines_called["ok"] = True
        return np.full((N, N), 5.0 + 0.0j, dtype=np.complex64)

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
        fake_shift_sum,
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner._stitch_for_metrics",
        fake_stitch,
    )

    output, effective_mode, runtime_contract = _reassemble_predictions_for_metrics(
        pred_chw, ground_truth, test_data, metadata, cfg
    )

    assert effective_mode == "grid_lines"
    assert grid_lines_called["ok"] is True
    assert position_called["ok"] is False
    assert np.squeeze(output).shape == (N, N)


def test_normalize_position_inputs_builds_per_patch_offsets_gs2():
    """gridsize>1 mismatch branch: coords_offsets (B,1,2,C) must expand to one
    global (x,y) offset per flattened patch (B*C,1,1,2), index-aligned to the
    (B*C,H,W,1) patch flattening (patch g*C+c -> coords_offsets[g,0,:,c]) with the
    stored (y,x) axis swapped to (x,y) (docs/findings.md#TORCH-GS2-POSITION-OFFSETS-001).
    """
    B, C, H, W = 2, 4, 10, 10
    # Channels-first complex predictions matching forward_predict's (B, C, H, W).
    pred = np.ones((B, C, H, W), dtype=np.complex64)
    # Distinct value per (group, channel, coord-row): row0 is the stored 'y',
    # row1 the stored 'x'.
    coords = np.zeros((B, 1, 2, C), dtype=np.float64)
    for g in range(B):
        for c in range(C):
            coords[g, 0, 0, c] = 1000.0 + 100.0 * g + 10.0 * c  # stored y
            coords[g, 0, 1, c] = 2000.0 + 100.0 * g + 10.0 * c  # stored x

    patches, offsets_b12c, offsets_b112, _crop = _normalize_position_inputs(
        pred, {"coords_offsets": coords}, position_crop_border=0
    )

    assert patches.shape == (B * C, H, W, 1)
    assert offsets_b112.shape == (B * C, 1, 1, 2)
    # Each flattened patch g*C+c carries its own offset, swapped to (x, y).
    for g in range(B):
        for c in range(C):
            expected_xy = coords[g, 0, ::-1, c]  # (x, y)
            np.testing.assert_allclose(offsets_b112[g * C + c, 0, 0, :], expected_xy)
    # Batched backend re-derives offsets via transpose(offsets_b12c,(0,1,3,2));
    # it must reproduce offsets_b112 exactly.
    assert offsets_b12c.shape == (B * C, 1, 2, 1)
    np.testing.assert_array_equal(
        np.transpose(offsets_b12c, (0, 1, 3, 2)), offsets_b112
    )


def test_normalize_position_inputs_rejects_channel_count_mismatch():
    """Fail fast (no silent repeat) when the patch/offset ratio is not the coords
    channel dimension (docs/findings.md#TORCH-GS2-POSITION-OFFSETS-001)."""
    B, C, H, W = 2, 4, 10, 10
    pred = np.ones((B, C, H, W), dtype=np.complex64)
    # coords channel dim (3) != prediction channel count (4) -> ratio 8/2=4 != 3.
    coords = np.zeros((B, 1, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot align coords_offsets"):
        _normalize_position_inputs(
            pred, {"coords_offsets": coords}, position_crop_border=0
        )


def test_position_path_reconstructs_synthetic_object_gs2():
    """Oracle: C=4 overlapping ground-truth patches placed at stored-convention
    (y,x) offsets must reconstruct the underlying object through the runner's real
    position path (backend 'shift_sum'). Pre-fix the offset collapse destroys the
    reconstruction; post-fix the central region matches the object exactly.
    """
    from ptycho import params as p_mod

    N = 16
    obj_size = 48
    p_mod.cfg["N"] = N
    rng = np.random.default_rng(12345)
    obj = (
        rng.standard_normal((obj_size, obj_size))
        + 1j * rng.standard_normal((obj_size, obj_size))
    ).astype(np.complex64)

    cx0, cy0 = 24, 24
    channel_offsets = [(-2, -2), (2, -2), (-2, 2), (2, 2)]  # (dx, dy)
    channels = len(channel_offsets)

    def window(cx, cy, size):
        return obj[cy - size // 2: cy + size // 2, cx - size // 2: cx + size // 2]

    # Channels-first per-patch predictions: patch c is the NxN crop of the object
    # centered at (cx0+dx, cy0+dy). Integer offsets keep translate exact.
    pred = np.zeros((1, channels, N, N), dtype=np.complex64)
    coords = np.zeros((1, 1, 2, channels), dtype=np.float64)
    for c, (dx, dy) in enumerate(channel_offsets):
        pred[0, c] = window(cx0 + dx, cy0 + dy, N)
        coords[0, 0, 0, c] = cy0 + dy  # stored y
        coords[0, 0, 1, c] = cx0 + dx  # stored x

    recon = np.squeeze(
        np.asarray(
            _reassemble_with_coords_offsets(
                pred,
                {"coords_offsets": coords},
                M=N,
                backend="shift_sum",
                position_crop_border=0,
            )
        )
    )

    K = 8
    center = recon.shape[0] // 2
    recon_center = recon[center - K // 2: center + K // 2, center - K // 2: center + K // 2]
    obj_center = window(cx0, cy0, K)

    def correlation(a, b):
        a = np.abs(a).ravel() - np.abs(a).mean()
        b = np.abs(b).ravel() - np.abs(b).mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    nmae = float(
        np.abs(np.abs(recon_center) - np.abs(obj_center)).mean()
        / (np.abs(obj_center).mean() + 1e-12)
    )
    assert correlation(recon_center, obj_center) > 0.99
    assert nmae < 0.02


def test_normalize_position_inputs_matching_batch_no_swap_pin():
    """Pin: a batch-matching C==1 input keeps today's behavior byte-identical --
    offsets_b112 = transpose(coords,(0,1,3,2)) with NO (y,x)->(x,y) swap, and
    coords_offsets forwarded unchanged."""
    B, H, W = 3, 10, 10
    pred = np.ones((B, 1, H, W), dtype=np.complex64)  # channels-first, C==1
    coords = np.zeros((B, 1, 2, 1), dtype=np.float64)
    for b in range(B):
        coords[b, 0, 0, 0] = 10.0 + b  # row0
        coords[b, 0, 1, 0] = 90.0 + b  # row1

    patches, offsets_b12c, offsets_b112, _crop = _normalize_position_inputs(
        pred, {"coords_offsets": coords}, position_crop_border=0
    )

    assert patches.shape == (B, H, W, 1)
    assert offsets_b112.shape == (B, 1, 1, 2)
    # No swap: offsets_b112[b] == [coords row0, coords row1] (not reversed).
    expected = np.transpose(coords, (0, 1, 3, 2))
    np.testing.assert_array_equal(offsets_b112, expected)
    np.testing.assert_array_equal(offsets_b12c, coords)

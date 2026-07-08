"""Regression coverage for the central_mask object_big training-merge normalization.

Root cause (docs/findings.md TORCH-GS2-CENTRAL-MASK-MERGE-001): at gridsize>1
(``object_big=True``) the default ``training_patch_weighting='central_mask'`` merge
in ``reassemble_patches_position_real`` divided the sum of FULL (unmasked) patches by
a count built from a central-N/2-only prototype mask (+0.001). Outside every patch's
central window the count is ~0, so the ~73% peripheral canvas was divided by 0.001 and
amplified up to ~1000x. Feeding the true object through this merge exploded it and made
a flat-amplitude object the training-loss minimizer -- the cnn@gs2 flat-amplitude gate
failure.

These tests use INDEPENDENT references (an analytic O(1) bound, and a merge-OFF /
per-position diffraction oracle), never the central_mask merge compared against itself.
Both FAIL against the pre-fix (defective) normalization and PASS once the numerator is
masked by the same central mask (periphery zeroed), making numerator and denominator
share support -- mirroring ``reassemble_patches_position_real_probe`` and TF's
self-consistent overlap normalization (ptycho/tf_helper.py reassemble_position:1335-1361,
mk_norm:1092-1101).
"""
import numpy as np
import pytest
import torch

import ptycho_torch.helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig

N = 32
C = 4
GRID = (2, 2)
MAX_NEIGHBOR = 4.0  # -> get_bigN = N + (2-1)*4 = 36 padded canvas


def _configs():
    data_cfg = DataConfig(N=N, C=C, grid_size=GRID, max_neighbor_distance=MAX_NEIGHBOR)
    model_cfg = ModelConfig(object_big=True, C_forward=C, C_model=C, max_position_jitter=0)
    return data_cfg, model_cfg


def _grid_offsets(B):
    """Small, distinct integer 2x2-grid offsets (x, y) -> (B, C, 1, 2)."""
    base = torch.tensor(
        [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]], dtype=torch.float32
    )
    return base.view(1, C, 1, 2).expand(B, C, 1, 2).contiguous()


def test_central_mask_merge_does_not_explode_periphery():
    """Analytic O(1) bound: merging bounded patches (|amp| <= 1, nonzero periphery)
    must yield an O(1) canvas, never the ~1000x peripheral explosion of the
    defective full-numerator / central-count normalization."""
    data_cfg, model_cfg = _configs()
    B = 2
    torch.manual_seed(0)

    # |amp| in [0.5, 1.0] everywhere (nonzero periphery), arbitrary phase.
    amp = 0.5 + 0.5 * torch.rand(B, C, N, N)
    phase = (torch.rand(B, C, N, N) - 0.5) * np.pi
    patches = (amp * torch.exp(1j * phase)).to(torch.complex64)
    assert patches.abs().max() <= 1.0 + 1e-6

    offsets = _grid_offsets(B)
    merged, _mask, _M = hh.reassemble_patches_position_real(
        patches, offsets, data_config=data_cfg, model_config=model_cfg
    )

    merged_max = float(merged.abs().max())
    # Correct self-consistent normalization keeps overlaps averaged: |merged| <= ~1.
    # The defective normalization drives the periphery to ~amp/0.001 ~ O(100-1000).
    assert merged_max < 2.0, (
        f"merged |amp| max={merged_max:.3f} indicates the peripheral explosion "
        f"(central-count denominator applied to an unmasked full-patch numerator)"
    )


def _scale_fair_loss(pred, obs):
    """min_k L1(k*pred, obs).mean() / mean(obs) -- the amp head sets global scale
    freely, so the scale-fair floor is the fair comparison (RCA h4/h7)."""
    p = pred.reshape(-1)
    o = obs.reshape(-1)
    denom = o.abs().mean() + 1e-8
    ks = torch.logspace(-5, 3, 120)
    return min((torch.abs(k * p - o).mean() / denom).item() for k in ks)


def _diffract(obj_bcn, probe_n):
    """Per-channel |FFT(probe * object)| amplitude via the production helper."""
    ill = obj_bcn[:, :, None, :, :] * probe_n.view(1, 1, 1, N, N)
    pred, _ = hh.pad_and_diffract(ill, pad=False)
    return pred


def test_true_object_beats_flat_through_central_mask_forward():
    """Independent-reference oracle (compact RCA probe): the true object must score a
    LOWER scale-fair loss than a flat-amplitude control when both pass through the
    central_mask object_big forward, measured against a merge-OFF (per-position)
    diffraction reference the merge never touches.

    Pre-fix the explosion inflates the true object's loss above the flat control's
    (RCA: 0.872 > 0.196) -> this assertion FAILS. Post-fix the true object is again
    the minimizer -> it PASSES.
    """
    data_cfg, model_cfg = _configs()
    B = 3
    torch.manual_seed(1)

    # Structured true object with strong peripheral contrast (positive amplitude).
    gt_amp = 0.3 + torch.rand(B, C, N, N)
    gt_phase = (torch.rand(B, C, N, N) - 0.5) * np.pi
    gt = (gt_amp * torch.exp(1j * gt_phase)).to(torch.complex64)

    # Degenerate flat-amplitude control: the peripherally-suppressed, low-contrast
    # object the DEFECTIVE loss descends to (RCA: the optimizer drives peripheral
    # amplitude -> 0 to keep the exploded loss finite). Constant |amp| = mean GT amp
    # confined to the central N/2 window (what the pad_object amplitude decoder's
    # central branch produces), GT phase preserved. Its zero periphery means it does
    # NOT explode through the defective merge, so pre-fix it under-scores the true
    # object; post-fix (periphery zeroed for both) the true object's central contrast
    # wins.
    window = torch.zeros(N, N)
    central = slice(N // 4, N // 4 + N // 2)
    window[central, central] = 1.0
    flat_amp = float(gt_amp.mean()) * window
    flat = (flat_amp * torch.exp(1j * gt_phase)).to(torch.complex64)

    # Centrally-concentrated real probe (Gaussian) so the independent reference is
    # limited to the central object -- exactly where a flat control loses contrast.
    yy, xx = torch.meshgrid(
        torch.arange(N) - N / 2, torch.arange(N) - N / 2, indexing="ij"
    )
    probe = torch.exp(-(xx**2 + yy**2) / (2 * (N / 6) ** 2)).to(torch.complex64)

    offsets = _grid_offsets(B)

    def forward(patches):
        merged, _, _ = hh.reassemble_patches_position_real(
            patches, offsets, data_config=data_cfg, model_config=model_cfg
        )
        extracted = hh.extract_channels_from_region(
            merged[:, None, :, :], offsets, data_config=data_cfg, model_config=model_cfg
        )
        return _diffract(extracted, probe)

    # Independent reference: per-position (merge-OFF) diffraction of the true object.
    observed = _diffract(gt, probe)

    loss_gt = _scale_fair_loss(forward(gt), observed)
    loss_flat = _scale_fair_loss(forward(flat), observed)

    assert loss_gt < loss_flat, (
        f"true-object loss {loss_gt:.4f} must be below the flat-amplitude control "
        f"{loss_flat:.4f}; loss_gt >= loss_flat means the merge still makes a flat "
        f"object the training-loss minimizer"
    )

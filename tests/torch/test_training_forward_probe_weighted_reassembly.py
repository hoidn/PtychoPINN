"""Tests for the probe-weighted reassembly helper merged from main (Task 2.2)
plus the Task 2.5 (B3) config-gated dispatch inside ``ForwardModel.forward``.

Covers the additive `reassemble_patches_position_real_probe` function in
`ptycho_torch/helper.py`, exercised through its real signature/return contract
(the one consumed by the dangling call in `beta_modules/model.py`):
`probe=`, `use_probe_weights=` kwargs, returning `(imgs_merged, boolean_mask, M)`.

The Task 2.5 additions:
  * the ``training_patch_weighting`` ModelConfig field and its dispatch at the
    single plug point (`ptycho_torch/model.py` ``ForwardModel.forward``:
    'central_mask' -> unchanged default helper; 'probe'/'uniform' -> the merged
    probe helper with ``use_probe_weights=(mode=='probe')``);
  * three deferred-coverage cases carried over from the Task 2.2 review, whose
    original tests only exercised full-overlap / P=1 / explicit-padded_size:
    (a) a genuine offset/seam two-patch geometry, (b) the ``padded_size=None``
    default branch, (c) a P>1 multi-mode probe exercising the
    ``sum(|probe|^2, dim=2)`` mode-summation line.
"""

import pytest
import torch

from ptycho_torch import helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch.model import ForwardModel


def test_probe_weighting_reduces_corrupted_overlap_seam_error():
    """Two fully-overlapping patches share a solution region; one patch has a
    corrupted stripe. A probe that down-weights the corrupted stripe should
    recover the true value more accurately (lower MAE) than uniform (binary
    center-mask) weighting, which averages the corrupted and correct patches
    equally.
    """
    N = 8
    C = 2
    correct_value = 1 + 0j
    corrupted_value = 5 + 0j
    corrupted_cols = slice(4, 6)  # inside the N//4:N//4+N//2 center mask (2:6)

    patch_correct = torch.full((N, N), correct_value, dtype=torch.complex64)
    patch_corrupted = patch_correct.clone()
    patch_corrupted[:, corrupted_cols] = corrupted_value

    # (B=1, C=2, N, N): channel 0 corrupted, channel 1 correct, same offset -> full overlap
    inputs = torch.stack([patch_corrupted, patch_correct]).unsqueeze(0)
    offsets = torch.zeros((1, C, 1, 2), dtype=torch.float32)

    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    probe_corrupted = torch.ones((1, N, N), dtype=torch.float32)
    probe_corrupted[:, :, corrupted_cols] = 0.05  # low weight where patch 0 is corrupted
    probe_correct = torch.ones((1, N, N), dtype=torch.float32)
    probe = torch.stack([probe_corrupted, probe_correct]).unsqueeze(0)  # (B, C, P, N, N)

    probe_weighted, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe, use_probe_weights=True, padded_size=N,
    )
    uniform, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=None, use_probe_weights=False, padded_size=N,
    )

    center = slice(N // 4, N // 4 + N // 2)
    expected = torch.full((N // 2, N // 2), correct_value, dtype=torch.complex64)

    probe_mae = torch.mean(torch.abs(probe_weighted[0, center, center] - expected))
    uniform_mae = torch.mean(torch.abs(uniform[0, center, center] - expected))

    assert probe_mae < uniform_mae


def test_reassemble_patches_position_real_probe_c1_no_nan_support_preserved():
    """Single-patch (C=1) case: probe-weighted assembly must not produce
    NaN/Inf anywhere (including the zero-weight border outside the patch's
    footprint), and the patch's own footprint must remain part of the support
    (boolean_mask)."""
    N = 8
    M = 16
    pad = (M - N) // 2

    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=1, max_position_jitter=0)

    inputs = torch.ones((1, 1, N, N), dtype=torch.complex64)
    offsets = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    probe = torch.ones((1, 1, 1, N, N), dtype=torch.float32)  # (B, C, P, N, N)

    merged, mask, M_out = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe, use_probe_weights=True, padded_size=M,
    )

    assert M_out == M
    assert torch.isfinite(merged).all()
    assert not torch.isnan(merged).any()

    # Support preserved inside the patch footprint, absent in the untouched border corner.
    assert mask[0, pad:pad + N, pad:pad + N].all()
    assert not mask[0, :pad, :pad].any()

    torch.testing.assert_close(
        merged[0, pad:pad + N, pad:pad + N], inputs[0, 0], atol=1e-5, rtol=1e-5
    )


def test_reassemble_patches_position_real_probe_return_contract_shapes():
    """Return contract: (imgs_merged, boolean_mask, M) with the shapes/dtypes
    expected by the beta_modules/model.py call site (unpacked as
    `reassembled_obj, _, _`)."""
    N = 8
    M = 12
    C = 3
    B = 2

    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    inputs = torch.ones((B, C, N, N), dtype=torch.complex64)
    offsets = torch.zeros((B, C, 1, 2), dtype=torch.float32)
    probe = torch.ones((B, C, 1, N, N), dtype=torch.float32)

    imgs_merged, boolean_mask, M_out = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe, use_probe_weights=True, padded_size=M,
    )

    assert isinstance(M_out, int)
    assert M_out == M
    assert imgs_merged.shape == (B, M, M)
    assert imgs_merged.dtype == torch.complex64
    assert boolean_mask.shape == (B, M, M)
    assert boolean_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Task 2.5 (B3): config-gated dispatch inside ForwardModel.forward
# ---------------------------------------------------------------------------


def test_training_patch_weighting_defaults_to_central_mask():
    """DEFAULTS NEVER CHANGE: the new knob must default to 'central_mask', i.e.
    the pre-existing binary-center-mask reassembly (``reassemble_patches_position_real``)."""
    assert ModelConfig().training_patch_weighting == 'central_mask'


def _forward_case_configs(mode: str):
    data_cfg = DataConfig(N=8, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(
        C_forward=2, C_model=2, object_big=True, max_position_jitter=0,
        intensity_scale_trainable=False, training_patch_weighting=mode,
    )
    return data_cfg, model_cfg


def test_forward_probe_dispatch_differs_from_central_mask_default():
    """``training_patch_weighting='probe'`` must route reassembly through the
    probe helper, producing a *different* predicted diffraction than the
    'central_mask' default when a probe down-weights a corrupted patch stripe.

    (Before Task 2.5, constructing ``ModelConfig(training_patch_weighting=...)``
    raises ``TypeError`` -- the RED state; the field + dispatch make it GREEN.)"""
    N, C = 8, 2
    corrupted_cols = slice(4, 6)

    patch_correct = torch.ones((N, N), dtype=torch.complex64)
    patch_corrupted = patch_correct.clone()
    patch_corrupted[:, corrupted_cols] = 5 + 0j
    # channel 0 corrupted, channel 1 correct, same (zero) offset -> full overlap
    x = torch.stack([patch_corrupted, patch_correct]).unsqueeze(0)  # (1, C, N, N)
    positions = torch.zeros((1, C, 1, 2), dtype=torch.float32)

    probe_corrupted = torch.ones((1, N, N), dtype=torch.complex64)
    probe_corrupted[:, :, corrupted_cols] = 0.05  # down-weight patch 0's corruption
    probe_correct = torch.ones((1, N, N), dtype=torch.complex64)
    probe = torch.stack([probe_corrupted, probe_correct]).unsqueeze(0)  # (1, C, P=1, N, N)

    scale = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    outputs = {}
    for mode in ('central_mask', 'probe'):
        data_cfg, model_cfg = _forward_case_configs(mode)
        model = ForwardModel(model_cfg, data_cfg).eval()
        with torch.no_grad():
            # I_measured=None: only consumed by the rectangular_scaled
            # variable-projection branch, dead for this amplitude path (Task 2.6
            # added I_measured at arg-2 to match main's ForwardModel.forward).
            outputs[mode] = model.forward(x, None, positions, probe, scale)

    assert torch.isfinite(outputs['central_mask']).all()
    assert torch.isfinite(outputs['probe']).all()
    # Dispatch is live: the probe-weighted reassembly changes the object, hence
    # the predicted diffraction, relative to the binary-center-mask default.
    assert not torch.allclose(outputs['central_mask'], outputs['probe'])


def test_forward_default_matches_explicit_central_mask():
    """The default-config forward must be bit-identical to explicitly requesting
    'central_mask' (proves the default routes to the unchanged helper)."""
    N, C = 8, 2
    x = torch.ones((1, C, N, N), dtype=torch.complex64)
    x[0, 0, :, 3:5] = 2 + 0j
    positions = torch.zeros((1, C, 1, 2), dtype=torch.float32)
    probe = torch.ones((1, C, 1, N, N), dtype=torch.complex64)
    scale = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    default_cfg = ModelConfig(C_forward=C, C_model=C, object_big=True, max_position_jitter=0)
    explicit_cfg = ModelConfig(C_forward=C, C_model=C, object_big=True, max_position_jitter=0,
                               training_patch_weighting='central_mask')

    with torch.no_grad():
        # I_measured=None (arg-2, added by Task 2.6 for main-signature parity).
        out_default = ForwardModel(default_cfg, data_cfg).eval().forward(x, None, positions, probe, scale)
        out_explicit = ForwardModel(explicit_cfg, data_cfg).eval().forward(x, None, positions, probe, scale)

    torch.testing.assert_close(out_default, out_explicit, rtol=0, atol=0)


def test_forward_model_seals_training_assembly_before_runtime_mutation(monkeypatch):
    """Training assembly is resolved once from structural model fields and does
    not consult inference policy or later mutable config state."""
    data_cfg, model_cfg = _forward_case_configs("probe")
    model = ForwardModel(model_cfg, data_cfg).eval()
    model_cfg.training_patch_weighting = "central_mask"

    calls = []

    def central(*_args, **_kwargs):
        calls.append("central_mask")
        return torch.ones((1, 8, 8), dtype=torch.complex64), None, 8

    def weighted(*_args, **kwargs):
        calls.append(("weighted", kwargs["use_probe_weights"]))
        return torch.ones((1, 8, 8), dtype=torch.complex64), None, 8

    def extract(canvas, *_args, **_kwargs):
        return canvas

    monkeypatch.setattr(hh, "reassemble_patches_position_real", central)
    monkeypatch.setattr(hh, "reassemble_patches_position_real_probe", weighted)
    monkeypatch.setattr(hh, "extract_channels_from_region", extract)

    x = torch.ones((1, 2, 8, 8), dtype=torch.complex64)
    positions = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
    probe = torch.ones((1, 2, 1, 8, 8), dtype=torch.complex64)

    assembled = model._assemble_training_patches(x, positions, probe)

    assert model.training_assembly_spec.configured_weighting == "probe"
    assert calls == [("weighted", True)]
    assert assembled.shape == (1, 1, 8, 8)


# ---------------------------------------------------------------------------
# Deferred Task 2.2 coverage for the probe helper (offset/seam, padded_size=None,
# P>1 multi-mode)
# ---------------------------------------------------------------------------


def test_probe_helper_offset_seam_two_patch_geometry():
    """Two patches at DIFFERENT non-zero offsets form a real seam: each patch has
    an exclusive region carrying its own value and a shared overlap band carrying
    their weighted average. Exercises the Translation-at-offset path (the original
    2.2 tests only used offset=0)."""
    N, C, M = 8, 2, 16
    a, b = 1.0, 3.0
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    inputs = torch.stack([
        torch.full((N, N), a, dtype=torch.complex64),   # channel 0 -> offset +3
        torch.full((N, N), b, dtype=torch.complex64),   # channel 1 -> offset -3
    ]).unsqueeze(0)
    offsets = torch.zeros((1, C, 1, 2), dtype=torch.float32)
    offsets[0, 0, 0, 0] = 3.0
    offsets[0, 1, 0, 0] = -3.0
    probe = torch.ones((1, C, 1, N, N), dtype=torch.float32)

    merged, mask, M_out = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe, use_probe_weights=True, padded_size=M,
    )

    assert M_out == M
    assert torch.isfinite(merged).all()

    row = merged[0, M // 2].real
    # channel-0 (a, offset +3) translates to -3 under the corrected convention
    # (TORCH-REASSEMBLY-SIGN-001) and lands on the LEFT; channel-1 (b, offset -3)
    # lands on the RIGHT. Overlap seam average is unchanged.
    assert float(row[3]) == pytest.approx(a, abs=1e-4)
    assert float(row[8]) == pytest.approx((a + b) / 2, abs=1e-4)
    assert float(row[12]) == pytest.approx(b, abs=1e-4)
    # A genuine seam exists: an overlap band strictly between the two patch values.
    overlap = (merged[0].real > a + 1e-3) & (merged[0].real < b - 1e-3) & mask[0]
    assert overlap.any()


def test_probe_helper_padded_size_none_uses_get_padded_size():
    """``padded_size=None`` default branch must fall back to
    ``get_padded_size(data_config, model_config)`` (2.2 tests always passed an
    explicit padded_size)."""
    N = 8
    data_cfg = DataConfig(N=N, C=1, grid_size=(2, 2), max_neighbor_distance=3.0)
    model_cfg = ModelConfig(C_forward=1, max_position_jitter=0)

    expected_M = hh.get_padded_size(data_cfg, model_cfg)
    assert expected_M > N  # this config genuinely pads, so the branch is meaningful

    inputs = torch.ones((1, 1, N, N), dtype=torch.complex64)
    offsets = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    probe = torch.ones((1, 1, 1, N, N), dtype=torch.float32)

    merged, mask, M_out = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe, use_probe_weights=True, padded_size=None,
    )

    assert M_out == expected_M
    assert merged.shape == (1, expected_M, expected_M)


def test_probe_helper_multimode_probe_sums_over_modes():
    """A P>1 multi-mode probe must weight by ``sum_p |P_p|^2``; the merged result
    must equal that of a single-mode probe whose intensity equals the mode sum.
    Exercises the ``torch.sum(torch.abs(probe)**2, dim=2)`` line (2.2 tests were
    P=1 only)."""
    N, C = 8, 2
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    # Two fully-overlapping patches with distinct values.
    inputs = torch.stack([
        torch.full((N, N), 1 + 0j, dtype=torch.complex64),
        torch.full((N, N), 5 + 0j, dtype=torch.complex64),
    ]).unsqueeze(0)
    offsets = torch.zeros((1, C, 1, 2), dtype=torch.float32)

    # Multi-mode (P=2): channel 0 modes {sqrt(3), 1} -> |.|^2 sum = 4; channel 1
    # modes {1, 0} -> |.|^2 sum = 1.
    probe_multi = torch.zeros((1, C, 2, N, N), dtype=torch.float32)
    probe_multi[0, 0, 0] = 3.0 ** 0.5
    probe_multi[0, 0, 1] = 1.0
    probe_multi[0, 1, 0] = 1.0
    probe_multi[0, 1, 1] = 0.0
    assert probe_multi.shape[2] == 2  # genuinely P>1

    # Equivalent single-mode probe: |2|^2 = 4 (channel 0), |1|^2 = 1 (channel 1).
    probe_single = torch.zeros((1, C, 1, N, N), dtype=torch.float32)
    probe_single[0, 0, 0] = 2.0
    probe_single[0, 1, 0] = 1.0

    merged_multi, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_multi, use_probe_weights=True, padded_size=N,
    )
    merged_single, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_single, use_probe_weights=True, padded_size=N,
    )

    torch.testing.assert_close(merged_multi, merged_single, rtol=1e-6, atol=1e-6)
    # And the mode weighting matters: it is not the same as uniform (1:1) blending.
    center = slice(N // 4, N // 4 + N // 2)
    expected_weighted = (1 * 4 + 5 * 1) / (4 + 1)  # = 1.8
    assert merged_multi[0, center, center].real.mean().item() == pytest.approx(expected_weighted, abs=1e-4)


# ---------------------------------------------------------------------------
# Task C1: plain / batch-broadcast probe layouts from the grid_lines
# dict-container pipeline
#
# The native mmap dataloader supplies a modes-layout probe (B, C, P, H, W)
# (ndim 5); the grid_lines dict-container loader supplies a plain probe --
# measured shape (16, 128, 128) i.e. (B, H, W) batch-broadcast (ndim 3),
# derived from a bare (H, W) probeGuess (ndim 2). Before C1 the weight
# derivation ``torch.sum(|probe|^2, dim=2).flatten(0, 1)`` assumed the modes
# axis at dim 2 and collapsed a spatial axis, yielding a 1-D tensor that
# crashed F.pad at helper.py:271 ("padding length 4 and input of dimension 1").
# ---------------------------------------------------------------------------


def test_probe_helper_batch_broadcast_layout_weights_equal_probe_intensity():
    """(a) Batch-broadcast probe ``(B, H, W)`` -- the grid_lines dict-container
    layout -- with ``use_probe_weights=True`` must not crash and must apply
    weights equal to ``|P|^2``. Equivalence oracle: a modes-layout call
    ``(B, C, P=1, H, W)`` carrying the same probe per channel must produce a
    byte-identical merge."""
    N, C, B = 8, 4, 2
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    # Distinct per-(b,c) patch values so channel/batch ordering matters.
    inputs = torch.arange(1, B * C + 1, dtype=torch.float32).reshape(B, C, 1, 1)
    inputs = inputs.expand(B, C, N, N).to(torch.complex64).contiguous()
    offsets = torch.zeros((B, C, 1, 2), dtype=torch.float32)

    # Batch-broadcast probe: a distinct probe per batch element, shared across C.
    probe_bhw = torch.ones((B, N, N), dtype=torch.complex64)
    probe_bhw[0, :, 4:6] = 0.5
    probe_bhw[1, 2:4, :] = 2.0

    merged_plain, mask_plain, M_plain = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_bhw, use_probe_weights=True, padded_size=N,
    )

    # Modes-layout equivalence oracle: same probe per channel, single mode.
    probe_modes = probe_bhw.unsqueeze(1).unsqueeze(2).expand(B, C, 1, N, N).contiguous()
    merged_modes, mask_modes, M_modes = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_modes, use_probe_weights=True, padded_size=N,
    )

    assert M_plain == N
    assert merged_plain.shape == (B, N, N)
    assert torch.isfinite(merged_plain).all()
    torch.testing.assert_close(merged_plain, merged_modes, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(mask_plain, mask_modes)


def test_probe_helper_bare_hw_layout_weights_equal_probe_intensity():
    """(a') Bare ``(H, W)`` probe (raw probeGuess, ndim 2) broadcasts to every
    flattened patch and equals the equivalent modes-layout merge."""
    N, C, B = 8, 2, 2
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    inputs = torch.arange(1, B * C + 1, dtype=torch.float32).reshape(B, C, 1, 1)
    inputs = inputs.expand(B, C, N, N).to(torch.complex64).contiguous()
    offsets = torch.zeros((B, C, 1, 2), dtype=torch.float32)

    probe_hw = torch.ones((N, N), dtype=torch.complex64)
    probe_hw[:, 4:6] = 0.25

    merged_plain, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_hw, use_probe_weights=True, padded_size=N,
    )

    probe_modes = probe_hw.reshape(1, 1, 1, N, N).expand(B, C, 1, N, N).contiguous()
    merged_modes, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_modes, use_probe_weights=True, padded_size=N,
    )

    torch.testing.assert_close(merged_plain, merged_modes, rtol=1e-6, atol=1e-6)


def test_object_big_shared_multimode_probe_expands_across_batch_and_channels():
    """A documented shared ``(1,1,P,H,W)`` probe must weight every object_big
    patch identically to explicit ``(B,C,P,H,W)`` expansion."""
    N, B, C, P = 8, 2, 2, 2
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(
        object_big=True,
        C_forward=C,
        training_patch_weighting="probe",
        max_position_jitter=0,
    )
    inputs = torch.arange(1, B * C + 1, dtype=torch.float32).reshape(B, C, 1, 1)
    inputs = inputs.expand(B, C, N, N).to(torch.complex64).contiguous()
    offsets = torch.zeros((B, C, 1, 2), dtype=torch.float32)

    shared_probe = torch.ones((1, 1, P, N, N), dtype=torch.complex64)
    shared_probe[:, :, 0, :, N // 2 :] = 0.5
    shared_probe[:, :, 1, N // 2 :, :] = 2.0
    expanded_probe = shared_probe.expand(B, C, P, N, N)

    merged_shared, mask_shared, size_shared = (
        hh.reassemble_patches_position_real_probe(
            inputs,
            offsets,
            data_cfg,
            model_cfg,
            probe=shared_probe,
            use_probe_weights=True,
            padded_size=N,
        )
    )
    merged_expanded, mask_expanded, size_expanded = (
        hh.reassemble_patches_position_real_probe(
            inputs,
            offsets,
            data_cfg,
            model_cfg,
            probe=expanded_probe,
            use_probe_weights=True,
            padded_size=N,
        )
    )

    assert size_shared == size_expanded == N
    torch.testing.assert_close(merged_shared, merged_expanded, rtol=0, atol=0)
    torch.testing.assert_close(mask_shared, mask_expanded, rtol=0, atol=0)


def test_probe_helper_batch_broadcast_hand_built_weighted_average():
    """(a'') Independent hand-built oracle: two fully-overlapping patches per
    batch, probe intensities w0 and w1, merged center == (v0*w0 + v1*w1)/(w0+w1)."""
    N, C = 8, 2
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    v0, v1 = 1.0, 5.0
    inputs = torch.stack([
        torch.full((N, N), v0, dtype=torch.complex64),
        torch.full((N, N), v1, dtype=torch.complex64),
    ]).unsqueeze(0)  # (1, C, N, N)
    offsets = torch.zeros((1, C, 1, 2), dtype=torch.float32)

    # Batch-broadcast probe shared across both channels -> both weighted equally.
    # To differentiate channels we cannot via a (B,H,W) probe (shared over C),
    # so this oracle verifies the shared-probe average is exactly the mean.
    probe_bhw = torch.full((1, N, N), 2.0, dtype=torch.complex64)

    merged, _, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_bhw, use_probe_weights=True, padded_size=N,
    )
    center = slice(N // 4, N // 4 + N // 2)
    # Equal weights (both channels share the same probe) -> simple mean.
    assert merged[0, center, center].real.mean().item() == pytest.approx((v0 + v1) / 2, abs=1e-4)


def test_probe_helper_uniform_weighting_byte_identical_with_plain_probe():
    """(c) ``use_probe_weights=False`` ignores the probe entirely, so passing a
    plain-layout probe (or a modes probe, or None) yields byte-identical output.
    Pins the uniform path against pre-captured values so a regression is caught."""
    N, C, B = 8, 4, 2
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    inputs = torch.arange(1, B * C + 1, dtype=torch.float32).reshape(B, C, 1, 1)
    inputs = inputs.expand(B, C, N, N).to(torch.complex64).contiguous()
    offsets = torch.zeros((B, C, 1, 2), dtype=torch.float32)
    probe_bhw = torch.ones((B, N, N), dtype=torch.complex64)

    merged_none, mask_none, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=None, use_probe_weights=False, padded_size=N,
    )
    merged_plain, mask_plain, _ = hh.reassemble_patches_position_real_probe(
        inputs, offsets, data_cfg, model_cfg,
        probe=probe_bhw, use_probe_weights=False, padded_size=N,
    )

    # Probe layout is irrelevant when uniform weighting is selected.
    torch.testing.assert_close(merged_plain, merged_none, rtol=0, atol=0)
    torch.testing.assert_close(mask_plain, mask_none)

    # Pin the actual value: uniform (binary center-mask) full-overlap average of
    # channel values 1..4 (batch 0) and 5..8 (batch 1) is their per-batch mean.
    center = slice(N // 4, N // 4 + N // 2)
    assert merged_none[0, center, center].real.mean().item() == pytest.approx((1 + 2 + 3 + 4) / 4, abs=1e-4)
    assert merged_none[1, center, center].real.mean().item() == pytest.approx((5 + 6 + 7 + 8) / 4, abs=1e-4)


def test_probe_helper_rejects_unsupported_probe_ndim():
    """(d) Fail-fast: a probe with an unsupported rank (1-D here) must raise
    ValueError naming the received shape and the accepted layout families,
    not silently mis-weight or crash inside F.pad."""
    N, C = 8, 1
    data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), max_neighbor_distance=0.0)
    model_cfg = ModelConfig(C_forward=C, max_position_jitter=0)

    inputs = torch.ones((1, C, N, N), dtype=torch.complex64)
    offsets = torch.zeros((1, C, 1, 2), dtype=torch.float32)
    bad_probe = torch.ones((N,), dtype=torch.complex64)  # 1-D nonsense

    with pytest.raises(ValueError, match=r"probe"):
        hh.reassemble_patches_position_real_probe(
            inputs, offsets, data_cfg, model_cfg,
            probe=bad_probe, use_probe_weights=True, padded_size=N,
        )

"""CPU physics safety net for gridsize>1 (object_big) generator support.

These tests pin the EXISTING machinery that Tasks 4/5 of the generator gridsize>1
plan must satisfy. They exercise only current product code; both are expected to
PASS unchanged. See docs/plans/2026-07-06-generator-gridsize2-support.md Task 2.

Two contracts are pinned:

1. ``test_merge_extract_roundtrip_pins_group_geometry`` -- the pure-geometry
   round-trip of the ``object_big`` physics branch
   (``ptycho_torch.model.ForwardModel.forward``, model.py:1584-1604): merging the
   C=gridsize^2 per-group patches into one shared bigN object via
   ``reassemble_patches_position_real`` and re-extracting per-position patches via
   ``extract_channels_from_region``. It pins the group offsets, the coordinate
   sign convention (reassemble uses ``-offsets`` to match TF
   ``_reassemble_patches_position_real``; extract uses ``+offsets``; see
   docs/findings.md TORCH-REASSEMBLY-SIGN-001), and the translation convention:
   consistent crops of a known object survive the round-trip in the deep interior,
   while crops built with the opposite sign do not (so a degenerate/symmetric pass
   cannot mask a sign regression).

2. ``test_oracle_generator_realimag_reproduces_reference_diffraction`` -- the
   ``object_big`` branch is generator-agnostic (Task 0 report, item 1): a stub
   generator emitting ground-truth patches in the generator ``real_imag`` output
   format ``(B, H, W, C, 2)`` (consumed by ``_predict_complex_patches``,
   model.py:105-116) reproduces the noiseless forward-model diffraction of the same
   ground-truth object to float precision, and a deliberately shifted oracle
   produces a strictly larger amplitude loss. This proves the physics branch is
   correct *for generator-format outputs specifically*, injected purely via the
   Lightning constructor's ``generator_module`` slot (no registry/model.py changes).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from ptycho_torch.config_params import (
    DataConfig, ModelConfig, TrainingConfig, InferenceConfig,
)
from ptycho_torch.helper import (
    reassemble_patches_position_real,
    extract_channels_from_region,
    get_padded_size,
)
from ptycho_torch.model import PtychoPINN_Lightning

# Small synthetic geometry: gridsize 2 (C = gridsize^2 = 4), N=32 patches.
# max_neighbor_distance=6.0 -> bigN = N + 6 = 38, so the padded canvas leaves a
# +/-3 px shift margin while the group offsets stay at +/-2 px (S below).
N = 32
GRID = (2, 2)
C = 4
MND = 6.0
S = 2  # per-axis magnitude of the (centered) group offsets, in pixels

# The central-mask reassembly normalization only reproduces the source object where
# every patch covering a pixel also covers it with its central mask. That "faithful"
# window is eroded by the shift magnitude S on each side, giving the deep interior
# [N//4 + 2S : 3N//4 - 2S]. Assertions are restricted to that window.
BORDER = N // 4 + 2 * S


def _group_offsets():
    """2x2-grid group offsets, centered so the group sits at the canvas center."""
    base = torch.tensor(
        [[-S, -S], [-S, S], [S, -S], [S, S]], dtype=torch.float32
    )  # (C, 2) as (x, y)
    offsets = base.view(1, C, 1, 2)  # (B=1, C, 1, 2)
    return base, offsets


def _smooth_object(M, seed=0):
    """A smooth, non-symmetric complex object filling the M x M canvas.

    Non-symmetry (distinct x/y structure) is required so a coordinate-sign
    regression cannot accidentally round-trip.
    """
    ys, xs = np.mgrid[0:M, 0:M]
    amp = 0.7 + 0.2 * np.sin(2 * np.pi * xs / M) * np.cos(2 * np.pi * ys / M)
    phase = 0.3 * np.sin(2 * np.pi * (xs + ys) / M)
    return torch.from_numpy((amp * np.exp(1j * phase)).astype(np.complex64))


def _consistent_crops(obj_t, base, M, sign=+1):
    """Crop C patches from the canvas consistent with reassemble's -offset placement.

    The object_big reassemble path pads each patch to the canvas center then shifts
    it by ``-offset`` (x->width, y->height), matching TF
    ``_reassemble_patches_position_real``; the matching crop of the shared object is
    the N x N window centered at (M//2 - oy, M//2 - ox). ``sign=-1`` builds the
    opposite (OLD ``+offset``) convention -- a negative control that must NOT
    round-trip.
    """
    c0 = M // 2
    patches = torch.zeros((1, C, N, N), dtype=torch.complex64)
    for c in range(C):
        ox = -sign * int(base[c, 0].item())
        oy = -sign * int(base[c, 1].item())
        r0 = c0 + oy - N // 2
        col0 = c0 + ox - N // 2
        patches[0, c] = obj_t[r0:r0 + N, col0:col0 + N]
    return patches


def test_merge_extract_roundtrip_pins_group_geometry():
    """Pin offsets / coordinate signs / translation of the object_big round-trip.

    Pins docs/plans/2026-07-06-generator-gridsize2-support.md Task 2 Step 1.
    """
    data_cfg = DataConfig(N=N, C=C, grid_size=GRID, max_neighbor_distance=MND)
    model_cfg = ModelConfig(object_big=True, C_model=C, C_forward=C)
    M = get_padded_size(data_cfg, model_cfg)

    obj_t = _smooth_object(M)
    base, offsets = _group_offsets()

    # Consistent crops survive reassemble -> extract in the deep interior.
    patches = _consistent_crops(obj_t, base, M, sign=+1)
    reassembled, _, _ = reassemble_patches_position_real(
        patches, offsets, data_cfg, model_cfg
    )
    extracted = extract_channels_from_region(
        reassembled[:, None, :, :], offsets, data_cfg, model_cfg
    )

    interior = slice(BORDER, N - BORDER)
    roundtrip_err = (
        extracted[:, :, interior, interior] - patches[:, :, interior, interior]
    ).abs()
    assert roundtrip_err.max().item() < 1e-3, (
        "object_big merge/extract failed to round-trip consistent group crops in "
        f"the deep interior (max err {roundtrip_err.max().item():.3e})"
    )

    # Negative control: crops built with the OLD (+offset-merge) sign convention are
    # mutually inconsistent under the corrected reassembly and must NOT round-trip --
    # this is what makes the test a genuine pin on the coordinate-sign convention.
    wrong = _consistent_crops(obj_t, base, M, sign=-1)
    reassembled_w, _, _ = reassemble_patches_position_real(
        wrong, offsets, data_cfg, model_cfg
    )
    extracted_w = extract_channels_from_region(
        reassembled_w[:, None, :, :], offsets, data_cfg, model_cfg
    )
    wrong_err = (
        extracted_w[:, :, interior, interior] - wrong[:, :, interior, interior]
    ).abs()
    assert wrong_err.max().item() > 1e-2, (
        "opposite-sign crops unexpectedly round-tripped; the geometry test is not "
        "actually pinning the coordinate-sign convention"
    )


def test_object_big_reassemble_matches_tf_offset_sign():
    """The object_big reassemble path must place content on the same side as TF.

    Single centered bright pixel, C=1 group, offset (dx=+5, dy=0), asserted directly
    against ``ptycho.tf_helper._reassemble_patches_position_real`` (which generated the
    grid-lines data conventions). A 2-sample batch with symmetric +/-offsets neutralizes
    TF's batch-mean recentering. Pins the corrected convention (TORCH-REASSEMBLY-SIGN-001):
    offset (dx=+5) displaces content to column ``center-5`` (matching TF), NOT ``center+5``.
    """
    tf = pytest.importorskip("tensorflow")
    from ptycho import tf_helper as hh_tf
    from ptycho import params as p

    n = 16
    padded = 32
    dx = 5.0
    p.cfg['N'] = n
    p.cfg['gridsize'] = 1

    patch = np.zeros((2, n, n), dtype=np.complex64)
    patch[0, n // 2, n // 2] = 1.0
    patch[1, n // 2, n // 2] = 1.0
    offsets_np = np.array([[[[dx, 0.0]]], [[[-dx, 0.0]]]], dtype=np.float32)  # (B=2,C=1,1,2)

    data_cfg = DataConfig(N=n, grid_size=(1, 1), C=1)
    model_cfg = ModelConfig(C_forward=1, C_model=1, object_big=True)
    torch_out, _, _ = reassemble_patches_position_real(
        torch.from_numpy(patch).reshape(2, 1, n, n),
        torch.from_numpy(offsets_np),
        data_cfg, model_cfg, agg=True, padded_size=padded,
    )
    torch_np = torch_out.detach().cpu().numpy()

    imgs_tf = tf.constant(patch.reshape(2, n, n, 1))
    offsets_tf = tf.constant(offsets_np.transpose(0, 1, 3, 2))  # (B,1,2,C=1)
    tf_out = hh_tf._reassemble_patches_position_real(
        imgs_tf, offsets_tf, agg=True, padded_size=padded
    ).numpy()[..., 0]

    center = padded // 2
    # Sample 0 (offset dx=+5) must land at column center-5 for both torch and TF.
    def bright_col(img2d):
        _, col = np.unravel_index(np.argmax(np.abs(img2d)), img2d.shape)
        return int(col - center)

    assert bright_col(tf_out[0]) == -int(dx), "TF reference did not place offset at center-dx"
    assert bright_col(torch_np[0]) == bright_col(tf_out[0]), (
        "torch object_big reassemble offset sign does not match TF "
        f"(_reassemble): torch col {bright_col(torch_np[0])} vs TF {bright_col(tf_out[0])}"
    )

    # Interior spatial-pattern parity: torch normalizes by the central-mask count
    # (``+0.001`` epsilon) while TF ``_reassemble`` only sums, so compare each output
    # normalized by its own peak (isolates placement, not the sum-vs-normalize scale).
    interior = slice(center - N // 4, center + N // 4)
    torch_amp = np.abs(torch_np[0])[interior, interior]
    tf_amp = np.abs(tf_out[0])[interior, interior]
    torch_norm = torch_amp / (torch_amp.max() + 1e-12)
    tf_norm = tf_amp / (tf_amp.max() + 1e-12)
    assert np.allclose(torch_norm, tf_norm, atol=1e-3), (
        "torch/TF object_big reassemble interior placement diverges "
        f"(max abs diff {np.abs(torch_norm - tf_norm).max():.3e})"
    )


def test_bridge_and_reassemble_net_convention_matches_tf():
    """Net-convention guard: the ingestion bridge and physics-layer sign flips must
    compose to TF's single convention end to end (four-surface reconciliation,
    TORCH-REASSEMBLY-SIGN-001).

    ``PtychoDataContainerTorch`` no longer negates ``coords_relative`` (verbatim
    pass-through, matching TF ``ptycho/loader.py`` per ptychodus_api_spec.md:172), and
    ``reassemble_patches_position_real`` now negates internally to match TF
    ``_reassemble_patches_position_real``. This threads synthetic grouped data with
    nonzero relative coords through the container and the object_big merge exactly as
    the training dataloader does (permute (B,1,2,C) -> (B,C,1,2), matching
    ``ptycho_torch/workflows/components.py``'s coords_relative reshape), and asserts the
    net placement matches TF ``_reassemble_patches_position_real`` fed the SAME raw
    ``coords_relative`` directly -- i.e. the bridge pass-through and the helper sign flip
    compose to the single TF-matching convention, not a double negation.
    """
    tf = pytest.importorskip("tensorflow")
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
    from ptycho import tf_helper as hh_tf
    from ptycho import params as p

    n = 16
    padded = 32
    dx = 5.0
    p.cfg['N'] = n
    p.cfg['gridsize'] = 1

    # Nonzero, asymmetric relative coords (C=1 group per sample); two samples with
    # symmetric +/-offsets neutralize TF's batch-mean recentering, matching the
    # single-pixel probe methodology used elsewhere in this file. Raw shape (B,1,2,C=1)
    # is the TF/grouped-data-contract shape PtychoDataContainerTorch's coords_relative
    # arrives in.
    coords_relative = np.array(
        [[[[dx], [0.0]]], [[[-dx], [0.0]]]], dtype=np.float32
    )  # (B=2, 1, 2, C=1)

    patch = np.zeros((2, n, n), dtype=np.complex64)
    patch[0, n // 2, n // 2] = 1.0
    patch[1, n // 2, n // 2] = 1.0

    grouped_data = {
        "X_full": np.zeros((2, n, n, 1), dtype=np.float32),
        "Y": None,
        "coords_relative": coords_relative,
        "coords_offsets": np.zeros((2, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.zeros((2, 1), dtype=np.int32),
    }
    probe = np.zeros((n, n), dtype=np.complex64)
    container = PtychoDataContainerTorch(grouped_data, probe)

    # Production reshape (ptycho_torch/workflows/components.py): (B,1,2,C) -> (B,C,1,2).
    positions = container.local_offsets.to(torch.float32).permute(0, 3, 1, 2).contiguous()

    data_cfg = DataConfig(N=n, grid_size=(1, 1), C=1)
    model_cfg = ModelConfig(C_forward=1, C_model=1, object_big=True)
    torch_out, _, _ = reassemble_patches_position_real(
        torch.from_numpy(patch).reshape(2, 1, n, n),
        positions, data_cfg, model_cfg, agg=True, padded_size=padded,
    )
    torch_np = torch_out.detach().cpu().numpy()

    # TF reference fed the SAME raw coords_relative directly (no bridge involved) --
    # this is the "TF net convention" the four-surface fix must converge on.
    imgs_tf = tf.constant(patch.reshape(2, n, n, 1))
    offsets_tf = tf.constant(coords_relative)  # already (B,1,2,C=1), TF's native shape
    tf_out = hh_tf._reassemble_patches_position_real(
        imgs_tf, offsets_tf, agg=True, padded_size=padded
    ).numpy()[..., 0]

    center = padded // 2
    def bright_col(img2d):
        _, col = np.unravel_index(np.argmax(np.abs(img2d)), img2d.shape)
        return int(col - center)

    assert bright_col(torch_np[0]) == bright_col(tf_out[0]), (
        "bridge+helper net placement does not match TF net convention: "
        f"torch col {bright_col(torch_np[0])} vs TF {bright_col(tf_out[0])}"
    )


class _RealImagStub(nn.Module):
    """Generator stub emitting fixed patches in the real_imag format (B, H, W, C, 2)."""

    def __init__(self, patches):  # complex (B, C, N, N)
        super().__init__()
        real_imag = torch.stack([patches.real, patches.imag], dim=-1)  # (B, C, N, N, 2)
        real_imag = real_imag.permute(0, 2, 3, 1, 4).contiguous()      # (B, N, N, C, 2)
        self.register_buffer("out", real_imag.to(torch.float32))

    def forward(self, x):
        return self.out


class _AmpPhaseStub(nn.Module):
    """Reference stub emitting the same patches via the amp_phase format (amp, phase)."""

    def __init__(self, patches):  # complex (B, C, N, N)
        super().__init__()
        self.register_buffer("amp", patches.abs().to(torch.float32))
        self.register_buffer("phase", patches.angle().to(torch.float32))

    def forward(self, x):
        return self.amp, self.phase


def _build_lightning(stub, data_cfg):
    """Inject a stub generator via the Lightning constructor (no registry changes)."""
    output = "amp_phase" if isinstance(stub, _AmpPhaseStub) else "real_imag"
    model_cfg = ModelConfig(
        object_big=True, C_model=C, C_forward=C, architecture="cnn",
        physics_forward_mode="amplitude", intensity_scale_trainable=False,
    )
    model = PtychoPINN_Lightning(
        model_cfg, data_cfg, TrainingConfig(), InferenceConfig(),
        generator_module=stub, generator_output=output,
    )
    model.eval()
    return model


def _forward_diffraction(model, offsets, probe):
    """Noiseless forward-model diffraction amplitude for the injected patches."""
    x_in = torch.zeros((1, C, N, N), dtype=torch.float32)  # ignored by the stubs
    ones = torch.ones((1, 1, 1, 1))
    with torch.no_grad():
        pred, _, _ = model(
            x_in, offsets, probe,
            input_scale_factor=ones, output_scale_factor=ones, experiment_ids=None,
        )
    return pred


def test_oracle_generator_realimag_reproduces_reference_diffraction():
    """A real_imag oracle reproduces the reference diffraction; a shift raises the loss.

    Pins docs/plans/2026-07-06-generator-gridsize2-support.md Task 2 Step 2. The
    reference "simulated noiseless data" is the object_big forward-model diffraction
    of the ground-truth patches supplied via the amp_phase format; the real_imag
    oracle must reproduce it to float precision, isolating the (B, H, W, C, 2)
    generator-format contract (the edge treatment of reassemble/extract is identical
    on both paths and cancels).
    """
    data_cfg = DataConfig(N=N, C=C, grid_size=GRID, max_neighbor_distance=MND)
    model_cfg = ModelConfig(object_big=True, C_model=C, C_forward=C)
    M = get_padded_size(data_cfg, model_cfg)

    obj_t = _smooth_object(M)
    base, offsets = _group_offsets()
    gt_patches = _consistent_crops(obj_t, base, M, sign=+1)
    shifted_patches = torch.roll(gt_patches, shifts=(S, S), dims=(2, 3))

    # Smooth Gaussian probe (identical on all paths; cancels in the comparison).
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    gauss = torch.exp(-(((xx - N / 2) ** 2 + (yy - N / 2) ** 2) / (2 * (N / 5) ** 2)))
    probe = (
        gauss.to(torch.complex64).view(1, 1, 1, N, N)
        .expand(1, C, 1, N, N).contiguous() * data_cfg.probe_scale
    )

    observed = _forward_diffraction(
        _build_lightning(_AmpPhaseStub(gt_patches), data_cfg), offsets, probe
    )
    pred_true = _forward_diffraction(
        _build_lightning(_RealImagStub(gt_patches), data_cfg), offsets, probe
    )
    pred_shifted = _forward_diffraction(
        _build_lightning(_RealImagStub(shifted_patches), data_cfg), offsets, probe
    )

    loss_true = (pred_true - observed).abs().mean().item()
    loss_shifted = (pred_shifted - observed).abs().mean().item()

    assert loss_true < 1e-3, (
        "real_imag oracle did not reproduce the reference diffraction of the "
        f"ground-truth object (amplitude loss {loss_true:.3e})"
    )
    # Absolute sanity floor for the shift-induced mismatch. Recalibrated from 1e-2
    # to 1e-3 after docs/findings.md TORCH-GS2-CENTRAL-MASK-MERGE-001: the old floor
    # was pinned to the DEFECTIVE ~1000x peripheral explosion, which inflated any
    # shift-induced difference. With the corrected (de-exploded) merge the same S=2
    # shift yields ~4e-3; the order-of-magnitude discrimination is enforced by the
    # relative assertion below (ratio ~4e5), which is the actual contract.
    assert loss_shifted > 1e-3, (
        f"shifted oracle produced an implausibly small loss ({loss_shifted:.3e})"
    )
    assert loss_shifted > 100 * loss_true, (
        "shifted oracle loss was not strictly larger than the aligned oracle loss "
        f"({loss_shifted:.3e} vs {loss_true:.3e})"
    )

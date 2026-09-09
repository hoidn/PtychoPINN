"""Overlap phase alignment: gather is the splat's adjoint, and per-patch phase constants are recovered from overlaps alone."""
import numpy as np
import pytest
import torch

from ptycho_torch.reassembly_accumulators import VectorizedWeightedAccumulator
from ptycho_torch.reassembly_phase_alignment import (
    PhaseAlignmentResult,
    _rotated,
    _wrap,
    align_patch_phases,
    gather_patches,
    spectral_constants,
)

CANVAS = (96, 96)
PATCH = 24


def _positions(n, generator):
    low, high = PATCH / 2 + 2, CANVAS[0] - PATCH / 2 - 3
    return low + (high - low) * torch.rand(n, 2, generator=generator)


def _smooth_field(generator):
    yy, xx = torch.meshgrid(
        torch.arange(CANVAS[0], dtype=torch.float32),
        torch.arange(CANVAS[1], dtype=torch.float32),
        indexing="ij",
    )
    phase = 1.2 * torch.sin(xx / 11.0) + 0.9 * torch.cos(yy / 9.0 + 0.4) + 0.02 * xx
    amplitude = 0.7 + 0.3 * torch.cos(xx / 13.0) * torch.sin(yy / 17.0)
    noise = 0.05 * torch.randn(CANVAS, generator=generator)
    return torch.polar(amplitude + noise, phase).to(torch.complex64)


def _probe_weight():
    grid = torch.arange(PATCH, dtype=torch.float32) - PATCH / 2 + 0.5
    radius = torch.hypot(grid.view(-1, 1), grid.view(1, -1))
    return torch.exp(-(radius / 7.0) ** 2)


def _smooth_truth_and_grid(canvas=160, patch=PATCH, n=200):
    yy, xx = torch.meshgrid(
        torch.arange(canvas, dtype=torch.float32),
        torch.arange(canvas, dtype=torch.float32),
        indexing="ij",
    )
    truth = torch.polar(
        0.8 + 0.15 * torch.cos(xx / 13.0) * torch.sin(yy / 17.0),
        0.7 * torch.sin(xx / 15.0) + 0.5 * torch.cos(yy / 11.0),
    ).to(torch.complex64)
    axis = torch.linspace(patch // 2 + 2, canvas - patch // 2 - 3, 15).round()
    positions = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), -1).reshape(-1, 2)
    return truth, positions[:n]

def test_gather_is_adjoint_of_splat():
    generator = torch.Generator().manual_seed(0)
    positions = _positions(12, generator)
    patches = torch.randn(12, PATCH, PATCH, dtype=torch.complex64, generator=generator)
    canvas_field = torch.randn(CANVAS, dtype=torch.complex64, generator=generator)
    splat = torch.zeros(CANVAS, dtype=torch.complex64)
    weights = torch.zeros(CANVAS)
    VectorizedWeightedAccumulator(CANVAS, torch.device("cpu")).accumulate_batch(
        splat, weights, patches, positions, torch.ones(PATCH, PATCH),
        patch_size=PATCH, uniform_weighting=True,
    )
    lhs = torch.vdot(canvas_field.reshape(-1), splat.reshape(-1))
    rhs = torch.vdot(
        gather_patches(canvas_field, positions, PATCH).reshape(-1), patches.reshape(-1)
    )
    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-3)


def test_alignment_recovers_per_patch_phase_constants():
    generator = torch.Generator().manual_seed(1)
    truth = _smooth_field(generator)
    positions = _positions(160, generator)
    injected = 2.5 * (torch.rand(160, generator=generator) - 0.5)
    patches = gather_patches(truth, positions, PATCH) * torch.polar(
        torch.ones(160), injected
    ).view(-1, 1, 1)
    probe = _probe_weight()
    result = align_patch_phases(
        patches, positions, probe, CANVAS, patch_size=PATCH, uniform_weighting=False
    )
    assert result.converged
    recovered = result.theta + injected
    recovered = recovered - torch.angle(torch.polar(torch.ones(160), recovered).mean())
    # sub-pixel placement leaves a bilinear resampling floor of ~0.02 rad; integer placement recovers to 1e-3
    assert float(torch.atan2(torch.sin(recovered), torch.cos(recovered)).abs().max()) < 5e-2

    plain = torch.zeros(CANVAS, dtype=torch.complex64)
    plain_weights = torch.zeros(CANVAS)
    VectorizedWeightedAccumulator(CANVAS, torch.device("cpu")).accumulate_batch(
        plain, plain_weights, patches, positions, probe,
        patch_size=PATCH, uniform_weighting=False,
    )
    covered = plain_weights > 0.5 * plain_weights.max()
    truth_np = truth.numpy()[covered.numpy()]

    def phase_mae(canvas, weights):
        field = (canvas / (weights + 1e-12)).numpy()[covered.numpy()]
        factor = np.vdot(field, truth_np)
        factor /= abs(factor)
        return float(np.abs(np.angle(field * factor * np.conj(truth_np))).mean())

    assert phase_mae(plain, plain_weights) > 0.1
    assert phase_mae(result.canvas, result.canvas_weights) < 0.02
    assert torch.allclose(result.canvas_weights, plain_weights, atol=1e-5)


def test_alignment_rejects_patch_outside_canvas():
    with pytest.raises(ValueError, match="inside the canvas"):
        align_patch_phases(
            torch.ones(1, PATCH, PATCH, dtype=torch.complex64), torch.tensor([[2.0, 2.0]]),
            torch.ones(PATCH, PATCH), CANVAS, patch_size=PATCH, uniform_weighting=True,
        )


def test_spectral_constants_recover_wrapped_constants():
    torch.manual_seed(0)
    truth, positions = _smooth_truth_and_grid()
    theta = 2 * torch.pi * torch.rand(len(positions)) - torch.pi
    patches = _rotated(gather_patches(truth, positions, PATCH), -theta)

    estimate = spectral_constants(patches, positions, _probe_weight(), PATCH)

    error = _wrap(estimate - theta)
    error = _wrap(
        error
        - torch.angle(torch.polar(torch.ones_like(error), error).mean())
    )
    assert error.abs().max() < 2e-2


def test_overlap_alignment_converges_in_few_sweeps_from_spectral_seed():
    torch.manual_seed(1)
    truth, positions = _smooth_truth_and_grid()
    theta = 2 * torch.pi * torch.rand(len(positions)) - torch.pi
    patches = _rotated(gather_patches(truth, positions, PATCH), -theta)

    result = align_patch_phases(
        patches,
        positions,
        _probe_weight(),
        (160, 160),
        patch_size=PATCH,
        uniform_weighting=False,
    )

    assert result.converged and result.sweeps <= 20
    error = _wrap(result.theta - theta)
    error = _wrap(
        error
        - torch.angle(torch.polar(torch.ones_like(error), error).mean())
    )
    assert error.abs().max() < 5e-3


def test_affine_method_is_withdrawn():
    from ptycho_torch.reassembly_phase_alignment import PHASE_ALIGNMENT_METHODS

    assert PHASE_ALIGNMENT_METHODS == ("none", "overlap")
    assert "tilt" not in PhaseAlignmentResult.__dataclass_fields__

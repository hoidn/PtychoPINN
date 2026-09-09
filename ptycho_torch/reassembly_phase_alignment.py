"""Per-patch phase gauge alignment for barycentric reassembly.

A diffraction pattern is blind to the constant phase of its own patch. This
module fits those constants against the overlap consensus without truth.

The constants-only objective is

    J(theta, O) = sum_k sum_x w(x) | exp(i theta_k) O_k(x) - (S_k O)(x) |^2

where ``S_k`` samples the canvas ``O`` at patch ``k``'s sub-pixel placement
with the same bilinear kernel the accumulator splats with (``S_k^T``). For
fixed ``theta`` the optimal ``O`` is the probe-weighted stitch of the rotated
patches; for fixed ``O`` the optimal ``theta_k`` is the angle of the weighted
overlap correlation.

Solver: spectral synchronisation of pairwise overlaps seeds all wrapped phase
constants at once, then Jacobi sweeps polish the same overlap objective until
the RMS wrapped step is below ``TOLERANCE_RAD`` or ``MAX_SWEEPS`` is reached.
``J`` is invariant to a common shift of all ``theta_k``, so the result is
gauged to zero circular mean. Alignment is deterministic up to GPU scatter-add
summation order.
"""
import warnings
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import torch
import torch.nn.functional as F

from ptycho_torch.reassembly_accumulators import (
    VectorizedWeightedAccumulator,
    bilinear_placement,
)

PHASE_ALIGNMENT_METHODS = ("none", "overlap")
MAX_SWEEPS = 300
TOLERANCE_RAD = 1e-3
REFINE_CHUNK = 256  # patches per gather/splat slice; bounds temporary memory
NEIGHBOURS = 12
PAIR_CHUNK = 256
_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class PatchBatch:
    """One accumulator batch: complex patches, canvas-pixel centres ``(x, y)``, and their weight map."""

    patches: torch.Tensor
    positions_px: torch.Tensor
    probe_mag_sq: torch.Tensor


@dataclass(frozen=True)
class PhaseAlignmentResult:
    canvas: torch.Tensor
    canvas_weights: torch.Tensor
    theta: torch.Tensor
    sweeps: int
    final_step_rad: float
    converged: bool


def gather_patches(
    reference: torch.Tensor,
    positions_px: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Adjoint of the accumulator's bilinear splat: sample ``reference`` (H, W) at each patch's placement."""
    xmin_wh, ymin_wh, corner_weights, valid = bilinear_placement(
        positions_px, patch_size, (int(reference.shape[0]), int(reference.shape[1]))
    )
    if not bool(valid.all()):
        raise ValueError("phase alignment requires every patch to lie inside the canvas")
    grid = torch.arange(patch_size, device=reference.device)
    rows = ymin_wh.view(-1, 1, 1) + grid.view(1, -1, 1)
    cols = xmin_wh.view(-1, 1, 1) + grid.view(1, 1, -1)
    flat = reference.reshape(-1)
    width = reference.shape[1]
    out = torch.zeros(
        (len(xmin_wh), patch_size, patch_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    for weight, (dy, dx) in zip(corner_weights, _CORNERS):
        out += weight.view(-1, 1, 1) * flat[(rows + dy) * width + cols + dx]
    return out


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rotated(patches: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return patches * torch.polar(torch.ones_like(theta), theta).view(-1, 1, 1)


def _overlap_edges(
    positions_px: torch.Tensor,
    patch_size: int,
    neighbours: int,
) -> torch.Tensor:
    """Unique overlapping pairs among each patch's nearest neighbours."""
    count = len(positions_px)
    if count < 2:
        return torch.empty((0, 2), dtype=torch.long, device=positions_px.device)
    distance = torch.cdist(positions_px, positions_px)
    distance.fill_diagonal_(float("inf"))
    near = distance.topk(min(neighbours, count - 1), largest=False).indices
    left = torch.arange(count, device=positions_px.device)[:, None].expand_as(near)
    edges = torch.stack((left.reshape(-1), near.reshape(-1)), dim=1).sort(1).values
    edges = torch.unique(edges, dim=0)
    delta = (positions_px[edges[:, 0]] - positions_px[edges[:, 1]]).abs()
    return edges[(delta < patch_size).all(1)]


def _sample_shifted(values: torch.Tensor, delta_xy: torch.Tensor) -> torch.Tensor:
    """Sample each image at local coordinates shifted by (dx, dy)."""
    _, height, width = values.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=torch.float32),
        torch.arange(width, device=values.device, dtype=torch.float32),
        indexing="ij",
    )
    x = x[None] + delta_xy[:, 0, None, None]
    y = y[None] + delta_xy[:, 1, None, None]
    grid = torch.stack(
        (2 * x / max(width - 1, 1) - 1, 2 * y / max(height - 1, 1) - 1),
        dim=-1,
    )
    channels = (
        torch.view_as_real(values).movedim(-1, 1)
        if values.is_complex()
        else values[:, None]
    )
    sampled = F.grid_sample(
        channels, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return (
        torch.view_as_complex(sampled.movedim(1, -1).contiguous())
        if values.is_complex()
        else sampled[:, 0]
    )


def spectral_constants(
    patches: torch.Tensor,
    positions_px: torch.Tensor,
    weight: torch.Tensor,
    patch_size: int,
    *,
    neighbours: int = NEIGHBOURS,
) -> torch.Tensor:
    """Seed wrapped patch constants from the leading overlap eigenvector."""
    count = len(patches)
    if count == 1:
        return torch.zeros(1, device=patches.device)
    edges = _overlap_edges(positions_px, patch_size, neighbours)
    matrix = torch.zeros(
        count, count, dtype=torch.complex64, device=patches.device
    )
    for chunk in edges.split(PAIR_CHUNK):
        left, right = chunk.T
        delta = positions_px[left] - positions_px[right]
        shifted = _sample_shifted(patches[right], delta)
        overlap_weight = weight * _sample_shifted(
            weight.expand(len(chunk), -1, -1), delta
        )
        correlation = (
            overlap_weight * torch.conj(patches[left]) * shifted
        ).sum(dim=(-2, -1)).to(torch.complex64)
        matrix[left, right] = correlation
        matrix[right, left] = torch.conj(correlation)
    degree = matrix.abs().sum(1)
    _, vectors = torch.linalg.eigh(
        (matrix + torch.diag(degree).to(matrix.dtype)).to(torch.complex128)
    )
    theta = torch.angle(vectors[:, -1]).to(torch.float32)
    return _wrap(
        theta - torch.angle(torch.polar(torch.ones_like(theta), theta).mean())
    )


class _Field:
    """The patch set plus the fixed pieces of the objective: weight map, canvas, splat and gather."""

    def __init__(self, patches, positions_px, probe_mag_sq, canvas_shape, patch_size, uniform_weighting):
        if not bool(bilinear_placement(positions_px, patch_size, canvas_shape)[3].all()):
            raise ValueError("phase alignment requires every patch to lie inside the canvas")
        self.patches = patches
        self.positions = positions_px
        self.probe_mag_sq = probe_mag_sq
        self.weight = torch.ones_like(probe_mag_sq) if uniform_weighting else probe_mag_sq
        self.canvas_shape = canvas_shape
        self.patch_size = patch_size
        self.uniform_weighting = uniform_weighting
        self.device = patches.device

    def empty_canvas(self):
        canvas = torch.zeros(self.canvas_shape, dtype=torch.complex64, device=self.device)
        weights = torch.zeros(self.canvas_shape, dtype=torch.float32, device=self.device)
        return canvas, weights, VectorizedWeightedAccumulator(self.canvas_shape, self.device)

    def splat(self, accumulator, canvas, weights, index, theta):
        accumulator.accumulate_batch(
            canvas, weights, _rotated(self.patches[index], theta), self.positions[index],
            self.probe_mag_sq, patch_size=self.patch_size,
            uniform_weighting=self.uniform_weighting,
        )

    def stitch(self, theta):
        canvas, weights, accumulator = self.empty_canvas()
        for index in torch.arange(len(self.patches), device=self.device).split(REFINE_CHUNK):
            self.splat(accumulator, canvas, weights, index, theta[index])
        return canvas, weights

    def best_constants(self, reference, index):
        """theta_k minimising the weighted disagreement between patch k and ``reference`` at its placement."""
        sampled = gather_patches(reference, self.positions[index], self.patch_size)
        correlation = (self.weight * torch.conj(self.patches[index]) * sampled).sum(dim=(-2, -1))
        return torch.angle(correlation)

    def jacobi_sweep(self, theta):
        canvas, weights = self.stitch(theta)
        reference = canvas / (weights + 1e-12)
        updates = [
            self.best_constants(reference, index)
            for index in torch.arange(len(self.patches), device=self.device).split(REFINE_CHUNK)
        ]
        return torch.cat(updates)

def align_patch_phases(
    patches: torch.Tensor,
    positions_px: torch.Tensor,
    probe_mag_sq: torch.Tensor,
    canvas_shape: Tuple[int, int],
    *,
    patch_size: int,
    uniform_weighting: bool,
    max_sweeps: int = MAX_SWEEPS,
    tolerance_rad: float = TOLERANCE_RAD,
) -> PhaseAlignmentResult:
    """Fit one phase constant per patch from overlaps and return the aligned stitch.

    ``patches`` is ``(N, m, m)`` complex, ``positions_px`` ``(N, 2)`` canvas
    pixel centres ``(x, y)``, ``probe_mag_sq`` the shared ``(m, m)`` weight map.
    ``canvas`` is the weighted numerator; divide by ``canvas_weights`` for the
    object, exactly as for the plain accumulator output.
    """
    if len(patches) == 0:
        raise ValueError("phase alignment needs at least one patch")
    field = _Field(patches, positions_px, probe_mag_sq, canvas_shape, patch_size, uniform_weighting)
    theta = spectral_constants(patches, positions_px, field.weight, patch_size)
    step = float("inf")
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        new_theta = field.jacobi_sweep(theta)
        step = float(torch.sqrt(_wrap(new_theta - theta).square().mean()))
        theta = new_theta
        if step < tolerance_rad:
            break
    mean_rotation = torch.angle(torch.polar(torch.ones_like(theta), theta).mean())
    theta = _wrap(theta - mean_rotation)
    canvas, weights = field.stitch(theta)
    return PhaseAlignmentResult(
        canvas=canvas,
        canvas_weights=weights,
        theta=theta,
        sweeps=sweeps,
        final_step_rad=step,
        converged=step < tolerance_rad,
    )


def _calibrated(patches: torch.Tensor, s1: Any, s2: Any, channels_swapped: bool) -> torch.Tensor:
    """Apply the channel-swap correction and VarPro (s1, s2) scaling to retained
    texture patches, mirroring what the canvas path does to the texture canvas."""
    if channels_swapped:
        patches = torch.complex(patches.imag, patches.real)
    return torch.complex(s1 * patches.real, s2 * patches.imag)


def align_calibrated_stitch(
    batches: Sequence[PatchBatch],
    canvas_shape: Tuple[int, int],
    *,
    s1: Any,
    s2: Any,
    channels_swapped: bool,
    patch_size: int,
    uniform_weighting: bool,
    verbose: bool,
) -> torch.Tensor:
    """Calibrate the retained texture batches, align their phases, and return the
    finished (weight-normalised) scaled canvas for ``reconstruct_image_barycentric``."""
    if not batches:
        raise ValueError("phase alignment needs at least one retained batch")
    probe_mag_sq = batches[0].probe_mag_sq
    if any(not torch.equal(batch.probe_mag_sq, probe_mag_sq) for batch in batches):
        raise ValueError("phase alignment requires one probe weight map for every batch")
    alignment = align_patch_phases(
        _calibrated(torch.cat([batch.patches for batch in batches]), s1, s2, channels_swapped),
        torch.cat([batch.positions_px for batch in batches]),
        probe_mag_sq,
        canvas_shape,
        patch_size=patch_size,
        uniform_weighting=uniform_weighting,
    )
    if not alignment.converged:
        warnings.warn(
            f"overlap phase alignment stopped after {alignment.sweeps} sweeps "
            f"with a final step of {alignment.final_step_rad:.2e} rad (not converged)",
            stacklevel=2,
        )
    if verbose:
        print(
            f"Overlap phase alignment: {alignment.sweeps} sweeps, final step "
            f"{alignment.final_step_rad:.2e} rad, per-patch constants std "
            f"{alignment.theta.std().item():.3f} rad"
        )
    return alignment.canvas / (alignment.canvas_weights + 1e-12)

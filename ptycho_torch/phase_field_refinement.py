"""Truth-free likelihood refinement of a smooth field on a stitched object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ptycho_torch.reconstruction_evaluation import canvas_positions


@dataclass(frozen=True)
class RefinementResult:
    canvas: np.ndarray
    phase_field: np.ndarray
    log_amplitude_field: np.ndarray | None
    seed: str
    seed_tilt_rms_rad_per_px: float | None
    seed_field_rms_rad: float | None
    gain_seed_rms: float | None
    log_amplitude_rms: float | None
    global_gain: float | None
    seed_field: np.ndarray
    nll_before: float
    nll_after_seed: float
    nll_after: float
    iterations: int
    converged: bool
    grid_pitch_px: int


def extract_canvas_patches(
    canvas: torch.Tensor, positions_px: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Bilinearly sample patches using the historical integer-centre convention."""
    if canvas.ndim != 2 or not torch.is_complex(canvas):
        raise ValueError("canvas must be a 2D complex tensor")
    if positions_px.ndim != 2 or positions_px.shape[1] != 2:
        raise ValueError("positions_px must have shape (frames, 2)")
    height, width = canvas.shape
    grid = (
        torch.arange(patch_size, dtype=torch.float32, device=canvas.device)
        - patch_size / 2
    )
    x = positions_px[:, 0, None, None] + grid[None, None, :]
    y = positions_px[:, 1, None, None] + grid[None, :, None]
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = x - x0, y - y0
    flat = canvas.reshape(-1)

    def tap(yy: torch.Tensor, xx: torch.Tensor, weight: torch.Tensor):
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        index = yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)
        return flat[index] * weight * inside

    return (
        tap(y0, x0, (1 - fy) * (1 - fx))
        + tap(y0, x0 + 1, (1 - fy) * fx)
        + tap(y0 + 1, x0, fy * (1 - fx))
        + tap(y0 + 1, x0 + 1, fy * fx)
    )


def expected_counts(
    canvas: torch.Tensor, positions_px: torch.Tensor, probe: torch.Tensor
) -> torch.Tensor:
    """Return incoherent count intensity for a 2D or modes-first probe."""
    if probe.ndim not in (2, 3) or probe.shape[-2] != probe.shape[-1]:
        raise ValueError("probe must have shape (m, m) or (modes, m, m)")
    patches = extract_canvas_patches(canvas, positions_px, int(probe.shape[-1]))
    modes = probe[None] if probe.ndim == 2 else probe
    exit_waves = patches[:, None] * modes[None]
    fields = torch.fft.fftshift(
        torch.fft.fft2(exit_waves, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
    )
    return fields.abs().square().sum(dim=1)


def poisson_nll(
    expected: torch.Tensor, counts: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Poisson negative log-likelihood without the data-only factorial term."""
    if expected.shape != counts.shape:
        raise ValueError("expected counts and observed counts must have matching shapes")
    # float64 accumulation: a float32 sum of ~4e6 terms at ~3e9 resolves only 256 counts,
    # which stalls the strong-Wolfe line search once the seed has removed most of the error.
    return (expected - counts * torch.log(expected + eps)).double().sum()


def _interpolation_matrix(size_out: int, size_in: int, device: torch.device) -> torch.Tensor:
    """1D bicubic interpolation matrix (size_out, size_in), align_corners=True.

    Built by pushing the identity through ``F.interpolate`` so it matches the 2D upsample exactly.
    """
    eye = torch.eye(size_in, device=device)
    return F.interpolate(
        eye[None, None], size=(size_out, size_in), mode="bicubic", align_corners=True
    )[0, 0]


def _bilinear_rows(matrix: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Rows of ``matrix`` bilinearly mixed at fractional ``coordinates`` (frames, patch).

    Coordinates outside [0, n) weigh zero, matching the taps of extract_canvas_patches.
    """
    n, k = matrix.shape
    padded = torch.cat([matrix, matrix.new_zeros(1, k)])
    lower = coordinates.floor().long()
    fraction = (coordinates - lower)[..., None]

    def row(index: torch.Tensor) -> torch.Tensor:
        inside = (index >= 0) & (index < n)
        return padded[index.clamp(0, n - 1).masked_fill(~inside, n)]

    return (1 - fraction) * row(lower) + fraction * row(lower + 1)


class SmoothField(torch.nn.Module):
    """A coarse bicubic field plus an exact global tilt, all zero-initialized.

    The field is the separable product A_y · C · A_xᵀ of the coarse grid with 1D bicubic
    interpolation matrices; it equals the bicubic upsample of the grid but has an explicit
    linear form, so patches can be sampled without a canvas gather and per-frame gradients
    integrate by least squares. ``tilt`` is stored in radians per half-canvas so its scale
    matches a coarse cell's: in rad/px it dominated the first L-BFGS steps. Parameters stay
    on the CPU because L-BFGS's per-iteration bookkeeping is dozens of tiny vector ops that
    are latency-bound on a shared GPU; the interpolation matrices live on ``device``.
    """

    def __init__(
        self, canvas_shape: tuple[int, int], pitch_px: int, device: str | torch.device = "cpu"
    ):
        super().__init__()
        if pitch_px <= 0:
            raise ValueError("pitch_px must be positive")
        self.canvas_shape = tuple(int(value) for value in canvas_shape)
        height, width = self.canvas_shape
        rows = (height + pitch_px - 1) // pitch_px + 1
        cols = (width + pitch_px - 1) // pitch_px + 1
        self.coarse = torch.nn.Parameter(torch.zeros(rows, cols))
        self.tilt = torch.nn.Parameter(torch.zeros(2))
        self.device = torch.device(device)
        self.row_matrix = _interpolation_matrix(height, rows, self.device)
        self.column_matrix = _interpolation_matrix(width, cols, self.device)
        self.tilt_scale = torch.tensor(
            [2 / max(width - 1, 1), 2 / max(height - 1, 1)], device=self.device
        )
        self.centre = ((width - 1) / 2, (height - 1) / 2)
        self._x = torch.arange(width, device=self.device) - self.centre[0]
        self._y = torch.arange(height, device=self.device) - self.centre[1]

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Coarse grid and tilt in rad/px, on the compute device."""
        return self.coarse.to(self.device), self.tilt.to(self.device) * self.tilt_scale

    def forward(self) -> torch.Tensor:
        coarse, tilt = self.coefficients()
        curved = self.row_matrix @ coarse @ self.column_matrix.T
        return curved + tilt[0] * self._x[None] + tilt[1] * self._y[:, None]

    def patch_weights(
        self, positions_px: torch.Tensor, patch_size: int
    ) -> tuple[torch.Tensor, ...]:
        """Constant sampling weights for sample_patches at the given frame centres."""
        grid = (
            torch.arange(patch_size, dtype=torch.float32, device=self.device)
            - patch_size / 2
        )
        x = positions_px[:, 0, None] + grid[None]
        y = positions_px[:, 1, None] + grid[None]
        return (
            _bilinear_rows(self.row_matrix, y),
            _bilinear_rows(self.column_matrix, x),
            x - self.centre[0],
            y - self.centre[1],
        )

    def sample_patches(self, weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Bilinear patches of the field, (frames, patch, patch), differentiable."""
        row_weights, column_weights, x, y = weights
        coarse, tilt = self.coefficients()
        curved = torch.einsum("jrk,kl,jcl->jrc", row_weights, coarse, column_weights)
        return curved + tilt[0] * x[:, None, :] + tilt[1] * y[:, :, None]


def frame_tilts(
    canvas: torch.Tensor,
    positions_px: torch.Tensor,
    probe: torch.Tensor,
    counts: torch.Tensor,
    *,
    chunk: int = 512,
    eps: float = 1e-6,
    floor: float = 10.0,
) -> torch.Tensor:
    """One Poisson Gauss-Newton ramp step per frame, as (x, y) rad/pixel."""
    size = int(probe.shape[-1])
    tilts = []
    modes = probe[None] if probe.ndim == 2 else probe
    y, x = torch.meshgrid(
        torch.arange(canvas.shape[0], device=canvas.device),
        torch.arange(canvas.shape[1], device=canvas.device),
        indexing="ij",
    )

    def fft(wave: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftshift(
            torch.fft.fft2(wave, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
        )

    for index in torch.arange(len(positions_px), device=canvas.device).split(chunk):
        patches = extract_canvas_patches(canvas, positions_px[index], size)
        waves = patches[:, None] * modes[None]
        fields = fft(waves)
        expected = fields.abs().square().sum(dim=1) + eps
        residual = counts[index] - expected
        derivative_x = extract_canvas_patches(1j * canvas * x, positions_px[index], size)
        derivative_y = extract_canvas_patches(1j * canvas * y, positions_px[index], size)
        gradient_x = 2 * (
            fields.conj()
            * fft(derivative_x[:, None] * modes[None])
        ).real.sum(dim=1)
        gradient_y = 2 * (
            fields.conj()
            * fft(derivative_y[:, None] * modes[None])
        ).real.sum(dim=1)
        inverse = (expected + floor).reciprocal()
        fxx = (gradient_x.square() * inverse).sum((-2, -1))
        fyy = (gradient_y.square() * inverse).sum((-2, -1))
        fxy = (gradient_x * gradient_y * inverse).sum((-2, -1))
        bx = (residual * inverse * gradient_x).sum((-2, -1))
        by = (residual * inverse * gradient_y).sum((-2, -1))
        determinant = (fxx * fyy - fxy.square()).clamp_min(eps)
        tilt_x = (fyy * bx - fxy * by) / determinant
        tilt_y = (fxx * by - fxy * bx) / determinant
        tilts.append(torch.stack((tilt_x, tilt_y), dim=1))
    return torch.cat(tilts)


def frame_gains(
    canvas: torch.Tensor,
    positions_px: torch.Tensor,
    probe: torch.Tensor,
    counts: torch.Tensor,
    *,
    chunk: int = 512,
) -> torch.Tensor:
    """Read the Poisson maximum-likelihood log-amplitude gain per frame."""
    gains = []
    for index in torch.arange(len(positions_px), device=canvas.device).split(chunk):
        predicted_total = expected_counts(
            canvas, positions_px[index], probe
        ).sum((-2, -1))
        observed_total = counts[index].sum((-2, -1))
        gains.append(0.5 * torch.log(observed_total / predicted_total))
    return torch.cat(gains)


def fit_gain_field(
    field: SmoothField,
    positions_px: torch.Tensor,
    gains: torch.Tensor,
    *,
    probe_intensity: torch.Tensor,
) -> float:
    """Fit probe-weighted frame gains with one smooth-field least-squares solve."""
    intensity = probe_intensity.to(field.device, dtype=torch.float32)
    if intensity.ndim != 2 or intensity.shape[0] != intensity.shape[1]:
        raise ValueError("probe_intensity must be a square 2D tensor")
    intensity = intensity / intensity.sum()
    positions = positions_px.to(field.device)
    weights = field.patch_weights(positions, int(intensity.shape[0]))
    row_weights, column_weights, x, y = weights
    curved = torch.einsum(
        "uv,juk,jvl->jkl", intensity, row_weights, column_weights
    ).flatten(1)
    tilt_x = (intensity[None] * x[:, None, :]).sum((-2, -1))
    tilt_y = (intensity[None] * y[:, :, None]).sum((-2, -1))
    design = torch.cat([curved, tilt_x[:, None], tilt_y[:, None]], dim=1).double()
    target = gains.to(field.device).double()
    normal = design.T @ design
    normal += 1e-6 * normal.diagonal().mean() * torch.eye(
        len(normal), device=field.device, dtype=normal.dtype
    )
    solution = torch.linalg.solve(normal, design.T @ target).float().cpu()
    with torch.no_grad():
        field.coarse.copy_(solution[:-2].reshape(field.coarse.shape))
        field.tilt.copy_(solution[-2:] / field.tilt_scale.cpu())
        fitted = (field.sample_patches(weights) * intensity).sum((-2, -1))
        return float((fitted - target).square().mean().sqrt())


def _field_gradient_at(
    field: SmoothField, positions_px: torch.Tensor
) -> torch.Tensor:
    phase = field()
    gradient_x = (
        torch.roll(phase, -1, dims=1) - torch.roll(phase, 1, dims=1)
    ) / 2
    gradient_y = (
        torch.roll(phase, -1, dims=0) - torch.roll(phase, 1, dims=0)
    ) / 2
    column = positions_px[:, 0].round().long().clamp(1, phase.shape[1] - 2)
    row = positions_px[:, 1].round().long().clamp(1, phase.shape[0] - 2)
    return torch.stack((gradient_x[row, column], gradient_y[row, column]), dim=1)


def fit_tilt_field(
    field: SmoothField, positions_px: torch.Tensor, tilts: torch.Tensor
) -> float:
    """Integrate per-frame tilts into the smooth field by least squares; return gradient RMS.

    The field gradient at a frame centre is linear in the coefficients, so the fit is one
    ridge-regularized normal-equation solve (the ridge resolves cells no frame covers and
    the tilt's overlap with the grid's own plane).
    """
    height, width = field.canvas_shape
    rows, columns = field.row_matrix, field.column_matrix
    column = positions_px[:, 0].round().long().clamp(1, width - 2)
    row = positions_px[:, 1].round().long().clamp(1, height - 2)
    d_column = (columns[column + 1] - columns[column - 1]) / 2
    d_row = (rows[row + 1] - rows[row - 1]) / 2
    design_x = (rows[row][:, :, None] * d_column[:, None, :]).flatten(1)
    design_y = (d_row[:, :, None] * columns[column][:, None, :]).flatten(1)
    ones = torch.ones(len(positions_px), 1, device=field.device)
    zeros = torch.zeros_like(ones)
    design = torch.cat(
        [
            torch.cat([design_x, ones, zeros], dim=1),
            torch.cat([design_y, zeros, ones], dim=1),
        ]
    ).double()
    target = torch.cat([tilts[:, 0], tilts[:, 1]]).double().to(field.device)
    normal = design.T @ design
    normal += 1e-6 * normal.diagonal().mean() * torch.eye(
        len(normal), device=field.device, dtype=normal.dtype
    )
    solution = torch.linalg.solve(normal, design.T @ target).float().cpu()
    with torch.no_grad():
        field.coarse.copy_(solution[:-2].reshape(field.coarse.shape))
        field.tilt.copy_(solution[-2:] / field.tilt_scale.cpu())
        return float(
            (_field_gradient_at(field, positions_px) - tilts.to(field.device))
            .square()
            .mean()
            .sqrt()
        )


def refine_phase_field(
    canvas: Any,
    anchor: Mapping[str, Any],
    counts: Any,
    xcoords: Any,
    ycoords: Any,
    probe: Any,
    *,
    support_mask: Any,
    grid_pitch_px: int = 16,
    seed: str = "tilt",
    fit_log_amplitude: bool = False,
    max_iter: int = 10,
    chunk: int = 512,
    device: str = "cuda",
) -> RefinementResult:
    """Fit a smooth phase field and, optionally, a bounded log-amplitude field."""
    base_array = np.asarray(canvas, dtype=np.complex64)
    count_array = np.asarray(counts, dtype=np.float32)
    probe_array = np.asarray(probe, dtype=np.complex64)
    support_array = np.asarray(support_mask)
    if base_array.ndim != 2 or count_array.ndim != 3:
        raise ValueError("canvas must be 2D and counts must have shape (frames, m, m)")
    if (
        support_array.shape != base_array.shape
        or support_array.dtype != np.bool_
        or not np.any(support_array)
    ):
        raise ValueError("support_mask must be boolean, canvas-shaped, and nonempty")
    if not np.isfinite(base_array).all() or not np.isfinite(count_array).all():
        raise ValueError("canvas and counts must be finite")
    if seed not in {"tilt", "none"}:
        raise ValueError("seed must be 'tilt' or 'none'")
    if max_iter < 0 or chunk <= 0:
        raise ValueError("max_iter must be nonnegative and chunk must be positive")

    target_device = torch.device(device)
    base = torch.as_tensor(base_array, device=target_device)
    data = torch.as_tensor(count_array, device=target_device)
    probe_tensor = torch.as_tensor(probe_array, device=target_device)
    positions = torch.as_tensor(
        canvas_positions(xcoords, ycoords, anchor, base_array.shape),
        dtype=torch.float32,
        device=target_device,
    )
    if len(positions) != len(data):
        raise ValueError("positions and count frames must have matching lengths")
    support = torch.as_tensor(support_array, device=target_device)
    phase = SmoothField(base_array.shape, grid_pitch_px, device=target_device)
    parameters = list(phase.parameters())
    gain = (
        SmoothField(base_array.shape, grid_pitch_px, device=target_device)
        if fit_log_amplitude
        else None
    )
    if gain is not None:
        parameters.extend(gain.parameters())
    size = int(probe_tensor.shape[-1])
    modes = probe_tensor[None] if probe_tensor.ndim == 2 else probe_tensor
    # Each frame is modelled as the bilinear patch of the frozen canvas times exp(i·phase)
    # sampled at the same taps; for a smooth field this differs from resampling the product
    # by O((∂phase)²). The frozen factors are computed once, so a likelihood evaluation is
    # one exponential, one FFT, and the coarse-grid einsum per chunk. The NLL is a sum over
    # pixels, so the counts are ifft-shifted once instead of fft-shifting every prediction.
    with torch.no_grad():
        shifted = torch.fft.ifftshift(data, dim=(-2, -1))
        frames = [
            (
                extract_canvas_patches(base, positions[indices], size)[:, None]
                * modes[None],
                phase.patch_weights(positions[indices], size),
                shifted[indices],
            )
            for indices in torch.arange(len(data), device=target_device).split(chunk)
        ]

    def total_nll(*, backward: bool) -> float:
        total = 0.0
        for exit_base, weights, observed in frames:
            field_patches = phase.sample_patches(weights)
            factor = torch.polar(
                torch.ones_like(field_patches), field_patches
            )
            if gain is not None:
                factor = factor * torch.exp(
                    0.5 * torch.tanh(gain.sample_patches(weights))
                )
            waves = exit_base * factor[:, None]
            expected = (
                torch.fft.fft2(waves, dim=(-2, -1), norm="ortho").abs().square().sum(dim=1)
            )
            loss = poisson_nll(expected, observed)
            if backward:
                (loss / data.numel()).backward()
            total += float(loss.detach())
        return total

    with torch.no_grad():
        nll_before = total_nll(backward=False)
    seed_tilt_rms = None
    if seed == "tilt":
        tilts = frame_tilts(base, positions, probe_tensor, data, chunk=chunk)
        seed_tilt_rms = float(tilts.square().sum(1).mean().sqrt())
        fit_tilt_field(phase, positions, tilts)
    gain_seed_rms = None
    if gain is not None:
        with torch.no_grad():
            phase_seed = phase()
            seeded_canvas = base * torch.polar(
                torch.ones_like(phase_seed), phase_seed
            )
            gain_values = frame_gains(
                seeded_canvas, positions, probe_tensor, data, chunk=chunk
            )
            unconstrained = torch.atanh(
                (2 * gain_values).clamp(-1 + 1e-6, 1 - 1e-6)
            )
            probe_intensity = modes.abs().square().sum(dim=0)
        fit_gain_field(
            gain,
            positions,
            unconstrained,
            probe_intensity=probe_intensity,
        )
        with torch.no_grad():
            normalized_probe = probe_intensity / probe_intensity.sum()
            fitted_values = torch.cat(
                [
                    (
                        0.5
                        * torch.tanh(gain.sample_patches(weights))
                        * normalized_probe
                    ).sum((-2, -1))
                    for _, weights, _ in frames
                ]
            )
            gain_seed_rms = float(
                (fitted_values - gain_values).square().mean().sqrt()
            )
    with torch.no_grad():
        seed_field = phase()
        seed_field = seed_field - seed_field[support].mean()
        seed_field_rms = (
            float(seed_field[support].square().mean().sqrt())
            if seed == "tilt"
            else None
        )
        nll_after_seed = total_nll(backward=False)

    iterations = 0
    if max_iter:
        optimizer = torch.optim.LBFGS(
            parameters,
            max_iter=int(max_iter),
            history_size=20,
            tolerance_grad=1e-9,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = total_nll(backward=True) / data.numel()
            return torch.tensor(loss, dtype=torch.float64)

        optimizer.step(closure)
        iterations = int(optimizer.state[parameters[0]].get("n_iter", 0))
    with torch.no_grad():
        nll_after = total_nll(backward=False)
        phase_field = phase()
        phase_field = phase_field - phase_field[support].mean()
        log_amplitude_field = (
            0.5 * torch.tanh(gain()) if gain is not None else None
        )
        amplitude_factor = (
            torch.exp(log_amplitude_field)
            if log_amplitude_field is not None
            else torch.ones_like(base.real)
        )
        result_canvas = base * amplitude_factor * torch.polar(
            torch.ones_like(base.real), phase_field
        )
        log_amplitude_rms = (
            float(log_amplitude_field[support].square().mean().sqrt())
            if log_amplitude_field is not None
            else None
        )
        global_gain = (
            float(torch.exp(log_amplitude_field[support]).mean())
            if log_amplitude_field is not None
            else None
        )

    return RefinementResult(
        canvas=result_canvas.cpu().numpy().astype(np.complex64),
        phase_field=phase_field.cpu().numpy().astype(np.float32),
        log_amplitude_field=(
            log_amplitude_field.cpu().numpy().astype(np.float32)
            if log_amplitude_field is not None
            else None
        ),
        seed=seed,
        seed_tilt_rms_rad_per_px=seed_tilt_rms,
        seed_field_rms_rad=seed_field_rms,
        gain_seed_rms=gain_seed_rms,
        log_amplitude_rms=log_amplitude_rms,
        global_gain=global_gain,
        seed_field=seed_field.cpu().numpy().astype(np.float32),
        nll_before=nll_before,
        nll_after_seed=nll_after_seed,
        nll_after=nll_after,
        iterations=iterations,
        converged=bool(
            nll_after <= nll_after_seed
            and (max_iter == 0 or iterations < max_iter)
        ),
        grid_pitch_px=int(grid_pitch_px),
    )

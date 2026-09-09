"""Global gauge quotient for reconstruction scoring.

A ptychographic reconstruction is determined only up to the global gauge group
``G = {s * exp(i * (c + a*x + b*y)) : s > 0}``: one positive amplitude scale
(the probe/object scalar ambiguity), one phase constant, and one linear phase
ramp (a uniform sub-pixel shift of every diffraction pattern). This module fits
that gauge between an aligned reconstruction and its truth on a boolean support
and applies it, so every metric downstream is invariant on the orbit. Nothing
higher-order is fitted here; per-patch constants and tilts are reconstruction
errors and stay with the reconstruction path.

Design: ``docs/superpowers/specs/2026-09-04-reconstruction-global-gauge-design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

GAUGE_METHOD = "global_scale_affine_phase_v1"
_PLANE_FIT_ROUNDS = 3


class GaugeError(ValueError):
    """Raised for invalid gauge inputs."""


@dataclass(frozen=True)
class GlobalGauge:
    """``s * exp(i * (c + ramp_x * x + ramp_y * y))``; ``x`` is the column index."""

    scale: float
    phase_offset: float
    ramp_x: float
    ramp_y: float

    def apply(self, array: Any) -> NDArray[np.complex128]:
        values = np.asarray(array, dtype=np.complex128)
        if values.ndim != 2:
            raise GaugeError("gauge applies to 2D arrays only")
        rows, cols = np.indices(values.shape, dtype=np.float64)
        phase = self.phase_offset + self.ramp_x * cols + self.ramp_y * rows
        return self.scale * np.exp(1j * phase) * values

    def to_jsonable(self, shape: tuple[int, int]) -> dict[str, Any]:
        height, width = int(shape[0]), int(shape[1])
        span = math.hypot(self.ramp_x * (width - 1), self.ramp_y * (height - 1))
        return {
            "method": GAUGE_METHOD,
            "scale": float(self.scale),
            "phase_offset": float(self.phase_offset),
            "ramp_x_rad_per_px": float(self.ramp_x),
            "ramp_y_rad_per_px": float(self.ramp_y),
            "ramp_span_rad": float(span),
        }


IDENTITY_GAUGE = GlobalGauge(scale=1.0, phase_offset=0.0, ramp_x=0.0, ramp_y=0.0)


def _validated(reconstruction: Any, target: Any, mask: Any):
    recon = np.asarray(reconstruction)
    truth = np.asarray(target)
    support = np.asarray(mask)
    if recon.ndim != 2 or recon.shape != truth.shape:
        raise GaugeError("gauge inputs must be equal-shape 2D arrays")
    if support.shape != recon.shape or support.dtype != np.bool_:
        raise GaugeError("gauge mask must be boolean and match the inputs")
    recon = recon.astype(np.complex128, copy=False)
    truth = truth.astype(np.complex128, copy=False)
    return recon, truth, support


def _peak_offset(values: NDArray[np.float64], index: int) -> float:
    """Sub-bin parabolic refinement of a periodic 1D peak."""
    size = values.size
    left, centre, right = (
        values[(index - 1) % size],
        values[index],
        values[(index + 1) % size],
    )
    denominator = left - 2.0 * centre + right
    if denominator >= 0.0:
        return 0.0
    return float(0.5 * (left - right) / denominator)


def _coarse_ramp(phasor: NDArray[np.complex128]) -> tuple[float, float]:
    spectrum = np.abs(np.fft.fft2(phasor))
    row, col = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    height, width = phasor.shape
    frequency_x = np.fft.fftfreq(width)[col] + _peak_offset(spectrum[row, :], col) / width
    frequency_y = np.fft.fftfreq(height)[row] + _peak_offset(spectrum[:, col], row) / height
    return 2.0 * math.pi * float(frequency_x), 2.0 * math.pi * float(frequency_y)


def fit_global_gauge(reconstruction: Any, target: Any, mask: Any) -> GlobalGauge:
    """Fit the gauge ``g`` minimising ``|g * reconstruction - target|`` on ``mask``.

    The phase part is fitted on the unit residual phasor
    ``reconstruction * conj(target)``: a coarse ramp from its FFT peak, then
    weighted least-squares plane fits on the wrapped residual, which are
    wrap-safe once the remaining ramp is below one FFT bin. The scale is the
    positive least-squares amplitude factor. A reconstruction with no energy
    on the support returns the identity gauge.
    """
    recon, truth, support = _validated(reconstruction, target, mask)
    if not support.any():
        raise GaugeError("gauge mask selects no pixels")
    # Nonfinite pixels take no part in the fit; the metric functions reject
    # them later with their own finiteness errors.
    support = support & np.isfinite(recon) & np.isfinite(truth)
    recon_scale = float(np.max(np.abs(recon[support]), initial=0.0))
    truth_scale = float(np.max(np.abs(truth[support]), initial=0.0))
    if recon_scale == 0.0 or truth_scale == 0.0:
        return IDENTITY_GAUGE
    # Work on max-normalised copies so near-overflow magnitudes stay finite;
    # the phase fit is scale-free and the scale is restored at the end.
    recon = np.where(support, recon / recon_scale, 0.0)
    truth = np.where(support, truth / truth_scale, 0.0)
    weight = np.abs(recon) * np.abs(truth)
    recon_energy = float(np.sum(np.abs(recon[support]) ** 2))
    if recon_energy == 0.0 or float(np.sum(weight)) == 0.0:
        return IDENTITY_GAUGE
    product = recon * np.conj(truth)
    phasor = np.zeros_like(product)
    nonzero = weight > 0.0
    phasor[nonzero] = product[nonzero] / np.abs(product[nonzero])

    rows, cols = np.indices(recon.shape, dtype=np.float64)
    beta, gamma = _coarse_ramp(phasor)
    points = nonzero
    design = np.column_stack(
        [np.ones(int(points.sum())), cols[points], rows[points]]
    )
    root_weight = np.sqrt(weight[points])
    for _ in range(_PLANE_FIT_ROUNDS):
        residual = phasor[points] * np.exp(-1j * (beta * cols[points] + gamma * rows[points]))
        constant = np.angle(np.sum(weight[points] * residual))
        wrapped = np.angle(residual * np.exp(-1j * constant))
        coefficients, *_ = np.linalg.lstsq(
            design * root_weight[:, None], wrapped * root_weight, rcond=None
        )
        beta += float(coefficients[1])
        gamma += float(coefficients[2])
    residual = phasor[points] * np.exp(-1j * (beta * cols[points] + gamma * rows[points]))
    alpha = float(np.angle(np.sum(weight[points] * residual)))
    scale = float(np.sum(weight[support]) / recon_energy) * truth_scale / recon_scale
    gauge = GlobalGauge(
        scale=scale,
        phase_offset=float(np.angle(np.exp(-1j * alpha))),
        ramp_x=-beta,
        ramp_y=-gamma,
    )
    if not all(
        math.isfinite(value)
        for value in (gauge.scale, gauge.phase_offset, gauge.ramp_x, gauge.ramp_y)
    ) or gauge.scale <= 0.0:
        raise GaugeError("gauge fit produced a nonfinite or nonpositive result")
    return gauge


__all__ = [
    "GAUGE_METHOD",
    "GaugeError",
    "GlobalGauge",
    "IDENTITY_GAUGE",
    "fit_global_gauge",
]

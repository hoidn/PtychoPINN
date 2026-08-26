"""Production raw-array reconstruction metrics and visual diagnostics.

The numerical API is intentionally independent of plotting.  Matplotlib and
Pillow are imported only by the comparison renderer after scoring has already
completed from the aligned complex arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as _structural_similarity


METRIC_CONTRACT_VERSION = "synthetic-quality-metrics-v1"
RENDERER_VERSION = "synthetic-comparison-renderer-v1"
PNG_WIDTH = 2100
PNG_HEIGHT = 1350
PANEL_NAMES = (
    "amplitude_truth",
    "amplitude_reconstruction",
    "amplitude_absolute_error",
    "phase_truth",
    "phase_reconstruction",
    "phase_wrapped_error",
)

_RELATIVE_ORTHOGONALITY_TOLERANCE = 1e-12
_RELATIVE_DEGENERACY_TOLERANCE = 8.0 * np.finfo(np.float64).eps


class MetricError(ValueError):
    """Raised when quality cannot be evaluated under the declared policy."""


class AlignmentError(MetricError):
    """Raised when reconstruction and truth coordinates cannot be aligned."""


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Recognizability metrics for one reconstruction canvas."""

    amplitude_ssim: float
    amplitude_pearson: float
    phase_ssim: float
    phase_wrapped_mae: float


@dataclass(frozen=True)
class ValidMaskDiagnostics:
    """Collapse and decoder-head saturation evidence on valid pixels."""

    amplitude_variance: float
    phase_variance: float
    amplitude_dynamic_range: float
    phase_dynamic_range: float
    real_head_saturation_fraction: float
    imag_head_saturation_fraction: float
    real_head_lower_saturation_fraction: float
    real_head_upper_saturation_fraction: float
    imag_head_lower_saturation_fraction: float
    imag_head_upper_saturation_fraction: float


@dataclass(frozen=True)
class ScanUtilizationMetrics:
    """Unique scan accounting and reconstructed-canvas coverage."""

    unique_scans_used: int
    unique_centers_used: int | None
    unique_scans_expected: int
    scan_utilization_fraction: float
    unique_scans_filtered_eligible: int
    filtered_scan_utilization_fraction: float | None
    canvas_coverage_fraction: float


@dataclass(frozen=True)
class Bounds:
    """Half-open image bounds ``[top:bottom, left:right]``."""

    top: int
    left: int
    bottom: int
    right: int

    def __post_init__(self) -> None:
        coordinates = (self.top, self.left, self.bottom, self.right)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in coordinates
        ):
            raise AlignmentError("bounds coordinates must be integers")
        for name, value in zip(
            ("top", "left", "bottom", "right"), coordinates, strict=True
        ):
            object.__setattr__(self, name, int(value))
        if min(self.top, self.left) < 0:
            raise AlignmentError("bounds must be nonnegative")
        if self.bottom <= self.top or self.right <= self.left:
            raise AlignmentError("bounds must be nonempty")

    @property
    def row_slice(self) -> slice:
        return slice(self.top, self.bottom)

    @property
    def col_slice(self) -> slice:
        return slice(self.left, self.right)

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    def to_jsonable(self) -> dict[str, int]:
        return {
            "top": self.top,
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
        }


def _readonly_copy(array: NDArray[Any]) -> NDArray[Any]:
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class PreparedComparison:
    """Immutable aligned canvas, sampled truth, mask, and metric footprints."""

    reconstruction: NDArray[np.complexfloating[Any, Any]]
    target: NDArray[np.complexfloating[Any, Any]]
    common_mask: NDArray[np.bool_]
    ssim_bounds: Bounds
    frc_bounds: Bounds

    def __post_init__(self) -> None:
        reconstruction = np.asarray(self.reconstruction)
        target = np.asarray(self.target)
        mask = np.asarray(self.common_mask)
        if reconstruction.ndim != 2 or not np.issubdtype(
            reconstruction.dtype, np.complexfloating
        ):
            raise AlignmentError("prepared reconstruction must be a 2D complex array")
        if target.ndim != 2 or not np.issubdtype(target.dtype, np.complexfloating):
            raise AlignmentError("prepared target must be a 2D complex array")
        if reconstruction.shape != target.shape:
            raise AlignmentError("prepared reconstruction and target shapes must match")
        if mask.ndim != 2 or mask.shape != reconstruction.shape:
            raise AlignmentError("prepared common mask must match the image shape")
        if mask.dtype != np.bool_:
            raise AlignmentError("prepared common mask must have boolean dtype")
        if not isinstance(self.ssim_bounds, Bounds) or not isinstance(
            self.frc_bounds, Bounds
        ):
            raise AlignmentError("prepared footprints must use Bounds")
        height, width = reconstruction.shape
        for name, bounds in (
            ("SSIM", self.ssim_bounds),
            ("FRC", self.frc_bounds),
        ):
            if bounds.bottom > height or bounds.right > width:
                raise AlignmentError(f"prepared {name} bounds exceed the image")
        if not mask[self.ssim_bounds.row_slice, self.ssim_bounds.col_slice].all():
            raise AlignmentError("prepared SSIM rectangle must be entirely common")
        if self.frc_bounds.height != self.frc_bounds.width:
            raise AlignmentError("prepared FRC bounds must be square")
        if (
            self.frc_bounds.top < self.ssim_bounds.top
            or self.frc_bounds.left < self.ssim_bounds.left
            or self.frc_bounds.bottom > self.ssim_bounds.bottom
            or self.frc_bounds.right > self.ssim_bounds.right
        ):
            raise AlignmentError("prepared FRC bounds must be nested in SSIM bounds")
        object.__setattr__(self, "reconstruction", _readonly_copy(reconstruction))
        object.__setattr__(self, "target", _readonly_copy(target))
        object.__setattr__(self, "common_mask", _readonly_copy(mask))


@dataclass(frozen=True)
class ReconstructionEvaluationResult:
    """Paths and JSON-safe records produced by one quality evaluation."""

    metrics_path: Path
    comparison_path: Path
    metrics: Mapping[str, Any]
    metric_validity: Mapping[str, Any]
    render: Mapping[str, Any]


def _complex_2d(value: Any, *, name: str) -> NDArray[np.complexfloating[Any, Any]]:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.complexfloating):
        raise AlignmentError(f"{name} must be a 2D complex array")
    return array


def _anchor_values(
    anchor: Mapping[str, Any], canvas_shape: tuple[int, int]
) -> tuple[float, float]:
    if not isinstance(anchor, Mapping):
        raise AlignmentError("anchor must be a mapping")
    required = {"scan_com", "canvas_shape", "canvas_origin_offset"}
    if not required.issubset(anchor):
        raise AlignmentError("anchor is missing required fields")
    raw_shape = anchor["canvas_shape"]
    if not isinstance(raw_shape, (tuple, list)) or len(raw_shape) != 2:
        raise AlignmentError("anchor canvas_shape must have length 2")
    if tuple(raw_shape) != canvas_shape:
        raise AlignmentError("anchor canvas_shape must exactly match reconstruction")
    scan_com = np.asarray(anchor["scan_com"])
    origin = np.asarray(anchor["canvas_origin_offset"])
    if scan_com.size != 2 or scan_com.ndim != 1 or not np.isfinite(scan_com).all():
        raise AlignmentError("anchor scan_com must be a finite length-2 vector")
    if origin.size != 2 or origin.ndim != 1 or not np.isfinite(origin).all():
        raise AlignmentError(
            "anchor canvas_origin_offset must be a finite length-2 vector"
        )
    scan_x, scan_y = float(scan_com[0]), float(scan_com[1])
    expected_origin = np.asarray(
        [canvas_shape[1] // 2 - scan_x, canvas_shape[0] // 2 - scan_y]
    )
    if not np.allclose(origin, expected_origin, rtol=0.0, atol=1e-12):
        raise AlignmentError(
            "anchor canvas_origin_offset is incompatible with scan_com"
        )
    return scan_x, scan_y


def _truth_origin_values(anchor: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return an optional exact ``(x, y)`` source origin from an anchor."""

    if "truth_origin" not in anchor:
        return None
    raw = np.asarray(anchor["truth_origin"])
    if (
        raw.ndim != 1
        or raw.size != 2
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or not np.isfinite(raw).all()
    ):
        raise AlignmentError("anchor truth_origin must be a finite length-2 vector")
    if (
        not np.equal(raw, np.floor(raw)).all()
        or np.any(raw < 0)
    ):
        raise AlignmentError(
            "anchor truth_origin must contain nonnegative integer coordinates"
        )
    return int(raw[0]), int(raw[1])


def _bilinear_sample(
    truth: NDArray[np.complexfloating[Any, Any]],
    y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> tuple[NDArray[np.complexfloating[Any, Any]], NDArray[np.bool_]]:
    height, width = truth.shape
    in_bounds = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    safe_x = np.clip(x, 0, width - 1)
    safe_y = np.clip(y, 0, height - 1)
    x0 = np.floor(safe_x).astype(np.intp)
    y0 = np.floor(safe_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    real_dtype = np.empty((), dtype=truth.dtype).real.dtype
    dx = (safe_x - x0).astype(real_dtype, copy=False)
    dy = (safe_y - y0).astype(real_dtype, copy=False)
    one = np.asarray(1, dtype=real_dtype)
    weights = (
        (one - dx) * (one - dy),
        dx * (one - dy),
        (one - dx) * dy,
        dx * dy,
    )
    samples = (truth[y0, x0], truth[y0, x1], truth[y1, x0], truth[y1, x1])
    sampled = np.zeros(x.shape, dtype=truth.dtype)
    for weight, values in zip(weights, samples, strict=True):
        contributes = weight > 0
        np.add(sampled, weight * np.where(contributes, values, 0), out=sampled)
    return sampled, in_bounds


def largest_true_rectangle(mask: NDArray[np.bool_]) -> Bounds:
    """Return the maximum all-true rectangle with deterministic lexical ties."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise AlignmentError("common mask must be 2D")
    if not np.any(array):
        raise AlignmentError("common mask is empty")
    heights = np.zeros(array.shape[1], dtype=np.int64)
    best_key: tuple[int, int, int, int, int] | None = None
    best: Bounds | None = None
    for bottom_index, row in enumerate(array.astype(bool, copy=False)):
        heights = np.where(row, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        bottom = bottom_index + 1
        for column in range(array.shape[1] + 1):
            current = int(heights[column]) if column < array.shape[1] else 0
            start = column
            while stack and stack[-1][1] > current:
                left, height = stack.pop()
                start = left
                candidate = Bounds(bottom - height, left, bottom, column)
                key = (
                    -(candidate.height * candidate.width),
                    candidate.top,
                    candidate.left,
                    candidate.bottom,
                    candidate.right,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = candidate
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    if best is None:
        raise AlignmentError("common mask is empty")
    return best


def centered_square_bounds(rectangle: Bounds) -> Bounds:
    """Center the largest square, assigning odd excess to bottom/right trim."""
    side = min(rectangle.height, rectangle.width)
    row_trim = (rectangle.height - side) // 2
    col_trim = (rectangle.width - side) // 2
    return Bounds(
        rectangle.top + row_trim,
        rectangle.left + col_trim,
        rectangle.top + row_trim + side,
        rectangle.left + col_trim + side,
    )


def _truth_origin_slice_fits(
    origin: tuple[int, int],
    canvas_shape: tuple[int, int],
    truth_shape: tuple[int, int],
) -> bool:
    """Whether the exact canvas-sized truth slice at ``origin`` stays in bounds."""
    origin_x, origin_y = origin
    return (
        origin_y + canvas_shape[0] <= truth_shape[0]
        and origin_x + canvas_shape[1] <= truth_shape[1]
    )


def prepare_anchor_aligned(
    reconstruction: Any,
    canvas_weights: Any,
    anchor: Mapping[str, Any],
    truth: Any,
    *,
    metric_crop_border: int = 0,
) -> PreparedComparison:
    """Align truth to a reconstruction and apply the declared metric border.

    ``truth_origin`` in the anchor names the source-origin of object
    coordinates in the truth array. When the exact canvas-sized truth slice at
    that origin fits, it is used directly (``truth_origin_slice_v1``); when the
    canvas extends beyond the truth from that origin, the object-centered
    bilinear mapping (canvas pixels at ``scan_com``, offset by ``truth_origin``)
    is used instead (``truth_origin_object_centered_v1``). Without
    ``truth_origin`` the truth is sampled array-relative to ``scan_com``.
    """
    recon = _complex_2d(reconstruction, name="reconstruction")
    target_source = _complex_2d(truth, name="truth")
    weights = np.asarray(canvas_weights)
    if weights.ndim != 2 or weights.shape != recon.shape:
        raise AlignmentError("canvas_weights must be 2D and match reconstruction")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise AlignmentError("canvas_weights must be finite and nonnegative")
    canvas_shape = (int(recon.shape[0]), int(recon.shape[1]))
    scan_x, scan_y = _anchor_values(anchor, canvas_shape)
    truth_origin = _truth_origin_values(anchor)
    if truth_origin is None:
        rows, cols = np.indices(recon.shape, dtype=np.float64)
        x = cols - math.floor(recon.shape[1] / 2) + scan_x
        y = rows - math.floor(recon.shape[0] / 2) + scan_y
        sampled, valid_truth = _bilinear_sample(target_source, y, x)
    else:
        origin_x, origin_y = truth_origin
        if _truth_origin_slice_fits(
            truth_origin, recon.shape, target_source.shape
        ):
            sampled = np.asarray(
                target_source[
                    origin_y : origin_y + recon.shape[0],
                    origin_x : origin_x + recon.shape[1],
                ]
            )
            valid_truth = np.ones(recon.shape, dtype=bool)
        else:
            rows, cols = np.indices(recon.shape, dtype=np.float64)
            x = cols - math.floor(recon.shape[1] / 2) + scan_x + origin_x
            y = rows - math.floor(recon.shape[0] / 2) + scan_y + origin_y
            sampled, valid_truth = _bilinear_sample(target_source, y, x)
    common = (weights > 0) & valid_truth
    if (
        isinstance(metric_crop_border, (bool, np.bool_))
        or not isinstance(metric_crop_border, (int, np.integer))
        or int(metric_crop_border) < 0
    ):
        raise AlignmentError("metric_crop_border must be a nonnegative integer")
    metric_crop_border = int(metric_crop_border)
    if metric_crop_border:
        if 2 * metric_crop_border >= min(recon.shape):
            raise AlignmentError("metric_crop_border leaves no comparison pixels")
        border_mask = np.zeros(recon.shape, dtype=bool)
        border_mask[
            metric_crop_border : recon.shape[0] - metric_crop_border,
            metric_crop_border : recon.shape[1] - metric_crop_border,
        ] = True
        common &= border_mask
    rectangle = largest_true_rectangle(common)
    square = centered_square_bounds(rectangle)
    return PreparedComparison(
        reconstruction=recon,
        target=sampled,
        common_mask=common,
        ssim_bounds=rectangle,
        frc_bounds=square,
    )


def _masked_points(
    prepared: PreparedComparison,
) -> tuple[NDArray[Any], NDArray[Any]]:
    return (
        prepared.reconstruction[prepared.common_mask],
        prepared.target[prepared.common_mask],
    )


def _positive_denominator(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise MetricError(f"{name} denominator is nonpositive or zero")
    return value


def _rms(value: Any) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return math.nan
    if np.issubdtype(array.dtype, np.complexfloating):
        magnitudes = np.abs(array.astype(np.complex128, copy=False))
    else:
        magnitudes = np.abs(array.astype(np.float64, copy=False))
    if not np.isfinite(magnitudes).all():
        return math.nan
    scale = float(np.max(magnitudes, initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean(np.square(magnitudes / scale))))


def _scaled_mean(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return math.nan
    scale = float(np.max(np.abs(array), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.mean(array / scale))


def _amplitude(value: Any) -> NDArray[np.float64]:
    return np.abs(np.asarray(value, dtype=np.complex128)).astype(np.float64, copy=False)


def _is_scale_degenerate(spread: float, scale: float) -> bool:
    if not math.isfinite(spread) or not math.isfinite(scale):
        return True
    if spread <= 0.0:
        return True
    return bool(scale > 0.0 and spread <= _RELATIVE_DEGENERACY_TOLERANCE * scale)


def global_phase_factor(
    reconstruction: Any, target: Any, mask: Any | None = None
) -> complex:
    """Return the unit global phase multiplying reconstruction toward target."""
    recon = np.asarray(reconstruction, dtype=np.complex128)
    expected = np.asarray(target, dtype=np.complex128)
    if recon.shape != expected.shape or recon.size == 0:
        raise MetricError("global phase inputs must be equal and nonempty")
    if mask is not None:
        selected = np.asarray(mask)
        if selected.shape != recon.shape or selected.dtype != np.bool_:
            raise MetricError("global phase mask must be boolean and match inputs")
        recon = recon[selected]
        expected = expected[selected]
    if recon.size == 0:
        raise MetricError("global phase correlation has no input points")
    if not np.isfinite(recon).all() or not np.isfinite(expected).all():
        raise MetricError("global phase correlation is nonfinite or near zero")
    recon_scale = float(np.max(np.abs(recon), initial=0.0))
    target_scale = float(np.max(np.abs(expected), initial=0.0))
    if recon_scale == 0.0 or target_scale == 0.0:
        raise MetricError("global phase correlation is zero or near orthogonal")
    scaled_recon = recon / recon_scale
    scaled_target = expected / target_scale
    correlation = np.mean(np.conj(scaled_recon) * scaled_target, dtype=np.complex128)
    scale = _rms(scaled_recon) * _rms(scaled_target)
    magnitude = float(abs(correlation))
    if (
        not np.isfinite(correlation)
        or not math.isfinite(scale)
        or scale == 0.0
        or magnitude <= _RELATIVE_ORTHOGONALITY_TOLERANCE * scale
    ):
        raise MetricError("global phase correlation is nonfinite or near orthogonal")
    return complex(correlation / magnitude)


def absolute_scale_metrics(prepared: PreparedComparison) -> dict[str, float]:
    """Compute amplitude-absolute metrics with phase-only complex alignment."""
    recon, target = _masked_points(prepared)
    recon_complex = np.asarray(recon, dtype=np.complex128)
    target_complex = np.asarray(target, dtype=np.complex128)
    recon_amp = _amplitude(recon_complex)
    target_amp = _amplitude(target_complex)
    amp_rms = _positive_denominator(_rms(target_amp), name="amplitude NRMSE")
    complex_rms = _positive_denominator(_rms(target_complex), name="complex NRMSE")
    target_mean = _positive_denominator(_scaled_mean(target_amp), name="mean ratio")
    factor = global_phase_factor(recon_complex, target_complex)
    amp_error = recon_amp - target_amp
    aligned_error = factor * recon_complex - target_complex
    values = {
        "absolute_amp_mae": _scaled_mean(np.abs(amp_error)),
        "absolute_amp_nrmse": float(_rms(amp_error) / amp_rms),
        "absolute_complex_nrmse": float(_rms(aligned_error) / complex_rms),
        "amp_mean_ratio": float(_scaled_mean(recon_amp) / target_mean),
    }
    for label, quantile in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95)):
        denominator = _positive_denominator(
            float(np.quantile(target_amp, quantile)), name=f"quantile {label} ratio"
        )
        values[f"amp_quantile_ratio_{label}"] = float(
            np.quantile(recon_amp, quantile) / denominator
        )
    if not all(math.isfinite(value) for value in values.values()):
        raise MetricError("absolute metric result is nonfinite")
    return values


def _pearson(first: NDArray[Any], second: NDArray[Any]) -> float:
    if first.shape != second.shape or first.size < 2:
        raise MetricError("Pearson inputs must be equal with at least two points")
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise MetricError("Pearson inputs must be finite")
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    x_scale = float(np.max(np.abs(x), initial=0.0))
    y_scale = float(np.max(np.abs(y), initial=0.0))
    centered_x_scale = float(np.max(np.abs(centered_x), initial=0.0))
    centered_y_scale = float(np.max(np.abs(centered_y), initial=0.0))
    if _is_scale_degenerate(centered_x_scale, x_scale) or _is_scale_degenerate(
        centered_y_scale, y_scale
    ):
        raise MetricError("Pearson variance is zero or near zero")
    scaled_x = centered_x / centered_x_scale
    scaled_y = centered_y / centered_y_scale
    denominator = _rms(scaled_x) * _rms(scaled_y)
    value = float(np.mean(scaled_x * scaled_y) / denominator)
    if not math.isfinite(value):
        raise MetricError("Pearson result is nonfinite")
    return value


def amplitude_pearson(prepared: PreparedComparison) -> float:
    recon, target = _masked_points(prepared)
    return _pearson(_amplitude(recon), _amplitude(target))


def _rectangle_arrays(
    prepared: PreparedComparison,
) -> tuple[NDArray[Any], NDArray[Any]]:
    bounds = prepared.ssim_bounds
    mask = prepared.common_mask[bounds.row_slice, bounds.col_slice]
    if not mask.all():
        raise MetricError("SSIM footprint is not entirely inside the common mask")
    return (
        prepared.reconstruction[bounds.row_slice, bounds.col_slice],
        prepared.target[bounds.row_slice, bounds.col_slice],
    )


def amplitude_similarity_inputs(
    prepared: PreparedComparison,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    recon, target = _rectangle_arrays(prepared)
    prediction_amp = _amplitude(recon)
    target_amp = _amplitude(target)
    prediction_mean = float(np.mean(prediction_amp))
    if not math.isfinite(prediction_mean) or prediction_mean <= 0.0:
        raise MetricError("amplitude prediction mean is nonpositive or near zero")
    target_range = float(np.max(target_amp) - np.min(target_amp))
    target_scale = float(np.max(np.abs(target_amp), initial=0.0))
    if _is_scale_degenerate(target_range, target_scale):
        raise MetricError("amplitude target data range is nonpositive or near zero")
    target_mean = float(np.mean(target_amp))
    scaled = prediction_amp * (target_mean / prediction_mean)
    return scaled, target_amp, target_range


def _ssim(
    prediction: NDArray[Any],
    target: NDArray[Any],
    *,
    data_range: float,
    win_size: int = 7,
) -> float:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise MetricError("SSIM inputs must be equal 2D arrays")
    if min(prediction.shape) < win_size:
        raise MetricError("SSIM footprint is too small for its window")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise MetricError("SSIM inputs must be finite")
    input_scale = float(
        max(
            np.max(np.abs(prediction), initial=0.0),
            np.max(np.abs(target), initial=0.0),
        )
    )
    if _is_scale_degenerate(data_range, input_scale):
        raise MetricError("SSIM data range is nonpositive or scale-degenerate")
    try:
        value = float(
            _structural_similarity(
                prediction,
                target,
                data_range=data_range,
                win_size=win_size,
            )
        )
    except (TypeError, ValueError) as error:
        raise MetricError(f"SSIM evaluation failed: {error}") from error
    if not math.isfinite(value):
        raise MetricError("SSIM result is nonfinite")
    return value


def amplitude_ssim(prepared: PreparedComparison) -> float:
    prediction, target, data_range = amplitude_similarity_inputs(prepared)
    return _ssim(prediction, target, data_range=data_range)


def _aligned_phase_arrays(
    prepared: PreparedComparison,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    recon_points, target_points = _masked_points(prepared)
    factor = global_phase_factor(recon_points, target_points)
    return (
        np.angle(factor * prepared.reconstruction),
        np.angle(prepared.target),
    )


def phase_wrapped_mae(prepared: PreparedComparison) -> float:
    prediction_phase, target_phase = _aligned_phase_arrays(prepared)
    residual = np.angle(np.exp(1j * (prediction_phase - target_phase)))
    value = float(np.mean(np.abs(residual[prepared.common_mask])))
    if not math.isfinite(value):
        raise MetricError("wrapped phase MAE result is nonfinite")
    return value


def phase_similarity_inputs(
    prepared: PreparedComparison,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prediction_phase, target_phase = _aligned_phase_arrays(prepared)
    bounds = prepared.ssim_bounds
    prediction = (prediction_phase[bounds.row_slice, bounds.col_slice] + np.pi) / (
        2 * np.pi
    )
    target = (target_phase[bounds.row_slice, bounds.col_slice] + np.pi) / (2 * np.pi)
    return prediction, target


def phase_ssim(prepared: PreparedComparison) -> float:
    prediction, target = phase_similarity_inputs(prepared)
    return _ssim(prediction, target, data_range=1.0)


def image_quality_metrics(prepared: PreparedComparison) -> ImageQualityMetrics:
    """Return typed amplitude and phase recognizability metrics."""
    try:
        amp_pearson = amplitude_pearson(prepared)
    except MetricError as error:
        if "variance is zero or near zero" not in str(error):
            raise
        amp_pearson = -1.0
    return ImageQualityMetrics(
        amplitude_ssim=amplitude_ssim(prepared),
        amplitude_pearson=amp_pearson,
        phase_ssim=phase_ssim(prepared),
        phase_wrapped_mae=phase_wrapped_mae(prepared),
    )


def _validated_complex_mask_points(
    value: Any, mask: Any, *, name: str
) -> NDArray[np.complex128]:
    array = np.asarray(value, dtype=np.complex128)
    selected = np.asarray(mask)
    if array.ndim != 2 or selected.shape != array.shape or selected.dtype != np.bool_:
        raise MetricError(f"{name} and valid mask must be equal 2D arrays")
    points = array[selected]
    if points.size == 0 or not np.isfinite(points).all():
        raise MetricError(f"{name} valid points must be nonempty and finite")
    return points


def valid_mask_diagnostics(
    reconstruction: Any,
    valid_mask: Any,
    *,
    decoder_output: Any | None = None,
    real_bounds: tuple[float, float] = (-0.8, 1.2),
    imag_bounds: tuple[float, float] = (-1.2, 1.2),
    saturation_tolerance: float = 1e-3,
) -> ValidMaskDiagnostics:
    """Measure texture collapse and rectangular-head saturation on valid pixels."""
    for name, bounds in (("real", real_bounds), ("imag", imag_bounds)):
        if (
            len(bounds) != 2
            or not all(math.isfinite(float(item)) for item in bounds)
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise MetricError(f"{name} saturation bounds are invalid")
    if not math.isfinite(saturation_tolerance) or not 0.0 <= saturation_tolerance < 0.5:
        raise MetricError("saturation tolerance is invalid")
    points = _validated_complex_mask_points(
        reconstruction, valid_mask, name="reconstruction"
    )
    amplitude = np.abs(points)
    phase = np.angle(points)
    phase_center = np.angle(np.mean(np.exp(1j * phase)))
    centered_phase = np.angle(np.exp(1j * (phase - phase_center)))
    head_points = (
        points
        if decoder_output is None
        else _validated_complex_mask_points(
            decoder_output, valid_mask, name="decoder output"
        )
    )
    real_tolerance = (real_bounds[1] - real_bounds[0]) * saturation_tolerance
    imag_tolerance = (imag_bounds[1] - imag_bounds[0]) * saturation_tolerance
    real_lower = float(np.mean(head_points.real <= real_bounds[0] + real_tolerance))
    real_upper = float(np.mean(head_points.real >= real_bounds[1] - real_tolerance))
    imag_lower = float(np.mean(head_points.imag <= imag_bounds[0] + imag_tolerance))
    imag_upper = float(np.mean(head_points.imag >= imag_bounds[1] - imag_tolerance))
    values = ValidMaskDiagnostics(
        amplitude_variance=float(np.var(amplitude)),
        phase_variance=float(np.var(centered_phase)),
        amplitude_dynamic_range=float(
            np.quantile(amplitude, 0.95) - np.quantile(amplitude, 0.05)
        ),
        phase_dynamic_range=float(
            np.quantile(centered_phase, 0.95) - np.quantile(centered_phase, 0.05)
        ),
        real_head_saturation_fraction=real_lower + real_upper,
        imag_head_saturation_fraction=imag_lower + imag_upper,
        real_head_lower_saturation_fraction=real_lower,
        real_head_upper_saturation_fraction=real_upper,
        imag_head_lower_saturation_fraction=imag_lower,
        imag_head_upper_saturation_fraction=imag_upper,
    )
    if not all(math.isfinite(value) for value in values.__dict__.values()):
        raise MetricError("valid-mask diagnostics are nonfinite")
    return values


def scan_utilization_metrics(
    used_scan_ids: Iterable[Any],
    used_center_scan_ids: Iterable[Any] | None,
    expected_scan_ids: Iterable[Any],
    filtered_eligible_scan_ids: Iterable[Any],
    canvas_weights: Any,
) -> ScanUtilizationMetrics:
    """Account for unique scans and positive-weight canvas pixels."""
    used = frozenset(used_scan_ids)
    centers = None if used_center_scan_ids is None else frozenset(used_center_scan_ids)
    expected = frozenset(expected_scan_ids)
    filtered = frozenset(filtered_eligible_scan_ids)
    if (
        not expected
        or not filtered
        or (centers is not None and not centers.issubset(filtered))
        or not used.issubset(expected)
        or not filtered.issubset(expected)
    ):
        raise MetricError(
            "used centers must be within filtered scans and participants within source scans"
        )
    weights = np.asarray(canvas_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0 or not np.isfinite(weights).all():
        raise MetricError("canvas weights must be a finite nonempty 2D array")
    if np.any(weights < 0.0):
        raise MetricError("canvas weights must be nonnegative")
    return ScanUtilizationMetrics(
        unique_scans_used=len(used),
        unique_centers_used=None if centers is None else len(centers),
        unique_scans_expected=len(expected),
        scan_utilization_fraction=float(len(used) / len(expected)),
        unique_scans_filtered_eligible=len(filtered),
        filtered_scan_utilization_fraction=(
            None if centers is None else float(len(centers) / len(filtered))
        ),
        canvas_coverage_fraction=float(np.mean(weights > 0.0)),
    )


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise MetricError(f"reassembly evidence is missing {name}")
        return record[name]
    if not hasattr(record, name):
        raise MetricError(f"reassembly evidence is missing {name}")
    return getattr(record, name)


def _count_diagnostics_are_legacy_not_applicable(record: Any) -> bool:
    status = (
        record.get("status")
        if isinstance(record, Mapping)
        else getattr(record, "status", None)
    )
    reason = (
        record.get("reason")
        if isinstance(record, Mapping)
        else getattr(record, "reason", None)
    )
    return status == "not_applicable" and reason == "legacy_normalized_amplitude"


def _count_diagnostics_record(record: Any) -> dict[str, Any]:
    """Copy one validated count-diagnostics record into JSON-ready evidence."""

    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "to_jsonable"):
        value = record.to_jsonable()
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(record, "__dataclass_fields__"):
        return asdict(record)
    return {
        name: getattr(record, name)
        for name in _FITTED_COUNT_FIELDS
        if hasattr(record, name)
    }


_FITTED_COUNT_FIELDS = (
    "relative_l2_intensity_error",
    "mean_raw_poisson_nll",
    "n_samples",
    "n_pixels",
    "effective_mask_digest",
    "sample_ids",
    "sample_identity_digest",
)


def _validate_fitted_count_diagnostics(
    record: Any,
    *,
    expected_sample_ids: Any,
    expected_mask_digest: Any,
) -> Any:
    """Require real fitted count evidence under the CI count-intensity profile.

    ``FittedCountMetrics`` carries no ``status`` field, so the legacy marker
    check cannot be reused.  A ``not_applicable``/``not_evaluated`` marker here
    would mean the CI diagnostics silently never ran, which must fail closed.
    """

    def value(name: str) -> Any:
        if isinstance(record, Mapping):
            if name not in record:
                raise MetricError(
                    f"count_intensity diagnostics are missing {name}; the CI "
                    "count metrics did not run"
                )
            return record[name]
        if not hasattr(record, name):
            raise MetricError(
                f"count_intensity diagnostics are missing {name}; the CI "
                "count metrics did not run"
            )
        return getattr(record, name)

    from ptycho_torch.reassembly_diagnostics import FittedCountMetrics

    try:
        fitted = FittedCountMetrics(
            **{name: value(name) for name in _FITTED_COUNT_FIELDS}
        )
    except (TypeError, ValueError) as error:
        raise MetricError(f"invalid count_intensity diagnostics: {error}") from error
    if fitted.relative_l2_intensity_error < 0.0:
        raise MetricError(
            "count_intensity diagnostics relative_l2_intensity_error must be "
            "nonnegative"
        )
    if (
        len(fitted.effective_mask_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in fitted.effective_mask_digest.lower()
        )
    ):
        raise MetricError(
            "count_intensity diagnostics effective_mask_digest must be SHA-256"
        )
    if (
        not isinstance(expected_mask_digest, str)
        or len(expected_mask_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_mask_digest.lower()
        )
    ):
        raise MetricError("reassembly mask_digest must be SHA-256")
    if fitted.effective_mask_digest.lower() != expected_mask_digest.lower():
        raise MetricError(
            "count_intensity diagnostics effective_mask_digest must match "
            "reassembly mask_digest"
        )
    expected_ids = tuple(
        int(item) for item in np.asarray(expected_sample_ids).reshape(-1)
    )
    if fitted.n_samples != len(expected_ids) or sorted(fitted.sample_ids) != sorted(
        expected_ids
    ):
        raise MetricError(
            "count_intensity diagnostics sample_ids must match channel_indices"
        )
    if fitted.n_pixels <= 0 or fitted.n_pixels % fitted.n_samples:
        raise MetricError(
            "count_intensity diagnostics n_pixels must be a positive whole "
            "detector-frame multiple"
        )
    return fitted


def _validate_channel_indices(
    channel_indices: Any,
    *,
    expected_channels: int = 4,
) -> tuple[dict[str, Any], NDArray[np.int64]]:
    if (
        isinstance(expected_channels, bool)
        or not isinstance(expected_channels, int)
        or expected_channels <= 0
    ):
        raise MetricError("expected_channels must be a positive integer")
    channel_label = f"C{expected_channels}"
    rows = np.asarray(channel_indices)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != expected_channels:
        raise MetricError(
            f"channel_indices must contain nonempty {channel_label} group rows"
        )
    if not np.issubdtype(rows.dtype, np.integer):
        if not np.issubdtype(rows.dtype, np.number) or not np.isfinite(rows).all():
            raise MetricError("channel_indices must contain finite integers")
        if not np.equal(rows, np.floor(rows)).all():
            raise MetricError("channel_indices must contain finite integers")
    normalized = rows.astype(np.int64, copy=False)
    if np.any(normalized < 0):
        raise MetricError("channel_indices must be nonnegative")
    if any(len(set(row.tolist())) != expected_channels for row in normalized):
        raise MetricError(
            f"every {channel_label} group must contain "
            f"{expected_channels} distinct channel indices"
        )
    return (
        {
            "group_count": int(normalized.shape[0]),
            "channel_count": expected_channels,
            "all_groups_distinct": True,
        },
        normalized,
    )


def _four_metrics(prepared: PreparedComparison) -> dict[str, float]:
    quality = image_quality_metrics(prepared)
    recon, target = _masked_points(prepared)
    amp_mae = _scaled_mean(np.abs(_amplitude(recon) - _amplitude(target)))
    values = {
        "amplitude_ssim": float(quality.amplitude_ssim),
        "phase_ssim": float(quality.phase_ssim),
        "absolute_amp_mae": float(amp_mae),
        "phase_wrapped_mae": float(quality.phase_wrapped_mae),
    }
    for name in ("amplitude_ssim", "phase_ssim"):
        value = values[name]
        if not math.isfinite(value) or not -1.0 - 1e-12 <= value <= 1.0 + 1e-12:
            raise MetricError(f"{name} must be finite and in [-1, 1]")
        values[name] = float(np.clip(value, -1.0, 1.0))
    for name in ("absolute_amp_mae", "phase_wrapped_mae"):
        if not math.isfinite(values[name]) or values[name] < 0.0:
            raise MetricError(f"{name} must be finite and nonnegative")
    return values


def _prescale_diagnostic_metrics(
    prepared: PreparedComparison,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return finite prescale diagnostics without turning them into a gate."""
    recon, target = _masked_points(prepared)
    values = {
        "absolute_amp_mae": float(
            _scaled_mean(np.abs(_amplitude(recon) - _amplitude(target)))
        )
    }
    undefined: dict[str, str] = {}
    try:
        values["amplitude_ssim"] = float(amplitude_ssim(prepared))
    except MetricError as error:
        values["amplitude_ssim"] = -1.0
        undefined["amplitude_ssim"] = str(error)
    try:
        values["phase_ssim"] = float(phase_ssim(prepared))
    except MetricError as error:
        values["phase_ssim"] = -1.0
        undefined["phase_ssim"] = str(error)
    try:
        values["phase_wrapped_mae"] = float(phase_wrapped_mae(prepared))
    except MetricError as error:
        values["phase_wrapped_mae"] = math.pi
        undefined["phase_wrapped_mae"] = str(error)
    ordered = {
        name: values[name]
        for name in (
            "amplitude_ssim",
            "phase_ssim",
            "absolute_amp_mae",
            "phase_wrapped_mae",
        )
    }
    if not all(math.isfinite(value) for value in ordered.values()):
        raise MetricError("prescale diagnostic metrics must be finite")
    status: dict[str, Any] = {
        "status": "complete" if not undefined else "partial_sentinel",
        "undefined": undefined,
        "sentinel_policy": {
            "undefined_ssim": -1.0,
            "undefined_wrapped_phase_mae": math.pi,
        },
    }
    return ordered, status


def _validate_raw_inputs(
    complex_canvas: Any,
    prescale_canvas: Any,
    canvas_weights: Any,
    truth: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    canvas = _complex_2d(complex_canvas, name="complex_canvas")
    prescale = _complex_2d(prescale_canvas, name="prescale_canvas")
    target = _complex_2d(truth, name="truth")
    weights = np.asarray(canvas_weights)
    if prescale.shape != canvas.shape:
        raise MetricError("prescale_canvas shape must match complex_canvas")
    if weights.ndim != 2 or weights.shape != canvas.shape:
        raise MetricError("canvas_weights shape must match complex_canvas")
    for name, value in (
        ("complex_canvas", canvas),
        ("prescale_canvas", prescale),
        ("truth", target),
        ("canvas_weights", weights),
    ):
        if not np.isfinite(value).all():
            raise MetricError(f"{name} must contain only finite values")
    if np.any(weights < 0):
        raise MetricError("canvas_weights must be nonnegative")
    if not np.any(weights > 0):
        raise MetricError("canvas_weights must contain positive support")
    return canvas, prescale, weights, target


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _restore_artifact(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _write_bytes_atomic(path, previous)


def render_comparison(
    prepared: PreparedComparison,
    output_path: str | os.PathLike[str],
    *,
    gauge_factor: complex | None = None,
) -> dict[str, Any]:
    """Render the fixed six-panel diagnostic from one prepared comparison."""
    if not isinstance(prepared, PreparedComparison):
        raise TypeError("prepared must be a PreparedComparison")
    factor = (
        global_phase_factor(
            prepared.reconstruction,
            prepared.target,
            prepared.common_mask,
        )
        if gauge_factor is None
        else complex(gauge_factor)
    )
    if not math.isfinite(factor.real) or not math.isfinite(factor.imag):
        raise MetricError("gauge_factor must be finite")
    if not math.isclose(abs(factor), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise MetricError("gauge_factor must have unit magnitude")

    mask = prepared.common_mask
    aligned = factor * prepared.reconstruction
    truth_amp = np.abs(prepared.target)
    recon_amp = np.abs(aligned)
    amp_error = np.abs(recon_amp - truth_amp)
    truth_phase = np.angle(prepared.target)
    recon_phase = np.angle(aligned)
    phase_error = np.angle(np.exp(1j * (recon_phase - truth_phase)))
    amp_vmax = float(max(np.max(truth_amp[mask]), np.max(recon_amp[mask])))
    amp_error_vmax = float(np.max(amp_error[mask]))
    if amp_vmax <= 0.0 or not math.isfinite(amp_vmax):
        raise MetricError("render amplitude scale must be finite and positive")
    if amp_error_vmax <= 0.0:
        amp_error_vmax = float(np.finfo(np.float64).eps * amp_vmax)

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)
    temporary_path: Path | None = None
    try:
        panels = (
            (truth_amp, "Amplitude truth", "viridis", 0.0, amp_vmax),
            (recon_amp, "Amplitude reconstruction", "viridis", 0.0, amp_vmax),
            (amp_error, "Amplitude absolute error", "magma", 0.0, amp_error_vmax),
            (truth_phase, "Phase truth", "twilight", -math.pi, math.pi),
            (recon_phase, "Phase reconstruction", "twilight", -math.pi, math.pi),
            (phase_error, "Phase wrapped error", "twilight", -math.pi, math.pi),
        )
        for axis, (array, title, cmap, vmin, vmax) in zip(
            axes.flat, panels, strict=True
        ):
            masked = np.ma.array(array, mask=~mask)
            image = axis.imshow(
                masked,
                origin="upper",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.canvas.draw()
        panel_bounds: dict[str, dict[str, int]] = {}
        for panel_name, axis in zip(PANEL_NAMES, axes.flat, strict=True):
            bounds = axis.get_window_extent()
            panel_bounds[panel_name] = {
                "left": max(0, int(math.floor(bounds.x0))),
                "top": max(0, PNG_HEIGHT - int(math.ceil(bounds.y1))),
                "right": min(PNG_WIDTH, int(math.ceil(bounds.x1))),
                "bottom": min(
                    PNG_HEIGHT,
                    PNG_HEIGHT - int(math.floor(bounds.y0)),
                ),
            }

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".png", dir=output.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        figure.savefig(temporary_path, dpi=150, format="png")
        from PIL import Image

        with Image.open(temporary_path) as image:
            if image.size != (PNG_WIDTH, PNG_HEIGHT):
                raise MetricError(
                    "comparison PNG dimensions disagree with renderer contract"
                )
            pixels = np.asarray(image)
        if not np.isfinite(pixels).all() or float(np.var(pixels)) <= 0.0:
            raise MetricError("comparison PNG is blank or nonfinite")
        for panel_name in PANEL_NAMES:
            bounds = panel_bounds[panel_name]
            region = pixels[
                bounds["top"] : bounds["bottom"],
                bounds["left"] : bounds["right"],
            ]
            if (
                region.size == 0
                or not np.isfinite(region).all()
                or not np.any(region[..., :3] < 250)
            ):
                raise MetricError(
                    f"comparison PNG is missing rendered panel {panel_name}"
                )
        os.replace(temporary_path, output)
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "renderer_version": RENDERER_VERSION,
        "aligned_shape": [int(size) for size in prepared.reconstruction.shape],
        "valid_pixel_count": int(np.count_nonzero(mask)),
        "panels": list(PANEL_NAMES),
        "panel_bounds": panel_bounds,
        "color_limits": {
            "amplitude": [0.0, amp_vmax],
            "amplitude_absolute_error": [0.0, amp_error_vmax],
            "phase": [-math.pi, math.pi],
            "phase_wrapped_error": [-math.pi, math.pi],
        },
        "png_width": PNG_WIDTH,
        "png_height": PNG_HEIGHT,
    }


def evaluate_reconstruction_quality(
    complex_canvas: Any,
    prescale_canvas: Any,
    canvas_weights: Any,
    canvas_anchor: Mapping[str, Any],
    truth: Any,
    reassembly: Any,
    channel_indices: Any,
    groups_per_center: int,
    output_dir: str | os.PathLike[str],
    expected_channels: int = 4,
    measurement_domain: str = "normalized_amplitude",
    metric_crop_border: int = 0,
) -> ReconstructionEvaluationResult:
    """Score raw arrays, render diagnostics, and publish two stage artifacts."""
    if measurement_domain not in {"normalized_amplitude", "count_intensity"}:
        raise MetricError(
            "measurement_domain must be 'normalized_amplitude' or "
            f"'count_intensity'; got {measurement_domain!r}"
        )
    if isinstance(groups_per_center, bool) or not isinstance(groups_per_center, int):
        raise MetricError("groups_per_center must be a positive integer")
    if groups_per_center <= 0:
        raise MetricError("groups_per_center must be a positive integer")
    canvas, prescale, weights, target = _validate_raw_inputs(
        complex_canvas,
        prescale_canvas,
        canvas_weights,
        truth,
    )
    channel_record, normalized_channel_indices = _validate_channel_indices(
        channel_indices,
        expected_channels=expected_channels,
    )
    accepted = int(_record_field(reassembly, "accepted_patches"))
    total = int(_record_field(reassembly, "total_patches"))
    if accepted <= 0 or accepted != total:
        raise MetricError("patch counts must satisfy accepted == total > 0")
    used_scan_ids = tuple(_record_field(reassembly, "used_scan_ids"))
    if normalized_channel_indices.size != total or set(
        normalized_channel_indices.reshape(-1).tolist()
    ) != set(used_scan_ids):
        raise MetricError(
            "channel_indices must match every accepted reassembly scan id"
        )
    s1 = float(_record_field(reassembly, "s1"))
    s2 = float(_record_field(reassembly, "s2"))
    if not math.isfinite(s1) or not math.isfinite(s2):
        raise MetricError("VarPro s1/s2 must be finite")
    if _record_field(reassembly, "effective_precision") != "32-true":
        raise MetricError("quality evaluation requires effective precision 32-true")
    count_record = _record_field(reassembly, "count_metrics")
    assembly_method = canvas_anchor.get("assembly_method")
    validated_count_record = count_record
    if measurement_domain == "count_intensity":
        validated_count_record = _validate_fitted_count_diagnostics(
            count_record,
            expected_sample_ids=normalized_channel_indices,
            expected_mask_digest=_record_field(reassembly, "mask_digest"),
        )
    elif not _count_diagnostics_are_legacy_not_applicable(count_record):
        raise MetricError("legacy count diagnostics must be explicitly not_applicable")

    prepared = prepare_anchor_aligned(
        canvas,
        weights,
        canvas_anchor,
        target,
        metric_crop_border=metric_crop_border,
    )
    prescale_prepared = prepare_anchor_aligned(
        prescale,
        weights,
        canvas_anchor,
        target,
        metric_crop_border=metric_crop_border,
    )
    if min(prepared.ssim_bounds.height, prepared.ssim_bounds.width) < 7:
        raise MetricError("SSIM rectangle must be at least 7-by-7")
    collapse = valid_mask_diagnostics(canvas, prepared.common_mask)
    if collapse.amplitude_variance <= 0.0:
        raise MetricError("reconstruction amplitude variance must be positive")
    if collapse.amplitude_dynamic_range <= 0.0:
        raise MetricError("reconstruction amplitude dynamic range must be positive")
    utilization = scan_utilization_metrics(
        used_scan_ids,
        _record_field(reassembly, "used_center_scan_ids"),
        _record_field(reassembly, "expected_scan_ids"),
        _record_field(reassembly, "filtered_eligible_scan_ids"),
        weights,
    )
    if groups_per_center == 1 and (
        utilization.scan_utilization_fraction != 1.0
        or utilization.filtered_scan_utilization_fraction != 1.0
    ):
        raise MetricError(
            "groups_per_center=1 requires complete scan and center utilization"
        )

    post_metrics = _four_metrics(prepared)
    # Prescale values are diagnostic-only: they must have a finite JSON
    # representation, but undefined components use declared sentinels rather
    # than passing through collapse, utilization, or quality gates.
    prescale_metrics, prescale_status = _prescale_diagnostic_metrics(prescale_prepared)
    factor = global_phase_factor(
        prepared.reconstruction,
        prepared.target,
        prepared.common_mask,
    )
    valid_pixel_count = int(np.count_nonzero(prepared.common_mask))
    truth_origin = _truth_origin_values(canvas_anchor)
    if truth_origin is None:
        alignment_method = "scan_com_bilinear_complex_v1"
    elif _truth_origin_slice_fits(
        truth_origin, prepared.reconstruction.shape, target.shape
    ):
        alignment_method = "truth_origin_slice_v1"
    else:
        alignment_method = "truth_origin_object_centered_v1"
    alignment = {
        "method": alignment_method,
        "translation_registration": "none",
        "object_center_crop": False,
        "valid_mask_policy": (
            "positive_weights_and_in_bounds_truth_then_symmetric_metric_crop"
            if metric_crop_border
            else "positive_weights_and_in_bounds_truth"
        ),
        "aligned_shape": [int(size) for size in prepared.reconstruction.shape],
        "valid_pixel_count": valid_pixel_count,
        "metric_crop_border": int(metric_crop_border),
        "ssim_bounds": prepared.ssim_bounds.to_jsonable(),
        "frc_bounds": prepared.frc_bounds.to_jsonable(),
        "canvas_anchor": {
            "scan_com": [float(item) for item in canvas_anchor["scan_com"]],
            "canvas_shape": [int(item) for item in canvas_anchor["canvas_shape"]],
            "canvas_origin_offset": [
                float(item) for item in canvas_anchor["canvas_origin_offset"]
            ],
        },
        "truth_shape": [int(size) for size in target.shape],
    }
    if truth_origin is not None:
        alignment["truth_origin"] = [int(item) for item in truth_origin]
    gauge = {
        "method": "unit_global_complex_phase",
        "real": float(factor.real),
        "imag": float(factor.imag),
        "magnitude": float(abs(factor)),
    }
    metrics: dict[str, Any] = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        **post_metrics,
        "valid_pixel_count": valid_pixel_count,
        "alignment": alignment,
        "gauge_factor": gauge,
    }
    metric_validity: dict[str, Any] = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "valid": True,
        "finite_raw_arrays": True,
        "valid_pixel_count": valid_pixel_count,
        "post_varpro_metrics": dict(post_metrics),
        "prescale_metrics": dict(prescale_metrics),
        "prescale_metrics_status": prescale_status,
        "prescale_role": "diagnostic_only",
        "quality_gate_canvas": (
            "tiled_source_object_gauge"
            if assembly_method == "tiled_raster_v1"
            else "post_varpro"
        ),
        "valid_mask": asdict(collapse),
        "scan_utilization": asdict(utilization),
        "channel_groups": channel_record,
        "patches_accepted": accepted,
        "patches_total": total,
        "groups_per_center": groups_per_center,
        "effective_precision": "32-true",
        "varpro": {"s1": s1, "s2": s2},
        "count_diagnostics": _count_diagnostics_record(validated_count_record),
    }
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.json"
    comparison_path = output_root / "comparison.png"
    previous_metrics = metrics_path.read_bytes() if metrics_path.is_file() else None
    previous_comparison = (
        comparison_path.read_bytes() if comparison_path.is_file() else None
    )
    with tempfile.TemporaryDirectory(
        prefix=".evaluation-stage-",
        dir=output_root,
    ) as staging_name:
        staging_root = Path(staging_name)
        staged_metrics = staging_root / "metrics.json"
        staged_comparison = staging_root / "comparison.png"
        _write_json_atomic(staged_metrics, metrics)
        render = render_comparison(
            prepared,
            staged_comparison,
            gauge_factor=factor,
        )
        try:
            os.replace(staged_metrics, metrics_path)
            os.replace(staged_comparison, comparison_path)
        except BaseException:
            _restore_artifact(metrics_path, previous_metrics)
            _restore_artifact(comparison_path, previous_comparison)
            raise
    return ReconstructionEvaluationResult(
        metrics_path=metrics_path,
        comparison_path=comparison_path,
        metrics=metrics,
        metric_validity=metric_validity,
        render=render,
    )


__all__ = [
    "AlignmentError",
    "Bounds",
    "ImageQualityMetrics",
    "METRIC_CONTRACT_VERSION",
    "MetricError",
    "PANEL_NAMES",
    "PNG_HEIGHT",
    "PNG_WIDTH",
    "PreparedComparison",
    "RENDERER_VERSION",
    "ReconstructionEvaluationResult",
    "ScanUtilizationMetrics",
    "ValidMaskDiagnostics",
    "absolute_scale_metrics",
    "amplitude_pearson",
    "amplitude_similarity_inputs",
    "amplitude_ssim",
    "centered_square_bounds",
    "evaluate_reconstruction_quality",
    "global_phase_factor",
    "image_quality_metrics",
    "largest_true_rectangle",
    "phase_similarity_inputs",
    "phase_ssim",
    "phase_wrapped_mae",
    "prepare_anchor_aligned",
    "render_comparison",
    "scan_utilization_metrics",
    "valid_mask_diagnostics",
]

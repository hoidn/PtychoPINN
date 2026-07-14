"""Anchor-aware metrics for manifest-driven ablation studies."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, gaussian_filter  # type: ignore[import-untyped]
from skimage.metrics import structural_similarity as _structural_similarity

from . import manifest as _manifest


_RELATIVE_ORTHOGONALITY_TOLERANCE = 1e-12
_RELATIVE_DEGENERACY_TOLERANCE = 8.0 * np.finfo(np.float64).eps
_CORRELATION_DOMAIN_TOLERANCE = 1e-12
_MS_SSIM_WEIGHTS = np.asarray(
    [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64
)
METRIC_PATHS = _manifest.METRIC_PATHS


class MetricError(ValueError):
    """Raised when a metric cannot be evaluated under its declared policy."""


class AlignmentError(MetricError):
    """Raised when reconstruction and target coordinates cannot be aligned."""


class RegistryError(MetricError):
    """Raised when a metric record violates the closed registry contract."""


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Recognizability metrics for one declared reconstruction stage."""

    amplitude_ssim: float
    amplitude_pearson: float
    phase_ssim: float
    phase_wrapped_mae: float


@dataclass(frozen=True)
class VarProQualityMetrics:
    """Structural quality immediately before and after VarPro scaling."""

    before: ImageQualityMetrics
    after: ImageQualityMetrics


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
class ConvergenceMetrics:
    """Tail behavior used to reject improving budget-boundary checkpoints."""

    tail_relative_improvement: float
    normalized_tail_slope: float
    budget_boundary_improving: float


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
class PoissonOracleMetrics:
    """Model count error relative to the truth-forward Poisson noise floor."""

    oracle_relative_l2_error: float
    model_relative_l2_error: float
    model_to_oracle_error_ratio: float


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


@dataclass(frozen=True)
class PreparedComparison:
    """Immutable aligned canvas, sampled target, mask, and metric footprints."""

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


def _readonly_copy(array: NDArray[Any]) -> NDArray[Any]:
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


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


def prepare_anchor_aligned(
    reconstruction: Any,
    canvas_weights: Any,
    anchor: Mapping[str, Any],
    truth: Any,
) -> PreparedComparison:
    """Sample truth into the canonical scan-COM canvas without cropping."""
    recon = _complex_2d(reconstruction, name="reconstruction")
    target_source = _complex_2d(truth, name="truth")
    weights = np.asarray(canvas_weights)
    if weights.ndim != 2 or weights.shape != recon.shape:
        raise AlignmentError("canvas_weights must be 2D and match reconstruction")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise AlignmentError("canvas_weights must be finite and nonnegative")
    canvas_shape = (int(recon.shape[0]), int(recon.shape[1]))
    scan_x, scan_y = _anchor_values(anchor, canvas_shape)

    rows, cols = np.indices(recon.shape, dtype=np.float64)
    x = cols - math.floor(recon.shape[1] / 2) + scan_x
    y = rows - math.floor(recon.shape[0] / 2) + scan_y
    sampled, valid_truth = _bilinear_sample(target_source, y, x)
    common = (weights > 0) & valid_truth
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
    """Compute amplitude-absolute metrics with global-phase-only complex alignment."""
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


def patch_amplitude_pearson(reconstruction: Any, target: Any) -> float:
    """Compute the separate object-frame patch amplitude diagnostic."""
    recon = np.asarray(reconstruction)
    expected = np.asarray(target)
    if recon.ndim != 2 or expected.ndim != 2 or recon.shape != expected.shape:
        raise MetricError("patch inputs must be equal 2D arrays")
    return _pearson(_amplitude(recon).ravel(), _amplitude(expected).ravel())


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
                prediction, target, data_range=data_range, win_size=win_size
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


def _average_pool_2x2(image: Any) -> NDArray[np.float64]:
    """Average nonoverlapping 2x2 cells, dropping odd bottom/right edges."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise MetricError("MS-SSIM pooling input must be 2D")
    even_height = array.shape[0] - array.shape[0] % 2
    even_width = array.shape[1] - array.shape[1] % 2
    if even_height == 0 or even_width == 0:
        raise MetricError("MS-SSIM pooling input is too small")
    trimmed = array[:even_height, :even_width]
    return trimmed.reshape(even_height // 2, 2, even_width // 2, 2).mean(axis=(1, 3))


def _gaussian_statistics_kernel(win_size: int) -> NDArray[np.float64]:
    if win_size < 3 or win_size % 2 == 0:
        raise MetricError("MS-SSIM window size must be odd and at least 3")
    radius = win_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    one_dimensional = np.exp(-0.5 * np.square(coordinates / 1.5))
    one_dimensional /= np.sum(one_dimensional)
    return np.outer(one_dimensional, one_dimensional)


def _ms_ssim_components(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    kernel: NDArray[np.float64],
) -> tuple[float, float]:
    """Return mean full SSIM and contrast-structure using standard K constants."""
    prediction_mean = convolve(prediction, kernel, mode="reflect")
    target_mean = convolve(target, kernel, mode="reflect")
    prediction_second = convolve(np.square(prediction), kernel, mode="reflect")
    target_second = convolve(np.square(target), kernel, mode="reflect")
    cross_second = convolve(prediction * target, kernel, mode="reflect")

    prediction_variance = prediction_second - np.square(prediction_mean)
    target_variance = target_second - np.square(target_mean)
    covariance = cross_second - prediction_mean * target_mean
    variance_scale = max(
        float(np.max(np.abs(prediction_second), initial=0.0)),
        float(np.max(np.abs(target_second), initial=0.0)),
        1.0,
    )
    variance_tolerance = 16.0 * np.finfo(np.float64).eps * variance_scale
    if np.any(prediction_variance < -variance_tolerance) or np.any(
        target_variance < -variance_tolerance
    ):
        raise MetricError("MS-SSIM local variance is materially negative")
    prediction_variance = np.where(prediction_variance < 0.0, 0.0, prediction_variance)
    target_variance = np.where(target_variance < 0.0, 0.0, target_variance)

    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (2.0 * prediction_mean * target_mean + c1) / (
        np.square(prediction_mean) + np.square(target_mean) + c1
    )
    contrast_structure = (2.0 * covariance + c2) / (
        prediction_variance + target_variance + c2
    )
    full_ssim = float(np.mean(luminance * contrast_structure))
    mean_contrast_structure = float(np.mean(contrast_structure))
    if not math.isfinite(full_ssim) or not math.isfinite(mean_contrast_structure):
        raise MetricError("MS-SSIM component mean is nonfinite")
    return full_ssim, mean_contrast_structure


def multiscale_ssim(
    prediction: Any,
    target: Any,
    *,
    data_range: float,
    sigma: float,
    levels: int = 5,
    win_size: int = 7,
) -> float:
    """Component MS-SSIM with Gaussian statistics and anti-aliased pooling.

    Local statistics use a Gaussian window with sigma 1.5 and standard
    ``K1=0.01``/``K2=0.03`` constants. The five Wang et al. weights are
    truncated to supported scales and renormalized. ``sigma`` is a one-time
    policy prefilter; scale transitions use deterministic 2x2 average pooling
    and drop odd bottom/right edges.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != expected.shape:
        raise MetricError("MS-SSIM inputs must be equal 2D arrays")
    if pred.size == 0 or min(pred.shape) < win_size:
        raise MetricError("MS-SSIM footprint is too small for its window")
    if not np.isfinite(pred).all() or not np.isfinite(expected).all():
        raise MetricError("MS-SSIM inputs must be finite")
    input_scale = float(
        max(
            np.max(np.abs(pred), initial=0.0),
            np.max(np.abs(expected), initial=0.0),
        )
    )
    if _is_scale_degenerate(data_range, input_scale):
        raise MetricError("MS-SSIM data range is nonpositive or scale-degenerate")
    if not math.isfinite(sigma) or sigma < 0:
        raise MetricError("MS-SSIM sigma must be finite and nonnegative")
    if levels <= 0:
        raise MetricError("MS-SSIM levels must be positive")
    if sigma > 0:
        pred = gaussian_filter(pred, sigma=sigma, mode="reflect")
        expected = gaussian_filter(expected, sigma=sigma, mode="reflect")

    pred = pred / data_range
    expected = expected / data_range
    kernel = _gaussian_statistics_kernel(win_size)
    components: list[tuple[float, float]] = []
    maximum_levels = min(levels, len(_MS_SSIM_WEIGHTS))
    while len(components) < maximum_levels and min(pred.shape) >= win_size:
        components.append(_ms_ssim_components(pred, expected, kernel=kernel))
        if len(components) == maximum_levels:
            break
        pooled_prediction = _average_pool_2x2(pred)
        pooled_target = _average_pool_2x2(expected)
        if min(pooled_prediction.shape) < win_size:
            break
        pred, expected = pooled_prediction, pooled_target
    if not components:
        raise MetricError("MS-SSIM footprint is too small for its window")

    weights = _MS_SSIM_WEIGHTS[: len(components)]
    weights = weights / np.sum(weights)
    factors = [item[1] for item in components[:-1]] + [components[-1][0]]
    if any(factor <= 0.0 for factor in factors):
        raise MetricError("MS-SSIM component mean is nonpositive")
    result = float(
        np.exp(np.dot(weights, np.log(np.asarray(factors, dtype=np.float64))))
    )
    if not math.isfinite(result):
        raise MetricError("MS-SSIM result is nonfinite")
    return result


def amplitude_ms_ssim(prepared: PreparedComparison) -> float:
    prediction, target, data_range = amplitude_similarity_inputs(prepared)
    return multiscale_ssim(prediction, target, data_range=data_range, sigma=1.0)


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


def phase_ms_ssim(prepared: PreparedComparison) -> float:
    prediction, target = phase_similarity_inputs(prepared)
    return multiscale_ssim(prediction, target, data_range=1.0, sigma=0.0)


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


def varpro_quality_metrics(
    before: PreparedComparison, after: PreparedComparison
) -> VarProQualityMetrics:
    """Evaluate recognizability on both sides of dataset-level VarPro."""
    return VarProQualityMetrics(
        before=image_quality_metrics(before),
        after=image_quality_metrics(after),
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


def convergence_metrics(
    losses: Any, *, tail_fraction: float = 0.4, improving_threshold: float = 0.02
) -> ConvergenceMetrics:
    """Summarize the loss tail without declaring an improving boundary converged."""
    values = np.asarray(losses, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 4
        or not np.isfinite(values).all()
        or not 0.0 < tail_fraction <= 1.0
        or not math.isfinite(improving_threshold)
        or improving_threshold < 0.0
    ):
        raise MetricError("convergence losses or policy are invalid")
    tail_size = max(4, int(math.ceil(values.size * tail_fraction)))
    tail = values[-tail_size:]
    midpoint = tail.size // 2
    early = float(np.mean(tail[:midpoint]))
    late = float(np.mean(tail[midpoint:]))
    denominator = max(abs(early), np.finfo(np.float64).tiny)
    improvement = float((early - late) / denominator)
    x = np.arange(tail.size, dtype=np.float64)
    slope = float(np.polyfit(x, tail, 1)[0] / denominator)
    return ConvergenceMetrics(
        tail_relative_improvement=improvement,
        normalized_tail_slope=slope,
        budget_boundary_improving=1.0 if improvement > improving_threshold else 0.0,
    )


def scan_utilization_metrics(
    used_scan_ids: Iterable[Any],
    used_center_scan_ids: Iterable[Any] | None,
    expected_scan_ids: Iterable[Any],
    filtered_eligible_scan_ids: Iterable[Any],
    canvas_weights: Any,
) -> ScanUtilizationMetrics:
    """Account for unique scans and positive-weight canvas pixels."""
    used = frozenset(used_scan_ids)
    centers = (
        None if used_center_scan_ids is None else frozenset(used_center_scan_ids)
    )
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


def poisson_noise_oracle_metrics(
    observed_counts: Any, expected_intensity: Any, model_intensity: Any
) -> PoissonOracleMetrics:
    """Compare model count error with the truth-forward Poisson realization floor."""
    observed = np.asarray(observed_counts, dtype=np.float64)
    expected = np.asarray(expected_intensity, dtype=np.float64)
    model = np.asarray(model_intensity, dtype=np.float64)
    if (
        observed.shape != expected.shape
        or observed.shape != model.shape
        or observed.size == 0
        or not np.isfinite(observed).all()
        or not np.isfinite(expected).all()
        or not np.isfinite(model).all()
        or np.any(observed < 0.0)
        or np.any(expected < 0.0)
        or np.any(model < 0.0)
    ):
        raise MetricError("Poisson oracle inputs must be equal finite nonnegative arrays")
    denominator = _positive_denominator(_rms(observed), name="count error")
    oracle_error = float(_rms(expected - observed) / denominator)
    model_error = float(_rms(model - observed) / denominator)
    oracle_floor = _positive_denominator(oracle_error, name="Poisson oracle")
    return PoissonOracleMetrics(
        oracle_relative_l2_error=oracle_error,
        model_relative_l2_error=model_error,
        model_to_oracle_error_ratio=float(model_error / oracle_floor),
    )


def poisson_noise_oracle_relative_l2(
    observed_counts: Any, expected_intensity: Any
) -> float:
    """Return the truth-forward Poisson realization error in count space."""
    observed = np.asarray(observed_counts, dtype=np.float64)
    expected = np.asarray(expected_intensity, dtype=np.float64)
    if (
        observed.shape != expected.shape
        or observed.size == 0
        or not np.isfinite(observed).all()
        or not np.isfinite(expected).all()
        or np.any(observed < 0.0)
        or np.any(expected < 0.0)
    ):
        raise MetricError("Poisson oracle arrays must be equal finite nonnegative arrays")
    denominator = _positive_denominator(_rms(observed), name="Poisson oracle")
    return float(_rms(expected - observed) / denominator)


@dataclass(frozen=True)
class FRCResult:
    curve: tuple[float, ...]
    frc50: float
    frc1over7: float

    def to_json(self) -> dict[str, float | list[float]]:
        return {
            "curve": list(self.curve),
            "frc50": self.frc50,
            "frc1over7": self.frc1over7,
        }


def _first_below_threshold_interpolated(
    curve: NDArray[np.float64], threshold: float
) -> float:
    below = np.flatnonzero(curve < threshold)
    if below.size == 0:
        return float(curve.size)
    index = int(below[0])
    if index == 0:
        return 0.0
    previous = float(curve[index - 1])
    current = float(curve[index])
    denominator = current - previous
    if denominator == 0.0:
        return float(index)
    crossing = (index - 1) + (threshold - previous) / denominator
    return float(np.clip(crossing, index - 1, index))


def _validated_correlation_curve(
    value: Any,
    *,
    error_type: type[MetricError],
    context: str,
) -> NDArray[np.float64]:
    try:
        entries = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise error_type(f"{context} must be a numeric sequence") from error
    if entries.ndim != 1 or entries.size == 0:
        raise error_type(f"{context} must be a nonempty 1D sequence")

    normalized: list[float] = []
    for entry in entries:
        if isinstance(entry, (bool, np.bool_)) or not isinstance(entry, Real):
            raise error_type(f"{context} entries must be real scalars, excluding bool")
        number = float(entry)
        if not math.isfinite(number):
            raise error_type(f"{context} entries must be finite")
        if not (
            -_CORRELATION_DOMAIN_TOLERANCE
            <= number
            <= 1.0 + _CORRELATION_DOMAIN_TOLERANCE
        ):
            raise error_type(
                f"{context} entries must be within the correlation-domain tolerance"
            )
        normalized.append(number)
    return np.asarray(normalized, dtype=np.float64)


def _run_fsc(target: NDArray[Any], prediction: NDArray[Any]) -> Any:
    """Load and call the legacy FSC backend without leaking its console output."""
    import contextlib
    import io

    from ptycho.FRC import fourier_ring_corr

    with (
        contextlib.redirect_stdout(io.StringIO()),
        np.errstate(all="ignore"),
    ):
        return fourier_ring_corr.FSC(target, prediction)


def frc_metrics(prediction: Any, target: Any) -> FRCResult:
    """Evaluate low-level FSC on already-square inputs without preprocessing."""
    pred = np.asarray(prediction)
    expected = np.asarray(target)
    if pred.ndim != 2 or expected.ndim != 2:
        raise MetricError("FRC inputs must be 2D")
    if pred.shape != expected.shape:
        raise MetricError("FRC inputs must have equal shapes")
    if pred.size == 0:
        raise MetricError("FRC inputs must be nonempty")
    if pred.shape[0] != pred.shape[1]:
        raise MetricError("FRC inputs must be square")
    if not np.isfinite(pred).all() or not np.isfinite(expected).all():
        raise MetricError("FRC inputs must be finite")
    curve = _validated_correlation_curve(
        _run_fsc(expected, pred),
        error_type=MetricError,
        context="FRC curve",
    )
    return FRCResult(
        curve=tuple(float(value) for value in curve),
        frc50=_first_below_threshold_interpolated(curve, 0.5),
        frc1over7=_first_below_threshold_interpolated(curve, 1.0 / 7.0),
    )


def _square_arrays(
    prepared: PreparedComparison,
) -> tuple[NDArray[Any], NDArray[Any]]:
    bounds = prepared.frc_bounds
    return (
        prepared.reconstruction[bounds.row_slice, bounds.col_slice],
        prepared.target[bounds.row_slice, bounds.col_slice],
    )


def amplitude_frc(prepared: PreparedComparison) -> FRCResult:
    prediction, target = _square_arrays(prepared)
    return frc_metrics(_amplitude(prediction), _amplitude(target))


def phase_frc(prepared: PreparedComparison) -> FRCResult:
    prediction_phase, target_phase = _aligned_phase_arrays(prepared)
    bounds = prepared.frc_bounds
    return frc_metrics(
        prediction_phase[bounds.row_slice, bounds.col_slice],
        target_phase[bounds.row_slice, bounds.col_slice],
    )


def metric_paths() -> frozenset[str]:
    """Return the manifest's immutable closed metric-path registry."""
    return METRIC_PATHS


@dataclass(frozen=True)
class MetricRecord:
    path: str
    value: float | tuple[float, ...]
    basis: str
    alignment: str

    def to_json(self) -> dict[str, Any]:
        value: float | list[float]
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "value": value,
            "metadata": {"basis": self.basis, "alignment": self.alignment},
        }


def _record_value(path: str, value: Any) -> float | tuple[float, ...]:
    is_curve = path.endswith("_frc_curve")
    if is_curve:
        curve = _validated_correlation_curve(
            value,
            error_type=RegistryError,
            context="metric curve",
        )
        return tuple(float(item) for item in curve)
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise RegistryError("metric value must be a finite scalar")
    return float(value)


def build_metric_record(
    path: str,
    value: Any,
    *,
    basis: str,
    alignment: str,
    truth_role: str | None = None,
) -> MetricRecord:
    """Build one immutable record after path, role, value, and policy validation."""
    if path not in METRIC_PATHS:
        raise RegistryError(f"path is not in the closed metric registry: {path!r}")
    normalized_value = _record_value(path, value)
    if not isinstance(basis, str) or not basis:
        raise RegistryError("metric basis must be a nonempty string")
    if not isinstance(alignment, str) or not alignment:
        raise RegistryError("metric alignment must be a nonempty string")

    namespace = path.split(".", 1)[0]
    image_namespaces = {"truth_quality", "reference_agreement"}
    if namespace in image_namespaces:
        expected_namespaces = {
            "object_truth": "truth_quality",
            "reference_reconstruction": "reference_agreement",
        }
        expected_namespace = (
            None if truth_role is None else expected_namespaces.get(truth_role)
        )
        if expected_namespace is None:
            raise RegistryError("image metric requires a truth or reference role")
        if namespace != expected_namespace:
            raise RegistryError(
                f"{truth_role} cannot emit metric in the {namespace} namespace"
            )
    elif truth_role is not None:
        raise RegistryError(
            f"{namespace} metrics cannot be relabeled with truth role {truth_role}"
        )
    return MetricRecord(path, normalized_value, basis, alignment)


def build_image_metric_record(
    name: str,
    value: Any,
    *,
    truth_role: str,
    basis: str,
    alignment: str,
) -> MetricRecord:
    if truth_role == "object_truth":
        namespace = "truth_quality"
    elif truth_role == "reference_reconstruction":
        if name.startswith("absolute_"):
            raise RegistryError("absolute correctness metrics require object truth")
        namespace = "reference_agreement"
    else:
        raise RegistryError(f"unsupported truth role: {truth_role!r}")
    return build_metric_record(
        f"{namespace}.{name}",
        value,
        basis=basis,
        alignment=alignment,
        truth_role=truth_role,
    )


def build_measurement_metric_record(
    name: str, value: Any, *, basis: str, alignment: str
) -> MetricRecord:
    return build_metric_record(
        f"measurement_consistency.{name}",
        value,
        basis=basis,
        alignment=alignment,
    )


@dataclass(frozen=True)
class MetricBundle:
    records: Iterable[MetricRecord]

    def __post_init__(self) -> None:
        try:
            records = tuple(self.records)
        except TypeError as error:
            raise RegistryError("metric bundle records must be iterable") from error
        if any(not isinstance(record, MetricRecord) for record in records):
            raise RegistryError("metric bundle entries must be MetricRecord instances")
        if len({record.path for record in records}) != len(records):
            raise RegistryError("metric bundle contains duplicate paths")
        object.__setattr__(self, "records", records)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for record in self.records:
            components = record.path.split(".")
            cursor = payload
            for component in components[:-1]:
                child = cursor.setdefault(component, {})
                if not isinstance(child, dict):
                    raise RegistryError("metric paths conflict during JSON encoding")
                cursor = child
            cursor[components[-1]] = record.to_json()
        return payload


def build_patch_amplitude_pearson_record(
    reconstruction: Any, target: Any, *, truth_role: str
) -> MetricRecord:
    return build_image_metric_record(
        "patch_amp_pearson",
        patch_amplitude_pearson(reconstruction, target),
        truth_role=truth_role,
        basis="object_frame_raw_amplitude",
        alignment="pearson_centering_only",
    )

"""Reusable, identity-bearing complex-probe transform pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from skimage.restoration import unwrap_phase


BOUNDARY_METHOD = "harmonic_dirichlet_c0"
BOUNDARY_SOLVER = "scipy.sparse.linalg.spsolve"
BOUNDARY_SOLVER_TOLERANCE = 1e-10


@dataclass(frozen=True)
class BoundaryMatchedProbeResult:
    """Numerical result and diagnostics for one boundary-matched extension."""

    probe: np.ndarray
    unwrapped_phase: np.ndarray
    quadratic_phase: np.ndarray
    correction: np.ndarray
    source_footprint_mask: np.ndarray
    inner_boundary_mask: np.ndarray
    outer_boundary_mask: np.ndarray
    free_annulus_mask: np.ndarray
    source_rows: tuple[int, int]
    source_columns: tuple[int, int]
    quadratic_coefficients: tuple[float, float]
    solver_tolerance: float
    laplacian_residual: float
    seam_residual: float

    def metadata(self) -> dict[str, Any]:
        return {
            "boundary_method": BOUNDARY_METHOD,
            "solver": BOUNDARY_SOLVER,
            "solver_tolerance": float(self.solver_tolerance),
            "laplacian_residual": float(self.laplacian_residual),
            "seam_residual": float(self.seam_residual),
            "source_rows": list(self.source_rows),
            "source_columns": list(self.source_columns),
            "quadratic_coefficients": list(self.quadratic_coefficients),
        }


@dataclass(frozen=True)
class ProbeTransformResult:
    """Transformed probe plus deterministic operation diagnostics."""

    probe: np.ndarray
    metadata: dict[str, Any]
    boundary_result: BoundaryMatchedProbeResult | None = None


def _center_slices(source_size: int, target_size: int) -> tuple[slice, slice]:
    if target_size < source_size:
        raise ValueError("target_N must be >= probe size")
    pad_total = target_size - source_size
    before = pad_total // 2
    after = before + source_size
    return slice(before, after), slice(before, after)


def _pad_complex_probe(probe: np.ndarray, target_N: int) -> np.ndarray:
    """Center-pad a complex probe with zeros."""

    height, width = probe.shape
    if height != width:
        raise ValueError("probe must be square")
    if target_N < height:
        raise ValueError("pad_complex requires target_N >= current probe size")
    pad_total = target_N - height
    before = pad_total // 2
    after = pad_total - before
    return np.pad(probe, ((before, after), (before, after)), mode="constant")


def _pad_amplitude(amplitude: np.ndarray, target_N: int) -> np.ndarray:
    """Center-pad amplitude using edge values."""

    height, width = amplitude.shape
    if height != width:
        raise ValueError("probe must be square")
    if target_N < height:
        raise ValueError("pad_extrapolate requires target_N >= probe size")
    pad_total = target_N - height
    before = pad_total // 2
    after = pad_total - before
    return np.pad(amplitude, ((before, after), (before, after)), mode="edge")


def _fit_quadratic_phase(phase: np.ndarray) -> tuple[float, float]:
    """Fit the established unweighted radial ``a*r^2+b`` phase model."""

    height, width = phase.shape
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    yy, xx = np.indices((height, width))
    radius_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
    design = np.stack(
        [radius_squared.ravel(), np.ones(radius_squared.size)], axis=1
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, phase.ravel(), rcond=None
    )
    return float(coefficients[0]), float(coefficients[1])


def _quadratic_phase(
    shape: tuple[int, int],
    coefficients: tuple[float, float],
    *,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    if center is None:
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
    else:
        center_y, center_x = center
    radius_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
    a, b = coefficients
    return a * radius_squared + b


def _inner_perimeter_mask(
    shape: tuple[int, int], row_slice: slice, column_slice: slice
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    y0, y1 = int(row_slice.start), int(row_slice.stop)
    x0, x1 = int(column_slice.start), int(column_slice.stop)
    mask[y0, x0:x1] = True
    mask[y1 - 1, x0:x1] = True
    mask[y0:y1, x0] = True
    mask[y0:y1, x1 - 1] = True
    return mask


def _outer_perimeter_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _wrap_phase(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def extend_probe_boundary_matched(
    probe: np.ndarray,
    target_N: int,
    *,
    solver_tolerance: float = BOUNDARY_SOLVER_TOLERANCE,
) -> BoundaryMatchedProbeResult:
    """Extend a probe with a C0 boundary-conditioned harmonic correction.

    The source complex array is copied exactly into the centered footprint.
    Its unwrapped phase supplies the inner Dirichlet boundary for the harmonic
    correction, while the correction is zero at the target perimeter. Corner
    pixels of the source footprint belong to the inner boundary; target corners
    belong to the outer boundary. These sets are disjoint because at least one
    padding pixel per side is required.
    """

    source = np.asarray(probe, dtype=np.complex64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("probe must be square")
    if not isinstance(target_N, int) or target_N <= source.shape[0]:
        raise ValueError("boundary-matched extension requires target_N > probe size")
    row_slice, column_slice = _center_slices(source.shape[0], target_N)
    if min(row_slice.start, column_slice.start) < 1 or max(
        row_slice.stop, column_slice.stop
    ) > target_N - 1:
        raise ValueError(
            "boundary-matched extension requires at least one outer pixel on every side"
        )
    if not np.isfinite(source).all():
        raise ValueError("probe must contain only finite values")
    if not np.isfinite(solver_tolerance) or solver_tolerance <= 0:
        raise ValueError("solver_tolerance must be finite and positive")

    source_phase = np.asarray(unwrap_phase(np.angle(source)), dtype=np.float64)
    coefficients = _fit_quadratic_phase(source_phase)
    source_center = (source.shape[0] - 1) / 2.0
    quadratic = _quadratic_phase(
        (target_N, target_N),
        coefficients,
        center=(
            float(row_slice.start) + source_center,
            float(column_slice.start) + source_center,
        ),
    )

    footprint = np.zeros((target_N, target_N), dtype=bool)
    footprint[row_slice, column_slice] = True
    inner_boundary = _inner_perimeter_mask(
        footprint.shape, row_slice, column_slice
    )
    outer_boundary = _outer_perimeter_mask(footprint.shape)
    if np.any(footprint & outer_boundary):
        raise ValueError("source footprint and target boundary must be disjoint")

    correction = np.zeros((target_N, target_N), dtype=np.float64)
    correction[row_slice, column_slice] = (
        source_phase - quadratic[row_slice, column_slice]
    )
    fixed = footprint | outer_boundary
    free = ~fixed
    free_coordinates = np.argwhere(free)
    free_index = np.full((target_N, target_N), -1, dtype=np.int64)
    if free_coordinates.size:
        free_index[free] = np.arange(free_coordinates.shape[0], dtype=np.int64)
        matrix = lil_matrix(
            (free_coordinates.shape[0], free_coordinates.shape[0]),
            dtype=np.float64,
        )
        right_hand_side = np.zeros(free_coordinates.shape[0], dtype=np.float64)
        for row, (y, x) in enumerate(free_coordinates):
            matrix[row, row] = 4.0
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                neighbor_index = free_index[neighbor_y, neighbor_x]
                if neighbor_index >= 0:
                    matrix[row, neighbor_index] = -1.0
                else:
                    right_hand_side[row] += correction[neighbor_y, neighbor_x]
        system: csr_matrix = matrix.tocsr()
        solution = spsolve(
            system,
            right_hand_side,
            permc_spec="NATURAL",
            use_umfpack=False,
        )
        correction[free] = solution
        residual = float(
            np.max(np.abs(system @ solution - right_hand_side), initial=0.0)
        )
    else:
        residual = 0.0
    if not np.isfinite(correction).all() or residual > solver_tolerance:
        raise RuntimeError(
            "boundary-matched harmonic solve failed: "
            f"residual {residual:.6g} exceeds tolerance {solver_tolerance:.6g}"
        )

    extended_phase = quadratic + correction
    seam_error = _wrap_phase(
        extended_phase[row_slice, column_slice] - source_phase
    )
    source_inner = _inner_perimeter_mask(
        source.shape, slice(0, source.shape[0]), slice(0, source.shape[1])
    )
    seam_residual = float(np.max(np.abs(seam_error[source_inner]), initial=0.0))
    amplitude = _pad_amplitude(np.abs(source), target_N)
    output = (amplitude * np.exp(1j * extended_phase)).astype(np.complex64)
    output[row_slice, column_slice] = source

    return BoundaryMatchedProbeResult(
        probe=output,
        unwrapped_phase=extended_phase,
        quadratic_phase=quadratic,
        correction=correction,
        source_footprint_mask=footprint,
        inner_boundary_mask=inner_boundary,
        outer_boundary_mask=outer_boundary,
        free_annulus_mask=free,
        source_rows=(int(row_slice.start), int(row_slice.stop)),
        source_columns=(int(column_slice.start), int(column_slice.stop)),
        quadratic_coefficients=coefficients,
        solver_tolerance=float(solver_tolerance),
        laplacian_residual=residual,
        seam_residual=seam_residual,
    )


def _pad_extrapolate_complex_probe(probe: np.ndarray, target_N: int) -> np.ndarray:
    """Legacy global-quadratic extension; keep byte behavior unchanged."""

    amplitude = np.abs(probe)
    phase = unwrap_phase(np.angle(probe))
    amplitude_padded = _pad_amplitude(amplitude, target_N)
    coefficients = _fit_quadratic_phase(phase)
    phase_extrapolated = _quadratic_phase((target_N, target_N), coefficients)
    phase_wrapped = (phase_extrapolated + np.pi) % (2 * np.pi) - np.pi
    return (amplitude_padded * np.exp(1j * phase_wrapped)).astype(np.complex64)


def smooth_complex_array(array: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth complex amplitude and unwrapped phase (legacy public contract)."""

    if not np.iscomplexobj(array):
        raise ValueError("Input array for smoothing must be complex.")
    amplitude = np.abs(array)
    phase = unwrap_phase(np.angle(array))
    smoothed_amplitude = gaussian_filter(amplitude, sigma=sigma)
    smoothed_phase = gaussian_filter(phase, sigma=sigma)
    return (smoothed_amplitude * np.exp(1j * smoothed_phase)).astype(array.dtype)


def interpolate_array(array: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Cubic real/imaginary interpolation (legacy public contract)."""

    if array.ndim != 2:
        raise ValueError("Interpolation only supports 2D arrays.")
    real_part = zoom(array.real, zoom_factor, order=3)
    imaginary_part = zoom(array.imag, zoom_factor, order=3)
    return (real_part + 1j * imaginary_part).astype(array.dtype)


def _interpolate_array_to_size(array: np.ndarray, target_N: int) -> np.ndarray:
    return interpolate_array(array, target_N / array.shape[0])


def _coerce_pipeline_value(raw_value: str) -> object:
    raw_value = raw_value.strip()
    if not raw_value:
        return raw_value
    try:
        integer = int(raw_value)
    except ValueError:
        integer = None
    if integer is not None and str(integer) == raw_value:
        return integer
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _serialize_numeric(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_probe_transform_pipeline(spec: str) -> list[dict[str, object]]:
    """Parse the public shorthand pipeline into explicit operation dictionaries."""

    if not spec or not spec.strip():
        raise ValueError("probe transform pipeline must be non-empty")
    steps: list[dict[str, object]] = []
    for raw_segment in spec.split("|"):
        segment = raw_segment.strip()
        if not segment:
            raise ValueError("probe transform pipeline contains an empty step")
        raw_op, separator, raw_parameters = segment.partition(":")
        op_name = raw_op.strip()
        parameters = raw_parameters.strip()
        if op_name == "smooth":
            if not separator or not parameters:
                raise ValueError("smooth requires a sigma value")
            steps.append({"op": "smooth_complex", "sigma": float(parameters)})
            continue
        if op_name in {"pad", "pad_preserve"}:
            if not separator or not parameters:
                raise ValueError(f"{op_name} requires a target_N value")
            steps.append({"op": "pad_complex", "target_N": int(parameters)})
            continue
        if op_name == "interp":
            if not separator or not parameters:
                raise ValueError("interp requires a target_N value")
            parts = [part.strip() for part in parameters.split(",") if part.strip()]
            step: dict[str, object] = {
                "op": "interpolate_complex",
                "order": 3,
                "representation": "real_imag",
            }
            first = parts.pop(0)
            if "=" in first:
                key, value = first.split("=", 1)
                step[key.strip()] = _coerce_pipeline_value(value)
            else:
                step["target_N"] = int(first)
            for part in parts:
                key, value = part.split("=", 1)
                step[key.strip()] = _coerce_pipeline_value(value)
            if "target_N" not in step:
                raise ValueError("interp requires target_N")
            steps.append(step)
            continue
        if op_name in {
            "pad_extrapolate",
            "pad_extrapolate_boundary_matched",
        }:
            if not separator or not parameters:
                raise ValueError(f"{op_name} requires a target_N value")
            explicit_name = (
                "pad_extrapolate_boundary_matched_complex"
                if op_name == "pad_extrapolate_boundary_matched"
                else "pad_extrapolate_complex"
            )
            steps.append({"op": explicit_name, "target_N": int(parameters)})
            continue

        step = {"op": op_name}
        if parameters:
            for part in [item.strip() for item in parameters.split(",") if item.strip()]:
                if "=" not in part:
                    raise ValueError(
                        f"Unsupported parameter syntax {part!r} in probe transform pipeline"
                    )
                key, value = part.split("=", 1)
                step[key.strip()] = _coerce_pipeline_value(value)
        if step["op"] not in {
            "smooth_complex",
            "pad_complex",
            "interpolate_complex",
            "pad_extrapolate_complex",
            "pad_extrapolate_boundary_matched_complex",
        }:
            raise ValueError(f"Unknown probe transform op {step['op']!r}")
        steps.append(step)
    return steps


def serialize_probe_transform_pipeline(
    steps: Sequence[Mapping[str, object]],
) -> str:
    """Serialize explicit operations using stable established spellings."""

    serialized: list[str] = []
    for step in steps:
        operation = step["op"]
        if operation == "smooth_complex":
            serialized.append(f"smooth:{_serialize_numeric(step['sigma'])}")
        elif operation == "pad_complex":
            serialized.append(f"pad:{int(step['target_N'])}")
        elif operation == "pad_extrapolate_complex":
            serialized.append(f"pad_extrapolate:{int(step['target_N'])}")
        elif operation == "pad_extrapolate_boundary_matched_complex":
            serialized.append(
                "pad_extrapolate_boundary_matched:"
                f"{int(step['target_N'])}"
            )
        elif operation == "interpolate_complex":
            fragment = f"interp:{int(step['target_N'])}"
            extras: list[str] = []
            if step.get("order", 3) != 3:
                extras.append(f"order={_serialize_numeric(step['order'])}")
            if step.get("representation", "real_imag") != "real_imag":
                extras.append(f"representation={step['representation']}")
            if extras:
                fragment = f"{fragment},{','.join(extras)}"
            serialized.append(fragment)
        else:
            raise ValueError(f"Unsupported probe transform op {operation!r}")
    return "|".join(serialized)


def _resolve_probe_size_after_step(
    current_size: int, step: Mapping[str, object]
) -> int:
    operation = step["op"]
    if operation == "smooth_complex":
        return current_size
    if operation in {
        "pad_complex",
        "interpolate_complex",
        "pad_extrapolate_complex",
        "pad_extrapolate_boundary_matched_complex",
    }:
        target_N = int(step["target_N"])
        if operation in {
            "pad_extrapolate_complex",
            "pad_extrapolate_boundary_matched_complex",
        } and target_N < current_size:
            raise ValueError("pad_extrapolate requires target_N >= probe size")
        if operation == "pad_complex" and target_N < current_size:
            raise ValueError("pad_complex requires target_N >= current probe size")
        return target_N
    raise ValueError(f"Unsupported probe transform op {operation!r}")


def normalize_probe_transform_pipeline(
    *,
    target_N: int,
    probe_shape: tuple[int, int],
    probe_scale_mode: str | None,
    probe_smoothing_sigma: float | None,
    probe_transform_pipeline: str | None,
) -> tuple[str, list[dict[str, object]]]:
    """Resolve legacy shorthands or an explicit pipeline deterministically."""

    if len(probe_shape) != 2 or probe_shape[0] != probe_shape[1]:
        raise ValueError("probe must be square")
    explicit_segments: list[str] | None = None
    if probe_transform_pipeline:
        explicit_segments = [segment.strip() for segment in probe_transform_pipeline.split("|")]
        steps = parse_probe_transform_pipeline(probe_transform_pipeline)
    else:
        sigma = float(probe_smoothing_sigma or 0.0)
        mode = probe_scale_mode or "pad_extrapolate"
        steps = []
        if mode == "interpolate":
            steps.append(
                {
                    "op": "interpolate_complex",
                    "target_N": target_N,
                    "order": 3,
                    "representation": "real_imag",
                }
            )
            if sigma > 0:
                steps.append({"op": "smooth_complex", "sigma": sigma})
        elif mode == "pad_preserve":
            if sigma > 0:
                steps.append({"op": "smooth_complex", "sigma": sigma})
            steps.append({"op": "pad_complex", "target_N": target_N})
        elif mode == "pad_extrapolate":
            steps.append({"op": "pad_extrapolate_complex", "target_N": target_N})
            if sigma > 0:
                steps.append({"op": "smooth_complex", "sigma": sigma})
        elif mode == "pipeline":
            raise ValueError(
                "probe_transform_pipeline must be provided when probe_scale_mode='pipeline'"
            )
        else:
            raise ValueError(f"Unknown scale_mode {mode!r}")

    boundary_indices = [
        index
        for index, step in enumerate(steps)
        if step["op"] == "pad_extrapolate_boundary_matched_complex"
    ]
    if len(boundary_indices) > 1:
        raise ValueError(
            "pad_extrapolate_boundary_matched may appear exactly once"
        )
    if boundary_indices and boundary_indices[-1] != len(steps) - 1:
        raise ValueError(
            "pad_extrapolate_boundary_matched must be the final operation"
        )
    current_size = probe_shape[0]
    for step in steps:
        if step["op"] == "pad_extrapolate_boundary_matched_complex":
            target_size = int(step["target_N"])
            total_padding = target_size - current_size
            before = total_padding // 2
            after = total_padding - before
            if before < 1 or after < 1:
                raise ValueError(
                    "pad_extrapolate_boundary_matched requires at least one "
                    "outer pixel on every side"
                )
        current_size = _resolve_probe_size_after_step(current_size, step)
    if current_size != target_N:
        raise ValueError(
            f"probe transform pipeline final probe size {current_size} "
            f"does not match target_N {target_N}"
        )
    normalized = serialize_probe_transform_pipeline(steps)
    if explicit_segments is not None:
        normalized_parts = normalized.split("|")
        for index, raw_segment in enumerate(explicit_segments):
            if raw_segment.partition(":")[0].strip() == "pad_preserve":
                normalized_parts[index] = (
                    f"pad_preserve:{int(steps[index]['target_N'])}"
                )
        normalized = "|".join(normalized_parts)
    return normalized, steps


def apply_probe_transform_pipeline_with_metadata(
    probe: np.ndarray,
    steps: Sequence[Mapping[str, object]],
) -> ProbeTransformResult:
    """Apply normalized operations and retain solver evidence when present."""

    transformed = np.asarray(probe, dtype=np.complex64)
    if transformed.ndim != 2 or transformed.shape[0] != transformed.shape[1]:
        raise ValueError("probe must be square")
    boundary_result: BoundaryMatchedProbeResult | None = None
    metadata: dict[str, Any] = {}
    for index, step in enumerate(steps):
        operation = step["op"]
        if boundary_result is not None:
            raise ValueError(
                "pad_extrapolate_boundary_matched must be the final operation"
            )
        if operation == "smooth_complex":
            transformed = smooth_complex_array(
                transformed, float(step["sigma"])
            )
        elif operation == "pad_complex":
            transformed = _pad_complex_probe(
                transformed, int(step["target_N"])
            ).astype(np.complex64)
        elif operation == "interpolate_complex":
            if transformed.shape[0] != int(step["target_N"]):
                transformed = _interpolate_array_to_size(
                    transformed, int(step["target_N"])
                ).astype(np.complex64)
        elif operation == "pad_extrapolate_complex":
            transformed = _pad_extrapolate_complex_probe(
                transformed, int(step["target_N"])
            )
        elif operation == "pad_extrapolate_boundary_matched_complex":
            if index != len(steps) - 1:
                raise ValueError(
                    "pad_extrapolate_boundary_matched must be the final operation"
                )
            boundary_result = extend_probe_boundary_matched(
                transformed,
                int(step["target_N"]),
            )
            transformed = boundary_result.probe
            metadata.update(boundary_result.metadata())
        else:
            raise ValueError(f"Unsupported probe transform op {operation!r}")
    return ProbeTransformResult(
        probe=transformed.astype(np.complex64),
        metadata=metadata,
        boundary_result=boundary_result,
    )


def apply_probe_transform_pipeline(
    probe: np.ndarray,
    steps: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Apply a normalized pipeline using the historical ndarray return type."""

    return apply_probe_transform_pipeline_with_metadata(probe, steps).probe


def make_disk_mask(N: int, diameter: float) -> np.ndarray:
    """Create a centered binary disk mask."""

    radius = diameter / 2.0
    yy, xx = np.ogrid[:N, :N]
    center = (N - 1) / 2.0
    distance_squared = (yy - center) ** 2 + (xx - center) ** 2
    return (distance_squared <= radius**2).astype(np.float32)


def apply_probe_mask(probe: np.ndarray, diameter: float | None) -> np.ndarray:
    """Apply a simulation-time centered disk mask when requested."""

    if diameter is None:
        return probe
    if probe.shape[0] != probe.shape[1]:
        raise ValueError("probe must be square")
    return (probe * make_disk_mask(probe.shape[0], diameter)).astype(probe.dtype)

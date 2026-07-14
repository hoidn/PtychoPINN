"""Typed metric-record and report-evidence assembly for canonical runs.

Separates evidence construction (closed-registry metric records, anchor-aligned
image comparisons, and plot-ready report arrays) from study orchestration so
each module keeps one responsibility. Image metrics are routed to the
``truth_quality`` or ``reference_agreement`` namespace strictly by the
dataset's declared truth role; count/VarPro metrics always live under
``measurement_consistency``.
"""

from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics
from .dataset_schema import DatasetDescriptor
from .metrics import MetricRecord, build_metric_record
from .runtime_execution import (
    RELOAD_ALLCLOSE_METRIC_PATH,
    RELOAD_METRIC_PATH,
    CanonicalRunResult,
    MilestoneRunResult,
    RuntimeExecutionError,
    _as_array,
)

_ANCHOR_ALIGNMENT = "anchor_common_mask"
_NO_ALIGNMENT = "none"

#: Per-attempt artifact persisting the raw training history (design step 4).
TRAINING_HISTORY_ARTIFACT = "training_history.json"
_TRAINING_HISTORY_SCHEMA = "ablation_training_history_v1"
#: Gradient-norm series preference order; the trainer logs pre-clip norms
#: under ``grad_norm_preclip`` when ``training.log_grad_norm`` is enabled.
_GRADIENT_SERIES_NAMES = (
    "grad_norm_preclip_step",
    "grad_norm_preclip",
    "grad_norm_preclip_epoch",
)


def npy_bytes(array: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, _as_array(array))
    return buffer.getvalue()


def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def records_to_payload(records: tuple[MetricRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": record.path,
            "value": list(record.value)
            if isinstance(record.value, tuple)
            else record.value,
            "basis": record.basis,
            "alignment": record.alignment,
        }
        for record in records
    ]


def records_from_payload(rows: Any, truth_role: str) -> tuple[MetricRecord, ...]:
    """Rebuild validated metric records from a stored metrics.json payload."""
    records = []
    for row in rows:
        path = row["path"]
        namespace = path.split(".", 1)[0]
        role = (
            truth_role
            if namespace in {"truth_quality", "reference_agreement"}
            else None
        )
        records.append(
            build_metric_record(
                path,
                row["value"],
                basis=row["basis"],
                alignment=row["alignment"],
                truth_role=role,
            )
        )
    return tuple(records)


def flat_metrics(records: tuple[MetricRecord, ...]) -> dict[str, Any]:
    return {record.path: record.value for record in records}


def load_truth_array(descriptor: DatasetDescriptor) -> np.ndarray | None:
    """Load the declared object truth or conventional reference array."""
    if descriptor.truth_location == "embedded_test":
        path, key = descriptor.test, descriptor.truth_key
    elif descriptor.truth_location == "external_npz":
        path, key = descriptor.reference, descriptor.truth_key
    else:
        return None
    if path is None or key is None:
        raise RuntimeExecutionError(
            "metrics", f"dataset {descriptor.id!r} declares truth without a source"
        )
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise RuntimeExecutionError(
                "metrics", f"truth key {key!r} missing from {path}"
            )
        return np.asarray(archive[key])


def _measurement(name: str, value: Any, *, basis: str) -> MetricRecord:
    return build_metric_record(
        f"measurement_consistency.{name}",
        value,
        basis=basis,
        alignment=_NO_ALIGNMENT,
    )


def _measurement_records(result: CanonicalRunResult) -> list[MetricRecord]:
    diagnostics = result.reloaded_diagnostics
    count = result.count_metrics
    records = [
        _measurement(
            "relative_l2_intensity_error",
            count.relative_l2_intensity_error,
            basis="physical_count_space",
        ),
        _measurement(
            "mean_raw_poisson_nll",
            count.mean_raw_poisson_nll,
            basis="physical_count_space",
        ),
        _measurement("varpro.s1", diagnostics.s1, basis="varpro_solve"),
        _measurement("varpro.s2", diagnostics.s2, basis="varpro_solve"),
    ]
    optional = (
        ("varpro.condition", diagnostics.condition),
        ("varpro.unit_objective", diagnostics.unit_objective),
        ("varpro.fitted_objective", diagnostics.fitted_objective),
    )
    for name, value in optional:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            records.append(_measurement(name, value, basis="varpro_solve"))
    return records


@lru_cache(maxsize=16)
def _cached_poisson_oracle(
    dataset_path: str,
    dataset_identity: str,
    measurement_key: str,
    sample_ids: tuple[int, ...],
    sample_identity_digest: str,
) -> float | None:
    """Compute once per immutable dataset and exact loader sample policy."""
    del dataset_identity, sample_identity_digest
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {measurement_key, "ground_truth_patches", "probeGuess"}
        if not required <= set(archive.files):
            return None
        observed = np.asarray(archive[measurement_key], dtype=np.float64)
        patches = np.asarray(archive["ground_truth_patches"])
        probe = np.asarray(archive["probeGuess"])
    if patches.ndim == 4 and patches.shape[-1] == 1:
        patches = patches[..., 0]
    if probe.ndim == 2:
        probe = probe[None]
    if (
        patches.ndim != 3
        or probe.ndim != 3
        or patches.shape[-2:] != probe.shape[-2:]
        or observed.shape != patches.shape
    ):
        raise RuntimeExecutionError(
            "metrics", "truth-forward Poisson oracle arrays have incompatible shapes"
        )
    ids = np.asarray(sample_ids, dtype=np.int64)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= patches.shape[0]):
        raise RuntimeExecutionError("metrics", "Poisson oracle sample ids are invalid")
    observed = observed[ids]
    patches = patches[ids]
    squared_error_sum = 0.0
    measured_square_sum = float(np.square(observed).sum(dtype=np.float64))
    for start in range(0, patches.shape[0], 128):
        patch_batch = patches[start : start + 128]
        exit_waves = patch_batch[:, None] * probe[None]
        fields = np.fft.fft2(exit_waves, axes=(-2, -1), norm="ortho")
        expected = np.fft.fftshift(
            np.square(np.abs(fields)).sum(axis=1), axes=(-2, -1)
        )
        residual = expected - observed[start : start + 128]
        squared_error_sum += float(np.square(residual).sum(dtype=np.float64))
    if measured_square_sum <= 0.0:
        raise RuntimeExecutionError(
            "metrics", "truth-forward Poisson oracle has zero measured energy"
        )
    oracle_error = float(np.sqrt(squared_error_sum / measured_square_sum))
    if not math.isfinite(oracle_error) or oracle_error <= 0.0:
        raise RuntimeExecutionError(
            "metrics", "truth-forward Poisson oracle floor must be positive and finite"
        )
    return oracle_error


def _truth_forward_poisson_oracle_records(
    descriptor: DatasetDescriptor, result: CanonicalRunResult
) -> list[MetricRecord]:
    """Evaluate the oracle over the exact model count-loader sample stream."""
    if descriptor.truth != "object_truth" or descriptor.test is None:
        return []
    sample_ids = tuple(getattr(result.count_metrics, "sample_ids", ()))
    sample_digest = getattr(result.count_metrics, "sample_identity_digest", "")
    if not sample_ids or not sample_digest:
        raise RuntimeExecutionError(
            "metrics", "count metrics are missing exact loader sample identity"
        )
    oracle_error = _cached_poisson_oracle(
        str(descriptor.test),
        getattr(descriptor, "test_sha256", str(descriptor.test)),
        descriptor.measurement_key,
        sample_ids,
        sample_digest,
    )
    if oracle_error is None:
        return []
    model_error = float(result.count_metrics.relative_l2_intensity_error)
    return [
        _measurement(
            "poisson_oracle_relative_l2_error",
            oracle_error,
            basis="truth_forward_poisson_noise_floor",
        ),
        _measurement(
            "model_to_poisson_oracle_error_ratio",
            model_error / oracle_error,
            basis="model_count_error_over_truth_forward_poisson_floor",
        ),
    ]


def _stability_records(result: CanonicalRunResult) -> list[MetricRecord]:
    diagnostics = result.reloaded_diagnostics
    weights = _as_array(diagnostics.canvas_weights)
    arrays = (
        result.reference_canvas,
        result.reference_texture,
        result.reloaded_canvas,
        result.reloaded_texture,
    )
    finite = all(bool(np.isfinite(_as_array(item)).all()) for item in arrays)

    def stability(name: str, value: Any) -> MetricRecord:
        return build_metric_record(
            f"stability.{name}",
            value,
            basis="canonical_reassembly",
            alignment=_NO_ALIGNMENT,
        )

    records = [
        build_metric_record(
            RELOAD_METRIC_PATH,
            result.reload_max_abs_error,
            basis="canonical_reassembly",
            alignment=_NO_ALIGNMENT,
        ),
        build_metric_record(
            RELOAD_ALLCLOSE_METRIC_PATH,
            1.0 if result.reload_allclose else 0.0,
            basis="canonical_reassembly_texture_and_canvas",
            alignment=_NO_ALIGNMENT,
        ),
        stability("finite", 1.0 if finite else 0.0),
        stability("patches_accepted", float(diagnostics.patches_accepted)),
        stability("patches_total", float(diagnostics.patches_total)),
    ]
    if weights.size:
        records.append(stability("coverage_fraction", float(np.mean(weights > 0))))
    texture = _as_array(result.reloaded_texture)
    if texture.ndim == 2 and weights.shape == texture.shape and np.any(weights > 0):
        collapse = metrics.valid_mask_diagnostics(
            texture,
            (weights > 0).astype(np.bool_),
            decoder_output=texture,
        )
        for name, value in (
            ("amp_variance", collapse.amplitude_variance),
            ("phase_variance", collapse.phase_variance),
            ("amp_dynamic_range", collapse.amplitude_dynamic_range),
            ("phase_dynamic_range", collapse.phase_dynamic_range),
        ):
            records.append(stability(name, value))
    for name, value in (
        (
            "real_head_saturation_fraction",
            getattr(diagnostics, "decoder_real_saturation_fraction", None),
        ),
        (
            "imag_head_saturation_fraction",
            getattr(diagnostics, "decoder_imag_saturation_fraction", None),
        ),
        (
            "real_head_lower_saturation_fraction",
            getattr(diagnostics, "decoder_real_lower_saturation_fraction", None),
        ),
        (
            "real_head_upper_saturation_fraction",
            getattr(diagnostics, "decoder_real_upper_saturation_fraction", None),
        ),
        (
            "imag_head_lower_saturation_fraction",
            getattr(diagnostics, "decoder_imag_lower_saturation_fraction", None),
        ),
        (
            "imag_head_upper_saturation_fraction",
            getattr(diagnostics, "decoder_imag_upper_saturation_fraction", None),
        ),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            records.append(stability(name, value))
    used_scan_ids = getattr(diagnostics, "used_scan_ids", None)
    used_center_scan_ids = getattr(diagnostics, "used_center_scan_ids", None)
    center_identity_available = getattr(
        diagnostics, "center_identity_available", True
    )
    expected_scan_ids = getattr(diagnostics, "expected_scan_ids", None)
    filtered_scan_ids = getattr(diagnostics, "filtered_eligible_scan_ids", None)
    if (
        used_scan_ids is not None
        and expected_scan_ids is not None
        and filtered_scan_ids is not None
    ):
        utilization = metrics.scan_utilization_metrics(
            used_scan_ids,
            used_center_scan_ids if center_identity_available else None,
            expected_scan_ids,
            filtered_scan_ids,
            weights,
        )
        records.extend(
            (
                stability("unique_scans_used", utilization.unique_scans_used),
                stability("unique_scans_expected", utilization.unique_scans_expected),
                stability(
                    "unique_scans_filtered_eligible",
                    utilization.unique_scans_filtered_eligible,
                ),
                stability(
                    "scan_utilization_fraction",
                    utilization.scan_utilization_fraction,
                ),
            )
        )
        if utilization.unique_centers_used is not None:
            records.append(
                stability("unique_centers_used", utilization.unique_centers_used)
            )
        if utilization.filtered_scan_utilization_fraction is not None:
            records.append(
                stability(
                    "filtered_scan_utilization_fraction",
                    utilization.filtered_scan_utilization_fraction,
                )
            )
    return records


def _history_mapping(history: Any) -> Mapping[str, Any] | None:
    """Return the history when it is a structurally usable mapping, else None."""
    if not isinstance(history, Mapping):
        return None
    if not isinstance(history.get("series"), Mapping):
        return None
    return history


def _series_values(
    history: Mapping[str, Any], names: tuple[str, ...]
) -> list[float] | None:
    series = history["series"]
    for name in names:
        entry = series.get(name)
        if not isinstance(entry, Mapping):
            continue
        values = entry.get("value")
        if not isinstance(values, (list, tuple)) or not values:
            continue
        try:
            return [float(item) for item in values]
        except (TypeError, ValueError):
            return None
    return None


def _loss_values(history: Mapping[str, Any]) -> list[float] | None:
    name = history.get("train_loss_name")
    if not isinstance(name, str) or not name:
        return None
    return _series_values(history, (f"{name}_epoch", name, f"{name}_step"))


def _validation_loss_values(history: Mapping[str, Any]) -> list[float] | None:
    name = history.get("val_loss_name")
    if not isinstance(name, str) or not name:
        return None
    return _series_values(history, (name, f"{name}_epoch", f"{name}_step"))


def _learning_rate_values(history: Mapping[str, Any]) -> list[float] | None:
    names = tuple(
        sorted(
            name
            for name in history["series"]
            if name.startswith("lr-") or name in {"lr", "learning_rate"}
        )
    )
    return _series_values(history, names)


def _gradient_values(history: Mapping[str, Any]) -> list[float] | None:
    return _series_values(history, _GRADIENT_SERIES_NAMES)


def _json_safe(value: Any) -> Any:
    """Make a history strict-JSON safe; non-finite floats become strings."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def training_history_payload(history: Any) -> dict[str, Any]:
    """Build the persisted training-history artifact payload.

    An absent or unusable history yields a typed absence record; it is never
    fabricated from other evidence.
    """
    mapping = _history_mapping(history)
    if mapping is None:
        return {
            "schema_version": _TRAINING_HISTORY_SCHEMA,
            "available": False,
            "reason": "trainer returned no training history",
        }
    return {
        "schema_version": _TRAINING_HISTORY_SCHEMA,
        "available": True,
        "history": _json_safe(mapping),
    }


def training_history_records(
    history: Any,
    *,
    checkpoint_sha256: str | None = None,
    checkpoint_epoch: int | None = None,
) -> tuple[MetricRecord, ...]:
    """Publish flattened training-stability operands from the history.

    Operand names (gate targets): ``stability.loss_final``,
    ``stability.loss_all_finite``, ``stability.gradient_norm_max``,
    ``stability.gradient_norm_final``, ``stability.gradient_norm_all_finite``,
    ``stability.clip_fraction``. Missing histories publish nothing.
    """
    def stability(name: str, value: float) -> MetricRecord:
        return build_metric_record(
            f"stability.{name}",
            value,
            basis="training_history",
            alignment=_NO_ALIGNMENT,
        )

    records: list[MetricRecord] = []
    if checkpoint_sha256 is not None:
        checkpoint_valid = (
            isinstance(checkpoint_sha256, str)
            and len(checkpoint_sha256) == 64
            and all(ch in "0123456789abcdef" for ch in checkpoint_sha256)
        )
        records.append(stability("checkpoint_identity_present", float(checkpoint_valid)))
    if type(checkpoint_epoch) is int and checkpoint_epoch >= 0:
        records.append(stability("checkpoint_epoch", float(checkpoint_epoch)))

    mapping = _history_mapping(history)
    if mapping is None:
        return tuple(records)

    losses = _loss_values(mapping)
    if losses:
        finite_losses = [value for value in losses if math.isfinite(value)]
        records.append(
            stability(
                "loss_all_finite",
                1.0 if len(finite_losses) == len(losses) else 0.0,
            )
        )
        if math.isfinite(losses[-1]):
            records.append(stability("loss_final", losses[-1]))
    validation_losses = _validation_loss_values(mapping)
    if validation_losses:
        finite_validation = all(math.isfinite(value) for value in validation_losses)
        records.append(stability("validation_loss_all_finite", float(finite_validation)))
        if math.isfinite(validation_losses[-1]):
            records.append(stability("validation_loss_final", validation_losses[-1]))
        if len(validation_losses) >= 4 and finite_validation:
            convergence = metrics.convergence_metrics(validation_losses)
            records.extend(
                (
                    stability(
                        "validation_loss_tail_relative_improvement",
                        convergence.tail_relative_improvement,
                    ),
                    stability(
                        "validation_loss_tail_normalized_slope",
                        convergence.normalized_tail_slope,
                    ),
                    stability(
                        "validation_budget_boundary_improving",
                        convergence.budget_boundary_improving,
                    ),
                )
            )
    learning_rates = _learning_rate_values(mapping)
    if learning_rates and all(
        math.isfinite(value) and value >= 0.0 for value in learning_rates
    ):
        records.extend(
            (
                stability("learning_rate_initial", learning_rates[0]),
                stability("learning_rate_final", learning_rates[-1]),
                stability("learning_rate_min", min(learning_rates)),
                stability(
                    "learning_rate_reduction_count",
                    float(
                        sum(
                            current < previous
                            for previous, current in zip(
                                learning_rates, learning_rates[1:]
                            )
                        )
                    ),
                ),
            )
        )
    gradients = _gradient_values(mapping)
    if gradients:
        finite_gradients = [value for value in gradients if math.isfinite(value)]
        records.append(
            stability(
                "gradient_norm_all_finite",
                1.0 if len(finite_gradients) == len(gradients) else 0.0,
            )
        )
        if finite_gradients:
            records.append(stability("gradient_norm_max", max(finite_gradients)))
            clip_value = mapping.get("gradient_clip_val")
            clip_fraction = 0.0
            if (
                isinstance(clip_value, (int, float))
                and not isinstance(clip_value, bool)
                and clip_value > 0
            ):
                clip_fraction = sum(
                    1 for value in finite_gradients if value > float(clip_value)
                ) / len(finite_gradients)
            records.append(stability("clip_fraction", clip_fraction))
        if math.isfinite(gradients[-1]):
            records.append(stability("gradient_norm_final", gradients[-1]))
    return tuple(records)


def history_report_curves(
    history: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (training_loss, gradient_norm) report curves; finite-only."""
    mapping = _history_mapping(history)
    if mapping is None:
        return (), ()

    def finite_curve(values: list[float] | None) -> tuple[float, ...]:
        if not values or not all(math.isfinite(value) for value in values):
            return ()
        return tuple(values)

    return finite_curve(_loss_values(mapping)), finite_curve(_gradient_values(mapping))


def stored_history_curves(
    attempt: Path,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Rebuild report curves from a completed attempt's stored history artifact."""
    path = Path(attempt) / TRAINING_HISTORY_ARTIFACT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (), ()
    if not isinstance(payload, dict) or payload.get("available") is not True:
        return (), ()
    return history_report_curves(payload.get("history"))


def _runtime_records(result: CanonicalRunResult) -> list[MetricRecord]:
    """Build timings and the canonical single-device CUDA allocator peak."""
    diagnostics = result.reloaded_diagnostics

    def runtime_record(
        name: str, value: Any, *, basis: str = "wall_clock"
    ) -> MetricRecord:
        return build_metric_record(
            f"runtime.{name}", value, basis=basis, alignment=_NO_ALIGNMENT
        )

    records = [
        runtime_record("train_seconds", result.train_seconds),
        runtime_record("inference_seconds", diagnostics.inference_time),
        runtime_record("assembly_seconds", diagnostics.assembly_time),
    ]
    if result.peak_memory_bytes is not None:
        records.append(
            runtime_record(
                "peak_memory_bytes",
                result.peak_memory_bytes,
                basis="framework_cuda_allocator",
            )
        )
    return records


def _image_records(
    result: CanonicalRunResult, truth_role: str, truth_array: np.ndarray
) -> tuple[list[MetricRecord], dict[str, np.ndarray]]:
    diagnostics = result.reloaded_diagnostics
    canvas = _as_array(result.reloaded_canvas)
    prepared = metrics.prepare_anchor_aligned(
        canvas,
        _as_array(diagnostics.canvas_weights),
        diagnostics.canvas_anchor,
        truth_array,
    )
    after_quality = metrics.image_quality_metrics(prepared)
    records = [
        metrics.build_image_metric_record(
            "amp_pearson",
            after_quality.amplitude_pearson,
            truth_role=truth_role,
            basis="raw_amplitude",
            alignment=_ANCHOR_ALIGNMENT,
        ),
        metrics.build_image_metric_record(
            "amp_ssim",
            after_quality.amplitude_ssim,
            truth_role=truth_role,
            basis="mean_scaled_amplitude",
            alignment="anchor_common_mask_largest_valid_rectangle",
        ),
    ]
    if truth_role == "object_truth":
        before_prepared = metrics.prepare_anchor_aligned(
            _as_array(result.reloaded_texture),
            _as_array(diagnostics.canvas_weights),
            diagnostics.canvas_anchor,
            truth_array,
        )
        staged = metrics.varpro_quality_metrics(before_prepared, prepared)
        for stage, quality in (("pre_varpro", staged.before), ("post_varpro", staged.after)):
            for name, value in (
                ("amp_pearson", quality.amplitude_pearson),
                ("amp_ssim", quality.amplitude_ssim),
                ("phase_ssim", quality.phase_ssim),
                ("phase_wrapped_mae", quality.phase_wrapped_mae),
            ):
                records.append(
                    metrics.build_metric_record(
                        f"truth_quality.{stage}.{name}",
                        value,
                        truth_role=truth_role,
                        basis="gauge_normalized_structure",
                        alignment="anchor_common_mask_global_phase",
                    )
                )
        for name, value in metrics.absolute_scale_metrics(prepared).items():
            records.append(
                metrics.build_image_metric_record(
                    name,
                    value,
                    truth_role=truth_role,
                    basis="absolute_amplitude",
                    alignment=f"{_ANCHOR_ALIGNMENT}_global_phase_only",
                )
            )
    mask = prepared.common_mask
    reconstruction = np.abs(canvas).astype(np.float64)
    target = np.where(mask, np.abs(prepared.target), 0.0).astype(np.float64)
    error = np.where(mask, np.abs(reconstruction - np.abs(prepared.target)), 0.0)
    arrays = {
        "reconstruction": reconstruction,
        "target": target,
        "error": error.astype(np.float64),
        "common_valid_mask": mask.astype(np.bool_),
    }
    return records, arrays


def _checkpoint_metric_records(
    resolved: Any,
    result: CanonicalRunResult | MilestoneRunResult,
    descriptor: DatasetDescriptor,
    *,
    runtime: bool,
) -> tuple[tuple[MetricRecord, ...], dict[str, np.ndarray]]:
    """Assemble the shared reconstruction/count records for one checkpoint."""
    records: list[MetricRecord] = []
    if resolved.ci_scaling_active:
        records.extend(_measurement_records(result))
        records.extend(_truth_forward_poisson_oracle_records(descriptor, result))
    records.extend(_stability_records(result))
    records.extend(
        training_history_records(
            result.training_history,
            checkpoint_sha256=getattr(
                result,
                "best_checkpoint_sha256",
                getattr(result, "checkpoint_sha256", None),
            ),
            checkpoint_epoch=getattr(
                result,
                "best_checkpoint_epoch",
                getattr(result, "checkpoint_epoch", None),
            ),
        )
    )
    if runtime:
        records.extend(_runtime_records(result))
    arrays: dict[str, np.ndarray] = {
        "reconstruction": np.abs(_as_array(result.reloaded_canvas)).astype(np.float64)
    }
    truth_array = load_truth_array(descriptor)
    if truth_array is not None:
        image_records, image_arrays = _image_records(
            result, descriptor.truth, truth_array
        )
        records.extend(image_records)
        arrays = image_arrays
    return tuple(records), arrays


def build_run_metric_records(
    resolved: Any, result: CanonicalRunResult, descriptor: DatasetDescriptor
) -> tuple[tuple[MetricRecord, ...], dict[str, np.ndarray]]:
    """Assemble all typed metric records and plot arrays for one run."""
    return _checkpoint_metric_records(resolved, result, descriptor, runtime=True)


def build_milestone_metric_records(
    resolved: Any, result: MilestoneRunResult, descriptor: DatasetDescriptor
) -> tuple[tuple[MetricRecord, ...], dict[str, np.ndarray]]:
    """Build the ordinary checkpoint metrics for one milestone result."""
    return _checkpoint_metric_records(resolved, result, descriptor, runtime=False)

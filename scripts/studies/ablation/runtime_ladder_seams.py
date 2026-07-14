"""Wired evidence seams for the bridge-ladder execution flow (Task 21b).

Task 21a declared these as fail-closed integration points; this revision
wires them to the canonical plumbing:

- ``apply_varpro_to_canvas``: the study runtime's CI VarPro machinery
  (``reassembly.compute_varpro_basis`` + ``VarProScaler`` +
  ``apply_varpro_canvas_scaling``), fed by the runner flow's own predicted
  patches and measured counts.
- ``compute_count_consistency``: the design's published count-space metric
  ``sqrt(sum((prediction - measured)^2) / sum(measured^2))`` with
  ``prediction = s1^2*X1 + s2^2*X2 + s1*s2*X3`` — the exact formula of
  ``reassembly.evaluate_fitted_count_metrics``.
- ``collect_scan_accounting``: the Task 19 reduction
  (``metrics.scan_utilization_metrics``) over the mmap loader's grouping
  record, expanded to the plan-mandated 8-field schema.
- ``resolve_normalization_reuse``: canonical-JSON hashes of the measured
  training/inference normalization records; the historical
  recomputed-held-out deviation is representable as
  ``recomputed_from_heldout=true`` (reuse=False), never a crash.

Missing evidence still fails closed; nothing here fabricates records.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np

from .metrics import scan_utilization_metrics
from .runtime_errors import RuntimeExecutionError

__all__ = [
    "apply_varpro_to_canvas",
    "build_inference_normalization_record",
    "collect_scan_accounting",
    "compute_count_consistency",
    "observe_inference_input_scales",
    "resolve_count_scale_operands",
    "resolve_normalization_reuse",
    "trivial_grouping_record",
]


def _channelled(array: np.ndarray, name: str) -> np.ndarray:
    """Coerce (B, H, W) / (B, H, W, C) / (B, C, H, W) inputs to (B, C, H, W)."""
    value = np.asarray(array)
    if value.ndim == 3:
        return value[:, np.newaxis]
    if value.ndim == 4:
        # Channel-last (runner dict convention) when the trailing axis is the
        # small grouping axis and the middle axes are the detector.
        if value.shape[1] == value.shape[2] and value.shape[3] != value.shape[2]:
            return np.moveaxis(value, -1, 1)
        return value
    raise RuntimeExecutionError(
        "count_consistency", f"{name} must be 3D or 4D; got shape {value.shape}"
    )


def _varpro_basis(probe: Any, patches: np.ndarray) -> tuple[Any, Any, Any, Any]:
    """Mode-summed VarPro basis from the canonical reassembly entry point."""
    import torch

    from ptycho_torch import reassembly

    textures = _channelled(patches, "patches")
    if not np.iscomplexobj(textures):
        raise RuntimeExecutionError(
            "count_consistency", "predicted patches must be complex textures"
        )
    tex = torch.as_tensor(np.ascontiguousarray(textures), dtype=torch.complex64)
    batch, channels, height, width = tex.shape
    probe_t = torch.as_tensor(
        np.ascontiguousarray(np.squeeze(np.asarray(probe))), dtype=torch.complex64
    )
    if probe_t.ndim != 2:
        raise RuntimeExecutionError(
            "count_consistency", f"probe must be 2D; got {tuple(probe_t.shape)}"
        )
    probe_b = probe_t.reshape(1, 1, 1, height, width).expand(
        batch, channels, 1, height, width
    )
    _, _, x1, x2, x3 = reassembly.compute_varpro_basis(probe_b, tex.real, tex.imag)
    return tex, x1, x2, x3


def apply_varpro_to_canvas(
    canvas: np.ndarray,
    test_data: Mapping[str, Any],
    runner_cfg: Any,
    patches: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """CI VarPro scaling of the gated canvas (canonical study machinery).

    Accumulates the ``VarProScaler`` sufficient statistics from the runner
    flow's own predicted patches and the measured count intensities, then
    applies ``reassembly.apply_varpro_canvas_scaling`` to the complex canvas
    — exactly the ``reconstruct_image_barycentric`` usage
    (``scaler.accumulate_batch_from_basis`` + LBFGS solve).
    """
    import torch

    from ptycho_torch import reassembly

    measured = test_data.get("diffraction")
    if measured is None:
        raise RuntimeExecutionError(
            "varpro", "VarPro scaling requires the measured count intensities"
        )
    _, x1, x2, x3 = _varpro_basis(test_data.get("probeGuess"), patches)
    measured_t = torch.as_tensor(
        _channelled(np.asarray(measured, dtype=np.float32), "diffraction")
    )
    scaler = reassembly.VarProScaler(torch.device("cpu"))
    scaler.accumulate_batch_from_basis(measured_t, x1, x2, x3)
    canvas_t = torch.as_tensor(np.asarray(canvas))
    scaled, s1, s2 = reassembly.apply_varpro_canvas_scaling(
        canvas_t, scaler, enabled=True, verbose=False
    )
    return scaled.cpu().numpy(), float(s1), float(s2)


def compute_count_consistency(
    measured: np.ndarray,
    probe: Any,
    patches: np.ndarray,
    *,
    s1: float,
    s2: float,
) -> dict[str, Any]:
    """Physical count-space relative L2 intensity error (design contract).

    ``prediction = s1^2*X1 + s2^2*X2 + s1*s2*X3`` over the canonical VarPro
    basis; ``relative_l2_intensity_error = sqrt(sum((prediction -
    measured)^2) / sum(measured^2))`` — the exact reduction published by the
    study runtime (``reassembly.evaluate_fitted_count_metrics``).
    """
    _, x1, x2, x3 = _varpro_basis(probe, patches)
    prediction = (
        float(s1) ** 2 * x1.double()
        + float(s2) ** 2 * x2.double()
        + float(s1) * float(s2) * x3.double()
    ).cpu().numpy()
    measured64 = _channelled(np.asarray(measured, dtype=np.float64), "measured")
    if measured64.shape != prediction.shape:
        raise RuntimeExecutionError(
            "count_consistency",
            f"measured intensities {measured64.shape} do not match the "
            f"prediction {prediction.shape}",
        )
    measured_square_sum = float(np.sum(measured64**2))
    if measured_square_sum <= 0.0:
        raise RuntimeExecutionError(
            "count_consistency", "measured intensities carry no energy"
        )
    residual = prediction - measured64
    relative = float(np.sqrt(np.sum(residual**2) / measured_square_sum))
    return {
        "relative_l2_intensity_error": relative,
        "basis": "physical_count_space",
        "s1": float(s1),
        "s2": float(s2),
        "n_samples": int(measured64.shape[0]),
        "n_pixels": int(measured64.size),
    }


def collect_scan_accounting(
    grouping_record: Mapping[str, Any], canvas_weights: np.ndarray
) -> dict[str, Any]:
    """Plan-mandated full scan accounting from the loader's grouping record.

    Reduces through the Task 19 machinery
    (``metrics.scan_utilization_metrics``) and expands to the 8-field schema:
    every source scan, duplicate use, group count, accepted patch, and
    reconstructed pixel plus utilization/coverage fractions.
    """
    if not isinstance(grouping_record, Mapping):
        raise RuntimeExecutionError(
            "scan_accounting", "grouping record is missing; cannot account scans"
        )
    try:
        expected = list(grouping_record["expected_scan_ids"])
        filtered = list(grouping_record["filtered_eligible_scan_ids"])
        used = list(grouping_record["used_scan_ids"])
        group_count = int(grouping_record["group_count"])
        slots = int(grouping_record["participant_slots"])
    except KeyError as error:
        raise RuntimeExecutionError(
            "scan_accounting", f"grouping record is missing {error}"
        ) from error
    weights = np.asarray(canvas_weights, dtype=np.float64)
    utilization = scan_utilization_metrics(used, None, expected, filtered, weights)
    return {
        "unique_scans_used": utilization.unique_scans_used,
        "unique_scans_expected": utilization.unique_scans_expected,
        "duplicate_scan_uses": int(slots - utilization.unique_scans_used),
        "group_count": group_count,
        "accepted_patch_count": slots,
        "reconstructed_pixel_count": int(np.count_nonzero(weights > 0.0)),
        "scan_utilization_fraction": utilization.scan_utilization_fraction,
        "canvas_coverage_fraction": utilization.canvas_coverage_fraction,
        # Grouping slots whose KDTree sentinel (-1) wrapped to the last scan.
        "sentinel_wrapped_slots": int(
            grouping_record.get("sentinel_wrapped_slots", 0)
        ),
    }


def _record_sha256(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {key: record[key] for key in record},
        sort_keys=True,
        separators=(",", ":"),
        default=float,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_normalization_reuse(
    training_payload: Mapping[str, Any],
    inference_record: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Normalization-statistics reuse evidence (plan checkbox 6).

    The training record is the loader's measured normalization state; the
    inference record declares the statistics the inference path applied and
    whether any were recomputed from held-out measurements. Returns
    ``(inference_reuses_training, training_sha256, inference_sha256)`` —
    ``False`` (a measured deviation, not a crash) when the inference path
    recomputed held-out statistics.
    """
    record = training_payload.get("training_normalization")
    if not isinstance(record, Mapping):
        raise RuntimeExecutionError(
            "normalization_evidence",
            "the training path did not record training_normalization; the "
            "loader evidence extraction must run before normalization-gated "
            "rungs",
        )
    if not isinstance(inference_record, Mapping):
        raise RuntimeExecutionError(
            "normalization_evidence", "inference normalization record is missing"
        )
    reuse = not bool(inference_record.get("recomputed_from_heldout", False))
    return reuse, _record_sha256(record), _record_sha256(inference_record)


def resolve_count_scale_operands(
    config: Mapping[str, Any],
    *,
    varpro_s1: float | None,
    varpro_s2: float | None,
    model: Any,
    training_payload: Mapping[str, Any],
) -> tuple[float, float, str]:
    """(s1, s2) operands for the count-consistency reduction, fail-closed.

    VarPro-fitted scales when the rung applied VarPro; the model's trained
    rectangular scales under rectangular_scaled; otherwise the resolved
    physics scaling constant (isotropic s1 = s2 = S).
    """
    if varpro_s1 is not None and varpro_s2 is not None:
        return float(varpro_s1), float(varpro_s2), "varpro_fit"
    if config["physics_forward_mode"] == "rectangular_scaled":
        modules = getattr(model, "modules", None)
        scaler = None
        if callable(modules):
            scaler = next(
                (
                    module
                    for module in modules()
                    if type(module).__name__ == "RectangularScaledDiffraction"
                ),
                None,
            )
        if scaler is None:
            raise RuntimeExecutionError(
                "count_consistency",
                "rectangular_scaled model exposes no RectangularScaledDiffraction "
                "module; cannot read the trained s1/s2 scales",
            )
        s1 = float(np.asarray(scaler.s1.detach().cpu().numpy()).reshape(-1)[0])
        s2 = float(np.asarray(scaler.s2.detach().cpu().numpy()).reshape(-1)[0])
        return s1, s2, "model_rect_parameters"
    constant = training_payload.get("physics_scaling_constant")
    if constant is None:
        raise RuntimeExecutionError(
            "count_consistency",
            "no count scaling operand source: neither a VarPro fit, trained "
            "rectangular parameters, nor a recorded physics scaling constant "
            "is available — refusing to default to unit scales silently",
        )
    value = float(constant)
    return value, value, "physics_scaling_constant"


def trivial_grouping_record(count: int) -> dict[str, Any]:
    """Ungrouped (C=1) accounting record: every scan is its own group."""
    ids = [int(i) for i in range(count)]
    return {
        "expected_scan_ids": ids,
        "filtered_eligible_scan_ids": ids,
        "used_scan_ids": ids,
        "group_count": count,
        "participant_slots": count,
    }


@contextmanager
def observe_inference_input_scales(model: Any) -> Any:
    """Record the input scale factors the REAL inference call consumes.

    Instruments ``model.forward_predict`` (the runner's inference entry) so
    the normalization evidence is MEASURED from the actual call, never
    declared. Restores the original attribute on exit.
    """
    observed: list[float] = []
    original = getattr(model, "forward_predict", None)
    if original is None:
        yield observed
        return

    def _spy(x: Any, positions: Any, probe: Any, input_scale_factor: Any) -> Any:
        values = np.unique(
            np.asarray(input_scale_factor.detach().cpu(), dtype=np.float64)
        )
        observed.extend(float(value) for value in values)
        return original(x, positions, probe, input_scale_factor)

    setattr(model, "forward_predict", _spy)
    try:
        yield observed
    finally:
        try:
            delattr(model, "forward_predict")
        except AttributeError:
            pass


def build_inference_normalization_record(
    observed_scales: list[float], test_data: Mapping[str, Any]
) -> dict[str, Any]:
    """Inference-side normalization record from the observed scale factors.

    Only a unit-constant observation attests that no held-out statistic was
    consumed; any other observation is conservatively recorded as a
    recomputed/deviating scale (a measured deviation, which FAILs the rung
    until explained). No observation at all fails closed.
    """
    if not observed_scales:
        raise RuntimeExecutionError(
            "normalization_evidence",
            "no inference input scale was observed on the real forward call; "
            "cannot attest normalization reuse",
        )
    unique = sorted({round(float(value), 12) for value in observed_scales})
    unit = unique == [1.0]
    norm_value = test_data.get("norm_Y_I")
    return {
        "source": "observed_forward_predict_input_scale",
        "observed_input_scale_values": unique[:8],
        "input_scale_is_unit_constant": unit,
        "recomputed_from_heldout": not unit,
        "stitch_norm_Y_I": (
            None if norm_value is None else float(np.asarray(norm_value))
        ),
    }

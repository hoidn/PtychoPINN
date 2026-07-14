"""Sealed per-rung evidence and the derived ladder report (plan Task 21).

Rung evidence is canonical JSON sealed exactly once (never overwritten in
place); its identity is the SHA-256 of the written bytes, recomputed by the
consumer-side parser rather than declared by the producer. The ladder report
is a derived artifact regenerated from sealed evidence on every invocation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .runtime_errors import RuntimeExecutionError
from .runtime_ladder_config import (
    LOCKED_MAX_ABS_AMP_SSIM_DELTA,
    LOCKED_MAX_ABS_PHASE_SSIM_DELTA,
    absolute_ssim_delta,
)
from .runtime_ladder_gate_policy import (
    adjudicate_absolute_ssim_delta,
    adjudicate_retained_ssim,
)
from .runtime_reference_execution import DECLARED_GAUGE_HANDLING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runtime_ladder_execution import LadderRunResult

__all__ = [
    "LADDER_EVIDENCE_SCHEMA_VERSION",
    "LADDER_REPORT_SCHEMA_VERSION",
    "assemble_rung_evidence",
    "parse_sealed_rung_evidence",
    "seal_rung_evidence",
    "write_ladder_report",
]

LADDER_EVIDENCE_SCHEMA_VERSION = "bridge_ladder_rung_evidence_v2"
HISTORICAL_LADDER_EVIDENCE_SCHEMA_VERSION = "bridge_ladder_rung_evidence_v1"
LADDER_REPORT_SCHEMA_VERSION = "grid_lines_bridge_ladder_report_v2"


def assemble_rung_evidence(
    spec: Any,
    rung: Any,
    result: "LadderRunResult",
    control: Any,
    recorded_differences: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the sealed per-rung evidence payload (gate section added later)."""
    if result.rung_id != rung.id:
        raise RuntimeExecutionError(
            "evidence", f"result rung {result.rung_id!r} does not match {rung.id!r}"
        )
    if result.gauge_handling != DECLARED_GAUGE_HANDLING:
        raise RuntimeExecutionError(
            "gauge_handling",
            f"gauge handling {result.gauge_handling!r} is undeclared; only "
            f"{DECLARED_GAUGE_HANDLING!r} is permitted",
        )
    if not result.no_resize_asserted:
        raise RuntimeExecutionError(
            "no_resize", "ladder evidence requires the no-resize assertion"
        )
    materialized = result.materialized
    return {
        "schema_version": LADDER_EVIDENCE_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "rung_id": rung.id,
        "group": rung.group,
        "control": {
            "rung_id": control.rung_id,
            "evidence_sha256": control.evidence_sha256,
            "amp_ssim": control.amp_ssim,
            "phase_ssim": control.phase_ssim,
        },
        "resolved_config": dict(result.resolved_config),
        "dataset": {
            "id": rung.dataset,
            "recipe_fingerprint_sha256": materialized.recipe_fingerprint_sha256,
            "train_sha256": materialized.train_sha256,
            "test_sha256": materialized.test_sha256,
            "probe_sha256": materialized.probe_sha256,
            "n_train": materialized.n_train,
            "n_test": materialized.n_test,
        },
        "checkpoint_sha256": result.checkpoint_sha256,
        "pre_stitch_patch_sha256": result.pre_stitch_patch_sha256,
        "historical_canvas_sha256": result.historical_canvas_sha256,
        "generic_canvas_sha256": result.generic_canvas_sha256,
        "historical_mask_sha256": result.historical_mask_sha256,
        "generic_mask_sha256": result.generic_mask_sha256,
        "canvases_equivalent": result.canvases_equivalent,
        "masks_equivalent": result.masks_equivalent,
        "no_resize_asserted": result.no_resize_asserted,
        "gauge_handling": result.gauge_handling,
        "gated_evaluator": result.gated_evaluator,
        "effective_probe_sha256": result.effective_probe_sha256,
        "effective_probe_matches_recipe": result.effective_probe_matches_recipe,
        "recorded_differences": recorded_differences,
        "metrics": {
            "amp_mae": result.amp_mae,
            "phase_mae": result.phase_mae,
            "amp_ssim": result.amp_ssim,
            "phase_ssim": result.phase_ssim,
        },
        "normalization": {
            "inference_reuses_training_normalization": (
                result.inference_reuses_training_normalization
            ),
            "training_normalization_sha256": result.training_normalization_sha256,
            "inference_normalization_sha256": result.inference_normalization_sha256,
        },
        "varpro": {
            "applied": result.varpro_applied,
            "s1": result.varpro_s1,
            "s2": result.varpro_s2,
        },
        "scan_accounting": (
            None if result.scan_accounting is None else dict(result.scan_accounting)
        ),
        # Plan checkbox 2: coverage on every rung; physical count-space error
        # on count rungs; scaling quantities beyond the VarPro s1/s2.
        "canvas_coverage_fraction": result.canvas_coverage_fraction,
        "count_consistency": (
            None
            if result.count_consistency is None
            else dict(result.count_consistency)
        ),
        "count_scaling": {
            "mode": str(result.resolved_config["count_scale_mode"]),
            "physics_scaling_constant": result.physics_scaling_constant,
        },
    }


def seal_rung_evidence(payload: Mapping[str, Any], path: Path) -> str:
    """Write canonical JSON bytes and return their recomputed SHA-256."""
    evidence_path = Path(path)
    if evidence_path.exists():
        raise RuntimeExecutionError(
            "evidence_seal",
            f"refusing to overwrite existing sealed rung evidence at "
            f"{evidence_path}",
        )
    data = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(data)
    written = evidence_path.read_bytes()
    if written != data:
        raise RuntimeExecutionError(
            "evidence_seal",
            f"sealed rung evidence readback mismatch at {evidence_path}",
        )
    return hashlib.sha256(written).hexdigest()


def parse_sealed_rung_evidence(data: bytes) -> tuple[dict[str, Any], str]:
    """Reparse sealed rung evidence, recomputing its seal hash from the bytes."""
    if not isinstance(data, bytes) or not data:
        raise RuntimeExecutionError(
            "evidence_parse", "sealed rung evidence must be nonempty bytes"
        )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeExecutionError(
            "evidence_parse", f"sealed rung evidence is invalid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeExecutionError(
            "evidence_parse", "sealed rung evidence must be a JSON object"
        )
    version = payload.get("schema_version")
    if version not in {
        LADDER_EVIDENCE_SCHEMA_VERSION,
        HISTORICAL_LADDER_EVIDENCE_SCHEMA_VERSION,
    }:
        raise RuntimeExecutionError(
            "evidence_parse",
            f"sealed rung evidence schema_version {version!r} is not "
            f"a supported ladder evidence schema",
        )
    if not isinstance(payload.get("rung_id"), str) or not payload["rung_id"]:
        raise RuntimeExecutionError(
            "evidence_parse", "sealed rung evidence must carry a rung_id"
        )
    if version == LADDER_EVIDENCE_SCHEMA_VERSION:
        _validate_v2_gate(payload.get("gate"))
    return payload, hashlib.sha256(data).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_v2_gate(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeExecutionError(
            "evidence_parse", "v2 sealed rung evidence must carry a gate object"
        )
    policy = value.get("policy")
    if policy == "absolute_ssim_delta_v1":
        required = {
            "policy",
            "threshold_provenance",
            "control",
            "current",
            "abs_amp_delta",
            "abs_phase_delta",
            "max_abs_amp_ssim_delta",
            "max_abs_phase_ssim_delta",
            "protocol_failure_reason",
            "verdict",
            "reason",
        }
        if set(value) != required:
            raise RuntimeExecutionError(
                "evidence_parse",
                "absolute gate evidence has missing or unexpected fields",
            )
        for operand_name in ("control", "current"):
            operand = value[operand_name]
            if not isinstance(operand, Mapping) or set(operand) != {
                "amp_ssim",
                "phase_ssim",
            }:
                raise RuntimeExecutionError(
                    "evidence_parse", f"absolute gate {operand_name} is malformed"
                )
            if not all(_finite_number(item) for item in operand.values()):
                raise RuntimeExecutionError(
                    "evidence_parse", f"absolute gate {operand_name} is nonfinite"
                )
        numeric = (
            "abs_amp_delta",
            "abs_phase_delta",
            "max_abs_amp_ssim_delta",
            "max_abs_phase_ssim_delta",
        )
        if any(not _finite_number(value[name]) or value[name] < 0 for name in numeric):
            raise RuntimeExecutionError(
                "evidence_parse", "absolute gate deltas and thresholds are invalid"
            )
        if value.get("threshold_provenance") != "locked":
            raise RuntimeExecutionError(
                "evidence_parse", "absolute gate threshold provenance is not locked"
            )
        if (
            float(value["max_abs_amp_ssim_delta"])
            != LOCKED_MAX_ABS_AMP_SSIM_DELTA
            or float(value["max_abs_phase_ssim_delta"])
            != LOCKED_MAX_ABS_PHASE_SSIM_DELTA
        ):
            raise RuntimeExecutionError(
                "evidence_parse", "absolute gate thresholds are not locked values"
            )
        expected_amp = absolute_ssim_delta(
            value["current"]["amp_ssim"], value["control"]["amp_ssim"]
        )
        expected_phase = absolute_ssim_delta(
            value["current"]["phase_ssim"], value["control"]["phase_ssim"]
        )
        if (
            float(value["abs_amp_delta"]) != expected_amp
            or float(value["abs_phase_delta"]) != expected_phase
        ):
            raise RuntimeExecutionError(
                "evidence_parse", "absolute gate delta does not match its operands"
            )
        try:
            adjudication = adjudicate_absolute_ssim_delta(
                current_amp_ssim=float(value["current"]["amp_ssim"]),
                current_phase_ssim=float(value["current"]["phase_ssim"]),
                control_amp_ssim=float(value["control"]["amp_ssim"]),
                control_phase_ssim=float(value["control"]["phase_ssim"]),
                max_abs_amp_ssim_delta=float(value["max_abs_amp_ssim_delta"]),
                max_abs_phase_ssim_delta=float(value["max_abs_phase_ssim_delta"]),
                protocol_failure_reason=value["protocol_failure_reason"],
            )
        except ValueError as error:
            raise RuntimeExecutionError(
                "evidence_parse", f"absolute gate protocol is invalid: {error}"
            ) from error
        if (
            value.get("verdict") != adjudication.verdict
            or value.get("reason") != adjudication.reason
        ):
            raise RuntimeExecutionError(
                "evidence_parse",
                "absolute gate verdict/reason is inconsistent with sealed "
                "operands, thresholds, and protocol failure",
            )
    elif policy == "retained_ssim_v1":
        required = {
            "policy",
            "threshold_provenance",
            "control",
            "current",
            "retained_amp_ssim_min_fraction",
            "retained_phase_ssim_min_fraction",
            "absolute_amp_ssim_floor",
            "retained_amp_ssim",
            "retained_phase_ssim",
            "protocol_failure_reason",
            "verdict",
            "reason",
        }
        if set(value) != required:
            raise RuntimeExecutionError(
                "evidence_parse",
                "retained gate evidence has missing or unexpected fields",
            )
        for operand_name in ("control", "current"):
            operand = value[operand_name]
            if not isinstance(operand, Mapping) or set(operand) != {
                "amp_ssim",
                "phase_ssim",
            }:
                raise RuntimeExecutionError(
                    "evidence_parse", f"retained gate {operand_name} is malformed"
                )
            if not all(_finite_number(item) for item in operand.values()):
                raise RuntimeExecutionError(
                    "evidence_parse", f"retained gate {operand_name} is nonfinite"
                )
        numeric = (
            "retained_amp_ssim_min_fraction",
            "retained_phase_ssim_min_fraction",
            "absolute_amp_ssim_floor",
            "retained_amp_ssim",
            "retained_phase_ssim",
        )
        if any(not _finite_number(value[name]) for name in numeric):
            raise RuntimeExecutionError(
                "evidence_parse", "retained gate ratios or thresholds are nonfinite"
            )
        if value.get("threshold_provenance") != "locked":
            raise RuntimeExecutionError(
                "evidence_parse", "retained gate threshold provenance is not locked"
            )
        try:
            adjudication = adjudicate_retained_ssim(
                current_amp_ssim=float(value["current"]["amp_ssim"]),
                current_phase_ssim=float(value["current"]["phase_ssim"]),
                control_amp_ssim=float(value["control"]["amp_ssim"]),
                control_phase_ssim=float(value["control"]["phase_ssim"]),
                retained_amp_ssim_min_fraction=float(
                    value["retained_amp_ssim_min_fraction"]
                ),
                retained_phase_ssim_min_fraction=float(
                    value["retained_phase_ssim_min_fraction"]
                ),
                absolute_amp_ssim_floor=float(value["absolute_amp_ssim_floor"]),
                protocol_failure_reason=value["protocol_failure_reason"],
            )
        except ValueError as error:
            raise RuntimeExecutionError(
                "evidence_parse", f"retained gate is invalid: {error}"
            ) from error
        if (
            float(value["retained_amp_ssim"])
            != adjudication.retained_amp_ssim
            or float(value["retained_phase_ssim"])
            != adjudication.retained_phase_ssim
        ):
            raise RuntimeExecutionError(
                "evidence_parse", "retained gate ratio does not match its operands"
            )
        if (
            value.get("verdict") != adjudication.verdict
            or value.get("reason") != adjudication.reason
        ):
            raise RuntimeExecutionError(
                "evidence_parse",
                "retained gate verdict/reason is inconsistent with sealed "
                f"operands and protocol failure; expected {adjudication.reason!r}",
            )
    else:
        raise RuntimeExecutionError(
            "evidence_parse", f"unsupported sealed gate policy {policy!r}"
        )


DIAGNOSTICS_REPORT_NAME = "ladder_diagnostics_report.json"


def write_diagnostics_report(
    spec: Any,
    output_root: Path,
    baseline: Any,
    entries: list[dict[str, Any]],
) -> Path:
    """Merge-on-write for the diagnostics report (task-21c review S-1).

    Every write path goes through here, killing the demotion class at the
    seam: previously adjudicated entries survive byte-for-byte unless the
    SAME rung is re-adjudicated in this invocation, and placeholders
    (skipped/pending) never enter the report. Entry order follows the spec's
    rung order.
    """
    incoming = {
        entry["id"]: entry
        for entry in entries
        if entry.get("status") == "adjudicated"
    }
    report_path = output_root / DIAGNOSTICS_REPORT_NAME
    existing: dict[str, dict[str, Any]] = {}
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        existing = {
            entry["id"]: entry
            for entry in previous.get("rungs", [])
            if entry.get("status") == "adjudicated"
        }
    merged = {**existing, **incoming}
    spec_order = {rung.id: index for index, rung in enumerate(spec.rungs)}
    ordered = [
        merged[rung_id]
        for rung_id in sorted(
            merged, key=lambda rung_id: spec_order.get(rung_id, len(spec_order))
        )
    ]
    return write_ladder_report(
        spec,
        output_root,
        baseline,
        ordered,
        None,
        report_name=DIAGNOSTICS_REPORT_NAME,
    )


def write_ladder_report(
    spec: Any,
    output_root: Path,
    baseline: Any,
    entries: list[dict[str, Any]],
    first_material_degradation: str | None,
    *,
    report_name: str = "ladder_report.json",
) -> Path:
    """Write a derived ladder report (rewritten on every invocation).

    The chain report (default name) carries only chain rungs; diagnostic
    invocations write ``ladder_diagnostics_report.json`` instead and never
    touch the chain report (task-21c review I3b).
    """
    report = {
        "schema_version": LADDER_REPORT_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "spec": spec.spec_declared,
        "gate": _report_gate(spec.gate),
        "baseline": {
            "id": baseline.rung_id,
            "evidence_sha256": baseline.evidence_sha256,
            "amp_ssim": baseline.amp_ssim,
            "phase_ssim": baseline.phase_ssim,
        },
        "rungs": entries,
        "first_material_degradation": first_material_degradation,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / report_name
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path


def _report_gate(gate: Any) -> dict[str, Any]:
    if gate.policy == "absolute_ssim_delta_v1":
        return {
            "policy": gate.policy,
            "threshold_provenance": gate.threshold_provenance,
            "max_abs_amp_ssim_delta": gate.max_abs_amp_ssim_delta,
            "max_abs_phase_ssim_delta": gate.max_abs_phase_ssim_delta,
        }
    return {
        "policy": gate.policy,
        "threshold_provenance": gate.threshold_provenance,
        "retained_amp_ssim_min_fraction": gate.retained_amp_ssim_min_fraction,
        "retained_phase_ssim_min_fraction": gate.retained_phase_ssim_min_fraction,
        "absolute_amp_ssim_floor": gate.absolute_amp_ssim_floor,
    }

"""Rung gating for the bridge ladder: baseline and retained controls.

Rung 0's sealed reference evidence must hash to the spec's pin and
re-adjudicate PASS under the Task 19/20 bridge harness before it may serve
as the ladder's first control. The active policy gates absolute amplitude and
phase SSIM deltas against that dictionary baseline; historical retained gates
remain available for old specs and diagnostics. Protocol failures override the
metric gate and fail the rung with a dedicated reason.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .runtime_errors import StudyRequestError
from .runtime_ladder_config import (
    ABSOLUTE_GATE_POLICY,
    LADDER_DIFFERENCE_IDS,
    LadderGate,
)
from .runtime_ladder_gate_policy import (
    PROTOCOL_FAILURE_REASONS,
    REASON_INVALIDATING,
    REASON_NORMALIZATION,
    REASON_ABSOLUTE_FLOOR,
    REASON_RETAINED_AMP,
    REASON_RETAINED_PHASE,
    REASON_SCAN_OMISSION,
    REASON_UNCLASSIFIED,
    adjudicate_absolute_ssim_delta,
    adjudicate_retained_ssim,
)
from .runtime_ladder_spec import BridgeLadderSpec, LadderRung
from .runtime_reference import adjudicate_reference, load_reference_spec
from .verdicts import GateResult, IntegrationBridgeEvidence, Verdict

__all__ = [
    "REASON_ABSOLUTE_FLOOR",
    "REASON_INVALIDATING",
    "REASON_NORMALIZATION",
    "REASON_RETAINED_AMP",
    "REASON_RETAINED_PHASE",
    "REASON_SCAN_OMISSION",
    "REASON_UNCLASSIFIED",
    "LadderControl",
    "adjudicate_payload",
    "evaluate_rung_gate",
    "gate_evidence_record",
    "protocol_failure",
    "recorded_differences",
    "verify_baseline",
]

@dataclass(frozen=True)
class LadderControl:
    """The passing control a rung gates against (rung 0 or a passing rung)."""

    rung_id: str
    amp_ssim: float
    phase_ssim: float
    evidence_sha256: str


def verify_baseline(spec: BridgeLadderSpec) -> LadderControl:
    """Verify rung 0: pinned evidence bytes that re-adjudicate PASS."""
    if spec.baseline.status != "current":
        raise StudyRequestError(
            f"baseline {spec.baseline.id!r} is {spec.baseline.status}; current "
            "qualification evidence is unavailable and historical evidence "
            "cannot serve as the rung 0 control"
        )
    assert spec.baseline.evidence_sha256 is not None
    evidence_path = spec.baseline.evidence
    if not evidence_path.is_file():
        raise StudyRequestError(
            f"baseline evidence is missing: {evidence_path} (declared "
            f"{spec.baseline.evidence_declared!r})"
        )
    data = evidence_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.baseline.evidence_sha256:
        raise StudyRequestError(
            f"baseline evidence sha256 {digest} does not match the pinned "
            f"hash {spec.baseline.evidence_sha256}"
        )
    evidence = IntegrationBridgeEvidence.from_sealed_artifact_bytes(data)
    reference_spec = load_reference_spec(
        spec.baseline.reference_spec, base_dir=spec.base_dir
    )
    arm = reference_spec.arm(spec.baseline.reference_id)
    # Qualification evidence is sealed before Task 27 atomically promotes its
    # measured floors. First validate the sealed bytes against their original
    # numeric gates while retaining every current non-floor contract field.
    evidence_floor_bridge = dict(arm.bridge)
    for field in (
        "fixture_amp_mae_max",
        "fixture_phase_mae_max",
        "fixture_amp_ssim_min",
        "fixture_phase_ssim_min",
    ):
        evidence_floor_bridge[field] = getattr(evidence.contract, field)
    result = adjudicate_reference(
        replace(arm, bridge=evidence_floor_bridge), evidence
    )
    if result.verdict is not Verdict.PASS:
        raise StudyRequestError(
            "baseline reference evidence does not re-adjudicate PASS "
            f"(verdict={result.verdict.value if result.verdict else None}, "
            f"reason={result.reason}); the ladder has no valid rung 0 control"
        )
    if (
        evidence.fixture_amp_ssim < float(arm.bridge["fixture_amp_ssim_min"])
        or evidence.fixture_phase_ssim
        < float(arm.bridge["fixture_phase_ssim_min"])
        or evidence.fixture_amp_mae > float(arm.bridge["fixture_amp_mae_max"])
        or evidence.fixture_phase_mae > float(arm.bridge["fixture_phase_mae_max"])
    ):
        raise StudyRequestError(
            "baseline reference evidence does not pass the current re-pinned "
            "SSIM floors and MAE guards; the ladder has no valid rung 0 control"
        )
    return LadderControl(
        rung_id=spec.baseline.id,
        amp_ssim=float(evidence.fixture_amp_ssim),
        phase_ssim=float(evidence.fixture_phase_ssim),
        evidence_sha256=digest,
    )


def evaluate_rung_gate(
    gate: LadderGate,
    rung_id: str,
    *,
    amp_ssim: float,
    phase_ssim: float,
    control: LadderControl,
    failure_reason: str | None = None,
) -> GateResult:
    """Evaluate the configured SSIM gate against its control.

    ``failure_reason`` carries a protocol failure (unclassified difference,
    scan omission, normalization recompute) that overrides the metric gate.
    """
    operands = (amp_ssim, phase_ssim, control.amp_ssim, control.phase_ssim)
    if not all(math.isfinite(float(value)) for value in operands):
        raise StudyRequestError("gate SSIM operands must be finite")
    if gate.policy == ABSOLUTE_GATE_POLICY:
        assert gate.max_abs_amp_ssim_delta is not None
        assert gate.max_abs_phase_ssim_delta is not None
        adjudication = adjudicate_absolute_ssim_delta(
            current_amp_ssim=amp_ssim,
            current_phase_ssim=phase_ssim,
            control_amp_ssim=control.amp_ssim,
            control_phase_ssim=control.phase_ssim,
            max_abs_amp_ssim_delta=gate.max_abs_amp_ssim_delta,
            max_abs_phase_ssim_delta=gate.max_abs_phase_ssim_delta,
            protocol_failure_reason=failure_reason,
        )
        return GateResult.active(
            rung_id,
            Verdict(adjudication.verdict),
            category="ladder_rung",
            reason=adjudication.reason,
            observed=adjudication.observed,
            threshold=adjudication.threshold,
        )
    assert gate.retained_amp_ssim_min_fraction is not None
    assert gate.retained_phase_ssim_min_fraction is not None
    assert gate.absolute_amp_ssim_floor is not None
    try:
        adjudication = adjudicate_retained_ssim(
            current_amp_ssim=amp_ssim,
            current_phase_ssim=phase_ssim,
            control_amp_ssim=control.amp_ssim,
            control_phase_ssim=control.phase_ssim,
            retained_amp_ssim_min_fraction=gate.retained_amp_ssim_min_fraction,
            retained_phase_ssim_min_fraction=gate.retained_phase_ssim_min_fraction,
            absolute_amp_ssim_floor=gate.absolute_amp_ssim_floor,
            protocol_failure_reason=failure_reason,
        )
    except ValueError as error:
        raise StudyRequestError(str(error)) from error
    return GateResult.active(
        rung_id,
        Verdict(adjudication.verdict),
        category="ladder_rung",
        reason=adjudication.reason,
        observed=adjudication.observed,
        threshold=adjudication.threshold,
    )


def gate_evidence_record(
    gate: LadderGate,
    control: LadderControl,
    *,
    amp_ssim: float,
    phase_ssim: float,
    result: GateResult,
) -> dict[str, Any]:
    """Canonical policy operands and verdict sealed with each rung."""
    if gate.policy == ABSOLUTE_GATE_POLICY:
        assert gate.max_abs_amp_ssim_delta is not None
        assert gate.max_abs_phase_ssim_delta is not None
        protocol_reason = (
            result.reason if result.reason in PROTOCOL_FAILURE_REASONS else None
        )
        adjudication = adjudicate_absolute_ssim_delta(
            current_amp_ssim=amp_ssim,
            current_phase_ssim=phase_ssim,
            control_amp_ssim=control.amp_ssim,
            control_phase_ssim=control.phase_ssim,
            max_abs_amp_ssim_delta=gate.max_abs_amp_ssim_delta,
            max_abs_phase_ssim_delta=gate.max_abs_phase_ssim_delta,
            protocol_failure_reason=protocol_reason,
        )
        return {
            "policy": gate.policy,
            "threshold_provenance": gate.threshold_provenance,
            "control": {
                "amp_ssim": control.amp_ssim,
                "phase_ssim": control.phase_ssim,
            },
            "current": {"amp_ssim": amp_ssim, "phase_ssim": phase_ssim},
            "abs_amp_delta": adjudication.amp_delta,
            "abs_phase_delta": adjudication.phase_delta,
            "max_abs_amp_ssim_delta": gate.max_abs_amp_ssim_delta,
            "max_abs_phase_ssim_delta": gate.max_abs_phase_ssim_delta,
            "protocol_failure_reason": protocol_reason,
            "verdict": result.verdict.value if result.verdict else None,
            "reason": result.reason,
        }
    assert gate.retained_amp_ssim_min_fraction is not None
    assert gate.retained_phase_ssim_min_fraction is not None
    assert gate.absolute_amp_ssim_floor is not None
    protocol_reason = result.reason if result.reason in PROTOCOL_FAILURE_REASONS else None
    adjudication = adjudicate_retained_ssim(
        current_amp_ssim=amp_ssim,
        current_phase_ssim=phase_ssim,
        control_amp_ssim=control.amp_ssim,
        control_phase_ssim=control.phase_ssim,
        retained_amp_ssim_min_fraction=gate.retained_amp_ssim_min_fraction,
        retained_phase_ssim_min_fraction=gate.retained_phase_ssim_min_fraction,
        absolute_amp_ssim_floor=gate.absolute_amp_ssim_floor,
        protocol_failure_reason=protocol_reason,
    )
    return {
        "policy": gate.policy,
        "threshold_provenance": gate.threshold_provenance,
        "control": {
            "amp_ssim": control.amp_ssim,
            "phase_ssim": control.phase_ssim,
        },
        "current": {"amp_ssim": amp_ssim, "phase_ssim": phase_ssim},
        "retained_amp_ssim_min_fraction": gate.retained_amp_ssim_min_fraction,
        "retained_phase_ssim_min_fraction": gate.retained_phase_ssim_min_fraction,
        "absolute_amp_ssim_floor": gate.absolute_amp_ssim_floor,
        "protocol_failure_reason": protocol_reason,
        "verdict": result.verdict.value if result.verdict else None,
        "reason": result.reason,
        "retained_amp_ssim": adjudication.retained_amp_ssim,
        "retained_phase_ssim": adjudication.retained_phase_ssim,
    }


def _observed_differences(payload: Mapping[str, Any]) -> set[str]:
    observed: set[str] = set()
    if not payload["canvases_equivalent"]:
        observed.add("canvas_equivalence")
    if not payload["masks_equivalent"]:
        observed.add("mask_equivalence")
    if not payload["effective_probe_matches_recipe"]:
        observed.add("effective_probe_identity")
    return observed & LADDER_DIFFERENCE_IDS


def recorded_differences(
    rung: LadderRung, payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Predeclared classifications for the differences a rung actually shows."""
    return [
        {
            "field": field,
            "classification": rung.expected_differences[field].classification,
            "justification": rung.expected_differences[field].justification,
        }
        for field in sorted(_observed_differences(payload))
        if field in rung.expected_differences
    ]


def _require_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: The plan-mandated full accounting: every source scan, duplicate use,
#: group count, accepted patch, and reconstructed pixel plus coverage.
SCAN_ACCOUNTING_REQUIRED_FIELDS = (
    "unique_scans_used",
    "unique_scans_expected",
    "duplicate_scan_uses",
    "group_count",
    "accepted_patch_count",
    "reconstructed_pixel_count",
    "scan_utilization_fraction",
    "canvas_coverage_fraction",
)


def protocol_failure(rung: LadderRung, payload: Mapping[str, Any]) -> str | None:
    """Protocol failure reason for a rung payload, or None.

    Raises fail-closed when mandatory rung evidence (scan accounting,
    normalization reuse, count consistency, coverage, scaling constants) is
    missing entirely.
    """
    for field in sorted(_observed_differences(payload)):
        declared = rung.expected_differences.get(field)
        if declared is None:
            return REASON_UNCLASSIFIED
        if declared.classification == "comparison_invalidating":
            return REASON_INVALIDATING
    if not _require_number(payload.get("canvas_coverage_fraction")):
        raise StudyRequestError(
            f"rung {rung.id} evidence lacks the mandatory "
            "canvas_coverage_fraction field"
        )
    if rung.requires_scan_accounting:
        accounting = payload.get("scan_accounting")
        if not isinstance(accounting, Mapping):
            raise StudyRequestError(
                f"rung {rung.id} requires full scan accounting evidence "
                "(every source scan, duplicate use, group count, accepted "
                "patch, reconstructed pixel); none was recorded"
            )
        missing = [
            field
            for field in SCAN_ACCOUNTING_REQUIRED_FIELDS
            if not _require_number(accounting.get(field))
        ]
        if missing:
            raise StudyRequestError(
                f"rung {rung.id} scan accounting is missing mandated fields "
                f"{missing}"
            )
        if float(accounting["scan_utilization_fraction"]) < 1.0:
            return REASON_SCAN_OMISSION
    if rung.requires_count_error_evidence:
        consistency = payload.get("count_consistency")
        if not isinstance(consistency, Mapping) or not _require_number(
            consistency.get("relative_l2_intensity_error")
        ):
            raise StudyRequestError(
                f"rung {rung.id} requires physical count-consistency evidence "
                "(relative_l2_intensity_error); none was recorded"
            )
    if payload["resolved_config"].get("count_scale_mode") == "auto":
        scaling = payload.get("count_scaling")
        if not isinstance(scaling, Mapping) or not _require_number(
            scaling.get("physics_scaling_constant")
        ):
            raise StudyRequestError(
                f"rung {rung.id} ran count_scale_mode=auto without recording "
                "the resolved physics scaling constant"
            )
    if rung.requires_normalization_evidence:
        normalization = payload.get("normalization")
        reuse = (
            normalization.get("inference_reuses_training_normalization")
            if isinstance(normalization, Mapping)
            else None
        )
        if reuse is None:
            raise StudyRequestError(
                f"rung {rung.id} requires normalization-statistics reuse "
                "evidence; none was recorded"
            )
        if reuse is not True:
            return REASON_NORMALIZATION
    return None


def adjudicate_payload(
    spec: BridgeLadderSpec,
    rung: LadderRung,
    payload: Mapping[str, Any],
    control: LadderControl,
) -> GateResult:
    """Adjudicate one rung payload: protocol failures, then retained SSIM."""
    failure = protocol_failure(rung, payload)
    metrics = payload["metrics"]
    return evaluate_rung_gate(
        spec.gate,
        rung.id,
        amp_ssim=float(metrics["amp_ssim"]),
        phase_ssim=float(metrics["phase_ssim"]),
        control=control,
        failure_reason=failure,
    )

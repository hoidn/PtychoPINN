"""Pure policy adjudication shared by ladder runtime and evidence parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .runtime_ladder_config import absolute_ssim_delta

REASON_ABSOLUTE_AMP_DELTA = "ladder_absolute_amp_ssim_delta_exceeded"
REASON_ABSOLUTE_PHASE_DELTA = "ladder_absolute_phase_ssim_delta_exceeded"
REASON_RETAINED_AMP = "ladder_retained_amp_ssim_below_threshold"
REASON_RETAINED_PHASE = "ladder_retained_phase_ssim_below_threshold"
REASON_ABSOLUTE_FLOOR = "ladder_absolute_amp_ssim_floor_failed"
REASON_UNCLASSIFIED = "ladder_unclassified_difference"
REASON_INVALIDATING = "ladder_comparison_invalidating_difference"
REASON_SCAN_OMISSION = "ladder_scan_omission"
REASON_NORMALIZATION = "ladder_normalization_not_reused"

PROTOCOL_FAILURE_REASONS = frozenset(
    {
        REASON_UNCLASSIFIED,
        REASON_INVALIDATING,
        REASON_SCAN_OMISSION,
        REASON_NORMALIZATION,
    }
)


@dataclass(frozen=True)
class AbsoluteGateAdjudication:
    verdict: str
    reason: str | None
    amp_delta: float
    phase_delta: float
    observed: float | None
    threshold: float | None


@dataclass(frozen=True)
class RetainedGateAdjudication:
    verdict: str
    reason: str | None
    retained_amp_ssim: float
    retained_phase_ssim: float
    observed: float | None
    threshold: float | None


def _validate_protocol_failure(
    protocol_failure_reason: str | None, *, policy: str
) -> None:
    if protocol_failure_reason is None:
        return
    if not isinstance(protocol_failure_reason, str):
        raise ValueError(f"{policy} gate protocol failure must be text or null")
    if protocol_failure_reason not in PROTOCOL_FAILURE_REASONS:
        raise ValueError(
            f"unknown {policy} gate protocol failure {protocol_failure_reason!r}"
        )


def adjudicate_absolute_ssim_delta(
    *,
    current_amp_ssim: float,
    current_phase_ssim: float,
    control_amp_ssim: float,
    control_phase_ssim: float,
    max_abs_amp_ssim_delta: float,
    max_abs_phase_ssim_delta: float,
    protocol_failure_reason: str | None,
) -> AbsoluteGateAdjudication:
    """Derive the one canonical absolute-gate verdict and reason."""
    operands = (
        current_amp_ssim,
        current_phase_ssim,
        control_amp_ssim,
        control_phase_ssim,
        max_abs_amp_ssim_delta,
        max_abs_phase_ssim_delta,
    )
    if not all(math.isfinite(float(value)) for value in operands):
        raise ValueError("absolute gate operands and thresholds must be finite")
    amp_delta = absolute_ssim_delta(current_amp_ssim, control_amp_ssim)
    phase_delta = absolute_ssim_delta(current_phase_ssim, control_phase_ssim)
    _validate_protocol_failure(protocol_failure_reason, policy="absolute")
    if protocol_failure_reason is not None:
        return AbsoluteGateAdjudication(
            verdict="fail",
            reason=protocol_failure_reason,
            amp_delta=amp_delta,
            phase_delta=phase_delta,
            observed=None,
            threshold=None,
        )
    if amp_delta > max_abs_amp_ssim_delta:
        return AbsoluteGateAdjudication(
            verdict="fail",
            reason=REASON_ABSOLUTE_AMP_DELTA,
            amp_delta=amp_delta,
            phase_delta=phase_delta,
            observed=amp_delta,
            threshold=max_abs_amp_ssim_delta,
        )
    if phase_delta > max_abs_phase_ssim_delta:
        return AbsoluteGateAdjudication(
            verdict="fail",
            reason=REASON_ABSOLUTE_PHASE_DELTA,
            amp_delta=amp_delta,
            phase_delta=phase_delta,
            observed=phase_delta,
            threshold=max_abs_phase_ssim_delta,
        )
    return AbsoluteGateAdjudication(
        verdict="pass",
        reason=None,
        amp_delta=amp_delta,
        phase_delta=phase_delta,
        observed=amp_delta,
        threshold=max_abs_amp_ssim_delta,
    )


def adjudicate_retained_ssim(
    *,
    current_amp_ssim: float,
    current_phase_ssim: float,
    control_amp_ssim: float,
    control_phase_ssim: float,
    retained_amp_ssim_min_fraction: float,
    retained_phase_ssim_min_fraction: float,
    absolute_amp_ssim_floor: float,
    protocol_failure_reason: str | None,
) -> RetainedGateAdjudication:
    """Derive the canonical historical retained-SSIM verdict and reason."""
    operands = (
        current_amp_ssim,
        current_phase_ssim,
        control_amp_ssim,
        control_phase_ssim,
        retained_amp_ssim_min_fraction,
        retained_phase_ssim_min_fraction,
        absolute_amp_ssim_floor,
    )
    if not all(math.isfinite(float(value)) for value in operands):
        raise ValueError("retained gate operands and thresholds must be finite")
    if control_amp_ssim <= 0.0 or control_phase_ssim <= 0.0:
        raise ValueError("retained gate control SSIM values must be positive")
    if not 0.0 < retained_amp_ssim_min_fraction <= 1.0:
        raise ValueError("retained amplitude threshold must be in (0, 1]")
    if not 0.0 < retained_phase_ssim_min_fraction <= 1.0:
        raise ValueError("retained phase threshold must be in (0, 1]")
    if not -1.0 <= absolute_amp_ssim_floor <= 1.0:
        raise ValueError("retained absolute amplitude floor must be in [-1, 1]")
    retained_amp = float(current_amp_ssim) / float(control_amp_ssim)
    retained_phase = float(current_phase_ssim) / float(control_phase_ssim)
    _validate_protocol_failure(protocol_failure_reason, policy="retained")
    if protocol_failure_reason is not None:
        return RetainedGateAdjudication(
            verdict="fail",
            reason=protocol_failure_reason,
            retained_amp_ssim=retained_amp,
            retained_phase_ssim=retained_phase,
            observed=None,
            threshold=None,
        )
    if retained_amp < retained_amp_ssim_min_fraction:
        return RetainedGateAdjudication(
            verdict="fail",
            reason=REASON_RETAINED_AMP,
            retained_amp_ssim=retained_amp,
            retained_phase_ssim=retained_phase,
            observed=retained_amp,
            threshold=retained_amp_ssim_min_fraction,
        )
    if retained_phase < retained_phase_ssim_min_fraction:
        return RetainedGateAdjudication(
            verdict="fail",
            reason=REASON_RETAINED_PHASE,
            retained_amp_ssim=retained_amp,
            retained_phase_ssim=retained_phase,
            observed=retained_phase,
            threshold=retained_phase_ssim_min_fraction,
        )
    if current_amp_ssim < absolute_amp_ssim_floor:
        return RetainedGateAdjudication(
            verdict="fail",
            reason=REASON_ABSOLUTE_FLOOR,
            retained_amp_ssim=retained_amp,
            retained_phase_ssim=retained_phase,
            observed=current_amp_ssim,
            threshold=absolute_amp_ssim_floor,
        )
    return RetainedGateAdjudication(
        verdict="pass",
        reason=None,
        retained_amp_ssim=retained_amp,
        retained_phase_ssim=retained_phase,
        observed=retained_amp,
        threshold=retained_amp_ssim_min_fraction,
    )

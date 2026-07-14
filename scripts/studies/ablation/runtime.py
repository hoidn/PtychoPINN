"""Public facade for the architecture-neutral ablation runtime.

Consumers (the thin CLI and tests) import from this module only. The
implementation is split into single-responsibility helpers:

- :mod:`runtime_execution` — canonical per-run Torch execution flow;
- :mod:`runtime_records` — typed metric-record and report-evidence assembly;
- :mod:`runtime_planning` — request loading, dataset-spec merge, dry-run plan;
- :mod:`runtime_attempts` — attempt fingerprints, artifacts, resume reuse;
- :mod:`runtime_study` — study orchestration, verdicts, and reporting;
- :mod:`runtime_reference` — grid-lines reference-performance qualification
  (Task 20 dictionary-loader bridge plumbing);
- :mod:`runtime_ladder` — one-variable configuration bridge ladder
  (Task 21 rung walk, retained-SSIM gating, sealed rung evidence).
"""

from __future__ import annotations

from .runtime_execution import (
    RELOAD_ATOL,
    RELOAD_METRIC_PATH,
    RELOAD_RTOL,
    CanonicalRunResult,
    MilestoneRunResult,
    RuntimeExecutionError,
    execute_canonical_run,
)
from .runtime_records import (
    TRAINING_HISTORY_ARTIFACT,
    build_milestone_metric_records,
    build_run_metric_records,
    history_report_curves,
    stored_history_curves,
    training_history_payload,
    training_history_records,
)
from .runtime_ladder import (
    LadderControl,
    LadderOutcome,
    LadderRequest,
    evaluate_rung_gate,
    load_ladder_spec,
    render_ladder_dry_run,
    run_bridge_ladder,
    verify_baseline,
)
from .runtime_reference import (
    DECLARED_GAUGE_HANDLING,
    ReferenceQualificationOutcome,
    ReferenceQualificationRequest,
    ReferenceQualificationSpec,
    ReferenceRunResult,
    adjudicate_reference,
    arm_requirement,
    assemble_reference_evidence,
    execute_reference_run,
    derive_floor_candidate,
    load_locked_ssim_floors,
    load_reference_spec,
    render_reference_dry_run,
    run_reference_qualification,
    seal_reference_evidence,
)
from .runtime_study import (
    USAGE_ERRORS,
    LoadedStudy,
    StudyOutcome,
    StudyRequest,
    StudyRequestError,
    load_study,
    render_dry_run,
    run_study,
)

__all__ = [
    "DECLARED_GAUGE_HANDLING",
    "RELOAD_ATOL",
    "RELOAD_METRIC_PATH",
    "RELOAD_RTOL",
    "TRAINING_HISTORY_ARTIFACT",
    "USAGE_ERRORS",
    "CanonicalRunResult",
    "LadderControl",
    "LadderOutcome",
    "LadderRequest",
    "LoadedStudy",
    "MilestoneRunResult",
    "ReferenceQualificationOutcome",
    "ReferenceQualificationRequest",
    "ReferenceQualificationSpec",
    "ReferenceRunResult",
    "RuntimeExecutionError",
    "StudyOutcome",
    "StudyRequest",
    "StudyRequestError",
    "adjudicate_reference",
    "arm_requirement",
    "assemble_reference_evidence",
    "build_milestone_metric_records",
    "build_run_metric_records",
    "evaluate_rung_gate",
    "execute_canonical_run",
    "execute_reference_run",
    "derive_floor_candidate",
    "history_report_curves",
    "load_ladder_spec",
    "load_locked_ssim_floors",
    "load_reference_spec",
    "load_study",
    "render_dry_run",
    "render_ladder_dry_run",
    "render_reference_dry_run",
    "run_bridge_ladder",
    "run_reference_qualification",
    "run_study",
    "seal_reference_evidence",
    "verify_baseline",
    "stored_history_curves",
    "training_history_payload",
    "training_history_records",
]

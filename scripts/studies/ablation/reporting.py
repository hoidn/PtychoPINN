"""Deterministic scientific report rendering for ablation evidence.

The report layer is intentionally independent of model, training, and runtime
modules. It consumes completed rows and emits human-readable and machine-ready
artifacts, including visible placeholders for failed and missing attempts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .metrics import MetricRecord
from .manifest import CLAIM_GRADE_DISQUALIFYING_REASONS
from .reporting_figures import render_all_figures, render_milestone_grid
from .reporting_scatter_layout import REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
from .verdicts import (
    AttemptRow,
    AttemptStatus,
    CompletionState,
    GateResult,
    Verdict,
    aggregate_verdict,
)
from .visual_review import (
    PENDING_REVIEW_SCHEMA_VERSION,
    REVIEW_SCHEMA_VERSION,
    ReviewError,
    VisualReview,
    parse_review,
    pending_review_template,
)


class ReportingError(ValueError):
    """Raised when typed report evidence is malformed."""


_TRUTH_ROLES = frozenset({"object_truth", "reference_reconstruction", "none"})
_GRID_FILENAME = "reconstruction_truth_error_grid.png"
_FIGURE_FILENAMES = (
    _GRID_FILENAME,
    "structural_quality_grid.png",
    "training_gradient_curves.png",
    "seed_distribution.png",
    "varpro_scale.png",
    "absolute_scale_stability_dashboard.png",
)
_SEMANTIC_SIDECARS = ("figure_row_mapping.json", "plot_metadata.json")
_PROVENANCE_FILENAMES = (
    "source_manifest.toml",
    "source_config.json",
    "invocation.json",
    "expansion.json",
)
_REPORT_FILENAMES = (
    *_PROVENANCE_FILENAMES,
    "report.md",
    "aggregate_metrics.json",
    "aggregate_metrics.csv",
    "arm_seed_status.json",
    "arm_seed_status.csv",
    "verdicts.json",
    "verdicts.csv",
    *_SEMANTIC_SIDECARS,
    "visual_review.json",
    *_FIGURE_FILENAMES,
)
_COMPLETION_FILENAME = "report_completion.json"
_COMPLETION_SCHEMA_VERSION = "ablation_report_completion_v1"
REQUIRED_REPORT_ARTIFACTS = frozenset(_REPORT_FILENAMES)
_CLAIM_GRADE_REASONS = CLAIM_GRADE_DISQUALIFYING_REASONS


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    arm_id: str
    dataset_id: str
    seed: int
    truth_role: str = "none"
    capabilities: frozenset[str] = frozenset()
    ci_scaling_active: bool = False
    contract_declared: bool = False
    object_family: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "arm_id", "dataset_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ReportingError(f"{name} must be a nonempty string")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ReportingError("seed must be a nonnegative integer")
        if self.truth_role not in _TRUTH_ROLES:
            raise ReportingError(
                "truth_role must be object_truth, reference_reconstruction, or none"
            )
        if not isinstance(self.capabilities, frozenset) or any(
            not isinstance(capability, str) or not capability
            for capability in self.capabilities
        ):
            raise ReportingError("capabilities must be a frozenset of nonempty strings")
        if not isinstance(self.ci_scaling_active, bool):
            raise ReportingError("ci_scaling_active must be boolean")
        if not isinstance(self.contract_declared, bool):
            raise ReportingError("contract_declared must be boolean")
        family = self.dataset_id if self.object_family is None else self.object_family
        if not isinstance(family, str) or not family:
            raise ReportingError("object_family must be a nonempty string")
        object.__setattr__(self, "object_family", family)


def _finite_history(values: Iterable[float], name: str) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ReportingError(f"{name} must contain numeric values") from error
    if not all(math.isfinite(value) for value in normalized):
        raise ReportingError(f"{name} must contain finite values")
    return normalized


def _image(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ReportingError(f"{name} must be a numeric 2D array")
    if not np.isfinite(array).all():
        raise ReportingError(f"{name} must be finite")
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ReportRow:
    """Plot-ready evidence for one completed or terminal-failed runtime row."""

    attempt: AttemptRow
    truth_role: str
    reconstruction: np.ndarray | None = None
    target: np.ndarray | None = None
    error: np.ndarray | None = None
    common_valid_mask: np.ndarray | None = None
    training_loss: tuple[float, ...] = ()
    gradient_norm: tuple[float, ...] = ()
    metric_records: tuple[MetricRecord, ...] = ()
    dose_points: tuple[tuple[float, float], ...] = ()
    varpro_scales: tuple[float, ...] = ()
    source_fingerprint: str | None = None
    failure_stage: str | None = None
    failure_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptRow):
            raise ReportingError("attempt must be AttemptRow")
        if self.truth_role not in _TRUTH_ROLES:
            raise ReportingError(
                "truth_role must be object_truth, reference_reconstruction, or none"
            )
        records = tuple(self.metric_records)
        if any(not isinstance(record, MetricRecord) for record in records):
            raise ReportingError("metric_records must contain MetricRecord values")
        if len({record.path for record in records}) != len(records):
            raise ReportingError("metric_records must not duplicate paths")
        object.__setattr__(self, "metric_records", records)
        object.__setattr__(
            self, "training_loss", _finite_history(self.training_loss, "training_loss")
        )
        object.__setattr__(
            self, "gradient_norm", _finite_history(self.gradient_norm, "gradient_norm")
        )
        dose_points = tuple(
            (float(dose), float(scale)) for dose, scale in self.dose_points
        )
        if any(
            not math.isfinite(dose) or not math.isfinite(scale)
            for dose, scale in dose_points
        ):
            raise ReportingError("dose_points must be finite")
        object.__setattr__(self, "dose_points", dose_points)
        object.__setattr__(
            self, "varpro_scales", _finite_history(self.varpro_scales, "varpro_scales")
        )
        if self.source_fingerprint is not None and (
            not isinstance(self.source_fingerprint, str)
            or len(self.source_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_fingerprint
            )
        ):
            raise ReportingError("source_fingerprint must be a lowercase SHA-256")
        if self.attempt.terminal_success:
            if self.reconstruction is None:
                raise ReportingError("successful rows require a reconstruction")
            reconstruction = _image(self.reconstruction, "reconstruction")
            target = None if self.target is None else _image(self.target, "target")
            error = None if self.error is None else _image(self.error, "error")
            common_valid_mask = (
                None
                if self.common_valid_mask is None
                else np.asarray(self.common_valid_mask)
            )
            if self.truth_role != "none" and target is None:
                raise ReportingError("truth/reference rows require a target")
            if target is not None and target.shape != reconstruction.shape:
                raise ReportingError(
                    "reconstruction and target shapes must match exactly"
                )
            if error is not None and error.shape != reconstruction.shape:
                raise ReportingError("error shape must exactly match reconstruction")
            if common_valid_mask is not None:
                if (
                    common_valid_mask.dtype != np.bool_
                    or common_valid_mask.shape != reconstruction.shape
                    or not common_valid_mask.any()
                ):
                    raise ReportingError(
                        "common_valid_mask must be a nonempty boolean reconstruction mask"
                    )
                common_valid_mask = np.array(common_valid_mask, copy=True)
                common_valid_mask.setflags(write=False)
            object.__setattr__(self, "reconstruction", reconstruction)
            object.__setattr__(self, "target", target)
            object.__setattr__(self, "error", error)
            object.__setattr__(self, "common_valid_mask", common_valid_mask)
        elif any(
            value is not None
            for value in (
                self.reconstruction,
                self.target,
                self.error,
                self.common_valid_mask,
            )
        ):
            raise ReportingError("failed/incomplete rows must not carry plot arrays")
        if self.failure_stage is not None and (
            not isinstance(self.failure_stage, str) or not self.failure_stage
        ):
            raise ReportingError(
                "failure_stage must be a nonempty string when provided"
            )
        if self.failure_error is not None and not isinstance(self.failure_error, str):
            raise ReportingError("failure_error must be a string when provided")

    @classmethod
    def failed(
        cls,
        run_id: str,
        arm_id: str,
        dataset_id: str,
        seed: int,
        *,
        stage: str,
        error: str,
    ) -> ReportRow:
        return cls(
            attempt=AttemptRow(
                run_id=run_id,
                arm_id=arm_id,
                dataset_id=dataset_id,
                seed=seed,
                status=AttemptStatus.FAILED,
                completion=CompletionState.TERMINAL,
                metrics={},
            ),
            truth_role="none",
            failure_stage=stage,
            failure_error=error,
        )


@dataclass(frozen=True)
class ReportInput:
    study_id: str
    rows: tuple[ReportRow, ...]
    requested_runs: tuple[RunIdentity, ...]
    gate_results: tuple[GateResult, ...]
    review: VisualReview | None = None
    claim_grade_eligible: bool = False
    claim_grade_disqualifying_reasons: tuple[str, ...] = ("manifest_budget_mismatch",)
    actual_protocol_sha256: str | None = None
    expected_protocol_sha256: str | None = None
    preserve_visual_evidence: bool = True
    source_manifest: bytes | None = None
    source_config: Mapping[str, object] | None = None
    invocation: Mapping[str, object] | None = None
    expansion: Mapping[str, object] | None = None
    in_place_visual_review_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id:
            raise ReportingError("study_id must be a nonempty string")
        rows = tuple(self.rows)
        requested = tuple(self.requested_runs)
        results = tuple(self.gate_results)
        if any(not isinstance(row, ReportRow) for row in rows):
            raise ReportingError("rows must contain ReportRow values")
        if any(not isinstance(item, RunIdentity) for item in requested):
            raise ReportingError("requested_runs must contain RunIdentity values")
        if any(not isinstance(item, GateResult) for item in results):
            raise ReportingError("gate_results must contain GateResult values")
        if self.review is not None and not isinstance(self.review, VisualReview):
            raise ReportingError("review must be VisualReview when provided")
        recovery_digest = self.in_place_visual_review_sha256
        if recovery_digest is not None and (
            self.review is None
            or not isinstance(recovery_digest, str)
            or len(recovery_digest) != 64
            or any(character not in "0123456789abcdef" for character in recovery_digest)
        ):
            raise ReportingError(
                "in-place visual review recovery requires a parsed review and SHA-256"
            )
        if not isinstance(self.claim_grade_eligible, bool):
            raise ReportingError("claim_grade_eligible must be boolean")
        reasons = tuple(self.claim_grade_disqualifying_reasons)
        if (
            any(reason not in _CLAIM_GRADE_REASONS for reason in reasons)
            or len(set(reasons)) != len(reasons)
            or tuple(sorted(reasons, key=_CLAIM_GRADE_REASONS.index)) != reasons
        ):
            raise ReportingError(
                "claim-grade reasons must be closed, unique, and ordered"
            )
        if self.claim_grade_eligible != (not reasons):
            raise ReportingError("claim-grade eligibility contradicts its reasons")
        for name in ("actual_protocol_sha256", "expected_protocol_sha256"):
            digest = getattr(self, name)
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ReportingError(f"{name} must be a lowercase SHA-256")
        if self.claim_grade_eligible and (
            self.actual_protocol_sha256 is None
            or self.expected_protocol_sha256 is None
            or self.actual_protocol_sha256 != self.expected_protocol_sha256
        ):
            raise ReportingError(
                "claim-grade protocol hashes must be present and equal"
            )
        if not isinstance(self.preserve_visual_evidence, bool):
            raise ReportingError("preserve_visual_evidence must be boolean")
        if self.source_manifest is not None and not isinstance(
            self.source_manifest, bytes
        ):
            raise ReportingError("source_manifest must be bytes when provided")
        for name in ("source_config", "invocation", "expansion"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise ReportingError(f"{name} must be a mapping when provided")
        if len({row.attempt.run_id for row in rows}) != len(rows):
            raise ReportingError("rows contain duplicate run_id values")
        if len({item.run_id for item in requested}) != len(requested):
            raise ReportingError("requested_runs contain duplicate run_id values")
        requested_by_run_id = {item.run_id: item for item in requested}
        for row in rows:
            expected = requested_by_run_id.get(row.attempt.run_id)
            if expected is None:
                continue
            for field in ("arm_id", "dataset_id", "seed"):
                if getattr(row.attempt, field) != getattr(expected, field):
                    raise ReportingError(
                        f"requested run_id {expected.run_id!r} has conflicting {field}"
                    )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "requested_runs", requested)
        object.__setattr__(self, "gate_results", results)
        object.__setattr__(self, "claim_grade_disqualifying_reasons", reasons)


@dataclass(frozen=True)
class ReportArtifacts:
    output_root: Path
    aggregate_verdict: Verdict
    paths: tuple[Path, ...]


MILESTONE_TRAJECTORY_COLUMNS = (
    "epoch",
    "validation_loss",
    "learning_rate",
    "amplitude_ssim",
    "phase_ssim",
    "stitched_amplitude_std",
    "centered_phase_variance",
    "cnn_rail_occupancy",
    "ci_poisson_nll",
    "ci_relative_count_error",
    "ci_fitted_scales",
)
_MILESTONE_TRAJECTORY_JSON = "milestone_trajectory.json"
_MILESTONE_TRAJECTORY_CSV = "milestone_trajectory.csv"
_MILESTONE_GRID = "milestone_reconstruction_grid.png"
_MILESTONE_REVIEW = "milestone_review.json"


@dataclass(frozen=True)
class MilestoneEvidence:
    """Canonical metric records and plot arrays for one post-epoch checkpoint."""

    epoch: int
    checkpoint_sha256: str
    records: tuple[MetricRecord, ...]
    arrays: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ReportingError("milestone epoch must be a positive integer")
        digest = self.checkpoint_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReportingError("milestone checkpoint identity must be a SHA-256")
        if "reconstruction" not in self.arrays:
            raise ReportingError("milestone arrays require canonical reconstruction")


def _ordered_milestones(
    milestones: Iterable[MilestoneEvidence],
) -> tuple[MilestoneEvidence, ...]:
    ordered = tuple(milestones)
    epochs = tuple(item.epoch for item in ordered)
    if not ordered:
        raise ReportingError("milestone evidence must not be empty")
    if tuple(sorted(set(epochs))) != epochs:
        raise ReportingError("milestone evidence must be strictly increasing")
    return ordered


def _numeric_metric(records: tuple[MetricRecord, ...], path: str) -> float | None:
    value = next((record.value for record in records if record.path == path), None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _first_numeric_metric(
    records: tuple[MetricRecord, ...], *paths: str
) -> float | None:
    for path in paths:
        value = _numeric_metric(records, path)
        if value is not None:
            return value
    return None


def _rail_occupancy(records: tuple[MetricRecord, ...]) -> dict[str, float] | None:
    paths = {
        "real_lower": "stability.real_head_lower_saturation_fraction",
        "real_upper": "stability.real_head_upper_saturation_fraction",
        "imag_lower": "stability.imag_head_lower_saturation_fraction",
        "imag_upper": "stability.imag_head_upper_saturation_fraction",
    }
    values = {
        name: value
        for name, path in paths.items()
        if (value := _numeric_metric(records, path)) is not None
    }
    return values or None


def collate_milestone_trajectory(
    milestones: Iterable[MilestoneEvidence],
) -> list[dict[str, object]]:
    """Collate compact rows exclusively from canonical milestone records."""
    rows: list[dict[str, object]] = []
    for milestone in _ordered_milestones(milestones):
        records = milestone.records
        amplitude_variance = _numeric_metric(records, "stability.amp_variance")
        s1 = _numeric_metric(records, "measurement_consistency.varpro.s1")
        s2 = _numeric_metric(records, "measurement_consistency.varpro.s2")
        scales = None if s1 is None or s2 is None else {"s1": s1, "s2": s2}
        rows.append(
            {
                "epoch": milestone.epoch,
                "validation_loss": _numeric_metric(
                    records, "stability.validation_loss_final"
                ),
                "learning_rate": _numeric_metric(
                    records, "stability.learning_rate_final"
                ),
                "amplitude_ssim": _first_numeric_metric(
                    records,
                    "truth_quality.post_varpro.amp_ssim",
                    "truth_quality.amp_ssim",
                ),
                "phase_ssim": _first_numeric_metric(
                    records,
                    "truth_quality.post_varpro.phase_ssim",
                    "truth_quality.phase_ssim",
                ),
                "stitched_amplitude_std": (
                    math.sqrt(amplitude_variance)
                    if amplitude_variance is not None and amplitude_variance >= 0.0
                    else None
                ),
                "centered_phase_variance": _numeric_metric(
                    records, "stability.phase_variance"
                ),
                "cnn_rail_occupancy": _rail_occupancy(records),
                "ci_poisson_nll": _numeric_metric(
                    records, "measurement_consistency.mean_raw_poisson_nll"
                ),
                "ci_relative_count_error": _numeric_metric(
                    records, "measurement_consistency.relative_l2_intensity_error"
                ),
                "ci_fitted_scales": scales,
            }
        )
    return rows


def write_milestone_trajectory(
    output_dir: str | Path,
    milestones: Iterable[MilestoneEvidence],
) -> tuple[Path, Path]:
    """Write one ordered compact JSON/CSV trajectory pair."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = collate_milestone_trajectory(milestones)
    json_path = root / _MILESTONE_TRAJECTORY_JSON
    try:
        json_path.write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ReportingError(
            f"JSON serialization failed for {json_path.name}: {error}"
        ) from error
    csv_rows = [
        {
            key: (
                ""
                if value is None
                else _json_cell(value)
                if isinstance(value, Mapping)
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    csv_path = root / _MILESTONE_TRAJECTORY_CSV
    _write_csv(csv_path, list(MILESTONE_TRAJECTORY_COLUMNS), csv_rows)
    return json_path, csv_path


def write_milestone_visuals(
    output_dir: str | Path,
    run: object,
    milestones: Iterable[MilestoneEvidence],
) -> tuple[str, str]:
    """Write the compact four-column grid and pending three-status review."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_milestones(milestones)
    render_milestone_grid(
        ordered,
        root / _MILESTONE_GRID,
        title=str(getattr(run, "arm_id")),
    )
    _stable_json(
        root / _MILESTONE_REVIEW,
        {
            "arm_id": str(getattr(run, "arm_id")),
            "dataset_id": str(getattr(run, "dataset_id")),
            "milestone_epochs": [item.epoch for item in ordered],
            "run_id": str(getattr(run, "id")),
            "seed": int(getattr(run, "seed")),
            "recognizable": "pending",
            "collapsed": "pending",
            "saturated": "pending",
        },
    )
    return _MILESTONE_GRID, _MILESTONE_REVIEW


def write_milestone_artifacts(
    output_dir: str | Path,
    run: object,
    milestones: Iterable[MilestoneEvidence],
) -> tuple[str, ...]:
    """Write all Task 22 compact outputs and return completion-relative names."""
    ordered = _ordered_milestones(milestones)
    write_milestone_trajectory(output_dir, ordered)
    visual_names = write_milestone_visuals(output_dir, run, ordered)
    return (
        _MILESTONE_TRAJECTORY_JSON,
        _MILESTONE_TRAJECTORY_CSV,
        *visual_names,
    )


def _stable_json(path: Path, payload: object) -> None:
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        path.write_text(encoded, encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise ReportingError(
            f"JSON serialization failed for {path.name}: {error}"
        ) from error


def _json_cell(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ReportingError(
            f"JSON serialization failed for CSV value: {error}"
        ) from error


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ReportingError(f"cannot hash {path.name}: {error}") from error


def _write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _quantity_kind(path: str) -> str:
    if path.startswith("truth_quality.absolute_"):
        return "absolute_quantity"
    if path.startswith("truth_quality."):
        return "mean_normalized_recognizability"
    if path.startswith("reference_agreement."):
        return "reference_agreement_not_truth"
    if path.startswith("measurement_consistency."):
        return "absolute_measurement_quantity"
    return "stability_or_runtime_quantity"


def _status_rows(study: ReportInput) -> list[dict[str, object]]:
    actual = {row.attempt.run_id: row for row in study.rows}
    output: list[dict[str, object]] = []
    for identity in sorted(
        study.requested_runs, key=lambda item: (item.arm_id, item.seed, item.run_id)
    ):
        row = actual.get(identity.run_id)
        if row is None:
            output.append(
                {
                    "run_id": identity.run_id,
                    "arm_id": identity.arm_id,
                    "dataset_id": identity.dataset_id,
                    "seed": identity.seed,
                    "status": "missing",
                    "completion": "missing",
                    "failure_stage": "missing",
                    "failure_error": "requested attempt has no row",
                }
            )
            continue
        output.append(
            {
                "run_id": row.attempt.run_id,
                "arm_id": row.attempt.arm_id,
                "dataset_id": row.attempt.dataset_id,
                "seed": row.attempt.seed,
                "status": row.attempt.status.value,
                "completion": row.attempt.completion.value,
                "failure_stage": row.failure_stage or "",
                "failure_error": row.failure_error or "",
            }
        )
    for row in sorted(
        study.rows,
        key=lambda item: (item.attempt.arm_id, item.attempt.seed, item.attempt.run_id),
    ):
        if row.attempt.run_id not in {item.run_id for item in study.requested_runs}:
            output.append(
                {
                    "run_id": row.attempt.run_id,
                    "arm_id": row.attempt.arm_id,
                    "dataset_id": row.attempt.dataset_id,
                    "seed": row.attempt.seed,
                    "status": row.attempt.status.value,
                    "completion": row.attempt.completion.value,
                    "failure_stage": row.failure_stage or "",
                    "failure_error": row.failure_error or "",
                }
            )
    return output


def _metric_rows(study: ReportInput) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in study.rows:
        for record in row.metric_records:
            payload = record.to_json()
            output.append(
                {
                    "run_id": row.attempt.run_id,
                    "arm_id": row.attempt.arm_id,
                    "dataset_id": row.attempt.dataset_id,
                    "seed": row.attempt.seed,
                    "metric_path": record.path,
                    "value": payload["value"],
                    "basis": record.basis,
                    "alignment": record.alignment,
                    "quantity_kind": _quantity_kind(record.path),
                }
            )
    return sorted(
        output, key=lambda item: (str(item["run_id"]), str(item["metric_path"]))
    )


_REPORT_NAMESPACES = (
    "truth_quality",
    "reference_agreement",
    "measurement_consistency",
    "stability",
    "runtime",
)


def _arm_seed_accounting(
    study: ReportInput, statuses: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_run_id = {str(row["run_id"]): row for row in statuses}
    counts: dict[str, dict[str, int]] = {}
    for requested in study.requested_runs:
        count = counts.setdefault(
            requested.arm_id,
            {"requested": 0, "successful": 0, "failed": 0, "missing": 0},
        )
        count["requested"] += 1
        status = str(by_run_id[requested.run_id]["status"])
        if status == AttemptStatus.SUCCESS.value:
            count["successful"] += 1
        elif status == AttemptStatus.FAILED.value:
            count["failed"] += 1
        else:
            count["missing"] += 1
    return [{"arm_id": arm_id, **counts[arm_id]} for arm_id in sorted(counts)]


def _namespace_disclosure(study: ReportInput) -> list[str]:
    contracts_by_arm: dict[str, RunIdentity] = {}
    for requested in study.requested_runs:
        existing = contracts_by_arm.setdefault(requested.arm_id, requested)
        contract = (
            requested.dataset_id,
            requested.truth_role,
            requested.capabilities,
            requested.ci_scaling_active,
            requested.contract_declared,
        )
        existing_contract = (
            existing.dataset_id,
            existing.truth_role,
            existing.capabilities,
            existing.ci_scaling_active,
            existing.contract_declared,
        )
        if contract != existing_contract:
            raise ReportingError(
                f"arm {requested.arm_id!r} has inconsistent namespace contracts"
            )

    records_by_arm: dict[str, dict[str, set[str]]] = {
        arm_id: {namespace: set() for namespace in _REPORT_NAMESPACES}
        for arm_id in contracts_by_arm
    }
    for row in study.rows:
        arm_records = records_by_arm.get(row.attempt.arm_id)
        if arm_records is None:
            continue
        for record in row.metric_records:
            namespace = record.path.split(".", 1)[0]
            if namespace in arm_records:
                arm_records[namespace].add(record.path)

    def applicability(identity: RunIdentity, namespace: str) -> tuple[bool, str]:
        if not identity.contract_declared:
            return True, ""
        if namespace == "truth_quality":
            if identity.truth_role != "object_truth":
                return False, f"declared truth role is {identity.truth_role}"
            if "has_object_truth" not in identity.capabilities:
                return False, "missing capability has_object_truth"
        elif namespace == "reference_agreement":
            if identity.truth_role != "reference_reconstruction":
                return False, f"declared truth role is {identity.truth_role}"
            if "has_reference" not in identity.capabilities:
                return False, "missing capability has_reference"
        elif namespace == "measurement_consistency":
            if not identity.ci_scaling_active:
                return False, "legacy/non-CI run contract"
            if "supports_count_metrics" not in identity.capabilities:
                return False, "missing capability supports_count_metrics"
        return True, ""

    lines = ["## Metric namespaces", ""]
    for namespace in _REPORT_NAMESPACES:
        lines.extend((f"### {namespace}", ""))
        if not contracts_by_arm:
            lines.append("- NO_EVIDENCE (no selected run contracts)")
        for arm_id, identity in sorted(contracts_by_arm.items()):
            applies, reason = applicability(identity, namespace)
            paths = sorted(records_by_arm[arm_id][namespace])
            if paths:
                if identity.contract_declared and not applies:
                    raise ReportingError(
                        "artifact consistency error: "
                        f"arm {arm_id!r} emitted {namespace} records "
                        f"{paths!r} but its declared contract marks the namespace "
                        f"NOT_APPLICABLE ({reason})"
                    )
                lines.append(f"- {arm_id}: AVAILABLE ({', '.join(paths)})")
            elif not applies:
                lines.append(f"- {arm_id}: NOT_APPLICABLE ({reason})")
            else:
                lines.append(
                    f"- {arm_id}: NO_EVIDENCE (applicable namespace has no emitted records)"
                )
        lines.append("")
    return lines


def _report_text(
    study: ReportInput, statuses: list[dict[str, object]], overall: Verdict
) -> str:
    failed_or_missing = [
        row for row in statuses if row["status"] in {"failed", "incomplete", "missing"}
    ]
    numeric = [row for row in study.gate_results if row.category != "manual_review"]
    visual = [row for row in study.gate_results if row.category == "manual_review"]
    conclusion = (
        overall.value.upper() if study.claim_grade_eligible else "NON_CLAIM_GRADE"
    )
    lines = [
        f"# {study.study_id} compatibility report",
        "",
        f"Bounded compatibility conclusion: **{conclusion}**. This report assesses the declared study contract; it does not establish model superiority.",
        "",
    ]
    if not study.claim_grade_eligible:
        lines.extend(
            [
                "**NON_CLAIM_GRADE**: diagnostic verdicts below cannot publish a bounded compatibility PASS.",
                "Disqualifying reasons: "
                + ", ".join(study.claim_grade_disqualifying_reasons),
                "",
            ]
        )
    lines.extend(["## Numeric gates", ""])
    lines.extend(
        f"- {result.id}: {result.verdict.value if result.verdict else 'NOT_APPLICABLE'} ({result.reason or 'evaluated'})"
        for result in numeric
    )
    lines.extend(["", "## Manual review", ""])
    lines.extend(
        f"- {result.id}: {result.verdict.value if result.verdict else 'NOT_APPLICABLE'} ({result.reason or 'evaluated'})"
        for result in visual
    )
    lines.extend([""])
    lines.extend(_namespace_disclosure(study))
    lines.extend(["## Arm seed accounting", ""])
    lines.extend(
        (
            "| Arm | Requested | Successful | Failed | Missing/incomplete | Summary |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        )
    )
    for count in _arm_seed_accounting(study, statuses):
        summary = (
            f"{count['requested']} requested / {count['successful']} successful / "
            f"{count['failed']} failed / {count['missing']} missing"
        )
        lines.append(
            "| {arm_id} | {requested} | {successful} | {failed} | {missing} | {summary} |".format(
                **count, summary=summary
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation labels",
            "",
            "- absolute quantities use unnormalized physical/count-scale axes.",
            "- mean-normalized/recognizability quantities are not absolute correctness evidence.",
            "- reference agreement (not truth) is never labeled as absolute correctness.",
            "",
            "## Failed or missing arms",
            "",
        ]
    )
    lines.extend(
        f"- {row['run_id']}: {row['status']} ({row['failure_stage'] or row['failure_error']})"
        for row in failed_or_missing
    )
    if not failed_or_missing:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _review_payload(review: VisualReview) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "reviewer": review.reviewer,
        "timestamp": review.timestamp,
        "figure_sha256": review.figure_sha256,
    }
    if review.families:
        payload["families"] = {
            family: {
                "decision": record.decision.value,
                "recognizable": record.recognizable,
                "flat": record.flat,
                "checkerboard": record.checkerboard,
                "mirrored": record.mirrored,
                "saturation": record.saturation,
                "collapse": record.collapse,
                "notes": record.notes,
            }
            for family, record in review.families.items()
        }
        return payload
    if review.legacy is None:
        raise ReportingError("visual review has neither family nor legacy records")
    payload.update(
        {
            "decision": review.legacy.decision.value,
            "recognizable": review.legacy.recognizable,
            "flat": review.legacy.flat,
            "checkerboard": review.legacy.checkerboard,
            "mirrored": review.legacy.mirrored,
            "saturation": review.legacy.saturation,
            "collapse": review.legacy.collapse,
            "notes": review.legacy.notes,
        }
    )
    return payload


def _existing_completed_review(root: Path) -> VisualReview | None:
    path = root / "visual_review.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingError(
            f"cannot validate existing visual review: {error}"
        ) from error
    if (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == PENDING_REVIEW_SCHEMA_VERSION
    ):
        return None
    try:
        return parse_review(payload)
    except ReviewError as error:
        raise ReportingError(
            f"cannot validate existing visual review: {error}"
        ) from error


def _copy_preserved_visual_bundle(
    root: Path, staging: Path, review: VisualReview, *, preserve_review: bool
) -> None:
    _validate_renderer_semantic_sidecars(root)
    for filename in (*_FIGURE_FILENAMES, *_SEMANTIC_SIDECARS):
        source = root / filename
        if not source.is_file():
            raise ReportingError(
                f"reviewed report is missing required artifact {filename}"
            )
        if filename.endswith(".json"):
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ReportingError(
                    f"reviewed report has invalid semantic sidecar {filename}: {error}"
                ) from error
            if not isinstance(payload, Mapping) or not set(_FIGURE_FILENAMES) <= set(
                payload
            ):
                raise ReportingError(
                    f"reviewed report has incomplete semantic sidecar {filename}"
                )
        shutil.copy2(source, staging / filename)
    grid_sha256 = _sha256(staging / _GRID_FILENAME)
    if grid_sha256 != review.figure_sha256:
        raise ReportingError(
            "reviewed grid SHA-256 does not match the completed review"
        )
    if preserve_review:
        shutil.copy2(root / "visual_review.json", staging / "visual_review.json")
    else:
        _stable_json(staging / "visual_review.json", _review_payload(review))


def _array_evidence_digest(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(json.dumps(contiguous.shape, separators=(",", ":")).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _row_evidence_digest(row: ReportRow) -> str:
    payload = {
        "run_id": row.attempt.run_id,
        "status": row.attempt.status.value,
        "completion": row.attempt.completion.value,
        "source_fingerprint": row.source_fingerprint,
        "truth_role": row.truth_role,
        "training_loss": list(row.training_loss),
        "gradient_norm": list(row.gradient_norm),
        "metrics": [
            {"path": record.path, **record.to_json()}
            for record in sorted(row.metric_records, key=lambda item: item.path)
        ],
        "arrays": {
            "reconstruction": _array_evidence_digest(row.reconstruction),
            "target": _array_evidence_digest(row.target),
            "error": _array_evidence_digest(row.error),
            "common_valid_mask": _array_evidence_digest(row.common_valid_mask),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _visual_evidence_identity(
    study: ReportInput, metadata: Mapping[str, object]
) -> dict[str, object]:
    rows = sorted(
        study.rows,
        key=lambda row: row.attempt.run_id,
    )
    successful = [row for row in rows if row.attempt.terminal_success]
    fingerprints = {row.attempt.run_id: row.source_fingerprint for row in successful}
    varpro = metadata.get("varpro_scale.png")
    eligible = varpro.get("eligible_run_ids", []) if isinstance(varpro, Mapping) else []
    return {
        "renderer_layout_schema_version": REPORT_RENDERER_LAYOUT_SCHEMA_VERSION,
        "eligible_run_ids": list(eligible),
        "source_run_fingerprints": fingerprints,
        "source_run_fingerprints_complete": bool(fingerprints)
        and all(value is not None for value in fingerprints.values()),
        "requested_runs": [
            {
                "run_id": identity.run_id,
                "arm_id": identity.arm_id,
                "dataset_id": identity.dataset_id,
                "seed": identity.seed,
                "object_family": identity.object_family,
                "ci_scaling_active": identity.ci_scaling_active,
            }
            for identity in sorted(
                study.requested_runs, key=lambda identity: identity.run_id
            )
        ],
        "row_evidence_sha256": {
            row.attempt.run_id: _row_evidence_digest(row) for row in rows
        },
    }


def _preserved_visual_identity_matches(
    root: Path, current: Mapping[str, object]
) -> bool:
    if not current.get("source_run_fingerprints_complete"):
        return False
    try:
        sidecars = _validate_renderer_semantic_sidecars(root)
        existing = sidecars["plot_metadata.json"]
    except ReportingError:
        return False
    return (
        isinstance(existing, Mapping)
        and existing.get("_visual_evidence_identity") == current
    )


def _validate_renderer_semantic_sidecars(
    root: Path,
) -> dict[str, Mapping[str, object]]:
    scatter_fields = {
        "renderer_layout_schema_version",
        "run_id",
        "arm_id",
        "visual_role_id",
        "visual_style_id",
        "arm_display_label",
        "object_family",
        "panel",
        "seed",
        "exact_anchor",
        "marker_display_offset_points",
        "marker_artist_id",
        "connector_id",
    }
    payloads: dict[str, Mapping[str, object]] = {}
    for filename in _SEMANTIC_SIDECARS:
        try:
            payload = json.loads((root / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReportingError(
                f"renderer semantic schema is invalid in {filename}: {error}"
            ) from error
        if (
            not isinstance(payload, Mapping)
            or payload.get("renderer_layout_schema_version")
            != REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
            or not set(_FIGURE_FILENAMES) <= set(payload)
        ):
            raise ReportingError(
                f"renderer semantic schema is stale or incomplete in {filename}"
            )
        payloads[filename] = payload
    plot_metadata = payloads["plot_metadata.json"]
    for figure_name in ("varpro_scale.png", "absolute_scale_stability_dashboard.png"):
        figure = plot_metadata.get(figure_name)
        layouts = figure.get("scatter_layout") if isinstance(figure, Mapping) else None
        if not isinstance(layouts, list) or any(
            not isinstance(record, Mapping)
            or not scatter_fields <= set(record)
            or record.get("renderer_layout_schema_version")
            != REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
            for record in layouts
        ):
            raise ReportingError(
                f"renderer semantic schema has stale scatter layout in {figure_name}"
            )
    dashboard = plot_metadata.get("absolute_scale_stability_dashboard.png")
    zero_annotations = (
        dashboard.get("zero_annotations") if isinstance(dashboard, Mapping) else None
    )
    if not isinstance(zero_annotations, list) or any(
        not isinstance(record, Mapping)
        or not scatter_fields
        | {
            "annotation_slot",
            "annotation_display_offset_points",
            "annotation_artist_id",
            "connectors",
        }
        <= set(record)
        or record.get("renderer_layout_schema_version")
        != REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
        or not isinstance(record.get("connectors"), Mapping)
        for record in zero_annotations
    ):
        raise ReportingError("renderer semantic schema has stale zero annotations")
    return payloads


def _prepare_visual_artifacts(study: ReportInput, root: Path, staging: Path) -> None:
    metadata, mappings = render_all_figures(study, staging)
    evidence_identity = _visual_evidence_identity(study, metadata)
    metadata["_visual_evidence_identity"] = evidence_identity
    _stable_json(staging / "figure_row_mapping.json", mappings)
    _stable_json(staging / "plot_metadata.json", metadata)
    review = study.review
    if review is None and study.preserve_visual_evidence:
        review = _existing_completed_review(root)
    if review is not None and _preserved_visual_identity_matches(
        root, evidence_identity
    ):
        _copy_preserved_visual_bundle(
            root, staging, review, preserve_review=study.review is None
        )
        return
    grid = staging / _GRID_FILENAME
    if not grid.is_file():
        raise ReportingError("report renderer did not create the reconstruction grid")
    _stable_json(
        staging / "visual_review.json",
        pending_review_template(_GRID_FILENAME, _sha256(grid)),
    )


def _validate_staged_report(staging: Path) -> None:
    missing = [name for name in _REPORT_FILENAMES if not (staging / name).is_file()]
    if missing:
        raise ReportingError(f"staged report is missing artifacts: {missing!r}")
    for filename in _FIGURE_FILENAMES:
        if (staging / filename).stat().st_size == 0:
            raise ReportingError(f"staged report figure is empty: {filename}")
    try:
        if not (staging / "report.md").read_text(encoding="utf-8").strip():
            raise ReportingError("staged report prose is empty")
    except OSError as error:
        raise ReportingError(f"cannot validate staged report prose: {error}") from error
    for filename in (
        "aggregate_metrics.csv",
        "arm_seed_status.csv",
        "verdicts.csv",
    ):
        try:
            with (staging / filename).open(newline="", encoding="utf-8") as handle:
                if not next(csv.reader(handle), None):
                    raise ReportingError(f"staged report CSV is empty: {filename}")
        except OSError as error:
            raise ReportingError(
                f"cannot validate staged report CSV {filename}: {error}"
            ) from error
    for filename in (
        *_SEMANTIC_SIDECARS,
        "source_config.json",
        "invocation.json",
        "expansion.json",
        "aggregate_metrics.json",
        "arm_seed_status.json",
        "verdicts.json",
    ):
        try:
            json.loads((staging / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReportingError(
                f"staged report has invalid JSON {filename}: {error}"
            ) from error
    _validate_renderer_semantic_sidecars(staging)
    try:
        review_payload = json.loads((staging / "visual_review.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingError(
            f"staged report has invalid visual review: {error}"
        ) from error
    if (
        isinstance(review_payload, Mapping)
        and review_payload.get("schema_version") == PENDING_REVIEW_SCHEMA_VERSION
    ):
        if review_payload.get("figure_path") != _GRID_FILENAME or review_payload.get(
            "figure_sha256"
        ) != _sha256(staging / _GRID_FILENAME):
            raise ReportingError(
                "pending visual review is not bound to the staged grid"
            )
    else:
        try:
            review = parse_review(review_payload)
        except ReviewError as error:
            raise ReportingError(
                f"staged report has invalid visual review: {error}"
            ) from error
        if review.figure_sha256 != _sha256(staging / _GRID_FILENAME):
            raise ReportingError(
                "completed visual review is not bound to the staged grid"
            )


def _completion_payload(staging: Path, study: ReportInput) -> dict[str, object]:
    return {
        "schema_version": _COMPLETION_SCHEMA_VERSION,
        "claim_grade_eligible": study.claim_grade_eligible,
        "claim_grade_disqualifying_reasons": list(
            study.claim_grade_disqualifying_reasons
        ),
        "actual_protocol_sha256": study.actual_protocol_sha256,
        "expected_protocol_sha256": study.expected_protocol_sha256,
        "artifacts": [
            {"path": name, "sha256": _sha256(staging / name)}
            for name in sorted(_REPORT_FILENAMES)
        ],
    }


def _fsync_directory(root: Path) -> None:
    try:
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ReportingError(
            f"cannot fsync report directory {root}: {error}"
        ) from error


def _validate_in_place_visual_review(
    root: Path, expected_sha256: str, expected_review: VisualReview
) -> None:
    if expected_review.legacy is not None or set(expected_review.families) != {
        "deadleaves",
        "lines",
    }:
        raise ReportingError(
            "in-place visual review recovery requires a family-aware review"
        )
    review_path = root / "visual_review.json"
    if review_path.is_symlink() or not review_path.is_file():
        raise ReportingError("in-place visual review must be a regular file")
    try:
        review_bytes = review_path.read_bytes()
        payload = json.loads(review_bytes)
        review = parse_review(payload)
    except (OSError, json.JSONDecodeError, ReviewError) as error:
        raise ReportingError(
            f"in-place visual review replacement is invalid: {error}"
        ) from error
    if hashlib.sha256(review_bytes).hexdigest() != expected_sha256:
        raise ReportingError("in-place visual review replacement bytes changed")
    if review.legacy is not None or set(review.families) != {"deadleaves", "lines"}:
        raise ReportingError("in-place visual review replacement must be family-aware")
    if review != expected_review:
        raise ReportingError("in-place visual review replacement changed after parsing")
    grid = root / _GRID_FILENAME
    if grid.is_symlink() or not grid.is_file():
        raise ReportingError("in-place visual review grid must be a regular file")
    if review.figure_sha256 != _sha256(grid):
        raise ReportingError(
            "in-place visual review does not match the current reconstruction grid"
        )


def _verify_completed_report(
    root: Path,
    *,
    in_place_visual_review_sha256: str | None = None,
    in_place_visual_review: VisualReview | None = None,
) -> dict[str, object]:
    root = Path(root)
    completion = root / _COMPLETION_FILENAME
    if completion.is_symlink():
        raise ReportingError("report completion trust anchor must not be a symlink")
    if not completion.is_file():
        raise ReportingError("report completion marker is missing")
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingError(
            f"cannot validate existing report completion: {error}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema_version",
            "claim_grade_eligible",
            "claim_grade_disqualifying_reasons",
            "actual_protocol_sha256",
            "expected_protocol_sha256",
            "artifacts",
        }
        or payload.get("schema_version") != _COMPLETION_SCHEMA_VERSION
        or not isinstance(payload.get("claim_grade_eligible"), bool)
        or not isinstance(payload.get("claim_grade_disqualifying_reasons"), list)
        or payload.get("actual_protocol_sha256") is not None
        and not isinstance(payload.get("actual_protocol_sha256"), str)
        or payload.get("expected_protocol_sha256") is not None
        and not isinstance(payload.get("expected_protocol_sha256"), str)
        or not isinstance(payload.get("artifacts"), list)
    ):
        raise ReportingError("existing report completion has an invalid schema")
    for name in ("actual_protocol_sha256", "expected_protocol_sha256"):
        digest = payload[name]
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReportingError("existing report completion has invalid protocol hash")
    if payload["claim_grade_eligible"] and (
        payload["actual_protocol_sha256"] is None
        or payload["expected_protocol_sha256"] is None
        or payload["actual_protocol_sha256"] != payload["expected_protocol_sha256"]
    ):
        raise ReportingError(
            "existing report completion has invalid claim-grade protocol hashes"
        )
    reasons = payload["claim_grade_disqualifying_reasons"]
    if (
        any(
            not isinstance(reason, str) or reason not in _CLAIM_GRADE_REASONS
            for reason in reasons
        )
        or len(set(reasons)) != len(reasons)
        or sorted(reasons, key=_CLAIM_GRADE_REASONS.index) != reasons
        or payload["claim_grade_eligible"] != (not reasons)
    ):
        raise ReportingError(
            "existing report completion has invalid claim-grade eligibility"
        )
    artifacts = payload["artifacts"]
    expected = REQUIRED_REPORT_ARTIFACTS
    recorded: dict[str, str] = {}
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ReportingError("existing report completion has an invalid artifact")
        path = artifact["path"]
        digest = artifact["sha256"]
        if (
            path in recorded
            or path not in expected
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReportingError(
                "existing report completion has invalid artifact paths"
            )
        recorded[path] = digest
    if set(recorded) != expected:
        raise ReportingError(
            "existing report completion does not cover the report bundle"
        )
    if in_place_visual_review_sha256 is not None:
        if in_place_visual_review is None:
            raise ReportingError("in-place visual review recovery is incomplete")
        _validate_in_place_visual_review(
            root, in_place_visual_review_sha256, in_place_visual_review
        )
    for path, digest in recorded.items():
        if not (root / path).is_file() or (root / path).is_symlink():
            raise ReportingError(
                "existing report completion has invalid artifact paths"
            )
        if _sha256(root / path) != digest and not (
            path == "visual_review.json" and in_place_visual_review_sha256 is not None
        ):
            raise ReportingError(
                "existing report completion does not match report bytes"
            )
    _validate_renderer_semantic_sidecars(root)
    return dict(payload)


def verify_completed_report(root: str | Path) -> dict[str, object]:
    """Validate the exact sealed report bundle and return its completion record."""
    return _verify_completed_report(Path(root))


def _validate_existing_completion(root: Path, study: ReportInput) -> bool:
    if not (root / _COMPLETION_FILENAME).exists():
        if study.in_place_visual_review_sha256 is not None:
            raise ReportingError(
                "in-place visual review recovery requires an existing completion marker"
            )
        return False
    _verify_completed_report(
        root,
        in_place_visual_review_sha256=study.in_place_visual_review_sha256,
        in_place_visual_review=study.review,
    )
    return True


def _backup_published_files(root: Path, backup: Path) -> set[str]:
    prior: set[str] = set()
    for filename in _REPORT_FILENAMES:
        source = root / filename
        if source.is_file():
            shutil.copy2(source, backup / filename)
            prior.add(filename)
    return prior


def _rollback_publication(
    root: Path,
    backup: Path,
    prior: set[str],
    applied: list[str],
    had_completion: bool,
) -> None:
    errors: list[Exception] = []
    completion = root / _COMPLETION_FILENAME
    try:
        completion.unlink(missing_ok=True)
    except OSError as error:
        errors.append(error)
    for filename in reversed(applied):
        destination = root / filename
        try:
            if filename in prior:
                os.replace(backup / filename, destination)
            else:
                destination.unlink(missing_ok=True)
        except OSError as error:
            errors.append(error)
    if had_completion:
        try:
            os.replace(backup / _COMPLETION_FILENAME, completion)
        except OSError as error:
            errors.append(error)
    try:
        _fsync_directory(root)
    except ReportingError as error:
        errors.append(error)
    if errors:
        raise ReportingError("report publication rollback failed") from errors[0]


def _publish_staged_report(
    staging: Path, root: Path, study: ReportInput
) -> tuple[Path, ...]:
    _stable_json(staging / _COMPLETION_FILENAME, _completion_payload(staging, study))
    root.mkdir(parents=True, exist_ok=True)
    had_completion = _validate_existing_completion(root, study)
    backup = Path(tempfile.mkdtemp(prefix=f".{root.name}.backup-", dir=root.parent))
    prior: set[str] = set()
    applied: list[str] = []
    completion_withdrawn = False
    try:
        prior = _backup_published_files(root, backup)
        if had_completion:
            os.replace(root / _COMPLETION_FILENAME, backup / _COMPLETION_FILENAME)
            completion_withdrawn = True
        for filename in _REPORT_FILENAMES:
            os.replace(staging / filename, root / filename)
            applied.append(filename)
        os.replace(staging / _COMPLETION_FILENAME, root / _COMPLETION_FILENAME)
        _fsync_directory(root)
    except OSError as error:
        if completion_withdrawn:
            try:
                _rollback_publication(root, backup, prior, applied, had_completion)
            except ReportingError as rollback_error:
                raise ReportingError(
                    "report publication failed and rollback could not restore the prior bundle"
                ) from rollback_error
        elif applied:
            try:
                _rollback_publication(root, backup, prior, applied, had_completion)
            except ReportingError as rollback_error:
                raise ReportingError(
                    "report publication failed and rollback could not restore the prior bundle"
                ) from rollback_error
        raise ReportingError(f"report publication failed: {error}") from error
    except ReportingError:
        if completion_withdrawn or applied:
            _rollback_publication(root, backup, prior, applied, had_completion)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
    return tuple(
        sorted(root / name for name in (*_REPORT_FILENAMES, _COMPLETION_FILENAME))
    )


def write_report(study: ReportInput, output_root: str | Path) -> ReportArtifacts:
    """Write deterministic study tables, report prose, figures, and sidecars."""
    if not isinstance(study, ReportInput):
        raise ReportingError("study must be ReportInput")
    root = Path(output_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.report-", dir=root.parent))
    overall = aggregate_verdict(study.gate_results)
    try:
        (staging / "source_manifest.toml").write_bytes(
            study.source_manifest
            if study.source_manifest is not None
            else b"# source manifest unavailable in report-only rendering\n"
        )
        _stable_json(staging / "source_config.json", study.source_config or {})
        invocation = dict(study.invocation or {})
        invocation.update(
            {
                "claim_grade_eligible": study.claim_grade_eligible,
                "claim_grade_disqualifying_reasons": list(
                    study.claim_grade_disqualifying_reasons
                ),
                "actual_protocol_sha256": study.actual_protocol_sha256,
                "expected_protocol_sha256": study.expected_protocol_sha256,
            }
        )
        _stable_json(staging / "invocation.json", invocation)
        _stable_json(staging / "expansion.json", study.expansion or {})
        statuses = _status_rows(study)
        metrics = _metric_rows(study)
        _stable_json(
            staging / "aggregate_metrics.json",
            {"schema_version": "ablation_metrics_v1", "rows": metrics},
        )
        _write_csv(
            staging / "aggregate_metrics.csv",
            [
                "run_id",
                "arm_id",
                "dataset_id",
                "seed",
                "metric_path",
                "value",
                "basis",
                "alignment",
                "quantity_kind",
            ],
            ({**row, "value": _json_cell(row["value"])} for row in metrics),
        )
        _stable_json(
            staging / "arm_seed_status.json",
            {"schema_version": "ablation_status_v1", "rows": statuses},
        )
        _write_csv(
            staging / "arm_seed_status.csv",
            [
                "run_id",
                "arm_id",
                "dataset_id",
                "seed",
                "status",
                "completion",
                "failure_stage",
                "failure_error",
            ],
            statuses,
        )
        verdict_rows = [
            {
                "id": result.id,
                "category": result.category,
                "applicability": result.applicability.value,
                "verdict": "" if result.verdict is None else result.verdict.value,
                "reason": result.reason or "",
                "observed": result.observed,
                "threshold": result.threshold,
                "contributing_run_ids": list(result.contributing_run_ids),
            }
            for result in sorted(study.gate_results, key=lambda item: item.id)
        ]
        _stable_json(
            staging / "verdicts.json",
            {
                "schema_version": "ablation_verdicts_v1",
                "aggregate_verdict": overall.value,
                "published_conclusion": (
                    overall.value if study.claim_grade_eligible else "NON_CLAIM_GRADE"
                ),
                "rows": verdict_rows,
            },
        )
        _write_csv(
            staging / "verdicts.csv",
            [
                "id",
                "category",
                "applicability",
                "verdict",
                "reason",
                "observed",
                "threshold",
                "contributing_run_ids",
            ],
            (
                {**row, "contributing_run_ids": _json_cell(row["contributing_run_ids"])}
                for row in verdict_rows
            ),
        )
        (staging / "report.md").write_text(
            _report_text(study, statuses, overall), encoding="utf-8"
        )
        _prepare_visual_artifacts(study, root, staging)
        _validate_staged_report(staging)
        paths = _publish_staged_report(staging, root, study)
    except ReportingError:
        raise
    except Exception as error:
        raise ReportingError(f"report render failed: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return ReportArtifacts(root, overall, paths)


render_report = write_report

"""Study-level orchestration for the reusable Torch ablation driver.

Executes (or resumes) every selected run through the canonical execution flow,
evaluates gates/comparisons conservatively, and writes the typed study report
plus machine-readable expansion/invocation records. Request loading and
dry-run planning live in :mod:`runtime_planning`; the per-run attempt
lifecycle lives in :mod:`runtime_attempts`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    CompletionRefusedError,
    FingerprintPair,
    GitIdentity,
    InferenceFingerprintInput,
    PrepareOutcome,
    TrainingFingerprintInput,
    UntrackedSource,
    complete_attempt,
    inference_fingerprint,
    training_fingerprint,
)
from .configuration import ConfigResolutionError, resolve_torch_configs
from .datasets import (
    ValidatedDataset,
    ValidatedDatasetBundle,
    load_checked_dataset_bundle,
    load_standalone_dataset,
)
from .manifest import (
    ResolvedRun,
    ResolvedStudy,
    RuleApplicability,
    _thaw,
    _stat_identity,
    claim_grade_eligibility,
    claim_grade_protocol_fingerprints,
)
from .reporting import (
    MilestoneEvidence,
    ReportArtifacts,
    ReportInput,
    ReportRow,
    RunIdentity,
    write_report,
    write_milestone_artifacts,
)
from .runtime_attempts import (
    content_sha256s,
    prepare_run_attempt,
    record_failure,
    rows_from_completed_attempt,
    success_rows,
    write_run_artifacts,
)
from .runtime_execution import RuntimeExecutionError, execute_canonical_run
from .runtime_planning import (
    USAGE_ERRORS,
    LoadedStudy,
    StudyRequest,
    StudyRequestError,
    load_study,
    render_dry_run,
    study_output_root,
)
from .runtime_records import (
    build_milestone_metric_records,
    build_run_metric_records,
    history_report_curves,
)
from .verdicts import (
    AttemptRow,
    GateResult,
    Verdict,
    VerdictInputError,
    aggregate_verdict,
    evaluate_comparison,
    evaluate_gate,
)
from .visual_review import ReviewError, VisualReview, parse_review

__all__ = [
    "USAGE_ERRORS",
    "LoadedStudy",
    "StudyOutcome",
    "StudyRequest",
    "StudyRequestError",
    "load_study",
    "render_dry_run",
    "run_study",
]

_EXPANSION_SCHEMA = "ablation_expansion_v1"
_SOURCE_PREFIXES = ("ptycho/", "ptycho_torch/", "scripts/")


@dataclass(frozen=True)
class StudyOutcome:
    verdict: Verdict
    output_root: Path
    report: ReportArtifacts
    failed_run_ids: tuple[str, ...]
    aborted: bool


@dataclass(frozen=True)
class _FrozenSourceManifest:
    path: Path
    data: bytes
    sha256: str
    stat_identity: tuple[int, int, int, int, int]

    def verify_unchanged(self) -> None:
        try:
            current = _stat_identity(self.path.stat())
        except OSError as error:
            raise StudyRequestError(
                f"source manifest changed during study: {error}"
            ) from error
        if current != self.stat_identity:
            raise StudyRequestError("source manifest changed during study")


def _freeze_source_manifest(
    path: Path, expected_sha256: str | None = None
) -> _FrozenSourceManifest:
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise StudyRequestError(
            f"cannot freeze source manifest {path}: {error}"
        ) from error
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    if before_identity != after_identity:
        raise StudyRequestError("source manifest changed while being frozen")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise StudyRequestError(
            "source manifest hash does not match parsed manifest hash"
        )
    return _FrozenSourceManifest(path, data, digest, after_identity)


def _frozen_source_from_loaded(
    loaded: LoadedStudy, path: Path
) -> _FrozenSourceManifest:
    manifest = loaded.manifest
    if (
        manifest.source_bytes is not None
        and manifest.source_sha256 is not None
        and manifest.source_stat is not None
    ):
        if hashlib.sha256(manifest.source_bytes).hexdigest() != manifest.source_sha256:
            raise StudyRequestError(
                "parsed source manifest hash is internally inconsistent"
            )
        frozen = _FrozenSourceManifest(
            path, manifest.source_bytes, manifest.source_sha256, manifest.source_stat
        )
        frozen.verify_unchanged()
        return frozen
    return _freeze_source_manifest(path)


def _repository_root() -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd()


def _git_identity() -> GitIdentity:
    """Identify the driver's own source checkout for run fingerprints."""
    code_root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=code_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        raise StudyRequestError(
            f"run fingerprints require a git checkout at {code_root}: {error}"
        ) from error
    if not status.strip():
        return GitIdentity(commit=commit, clean=True)
    patch = git("diff", "HEAD")
    untracked: list[UntrackedSource] = []
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip()
        if not relative.endswith(".py") or not relative.startswith(_SOURCE_PREFIXES):
            continue
        source = code_root / relative
        if source.is_file():
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            untracked.append(UntrackedSource(path=relative, sha256=digest))
    return GitIdentity(
        commit=commit,
        clean=False,
        tracked_patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        untracked_sources=tuple(untracked),
    )


def _environment_digest() -> str:
    import lightning
    import numpy
    import torch

    payload = {
        "lightning": lightning.__version__,
        "numpy": numpy.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_visual_review(
    path: Path | None, output_root: Path
) -> tuple[VisualReview | None, str | None]:
    if path is None:
        return None, None
    review_path = Path(path)
    try:
        review_bytes = review_path.read_bytes()
        payload = json.loads(review_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise StudyRequestError(
            f"cannot read --visual-review {path}: {error}"
        ) from error
    review = parse_review(payload)
    grid = output_root / "reconstruction_truth_error_grid.png"
    try:
        grid_sha256 = hashlib.sha256(grid.read_bytes()).hexdigest()
    except OSError as error:
        raise StudyRequestError(
            f"cannot bind visual review: missing reconstruction grid {grid}: {error}"
        ) from error
    if review.figure_sha256 != grid_sha256:
        raise ReviewError(
            "visual review figure_sha256 does not match the output-root reconstruction grid"
        )
    same_path = (
        review_path.absolute() == (output_root / "visual_review.json").absolute()
    )
    family_aware = review.legacy is None and set(review.families) == {
        "deadleaves",
        "lines",
    }
    return (
        review,
        hashlib.sha256(review_bytes).hexdigest()
        if same_path
        and family_aware
        and review_path.is_file()
        and not review_path.is_symlink()
        else None,
    )


def _load_validated_datasets(loaded: LoadedStudy) -> dict[str, ValidatedDataset]:
    declared = {dataset.id: dataset for dataset in loaded.manifest.datasets}
    repo_root = _repository_root()
    validated: dict[str, ValidatedDataset] = {}
    checked_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    checked_group_by_id: dict[str, tuple[str, str]] = {}
    for dataset_id, declaration in declared.items():
        if dataset_id in loaded.standalone_specs:
            continue
        metadata = _thaw(declaration.metadata)
        group_key = (metadata["provenance"], metadata["provenance_sha256"])
        checked_groups.setdefault(group_key, {})[dataset_id] = metadata
        checked_group_by_id[dataset_id] = group_key
    loaded_checked_groups: dict[tuple[str, str], ValidatedDatasetBundle] = {}
    for dataset_id in dict.fromkeys(run.dataset_id for run in loaded.selected):
        if dataset_id in loaded.standalone_specs:
            validated[dataset_id] = load_standalone_dataset(
                loaded.standalone_specs[dataset_id]
            )
        else:
            group_key = checked_group_by_id[dataset_id]
            bundle = loaded_checked_groups.get(group_key)
            if bundle is None:
                bundle = load_checked_dataset_bundle(
                    checked_groups[group_key], repo_root=repo_root
                )
                loaded_checked_groups[group_key] = bundle
            validated[dataset_id] = bundle[dataset_id]
    return validated


def _selected_dataset_materialization_profiles(
    selected: tuple[ResolvedRun, ...],
    validated: Mapping[str, ValidatedDataset],
) -> dict[str, str | None]:
    selected_ids = {run.dataset_id for run in selected}
    if set(validated) != selected_ids:
        raise StudyRequestError(
            "validated datasets must cover exactly selected dataset ids"
        )
    return {
        dataset_id: validated[dataset_id].bundle.materialization_profile
        for dataset_id in sorted(selected_ids)
    }


def _preflight_selected_configs(
    loaded: LoadedStudy,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    """Reject invalid selected-arm configuration before dataset or output access."""
    seen_arms: set[str] = set()
    ci_scaling_by_arm: dict[str, bool] = {}
    resolved_by_arm: dict[str, dict[str, Any]] = {}
    for run in loaded.study.runs:
        if run.arm_id in seen_arms:
            continue
        seen_arms.add(run.arm_id)
        try:
            resolved = resolve_torch_configs(
                dict(run.overrides),
                require_all_explicit=loaded.manifest.require_all_explicit,
            )
        except ConfigResolutionError as error:
            raise StudyRequestError(
                f"arm {run.arm_id} failed config validation: {error}"
            ) from error
        ci_scaling_by_arm[run.arm_id] = resolved.ci_scaling_active
        resolved_by_arm[run.arm_id] = json.loads(resolved.canonical_json)
    return (
        ci_scaling_by_arm,
        {run.id: resolved_by_arm[run.arm_id] for run in loaded.study.runs},
    )


def _required_capabilities(study: ResolvedStudy, run: ResolvedRun) -> tuple[str, ...]:
    capabilities: set[str] = set()
    for gate in study.gates:
        if (
            gate.applicability is RuleApplicability.ACTIVE
            and gate.target_arm_id == run.arm_id
            and gate.dataset_id == run.dataset_id
        ):
            capabilities.update(gate.gate.requires)
    for comparison in study.comparisons:
        if comparison.applicability is RuleApplicability.ACTIVE and run.arm_id in (
            comparison.left_arm_id,
            comparison.right_arm_id,
        ):
            capabilities.update(comparison.comparison.requires)
    return tuple(sorted(capabilities))


def _refuse_output_collisions(
    loaded: LoadedStudy, output_root: Path, request: StudyRequest
) -> None:
    if request.resume or request.rerun:
        return
    for run in loaded.selected:
        run_root = output_root / "runs" / run.id
        if run_root.is_dir() and any(run_root.iterdir()):
            raise StudyRequestError(
                f"output collision: {run_root} already contains attempts; pass "
                "--resume to reuse completed runs or --rerun to archive them"
            )


def _build_milestone_evidence(
    resolved: object,
    result: object,
    descriptor: object,
) -> tuple[MilestoneEvidence, ...]:
    """Reuse canonical milestone record/array assembly without re-evaluation."""
    evidence: list[MilestoneEvidence] = []
    for milestone in result.milestones:
        records, arrays = build_milestone_metric_records(
            resolved, milestone, descriptor
        )
        evidence.append(
            MilestoneEvidence(
                epoch=milestone.milestone_epoch,
                checkpoint_sha256=milestone.checkpoint_sha256,
                records=records,
                arrays=arrays,
            )
        )
    return tuple(evidence)


def _execute_or_reuse_run(
    loaded: LoadedStudy,
    run: ResolvedRun,
    validated: ValidatedDataset,
    output_root: Path,
    request: StudyRequest,
    git: GitIdentity,
    environment_digest: str,
    source_manifest: _FrozenSourceManifest | None = None,
) -> tuple[AttemptRow, ReportRow]:
    if source_manifest is None:
        source_manifest = _freeze_source_manifest(Path(request.spec))
    resolved = resolve_torch_configs(
        dict(run.overrides),
        dataset=validated,
        dataset_id=run.dataset_id,
        required_capabilities=_required_capabilities(loaded.study, run),
        require_all_explicit=loaded.manifest.require_all_explicit,
    )
    training_input = TrainingFingerprintInput(
        schema_version=1,
        manifest_sha256=loaded.manifest_sha256,
        logical_run_id=run.id,
        resolved_configs=json.loads(resolved.canonical_json),
        seed=run.seed,
        git=git,
        environment_digest=environment_digest,
        content_sha256s=content_sha256s(validated.descriptor),
    )
    training_fp = training_fingerprint(training_input)
    run_root = output_root / "runs" / run.id
    prepared = prepare_run_attempt(
        run_root, training_input, training_fp, rerun=request.rerun
    )
    if prepared.outcome is PrepareOutcome.REUSABLE:
        source_manifest.verify_unchanged()
        return rows_from_completed_attempt(run, prepared.attempt)
    attempt = prepared.attempt
    try:
        result = execute_canonical_run(
            resolved,
            seed=run.seed,
            train_npz=validated.descriptor.train,
            test_npz=validated.descriptor.test,
            work_dir=attempt / "work",
            milestone_epochs=loaded.effective_milestones,
        )
        records, arrays = build_run_metric_records(
            resolved, result, validated.descriptor
        )
        milestone_evidence = _build_milestone_evidence(
            resolved, result, validated.descriptor
        )
        if tuple(item.epoch for item in milestone_evidence) != (
            loaded.effective_milestones
        ):
            raise RuntimeExecutionError(
                "milestone_outputs",
                "canonical milestone results do not match requested epochs",
            )
    except Exception as error:
        record_failure(attempt, _failure_stage(error), error)
        raise
    inference_fp = inference_fingerprint(
        InferenceFingerprintInput(
            training=training_input,
            selected_checkpoint_sha256=result.best_checkpoint_sha256,
        )
    )
    required = write_run_artifacts(
        attempt,
        run,
        resolved,
        result,
        records,
        arrays,
        training_input,
        training_fp,
        inference_fp,
        source_manifest=source_manifest.data,
    )
    if milestone_evidence:
        required += write_milestone_artifacts(
            attempt, run, milestone_evidence
        )
    source_manifest.verify_unchanged()
    complete_attempt(attempt, FingerprintPair(training_fp, inference_fp), required)
    return success_rows(
        run,
        validated.descriptor.truth,
        records,
        arrays,
        history_report_curves(result.training_history),
        inference_fp,
    )


def _failure_stage(error: Exception) -> str:
    if isinstance(error, RuntimeExecutionError):
        return error.stage
    if isinstance(error, ConfigResolutionError):
        return "config_resolution"
    if isinstance(error, CompletionRefusedError):
        return "resume_validation"
    return "runtime"


def _safe_rule(evaluate: Any, rule_id: str) -> GateResult:
    try:
        return evaluate()
    except VerdictInputError as error:
        return GateResult.active(
            rule_id, Verdict.INCONCLUSIVE, reason=f"verdict_input_error: {error}"
        )


def _evaluate_rules(
    loaded: LoadedStudy,
    rows: tuple[AttemptRow, ...],
    review: VisualReview | None,
) -> tuple[GateResult, ...]:
    seeds = loaded.requested_seeds
    results: list[GateResult] = []
    status_by_arm: dict[str, GateResult] = {}
    status_gates = [
        gate for gate in loaded.study.gates if gate.gate.operator == "status_count_ge"
    ]
    other_gates = [
        gate for gate in loaded.study.gates if gate.gate.operator != "status_count_ge"
    ]
    for gate in status_gates:
        result = _safe_rule(
            lambda gate=gate: evaluate_gate(gate, rows, requested_seeds=seeds),
            gate.id,
        )
        if gate.applicability is RuleApplicability.ACTIVE:
            status_by_arm[gate.target_arm_id] = result
        results.append(result)
    for gate in other_gates:
        results.append(
            _safe_rule(
                lambda gate=gate: evaluate_gate(
                    gate,
                    rows,
                    requested_seeds=seeds,
                    status_result=status_by_arm.get(gate.target_arm_id),
                    review=review,
                ),
                gate.id,
            )
        )
    for comparison in loaded.study.comparisons:
        results.append(
            _safe_rule(
                lambda comparison=comparison: evaluate_comparison(
                    comparison, rows, requested_seeds=seeds
                ),
                comparison.id,
            )
        )
    return tuple(results)


def _study_records(
    loaded: LoadedStudy,
    request: StudyRequest,
    output_root: Path,
    dataset_profiles: Mapping[str, str | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expansion = {
        "schema_version": _EXPANSION_SCHEMA,
        "study_id": loaded.manifest.study_id,
        "manifest_sha256": loaded.manifest_sha256,
        "requested_seeds": list(loaded.requested_seeds),
        "dataset_materialization_profiles": dict(dataset_profiles),
        "selected_runs": [
            {
                "id": run.id,
                "arm_id": run.arm_id,
                "dataset_id": run.dataset_id,
                "seed": run.seed,
                "dimensions": dict(run.dimensions),
                "overrides": dict(run.overrides),
            }
            for run in loaded.selected
        ],
        "excludes": [dict(item) for item in loaded.manifest.excludes],
        "gates": [
            {
                "id": gate.id,
                "dataset_id": gate.dataset_id,
                "target_arm_id": gate.target_arm_id,
                "applicability": gate.applicability.value,
                "reason": gate.reason,
            }
            for gate in loaded.study.gates
        ],
        "comparisons": [
            {
                "id": comparison.id,
                "left_arm_id": comparison.left_arm_id,
                "right_arm_id": comparison.right_arm_id,
                "applicability": comparison.applicability.value,
                "reason": comparison.reason,
            }
            for comparison in loaded.study.comparisons
        ],
    }
    invocation = {
        "spec": str(request.spec),
        "dataset": request.dataset,
        "dataset_spec": None
        if request.dataset_spec is None
        else str(request.dataset_spec),
        "only": request.only,
        "seeds": None if request.seeds is None else list(request.seeds),
        "epochs": request.epochs,
        "output_root": str(output_root),
        "resume": request.resume,
        "rerun": request.rerun,
        "fail_fast": request.fail_fast,
        "visual_review": None
        if request.visual_review is None
        else str(request.visual_review),
        "integration_bridge_evidence": None
        if request.integration_bridge_evidence is None
        else str(request.integration_bridge_evidence),
        "integration_bridge_evidence_sha256": (
            loaded.integration_bridge_evidence_sha256
        ),
        "dataset_materialization_profiles": dict(dataset_profiles),
    }
    return expansion, invocation


def run_study(request: StudyRequest) -> StudyOutcome:
    """Execute (or resume) every selected run and write the study report."""
    if request.dry_run:
        raise StudyRequestError("run_study cannot execute a --dry-run request")
    loaded = load_study(request)
    if (
        loaded.manifest.budget_threshold_contract_locked
        and loaded.manifest.integration_bridge_requirement is not None
        and loaded.integration_bridge_evidence is None
    ):
        raise StudyRequestError(
            "claim-locked execution requires --integration-bridge-evidence "
            "that passes the manifest requirement"
        )
    source_manifest = _frozen_source_from_loaded(loaded, Path(request.spec))
    ci_scaling_by_arm, resolved_run_configs = _preflight_selected_configs(loaded)
    output_root = study_output_root(loaded, required=True)
    assert output_root is not None
    review, in_place_visual_review_sha256 = (
        (None, None)
        if request.rerun
        else _load_visual_review(request.visual_review, output_root)
    )
    validated = _load_validated_datasets(loaded)
    dataset_profiles = _selected_dataset_materialization_profiles(
        loaded.selected, validated
    )
    _refuse_output_collisions(loaded, output_root, request)
    git = _git_identity()
    environment_digest = _environment_digest()
    claim_grade_eligible, claim_grade_reasons = claim_grade_eligibility(
        loaded.manifest,
        loaded.study.runs,
        loaded.selected,
        epochs_override=request.epochs is not None,
        seeds_override=request.seeds is not None,
        matrix_filter=request.only is not None,
        dataset_override=request.dataset is not None,
        external_dataset_spec=request.dataset_spec is not None,
        dirty_checkout=not git.clean,
        resolved_run_configs=resolved_run_configs,
        dataset_profiles=dataset_profiles,
        integration_bridge_evidence=loaded.integration_bridge_evidence,
    )
    actual_protocol_sha256, expected_protocol_sha256 = (
        claim_grade_protocol_fingerprints(
            loaded.manifest,
            loaded.study.runs,
            resolved_run_configs,
            dataset_profiles=dataset_profiles,
        )
    )

    rows: list[AttemptRow] = []
    report_rows: list[ReportRow] = []
    failed: list[str] = []
    aborted = False
    for run in loaded.selected:
        try:
            attempt_row, report_row = _execute_or_reuse_run(
                loaded,
                run,
                validated[run.dataset_id],
                output_root,
                request,
                git,
                environment_digest,
                source_manifest,
            )
        except Exception as error:  # arm failures stay visible in the report
            report_row = ReportRow.failed(
                run.id,
                run.arm_id,
                run.dataset_id,
                run.seed,
                stage=_failure_stage(error),
                error=f"{type(error).__name__}: {error}",
            )
            attempt_row = report_row.attempt
            failed.append(run.id)
        rows.append(attempt_row)
        report_rows.append(report_row)
        if failed and request.fail_fast and failed[-1] == run.id:
            aborted = True
            break

    results = _evaluate_rules(loaded, tuple(rows), review)
    source_manifest.verify_unchanged()
    expansion, invocation = _study_records(
        loaded, request, output_root, dataset_profiles
    )
    report = write_report(
        ReportInput(
            study_id=loaded.manifest.study_id,
            rows=tuple(report_rows),
            requested_runs=tuple(
                RunIdentity(
                    run.id,
                    run.arm_id,
                    run.dataset_id,
                    run.seed,
                    truth_role=run.dataset.truth,
                    capabilities=run.dataset.capabilities,
                    ci_scaling_active=ci_scaling_by_arm[run.arm_id],
                    contract_declared=True,
                    object_family=str(
                        run.dimensions.get("object_family", run.dataset_id)
                    ),
                )
                for run in loaded.selected
            ),
            gate_results=results,
            review=review,
            claim_grade_eligible=claim_grade_eligible,
            claim_grade_disqualifying_reasons=claim_grade_reasons,
            actual_protocol_sha256=actual_protocol_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            preserve_visual_evidence=not request.rerun,
            source_manifest=source_manifest.data,
            source_config={
                "require_all_explicit": loaded.manifest.require_all_explicit,
                "base_overrides": _thaw(loaded.manifest.base_overrides),
            },
            invocation=invocation,
            expansion=expansion,
            in_place_visual_review_sha256=in_place_visual_review_sha256,
        ),
        output_root,
    )
    return StudyOutcome(
        verdict=aggregate_verdict(results),
        output_root=output_root,
        report=report,
        failed_run_ids=tuple(failed),
        aborted=aborted,
    )

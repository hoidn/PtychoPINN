"""Grid-lines reference-performance qualification (plan Task 20 plumbing).

Public facade tying together the checked reference spec
(:mod:`runtime_reference_spec`), the dictionary-loader execution flow
(:mod:`runtime_reference_execution`), and the versioned bridge adjudication
(:func:`scripts.studies.ablation.verdicts.evaluate_integration_bridge`).

Fail-closed contract:

- a frozen-source arm whose floors are not yet locked (no pinned artifact
  hash, or missing artifact bytes) is never executed and adjudicates as
  ``integration_bridge_ssim_floors_unlocked``;
- locked floors are passed to the harness only for frozen-source arms —
  fixture-sourced requirements reject external floors by design;
- sealed evidence is never overwritten in place.

Task 20b GPU invocation::

    python -m scripts.studies.ablation.runtime_reference \\
        --spec scripts/studies/specs/grid_lines_reference_performance.toml \\
        --reference grid_lines_hybrid_resnet_reference \\
        --train-npz <train.npz> --test-npz <test.npz> \\
        --output-root <artifact-root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .runtime_errors import StudyRequestError
from .runtime_atomic import atomic_write_bytes_no_clobber
from .runtime_reference_execution import (
    DECLARED_GAUGE_HANDLING,
    ReferenceRunResult,
    assemble_reference_evidence,
    execute_reference_run,
    seal_reference_evidence,
)
from .runtime_reference_spec import (
    REFERENCE_SPEC_KIND,
    ExpectedDifference,
    ReferenceArm,
    ReferenceQualificationSpec,
    load_reference_spec,
    render_reference_dry_run,
)
from .runtime_reference_floors import derive_floor_candidate
from .runtime_reference_visuals import publish_reference_visual
from .verdicts import (
    GateResult,
    IntegrationBridgeEvidence,
    IntegrationBridgeRequirement,
    LockedSsimFloors,
    Verdict,
    evaluate_integration_bridge,
)

__all__ = [
    "DECLARED_GAUGE_HANDLING",
    "REFERENCE_SPEC_KIND",
    "ExpectedDifference",
    "ReferenceArm",
    "ReferenceQualificationOutcome",
    "ReferenceQualificationRequest",
    "ReferenceQualificationSpec",
    "ReferenceRunResult",
    "adjudicate_reference",
    "arm_requirement",
    "assemble_reference_evidence",
    "execute_reference_run",
    "derive_floor_candidate",
    "load_locked_ssim_floors",
    "load_reference_spec",
    "main",
    "render_reference_dry_run",
    "run_reference_qualification",
    "seal_reference_evidence",
]

_REPORT_SCHEMA = "grid_lines_reference_qualification_report_v2"
_UNLOCKED_REASON = "integration_bridge_ssim_floors_unlocked"


@dataclass(frozen=True)
class ReferenceQualificationRequest:
    """Typed request consumed by :func:`run_reference_qualification`."""

    spec: Path
    reference: str | None = None
    train_npz: Path | None = None
    test_npz: Path | None = None
    output_root: Path | None = None
    dry_run: bool = False
    base_dir: Path | None = None


@dataclass(frozen=True)
class ReferenceQualificationOutcome:
    passed: bool
    results: tuple[GateResult, ...]
    recipe_fingerprint_sha256: str
    plan: str | None = None
    report_path: Path | None = None


def arm_requirement(arm: ReferenceArm) -> IntegrationBridgeRequirement:
    """Build the locked v4 requirement; unlocked frozen floors fail closed."""
    if (
        arm.floor_source == "frozen_reference_artifact"
        and arm.bridge.get("ssim_floor_reference_artifact_sha256") is None
    ):
        raise StudyRequestError(
            f"reference {arm.id} floors are unlocked: freeze the reference "
            "artifact and pin ssim_floor_reference_artifact_sha256 before "
            "building its requirement"
        )
    return IntegrationBridgeRequirement.from_mapping(dict(arm.bridge))


def load_locked_ssim_floors(arm: ReferenceArm) -> LockedSsimFloors | None:
    """Parse locked floors from the frozen artifact bytes; None when unusable.

    Only frozen-source arms may carry locked floors; fixture arms always
    return None so the harness's fixture/locked exclusivity holds.
    """
    if arm.floor_source != "frozen_reference_artifact":
        return None
    artifact = arm.frozen_reference_artifact
    if (
        arm.bridge.get("ssim_floor_reference_artifact_sha256") is None
        or artifact is None
        or not artifact.is_file()
    ):
        return None
    return LockedSsimFloors.from_frozen_reference_artifact_bytes(artifact.read_bytes())


def adjudicate_reference(
    arm: ReferenceArm, evidence: IntegrationBridgeEvidence | None
) -> GateResult:
    """Adjudicate one arm under the Task 19 bridge harness, failing closed."""
    locked = load_locked_ssim_floors(arm)
    if arm.floor_source == "frozen_reference_artifact" and locked is None:
        return GateResult.active(
            str(arm.bridge["bridge_id"]),
            Verdict.FAIL if evidence is not None else Verdict.INCONCLUSIVE,
            category="claim_prerequisite",
            reason=(
                _UNLOCKED_REASON
                if evidence is not None
                else "missing_integration_bridge_evidence"
            ),
        )
    return evaluate_integration_bridge(
        arm_requirement(arm), evidence, locked_ssim_floors=locked
    )


def _selected_arms(
    spec: ReferenceQualificationSpec, reference: str | None
) -> tuple[ReferenceArm, ...]:
    if reference is None:
        return spec.arms
    return (spec.arm(reference),)


def _execute_arm(
    spec: ReferenceQualificationSpec,
    arm: ReferenceArm,
    request: ReferenceQualificationRequest,
    output_root: Path,
) -> tuple[GateResult, dict[str, object]]:
    report_entry: dict[str, object] = {
        "id": arm.id,
        "floor_source": arm.floor_source,
        "floors_locked": arm.floors_locked,
    }
    if arm.floor_source == "frozen_reference_artifact" and not arm.floors_locked:
        # Fail closed without training: floors must be locked from the frozen
        # artifact BEFORE execution, so an unlocked arm may not even run.
        result = GateResult.active(
            str(arm.bridge["bridge_id"]),
            Verdict.FAIL,
            category="claim_prerequisite",
            reason=_UNLOCKED_REASON,
        )
        report_entry.update({"executed": False, "reason": result.reason})
        return result, report_entry
    assert request.train_npz is not None and request.test_npz is not None
    evidence_path = output_root / arm.id / "reference_evidence.json"
    if evidence_path.exists():
        raise StudyRequestError(
            f"sealed evidence already exists at {evidence_path}; refusing to "
            "overwrite reference-qualification evidence in place"
        )
    run_result = execute_reference_run(
        spec,
        arm,
        train_npz=Path(request.train_npz),
        test_npz=Path(request.test_npz),
        work_dir=output_root / arm.id / "work",
    )
    payload = assemble_reference_evidence(spec, arm, run_result)
    evidence = seal_reference_evidence(payload, evidence_path)
    visual_manifest = publish_reference_visual(
        run_result,
        architecture=str(arm.runner["architecture"]),
        output_dir=output_root / arm.id,
    )
    result = adjudicate_reference(arm, evidence)
    report_entry.update(
        {
            "executed": True,
            "reason": result.reason,
            "evidence_path": str(evidence_path),
            "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": run_result.checkpoint_sha256,
            "selected_checkpoint": str(run_result.best_checkpoint),
            "train_sha256": run_result.materialized.train_sha256,
            "test_sha256": run_result.materialized.test_sha256,
            "visual_manifest_path": str(
                output_root / arm.id / "visual_bundle" / "visual_manifest.json"
            ),
            "visual_sha256": visual_manifest["visual_sha256"],
            "command": dict(run_result.command),
            "metrics": {
                "fixture_amp_mae": run_result.fixture_amp_mae,
                "fixture_phase_mae": run_result.fixture_phase_mae,
                "fixture_amp_ssim": run_result.fixture_amp_ssim,
                "fixture_phase_ssim": run_result.fixture_phase_ssim,
            },
        }
    )
    return result, report_entry


def run_reference_qualification(
    request: ReferenceQualificationRequest,
) -> ReferenceQualificationOutcome:
    """Qualify the selected reference arms and write the sealed report."""
    spec = load_reference_spec(request.spec, base_dir=request.base_dir)
    arms = _selected_arms(spec, request.reference)
    if request.dry_run:
        return ReferenceQualificationOutcome(
            passed=False,
            results=(),
            recipe_fingerprint_sha256=spec.recipe.fingerprint_sha256,
            plan=render_reference_dry_run(spec),
        )
    if request.train_npz is None or request.test_npz is None:
        raise StudyRequestError(
            "reference qualification requires --train-npz and --test-npz"
        )
    if request.output_root is None:
        raise StudyRequestError("reference qualification requires --output-root")
    output_root = Path(request.output_root)
    if output_root.exists():
        raise StudyRequestError(
            f"reference qualification requires a fresh output root; "
            f"{output_root} already exists"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.mkdir()
    except FileExistsError as error:
        raise StudyRequestError(
            f"reference qualification requires a fresh output root; "
            f"{output_root} was created concurrently"
        ) from error
    results: list[GateResult] = []
    report_entries: list[dict[str, object]] = []
    for arm in arms:
        result, report_entry = _execute_arm(spec, arm, request, output_root)
        report_entry["verdict"] = result.verdict.value if result.verdict else None
        results.append(result)
        report_entries.append(report_entry)
    report_path = output_root / "reference_qualification_report.json"
    if report_path.exists():
        raise StudyRequestError(
            f"report already exists at {report_path}; refusing to overwrite"
        )
    report = {
        "schema_version": _REPORT_SCHEMA,
        "study_id": spec.study_id,
        "spec": str(spec.spec_path),
        "recipe_id": spec.recipe.id,
        "recipe_fingerprint_sha256": spec.recipe.fingerprint_sha256,
        "declared_gauge_handling": DECLARED_GAUGE_HANDLING,
        "references": report_entries,
    }
    atomic_write_bytes_no_clobber(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        stage="reference_report",
    )
    return ReferenceQualificationOutcome(
        passed=bool(results)
        and all(result.verdict is Verdict.PASS for result in results),
        results=tuple(results),
        recipe_fingerprint_sha256=spec.recipe.fingerprint_sha256,
        report_path=report_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid_lines_reference_qualification",
        description=(
            "Run the grid-lines reference-performance qualification through "
            "the dictionary-loader study flow."
        ),
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--reference", help="qualify only this declared reference id")
    parser.add_argument("--train-npz", dest="train_npz", type=Path)
    parser.add_argument("--test-npz", dest="test_npz", type=Path)
    parser.add_argument("--output-root", dest="output_root", type=Path)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exit_error:
        code = exit_error.code
        return code if isinstance(code, int) else 2
    request = ReferenceQualificationRequest(
        spec=args.spec,
        reference=args.reference,
        train_npz=args.train_npz,
        test_npz=args.test_npz,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    try:
        outcome = run_reference_qualification(request)
    except StudyRequestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if outcome.plan is not None:
        print(outcome.plan)
        return 0
    for result in outcome.results:
        verdict = result.verdict.value if result.verdict else "not_applicable"
        line = f"reference {result.id} {verdict}"
        if result.reason:
            line += f" reason={result.reason}"
        print(line)
    print(f"report {outcome.report_path}")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""One-variable configuration bridge ladder orchestration (plan Task 21).

Sequential rung walk with policy-specific control linkage:

- Rung 0 is the already-qualified Hybrid reference. Its sealed evidence bytes
  must hash to the spec's pin and re-adjudicate PASS under the Task 19/20
  bridge harness before any rung may run.
- The active absolute-delta policy compares every mmap rung with the dictionary
  baseline. Historical retained-SSIM specs continue to use the immediately
  preceding passing rung plus an absolute amplitude floor. Internal tensor/hash
  differences are classified diagnostics; protocol failures override metrics.
- Per-rung evidence is sealed once and never overwritten; re-invocations
  re-verify and reuse it, so Task 21b can execute the GPU rungs one at a
  time. The ladder report is a derived artifact rewritten on each invocation.
- Execution refuses to run while the gate thresholds are still ``proposed``;
  the controller must lock them in the spec first.

CLI (Task 21b, one rung per GPU job)::

    python -m scripts.studies.ablation.runtime_ladder \\
        --spec scripts/studies/specs/grid_lines_bridge_ladder.toml \\
        --datasets-root <root with <dataset_id>/{train,test}.npz> \\
        --output-root <artifact-root> [--rung RUNG_ID] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .dataset_content import file_sha256
from .runtime_errors import StudyRequestError
from .runtime_ladder_evidence import (
    assemble_rung_evidence,
    seal_rung_evidence,
    write_diagnostics_report,
    write_ladder_report,
)
from .runtime_ladder_execution import execute_ladder_rung
from .runtime_ladder_reuse import (
    assert_dataset_bytes_bound,
    control_from_sealed_evidence,
    diagnostic_binding_reference,
    reuse_existing_evidence,
    rung_paths,
    validate_staged_datasets,
)
from .runtime_ladder_gating import (
    LadderControl,
    adjudicate_payload,
    evaluate_rung_gate,
    gate_evidence_record,
    recorded_differences,
    verify_baseline,
)
from .runtime_ladder_spec import (
    LADDER_SPEC_KIND,
    BridgeLadderSpec,
    LadderRung,
    load_ladder_spec,
    render_ladder_dry_run,
)
from .verdicts import GateResult, Verdict

__all__ = [
    "LADDER_SPEC_KIND",
    "LadderControl",
    "LadderOutcome",
    "LadderRequest",
    "evaluate_rung_gate",
    "load_ladder_spec",
    "main",
    "render_ladder_dry_run",
    "run_bridge_ladder",
    "verify_baseline",
]

_EVIDENCE_NAME = "rung_evidence.json"


@dataclass(frozen=True)
class LadderRequest:
    """Typed request consumed by :func:`run_bridge_ladder`."""

    spec: Path
    datasets_root: Path | None = None
    output_root: Path | None = None
    rung: str | None = None
    seed: int | None = None
    dry_run: bool = False
    base_dir: Path | None = None


@dataclass(frozen=True)
class LadderOutcome:
    passed: bool
    results: tuple[GateResult, ...]
    first_material_degradation: str | None = None
    report_path: Path | None = None
    plan: str | None = None


def _process_rung(
    spec: BridgeLadderSpec,
    rung: LadderRung,
    request: LadderRequest,
    control: LadderControl,
    output_root: Path,
    *,
    execute: bool,
    predecessor: tuple[LadderRung, Mapping[str, Any]] | None,
) -> tuple[GateResult, dict[str, Any], str]:
    evidence_path = output_root / rung.id / _EVIDENCE_NAME
    if rung.diagnostic and predecessor is None:
        # I1: diagnostics bind against the sealed chain evidence sharing
        # their dataset id (when present in the output root).
        predecessor = diagnostic_binding_reference(spec, rung, output_root)
    if evidence_path.is_file():
        payload, digest = reuse_existing_evidence(
            rung, evidence_path, control, request.seed
        )
        assert_dataset_bytes_bound(
            rung,
            predecessor,
            payload["dataset"]["train_sha256"],
            payload["dataset"]["test_sha256"],
        )
        result = adjudicate_payload(spec, rung, payload, control)
        metrics = payload["metrics"]
        expected_gate = gate_evidence_record(
            spec.gate,
            control,
            amp_ssim=float(metrics["amp_ssim"]),
            phase_ssim=float(metrics["phase_ssim"]),
            result=result,
        )
        if payload.get("gate") != expected_gate:
            raise StudyRequestError(
                f"sealed evidence at {evidence_path} has missing or tampered "
                "gate operands, thresholds, or verdict"
            )
        return result, payload, digest
    if not execute:
        raise StudyRequestError(
            f"rung {rung.id} has no sealed evidence at {evidence_path}; "
            "prior rungs must be executed (or their evidence staged) first"
        )
    if request.datasets_root is None:
        raise StudyRequestError("ladder execution requires --datasets-root")
    train_npz, test_npz = rung_paths(rung, Path(request.datasets_root))
    # Bind the bytes BEFORE any training happens (fail fast, seal nothing).
    assert_dataset_bytes_bound(
        rung, predecessor, file_sha256(train_npz), file_sha256(test_npz)
    )
    run_result = execute_ladder_rung(
        spec,
        rung,
        train_npz=train_npz,
        test_npz=test_npz,
        work_dir=output_root / rung.id / "work",
        seed=request.seed,
    )
    observed = {
        "canvases_equivalent": run_result.canvases_equivalent,
        "masks_equivalent": run_result.masks_equivalent,
        "effective_probe_matches_recipe": (
            run_result.effective_probe_matches_recipe
        ),
    }
    payload = assemble_rung_evidence(
        spec, rung, run_result, control, recorded_differences(rung, observed)
    )
    result = adjudicate_payload(spec, rung, payload, control)
    payload["gate"] = gate_evidence_record(
        spec.gate,
        control,
        amp_ssim=float(run_result.amp_ssim),
        phase_ssim=float(run_result.phase_ssim),
        result=result,
    )
    digest = seal_rung_evidence(payload, evidence_path)
    return result, payload, digest


def run_bridge_ladder(request: LadderRequest) -> LadderOutcome:
    """Walk the ladder sequentially, gating each rung on its passing control."""
    spec = load_ladder_spec(request.spec, base_dir=request.base_dir)
    selected = spec.rung(request.rung) if request.rung is not None else None
    if selected is not None and not selected.runnable:
        raise StudyRequestError(
            f"rung {selected.id!r} is {selected.execution_status} and is not runnable"
        )
    if request.dry_run:
        plan = render_ladder_dry_run(spec)
        if request.datasets_root is not None:
            plan = "\n".join(
                [plan, *validate_staged_datasets(spec, Path(request.datasets_root))]
            )
        return LadderOutcome(passed=False, results=(), plan=plan)
    if not spec.gate.locked:
        raise StudyRequestError(
            "gate thresholds are still "
            f"{spec.gate.threshold_provenance!r}; the controller must set "
            "gate.threshold_provenance = 'locked' before any rung executes"
        )
    if request.output_root is None:
        raise StudyRequestError("ladder execution requires --output-root")
    selected_index = (
        None if selected is None else [r.id for r in spec.rungs].index(selected.id)
    )
    output_root = Path(request.output_root)
    baseline = verify_baseline(spec)
    control = baseline
    results: list[GateResult] = []
    chain_results: list[GateResult] = []
    entries: list[dict[str, Any]] = []
    first_material_degradation: str | None = None
    predecessor: tuple[LadderRung, dict[str, Any]] | None = None
    chain_total = sum(
        1 for rung in spec.rungs if rung.runnable and not rung.diagnostic
    )
    for index, rung in enumerate(spec.rungs):
        if not rung.runnable:
            entries.append(
                {
                    "id": rung.id,
                    "group": rung.group,
                    "status": rung.execution_status,
                    "diagnostic": rung.diagnostic,
                }
            )
            continue
        if selected_index is not None and index > selected_index:
            entries.append(
                {
                    "id": rung.id,
                    "group": rung.group,
                    "status": "pending",
                    "diagnostic": rung.diagnostic,
                }
            )
            continue
        if selected_index is not None and index != selected_index and (
            rung.diagnostic or (selected is not None and selected.diagnostic)
        ):
            # Diagnostic branches are standalone: never a prerequisite, and
            # selecting one requires no prior chain evidence.
            entries.append(
                {
                    "id": rung.id,
                    "group": rung.group,
                    "status": "skipped",
                    "diagnostic": rung.diagnostic,
                }
            )
            continue
        execute = selected_index is None or index == selected_index
        # Diagnostic rungs gate against the baseline (rung 0) by default;
        # a declared control_rung gates against that rung's sealed evidence
        # instead (sampler-isolation rungs 1d/1e). They never consume or
        # produce chain state either way.
        if rung.diagnostic and rung.control_rung is not None:
            rung_control = control_from_sealed_evidence(
                spec.rung(rung.control_rung), output_root, request.seed
            )
        elif rung.diagnostic:
            rung_control = baseline
        elif spec.gate.policy == "absolute_ssim_delta_v1":
            rung_control = baseline
        else:
            rung_control = control
        result, payload, digest = _process_rung(
            spec,
            rung,
            request,
            rung_control,
            output_root,
            execute=execute,
            predecessor=None if rung.diagnostic else predecessor,
        )
        if not rung.diagnostic:
            predecessor = (rung, payload)
            chain_results.append(result)
        results.append(result)
        metrics = payload["metrics"]
        entries.append(
            {
                "id": rung.id,
                "group": rung.group,
                "dataset": rung.dataset,
                "status": "adjudicated",
                "diagnostic": rung.diagnostic,
                "verdict": result.verdict.value if result.verdict else None,
                "reason": result.reason,
                "control_rung_id": rung_control.rung_id,
                "retained_amp_ssim": (
                    float(metrics["amp_ssim"]) / rung_control.amp_ssim
                ),
                "retained_phase_ssim": (
                    float(metrics["phase_ssim"]) / rung_control.phase_ssim
                ),
                "amp_ssim": metrics["amp_ssim"],
                "phase_ssim": metrics["phase_ssim"],
                "gate": payload["gate"],
                "evidence_path": f"{rung.id}/{_EVIDENCE_NAME}",
                "evidence_sha256": digest,
            }
        )
        if rung.diagnostic:
            continue
        if (
            spec.gate.policy == "retained_ssim_v1"
            and result.verdict is Verdict.PASS
        ):
            control = LadderControl(
                rung_id=rung.id,
                amp_ssim=float(metrics["amp_ssim"]),
                phase_ssim=float(metrics["phase_ssim"]),
                evidence_sha256=digest,
            )
        if (
            result.verdict is not Verdict.PASS
            and first_material_degradation is None
        ):
            first_material_degradation = rung.id
    diagnostic_entries = [
        entry for entry in entries if entry.get("diagnostic", False)
    ]
    chain_entries = [
        entry for entry in entries if not entry.get("diagnostic", False)
    ]
    if selected is not None and selected.diagnostic:
        # I3b: a diagnostic invocation must never clobber the chain report.
        # S-1: the diagnostics report merges on write — sibling diagnostics
        # never demote each other's adjudicated entries.
        report_path = write_diagnostics_report(
            spec, output_root, baseline, diagnostic_entries
        )
    else:
        report_path = write_ladder_report(
            spec, output_root, baseline, chain_entries, first_material_degradation
        )
        adjudicated_diagnostics = [
            entry
            for entry in diagnostic_entries
            if entry.get("status") == "adjudicated"
        ]
        # R1 + S-1: adjudicated-only, merged on write.
        if adjudicated_diagnostics:
            write_diagnostics_report(
                spec, output_root, baseline, adjudicated_diagnostics
            )
    passed = len(chain_results) == chain_total and all(
        result.verdict is Verdict.PASS for result in chain_results
    )
    return LadderOutcome(
        passed=passed,
        results=tuple(results),
        first_material_degradation=first_material_degradation,
        report_path=report_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid_lines_bridge_ladder",
        description=(
            "Walk the one-variable configuration bridge ladder with per-rung "
            "sealed evidence and policy-specific SSIM gating."
        ),
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--datasets-root",
        dest="datasets_root",
        type=Path,
        help="root containing <dataset_id>/{train,test}.npz per rung dataset",
    )
    parser.add_argument("--output-root", dest="output_root", type=Path)
    parser.add_argument("--rung", help="execute only this rung (priors must exist)")
    parser.add_argument("--seed", type=int, help="seed override for every rung")
    parser.add_argument(
        "--base-dir",
        dest="base_dir",
        type=Path,
        help="base directory for relative spec/dataset paths (default: cwd)",
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exit_error:
        code = exit_error.code
        return code if isinstance(code, int) else 2
    request = LadderRequest(
        spec=args.spec,
        datasets_root=args.datasets_root,
        output_root=args.output_root,
        rung=args.rung,
        seed=args.seed,
        dry_run=args.dry_run,
        base_dir=args.base_dir,
    )
    try:
        outcome = run_bridge_ladder(request)
    except StudyRequestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if outcome.plan is not None:
        print(outcome.plan)
        return 0
    for result in outcome.results:
        verdict = result.verdict.value if result.verdict else "not_applicable"
        line = f"rung {result.id} {verdict}"
        if result.reason:
            line += f" reason={result.reason}"
        print(line)
    if outcome.first_material_degradation is not None:
        print(f"first_material_degradation {outcome.first_material_degradation}")
    print(f"report {outcome.report_path}")
    if args.rung is not None:
        # Per-rung invocations (the 21b automation contract) succeed iff the
        # SELECTED rung adjudicates PASS; ladder-wide completeness lives in
        # the report and the final full-walk adjudication.
        selected = next(
            (result for result in outcome.results if result.id == args.rung), None
        )
        return 0 if selected is not None and selected.verdict is Verdict.PASS else 1
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    sys.exit(main())

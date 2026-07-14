#!/usr/bin/env python
"""Thin CLI for manifest-driven Torch ablation studies.

All study semantics live in ``scripts/studies/ablation``; this shell only
parses arguments, renders the dry-run plan, and maps typed errors to exit
codes: 0 = report written (or plan printed), 1 = aborted via --fail-fast,
2 = usage/validation error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch_ablation_driver",
        description="Run a reusable manifest-driven Torch ablation study.",
    )
    parser.add_argument("--spec", type=Path, required=True, help="study TOML manifest")
    parser.add_argument("--dataset", help="declared dataset id to select")
    parser.add_argument(
        "--dataset-spec",
        dest="dataset_spec",
        type=Path,
        help="standalone versioned dataset descriptor TOML to register",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="print the expanded plan without loading data or training",
    )
    parser.add_argument("--only", help="exact arm/run id or dimension=value[,...]")
    parser.add_argument("--seeds", help="CSV of seeds overriding study.seeds")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument(
        "--output-root", dest="output_root", type=Path, help="study artifact root"
    )
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed runs whose fingerprints and artifacts validate",
    )
    exclusive.add_argument(
        "--rerun",
        action="store_true",
        help="archive completed attempts and execute fresh ones",
    )
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        help="stop after the first failed run",
    )
    parser.add_argument(
        "--visual-review",
        dest="visual_review",
        type=Path,
        help="completed machine-readable visual review JSON to import",
    )
    parser.add_argument(
        "--integration-bridge-evidence",
        dest="integration_bridge_evidence",
        type=Path,
        help="sealed reference-qualification evidence required by claim lock",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_error:
        code = exit_error.code
        return code if isinstance(code, int) else 2

    from scripts.studies.ablation import runtime

    try:
        seeds = _parse_seeds(args.seeds, runtime.StudyRequestError)
        request = runtime.StudyRequest(
            spec=args.spec,
            dataset=args.dataset,
            dataset_spec=args.dataset_spec,
            dry_run=args.dry_run,
            only=args.only,
            seeds=seeds,
            epochs=args.epochs,
            output_root=args.output_root,
            resume=args.resume,
            rerun=args.rerun,
            fail_fast=args.fail_fast,
            visual_review=args.visual_review,
            integration_bridge_evidence=args.integration_bridge_evidence,
        )
        if args.dry_run:
            print(runtime.render_dry_run(runtime.load_study(request), request))
            return 0
        outcome = runtime.run_study(request)
    except runtime.USAGE_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"aggregate_verdict {outcome.verdict.value}")
    print(f"report {outcome.report.output_root}")
    if outcome.failed_run_ids:
        print("failed_runs " + ",".join(outcome.failed_run_ids))
    return 1 if outcome.aborted else 0


def _parse_seeds(text: str | None, error_type: type[Exception]) -> tuple[int, ...] | None:
    if text is None:
        return None
    parts = [part for part in text.split(",") if part]
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as error:
        raise error_type(f"--seeds must be a CSV of integers, got {text!r}") from error
    if not seeds:
        raise error_type("--seeds must not be empty")
    return seeds


if __name__ == "__main__":
    sys.exit(main())

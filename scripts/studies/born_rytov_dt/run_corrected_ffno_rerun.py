"""Corrected BRDT FFNO rerun entrypoint.

Wraps the historical FFNO-extension runner so the new append-only
``2026-05-06-brdt-corrected-ffno-row-rerun`` root can reuse the same
execution path while publishing truthful backlog metadata for the
corrected pure-FFNO authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from scripts.studies.born_rytov_dt.models import build_neural_adapter
from scripts.studies.born_rytov_dt import run_ffno_extension as historical_mod
from scripts.studies.invocation_logging import (
    build_command_line,
    capture_runtime_provenance,
    update_invocation_artifacts,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_corrected_ffno_rerun.py"
BACKLOG_ITEM = "2026-05-06-brdt-corrected-ffno-row-rerun"
CLAIM_BOUNDARY = historical_mod.CLAIM_BOUNDARY
FFNO_ROW_ID = historical_mod.FFNO_ROW_ID


def _rewrite_json(path: Path, mutator) -> None:
    payload = json.loads(path.read_text())
    mutator(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rewrite_top_level_artifacts(output_root: Path, argv: List[str]) -> None:
    invocation_json = output_root / "invocation.json"
    invocation_sh = output_root / "invocation.sh"
    if invocation_json.exists():
        update_invocation_artifacts(
            invocation_json,
            script=SCRIPT_PATH,
            command=build_command_line(SCRIPT_PATH, argv),
            extra={
                "backlog_item": BACKLOG_ITEM,
                "claim_boundary": CLAIM_BOUNDARY,
                "supersedes_backlog_item": historical_mod.BACKLOG_ITEM,
            },
        )
    invocation_sh.write_text(build_command_line(SCRIPT_PATH, argv) + "\n")


def _rewrite_row_invocation(output_root: Path, argv: List[str]) -> None:
    row_json = output_root / "rows" / FFNO_ROW_ID / "invocation.json"
    row_sh = output_root / "rows" / FFNO_ROW_ID / "invocation.sh"
    if row_json.exists():
        update_invocation_artifacts(
            row_json,
            script=SCRIPT_PATH,
            command=build_command_line(SCRIPT_PATH, argv),
            extra={
                "backlog_item": BACKLOG_ITEM,
                "claim_boundary": CLAIM_BOUNDARY,
                "supersedes_backlog_item": historical_mod.BACKLOG_ITEM,
            },
        )
    if row_sh.exists():
        row_sh.write_text(build_command_line(SCRIPT_PATH, argv) + "\n")


def _ensure_model_profile(output_root: Path) -> None:
    row_dir = output_root / "rows" / FFNO_ROW_ID
    profile_path = row_dir / "model_profile.json"
    if profile_path.exists():
        return
    row_summary_path = row_dir / "row_summary.json"
    manifest_path = output_root / "preflight_manifest.json"
    if not row_summary_path.exists() or not manifest_path.exists():
        return

    row_summary = json.loads(row_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    in_channels = int(
        (manifest.get("input_contract") or {}).get("in_channels", 1)
    )
    info = build_neural_adapter("ffno", in_channels=in_channels).info()
    profile_path.write_text(
        json.dumps(
            {
                "row_id": FFNO_ROW_ID,
                "architecture": "ffno",
                "parameter_count": int(row_summary.get("parameter_count", info.parameter_count)),
                "arch_kwargs": dict(info.arch_kwargs),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _rewrite_corrected_metadata(output_root: Path, argv: List[str]) -> None:
    manifest_path = output_root / "preflight_manifest.json"
    if manifest_path.exists():
        _rewrite_json(
            manifest_path,
            lambda payload: payload.update(
                {
                    "backlog_item": BACKLOG_ITEM,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ),
        )

    combined_manifest = output_root / "combined_manifest.json"
    if combined_manifest.exists():
        def _mutate_combined_manifest(payload: Dict[str, Any]) -> None:
            payload["backlog_item"] = BACKLOG_ITEM
            payload["claim_boundary"] = CLAIM_BOUNDARY
            extension = dict(payload.get("extension") or {})
            extension["backlog_item"] = BACKLOG_ITEM
            payload["extension"] = extension

        _rewrite_json(combined_manifest, _mutate_combined_manifest)

    combined_metrics = output_root / "combined_metrics.json"
    if combined_metrics.exists():
        def _mutate_combined_metrics(payload: Dict[str, Any]) -> None:
            payload["claim_boundary"] = CLAIM_BOUNDARY
            extension = dict(payload.get("extension") or {})
            extension["backlog_item"] = BACKLOG_ITEM
            payload["extension"] = extension

        _rewrite_json(combined_metrics, _mutate_combined_metrics)

    _rewrite_top_level_artifacts(output_root, argv)
    _rewrite_row_invocation(output_root, argv)
    _ensure_model_profile(output_root)


def run_corrected_ffno_rerun(
    *,
    baseline_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[historical_mod.preflight_mod.TrainingContract] = None,
    fixed_sample_seed: int = 17,
    fixed_sample_count: int = 4,
    in_channels: int = 1,
    device_choice: str = "auto",
    dry_run: bool = False,
    parent_argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    parent_argv = list(parent_argv) if parent_argv is not None else []
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    parsed_args = {
        "baseline_root": str(Path(baseline_root)),
        "manifest": str(Path(manifest_path)),
        "output_root": str(output_root),
        "epochs": getattr(contract, "epochs", None),
        "batch_size": getattr(contract, "batch_size", None),
        "learning_rate": getattr(contract, "learning_rate", None),
        "seed": getattr(contract, "seed", None),
        "fixed_sample_seed": int(fixed_sample_seed),
        "fixed_sample_count": int(fixed_sample_count),
        "in_channels": int(in_channels),
        "device": str(device_choice),
        "dry_run": bool(dry_run),
    }
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=parent_argv,
        parsed_args=parsed_args,
        extra={
            "backlog_item": BACKLOG_ITEM,
            "claim_boundary": CLAIM_BOUNDARY,
            "supersedes_backlog_item": historical_mod.BACKLOG_ITEM,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )

    result = historical_mod.run_ffno_extension(
        baseline_root=baseline_root,
        manifest_path=manifest_path,
        output_root=output_root,
        contract=contract,
        fixed_sample_seed=fixed_sample_seed,
        fixed_sample_count=fixed_sample_count,
        in_channels=in_channels,
        device_choice=device_choice,
        dry_run=dry_run,
        parent_argv=parent_argv,
    )
    _rewrite_corrected_metadata(output_root, parent_argv)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_run_corrected_ffno_rerun",
        description=(
            "Run the corrected BRDT pure-FFNO row under the locked four-row "
            "preflight contract and publish a truthful 2026-05-06 append-only root."
        ),
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fixed-sample-seed", type=int, default=17)
    parser.add_argument("--fixed-sample-count", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        baseline_manifest = historical_mod.ext_bundle.validate_baseline_bundle(
            Path(args.baseline_root)
        )
    except historical_mod.ext_bundle.BaselineContractMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    contract = historical_mod._maybe_override_contract(args, baseline_manifest)

    try:
        result = run_corrected_ffno_rerun(
            baseline_root=Path(args.baseline_root),
            manifest_path=Path(args.manifest),
            output_root=Path(args.output_root),
            contract=contract,
            fixed_sample_seed=int(args.fixed_sample_seed),
            fixed_sample_count=int(args.fixed_sample_count),
            in_channels=int(args.in_channels),
            device_choice=str(args.device),
            dry_run=bool(args.dry_run),
            parent_argv=sys.argv[1:] if argv is None else list(argv),
        )
    except (
        ValueError,
        historical_mod.ext_bundle.BaselineContractMismatchError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

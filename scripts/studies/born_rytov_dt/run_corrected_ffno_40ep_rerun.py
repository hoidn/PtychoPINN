"""Corrected BRDT 40-epoch FFNO rerun entrypoint.

Reuses the hardened BRDT 40-epoch paper-evidence runner while rebinding it to
the corrected pure-FFNO 20-epoch authority and publishing truthful
``2026-05-06-brdt-corrected-ffno-40ep-rerun`` metadata on the durable bundle
surfaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from scripts.studies.born_rytov_dt import (
    run_brdt_40ep_paper_evidence as historical_mod,
)
from scripts.studies.invocation_logging import (
    build_command_line,
    update_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_corrected_ffno_40ep_rerun.py"
BACKLOG_ITEM = "2026-05-06-brdt-corrected-ffno-40ep-rerun"
PRE_GATE_CLAIM_BOUNDARY = historical_mod.PRE_GATE_CLAIM_BOUNDARY
PASSED_CLAIM_BOUNDARY = historical_mod.PASSED_CLAIM_BOUNDARY
CLAIM_BOUNDARY = PRE_GATE_CLAIM_BOUNDARY
CORRECTED_FFNO_BACKLOG_ITEM = "2026-05-06-brdt-corrected-ffno-row-rerun"
DURABLE_SUMMARY_PATH = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/"
    "brdt_corrected_ffno_40ep_rerun_summary.md"
)
CANONICAL_ARTIFACT_ROOT = (
    f".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/{BACKLOG_ITEM}"
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _rewrite_json(path: Path, mutator) -> None:
    payload = _read_json(path)
    mutator(payload)
    _write_json(path, payload)


def _validate_corrected_ffno_root(
    ffno_root: Path,
    *,
    baseline_root: Path,
) -> Dict[str, Any]:
    root = Path(ffno_root).resolve()
    manifest_path = root / "preflight_manifest.json"
    metrics_path = root / "metrics.json"
    combined_metrics_path = root / "combined_metrics.json"
    profile_path = root / "rows" / "ffno" / "model_profile.json"
    if not manifest_path.exists():
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            f"corrected ffno root missing preflight_manifest.json at {manifest_path}"
        )
    if not metrics_path.exists():
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            f"corrected ffno root missing metrics.json at {metrics_path}"
        )
    if not combined_metrics_path.exists():
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            "corrected ffno root missing combined_metrics.json at "
            f"{combined_metrics_path}"
        )
    if not profile_path.exists():
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            f"corrected ffno root missing model_profile.json at {profile_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("backlog_item") != CORRECTED_FFNO_BACKLOG_ITEM:
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            "corrected ffno root backlog_item mismatch"
        )
    lineage = manifest.get("baseline_lineage") or {}
    declared_baseline = lineage.get("baseline_root")
    if declared_baseline is not None and Path(declared_baseline).resolve() != Path(
        baseline_root
    ).resolve():
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            "corrected ffno root baseline_root does not match the requested baseline root"
        )
    profile = _read_json(profile_path)
    if str(profile.get("architecture") or "") != "ffno":
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            "corrected ffno root model_profile architecture mismatch"
        )
    if int(profile.get("parameter_count", -1)) != 27394:
        raise historical_mod.ext_bundle.BaselineContractMismatchError(
            "corrected ffno root no-refiner parameter_count mismatch"
        )
    return manifest


class _HistoricalPatch:
    def __init__(self, *, ffno_root: Path) -> None:
        self._ffno_root = Path(ffno_root).resolve()
        self._saved: Dict[str, Any] = {}

    def __enter__(self) -> None:
        patch_values = {
            "SCRIPT_PATH": SCRIPT_PATH,
            "BACKLOG_ITEM": BACKLOG_ITEM,
            "CLAIM_BOUNDARY": CLAIM_BOUNDARY,
            "DURABLE_SUMMARY_PATH": DURABLE_SUMMARY_PATH,
            "CANONICAL_ARTIFACT_ROOT": CANONICAL_ARTIFACT_ROOT,
        }
        for key, value in patch_values.items():
            self._saved[key] = getattr(historical_mod, key)
            setattr(historical_mod, key, value)
        self._saved["_validate_ffno_extension_bundle"] = (
            historical_mod._validate_ffno_extension_bundle
        )
        historical_mod._validate_ffno_extension_bundle = (
            lambda ffno_extension_root, *, baseline_root: _validate_corrected_ffno_root(
                self._ffno_root, baseline_root=baseline_root
            )
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self._saved.items():
            setattr(historical_mod, key, value)


def _rewrite_invocations(output_root: Path, argv: List[str]) -> None:
    command = build_command_line(SCRIPT_PATH, argv)
    invocation_paths = [
        output_root / "invocation.json",
        output_root / "rows" / "hybrid_resnet" / "invocation.json",
        output_root / "rows" / "ffno" / "invocation.json",
    ]
    for path in invocation_paths:
        if not path.exists():
            continue
        update_invocation_artifacts(
            path,
            script=SCRIPT_PATH,
            command=command,
            extra={
                "backlog_item": BACKLOG_ITEM,
                "claim_boundary": CLAIM_BOUNDARY,
                "corrected_ffno_backlog_item": CORRECTED_FFNO_BACKLOG_ITEM,
                "supersedes_backlog_item": historical_mod.BACKLOG_ITEM,
            },
        )
        payload = _read_json(path)
        parsed_args = dict(payload.get("parsed_args") or {})
        if "ffno_extension_root" in parsed_args:
            parsed_args["ffno_root"] = parsed_args.pop("ffno_extension_root")
        payload["parsed_args"] = parsed_args
        _write_json(path, payload)
    for path in (
        output_root / "invocation.sh",
        output_root / "rows" / "hybrid_resnet" / "invocation.sh",
        output_root / "rows" / "ffno" / "invocation.sh",
    ):
        if path.exists():
            path.write_text(command + "\n")


def _write_combined_manifest(
    *,
    output_root: Path,
    baseline_root: Path,
    ffno_root: Path,
    claim_boundary: str,
) -> Path:
    metrics_path = output_root / "combined_metrics.json"
    preflight_manifest_path = output_root / "preflight_manifest.json"
    rows = []
    if metrics_path.exists():
        try:
            rows = [
                {
                    "row_id": row.get("row_id"),
                    "row_status": row.get("row_status"),
                    "paper_label": row.get("paper_label"),
                    "architecture": row.get("architecture"),
                }
                for row in (_read_json(metrics_path).get("rows") or [])
            ]
        except Exception:
            rows = []
    payload = {
        "schema_version": "brdt_corrected_40ep_combined_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": str(claim_boundary),
        "output_root": str(output_root),
        "baseline_root": str(Path(baseline_root).resolve()),
        "ffno_root": str(Path(ffno_root).resolve()),
        "paper_evidence_gate_path": str(output_root / "paper_evidence_gate.json"),
        "preflight_manifest_path": str(preflight_manifest_path),
        "rows": rows,
    }
    out_path = output_root / "combined_manifest.json"
    _write_json(out_path, payload)
    return out_path


def _rewrite_corrected_metadata(
    *,
    output_root: Path,
    baseline_root: Path,
    ffno_root: Path,
    argv: List[str],
    dry_run: bool,
) -> None:
    manifest_path = output_root / "preflight_manifest.json"
    if manifest_path.exists():
        _rewrite_json(
            manifest_path,
            lambda payload: payload.update(
                {
                    "backlog_item": BACKLOG_ITEM,
                    "claim_boundary": str(payload.get("claim_boundary") or CLAIM_BOUNDARY),
                    "baseline_lineage": {
                        **dict(payload.get("baseline_lineage") or {}),
                        "ffno_root": str(Path(ffno_root).resolve()),
                    },
                }
            ),
        )
    claim_boundary = CLAIM_BOUNDARY
    gate_path = output_root / "paper_evidence_gate.json"
    if gate_path.exists():
        claim_boundary = str(
            (_read_json(gate_path).get("claim_boundary") or CLAIM_BOUNDARY)
        )
    combined_metrics_path = output_root / "combined_metrics.json"
    if combined_metrics_path.exists():
        _rewrite_json(
            combined_metrics_path,
            lambda payload: payload.update(
                {
                    "backlog_item": BACKLOG_ITEM,
                    "claim_boundary": claim_boundary,
                    "ffno_root": str(Path(ffno_root).resolve()),
                }
            ),
        )
    elif dry_run and manifest_path.exists():
        manifest_payload = _read_json(manifest_path)
        _write_json(
            combined_metrics_path,
            {
                "schema_version": "brdt_corrected_40ep_dry_run_v1",
                "backlog_item": BACKLOG_ITEM,
                "claim_boundary": claim_boundary,
                "output_root": str(output_root),
                "baseline_root": str(Path(baseline_root).resolve()),
                "ffno_root": str(Path(ffno_root).resolve()),
                "rows": list(manifest_payload.get("rows") or []),
            },
        )
    visual_manifest_path = output_root / "visual_manifest.json"
    if visual_manifest_path.exists():
        _rewrite_json(
            visual_manifest_path,
            lambda payload: payload.update(
                {
                    "backlog_item": BACKLOG_ITEM,
                    "claim_boundary": claim_boundary,
                    "output_root": str(output_root),
                    "ffno_root": str(Path(ffno_root).resolve()),
                }
            ),
        )
    elif dry_run and manifest_path.exists():
        manifest_payload = _read_json(manifest_path)
        _write_json(
            visual_manifest_path,
            {
                "schema_version": "brdt_corrected_40ep_dry_run_visuals_v1",
                "backlog_item": BACKLOG_ITEM,
                "claim_boundary": claim_boundary,
                "output_root": str(output_root),
                "ffno_root": str(Path(ffno_root).resolve()),
                "required_paper_sample": int(
                    manifest_payload.get("required_paper_sample", 255)
                ),
                "rows_present": [
                    row.get("row_id") for row in (manifest_payload.get("rows") or [])
                ],
                "figures": [],
                "classical_present": False,
            },
        )
    _write_combined_manifest(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_root=ffno_root,
        claim_boundary=claim_boundary,
    )
    _rewrite_invocations(output_root, argv)


def run_corrected_ffno_40ep_rerun(
    *,
    baseline_root: Path,
    ffno_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[historical_mod.preflight_mod.TrainingContract] = None,
    device_choice: str = "auto",
    dry_run: bool = False,
    fixed_sample_ids: Optional[List[int]] = None,
    required_paper_sample: int = 255,
    parent_argv: Optional[List[str]] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    baseline_root = Path(baseline_root).resolve()
    ffno_root = Path(ffno_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parent_argv = list(parent_argv) if parent_argv is not None else []
    with _HistoricalPatch(ffno_root=ffno_root):
        result = historical_mod.run_paper_evidence(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_root,
            manifest_path=Path(manifest_path),
            output_root=output_root,
            contract=contract,
            device_choice=device_choice,
            dry_run=dry_run,
            fixed_sample_ids=fixed_sample_ids,
            required_paper_sample=int(required_paper_sample),
            parent_argv=parent_argv,
            force_overwrite=force_overwrite,
        )
    _rewrite_corrected_metadata(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_root=ffno_root,
        argv=parent_argv,
        dry_run=bool(dry_run),
    )
    result = dict(result)
    result["combined_manifest_json_path"] = str(output_root / "combined_manifest.json")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_run_corrected_ffno_40ep_rerun",
        description=(
            "Run the corrected BRDT 40-epoch paired bundle using the corrected "
            "pure-FFNO 20-epoch authority."
        ),
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--ffno-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-meta-only", action="store_true")
    parser.add_argument(
        "--reconstruct-runtime-provenance-from-invocation", action="store_true"
    )
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--required-paper-sample", type=int, default=255)
    parser.add_argument(
        "--fixed-sample-ids", nargs="+", type=int, default=[145, 83, 255, 126]
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--plateau-factor", type=float, default=None)
    parser.add_argument("--plateau-patience", type=int, default=None)
    parser.add_argument("--plateau-threshold", type=float, default=None)
    parser.add_argument("--plateau-min-lr", type=float, default=None)
    return parser


def _maybe_override_contract(
    args: argparse.Namespace,
) -> historical_mod.preflight_mod.TrainingContract:
    contract = historical_mod._default_contract()
    if args.epochs is not None:
        contract.epochs = int(args.epochs)
    if args.batch_size is not None:
        contract.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        contract.learning_rate = float(args.learning_rate)
    if args.seed is not None:
        contract.seed = int(args.seed)
    if args.scheduler is not None:
        contract.scheduler = str(args.scheduler)
    if args.plateau_factor is not None:
        contract.plateau_factor = float(args.plateau_factor)
    if args.plateau_patience is not None:
        contract.plateau_patience = int(args.plateau_patience)
    if args.plateau_threshold is not None:
        contract.plateau_threshold = float(args.plateau_threshold)
    if args.plateau_min_lr is not None:
        contract.plateau_min_lr = float(args.plateau_min_lr)
    return contract


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    parent_argv = sys.argv[1:] if argv is None else list(argv)
    contract = _maybe_override_contract(args)

    try:
        _validate_corrected_ffno_root(
            Path(args.ffno_root), baseline_root=Path(args.baseline_root)
        )
    except historical_mod.ext_bundle.BaselineContractMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    with _HistoricalPatch(ffno_root=Path(args.ffno_root)):
        if args.reconstruct_runtime_provenance_from_invocation:
            historical_mod.reconstruct_runtime_provenance_from_invocation(
                output_root=Path(args.output_root).resolve()
            )
        if (
            args.rebuild_meta_only
            or args.reconstruct_runtime_provenance_from_invocation
        ):
            result = historical_mod.rebuild_meta_only(
                baseline_root=Path(args.baseline_root),
                ffno_extension_root=Path(args.ffno_root),
                manifest_path=Path(args.manifest),
                output_root=Path(args.output_root),
                contract=contract,
                fixed_sample_ids=[int(i) for i in args.fixed_sample_ids],
                required_paper_sample=int(args.required_paper_sample),
                parent_argv=parent_argv,
                force_overwrite=bool(args.force_overwrite),
            )
        else:
            result = historical_mod.run_paper_evidence(
                baseline_root=Path(args.baseline_root),
                ffno_extension_root=Path(args.ffno_root),
                manifest_path=Path(args.manifest),
                output_root=Path(args.output_root),
                contract=contract,
                device_choice=str(args.device),
                dry_run=bool(args.dry_run),
                fixed_sample_ids=[int(i) for i in args.fixed_sample_ids],
                required_paper_sample=int(args.required_paper_sample),
                parent_argv=parent_argv,
                force_overwrite=bool(args.force_overwrite),
            )
    _rewrite_corrected_metadata(
        output_root=Path(args.output_root).resolve(),
        baseline_root=Path(args.baseline_root).resolve(),
        ffno_root=Path(args.ffno_root).resolve(),
        argv=parent_argv,
        dry_run=bool(args.dry_run),
    )
    result = dict(result)
    result["combined_manifest_json_path"] = str(
        Path(args.output_root).resolve() / "combined_manifest.json"
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preflight and bounded validation helpers for the lines128 paper benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare
from scripts.studies.metrics_tables import write_paper_benchmark_bundle


FNO_COMPARATOR_MODEL_IDS = {
    "fno": "pinn_fno",
    "fno_vanilla": "pinn_fno_vanilla",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_NPZ = REPO_ROOT / "datasets/Run1084_recon3_postPC_shrunk_3.npz"
FIXED_CONTRACT = {
    "N": 128,
    "gridsize": 1,
    "nimgs_train": 2,
    "nimgs_test": 2,
    "nphotons": 1e9,
    "set_phi": True,
    "probe_scale_mode": "pad_extrapolate",
    "probe_smoothing_sigma": 0.5,
    "seed": 3,
    "torch_epochs": 40,
    "torch_learning_rate": 2e-4,
    "torch_scheduler": "ReduceLROnPlateau",
    "torch_plateau_factor": 0.5,
    "torch_plateau_patience": 2,
    "torch_plateau_min_lr": 1e-4,
    "torch_plateau_threshold": 0.0,
    "torch_loss_mode": "mae",
    "torch_output_mode": "real_imag",
    "fno_modes": 12,
    "fno_width": 32,
    "fno_blocks": 4,
    "fno_cnn_blocks": 2,
}


def _load_decision_artifact(path: Path) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing decision artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Decision artifact must be a JSON object")
    return payload


def _resolve_repo_path(path_value: object) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Decision artifact contract note is missing a valid path")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _validate_contract_note(payload: Mapping[str, object]) -> Path:
    contract_note = payload.get("contract_note")
    if not isinstance(contract_note, Mapping):
        raise ValueError("Decision artifact missing contract note metadata")
    if contract_note.get("status") != "resolved":
        raise ValueError("Preflight contract note is unresolved")
    contract_note_path = _resolve_repo_path(contract_note.get("path"))
    if not contract_note_path.exists():
        raise FileNotFoundError(f"Missing resolved contract note for harness preflight: {contract_note_path}")
    return contract_note_path


def _validate_fno_comparator(payload: Mapping[str, object], rows_by_id: Mapping[str, Mapping[str, object]]) -> str:
    comparator = payload.get("selected_fno_comparator")
    if comparator not in FNO_COMPARATOR_MODEL_IDS:
        raise ValueError("Decision artifact must select an explicit FNO comparator")
    comparator_model_id = FNO_COMPARATOR_MODEL_IDS[str(comparator)]
    comparator_row = rows_by_id.get(comparator_model_id)
    if comparator_row is None or comparator_row.get("status") != "supported_for_harness":
        raise ValueError("Selected FNO comparator is not supported_for_harness")
    return str(comparator)


def _row_statuses(rows_by_id: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    statuses: Dict[str, Dict[str, object]] = {}
    for model_id, row in rows_by_id.items():
        status = str(row.get("status", "row_blocker"))
        statuses[model_id] = {"status": status}
        if "blocker_reason" in row:
            statuses[model_id]["reason"] = row["blocker_reason"]
    return statuses


def _required_rows(rows_by_id: Mapping[str, Mapping[str, object]]) -> List[str]:
    return [
        model_id
        for model_id, row in rows_by_id.items()
        if row.get("required_for_minimum_subset")
    ]


def _supported_rows(rows_by_id: Mapping[str, Mapping[str, object]]) -> List[Mapping[str, object]]:
    return [
        row
        for row in rows_by_id.values()
        if row.get("status") == "supported_for_harness"
    ]


def _validate_required_rows(required_rows: Iterable[str], row_statuses: Mapping[str, Mapping[str, object]]) -> None:
    for required_model in required_rows:
        status = row_statuses.get(required_model, {}).get("status")
        if status != "supported_for_harness":
            raise ValueError(f"Minimum required row is not supported_for_harness: {required_model}")


def _run_compare_preflight(
    *,
    supported_rows: Iterable[Mapping[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    supported_rows = list(supported_rows)
    selected_models = tuple(str(row["model_id"]) for row in supported_rows)
    model_n = {
        str(row["model_id"]): int(row.get("N", FIXED_CONTRACT["N"]))
        for row in supported_rows
    }
    return run_grid_lines_compare(
        N=int(FIXED_CONTRACT["N"]),
        gridsize=int(FIXED_CONTRACT["gridsize"]),
        output_dir=output_dir / "compare_wrapper_preflight",
        probe_npz=DEFAULT_PROBE_NPZ,
        architectures=(),
        models=selected_models,
        model_n=model_n,
        seed=int(FIXED_CONTRACT["seed"]),
        nimgs_train=int(FIXED_CONTRACT["nimgs_train"]),
        nimgs_test=int(FIXED_CONTRACT["nimgs_test"]),
        nphotons=float(FIXED_CONTRACT["nphotons"]),
        set_phi=bool(FIXED_CONTRACT["set_phi"]),
        probe_scale_mode=str(FIXED_CONTRACT["probe_scale_mode"]),
        probe_smoothing_sigma=float(FIXED_CONTRACT["probe_smoothing_sigma"]),
        torch_epochs=int(FIXED_CONTRACT["torch_epochs"]),
        torch_learning_rate=float(FIXED_CONTRACT["torch_learning_rate"]),
        torch_scheduler=str(FIXED_CONTRACT["torch_scheduler"]),
        torch_plateau_factor=float(FIXED_CONTRACT["torch_plateau_factor"]),
        torch_plateau_patience=int(FIXED_CONTRACT["torch_plateau_patience"]),
        torch_plateau_min_lr=float(FIXED_CONTRACT["torch_plateau_min_lr"]),
        torch_plateau_threshold=float(FIXED_CONTRACT["torch_plateau_threshold"]),
        torch_loss_mode=str(FIXED_CONTRACT["torch_loss_mode"]),
        torch_output_mode=str(FIXED_CONTRACT["torch_output_mode"]),
        fno_modes=int(FIXED_CONTRACT["fno_modes"]),
        fno_width=int(FIXED_CONTRACT["fno_width"]),
        fno_blocks=int(FIXED_CONTRACT["fno_blocks"]),
        fno_cnn_blocks=int(FIXED_CONTRACT["fno_cnn_blocks"]),
        preflight_only=True,
    )


def _build_row_payloads(
    *,
    rows_by_id: Mapping[str, Mapping[str, object]],
    compare_preflight: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    row_plan = compare_preflight.get("row_plan")
    if not isinstance(row_plan, list):
        raise ValueError("Compare wrapper preflight did not return row_plan")
    preflight_rows = {
        str(item["model_id"]): item
        for item in row_plan
        if isinstance(item, Mapping) and "model_id" in item
    }

    row_payloads: Dict[str, Dict[str, object]] = {}
    for model_id, row in rows_by_id.items():
        if row.get("status") != "supported_for_harness":
            continue
        preflight_row = preflight_rows.get(model_id, {})
        row_payloads[model_id] = {
            "model_label": row.get("model_label", model_id),
            "N": int(preflight_row.get("N", row.get("N", FIXED_CONTRACT["N"]))),
            # Readiness bundles must use the real preflight route but remain
            # benchmark_incomplete until later execution produces row metrics.
            "metrics": {},
        }
    return row_payloads


def run_lines128_paper_benchmark_preflight(
    *,
    decision_artifact: Path,
    output_dir: Path,
) -> Dict[str, object]:
    payload = _load_decision_artifact(decision_artifact)
    contract_note_path = _validate_contract_note(payload)

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Decision artifact must define at least one row")
    rows_by_id = {
        str(row["model_id"]): row
        for row in rows
        if isinstance(row, Mapping) and "model_id" in row
    }
    comparator = _validate_fno_comparator(payload, rows_by_id)
    row_statuses = _row_statuses(rows_by_id)
    required_rows = _required_rows(rows_by_id)
    _validate_required_rows(required_rows, row_statuses)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_preflight = _run_compare_preflight(
        supported_rows=_supported_rows(rows_by_id),
        output_dir=output_dir,
    )
    compare_preflight_path = output_dir / "compare_wrapper_preflight.json"
    compare_preflight_path.write_text(json.dumps(compare_preflight, indent=2), encoding="utf-8")

    row_payloads = _build_row_payloads(
        rows_by_id=rows_by_id,
        compare_preflight=compare_preflight,
    )
    bundle_paths = write_paper_benchmark_bundle(
        output_dir=output_dir,
        row_payloads=row_payloads,
        required_rows=tuple(required_rows),
        fixed_sample_ids=payload.get("fixed_sample_ids", []),
        shared_visual_scales=payload.get("shared_visual_scales", {}),
        selected_fno_comparator=comparator,
        row_statuses=row_statuses,
        evidence_scope="readiness_only_not_benchmark_performance",
    )
    bundle_paths["compare_preflight_json"] = str(compare_preflight_path)
    return {
        "contract_note_path": str(contract_note_path),
        "selected_models": list(compare_preflight.get("selected_models", [])),
        "required_rows": required_rows,
        "compare_preflight": compare_preflight,
        "bundle_paths": bundle_paths,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run lines128 paper benchmark preflight validation")
    parser.add_argument("--decision-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    from scripts.studies.invocation_logging import (
        capture_runtime_provenance,
        get_git_commit,
        update_invocation_artifacts,
        write_invocation_artifacts,
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    invocation_json, _ = write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/lines128_paper_benchmark.py",
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "runtime_provenance": capture_runtime_provenance(),
            "git_commit": get_git_commit(REPO_ROOT),
        },
    )
    try:
        run_lines128_paper_benchmark_preflight(
            decision_artifact=args.decision_artifact,
            output_dir=args.output_dir,
        )
        update_invocation_artifacts(
            invocation_json,
            status="completed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        update_invocation_artifacts(
            invocation_json,
            status="failed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()

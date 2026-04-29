#!/usr/bin/env python3
"""Preflight and bounded validation helpers for the lines128 paper benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

from scripts.studies.metrics_tables import write_paper_benchmark_bundle


FNO_COMPARATOR_MODEL_IDS = {
    "fno": "pinn_fno",
    "fno_vanilla": "pinn_fno_vanilla",
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


def _validate_contract_note(payload: Mapping[str, object]) -> None:
    contract_note = payload.get("contract_note")
    if not isinstance(contract_note, Mapping):
        raise ValueError("Decision artifact missing contract note metadata")
    if contract_note.get("status") != "resolved":
        raise ValueError("Preflight contract note is unresolved")


def _validate_fno_comparator(payload: Mapping[str, object], rows_by_id: Mapping[str, Mapping[str, object]]) -> str:
    comparator = payload.get("selected_fno_comparator")
    if comparator not in FNO_COMPARATOR_MODEL_IDS:
        raise ValueError("Decision artifact must select an explicit FNO comparator")
    comparator_model_id = FNO_COMPARATOR_MODEL_IDS[str(comparator)]
    comparator_row = rows_by_id.get(comparator_model_id)
    if comparator_row is None or comparator_row.get("status") != "supported_for_harness":
        raise ValueError("Selected FNO comparator is not supported_for_harness")
    return str(comparator)


def _mock_metrics(index: int) -> Dict[str, List[float]]:
    base = 0.01 * (index + 1)
    return {
        "mae": [base, base + 0.01],
        "mse": [base / 10.0, (base + 0.01) / 10.0],
        "psnr": [70.0 - index, 65.0 - index],
        "ssim": [0.90 - index * 0.01, 0.85 - index * 0.01],
        "ms_ssim": [0.88 - index * 0.01, 0.83 - index * 0.01],
        "frc50": [64 - index, 48 - index],
    }


def run_lines128_paper_benchmark_preflight(
    *,
    decision_artifact: Path,
    output_dir: Path,
) -> Dict[str, object]:
    payload = _load_decision_artifact(decision_artifact)
    _validate_contract_note(payload)

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Decision artifact must define at least one row")
    rows_by_id = {
        str(row["model_id"]): row
        for row in rows
        if isinstance(row, Mapping) and "model_id" in row
    }
    comparator = _validate_fno_comparator(payload, rows_by_id)

    selected_models: List[str] = []
    required_rows: List[str] = []
    row_statuses: Dict[str, Dict[str, object]] = {}
    row_payloads: Dict[str, Dict[str, object]] = {}

    for index, row in enumerate(rows_by_id.values()):
        model_id = str(row["model_id"])
        status = str(row.get("status", "row_blocker"))
        row_statuses[model_id] = {"status": status}
        if "blocker_reason" in row:
            row_statuses[model_id]["reason"] = row["blocker_reason"]
        if row.get("required_for_minimum_subset"):
            required_rows.append(model_id)
        if status != "supported_for_harness":
            continue
        selected_models.append(model_id)
        row_payloads[model_id] = {
            "model_label": row.get("model_label", model_id),
            "N": int(row.get("N", 128)),
            "metrics": _mock_metrics(index),
        }

    for required_model in required_rows:
        status = row_statuses.get(required_model, {}).get("status")
        if status != "supported_for_harness":
            raise ValueError(f"Minimum required row is not supported_for_harness: {required_model}")

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
    return {
        "selected_models": selected_models,
        "required_rows": required_rows,
        "bundle_paths": bundle_paths,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run lines128 paper benchmark preflight validation")
    parser.add_argument("--decision-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_lines128_paper_benchmark_preflight(
        decision_artifact=args.decision_artifact,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

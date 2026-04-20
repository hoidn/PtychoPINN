"""Comparison collation for OpenFWI FlatVel-A smoke runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "profile_id",
    "status",
    "test_MAE",
    "test_RMSE",
    "test_SSIM",
    "val_MAE",
    "val_RMSE",
    "val_SSIM",
    "runtime_sec",
    "parameter_count",
    "blocker_reason",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _profile_record(run_root: Path, profile_id: str, *, run_id: str | None) -> dict[str, Any]:
    profile_root = run_root / "runs" / profile_id
    metrics_path = profile_root / "metrics.json"
    blocker_path = profile_root / "blocker.json"
    if metrics_path.exists():
        payload = _read_json(metrics_path)
        payload_run_id = str(payload.get("run_id")) if payload.get("run_id") is not None else None
        if run_id is not None and payload_run_id != str(run_id):
            raise ValueError(f"{profile_id} metrics run_id {payload_run_id!r} does not match {run_id!r}")
        eval_payload = payload.get("eval", {})
        val_payload = eval_payload.get("val", {})
        test_payload = eval_payload.get("test", payload)
        description = payload.get("model_description", {})
        return {
            "profile_id": profile_id,
            "status": "metrics",
            "run_id": payload_run_id,
            "split_manifest": payload.get("split_manifest"),
            "normalization_stats": payload.get("normalization_stats"),
            "metric_units": payload.get("metric_units") or test_payload.get("metric_units"),
            "test_MAE": test_payload.get("MAE", payload.get("MAE")),
            "test_RMSE": test_payload.get("RMSE", payload.get("RMSE")),
            "test_SSIM": test_payload.get("SSIM", payload.get("SSIM")),
            "val_MAE": val_payload.get("MAE"),
            "val_RMSE": val_payload.get("RMSE"),
            "val_SSIM": val_payload.get("SSIM"),
            "runtime_sec": payload.get("runtime_sec"),
            "parameter_count": description.get("parameter_count"),
            "blocker_reason": "",
        }
    if blocker_path.exists():
        payload = _read_json(blocker_path)
        payload_run_id = str(payload.get("run_id")) if payload.get("run_id") is not None else None
        if run_id is not None and payload_run_id != str(run_id):
            raise ValueError(f"{profile_id} blocker run_id {payload_run_id!r} does not match {run_id!r}")
        return {
            "profile_id": profile_id,
            "status": "blocker",
            "run_id": payload_run_id,
            "blocker_reason": payload.get("reason"),
            "blocker_message": payload.get("message"),
        }
    return {"profile_id": profile_id, "status": "missing", "blocker_reason": "missing_metrics_or_blocker"}


def _validate_comparability(records: list[dict[str, Any]]) -> None:
    metrics = [record for record in records if record["status"] == "metrics"]
    if not metrics:
        return
    reference = metrics[0]
    for record in metrics[1:]:
        for key in ["run_id", "split_manifest", "normalization_stats", "metric_units"]:
            if record.get(key) != reference.get(key):
                raise ValueError(f"incomparable profile artifacts for {key}: {reference.get(key)!r} != {record.get(key)!r}")


def _write_outputs(run_root: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    (run_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_root / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def collate_comparison(run_root: Path, *, profiles: list[str], run_id: str | None) -> dict[str, Any]:
    """Collate local smoke metrics/blockers and write CSV/JSON summaries."""
    run_root = Path(run_root)
    data_blocker_path = run_root / "data_access_blocker.json"
    if data_blocker_path.exists():
        blocker = _read_json(data_blocker_path)
        rows = [
            {"profile_id": profile, "status": "blocked", "blocker_reason": blocker.get("reason")}
            for profile in profiles
        ]
        summary = {
            "schema_version": "openfwi_flatvel_a_comparison_summary_v1",
            "run_id": run_id,
            "data_access_complete": False,
            "shape_validation_complete": False,
            "hybrid_profile_complete": False,
            "local_baseline_complete": False,
            "official_inversionnet_status": "not_attempted",
            "recommended_decision_input": "block_data_access",
            "data_access_blocker": blocker,
            "profiles": {row["profile_id"]: row for row in rows},
        }
        _write_outputs(run_root, summary, rows)
        return summary

    rows = [_profile_record(run_root, profile_id, run_id=run_id) for profile_id in profiles]
    _validate_comparability(rows)
    profiles_by_id = {row["profile_id"]: row for row in rows}
    hybrid = profiles_by_id.get("hybrid_resnet_smoke", {})
    baseline_rows = [
        row
        for row in rows
        if row["profile_id"] != "hybrid_resnet_smoke" and row["status"] == "metrics"
    ]
    hybrid_mae = _finite(hybrid.get("test_MAE"))
    baseline_maes = [_finite(row.get("test_MAE")) for row in baseline_rows]
    baseline_maes = [item for item in baseline_maes if item is not None]
    best_baseline = min(baseline_maes) if baseline_maes else None
    relative_gap = None
    if hybrid_mae is not None and best_baseline is not None and best_baseline > 0:
        relative_gap = (hybrid_mae - best_baseline) / best_baseline
    official_path = run_root / "official_inversionnet_compatibility.json"
    official_blocker_path = run_root / "official_inversionnet_blocker.json"
    if official_path.exists():
        official_status = _read_json(official_path).get("status", "complete")
    elif official_blocker_path.exists():
        official_status = "blocked"
    else:
        official_status = "not_attempted"

    hybrid_complete = hybrid.get("status") == "metrics"
    baseline_complete = bool(baseline_rows)
    metrics_complete = hybrid_mae is not None and best_baseline is not None
    if not baseline_complete:
        decision = "block_baseline_incomplete"
    elif not hybrid_complete or not metrics_complete:
        decision = "block_metrics_incomplete"
    else:
        decision = "proceed_candidate"
    summary = {
        "schema_version": "openfwi_flatvel_a_comparison_summary_v1",
        "run_id": run_id,
        "data_access_complete": True,
        "shape_validation_complete": (run_root / "shard_shapes.json").exists(),
        "hybrid_profile_complete": hybrid_complete,
        "local_baseline_complete": baseline_complete,
        "official_inversionnet_status": official_status,
        "best_baseline_MAE": best_baseline,
        "hybrid_MAE": hybrid_mae,
        "relative_gap_vs_best_baseline": relative_gap,
        "recommended_decision_input": decision,
        "profiles": profiles_by_id,
    }
    _write_outputs(run_root, summary, rows)
    return summary

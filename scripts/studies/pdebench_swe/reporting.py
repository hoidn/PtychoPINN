"""Result collation for PDEBench SWE longer execution."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.studies.pdebench_swe.run_config import REQUIRED_BASELINE_PROFILE_IDS


CSV_COLUMNS = [
    "profile_id",
    "status",
    "test_err_nRMSE",
    "test_err_RMSE",
    "val_err_nRMSE",
    "val_err_RMSE",
    "runtime_sec",
    "parameter_count",
    "blocker_reason",
]

REQUIRED_BASELINE_PROFILES = list(REQUIRED_BASELINE_PROFILE_IDS)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_record(run_root: Path, profile_id: str, *, run_id: str | None) -> dict[str, Any]:
    profile_root = run_root / "runs" / profile_id
    metrics_path = profile_root / "metrics.json"
    blocker_path = profile_root / "blocker.json"
    if metrics_path.exists():
        payload = _read_json(metrics_path)
        status = "metrics"
    elif blocker_path.exists():
        payload = _read_json(blocker_path)
        status = "blocker"
    else:
        return {
            "profile_id": profile_id,
            "status": "missing",
            "blocker_reason": "missing_metrics_or_blocker",
        }
    payload_run_id = str(payload.get("run_id")) if payload.get("run_id") is not None else None
    if run_id is not None and payload_run_id != str(run_id):
        raise ValueError(f"{profile_id} {status} run_id {payload_run_id!r} does not match {run_id!r}")
    if status == "blocker":
        return {
            "profile_id": profile_id,
            "status": status,
            "run_id": payload_run_id,
            "blocker_reason": payload.get("reason"),
            "blocker_message": payload.get("message"),
        }

    eval_payload = payload.get("eval", {})
    test_payload = eval_payload.get("test", {})
    val_payload = eval_payload.get("val", {})
    description = payload.get("model_description", {})
    return {
        "profile_id": profile_id,
        "status": status,
        "run_id": payload_run_id,
        "data_file": payload.get("data_file"),
        "split_manifest_run": payload.get("split_manifest_run"),
        "normalization_stats": payload.get("normalization_stats"),
        "horizon": payload.get("horizon") or test_payload.get("horizon"),
        "metric_units": payload.get("metric_units") or test_payload.get("metric_units"),
        "test_err_nRMSE": test_payload.get("err_nRMSE", payload.get("err_nRMSE")),
        "test_err_RMSE": test_payload.get("err_RMSE", payload.get("err_RMSE")),
        "val_err_nRMSE": val_payload.get("err_nRMSE"),
        "val_err_RMSE": val_payload.get("err_RMSE"),
        "runtime_sec": payload.get("runtime_sec"),
        "parameter_count": description.get("parameter_count"),
        "blocker_reason": "",
    }


def _validate_comparability(records: list[dict[str, Any]]) -> None:
    metrics_records = [record for record in records if record["status"] == "metrics"]
    if not metrics_records:
        return
    keys = ["run_id", "data_file", "split_manifest_run", "normalization_stats", "horizon", "metric_units"]
    reference = metrics_records[0]
    for record in metrics_records[1:]:
        for key in keys:
            if record.get(key) != reference.get(key):
                raise ValueError(
                    f"incomparable profile artifacts for {key}: "
                    f"{reference.get(key)!r} != {record.get(key)!r}"
                )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _write_outputs(run_root: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    json_path = run_root / "comparison_summary.json"
    csv_path = run_root / "comparison_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def collate_comparison(
    run_root: Path,
    *,
    primary_profiles: list[str],
    ablation_profiles: list[str],
    run_id: str | None = None,
    ablation_skip_reason: str | None = None,
) -> dict[str, Any]:
    """Collate per-profile metrics/blockers and write comparison summaries."""
    run_root = Path(run_root)
    profile_ids = list(primary_profiles) + [profile for profile in ablation_profiles if profile not in primary_profiles]
    rows = [_profile_record(run_root, profile_id, run_id=run_id) for profile_id in profile_ids]
    _validate_comparability(rows)

    profiles = {row["profile_id"]: row for row in rows}
    primary_complete = all(profiles[profile]["status"] == "metrics" for profile in primary_profiles)
    baseline_profiles = list(REQUIRED_BASELINE_PROFILES)
    baseline_complete = all(profiles.get(profile, {}).get("status") == "metrics" for profile in baseline_profiles)
    ablation_complete = bool(ablation_profiles) and all(
        profiles.get(profile, {}).get("status") == "metrics" for profile in ablation_profiles
    )

    hybrid_test = _finite_float(profiles.get("hybrid_resnet_base", {}).get("test_err_nRMSE"))
    baseline_values = [
        _finite_float(profiles.get(profile, {}).get("test_err_nRMSE"))
        for profile in baseline_profiles
    ]
    baseline_values = [value for value in baseline_values if value is not None]
    best_baseline = min(baseline_values) if baseline_values else None
    relative_gap = None
    if hybrid_test is not None and best_baseline is not None and best_baseline > 0:
        relative_gap = (hybrid_test - best_baseline) / best_baseline

    if not primary_complete:
        decision_input = "block_local_baseline_incomplete" if not baseline_complete else "block_primary_incomplete"
    elif not baseline_complete:
        decision_input = "block_local_baseline_incomplete"
    elif hybrid_test is None or best_baseline is None:
        decision_input = "block_metrics_nonfinite"
    elif hybrid_test <= 1.10 * best_baseline:
        decision_input = "primary_viable"
    else:
        decision_input = "primary_noncompetitive"

    summary = {
        "schema_version": "pdebench_swe_comparison_summary_v1",
        "run_id": run_id,
        "primary_profiles": list(primary_profiles),
        "required_baseline_profiles": list(REQUIRED_BASELINE_PROFILES),
        "ablation_profiles": list(ablation_profiles),
        "primary_profiles_complete": primary_complete,
        "baseline_profiles_complete": baseline_complete,
        "ablation_profiles_complete": ablation_complete,
        "ablation_skip_reason": ablation_skip_reason,
        "hybrid_test_err_nRMSE": hybrid_test,
        "best_baseline_test_err_nRMSE": best_baseline,
        "relative_gap_vs_best_baseline": relative_gap,
        "recommended_decision_input": decision_input,
        "profiles": profiles,
    }
    _write_outputs(run_root, summary, rows)
    return summary

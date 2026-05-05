"""Comparison emission for BRDT physics-only objective ablation.

Reads two BRDT preflight ``metrics.json`` payloads (a baseline and an
ablation) and writes ``comparison_to_supervised_plus_born.{json,csv}``
under the ablation root. Per-row deltas cover the blocking image and
measurement metrics, the supporting metrics, runtime, parameter count,
final loss breakdown, and output dynamic-range diagnostics when present.

This helper is task-local and is intentionally narrow: it only compares
neural rows, never re-runs anything, and never overwrites the baseline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


COMPARISON_SCHEMA_VERSION: str = "brdt_objective_ablation_comparison_v1"
JSON_NAME: str = "comparison_to_supervised_plus_born.json"
CSV_NAME: str = "comparison_to_supervised_plus_born.csv"

_BLOCKING_IMAGE_KEYS = ("image_mae_phys", "image_rmse_phys", "image_relative_l2_phys")
_BLOCKING_MEAS_KEYS = ("meas_mae", "meas_rmse", "meas_relative_l2")
_SUPPORTING_KEYS = ("psnr_phys", "ssim_phys")


def _index_rows(metrics: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = metrics.get("rows") or []
    return {str(row["row_id"]): row for row in rows if "row_id" in row}


def _row_metrics(row: Mapping[str, Any], bucket: str, key: str) -> Optional[float]:
    block = row.get(bucket)
    if not isinstance(block, Mapping):
        return None
    val = block.get(key)
    if val is None or isinstance(val, str):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _delta(ablation: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if ablation is None or baseline is None:
        return None
    return float(ablation) - float(baseline)


def _row_runtime(row: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = row.get("runtime")
    return runtime if isinstance(runtime, Mapping) else {}


def _output_dynamic_range(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    runtime = _row_runtime(row)
    extras = runtime.get("extras")
    if not isinstance(extras, Mapping):
        return None
    odr = extras.get("output_dynamic_range")
    return odr if isinstance(odr, Mapping) else None


def _final_loss_breakdown(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    runtime = _row_runtime(row)
    extras = runtime.get("extras")
    if not isinstance(extras, Mapping):
        return None
    fl = extras.get("final_loss_breakdown")
    return fl if isinstance(fl, Mapping) else None


def build_comparison(
    *,
    baseline_metrics: Mapping[str, Any],
    ablation_metrics: Mapping[str, Any],
    selected_row_ids: Iterable[str],
    baseline_root: str,
    ablation_root: str,
    baseline_objective_preset: str = "supervised_plus_born",
    ablation_objective_preset: str = "relative_physics_only",
) -> Dict[str, Any]:
    """Build the machine-readable comparison payload for the selected rows."""
    baseline_idx = _index_rows(baseline_metrics)
    ablation_idx = _index_rows(ablation_metrics)
    rows_payload: List[Dict[str, Any]] = []
    for row_id in selected_row_ids:
        baseline_row = baseline_idx.get(row_id)
        ablation_row = ablation_idx.get(row_id)
        entry: Dict[str, Any] = {
            "row_id": row_id,
            "baseline_present": baseline_row is not None,
            "ablation_present": ablation_row is not None,
        }
        if baseline_row is None or ablation_row is None:
            rows_payload.append(entry)
            continue
        entry["paper_label"] = ablation_row.get("paper_label") or baseline_row.get(
            "paper_label"
        )
        entry["architecture"] = ablation_row.get("architecture") or baseline_row.get(
            "architecture"
        )
        # Blocking + supporting metric deltas.
        metric_deltas: Dict[str, Dict[str, Any]] = {}
        for bucket, keys in (
            ("image", _BLOCKING_IMAGE_KEYS),
            ("measurement", _BLOCKING_MEAS_KEYS),
            ("supporting", _SUPPORTING_KEYS),
        ):
            bucket_payload: Dict[str, Any] = {}
            for key in keys:
                base_v = _row_metrics(baseline_row, bucket, key)
                ablate_v = _row_metrics(ablation_row, bucket, key)
                bucket_payload[key] = {
                    "baseline": base_v,
                    "ablation": ablate_v,
                    "delta": _delta(ablate_v, base_v),
                }
            metric_deltas[bucket] = bucket_payload
        entry["metrics"] = metric_deltas
        # Runtime fields.
        baseline_runtime = _row_runtime(baseline_row)
        ablation_runtime = _row_runtime(ablation_row)
        entry["runtime"] = {
            "baseline": {
                "parameter_count": baseline_runtime.get("parameter_count"),
                "wall_time_train_s": baseline_runtime.get("wall_time_train_s"),
                "wall_time_eval_s": baseline_runtime.get("wall_time_eval_s"),
                "epochs": baseline_runtime.get("epochs"),
                "batch_size": baseline_runtime.get("batch_size"),
                "learning_rate": baseline_runtime.get("learning_rate"),
            },
            "ablation": {
                "parameter_count": ablation_runtime.get("parameter_count"),
                "wall_time_train_s": ablation_runtime.get("wall_time_train_s"),
                "wall_time_eval_s": ablation_runtime.get("wall_time_eval_s"),
                "epochs": ablation_runtime.get("epochs"),
                "batch_size": ablation_runtime.get("batch_size"),
                "learning_rate": ablation_runtime.get("learning_rate"),
            },
            "parameter_count_delta": _delta(
                ablation_runtime.get("parameter_count"),
                baseline_runtime.get("parameter_count"),
            ),
        }
        # Final loss breakdown for both, where present.
        entry["final_loss_breakdown"] = {
            "baseline": _final_loss_breakdown(baseline_row),
            "ablation": _final_loss_breakdown(ablation_row),
        }
        # Output dynamic-range diagnostics for collapse detection.
        entry["output_dynamic_range"] = {
            "baseline": _output_dynamic_range(baseline_row),
            "ablation": _output_dynamic_range(ablation_row),
        }
        rows_payload.append(entry)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "claim_boundary": "decision_support_append_only",
        "baseline": {
            "objective_preset": baseline_objective_preset,
            "root": baseline_root,
        },
        "ablation": {
            "objective_preset": ablation_objective_preset,
            "root": ablation_root,
        },
        "rows": rows_payload,
    }


def write_comparison_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_comparison_csv(payload: Mapping[str, Any], path: Path) -> None:
    fieldnames = [
        "row_id",
        "paper_label",
        "architecture",
        "image_mae_phys_baseline",
        "image_mae_phys_ablation",
        "image_mae_phys_delta",
        "image_rmse_phys_baseline",
        "image_rmse_phys_ablation",
        "image_rmse_phys_delta",
        "image_relative_l2_phys_baseline",
        "image_relative_l2_phys_ablation",
        "image_relative_l2_phys_delta",
        "meas_mae_baseline",
        "meas_mae_ablation",
        "meas_mae_delta",
        "meas_rmse_baseline",
        "meas_rmse_ablation",
        "meas_rmse_delta",
        "meas_relative_l2_baseline",
        "meas_relative_l2_ablation",
        "meas_relative_l2_delta",
        "psnr_phys_baseline",
        "psnr_phys_ablation",
        "psnr_phys_delta",
        "ssim_phys_baseline",
        "ssim_phys_ablation",
        "ssim_phys_delta",
        "parameter_count_baseline",
        "parameter_count_ablation",
        "wall_time_train_s_baseline",
        "wall_time_train_s_ablation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("rows") or []:
            metrics = row.get("metrics") or {}
            runtime = row.get("runtime") or {}
            base_rt = runtime.get("baseline") or {}
            ablate_rt = runtime.get("ablation") or {}
            flat: Dict[str, Any] = {
                "row_id": row.get("row_id"),
                "paper_label": row.get("paper_label", ""),
                "architecture": row.get("architecture", ""),
                "parameter_count_baseline": base_rt.get("parameter_count", ""),
                "parameter_count_ablation": ablate_rt.get("parameter_count", ""),
                "wall_time_train_s_baseline": base_rt.get("wall_time_train_s", ""),
                "wall_time_train_s_ablation": ablate_rt.get("wall_time_train_s", ""),
            }
            for bucket, keys in (
                ("image", _BLOCKING_IMAGE_KEYS),
                ("measurement", _BLOCKING_MEAS_KEYS),
                ("supporting", _SUPPORTING_KEYS),
            ):
                bucket_data = (metrics.get(bucket) or {})
                for key in keys:
                    md = bucket_data.get(key) or {}
                    flat[f"{key}_baseline"] = md.get("baseline", "")
                    flat[f"{key}_ablation"] = md.get("ablation", "")
                    flat[f"{key}_delta"] = md.get("delta", "")
            writer.writerow(flat)


def emit_comparison_artifacts(
    *,
    baseline_metrics_path: Path,
    ablation_metrics_path: Path,
    output_root: Path,
    selected_row_ids: Iterable[str],
    baseline_root: str,
    ablation_root: str,
    baseline_objective_preset: str = "supervised_plus_born",
    ablation_objective_preset: str = "relative_physics_only",
) -> Tuple[Path, Path]:
    """Read the two metrics files and write JSON+CSV comparison artifacts."""
    baseline = json.loads(Path(baseline_metrics_path).read_text())
    ablation = json.loads(Path(ablation_metrics_path).read_text())
    payload = build_comparison(
        baseline_metrics=baseline,
        ablation_metrics=ablation,
        selected_row_ids=list(selected_row_ids),
        baseline_root=baseline_root,
        ablation_root=ablation_root,
        baseline_objective_preset=baseline_objective_preset,
        ablation_objective_preset=ablation_objective_preset,
    )
    json_path = Path(output_root) / JSON_NAME
    csv_path = Path(output_root) / CSV_NAME
    write_comparison_json(payload, json_path)
    write_comparison_csv(payload, csv_path)
    return json_path, csv_path

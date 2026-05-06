"""Convergence audit and promotion-gate helpers for BRDT paper evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


CONVERGENCE_AUDIT_SCHEMA_VERSION = "brdt_convergence_audit_v1"
PAPER_EVIDENCE_GATE_SCHEMA_VERSION = "brdt_paper_evidence_gate_v1"

_METRIC_BUCKETS = ("image_metrics", "measurement_metrics", "supporting")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_history(path: Path) -> Dict[str, Any]:
    return _read_json(Path(path))


def summarize_history(*, row_id: str, history_payload: Mapping[str, Any]) -> Dict[str, Any]:
    epochs = list(history_payload.get("epochs") or [])
    losses = [float(entry.get("train_total_loss", 0.0)) for entry in epochs]
    lrs = [float(entry.get("learning_rate", 0.0)) for entry in epochs]
    lr_reduction_count = sum(1 for entry in epochs if bool(entry.get("lr_reduced")))

    def _window_delta(window: int) -> Optional[float]:
        if len(losses) < window:
            return None
        return float(losses[-1] - losses[-window])

    last5_delta = _window_delta(5)
    last10_delta = _window_delta(10)
    improving = False
    if last5_delta is not None:
        improving = bool(last5_delta < -1e-6)
    elif len(losses) >= 2:
        improving = bool(losses[-1] < (losses[-2] - 1e-6))

    return {
        "row_id": row_id,
        "history_records": int(len(epochs)),
        "final_learning_rate": float(lrs[-1]) if lrs else None,
        "lr_reduction_count": int(lr_reduction_count),
        "last5_loss_delta": last5_delta,
        "last10_loss_delta": last10_delta,
        "materially_improving_at_stop": bool(improving),
    }


def build_convergence_audit(
    *,
    backlog_item: str,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    current_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = []
    for row_id, current in current_rows.items():
        baseline = dict(baseline_rows.get(row_id) or {})
        current_summary = dict(current.get("row_summary") or {})
        history_summary = dict(current.get("history_summary") or {})
        row_payload: Dict[str, Any] = {
            "row_id": row_id,
            "paper_label": current_summary.get("paper_label", row_id),
            "history_records": history_summary.get("history_records"),
            "convergence": history_summary,
            "status": current_summary.get("row_status"),
            "metric_deltas": {},
        }
        for bucket in _METRIC_BUCKETS:
            current_bucket = dict(current_summary.get(bucket) or {})
            baseline_bucket = dict(baseline.get(bucket) or {})
            deltas: Dict[str, Any] = {}
            for key, current_value in current_bucket.items():
                base_value = baseline_bucket.get(key)
                if isinstance(current_value, (int, float)) and isinstance(
                    base_value, (int, float)
                ):
                    deltas[key] = float(current_value) - float(base_value)
                else:
                    deltas[key] = None
            row_payload["metric_deltas"][bucket] = deltas
        current_runtime = dict(current_summary.get("runtime") or {})
        baseline_runtime = dict(baseline.get("runtime") or {})
        row_payload["runtime_deltas"] = {
            "parameter_count": (
                float(current_runtime.get("parameter_count"))
                - float(baseline_runtime.get("parameter_count"))
                if isinstance(current_runtime.get("parameter_count"), (int, float))
                and isinstance(baseline_runtime.get("parameter_count"), (int, float))
                else None
            ),
            "wall_time_train_s": (
                float(current_runtime.get("wall_time_train_s"))
                - float(baseline_runtime.get("wall_time_train_s"))
                if isinstance(current_runtime.get("wall_time_train_s"), (int, float))
                and isinstance(baseline_runtime.get("wall_time_train_s"), (int, float))
                else None
            ),
        }
        rows.append(row_payload)
    return {
        "schema_version": CONVERGENCE_AUDIT_SCHEMA_VERSION,
        "backlog_item": backlog_item,
        "rows": rows,
    }


def write_convergence_audit_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_convergence_audit_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "row_id",
                "paper_label",
                "status",
                "history_records",
                "final_learning_rate",
                "lr_reduction_count",
                "last5_loss_delta",
                "last10_loss_delta",
                "materially_improving_at_stop",
            ],
        )
        writer.writeheader()
        for row in payload.get("rows") or []:
            conv = dict(row.get("convergence") or {})
            writer.writerow(
                {
                    "row_id": row.get("row_id"),
                    "paper_label": row.get("paper_label"),
                    "status": row.get("status"),
                    "history_records": row.get("history_records"),
                    "final_learning_rate": conv.get("final_learning_rate"),
                    "lr_reduction_count": conv.get("lr_reduction_count"),
                    "last5_loss_delta": conv.get("last5_loss_delta"),
                    "last10_loss_delta": conv.get("last10_loss_delta"),
                    "materially_improving_at_stop": conv.get(
                        "materially_improving_at_stop"
                    ),
                }
            )


def build_paper_evidence_gate(
    *,
    backlog_item: str,
    expected_epochs: int,
    rows: Mapping[str, Mapping[str, Any]],
    provenance_checks: Mapping[str, bool],
) -> Dict[str, Any]:
    failed_gate_checks = []
    any_blocked = False
    for row_id, payload in rows.items():
        status = str(payload.get("row_status", "unknown"))
        if status == "blocked":
            any_blocked = True
        if status != "completed":
            failed_gate_checks.append(f"{row_id}.row_status")
        if int(payload.get("history_records", 0)) != int(expected_epochs):
            failed_gate_checks.append(f"{row_id}.history_records")
        if not bool(payload.get("scheduler_matches_contract", False)):
            failed_gate_checks.append(f"{row_id}.scheduler")
    for key, ok in provenance_checks.items():
        if not bool(ok):
            failed_gate_checks.append(str(key))
    promotion_status = "passed"
    claim_boundary = "paper_evidence_brdt_additive"
    row_status = "completed"
    if any_blocked:
        promotion_status = "blocked"
        claim_boundary = "decision_support_convergence_followup"
        row_status = "blocked"
    elif failed_gate_checks:
        promotion_status = "failed"
        claim_boundary = "decision_support_convergence_followup"
    return {
        "schema_version": PAPER_EVIDENCE_GATE_SCHEMA_VERSION,
        "backlog_item": backlog_item,
        "expected_epochs": int(expected_epochs),
        "claim_boundary": claim_boundary,
        "promotion_status": promotion_status,
        "row_status": row_status,
        "failed_gate_checks": failed_gate_checks,
        "rows": {str(k): dict(v) for k, v in rows.items()},
        "provenance_checks": {str(k): bool(v) for k, v in provenance_checks.items()},
    }


def write_paper_evidence_gate(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")

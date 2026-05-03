"""Metric helpers for the BRDT four-row preflight.

This module owns the metric schema, per-sample reductions, and the
machine-readable JSON/CSV/schema serialization for the bounded
decision-support preflight bundle. Image-space metrics are computed on
physical ``q``; measurement-space metrics are computed on the (real,
imag) sinogram tensor produced by the locked operator.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


METRIC_SCHEMA_VERSION: str = "brdt_preflight_metrics_v1"

IMAGE_METRICS = ("image_mae_phys", "image_rmse_phys", "image_relative_l2_phys")
MEASUREMENT_METRICS = ("meas_mae", "meas_rmse", "meas_relative_l2")
SUPPORTING_METRICS = ("psnr_phys", "ssim_phys")
RUNTIME_FIELDS = (
    "wall_time_train_s",
    "wall_time_eval_s",
    "device",
    "device_name",
    "epochs",
    "batch_size",
    "learning_rate",
    "parameter_count",
    "row_status",
)


@dataclass
class RowMetrics:
    """Aggregated per-row metric payload."""

    row_id: str
    paper_label: str
    architecture: str
    row_status: str
    image: Dict[str, float] = field(default_factory=dict)
    measurement: Dict[str, float] = field(default_factory=dict)
    supporting: Dict[str, float] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    blocker_reason: Optional[str] = None
    blocker_message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "row_id": self.row_id,
            "paper_label": self.paper_label,
            "architecture": self.architecture,
            "row_status": self.row_status,
            "image": dict(self.image),
            "measurement": dict(self.measurement),
            "runtime": dict(self.runtime),
        }
        if self.supporting:
            payload["supporting"] = dict(self.supporting)
        if self.blocker_reason:
            payload["blocker_reason"] = self.blocker_reason
        if self.blocker_message:
            payload["blocker_message"] = self.blocker_message
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def _safe_norm(arr: np.ndarray) -> float:
    n = float(np.linalg.norm(arr.astype(np.float64).reshape(-1)))
    return n if n > 1e-30 else 0.0


def image_space_metrics_phys(
    pred_phys: np.ndarray, target_phys: np.ndarray
) -> Dict[str, float]:
    """MAE/RMSE/relative-L2 on physical ``q`` images.

    Both inputs must share shape ``(B, 1, N, N)`` or ``(B, N, N)`` and be
    in physical-q units. Returns a flat dict keyed by ``image_*``.
    """
    pred = pred_phys.astype(np.float64)
    targ = target_phys.astype(np.float64)
    diff = pred - targ
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = _safe_norm(targ)
    rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / denom) if denom > 0 else float("inf")
    return {
        IMAGE_METRICS[0]: mae,
        IMAGE_METRICS[1]: rmse,
        IMAGE_METRICS[2]: rel_l2,
    }


def measurement_space_metrics(
    pred_sino: np.ndarray, target_sino: np.ndarray
) -> Dict[str, float]:
    """MAE/RMSE/relative-L2 on the (real, imag) sinogram tensor."""
    pred = pred_sino.astype(np.float64)
    targ = target_sino.astype(np.float64)
    diff = pred - targ
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = _safe_norm(targ)
    rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / denom) if denom > 0 else float("inf")
    return {
        MEASUREMENT_METRICS[0]: mae,
        MEASUREMENT_METRICS[1]: rmse,
        MEASUREMENT_METRICS[2]: rel_l2,
    }


def supporting_image_metrics(
    pred_phys: np.ndarray, target_phys: np.ndarray
) -> Dict[str, float]:
    """Optional supporting diagnostics: PSNR proxy and naive SSIM substitute.

    These are diagnostic only; the blocking metrics are MAE/RMSE/rel-L2.
    PSNR uses the target peak-to-peak amplitude as the dynamic range so
    rows can be compared on the same scale even when ``q`` ranges differ.
    """
    pred = pred_phys.astype(np.float64)
    targ = target_phys.astype(np.float64)
    diff = pred - targ
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        psnr = float("inf")
    else:
        peak = float(np.max(targ) - np.min(targ))
        psnr = float(20.0 * np.log10(peak / np.sqrt(mse))) if peak > 0 else float("nan")
    # Mean correlation as a lightweight similarity proxy (not full SSIM).
    p_flat = pred.reshape(pred.shape[0], -1)
    t_flat = targ.reshape(targ.shape[0], -1)
    pf = p_flat - p_flat.mean(axis=1, keepdims=True)
    tf = t_flat - t_flat.mean(axis=1, keepdims=True)
    num = float(np.mean(np.sum(pf * tf, axis=1)))
    denom = float(
        np.mean(np.sqrt(np.sum(pf * pf, axis=1) * np.sum(tf * tf, axis=1)))
    )
    sim = float(num / denom) if denom > 0 else float("nan")
    return {"psnr_phys": psnr, "ssim_phys": sim}


def aggregate_image_metrics_per_sample(
    per_sample_pred: Iterable[np.ndarray],
    per_sample_target: Iterable[np.ndarray],
) -> Dict[str, float]:
    """Per-sample then mean-reduced image metrics."""
    preds = list(per_sample_pred)
    targs = list(per_sample_target)
    if len(preds) != len(targs):
        raise ValueError("per-sample arrays must share length")
    if not preds:
        return {k: float("nan") for k in IMAGE_METRICS}
    rows = [image_space_metrics_phys(p, t) for p, t in zip(preds, targs)]
    return {
        k: float(np.mean([r[k] for r in rows])) for k in IMAGE_METRICS
    }


def write_metric_schema(path: Path) -> None:
    schema = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "blocking_metrics": {
            "image_space_physical_q": list(IMAGE_METRICS),
            "measurement_space": list(MEASUREMENT_METRICS),
        },
        "supporting_metrics": list(SUPPORTING_METRICS),
        "runtime_fields": list(RUNTIME_FIELDS),
        "claim_boundary": "decision_support_preflight_only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")


def write_metrics_json(path: Path, rows: List[RowMetrics]) -> None:
    payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "claim_boundary": "decision_support_preflight_only",
        "rows": [r.to_dict() for r in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_metrics_csv(path: Path, rows: List[RowMetrics]) -> None:
    """Flat CSV with one row per preflight row."""
    fieldnames = [
        "row_id",
        "paper_label",
        "architecture",
        "row_status",
        *IMAGE_METRICS,
        *MEASUREMENT_METRICS,
        *SUPPORTING_METRICS,
        "parameter_count",
        "wall_time_train_s",
        "wall_time_eval_s",
        "epochs",
        "batch_size",
        "learning_rate",
        "device",
        "device_name",
        "blocker_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row: Dict[str, Any] = {
                "row_id": r.row_id,
                "paper_label": r.paper_label,
                "architecture": r.architecture,
                "row_status": r.row_status,
                "blocker_reason": r.blocker_reason or "",
            }
            for k in IMAGE_METRICS:
                row[k] = r.image.get(k, "")
            for k in MEASUREMENT_METRICS:
                row[k] = r.measurement.get(k, "")
            for k in SUPPORTING_METRICS:
                row[k] = r.supporting.get(k, "")
            for k in (
                "parameter_count",
                "wall_time_train_s",
                "wall_time_eval_s",
                "epochs",
                "batch_size",
                "learning_rate",
                "device",
                "device_name",
            ):
                row[k] = r.runtime.get(k, "")
            writer.writerow(row)


def collect_runtime_metadata(
    *,
    device: str,
    device_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    parameter_count: int,
    wall_time_train_s: float,
    wall_time_eval_s: float,
    row_status: str,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "device": str(device),
        "device_name": str(device_name),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "parameter_count": int(parameter_count),
        "wall_time_train_s": float(wall_time_train_s),
        "wall_time_eval_s": float(wall_time_eval_s),
        "row_status": str(row_status),
    }
    if extras:
        payload["extras"] = dict(extras)
    return payload

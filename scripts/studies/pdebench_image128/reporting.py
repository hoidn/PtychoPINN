"""Reporting payloads for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def darcy_literature_context() -> dict[str, Any]:
    caveat = (
        "Published values are calibration context, not exact native-128 local pass/fail thresholds; "
        "PDEBench/HAMLET commonly use x2 spatial subsampling and different training protocols."
    )
    return {
        "schema_version": "pdebench_darcy_literature_context_v1",
        "task_id": "darcy",
        "access_date": "2026-04-20",
        "sources": {
            "pdebench_repository": "https://github.com/pdebench/PDEBench",
            "pdebench_supplement_table_8": "https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Supplemental-Datasets_and_Benchmarks.pdf",
            "fno_jmlr": "https://jmlr.org/papers/volume24/21-1524/21-1524.pdf",
            "hamlet_openreview": "https://openreview.net/pdf/2a77316b975457831c5d7c21021f5bb87c99eff1.pdf",
            "pdebench_darcy_config": "https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/models/config/args/config_Darcy.yaml",
        },
        "calibration_targets": {
            "pdebench_unet": {"model": "U-Net", "RMSE": 6.4e-3, "nRMSE": 3.3e-2, "protocol_caveat": caveat},
            "pdebench_fno": {"model": "FNO", "RMSE": 1.2e-2, "nRMSE": 6.4e-2, "protocol_caveat": caveat},
            "oformer": {"model": "OFormer", "nRMSE": 2.05e-2, "protocol_caveat": caveat},
            "hamlet": {"model": "HAMLET", "nRMSE": 1.40e-2, "protocol_caveat": caveat},
            "fno_jmlr": {
                "model": "FNO",
                "relative_error_range": [1.08e-2, 9.8e-3],
                "protocol_caveat": "Canonical Darcy neural-operator context, not the exact PDEBench beta-file protocol.",
            },
        },
    }


def write_literature_context(output_root: Path, *, task_id: str = "darcy") -> Path:
    if task_id != "darcy":
        raise ValueError("only Darcy literature context is implemented")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "literature_context.json"
    path.write_text(json.dumps(darcy_literature_context(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_comparison_summary(
    *,
    task_id: str,
    mode: str,
    output_root: Path,
    profile_results: list[dict[str, Any]],
    run_id: str | None = None,
    blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    profile_ids = [str(item.get("profile_id")) for item in profile_results]
    if mode == "benchmark" and "unet_tiny_smoke" in profile_ids:
        raise ValueError("unet_tiny_smoke is readiness-only and cannot be included as a strong benchmark baseline")
    required = {"hybrid_resnet_base", "fno_base", "unet_strong"}
    completed = {str(item.get("profile_id")) for item in profile_results if item.get("status") == "completed"}
    performance_complete = mode == "benchmark" and required.issubset(completed)
    evidence_scope = "benchmark_performance" if performance_complete else "smoke_feasibility_only"
    if mode == "benchmark" and not performance_complete:
        evidence_scope = "blocked_or_incomplete_benchmark"
    return {
        "schema_version": "pdebench_image128_comparison_summary_v1",
        "task_id": task_id,
        "mode": mode,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(Path(output_root)),
        "profile_results": profile_results,
        "required_primary_profiles": sorted(required),
        "performance_assessment_complete": bool(performance_complete),
        "evidence_scope": evidence_scope,
        "metric_interpretation": (
            "benchmark_performance" if performance_complete else "sanity_only_not_benchmark_performance"
        ),
        "literature_context": darcy_literature_context(),
        "blockers": blockers or [],
    }


def write_comparison_summary(payload: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "comparison_summary.json"
    csv_path = output_root / "comparison_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = payload.get("profile_results", [])
    fields = ["profile_id", "status", "err_RMSE", "err_nRMSE", "relative_l2", "parameter_count", "blocker_reason"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return json_path, csv_path

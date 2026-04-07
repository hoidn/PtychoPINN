#!/usr/bin/env python3
"""Collate metrics and comparison PNGs from NERSC N=128 factorial runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

DATASETS = ("cameraman256", "scan807")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_factorial_manifest(factorial_root: Path) -> dict[str, Any]:
    manifest_path = factorial_root / "factorial_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing factorial manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _extract_factors(run_entry: dict[str, Any]) -> dict[str, Any]:
    factors = dict(run_entry.get("factors", {}))
    return {
        "probe_mode": factors.get("probe_mode"),
        "probe_mask": factors.get("probe_mask"),
        "probe_mask_sigma": factors.get("probe_mask_sigma"),
        "probe_mask_diameter": factors.get("probe_mask_diameter"),
        "torch_mae_pred_l2_match_target": factors.get("torch_mae_pred_l2_match_target"),
        "downsample_policy": factors.get("downsample_policy"),
    }


def collate_factorial_results(
    *,
    factorial_root: Path,
    shared_dir: Path,
) -> dict[str, Any]:
    manifest = _load_factorial_manifest(factorial_root)
    shared_png_dir = shared_dir / "shared_pngs"
    shared_png_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    copied_pngs: list[str] = []

    for run in manifest.get("runs", []):
        run_id = run.get("run_id")
        run_output = Path(run.get("output_dir", factorial_root / "runs" / str(run_id)))
        factors = _extract_factors(run)

        for dataset in DATASETS:
            metrics_path = run_output / dataset / "metrics_by_model.json"
            png_path = run_output / dataset / "visuals" / "compare_amp_phase.png"

            if metrics_path.exists():
                metrics_by_model = json.loads(metrics_path.read_text())
                for model_id, payload in metrics_by_model.items():
                    metrics = payload.get("metrics", {})
                    row: dict[str, Any] = {
                        "run_id": run_id,
                        "dataset": dataset,
                        "model_id": model_id,
                        **factors,
                    }
                    for key, value in metrics.items():
                        numeric = _safe_float(value)
                        if numeric is not None:
                            row[key] = numeric
                    metric_rows.append(row)
            else:
                missing_artifacts.append(str(metrics_path))

            if png_path.exists():
                dest_name = f"{run_id}__dataset-{dataset}__compare_amp_phase.png"
                dest_path = shared_png_dir / dest_name
                shutil.copy2(png_path, dest_path)
                copied_pngs.append(str(dest_path))
            else:
                missing_artifacts.append(str(png_path))

    shared_dir.mkdir(parents=True, exist_ok=True)
    csv_path = shared_dir / "metrics_summary.csv"
    md_path = shared_dir / "metrics_summary.md"

    factor_columns = [
        "run_id",
        "dataset",
        "model_id",
        "probe_mode",
        "probe_mask",
        "probe_mask_sigma",
        "probe_mask_diameter",
        "torch_mae_pred_l2_match_target",
        "downsample_policy",
    ]
    metric_columns = sorted(
        {
            key
            for row in metric_rows
            for key in row.keys()
            if key not in set(factor_columns)
        }
    )
    fieldnames = factor_columns + metric_columns

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    preview_cols = [
        "run_id",
        "dataset",
        "model_id",
        "probe_mode",
        "torch_mae_pred_l2_match_target",
        "downsample_policy",
        "mae",
        "mse",
        "psnr",
        "ssim",
        "ms_ssim",
    ]
    preview_cols = [col for col in preview_cols if col in fieldnames]

    lines = [
        "# NERSC N128 Factorial Summary",
        "",
        f"- Factorial root: `{factorial_root}`",
        f"- Runs listed: {len(manifest.get('runs', []))}",
        f"- Metric rows: {len(metric_rows)}",
        f"- Copied PNGs: {len(copied_pngs)}",
        f"- Missing artifacts: {len(missing_artifacts)}",
        "",
        f"- CSV: `{csv_path}`",
        f"- PNG directory: `{shared_png_dir}`",
        "",
    ]
    if missing_artifacts:
        lines.append("## Missing Artifacts")
        lines.append("")
        for path in missing_artifacts:
            lines.append(f"- `{path}`")
        lines.append("")

    lines.append("## Metrics Preview")
    lines.append("")
    if metric_rows and preview_cols:
        lines.append("| " + " | ".join(preview_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(preview_cols)) + "|")
        for row in metric_rows:
            values = [str(row.get(col, "")) for col in preview_cols]
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("_No metric rows found._")
    lines.append("")
    md_path.write_text("\n".join(lines))

    return {
        "factorial_root": str(factorial_root),
        "shared_dir": str(shared_dir),
        "metrics_summary_csv": str(csv_path),
        "metrics_summary_md": str(md_path),
        "shared_png_dir": str(shared_png_dir),
        "metric_rows": len(metric_rows),
        "copied_png_count": len(copied_pngs),
        "missing_artifacts": missing_artifacts,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collate metrics and compare_amp_phase PNGs from an N=128 factorial NERSC study."
        )
    )
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = collate_factorial_results(
        factorial_root=args.factorial_root,
        shared_dir=args.shared_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Capture photon_grid dose baseline facts for maintainer coordination.

Walks dataset-root for data_p1e*.npz files, loads baseline params via dill,
and emits JSON + Markdown summaries documenting the scenario metadata,
dataset table, and metrics.

Usage:
    python plans/active/seed/bin/dose_baseline_snapshot.py \
        --dataset-root photon_grid_study_20250826_152459 \
        --baseline-params photon_grid_study_20250826_152459/results_p1e5/.../params.dill \
        --scenario-id PGRID-20250826-P1E5-T1024 \
        --output plans/active/seed/reports/2026-01-22T024002Z
"""
import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np


def sha256_file(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_for_json(obj):
    """Convert numpy/tensor types to JSON-serializable Python types."""
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return float(obj.item())
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, tuple):
        return [convert_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description="Capture dose baseline snapshot")
    parser.add_argument(
        "--dataset-root",
        default="photon_grid_study_20250826_152459",
        help="Root directory containing data_p1e*.npz files",
    )
    parser.add_argument(
        "--baseline-params",
        default="photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill",
        help="Path to baseline params.dill file",
    )
    parser.add_argument(
        "--scenario-id",
        default="PGRID-20250826-P1E5-T1024",
        help="Scenario identifier string",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for JSON and Markdown files",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect dataset files
    npz_files = sorted(dataset_root.glob("data_p1e*.npz"))
    if not npz_files:
        print(f"ERROR: No data_p1e*.npz files found in {dataset_root}", file=sys.stderr)
        sys.exit(1)

    dataset_table = []
    for npz_path in npz_files:
        sha = sha256_file(str(npz_path))
        with np.load(str(npz_path), allow_pickle=True) as d:
            arrays_info = {}
            for k in d.keys():
                arr = d[k]
                arrays_info[k] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }

        # Extract photon dose from filename (e.g., data_p1e5.npz -> 1e5)
        fname = npz_path.name
        dose_str = fname.replace("data_p", "").replace(".npz", "")

        dataset_table.append({
            "filename": fname,
            "path": str(npz_path),
            "sha256": sha,
            "photon_dose": dose_str,
            "arrays": arrays_info,
            "size_bytes": npz_path.stat().st_size,
        })

    # Load baseline params
    import dill

    with open(args.baseline_params, "rb") as f:
        params = dill.load(f)

    # Extract key metrics (handle potential tensor types)
    key_params = {
        "N": params.get("N"),
        "gridsize": params.get("gridsize"),
        "nimgs_train": params.get("nimgs_train"),
        "nimgs_test": params.get("nimgs_test"),
        "batch_size": params.get("batch_size"),
        "nepochs": params.get("nepochs"),
        "mae_weight": params.get("mae_weight"),
        "nll_weight": params.get("nll_weight"),
        "default_probe_scale": params.get("default_probe_scale"),
        "intensity_scale.trainable": params.get("intensity_scale.trainable"),
        "probe.trainable": params.get("probe.trainable"),
        "probe.mask": params.get("probe.mask"),
        "label": params.get("label"),
        "output_prefix": params.get("output_prefix"),
        "nphotons_in_params": params.get("nphotons"),
        "timestamp": params.get("timestamp"),
    }

    # Extract metrics (tuples of train/test values)
    metrics = {
        "mae": params.get("mae"),
        "ms_ssim": params.get("ms_ssim"),
        "psnr": params.get("psnr"),
        "frc50": params.get("frc50"),
        "mse": params.get("mse"),
    }

    # Convert intensity_scale tensor to float if present
    intensity_scale_val = params.get("intensity_scale")
    if intensity_scale_val is not None:
        if hasattr(intensity_scale_val, "numpy"):
            intensity_scale_val = float(intensity_scale_val.numpy())
        elif isinstance(intensity_scale_val, np.ndarray):
            intensity_scale_val = float(intensity_scale_val)
    key_params["intensity_scale_value"] = intensity_scale_val

    # Build summary
    summary = {
        "scenario_id": args.scenario_id,
        "dataset_root": str(dataset_root),
        "baseline_params_path": args.baseline_params,
        "key_params": convert_for_json(key_params),
        "metrics": convert_for_json(metrics),
        "datasets": dataset_table,
        "total_datasets": len(dataset_table),
    }

    # Write JSON
    json_path = output_dir / "dose_baseline_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {json_path}")

    # Write Markdown
    md_path = output_dir / "dose_baseline_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Dose Baseline Summary\n\n")
        f.write(f"**Scenario ID:** {args.scenario_id}\n\n")
        f.write(f"**Dataset Root:** `{dataset_root}`\n\n")
        f.write(f"**Baseline Params:** `{args.baseline_params}`\n\n")

        f.write("## Key Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for k, v in key_params.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n## Metrics (train, test)\n\n")
        f.write("| Metric | Train | Test |\n")
        f.write("|--------|-------|------|\n")
        for k, v in metrics.items():
            if v is not None:
                train_v = convert_for_json(v[0]) if len(v) > 0 else "N/A"
                test_v = convert_for_json(v[1]) if len(v) > 1 else "N/A"
                # Format floats nicely
                if isinstance(train_v, float):
                    train_v = f"{train_v:.6g}"
                if isinstance(test_v, float):
                    test_v = f"{test_v:.6g}"
                f.write(f"| {k} | {train_v} | {test_v} |\n")

        f.write("\n## Dataset Files\n\n")
        f.write("| File | Photon Dose | Patterns | Diff Shape | SHA256 (truncated) |\n")
        f.write("|------|-------------|----------|------------|--------------------|\n")
        for ds in dataset_table:
            diff_shape = ds["arrays"].get("diff3d", {}).get("shape", "N/A")
            n_patterns = diff_shape[0] if isinstance(diff_shape, list) and len(diff_shape) > 0 else "N/A"
            sha_short = ds["sha256"][:16] + "..."
            f.write(f"| {ds['filename']} | {ds['photon_dose']} | {n_patterns} | {diff_shape} | {sha_short} |\n")

        f.write(f"\n**Total datasets:** {len(dataset_table)}\n")
        f.write(f"\n**Full JSON:** See `dose_baseline_summary.json` in the same directory.\n")

    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()

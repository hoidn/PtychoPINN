#!/usr/bin/env python
"""D0 Parity Logger — Capture dose parity evidence for maintainer coordination.

Walks dataset-root for data_p1e*.npz files, loads baseline params via dill,
and emits JSON + Markdown + CSV summaries documenting the scenario metadata,
dataset statistics (raw, grouped, normalized), probe stats, and metrics.

CLI reference: plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md
Data contracts: specs/data_contracts.md

Usage:
    python scripts/tools/d0_parity_logger.py \\
        --dataset-root photon_grid_study_20250826_152459 \\
        --baseline-params photon_grid_study_20250826_152459/results_p1e5/.../params.dill \\
        --scenario-id PGRID-20250826-P1E5-T1024 \\
        --output plans/active/seed/reports/2026-01-22T042640Z
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np


def sha256_file(filepath: str) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to the file to hash.

    Returns:
        Hex digest of the SHA256 hash.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_for_json(obj: Any) -> Any:
    """Convert numpy/tensor types to JSON-serializable Python types.

    Args:
        obj: Object to convert (may be numpy array, tensor, dict, etc.)

    Returns:
        JSON-serializable equivalent.
    """
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return float(obj.item())
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, tuple):
        return [convert_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(x) for x in obj]
    return obj


def summarize_array(arr: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a numeric array.

    Handles complex arrays by computing magnitude.

    Args:
        arr: Input numpy array.

    Returns:
        Dictionary with min, max, mean, std, median, percentiles,
        and nonzero_fraction.
    """
    # Handle complex arrays by taking magnitude
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    flat = arr.ravel().astype(np.float64)

    if flat.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p01": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "nonzero_fraction": 0.0,
            "count": 0,
        }

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p10": float(np.percentile(flat, 10)),
        "p90": float(np.percentile(flat, 90)),
        "p99": float(np.percentile(flat, 99)),
        "nonzero_fraction": float(np.count_nonzero(flat) / flat.size),
        "count": int(flat.size),
    }


def summarize_grouped(diff3d: np.ndarray, scan_index: np.ndarray) -> Dict[str, Any]:
    """Compute per-scan grouped intensity statistics.

    For each unique scan index, computes the mean intensity across that scan's
    patterns. Then reports statistics over those per-scan averages.

    Args:
        diff3d: Diffraction patterns array (num_patterns x N x N).
        scan_index: Array mapping each pattern to its scan group.

    Returns:
        Dictionary with grouped statistics and number of unique scans.
    """
    # Handle complex arrays
    if np.iscomplexobj(diff3d):
        diff3d = np.abs(diff3d)

    diff3d = diff3d.astype(np.float64)
    scan_index = np.asarray(scan_index).ravel()

    # Compute per-pattern mean intensity
    per_pattern_mean = np.mean(diff3d, axis=(1, 2)) if diff3d.ndim == 3 else diff3d.ravel()

    # Group by scan_index using bincount
    n_scans = int(np.max(scan_index)) + 1
    counts = np.bincount(scan_index.astype(int), minlength=n_scans)
    sums = np.bincount(scan_index.astype(int), weights=per_pattern_mean, minlength=n_scans)

    # Avoid division by zero for unused bins
    valid_mask = counts > 0
    per_scan_means = np.zeros(n_scans)
    per_scan_means[valid_mask] = sums[valid_mask] / counts[valid_mask]

    # Only consider scans that have patterns
    active_scan_means = per_scan_means[valid_mask]

    stats = summarize_array(active_scan_means)
    stats["n_unique_scans"] = int(np.sum(valid_mask))
    stats["n_patterns"] = int(len(scan_index))

    return stats


def summarize_probe(probe_array: np.ndarray) -> Dict[str, Any]:
    """Compute amplitude and phase statistics for a probe array.

    Args:
        probe_array: Complex probe array.

    Returns:
        Dictionary with amplitude and phase stats, each containing
        min, max, mean, std, and percentiles.
    """
    amplitude = np.abs(probe_array)
    phase = np.angle(probe_array)

    def percentile_stats(arr: np.ndarray) -> Dict[str, float]:
        flat = arr.ravel().astype(np.float64)
        if flat.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                    "p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "p01": float(np.percentile(flat, 1)),
            "p05": float(np.percentile(flat, 5)),
            "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)),
            "p99": float(np.percentile(flat, 99)),
        }

    return {
        "amplitude": percentile_stats(amplitude),
        "phase": percentile_stats(phase),
        "l2_norm": float(np.linalg.norm(probe_array.ravel())),
        "shape": list(probe_array.shape),
        "dtype": str(probe_array.dtype),
    }


def load_params(path: str) -> Dict[str, Any]:
    """Load params from a dill file and convert to plain Python types.

    Args:
        path: Path to the params.dill file.

    Returns:
        Dictionary with params converted to JSON-serializable types.
    """
    import dill

    with open(path, "rb") as f:
        params = dill.load(f)

    # Convert to dict if needed
    if hasattr(params, "__dict__"):
        params = dict(params)
    elif hasattr(params, "items"):
        params = dict(params)
    else:
        # Try to iterate as a dict-like object
        try:
            params = {k: params[k] for k in params.keys()}
        except (AttributeError, TypeError):
            params = {"raw": str(params)}

    return convert_for_json(params)


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def process_dataset(npz_path: Path) -> Dict[str, Any]:
    """Process a single NPZ dataset file.

    Args:
        npz_path: Path to the NPZ file.

    Returns:
        Dictionary with dataset metadata and statistics.
    """
    sha = sha256_file(str(npz_path))

    result = {
        "filename": npz_path.name,
        "path": str(npz_path),
        "sha256": sha,
        "size_bytes": npz_path.stat().st_size,
        "arrays": {},
        "stats": {},
    }

    # Extract photon dose from filename (e.g., data_p1e5.npz -> 1e5)
    fname = npz_path.name
    dose_str = fname.replace("data_p", "").replace(".npz", "")
    result["photon_dose"] = dose_str

    with np.load(str(npz_path), allow_pickle=True) as data:
        # Record array shapes/dtypes
        for key in data.keys():
            arr = data[key]
            result["arrays"][key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        # Compute stage-level stats if diff3d exists
        if "diff3d" in data:
            diff3d = data["diff3d"]

            # Raw diffraction stats
            result["stats"]["raw"] = summarize_array(diff3d)

            # Normalized diffraction stats (divide by max + epsilon)
            max_val = np.max(np.abs(diff3d)) if np.iscomplexobj(diff3d) else np.max(diff3d)
            normalized = diff3d / (max_val + 1e-12)
            result["stats"]["normalized"] = summarize_array(normalized)

            # Grouped stats if scan_index exists
            if "scan_index" in data:
                scan_index = data["scan_index"]
                result["stats"]["grouped"] = summarize_grouped(diff3d, scan_index)

        # Probe stats if probeGuess exists
        if "probeGuess" in data:
            result["probe"] = summarize_probe(data["probeGuess"])

    return result


def extract_baseline_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key baseline parameters for the summary.

    Args:
        params: Full params dictionary from dill file.

    Returns:
        Dictionary with relevant baseline params.
    """
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

    # Handle intensity_scale value
    intensity_scale_val = params.get("intensity_scale")
    if intensity_scale_val is not None:
        if isinstance(intensity_scale_val, (list, tuple)) and len(intensity_scale_val) > 0:
            intensity_scale_val = intensity_scale_val[0]
        if hasattr(intensity_scale_val, "numpy"):
            intensity_scale_val = float(intensity_scale_val.numpy())
        elif isinstance(intensity_scale_val, np.ndarray):
            intensity_scale_val = float(intensity_scale_val.flat[0])
        elif isinstance(intensity_scale_val, (int, float)):
            intensity_scale_val = float(intensity_scale_val)
    key_params["intensity_scale_value"] = intensity_scale_val

    return key_params


def extract_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inference metrics from params.

    Args:
        params: Full params dictionary.

    Returns:
        Dictionary with metrics as [train, test] pairs.
    """
    metrics = {}
    for metric_name in ["mae", "ms_ssim", "psnr", "mse", "frc50"]:
        val = params.get(metric_name)
        if val is not None:
            metrics[metric_name] = convert_for_json(val)
    return metrics


def write_probe_csv(datasets: List[Dict[str, Any]], output_path: Path) -> None:
    """Write probe statistics to CSV.

    Args:
        datasets: List of dataset dictionaries.
        output_path: Path to write CSV file.
    """
    rows = []
    for ds in datasets:
        if "probe" not in ds:
            continue
        probe = ds["probe"]
        amp = probe.get("amplitude", {})
        phase = probe.get("phase", {})
        rows.append({
            "filename": ds["filename"],
            "photon_dose": ds["photon_dose"],
            "amp_min": amp.get("min"),
            "amp_max": amp.get("max"),
            "amp_mean": amp.get("mean"),
            "amp_p50": amp.get("p50"),
            "phase_min": phase.get("min"),
            "phase_max": phase.get("max"),
            "phase_mean": phase.get("mean"),
            "phase_p50": phase.get("p50"),
            "l2_norm": probe.get("l2_norm"),
        })

    if not rows:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    """Write human-readable Markdown summary.

    Args:
        summary: Full summary dictionary.
        output_path: Path to write Markdown file.
    """
    with open(output_path, "w") as f:
        f.write("# D0 Parity Log\n\n")

        # Metadata
        meta = summary["metadata"]
        f.write("## Metadata\n\n")
        f.write(f"- **Scenario ID:** {meta['scenario_id']}\n")
        f.write(f"- **Dataset Root:** `{meta['dataset_root']}`\n")
        f.write(f"- **Baseline Params:** `{meta['baseline_params']}`\n")
        f.write(f"- **Timestamp:** {meta['timestamp']}\n")
        f.write(f"- **Git SHA:** `{meta['git_sha']}`\n\n")

        # Key Parameters
        f.write("## Key Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for k, v in summary.get("baseline_params", {}).items():
            f.write(f"| {k} | {v} |\n")

        # Metrics
        if summary.get("metrics"):
            f.write("\n## Metrics (train, test)\n\n")
            f.write("| Metric | Train | Test |\n")
            f.write("|--------|-------|------|\n")
            for k, v in summary["metrics"].items():
                if v is not None and isinstance(v, (list, tuple)) and len(v) >= 2:
                    train_v = f"{v[0]:.6g}" if isinstance(v[0], float) else str(v[0])
                    test_v = f"{v[1]:.6g}" if isinstance(v[1], float) else str(v[1])
                    f.write(f"| {k} | {train_v} | {test_v} |\n")

        # Probe flags
        f.write("\n## Probe Flags\n\n")
        f.write(f"- **probe.trainable:** {summary.get('baseline_params', {}).get('probe.trainable')}\n")
        f.write(f"- **probe.mask:** {summary.get('baseline_params', {}).get('probe.mask')}\n")
        f.write(f"- **intensity_scale.trainable:** {summary.get('baseline_params', {}).get('intensity_scale.trainable')}\n")
        f.write(f"- **intensity_scale_value:** {summary.get('baseline_params', {}).get('intensity_scale_value')}\n\n")

        # Dataset table
        f.write("## Dataset Files\n\n")
        f.write("| File | Photon Dose | Patterns | Diff Shape | SHA256 (truncated) |\n")
        f.write("|------|-------------|----------|------------|--------------------|\n")
        for ds in summary.get("datasets", []):
            diff_info = ds.get("arrays", {}).get("diff3d", {})
            diff_shape = diff_info.get("shape", "N/A")
            n_patterns = diff_shape[0] if isinstance(diff_shape, list) and len(diff_shape) > 0 else "N/A"
            sha_short = ds["sha256"][:16] + "..."
            f.write(f"| {ds['filename']} | {ds['photon_dose']} | {n_patterns} | {diff_shape} | {sha_short} |\n")

        f.write(f"\n**Total datasets:** {summary.get('total_datasets', 0)}\n")

        # Stage-level stats for every dataset
        if summary.get("datasets"):
            f.write("\n## Stage-Level Stats by Dataset\n")

            for ds in summary["datasets"]:
                if not ds.get("stats"):
                    continue

                f.write(f"\n### {ds['filename']} (photon dose: {ds['photon_dose']})\n\n")

                # Raw diffraction table
                if ds["stats"].get("raw"):
                    f.write("#### Raw Diffraction\n\n")
                    f.write("| Stat | Value |\n")
                    f.write("|------|-------|\n")
                    stats = ds["stats"]["raw"]
                    f.write(f"| min | {stats['min']:.6g} |\n")
                    f.write(f"| max | {stats['max']:.6g} |\n")
                    f.write(f"| mean | {stats['mean']:.6g} |\n")
                    f.write(f"| std | {stats['std']:.6g} |\n")
                    f.write(f"| median | {stats['median']:.6g} |\n")
                    f.write(f"| p01 | {stats['p01']:.6g} |\n")
                    f.write(f"| p10 | {stats['p10']:.6g} |\n")
                    f.write(f"| p90 | {stats['p90']:.6g} |\n")
                    f.write(f"| p99 | {stats['p99']:.6g} |\n")
                    f.write(f"| nonzero_fraction | {stats['nonzero_fraction']:.6g} |\n")
                    f.write(f"| count | {stats['count']} |\n\n")

                # Normalized diffraction table
                if ds["stats"].get("normalized"):
                    f.write("#### Normalized Diffraction\n\n")
                    f.write("| Stat | Value |\n")
                    f.write("|------|-------|\n")
                    stats = ds["stats"]["normalized"]
                    f.write(f"| min | {stats['min']:.6g} |\n")
                    f.write(f"| max | {stats['max']:.6g} |\n")
                    f.write(f"| mean | {stats['mean']:.6g} |\n")
                    f.write(f"| std | {stats['std']:.6g} |\n")
                    f.write(f"| median | {stats['median']:.6g} |\n")
                    f.write(f"| p01 | {stats['p01']:.6g} |\n")
                    f.write(f"| p10 | {stats['p10']:.6g} |\n")
                    f.write(f"| p90 | {stats['p90']:.6g} |\n")
                    f.write(f"| p99 | {stats['p99']:.6g} |\n")
                    f.write(f"| nonzero_fraction | {stats['nonzero_fraction']:.6g} |\n")
                    f.write(f"| count | {stats['count']} |\n\n")

                # Grouped intensity table
                if ds["stats"].get("grouped"):
                    f.write("#### Grouped Intensity\n\n")
                    f.write("| Stat | Value |\n")
                    f.write("|------|-------|\n")
                    stats = ds["stats"]["grouped"]
                    f.write(f"| n_unique_scans | {stats['n_unique_scans']} |\n")
                    f.write(f"| n_patterns | {stats['n_patterns']} |\n")
                    f.write(f"| min | {stats['min']:.6g} |\n")
                    f.write(f"| max | {stats['max']:.6g} |\n")
                    f.write(f"| mean | {stats['mean']:.6g} |\n")
                    f.write(f"| std | {stats['std']:.6g} |\n")
                    f.write(f"| median | {stats['median']:.6g} |\n\n")

        f.write("\n---\n")
        f.write("*Full details in `dose_parity_log.json`*\n")


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for D0 parity logger CLI.

    Args:
        args: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="D0 Parity Logger - Capture dose parity evidence"
    )
    parser.add_argument(
        "--dataset-root",
        default="photon_grid_study_20250826_152459",
        help="Root directory containing data_p1e*.npz files",
    )
    parser.add_argument(
        "--baseline-params",
        default=None,
        help="Path to baseline params.dill file",
    )
    parser.add_argument(
        "--scenario-id",
        default="UNKNOWN",
        help="Scenario identifier string",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for JSON, Markdown, and CSV files",
    )
    parser.add_argument(
        "--limit-datasets",
        default=None,
        help="Comma-separated list of dataset filenames to process (default: all)",
    )

    parsed = parser.parse_args(args)

    dataset_root = Path(parsed.dataset_root)
    output_dir = Path(parsed.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect dataset files
    npz_files = sorted(dataset_root.glob("data_p1e*.npz"))

    # Apply limit filter if specified
    if parsed.limit_datasets:
        allowed = set(parsed.limit_datasets.split(","))
        npz_files = [f for f in npz_files if f.name in allowed]

    if not npz_files:
        print(f"ERROR: No data_p1e*.npz files found in {dataset_root}", file=sys.stderr)
        return 1

    # Process datasets
    datasets = []
    for npz_path in npz_files:
        print(f"Processing: {npz_path.name}")
        ds_info = process_dataset(npz_path)
        datasets.append(ds_info)

    # Load baseline params if provided
    baseline_params = {}
    metrics = {}
    if parsed.baseline_params and Path(parsed.baseline_params).exists():
        print(f"Loading params: {parsed.baseline_params}")
        full_params = load_params(parsed.baseline_params)
        baseline_params = extract_baseline_params(full_params)
        metrics = extract_metrics(full_params)

    # Build summary
    summary = {
        "metadata": {
            "scenario_id": parsed.scenario_id,
            "dataset_root": str(dataset_root),
            "baseline_params": parsed.baseline_params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": get_git_sha(),
        },
        "baseline_params": baseline_params,
        "metrics": metrics,
        "datasets": datasets,
        "total_datasets": len(datasets),
    }

    # Write outputs
    json_path = output_dir / "dose_parity_log.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {json_path}")

    md_path = output_dir / "dose_parity_log.md"
    write_markdown(summary, md_path)
    print(f"Wrote: {md_path}")

    csv_path = output_dir / "probe_stats.csv"
    write_probe_csv(datasets, csv_path)
    if csv_path.exists():
        print(f"Wrote: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

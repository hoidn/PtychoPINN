#!/usr/bin/env python
"""
make_ground_truth_manifest.py - Generate manifest for photon_grid ground truth data.

Creates SHA256 checksums + size + metadata for:
- Dataset NPZ files (data_p1e*.npz)
- Baseline outputs (params.dill, baseline_model.h5, recon.dill)
- PINN weights (wts.h5.zip)

Validates NPZ keys against specs/data_contracts.md before emitting manifests.

Usage:
    python make_ground_truth_manifest.py \\
        --dataset-root photon_grid_study_20250826_152459 \\
        --baseline-params path/to/params.dill \\
        --baseline-files path/to/baseline_model.h5 path/to/recon.dill \\
        --pinn-weights path/to/wts.h5.zip \\
        --scenario-id PGRID-20250826-P1E5-T1024 \\
        --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/

See specs/data_contracts.md for required NPZ keys.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Required NPZ keys per specs/data_contracts.md RawData NPZ
REQUIRED_KEYS = frozenset({"diff3d", "probeGuess", "scan_index"})
# Optional keys
OPTIONAL_KEYS = frozenset({"objectGuess", "ground_truth_patches", "xcoords", "ycoords", "xcoords_start", "ycoords_start"})


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_npz_keys(path: Path) -> dict:
    """
    Load NPZ and validate keys against data_contracts.md.

    Returns dict with validation result and key info.
    Raises ValueError if required keys are missing.
    """
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.keys())

    missing = REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"{path.name}: missing required keys {sorted(missing)} (per specs/data_contracts.md)")

    present_optional = keys & OPTIONAL_KEYS
    extra_keys = keys - REQUIRED_KEYS - OPTIONAL_KEYS

    return {
        "required_keys": sorted(REQUIRED_KEYS),
        "present_optional_keys": sorted(present_optional),
        "extra_keys": sorted(extra_keys),
        "all_keys": sorted(keys),
    }


def extract_npz_metadata(path: Path) -> dict:
    """Extract array shapes and dtypes from NPZ file."""
    meta = {}
    with np.load(path, allow_pickle=True) as data:
        for key in sorted(data.keys()):
            arr = data[key]
            meta[key] = {
                "shape": list(arr.shape) if hasattr(arr, "shape") else None,
                "dtype": str(arr.dtype) if hasattr(arr, "dtype") else type(arr).__name__,
            }
    return meta


def gather_file_info(path: Path, file_type: str) -> dict:
    """Gather SHA256, size, and metadata for a single file."""
    if not path.exists():
        raise FileNotFoundError(f"{file_type}: {path}")

    info = {
        "path": str(path),
        "relative_path": str(path),
        "file_type": file_type,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }

    # For NPZ files, validate keys and extract metadata
    if path.suffix == ".npz":
        key_info = validate_npz_keys(path)
        info["keys"] = key_info
        info["array_metadata"] = extract_npz_metadata(path)

    return info


def load_params_dill_metadata(path: Path) -> dict:
    """Extract key metrics from params.dill if possible."""
    try:
        import dill
        with open(path, "rb") as f:
            params = dill.load(f)

        # Extract relevant fields (structure varies, be defensive)
        meta = {"loaded": True}
        if hasattr(params, "cfg"):
            cfg = params.cfg
            for field in ["N", "gridsize", "nepochs", "nconv", "offset_scale"]:
                if hasattr(cfg, field):
                    meta[field] = getattr(cfg, field)
        elif isinstance(params, dict):
            for field in ["N", "gridsize", "nepochs", "nconv", "offset_scale"]:
                if field in params:
                    meta[field] = params[field]
        return meta
    except Exception as e:
        return {"loaded": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest for photon_grid ground truth data bundle"
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Root directory containing data_p1e*.npz files"
    )
    parser.add_argument(
        "--baseline-params", required=True,
        help="Path to baseline params.dill file"
    )
    parser.add_argument(
        "--baseline-files", nargs="+", required=True,
        help="Paths to baseline output files (baseline_model.h5, recon.dill)"
    )
    parser.add_argument(
        "--pinn-weights", required=True,
        help="Path to PINN weights file (wts.h5.zip)"
    )
    parser.add_argument(
        "--scenario-id", required=True,
        help="Scenario identifier (e.g., PGRID-20250826-P1E5-T1024)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for manifest files"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scenario_id": args.scenario_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "spec_reference": "specs/data_contracts.md",
        "required_npz_keys": sorted(REQUIRED_KEYS),
        "datasets": [],
        "baseline_params": None,
        "baseline_files": [],
        "pinn_weights": None,
        "errors": [],
    }

    files_for_csv = []

    # Gather dataset NPZ files
    print(f"Scanning datasets in {dataset_root}...")
    dataset_files = sorted(dataset_root.glob("data_p1e*.npz"))
    if not dataset_files:
        manifest["errors"].append(f"No data_p1e*.npz files found in {dataset_root}")

    for npz_path in dataset_files:
        print(f"  Processing {npz_path.name}...")
        try:
            info = gather_file_info(npz_path, "dataset_npz")
            manifest["datasets"].append(info)
            files_for_csv.append({
                "file_type": "dataset_npz",
                "path": str(npz_path),
                "size_bytes": info["size_bytes"],
                "sha256": info["sha256"],
            })
        except Exception as e:
            manifest["errors"].append(f"dataset {npz_path.name}: {e}")
            print(f"    ERROR: {e}")

    # Baseline params.dill
    print(f"Processing baseline params: {args.baseline_params}...")
    params_path = Path(args.baseline_params)
    try:
        info = gather_file_info(params_path, "baseline_params")
        info["params_metadata"] = load_params_dill_metadata(params_path)
        manifest["baseline_params"] = info
        files_for_csv.append({
            "file_type": "baseline_params",
            "path": str(params_path),
            "size_bytes": info["size_bytes"],
            "sha256": info["sha256"],
        })
    except Exception as e:
        manifest["errors"].append(f"baseline_params: {e}")
        print(f"  ERROR: {e}")

    # Baseline output files
    for bf_path_str in args.baseline_files:
        bf_path = Path(bf_path_str)
        print(f"Processing baseline file: {bf_path}...")
        try:
            info = gather_file_info(bf_path, "baseline_output")
            manifest["baseline_files"].append(info)
            files_for_csv.append({
                "file_type": "baseline_output",
                "path": str(bf_path),
                "size_bytes": info["size_bytes"],
                "sha256": info["sha256"],
            })
        except Exception as e:
            manifest["errors"].append(f"baseline_file {bf_path}: {e}")
            print(f"  ERROR: {e}")

    # PINN weights
    print(f"Processing PINN weights: {args.pinn_weights}...")
    pinn_path = Path(args.pinn_weights)
    try:
        info = gather_file_info(pinn_path, "pinn_weights")
        manifest["pinn_weights"] = info
        files_for_csv.append({
            "file_type": "pinn_weights",
            "path": str(pinn_path),
            "size_bytes": info["size_bytes"],
            "sha256": info["sha256"],
        })
    except Exception as e:
        manifest["errors"].append(f"pinn_weights: {e}")
        print(f"  ERROR: {e}")

    # Emit JSON manifest
    json_path = output_dir / "ground_truth_manifest.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    # Emit CSV file list
    csv_path = output_dir / "files.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_type", "path", "size_bytes", "sha256"])
        writer.writeheader()
        for row in files_for_csv:
            writer.writerow(row)
    print(f"Wrote: {csv_path}")

    # Emit Markdown summary
    md_path = output_dir / "ground_truth_manifest.md"
    with open(md_path, "w") as f:
        f.write(f"# Ground Truth Manifest\n\n")
        f.write(f"**Scenario ID:** {args.scenario_id}\n\n")
        f.write(f"**Generated:** {manifest['generated_utc']}\n\n")
        f.write(f"**Dataset Root:** `{dataset_root}`\n\n")
        f.write(f"**Spec Reference:** `{manifest['spec_reference']}`\n\n")

        f.write("## Datasets\n\n")
        f.write("| File | Size | SHA256 |\n")
        f.write("|------|------|--------|\n")
        for ds in manifest["datasets"]:
            name = Path(ds["path"]).name
            size_mb = ds["size_bytes"] / (1024 * 1024)
            sha_short = ds["sha256"][:16] + "..."
            f.write(f"| `{name}` | {size_mb:.2f} MB | `{sha_short}` |\n")

        f.write("\n### NPZ Key Validation\n\n")
        f.write(f"Required keys (per `specs/data_contracts.md`): `{sorted(REQUIRED_KEYS)}`\n\n")
        if manifest["datasets"]:
            sample = manifest["datasets"][0]
            if "keys" in sample:
                f.write(f"**Sample ({Path(sample['path']).name}):**\n")
                f.write(f"- Present optional keys: `{sample['keys']['present_optional_keys']}`\n")
                f.write(f"- Extra keys: `{sample['keys']['extra_keys']}`\n\n")

        f.write("## Baseline Outputs\n\n")
        if manifest["baseline_params"]:
            bp = manifest["baseline_params"]
            f.write(f"**params.dill:** `{Path(bp['path']).name}`\n")
            f.write(f"- SHA256: `{bp['sha256'][:16]}...`\n")
            if "params_metadata" in bp and bp["params_metadata"].get("loaded"):
                pm = bp["params_metadata"]
                f.write(f"- N: {pm.get('N', 'N/A')}\n")
                f.write(f"- gridsize: {pm.get('gridsize', 'N/A')}\n")
                f.write(f"- nepochs: {pm.get('nepochs', 'N/A')}\n")
            f.write("\n")

        f.write("| File | Type | Size | SHA256 |\n")
        f.write("|------|------|------|--------|\n")
        for bf in manifest["baseline_files"]:
            name = Path(bf["path"]).name
            size_kb = bf["size_bytes"] / 1024
            sha_short = bf["sha256"][:16] + "..."
            f.write(f"| `{name}` | baseline_output | {size_kb:.1f} KB | `{sha_short}` |\n")

        f.write("\n## PINN Weights\n\n")
        if manifest["pinn_weights"]:
            pw = manifest["pinn_weights"]
            name = Path(pw["path"]).name
            size_mb = pw["size_bytes"] / (1024 * 1024)
            f.write(f"| `{name}` | {size_mb:.2f} MB | `{pw['sha256'][:16]}...` |\n")

        if manifest["errors"]:
            f.write("\n## Errors\n\n")
            for err in manifest["errors"]:
                f.write(f"- {err}\n")

    print(f"Wrote: {md_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Manifest complete: {len(manifest['datasets'])} datasets, "
          f"{len(manifest['baseline_files'])} baseline files")
    if manifest["errors"]:
        print(f"ERRORS: {len(manifest['errors'])}")
        for err in manifest["errors"]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("All files validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()

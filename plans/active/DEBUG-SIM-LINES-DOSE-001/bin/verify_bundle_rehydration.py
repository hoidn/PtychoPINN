#!/usr/bin/env python
"""
verify_bundle_rehydration.py - Verify tarball extraction preserves manifest integrity.

Extracts dose_experiments_ground_truth.tar.gz into a temp directory,
regenerates the manifest, and compares SHA256 + size metadata against
the original manifest. Emits rehydration_diff.json and rehydration_summary.md.

Usage:
    python verify_bundle_rehydration.py \
        --tarball path/to/dose_experiments_ground_truth.tar.gz \
        --manifest path/to/ground_truth_manifest.json \
        --output path/to/output_dir \
        [--keep-extracted]

See docs/fix_plan.md:304 for the rehydration verification checklist.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_tarball(tarball_path: Path, dest_dir: Path) -> Path:
    """Extract tarball to destination directory, return extraction root."""
    print(f"Extracting {tarball_path.name} to {dest_dir}...")
    with tarfile.open(tarball_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")

    # Find the root directory inside the extraction
    subdirs = list(dest_dir.iterdir())
    if len(subdirs) == 1 and subdirs[0].is_dir():
        return subdirs[0]
    return dest_dir


def regenerate_manifest(extracted_root: Path, output_dir: Path, scenario_id: str) -> Path:
    """Run make_ground_truth_manifest.py on extracted bundle, return manifest path."""
    manifest_script = Path(__file__).parent / "make_ground_truth_manifest.py"

    # Map extracted structure to manifest script args
    simulation_dir = extracted_root / "simulation"
    training_dir = extracted_root / "training"
    inference_dir = extracted_root / "inference"

    # Find the files
    params_dill = training_dir / "params.dill"
    baseline_files = [
        training_dir / "baseline_model.h5",
        training_dir / "recon.dill",
    ]
    pinn_weights = inference_dir / "wts.h5.zip"

    # Validate files exist
    missing = []
    if not simulation_dir.exists():
        missing.append(str(simulation_dir))
    if not params_dill.exists():
        missing.append(str(params_dill))
    for bf in baseline_files:
        if not bf.exists():
            missing.append(str(bf))
    if not pinn_weights.exists():
        missing.append(str(pinn_weights))

    if missing:
        raise FileNotFoundError(f"Missing files in extracted bundle: {missing}")

    # Run the manifest script
    manifest_output = output_dir / "manifest"
    manifest_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(manifest_script),
        "--dataset-root", str(simulation_dir),
        "--baseline-params", str(params_dill),
        "--baseline-files", str(baseline_files[0]), str(baseline_files[1]),
        "--pinn-weights", str(pinn_weights),
        "--scenario-id", scenario_id,
        "--output", str(manifest_output),
    ]

    print(f"Regenerating manifest via: {' '.join(cmd[:3])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Manifest regeneration failed: {result.returncode}")

    print(result.stdout)
    return manifest_output / "ground_truth_manifest.json"


def load_manifest(path: Path) -> dict:
    """Load manifest JSON."""
    with open(path) as f:
        return json.load(f)


def extract_file_checksums(manifest: dict) -> dict:
    """Extract filename -> (sha256, size_bytes, file_type) from manifest."""
    checksums = {}

    # Datasets
    for ds in manifest.get("datasets", []):
        name = Path(ds["path"]).name
        checksums[name] = {
            "sha256": ds["sha256"],
            "size_bytes": ds["size_bytes"],
            "file_type": ds["file_type"],
        }

    # Baseline params
    if manifest.get("baseline_params"):
        bp = manifest["baseline_params"]
        name = Path(bp["path"]).name
        checksums[name] = {
            "sha256": bp["sha256"],
            "size_bytes": bp["size_bytes"],
            "file_type": bp["file_type"],
        }

    # Baseline files
    for bf in manifest.get("baseline_files", []):
        name = Path(bf["path"]).name
        checksums[name] = {
            "sha256": bf["sha256"],
            "size_bytes": bf["size_bytes"],
            "file_type": bf["file_type"],
        }

    # PINN weights
    if manifest.get("pinn_weights"):
        pw = manifest["pinn_weights"]
        name = Path(pw["path"]).name
        checksums[name] = {
            "sha256": pw["sha256"],
            "size_bytes": pw["size_bytes"],
            "file_type": pw["file_type"],
        }

    return checksums


def compare_manifests(original: dict, rehydrated: dict) -> dict:
    """Compare two manifest file checksums, return diff report."""
    orig_checksums = extract_file_checksums(original)
    rehyd_checksums = extract_file_checksums(rehydrated)

    all_files = set(orig_checksums.keys()) | set(rehyd_checksums.keys())

    matches = []
    mismatches = []
    missing_in_rehydrated = []
    extra_in_rehydrated = []

    for filename in sorted(all_files):
        orig = orig_checksums.get(filename)
        rehyd = rehyd_checksums.get(filename)

        if orig and not rehyd:
            missing_in_rehydrated.append({
                "filename": filename,
                "original": orig,
            })
        elif rehyd and not orig:
            extra_in_rehydrated.append({
                "filename": filename,
                "rehydrated": rehyd,
            })
        else:
            # Both exist, compare
            sha_match = orig["sha256"] == rehyd["sha256"]
            size_match = orig["size_bytes"] == rehyd["size_bytes"]

            if sha_match and size_match:
                matches.append({
                    "filename": filename,
                    "sha256": orig["sha256"],
                    "size_bytes": orig["size_bytes"],
                    "file_type": orig["file_type"],
                })
            else:
                mismatches.append({
                    "filename": filename,
                    "original": orig,
                    "rehydrated": rehyd,
                    "sha256_match": sha_match,
                    "size_match": size_match,
                })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario_id": original.get("scenario_id"),
        "total_files": len(all_files),
        "matches": len(matches),
        "mismatches": len(mismatches),
        "missing_in_rehydrated": len(missing_in_rehydrated),
        "extra_in_rehydrated": len(extra_in_rehydrated),
        "status": "PASS" if not mismatches and not missing_in_rehydrated else "FAIL",
        "details": {
            "matches": matches,
            "mismatches": mismatches,
            "missing_in_rehydrated": missing_in_rehydrated,
            "extra_in_rehydrated": extra_in_rehydrated,
        },
    }


def write_summary(diff: dict, output_dir: Path) -> Path:
    """Write human-readable summary markdown."""
    summary_path = output_dir / "rehydration_summary.md"

    with open(summary_path, "w") as f:
        f.write("# Tarball Rehydration Verification\n\n")
        f.write(f"**Timestamp:** {diff['timestamp']}\n\n")
        f.write(f"**Scenario ID:** {diff['scenario_id']}\n\n")
        f.write(f"**Status:** `{diff['status']}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total files | {diff['total_files']} |\n")
        f.write(f"| Matches | {diff['matches']} |\n")
        f.write(f"| Mismatches | {diff['mismatches']} |\n")
        f.write(f"| Missing in rehydrated | {diff['missing_in_rehydrated']} |\n")
        f.write(f"| Extra in rehydrated | {diff['extra_in_rehydrated']} |\n")

        if diff["status"] == "PASS":
            f.write("\n## Verification Result\n\n")
            f.write("All files in the rehydrated bundle match the original manifest.\n")
            f.write("SHA256 checksums and file sizes are identical.\n\n")

            f.write("### Verified Files\n\n")
            f.write("| File | Type | Size | SHA256 (first 16) |\n")
            f.write("|------|------|------|-------------------|\n")
            for m in diff["details"]["matches"]:
                size_mb = m["size_bytes"] / (1024 * 1024)
                sha_short = m["sha256"][:16] + "..."
                f.write(f"| `{m['filename']}` | {m['file_type']} | {size_mb:.2f} MB | `{sha_short}` |\n")
        else:
            f.write("\n## Mismatches\n\n")
            for mm in diff["details"]["mismatches"]:
                f.write(f"### `{mm['filename']}`\n\n")
                f.write(f"- SHA256 match: {mm['sha256_match']}\n")
                f.write(f"- Size match: {mm['size_match']}\n")
                f.write(f"- Original: `{mm['original']['sha256'][:16]}...` ({mm['original']['size_bytes']} bytes)\n")
                f.write(f"- Rehydrated: `{mm['rehydrated']['sha256'][:16]}...` ({mm['rehydrated']['size_bytes']} bytes)\n\n")

            if diff["details"]["missing_in_rehydrated"]:
                f.write("\n## Missing in Rehydrated\n\n")
                for m in diff["details"]["missing_in_rehydrated"]:
                    f.write(f"- `{m['filename']}`\n")

            if diff["details"]["extra_in_rehydrated"]:
                f.write("\n## Extra in Rehydrated\n\n")
                for e in diff["details"]["extra_in_rehydrated"]:
                    f.write(f"- `{e['filename']}`\n")

        f.write("\n---\n\n")
        f.write("*Generated by `verify_bundle_rehydration.py`*\n")

    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Verify tarball extraction preserves manifest integrity"
    )
    parser.add_argument(
        "--tarball", required=True,
        help="Path to dose_experiments_ground_truth.tar.gz"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to original ground_truth_manifest.json"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for verification artifacts"
    )
    parser.add_argument(
        "--keep-extracted", action="store_true",
        help="Keep extracted files after verification (default: delete)"
    )

    args = parser.parse_args()

    tarball_path = Path(args.tarball)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)

    # Validate inputs
    if not tarball_path.exists():
        print(f"ERROR: Tarball not found: {tarball_path}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original manifest
    print(f"Loading original manifest: {manifest_path}")
    original_manifest = load_manifest(manifest_path)
    scenario_id = original_manifest.get("scenario_id", "UNKNOWN")

    # Create temp directory for extraction
    temp_dir = tempfile.mkdtemp(prefix="rehydration_")
    temp_path = Path(temp_dir)

    try:
        # Extract tarball
        extracted_root = extract_tarball(tarball_path, temp_path)
        print(f"Extracted to: {extracted_root}")

        # Regenerate manifest
        rehydrated_manifest_path = regenerate_manifest(
            extracted_root, output_dir, scenario_id
        )
        print(f"Rehydrated manifest: {rehydrated_manifest_path}")

        # Load rehydrated manifest
        rehydrated_manifest = load_manifest(rehydrated_manifest_path)

        # Compare manifests
        print("\nComparing manifests...")
        diff = compare_manifests(original_manifest, rehydrated_manifest)

        # Write diff JSON
        diff_path = output_dir / "rehydration_diff.json"
        with open(diff_path, "w") as f:
            json.dump(diff, f, indent=2, default=str)
        print(f"Wrote: {diff_path}")

        # Write summary
        summary_path = write_summary(diff, output_dir)
        print(f"Wrote: {summary_path}")

        # Print result
        print(f"\n{'='*60}")
        print(f"Rehydration verification: {diff['status']}")
        print(f"  Total files: {diff['total_files']}")
        print(f"  Matches: {diff['matches']}")
        print(f"  Mismatches: {diff['mismatches']}")

        if diff["status"] == "FAIL":
            print("\nERROR: Rehydration verification failed!")
            for mm in diff["details"]["mismatches"]:
                print(f"  - {mm['filename']}: SHA256={mm['sha256_match']}, Size={mm['size_match']}")
            sys.exit(1)
        else:
            print("\nSUCCESS: All files match original manifest.")
            sys.exit(0)

    finally:
        # Cleanup
        if args.keep_extracted:
            print(f"\nKeeping extracted files at: {temp_path}")
        else:
            print(f"\nCleaning up: {temp_path}")
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    main()

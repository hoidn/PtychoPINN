#!/usr/bin/env python
"""Package dose_experiments ground-truth bundle for maintainer delivery.

This CLI loads the Phase-A manifest JSON and README, copies all datasets/baseline/pinn
artifacts into a structured drop directory, verifies SHA256 checksums, and produces
a tarball with verification logs.

Spec references:
- specs/data_contracts.md: RawData NPZ key requirements
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md: Phase C checklist

Usage:
    python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py \
        --manifest-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json \
        --manifest-md plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.md \
        --baseline-summary plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json \
        --readme plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/README.md \
        --drop-root plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth \
        --reports-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z \
        --tarball plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA256 hash of a file using chunked reading to avoid memory issues."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file, raising error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} bytes"


def copy_and_verify(
    src_path: Path,
    dst_path: Path,
    expected_sha: str,
    file_desc: str,
) -> dict[str, Any]:
    """Copy file and verify SHA256 matches expected value.

    Returns dict with verification result.
    """
    result = {
        "file": str(dst_path.name),
        "source": str(src_path),
        "destination": str(dst_path),
        "expected_sha256": expected_sha,
        "verified": False,
        "error": None,
    }

    # Check source exists
    if not src_path.exists():
        result["error"] = f"Source file not found: {src_path}"
        return result

    # Verify source SHA before copying
    src_sha = sha256_file(src_path)
    result["source_sha256"] = src_sha

    if src_sha != expected_sha:
        result["error"] = f"Source SHA mismatch: expected {expected_sha[:12]}..., got {src_sha[:12]}..."
        return result

    # Create destination directory if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy file preserving metadata
    shutil.copy2(src_path, dst_path)

    # Verify destination SHA
    dst_sha = sha256_file(dst_path)
    result["destination_sha256"] = dst_sha

    if dst_sha != expected_sha:
        result["error"] = f"Destination SHA mismatch after copy: expected {expected_sha[:12]}..., got {dst_sha[:12]}..."
        return result

    result["verified"] = True
    result["size_bytes"] = dst_path.stat().st_size
    print(f"  OK: {file_desc} ({format_bytes(result['size_bytes'])})")
    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Package dose_experiments ground-truth bundle for maintainer delivery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        required=True,
        help="Path to ground_truth_manifest.json from Phase A",
    )
    parser.add_argument(
        "--manifest-md",
        type=Path,
        required=True,
        help="Path to ground_truth_manifest.md from Phase A",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        required=True,
        help="Path to dose_baseline_summary.json",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        required=True,
        help="Path to README.md to include in bundle",
    )
    parser.add_argument(
        "--drop-root",
        type=Path,
        required=True,
        help="Root directory for the bundle (e.g., dose_experiments_ground_truth/)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory to write verification logs",
    )
    parser.add_argument(
        "--tarball",
        type=Path,
        required=True,
        help="Path for output tarball (e.g., dose_experiments_ground_truth.tar.gz)",
    )

    args = parser.parse_args()

    # Validate inputs exist
    for input_file, desc in [
        (args.manifest_json, "Manifest JSON"),
        (args.manifest_md, "Manifest MD"),
        (args.baseline_summary, "Baseline summary"),
        (args.readme, "README"),
    ]:
        if not input_file.exists():
            print(f"ERROR: {desc} not found: {input_file}", file=sys.stderr)
            return 1

    # Load manifest
    print(f"Loading manifest: {args.manifest_json}")
    manifest = load_json(args.manifest_json)

    # Remove stale drop if exists (per input.md pitfalls)
    if args.drop_root.exists():
        print(f"Removing stale drop directory: {args.drop_root}")
        shutil.rmtree(args.drop_root)

    # Create directory structure
    dirs = {
        "simulation": args.drop_root / "simulation",
        "training": args.drop_root / "training",
        "inference": args.drop_root / "inference",
        "docs": args.drop_root / "docs",
    }
    for name, dir_path in dirs.items():
        print(f"Creating {name}/: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure reports dir exists
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # Track verification results
    verification = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_id": manifest.get("scenario_id", "N/A"),
        "drop_root": str(args.drop_root),
        "files": [],
        "summary": {
            "total_files": 0,
            "verified_files": 0,
            "failed_files": 0,
            "total_size_bytes": 0,
        },
    }

    # === Copy datasets to simulation/ ===
    print("\n=== Copying datasets to simulation/ ===")
    datasets = manifest.get("datasets", [])
    for ds in datasets:
        src_path = Path(ds["path"])
        dst_path = dirs["simulation"] / src_path.name
        expected_sha = ds["sha256"]

        result = copy_and_verify(src_path, dst_path, expected_sha, src_path.name)
        verification["files"].append(result)

    # === Copy baseline params + outputs to training/ ===
    print("\n=== Copying baseline artifacts to training/ ===")

    # params.dill
    baseline_params = manifest.get("baseline_params", {})
    if baseline_params:
        src_path = Path(baseline_params["path"])
        dst_path = dirs["training"] / src_path.name
        result = copy_and_verify(
            src_path, dst_path, baseline_params["sha256"], src_path.name
        )
        verification["files"].append(result)

    # baseline_model.h5, recon.dill
    baseline_files = manifest.get("baseline_files", [])
    for bf in baseline_files:
        src_path = Path(bf["path"])
        dst_path = dirs["training"] / src_path.name
        result = copy_and_verify(src_path, dst_path, bf["sha256"], src_path.name)
        verification["files"].append(result)

    # === Copy PINN weights to inference/ ===
    print("\n=== Copying PINN weights to inference/ ===")
    pinn_weights = manifest.get("pinn_weights", {})
    if pinn_weights:
        src_path = Path(pinn_weights["path"])
        dst_path = dirs["inference"] / src_path.name
        result = copy_and_verify(
            src_path, dst_path, pinn_weights["sha256"], src_path.name
        )
        verification["files"].append(result)

    # === Copy docs to docs/ ===
    print("\n=== Copying documentation to docs/ ===")

    # Copy manifest JSON/MD (no SHA verification, these are generated docs)
    for doc_src in [args.manifest_json, args.manifest_md, args.baseline_summary]:
        doc_dst = dirs["docs"] / doc_src.name
        shutil.copy2(doc_src, doc_dst)
        print(f"  OK: {doc_src.name} (doc)")
        verification["files"].append({
            "file": doc_src.name,
            "source": str(doc_src),
            "destination": str(doc_dst),
            "verified": True,  # No SHA check for generated docs
            "type": "documentation",
        })

    # README will be re-generated and copied separately
    readme_dst = dirs["docs"] / "README.md"
    shutil.copy2(args.readme, readme_dst)
    print(f"  OK: README.md (doc)")
    verification["files"].append({
        "file": "README.md",
        "source": str(args.readme),
        "destination": str(readme_dst),
        "verified": True,
        "type": "documentation",
    })

    # === Compute summary ===
    for f in verification["files"]:
        verification["summary"]["total_files"] += 1
        if f.get("verified", False):
            verification["summary"]["verified_files"] += 1
            if "size_bytes" in f:
                verification["summary"]["total_size_bytes"] += f["size_bytes"]
        else:
            verification["summary"]["failed_files"] += 1

    # === Check for failures ===
    failed = [f for f in verification["files"] if not f.get("verified", False)]
    if failed:
        print(f"\nERROR: {len(failed)} file(s) failed verification:")
        for f in failed:
            print(f"  - {f['file']}: {f.get('error', 'unknown error')}")
        # Still write verification log for debugging
        verification_json_path = args.reports_dir / "bundle_verification.json"
        with open(verification_json_path, "w", encoding="utf-8") as vf:
            json.dump(verification, vf, indent=2)
        return 1

    # === Create tarball ===
    print(f"\n=== Creating tarball: {args.tarball} ===")
    args.tarball.parent.mkdir(parents=True, exist_ok=True)

    # Use tarfile to create .tar.gz with the bundle directory name preserved
    with tarfile.open(args.tarball, "w:gz") as tar:
        # Add the entire drop_root directory with its name
        tar.add(args.drop_root, arcname=args.drop_root.name)

    tarball_size = args.tarball.stat().st_size
    tarball_sha = sha256_file(args.tarball)

    verification["tarball"] = {
        "path": str(args.tarball),
        "size_bytes": tarball_size,
        "sha256": tarball_sha,
    }

    print(f"  Tarball size: {format_bytes(tarball_size)}")
    print(f"  Tarball SHA256: {tarball_sha}")

    # Write .sha256 file next to tarball
    sha_file = args.tarball.with_suffix(args.tarball.suffix + ".sha256")
    sha_file.write_text(f"{tarball_sha}  {args.tarball.name}\n")
    print(f"  SHA256 file: {sha_file}")

    # === Write verification logs ===
    print(f"\n=== Writing verification logs ===")

    # JSON
    verification_json_path = args.reports_dir / "bundle_verification.json"
    with open(verification_json_path, "w", encoding="utf-8") as vf:
        json.dump(verification, vf, indent=2)
    print(f"  JSON: {verification_json_path}")

    # Markdown
    verification_md_path = args.reports_dir / "bundle_verification.md"
    md_lines = [
        "# Bundle Verification Report",
        "",
        f"**Generated:** {verification['generated_utc']}",
        f"**Scenario ID:** {verification['scenario_id']}",
        f"**Drop Root:** `{verification['drop_root']}`",
        "",
        "## Summary",
        "",
        f"- Total files: {verification['summary']['total_files']}",
        f"- Verified: {verification['summary']['verified_files']}",
        f"- Failed: {verification['summary']['failed_files']}",
        f"- Total size: {format_bytes(verification['summary']['total_size_bytes'])}",
        "",
        "## Tarball",
        "",
        f"- Path: `{verification['tarball']['path']}`",
        f"- Size: {format_bytes(verification['tarball']['size_bytes'])}",
        f"- SHA256: `{verification['tarball']['sha256']}`",
        "",
        "## File Verification Details",
        "",
        "| File | Source | Verified | Size |",
        "|------|--------|----------|------|",
    ]

    for f in verification["files"]:
        status = "YES" if f.get("verified") else "FAIL"
        size = format_bytes(f.get("size_bytes", 0)) if "size_bytes" in f else "-"
        md_lines.append(f"| `{f['file']}` | `{Path(f.get('source', '')).name}` | {status} | {size} |")

    md_lines.append("")
    verification_md_path.write_text("\n".join(md_lines))
    print(f"  Markdown: {verification_md_path}")

    # === Final summary ===
    print("\n" + "=" * 60)
    print("BUNDLE PACKAGING COMPLETE")
    print("=" * 60)
    print(f"Drop root: {args.drop_root}")
    print(f"Total files: {verification['summary']['total_files']}")
    print(f"Verified: {verification['summary']['verified_files']}")
    print(f"Total size: {format_bytes(verification['summary']['total_size_bytes'])}")
    print(f"Tarball: {args.tarball} ({format_bytes(tarball_size)})")
    print(f"Tarball SHA256: {tarball_sha}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Validate that a candidate training_manifest.json reports a finite final_loss
that is within an acceptable tolerance of a reference (visually verified) run.

Inputs:
    --reference  Path to the blessed manifest (copied from a “good” run)
    --candidate  Path to the current manifest being validated
    --dose       Dose value to locate inside both manifests
    --view       View string (baseline, dense, sparse)
    --gridsize   Gridsize integer (1 or 2)
    --tolerance  Fractional tolerance (default: 0.25 → <= 1.25x reference)

The tool exits with status 0 when the candidate run’s loss is finite and
within tolerance, and non-zero otherwise. On failure it prints a descriptive
message so supervisors can mark the Attempt as blocked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training losses against a reference manifest.")
    parser.add_argument("--reference", required=True, type=Path, help="Path to the blessed training_manifest.json")
    parser.add_argument("--candidate", required=True, type=Path, help="Path to the new training_manifest.json")
    parser.add_argument("--dose", required=True, type=float, help="Dose value to locate inside the manifest")
    parser.add_argument("--view", required=True, help="View name (baseline, dense, sparse)")
    parser.add_argument("--gridsize", required=True, type=int, help="Gridsize value (1 or 2)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Allowed fractional increase over the reference loss (default: 0.25 = 25%%)",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def find_job(manifest: Dict[str, Any], dose: float, view: str, gridsize: int) -> Dict[str, Any]:
    for job in manifest.get("jobs", []):
        if (
            float(job.get("dose")) == dose
            and str(job.get("view")) == view
            and int(job.get("gridsize")) == gridsize
        ):
            return job
    raise ValueError(f"No job found for dose={dose}, view={view}, gridsize={gridsize} in manifest.")


def extract_loss(job: Dict[str, Any], manifest_path: Path) -> float:
    result = job.get("result") or {}
    status = result.get("status")
    if status != "success":
        raise ValueError(f"Job status {status!r} in {manifest_path} is not 'success'.")
    loss = result.get("final_loss")
    if loss is None:
        raise ValueError(f"final_loss missing in {manifest_path}.")
    if not isinstance(loss, (int, float)):
        raise ValueError(f"final_loss is not numeric in {manifest_path}: {loss!r}")
    if not math.isfinite(loss):
        raise ValueError(f"final_loss is not finite in {manifest_path}: {loss}")
    return float(loss)


def main() -> int:
    args = parse_args()
    reference_manifest = load_manifest(args.reference)
    candidate_manifest = load_manifest(args.candidate)

    ref_job = find_job(reference_manifest, args.dose, args.view, args.gridsize)
    cand_job = find_job(candidate_manifest, args.dose, args.view, args.gridsize)

    ref_loss = extract_loss(ref_job, args.reference)
    cand_loss = extract_loss(cand_job, args.candidate)

    limit = ref_loss * (1.0 + args.tolerance)
    print(
        f"[training-loss-check] reference={ref_loss:.6f} candidate={cand_loss:.6f} "
        f"tolerance={args.tolerance:.3f} limit={limit:.6f}"
    )

    if cand_loss <= limit:
        print("[training-loss-check] OK: candidate loss is within tolerance.")
        return 0

    print(
        "[training-loss-check] FAILURE: candidate loss exceeds tolerance "
        f"(candidate={cand_loss:.6f} > limit={limit:.6f})."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

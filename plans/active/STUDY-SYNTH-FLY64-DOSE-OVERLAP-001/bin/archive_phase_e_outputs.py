#!/usr/bin/env python3
"""
Archive Phase E training outputs for dose overlap study (initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001, owner: galph).

Inputs:
  --phase-e-root  Path to tmp Phase E artifact root (e.g. tmp/phase_e_training_gs2)
  --hub           Destination hub (plans/active/.../reports/<timestamp>/phase_e_training_bundle_real_runs_exec)
  --dose          Dose value used for CLI run (e.g. 1000)
  --views         One or more view identifiers to archive (baseline, dense, sparse)

Outputs:
  - Copies training_manifest.json and skip_summary.json into <hub>/data/ with dose-prefixed names.
  - Copies bundle archives for each view into <hub>/data/ as wts_<view>_gs<gridsize>.h5.zip.
  - Writes pretty-printed manifest to <hub>/analysis/training_manifest_pretty.json.
  - Emits SHA256 checksums for copied bundles to <hub>/analysis/bundle_checksums.txt.

Repro:
  python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py \
    --phase-e-root tmp/phase_e_training_gs2 \
    --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/phase_e_training_bundle_real_runs_exec \
    --dose 1000 --views dense baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


def compute_sha256(path: Path) -> str:
    """Return SHA256 hex digest for the given file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_manifest_files(phase_e_root: Path, hub: Path, dose: int) -> dict:
    """Copy manifest + skip summary into hub/data and return manifest JSON."""
    data_dir = hub / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_src = phase_e_root / "training_manifest.json"
    skip_src = phase_e_root / "skip_summary.json"
    if not manifest_src.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_src}")
    if not skip_src.exists():
        raise FileNotFoundError(f"Skip summary not found: {skip_src}")

    manifest_dest = data_dir / f"training_manifest_dose{dose}.json"
    skip_dest = data_dir / f"skip_summary_dose{dose}.json"
    shutil.copy2(manifest_src, manifest_dest)
    shutil.copy2(skip_src, skip_dest)

    with manifest_dest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    analysis_dir = hub / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pretty_manifest = analysis_dir / "training_manifest_pretty.json"
    with pretty_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return manifest


def archive_bundles(
    phase_e_root: Path,
    hub: Path,
    dose: int,
    views: Iterable[str],
    manifest: dict,
) -> None:
    """Copy bundle archives into hub/data and record checksums."""
    data_dir = hub / "data"
    analysis_dir = hub / "analysis"
    checksum_path = analysis_dir / "bundle_checksums.txt"

    entries = []
    for view in views:
        if view == "baseline":
            rel_dir = Path(f"dose_{dose}") / "baseline" / "gs1"
            gridsize = "1"
        else:
            rel_dir = Path(f"dose_{dose}") / view / "gs2"
            gridsize = "2"

        bundle_src = phase_e_root / rel_dir / "wts.h5.zip"
        if not bundle_src.exists():
            raise FileNotFoundError(f"Expected bundle missing: {bundle_src}")

        bundle_dest = data_dir / f"wts_{view}_gs{gridsize}.h5.zip"
        shutil.copy2(bundle_src, bundle_dest)

        sha256 = compute_sha256(bundle_dest)
        entries.append((bundle_dest.name, sha256))

        # Validate manifest contains matching checksum for this job
        matched_jobs = [
            job for job in manifest.get("jobs", [])
            if job.get("view") == view and job.get("dose") == dose
        ]
        if not matched_jobs:
            raise ValueError(f"Manifest missing job entry for view={view}, dose={dose}")

        for job in matched_jobs:
            result = job.get("result", {})
            manifest_sha = result.get("bundle_sha256")
            if manifest_sha is None:
                raise ValueError(f"Manifest missing bundle_sha256 for view={view}")
            if manifest_sha != sha256:
                raise ValueError(
                    f"SHA mismatch for view={view}: manifest={manifest_sha}, computed={sha256}"
                )

    with checksum_path.open("w", encoding="utf-8") as handle:
        for name, sha256 in entries:
            handle.write(f"{sha256}  {name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive Phase E training outputs.")
    parser.add_argument("--phase-e-root", required=True, type=Path)
    parser.add_argument("--hub", required=True, type=Path)
    parser.add_argument("--dose", required=True, type=int)
    parser.add_argument(
        "--views",
        required=True,
        nargs="+",
        help="Views to archive (e.g. baseline dense sparse)",
    )
    args = parser.parse_args()

    manifest = copy_manifest_files(args.phase_e_root, args.hub, args.dose)
    archive_bundles(args.phase_e_root, args.hub, args.dose, args.views, manifest)


if __name__ == "__main__":
    main()


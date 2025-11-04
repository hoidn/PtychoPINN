"""
Phase F: PtyChi LSQML reconstruction job builder.

This module constructs a manifest of reconstruction jobs for pty-chi LSQML
baseline comparisons against PtychoPINN trained models. Each job targets
Phase C baseline datasets (gs1) or Phase D overlap views (gs2 dense/sparse).

Job manifest includes CLI arguments for scripts/reconstruction/ptychi_reconstruct_tike.py
with algorithm='LSQML', num_epochs=100 (baseline; parameterizable), and
artifact paths derived from dose/view/split combinations.

CONFIG-001 compliance: This builder remains pure (no params.cfg mutation).
The CONFIG-001 bridge is deferred to the actual LSQML runner invocation in Phase F2.

DATA-001 compliance: Builder validates NPZ paths against Phase C/D outputs
and assumes canonical contract (amplitude diffraction, complex64 Y patches).

POLICY-001 note: Pty-chi uses PyTorch internally, which is acceptable per
study design. No PtychoPINN backend switch is required.

OVERSAMPLING-001 note: Reconstruction jobs inherit neighbor_count=7 from
Phase D/E artifacts. No additional K≥C validation is needed in the builder.
"""

from pathlib import Path
from typing import List
from dataclasses import dataclass, field
import subprocess
import sys
from enum import Enum

from studies.fly64_dose_overlap.design import get_study_design


class ViewType(str, Enum):
    """Overlap view types for reconstruction jobs."""
    BASELINE = "baseline"
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass
class ReconstructionJob:
    """
    Single pty-chi LSQML reconstruction job specification.

    Attributes:
        dose: Photon dose (photons per exposure)
        view: View type ('baseline', 'dense', or 'sparse')
        split: Data split ('train' or 'test')
        input_npz: Path to input NPZ (Phase C baseline or Phase D overlap)
        output_dir: Artifact directory for LSQML outputs
        algorithm: Reconstruction algorithm ('LSQML')
        num_epochs: Number of LSQML iterations
        cli_args: Assembled arguments for scripts/reconstruction/ptychi_reconstruct_tike.py
    """
    dose: float
    view: str
    split: str
    input_npz: Path
    output_dir: Path
    algorithm: str = "LSQML"
    num_epochs: int = 100
    cli_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and assemble CLI arguments."""
        # Ensure paths are Path objects
        self.input_npz = Path(self.input_npz)
        self.output_dir = Path(self.output_dir)

        # Assemble CLI arguments if not provided
        if not self.cli_args:
            self.cli_args = [
                "python",
                "scripts/reconstruction/ptychi_reconstruct_tike.py",
                "--algorithm", self.algorithm,
                "--num-epochs", str(self.num_epochs),
                "--input-npz", str(self.input_npz),
                "--output-dir", str(self.output_dir),
            ]


def build_ptychi_jobs(
    phase_c_root: Path,
    phase_d_root: Path,
    artifact_root: Path,
    allow_missing: bool = False,
) -> List[ReconstructionJob]:
    """
    Enumerate pty-chi LSQML reconstruction jobs for study doses and views.

    Expected manifest structure:
    - 3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose
    - Total: 21 jobs for the full study (3 doses × 7 jobs)

    Each job contains:
    - dose (float): photon dose from StudyDesign.dose_list
    - view (str): 'baseline', 'dense', or 'sparse'
    - split (str): 'train' or 'test'
    - input_npz (Path): Phase C baseline or Phase D overlap NPZ
    - output_dir (Path): artifact directory for LSQML outputs
    - algorithm (str): 'LSQML'
    - num_epochs (int): 100 (baseline; parameterizable in future)
    - cli_args (list): arguments for scripts/reconstruction/ptychi_reconstruct_tike.py

    Args:
        phase_c_root: Root directory for Phase C baseline datasets
                      (dose_{dose}/patched_{split}.npz)
        phase_d_root: Root directory for Phase D overlap views
                      (dose_{dose}/{view}/{view}_{split}.npz)
        artifact_root: Root directory for reconstruction outputs
                       (phase_f_reconstruction/dose_{dose}/{view}/{split}/)
        allow_missing: If False, raise FileNotFoundError for missing NPZ files

    Returns:
        List of ReconstructionJob dataclasses

    Raises:
        FileNotFoundError: If allow_missing=False and input NPZ does not exist
    """
    design = get_study_design()
    jobs = []

    # Deterministic ordering: doses ascending, views (baseline→dense→sparse), splits (train→test)
    for dose in sorted(design.dose_list):
        dose_int = int(dose)

        # Process each split
        for split in ["train", "test"]:
            # Baseline jobs (Phase C patched datasets, gs=1)
            phase_c_dose_dir = phase_c_root / f"dose_{dose_int}"
            baseline_npz = phase_c_dose_dir / f"patched_{split}.npz"

            if not allow_missing and not baseline_npz.exists():
                raise FileNotFoundError(
                    f"Phase C baseline NPZ not found: {baseline_npz}. "
                    f"Expected from Phase C patching."
                )

            baseline_output = artifact_root / f"dose_{dose_int}" / "baseline" / split
            jobs.append(ReconstructionJob(
                dose=dose,
                view=ViewType.BASELINE.value,
                split=split,
                input_npz=baseline_npz,
                output_dir=baseline_output,
            ))

            # Overlap view jobs (Phase D datasets, gs=2)
            phase_d_dose_dir = phase_d_root / f"dose_{dose_int}"
            for view in ["dense", "sparse"]:
                view_dir = phase_d_dose_dir / view
                overlap_npz = view_dir / f"{view}_{split}.npz"

                if not allow_missing and not overlap_npz.exists():
                    raise FileNotFoundError(
                        f"Phase D overlap NPZ not found: {overlap_npz}. "
                        f"Expected from Phase D overlap filtering."
                    )

                overlap_output = artifact_root / f"dose_{dose_int}" / view / split
                jobs.append(ReconstructionJob(
                    dose=dose,
                    view=view,
                    split=split,
                    input_npz=overlap_npz,
                    output_dir=overlap_output,
                ))

    return jobs


def run_ptychi_job(job: ReconstructionJob, dry_run: bool = True) -> subprocess.CompletedProcess:
    """
    Execute a single pty-chi LSQML reconstruction job.

    Args:
        job: ReconstructionJob to execute
        dry_run: If True, skip actual execution and return mock result

    Returns:
        subprocess.CompletedProcess with execution results

    Note:
        CONFIG-001 bridge (update_legacy_dict) is handled by the reconstruction
        script itself. This runner simply dispatches the subprocess.
    """
    if dry_run:
        # Return mock result without executing
        return subprocess.CompletedProcess(
            args=job.cli_args,
            returncode=0,
            stdout=f"[DRY RUN] Would execute: {' '.join(job.cli_args)}",
            stderr="",
        )

    # Ensure output directory exists
    job.output_dir.mkdir(parents=True, exist_ok=True)

    # Execute reconstruction script
    result = subprocess.run(
        job.cli_args,
        capture_output=True,
        text=True,
        check=False,
    )

    return result


def main():
    """
    CLI entry point for Phase F pty-chi LSQML reconstruction orchestrator.

    Filters reconstruction jobs by dose/view/split/gridsize and executes them
    via run_ptychi_job(). Emits manifest + skip summary JSONs to artifact root.

    Example usage:
        python -m studies.fly64_dose_overlap.reconstruction \\
            --phase-c-root tmp/phase_c \\
            --phase-d-root tmp/phase_d \\
            --artifact-root plans/.../phase_f_reconstruction \\
            --dose 1000 \\
            --view dense \\
            --split train \\
            --dry-run \\
            --allow-missing-phase-d
    """
    import argparse
    import json
    import sys
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Phase F: PtyChi LSQML reconstruction job orchestrator"
    )
    parser.add_argument(
        "--phase-c-root",
        type=Path,
        required=True,
        help="Root directory for Phase C baseline datasets (dose_{dose}/patched_{split}.npz)",
    )
    parser.add_argument(
        "--phase-d-root",
        type=Path,
        required=True,
        help="Root directory for Phase D overlap views (dose_{dose}/{view}/{view}_{split}.npz)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root directory for reconstruction outputs and manifest",
    )
    parser.add_argument(
        "--dose",
        type=float,
        default=None,
        help="Filter by specific dose (e.g., 1000, 10000, 100000)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        choices=["baseline", "dense", "sparse"],
        help="Filter by specific view type",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "test"],
        help="Filter by specific data split",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=None,
        help="Filter by gridsize (1 for baseline, 2 for overlap views)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual job execution (validation only)",
    )
    parser.add_argument(
        "--allow-missing-phase-d",
        action="store_true",
        help="Allow missing Phase D datasets (skip with metadata instead of failing)",
    )

    args = parser.parse_args()

    # Ensure artifact root exists
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    # Build full job manifest with allow_missing flag
    print(f"Building job manifest from Phase C: {args.phase_c_root}")
    print(f"                      and Phase D: {args.phase_d_root}")

    all_jobs = build_ptychi_jobs(
        phase_c_root=args.phase_c_root,
        phase_d_root=args.phase_d_root,
        artifact_root=args.artifact_root,
        allow_missing=args.allow_missing_phase_d,
    )

    print(f"Total jobs enumerated: {len(all_jobs)}")

    # Apply filters
    filtered_jobs = all_jobs
    skipped_jobs = []

    if args.dose is not None:
        jobs_before = len(filtered_jobs)
        filtered_jobs = [j for j in filtered_jobs if j.dose == args.dose]
        for j in all_jobs:
            if j.dose != args.dose and j not in skipped_jobs:
                skipped_jobs.append({
                    "dose": j.dose,
                    "view": j.view,
                    "split": j.split,
                    "reason": f"dose filter (requested: {args.dose})",
                })
        print(f"Filter --dose={args.dose}: {jobs_before} → {len(filtered_jobs)} jobs")

    if args.view is not None:
        jobs_before = len(filtered_jobs)
        filtered_jobs = [j for j in filtered_jobs if j.view == args.view]
        for j in all_jobs:
            if j.view != args.view and j not in [s["dose"] == j.dose and s["view"] == j.view and s["split"] == j.split for s in skipped_jobs]:
                if not any(s["dose"] == j.dose and s["view"] == j.view and s["split"] == j.split for s in skipped_jobs):
                    skipped_jobs.append({
                        "dose": j.dose,
                        "view": j.view,
                        "split": j.split,
                        "reason": f"view filter (requested: {args.view})",
                    })
        print(f"Filter --view={args.view}: {jobs_before} → {len(filtered_jobs)} jobs")

    if args.split is not None:
        jobs_before = len(filtered_jobs)
        filtered_jobs = [j for j in filtered_jobs if j.split == args.split]
        for j in all_jobs:
            if j.split != args.split and not any(s["dose"] == j.dose and s["view"] == j.view and s["split"] == j.split for s in skipped_jobs):
                skipped_jobs.append({
                    "dose": j.dose,
                    "view": j.view,
                    "split": j.split,
                    "reason": f"split filter (requested: {args.split})",
                })
        print(f"Filter --split={args.split}: {jobs_before} → {len(filtered_jobs)} jobs")

    if args.gridsize is not None:
        jobs_before = len(filtered_jobs)
        # Gridsize semantics: 1 = baseline, 2 = dense/sparse
        if args.gridsize == 1:
            filtered_jobs = [j for j in filtered_jobs if j.view == "baseline"]
        elif args.gridsize == 2:
            filtered_jobs = [j for j in filtered_jobs if j.view in ["dense", "sparse"]]
        else:
            print(f"Warning: Invalid gridsize {args.gridsize} (expected 1 or 2), ignoring filter")

        for j in all_jobs:
            is_baseline = j.view == "baseline"
            matches_gridsize = (args.gridsize == 1 and is_baseline) or (args.gridsize == 2 and not is_baseline)
            if not matches_gridsize and not any(s["dose"] == j.dose and s["view"] == j.view and s["split"] == j.split for s in skipped_jobs):
                skipped_jobs.append({
                    "dose": j.dose,
                    "view": j.view,
                    "split": j.split,
                    "reason": f"gridsize filter (requested: {args.gridsize})",
                })
        print(f"Filter --gridsize={args.gridsize}: {jobs_before} → {len(filtered_jobs)} jobs")

    print(f"\nFiltered jobs: {len(filtered_jobs)} selected, {len(skipped_jobs)} skipped")

    # Execute filtered jobs
    execution_results = []
    for i, job in enumerate(filtered_jobs, 1):
        print(f"\n[{i}/{len(filtered_jobs)}] Job: dose={job.dose}, view={job.view}, split={job.split}")
        result = run_ptychi_job(job, dry_run=args.dry_run)

        execution_results.append({
            "dose": job.dose,
            "view": job.view,
            "split": job.split,
            "returncode": result.returncode,
            "stdout_preview": result.stdout[:200] if result.stdout else "",
        })

        if args.dry_run:
            print(f"  [DRY RUN] {result.stdout}")
        else:
            print(f"  Return code: {result.returncode}")
            if result.returncode != 0:
                print(f"  STDERR: {result.stderr}")

    # Emit manifest JSON
    manifest_path = args.artifact_root / "reconstruction_manifest.json"
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "phase_c_root": str(args.phase_c_root),
        "phase_d_root": str(args.phase_d_root),
        "artifact_root": str(args.artifact_root),
        "filters": {
            "dose": args.dose,
            "view": args.view,
            "split": args.split,
            "gridsize": args.gridsize,
        },
        "total_jobs": len(all_jobs),
        "filtered_jobs": len(filtered_jobs),
        "jobs": [
            {
                "dose": job.dose,
                "view": job.view,
                "split": job.split,
                "input_npz": str(job.input_npz),
                "output_dir": str(job.output_dir),
                "algorithm": job.algorithm,
                "num_epochs": job.num_epochs,
            }
            for job in filtered_jobs
        ],
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")

    # Emit skip summary JSON
    skip_summary_path = args.artifact_root / "skip_summary.json"
    skip_summary = {
        "timestamp": datetime.now().isoformat(),
        "skipped_count": len(skipped_jobs),
        "skipped_jobs": skipped_jobs,
    }

    with open(skip_summary_path, 'w') as f:
        json.dump(skip_summary, f, indent=2)

    print(f"Skip summary written to: {skip_summary_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Phase F Reconstruction Orchestrator Summary")
    print(f"{'='*60}")
    print(f"Total jobs:     {len(all_jobs)}")
    print(f"Filtered jobs:  {len(filtered_jobs)}")
    print(f"Skipped jobs:   {len(skipped_jobs)}")
    print(f"Dry run:        {args.dry_run}")
    print(f"Artifacts:      {args.artifact_root}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

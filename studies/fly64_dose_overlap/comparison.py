"""
Phase G comparison orchestration for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.

Builds comparison jobs for three-way comparisons (PINN vs baseline vs pty-chi)
across all dose/view/split conditions.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class ComparisonJob:
    """Dataclass representing a single comparison job."""
    dose: int
    view: str
    split: str
    phase_c_npz: Path
    pinn_checkpoint: Path
    baseline_checkpoint: Path
    phase_f_manifest: Path
    ms_ssim_sigma: float = 1.0
    skip_registration: bool = False
    register_ptychi_only: bool = True


def build_comparison_jobs(
    phase_c_root: Path,
    phase_e_root: Path,
    phase_f_root: Path,
    dose_filter: Optional[int] = None,
    view_filter: Optional[str] = None,
    split_filter: Optional[str] = None,
) -> List[ComparisonJob]:
    """
    Build deterministic comparison jobs for all dose/view/split conditions.

    Parameters
    ----------
    phase_c_root : Path
        Root directory containing Phase C datasets (dose_{dose}/patched_{split}.npz)
    phase_e_root : Path
        Root directory containing Phase E checkpoints (pinn/baseline subdirs)
    phase_f_root : Path
        Root directory containing Phase F pty-chi manifests
    dose_filter : Optional[int]
        If provided, only include jobs for this dose
    view_filter : Optional[str]
        If provided, only include jobs for this view (dense/sparse)
    split_filter : Optional[str]
        If provided, only include jobs for this split (train/test)

    Returns
    -------
    List[ComparisonJob]
        List of comparison jobs with pointers to Phase C/E/F artifacts.
        Jobs are ordered deterministically: by dose (asc), then view (dense before sparse),
        then split (train before test).
    """
    phase_c_root = Path(phase_c_root)
    phase_e_root = Path(phase_e_root)
    phase_f_root = Path(phase_f_root)

    # Define deterministic ordering
    doses = [500, 1000, 2000]
    views = ['dense', 'sparse']
    splits = ['train', 'test']

    # Apply filters
    if dose_filter is not None:
        doses = [d for d in doses if d == dose_filter]
    if view_filter is not None:
        views = [v for v in views if v == view_filter]
    if split_filter is not None:
        splits = [s for s in splits if s == split_filter]

    # Build jobs
    jobs = []
    for dose in doses:
        for view in views:
            for split in splits:
                # Phase C dataset path
                dose_dir = phase_c_root / f'dose_{dose}'
                if view == 'dense':
                    # Dense view uses patched data
                    phase_c_npz = dose_dir / f'patched_{split}.npz'
                else:
                    # Sparse view uses overlap data
                    phase_c_npz = dose_dir / view / f'{view}_{split}.npz'

                # Phase E checkpoints
                pinn_checkpoint = phase_e_root / 'pinn' / 'checkpoint.h5'
                baseline_checkpoint = phase_e_root / 'baseline' / 'checkpoint.h5'

                # Phase F manifest
                phase_f_manifest = phase_f_root / f'dose_{dose}_{view}_{split}' / 'manifest.json'

                # Validate paths exist (fail fast)
                if not phase_c_npz.exists():
                    raise FileNotFoundError(
                        f"Phase C dataset not found: {phase_c_npz}\n"
                        f"Expected at: {phase_c_root}/dose_{dose}/{view if view != 'dense' else ''}..."
                    )
                if not pinn_checkpoint.exists():
                    raise FileNotFoundError(f"PINN checkpoint not found: {pinn_checkpoint}")
                if not baseline_checkpoint.exists():
                    raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")
                if not phase_f_manifest.exists():
                    raise FileNotFoundError(f"Phase F manifest not found: {phase_f_manifest}")

                jobs.append(ComparisonJob(
                    dose=dose,
                    view=view,
                    split=split,
                    phase_c_npz=phase_c_npz,
                    pinn_checkpoint=pinn_checkpoint,
                    baseline_checkpoint=baseline_checkpoint,
                    phase_f_manifest=phase_f_manifest,
                    ms_ssim_sigma=1.0,
                    skip_registration=False,
                    register_ptychi_only=True,
                ))

    return jobs


def main():
    """CLI entry point for comparison orchestration."""
    parser = argparse.ArgumentParser(
        description="Phase G comparison orchestration - build three-way comparison jobs"
    )
    parser.add_argument('--phase-c-root', type=Path, required=True,
                        help='Root directory for Phase C datasets')
    parser.add_argument('--phase-e-root', type=Path, required=True,
                        help='Root directory for Phase E checkpoints')
    parser.add_argument('--phase-f-root', type=Path, required=True,
                        help='Root directory for Phase F pty-chi manifests')
    parser.add_argument('--artifact-root', type=Path, required=True,
                        help='Output directory for manifest and summary')
    parser.add_argument('--dose', type=int, choices=[500, 1000, 2000],
                        help='Filter to specific dose')
    parser.add_argument('--view', choices=['dense', 'sparse'],
                        help='Filter to specific view')
    parser.add_argument('--split', choices=['train', 'test'],
                        help='Filter to specific split')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build jobs but do not execute comparisons')

    args = parser.parse_args()

    # Ensure artifact root exists
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    # Build jobs
    print(f"Building comparison jobs...")
    print(f"  Phase C root: {args.phase_c_root}")
    print(f"  Phase E root: {args.phase_e_root}")
    print(f"  Phase F root: {args.phase_f_root}")
    print(f"  Filters: dose={args.dose}, view={args.view}, split={args.split}")

    try:
        jobs = build_comparison_jobs(
            phase_c_root=args.phase_c_root,
            phase_e_root=args.phase_e_root,
            phase_f_root=args.phase_f_root,
            dose_filter=args.dose,
            view_filter=args.view,
            split_filter=args.split,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nFailed to build comparison jobs due to missing prerequisites.")
        return 1

    print(f"\nBuilt {len(jobs)} comparison jobs")

    # Emit manifest
    manifest_path = args.artifact_root / 'comparison_manifest.json'
    manifest = {
        'n_jobs': len(jobs),
        'filters': {
            'dose': args.dose,
            'view': args.view,
            'split': args.split,
        },
        'jobs': [
            {
                'dose': job.dose,
                'view': job.view,
                'split': job.split,
                'phase_c_npz': str(job.phase_c_npz),
                'pinn_checkpoint': str(job.pinn_checkpoint),
                'baseline_checkpoint': str(job.baseline_checkpoint),
                'phase_f_manifest': str(job.phase_f_manifest),
                'ms_ssim_sigma': job.ms_ssim_sigma,
                'skip_registration': job.skip_registration,
                'register_ptychi_only': job.register_ptychi_only,
            }
            for job in jobs
        ],
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to: {manifest_path}")

    # Emit summary
    summary_path = args.artifact_root / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Phase G Comparison Jobs Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total jobs: {len(jobs)}\n")
        f.write(f"Filters: dose={args.dose}, view={args.view}, split={args.split}\n\n")
        f.write(f"Job Enumeration:\n")
        for i, job in enumerate(jobs, 1):
            f.write(f"  {i:2d}. dose={job.dose:4d} view={job.view:6s} split={job.split:5s}\n")
        if args.dry_run:
            f.write(f"\nDry-run mode: no comparison scripts executed\n")
    print(f"Wrote summary to: {summary_path}")

    if args.dry_run:
        print("\nDry-run mode: skipping comparison execution")
        return 0

    print("\nComparison execution not yet implemented (Phase G2)")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

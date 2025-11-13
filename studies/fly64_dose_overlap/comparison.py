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
    phase_f_root: Path
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
        Root directory containing Phase E checkpoints (dose-specific structure:
        dose_{dose}/baseline/gs1/wts.h5.zip and dose_{dose}/{view}/gs2/wts.h5.zip)
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

    # Define deterministic ordering of doses by scanning Phase C root
    # Accept any dose_* folders (e.g., dose_1000, dose_100000) and sort ascending
    doses = []
    for p in phase_c_root.iterdir():
        if p.is_dir() and p.name.startswith('dose_'):
            try:
                doses.append(int(p.name.split('_', 1)[1]))
            except (ValueError, IndexError):
                continue
    doses = sorted(set(doses))
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

                # Phase E checkpoints (dose-specific structure per training.py:184,226)
                # Baseline: dose_{dose}/baseline/gs1/wts.h5.zip
                # View (dense/sparse): dose_{dose}/{view}/gs2/wts.h5.zip
                dose_suffix = f'dose_{dose}'
                pinn_checkpoint = phase_e_root / dose_suffix / view / 'gs2' / 'wts.h5.zip'
                baseline_checkpoint = phase_e_root / dose_suffix / 'baseline' / 'gs1' / 'wts.h5.zip'

                # Phase F manifest (legacy per-split) — may not exist if Phase F writes
                # a single reconstruction_manifest.json at artifact root. We'll tolerate
                # missing per-split manifests and fall back later.
                phase_f_manifest = phase_f_root / f'dose_{dose}_{view}_{split}' / 'manifest.json'

                # Validate required Phase C/E paths (fail fast). Phase F is validated later.
                if not phase_c_npz.exists():
                    raise FileNotFoundError(
                        f"Phase C dataset not found: {phase_c_npz}\n"
                        f"Expected at: {phase_c_root}/dose_{dose}/{view if view != 'dense' else ''}..."
                    )
                if not pinn_checkpoint.exists():
                    raise FileNotFoundError(f"PINN checkpoint not found: {pinn_checkpoint}")
                if not baseline_checkpoint.exists():
                    raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")

                jobs.append(ComparisonJob(
                    dose=dose,
                    view=view,
                    split=split,
                    phase_c_npz=phase_c_npz,
                    pinn_checkpoint=pinn_checkpoint,
                    baseline_checkpoint=baseline_checkpoint,
                    phase_f_manifest=phase_f_manifest,
                    phase_f_root=phase_f_root,
                    ms_ssim_sigma=1.0,
                    skip_registration=False,
                    register_ptychi_only=True,
                ))

    return jobs


def execute_comparison_jobs(
    jobs: List[ComparisonJob],
    artifact_root: Path,
) -> dict:
    """
    Execute comparison jobs by shelling out to scripts/compare_models.py.

    Parameters
    ----------
    jobs : List[ComparisonJob]
        List of comparison jobs to execute
    artifact_root : Path
        Root directory for artifacts (logs, execution results)

    Returns
    -------
    dict
        Manifest with execution results including return codes and log paths
    """
    import subprocess
    import sys

    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    execution_results = []

    # Cache for parsed Phase F manifests (avoid redundant IO per TYPE-PATH-001 guidance)
    phase_f_manifest_cache = {}

    for job in jobs:
        # Construct output directory for this comparison
        output_dir = artifact_root / f"dose_{job.dose}" / job.view / job.split
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct log path
        log_path = output_dir / "comparison.log"

        # Resolve Phase F reconstruction path. Prefer legacy per-split manifest if present;
        # otherwise, fall back to the known output layout used by reconstruction.py.
        phase_f_manifest_path = Path(job.phase_f_manifest)
        tike_recon_path = None
        if phase_f_manifest_path.exists():
            if phase_f_manifest_path not in phase_f_manifest_cache:
                try:
                    with open(phase_f_manifest_path, 'r') as f:
                        phase_f_manifest_cache[phase_f_manifest_path] = json.load(f)
                except (json.JSONDecodeError,) as e:
                    # Fall back if manifest cannot be parsed
                    phase_f_manifest_cache[phase_f_manifest_path] = None
            phase_f_manifest = phase_f_manifest_cache.get(phase_f_manifest_path)
            if isinstance(phase_f_manifest, dict) and 'output_dir' in phase_f_manifest:
                phase_f_output_dir = Path(phase_f_manifest['output_dir'])
                tike_recon_path = phase_f_output_dir / 'ptychi_reconstruction.npz'

        # Fallback: derive expected reconstruction path directly under Phase F root
        # using the known per-split directory that would contain manifest.json
        if tike_recon_path is None:
            # Construct from known Phase F root canonical layout
            tike_recon_path = Path(job.phase_f_root) / f'dose_{job.dose}' / job.view / job.split / 'ptychi_reconstruction.npz'

        # Fail fast if reconstruction file does not exist
        if not tike_recon_path.exists():
            raise FileNotFoundError(
                f"Phase F reconstruction NPZ not found: {tike_recon_path}\n"
                f"Expected from Phase F LSQML reconstruction (dose={job.dose}, view={job.view}, split={job.split})"
            )

        # Build command to invoke scripts/compare_models.py
        # Use sys.executable to ensure same Python interpreter
        cmd = [
            sys.executable,
            "-m", "scripts.compare_models",
            "--pinn_dir", str(job.pinn_checkpoint.parent),
            "--baseline_dir", str(job.baseline_checkpoint.parent),
            "--test_data", str(job.phase_c_npz),
            "--output_dir", str(output_dir),
            "--ms-ssim-sigma", str(job.ms_ssim_sigma),
        ]

        # Append --tike_recon_path for three-way comparison (PINN vs baseline vs pty-chi LSQML)
        cmd.extend(["--tike_recon_path", str(tike_recon_path)])

        # Add registration flags
        if job.skip_registration:
            cmd.append("--skip-registration")
        if job.register_ptychi_only:
            cmd.append("--register-ptychi-only")

        # Execute subprocess
        print(f"\nExecuting comparison for dose={job.dose}, view={job.view}, split={job.split}")
        print(f"  Output: {output_dir}")
        print(f"  Log: {log_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            # Write log
            with open(log_path, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            execution_results.append({
                'dose': job.dose,
                'view': job.view,
                'split': job.split,
                'returncode': result.returncode,
                'log_path': str(log_path.relative_to(artifact_root)),
            })

            if result.returncode == 0:
                print(f"  SUCCESS (returncode={result.returncode})")
            else:
                print(f"  FAILED (returncode={result.returncode})")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 600s")
            execution_results.append({
                'dose': job.dose,
                'view': job.view,
                'split': job.split,
                'returncode': -1,
                'log_path': str(log_path.relative_to(artifact_root)),
                'error': 'timeout',
            })
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            execution_results.append({
                'dose': job.dose,
                'view': job.view,
                'split': job.split,
                'returncode': -1,
                'log_path': str(log_path.relative_to(artifact_root)),
                'error': str(e),
            })

    # Calculate success/failure counts
    n_success = sum(1 for r in execution_results if r['returncode'] == 0)
    n_failed = len(execution_results) - n_success

    # Build manifest with execution results
    manifest = {
        'n_jobs': len(jobs),
        'n_executed': len(execution_results),
        'n_success': n_success,
        'n_failed': n_failed,
        'execution_results': execution_results,
    }

    return manifest


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
    parser.add_argument('--dose', type=int,
                        help='Filter to specific dose (any integer present under phase-c-root)')
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

    # Execute comparison jobs
    print("\n" + "=" * 50)
    print("EXECUTING COMPARISON JOBS")
    print("=" * 50)

    execution_manifest = execute_comparison_jobs(
        jobs=jobs,
        artifact_root=args.artifact_root,
    )

    # Update manifest with execution results (including summary counts)
    manifest['execution_results'] = execution_manifest['execution_results']
    manifest['n_success'] = execution_manifest['n_success']
    manifest['n_failed'] = execution_manifest['n_failed']

    # Rewrite manifest with execution results
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nUpdated manifest with execution results: {manifest_path}")

    # Emit execution summary
    print(f"\nExecution Summary:")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Executed: {execution_manifest['n_executed']}")
    print(f"  Success: {execution_manifest['n_success']}")
    print(f"  Failed: {execution_manifest['n_failed']}")

    # Return non-zero if any failures
    return 1 if execution_manifest['n_failed'] > 0 else 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

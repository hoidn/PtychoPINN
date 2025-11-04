"""
Phase E training job matrix builder for fly64 dose/overlap study.

This module provides the job enumeration logic for training PtychoPINN across
all dose/view/gridsize combinations in the study design. It generates TrainingJob
instances that encapsulate dataset paths, configuration metadata, and artifact
destinations without executing any training.

Key responsibilities:
- Enumerate 9 jobs per dose (3 doses × 3 variants: baseline gs1, dense gs2, sparse gs2)
- Validate dataset file existence from Phase C and Phase D outputs
- Derive deterministic artifact paths for logs and checkpoints
- Maintain CONFIG-001 boundaries (no params.cfg mutation; bridge deferred to execution)

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:133-144
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-115
- specs/data_contracts.md:190-260 (DATA-001 NPZ requirements)
- docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001 bridge ordering)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Dict, Any
from studies.fly64_dose_overlap.design import StudyDesign, get_study_design
from ptycho.config.config import update_legacy_dict
from ptycho import params as p


@dataclass
class TrainingJob:
    """
    Encapsulates metadata for a single PtychoPINN training run.

    Attributes:
        dose: Photon dose level (nphotons per exposure)
        view: Dataset view ('baseline', 'dense', or 'sparse')
        gridsize: Gridsize configuration (1 or 2)
        train_data_path: Path to training NPZ (Phase C or D output)
        test_data_path: Path to test NPZ (Phase C or D output)
        artifact_dir: Directory for checkpoints and outputs
        log_path: Path to training log file

    Invariants:
        - baseline view uses gridsize=1 with Phase C patched NPZs
        - dense/sparse views use gridsize=2 with Phase D filtered NPZs
        - train_data_path and test_data_path must exist on disk
        - artifact_dir is deterministic from dose/view/gridsize
    """
    dose: float
    view: str
    gridsize: int
    train_data_path: str
    test_data_path: str
    artifact_dir: Path
    log_path: Path

    def __post_init__(self):
        """Validate job invariants."""
        # Validate view
        valid_views = {'baseline', 'dense', 'sparse'}
        if self.view not in valid_views:
            raise ValueError(
                f"Invalid view '{self.view}'. Expected one of: {valid_views}"
            )

        # Validate gridsize matches view
        if self.view == 'baseline' and self.gridsize != 1:
            raise ValueError(
                f"Baseline jobs must use gridsize=1, got {self.gridsize}"
            )
        if self.view in {'dense', 'sparse'} and self.gridsize != 2:
            raise ValueError(
                f"Overlap jobs ({self.view}) must use gridsize=2, got {self.gridsize}"
            )

        # Validate dataset paths exist
        if not Path(self.train_data_path).exists():
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_data_path}"
            )
        if not Path(self.test_data_path).exists():
            raise FileNotFoundError(
                f"Test dataset not found: {self.test_data_path}"
            )


def build_training_jobs(
    phase_c_root: Path,
    phase_d_root: Path,
    artifact_root: Path,
    design: StudyDesign | None = None,
) -> List[TrainingJob]:
    """
    Enumerate all training jobs for the dose/overlap study.

    This function generates a job matrix covering:
    - 3 doses (from StudyDesign.dose_list)
    - 3 variants per dose:
      - baseline: gs1, Phase C patched_{train,test}.npz
      - dense: gs2, Phase D dense_{train,test}.npz
      - sparse: gs2, Phase D sparse_{train,test}.npz

    Total: 3 doses × 3 variants = 9 jobs

    Args:
        phase_c_root: Root directory for Phase C dataset outputs
                      (expects dose_{dose}/patched_{train,test}.npz)
        phase_d_root: Root directory for Phase D overlap views
                      (expects dose_{dose}/{view}_{train,test}.npz)
        artifact_root: Root directory for training artifacts
                       (job-specific subdirs created here)
        design: StudyDesign instance (default: get_study_design())

    Returns:
        List of TrainingJob instances, one per dose/view/gridsize combination

    Raises:
        FileNotFoundError: If any required dataset file is missing

    References:
        - CONFIG-001: This function remains pure (no params.cfg mutation).
                      Legacy bridge via update_legacy_dict() is deferred to
                      execution helper (run_training_job in task E3).
        - DATA-001: Dataset paths validated for existence; actual NPZ contract
                    enforcement occurs during training via loader.
        - OVERSAMPLING-001: Gridsize=2 jobs assume neighbor_count=7 from Phase D.

    Examples:
        >>> design = get_study_design()
        >>> jobs = build_training_jobs(
        ...     phase_c_root=Path("outputs/phase_c"),
        ...     phase_d_root=Path("outputs/phase_d"),
        ...     artifact_root=Path("artifacts/training"),
        ...     design=design,
        ... )
        >>> len(jobs)
        9
        >>> {j.view for j in jobs}
        {'baseline', 'dense', 'sparse'}
    """
    if design is None:
        design = get_study_design()

    jobs = []

    for dose in design.dose_list:
        dose_int = int(dose)
        dose_suffix = f"dose_{dose_int}"

        # Job 1: Baseline (gs1, Phase C patched NPZs)
        baseline_job = TrainingJob(
            dose=dose,
            view='baseline',
            gridsize=1,
            train_data_path=str(phase_c_root / dose_suffix / "patched_train.npz"),
            test_data_path=str(phase_c_root / dose_suffix / "patched_test.npz"),
            artifact_dir=artifact_root / dose_suffix / "baseline" / "gs1",
            log_path=artifact_root / dose_suffix / "baseline" / "gs1" / "train.log",
        )
        jobs.append(baseline_job)

        # Jobs 2-3: Overlap views (gs2, Phase D filtered NPZs)
        for view in ['dense', 'sparse']:
            overlap_job = TrainingJob(
                dose=dose,
                view=view,
                gridsize=2,
                train_data_path=str(phase_d_root / dose_suffix / f"{view}_train.npz"),
                test_data_path=str(phase_d_root / dose_suffix / f"{view}_test.npz"),
                artifact_dir=artifact_root / dose_suffix / view / "gs2",
                log_path=artifact_root / dose_suffix / view / "gs2" / "train.log",
            )
            jobs.append(overlap_job)

    return jobs


def run_training_job(
    job: TrainingJob,
    runner: Callable,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single training job with CONFIG-001 compliance and logging.

    This helper orchestrates the execution of one PtychoPINN training run by:
    1. Creating artifact and log directories
    2. Constructing a TrainingConfig dataclass with job metadata
    3. Bridging params.cfg via update_legacy_dict (CONFIG-001 compliance)
    4. Invoking the injected runner callable (or skipping if dry_run=True)
    5. Ensuring log_path is touched/written for downstream traceability

    Args:
        job: TrainingJob instance with dataset paths and artifact destinations
        runner: Callable that executes training, signature:
                runner(*, config, job, log_path) -> Dict[str, Any]
                The runner receives:
                - config: A TrainingConfig dataclass instance
                - job: The original TrainingJob instance
                - log_path: Path where training logs should be written
        dry_run: If True, skip runner invocation and return summary dict instead

    Returns:
        If dry_run=False: Result dict returned by runner (e.g., {'status': 'success'})
        If dry_run=True: Summary dict describing what would be executed:
            {
                'dose': float,
                'view': str,
                'gridsize': int,
                'train_data_path': str,
                'test_data_path': str,
                'log_path': Path,
                'dry_run': True,
            }

    Raises:
        Any exception raised by runner is propagated after ensuring logs persisted

    References:
        - CONFIG-001: update_legacy_dict must be called with TrainingConfig before any legacy loaders
        - DATA-001: Dataset paths validated at TrainingJob construction time
        - OVERSAMPLING-001: Gridsize semantics preserved in job metadata
        - input.md:10 (Phase E4: upgrade to TrainingConfig + update_legacy_dict)

    Examples:
        >>> # Real training execution
        >>> def actual_trainer(*, config, job, log_path):
        ...     # Train model...
        ...     return {'status': 'success', 'final_loss': 0.123}
        >>> result = run_training_job(job, runner=actual_trainer, dry_run=False)

        >>> # Dry-run preview
        >>> result = run_training_job(job, runner=actual_trainer, dry_run=True)
        >>> result['dry_run']
        True
    """
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Step 1: Create artifact and log directories
    job.artifact_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 2: Touch log file to ensure it exists for downstream tools
    job.log_path.touch(exist_ok=True)

    # Step 3: If dry_run, return summary without executing
    if dry_run:
        summary = {
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
            'train_data_path': job.train_data_path,
            'test_data_path': job.test_data_path,
            'log_path': job.log_path,
            'artifact_dir': job.artifact_dir,
            'dry_run': True,
        }
        # Write dry-run marker to log
        with job.log_path.open('w') as f:
            f.write(f"DRY RUN: {job.view} dose={job.dose} gridsize={job.gridsize}\n")
            f.write(f"Train: {job.train_data_path}\n")
            f.write(f"Test: {job.test_data_path}\n")
        return summary

    # Step 4: Construct TrainingConfig dataclass for CONFIG-001 bridge
    # Create ModelConfig with gridsize
    model_config = ModelConfig(gridsize=job.gridsize)

    # Create TrainingConfig with essential fields
    config = TrainingConfig(
        train_data_file=job.train_data_path,
        test_data_file=job.test_data_path,
        output_dir=str(job.artifact_dir),
        model=model_config,
        nphotons=job.dose,  # Pass dose as nphotons for CONFIG-001 bridge
    )

    # Step 5: Bridge params.cfg (CONFIG-001 compliance)
    # Must call update_legacy_dict BEFORE any runner invocation
    update_legacy_dict(p.cfg, config)

    # Step 6: Invoke runner with standard kwargs
    result = runner(config=config, job=job, log_path=job.log_path)

    # Step 7: Return runner result
    return result


def main():
    """
    CLI entrypoint for fly64 dose/overlap study training jobs.

    This CLI orchestrates training across the full job matrix by:
    1. Parsing command-line arguments (Phase C/D roots, artifact root, filters)
    2. Enumerating all jobs via build_training_jobs()
    3. Filtering jobs by optional --dose, --view, --gridsize flags
    4. Executing each matching job via run_training_job() with CONFIG-001 bridge
    5. Emitting training_manifest.json with job metadata for downstream analysis

    CLI Arguments:
        --phase-c-root: Root directory for Phase C dataset outputs (required)
        --phase-d-root: Root directory for Phase D overlap views (required)
        --artifact-root: Root directory for training artifacts/logs (required)
        --dose: Optional dose filter (e.g., 1000, 10000, 100000)
        --view: Optional view filter (baseline, dense, or sparse)
        --gridsize: Optional gridsize filter (1 or 2)
        --dry-run: If set, skip training execution and emit summary only

    Examples:
        # Train all jobs (9 total: 3 doses × 3 variants)
        python -m studies.fly64_dose_overlap.training \\
            --phase-c-root outputs/phase_c \\
            --phase-d-root outputs/phase_d \\
            --artifact-root artifacts/training

        # Train only baseline jobs (gridsize=1)
        python -m studies.fly64_dose_overlap.training \\
            --phase-c-root outputs/phase_c \\
            --phase-d-root outputs/phase_d \\
            --artifact-root artifacts/training \\
            --view baseline

        # Dry-run for dose=1e4, dense view
        python -m studies.fly64_dose_overlap.training \\
            --phase-c-root outputs/phase_c \\
            --phase-d-root outputs/phase_d \\
            --artifact-root artifacts/training \\
            --dose 10000 \\
            --view dense \\
            --dry-run

    References:
        - input.md:10 (Phase E4 CLI requirements)
        - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-115
    """
    import argparse
    import json
    from datetime import datetime

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Train PtychoPINN across fly64 dose/overlap study matrix"
    )
    parser.add_argument(
        '--phase-c-root',
        type=Path,
        required=True,
        help='Root directory for Phase C dataset outputs (dose_*/patched_{train,test}.npz)'
    )
    parser.add_argument(
        '--phase-d-root',
        type=Path,
        required=True,
        help='Root directory for Phase D overlap views (dose_*/{dense,sparse}_{train,test}.npz)'
    )
    parser.add_argument(
        '--artifact-root',
        type=Path,
        required=True,
        help='Root directory for training artifacts (logs, checkpoints, manifest)'
    )
    parser.add_argument(
        '--dose',
        type=float,
        default=None,
        help='Optional dose filter (e.g., 1000, 10000, 100000)'
    )
    parser.add_argument(
        '--view',
        type=str,
        default=None,
        choices=['baseline', 'dense', 'sparse'],
        help='Optional view filter (baseline, dense, or sparse)'
    )
    parser.add_argument(
        '--gridsize',
        type=int,
        default=None,
        choices=[1, 2],
        help='Optional gridsize filter (1 or 2)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Skip training execution; emit summary only'
    )

    args = parser.parse_args()

    # Ensure artifact root exists
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Build full job matrix
    print(f"Enumerating training jobs from Phase C ({args.phase_c_root}) and Phase D ({args.phase_d_root})...")
    all_jobs = build_training_jobs(
        phase_c_root=args.phase_c_root,
        phase_d_root=args.phase_d_root,
        artifact_root=args.artifact_root,
    )
    print(f"  → {len(all_jobs)} total jobs enumerated")

    # Step 2: Apply filters
    filtered_jobs = all_jobs

    if args.dose is not None:
        filtered_jobs = [j for j in filtered_jobs if j.dose == args.dose]
        print(f"  → Filtered by dose={args.dose}: {len(filtered_jobs)} jobs remain")

    if args.view is not None:
        filtered_jobs = [j for j in filtered_jobs if j.view == args.view]
        print(f"  → Filtered by view={args.view}: {len(filtered_jobs)} jobs remain")

    if args.gridsize is not None:
        filtered_jobs = [j for j in filtered_jobs if j.gridsize == args.gridsize]
        print(f"  → Filtered by gridsize={args.gridsize}: {len(filtered_jobs)} jobs remain")

    # Handle case where no jobs match filters
    if not filtered_jobs:
        print("\n⚠ No jobs match the specified filters. Exiting without training.")
        return

    # Step 3: Define stub runner for CLI execution
    # In a real training loop, this would invoke ptycho_train or similar
    def stub_runner(*, config, job, log_path):
        """Placeholder runner for Phase E4 CLI; will be wired to actual trainer in E5."""
        return {'status': 'stub_complete', 'job': job.view}

    # Step 4: Execute filtered jobs
    print(f"\nExecuting {len(filtered_jobs)} training job(s)...")
    job_results = []

    for i, job in enumerate(filtered_jobs, start=1):
        print(f"  [{i}/{len(filtered_jobs)}] {job.view} (dose={job.dose:.0e}, gridsize={job.gridsize})")
        result = run_training_job(job, runner=stub_runner, dry_run=args.dry_run)

        # Convert result dict paths to strings for JSON serialization
        result_serializable = {}
        for k, v in result.items():
            if isinstance(v, Path):
                result_serializable[k] = str(v)
            else:
                result_serializable[k] = v

        job_results.append({
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
            'train_data_path': job.train_data_path,
            'test_data_path': job.test_data_path,
            'log_path': str(job.log_path),
            'artifact_dir': str(job.artifact_dir),
            'result': result_serializable,
        })

    # Step 5: Emit training_manifest.json
    manifest = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'phase_c_root': str(args.phase_c_root),
        'phase_d_root': str(args.phase_d_root),
        'artifact_root': str(args.artifact_root),
        'filters': {
            'dose': args.dose,
            'view': args.view,
            'gridsize': args.gridsize,
        },
        'dry_run': args.dry_run,
        'jobs': job_results,
    }

    manifest_path = args.artifact_root / "training_manifest.json"
    with manifest_path.open('w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Training manifest written to {manifest_path}")
    print(f"  → {len(job_results)} job(s) completed")


if __name__ == '__main__':
    main()

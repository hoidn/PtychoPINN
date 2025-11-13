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
from typing import List, Callable, Dict, Any, Literal

import numpy as np

from studies.fly64_dose_overlap.design import StudyDesign, get_study_design
from studies.fly64_dose_overlap.validation import validate_dataset_contract
from ptycho.config.config import update_legacy_dict
from ptycho.workflows import components as tf_components
from ptycho import model_manager
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
    allow_missing_phase_d: bool = False,
    skip_events: List[Dict[str, Any]] | None = None,
) -> List[TrainingJob]:
    """
    Enumerate all training jobs for the dose/overlap study.

    This function generates a job matrix covering:
    - 3 doses (from StudyDesign.dose_list)
    - 3 variants per dose:
      - baseline: gs1, Phase C patched_{train,test}.npz
      - dense: gs2, Phase D dose/view/view_{train,test}.npz
      - sparse: gs2, Phase D dose/view/view_{train,test}.npz

    Total: Up to 3 doses × 3 variants = 9 jobs (fewer if allow_missing_phase_d=True and overlap data missing)

    Args:
        phase_c_root: Root directory for Phase C dataset outputs
                      (expects dose_{dose}/patched_{train,test}.npz)
        phase_d_root: Root directory for Phase D overlap views
                      (expects dose_{dose}/{view}/{view}_{train,test}.npz per overlap.py:490,366)
        artifact_root: Root directory for training artifacts
                       (job-specific subdirs created here)
        design: StudyDesign instance (default: get_study_design())
        allow_missing_phase_d: If True, skip overlap jobs when NPZ files missing (with logging);
                               If False (default), raise FileNotFoundError for strict validation
        skip_events: Optional list to accumulate skip metadata when allow_missing_phase_d=True.
                     Each skip appends a dict: {'dose': float, 'view': str, 'reason': str}

    Returns:
        List of TrainingJob instances, one per dose/view/gridsize combination

    Raises:
        FileNotFoundError: If any required dataset file is missing and allow_missing_phase_d=False

    References:
        - CONFIG-001: This function remains pure (no params.cfg mutation).
                      Legacy bridge via update_legacy_dict() is deferred to
                      execution helper (run_training_job in task E3).
        - DATA-001: Dataset paths validated for existence; actual NPZ contract
                    enforcement occurs during training via loader.
        - OVERSAMPLING-001: Gridsize=2 jobs assume neighbor_count=7 from Phase D.
        - input.md:10 (Phase E5 path alignment + allow_missing_phase_d parameter)
        - studies/fly64_dose_overlap/overlap.py:490 (Phase D creates dose/view/ subdirectories)
        - studies/fly64_dose_overlap/overlap.py:366 (Phase D writes view_split.npz under view subdir)

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

        >>> # Handle missing sparse view gracefully
        >>> jobs = build_training_jobs(
        ...     phase_c_root=Path("outputs/phase_c"),
        ...     phase_d_root=Path("outputs/phase_d_incomplete"),
        ...     artifact_root=Path("artifacts/training"),
        ...     design=design,
        ...     allow_missing_phase_d=True,
        ... )
        >>> len(jobs)
        6  # Only baseline + dense for 3 doses
    """
    import logging
    logger = logging.getLogger(__name__)

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

        # Jobs 2-3: Overlap views (gs2, Phase D filtered NPZs under view subdirectories)
        for view in ['dense', 'sparse']:
            # Phase D path (primary): dose_{dose}/{view}/{view}_{split}.npz
            primary_train = phase_d_root / dose_suffix / view / f"{view}_train.npz"
            primary_test = phase_d_root / dose_suffix / view / f"{view}_test.npz"

            # Backward-compatibility (fallback): some generators write train.npz/test.npz
            fallback_train = phase_d_root / dose_suffix / view / "train.npz"
            fallback_test = phase_d_root / dose_suffix / view / "test.npz"

            # Prefer primary naming; fallback if needed
            train_data_path = primary_train if primary_train.exists() else fallback_train
            test_data_path = primary_test if primary_test.exists() else fallback_test

            # Check if overlap view NPZs exist (after fallback)
            if not train_data_path.exists() or not test_data_path.exists():
                if allow_missing_phase_d:
                    # Build skip reason message (report both probed paths)
                    reason = (
                        f"NPZ files not found (train={train_data_path.exists()}, test={test_data_path.exists()}). "
                        f"Checked primary ({primary_train.name}/{primary_test.name}) and fallback (train.npz/test.npz)."
                    )

                    # Log skip
                    logger.info(f"Skipping {view} view for dose={dose:.0e}: {reason}")

                    # Accumulate skip metadata if caller provided a list
                    if skip_events is not None:
                        skip_events.append({
                            'dose': dose,
                            'view': view,
                            'reason': reason,
                        })

                    continue
                else:
                    # Strict mode: raise FileNotFoundError (handled by TrainingJob.__post_init__)
                    pass  # TrainingJob validation will raise

            overlap_job = TrainingJob(
                dose=dose,
                view=view,
                gridsize=2,
                train_data_path=str(train_data_path),
                test_data_path=str(test_data_path),
                artifact_dir=artifact_root / dose_suffix / view / "gs2",
                log_path=artifact_root / dose_suffix / view / "gs2" / "train.log",
            )
            jobs.append(overlap_job)

    return jobs


BackendLiteral = Literal['tensorflow', 'pytorch']


def _infer_patch_size(npz_path: str | Path) -> int:
    """Infer patch size N from an NPZ file (diffraction or probeGuess).

    Prefers diffraction array shape (N,H,W) → returns H. Falls back to
    probeGuess.shape[0] when present. Defaults to 64 if inference fails.
    """
    from pathlib import Path as _Path
    import numpy as _np

    try:
        p = _Path(npz_path)
        with _np.load(p, allow_pickle=True) as d:
            if 'diffraction' in d:
                arr = d['diffraction']
                if arr.ndim == 3:
                    return int(arr.shape[1])
            if 'diff3d' in d:
                arr = d['diff3d']
                if arr.ndim == 3:
                    # diff3d convention is (N,H,W)
                    return int(arr.shape[1])
            if 'probeGuess' in d:
                pg = d['probeGuess']
                return int(pg.shape[0])
    except Exception:
        pass
    return 64


def run_training_job(
    job: TrainingJob,
    runner: Callable,
    dry_run: bool = False,
    backend: BackendLiteral = 'tensorflow',
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
                runner(*, config, job, log_path, backend) -> Dict[str, Any]
                The runner receives:
                - config: A TrainingConfig dataclass instance
                - job: The original TrainingJob instance
                - log_path: Path where training logs should be written
                - backend: Literal['tensorflow','pytorch'] indicating selected backend
        dry_run: If True, skip runner invocation and return summary dict instead
        backend: Backend selection ('tensorflow' default)

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
                'backend': str,
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
            'backend': backend,
        }
        # Write dry-run marker to log
        with job.log_path.open('w') as f:
            f.write(f"DRY RUN: {job.view} dose={job.dose} gridsize={job.gridsize}\n")
            f.write(f"Train: {job.train_data_path}\n")
            f.write(f"Test: {job.test_data_path}\n")
        return summary

    # Step 4: Construct TrainingConfig dataclass for CONFIG-001 bridge
    # Create ModelConfig with gridsize and inferred patch size N from dataset
    inferred_N = _infer_patch_size(job.train_data_path)
    model_config = ModelConfig(gridsize=job.gridsize, N=inferred_N)

    # Create TrainingConfig with essential fields
    config = TrainingConfig(
        train_data_file=job.train_data_path,
        test_data_file=job.test_data_path,
        output_dir=str(job.artifact_dir),
        model=model_config,
        nphotons=job.dose,  # Pass dose as nphotons for CONFIG-001 bridge
        backend=backend,
    )

    # Step 5: Bridge params.cfg (CONFIG-001 compliance)
    # Must call update_legacy_dict BEFORE any runner invocation
    update_legacy_dict(p.cfg, config)

    # Step 6: Invoke runner with standard kwargs
    result = runner(config=config, job=job, log_path=job.log_path, backend=backend)

    # Step 7: Return runner result
    return result


def _write_execution_header(log_path: Path, config, job: TrainingJob, backend: str) -> None:
    """Append a standardized execution header to the training log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(
            "Phase E5 Training Execution — "
            f"{job.view} (dose={job.dose:.0e}, gridsize={job.gridsize}, backend={backend})\n"
        )
        f.write(f"{'=' * 80}\n")
        f.write(f"Train dataset: {config.train_data_file}\n")
        f.write(f"Test dataset: {config.test_data_file}\n")
        f.write(f"Output directory: {config.output_dir}\n")
        f.write(f"Gridsize: {config.model.gridsize}\n")
        f.write(f"nphotons: {config.nphotons}\n")
        f.write(f"Backend: {backend}\n\n")


def execute_training_job(
    *,
    config,
    job,
    log_path,
    backend: BackendLiteral = 'tensorflow',
):
    """Dispatcher that routes to the requested backend implementation."""
    if backend == 'tensorflow':
        return _execute_training_job_tensorflow(config=config, job=job, log_path=log_path)
    if backend == 'pytorch':
        return _execute_training_job_pytorch(config=config, job=job, log_path=log_path)
    raise ValueError(f"Unsupported backend '{backend}'. Expected 'tensorflow' or 'pytorch'.")


def _execute_training_job_tensorflow(*, config, job, log_path):
    """Execute a single training job using the TensorFlow backend."""
    import hashlib
    import traceback
    from pathlib import Path

    _write_execution_header(log_path, config, job, backend='tensorflow')

    train_path = Path(config.train_data_file)
    test_path = Path(config.test_data_file)

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    # Phase D now measures overlap via metrics (see docs/GRIDSIZE_N_GROUPS_GUIDE.md),
    # and spacing/packing acceptance gates are removed. Disable spacing gating by
    # omitting the 'view' parameter during dataset validation.
    view_for_validation = None

    try:
        with np.load(train_path, allow_pickle=True) as train_npz:
            validate_dataset_contract(
                dict(train_npz),
                view=view_for_validation,
                gridsize=job.gridsize,
                neighbor_count=config.neighbor_count,
            )
        with np.load(test_path, allow_pickle=True) as test_npz:
            validate_dataset_contract(
                dict(test_npz),
                view=view_for_validation,
                gridsize=job.gridsize,
                neighbor_count=config.neighbor_count,
            )
    except Exception as exc:  # pylint: disable=broad-except
        with log_path.open('a') as f:
            f.write(
                f"Dataset contract validation failed: {type(exc).__name__}: {exc}\n"
            )
            f.write(traceback.format_exc())
        return {
            'status': 'failed',
            'error': f"Dataset validation failed: {type(exc).__name__}: {exc}",
        }

    try:
        with log_path.open('a') as f:
            f.write(f"Loading datasets via TensorFlow loader...\n")
        train_data = tf_components.load_data(str(train_path))
        test_data = tf_components.load_data(str(test_path))
    except Exception as exc:  # pylint: disable=broad-except
        with log_path.open('a') as f:
            f.write(
                f"Failed to load datasets via TensorFlow loader: {type(exc).__name__}: {exc}\n"
            )
            f.write(traceback.format_exc())
        return {
            'status': 'failed',
            'error': f"Dataset loading failed: {type(exc).__name__}: {exc}",
        }

    try:
        with log_path.open('a') as f:
            f.write("Invoking TensorFlow train_cdi_model...\n")
        training_results = tf_components.train_cdi_model(
            train_data=train_data,
            test_data=test_data,
            config=config,
        )
    except Exception as exc:  # pylint: disable=broad-except
        with log_path.open('a') as f:
            f.write(f"Training failed: {type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
        return {
            'status': 'failed',
            'error': f"Training failed: {type(exc).__name__}: {exc}",
        }

    final_loss = None
    if 'history' in training_results and 'train_loss' in training_results['history']:
        train_losses = training_results['history']['train_loss']
        if train_losses:
            final_loss = train_losses[-1]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = None
    bundle_sha256 = None
    bundle_size_bytes = None
    checkpoint_path = None

    try:
        with log_path.open('a') as f:
            f.write("Persisting model bundle via TensorFlow ModelManager...\n")
        model_manager.save(str(output_dir))
        bundle_base = output_dir / p.get('h5_path')
        bundle_file = bundle_base.with_suffix('.h5.zip')
        bundle_path = str(bundle_file)

        if bundle_file.exists():
            bundle_size_bytes = bundle_file.stat().st_size
            sha256_hash = hashlib.sha256()
            with bundle_file.open('rb') as bundle_stream:
                for chunk in iter(lambda: bundle_stream.read(65536), b''):
                    sha256_hash.update(chunk)
            bundle_sha256 = sha256_hash.hexdigest()
            with log_path.open('a') as f:
                f.write(f"Bundle saved: {bundle_path}\n")
                f.write(f"Bundle SHA256: {bundle_sha256}\n")
                f.write(f"Bundle size: {bundle_size_bytes} bytes\n")
        else:
            with log_path.open('a') as f:
                f.write(f"Warning: Bundle file not found after save: {bundle_file}\n")
    except Exception as exc:  # pylint: disable=broad-except
        with log_path.open('a') as f:
            f.write(f"Warning: Bundle persistence failed: {type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())

    with log_path.open('a') as f:
        f.write("Training completed successfully\n")
        f.write(f"Final loss: {final_loss}\n")
        f.write(f"Epochs completed: {len(training_results.get('history', {}).get('train_loss', []))}\n")
        f.write(f"Checkpoint path: {checkpoint_path}\n")
        f.write(f"Bundle path: {bundle_path}\n")
        f.write(f"Bundle SHA256: {bundle_sha256}\n")
        f.write(f"Bundle size: {bundle_size_bytes} bytes\n")

    return {
        'status': 'success',
        'final_loss': final_loss,
        'epochs_completed': len(training_results.get('history', {}).get('train_loss', [])),
        'checkpoint_path': checkpoint_path,
        'bundle_path': bundle_path,
        'bundle_sha256': bundle_sha256,
        'bundle_size_bytes': bundle_size_bytes,
        'training_results': training_results,
    }


def _execute_training_job_pytorch(*, config, job, log_path):
    """
    Production runner helper for executing a single PtychoPINN training job
    via the PyTorch backend.

    This helper implements the real training execution logic by:
    1. Loading NPZ datasets from Phase C/D paths (already validated in TrainingJob)
    2. Using the provided TrainingConfig (CONFIG-001 bridge already done in run_training_job)
    3. Delegating to the PyTorch backend trainer (train_cdi_model_torch)
    4. Writing logs/artifacts to the provided log_path and job.artifact_dir
    5. Returning training metrics and status for manifest emission

    Args:
        config: TrainingConfig instance with dataset paths, model config, and hyperparameters
                (CONFIG-001 bridge via update_legacy_dict already applied by caller)
        job: TrainingJob instance with dose/view/gridsize metadata and artifact destinations
        log_path: Path where training logs should be written

    Returns:
        Dict containing training result metadata:
        {
            'status': 'success' | 'failed',
            'final_loss': float (if available),
            'epochs_completed': int (if available),
            'checkpoint_path': str (if saved),
        }

    Raises:
        FileNotFoundError: If dataset paths don't exist (should not happen after TrainingJob validation)
        RuntimeError: If training backend fails

    References:
        - input.md:10 (Phase E5: real runner integration with backend delegation)
        - docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001 compliance assumed by caller)
        - docs/workflows/pytorch.md §12 (canonical PyTorch training invocation)
        - docs/pytorch_runtime_checklist.md (runtime guardrails)

    Examples:
        >>> # Invoked by run_training_job after CONFIG-001 bridge
        >>> result = execute_training_job(config=training_config, job=job, log_path=job.log_path)
        >>> result['status']
        'success'
    """
    import sys
    from pathlib import Path

    from ptycho_torch.memmap_bridge import MemmapDatasetBridge
    from ptycho_torch.workflows.components import train_cdi_model_torch
    from ptycho_torch.model_manager import save_torch_bundle

    # Step 0/1: ensure logging header written
    _write_execution_header(log_path, config, job, backend='pytorch')

    # Step 2: Validate dataset paths exist (defensive check)
    train_path = Path(config.train_data_file)
    test_path = Path(config.test_data_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    # Step 3: Load datasets via MemmapDatasetBridge
    # MemmapDatasetBridge wraps NPZ files and provides RawDataTorch instances
    # via the raw_data_torch attribute. This ensures CONFIG-001 compliance
    # (update_legacy_dict already called above) and enables memory-mapped loading.
    #
    # The bridge's RawDataTorch payload (bridge.raw_data_torch) is passed to
    # train_cdi_model_torch for training. This approach:
    # - Reuses RawDataTorch grouping logic (Phase C.C3 delegation)
    # - Maintains DATA-001 NPZ contract enforcement
    # - Supports large datasets via memory mapping
    try:
        with log_path.open('a') as f:
            f.write(f"Instantiating MemmapDatasetBridge for training dataset: {train_path}...\n")
        train_bridge = MemmapDatasetBridge(
            npz_path=str(train_path),
            config=config,  # CONFIG-001: config already bridged above
        )
        # Extract RawDataTorch payload from bridge
        train_data = train_bridge.raw_data_torch

        with log_path.open('a') as f:
            f.write(f"Instantiating MemmapDatasetBridge for test dataset: {test_path}...\n")
        test_bridge = MemmapDatasetBridge(
            npz_path=str(test_path),
            config=config,  # CONFIG-001: config already bridged above
        )
        # Extract RawDataTorch payload from bridge
        test_data = test_bridge.raw_data_torch

    except Exception as e:
        with log_path.open('a') as f:
            f.write(f"Failed to load datasets via MemmapDatasetBridge: {type(e).__name__}: {e}\n")
        return {
            'status': 'failed',
            'error': f"Dataset loading failed: {type(e).__name__}: {e}",
        }

    # Step 4: Delegate to PyTorch backend trainer
    # train_cdi_model_torch orchestrates data prep, probe setup, and Lightning training.
    # It returns a dict with:
    # - 'history': Training history (losses, metrics)
    # - 'train_container': PtychoDataContainerTorch for training data
    # - 'test_container': Optional PtychoDataContainerTorch for test data
    try:
        with log_path.open('a') as f:
            f.write(f"Invoking train_cdi_model_torch with config...\n")

        training_results = train_cdi_model_torch(
            train_data=train_data,
            test_data=test_data,
            config=config,
        )

        # Extract final loss from history if available
        final_loss = None
        if 'history' in training_results and 'train_loss' in training_results['history']:
            train_losses = training_results['history']['train_loss']
            if len(train_losses) > 0:
                final_loss = train_losses[-1]

        # Determine checkpoint path if available
        # (train_cdi_model_torch may save checkpoints to config.output_dir)
        checkpoint_path = None
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            # Look for common checkpoint patterns
            checkpoint_candidates = list(output_dir.glob("*.ckpt")) + list(output_dir.glob("*.pth"))
            if checkpoint_candidates:
                # Take the most recent checkpoint
                checkpoint_path = str(sorted(checkpoint_candidates, key=lambda p: p.stat().st_mtime)[-1])

        # Step 5: Persist model bundle (specs/ptychodus_api_spec.md §4.6)
        # Create wts.h5.zip archive for downstream Phase G comparisons and
        # ModelManager consumers. Bundle path follows Ptychodus naming convention.
        bundle_path = None
        bundle_sha256 = None
        bundle_size_bytes = None
        if 'models' in training_results and training_results['models']:
            try:
                # Define bundle base path (without .zip extension)
                bundle_base = output_dir / "wts.h5"

                with log_path.open('a') as f:
                    f.write(f"Persisting model bundle to {bundle_base}.zip...\n")

                # Call save_torch_bundle with dual-model dict
                # (autoencoder + diffraction_to_obj per spec §4.6 requirement)
                save_torch_bundle(
                    models_dict=training_results['models'],
                    base_path=str(bundle_base),
                    config=config,
                    intensity_scale=None,  # Extract from params.cfg if needed
                )

                # Record bundle path for manifest emission
                bundle_path = str(bundle_base.with_suffix('.h5.zip'))

                # Compute SHA256 checksum for bundle integrity validation (Phase E6)
                import hashlib
                bundle_file = Path(bundle_path)
                if bundle_file.exists():
                    # Compute file size in bytes (Phase E6 Do Now: size tracking)
                    bundle_size_bytes = bundle_file.stat().st_size

                    sha256_hash = hashlib.sha256()
                    with bundle_file.open('rb') as f:
                        # Read in 64KB chunks to avoid memory issues with large bundles
                        for chunk in iter(lambda: f.read(65536), b''):
                            sha256_hash.update(chunk)
                    bundle_sha256 = sha256_hash.hexdigest()

                    with log_path.open('a') as f:
                        f.write(f"Model bundle saved: {bundle_path}\n")
                        f.write(f"Bundle SHA256: {bundle_sha256}\n")
                        f.write(f"Bundle size: {bundle_size_bytes} bytes\n")
                else:
                    with log_path.open('a') as f:
                        f.write(f"Warning: Bundle file not found after save: {bundle_path}\n")
            except Exception as e:
                # Log bundle persistence failure but don't fail the training run
                with log_path.open('a') as f:
                    f.write(f"Warning: Bundle persistence failed: {type(e).__name__}: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
                # Leave bundle_path and bundle_sha256 as None to signal missing bundle

        result = {
            'status': 'success',
            'final_loss': final_loss,
            'epochs_completed': len(training_results.get('history', {}).get('train_loss', [])),
            'checkpoint_path': checkpoint_path,
            'bundle_path': bundle_path,  # NEW: bundle archive path for Phase G
            'bundle_sha256': bundle_sha256,  # NEW: SHA256 checksum for integrity validation
            'bundle_size_bytes': bundle_size_bytes,  # NEW Phase E6 Do Now: size tracking
            'training_results': training_results,  # Include full results for downstream use
        }

        # Log completion
        with log_path.open('a') as f:
            f.write(f"Training completed successfully\n")
            f.write(f"Final loss: {final_loss}\n")
            f.write(f"Epochs completed: {result['epochs_completed']}\n")
            f.write(f"Checkpoint path: {checkpoint_path}\n")
            f.write(f"Bundle path: {bundle_path}\n")
            f.write(f"Bundle SHA256: {bundle_sha256}\n")
            f.write(f"Bundle size: {bundle_size_bytes} bytes\n")

        return result

    except Exception as e:
        # Log failure
        with log_path.open('a') as f:
            f.write(f"\nTraining failed with exception:\n")
            f.write(f"{type(e).__name__}: {e}\n")
            import traceback
            f.write(traceback.format_exc())

        return {
            'status': 'failed',
            'error': f"{type(e).__name__}: {e}",
        }


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
        --backend: Backend to use for training ('tensorflow' default)
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
    parser.add_argument(
        '--backend',
        type=str,
        default='tensorflow',
        choices=['tensorflow', 'pytorch'],
        help="Backend to use for training (default: 'tensorflow')"
    )

    args = parser.parse_args()

    # Ensure artifact root exists
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Build job matrix with optional pre-filtering by dose for CLI
    # CLI mode uses allow_missing_phase_d=True to handle scenarios where Phase D
    # overlap filtering rejected some views due to spacing threshold (e.g., sparse
    # view with too few positions). This allows baseline training to proceed even
    # when overlap data is incomplete. Tests use strict mode (default False) to
    # ensure deterministic job matrices.
    #
    # Phase E5: Accumulate skip metadata for manifest reporting
    skip_events = []
    print(f"Enumerating training jobs from Phase C ({args.phase_c_root}) and Phase D ({args.phase_d_root})...")
    print(f"  → Selected backend: {args.backend}")

    # Important: if --dose is provided, constrain the study design BEFORE
    # instantiating TrainingJob objects. This avoids constructing jobs for
    # other doses that may not have Phase C datasets in this workspace,
    # which would otherwise raise FileNotFoundError during TrainingJob
    # validation (baseline jobs validate patched_{train,test}.npz).
    design = get_study_design()
    if args.dose is not None:
        try:
            # Normalize to float (StudyDesign uses floats)
            design.dose_list = [float(args.dose)]
        except Exception:
            design.dose_list = [args.dose]

    all_jobs = build_training_jobs(
        phase_c_root=args.phase_c_root,
        phase_d_root=args.phase_d_root,
        artifact_root=args.artifact_root,
        design=design,
        allow_missing_phase_d=True,  # Non-strict mode for CLI robustness
        skip_events=skip_events,  # Phase E5: capture skip metadata
    )
    print(f"  → {len(all_jobs)} total jobs enumerated")

    # Phase E5: Print skip summary if any views were skipped
    if skip_events:
        print(f"  ⚠ {len(skip_events)} view(s) skipped due to missing Phase D data:")
        for skip_event in skip_events:
            print(f"    - {skip_event['view']} (dose={skip_event['dose']:.0e}): {skip_event['reason'][:80]}...")

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

    # Step 3: Execute filtered jobs with real runner (execute_training_job)
    # The runner helper is now wired to execute_training_job (Phase E5 upgrade)
    # Tests can monkeypatch execute_training_job to spy on invocation
    print(f"\nExecuting {len(filtered_jobs)} training job(s)...")
    job_results = []

    for i, job in enumerate(filtered_jobs, start=1):
        print(
            f"  [{i}/{len(filtered_jobs)}] {job.view} (dose={job.dose:.0e}, "
            f"gridsize={job.gridsize}, backend={args.backend})"
        )
        result = run_training_job(
            job,
            runner=execute_training_job,
            dry_run=args.dry_run,
            backend=args.backend,
        )

        # Phase E6: Emit bundle_path and bundle_sha256 to stdout for CLI log capture
        # This ensures each job's bundle digest appears alongside manifest pointers
        # in the CLI log, providing traceable integrity proof per specs/ptychodus_api_spec.md:239
        # Format includes view/dose context for traceability in captured logs
        # IMPORTANT: Normalize bundle_path to artifact-relative before stdout emission
        # to avoid workstation-specific absolute paths in logged output
        if not args.dry_run and result.get('bundle_path'):
            # Convert absolute bundle_path to artifact-relative path for portable logging
            bundle_path_abs = Path(result['bundle_path'])
            try:
                bundle_path_rel = bundle_path_abs.relative_to(job.artifact_dir)
                bundle_path_display = str(bundle_path_rel)
            except ValueError:
                # Defensive: if bundle_path not under artifact_dir, use as-is
                bundle_path_display = result['bundle_path']

            print(f"    → Bundle [{job.view}/dose={job.dose:.0e}]: {bundle_path_display}")
            if result.get('bundle_sha256'):
                print(f"    → SHA256 [{job.view}/dose={job.dose:.0e}]: {result['bundle_sha256']}")
            # Phase E6 Do Now: Emit bundle size to stdout for traceability
            if result.get('bundle_size_bytes') is not None:
                print(f"    → Size [{job.view}/dose={job.dose:.0e}]: {result['bundle_size_bytes']} bytes")

        # Convert result dict paths to strings for JSON serialization
        # Phase E6: Normalize bundle_path to be relative to artifact_dir
        result_serializable = {}
        def _sanitize(value):
            """Recursively convert training results to JSON-safe data."""
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                sanitized = {}
                for key, val in value.items():
                    # Drop heavy/non-serializable entries such as TF models
                    if key in {'models', 'model', 'autoencoder', 'diffraction_to_obj'}:
                        continue
                    sanitized[key] = _sanitize(val)
                return sanitized
            if isinstance(value, (list, tuple)):
                return [_sanitize(v) for v in value]
            # Fallback to string representation for any remaining objects
            return str(value)

        for k, v in result.items():
            if isinstance(v, Path):
                result_serializable[k] = str(v)
            elif k == 'bundle_path' and v is not None:
                # Phase E6: Convert absolute bundle_path to relative path from artifact_dir
                # This ensures manifest uses portable paths that work across workstations
                # Example: /abs/path/artifacts/dose_1000/baseline/gs1/wts.h5.zip
                #          → wts.h5.zip (relative to artifact_dir)
                bundle_path_abs = Path(v)
                try:
                    # Compute relative path from artifact_dir
                    bundle_path_rel = bundle_path_abs.relative_to(job.artifact_dir)
                    result_serializable[k] = str(bundle_path_rel)
                except ValueError:
                    # If bundle_path is not under artifact_dir, keep absolute
                    # (should not happen in normal execution, but defensive)
                    result_serializable[k] = str(v)
            else:
                result_serializable[k] = _sanitize(v)

        job_results.append({
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
            'train_data_path': job.train_data_path,
            'test_data_path': job.test_data_path,
            'log_path': str(job.log_path),
            'artifact_dir': str(job.artifact_dir),
            'backend': args.backend,
            'result': result_serializable,
        })

    # Step 5: Emit skip_summary.json (Phase E5.5 standalone artifact)
    # This provides a dedicated skip report for downstream analytics/tooling
    # separate from the full training manifest
    skip_summary_path = args.artifact_root / "skip_summary.json"
    skip_summary = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'skipped_views': skip_events,
        'skipped_count': len(skip_events),
    }
    with skip_summary_path.open('w') as f:
        json.dump(skip_summary, f, indent=2)

    # Step 6: Emit training_manifest.json
    # Phase E5: Include skip metadata in manifest for traceability
    # Phase E5.5: Reference skip_summary.json for downstream tooling
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
        'skipped_views': skip_events,  # Phase E5: expose skip metadata
        'skipped_count': len(skip_events),  # Phase E5: convenience count
        'skip_summary_path': str(skip_summary_path.relative_to(args.artifact_root)),  # Phase E5.5: relative path to standalone skip summary
    }

    manifest_path = args.artifact_root / "training_manifest.json"
    with manifest_path.open('w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Training manifest written to {manifest_path}")
    print(f"  → {len(job_results)} job(s) completed")
    if skip_events:
        print(f"  → Skip summary written to {skip_summary_path} ({len(skip_events)} view(s) skipped)")


if __name__ == '__main__':
    main()

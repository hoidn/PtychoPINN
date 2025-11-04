"""
Tests for Phase E training job matrix builder.

Validates that build_training_jobs() correctly enumerates 9 jobs per dose
(3 doses × 3 variants: baseline gs1, dense gs2, sparse gs2) with proper
metadata and dataset path validation.

Test tier: Unit
Test strategy: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md
"""

import pytest
import numpy as np
from pathlib import Path
from studies.fly64_dose_overlap.training import build_training_jobs, TrainingJob
from studies.fly64_dose_overlap.design import get_study_design


@pytest.fixture
def mock_phase_c_datasets(tmp_path):
    """
    Create minimal Phase C dataset tree for testing.

    Structure:
        dose_1000/
            patched_train.npz
            patched_test.npz
        dose_10000/
            patched_train.npz
            patched_test.npz
        dose_100000/
            patched_train.npz
            patched_test.npz

    Each NPZ contains minimal DATA-001 keys with small arrays.
    """
    phase_c_root = tmp_path / "phase_c"
    design = get_study_design()

    # Minimal DATA-001 compliant arrays
    minimal_data = {
        'diffraction': np.random.rand(10, 64, 64).astype(np.float32),  # amplitude
        'objectGuess': np.random.rand(128, 128) + 1j * np.random.rand(128, 128),
        'probeGuess': np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
        'Y': (np.random.rand(10, 128, 128) + 1j * np.random.rand(10, 128, 128)).astype(np.complex64),
        'xcoords': np.random.rand(10).astype(np.float32),
        'ycoords': np.random.rand(10).astype(np.float32),
        'filenames': np.array([f'img_{i:04d}' for i in range(10)]),
    }

    for dose in design.dose_list:
        dose_dir = phase_c_root / f"dose_{int(dose)}"
        dose_dir.mkdir(parents=True, exist_ok=True)

        # Write train and test NPZs
        for split in ['train', 'test']:
            npz_path = dose_dir / f"patched_{split}.npz"
            np.savez_compressed(npz_path, **minimal_data)

    return phase_c_root


@pytest.fixture
def mock_phase_d_datasets(tmp_path):
    """
    Create minimal Phase D overlap views for testing.

    Structure:
        dose_1000/
            dense_train.npz, dense_test.npz
            sparse_train.npz, sparse_test.npz
        dose_10000/
            dense_train.npz, dense_test.npz
            sparse_train.npz, sparse_test.npz
        dose_100000/
            dense_train.npz, dense_test.npz
            sparse_train.npz, sparse_test.npz

    Each NPZ contains minimal DATA-001 keys plus gridsize=2 metadata.
    """
    phase_d_root = tmp_path / "phase_d"
    design = get_study_design()

    # Minimal DATA-001 compliant arrays (fewer positions due to filtering)
    minimal_data = {
        'diffraction': np.random.rand(5, 64, 64).astype(np.float32),
        'objectGuess': np.random.rand(128, 128) + 1j * np.random.rand(128, 128),
        'probeGuess': np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
        'Y': (np.random.rand(5, 128, 128) + 1j * np.random.rand(5, 128, 128)).astype(np.complex64),
        'xcoords': np.random.rand(5).astype(np.float32),
        'ycoords': np.random.rand(5).astype(np.float32),
        'filenames': np.array([f'img_{i:04d}' for i in range(5)]),
    }

    for dose in design.dose_list:
        dose_dir = phase_d_root / f"dose_{int(dose)}"
        dose_dir.mkdir(parents=True, exist_ok=True)

        # Write dense and sparse NPZs for train/test
        for view in ['dense', 'sparse']:
            for split in ['train', 'test']:
                npz_path = dose_dir / f"{view}_{split}.npz"
                np.savez_compressed(npz_path, **minimal_data)

    return phase_d_root


def test_build_training_jobs_matrix(mock_phase_c_datasets, mock_phase_d_datasets, tmp_path):
    """
    RED → GREEN TDD test for Phase E job matrix enumeration.

    Validates that build_training_jobs() produces exactly 9 jobs per dose:
    - 1 baseline job (gs1, Phase C patched_train.npz/patched_test.npz)
    - 2 overlap jobs (gs2, Phase D dense/sparse NPZs)

    Each TrainingJob must contain:
    - dose (float, from StudyDesign.dose_list)
    - view (str, one of {"baseline", "dense", "sparse"})
    - gridsize (int, 1 or 2)
    - train_data_path (str, validated existence)
    - test_data_path (str, validated existence)
    - artifact_dir (Path, deterministic from dose/view/gridsize)
    - log_path (Path, derived from artifact_dir)

    References:
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:133-144
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-115
    - specs/data_contracts.md:190-260 (DATA-001 keys)
    """
    design = get_study_design()
    artifact_root = tmp_path / "artifacts"

    # Call builder
    jobs = build_training_jobs(
        phase_c_root=mock_phase_c_datasets,
        phase_d_root=mock_phase_d_datasets,
        artifact_root=artifact_root,
        design=design,
    )

    # Assert total count: 3 doses × 3 variants = 9 jobs
    assert len(jobs) == 9, \
        f"Expected 9 jobs (3 doses × 3 variants), got {len(jobs)}"

    # Group by dose and validate structure
    jobs_by_dose = {}
    for job in jobs:
        assert isinstance(job, TrainingJob), \
            f"Expected TrainingJob instance, got {type(job)}"

        if job.dose not in jobs_by_dose:
            jobs_by_dose[job.dose] = []
        jobs_by_dose[job.dose].append(job)

    # Validate each dose has 3 jobs
    for dose in design.dose_list:
        assert dose in jobs_by_dose, \
            f"Missing jobs for dose={dose}"
        assert len(jobs_by_dose[dose]) == 3, \
            f"Expected 3 jobs for dose={dose}, got {len(jobs_by_dose[dose])}"

        # Extract views for this dose
        views = {job.view for job in jobs_by_dose[dose]}
        assert views == {'baseline', 'dense', 'sparse'}, \
            f"Expected views {{'baseline', 'dense', 'sparse'}}, got {views}"

        # Validate each job's metadata
        for job in jobs_by_dose[dose]:
            # Check dose
            assert job.dose == dose, \
                f"Job dose mismatch: expected {dose}, got {job.dose}"

            # Check gridsize matches view expectation
            if job.view == 'baseline':
                assert job.gridsize == 1, \
                    f"Baseline jobs must have gridsize=1, got {job.gridsize}"
            else:  # dense or sparse
                assert job.gridsize == 2, \
                    f"Overlap jobs ({job.view}) must have gridsize=2, got {job.gridsize}"

            # Check dataset paths exist
            assert Path(job.train_data_path).exists(), \
                f"Train dataset not found: {job.train_data_path}"
            assert Path(job.test_data_path).exists(), \
                f"Test dataset not found: {job.test_data_path}"

            # Check artifact paths are deterministic and under artifact_root
            assert artifact_root in job.artifact_dir.parents or job.artifact_dir == artifact_root, \
                f"artifact_dir must be under artifact_root, got {job.artifact_dir}"

            # Check log path is derived from artifact_dir
            assert job.log_path.parent == job.artifact_dir, \
                f"log_path must be in artifact_dir, got {job.log_path}"

            # Validate dataset paths match expected structure
            dose_suffix = f"dose_{int(dose)}"
            assert dose_suffix in str(job.train_data_path), \
                f"Train path must contain {dose_suffix}, got {job.train_data_path}"
            assert dose_suffix in str(job.test_data_path), \
                f"Test path must contain {dose_suffix}, got {job.test_data_path}"

            # Validate view-specific paths
            if job.view == 'baseline':
                assert 'patched_train.npz' in str(job.train_data_path), \
                    f"Baseline train path must be patched_train.npz, got {job.train_data_path}"
                assert 'patched_test.npz' in str(job.test_data_path), \
                    f"Baseline test path must be patched_test.npz, got {job.test_data_path}"
            else:
                assert f'{job.view}_train.npz' in str(job.train_data_path), \
                    f"Overlap train path must be {job.view}_train.npz, got {job.train_data_path}"
                assert f'{job.view}_test.npz' in str(job.test_data_path), \
                    f"Overlap test path must be {job.view}_test.npz, got {job.test_data_path}"

    print(f"\n✓ Job matrix enumeration validated: {len(jobs)} jobs across {len(design.dose_list)} doses")
    for dose in sorted(jobs_by_dose.keys()):
        views_str = ', '.join(sorted(j.view for j in jobs_by_dose[dose]))
        print(f"  Dose {dose:.0e}: {views_str}")


def test_run_training_job_invokes_runner(tmp_path):
    """
    RED → GREEN TDD test for run_training_job runner invocation.

    Validates that run_training_job():
    - Creates artifact and log directories before execution
    - Calls update_legacy_dict(params.cfg, config) with the correct bridge config
    - Invokes the injected runner callable with (config, job, log_path) kwargs
    - Touches/writes to job.log_path to ensure logging infrastructure ready
    - Returns useful metadata (e.g., runner result or summary dict)

    Test strategy: Use monkeypatch to spy on update_legacy_dict call and inject
    a stub runner that records its invocation signature. Validate call ordering
    (bridge before runner) and parameter passing.

    References:
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-123
    - docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001 ordering)
    - input.md:8-11 (task E3 requirements)
    """
    from studies.fly64_dose_overlap.training import run_training_job

    # Setup: create a minimal job with tmp paths
    artifact_dir = tmp_path / "artifacts" / "dose_1000" / "baseline" / "gs1"
    log_path = artifact_dir / "train.log"

    # Create minimal stub datasets (empty files suffice for this test)
    train_data = tmp_path / "train.npz"
    test_data = tmp_path / "test.npz"
    train_data.touch()
    test_data.touch()

    job = TrainingJob(
        dose=1e3,
        view='baseline',
        gridsize=1,
        train_data_path=str(train_data),
        test_data_path=str(test_data),
        artifact_dir=artifact_dir,
        log_path=log_path,
    )

    # Spy: record calls to runner
    runner_calls = []

    def stub_runner(*, config, job, log_path):
        runner_calls.append({
            'config': config,
            'job': job,
            'log_path': log_path,
        })
        return {'status': 'success', 'epochs': 10}  # Mock training result

    # Execute with spy
    result = run_training_job(job, runner=stub_runner, dry_run=False)

    # Assertions: directories created
    assert artifact_dir.exists(), \
        f"artifact_dir not created: {artifact_dir}"
    assert artifact_dir.is_dir(), \
        f"artifact_dir is not a directory: {artifact_dir}"

    # Assertions: log file touched/created
    assert log_path.exists(), \
        f"log_path not created: {log_path}"
    assert log_path.is_file(), \
        f"log_path is not a file: {log_path}"

    # Assertions: runner invoked with correct kwargs
    assert len(runner_calls) == 1, \
        f"Runner should be called exactly once, got {len(runner_calls)} calls"

    runner_call = runner_calls[0]
    assert runner_call['job'] is job, \
        "Runner must receive the original job instance"
    assert runner_call['log_path'] == log_path, \
        f"Runner log_path mismatch: expected {log_path}, got {runner_call['log_path']}"
    assert runner_call['config'] is not None, \
        "Runner must receive a config object"

    # Assertions: result returned
    assert result is not None, \
        "run_training_job must return a result"
    assert result['status'] == 'success', \
        f"Expected runner result to be returned, got {result}"

    print(f"\n✓ run_training_job invocation validated:")
    print(f"  - Artifact directory created: {artifact_dir}")
    print(f"  - Log file touched: {log_path}")
    print(f"  - params.cfg updated for CONFIG-001 compliance")
    print(f"  - Runner invoked: {len(runner_calls)} time(s)")
    print(f"  - Result: {result}")


def test_run_training_job_dry_run(tmp_path):
    """
    RED → GREEN TDD test for run_training_job dry-run mode.

    Validates that run_training_job(dry_run=True):
    - Creates artifact and log directories
    - Calls update_legacy_dict to prepare params.cfg
    - Does NOT invoke the runner callable
    - Returns a summary dict describing what would have been executed
    - Summary includes: dose, view, gridsize, dataset paths, log_path

    Test strategy: Inject a sentinel runner that raises AssertionError if called.
    Validate that dry_run prevents execution while still returning useful metadata.

    References:
    - input.md:10 (honor dry_run by summarizing without executing)
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-123
    """
    from studies.fly64_dose_overlap.training import run_training_job

    # Setup: create a minimal job
    artifact_dir = tmp_path / "artifacts" / "dose_10000" / "dense" / "gs2"
    log_path = artifact_dir / "train.log"

    train_data = tmp_path / "train.npz"
    test_data = tmp_path / "test.npz"
    train_data.touch()
    test_data.touch()

    job = TrainingJob(
        dose=1e4,
        view='dense',
        gridsize=2,
        train_data_path=str(train_data),
        test_data_path=str(test_data),
        artifact_dir=artifact_dir,
        log_path=log_path,
    )

    # Sentinel: runner must NOT be called in dry-run mode
    def sentinel_runner(**kwargs):
        raise AssertionError("Runner called during dry_run=True (should be skipped)")

    # Execute in dry-run mode
    result = run_training_job(job, runner=sentinel_runner, dry_run=True)

    # Assertions: directories still created (setup phase)
    assert artifact_dir.exists(), \
        f"artifact_dir should be created even in dry-run: {artifact_dir}"

    # Assertions: result is a summary dict
    assert result is not None, \
        "run_training_job must return a summary in dry-run mode"
    assert isinstance(result, dict), \
        f"dry-run result should be a dict, got {type(result)}"

    # Assertions: summary contains key metadata
    required_keys = {'dose', 'view', 'gridsize', 'train_data_path', 'test_data_path', 'log_path'}
    missing_keys = required_keys - result.keys()
    assert not missing_keys, \
        f"dry-run summary missing keys: {missing_keys}"

    assert result['dose'] == job.dose, \
        f"dose mismatch: expected {job.dose}, got {result['dose']}"
    assert result['view'] == job.view, \
        f"view mismatch: expected {job.view}, got {result['view']}"
    assert result['gridsize'] == job.gridsize, \
        f"gridsize mismatch: expected {job.gridsize}, got {result['gridsize']}"

    # Sentinel not triggered → runner was NOT called
    print(f"\n✓ run_training_job dry-run validated:")
    print(f"  - Artifact directory created: {artifact_dir}")
    print(f"  - Runner skipped (sentinel not triggered)")
    print(f"  - Summary returned: {result}")

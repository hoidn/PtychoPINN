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


def test_run_training_job_invokes_runner(tmp_path, monkeypatch):
    """
    RED → GREEN TDD test for run_training_job runner invocation.

    Validates that run_training_job():
    - Creates artifact and log directories before execution
    - Constructs a TrainingConfig dataclass with job metadata
    - Calls update_legacy_dict(params.cfg, config) with the TrainingConfig instance
    - Invokes the injected runner callable with (config, job, log_path) kwargs
    - Touches/writes to job.log_path to ensure logging infrastructure ready
    - Returns useful metadata (e.g., runner result or summary dict)

    Test strategy: Use monkeypatch to spy on update_legacy_dict call and inject
    a stub runner that records its invocation signature. Validate call ordering
    (bridge before runner) and parameter passing.

    References:
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-123
    - docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001 ordering)
    - input.md:8-11 (task E4 tightened requirements: assert TrainingConfig)
    """
    from studies.fly64_dose_overlap.training import run_training_job
    from ptycho.config.config import TrainingConfig

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

    # Spy: record calls to update_legacy_dict
    bridge_calls = []
    original_update_legacy_dict = __import__('ptycho.config.config', fromlist=['update_legacy_dict']).update_legacy_dict

    def spy_update_legacy_dict(cfg_dict, config):
        bridge_calls.append({
            'cfg_dict': cfg_dict,
            'config': config,
        })
        # Call original to maintain CONFIG-001 behavior
        return original_update_legacy_dict(cfg_dict, config)

    monkeypatch.setattr('studies.fly64_dose_overlap.training.update_legacy_dict', spy_update_legacy_dict)

    # Spy: record calls to runner
    runner_calls = []

    def stub_runner(*, config, job, log_path):
        runner_calls.append({
            'config': config,
            'job': job,
            'log_path': log_path,
        })
        return {'status': 'success', 'epochs': 10}  # Mock training result

    # Execute with spies
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

    # Assertions: update_legacy_dict called with TrainingConfig
    assert len(bridge_calls) == 1, \
        f"update_legacy_dict should be called exactly once, got {len(bridge_calls)} calls"

    bridge_call = bridge_calls[0]
    assert isinstance(bridge_call['config'], TrainingConfig), \
        f"update_legacy_dict must receive TrainingConfig instance, got {type(bridge_call['config'])}"

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
    print(f"  - update_legacy_dict called with TrainingConfig (CONFIG-001 compliance)")
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


def test_training_cli_filters_jobs(tmp_path, monkeypatch):
    """
    RED → GREEN TDD test for training CLI job filtering.

    Validates that the training CLI main() function:
    - Parses command-line arguments (--phase-c-root, --phase-d-root, --artifact-root)
    - Accepts optional filter flags (--dose, --view, --gridsize)
    - Calls build_training_jobs() to enumerate full job matrix
    - Filters jobs based on provided CLI flags
    - Invokes run_training_job() only for matching jobs
    - Handles gracefully when filters match nothing (informative error)

    Test strategy: Mock sys.argv to inject CLI arguments, monkeypatch build_training_jobs
    and run_training_job to return predictable stubs, validate filtering logic.

    References:
    - input.md:10 (CLI filtering requirements for Phase E4)
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-115
    """
    import sys
    import json
    from studies.fly64_dose_overlap import training

    # Setup: Create mock Phase C and Phase D directories
    phase_c_root = tmp_path / "phase_c"
    phase_d_root = tmp_path / "phase_d"
    artifact_root = tmp_path / "artifacts"

    phase_c_root.mkdir()
    phase_d_root.mkdir()

    # Create minimal dataset structure (empty files suffice for CLI test)
    for dose in [1000, 10000, 100000]:
        dose_dir_c = phase_c_root / f"dose_{dose}"
        dose_dir_d = phase_d_root / f"dose_{dose}"
        dose_dir_c.mkdir()
        dose_dir_d.mkdir()

        # Phase C patched datasets
        (dose_dir_c / "patched_train.npz").touch()
        (dose_dir_c / "patched_test.npz").touch()

        # Phase D overlap views
        for view in ['dense', 'sparse']:
            (dose_dir_d / f"{view}_train.npz").touch()
            (dose_dir_d / f"{view}_test.npz").touch()

    # Spy: Track which jobs were executed
    executed_jobs = []

    def mock_run_training_job(job, runner, dry_run=False):
        executed_jobs.append({
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
            'dry_run': dry_run,
        })
        return {'status': 'mock_success'}

    monkeypatch.setattr(training, 'run_training_job', mock_run_training_job)

    # Test case 1: No filters (all 9 jobs executed)
    executed_jobs.clear()
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--dry-run',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    assert len(executed_jobs) == 9, \
        f"Expected 9 jobs without filters, got {len(executed_jobs)}"

    # Test case 2: Filter by dose (3 jobs: baseline + dense + sparse for dose=1e3)
    executed_jobs.clear()
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--dose', '1000',
        '--dry-run',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    assert len(executed_jobs) == 3, \
        f"Expected 3 jobs for dose=1000, got {len(executed_jobs)}"
    assert all(j['dose'] == 1e3 for j in executed_jobs), \
        "All executed jobs must have dose=1e3"

    # Test case 3: Filter by view (3 jobs: baseline across all doses)
    executed_jobs.clear()
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--view', 'baseline',
        '--dry-run',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    assert len(executed_jobs) == 3, \
        f"Expected 3 baseline jobs, got {len(executed_jobs)}"
    assert all(j['view'] == 'baseline' for j in executed_jobs), \
        "All executed jobs must have view=baseline"

    # Test case 4: Filter by gridsize (6 jobs: dense + sparse across all doses)
    executed_jobs.clear()
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--gridsize', '2',
        '--dry-run',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    assert len(executed_jobs) == 6, \
        f"Expected 6 jobs for gridsize=2, got {len(executed_jobs)}"
    assert all(j['gridsize'] == 2 for j in executed_jobs), \
        "All executed jobs must have gridsize=2"

    # Test case 5: Combined filters (1 job: dense view, dose=1e4, gridsize=2)
    executed_jobs.clear()
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--dose', '10000',
        '--view', 'dense',
        '--gridsize', '2',
        '--dry-run',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    assert len(executed_jobs) == 1, \
        f"Expected 1 job for dose=1e4, view=dense, gridsize=2, got {len(executed_jobs)}"
    assert executed_jobs[0]['dose'] == 1e4, "Job dose must be 1e4"
    assert executed_jobs[0]['view'] == 'dense', "Job view must be dense"
    assert executed_jobs[0]['gridsize'] == 2, "Job gridsize must be 2"

    print(f"\n✓ CLI job filtering validated:")
    print(f"  - No filters: 9 jobs executed")
    print(f"  - Filter by dose: 3 jobs")
    print(f"  - Filter by view: 3 jobs")
    print(f"  - Filter by gridsize: 6 jobs")
    print(f"  - Combined filters: 1 job")


def test_training_cli_manifest_and_bridging(tmp_path, monkeypatch):
    """
    RED → GREEN TDD test for training CLI manifest emission and CONFIG-001 bridging.

    Validates that the training CLI main() function:
    - Emits a training_manifest.json file under --artifact-root
    - Manifest contains job metadata (dose, view, gridsize, dataset paths, log paths)
    - Ensures run_training_job is called with proper CONFIG-001 bridge
    - Writes CLI stdout/stderr log to --artifact-root for traceability

    Test strategy: Mock run_training_job to verify it's called (CONFIG-001 handled internally),
    execute CLI with --dry-run, validate manifest JSON structure and content.

    References:
    - input.md:10 (manifest emission requirement for Phase E4)
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-115
    """
    import sys
    import json
    from studies.fly64_dose_overlap import training

    # Setup: Create mock Phase C and Phase D directories
    phase_c_root = tmp_path / "phase_c"
    phase_d_root = tmp_path / "phase_d"
    artifact_root = tmp_path / "artifacts"

    phase_c_root.mkdir()
    phase_d_root.mkdir()

    # Create minimal dataset structure for all doses (to match StudyDesign defaults)
    for dose in [1000, 10000, 100000]:
        dose_dir_c = phase_c_root / f"dose_{dose}"
        dose_dir_d = phase_d_root / f"dose_{dose}"
        dose_dir_c.mkdir()
        dose_dir_d.mkdir()

        # Phase C patched datasets
        (dose_dir_c / "patched_train.npz").touch()
        (dose_dir_c / "patched_test.npz").touch()

        # Phase D overlap views
        for view in ['dense', 'sparse']:
            (dose_dir_d / f"{view}_train.npz").touch()
            (dose_dir_d / f"{view}_test.npz").touch()

    # Spy: Track run_training_job calls
    run_calls = []

    def mock_run_training_job(job, runner, dry_run=False):
        run_calls.append({
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
        })
        return {'status': 'mock_success'}

    monkeypatch.setattr(training, 'run_training_job', mock_run_training_job)

    # Execute CLI with dose filter to reduce output
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--dose', '1000',
        '--dry-run',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    # Assertions: manifest file created
    manifest_path = artifact_root / "training_manifest.json"
    assert manifest_path.exists(), \
        f"training_manifest.json not found at {manifest_path}"

    # Assertions: manifest content is valid JSON
    with manifest_path.open('r') as f:
        manifest = json.load(f)

    assert isinstance(manifest, dict), \
        f"Manifest must be a dict, got {type(manifest)}"

    # Assertions: manifest contains expected keys
    required_keys = {'timestamp', 'phase_c_root', 'phase_d_root', 'artifact_root', 'jobs'}
    missing_keys = required_keys - manifest.keys()
    assert not missing_keys, \
        f"Manifest missing keys: {missing_keys}"

    # Assertions: jobs list matches executed jobs (3 for dose=1000)
    assert isinstance(manifest['jobs'], list), \
        f"Manifest 'jobs' must be a list, got {type(manifest['jobs'])}"
    assert len(manifest['jobs']) == 3, \
        f"Expected 3 jobs in manifest for dose=1000, got {len(manifest['jobs'])}"

    # Assertions: each job entry contains required metadata
    for job_entry in manifest['jobs']:
        required_job_keys = {'dose', 'view', 'gridsize', 'train_data_path', 'test_data_path', 'log_path'}
        missing_job_keys = required_job_keys - job_entry.keys()
        assert not missing_job_keys, \
            f"Job entry missing keys: {missing_job_keys}"

        assert job_entry['dose'] == 1000.0, \
            f"Job dose must be 1000.0, got {job_entry['dose']}"
        assert job_entry['view'] in {'baseline', 'dense', 'sparse'}, \
            f"Job view must be one of baseline/dense/sparse, got {job_entry['view']}"

    # Assertions: run_training_job was called for each job
    assert len(run_calls) == 3, \
        f"Expected run_training_job called 3 times, got {len(run_calls)}"

    print(f"\n✓ CLI manifest and bridging validated:")
    print(f"  - training_manifest.json created at {manifest_path}")
    print(f"  - Manifest contains {len(manifest['jobs'])} job entries")
    print(f"  - run_training_job called {len(run_calls)} times (CONFIG-001 bridging implicit)")
    print(f"  - Each job entry has required metadata keys")


def test_execute_training_job_delegates_to_pytorch_trainer(tmp_path, monkeypatch):
    """
    RED → GREEN TDD test for Phase E5 execute_training_job backend delegation.

    Validates that execute_training_job():
    - Loads NPZ datasets from job.train_data_path and job.test_data_path
    - Uses the provided TrainingConfig (CONFIG-001 bridge already done by caller)
    - Calls ptycho_torch.workflows.components.train_cdi_model_torch with:
      - train_data: RawData or container loaded from job.train_data_path
      - test_data: RawData or container loaded from job.test_data_path
      - config: The TrainingConfig instance passed to execute_training_job
    - Writes logs/artifacts to job.log_path and job.artifact_dir
    - Returns training metrics (status, final_loss, epochs_completed, checkpoint_path)

    Test strategy: Monkeypatch train_cdi_model_torch to spy on invocation.
    Use minimal Phase C/D fixture NPZs (DATA-001 compliant) to avoid heavy I/O.
    Validate that the spy receives correct data containers and config.

    References:
    - input.md:9 (Phase E5 RED test requirements)
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163-166
    - docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001 compliance assumed by caller)
    - docs/workflows/pytorch.md §12 (train_cdi_model_torch signature)
    """
    from studies.fly64_dose_overlap.training import execute_training_job, TrainingJob
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Setup: Create minimal Phase C fixture NPZ (DATA-001 compliant)
    train_npz = tmp_path / "phase_c_train.npz"
    test_npz = tmp_path / "phase_c_test.npz"

    # Minimal DATA-001 compliant arrays plus xcoords_start/ycoords_start for load_data
    minimal_data = {
        'diffraction': np.random.rand(10, 64, 64).astype(np.float32),  # amplitude
        'objectGuess': np.random.rand(128, 128) + 1j * np.random.rand(128, 128),
        'probeGuess': np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
        'Y': (np.random.rand(10, 128, 128) + 1j * np.random.rand(10, 128, 128)).astype(np.complex64),
        'xcoords': np.random.rand(10).astype(np.float32),
        'ycoords': np.random.rand(10).astype(np.float32),
        'xcoords_start': np.random.rand(10).astype(np.float32),
        'ycoords_start': np.random.rand(10).astype(np.float32),
        'filenames': np.array([f'img_{i:04d}' for i in range(10)]),
    }

    np.savez_compressed(train_npz, **minimal_data)
    np.savez_compressed(test_npz, **minimal_data)

    # Setup: Create TrainingJob
    artifact_dir = tmp_path / "artifacts" / "dose_1000" / "baseline" / "gs1"
    log_path = artifact_dir / "train.log"

    job = TrainingJob(
        dose=1e3,
        view='baseline',
        gridsize=1,
        train_data_path=str(train_npz),
        test_data_path=str(test_npz),
        artifact_dir=artifact_dir,
        log_path=log_path,
    )

    # Setup: Create TrainingConfig (CONFIG-001 bridge assumed done by caller: run_training_job)
    model_config = ModelConfig(gridsize=1)
    config = TrainingConfig(
        train_data_file=str(train_npz),
        test_data_file=str(test_npz),
        output_dir=str(artifact_dir),
        model=model_config,
        nphotons=1e3,
    )

    # Spy: record calls to train_cdi_model_torch
    trainer_calls = []

    def spy_train_cdi_model_torch(train_data, test_data, config):
        """Spy that records invocation signature for validation."""
        trainer_calls.append({
            'train_data': train_data,
            'test_data': test_data,
            'config': config,
        })
        # Return minimal success result
        return {
            'history': {'train_loss': [0.5, 0.3, 0.1]},
            'train_container': train_data,
            'test_container': test_data,
        }

    # Monkeypatch load_data to return mock RawData objects
    # This avoids needing to create perfectly formatted NPZ fixtures
    from ptycho.raw_data import RawData

    def mock_load_data(file_path):
        """Return a minimal RawData instance for testing."""
        return RawData(
            xcoords=np.array([0.0]),
            ycoords=np.array([0.0]),
            xcoords_start=np.array([0.0]),
            ycoords_start=np.array([0.0]),
            diff3d=np.random.rand(1, 64, 64).astype(np.float32),
            probeGuess=np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
            scan_index=np.array([0]),
        )

    # Monkeypatch the PyTorch trainer in the training module's namespace
    # The training module imports it at module level: from ptycho_torch.workflows.components import train_cdi_model_torch
    # So we need to patch the reference in studies.fly64_dose_overlap.training
    from studies.fly64_dose_overlap import training as training_module
    monkeypatch.setattr(training_module, 'load_data', mock_load_data)
    monkeypatch.setattr(training_module, 'train_cdi_model_torch', spy_train_cdi_model_torch)

    # Execute: call execute_training_job
    result = execute_training_job(config=config, job=job, log_path=log_path)

    # Debug: print result to see if there was an error
    print(f"\nResult from execute_training_job: {result}")
    if log_path.exists():
        print(f"Log contents:\n{log_path.read_text()}")

    # Assertions: train_cdi_model_torch was called
    assert len(trainer_calls) == 1, \
        f"train_cdi_model_torch should be called exactly once, got {len(trainer_calls)} calls"

    call = trainer_calls[0]

    # Assertions: train_data is not None (RawData or container)
    assert call['train_data'] is not None, \
        "train_cdi_model_torch must receive train_data"

    # Assertions: test_data is not None (RawData or container)
    assert call['test_data'] is not None, \
        "train_cdi_model_torch must receive test_data"

    # Assertions: config is the TrainingConfig instance
    assert call['config'] is config, \
        "train_cdi_model_torch must receive the same TrainingConfig instance"

    # Assertions: config has correct fields
    assert call['config'].train_data_file == str(train_npz), \
        f"config.train_data_file mismatch: expected {train_npz}, got {call['config'].train_data_file}"
    assert call['config'].test_data_file == str(test_npz), \
        f"config.test_data_file mismatch: expected {test_npz}, got {call['config'].test_data_file}"
    assert call['config'].model.gridsize == 1, \
        f"config.model.gridsize must be 1, got {call['config'].model.gridsize}"

    # Assertions: result contains expected keys
    assert result is not None, \
        "execute_training_job must return a result dict"
    assert 'status' in result, \
        "result must contain 'status' key"

    # Assertions: log_path exists and contains execution metadata
    assert log_path.exists(), \
        f"log_path not written: {log_path}"
    log_content = log_path.read_text()
    assert 'Phase E5 Training Execution' in log_content, \
        "log must contain Phase E5 execution marker"

    print(f"\n✓ execute_training_job delegation validated:")
    print(f"  - train_cdi_model_torch called: {len(trainer_calls)} time(s)")
    print(f"  - Received train_data: {type(call['train_data'])}")
    print(f"  - Received test_data: {type(call['test_data'])}")
    print(f"  - Received config with gridsize={call['config'].model.gridsize}, nphotons={call['config'].nphotons}")
    print(f"  - Result: {result}")
    print(f"  - Log written: {log_path}")


def test_training_cli_invokes_real_runner(tmp_path, monkeypatch):
    """
    RED → GREEN TDD test for Phase E5 real training runner integration.

    Validates that the training CLI main() function, when invoked without --dry-run:
    - Invokes a production runner helper (execute_training_job) instead of stub_runner
    - Passes resolved TrainingJob and TrainingConfig to the runner
    - Runner helper performs CONFIG-001 bridging (update_legacy_dict)
    - Runner helper delegates to actual backend trainer (e.g., train_cdi_model_torch)
    - Artifacts (logs, manifests) are written under --artifact-root

    Test strategy: Monkeypatch execute_training_job to spy on invocation without
    executing full training. Validate that CLI calls the real runner with proper
    parameters when --dry-run is NOT set.

    References:
    - input.md:10 (Phase E5: wire CLI to real runner with deterministic execution)
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md Phase E future item 0
    - docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001 bridging)
    - docs/workflows/pytorch.md §12 (canonical PyTorch training invocation)
    """
    import sys
    import json
    from studies.fly64_dose_overlap import training
    from ptycho.config.config import TrainingConfig

    # Setup: Create mock Phase C and Phase D directories
    phase_c_root = tmp_path / "phase_c"
    phase_d_root = tmp_path / "phase_d"
    artifact_root = tmp_path / "artifacts"

    phase_c_root.mkdir()
    phase_d_root.mkdir()

    # Create minimal dataset structure for ALL doses (build_training_jobs enumerates all)
    # StudyDesign.dose_list = [1e3, 1e4, 1e5] per design.py
    for dose in [1000, 10000, 100000]:
        dose_dir_c = phase_c_root / f"dose_{dose}"
        dose_dir_d = phase_d_root / f"dose_{dose}"
        dose_dir_c.mkdir()
        dose_dir_d.mkdir()

        # Phase C: baseline patched datasets
        (dose_dir_c / "patched_train.npz").touch()
        (dose_dir_c / "patched_test.npz").touch()

        # Phase D: overlap view datasets (needed even though we filter to baseline)
        for view in ['dense', 'sparse']:
            (dose_dir_d / f"{view}_train.npz").touch()
            (dose_dir_d / f"{view}_test.npz").touch()

    # Spy: record calls to execute_training_job (the real runner helper)
    runner_calls = []

    def spy_execute_training_job(*, config, job, log_path):
        """Spy that records invocation signature for validation."""
        runner_calls.append({
            'config': config,
            'job': job,
            'log_path': log_path,
        })
        # Return minimal success result to satisfy CLI expectations
        return {'status': 'success', 'final_loss': 0.123}

    # Monkeypatch the real runner helper (NOT stub_runner)
    # Assumption: execute_training_job is the production helper added in E5
    monkeypatch.setattr(training, 'execute_training_job', spy_execute_training_job)

    # Monkeypatch CLI to default to execute_training_job instead of stub_runner
    # This requires main() to be updated to accept runner injection or use execute_training_job
    # For now, we'll assume main() is refactored to call execute_training_job by default

    # Execute CLI without --dry-run for baseline job
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--dose', '1000',
        '--view', 'baseline',
        # No --dry-run flag → should invoke real runner
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    training.main()

    # Assertions: execute_training_job was called
    assert len(runner_calls) == 1, \
        f"execute_training_job should be called exactly once, got {len(runner_calls)} calls"

    call = runner_calls[0]

    # Assertions: config is TrainingConfig instance
    assert isinstance(call['config'], TrainingConfig), \
        f"Runner must receive TrainingConfig instance, got {type(call['config'])}"

    # Assertions: config has correct fields
    assert call['config'].train_data_file.endswith('patched_train.npz'), \
        f"config.train_data_file must point to patched_train.npz, got {call['config'].train_data_file}"
    assert call['config'].test_data_file.endswith('patched_test.npz'), \
        f"config.test_data_file must point to patched_test.npz, got {call['config'].test_data_file}"
    assert call['config'].nphotons == 1000.0, \
        f"config.nphotons must match dose=1000, got {call['config'].nphotons}"
    assert call['config'].model.gridsize == 1, \
        f"config.model.gridsize must be 1 for baseline, got {call['config'].model.gridsize}"

    # Assertions: job metadata matches
    assert call['job'].dose == 1e3, \
        f"job.dose must be 1e3, got {call['job'].dose}"
    assert call['job'].view == 'baseline', \
        f"job.view must be 'baseline', got {call['job'].view}"
    assert call['job'].gridsize == 1, \
        f"job.gridsize must be 1, got {call['job'].gridsize}"

    # Assertions: log_path is Path instance pointing to artifact tree
    assert call['log_path'] is not None, \
        "log_path must be provided to runner"
    assert artifact_root in call['log_path'].parents, \
        f"log_path must be under artifact_root, got {call['log_path']}"

    print(f"\n✓ CLI real runner integration validated:")
    print(f"  - execute_training_job called: {len(runner_calls)} time(s)")
    print(f"  - Received TrainingConfig with nphotons={call['config'].nphotons}, gridsize={call['config'].model.gridsize}")
    print(f"  - Job metadata: dose={call['job'].dose}, view={call['job'].view}")
    print(f"  - Log path: {call['log_path']}")

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
from types import SimpleNamespace
from studies.fly64_dose_overlap import training
from studies.fly64_dose_overlap.training import build_training_jobs, TrainingJob
from studies.fly64_dose_overlap.design import get_study_design


def test_train_cdi_model_normalizes_history(monkeypatch):
    """
    Ensure TensorFlow workflow returns dict-based history even if Keras produces a History object.
    """
    from ptycho.workflows import components as workflow_components

    train_raw = object()
    test_raw = object()
    container_map = {
        train_raw: SimpleNamespace(probe='train_probe'),
        test_raw: SimpleNamespace(probe='test_probe'),
    }

    def fake_create_container(data, config):
        return container_map[data]

    class FakeDataset:
        def __init__(self, train_container, test_container):
            self.train_data = train_container
            self.test_data = test_container

    class FakeHistory:
        def __init__(self):
            self.history = {'loss': [0.4, 0.2]}
            self.epoch = [0, 1]

    def fake_train_eval(dataset, model_instance=None):
        return {'history': FakeHistory(), 'model_instance': object()}

    monkeypatch.setattr(workflow_components, 'create_ptycho_data_container', fake_create_container)
    monkeypatch.setattr(workflow_components.probe, 'set_probe_guess', lambda *args, **kwargs: None)
    monkeypatch.setattr('ptycho.loader.PtychoDataset', FakeDataset)
    monkeypatch.setattr('ptycho.train_pinn.train_eval', fake_train_eval)

    class FakeGenerator:
        name = "fake"

        def build_models(self):
            return object(), object()

    monkeypatch.setattr(workflow_components, "resolve_generator", lambda config: FakeGenerator())

    from ptycho.config.config import TrainingConfig, ModelConfig
    config = TrainingConfig(model=ModelConfig())
    results = workflow_components.train_cdi_model(train_raw, test_raw, config)

    assert isinstance(results['history'], dict), "History payload must be normalized to a dict"
    assert 'loss' in results['history'], "Original loss series should be preserved"
    assert results['history'].get('train_loss') == results['history']['loss'], (
        "train_loss alias must mirror loss series for legacy consumers"
    )
    assert results['history_epochs'] == [0, 1], "Epoch metadata must be captured"
    assert results['train_container'] is container_map[train_raw]
    assert results['test_container'] is container_map[test_raw]


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

    Structure (matches actual Phase D overlap.py output):
        dose_1000/
            dense/
                dense_train.npz
                dense_test.npz
            sparse/
                sparse_train.npz
                sparse_test.npz
        dose_10000/
            dense/
                dense_train.npz
                dense_test.npz
            sparse/
                sparse_train.npz
                sparse_test.npz
        dose_100000/
            dense/
                dense_train.npz
                dense_test.npz
            sparse/
                sparse_train.npz
                sparse_test.npz

    Each NPZ contains minimal DATA-001 keys plus gridsize=2 metadata.

    References:
        - studies/fly64_dose_overlap/overlap.py:490 (output_dir = output_root / dose / view)
        - studies/fly64_dose_overlap/overlap.py:366 (output_path = output_dir / f"{view}_{split_name}.npz")
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

        # Write dense and sparse NPZs under view subdirectories
        for view in ['dense', 'sparse']:
            view_dir = dose_dir / view
            view_dir.mkdir(parents=True, exist_ok=True)

            for split in ['train', 'test']:
                # Match actual Phase D output pattern: dose_X/view/view_split.npz
                npz_path = view_dir / f"{view}_{split}.npz"
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

    def stub_runner(*, config, job, log_path, backend):
        runner_calls.append({
            'config': config,
            'job': job,
            'log_path': log_path,
            'backend': backend,
        })
        assert backend == 'tensorflow', \
            f"Expected default backend 'tensorflow', got {backend}"
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
    assert bridge_call['config'].backend == 'tensorflow', \
        f"TrainingConfig.backend should default to 'tensorflow', got {bridge_call['config'].backend}"

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
    assert runner_call['backend'] == 'tensorflow', \
        f"Runner backend mismatch: expected 'tensorflow', got {runner_call['backend']}"

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


def test_execute_training_job_dispatches_tensorflow_by_default(tmp_path, monkeypatch):
    """Default execute_training_job call should route to TensorFlow backend."""
    from ptycho.config.config import TrainingConfig, ModelConfig
    from studies.fly64_dose_overlap import training as training_module

    artifact_dir = tmp_path / "artifacts"
    log_path = artifact_dir / "train.log"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.touch()
    test_npz.touch()

    job = TrainingJob(
        dose=1e3,
        view='baseline',
        gridsize=1,
        train_data_path=str(train_npz),
        test_data_path=str(test_npz),
        artifact_dir=artifact_dir,
        log_path=log_path,
    )

    config = TrainingConfig(
        train_data_file=str(train_npz),
        test_data_file=str(test_npz),
        output_dir=str(artifact_dir),
        model=ModelConfig(gridsize=1),
    )

    tf_calls = []

    def fake_tf_runner(*, config, job, log_path):
        tf_calls.append({'config': config, 'job': job, 'log_path': log_path})
        return {'status': 'success', 'bundle_path': None}

    def fail_pytorch_runner(*args, **kwargs):
        pytest.fail("PyTorch backend should not be invoked when backend is default")

    monkeypatch.setattr(training_module, '_execute_training_job_tensorflow', fake_tf_runner, raising=False)
    monkeypatch.setattr(training_module, '_execute_training_job_pytorch', fail_pytorch_runner, raising=False)

    result = training_module.execute_training_job(
        config=config,
        job=job,
        log_path=log_path,
    )

    assert result['status'] == 'success', "TensorFlow runner should return success result"
    assert len(tf_calls) == 1, "TensorFlow runner should be called exactly once"
    assert tf_calls[0]['config'].backend == 'tensorflow', \
        f"TensorFlow runner must receive backend='tensorflow', got {tf_calls[0]['config'].backend}"


def test_execute_training_job_dispatches_pytorch_when_requested(tmp_path, monkeypatch):
    """execute_training_job should route to PyTorch backend when explicitly requested."""
    from ptycho.config.config import TrainingConfig, ModelConfig
    from studies.fly64_dose_overlap import training as training_module

    artifact_dir = tmp_path / "artifacts"
    log_path = artifact_dir / "train.log"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.touch()
    test_npz.touch()

    job = TrainingJob(
        dose=1e3,
        view='dense',
        gridsize=2,
        train_data_path=str(train_npz),
        test_data_path=str(test_npz),
        artifact_dir=artifact_dir,
        log_path=log_path,
    )

    config = TrainingConfig(
        train_data_file=str(train_npz),
        test_data_file=str(test_npz),
        output_dir=str(artifact_dir),
        model=ModelConfig(gridsize=2),
        backend='pytorch',
    )

    torch_calls = []

    def fake_pytorch_runner(*, config, job, log_path):
        torch_calls.append({'config': config, 'job': job, 'log_path': log_path})
        return {'status': 'success', 'bundle_path': None}

    def fail_tensorflow_runner(*args, **kwargs):
        pytest.fail("TensorFlow backend should not be invoked when backend='pytorch'")

    monkeypatch.setattr(training_module, '_execute_training_job_pytorch', fake_pytorch_runner, raising=False)
    monkeypatch.setattr(training_module, '_execute_training_job_tensorflow', fail_tensorflow_runner, raising=False)

    result = training_module.execute_training_job(
        config=config,
        job=job,
        log_path=log_path,
        backend='pytorch',
    )

    assert result['status'] == 'success', "PyTorch runner should return success result"
    assert len(torch_calls) == 1, "PyTorch runner should be called exactly once"
    assert torch_calls[0]['config'].backend == 'pytorch', \
        f"PyTorch runner must receive backend='pytorch', got {torch_calls[0]['config'].backend}"


def test_execute_training_job_tensorflow_persists_bundle(tmp_path, monkeypatch):
    """
    TensorFlow backend should save bundles via model_manager.save with manifest metadata.
    """
    from ptycho.config.config import TrainingConfig, ModelConfig
    from studies.fly64_dose_overlap import training as training_module

    artifact_dir = tmp_path / "artifacts"
    log_path = artifact_dir / "train.log"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    minimal_data = {
        'diffraction': np.random.rand(4, 32, 32).astype(np.float32),
        'objectGuess': np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
        'probeGuess': np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
        'xcoords': np.linspace(0, 1, 4).astype(np.float32),
        'ycoords': np.linspace(0, 1, 4).astype(np.float32),
        'xcoords_start': np.linspace(0, 1, 4).astype(np.float32),
        'ycoords_start': np.linspace(0, 1, 4).astype(np.float32),
        'Y': (np.random.rand(4, 64, 64) + 1j * np.random.rand(4, 64, 64)).astype(np.complex64),
    }
    np.savez_compressed(train_npz, **minimal_data)
    np.savez_compressed(test_npz, **minimal_data)

    job = TrainingJob(
        dose=1e3,
        view='baseline',
        gridsize=1,
        train_data_path=str(train_npz),
        test_data_path=str(test_npz),
        artifact_dir=artifact_dir,
        log_path=log_path,
    )

    config = TrainingConfig(
        train_data_file=str(train_npz),
        test_data_file=str(test_npz),
        output_dir=str(artifact_dir),
        model=ModelConfig(gridsize=1),
        backend='tensorflow',
    )

    load_calls = []
    def fake_load_data(path, **kwargs):
        load_calls.append(path)
        return SimpleNamespace()

    def fake_train_cdi_model(train_data, test_data, config):
        return {'history': {'train_loss': [0.5, 0.2, 0.1]}}

    saved_paths = []

    def fake_model_manager_save(out_prefix):
        bundle_base = Path(out_prefix) / 'wts.h5'
        bundle_path = bundle_base.with_suffix('.h5.zip')
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text("dummy bundle")
        saved_paths.append(bundle_path)

    monkeypatch.setattr(
        training_module,
        'tf_components',
        SimpleNamespace(load_data=fake_load_data, train_cdi_model=fake_train_cdi_model),
        raising=False,
    )
    monkeypatch.setattr(training_module, 'model_manager', SimpleNamespace(save=fake_model_manager_save), raising=False)
    monkeypatch.setattr(training_module, 'validate_dataset_contract', lambda *args, **kwargs: None, raising=False)

    result = training_module._execute_training_job_tensorflow(
        config=config,
        job=job,
        log_path=log_path,
    )

    assert result['status'] == 'success', "TensorFlow execution should report success"
    assert result['bundle_path'] is not None, "TensorFlow execution must return bundle_path"
    assert Path(result['bundle_path']).exists(), "Persisted bundle path must exist on disk"
    assert result['bundle_sha256'], "TensorFlow execution must compute bundle SHA256"
    assert result['bundle_size_bytes'] > 0, "TensorFlow bundle size must be positive"
    assert len(load_calls) == 2, "TensorFlow loader should be called for train and test NPZ files"
    assert len(saved_paths) == 1, "model_manager.save should be invoked exactly once"


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

        # Phase D overlap views (match actual Phase D layout: dose/view/view_split.npz)
        for view in ['dense', 'sparse']:
            view_dir = dose_dir_d / view
            view_dir.mkdir(parents=True, exist_ok=True)
            (view_dir / f"{view}_train.npz").touch()
            (view_dir / f"{view}_test.npz").touch()

    # Spy: Track which jobs were executed
    executed_jobs = []

    def mock_run_training_job(job, runner, dry_run=False, backend='tensorflow'):
        executed_jobs.append({
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
            'dry_run': dry_run,
            'backend': backend,
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
    assert {entry['backend'] for entry in executed_jobs} == {'tensorflow'}, \
        f"CLI should default to tensorflow backend, got backends: {executed_jobs}"

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
    assert {entry['backend'] for entry in executed_jobs} == {'tensorflow'}, \
        "Filtered run should still default to tensorflow backend"
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
    - Manifest exposes skipped_views array with skip metadata (dose, view, reason)
    - Ensures run_training_job is called with proper CONFIG-001 bridge
    - Writes CLI stdout/stderr log to --artifact-root for traceability

    Test strategy: Mock run_training_job to verify it's called (CONFIG-001 handled internally),
    execute CLI with --dry-run and deliberately missing sparse view data, validate manifest
    JSON structure and content including skipped_views field.

    References:
    - input.md:10 (Phase E5: manifest skip reporting requirement)
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
    # Phase E5: Deliberately omit sparse view for dose=1000 to test skip reporting
    for dose in [1000, 10000, 100000]:
        dose_dir_c = phase_c_root / f"dose_{dose}"
        dose_dir_d = phase_d_root / f"dose_{dose}"
        dose_dir_c.mkdir()
        dose_dir_d.mkdir()

        # Phase C patched datasets
        (dose_dir_c / "patched_train.npz").touch()
        (dose_dir_c / "patched_test.npz").touch()

        # Phase D overlap views (match actual Phase D layout: dose/view/view_split.npz)
        # Phase E5: Only create dense view for dose=1000; skip sparse to test skip reporting
        views = ['dense'] if dose == 1000 else ['dense', 'sparse']
        for view in views:
            view_dir = dose_dir_d / view
            view_dir.mkdir(parents=True, exist_ok=True)
            (view_dir / f"{view}_train.npz").touch()
            (view_dir / f"{view}_test.npz").touch()

    # Spy: Track run_training_job calls
    run_calls = []

    def mock_run_training_job(job, runner, dry_run=False, backend='tensorflow'):
        run_calls.append({
            'dose': job.dose,
            'view': job.view,
            'gridsize': job.gridsize,
            'backend': backend,
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

    # Runner should have been invoked for each executed job with tensorflow backend
    assert len(run_calls) == 2, \
        f"run_training_job should be called twice (baseline + dense), got {len(run_calls)} calls"
    assert {call['backend'] for call in run_calls} == {'tensorflow'}, \
        f"Expected run_training_job backend to default to tensorflow, got backends: {run_calls}"

    # Assertions: manifest contains expected keys
    required_keys = {'timestamp', 'phase_c_root', 'phase_d_root', 'artifact_root', 'jobs'}
    missing_keys = required_keys - manifest.keys()
    assert not missing_keys, \
        f"Manifest missing keys: {missing_keys}"

    # Assertions: jobs list matches executed jobs (2 for dose=1000: baseline + dense, sparse skipped)
    assert isinstance(manifest['jobs'], list), \
        f"Manifest 'jobs' must be a list, got {type(manifest['jobs'])}"
    assert len(manifest['jobs']) == 2, \
        f"Expected 2 jobs in manifest for dose=1000 (baseline + dense, sparse skipped), got {len(manifest['jobs'])}"

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

    # Assertions: run_training_job was called for each job (2 for dose=1000: baseline + dense)
    assert len(run_calls) == 2, \
        f"Expected run_training_job called 2 times (baseline + dense), got {len(run_calls)}"
    assert {call['backend'] for call in run_calls} == {'tensorflow'}, \
        f"Expected CLI to invoke training with tensorflow backend, got backends {run_calls}"

    # NEW Phase E5 Assertions: manifest contains skipped_views with missing sparse view
    assert 'skipped_views' in manifest, \
        "Manifest must contain 'skipped_views' field for Phase E5 skip reporting"
    assert isinstance(manifest['skipped_views'], list), \
        f"Manifest 'skipped_views' must be a list, got {type(manifest['skipped_views'])}"

    # Exactly 1 skip event: dose=1000 sparse view missing
    assert len(manifest['skipped_views']) == 1, \
        f"Expected 1 skipped view (dose=1000 sparse), got {len(manifest['skipped_views'])}"

    skip_event = manifest['skipped_views'][0]
    assert skip_event['dose'] == 1000.0, \
        f"Skipped event dose must be 1000.0, got {skip_event['dose']}"
    assert skip_event['view'] == 'sparse', \
        f"Skipped event view must be 'sparse', got {skip_event['view']}"
    assert 'reason' in skip_event, \
        "Skipped event must contain 'reason' field explaining why view was skipped"
    assert 'not found' in skip_event['reason'].lower() or 'missing' in skip_event['reason'].lower(), \
        f"Skipped event reason should mention missing files, got: {skip_event['reason']}"

    # NEW Phase E5 Assertions: manifest contains skipped_count convenience field
    assert 'skipped_count' in manifest, \
        "Manifest must contain 'skipped_count' field for Phase E5 skip reporting"
    assert manifest['skipped_count'] == 1, \
        f"Manifest 'skipped_count' must match len(skipped_views), expected 1, got {manifest['skipped_count']}"

    # NEW Phase E5.5 Assertions: skip_summary.json file exists alongside manifest
    skip_summary_path = artifact_root / "skip_summary.json"
    assert skip_summary_path.exists(), \
        f"skip_summary.json not found at {skip_summary_path} (Phase E5.5 skip persistence requirement)"

    # Assertions: skip_summary.json content is valid JSON
    with skip_summary_path.open('r') as f:
        skip_summary = json.load(f)

    assert isinstance(skip_summary, dict), \
        f"skip_summary.json must be a dict, got {type(skip_summary)}"

    # Assertions: skip_summary contains expected keys
    required_summary_keys = {'timestamp', 'skipped_views', 'skipped_count'}
    missing_summary_keys = required_summary_keys - skip_summary.keys()
    assert not missing_summary_keys, \
        f"skip_summary.json missing keys: {missing_summary_keys}"

    # Assertions: skip_summary.skipped_views matches manifest.skipped_views
    assert skip_summary['skipped_views'] == manifest['skipped_views'], \
        f"skip_summary.skipped_views must match manifest.skipped_views"
    assert skip_summary['skipped_count'] == manifest['skipped_count'], \
        f"skip_summary.skipped_count must match manifest.skipped_count"

    # Assertions: manifest contains skip_summary_path reference
    assert 'skip_summary_path' in manifest, \
        "Manifest must contain 'skip_summary_path' field pointing to skip_summary.json"
    assert manifest['skip_summary_path'] == str(skip_summary_path.relative_to(artifact_root)), \
        f"Manifest skip_summary_path should be relative path 'skip_summary.json', got {manifest.get('skip_summary_path')}"

    print(f"\n✓ CLI manifest and bridging validated:")
    print(f"  - training_manifest.json created at {manifest_path}")
    print(f"  - Manifest contains {len(manifest['jobs'])} job entries (baseline + dense)")
    print(f"  - run_training_job called {len(run_calls)} times (CONFIG-001 bridging implicit)")
    print(f"  - Each job entry has required metadata keys")
    print(f"  - Manifest contains {len(manifest['skipped_views'])} skipped view(s): {skip_event['view']} (dose={skip_event['dose']:.0e})")
    print(f"  - Skip reason: {skip_event['reason'][:100]}...")
    print(f"  - skip_summary.json created at {skip_summary_path} with matching content")
    print(f"  - Manifest references skip_summary via relative path: {manifest['skip_summary_path']}")


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
    from studies.fly64_dose_overlap import training as training_module
    from studies.fly64_dose_overlap import training as training_module
    from studies.fly64_dose_overlap import training
    from studies.fly64_dose_overlap import training
    from studies.fly64_dose_overlap import training
    from studies.fly64_dose_overlap.training import TrainingJob
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

    # Spy: record calls to MemmapDatasetBridge
    bridge_calls = []

    class SpyMemmapDatasetBridge:
        """Spy that records MemmapDatasetBridge instantiation and delegation."""
        def __init__(self, npz_path, config, memmap_dir="data/memmap"):
            bridge_calls.append({
                'npz_path': npz_path,
                'config': config,
                'memmap_dir': memmap_dir,
            })
            # Store for delegation to raw_data_torch
            self.npz_path = npz_path
            self.config = config
            # Mock raw_data_torch attribute (Phase C.C3 RawDataTorch delegation)
            from ptycho.raw_data import RawData
            self.raw_data_torch = RawData(
                xcoords=np.array([0.0]),
                ycoords=np.array([0.0]),
                xcoords_start=np.array([0.0]),
                ycoords_start=np.array([0.0]),
                diff3d=np.random.rand(1, 64, 64).astype(np.float32),
                probeGuess=np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
                scan_index=np.array([0]),
            )

    # Monkeypatch PyTorch backend components at their source modules
    monkeypatch.setattr('ptycho_torch.memmap_bridge.MemmapDatasetBridge', SpyMemmapDatasetBridge)
    monkeypatch.setattr('ptycho_torch.workflows.components.train_cdi_model_torch', spy_train_cdi_model_torch)

    # Execute: call execute_training_job
    result = training._execute_training_job_pytorch(
        config=config,
        job=job,
        log_path=log_path,
    )

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

    # NEW Assertions (Phase E5): MemmapDatasetBridge instantiation
    assert len(bridge_calls) == 2, \
        f"MemmapDatasetBridge should be instantiated twice (train + test), got {len(bridge_calls)} calls"

    # Validate train bridge call
    train_bridge_call = bridge_calls[0]
    assert str(train_bridge_call['npz_path']) == str(train_npz), \
        f"Train bridge should receive train_npz path, got {train_bridge_call['npz_path']}"
    assert train_bridge_call['config'] is config, \
        "Train bridge should receive the same TrainingConfig instance"

    # Validate test bridge call
    test_bridge_call = bridge_calls[1]
    assert str(test_bridge_call['npz_path']) == str(test_npz), \
        f"Test bridge should receive test_npz path, got {test_bridge_call['npz_path']}"
    assert test_bridge_call['config'] is config, \
        "Test bridge should receive the same TrainingConfig instance"

    # Assertions: RawDataTorch payload passed to trainer
    # The trainer should receive the raw_data_torch attribute from the bridge
    # (not the bridge itself, but the RawData instance it wraps)
    from ptycho.raw_data import RawData
    assert isinstance(call['train_data'], RawData), \
        f"train_cdi_model_torch should receive RawData from bridge.raw_data_torch, got {type(call['train_data'])}"
    assert isinstance(call['test_data'], RawData), \
        f"train_cdi_model_torch should receive RawData from bridge.raw_data_torch, got {type(call['test_data'])}"

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


def test_execute_training_job_persists_bundle(tmp_path, monkeypatch):
    """
    RED → GREEN TDD test for Phase E training bundle persistence.

    Validates that execute_training_job():
    - Calls save_torch_bundle after successful training
    - Creates wts.h5.zip archive in artifact_dir
    - Populates result['bundle_path'] with the bundle archive path
    - Writes bundle metadata to training manifest

    This test implements the gating prerequisite for Phase G comparisons,
    ensuring real training runs emit spec-compliant model bundles per
    specs/ptychodus_api_spec.md §4.6.

    Test Strategy:
    - Monkeypatch train_cdi_model_torch to return success with model stubs
    - Monkeypatch save_torch_bundle to spy on invocation
    - Verify bundle_path in result dict and file existence
    - Validate manifest fields include bundle_path

    References:
        - input.md:9 (Phase E5: bundle persistence requirement)
        - specs/ptychodus_api_spec.md:239 (§4.6 wts.h5.zip contract)
        - docs/fix_plan.md:31 (Phase G blocked on training bundles)
        - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163
    """
    from studies.fly64_dose_overlap.training import TrainingJob
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Setup: Create minimal Phase C fixture NPZ (DATA-001 compliant)
    train_npz = tmp_path / "phase_c_train.npz"
    test_npz = tmp_path / "phase_c_test.npz"

    minimal_data = {
        'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
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

    # Setup: Create TrainingConfig
    model_config = ModelConfig(gridsize=1)
    config = TrainingConfig(
        train_data_file=str(train_npz),
        test_data_file=str(test_npz),
        output_dir=str(artifact_dir),
        model=model_config,
        nphotons=1e3,
    )

    # Spy: record calls to save_torch_bundle
    bundle_calls = []

    def spy_save_torch_bundle(models_dict, base_path, config, intensity_scale=None):
        """Spy that records bundle save invocations."""
        bundle_calls.append({
            'models_dict': models_dict,
            'base_path': base_path,
            'config': config,
            'intensity_scale': intensity_scale,
        })
        # Create dummy bundle archive to simulate successful save
        bundle_path = Path(f"{base_path}.zip")
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text("dummy bundle content")

    # Mock train_cdi_model_torch to return success with model stubs
    def mock_train_cdi_model_torch(train_data, test_data, config):
        # Return minimal success with model stubs
        class DummyModel:
            pass
        return {
            'history': {'train_loss': [0.5, 0.3, 0.1]},
            'train_container': train_data,
            'test_container': test_data,
            'models': {
                'autoencoder': DummyModel(),
                'diffraction_to_obj': DummyModel(),
            },
        }

    # Mock MemmapDatasetBridge
    class SpyMemmapDatasetBridge:
        def __init__(self, npz_path, config, memmap_dir="data/memmap"):
            from ptycho.raw_data import RawData
            self.raw_data_torch = RawData(
                xcoords=np.array([0.0]),
                ycoords=np.array([0.0]),
                xcoords_start=np.array([0.0]),
                ycoords_start=np.array([0.0]),
                diff3d=np.random.rand(1, 64, 64).astype(np.float32),
                probeGuess=np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
                scan_index=np.array([0]),
            )

    # Monkeypatch PyTorch backend components
    monkeypatch.setattr('ptycho_torch.memmap_bridge.MemmapDatasetBridge', SpyMemmapDatasetBridge)
    monkeypatch.setattr('ptycho_torch.workflows.components.train_cdi_model_torch', mock_train_cdi_model_torch)
    monkeypatch.setattr('ptycho_torch.model_manager.save_torch_bundle', spy_save_torch_bundle)

    # Execute: call execute_training_job
    result = training._execute_training_job_pytorch(
        config=config,
        job=job,
        log_path=log_path,
    )

    # Debug output
    print(f"\nResult from execute_training_job: {result}")
    if log_path.exists():
        print(f"Log contents:\n{log_path.read_text()}")

    # RED Assertions: bundle persistence (expecting failure before implementation)
    assert result['status'] == 'success', \
        f"Training should succeed, got status={result['status']}"

    assert 'bundle_path' in result, \
        "result must contain 'bundle_path' key after successful training"

    assert result['bundle_path'] is not None, \
        "bundle_path must not be None after successful training"

    # Verify save_torch_bundle was called
    assert len(bundle_calls) == 1, \
        f"save_torch_bundle should be called exactly once, got {len(bundle_calls)} calls"

    call = bundle_calls[0]
    assert 'autoencoder' in call['models_dict'], \
        "models_dict must contain 'autoencoder' model"
    assert 'diffraction_to_obj' in call['models_dict'], \
        "models_dict must contain 'diffraction_to_obj' model"
    assert call['config'] is config, \
        "save_torch_bundle must receive the same TrainingConfig instance"

    # Verify bundle file exists
    bundle_path = Path(result['bundle_path'])
    assert bundle_path.exists(), \
        f"Bundle archive must exist at {bundle_path}"
    assert bundle_path.suffix == '.zip', \
        f"Bundle must be a .zip archive, got {bundle_path.suffix}"

    # NEW: Verify bundle_sha256 field (Phase E6)
    assert 'bundle_sha256' in result, \
        "result must contain 'bundle_sha256' key after successful bundle persistence"

    assert result['bundle_sha256'] is not None, \
        "bundle_sha256 must not be None when bundle_path is populated"

    # Validate SHA256 format (64-character lowercase hexadecimal)
    sha256_value = result['bundle_sha256']
    assert isinstance(sha256_value, str), \
        f"bundle_sha256 must be a string, got {type(sha256_value)}"
    assert len(sha256_value) == 64, \
        f"bundle_sha256 must be 64 characters (SHA256 hex digest), got {len(sha256_value)}"
    assert sha256_value.islower(), \
        f"bundle_sha256 must be lowercase hex, got {sha256_value}"
    assert all(c in '0123456789abcdef' for c in sha256_value), \
        f"bundle_sha256 must be hexadecimal, got {sha256_value}"

    # NEW: Verify on-disk SHA256 matches result['bundle_sha256'] (Phase E6 Do Now)
    # Recompute SHA256 from the actual bundle file using the same algorithm as production
    import hashlib
    sha256_hash = hashlib.sha256()
    with bundle_path.open('rb') as f:
        # Read in 64KB chunks to match production pattern (training.py:514-517)
        for chunk in iter(lambda: f.read(65536), b''):
            sha256_hash.update(chunk)
    on_disk_sha256 = sha256_hash.hexdigest()

    assert on_disk_sha256 == result['bundle_sha256'], \
        f"On-disk SHA256 ({on_disk_sha256}) must match result['bundle_sha256'] ({result['bundle_sha256']})"

    print(f"\n✓ execute_training_job bundle persistence validated:")
    print(f"  - save_torch_bundle called: {len(bundle_calls)} time(s)")
    print(f"  - Bundle path: {result['bundle_path']}")
    print(f"  - Bundle exists: {bundle_path.exists()}")
    print(f"  - Bundle SHA256 (result): {result['bundle_sha256']}")
    print(f"  - Bundle SHA256 (on-disk): {on_disk_sha256}")
    print(f"  - SHA256 match: {on_disk_sha256 == result['bundle_sha256']}")
    print(f"  - Models persisted: {list(call['models_dict'].keys())}")


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

    def spy_execute_training_job(*, config, job, log_path, backend):
        """Spy that records invocation signature for validation."""
        runner_calls.append({
            'config': config,
            'job': job,
            'log_path': log_path,
            'backend': backend,
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
    assert call['backend'] == 'tensorflow', \
        f"Runner backend should default to 'tensorflow', got {call['backend']}"

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


def test_build_training_jobs_skips_missing_view(mock_phase_c_datasets, tmp_path):
    """
    RED → GREEN TDD test for build_training_jobs with missing overlap view.

    Validates that build_training_jobs(..., allow_missing_phase_d=True):
    - Skips overlap jobs (dense/sparse) when their NPZ files are missing
    - Logs the omission with clear messaging
    - Still returns baseline jobs (Phase C always complete)
    - Does NOT raise FileNotFoundError when allow_missing_phase_d=True

    This scenario occurs when:
    - Phase C completed successfully for all doses
    - Phase D overlap filtering rejected some views due to spacing threshold
      (e.g., sparse view with too few positions after filtering)

    Expected behavior:
    - With allow_missing_phase_d=False (default): FileNotFoundError raised
    - With allow_missing_phase_d=True: missing views logged and skipped

    References:
        - input.md:10 (Phase E5 task: add allow_missing_phase_d switch)
        - docs/fix_plan.md:33 (Attempt #20 CLI failure on sparse view rejection)
        - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163
          (Phase E exit criteria: CLI must handle missing overlap data gracefully)
    """
    import logging
    from io import StringIO

    # Setup: Phase C complete, Phase D has dense but missing sparse view
    phase_d_root = tmp_path / "phase_d"
    artifact_root = tmp_path / "artifacts"
    design = get_study_design()

    # Create minimal DATA-001 arrays for Phase D dense view only
    minimal_data = {
        'diffraction': np.random.rand(5, 64, 64).astype(np.float32),
        'objectGuess': np.random.rand(128, 128) + 1j * np.random.rand(128, 128),
        'probeGuess': np.random.rand(64, 64) + 1j * np.random.rand(64, 64),
        'Y': (np.random.rand(5, 128, 128) + 1j * np.random.rand(5, 128, 128)).astype(np.complex64),
        'xcoords': np.random.rand(5).astype(np.float32),
        'ycoords': np.random.rand(5).astype(np.float32),
        'filenames': np.array([f'img_{i:04d}' for i in range(5)]),
    }

    # Create Phase D with dense view present but sparse view missing
    for dose in design.dose_list:
        dose_dir = phase_d_root / f"dose_{int(dose)}"

        # Dense view: present (matches actual Phase D layout)
        dense_dir = dose_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dense_dir / "dense_train.npz", **minimal_data)
        np.savez_compressed(dense_dir / "dense_test.npz", **minimal_data)

        # Sparse view: MISSING (simulates spacing threshold rejection)
        # Do NOT create dose_dir / "sparse" directory or NPZ files

    # Capture log output to verify skip messaging
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger('studies.fly64_dose_overlap.training')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        # Test 1: With allow_missing_phase_d=False (default), expect FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            jobs = build_training_jobs(
                phase_c_root=mock_phase_c_datasets,
                phase_d_root=phase_d_root,
                artifact_root=artifact_root,
                design=design,
                allow_missing_phase_d=False,  # strict validation
            )

        # Assert error message mentions the missing sparse view
        assert 'sparse' in str(exc_info.value).lower(), \
            f"FileNotFoundError should mention missing 'sparse' view, got: {exc_info.value}"

        print(f"\n✓ Strict mode (allow_missing_phase_d=False) correctly raised FileNotFoundError")
        print(f"  Error: {exc_info.value}")

        # Test 2: With allow_missing_phase_d=True, expect graceful skip
        jobs = build_training_jobs(
            phase_c_root=mock_phase_c_datasets,
            phase_d_root=phase_d_root,
            artifact_root=artifact_root,
            design=design,
            allow_missing_phase_d=True,  # non-strict mode
        )

        # Assertions: Only baseline + dense jobs returned (no sparse)
        # 3 doses × (1 baseline + 1 dense) = 6 jobs total
        assert len(jobs) == 6, \
            f"Expected 6 jobs (3 doses × [baseline + dense]), got {len(jobs)}"

        views_present = {job.view for job in jobs}
        assert views_present == {'baseline', 'dense'}, \
            f"Expected views {{'baseline', 'dense'}}, got {views_present}"
        assert 'sparse' not in views_present, \
            "Sparse jobs should be skipped when NPZs missing and allow_missing_phase_d=True"

        # Validate each dose has 2 jobs (baseline + dense)
        jobs_by_dose = {}
        for job in jobs:
            if job.dose not in jobs_by_dose:
                jobs_by_dose[job.dose] = []
            jobs_by_dose[job.dose].append(job)

        for dose in design.dose_list:
            assert dose in jobs_by_dose, \
                f"Missing jobs for dose={dose}"
            assert len(jobs_by_dose[dose]) == 2, \
                f"Expected 2 jobs (baseline + dense) for dose={dose}, got {len(jobs_by_dose[dose])}"

            views_for_dose = {job.view for job in jobs_by_dose[dose]}
            assert views_for_dose == {'baseline', 'dense'}, \
                f"Expected {{'baseline', 'dense'}} for dose={dose}, got {views_for_dose}"

        # Check log output mentions skipped sparse view
        log_output = log_stream.getvalue()
        assert 'sparse' in log_output.lower() and 'skip' in log_output.lower(), \
            f"Log should mention skipping 'sparse' view, got:\n{log_output}"

        print(f"\n✓ Non-strict mode (allow_missing_phase_d=True) gracefully skipped missing sparse view:")
        print(f"  - Jobs returned: {len(jobs)} (3 doses × 2 variants)")
        print(f"  - Views present: {sorted(views_present)}")
        print(f"  - Log excerpt: {log_output[:200]}...")

    finally:
        # Cleanup logger
        logger.removeHandler(handler)


def test_training_cli_records_bundle_path(tmp_path, monkeypatch, capsys):
    """
    RED → GREEN TDD test for Phase E6 CLI manifest bundle_path normalization and stdout format.

    Validates that the training CLI main() function:
    - Records bundle_path in manifest for each job (relative to job's artifact_dir)
    - Uses artifact-relative paths (not absolute workstation-specific paths)
    - Preserves skip_summary schema unchanged (no interference with bundle fields)
    - Handles missing bundles gracefully (None or omitted field)
    - NEW Phase E6: Emits bundle/SHA lines to stdout with view/dose context for traceability

    Test Strategy:
    - Monkeypatch execute_training_job to return mock results with bundle_path
    - Execute CLI without --dry-run to invoke execute_training_job (mocked)
    - Validate manifest JSON includes bundle_path field for each job entry
    - Verify paths are relative to artifact_dir (e.g., "wts.h5.zip" not "/abs/path/wts.h5.zip")
    - NEW: Capture stdout with capsys and assert bundle/SHA lines include view/dose context

    References:
        - input.md:10 (Phase E6: CLI stdout digest checks with view/dose context)
        - specs/ptychodus_api_spec.md:239 (§4.6 wts.h5.zip persistence contract)
        - docs/TESTING_GUIDE.md:101-140 (Phase E CLI testing requirements)
        - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268
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

    # Create minimal dataset structure for ALL doses (build_training_jobs enumerates all)
    # Then CLI filters by --dose 1000 to reduce job count
    for dose in [1000, 10000, 100000]:
        dose_dir_c = phase_c_root / f"dose_{dose}"
        dose_dir_d = phase_d_root / f"dose_{dose}"
        dose_dir_c.mkdir()
        dose_dir_d.mkdir()

        # Phase C patched datasets
        (dose_dir_c / "patched_train.npz").touch()
        (dose_dir_c / "patched_test.npz").touch()

        # Phase D dense view datasets
        dense_dir = dose_dir_d / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        (dense_dir / "dense_train.npz").touch()
        (dense_dir / "dense_test.npz").touch()

    # Spy: Track execute_training_job calls and return bundle_path in results
    runner_calls = []

    def mock_execute_training_job(*, config, job, log_path, backend):
        """Mock that returns success with bundle_path, bundle_sha256, and bundle_size_bytes."""
        runner_calls.append({
            'config': config,
            'job': job,
            'log_path': log_path,
            'backend': backend,
        })
        # Simulate bundle saved to artifact_dir/wts.h5.zip
        bundle_path_abs = job.artifact_dir / "wts.h5.zip"
        # Generate a mock 64-character hex SHA256 (deterministic for testing)
        # Use job.view to vary the checksum across jobs
        # abs() to avoid negative hash values that would produce '-' prefix
        mock_sha256 = f"{abs(hash(job.view)):064x}"[-64:]  # Ensure 64 chars
        # Generate a mock bundle size (deterministic, varies by view)
        # Use hash to create different sizes for different views
        mock_size = 10000 + abs(hash(job.view)) % 50000  # Range: 10KB - 60KB
        return {
            'status': 'success',
            'final_loss': 0.123,
            'bundle_path': str(bundle_path_abs),  # Absolute path from execute_training_job
            'bundle_sha256': mock_sha256,  # Phase E6: checksum for integrity validation
            'bundle_size_bytes': mock_size,  # Phase E6 Do Now: size tracking
        }

    monkeypatch.setattr(training, 'execute_training_job', mock_execute_training_job)

    # Execute CLI with dose filter (2 jobs: baseline + dense for dose=1000)
    # NOTE: We do NOT use --dry-run here, because dry-run skips execute_training_job entirely
    # and returns a summary dict without bundle_path. We need to invoke execute_training_job
    # (via our mock) to get bundle_path in the result.
    test_argv = [
        'training.py',
        '--phase-c-root', str(phase_c_root),
        '--phase-d-root', str(phase_d_root),
        '--artifact-root', str(artifact_root),
        '--dose', '1000',
        # No --dry-run → invokes execute_training_job (mocked)
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

    # Assertions: jobs list contains 2 entries (baseline + dense for dose=1000)
    assert 'jobs' in manifest, \
        "Manifest must contain 'jobs' list"
    assert len(manifest['jobs']) == 2, \
        f"Expected 2 jobs in manifest for dose=1000 (baseline + dense), got {len(manifest['jobs'])}"

    # Assertions: each job entry contains bundle_path field with relative path
    for job_entry in manifest['jobs']:
        assert 'result' in job_entry, \
            f"Job entry must contain 'result' dict from runner, got keys: {job_entry.keys()}"

        result_dict = job_entry['result']
        assert 'bundle_path' in result_dict, \
            f"Job result must contain 'bundle_path' field (Phase E6 requirement), got keys: {result_dict.keys()}"

        bundle_path = result_dict['bundle_path']
        assert bundle_path is not None, \
            f"bundle_path must not be None for successful training (job: {job_entry['view']})"

        # KEY ASSERTION: bundle_path must be relative to artifact_dir (not absolute)
        # Expected format: "wts.h5.zip" or relative path from artifact_dir
        # NOT acceptable: "/home/user/.../artifacts/dose_1000/baseline/gs1/wts.h5.zip"
        assert not Path(bundle_path).is_absolute(), \
            f"bundle_path must be relative to artifact_dir, got absolute path: {bundle_path}"

        # Validate bundle_path is just the filename (simplest case)
        assert bundle_path == "wts.h5.zip", \
            f"bundle_path should be 'wts.h5.zip' relative to artifact_dir, got: {bundle_path}"

        # Phase E6: Validate bundle_sha256 field is present and properly formatted
        assert 'bundle_sha256' in result_dict, \
            f"Job result must contain 'bundle_sha256' field (Phase E6 checksum requirement), got keys: {result_dict.keys()}"

        bundle_sha256 = result_dict['bundle_sha256']
        assert bundle_sha256 is not None, \
            f"bundle_sha256 must not be None for successful training (job: {job_entry['view']})"

        # Validate SHA256 format (64-character lowercase hexadecimal)
        assert isinstance(bundle_sha256, str), \
            f"bundle_sha256 must be a string, got {type(bundle_sha256)}"
        assert len(bundle_sha256) == 64, \
            f"bundle_sha256 must be 64 characters (SHA256 hex digest), got {len(bundle_sha256)}"
        assert bundle_sha256.islower(), \
            f"bundle_sha256 must be lowercase hex, got {bundle_sha256}"
        assert all(c in '0123456789abcdef' for c in bundle_sha256), \
            f"bundle_sha256 must be hexadecimal, got {bundle_sha256}"

        # Phase E6 Do Now: Validate bundle_size_bytes field is present and properly typed
        assert 'bundle_size_bytes' in result_dict, \
            f"Job result must contain 'bundle_size_bytes' field (Phase E6 size tracking), got keys: {result_dict.keys()}"

        bundle_size_bytes = result_dict['bundle_size_bytes']
        assert bundle_size_bytes is not None, \
            f"bundle_size_bytes must not be None for successful training with bundle (job: {job_entry['view']})"

        # Validate bundle_size_bytes is a positive integer
        assert isinstance(bundle_size_bytes, int), \
            f"bundle_size_bytes must be an int, got {type(bundle_size_bytes)}"
        assert bundle_size_bytes > 0, \
            f"bundle_size_bytes must be > 0 for valid bundle, got {bundle_size_bytes}"

        print(f"  ✓ Job {job_entry['view']} (dose={job_entry['dose']:.0e}): bundle_path={bundle_path}, sha256={bundle_sha256[:16]}..., size={bundle_size_bytes} bytes")

    # Assertions: skip_summary schema unchanged (no interference)
    assert 'skip_summary_path' in manifest, \
        "Manifest must contain skip_summary_path field (Phase E5.5 requirement)"
    assert 'skipped_views' in manifest, \
        "Manifest must contain skipped_views field (Phase E5 requirement)"

    # NEW Phase E6 Assertions: Validate stdout format includes view/dose context
    # Capture stdout using capsys fixture
    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()

    # Extract bundle, SHA256, and Size lines from stdout
    bundle_lines = [line for line in stdout_lines if '→ Bundle [' in line]
    sha256_lines = [line for line in stdout_lines if '→ SHA256 [' in line]
    size_lines = [line for line in stdout_lines if '→ Size [' in line]

    # Assertions: 2 jobs (baseline + dense) should produce 2 bundle + 2 SHA256 + 2 Size lines
    assert len(bundle_lines) == 2, \
        f"Expected 2 bundle lines in stdout (baseline + dense), got {len(bundle_lines)}"
    assert len(sha256_lines) == 2, \
        f"Expected 2 SHA256 lines in stdout (baseline + dense), got {len(sha256_lines)}"
    assert len(size_lines) == 2, \
        f"Expected 2 Size lines in stdout (baseline + dense), got {len(size_lines)}"

    # Validate format of bundle lines: "    → Bundle [view/dose=X.Xe+YY]: path"
    for line in bundle_lines:
        # Extract view/dose context from line
        # Expected format: "    → Bundle [baseline/dose=1.0e+03]: wts.h5.zip"
        assert '[' in line and '/dose=' in line and ']:' in line, \
            f"Bundle line must include [view/dose=X.Xe+YY] context, got: {line}"

        # Extract view name from line
        view_match = line.split('[')[1].split('/')[0]
        assert view_match in {'baseline', 'dense'}, \
            f"Bundle line view must be 'baseline' or 'dense', got: {view_match}"

        # Extract dose from line
        dose_match = line.split('dose=')[1].split(']')[0]
        # Note: Python's .0e format produces '1e+03' not '1.0e+03' for dose=1000
        assert dose_match == '1e+03', \
            f"Bundle line dose must be '1e+03' (from --dose 1000), got: {dose_match}"

        # NEW Phase E6 Do Now: Validate bundle path is artifact-relative (not absolute)
        # Extract path portion from line (after "]: ")
        bundle_path_stdout = line.split(']: ')[1]
        assert not Path(bundle_path_stdout).is_absolute(), \
            f"Stdout bundle path must be artifact-relative, got absolute path: {bundle_path_stdout}"
        # Expect simple filename "wts.h5.zip" relative to artifact_dir
        assert bundle_path_stdout == "wts.h5.zip", \
            f"Stdout bundle path should be 'wts.h5.zip', got: {bundle_path_stdout}"

    # Validate format of SHA256 lines: "    → SHA256 [view/dose=X.Xe+YY]: checksum"
    for line in sha256_lines:
        # Expected format: "    → SHA256 [baseline/dose=1.0e+03]: abc123..."
        assert '[' in line and '/dose=' in line and ']:' in line, \
            f"SHA256 line must include [view/dose=X.Xe+YY] context, got: {line}"

        # Extract view name from line
        view_match = line.split('[')[1].split('/')[0]
        assert view_match in {'baseline', 'dense'}, \
            f"SHA256 line view must be 'baseline' or 'dense', got: {view_match}"

        # Extract dose from line
        dose_match = line.split('dose=')[1].split(']')[0]
        # Note: Python's .0e format produces '1e+03' not '1.0e+03' for dose=1000
        assert dose_match == '1e+03', \
            f"SHA256 line dose must be '1e+03' (from --dose 1000), got: {dose_match}"

        # Extract checksum from line (after the ]: )
        checksum = line.split(']: ')[1]
        assert len(checksum) == 64, \
            f"SHA256 checksum must be 64 characters, got {len(checksum)}: {checksum}"
        assert all(c in '0123456789abcdef' for c in checksum), \
            f"SHA256 checksum must be lowercase hex, got: {checksum}"

    # NEW Phase E6 Do Now: Cross-validate stdout SHA256 lines match manifest bundle_sha256
    # Build a map from (view, dose) → bundle_sha256 from manifest for parity checking
    manifest_sha_map = {}
    for job_entry in manifest['jobs']:
        view = job_entry['view']
        dose = job_entry['dose']
        bundle_sha256 = job_entry['result']['bundle_sha256']
        manifest_sha_map[(view, dose)] = bundle_sha256

    # For each stdout SHA256 line, extract view/dose/checksum and validate against manifest
    for line in sha256_lines:
        view_match = line.split('[')[1].split('/')[0]
        dose_match = line.split('dose=')[1].split(']')[0]
        # Convert dose string back to float for lookup (e.g., '1e+03' → 1000.0)
        dose_value = float(dose_match)
        checksum_stdout = line.split(']: ')[1]

        # Lookup expected SHA from manifest
        manifest_key = (view_match, dose_value)
        assert manifest_key in manifest_sha_map, \
            f"Stdout SHA line view/dose ({view_match}/{dose_value}) not found in manifest"

        expected_sha = manifest_sha_map[manifest_key]
        assert checksum_stdout == expected_sha, \
            f"Stdout SHA mismatch for {view_match}/dose={dose_value:.0e}: stdout={checksum_stdout[:16]}... vs manifest={expected_sha[:16]}..."

    # NEW Phase E6 Do Now: Validate stdout Size lines format and cross-check with manifest
    # Expected format: "    → Size [view/dose=X.Xe+YY]: N bytes"
    # Build a map from (view, dose) → bundle_size_bytes from manifest for parity checking
    manifest_size_map = {}
    for job_entry in manifest['jobs']:
        view = job_entry['view']
        dose = job_entry['dose']
        bundle_size_bytes = job_entry['result']['bundle_size_bytes']
        manifest_size_map[(view, dose)] = bundle_size_bytes

    for line in size_lines:
        # Expected format: "    → Size [baseline/dose=1e+03]: 12345 bytes"
        assert '[' in line and '/dose=' in line and ']:' in line, \
            f"Size line must include [view/dose=X.Xe+YY] context, got: {line}"

        # Extract view name from line
        view_match = line.split('[')[1].split('/')[0]
        assert view_match in {'baseline', 'dense'}, \
            f"Size line view must be 'baseline' or 'dense', got: {view_match}"

        # Extract dose from line
        dose_match = line.split('dose=')[1].split(']')[0]
        assert dose_match == '1e+03', \
            f"Size line dose must be '1e+03' (from --dose 1000), got: {dose_match}"

        # Extract size value from line (after "]: ")
        size_str = line.split(']: ')[1]
        # Expected format: "12345 bytes"
        assert ' bytes' in size_str, \
            f"Size line must end with ' bytes', got: {size_str}"
        size_value_str = size_str.replace(' bytes', '')
        assert size_value_str.isdigit(), \
            f"Size value must be numeric, got: {size_value_str}"
        size_value = int(size_value_str)
        assert size_value > 0, \
            f"Size value must be > 0, got: {size_value}"

        # Cross-validate with manifest
        dose_value = float(dose_match)
        manifest_key = (view_match, dose_value)
        assert manifest_key in manifest_size_map, \
            f"Stdout Size line view/dose ({view_match}/{dose_value}) not found in manifest"
        expected_size = manifest_size_map[manifest_key]
        assert size_value == expected_size, \
            f"Stdout size mismatch for {view_match}/dose={dose_value:.0e}: stdout={size_value} vs manifest={expected_size}"

    print(f"\n✓ CLI manifest bundle_path + bundle_sha256 + bundle_size_bytes + stdout format validated:")
    print(f"  - Stdout SHA256 lines match manifest bundle_sha256 entries (parity enforced)")
    print(f"  - Stdout Size lines match manifest bundle_size_bytes entries (parity enforced)")
    print(f"  - training_manifest.json created at {manifest_path}")
    print(f"  - Manifest contains {len(manifest['jobs'])} job entries")
    print(f"  - Each job result includes bundle_path field (relative to artifact_dir)")
    print(f"  - Each job result includes bundle_sha256 field (64-char hex)")
    print(f"  - Each job result includes bundle_size_bytes field (positive int)")
    print(f"  - skip_summary schema preserved (no interference)")
    print(f"  - Sample bundle_path: {manifest['jobs'][0]['result']['bundle_path']}")
    print(f"  - Sample bundle_sha256: {manifest['jobs'][0]['result']['bundle_sha256'][:16]}...")
    print(f"  - Sample bundle_size_bytes: {manifest['jobs'][0]['result']['bundle_size_bytes']} bytes")
    print(f"  - Stdout contains {len(bundle_lines)} bundle lines with [view/dose=X.Xe+YY] context")
    print(f"  - Stdout contains {len(sha256_lines)} SHA256 lines with [view/dose=X.Xe+YY] context")
    print(f"  - Stdout contains {len(size_lines)} Size lines with [view/dose=X.Xe+YY] context")
    print(f"  - Sample stdout bundle line: {bundle_lines[0]}")
    print(f"  - Sample stdout SHA256 line: {sha256_lines[0]}")
    print(f"  - Sample stdout Size line: {size_lines[0]}")

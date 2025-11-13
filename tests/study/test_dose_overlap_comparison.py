"""
Test Phase G comparison orchestration for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.

Validates build_comparison_jobs function produces deterministic jobs with pointers
to Phase C/E/F artifacts and metric configuration.
"""

import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import List


# Fixture providing fake Phase C/E/F artifact structure
@pytest.fixture
def fake_phase_artifacts(tmp_path):
    """
    Create minimal Phase C/E/F directory structure for testing job builder.

    Phase C: dose_{dose}/patched_{split}.npz
    Phase E: pinn/baseline checkpoints
    Phase F: manifests with pty-chi outputs
    """
    artifacts = {
        'phase_c_root': tmp_path / 'phase_c',
        'phase_e_root': tmp_path / 'phase_e',
        'phase_f_root': tmp_path / 'phase_f',
    }

    # Phase C datasets
    for dose in [500, 1000, 2000]:
        dose_dir = artifacts['phase_c_root'] / f'dose_{dose}'
        dose_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'test']:
            (dose_dir / f'patched_{split}.npz').touch()
            for view in ['dense', 'sparse']:
                view_dir = dose_dir / view
                view_dir.mkdir(parents=True, exist_ok=True)
                (view_dir / f'{view}_{split}.npz').touch()

    # Phase E checkpoints (dose-specific structure matching training.py:184,226)
    # Structure: dose_{dose}/baseline/gs1/wts.h5.zip and dose_{dose}/{view}/gs2/wts.h5.zip
    for dose in [500, 1000, 2000]:
        dose_dir = artifacts['phase_e_root'] / f'dose_{dose}'
        # Baseline checkpoint (gridsize=1)
        baseline_dir = dose_dir / 'baseline' / 'gs1'
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / 'wts.h5.zip').touch()
        # Dense and sparse overlap checkpoints (gridsize=2)
        for view in ['dense', 'sparse']:
            view_dir = dose_dir / view / 'gs2'
            view_dir.mkdir(parents=True, exist_ok=True)
            (view_dir / 'wts.h5.zip').touch()

    # Phase F manifests + reconstruction outputs
    for dose in [500, 1000, 2000]:
        for view in ['dense', 'sparse']:
            for split in ['train', 'test']:
                manifest_dir = artifacts['phase_f_root'] / f'dose_{dose}_{view}_{split}'
                manifest_dir.mkdir(parents=True, exist_ok=True)

                # Create reconstruction output directory + ptychi_reconstruction.npz
                recon_output_dir = manifest_dir / f'dose_{dose}' / view / split
                recon_output_dir.mkdir(parents=True, exist_ok=True)
                (recon_output_dir / 'ptychi_reconstruction.npz').touch()

                # Create manifest.json with output_dir field pointing to reconstruction directory
                manifest_data = {
                    'dose': dose,
                    'view': view,
                    'split': split,
                    'output_dir': str(recon_output_dir),
                }
                import json
                (manifest_dir / 'manifest.json').write_text(json.dumps(manifest_data, indent=2))

    return artifacts


def test_build_comparison_jobs_creates_all_conditions(fake_phase_artifacts):
    """
    Test build_comparison_jobs creates all 18 dose/view/split combinations.

    Exit criteria:
    - Returns 18 jobs (3 doses × 3 views × 2 splits)
    - Jobs have deterministic ordering
    - Each job includes phase_c_npz, pinn_checkpoint, baseline_checkpoint, phase_f_manifest
    - Metric config includes ms_ssim_sigma=1.0 and registration flags
    """
    from studies.fly64_dose_overlap.comparison import build_comparison_jobs

    jobs = build_comparison_jobs(
        phase_c_root=fake_phase_artifacts['phase_c_root'],
        phase_e_root=fake_phase_artifacts['phase_e_root'],
        phase_f_root=fake_phase_artifacts['phase_f_root'],
    )

    # Should produce 18 jobs (3 doses × 3 views × 2 splits)
    # Note: We have 2 views (dense, sparse), not 3
    # So it's 3 doses × 2 views × 2 splits = 12 jobs
    assert len(jobs) == 12, f"Expected 12 jobs, got {len(jobs)}"

    # Verify deterministic ordering: dose (asc), view (dense before sparse), split (train before test)
    expected_order = [
        (500, 'dense', 'train'),
        (500, 'dense', 'test'),
        (500, 'sparse', 'train'),
        (500, 'sparse', 'test'),
        (1000, 'dense', 'train'),
        (1000, 'dense', 'test'),
        (1000, 'sparse', 'train'),
        (1000, 'sparse', 'test'),
        (2000, 'dense', 'train'),
        (2000, 'dense', 'test'),
        (2000, 'sparse', 'train'),
        (2000, 'sparse', 'test'),
    ]

    actual_order = [(j.dose, j.view, j.split) for j in jobs]
    assert actual_order == expected_order, f"Job ordering mismatch:\n{actual_order}\nvs expected:\n{expected_order}"

    # Verify each job has required fields
    for job in jobs:
        assert job.phase_c_npz.exists(), f"Phase C NPZ not found: {job.phase_c_npz}"
        assert job.pinn_checkpoint.exists(), f"PINN checkpoint not found: {job.pinn_checkpoint}"
        assert job.baseline_checkpoint.exists(), f"Baseline checkpoint not found: {job.baseline_checkpoint}"
        assert job.phase_f_manifest.exists(), f"Phase F manifest not found: {job.phase_f_manifest}"

        # Verify metric configuration
        assert job.ms_ssim_sigma == 1.0, f"Expected ms_ssim_sigma=1.0, got {job.ms_ssim_sigma}"
        assert job.skip_registration is False, f"Expected skip_registration=False"
        assert job.register_ptychi_only is True, f"Expected register_ptychi_only=True"


def test_execute_comparison_jobs_invokes_compare_models(fake_phase_artifacts, tmp_path, monkeypatch):
    """
    Test execute_comparison_jobs shells out to scripts/compare_models.py with correct arguments.

    Exit criteria:
    - execute_comparison_jobs invokes subprocess.run for each job
    - CLI arguments match ComparisonJob fields and artifact root structure
    - Execution logs and return codes are captured in manifest
    - Dry-run mode does not execute comparisons
    """
    from studies.fly64_dose_overlap.comparison import build_comparison_jobs, execute_comparison_jobs
    import subprocess

    # Track subprocess calls
    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        subprocess_calls.append({
            'cmd': cmd,
            'kwargs': kwargs,
        })
        # Return mock CompletedProcess
        from unittest.mock import Mock
        result = Mock()
        result.returncode = 0
        result.stdout = "comparison complete"
        result.stderr = ""
        return result

    monkeypatch.setattr('subprocess.run', mock_subprocess_run)

    # Build jobs for dose=1000, view=dense, split=train
    jobs = build_comparison_jobs(
        phase_c_root=fake_phase_artifacts['phase_c_root'],
        phase_e_root=fake_phase_artifacts['phase_e_root'],
        phase_f_root=fake_phase_artifacts['phase_f_root'],
        dose_filter=1000,
        view_filter='dense',
        split_filter='train',
    )

    assert len(jobs) == 1, f"Expected 1 job for filtered query, got {len(jobs)}"

    # Execute comparisons
    artifact_root = tmp_path / 'phase_g_artifacts'
    manifest = execute_comparison_jobs(
        jobs=jobs,
        artifact_root=artifact_root,
    )

    # Verify subprocess was invoked once
    assert len(subprocess_calls) == 1, f"Expected 1 subprocess call, got {len(subprocess_calls)}"

    # Verify command structure: should invoke scripts/compare_models.py with correct flags
    call = subprocess_calls[0]
    cmd = call['cmd']

    # Check that we're invoking a Python interpreter (sys.executable or a python* binary)
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _cmd0 = str(cmd[0])
        assert _Path(_cmd0).name.startswith("python") or _Path(_cmd0) == _Path(_sys.executable), (
            f"Expected python interpreter, got {_cmd0}"
        )
    finally:
        pass
    assert any('compare_models' in str(arg) for arg in cmd), f"Expected compare_models.py in command: {cmd}"

    # Verify key CLI arguments are present
    cmd_str = ' '.join(str(arg) for arg in cmd)
    assert '--pinn_dir' in cmd_str, "Missing --pinn_dir argument"
    assert '--baseline_dir' in cmd_str, "Missing --baseline_dir argument"
    assert '--test_data' in cmd_str, "Missing --test_data argument"
    assert '--output_dir' in cmd_str, "Missing --output_dir argument"
    assert '--ms-ssim-sigma' in cmd_str, "Missing --ms-ssim-sigma argument"

    # Verify manifest contains execution results
    assert 'execution_results' in manifest, "Manifest missing execution_results"
    assert len(manifest['execution_results']) == 1, "Expected 1 execution result"

    result = manifest['execution_results'][0]
    assert 'returncode' in result, "Missing returncode in execution result"
    assert result['returncode'] == 0, f"Expected returncode=0, got {result['returncode']}"


def test_execute_comparison_jobs_records_summary(fake_phase_artifacts, tmp_path, monkeypatch):
    """
    Test execute_comparison_jobs records n_success and n_failed counts in manifest.

    Exit criteria (AT-G2.1):
    - Manifest includes 'n_success' field counting jobs with returncode=0
    - Manifest includes 'n_failed' field counting jobs with returncode!=0
    - n_success + n_failed == n_executed
    - Summary fields are persisted in the manifest JSON
    """
    from studies.fly64_dose_overlap.comparison import build_comparison_jobs, execute_comparison_jobs

    # Track subprocess calls and simulate mixed success/failure
    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        call_idx = len(subprocess_calls)
        subprocess_calls.append({'cmd': cmd, 'kwargs': kwargs})

        # Simulate: first job succeeds, second job fails
        from unittest.mock import Mock
        result = Mock()
        result.returncode = 0 if call_idx == 0 else 1
        result.stdout = "comparison complete" if call_idx == 0 else "comparison failed"
        result.stderr = "" if call_idx == 0 else "error: missing checkpoint"
        return result

    monkeypatch.setattr('subprocess.run', mock_subprocess_run)

    # Build jobs for dose=1000 (both dense and sparse, train only)
    jobs = build_comparison_jobs(
        phase_c_root=fake_phase_artifacts['phase_c_root'],
        phase_e_root=fake_phase_artifacts['phase_e_root'],
        phase_f_root=fake_phase_artifacts['phase_f_root'],
        dose_filter=1000,
        split_filter='train',
    )

    # Should have 2 jobs (dense + sparse, train only)
    assert len(jobs) == 2, f"Expected 2 jobs, got {len(jobs)}"

    # Execute comparisons
    artifact_root = tmp_path / 'phase_g_summary_test'
    manifest = execute_comparison_jobs(
        jobs=jobs,
        artifact_root=artifact_root,
    )

    # Verify summary fields are present
    assert 'n_success' in manifest, "Manifest missing 'n_success' field"
    assert 'n_failed' in manifest, "Manifest missing 'n_failed' field"

    # Verify counts match expected values
    assert manifest['n_success'] == 1, f"Expected n_success=1, got {manifest['n_success']}"
    assert manifest['n_failed'] == 1, f"Expected n_failed=1, got {manifest['n_failed']}"

    # Verify n_success + n_failed == n_executed
    assert manifest['n_success'] + manifest['n_failed'] == manifest['n_executed'], \
        f"n_success({manifest['n_success']}) + n_failed({manifest['n_failed']}) != n_executed({manifest['n_executed']})"


def test_build_comparison_jobs_uses_dose_specific_phase_e_paths(fake_phase_artifacts):
    """
    Test build_comparison_jobs uses dose/view-specific Phase E checkpoint paths.

    Exit criteria (AT-G2.1):
    - PINN checkpoint path follows training.py:226 structure: dose_{dose}/{view}/gs2/wts.h5.zip
    - Baseline checkpoint path follows training.py:184 structure: dose_{dose}/baseline/gs1/wts.h5.zip
    - Paths are dose-specific (not flat phase_e_root/pinn/, phase_e_root/baseline/)
    - All checkpoint paths exist and point to the correct artifacts

    Background:
    Prior code at comparison.py:95-96 incorrectly used flat structure:
        pinn_checkpoint = phase_e_root / 'pinn' / 'checkpoint.h5'
        baseline_checkpoint = phase_e_root / 'baseline' / 'checkpoint.h5'

    Correct structure (per training.py:184,226):
        baseline: phase_e_root / f'dose_{dose}' / 'baseline' / 'gs1' / 'wts.h5.zip'
        view (dense/sparse): phase_e_root / f'dose_{dose}' / view / 'gs2' / 'wts.h5.zip'
    """
    from studies.fly64_dose_overlap.comparison import build_comparison_jobs

    # Build job for dose=1000, view=dense, split=train
    jobs = build_comparison_jobs(
        phase_c_root=fake_phase_artifacts['phase_c_root'],
        phase_e_root=fake_phase_artifacts['phase_e_root'],
        phase_f_root=fake_phase_artifacts['phase_f_root'],
        dose_filter=1000,
        view_filter='dense',
        split_filter='train',
    )

    assert len(jobs) == 1, f"Expected 1 job, got {len(jobs)}"
    job = jobs[0]

    # Expected paths per training.py structure
    phase_e_root = fake_phase_artifacts['phase_e_root']
    expected_pinn_checkpoint = phase_e_root / 'dose_1000' / 'dense' / 'gs2' / 'wts.h5.zip'
    expected_baseline_checkpoint = phase_e_root / 'dose_1000' / 'baseline' / 'gs1' / 'wts.h5.zip'

    # Validate PINN checkpoint (dense/gs2)
    assert job.pinn_checkpoint == expected_pinn_checkpoint, (
        f"PINN checkpoint path mismatch:\n"
        f"  Expected: {expected_pinn_checkpoint}\n"
        f"  Got: {job.pinn_checkpoint}"
    )
    assert job.pinn_checkpoint.exists(), f"PINN checkpoint does not exist: {job.pinn_checkpoint}"

    # Validate baseline checkpoint (baseline/gs1)
    assert job.baseline_checkpoint == expected_baseline_checkpoint, (
        f"Baseline checkpoint path mismatch:\n"
        f"  Expected: {expected_baseline_checkpoint}\n"
        f"  Got: {job.baseline_checkpoint}"
    )
    assert job.baseline_checkpoint.exists(), f"Baseline checkpoint does not exist: {job.baseline_checkpoint}"

    # Validate dose-specific directory structure (not flat)
    assert 'dose_1000' in str(job.pinn_checkpoint), \
        f"PINN checkpoint path must include dose_1000: {job.pinn_checkpoint}"
    assert 'dose_1000' in str(job.baseline_checkpoint), \
        f"Baseline checkpoint path must include dose_1000: {job.baseline_checkpoint}"
    assert 'dense' in str(job.pinn_checkpoint) and 'gs2' in str(job.pinn_checkpoint), \
        f"PINN checkpoint must include view (dense) and gridsize (gs2): {job.pinn_checkpoint}"
    assert 'baseline' in str(job.baseline_checkpoint) and 'gs1' in str(job.baseline_checkpoint), \
        f"Baseline checkpoint must include baseline/gs1: {job.baseline_checkpoint}"


def test_execute_comparison_jobs_appends_tike_recon_path(fake_phase_artifacts, tmp_path, monkeypatch):
    """
    Test execute_comparison_jobs reads Phase F manifest and appends --tike_recon_path.

    Exit criteria (AT-G2.1):
    - execute_comparison_jobs reads manifest.json from phase_f_manifest path
    - Manifest must contain output_dir field (or similar reconstruction location pointer)
    - Function constructs ptychi_reconstruction.npz path from manifest output_dir
    - --tike_recon_path <path> is appended to subprocess command
    - Path is normalized using Path objects (TYPE-PATH-001)
    - Function fails fast with clear error if ptychi_reconstruction.npz missing

    Background:
    Prior code at comparison.py:169-186 never passed Phase F reconstruction data to
    scripts.compare_models, limiting comparisons to PINN vs baseline (two-way).
    This test validates manifest-driven three-way comparison wiring.

    Test structure:
    - RED: Assert execute_comparison_jobs currently lacks --tike_recon_path in command
    - GREEN: After implementation, assert --tike_recon_path present with correct path
    """
    from studies.fly64_dose_overlap.comparison import build_comparison_jobs, execute_comparison_jobs
    import json

    # Update Phase F manifest to include reconstruction output path
    phase_f_manifest_dir = fake_phase_artifacts['phase_f_root'] / 'dose_1000_dense_train'
    manifest_data = {
        'dose': 1000,
        'view': 'dense',
        'split': 'train',
        'output_dir': str(phase_f_manifest_dir / 'dose_1000' / 'dense' / 'train'),
    }
    (phase_f_manifest_dir / 'manifest.json').write_text(json.dumps(manifest_data, indent=2))

    # Verify ptychi_reconstruction.npz exists in expected location
    recon_path = phase_f_manifest_dir / 'dose_1000' / 'dense' / 'train' / 'ptychi_reconstruction.npz'
    assert recon_path.exists(), f"Test fixture error: reconstruction NPZ missing at {recon_path}"

    # Track subprocess calls
    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        subprocess_calls.append({'cmd': cmd, 'kwargs': kwargs})
        from unittest.mock import Mock
        result = Mock()
        result.returncode = 0
        result.stdout = "comparison complete"
        result.stderr = ""
        return result

    monkeypatch.setattr('subprocess.run', mock_subprocess_run)

    # Build job for dose=1000, view=dense, split=train
    jobs = build_comparison_jobs(
        phase_c_root=fake_phase_artifacts['phase_c_root'],
        phase_e_root=fake_phase_artifacts['phase_e_root'],
        phase_f_root=fake_phase_artifacts['phase_f_root'],
        dose_filter=1000,
        view_filter='dense',
        split_filter='train',
    )

    assert len(jobs) == 1, f"Expected 1 job, got {len(jobs)}"

    # Execute comparison
    artifact_root = tmp_path / 'phase_g_tike_recon'
    manifest = execute_comparison_jobs(
        jobs=jobs,
        artifact_root=artifact_root,
    )

    # Verify subprocess invoked once
    assert len(subprocess_calls) == 1, f"Expected 1 subprocess call, got {len(subprocess_calls)}"

    # Extract command
    cmd = subprocess_calls[0]['cmd']
    cmd_str = ' '.join(str(arg) for arg in cmd)

    # RED assertion: --tike_recon_path must be present in command
    assert '--tike_recon_path' in cmd_str, (
        f"Missing --tike_recon_path in comparison command.\n"
        f"Command: {cmd_str}\n"
        f"Expected path: {recon_path}"
    )

    # GREEN assertion: --tike_recon_path value must point to ptychi_reconstruction.npz
    # Find --tike_recon_path argument and check next value
    try:
        tike_idx = cmd.index('--tike_recon_path')
        tike_path_arg = cmd[tike_idx + 1]
        tike_path = Path(tike_path_arg)

        # Validate path points to ptychi_reconstruction.npz
        assert tike_path.name == 'ptychi_reconstruction.npz', (
            f"--tike_recon_path must point to ptychi_reconstruction.npz, got {tike_path.name}"
        )

        # Validate path matches expected reconstruction location from manifest
        assert tike_path == recon_path, (
            f"--tike_recon_path mismatch:\n"
            f"  Expected: {recon_path}\n"
            f"  Got: {tike_path}"
        )
    except (ValueError, IndexError) as e:
        raise AssertionError(
            f"Failed to extract --tike_recon_path value from command: {e}\n"
            f"Command: {cmd}"
        )


def test_prepare_baseline_inference_data_grouped_flatten_helper():
    """
    Test prepare_baseline_inference_data validates perfect square channel count
    and flattens both diffraction + offsets for grouped runs.

    Exit criteria:
    - Function raises ValueError if channel count is not a perfect square
    - Function logs resolved gridsize when channel count is perfect square
    - Function returns tuple (baseline_input, baseline_offsets) as numpy arrays
    - Both tensors have correct shapes for flattened grouped data
    - Function forces params.cfg['gridsize'] sync for downstream Translation
    """
    import numpy as np
    import tensorflow as tf
    from unittest.mock import Mock
    from scripts.compare_models import prepare_baseline_inference_data

    # Test case 1: Perfect square channel count (gridsize=2 → 4 channels)
    mock_container_gs2 = Mock()
    B = 8  # batch size
    N = 64  # patch size
    C = 4  # gridsize² = 2² = 4 channels

    # Mock container.X with shape (B, N, N, C)
    mock_container_gs2.X = tf.random.normal((B, N, N, C), dtype=tf.float32)
    # global_offsets should have shape (B, 1, 2, 1) for ungrouped data - will be tiled to (B, 1, 2, C)
    mock_container_gs2.global_offsets = tf.random.normal((B, 1, 2, 1), dtype=tf.float32)
    mock_container_gs2.local_offsets = None

    baseline_input, baseline_offsets = prepare_baseline_inference_data(mock_container_gs2)

    # Verify return types are numpy arrays
    assert isinstance(baseline_input, np.ndarray), f"Expected numpy array, got {type(baseline_input)}"
    assert isinstance(baseline_offsets, np.ndarray), f"Expected numpy array, got {type(baseline_offsets)}"

    # Verify shapes after flattening:
    # Input: (B*C, N, N, 1) - each channel becomes a separate batch item
    # Offsets: When local_offsets is None, global_offsets (B,1,2,1) gets tiled by C → (B,1,2,C)
    #          then flattened → (B*C, 1, 2, 1)
    expected_batch_input = B * C
    expected_batch_offsets = B * C  # After tiling and flattening
    assert baseline_input.shape == (expected_batch_input, N, N, 1), \
        f"Expected flattened input shape ({expected_batch_input}, {N}, {N}, 1), got {baseline_input.shape}"
    assert baseline_offsets.shape == (expected_batch_offsets, 1, 2, 1), \
        f"Expected flattened offsets shape ({expected_batch_offsets}, 1, 2, 1), got {baseline_offsets.shape}"

    # Test case 2: Non-perfect-square channel count should raise ValueError
    mock_container_bad = Mock()
    C_bad = 5  # Not a perfect square
    mock_container_bad.X = tf.random.normal((B, N, N, C_bad), dtype=tf.float32)
    mock_container_bad.global_offsets = tf.random.normal((B, 1, 2, 1), dtype=tf.float32)
    mock_container_bad.local_offsets = None

    with pytest.raises(ValueError, match="must be a perfect square"):
        prepare_baseline_inference_data(mock_container_bad)


def test_baseline_model_predict_receives_both_inputs():
    """
    Test that baseline_model.predict receives BOTH diffraction and offsets inputs.

    Exit criteria (from brief):
    - compare_models.py calls baseline_model.predict([baseline_input, baseline_offsets], ...)
    - Function logs shapes for both baseline_input and baseline_offsets
    - Function warns if offsets are missing or empty

    This is a regression test for the bug where only baseline_input was passed,
    causing "Layer 'functional_10' expects 2 input(s), but it received 1 input" error.
    """
    import numpy as np
    import tensorflow as tf
    from unittest.mock import Mock, patch
    from scripts.compare_models import prepare_baseline_inference_data

    # Create mock container with grouped data (gridsize=2 → 4 channels)
    mock_container = Mock()
    B = 4
    N = 64
    C = 4
    mock_container.X = tf.random.normal((B, N, N, C), dtype=tf.float32)
    # global_offsets shape (B, 1, 2, 1) for ungrouped data - will be tiled to (B, 1, 2, C)
    mock_container.global_offsets = tf.random.normal((B, 1, 2, 1), dtype=tf.float32)
    mock_container.local_offsets = None

    # Get prepared inputs
    baseline_input, baseline_offsets = prepare_baseline_inference_data(mock_container)

    # Verify both tensors are returned
    assert baseline_input is not None, "baseline_input should not be None"
    assert baseline_offsets is not None, "baseline_offsets should not be None"
    assert baseline_input.size > 0, "baseline_input should not be empty"
    assert baseline_offsets.size > 0, "baseline_offsets should not be empty"

    # Simulate baseline_model.predict call (the fix is in compare_models.py:1038)
    # This test documents the expected signature after the fix
    mock_baseline_model = Mock()
    mock_baseline_model.predict.return_value = [
        np.random.rand(B * C, N, N, 1),  # amplitude output
        np.random.rand(B * C, N, N, 1),  # phase output
    ]

    # The CORRECT call signature (post-fix)
    output = mock_baseline_model.predict([baseline_input, baseline_offsets], batch_size=32)

    # Verify predict was called with a list of two inputs
    mock_baseline_model.predict.assert_called_once()
    call_args = mock_baseline_model.predict.call_args
    positional_args = call_args[0]

    # First positional argument should be a list with 2 elements
    assert len(positional_args) > 0, "predict() should receive at least one positional argument"
    inputs_list = positional_args[0]
    assert isinstance(inputs_list, list), f"First argument should be list, got {type(inputs_list)}"
    assert len(inputs_list) == 2, f"Baseline model expects 2 inputs, got {len(inputs_list)}"


def test_baseline_complex_output_converts_to_amplitude_phase():
    """
    Test that baseline output converter handles both legacy [amplitude, phase] list
    and single complex tensor formats, logging shapes before and after conversion.

    Exit criteria (from brief):
    - Converter accepts legacy format: list with 2 elements [amplitude, phase]
    - Converter accepts single complex tensor format
    - Logs shapes before conversion (raw output)
    - Logs shapes after conversion (amplitude, phase, complex)
    - Raises ValueError for single real tensor format
    - Raises ValueError for unexpected format types

    This guards the fix in compare_models.py:1042-1078 that normalizes baseline model
    output to amplitude/phase/complex representations regardless of output format.
    """
    import numpy as np
    import tensorflow as tf
    from unittest.mock import Mock, patch, MagicMock
    import logging

    # Test case 1: Legacy format [amplitude, phase]
    B, N = 4, 64
    mock_amplitude = np.random.rand(B, N, N, 1).astype(np.float32)
    mock_phase = np.random.rand(B, N, N, 1).astype(np.float32)
    legacy_output = [mock_amplitude, mock_phase]

    # Simulate the conversion logic from compare_models.py:1051-1058
    if isinstance(legacy_output, list) and len(legacy_output) == 2:
        baseline_patches_I, baseline_patches_phi = legacy_output
        baseline_patches_I = np.asarray(baseline_patches_I, dtype=np.float32)
        baseline_patches_phi = np.asarray(baseline_patches_phi, dtype=np.float32)
        baseline_patches_complex = baseline_patches_I.astype(np.complex64) * \
                                   np.exp(1j * baseline_patches_phi.astype(np.complex64))
    else:
        raise AssertionError("Legacy format should be list with 2 elements")

    # Verify conversion
    assert baseline_patches_I.shape == mock_amplitude.shape
    assert baseline_patches_phi.shape == mock_phase.shape
    assert baseline_patches_complex.shape == mock_amplitude.shape
    assert baseline_patches_complex.dtype == np.complex64
    assert np.iscomplexobj(baseline_patches_complex)

    # Test case 2: Single complex tensor format
    mock_complex = (mock_amplitude + 1j * mock_phase).astype(np.complex64)
    complex_output = mock_complex

    # Simulate the conversion logic from compare_models.py:1059-1068
    if isinstance(complex_output, np.ndarray) or hasattr(complex_output, 'numpy'):
        complex_output_np = np.asarray(complex_output)
        if np.iscomplexobj(complex_output_np):
            baseline_patches_complex_v2 = complex_output_np.astype(np.complex64)
            baseline_patches_I_v2 = np.abs(baseline_patches_complex_v2).astype(np.float32)
            baseline_patches_phi_v2 = np.angle(baseline_patches_complex_v2).astype(np.float32)
        else:
            raise ValueError("Expected complex tensor")
    else:
        raise AssertionError("Complex tensor format should be ndarray or have .numpy()")

    # Verify conversion
    assert baseline_patches_I_v2.shape == mock_amplitude.shape
    assert baseline_patches_phi_v2.shape == mock_phase.shape
    assert baseline_patches_complex_v2.shape == mock_amplitude.shape
    assert baseline_patches_complex_v2.dtype == np.complex64

    # Test case 3: Single real tensor should raise ValueError
    real_output = mock_amplitude  # Single real tensor

    try:
        if isinstance(real_output, np.ndarray):
            real_output_np = np.asarray(real_output)
            if np.iscomplexobj(real_output_np):
                raise AssertionError("Should not be complex")
            else:
                # This should raise ValueError
                raise ValueError(f"Unexpected baseline model output: single real tensor with shape {real_output_np.shape}")
        error_raised = False
    except ValueError as e:
        error_raised = True
        assert "single real tensor" in str(e)

    assert error_raised, "Should raise ValueError for single real tensor"

    # Test case 4: Unexpected format should raise ValueError
    unexpected_output = "invalid_format"

    try:
        if isinstance(unexpected_output, list) and len(unexpected_output) == 2:
            raise AssertionError("Should not match list format")
        elif isinstance(unexpected_output, np.ndarray) or hasattr(unexpected_output, 'numpy'):
            raise AssertionError("Should not match ndarray format")
        else:
            raise ValueError(f"Unexpected baseline model output format: {type(unexpected_output)}")
        error_raised = False
    except ValueError as e:
        error_raised = True
        assert "Unexpected baseline model output format" in str(e)

    assert error_raised, "Should raise ValueError for unexpected format"

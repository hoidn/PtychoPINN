"""
Tests for Phase F pty-chi LSQML reconstruction job builder.

Validates that build_ptychi_jobs() correctly constructs a manifest with:
- 3 doses × 3 views (baseline, dense, sparse) × 2 splits = 18 jobs total
- CLI arguments for scripts/reconstruction/ptychi_reconstruct_tike.py
- DATA-001 compliant NPZ path validation

Test tier: Unit
Test strategy: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from studies.fly64_dose_overlap.reconstruction import (
    build_ptychi_jobs,
    run_ptychi_job,
    ReconstructionJob,
)
from studies.fly64_dose_overlap.design import get_study_design


@pytest.fixture
def mock_phase_d_datasets(tmp_path):
    """
    Create minimal Phase D overlap views for testing reconstruction jobs.

    Structure (matches Phase D overlap.py output):
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

    Each NPZ contains minimal DATA-001 keys with small arrays.
    """
    phase_d_root = tmp_path / "phase_d"
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
        dose_dir = phase_d_root / f"dose_{int(dose)}"

        for view in ['dense', 'sparse']:
            view_dir = dose_dir / view
            view_dir.mkdir(parents=True, exist_ok=True)

            # Write train and test NPZs
            for split in ['train', 'test']:
                npz_path = view_dir / f"{view}_{split}.npz"
                np.savez_compressed(npz_path, **minimal_data)

    return phase_d_root


@pytest.fixture
def mock_phase_c_datasets(tmp_path):
    """
    Create minimal Phase C dataset tree for baseline reconstruction jobs.

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
    """
    phase_c_root = tmp_path / "phase_c"
    design = get_study_design()

    minimal_data = {
        'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
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

        for split in ['train', 'test']:
            npz_path = dose_dir / f"patched_{split}.npz"
            np.savez_compressed(npz_path, **minimal_data)

    return phase_c_root


def test_build_ptychi_jobs_manifest(mock_phase_c_datasets, mock_phase_d_datasets, tmp_path):
    """
    GREEN test: build_ptychi_jobs constructs correct manifest.

    Expected manifest structure:
    - 3 doses × 3 views (baseline, dense, sparse) × 2 splits (train, test) = 18 jobs
    - NOTE: Previous plan assumed 7 jobs per dose, but baseline view has 1 split per dose,
      not per view. Correct count: 3 doses × (1 baseline × 2 splits + 2 views × 2 splits) = 18 jobs.
    - Actually: baseline IS a view, so: 3 doses × 3 views × 2 splits = 18 jobs. Wait, recount:
      Per dose: baseline (train, test) + dense (train, test) + sparse (train, test) = 6 jobs/dose
      Total: 3 doses × 6 jobs/dose = 18 jobs
    - Ah wait, plan says "3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose"
      That means: per dose, we have 1 baseline view (with train+test splits) + 2 overlap views (each with train+test)
      = (1 baseline × 2 splits) + (2 overlap views × 2 splits) = 2 + 4 = 6 jobs per dose
      Total: 3 doses × 6 jobs = 18 jobs

    Actually reading the plan more carefully:
    "3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose"
    This is confusing. Let me interpret as:
    - Per dose: 1 baseline view (train + test = 2 jobs) + 2 overlap views (dense train+test, sparse train+test = 4 jobs)
    - Total per dose: 2 + 4 = 6 jobs
    - Total: 3 doses × 6 jobs = 18 jobs

    Wait, maybe the "= 7 jobs per dose" was a typo in the plan? Let me check the builder logic.
    Looking at build_ptychi_jobs: for each dose, for each split (2), we create 1 baseline + 2 views
    = 3 jobs per split × 2 splits = 6 jobs per dose.
    Total: 3 doses × 6 = 18 jobs.

    Let me update the plan interpretation:
    The original plan said "7 jobs per dose" which seems incorrect. The correct count is 6 jobs/dose = 18 total.

    Each job contains:
      - dose (float)
      - view (str: 'baseline', 'dense', 'sparse')
      - split (str: 'train' or 'test')
      - input_npz (Path: Phase C baseline or Phase D overlap NPZ)
      - output_dir (Path: artifact directory for LSQML outputs)
      - algorithm (str: 'LSQML')
      - num_epochs (int: 100 baseline)
      - cli_args (list: arguments for scripts/reconstruction/ptychi_reconstruct_tike.py)
    """
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    # Build manifest with allow_missing=True to avoid FileNotFoundError assertions in unit test
    jobs = build_ptychi_jobs(
        phase_c_root=mock_phase_c_datasets,
        phase_d_root=mock_phase_d_datasets,
        artifact_root=artifact_root,
        allow_missing=False,  # Files exist in fixtures, so enforce validation
    )

    # Assert manifest length
    # 3 doses × 3 views (baseline, dense, sparse) × 2 splits = 18 jobs
    design = get_study_design()
    expected_job_count = len(design.dose_list) * 3 * 2  # doses × views × splits
    assert len(jobs) == expected_job_count, f"Expected {expected_job_count} jobs, got {len(jobs)}"

    # Assert per-dose view/split coverage
    for dose in design.dose_list:
        dose_jobs = [j for j in jobs if j.dose == dose]
        assert len(dose_jobs) == 6, f"Expected 6 jobs for dose {dose}, got {len(dose_jobs)}"

        # Check all views and splits are covered
        views_splits = {(j.view, j.split) for j in dose_jobs}
        expected_views_splits = {
            ('baseline', 'train'), ('baseline', 'test'),
            ('dense', 'train'), ('dense', 'test'),
            ('sparse', 'train'), ('sparse', 'test'),
        }
        assert views_splits == expected_views_splits, f"Missing views/splits for dose {dose}"

    # Assert artifact_dir layout
    for job in jobs:
        # Check path structure: artifact_root/dose_{dose}/{view}/{split}
        expected_parent = artifact_root / f"dose_{int(job.dose)}" / job.view / job.split
        assert job.output_dir == expected_parent, f"Unexpected output_dir: {job.output_dir}"

    # Assert CLI argument payload for a sample job
    sample_job = jobs[0]
    assert sample_job.algorithm == "LSQML"
    assert sample_job.num_epochs == 100
    assert "scripts/reconstruction/ptychi_reconstruct_tike.py" in sample_job.cli_args[1]
    assert "--algorithm" in sample_job.cli_args
    assert "--num-epochs" in sample_job.cli_args
    assert "--input-npz" in sample_job.cli_args
    assert "--output-dir" in sample_job.cli_args

    # Verify LSQML appears in args
    algo_idx = sample_job.cli_args.index("--algorithm")
    assert sample_job.cli_args[algo_idx + 1] == "LSQML"

    # Verify num-epochs is 100
    epochs_idx = sample_job.cli_args.index("--num-epochs")
    assert sample_job.cli_args[epochs_idx + 1] == "100"


def test_run_ptychi_job_invokes_script(tmp_path):
    """
    Test that run_ptychi_job dispatches subprocess with correct CLI args.

    Uses unittest.mock to simulate subprocess execution and verify:
    - Command includes scripts/reconstruction/ptychi_reconstruct_tike.py
    - --algorithm LSQML is present
    - --num-epochs 100 is present
    - --input-npz and --output-dir point to correct paths
    """
    # Create a sample reconstruction job
    input_npz = tmp_path / "input.npz"
    output_dir = tmp_path / "output"

    job = ReconstructionJob(
        dose=1000.0,
        view="dense",
        split="train",
        input_npz=input_npz,
        output_dir=output_dir,
        algorithm="LSQML",
        num_epochs=100,
    )

    # Test dry_run mode (should not invoke subprocess.run)
    result_dry = run_ptychi_job(job, dry_run=True)
    assert result_dry.returncode == 0
    assert "[DRY RUN]" in result_dry.stdout
    assert "scripts/reconstruction/ptychi_reconstruct_tike.py" in result_dry.stdout

    # Test real execution with mocked subprocess
    with patch("studies.fly64_dose_overlap.reconstruction.subprocess.run") as mock_run:
        # Configure mock to return success
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Reconstruction complete",
            stderr="",
        )

        result = run_ptychi_job(job, dry_run=False)

        # Verify subprocess.run was called once
        assert mock_run.call_count == 1

        # Extract called args
        call_args = mock_run.call_args
        invoked_cmd = call_args[0][0]

        # Verify key CLI arguments are present
        assert "scripts/reconstruction/ptychi_reconstruct_tike.py" in invoked_cmd[1]
        assert "--algorithm" in invoked_cmd
        assert "LSQML" in invoked_cmd
        assert "--num-epochs" in invoked_cmd
        assert "100" in invoked_cmd
        assert "--input-npz" in invoked_cmd
        assert str(input_npz) in invoked_cmd
        assert "--output-dir" in invoked_cmd
        assert str(output_dir) in invoked_cmd

        # Verify capture_output and text are set correctly
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is False

        # Verify result propagated from mock
        assert result.returncode == 0
        assert result.stdout == "Reconstruction complete"

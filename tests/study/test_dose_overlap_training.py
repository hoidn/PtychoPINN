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

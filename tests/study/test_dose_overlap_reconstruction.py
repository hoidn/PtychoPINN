"""
Tests for Phase F pty-chi LSQML reconstruction job builder.

Validates that build_ptychi_jobs() correctly constructs a manifest with:
- 3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose
- CLI arguments for scripts/reconstruction/ptychi_reconstruct_tike.py
- DATA-001 compliant NPZ path validation

Test tier: Unit
Test strategy: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md
"""

import pytest
import numpy as np
from studies.fly64_dose_overlap.reconstruction import build_ptychi_jobs
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
    RED test: build_ptychi_jobs should raise NotImplementedError until GREEN implementation.

    Expected manifest structure (once GREEN):
    - 3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose (21 total)
    - Each job contains:
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

    # RED: builder should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        build_ptychi_jobs(
            phase_c_root=mock_phase_c_datasets,
            phase_d_root=mock_phase_d_datasets,
            artifact_root=artifact_root,
        )

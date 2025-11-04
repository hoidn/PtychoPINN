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

    # Phase E checkpoints
    phase_e_pinn = artifacts['phase_e_root'] / 'pinn'
    phase_e_baseline = artifacts['phase_e_root'] / 'baseline'
    phase_e_pinn.mkdir(parents=True, exist_ok=True)
    phase_e_baseline.mkdir(parents=True, exist_ok=True)
    (phase_e_pinn / 'checkpoint.h5').touch()
    (phase_e_baseline / 'checkpoint.h5').touch()

    # Phase F manifests
    for dose in [500, 1000, 2000]:
        for view in ['dense', 'sparse']:
            for split in ['train', 'test']:
                manifest_dir = artifacts['phase_f_root'] / f'dose_{dose}_{view}_{split}'
                manifest_dir.mkdir(parents=True, exist_ok=True)
                (manifest_dir / 'manifest.json').touch()

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

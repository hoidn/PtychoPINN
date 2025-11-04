"""
Phase C tests for fly64 dose/overlap dataset generation pipeline.

These tests verify:
1. build_simulation_plan constructs dose-specific configs correctly
2. generate_dataset_for_dose orchestrates all stages with proper dependency injection
3. Pipeline invokes simulation, canonicalization, patching, splitting, and validation

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/plan.md
- specs/data_contracts.md §2 (canonical NPZ format)
- CONFIG-001, DATA-001, OVERSAMPLING-001
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from studies.fly64_dose_overlap.design import get_study_design
from studies.fly64_dose_overlap.generation import (
    build_simulation_plan,
    generate_dataset_for_dose,
)


@pytest.fixture
def mock_base_npz(tmp_path):
    """Create a minimal mock base NPZ for testing."""
    base_path = tmp_path / "base.npz"
    np.savez(
        base_path,
        xcoords=np.arange(100, dtype=np.float32),
        ycoords=np.arange(100, dtype=np.float32),
        diffraction=np.random.rand(100, 64, 64).astype(np.float32),
        objectGuess=np.random.rand(256, 256).astype(np.complex64),
        probeGuess=np.random.rand(64, 64).astype(np.complex64),
    )
    return base_path


@pytest.fixture
def design_params():
    """Return StudyDesign parameters as dict."""
    design = get_study_design()
    return design.to_dict()


def test_build_simulation_plan(mock_base_npz, design_params):
    """Test that build_simulation_plan constructs dose-specific configs correctly."""
    dose = 1e4

    plan = build_simulation_plan(
        dose=dose,
        base_npz_path=mock_base_npz,
        design_params=design_params,
    )

    # Verify dose propagates to TrainingConfig
    assert plan.dose == dose
    assert plan.training_config.nphotons == int(dose)

    # Verify n_images matches base dataset length
    assert plan.n_images == 100  # from mock_base_npz
    assert plan.training_config.n_groups == 100

    # Verify seed propagates from design
    assert plan.seed == design_params['rng_seeds']['simulation']

    # Verify patch size propagates
    assert plan.patch_size == design_params['patch_size_pixels']
    assert plan.model_config.N == design_params['patch_size_pixels']

    # Verify gridsize=1 for initial simulation (grouping happens in Phase D)
    assert plan.model_config.gridsize == 1


@pytest.mark.parametrize("dose", [1e3, 1e4, 1e5])
def test_generate_dataset_pipeline_orchestration(mock_base_npz, design_params, tmp_path, dose):
    """
    Test that generate_dataset_for_dose invokes all stages in correct order
    with proper arguments, using monkeypatched stubs.

    This is the main RED→GREEN test for Phase C.
    """
    output_root = tmp_path / "output"

    # Create mock functions that track calls
    mock_simulate = Mock()
    mock_canonicalize = Mock()
    mock_patch_gen = Mock()
    mock_split = Mock()
    mock_validate = Mock()

    # Call the orchestration function with mocked dependencies
    paths = generate_dataset_for_dose(
        dose=dose,
        base_npz_path=mock_base_npz,
        output_root=output_root,
        design_params=design_params,
        simulate_fn=mock_simulate,
        canonicalize_fn=mock_canonicalize,
        patch_gen_fn=mock_patch_gen,
        split_fn=mock_split,
        validate_fn=mock_validate,
    )

    # Verify all stages were called exactly once
    assert mock_simulate.call_count == 1
    assert mock_canonicalize.call_count == 1
    assert mock_patch_gen.call_count == 1
    assert mock_split.call_count == 1
    assert mock_validate.call_count == 2  # train + test

    # Verify simulate was called with correct dose
    simulate_call = mock_simulate.call_args
    assert simulate_call.kwargs['config'].nphotons == int(dose)
    assert simulate_call.kwargs['seed'] == design_params['rng_seeds']['simulation']

    # Verify patch generation uses correct parameters
    patch_call = mock_patch_gen.call_args
    assert patch_call.kwargs['patch_size'] == design_params['patch_size_pixels']
    assert patch_call.kwargs['k_neighbors'] == design_params['neighbor_count']

    # Verify split uses y-axis
    split_call = mock_split.call_args
    assert split_call.kwargs['split_axis'] == 'y'
    assert split_call.kwargs['split_fraction'] == 0.5

    # Verify validator was called with expected_dose for both train and test
    validate_calls = mock_validate.call_args_list
    assert len(validate_calls) == 2
    for call in validate_calls:
        assert call.kwargs['expected_dose'] == dose
        assert call.kwargs['design_params'] == design_params

    # Verify return paths structure
    assert 'train' in paths
    assert 'test' in paths
    assert 'intermediate_dir' in paths
    assert paths['intermediate_dir'].name == f"dose_{int(dose)}"


def test_generate_dataset_config_construction(mock_base_npz, design_params, tmp_path):
    """Test that TrainingConfig is correctly constructed for simulation."""
    dose = 1e4
    output_root = tmp_path / "output"

    # Mock only validate to let config construction run
    mock_validate = Mock()

    # Create a mock simulate that captures the config
    captured_config = None
    def capture_simulate(config, **kwargs):
        nonlocal captured_config
        captured_config = config

    mock_simulate = Mock(side_effect=capture_simulate)
    mock_canonicalize = Mock()
    mock_patch_gen = Mock()
    mock_split = Mock()

    generate_dataset_for_dose(
        dose=dose,
        base_npz_path=mock_base_npz,
        output_root=output_root,
        design_params=design_params,
        simulate_fn=mock_simulate,
        canonicalize_fn=mock_canonicalize,
        patch_gen_fn=mock_patch_gen,
        split_fn=mock_split,
        validate_fn=mock_validate,
    )

    # Verify config was captured and has correct values
    assert captured_config is not None
    assert captured_config.nphotons == int(dose)
    assert captured_config.n_groups == 100  # from mock_base_npz
    assert captured_config.model.gridsize == 1
    # Verify n_images is set (required for legacy simulator coordinate arrays)
    assert captured_config.n_images == 100  # must match base dataset length

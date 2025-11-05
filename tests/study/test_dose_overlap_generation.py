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
    mock_validate = Mock()

    # Create mock split function that produces stub NPZ files for validation
    def mock_split_with_files(input_path, output_dir, split_fraction, split_axis):
        """Mock split that creates stub NPZ files so validation can load them."""
        output_dir = Path(output_dir)
        train_npz = output_dir / f"{Path(input_path).stem}_train.npz"
        test_npz = output_dir / f"{Path(input_path).stem}_test.npz"
        # Create minimal stub NPZ files with DATA-001 keys
        for split_path in [train_npz, test_npz]:
            np.savez(
                split_path,
                diffraction=np.random.rand(10, 64, 64).astype(np.float32),
                objectGuess=np.random.rand(128, 128).astype(np.complex64),
                probeGuess=np.random.rand(64, 64).astype(np.complex64),
                xcoords=np.arange(10, dtype=np.float32),
                ycoords=np.arange(10, dtype=np.float32),
            )

    # Call the orchestration function with mocked dependencies
    paths = generate_dataset_for_dose(
        dose=dose,
        base_npz_path=mock_base_npz,
        output_root=output_root,
        design_params=design_params,
        simulate_fn=mock_simulate,
        canonicalize_fn=mock_canonicalize,
        patch_gen_fn=mock_patch_gen,
        split_fn=mock_split_with_files,
        validate_fn=mock_validate,
    )

    # Verify all stages were called exactly once
    assert mock_simulate.call_count == 1
    assert mock_canonicalize.call_count == 1
    assert mock_patch_gen.call_count == 1
    # Note: mock_split_with_files is a function, not a Mock, so no call_count check
    assert mock_validate.call_count == 2  # train + test

    # Verify simulate was called with correct dose
    simulate_call = mock_simulate.call_args
    assert simulate_call.kwargs['config'].nphotons == int(dose)
    assert simulate_call.kwargs['seed'] == design_params['rng_seeds']['simulation']

    # Verify patch generation uses correct parameters
    patch_call = mock_patch_gen.call_args
    assert patch_call.kwargs['patch_size'] == design_params['patch_size_pixels']
    assert patch_call.kwargs['k_neighbors'] == design_params['neighbor_count']

    # Verify validator was called twice (train + test)
    validate_calls = mock_validate.call_args_list
    assert len(validate_calls) == 2
    # Note: validator now expects in-memory dict (data kwarg), not file path
    # See test_generate_dataset_validates_with_real_contract for signature validation

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

    # Create mock split function that produces stub NPZ files for validation
    def mock_split_with_files(input_path, output_dir, split_fraction, split_axis):
        """Mock split that creates stub NPZ files so validation can load them."""
        output_dir = Path(output_dir)
        train_npz = output_dir / f"{Path(input_path).stem}_train.npz"
        test_npz = output_dir / f"{Path(input_path).stem}_test.npz"
        # Create minimal stub NPZ files with DATA-001 keys
        for split_path in [train_npz, test_npz]:
            np.savez(
                split_path,
                diffraction=np.random.rand(10, 64, 64).astype(np.float32),
                objectGuess=np.random.rand(128, 128).astype(np.complex64),
                probeGuess=np.random.rand(64, 64).astype(np.complex64),
                xcoords=np.arange(10, dtype=np.float32),
                ycoords=np.arange(10, dtype=np.float32),
            )

    generate_dataset_for_dose(
        dose=dose,
        base_npz_path=mock_base_npz,
        output_root=output_root,
        design_params=design_params,
        simulate_fn=mock_simulate,
        canonicalize_fn=mock_canonicalize,
        patch_gen_fn=mock_patch_gen,
        split_fn=mock_split_with_files,
        validate_fn=mock_validate,
    )

    # Verify config was captured and has correct values
    assert captured_config is not None
    assert captured_config.nphotons == int(dose)
    assert captured_config.n_groups == 100  # from mock_base_npz
    assert captured_config.model.gridsize == 1
    # Verify n_images is set (required for legacy simulator coordinate arrays)
    assert captured_config.n_images == 100  # must match base dataset length


def test_generate_dataset_validates_with_real_contract(mock_base_npz, design_params, tmp_path):
    """
    Regression test ensuring generate_dataset_for_dose calls the refactored
    validator with the correct signature (data dict, not file path).

    This test validates the fix for the TypeError:
    'validate_dataset_contract() got an unexpected keyword argument dataset_path'
    that occurred when Stage 5 validation used the old file-path interface.

    References:
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/plan/plan.md
    - studies/fly64_dose_overlap/validation.py:33-39 (refactored signature)
    """
    dose = 1e3
    output_root = tmp_path / "output"

    # Create a custom validator mock that verifies the correct signature
    validator_calls = []

    def mock_validator(data, view=None, gridsize=1, neighbor_count=None, design=None):
        """Mock validator matching the refactored signature."""
        # Verify we receive in-memory dict, not file path
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        # Verify required keys present (DATA-001)
        assert 'diffraction' in data
        assert 'xcoords' in data
        assert 'ycoords' in data
        # Verify design params passed correctly (if present in design_params)
        if 'view' in design_params:
            assert view == design_params['view']
        assert gridsize == design_params.get('gridsize', 1)
        assert neighbor_count == design_params.get('neighbor_count')
        validator_calls.append({'data': data, 'view': view, 'gridsize': gridsize, 'neighbor_count': neighbor_count})

    # Create mock split function that produces stub NPZ files for validation
    def mock_split_with_files(input_path, output_dir, split_fraction, split_axis):
        """Mock split that creates stub NPZ files so validation can load them."""
        output_dir = Path(output_dir)
        train_npz = output_dir / f"{Path(input_path).stem}_train.npz"
        test_npz = output_dir / f"{Path(input_path).stem}_test.npz"
        # Create minimal stub NPZ files with DATA-001 keys
        for split_path in [train_npz, test_npz]:
            np.savez(
                split_path,
                diffraction=np.random.rand(10, 64, 64).astype(np.float32),
                objectGuess=np.random.rand(128, 128).astype(np.complex64),
                probeGuess=np.random.rand(64, 64).astype(np.complex64),
                xcoords=np.arange(10, dtype=np.float32),
                ycoords=np.arange(10, dtype=np.float32),
            )

    # Stub out heavy pipeline stages
    mock_simulate = Mock()
    mock_canonicalize = Mock()
    mock_patch_gen = Mock()

    generate_dataset_for_dose(
        dose=dose,
        base_npz_path=mock_base_npz,
        output_root=output_root,
        design_params=design_params,
        simulate_fn=mock_simulate,
        canonicalize_fn=mock_canonicalize,
        patch_gen_fn=mock_patch_gen,
        split_fn=mock_split_with_files,  # Use file-creating mock
        validate_fn=mock_validator,  # Use our signature-checking mock
    )

    # Verify validator was called twice (train + test) with correct signature
    assert len(validator_calls) == 2, f"Expected 2 validator calls, got {len(validator_calls)}"

    # Verify no old-style kwargs (dataset_path, design_params, expected_dose)
    # would have been passed - the mock would have raised TypeError if so
    for call in validator_calls:
        assert 'data' in call
        assert isinstance(call['data'], dict)
        # Verify view passed if present in design_params
        if 'view' in design_params:
            assert call['view'] == design_params['view']
        assert call['gridsize'] == design_params.get('gridsize', 1)
        assert call['neighbor_count'] == design_params.get('neighbor_count')

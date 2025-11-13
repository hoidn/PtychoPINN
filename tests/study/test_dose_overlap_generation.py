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


def test_build_simulation_plan_handles_metadata_pickle_guard(tmp_path, design_params):
    """
    Test that build_simulation_plan can load base NPZ files with metadata
    without triggering allow_pickle=False errors.

    This validates the fix for:
    ValueError: Object arrays cannot be loaded when allow_pickle=False

    Prior to fix, build_simulation_plan used raw np.load() which fails
    on metadata-bearing NPZs. After fix, it uses MetadataManager.load_with_metadata()
    which correctly handles pickle=True for _metadata field.

    References:
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/plan/plan.md
    - DATA-001 (metadata must be preserved across pipeline stages)
    """
    from ptycho.metadata import MetadataManager
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Create a base NPZ with embedded metadata (simulating simulate_and_save output)
    base_path = tmp_path / "base_with_metadata.npz"

    # Create minimal config for metadata
    model_cfg = ModelConfig(gridsize=1, N=64)
    train_cfg = TrainingConfig(
        model=model_cfg,
        train_data_file=str(base_path),
        n_images=100,
        nphotons=5000,
    )

    # Create metadata
    metadata = MetadataManager.create_metadata(
        config=train_cfg,
        script_name="test_metadata_guard",
        seed=42,
    )

    # Save NPZ with metadata
    data_dict = {
        'xcoords': np.arange(100, dtype=np.float32),
        'ycoords': np.arange(100, dtype=np.float32),
        'diffraction': np.random.rand(100, 64, 64).astype(np.float32),
        'objectGuess': np.random.rand(256, 256).astype(np.complex64),
        'probeGuess': np.random.rand(64, 64).astype(np.complex64),
    }

    MetadataManager.save_with_metadata(
        file_path=str(base_path),
        data_dict=data_dict,
        metadata=metadata,
    )

    # This should NOT raise ValueError about allow_pickle
    plan = build_simulation_plan(
        dose=1e4,
        base_npz_path=base_path,
        design_params=design_params,
    )

    # Verify plan was constructed successfully
    assert plan.n_images == 100
    assert plan.dose == 1e4


def test_load_data_for_sim_handles_metadata_pickle_guard(tmp_path):
    """
    Test that scripts/simulation/simulate_and_save.py::load_data_for_sim
    can load NPZ files with metadata without triggering allow_pickle=False errors.

    This validates the fix for:
    ValueError: Object arrays cannot be loaded when allow_pickle=False

    Prior to fix, load_data_for_sim used raw np.load() which fails on metadata-bearing
    NPZs. After fix, it uses MetadataManager.load_with_metadata() which correctly
    handles pickle=True for _metadata field.

    References:
    - scripts/simulation/simulate_and_save.py:37 (load_data_for_sim function)
    - DATA-001 (metadata must be preserved)
    """
    from scripts.simulation.simulate_and_save import load_data_for_sim
    from ptycho.metadata import MetadataManager
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Create a NPZ with embedded metadata
    npz_path = tmp_path / "data_with_metadata.npz"

    # Create minimal config for metadata
    model_cfg = ModelConfig(gridsize=1, N=64)
    train_cfg = TrainingConfig(
        model=model_cfg,
        train_data_file=str(npz_path),
        n_images=50,
        nphotons=3000,
    )

    metadata = MetadataManager.create_metadata(
        config=train_cfg,
        script_name="test_load_data_for_sim_guard",
        seed=123,
    )

    # Save NPZ with metadata
    data_dict = {
        'objectGuess': np.random.rand(256, 256).astype(np.complex64),
        'probeGuess': np.random.rand(64, 64).astype(np.complex64),
        'xcoords': np.arange(50, dtype=np.float32),
        'ycoords': np.arange(50, dtype=np.float32),
    }

    MetadataManager.save_with_metadata(
        file_path=str(npz_path),
        data_dict=data_dict,
        metadata=metadata,
    )

    # This should NOT raise ValueError about allow_pickle
    objectGuess, probeGuess, all_data = load_data_for_sim(str(npz_path), load_all=True)

    # Verify data was loaded correctly
    assert objectGuess.shape == (256, 256)
    assert probeGuess.shape == (64, 64)
    assert all_data is not None
    assert 'xcoords' in all_data
    # Metadata should NOT be in the returned data dict (filtered by MetadataManager)
    assert '_metadata' not in all_data


def test_generate_dataset_for_dose_handles_metadata_splits(tmp_path, design_params):
    """
    Test that generate_dataset_for_dose Stage 5 loads metadata-bearing train/test splits
    via MetadataManager and passes clean data to validator.

    This validates the fix for:
    - ValueError: Object arrays cannot be loaded when allow_pickle=False
    - Ensures validator receives dict without '_metadata' key (DATA-001 compliance)

    Stage 5 (lines 219-235) must use MetadataManager.load_with_metadata() to:
    1. Load train/test NPZs with embedded metadata safely (allow_pickle=True)
    2. Filter '_metadata' before passing to validator
    3. Preserve TYPE-PATH-001 by passing Path objects where required

    References:
    - studies/fly64_dose_overlap/generation.py:224 (Stage 5 validation block)
    - DATA-001 (metadata filtering requirement)
    - input.md (2025-11-08T230500Z Do Now)
    """
    from ptycho.metadata import MetadataManager
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Create base NPZ with metadata
    base_npz_path = tmp_path / "base.npz"
    model_cfg = ModelConfig(gridsize=2, N=64)
    train_cfg = TrainingConfig(
        model=model_cfg,
        train_data_file=str(base_npz_path),
        n_images=100,
        nphotons=5000,
    )
    metadata = MetadataManager.create_metadata(
        config=train_cfg,
        script_name="test_metadata_splits",
        seed=42,
    )
    base_data = {
        'xcoords': np.arange(100, dtype=np.float32),
        'ycoords': np.arange(100, dtype=np.float32),
        'diffraction': np.random.rand(100, 64, 64).astype(np.float32),
        'objectGuess': np.random.rand(256, 256).astype(np.complex64),
        'probeGuess': np.random.rand(64, 64).astype(np.complex64),
    }
    MetadataManager.save_with_metadata(
        file_path=str(base_npz_path),
        data_dict=base_data,
        metadata=metadata,
    )

    # Create mock split function that produces metadata-bearing NPZs
    def mock_split_with_metadata(input_path, output_dir, split_fraction, split_axis):
        """Create train/test splits with embedded metadata."""
        train_path = output_dir / "patched_train.npz"
        test_path = output_dir / "patched_test.npz"

        # Create minimal splits with metadata
        split_metadata = MetadataManager.create_metadata(
            config=train_cfg,
            script_name="mock_split",
            seed=99,
        )
        MetadataManager.add_transformation_record(
            metadata=split_metadata,
            tool_name='transpose_rename_convert',
            operation='canonical_layout',
            parameters={'layout': 'NHW'},
        )

        split_data = {
            'diffraction': np.random.rand(50, 64, 64).astype(np.float32),
            'Y': np.random.rand(50, 10, 64, 64).astype(np.complex64),
            'nn_indices': np.random.randint(0, 50, size=(50, 10), dtype=np.int32),
            'xcoords': np.arange(50, dtype=np.float32),
            'ycoords': np.arange(50, dtype=np.float32),
            'objectGuess': np.random.rand(256, 256).astype(np.complex64),
            'probeGuess': np.random.rand(64, 64).astype(np.complex64),
        }

        MetadataManager.save_with_metadata(str(train_path), split_data, split_metadata)
        MetadataManager.save_with_metadata(str(test_path), split_data, split_metadata)

    # Mock validator to capture what it receives
    validator_calls = []
    def mock_validator(**kwargs):
        validator_calls.append(kwargs)
        # Assert no '_metadata' key leaked through
        assert '_metadata' not in kwargs.get('data', {}), \
            "Stage 5 validator received _metadata key; MetadataManager filter failed"

    # Stub heavy pipeline stages
    mock_simulate = Mock()
    mock_canonicalize = Mock()
    mock_patch_gen = Mock()

    output_root = tmp_path / "output"
    dose = 1000

    # Execute pipeline
    generate_dataset_for_dose(
        dose=dose,
        base_npz_path=base_npz_path,
        output_root=output_root,
        design_params=design_params,
        simulate_fn=mock_simulate,
        canonicalize_fn=mock_canonicalize,
        patch_gen_fn=mock_patch_gen,
        split_fn=mock_split_with_metadata,
        validate_fn=mock_validator,
    )

    # Verify validator was called twice (train + test) with clean data
    assert len(validator_calls) == 2, f"Expected 2 validator calls, got {len(validator_calls)}"

    # Verify each call received dict without '_metadata'
    for i, call in enumerate(validator_calls):
        assert 'data' in call, f"Call {i}: missing 'data' kwarg"
        assert isinstance(call['data'], dict), f"Call {i}: 'data' not a dict"
        assert '_metadata' not in call['data'], \
            f"Call {i}: validator received '_metadata' key (DATA-001 violation)"
        # Verify DATA-001 keys present
        assert 'diffraction' in call['data'], f"Call {i}: missing 'diffraction'"
        assert 'Y' in call['data'], f"Call {i}: missing 'Y'"

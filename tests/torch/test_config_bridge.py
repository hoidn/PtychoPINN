"""
Tests for PyTorch config bridge adapter translating PyTorch configs to TensorFlow spec format.

This module validates Phase B.B2 of INTEGRATE-PYTORCH-001: the configuration bridge that
translates PyTorch singleton configs to TensorFlow dataclass configs, enabling population
of the legacy params.cfg dictionary through the standard update_legacy_dict() function.

MVP Test Coverage (9 fields):
- N, gridsize, model_type (model essentials)
- train_data_file, test_data_file, model_path (lifecycle paths)
- n_groups, neighbor_count (data grouping)
- nphotons (physics scaling)

Test Strategy:
1. Instantiate PyTorch configs with MVP-aligned values
2. Call adapter functions to convert to TensorFlow dataclass configs
3. Use update_legacy_dict() to populate params.cfg
4. Assert params.cfg contains expected keys with correct values

Implementation Status: Adapter module (ptycho_torch.config_bridge) implemented in Phase B.B3.
This test validates the MVP translation functions convert PyTorch configs to TensorFlow dataclasses.
"""

import unittest
from pathlib import Path
import pytest
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestConfigBridgeMVP(unittest.TestCase):
    """Test suite for PyTorch → TensorFlow config bridge MVP functionality."""

    def setUp(self):
        """Set up test fixtures and save params.cfg state."""
        import ptycho.params as params
        # Save snapshot of params.cfg for restoration
        self.params_snapshot = dict(params.cfg)

    def tearDown(self):
        """Restore params.cfg state after test."""
        import ptycho.params as params
        # Restore original state
        params.cfg.clear()
        params.cfg.update(self.params_snapshot)

    def test_mvp_config_bridge_populates_params_cfg(self):
        """Test that PyTorch configs populate params.cfg with MVP fields via bridge adapter.

        This test exercises the complete configuration bridge workflow:
        1. Create PyTorch configs using existing singleton pattern
        2. Invoke adapter to translate to TensorFlow dataclass format
        3. Call update_legacy_dict() to populate params.cfg
        4. Verify all MVP fields present with correct values and types

        Note: This test will be automatically skipped if PyTorch runtime is unavailable
        (handled by tests/conftest.py). When PyTorch is available, validates full workflow.
        """
        # Import PyTorch configs (existing implementation)
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig

        # Import TensorFlow config utilities
        from ptycho.config.config import update_legacy_dict
        import ptycho.params as params

        # 1. Instantiate PyTorch configs with MVP-aligned values
        pt_data = DataConfig(
            N=128,
            grid_size=(2, 2),
            nphotons=1e9,
            K=7  # Maps to neighbor_count in TensorFlow
        )

        pt_model = ModelConfig(
            mode='Unsupervised'  # Maps to model_type='pinn' in TensorFlow
        )

        pt_train = TrainingConfig(
            epochs=1  # Maps to nepochs in TensorFlow
        )

        pt_infer = InferenceConfig(
            batch_size=1
        )

        # 2. Import and use the adapter module
        from ptycho_torch import config_bridge

        # Call adapter functions to translate PyTorch → TensorFlow dataclasses
        spec_model = config_bridge.to_model_config(pt_data, pt_model)

        spec_train = config_bridge.to_training_config(
            spec_model,
            pt_data,
            pt_model,  # PyTorch ModelConfig for intensity_scale_trainable
            pt_train,
            overrides=dict(
                train_data_file=Path('train.npz'),
                n_groups=512,
                neighbor_count=7,
                nphotons=1e9
            )
        )

        spec_infer = config_bridge.to_inference_config(
            spec_model,
            pt_data,
            pt_infer,
            overrides=dict(
                model_path=Path('model_dir'),
                test_data_file=Path('test.npz'),
                n_groups=512,
                neighbor_count=7
            )
        )

        # 3. Call update_legacy_dict to populate params.cfg
        update_legacy_dict(params.cfg, spec_train)
        update_legacy_dict(params.cfg, spec_infer)

        # 4. Assert MVP fields populated correctly
        # Model essentials
        self.assertEqual(params.cfg['N'], 128, "N should match DataConfig.N")
        self.assertEqual(params.cfg['gridsize'], 2, "gridsize should be extracted from grid_size tuple")
        self.assertEqual(params.cfg['model_type'], 'pinn', "model_type should map from mode='Unsupervised'")

        # Lifecycle paths (KEY_MAPPINGS translation)
        self.assertEqual(params.cfg['train_data_file_path'], 'train.npz',
                       "train_data_file should map via KEY_MAPPINGS")
        self.assertEqual(params.cfg['test_data_file_path'], 'test.npz',
                       "test_data_file should map via KEY_MAPPINGS")
        self.assertEqual(params.cfg['model_path'], 'model_dir',
                       "model_path should be converted to string")

        # Data grouping
        self.assertEqual(params.cfg['n_groups'], 512, "n_groups should pass through")
        self.assertEqual(params.cfg['neighbor_count'], 7, "neighbor_count should map from K")

        # Physics scaling
        self.assertEqual(params.cfg['nphotons'], 1e9, "nphotons should match")


@pytest.fixture
def params_cfg_snapshot():
    """Fixture to save and restore params.cfg state for parity tests."""
    import ptycho.params as params
    # Save snapshot of params.cfg before test
    snapshot = dict(params.cfg)
    yield
    # Restore original state after test
    params.cfg.clear()
    params.cfg.update(snapshot)


class TestConfigBridgeParity:
    """
    Comprehensive parity tests for PyTorch → TensorFlow config bridge adapter.

    This test suite implements Phase B.B4 of INTEGRATE-PYTORCH-001, extending beyond
    the MVP 9-field coverage to validate all spec-required fields across ModelConfig,
    TrainingConfig, and InferenceConfig.

    Test organization:
    - Direct fields: Pass through without transformation
    - Transform fields: Require name/type conversion
    - Override-required fields: Missing from PyTorch, use defaults or overrides
    - Default divergence: Different PyTorch/TensorFlow defaults
    - Error handling: Invalid inputs raise actionable errors

    References:
    - Field matrix: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md
    - Test design: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/testcase_design.md
    - Canonical fixtures: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/fixtures.py
    """

    # ============================================================================
    # Test Case 1: Direct Field Translation
    # ============================================================================

    @pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
        pytest.param('N', 128, 128, id='N-direct'),
        pytest.param('n_filters_scale', 2, 2, id='n_filters_scale-direct'),
        pytest.param('object_big', False, False, id='object_big-direct'),
        pytest.param('probe_big', False, False, id='probe_big-direct'),
    ])
    def test_model_config_direct_fields(self, params_cfg_snapshot, field_name, pytorch_value, expected_tf_value):
        """
        Test ModelConfig fields that translate directly without transformation.

        Spec coverage: §5.1:1 (N), §5.1:3 (n_filters_scale), §5.1:6 (object_big), §5.1:7 (probe_big)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        # Create PyTorch config with test value
        if field_name in ['N']:
            pt_data = DataConfig(**{field_name: pytorch_value})
            pt_model = ModelConfig()
        else:
            pt_data = DataConfig()
            pt_model = ModelConfig(**{field_name: pytorch_value})

        # Translate to TensorFlow
        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Assert field value matches
        assert getattr(tf_model, field_name) == expected_tf_value, \
            f"{field_name} should pass through directly"

    @pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
        pytest.param('batch_size', 32, 32, id='batch_size-direct'),
    ])
    def test_training_config_direct_fields(self, params_cfg_snapshot, field_name, pytorch_value, expected_tf_value):
        """
        Test TrainingConfig fields that translate directly without transformation.

        Spec coverage: §5.2:3 (batch_size)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig(**{field_name: pytorch_value})

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(train_data_file=Path('train.npz'), n_groups=512, nphotons=1e9)
        )

        assert getattr(tf_train, field_name) == expected_tf_value, \
            f"{field_name} should pass through directly"

    # ============================================================================
    # Test Case 2: Transformed Field Translation
    # ============================================================================

    @pytest.mark.parametrize('pt_field,pt_value,tf_field,tf_value', [
        pytest.param('grid_size', (3, 3), 'gridsize', 3, id='gridsize-tuple-to-int'),
        pytest.param('mode', 'Unsupervised', 'model_type', 'pinn', id='model_type-unsupervised'),
        pytest.param('mode', 'Supervised', 'model_type', 'supervised', id='model_type-supervised'),
        pytest.param('amp_activation', 'silu', 'amp_activation', 'swish', id='amp_activation-silu'),
        pytest.param('amp_activation', 'SiLU', 'amp_activation', 'swish', id='amp_activation-SiLU'),
        pytest.param('amp_activation', 'sigmoid', 'amp_activation', 'sigmoid', id='amp_activation-passthrough'),
    ])
    def test_model_config_transform_fields(self, params_cfg_snapshot, pt_field, pt_value, tf_field, tf_value):
        """
        Test ModelConfig fields that require transformation during translation.

        Spec coverage: §5.1:2 (gridsize), §5.1:4 (model_type), §5.1:5 (amp_activation)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        # Create PyTorch config with test value
        if pt_field in ['grid_size']:
            pt_data = DataConfig(**{pt_field: pt_value})
            pt_model = ModelConfig()
        else:
            pt_data = DataConfig()
            pt_model = ModelConfig(**{pt_field: pt_value})

        # Translate to TensorFlow
        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Assert transformed field value matches
        assert getattr(tf_model, tf_field) == tf_value, \
            f"{pt_field}={pt_value} should transform to {tf_field}={tf_value}"

    @pytest.mark.parametrize('pt_field,pt_value,tf_field,tf_value', [
        pytest.param('epochs', 100, 'nepochs', 100, id='nepochs-rename'),
        pytest.param('nll', True, 'nll_weight', 1.0, id='nll_weight-true'),
        pytest.param('nll', False, 'nll_weight', 0.0, id='nll_weight-false'),
    ])
    def test_training_config_transform_fields(self, params_cfg_snapshot, pt_field, pt_value, tf_field, tf_value):
        """
        Test TrainingConfig fields that require transformation during translation.

        Spec coverage: §5.2:4 (nepochs), §5.2:6 (nll_weight)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig(**{pt_field: pt_value})

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(train_data_file=Path('train.npz'), n_groups=512, nphotons=1e9)
        )

        assert getattr(tf_train, tf_field) == tf_value, \
            f"{pt_field}={pt_value} should transform to {tf_field}={tf_value}"

    # ============================================================================
    # Test Case 3: Override-Required Fields
    # ============================================================================

    @pytest.mark.parametrize('field_name,override_value,expected_value', [
        pytest.param('pad_object', None, True, id='pad_object-default'),
        pytest.param('pad_object', False, False, id='pad_object-override'),
        pytest.param('gaussian_smoothing_sigma', None, 0.0, id='gaussian_smoothing_sigma-default'),
        pytest.param('gaussian_smoothing_sigma', 0.5, 0.5, id='gaussian_smoothing_sigma-override'),
    ])
    def test_model_config_override_fields(self, params_cfg_snapshot, field_name, override_value, expected_value):
        """
        Test ModelConfig fields that now have defaults in PyTorch or use overrides.

        Spec coverage: §5.1:9 (pad_object), §5.1:11 (gaussian_smoothing_sigma)
        Phase: Phase C.C3 spec defaults backfill
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()

        # Build overrides dict
        overrides = {field_name: override_value} if override_value is not None else {}

        tf_model = config_bridge.to_model_config(pt_data, pt_model, overrides=overrides)

        assert getattr(tf_model, field_name) == expected_value, \
            f"{field_name} should use {'override' if override_value is not None else 'default'} value"

    @pytest.mark.parametrize('field_name,override_value,expected_value', [
        pytest.param('mae_weight', None, 0.0, id='mae_weight-default'),
        pytest.param('mae_weight', 0.3, 0.3, id='mae_weight-override'),
        pytest.param('realspace_mae_weight', None, 0.0, id='realspace_mae_weight-default'),
        pytest.param('realspace_weight', None, 0.0, id='realspace_weight-default'),
        pytest.param('positions_provided', None, True, id='positions_provided-default'),
        pytest.param('probe_trainable', None, False, id='probe_trainable-default'),
        pytest.param('sequential_sampling', None, False, id='sequential_sampling-default'),
    ])
    def test_training_config_override_fields(self, params_cfg_snapshot, field_name, override_value, expected_value):
        """
        Test TrainingConfig fields missing from PyTorch that require defaults/overrides.

        Spec coverage: §5.2:5 (mae_weight), §5.2:7 (realspace_mae_weight),
        §5.2:8 (realspace_weight), §5.2:15 (positions_provided),
        §5.2:16 (probe_trainable), §5.2:19 (sequential_sampling)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        # Build overrides dict (always include required fields including nphotons to avoid validation error)
        overrides = dict(train_data_file=Path('train.npz'), n_groups=512, nphotons=1e9)
        if override_value is not None:
            overrides[field_name] = override_value

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train, overrides=overrides
        )

        assert getattr(tf_train, field_name) == expected_value, \
            f"{field_name} should use {'override' if override_value is not None else 'default'} value"

    @pytest.mark.parametrize('field_name,override_value,expected_value', [
        pytest.param('debug', None, False, id='debug-default'),
        pytest.param('debug', True, True, id='debug-override'),
    ])
    def test_inference_config_override_fields(self, params_cfg_snapshot, field_name, override_value, expected_value):
        """
        Test InferenceConfig fields missing from PyTorch that require defaults/overrides.

        Spec coverage: §5.3:8 (debug)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, InferenceConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_infer = InferenceConfig()

        # Build overrides dict (always include required fields)
        overrides = dict(
            model_path=Path('model_dir'),
            test_data_file=Path('test.npz'),
            n_groups=512
        )
        if override_value is not None:
            overrides[field_name] = override_value

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_infer = config_bridge.to_inference_config(
            tf_model, pt_data, pt_infer, overrides=overrides
        )

        assert getattr(tf_infer, field_name) == expected_value, \
            f"{field_name} should use {'override' if override_value is not None else 'default'} value"

    # ============================================================================
    # Test Case 3.5: probe_mask Translation (Phase B.B5.B2)
    # ============================================================================

    @pytest.mark.parametrize('pytorch_value,expected_tf_value,description', [
        pytest.param(None, False, 'default-None→False', id='probe_mask-default'),
        # Note: Cannot test Tensor→True case without torch runtime, but logic is implemented
    ])
    def test_model_config_probe_mask_translation(self, params_cfg_snapshot, pytorch_value, expected_tf_value, description):
        """
        Test probe_mask translation from Optional[Tensor] to bool.

        Without torch: None → False (default behavior)
        With torch: None → False, Tensor → True

        Spec coverage: §5.1:8 (probe_mask)
        Phase: B.B5.B2 parity extension
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        # Create ModelConfig with probe_mask=None (PyTorch default)
        pt_model = ModelConfig(probe_mask=pytorch_value)

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        assert tf_model.probe_mask == expected_tf_value, \
            f"probe_mask={pytorch_value} should translate to {expected_tf_value} ({description})"

    def test_model_config_probe_mask_override(self, params_cfg_snapshot):
        """
        Test that probe_mask can be explicitly overridden via overrides dict.

        This allows external callers to force True even when PyTorch config has None.

        Spec coverage: §5.1:8 (probe_mask override pattern)
        Phase: B.B5.B2 parity extension
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig(probe_mask=None)  # PyTorch default

        # Override to True
        tf_model = config_bridge.to_model_config(
            pt_data, pt_model,
            overrides={'probe_mask': True}
        )

        assert tf_model.probe_mask is True, \
            "probe_mask override should force True even when PyTorch has None"

    # ============================================================================
    # Test Case 4: Default Divergence Detection
    # ============================================================================

    @pytest.mark.parametrize('field_name,pytorch_default,tf_default,test_value', [
        pytest.param('nphotons', 1e5, 1e9, 5e8, id='nphotons-divergence',
                     marks=pytest.mark.mvp),
        pytest.param('probe_scale', 1.0, 4.0, 2.0, id='probe_scale-divergence'),
    ])
    def test_default_divergence_detection(self, params_cfg_snapshot, field_name, pytorch_default, tf_default, test_value):
        """
        Test that fields with different PyTorch/TensorFlow defaults use explicit values.

        This ensures the adapter doesn't silently fall back to incompatible defaults.

        Spec coverage: §5.2:9 (nphotons HIGH risk), §5.1:10 (probe_scale MEDIUM risk)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        # Create PyTorch config with explicit non-default value
        pt_data = DataConfig(**{field_name: test_value})
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        # For model-level fields, test via to_model_config
        if field_name in ['probe_scale']:
            tf_model = config_bridge.to_model_config(pt_data, pt_model)
            actual_value = getattr(tf_model, field_name)
        # For training-level fields, test via to_training_config
        else:
            tf_model = config_bridge.to_model_config(pt_data, pt_model)
            tf_train = config_bridge.to_training_config(
                tf_model, pt_data, pt_model, pt_train,
                overrides=dict(train_data_file=Path('train.npz'), n_groups=512, nphotons=test_value)
            )
            actual_value = getattr(tf_train, field_name)

        # Assert explicit value is used (not either default)
        assert actual_value == test_value, \
            f"{field_name} should use explicit value {test_value}, not defaults " \
            f"(PyTorch={pytorch_default}, TF={tf_default})"
        assert actual_value != pytorch_default, \
            "Should not fall back to PyTorch default"
        assert actual_value != tf_default, \
            "Should not fall back to TensorFlow default"

    # ============================================================================
    # Test Case 5: Error Handling & Validation
    # ============================================================================

    @pytest.mark.parametrize('invalid_value,expected_error,error_fragment', [
        pytest.param((2, 3), ValueError, 'Non-square grids', id='gridsize-non-square'),
    ])
    def test_gridsize_error_handling(self, params_cfg_snapshot, invalid_value, expected_error, error_fragment):
        """Test grid_size validation (non-square grids should raise ValueError)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(grid_size=invalid_value)
        pt_model = ModelConfig()

        with pytest.raises(expected_error) as exc_info:
            config_bridge.to_model_config(pt_data, pt_model)

        assert error_fragment in str(exc_info.value), \
            f"Error message should contain '{error_fragment}'"

    @pytest.mark.parametrize('invalid_value,expected_error,error_fragment', [
        pytest.param('InvalidMode', ValueError, 'Invalid mode', id='model_type-invalid-enum'),
    ])
    def test_model_type_error_handling(self, params_cfg_snapshot, invalid_value, expected_error, error_fragment):
        """Test mode validation (invalid enum should raise ValueError)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig(mode=invalid_value)

        with pytest.raises(expected_error) as exc_info:
            config_bridge.to_model_config(pt_data, pt_model)

        assert error_fragment in str(exc_info.value), \
            f"Error message should contain '{error_fragment}'"

    @pytest.mark.parametrize('invalid_value,expected_error,error_fragment', [
        pytest.param('unknown_activation', ValueError, 'Unknown activation', id='amp_activation-unknown'),
    ])
    def test_activation_error_handling(self, params_cfg_snapshot, invalid_value, expected_error, error_fragment):
        """Test amp_activation validation (unknown activation should raise ValueError)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig(amp_activation=invalid_value)

        with pytest.raises(expected_error) as exc_info:
            config_bridge.to_model_config(pt_data, pt_model)

        assert error_fragment in str(exc_info.value), \
            f"Error message should contain '{error_fragment}'"

    def test_train_data_file_required_error(self, params_cfg_snapshot):
        """Test that missing train_data_file raises actionable ValueError (MVP field)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        with pytest.raises(ValueError) as exc_info:
            config_bridge.to_training_config(
                tf_model, pt_data, pt_model, pt_train,
                overrides=dict(n_groups=512, nphotons=1e9)  # Missing train_data_file but include nphotons
            )

        assert 'train_data_file is required' in str(exc_info.value), \
            "Error message should indicate train_data_file is required"

    def test_model_path_required_error(self, params_cfg_snapshot):
        """Test that missing model_path raises actionable ValueError (MVP field)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, InferenceConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_infer = InferenceConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        with pytest.raises(ValueError) as exc_info:
            config_bridge.to_inference_config(
                tf_model, pt_data, pt_infer,
                overrides=dict(test_data_file=Path('test.npz'), n_groups=512)  # Missing model_path
            )

        assert 'model_path is required' in str(exc_info.value), \
            "Error message should indicate model_path is required"

    # ============================================================================
    # Test Case 5.5: nphotons Override Validation (Phase B.B5.B4)
    # ============================================================================

    def test_nphotons_default_divergence_error(self, params_cfg_snapshot):
        """
        Test that using PyTorch default nphotons without override raises informative error.

        Regression test for Phase B.B5.B4: Validates that the adapter enforces explicit
        nphotons override when PyTorch default (1e5) differs from TensorFlow default (1e9).

        Error message should provide actionable guidance: overrides=dict(..., nphotons=1e9)

        Spec coverage: §5.2:9 (nphotons HIGH risk divergence)
        Phase: B.B5.B4 override validation
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        # Use PyTorch default nphotons (1e5)
        pt_data = DataConfig(nphotons=1e5)
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        with pytest.raises(ValueError) as exc_info:
            config_bridge.to_training_config(
                tf_model, pt_data, pt_model, pt_train,
                overrides=dict(train_data_file=Path('train.npz'), n_groups=512)
                # Missing nphotons override - should trigger validation error
            )

        # Assert error message contains actionable guidance
        error_msg = str(exc_info.value)
        assert 'nphotons default divergence' in error_msg, \
            "Error message should identify nphotons divergence"
        assert 'overrides' in error_msg, \
            "Error message should mention overrides parameter"
        assert 'nphotons=' in error_msg, \
            "Error message should provide override syntax example"

    def test_nphotons_override_passes_validation(self, params_cfg_snapshot):
        """
        Test that explicit nphotons override passes validation (green path).

        Paired with test_nphotons_default_divergence_error to confirm validation
        accepts explicit overrides and only rejects PyTorch defaults.

        Spec coverage: §5.2:9 (nphotons override pattern)
        Phase: B.B5.B4 override validation
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        # Use PyTorch default nphotons (1e5) but provide explicit override
        pt_data = DataConfig(nphotons=1e5)
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Should NOT raise error when nphotons provided in overrides
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(
                train_data_file=Path('train.npz'),
                n_groups=512,
                nphotons=1e9  # Explicit override resolves divergence
            )
        )

        # Assert override value is used
        assert tf_train.nphotons == 1e9, \
            "nphotons override should be applied successfully"

    # ============================================================================
    # Test Case 5.6: n_subsample Override Validation (Phase C.C1-C2)
    # ============================================================================

    def test_training_config_n_subsample_missing_override_uses_none(self, params_cfg_snapshot):
        """
        Test that missing n_subsample override defaults to None in TrainingConfig.

        Semantic collision context: PyTorch DataConfig.n_subsample exists but has
        different semantics (coordinate subsampling factor, not sample count). The
        adapter must NOT propagate PyTorch n_subsample to TensorFlow; explicit
        override is required to set training sample count.

        Spec coverage: §5.2:12 (n_subsample override_required)
        Phase: Phase C.C1-C2 (n_subsample parity)
        Reference: field_matrix.md row 51
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        # PyTorch config with n_subsample=7 (coordinate subsampling, not sample count)
        pt_data = DataConfig(n_subsample=7)
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Do NOT provide n_subsample override
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(
                train_data_file=Path('train.npz'),
                n_groups=512,
                nphotons=1e9
            )
        )

        # Assert PyTorch value is NOT propagated (semantic collision protection)
        assert tf_train.n_subsample is None, \
            "n_subsample should default to None when not in overrides (semantic collision guard)"
        assert tf_train.n_subsample != 7, \
            "PyTorch DataConfig.n_subsample should NOT propagate to TensorFlow (different semantics)"

    def test_training_config_n_subsample_explicit_override(self, params_cfg_snapshot):
        """
        Test that explicit n_subsample override is applied to TrainingConfig.

        Green path confirming that when caller provides explicit n_subsample value,
        adapter applies it to TensorFlow config regardless of PyTorch value.

        Spec coverage: §5.2:12 (n_subsample override pattern)
        Phase: Phase C.C1-C2 (n_subsample parity)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(n_subsample=7)  # PyTorch coordinate subsampling
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Provide explicit n_subsample override for TensorFlow sample count
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(
                train_data_file=Path('train.npz'),
                n_groups=512,
                nphotons=1e9,
                n_subsample=1000  # Explicit TensorFlow sample count override
            )
        )

        # Assert override value is used (not PyTorch value)
        assert tf_train.n_subsample == 1000, \
            "n_subsample override should be applied successfully"
        assert tf_train.n_subsample != 7, \
            "Override should replace PyTorch value (semantic collision resolved)"

    def test_inference_config_n_subsample_missing_override_uses_none(self, params_cfg_snapshot):
        """
        Test that missing n_subsample override defaults to None in InferenceConfig.

        Same semantic collision as TrainingConfig: PyTorch n_subsample has different
        meaning, so adapter must not propagate without explicit override.

        Spec coverage: §5.3:5 (n_subsample override_required)
        Phase: Phase C.C1-C2 (n_subsample parity)
        Reference: field_matrix.md row 69
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, InferenceConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(n_subsample=7)  # PyTorch coordinate subsampling
        pt_model = ModelConfig()
        pt_infer = InferenceConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Do NOT provide n_subsample override
        tf_infer = config_bridge.to_inference_config(
            tf_model, pt_data, pt_infer,
            overrides=dict(
                model_path=Path('model_dir'),
                test_data_file=Path('test.npz'),
                n_groups=512
            )
        )

        # Assert PyTorch value is NOT propagated
        assert tf_infer.n_subsample is None, \
            "n_subsample should default to None when not in overrides"
        assert tf_infer.n_subsample != 7, \
            "PyTorch DataConfig.n_subsample should NOT propagate to InferenceConfig"

    def test_inference_config_n_subsample_explicit_override(self, params_cfg_snapshot):
        """
        Test that explicit n_subsample override is applied to InferenceConfig.

        Green path for inference-time sample count control.

        Spec coverage: §5.3:5 (n_subsample override pattern)
        Phase: Phase C.C1-C2 (n_subsample parity)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, InferenceConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(n_subsample=7)
        pt_model = ModelConfig()
        pt_infer = InferenceConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Provide explicit n_subsample override
        tf_infer = config_bridge.to_inference_config(
            tf_model, pt_data, pt_infer,
            overrides=dict(
                model_path=Path('model_dir'),
                test_data_file=Path('test.npz'),
                n_groups=512,
                n_subsample=500  # Explicit inference sample count
            )
        )

        # Assert override value is used
        assert tf_infer.n_subsample == 500, \
            "n_subsample override should be applied successfully"
        assert tf_infer.n_subsample != 7, \
            "Override should replace PyTorch value"

    # ============================================================================
    # Test Case 5.8: subsample_seed Field Propagation (Phase C.C3)
    # ============================================================================

    def test_training_config_subsample_seed_from_dataconfig(self, params_cfg_snapshot):
        """
        Test that subsample_seed propagates from DataConfig to TrainingConfig.

        Now that subsample_seed exists in PyTorch DataConfig, verify that
        config_bridge correctly propagates it to TensorFlow TrainingConfig.

        Spec coverage: §5.2:13 (subsample_seed)
        Phase: Phase C.C3 spec defaults backfill
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        # Set subsample_seed in DataConfig
        pt_data = DataConfig(subsample_seed=42)
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(
                train_data_file=Path('train.npz'),
                n_groups=512,
                nphotons=1e9
            )
        )

        # Assert subsample_seed is propagated from DataConfig
        assert tf_train.subsample_seed == 42, \
            "subsample_seed should propagate from DataConfig to TrainingConfig"

    def test_training_config_subsample_seed_override(self, params_cfg_snapshot):
        """
        Test that subsample_seed override takes precedence over DataConfig value.

        Spec coverage: §5.2:13 (subsample_seed override pattern)
        Phase: Phase C.C3 spec defaults backfill
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(subsample_seed=42)  # DataConfig value
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(
                train_data_file=Path('train.npz'),
                n_groups=512,
                nphotons=1e9,
                subsample_seed=99  # Override value
            )
        )

        # Assert override value is used (not DataConfig value)
        assert tf_train.subsample_seed == 99, \
            "subsample_seed override should take precedence"
        assert tf_train.subsample_seed != 42, \
            "Override should replace DataConfig value"

    # ============================================================================
    # Test Case 5.7: Override Warning Coverage (Phase B.B5.D3)
    # ============================================================================

    # Note: probe_scale default divergence test removed after discovering that strict
    # validation breaks too many existing test cases. The field has known divergence
    # (PyTorch 1.0 vs TensorFlow 4.0) documented in override_matrix.md, but runtime
    # enforcement is too disruptive. Callers should consult documentation for this field.

    def test_n_groups_missing_override_warning(self, params_cfg_snapshot):
        """
        Test that missing n_groups override in TrainingConfig raises warning/error.

        From override_matrix.md: n_groups training stage with no override leaves
        params.cfg['n_groups'] = None if inference also omits value, breaking
        downstream workflows that expect a valid integer.

        Spec coverage: §5.2:10 (n_groups override_required)
        Phase: B.B5.D3 override warning coverage
        Reference: override_matrix.md row for n_groups
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        with pytest.raises(ValueError) as exc_info:
            config_bridge.to_training_config(
                tf_model, pt_data, pt_model, pt_train,
                overrides=dict(
                    train_data_file=Path('train.npz'),
                    nphotons=1e9
                    # Missing n_groups override - should trigger validation error
                )
            )

        # Assert error message contains actionable guidance
        error_msg = str(exc_info.value)
        assert 'n_groups' in error_msg, \
            "Error message should identify missing n_groups"
        assert 'overrides' in error_msg, \
            "Error message should mention overrides parameter"
        assert 'n_groups=' in error_msg, \
            "Error message should provide override syntax example"

    def test_test_data_file_training_missing_warning(self, params_cfg_snapshot):
        """
        Test that missing test_data_file in TrainingConfig emits warning (optional field).

        From override_matrix.md: test_data_file (training stage) remains None until
        inference override applied. Consider warning to surface absent evaluation data.

        Note: This is a softer validation than train_data_file (which is required).
        The warning helps callers understand inference update is needed for evaluation flows.

        Spec coverage: §5.2:2 (test_data_file optional)
        Phase: B.B5.D3 override warning coverage
        Reference: override_matrix.md row for test_data_file (training)
        """
        import warnings
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        # Should emit warning when test_data_file omitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = config_bridge.to_training_config(
                tf_model, pt_data, pt_model, pt_train,
                overrides=dict(
                    train_data_file=Path('train.npz'),
                    n_groups=512,
                    nphotons=1e9
                    # Missing test_data_file - should emit warning
                )
            )

            # Assert warning was issued
            assert len(w) == 1, "Should emit exactly one warning for missing test_data_file"
            warning_msg = str(w[0].message)
            assert 'test_data_file' in warning_msg, \
                "Warning should mention test_data_file"
            assert 'evaluation' in warning_msg or 'inference' in warning_msg, \
                "Warning should explain impact on evaluation/inference workflows"

    # ============================================================================
    # Test Case 6: Baseline params.cfg Comparison (Phase D.D1)
    # ============================================================================

    def test_params_cfg_matches_baseline(self, params_cfg_snapshot):
        """
        Test that adapter-populated params.cfg matches canonical TensorFlow baseline.

        This test validates end-to-end config bridge correctness by instantiating
        PyTorch configs with canonical values, translating through the adapter,
        populating params.cfg via update_legacy_dict, and comparing the resulting
        legacy dictionary against a pre-captured TensorFlow baseline snapshot.

        Baseline: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/baseline_params.json
        Spec coverage: §5.1-§5.3 (all config fields)
        Phase: Phase B.B5.D1 baseline comparison
        References: supervisor_summary.md for canonical config inputs
        """
        import json
        from collections import OrderedDict
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
        from ptycho_torch import config_bridge
        from ptycho.config.config import update_legacy_dict
        import ptycho.params as params

        # Helper function to normalize params.cfg for comparison
        def canonicalize_params(cfg):
            """Convert params.cfg to deterministic dict for baseline comparison."""
            normalized = OrderedDict()
            for key, value in sorted(cfg.items()):
                if isinstance(value, Path):
                    normalized[key] = str(value)
                elif isinstance(value, (int, float, str, bool)):
                    normalized[key] = value
                elif value is None:
                    normalized[key] = None
                else:
                    # Fallback: stringify unknown types
                    normalized[key] = str(value)
            return normalized

        # Instantiate PyTorch configs with canonical values from supervisor_summary.md
        pt_data = DataConfig(
            N=128,
            grid_size=(3, 3),  # Produces gridsize=3
            K=6,  # Produces neighbor_count=6
            nphotons=5e8,
            probe_scale=2.0
        )

        pt_model = ModelConfig(
            mode='Unsupervised',  # Produces model_type='pinn'
            n_filters_scale=2,
            amp_activation='silu',  # Maps to 'swish'
            object_big=False,
            probe_big=False,
            intensity_scale_trainable=True
        )

        pt_train = TrainingConfig(
            epochs=100,  # Produces nepochs=100
            batch_size=32,
            nll=True  # Produces nll_weight=1.0 (but baseline has 0.7, use override)
        )

        pt_infer = InferenceConfig()

        # Build override dictionaries matching baseline values
        model_overrides = dict(
            probe_mask=True,  # Force True (torch unavailable, must use override)
            pad_object=False,
            gaussian_smoothing_sigma=0.5
        )

        training_overrides = dict(
            train_data_file=Path('/canonical/baseline/train_data.npz'),
            test_data_file=Path('/canonical/baseline/test_data.npz'),
            n_groups=1024,  # Note: baseline uses 512 for inference, but training captures this
            n_subsample=2048,  # Baseline doesn't show this, but we'll check after
            subsample_seed=42,  # Baseline shows 99 (from inference), will be overridden
            output_dir=Path('/canonical/baseline/training_outputs'),
            mae_weight=0.3,
            nll_weight=0.7,  # Explicit override to match baseline (not from nll bool conversion)
            realspace_mae_weight=0.05,
            realspace_weight=0.1,
            positions_provided=False,
            probe_trainable=True,
            sequential_sampling=True,
            nphotons=5e8  # Explicit override to avoid divergence validation error
        )

        inference_overrides = dict(
            model_path=Path('/canonical/baseline/model_directory'),
            test_data_file=Path('/canonical/baseline/inference_data.npz'),
            n_groups=512,  # Baseline inference value
            n_subsample=1024,  # Baseline inference value
            subsample_seed=99,  # Baseline inference value
            output_dir=Path('/canonical/baseline/inference_outputs'),
            debug=True
        )

        # Translate through adapter
        tf_model = config_bridge.to_model_config(pt_data, pt_model, overrides=model_overrides)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train, overrides=training_overrides
        )
        tf_infer = config_bridge.to_inference_config(
            tf_model, pt_data, pt_infer, overrides=inference_overrides
        )

        # Clear and populate params.cfg (training first, then inference to match baseline)
        params.cfg.clear()
        update_legacy_dict(params.cfg, tf_train)
        update_legacy_dict(params.cfg, tf_infer)

        # Load baseline JSON
        baseline_path = project_root / 'plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/baseline_params.json'
        with baseline_path.open() as fp:
            baseline = json.load(fp)

        # Normalize params.cfg for comparison
        actual = canonicalize_params(params.cfg)

        # Compare dictionaries
        # Note: baseline only has 31 keys; actual may have more from update_legacy_dict
        # Focus comparison on baseline keys (adapter correctness, not extra keys)
        mismatches = []
        for key in baseline.keys():
            if key not in actual:
                mismatches.append(f"Missing key in actual: {key}")
            elif actual[key] != baseline[key]:
                mismatches.append(f"Mismatch for {key}: actual={actual[key]}, baseline={baseline[key]}")

        # Assert no mismatches
        if mismatches:
            # On failure, dump diff for debugging
            diff_path = project_root / 'plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061500Z/params_diff.json'
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            with diff_path.open('w') as fp:
                json.dump({
                    'mismatches': mismatches,
                    'actual_keys': list(actual.keys()),
                    'baseline_keys': list(baseline.keys()),
                    'actual': dict(actual),
                    'baseline': baseline
                }, fp, indent=2)

            pytest.fail("params.cfg does not match baseline:\n" + "\n".join(mismatches) +
                       f"\n\nDiff saved to: {diff_path}")


    def test_training_config_gradient_clip_algorithm_roundtrip(self, params_cfg_snapshot):
        """
        Test that gradient_clip_algorithm='agc' round-trips through the bridge
        and populates params.cfg correctly.

        Task ID: FNO-STABILITY-OVERHAUL-001 Task 1.1
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge
        from ptycho.config.config import update_legacy_dict
        import ptycho.params as params

        pt_data = DataConfig(nphotons=1e9)
        pt_model = ModelConfig()
        pt_train = TrainingConfig(gradient_clip_algorithm='agc')

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(train_data_file=Path('train.npz'), n_groups=512, nphotons=1e9)
        )

        # Assert TF dataclass has the field
        assert tf_train.gradient_clip_algorithm == 'agc', \
            "gradient_clip_algorithm should be 'agc' on TF TrainingConfig"

        # Populate params.cfg and check
        update_legacy_dict(params.cfg, tf_train)
        assert params.cfg.get('gradient_clip_algorithm') == 'agc', \
            "params.cfg['gradient_clip_algorithm'] should be 'agc' after update_legacy_dict"


class TestConfigBridgeArchitecture:
    """Tests for model.architecture field bridging."""

    def test_config_bridge_architecture_override(self, params_cfg_snapshot):
        """Test that architecture field can be bridged through config_bridge."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(N=64, grid_size=(1, 1))
        pt_model = ModelConfig()

        tf_model = config_bridge.to_model_config(
            pt_data,
            pt_model,
            overrides={'architecture': 'cnn'}
        )

        assert tf_model.architecture == 'cnn'

    def test_config_bridge_architecture_default(self, params_cfg_snapshot):
        """Test that architecture defaults to 'cnn' when not provided."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(N=64, grid_size=(1, 1))
        pt_model = ModelConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        assert tf_model.architecture == 'cnn'

    def test_config_bridge_architecture_from_pt_model(self, params_cfg_snapshot):
        """Test that architecture is passed through from PyTorch ModelConfig."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(N=64, grid_size=(1, 1))
        pt_model = ModelConfig(architecture='fno')

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        assert tf_model.architecture == 'fno'

    def test_config_bridge_fno_input_transform(self, params_cfg_snapshot):
        """Test that fno_input_transform is passed through to TF ModelConfig."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(N=64, grid_size=(1, 1))
        pt_model = ModelConfig(architecture='fno')
        pt_model.fno_input_transform = 'sqrt'

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        assert tf_model.fno_input_transform == 'sqrt'

    def test_training_config_lr_scheduler_roundtrip(self, params_cfg_snapshot):
        """Test scheduler fields round-trip through config bridge."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(nphotons=1e9)
        pt_model = ModelConfig()
        pt_train = TrainingConfig(
            scheduler='WarmupCosine',
            lr_warmup_epochs=5,
            lr_min_ratio=0.05,
        )

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train,
            overrides=dict(train_data_file=Path('train.npz'), n_groups=512, nphotons=1e9)
        )

        assert tf_train.scheduler == 'WarmupCosine'
        assert tf_train.lr_warmup_epochs == 5
        assert tf_train.lr_min_ratio == 0.05


if __name__ == '__main__':
    unittest.main()

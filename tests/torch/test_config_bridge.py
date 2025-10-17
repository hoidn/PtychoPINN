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
        from ptycho.config.config import (
            ModelConfig as TFModelConfig,
            TrainingConfig as TFTrainingConfig,
            InferenceConfig as TFInferenceConfig,
            update_legacy_dict
        )
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


class TestConfigBridgeParity(unittest.TestCase):
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

    # ============================================================================
    # Test Case 1: Direct Field Translation
    # ============================================================================

    @pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
        pytest.param('N', 128, 128, id='N-direct'),
        pytest.param('n_filters_scale', 2, 2, id='n_filters_scale-direct'),
        pytest.param('object_big', False, False, id='object_big-direct'),
        pytest.param('probe_big', False, False, id='probe_big-direct'),
    ])
    def test_model_config_direct_fields(self, field_name, pytorch_value, expected_tf_value):
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
        self.assertEqual(getattr(tf_model, field_name), expected_tf_value,
                         f"{field_name} should pass through directly")

    @pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
        pytest.param('batch_size', 32, 32, id='batch_size-direct'),
    ])
    def test_training_config_direct_fields(self, field_name, pytorch_value, expected_tf_value):
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
            overrides=dict(train_data_file=Path('train.npz'), n_groups=512)
        )

        self.assertEqual(getattr(tf_train, field_name), expected_tf_value,
                         f"{field_name} should pass through directly")

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
    def test_model_config_transform_fields(self, pt_field, pt_value, tf_field, tf_value):
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
        self.assertEqual(getattr(tf_model, tf_field), tf_value,
                         f"{pt_field}={pt_value} should transform to {tf_field}={tf_value}")

    @pytest.mark.parametrize('pt_field,pt_value,tf_field,tf_value', [
        pytest.param('epochs', 100, 'nepochs', 100, id='nepochs-rename'),
        pytest.param('nll', True, 'nll_weight', 1.0, id='nll_weight-true'),
        pytest.param('nll', False, 'nll_weight', 0.0, id='nll_weight-false'),
    ])
    def test_training_config_transform_fields(self, pt_field, pt_value, tf_field, tf_value):
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
            overrides=dict(train_data_file=Path('train.npz'), n_groups=512)
        )

        self.assertEqual(getattr(tf_train, tf_field), tf_value,
                         f"{pt_field}={pt_value} should transform to {tf_field}={tf_value}")

    # ============================================================================
    # Test Case 3: Override-Required Fields
    # ============================================================================

    @pytest.mark.parametrize('field_name,override_value,expected_value', [
        pytest.param('pad_object', None, True, id='pad_object-default'),
        pytest.param('pad_object', False, False, id='pad_object-override'),
        pytest.param('gaussian_smoothing_sigma', None, 0.0, id='gaussian_smoothing_sigma-default'),
        pytest.param('gaussian_smoothing_sigma', 0.5, 0.5, id='gaussian_smoothing_sigma-override'),
    ])
    def test_model_config_override_fields(self, field_name, override_value, expected_value):
        """
        Test ModelConfig fields missing from PyTorch that use defaults or overrides.

        Spec coverage: §5.1:9 (pad_object), §5.1:11 (gaussian_smoothing_sigma)
        """
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()

        # Build overrides dict
        overrides = {field_name: override_value} if override_value is not None else {}

        tf_model = config_bridge.to_model_config(pt_data, pt_model, overrides=overrides)

        self.assertEqual(getattr(tf_model, field_name), expected_value,
                         f"{field_name} should use {'override' if override_value is not None else 'default'} value")

    @pytest.mark.parametrize('field_name,override_value,expected_value', [
        pytest.param('mae_weight', None, 0.0, id='mae_weight-default'),
        pytest.param('mae_weight', 0.3, 0.3, id='mae_weight-override'),
        pytest.param('realspace_mae_weight', None, 0.0, id='realspace_mae_weight-default'),
        pytest.param('realspace_weight', None, 0.0, id='realspace_weight-default'),
        pytest.param('positions_provided', None, True, id='positions_provided-default'),
        pytest.param('probe_trainable', None, False, id='probe_trainable-default'),
        pytest.param('sequential_sampling', None, False, id='sequential_sampling-default'),
    ])
    def test_training_config_override_fields(self, field_name, override_value, expected_value):
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

        # Build overrides dict (always include required fields)
        overrides = dict(train_data_file=Path('train.npz'), n_groups=512)
        if override_value is not None:
            overrides[field_name] = override_value

        tf_model = config_bridge.to_model_config(pt_data, pt_model)
        tf_train = config_bridge.to_training_config(
            tf_model, pt_data, pt_model, pt_train, overrides=overrides
        )

        self.assertEqual(getattr(tf_train, field_name), expected_value,
                         f"{field_name} should use {'override' if override_value is not None else 'default'} value")

    @pytest.mark.parametrize('field_name,override_value,expected_value', [
        pytest.param('debug', None, False, id='debug-default'),
        pytest.param('debug', True, True, id='debug-override'),
    ])
    def test_inference_config_override_fields(self, field_name, override_value, expected_value):
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

        self.assertEqual(getattr(tf_infer, field_name), expected_value,
                         f"{field_name} should use {'override' if override_value is not None else 'default'} value")

    # ============================================================================
    # Test Case 4: Default Divergence Detection
    # ============================================================================

    @pytest.mark.parametrize('field_name,pytorch_default,tf_default,test_value', [
        pytest.param('nphotons', 1e5, 1e9, 5e8, id='nphotons-divergence',
                     marks=pytest.mark.mvp),
        pytest.param('probe_scale', 1.0, 4.0, 2.0, id='probe_scale-divergence'),
    ])
    def test_default_divergence_detection(self, field_name, pytorch_default, tf_default, test_value):
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
        self.assertEqual(actual_value, test_value,
                         f"{field_name} should use explicit value {test_value}, not defaults "
                         f"(PyTorch={pytorch_default}, TF={tf_default})")
        self.assertNotEqual(actual_value, pytorch_default,
                            f"Should not fall back to PyTorch default")
        self.assertNotEqual(actual_value, tf_default,
                            f"Should not fall back to TensorFlow default")

    # ============================================================================
    # Test Case 5: Error Handling & Validation
    # ============================================================================

    @pytest.mark.parametrize('invalid_value,expected_error,error_fragment', [
        pytest.param((2, 3), ValueError, 'Non-square grids', id='gridsize-non-square'),
    ])
    def test_gridsize_error_handling(self, invalid_value, expected_error, error_fragment):
        """Test grid_size validation (non-square grids should raise ValueError)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig(grid_size=invalid_value)
        pt_model = ModelConfig()

        with self.assertRaises(expected_error) as cm:
            config_bridge.to_model_config(pt_data, pt_model)

        self.assertIn(error_fragment, str(cm.exception),
                      f"Error message should contain '{error_fragment}'")

    @pytest.mark.parametrize('invalid_value,expected_error,error_fragment', [
        pytest.param('InvalidMode', ValueError, 'Invalid mode', id='model_type-invalid-enum'),
    ])
    def test_model_type_error_handling(self, invalid_value, expected_error, error_fragment):
        """Test mode validation (invalid enum should raise ValueError)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig(mode=invalid_value)

        with self.assertRaises(expected_error) as cm:
            config_bridge.to_model_config(pt_data, pt_model)

        self.assertIn(error_fragment, str(cm.exception),
                      f"Error message should contain '{error_fragment}'")

    @pytest.mark.parametrize('invalid_value,expected_error,error_fragment', [
        pytest.param('unknown_activation', ValueError, 'Unknown activation', id='amp_activation-unknown'),
    ])
    def test_activation_error_handling(self, invalid_value, expected_error, error_fragment):
        """Test amp_activation validation (unknown activation should raise ValueError)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig(amp_activation=invalid_value)

        with self.assertRaises(expected_error) as cm:
            config_bridge.to_model_config(pt_data, pt_model)

        self.assertIn(error_fragment, str(cm.exception),
                      f"Error message should contain '{error_fragment}'")

    def test_train_data_file_required_error(self):
        """Test that missing train_data_file raises actionable ValueError (MVP field)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_train = TrainingConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        with self.assertRaises(ValueError) as cm:
            config_bridge.to_training_config(
                tf_model, pt_data, pt_model, pt_train,
                overrides=dict(n_groups=512)  # Missing train_data_file
            )

        self.assertIn('train_data_file is required', str(cm.exception),
                      "Error message should indicate train_data_file is required")

    def test_model_path_required_error(self):
        """Test that missing model_path raises actionable ValueError (MVP field)."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, InferenceConfig
        from ptycho_torch import config_bridge

        pt_data = DataConfig()
        pt_model = ModelConfig()
        pt_infer = InferenceConfig()

        tf_model = config_bridge.to_model_config(pt_data, pt_model)

        with self.assertRaises(ValueError) as cm:
            config_bridge.to_inference_config(
                tf_model, pt_data, pt_infer,
                overrides=dict(test_data_file=Path('test.npz'), n_groups=512)  # Missing model_path
            )

        self.assertIn('model_path is required', str(cm.exception),
                      "Error message should indicate model_path is required")


if __name__ == '__main__':
    unittest.main()

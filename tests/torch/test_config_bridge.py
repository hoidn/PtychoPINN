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


if __name__ == '__main__':
    unittest.main()

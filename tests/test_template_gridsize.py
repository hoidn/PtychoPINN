"""
Template for testing gridsize-dependent functionality.
Copy this file when creating new tests that involve different gridsize values.

This template ensures proper params.cfg initialization to avoid shape mismatch bugs.
"""

import unittest
import numpy as np
from ptycho import params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.workflows.components import create_ptycho_data_container, load_data
from ptycho.raw_data import RawData


class TestGridsizeTemplate(unittest.TestCase):
    """Template test class for gridsize-dependent tests."""
    
    def setUp(self):
        """Set up test fixtures - CRITICAL for params initialization."""
        # Clear any existing params to ensure clean state
        params.cfg.clear()
        
        # Create test data (you may want to use actual test files)
        self.test_data_path = 'test_data.npz'  # Update with your test data
        
    def tearDown(self):
        """Clean up after tests."""
        # Clear params to avoid interference with other tests
        params.cfg.clear()
        
    def test_gridsize_1_produces_correct_shape(self):
        """Test that gridsize=1 produces single-channel output."""
        # Setup config with gridsize=1
        config = TrainingConfig(
            model=ModelConfig(gridsize=1, N=64),
            n_subsample=128,
            n_groups=100
        )
        
        # CRITICAL: Update legacy params before any data operations
        update_legacy_dict(params.cfg, config)
        
        # Verify params are set correctly
        self.assertEqual(params.cfg.get('gridsize'), 1)
        
        # Load and process data
        data = load_data(self.test_data_path, n_subsample=128)
        container = create_ptycho_data_container(data, config)
        
        # Verify shape
        self.assertEqual(container.X.shape, (100, 64, 64, 1))
        self.assertEqual(container.X.shape[-1], 1, "Should have 1 channel for gridsize=1")
        
    def test_gridsize_2_produces_correct_shape(self):
        """Test that gridsize=2 produces 4-channel output."""
        # Setup config with gridsize=2
        config = TrainingConfig(
            model=ModelConfig(gridsize=2, N=64),
            n_subsample=128,
            n_groups=100
        )
        
        # CRITICAL: Update legacy params before any data operations
        update_legacy_dict(params.cfg, config)
        
        # Verify params are set correctly
        self.assertEqual(params.cfg.get('gridsize'), 2)
        
        # Load and process data
        data = load_data(self.test_data_path, n_subsample=128)
        container = create_ptycho_data_container(data, config)
        
        # Verify shape
        self.assertEqual(container.X.shape, (100, 64, 64, 4))
        self.assertEqual(container.X.shape[-1], 4, "Should have 4 channels for gridsize=2")
        
    def test_oversampling_with_gridsize_2(self):
        """Test that oversampling works correctly with gridsize=2."""
        # Setup config requesting more groups than images
        config = TrainingConfig(
            model=ModelConfig(gridsize=2, N=64),
            n_subsample=128,
            n_groups=1024,  # 8x oversampling
            neighbor_count=7  # Enables K-choose-C oversampling
        )
        
        # CRITICAL: Update legacy params before any data operations
        update_legacy_dict(params.cfg, config)
        
        # Load and process data
        data = load_data(self.test_data_path, n_subsample=128)
        container = create_ptycho_data_container(data, config)
        
        # Verify oversampling worked
        self.assertEqual(container.X.shape[0], 1024, "Should create 1024 groups")
        self.assertEqual(container.X.shape[-1], 4, "Should have 4 channels for gridsize=2")
        
    def test_params_not_initialized_fails_gracefully(self):
        """Test that missing params initialization is caught."""
        # Intentionally skip update_legacy_dict to test error handling
        config = TrainingConfig(
            model=ModelConfig(gridsize=2, N=64),
            n_subsample=128,
            n_groups=100
        )
        
        # Do NOT call update_legacy_dict - simulating the bug
        # update_legacy_dict(params.cfg, config)  # SKIPPED!
        
        # This should either:
        # 1. Fail with a clear error message about params not being initialized
        # 2. Use the fallback behavior (gridsize=None → params.get → default)
        
        data = load_data(self.test_data_path, n_subsample=128)
        
        # With the new gridsize parameter, this should still work
        # but might use default gridsize=1 if params not set
        container = create_ptycho_data_container(data, config)
        
        # Document the expected behavior
        if params.cfg.get('gridsize') is None:
            # Fallback behavior - should get default gridsize=1
            self.assertEqual(container.X.shape[-1], 1, 
                           "Without params init, should fallback to gridsize=1")
        
    def test_direct_generate_grouped_data_with_explicit_gridsize(self):
        """Test the refactored generate_grouped_data with explicit gridsize."""
        # Create raw data
        raw_data = RawData.from_file(self.test_data_path)
        
        # Test with explicit gridsize parameter (new behavior)
        result = raw_data.generate_grouped_data(
            N=64,
            K=7, 
            nsamples=100,
            gridsize=2  # Explicit parameter - doesn't need params.cfg!
        )
        
        # Should work without params.cfg being set
        self.assertEqual(result['diffraction'].shape, (100, 64, 64, 4))


# Test runner for debugging shape issues
def debug_shape_mismatch():
    """
    Diagnostic function to debug shape mismatches.
    Run this when you get unexpected shapes.
    """
    from ptycho import params
    from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
    
    print("=== Debugging Shape Mismatch ===")
    print(f"Current params.cfg: {params.cfg}")
    
    # Test without initialization
    print("\n1. Testing WITHOUT update_legacy_dict:")
    config = TrainingConfig(model=ModelConfig(gridsize=2))
    print(f"   Config gridsize: {config.model.gridsize}")
    print(f"   Params gridsize: {params.cfg.get('gridsize', 'NOT SET')}")
    
    # Test with initialization  
    print("\n2. Testing WITH update_legacy_dict:")
    update_legacy_dict(params.cfg, config)
    print(f"   Config gridsize: {config.model.gridsize}")
    print(f"   Params gridsize: {params.cfg.get('gridsize', 'NOT SET')}")
    
    # Test data generation
    print("\n3. Testing data generation:")
    try:
        from ptycho.workflows.components import load_data, create_ptycho_data_container
        data = load_data('test_data.npz', n_subsample=128)
        container = create_ptycho_data_container(data, config)
        print(f"   Result shape: {container.X.shape}")
        print(f"   Expected: (*, 64, 64, 4) for gridsize=2")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(argv=[''], exit=False)
    
    # Run diagnostic
    print("\n" + "="*50)
    debug_shape_mismatch()
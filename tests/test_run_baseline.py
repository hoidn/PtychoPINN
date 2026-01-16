import unittest
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ptycho.loader import PtychoDataContainer
# Import the function from our test utilities
from tests.test_utilities import _prepare_baseline_data_inputs

class TestRunBaselineDataPrep(unittest.TestCase):
    def test_prepare_baseline_data_flattens_gridsize2_data(self):
        """Test that data shaping logic correctly flattens multi-channel data."""
        # 1. Arrange: Create mock PtychoDataContainer with 4-channel data
        mock_X = tf.random.normal((16, 64, 64, 4))
        mock_Y_I = tf.random.normal((16, 64, 64, 4))
        mock_Y_phi = tf.random.normal((16, 64, 64, 4))
        mock_global_offsets = tf.random.normal((16, 1, 2, 4))  # Scan coordinates
        
        # Create mock train and test data containers
        mock_train_data = PtychoDataContainer(
            X=mock_X, Y_I=mock_Y_I, Y_phi=mock_Y_phi,
            norm_Y_I=1.0, YY_full=None, coords_nominal=None,
            coords_true=None, nn_indices=None,
            global_offsets=None, local_offsets=None, probeGuess=None
        )
        
        mock_test_data = PtychoDataContainer(
            X=mock_X, Y_I=mock_Y_I, Y_phi=mock_Y_phi,
            norm_Y_I=1.0, YY_full=None, coords_nominal=None,
            coords_true=None, nn_indices=None,
            global_offsets=mock_global_offsets, local_offsets=None, probeGuess=None
        )
        
        # Create mock dataset container
        class MockDataset:
            def __init__(self):
                self.train_data = mock_train_data
                self.test_data = mock_test_data
        
        mock_dataset = MockDataset()
        
        # Mock config
        class MockConfig:
            class MockModelConfig:
                gridsize = 2
            model = MockModelConfig()
        
        # 2. Act: Call the data preparation function
        X_out, Y_I_out, Y_phi_out, X_test_out, global_offsets_out = _prepare_baseline_data_inputs(
            mock_dataset, MockConfig()
        )
        
        # 3. Assert: Check that the output tensors are flattened
        # Batch dimension should be 16 * 4 = 64
        # Channel dimension should be 1
        self.assertEqual(X_out.shape, (64, 64, 64, 1), 
                        f"X_train shape incorrect: {X_out.shape}")
        self.assertEqual(Y_I_out.shape, (64, 64, 64, 1),
                        f"Y_I_train shape incorrect: {Y_I_out.shape}")
        self.assertEqual(Y_phi_out.shape, (64, 64, 64, 1),
                        f"Y_phi_train shape incorrect: {Y_phi_out.shape}")
        self.assertEqual(X_test_out.shape, (64, 64, 64, 1),
                        f"X_test shape incorrect: {X_test_out.shape}")
        self.assertEqual(global_offsets_out.shape, (64, 1, 2, 1),
                        f"global_offsets shape incorrect: {global_offsets_out.shape}")
    
    def test_prepare_baseline_data_passes_through_gridsize1(self):
        """Test that gridsize=1 data passes through unchanged."""
        # 1. Arrange: Create mock PtychoDataContainer with single-channel data
        mock_X = tf.random.normal((16, 64, 64, 1))
        mock_Y_I = tf.random.normal((16, 64, 64, 1))
        mock_Y_phi = tf.random.normal((16, 64, 64, 1))
        mock_global_offsets = tf.random.normal((16, 1, 2, 1))
        
        # Create mock containers
        mock_train_data = PtychoDataContainer(
            X=mock_X, Y_I=mock_Y_I, Y_phi=mock_Y_phi,
            norm_Y_I=1.0, YY_full=None, coords_nominal=None,
            coords_true=None, nn_indices=None,
            global_offsets=None, local_offsets=None, probeGuess=None
        )
        
        mock_test_data = PtychoDataContainer(
            X=mock_X, Y_I=mock_Y_I, Y_phi=mock_Y_phi,
            norm_Y_I=1.0, YY_full=None, coords_nominal=None,
            coords_true=None, nn_indices=None,
            global_offsets=mock_global_offsets, local_offsets=None, probeGuess=None
        )
        
        # Create mock dataset container
        class MockDataset:
            def __init__(self):
                self.train_data = mock_train_data
                self.test_data = mock_test_data
        
        mock_dataset = MockDataset()
        
        # Mock config for gridsize=1
        class MockConfig:
            class MockModelConfig:
                gridsize = 1
            model = MockModelConfig()
        
        # 2. Act: Call the data preparation function
        X_out, Y_I_out, Y_phi_out, X_test_out, global_offsets_out = _prepare_baseline_data_inputs(
            mock_dataset, MockConfig()
        )
        
        # 3. Assert: Check that shapes are unchanged
        self.assertEqual(X_out.shape, (16, 64, 64, 1))
        self.assertEqual(Y_I_out.shape, (16, 64, 64, 1))
        self.assertEqual(Y_phi_out.shape, (16, 64, 64, 1))
        self.assertEqual(X_test_out.shape, (16, 64, 64, 1))
        self.assertEqual(global_offsets_out.shape, (16, 1, 2, 1))

if __name__ == '__main__':
    unittest.main()
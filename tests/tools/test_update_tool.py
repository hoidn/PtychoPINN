#!/usr/bin/env python3
"""Test script for update_tool.py"""

import numpy as np
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import from correct location
from scripts.tools.update_tool import update_object_guess


def create_test_npz(filepath):
    """Create a test NPZ file with typical ptychography data structure"""
    # Create test data
    n_positions = 100
    detector_size = 128
    object_size = 256
    
    test_data = {
        'xcoords': np.random.rand(n_positions) * 100,
        'ycoords': np.random.rand(n_positions) * 100,
        'diff3d': np.random.rand(n_positions, detector_size, detector_size).astype(np.float64),
        'probeGuess': np.random.rand(detector_size, detector_size).astype(np.complex128),
        'objectGuess': np.random.rand(object_size, object_size).astype(np.complex128),
        'scan_index': np.arange(n_positions)
    }
    
    np.savez_compressed(filepath, **test_data)
    return test_data


class TestUpdateTool(unittest.TestCase):
    """Test suite for update_tool.py functions."""

    def test_update_function(self):
        """Test the update_object_guess function"""
        print("Testing update_object_guess function...")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            original_path = os.path.join(tmpdir, 'original.npz')
            recon_path = os.path.join(tmpdir, 'reconstruction.npy')
            output_path = os.path.join(tmpdir, 'updated.npz')

            # Create original NPZ
            original_data = create_test_npz(original_path)
            print(f"Created test NPZ: {original_path}")

            # Create new reconstruction (3D format like from Tike)
            new_recon = np.random.rand(1, 256, 256).astype(np.complex128)
            np.save(recon_path, new_recon)
            print(f"Created test reconstruction: {recon_path}")

            # Test 1: Update with file path
            print("\nTest 1: Update with file path")
            update_object_guess(original_path, recon_path, output_path)

            # Verify the update
            with np.load(output_path) as updated:
                self.assertIn('objectGuess', updated.files)
                self.assertEqual(updated['objectGuess'].shape, (256, 256))  # Should be squeezed
                self.assertEqual(updated['diff3d'].dtype, np.float32)  # Should be converted
                print("✓ File path update successful")

            # Test 2: Update with numpy array directly
            print("\nTest 2: Update with numpy array")
            output_path2 = os.path.join(tmpdir, 'updated2.npz')
            new_recon_2d = np.random.rand(256, 256).astype(np.complex128)
            update_object_guess(original_path, new_recon_2d, output_path2)

            with np.load(output_path2) as updated:
                self.assertTrue(np.array_equal(updated['objectGuess'], new_recon_2d))
                print("✓ Direct array update successful")

            # Test 3: Verify all other data is preserved
            print("\nTest 3: Verify data preservation")
            with np.load(output_path) as updated:
                for key in original_data:
                    if key not in ['objectGuess', 'diff3d']:
                        self.assertTrue(np.array_equal(updated[key], original_data[key]))
                print("✓ All other data preserved correctly")

        print("\nAll tests passed! ✓")


def test_update_function():
    """Legacy function for backward compatibility."""
    suite = unittest.TestSuite()
    test_case = TestUpdateTool('test_update_function')
    suite.addTest(test_case)

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if result.wasSuccessful():
        print("Update tool test passed")
    else:
        print("Update tool test failed")


if __name__ == "__main__":
    unittest.main()

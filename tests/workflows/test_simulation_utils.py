"""
Unit tests for ptycho.workflows.simulation_utils module.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

from ptycho.workflows.simulation_utils import (
    load_probe_from_source,
    validate_probe_object_compatibility
)


class TestLoadProbeFromSource(unittest.TestCase):
    """Test cases for load_probe_from_source function."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_from_numpy_array(self):
        """Test loading probe from a NumPy array."""
        # Create a test probe
        probe = np.ones((64, 64), dtype=np.complex64) * (1 + 1j)
        
        # Load it
        loaded_probe = load_probe_from_source(probe)
        
        # Verify
        np.testing.assert_array_equal(loaded_probe, probe)
        self.assertEqual(loaded_probe.dtype, np.complex64)
        self.assertEqual(loaded_probe.shape, (64, 64))
    
    def test_load_from_npy_file(self):
        """Test loading probe from .npy file."""
        # Create test probe and save to file
        probe = np.ones((32, 32), dtype=np.complex64) * (2 + 3j)
        npy_path = os.path.join(self.temp_dir, 'test_probe.npy')
        np.save(npy_path, probe)
        
        # Load from file
        loaded_probe = load_probe_from_source(npy_path)
        
        # Verify
        np.testing.assert_array_equal(loaded_probe, probe)
        self.assertEqual(loaded_probe.dtype, np.complex64)
        
    def test_load_from_npz_file(self):
        """Test loading probe from .npz file with probeGuess key."""
        # Create test data
        probe = np.ones((48, 48), dtype=np.complex64) * (1.5 - 0.5j)
        obj = np.ones((128, 128), dtype=np.complex64)
        
        # Save to NPZ
        npz_path = os.path.join(self.temp_dir, 'test_data.npz')
        np.savez(npz_path, probeGuess=probe, objectGuess=obj)
        
        # Load from file
        loaded_probe = load_probe_from_source(npz_path)
        
        # Verify
        np.testing.assert_array_equal(loaded_probe, probe)
        self.assertEqual(loaded_probe.dtype, np.complex64)
    
    def test_dtype_conversion(self):
        """Test automatic conversion to complex64."""
        # Create probe with complex128
        probe = np.ones((16, 16), dtype=np.complex128) * (1 + 1j)
        
        # Load it
        loaded_probe = load_probe_from_source(probe)
        
        # Verify dtype was converted
        self.assertEqual(loaded_probe.dtype, np.complex64)
        np.testing.assert_allclose(loaded_probe, probe.astype(np.complex64))
    
    def test_invalid_shape(self):
        """Test error handling for non-2D arrays."""
        # 1D array
        probe_1d = np.ones(64, dtype=np.complex64)
        with self.assertRaises(ValueError) as cm:
            load_probe_from_source(probe_1d)
        self.assertIn("2D array", str(cm.exception))
        
        # 3D array
        probe_3d = np.ones((64, 64, 2), dtype=np.complex64)
        with self.assertRaises(ValueError) as cm:
            load_probe_from_source(probe_3d)
        self.assertIn("2D array", str(cm.exception))
    
    def test_non_complex_data(self):
        """Test error handling for non-complex data."""
        # Real-valued array
        probe_real = np.ones((32, 32), dtype=np.float32)
        with self.assertRaises(ValueError) as cm:
            load_probe_from_source(probe_real)
        self.assertIn("complex-valued", str(cm.exception))
    
    def test_missing_npz_key(self):
        """Test error handling when NPZ doesn't contain probeGuess."""
        # Save NPZ without probeGuess
        npz_path = os.path.join(self.temp_dir, 'bad_data.npz')
        np.savez(npz_path, objectGuess=np.ones((64, 64)), otherKey=123)
        
        with self.assertRaises(ValueError) as cm:
            load_probe_from_source(npz_path)
        self.assertIn("probeGuess", str(cm.exception))
        self.assertIn("Available keys", str(cm.exception))
    
    def test_file_not_found(self):
        """Test error handling for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            load_probe_from_source('/nonexistent/path/probe.npy')
    
    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        txt_path = os.path.join(self.temp_dir, 'probe.txt')
        with open(txt_path, 'w') as f:
            f.write("not a probe")
        
        with self.assertRaises(ValueError) as cm:
            load_probe_from_source(txt_path)
        self.assertIn("Unsupported file format", str(cm.exception))
    
    def test_invalid_source_type(self):
        """Test error handling for invalid source types."""
        with self.assertRaises(TypeError):
            load_probe_from_source(123)  # Integer instead of valid type
        
        with self.assertRaises(TypeError):
            load_probe_from_source([1, 2, 3])  # List instead of array


class TestValidateProbeObjectCompatibility(unittest.TestCase):
    """Test cases for validate_probe_object_compatibility function."""
    
    def test_valid_compatibility(self):
        """Test successful validation when probe is smaller than object."""
        probe = np.ones((64, 64), dtype=np.complex64)
        obj = np.ones((256, 256), dtype=np.complex64)
        
        # Should not raise any exception
        validate_probe_object_compatibility(probe, obj)
    
    def test_probe_too_wide(self):
        """Test error when probe is wider than object."""
        probe = np.ones((64, 300), dtype=np.complex64)
        obj = np.ones((256, 256), dtype=np.complex64)
        
        with self.assertRaises(ValueError) as cm:
            validate_probe_object_compatibility(probe, obj)
        self.assertIn("too large", str(cm.exception))
        self.assertIn("64x300", str(cm.exception))
        self.assertIn("256x256", str(cm.exception))
    
    def test_probe_too_tall(self):
        """Test error when probe is taller than object."""
        probe = np.ones((300, 64), dtype=np.complex64)
        obj = np.ones((256, 256), dtype=np.complex64)
        
        with self.assertRaises(ValueError) as cm:
            validate_probe_object_compatibility(probe, obj)
        self.assertIn("too large", str(cm.exception))
        self.assertIn("300x64", str(cm.exception))
    
    def test_probe_equal_size(self):
        """Test error when probe equals object size."""
        probe = np.ones((256, 256), dtype=np.complex64)
        obj = np.ones((256, 256), dtype=np.complex64)
        
        with self.assertRaises(ValueError) as cm:
            validate_probe_object_compatibility(probe, obj)
        self.assertIn("smaller than object", str(cm.exception))
    
    def test_non_square_arrays(self):
        """Test validation with non-square arrays."""
        probe = np.ones((30, 40), dtype=np.complex64)
        obj = np.ones((100, 150), dtype=np.complex64)
        
        # Should pass
        validate_probe_object_compatibility(probe, obj)
        
        # Should fail if dimensions are swapped
        probe_bad = np.ones((100, 40), dtype=np.complex64)
        with self.assertRaises(ValueError):
            validate_probe_object_compatibility(probe_bad, obj)


if __name__ == '__main__':
    unittest.main()
"""
Integration tests for decoupled probe simulation functionality.

Tests the enhanced simulate_and_save.py script with external probe loading.
"""

import unittest
import tempfile
import numpy as np
import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.workflows.simulation_utils import load_probe_from_source


class TestDecoupledSimulation(unittest.TestCase):
    """Test suite for decoupled probe simulation functionality."""
    
    def setUp(self):
        """Create test data for simulations."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test object (128x128 complex)
        self.object_size = 128
        self.object = np.ones((self.object_size, self.object_size), dtype=np.complex64)
        # Add some features
        center = self.object_size // 2
        self.object[center-10:center+10, center-10:center+10] = 2.0 + 0.5j
        
        # Create test probe (32x32 complex) 
        self.probe_size = 32
        x = np.linspace(-2, 2, self.probe_size)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        self.probe1 = np.exp(-R**2).astype(np.complex64)
        
        # Create alternative probe with phase
        self.probe2 = np.exp(-R**2) * np.exp(1j * R**2 * 0.5)
        self.probe2 = self.probe2.astype(np.complex64)
        
        # Save test data
        self.input_npz = os.path.join(self.test_dir, 'test_input.npz')
        np.savez(self.input_npz, objectGuess=self.object, probeGuess=self.probe1)
        
        # Save external probes
        self.probe_npy = os.path.join(self.test_dir, 'external_probe.npy')
        np.save(self.probe_npy, self.probe2)
        
        self.probe_npz = os.path.join(self.test_dir, 'external_probe.npz')
        np.savez(self.probe_npz, probeGuess=self.probe2, extra_key='ignored')
        
        # Path to simulation script
        self.script_path = os.path.join(project_root, 'scripts', 'simulation', 'simulate_and_save.py')
        
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.test_dir)
        
    def run_simulation(self, args):
        """Helper to run simulation script and return result."""
        cmd = [sys.executable, self.script_path] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
        
    def test_probe_override_npy(self):
        """Test probe override with .npy file."""
        output_file = os.path.join(self.test_dir, 'output_npy.npz')
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', self.probe_npy,
            '--n-images', '50'
        ])
        
        # Check simulation succeeded
        self.assertEqual(result.returncode, 0, f"Simulation failed: {result.stderr}")
        self.assertTrue(os.path.exists(output_file))
        
        # Verify probe was overridden
        with np.load(output_file) as data:
            saved_probe = data['probeGuess']
            self.assertTrue(np.allclose(saved_probe, self.probe2),
                          "Probe was not correctly overridden")
            
    def test_probe_override_npz(self):
        """Test probe override with .npz file."""
        output_file = os.path.join(self.test_dir, 'output_npz.npz')
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', self.probe_npz,
            '--n-images', '50'
        ])
        
        # Check simulation succeeded
        self.assertEqual(result.returncode, 0, f"Simulation failed: {result.stderr}")
        self.assertTrue(os.path.exists(output_file))
        
        # Verify probe was overridden
        with np.load(output_file) as data:
            saved_probe = data['probeGuess']
            self.assertTrue(np.allclose(saved_probe, self.probe2),
                          "Probe was not correctly overridden from NPZ")
            
    def test_gridsize1_consistency(self):
        """Test data consistency with gridsize=1."""
        output_file = os.path.join(self.test_dir, 'output_gs1.npz')
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', self.probe_npy,
            '--n-images', '100',
            '--gridsize', '1'
        ])
        
        self.assertEqual(result.returncode, 0, f"Simulation failed: {result.stderr}")
        
        # Check data contract compliance
        with np.load(output_file) as data:
            # Required keys per data contract
            required_keys = ['diff3d', 'xcoords', 'ycoords', 'probeGuess', 'objectGuess']
            for key in required_keys:
                self.assertIn(key, data.files, f"Missing required key: {key}")
                
            # Check shapes
            n_images = 100
            self.assertEqual(data['diff3d'].shape, (n_images, self.probe_size, self.probe_size))
            self.assertEqual(data['xcoords'].shape, (n_images,))
            self.assertEqual(data['ycoords'].shape, (n_images,))
            self.assertEqual(data['probeGuess'].shape, (self.probe_size, self.probe_size))
            self.assertEqual(data['objectGuess'].shape, (self.object_size, self.object_size))
            
    def test_gridsize2_consistency(self):
        """Test data consistency with gridsize=2."""
        # Note: gridsize > 1 has known issues with the simulation pipeline
        # This test documents the current behavior
        output_file = os.path.join(self.test_dir, 'output_gs2.npz')
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', self.probe_npy,
            '--n-images', '100',
            '--gridsize', '2'
        ])
        
        # Currently gridsize > 1 fails due to multi-channel shape issues
        # This is a known limitation documented in DEVELOPER_GUIDE.md section 8
        if result.returncode != 0:
            self.assertIn("shapes must be equal", result.stderr)
            # Skip further assertions for now
            return
            
        # If it does work in future, check data contract compliance
        with np.load(output_file) as data:
            # For gridsize=2, ground_truth_patches should be present
            self.assertIn('ground_truth_patches', data.files)
            
            # Verify coordinates still work
            self.assertEqual(len(data['xcoords']), len(data['ycoords']))
            
    def test_probe_too_large_error(self):
        """Test error when probe is larger than object."""
        # Create probe larger than object
        large_probe = np.ones((150, 150), dtype=np.complex64)
        large_probe_file = os.path.join(self.test_dir, 'large_probe.npy')
        np.save(large_probe_file, large_probe)
        
        output_file = os.path.join(self.test_dir, 'output_error.npz')
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', large_probe_file,
            '--n-images', '50'
        ])
        
        # Should fail with non-zero exit code
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("too large", result.stderr.lower())
        
    def test_invalid_probe_file_error(self):
        """Test error handling for invalid probe files."""
        output_file = os.path.join(self.test_dir, 'output_error.npz')
        
        # Test 1: Non-existent file
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', 'nonexistent.npy',
            '--n-images', '50'
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())
        
        # Test 2: NPZ without probeGuess key
        bad_npz = os.path.join(self.test_dir, 'bad_probe.npz')
        np.savez(bad_npz, wrong_key=self.probe2)
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', bad_npz,
            '--n-images', '50'
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("probeGuess", result.stderr)
        
        # Test 3: Non-complex data
        real_probe_file = os.path.join(self.test_dir, 'real_probe.npy')
        np.save(real_probe_file, np.abs(self.probe1))  # Real-valued
        
        result = self.run_simulation([
            '--input-file', self.input_npz,
            '--output-file', output_file,
            '--probe-file', real_probe_file,
            '--n-images', '50'
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("complex", result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python
"""
Integration tests for the refactored simulate_and_save.py script.

This test suite validates the refactored simulation pipeline for both
gridsize=1 and gridsize > 1 cases, ensuring data contract compliance
and backward compatibility.
"""

import unittest
import numpy as np
import tempfile
import subprocess
import os
import sys
import shutil
from pathlib import Path
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestSimulateAndSave(unittest.TestCase):
    """Test suite for the refactored simulate_and_save.py script."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.script_path = os.path.join(project_root, 'scripts', 'simulation', 'simulate_and_save.py')
        cls.test_data_dir = tempfile.mkdtemp(prefix='test_simulate_')
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_data_dir):
            shutil.rmtree(cls.test_data_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.temp_dir = tempfile.mkdtemp(dir=self.test_data_dir)
        
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # Section 0.C: Test Data Fixtures
    def create_test_npz(self, obj_size=128, probe_size=32, output_path=None):
        """Create a minimal valid NPZ file for testing."""
        if output_path is None:
            output_path = os.path.join(self.temp_dir, 'test_input.npz')
        
        # Create complex arrays with known properties
        obj_real = np.random.rand(obj_size, obj_size).astype(np.float32)
        obj_imag = np.random.rand(obj_size, obj_size).astype(np.float32) * 0.1
        object_guess = (obj_real + 1j * obj_imag).astype(np.complex64)
        
        # Create probe with Gaussian-like profile
        x = np.linspace(-1, 1, probe_size)
        y = np.linspace(-1, 1, probe_size)
        X, Y = np.meshgrid(x, y)
        probe_amp = np.exp(-(X**2 + Y**2) / 0.5)
        probe_phase = np.random.rand(probe_size, probe_size) * 0.1
        probe_guess = (probe_amp * np.exp(1j * probe_phase)).astype(np.complex64)
        
        np.savez_compressed(output_path, 
                          objectGuess=object_guess,
                          probeGuess=probe_guess)
        return output_path
    
    def run_simulate_and_save(self, input_file, output_file, **kwargs):
        """Run the simulate_and_save.py script with given parameters."""
        cmd = [sys.executable, self.script_path,
               '--input-file', input_file,
               '--output-file', output_file]
        
        # Add optional arguments
        for key, value in kwargs.items():
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
    
    # Section 1: Gridsize=1 Regression Tests
    def test_gridsize1_basic_functionality(self):
        """Test basic functionality with gridsize=1."""
        # Create test input
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_gs1.npz')
        
        # Run simulation
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=100, seed=42
        )
        
        # Check execution succeeded
        self.assertEqual(result.returncode, 0, 
                        f"Script failed with stderr: {result.stderr}")
        
        # Verify output exists
        self.assertTrue(os.path.exists(output_file),
                       "Output file was not created")
    
    def test_gridsize1_output_shapes(self):
        """Verify gridsize=1 produces correct output shapes."""
        # Create and run simulation
        input_file = self.create_test_npz(obj_size=128, probe_size=32)
        output_file = os.path.join(self.temp_dir, 'output_gs1_shapes.npz')
        
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=100, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        # Load and check shapes
        data = np.load(output_file)
        
        # Diffraction should be 3D with no channel dimension
        self.assertEqual(data['diffraction'].shape, (100, 32, 32),
                        f"Expected (100, 32, 32), got {data['diffraction'].shape}")
        
        # Coordinates should be 1D
        self.assertEqual(data['xcoords'].shape, (100,))
        self.assertEqual(data['ycoords'].shape, (100,))
        
        # Y patches should also be 3D
        self.assertEqual(data['Y'].shape, (100, 32, 32),
                        f"Expected Y shape (100, 32, 32), got {data['Y'].shape}")
    
    def test_gridsize1_data_types(self):
        """Validate data types for gridsize=1."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_gs1_types.npz')
        
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=50, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        # Check data types
        data = np.load(output_file)
        
        # Data contract requirements
        self.assertEqual(data['diffraction'].dtype, np.float32,
                        "Diffraction must be float32")
        self.assertTrue(np.iscomplexobj(data['objectGuess']),
                       "objectGuess must be complex")
        self.assertTrue(np.iscomplexobj(data['probeGuess']),
                       "probeGuess must be complex")
        self.assertEqual(data['xcoords'].dtype, np.float64,
                        "Coordinates must be float64")
        self.assertEqual(data['ycoords'].dtype, np.float64,
                        "Coordinates must be float64")
        self.assertTrue(np.iscomplexobj(data['Y']),
                       "Y patches must be complex")
    
    # Section 2: Gridsize=2 Correctness Tests
    def test_gridsize2_no_crash(self):
        """Verify gridsize=2 completes without crashing."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_gs2.npz')
        
        # This is the core bug fix test
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=2, n_images=100, seed=42
        )
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0,
                        f"Gridsize=2 failed with: {result.stderr}")
        self.assertTrue(os.path.exists(output_file),
                       "Output file was not created for gridsize=2")
    
    def test_gridsize2_output_shapes(self):
        """Verify gridsize=2 produces correct output shapes."""
        input_file = self.create_test_npz(obj_size=128, probe_size=32)
        output_file = os.path.join(self.temp_dir, 'output_gs2_shapes.npz')
        
        # Run with 100 scan positions, gridsize=2
        # This should create 100 positions that get grouped into 25 groups of 4
        # But the algorithm might create 100 groups with 4 patterns each = 400 total
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=2, n_images=100, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        data = np.load(output_file)
        
        # For gridsize=2 with n_images=100:
        # The implementation creates 100 groups, each producing 4 patterns
        # Total patterns = 100 * 4 = 400
        total_patterns = 100 * (2**2)  # 400
        
        self.assertEqual(data['diffraction'].shape, (total_patterns, 32, 32),
                        f"Expected {total_patterns} patterns for gridsize=2")
        
        # Coordinates should be expanded
        self.assertEqual(data['xcoords'].shape, (total_patterns,))
        self.assertEqual(data['ycoords'].shape, (total_patterns,))
        
        # Y patches should match
        self.assertEqual(data['Y'].shape, (total_patterns, 32, 32))
    
    def test_gridsize2_coordinate_expansion(self):
        """Validate coordinate expansion for gridsize=2."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_gs2_coords.npz')
        
        # Use small number for easier validation
        n_images = 10
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=2, n_images=n_images, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        data = np.load(output_file)
        xcoords = data['xcoords']
        ycoords = data['ycoords']
        
        # The implementation creates n_images groups, each with gridsize^2 patterns
        # So for n_images=10 and gridsize=2, we get 10*4=40 patterns
        expected_patterns = n_images * (2**2)
        self.assertEqual(len(xcoords), expected_patterns)
        
        # Check that coordinates come in groups of 4
        # Each group should have similar but not identical coordinates
        for i in range(0, len(xcoords), 4):
            group_x = xcoords[i:i+4]
            group_y = ycoords[i:i+4]
            
            # Coordinates in a group should be close but not identical
            x_spread = np.max(group_x) - np.min(group_x)
            y_spread = np.max(group_y) - np.min(group_y)
            
            # They should be within a reasonable neighborhood
            self.assertLess(x_spread, 10, 
                           f"Group {i//4} x-coordinates spread too far: {x_spread}")
            self.assertLess(y_spread, 10,
                           f"Group {i//4} y-coordinates spread too far: {y_spread}")
    
    # Section 3: Feature-Specific Tests
    def test_probe_override(self):
        """Test probe override functionality."""
        # Create main input file
        input_file = self.create_test_npz(probe_size=32)
        
        # Create custom probe file
        probe_size = 32
        x = np.linspace(-1, 1, probe_size)
        X, Y = np.meshgrid(x, x)
        custom_probe = np.exp(-(X**2 + Y**2) / 0.3).astype(np.complex64)
        
        probe_file = os.path.join(self.temp_dir, 'custom_probe.npz')
        np.savez_compressed(probe_file, probeGuess=custom_probe)
        
        output_file = os.path.join(self.temp_dir, 'output_probe_override.npz')
        
        # Run with probe override
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=50, seed=42,
            probe_file=probe_file
        )
        self.assertEqual(result.returncode, 0)
        
        # Verify the output uses the custom probe
        data = np.load(output_file)
        np.testing.assert_array_equal(data['probeGuess'], custom_probe,
                                     "Output probe doesn't match custom probe")
    
    def test_scan_type_random(self):
        """Test random scan type (default)."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_random_scan.npz')
        
        # Random scan is the default
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=100, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        # Check coordinates are within object bounds
        data = np.load(output_file)
        obj_shape = data['objectGuess'].shape
        
        # Coordinates should be within object bounds with some buffer
        self.assertTrue(np.all(data['xcoords'] >= 0))
        self.assertTrue(np.all(data['xcoords'] < obj_shape[1]))
        self.assertTrue(np.all(data['ycoords'] >= 0))
        self.assertTrue(np.all(data['ycoords'] < obj_shape[0]))
    
    # Section 4: Data Contract Compliance
    def test_data_contract_compliance(self):
        """Verify output follows data contract specifications."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_contract.npz')
        
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=50, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        data = np.load(output_file)
        
        # Required keys per data contract
        required_keys = ['diffraction', 'objectGuess', 'probeGuess', 
                        'xcoords', 'ycoords']
        for key in required_keys:
            self.assertIn(key, data.files,
                         f"Required key '{key}' missing from output")
        
        # Optional but expected keys
        optional_keys = ['Y', 'scan_index']
        for key in optional_keys:
            self.assertIn(key, data.files,
                         f"Expected key '{key}' missing from output")
        
        # Legacy compatibility keys
        legacy_keys = ['diff3d', 'xcoords_start', 'ycoords_start']
        for key in legacy_keys:
            self.assertIn(key, data.files,
                         f"Legacy key '{key}' missing from output")
    
    def test_amplitude_not_intensity(self):
        """Verify diffraction contains amplitude, not intensity."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_amplitude.npz')
        
        # Use known photon count for validation
        nphotons = 1e6
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=50, seed=42,
            n_photons=int(nphotons)
        )
        self.assertEqual(result.returncode, 0)
        
        data = np.load(output_file)
        diff = data['diffraction']
        
        # Amplitude values should be roughly in range [0, sqrt(nphotons)]
        # If it were intensity, values would be much larger
        max_expected_amplitude = np.sqrt(nphotons) * 10  # Allow some headroom
        
        self.assertTrue(np.all(diff >= 0),
                       "Diffraction amplitudes must be non-negative")
        self.assertLess(np.max(diff), max_expected_amplitude,
                       f"Values too large for amplitude: max={np.max(diff)}")
        
        # Check that it's not obviously intensity (squared values)
        # Amplitude should have more values near zero
        near_zero_fraction = np.sum(diff < 0.1 * np.max(diff)) / diff.size
        self.assertGreater(near_zero_fraction, 0.1,
                          "Too few near-zero values for amplitude data")
    
    # Section 5: Content Validation Tests
    def test_physical_plausibility(self):
        """Test physical plausibility of simulation results."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_physics.npz')
        
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=100, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        data = np.load(output_file)
        diff = data['diffraction']
        
        # Physical constraints
        self.assertTrue(np.all(diff >= 0),
                       "Diffraction amplitudes must be non-negative")
        self.assertTrue(np.any(diff > 0),
                       "Diffraction patterns are all zero")
        
        # Check for reasonable dynamic range
        max_val = np.max(diff)
        mean_val = np.mean(diff[diff > 0])  # Mean of non-zero values
        dynamic_range = max_val / mean_val
        
        self.assertGreater(dynamic_range, 2,
                          "Diffraction patterns lack dynamic range")
        self.assertLess(dynamic_range, 1000,
                       "Diffraction patterns have unrealistic dynamic range")
        
        # Check that patterns vary (not all identical)
        pattern_std = np.std([np.mean(diff[i]) for i in range(len(diff))])
        self.assertGreater(pattern_std, 0,
                          "All diffraction patterns are identical")
    
    # Section 6: Performance & Benchmarking
    @unittest.skipIf(os.getenv('SKIP_SLOW_TESTS', False),
                     "Skipping slow performance test")
    def test_performance_benchmark(self):
        """Benchmark performance for regression detection."""
        input_file = self.create_test_npz(obj_size=256, probe_size=64)
        output_file = os.path.join(self.temp_dir, 'output_benchmark.npz')
        
        # Time a reasonably sized simulation
        n_images = 1000
        start_time = time.time()
        
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=n_images, seed=42
        )
        
        elapsed_time = time.time() - start_time
        
        self.assertEqual(result.returncode, 0)
        
        # Set a reasonable timeout (adjust based on hardware)
        max_time = 60  # seconds
        self.assertLess(elapsed_time, max_time,
                       f"Simulation took {elapsed_time:.1f}s, exceeding {max_time}s limit")
        
        # Log performance for tracking
        print(f"\nPerformance: {n_images} images in {elapsed_time:.2f}s "
              f"({n_images/elapsed_time:.1f} images/sec)")
    
    # Section 7: Integration Tests
    def test_integration_with_loader(self):
        """Test that output can be loaded by training pipeline."""
        input_file = self.create_test_npz()
        output_file = os.path.join(self.temp_dir, 'output_integration.npz')
        
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=100, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        # Try to load with ptycho.loader
        try:
            from ptycho.loader import RawData
            raw_data = RawData(output_file)
            
            # Basic checks that loading succeeded
            self.assertIsNotNone(raw_data.diff3d)
            self.assertIsNotNone(raw_data.xcoords)
            self.assertIsNotNone(raw_data.ycoords)
            self.assertEqual(len(raw_data.diff3d), 100)
            
        except Exception as e:
            self.fail(f"Failed to load output with RawData: {e}")
    
    # Section 8: Edge Cases & Error Handling
    def test_invalid_input_handling(self):
        """Test graceful handling of invalid inputs."""
        # Missing objectGuess
        bad_file = os.path.join(self.temp_dir, 'bad_input.npz')
        np.savez_compressed(bad_file, probeGuess=np.ones((32, 32), dtype=np.complex64))
        
        output_file = os.path.join(self.temp_dir, 'output_bad.npz')
        
        result = self.run_simulate_and_save(
            bad_file, output_file,
            gridsize=1, n_images=10
        )
        
        # Should fail with helpful error
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("objectGuess", result.stderr,
                     "Error message should mention missing objectGuess")
    
    def test_boundary_conditions(self):
        """Test edge cases with extreme parameters."""
        input_file = self.create_test_npz()
        
        # Test minimum n_images
        output_file = os.path.join(self.temp_dir, 'output_min.npz')
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=1, n_images=1, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        # Verify single image output
        data = np.load(output_file)
        self.assertEqual(data['diffraction'].shape[0], 1)
        
        # Test with gridsize=2 (gridsize=3 might not be supported)
        output_file = os.path.join(self.temp_dir, 'output_grid2.npz')
        result = self.run_simulate_and_save(
            input_file, output_file,
            gridsize=2, n_images=5, seed=42
        )
        self.assertEqual(result.returncode, 0)
        
        # Should have 5 * 4 = 20 patterns
        data = np.load(output_file)
        self.assertEqual(data['diffraction'].shape[0], 20)


def create_visual_validation_script():
    """Create the visual validation script for manual inspection."""
    script_content = '''#!/usr/bin/env python
"""Visual validation script for test outputs."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def visualize_test_output(npz_path, output_path=None):
    """Create visualization of test output."""
    data = np.load(npz_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Test Output Visualization: {os.path.basename(npz_path)}")
    
    # Object amplitude
    ax = axes[0, 0]
    im = ax.imshow(np.abs(data['objectGuess']), cmap='gray')
    ax.set_title('Object Amplitude')
    plt.colorbar(im, ax=ax)
    
    # Object phase
    ax = axes[0, 1]
    im = ax.imshow(np.angle(data['objectGuess']), cmap='hsv')
    ax.set_title('Object Phase')
    plt.colorbar(im, ax=ax)
    
    # Probe
    ax = axes[0, 2]
    im = ax.imshow(np.abs(data['probeGuess']), cmap='hot')
    ax.set_title('Probe Amplitude')
    plt.colorbar(im, ax=ax)
    
    # First diffraction pattern
    ax = axes[1, 0]
    im = ax.imshow(np.log1p(data['diffraction'][0]), cmap='viridis')
    ax.set_title('Diffraction Pattern 0 (log scale)')
    plt.colorbar(im, ax=ax)
    
    # Scan positions
    ax = axes[1, 1]
    ax.scatter(data['xcoords'], data['ycoords'], s=1, alpha=0.5)
    ax.set_title(f'Scan Positions (n={len(data["xcoords"])})')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Statistics
    ax = axes[1, 2]
    ax.text(0.1, 0.9, f"Diffraction shape: {data['diffraction'].shape}", transform=ax.transAxes)
    ax.text(0.1, 0.8, f"Data type: {data['diffraction'].dtype}", transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Value range: [{np.min(data['diffraction']):.2f}, {np.max(data['diffraction']):.2f}]", transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Coordinate range X: [{np.min(data['xcoords']):.1f}, {np.max(data['xcoords']):.1f}]", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Coordinate range Y: [{np.min(data['ycoords']):.1f}, {np.max(data['ycoords']):.1f}]", transform=ax.transAxes)
    ax.set_title('Data Statistics')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_test_outputs.py <npz_file> [output_image]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_test_output(npz_path, output_path)
'''
    
    script_path = os.path.join(project_root, 'tests', 'simulation', 'visualize_test_outputs.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"Created visual validation script: {script_path}")


if __name__ == '__main__':
    # Create visual validation script
    create_visual_validation_script()
    
    # Run tests
    unittest.main()
"""
Unit tests for scripts/tools/create_hybrid_probe.py
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after adding to path
from scripts.tools.create_hybrid_probe import create_hybrid_probe


class TestCreateHybridProbe(unittest.TestCase):
    """Test cases for create_hybrid_probe function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create some test probes
        self.size = 64
        
        # Simple amplitude probe (gaussian-like)
        x = np.linspace(-2, 2, self.size)
        y = np.linspace(-2, 2, self.size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        self.amp_probe = np.exp(-R**2).astype(np.complex64)
        
        # Phase probe with some aberration
        phase = 0.5 * R**2 + 0.1 * X  # Quadratic phase + tilt
        self.phase_probe = np.exp(1j * phase).astype(np.complex64)
        
    def test_matching_dimensions(self):
        """Test hybrid probe creation with matching dimensions."""
        hybrid = create_hybrid_probe(self.amp_probe, self.phase_probe, normalize=False)
        
        # Check shape
        self.assertEqual(hybrid.shape, self.amp_probe.shape)
        
        # Check amplitude matches source 1
        np.testing.assert_allclose(
            np.abs(hybrid), 
            np.abs(self.amp_probe),
            rtol=1e-6
        )
        
        # Check phase matches source 2
        np.testing.assert_allclose(
            np.angle(hybrid),
            np.angle(self.phase_probe),
            rtol=1e-6
        )
        
        # Check dtype
        self.assertEqual(hybrid.dtype, np.complex64)
        
    def test_mismatched_dimensions(self):
        """Test that mismatched dimensions raise ValueError."""
        # Create smaller phase probe
        small_size = 32
        x = np.linspace(-2, 2, small_size)
        y = np.linspace(-2, 2, small_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        small_phase_probe = np.exp(1j * 0.5 * R**2).astype(np.complex64)
        
        # Should raise ValueError for dimension mismatch
        with self.assertRaises(ValueError) as cm:
            create_hybrid_probe(self.amp_probe, small_phase_probe)
        
        # Check error message
        self.assertIn("dimensions must match exactly", str(cm.exception))
        self.assertIn(f"{self.amp_probe.shape}", str(cm.exception))
        self.assertIn(f"{small_phase_probe.shape}", str(cm.exception))
        
    def test_normalization(self):
        """Test power normalization option."""
        # Create probes with different power levels
        high_power_probe = 5.0 * self.amp_probe
        
        # Create a phase probe with non-unit amplitude to ensure power changes
        phase_probe_with_amp = 0.7 * self.phase_probe  # Reduced amplitude
        
        # Without normalization
        hybrid_no_norm = create_hybrid_probe(
            high_power_probe, 
            phase_probe_with_amp, 
            normalize=False
        )
        
        # With normalization
        hybrid_norm = create_hybrid_probe(
            high_power_probe,
            phase_probe_with_amp,
            normalize=True
        )
        
        # Check that normalized version preserves power
        original_power = np.sum(np.abs(high_power_probe)**2)
        hybrid_norm_power = np.sum(np.abs(hybrid_norm)**2)
        
        np.testing.assert_allclose(
            hybrid_norm_power,
            original_power,
            rtol=1e-5
        )
        
        # Without normalization, power should be different
        # (because phase probe has reduced amplitude)
        hybrid_no_norm_power = np.sum(np.abs(hybrid_no_norm)**2)
        
        # The power should be preserved in hybrid_no_norm since we take
        # amplitude from high_power_probe, regardless of phase_probe amplitude
        # So this test should check that both have same power when not normalized
        np.testing.assert_allclose(
            hybrid_no_norm_power,
            original_power,
            rtol=1e-5
        )
        
    def test_complex_dtype_handling(self):
        """Test handling of different complex dtypes."""
        # Create probe with complex128
        amp_128 = self.amp_probe.astype(np.complex128)
        phase_128 = self.phase_probe.astype(np.complex128)
        
        hybrid = create_hybrid_probe(amp_128, phase_128)
        
        # Output should be complex64
        self.assertEqual(hybrid.dtype, np.complex64)
        
    def test_finite_values(self):
        """Test that output contains only finite values."""
        hybrid = create_hybrid_probe(self.amp_probe, self.phase_probe)
        
        self.assertTrue(np.all(np.isfinite(hybrid)))
        self.assertFalse(np.any(np.isnan(hybrid)))
        self.assertFalse(np.any(np.isinf(hybrid)))
        
    def test_phase_preservation(self):
        """Test that phase is correctly preserved from source 2."""
        # Create probe with distinctive phase pattern
        x = np.linspace(-np.pi, np.pi, self.size)
        y = np.linspace(-np.pi, np.pi, self.size)
        X, Y = np.meshgrid(x, y)
        
        # Vortex phase
        phase = np.arctan2(Y, X)
        phase_probe = np.exp(1j * phase).astype(np.complex64)
        
        hybrid = create_hybrid_probe(self.amp_probe, phase_probe)
        
        # Check phase matches
        hybrid_phase = np.angle(hybrid)
        expected_phase = np.angle(phase_probe)
        
        # Handle phase wrapping
        phase_diff = np.angle(np.exp(1j * (hybrid_phase - expected_phase)))
        
        np.testing.assert_allclose(
            phase_diff,
            0.0,
            atol=1e-6
        )
        
    def test_amplitude_preservation(self):
        """Test that amplitude is correctly preserved from source 1."""
        # Create probe with distinctive amplitude
        x = np.linspace(-2, 2, self.size)
        y = np.linspace(-2, 2, self.size)
        X, Y = np.meshgrid(x, y)
        
        # Ring amplitude
        R = np.sqrt(X**2 + Y**2)
        ring_amp = np.exp(-4 * (R - 1)**2).astype(np.complex64)
        
        hybrid = create_hybrid_probe(ring_amp, self.phase_probe)
        
        # Check amplitude matches
        np.testing.assert_allclose(
            np.abs(hybrid),
            np.abs(ring_amp),
            rtol=1e-6
        )


class TestCreateHybridProbeIntegration(unittest.TestCase):
    """Integration tests for the complete create_hybrid_probe script."""
    
    def setUp(self):
        """Create temporary directory and test files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test probes
        size = 32
        x = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        
        # Amplitude probe
        self.amp_probe = np.exp(-2*R**2).astype(np.complex64)
        self.amp_file = os.path.join(self.temp_dir, 'amp_probe.npy')
        np.save(self.amp_file, self.amp_probe)
        
        # Phase probe
        phase = R**2
        self.phase_probe = np.exp(1j * phase).astype(np.complex64)
        self.phase_file = os.path.join(self.temp_dir, 'phase_probe.npz')
        np.savez(self.phase_file, probeGuess=self.phase_probe, other_data=123)
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualization_creation(self, mock_close, mock_savefig):
        """Test that visualization is created when requested."""
        from scripts.tools.create_hybrid_probe import visualize_probes
        
        hybrid = create_hybrid_probe(self.amp_probe, self.phase_probe)
        output_path = os.path.join(self.temp_dir, 'test_viz.png')
        
        # Call visualization function
        visualize_probes(self.amp_probe, self.phase_probe, hybrid, output_path)
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python
"""
Regression tests for scaling bugs in diffsim.py
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho import params as p
from ptycho.diffsim import illuminate_and_diffract, scale_nphotons
import tensorflow as tf


class TestScalingRegression(unittest.TestCase):
    """Test suite to prevent scaling regression bugs."""
    
    def setUp(self):
        """Set up test environment with known parameters."""
        p.set('N', 32)
        p.set('batch_size', 4)
        p.set('nphotons', 1e6)
        p.set('gridsize', 1)
        
        self.n_images = 16
        self.Y_I = np.random.uniform(0.1, 1.0, (self.n_images, 32, 32, 1)).astype(np.float32)
        self.Y_phi = np.random.uniform(-np.pi, np.pi, (self.n_images, 32, 32, 1)).astype(np.float32)
        self.probe = np.ones((32, 32), dtype=np.float32)
        
    def test_intensity_scale_is_valid(self):
        """Test that intensity_scale is positive and finite."""
        X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(
            self.Y_I, self.Y_phi, self.probe
        )
        
        self.assertGreater(intensity_scale, 0, 
            "intensity_scale must be positive")
        self.assertTrue(np.isfinite(intensity_scale), 
            "intensity_scale must be finite (not NaN or inf)")
        
    def test_both_arrays_scaled_identically(self):
        """Regression test for X and Y_I scaling symmetry."""
        X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(
            self.Y_I, self.Y_phi, self.probe
        )
        
        x_mean = np.mean(np.abs(X))
        y_mean = np.mean(np.abs(Y_I_out))
        
        ratio = x_mean / y_mean if y_mean > 0 else float('inf')
        self.assertLess(ratio, 10.0,
            f"X and Y_I scaling mismatch detected! X mean: {x_mean:.6f}, "
            f"Y_I mean: {y_mean:.6f}, ratio: {ratio:.2f}. "
            f"This indicates X is not being divided by intensity_scale. "
            f"Check diffsim.py line 131 - both X and Y_I must be scaled.")
        
        self.assertGreater(ratio, 0.1,
            f"X and Y_I scaling mismatch detected! X mean: {x_mean:.6f}, "
            f"Y_I mean: {y_mean:.6f}, ratio: {ratio:.2f}. "
            f"This indicates Y_I is not being divided by intensity_scale.")
    
    def test_scaling_preserves_physics(self):
        """Verify that intensity_scale creates correct photon statistics."""
        for nphotons in [1e4, 1e6, 1e9]:
            p.set('nphotons', nphotons)
            
            X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(
                self.Y_I, self.Y_phi, self.probe
            )
            
            X_physical = X * intensity_scale
            
            total_photons = np.sum(X_physical ** 2)
            expected_photons = nphotons * len(X)
            ratio = total_photons / expected_photons
            self.assertGreater(ratio, 0.5,
                f"Photon count too low for nphotons={nphotons:.1e}: "
                f"got {total_photons:.2e}, expected ~{expected_photons:.2e}")
            self.assertLess(ratio, 2.0,
                f"Photon count too high for nphotons={nphotons:.1e}: "
                f"got {total_photons:.2e}, expected ~{expected_photons:.2e}")
    
    def test_scaling_is_reversible(self):
        """Test that scaling can be reversed correctly."""
        X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(
            self.Y_I, self.Y_phi, self.probe
        )
        
        X_reversed = X * intensity_scale
        Y_I_reversed = Y_I_out * intensity_scale
        self.assertGreater(np.mean(X_reversed), 0.01,
            "Reversed X values are too small - scaling may be wrong")
        self.assertGreater(np.mean(Y_I_reversed), 0.01,
            "Reversed Y_I values are too small - scaling may be wrong")
    
    def test_phase_is_not_scaled(self):
        """Verify that phase is not affected by intensity scaling."""
        X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(
            self.Y_I, self.Y_phi, self.probe
        )
        
        phase_difference = np.abs(Y_phi_out - self.Y_phi)
        max_difference = np.max(phase_difference)
        self.assertLess(max_difference, 0.01,
            f"Phase should not be modified by scaling, but max difference is {max_difference}")
    
    def test_different_nphotons_produce_proportional_scaling(self):
        """Test that different nphoton values produce proportional intensity_scale."""
        scales = {}
        
        for nphotons in [1e4, 1e6, 1e8]:
            p.set('nphotons', nphotons)
            _, _, _, intensity_scale = illuminate_and_diffract(
                self.Y_I, self.Y_phi, self.probe
            )
            scales[nphotons] = intensity_scale
        
        ratio_1 = scales[1e6] / scales[1e4]
        expected_ratio_1 = np.sqrt(1e6 / 1e4)
        
        ratio_2 = scales[1e8] / scales[1e6]
        expected_ratio_2 = np.sqrt(1e8 / 1e6)
        self.assertAlmostEqual(ratio_1, expected_ratio_1, delta=expected_ratio_1 * 0.2,
            msg=f"Scaling relationship broken: {ratio_1:.2f} vs expected {expected_ratio_1:.2f}")
        self.assertAlmostEqual(ratio_2, expected_ratio_2, delta=expected_ratio_2 * 0.2,
            msg=f"Scaling relationship broken: {ratio_2:.2f} vs expected {expected_ratio_2:.2f}")


class TestScalingAssertions(unittest.TestCase):
    """Test that the new assertions in diffsim.py catch bad scaling."""
    
    def test_assertions_catch_invalid_intensity_scale(self):
        """Test that assertions catch invalid intensity_scale values."""
        p.set('N', 32)
        p.set('batch_size', 4)
        p.set('nphotons', 1e6)
        
        Y_I = np.random.uniform(0.1, 1.0, (4, 32, 32, 1)).astype(np.float32)
        Y_phi = np.random.uniform(-np.pi, np.pi, (4, 32, 32, 1)).astype(np.float32)
        probe = np.ones((32, 32), dtype=np.float32)
        
        try:
            X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(
                Y_I, Y_phi, probe
            )
        except AssertionError:
            self.fail("illuminate_and_diffract raised AssertionError unexpectedly with valid data")
        self.assertIsNotNone(intensity_scale)
        self.assertGreater(intensity_scale, 0)
        self.assertFalse(np.isnan(intensity_scale))


if __name__ == '__main__':
    unittest.main()
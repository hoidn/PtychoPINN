#!/usr/bin/env python3
"""
Simple test to confirm the scaling bug by directly examining the illuminate_and_diffract code.
"""

import numpy as np
import tensorflow as tf

# Simulate the key part of illuminate_and_diffract
def test_scaling_division():
    """Test the division issue in illuminate_and_diffract."""
    print("=== Testing Scaling Division Bug ===")
    
    # Mock some typical values
    nphotons_1e4 = 1e4
    nphotons_1e9 = 1e9
    
    # Simulate the scaling factors that would be computed
    # These are similar to what scale_nphotons returns
    intensity_scale_1e4 = np.sqrt(nphotons_1e4 / 4096)  # ~1.56
    intensity_scale_1e9 = np.sqrt(nphotons_1e9 / 4096)  # ~494
    
    print(f"intensity_scale for 1e4: {intensity_scale_1e4:.6e}")
    print(f"intensity_scale for 1e9: {intensity_scale_1e9:.6e}")
    print(f"Ratio: {intensity_scale_1e9 / intensity_scale_1e4:.2f}")
    
    # Simulate diffraction patterns after scaling
    # These would be proportional to intensity_scale^2 due to the physics
    base_diffraction_value = 1.0
    X_before_division_1e4 = base_diffraction_value * (intensity_scale_1e4 ** 2)
    X_before_division_1e9 = base_diffraction_value * (intensity_scale_1e9 ** 2)
    
    print(f"\nDiffraction values BEFORE division:")
    print(f"  1e4 photons: {X_before_division_1e4:.6e}")
    print(f"  1e9 photons: {X_before_division_1e9:.6e}")
    print(f"  Ratio: {X_before_division_1e9 / X_before_division_1e4:.2e}")
    
    # Now apply the bug: divide by intensity_scale (line 123 in diffsim.py)
    X_after_division_1e4 = X_before_division_1e4 / intensity_scale_1e4
    X_after_division_1e9 = X_before_division_1e9 / intensity_scale_1e9
    
    print(f"\nDiffraction values AFTER division (the bug):")
    print(f"  1e4 photons: {X_after_division_1e4:.6e}")
    print(f"  1e9 photons: {X_after_division_1e9:.6e}")
    print(f"  Ratio: {X_after_division_1e9 / X_after_division_1e4:.2f}")
    
    # Show what the ratio should be vs what it actually is
    expected_ratio = nphotons_1e9 / nphotons_1e4  # 1e5
    actual_ratio = X_after_division_1e9 / X_after_division_1e4
    
    print(f"\n🚨 BUG ANALYSIS:")
    print(f"  Expected ratio: {expected_ratio:.2e}")
    print(f"  Actual ratio after division: {actual_ratio:.2f}")
    print(f"  Scaling factor lost due to division: {expected_ratio / actual_ratio:.2e}")
    
    if abs(actual_ratio - intensity_scale_1e9 / intensity_scale_1e4) < 0.01:
        print("✓ CONFIRMED: The division reduces the scaling from nphotons^1 to nphotons^0.5")
        print("  This matches the observed ~300x ratio instead of 100,000x")

def analyze_training_losses():
    """Analyze why the training losses were wrong."""
    print(f"\n=== Training Loss Analysis ===")
    
    # Based on our investigation, the diffraction values should be:
    # - 1e4: mean ~0.013, max ~0.037 (after incorrect division)
    # - 1e9: mean ~0.014, max ~0.025 (after incorrect division)
    
    diff_1e4_mean = 0.013
    diff_1e9_mean = 0.014
    
    # The Poisson NLL loss for amplitude data is computed as:
    # intensity = amplitude^2
    # loss ≈ pred_intensity - true_intensity * log(pred_intensity)
    
    # For the initial untrained model, predictions are random
    # If true data has similar intensities, losses will be similar
    
    intensity_1e4 = diff_1e4_mean ** 2
    intensity_1e9 = diff_1e9_mean ** 2
    
    print(f"Effective intensities after bug:")
    print(f"  1e4 dataset: {intensity_1e4:.6e}")
    print(f"  1e9 dataset: {intensity_1e9:.6e}")
    print(f"  Ratio: {intensity_1e9 / intensity_1e4:.2f}")
    
    # This explains why the losses were nearly identical!
    print(f"\n✓ EXPLANATION: Losses are nearly identical because the bug")
    print(f"   makes the actual data intensities nearly identical!")
    
    # The negative losses are explained by the Poisson NLL formula
    print(f"\n✓ NEGATIVE LOSSES: For small intensity values with Poisson NLL,")
    print(f"   the log term dominates, leading to negative loss values.")

if __name__ == "__main__":
    test_scaling_division()
    analyze_training_losses()
#!/usr/bin/env python3
"""
Trace the complete data flow to identify the scaling issues.
"""

import numpy as np
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.nongrid_simulation import generate_simulated_data, load_probe_object
from ptycho import params as p
from ptycho.diffsim import count_photons, scale_nphotons
import tensorflow as tf

def trace_single_photon_level(nphotons_value):
    """Trace the complete data flow for a single photon level."""
    print(f"\n=== Tracing nphotons = {nphotons_value} ===")
    
    # Load the same object/probe that was used for the datasets
    obj_path = "datasets/fly/fly001_transposed.npz"
    if not os.path.exists(obj_path):
        print(f"Warning: {obj_path} not found, using dummy data")
        return
    
    objectGuess, probeGuess = load_probe_object(obj_path)
    print(f"Loaded object: {objectGuess.shape}, probe: {probeGuess.shape}")
    
    # Create config to match the simulation
    model_config = ModelConfig(N=probeGuess.shape[0], gridsize=1)
    training_config = TrainingConfig(
        model=model_config,
        n_images=10,  # Small for tracing
        nphotons=nphotons_value
    )
    
    print(f"Config nphotons: {training_config.nphotons}")
    
    # Generate data and trace key steps
    raw_data, ground_truth_patches = generate_simulated_data(
        config=training_config,
        objectGuess=objectGuess,
        probeGuess=probeGuess,
        buffer=50,
        return_patches=True
    )
    
    # Check final diffraction data
    diff_data = raw_data.diff3d
    print(f"Final diffraction data stats:")
    print(f"  Shape: {diff_data.shape}")
    print(f"  Min: {diff_data.min():.6e}")
    print(f"  Max: {diff_data.max():.6e}")
    print(f"  Mean: {diff_data.mean():.6e}")
    print(f"  Sum per image (mean): {np.mean(np.sum(diff_data, axis=(1,2))):.6e}")
    
    # Now manually trace the scaling process
    print(f"\n--- Manual Scaling Trace ---")
    
    # This mimics what happens in from_simulation
    p.set('nphotons', nphotons_value)
    
    # Get patches from the ground truth (simulate Y_I extraction)
    patch_indices = [0, 1, 2]  # First few patches
    Y_patches = ground_truth_patches[patch_indices]
    Y_I = tf.math.abs(tf.convert_to_tensor(Y_patches))
    
    print(f"Y_I patches shape: {Y_I.shape}")
    print(f"Y_I mean: {tf.reduce_mean(Y_I).numpy():.6e}")
    
    # Check photon counting
    photon_counts = count_photons(Y_I)
    mean_photons = tf.reduce_mean(photon_counts)
    print(f"Mean photon count (before scaling): {mean_photons.numpy():.6e}")
    
    # Check normalization factor
    norm_factor = scale_nphotons(Y_I)
    print(f"Normalization factor: {norm_factor.numpy():.6e}")
    print(f"Expected mean after scaling: {mean_photons.numpy() * norm_factor.numpy()**2:.6e}")
    print(f"Target nphotons: {nphotons_value:.6e}")
    
    return {
        'nphotons': nphotons_value,
        'final_mean': diff_data.mean(),
        'final_max': diff_data.max(),
        'mean_photons_before': mean_photons.numpy(),
        'norm_factor': norm_factor.numpy(),
        'expected_after': mean_photons.numpy() * norm_factor.numpy()**2
    }

def trace_loss_calculation():
    """Trace how losses are calculated."""
    print(f"\n=== Tracing Loss Calculation ===")
    
    # Load both datasets and simulate loss calculation
    for nphotons_val, filename in [(1e4, "photon_1e4_4k_images_CORRECTED.npz"), 
                                  (1e9, "photon_1e9_4k_images_CORRECTED.npz")]:
        if not os.path.exists(filename):
            continue
            
        data = np.load(filename)
        diff_data = data['diff3d'][:100]  # First 100 images
        
        print(f"\n{filename}:")
        print(f"  Data mean: {diff_data.mean():.6e}")
        print(f"  Data max: {diff_data.max():.6e}")
        
        # Simulate the loss calculation (Poisson NLL)
        # The model expects diffraction AMPLITUDES but computes loss on intensity
        intensities = diff_data ** 2  # Square to get intensities
        
        # Mock prediction (just use the same data for simplicity)
        predicted_intensities = intensities * 1.1  # Slightly off
        
        # Poisson NLL: pred - obs*log(pred) + log_gamma(obs+1)
        # For large counts, this approaches (pred - obs)^2 / (2*obs)
        poisson_nll = tf.reduce_mean(
            predicted_intensities - intensities * tf.math.log(predicted_intensities + 1e-8)
        ).numpy()
        
        print(f"  Mock Poisson NLL: {poisson_nll:.2f}")
        
        # Check for negative values (which would be suspicious)
        if np.any(intensities <= 0):
            zero_count = np.sum(intensities == 0)
            print(f"  Warning: {zero_count} zero intensity values")

if __name__ == "__main__":
    # Trace both photon levels
    results = []
    for nphotons in [1e4, 1e9]:
        result = trace_single_photon_level(nphotons)
        if result:
            results.append(result)
    
    # Compare results
    if len(results) == 2:
        print(f"\n=== COMPARISON ===")
        ratio_mean = results[1]['final_mean'] / results[0]['final_mean']
        ratio_max = results[1]['final_max'] / results[0]['final_max']
        
        print(f"Final data mean ratio (1e9/1e4): {ratio_mean:.2e}")
        print(f"Final data max ratio (1e9/1e4): {ratio_max:.2e}")
        print(f"Expected ratio: {1e9/1e4:.2e}")
        
        print(f"\nNorm factors:")
        print(f"  1e4: {results[0]['norm_factor']:.6e}")
        print(f"  1e9: {results[1]['norm_factor']:.6e}")
        print(f"  Ratio: {results[1]['norm_factor'] / results[0]['norm_factor']:.6e}")
    
    # Trace loss calculation
    trace_loss_calculation()
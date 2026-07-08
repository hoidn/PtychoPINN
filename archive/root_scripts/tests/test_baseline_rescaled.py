#!/usr/bin/env python
"""
Test baseline inference with rescaled Run1084 data to match fly64 training scale.
This creates a temporary rescaled dataset without modifying the original.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

def main():
    print("Loading datasets for scale analysis...")
    
    # Load both datasets
    fly64 = np.load('datasets/fly64/fly64_shuffled.npz')
    run1084 = np.load('datasets/Run1084_recon3_postPC_shrunk_3.npz')
    
    # Get fly64 statistics for reference
    fly64_diff = fly64['diffraction']
    fly64_mean = fly64_diff.mean()
    fly64_std = fly64_diff.std()
    fly64_max = fly64_diff.max()
    
    print(f"\nfly64 diffraction statistics:")
    print(f"  Mean: {fly64_mean:.3f}")
    print(f"  Std:  {fly64_std:.3f}")
    print(f"  Max:  {fly64_max:.3f}")
    
    # Get Run1084 diffraction patterns
    run1084_diff = run1084['diffraction'].copy()  # Copy to avoid modifying original
    
    # Handle shape if needed (H,W,N) -> (N,H,W)
    if run1084_diff.shape[2] > run1084_diff.shape[0]:
        run1084_diff = np.transpose(run1084_diff, (2, 0, 1))
    
    original_mean = run1084_diff.mean()
    original_std = run1084_diff.std()
    original_max = run1084_diff.max()
    
    print(f"\nRun1084 original diffraction statistics:")
    print(f"  Mean: {original_mean:.3f}")
    print(f"  Std:  {original_std:.3f}")
    print(f"  Max:  {original_max:.3f}")
    
    # Method 1: Scale to match mean and std
    scale_factor = fly64_std / original_std
    shift_factor = fly64_mean - (original_mean * scale_factor)
    
    run1084_rescaled = run1084_diff * scale_factor + shift_factor
    run1084_rescaled = np.maximum(run1084_rescaled, 0)  # Ensure non-negative
    
    print(f"\nRescaling parameters:")
    print(f"  Scale factor: {scale_factor:.3f}")
    print(f"  Shift factor: {shift_factor:.3f}")
    
    print(f"\nRun1084 rescaled diffraction statistics:")
    print(f"  Mean: {run1084_rescaled.mean():.3f}")
    print(f"  Std:  {run1084_rescaled.std():.3f}")
    print(f"  Max:  {run1084_rescaled.max():.3f}")
    
    # Create temporary rescaled dataset
    temp_path = Path('datasets/Run1084_rescaled_temp.npz')
    
    # Save rescaled data with all other arrays unchanged
    np.savez_compressed(
        temp_path,
        diffraction=run1084_rescaled.T if run1084['diffraction'].shape[2] > run1084['diffraction'].shape[0] else run1084_rescaled,  # Keep original shape
        probeGuess=run1084['probeGuess'],
        objectGuess=run1084['objectGuess'],
        scan_index=run1084['scan_index'] if 'scan_index' in run1084 else None,
        xcoords=run1084['xcoords'],
        ycoords=run1084['ycoords'],
        xcoords_start=run1084['xcoords_start'] if 'xcoords_start' in run1084 else None,
        ycoords_start=run1084['ycoords_start'] if 'ycoords_start' in run1084 else None
    )
    
    print(f"\nCreated temporary rescaled dataset: {temp_path}")
    
    # Now run baseline inference with the rescaled data
    print("\n" + "="*60)
    print("Running baseline inference with RESCALED data...")
    print("="*60)
    
    import subprocess
    
    cmd = [
        'python', 'scripts/inference/baseline_inference.py',
        '--model_path', 'experiment_outputs/fly64_trained_models/baseline_run/baseline_model.h5',
        '--test_data_file', str(temp_path),
        '--output_dir', 'experiment_outputs/baseline_rescaled_test',
        '--gridsize', '1'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\nInference completed successfully!")
        print("\nOutputs saved to: experiment_outputs/baseline_rescaled_test/")
        
        # Load and compare the results
        print("\n" + "="*60)
        print("Comparing reconstruction quality...")
        print("="*60)
        
        rescaled_npz = np.load('experiment_outputs/baseline_rescaled_test/baseline_reconstruction.npz')
        original_test_npz_path = Path('experiment_outputs/fly64_trained_models/recon_on_run1084_baseline_rerun/baseline_reconstruction.npz')
        
        if original_test_npz_path.exists():
            original_npz = np.load(original_test_npz_path)
            
            rescaled_phase = rescaled_npz['reconstructed_phase']
            original_phase = original_npz['reconstructed_phase']
            
            rescaled_amp = rescaled_npz['reconstructed_amplitude']
            original_amp = original_npz['reconstructed_amplitude']
            
            print(f"\nOriginal inference (unscaled input):")
            print(f"  Phase unique values: {len(np.unique(original_phase))}")
            print(f"  Phase std dev: {original_phase.std():.6f}")
            print(f"  Amplitude unique values: {len(np.unique(original_amp))}")
            print(f"  Amplitude std dev: {original_amp.std():.6f}")
            
            print(f"\nRescaled inference:")
            print(f"  Phase unique values: {len(np.unique(rescaled_phase))}")
            print(f"  Phase std dev: {rescaled_phase.std():.6f}")
            print(f"  Amplitude unique values: {len(np.unique(rescaled_amp))}")
            print(f"  Amplitude std dev: {rescaled_amp.std():.6f}")
            
            # Visual comparison
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original reconstruction
            axes[0, 0].imshow(original_amp, cmap='gray')
            axes[0, 0].set_title(f'Original Scale - Amplitude\n({len(np.unique(original_amp))} unique values)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(original_phase, cmap='viridis')
            axes[0, 1].set_title(f'Original Scale - Phase\n({len(np.unique(original_phase))} unique values)')
            axes[0, 1].axis('off')
            
            # Rescaled reconstruction
            axes[1, 0].imshow(rescaled_amp, cmap='gray')
            axes[1, 0].set_title(f'Rescaled Input - Amplitude\n({len(np.unique(rescaled_amp))} unique values)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(rescaled_phase, cmap='viridis')
            axes[1, 1].set_title(f'Rescaled Input - Phase\n({len(np.unique(rescaled_phase))} unique values)')
            axes[1, 1].axis('off')
            
            plt.suptitle('Baseline Model: Original vs Rescaled Input Comparison', fontsize=14)
            plt.tight_layout()
            plt.savefig('experiment_outputs/baseline_scale_comparison.png', dpi=150, bbox_inches='tight')
            print(f"\nComparison plot saved to: experiment_outputs/baseline_scale_comparison.png")
            
    else:
        print("\nInference failed!")
        print("STDERR:", result.stderr)
    
    # Clean up temporary file
    if temp_path.exists():
        temp_path.unlink()
        print(f"\nCleaned up temporary file: {temp_path}")

if __name__ == "__main__":
    main()
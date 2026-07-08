#!/usr/bin/env python
"""Check what's really happening with pty-chi reconstruction."""

import numpy as np
import sys
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

# Load the actual comparison results
comparison_path = '3way_synthetic_ptychi_1e4_128/train_128/trial_1/reconstructions_aligned.npz'
with np.load(comparison_path, allow_pickle=True) as data:
    print("Keys in reconstructions_aligned.npz:", list(data.keys()))
    
    if 'ptychi_amp_aligned' in data:
        ptychi_amp = data['ptychi_amp_aligned']
        print(f"\nPty-chi aligned amplitude shape: {ptychi_amp.shape}")
        print(f"Min: {np.min(ptychi_amp):.4f}, Max: {np.max(ptychi_amp):.4f}")
        print(f"Mean: {np.mean(ptychi_amp):.4f}, Std: {np.std(ptychi_amp):.4f}")
    
    if 'pinn_amp_aligned' in data:
        pinn_amp = data['pinn_amp_aligned']
        print(f"\nPINN aligned amplitude shape: {pinn_amp.shape}")
        print(f"Min: {np.min(pinn_amp):.4f}, Max: {np.max(pinn_amp):.4f}")
        print(f"Mean: {np.mean(pinn_amp):.4f}, Std: {np.std(pinn_amp):.4f}")
    
    if 'ground_truth_amp' in data:
        gt_amp = data['ground_truth_amp']
        print(f"\nGround truth amplitude shape: {gt_amp.shape}")
        print(f"Min: {np.min(gt_amp):.4f}, Max: {np.max(gt_amp):.4f}")
        print(f"Mean: {np.mean(gt_amp):.4f}, Std: {np.std(gt_amp):.4f}")
        
    # Check if they're all the same shape after alignment
    if 'ptychi_amp_aligned' in data and 'pinn_amp_aligned' in data:
        print(f"\n✓ All aligned shapes match: {ptychi_amp.shape == pinn_amp.shape == gt_amp.shape}")
        
        # Calculate actual SSIM between aligned images
        from skimage.metrics import structural_similarity
        
        # Normalize for fair comparison
        def normalize(img):
            return (img - img.min()) / (img.max() - img.min() + 1e-10)
        
        ptychi_norm = normalize(ptychi_amp)
        gt_norm = normalize(gt_amp)
        
        ssim_val = structural_similarity(gt_norm, ptychi_norm, data_range=1.0)
        print(f"\nDirect SSIM on aligned images: {ssim_val:.4f}")
        
        # Check correlation
        correlation = np.corrcoef(gt_norm.flatten(), ptychi_norm.flatten())[0, 1]
        print(f"Correlation coefficient: {correlation:.4f}")
        
        if correlation < 0:
            print("⚠️ NEGATIVE CORRELATION - reconstruction is inverted/anti-correlated!")

# Also check the raw pty-chi file
print("\n" + "="*60)
print("RAW PTY-CHI FILE CHECK:")
ptychi_raw = '3way_synthetic_ptychi_1e4_128/train_128/trial_1/ptychi_run/ptychi_reconstruction.npz'
with np.load(ptychi_raw, allow_pickle=True) as data:
    raw_obj = data['reconstructed_object']
    print(f"Raw pty-chi shape: {raw_obj.shape}")
    print(f"Raw pty-chi dtype: {raw_obj.dtype}")
    
print("\nCONCLUSION:")
print("The comparison script likely crops/pads all reconstructions to match for visualization.")
print("The negative SSIM suggests the pty-chi reconstruction is poorly registered or inverted.")
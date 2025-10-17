#!/usr/bin/env python
"""Properly calculate pty-chi MS-SSIM by handling the shape mismatch correctly."""

import numpy as np
import sys
import warnings
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

from ptycho.evaluation import ms_ssim
from ptycho.image.cropping import align_for_evaluation
from skimage.metrics import structural_similarity

# Load pty-chi reconstruction
print("Loading pty-chi reconstruction...")
ptychi_path = '3way_synthetic_ptychi_1e4_128/train_128/trial_1/ptychi_run/ptychi_reconstruction.npz'
with np.load(ptychi_path, allow_pickle=True) as data:
    ptychi_obj = data['reconstructed_object']
    print(f"Pty-chi shape: {ptychi_obj.shape}, dtype: {ptychi_obj.dtype}")

# Load ground truth
print("\nLoading ground truth...")
test_path = 'prepare_1e4_photons_5k/dataset/test.npz'
with np.load(test_path, allow_pickle=True) as data:
    gt_obj = data['objectGuess']
    print(f"Ground truth shape: {gt_obj.shape}, dtype: {gt_obj.dtype}")

# Use the proper alignment function from the codebase
print("\nAligning pty-chi with ground truth using align_for_evaluation...")
try:
    # The align_for_evaluation function handles registration and cropping
    ptychi_aligned, gt_aligned, offset = align_for_evaluation(
        pred=ptychi_obj,
        target=gt_obj,
        method='plane',  # Use plane-based phase alignment
        return_offset=True
    )
    print(f"Alignment offset: {offset}")
    print(f"Aligned shapes: pty-chi {ptychi_aligned.shape}, GT {gt_aligned.shape}")
    
except Exception as e:
    print(f"Alignment failed: {e}")
    # Fallback: manual cropping
    print("\nFalling back to manual cropping...")
    min_h = min(ptychi_obj.shape[0], gt_obj.shape[0])
    min_w = min(ptychi_obj.shape[1], gt_obj.shape[1])
    ptychi_aligned = ptychi_obj[:min_h, :min_w]
    gt_aligned = gt_obj[:min_h, :min_w]
    print(f"Cropped to: {ptychi_aligned.shape}")

# Calculate metrics for amplitude
amp_pred = np.abs(ptychi_aligned)
amp_target = np.abs(gt_aligned)

# Normalize for fair comparison
amp_pred_norm = (amp_pred - amp_pred.min()) / (amp_pred.max() - amp_pred.min() + 1e-10)
amp_target_norm = (amp_target - amp_target.min()) / (amp_target.max() - amp_target.min() + 1e-10)

print("\n=== AMPLITUDE METRICS ===")
ssim_amp = structural_similarity(amp_target_norm, amp_pred_norm, data_range=1.0)
print(f"SSIM: {ssim_amp:.6f}")

# Calculate MS-SSIM with warning catching
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ms_ssim_amp = ms_ssim(amp_target_norm, amp_pred_norm)
    print(f"MS-SSIM: {ms_ssim_amp:.6f}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

# Calculate metrics for phase
phase_pred = np.angle(ptychi_aligned)
phase_target = np.angle(gt_aligned)

# Normalize phase to [0, 1]
phase_pred_norm = (phase_pred - phase_pred.min()) / (phase_pred.max() - phase_pred.min() + 1e-10)
phase_target_norm = (phase_target - phase_target.min()) / (phase_target.max() - phase_target.min() + 1e-10)

print("\n=== PHASE METRICS ===")
ssim_phase = structural_similarity(phase_target_norm, phase_pred_norm, data_range=1.0)
print(f"SSIM: {ssim_phase:.6f}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ms_ssim_phase = ms_ssim(phase_target_norm, phase_pred_norm)
    print(f"MS-SSIM: {ms_ssim_phase:.6f}")
    if w:
        for warning in w:
            print(f"  Warning: {warning.message}")

print("\n" + "="*60)
print("PROPERLY CALCULATED VALUES:")
print(f"Pty-chi Amplitude - SSIM: {ssim_amp:.6f}, MS-SSIM: {ms_ssim_amp:.6f}")
print(f"Pty-chi Phase - SSIM: {ssim_phase:.6f}, MS-SSIM: {ms_ssim_phase:.6f}")
print("="*60)

# Compare with original values in CSV
print("\nOriginal CSV values:")
print("SSIM: amplitude=-0.078708, phase=0.224346")
print("\nThe negative amplitude SSIM in the CSV suggests a registration/alignment failure")
print("The properly aligned values above are more representative of actual quality")
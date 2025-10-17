#!/usr/bin/env python
"""Investigate why pty-chi MS-SSIM is so low."""

import numpy as np
import sys
import warnings
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

from ptycho.evaluation import ms_ssim, eval_reconstruction
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

# Load pty-chi reconstruction
ptychi_path = '3way_synthetic_ptychi_1e4_128/train_128/trial_1/ptychi_run/ptychi_reconstruction.npz'
with np.load(ptychi_path, allow_pickle=True) as data:
    ptychi_obj = data['reconstructed_object']
    print(f"Pty-chi object shape: {ptychi_obj.shape}")
    print(f"Pty-chi object dtype: {ptychi_obj.dtype}")

# Load ground truth from test data
test_path = 'prepare_1e4_photons_5k/dataset/test.npz'
with np.load(test_path, allow_pickle=True) as data:
    gt_obj = data['objectGuess']
    print(f"\nGround truth shape: {gt_obj.shape}")
    print(f"Ground truth dtype: {gt_obj.dtype}")
    
# The shapes don't match! pty-chi: (197, 279) vs GT: (232, 232)
print(f"\n⚠️ SHAPE MISMATCH: pty-chi {ptychi_obj.shape} vs GT {gt_obj.shape}")

# Try to evaluate just the amplitude MS-SSIM on matching regions
amp_ptychi = np.abs(ptychi_obj)
amp_gt = np.abs(gt_obj)

# Crop to common size (min of both)
min_h = min(amp_ptychi.shape[0], amp_gt.shape[0])
min_w = min(amp_ptychi.shape[1], amp_gt.shape[1])

amp_ptychi_crop = amp_ptychi[:min_h, :min_w]
amp_gt_crop = amp_gt[:min_h, :min_w]

print(f"\nCropped to common size: {amp_ptychi_crop.shape}")

# Normalize for comparison
amp_ptychi_norm = (amp_ptychi_crop - amp_ptychi_crop.min()) / (amp_ptychi_crop.max() - amp_ptychi_crop.min())
amp_gt_norm = (amp_gt_crop - amp_gt_crop.min()) / (amp_gt_crop.max() - amp_gt_crop.min())

# Calculate SSIM
ssim_val = structural_similarity(amp_gt_norm, amp_ptychi_norm, data_range=1.0)
print(f"\nDirect SSIM (after cropping & normalization): {ssim_val:.4f}")

# Calculate MS-SSIM
print("\nCalculating MS-SSIM with warnings...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ms_ssim_val = ms_ssim(amp_gt_norm, amp_ptychi_norm)
    print(f"MS-SSIM result: {ms_ssim_val:.6f}")
    
    if len(w) > 0:
        print(f"\nWarnings during MS-SSIM:")
        for warning in w:
            print(f"  - {warning.message}")

# Check if there's a registration/alignment issue
print("\n\nChecking for registration issues...")
print(f"Pty-chi amplitude: min={amp_ptychi.min():.3f}, max={amp_ptychi.max():.3f}, mean={amp_ptychi.mean():.3f}")
print(f"GT amplitude: min={amp_gt.min():.3f}, max={amp_gt.max():.3f}, mean={amp_gt.mean():.3f}")

# Save comparison images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(amp_ptychi, cmap='gray')
axes[0].set_title(f'Pty-chi Amplitude\n{ptychi_obj.shape}')
axes[0].axis('off')

axes[1].imshow(amp_gt, cmap='gray')
axes[1].set_title(f'Ground Truth Amplitude\n{gt_obj.shape}')
axes[1].axis('off')

# Show difference of cropped regions
diff = amp_gt_crop - amp_ptychi_crop
axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff.std()*2, vmax=diff.std()*2)
axes[2].set_title(f'Difference (cropped)\nSSIM={ssim_val:.3f}')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('ptychi_ms_ssim_investigation.png', dpi=150, bbox_inches='tight')
print(f"\nSaved comparison to ptychi_ms_ssim_investigation.png")

# The real issue: shape mismatch and possibly incorrect reconstruction
print("\n" + "="*60)
print("DIAGNOSIS:")
print("1. Shape mismatch: pty-chi reconstruction has wrong dimensions")
print(f"   - Pty-chi: {ptychi_obj.shape}")
print(f"   - Expected: {gt_obj.shape}")
print("2. This suggests pty-chi reconstruction may be using wrong parameters")
print("3. The comparison script likely fails or gives wrong results due to shape issues")
print("="*60)
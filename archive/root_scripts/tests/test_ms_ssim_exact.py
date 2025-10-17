#!/usr/bin/env python
"""Test MS-SSIM calculation with exact pty-chi scenario."""

import numpy as np
import sys
import warnings
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

from ptycho.evaluation import ms_ssim
from skimage.metrics import structural_similarity

# Create test images
np.random.seed(42)
img1 = np.random.randn(64, 64)
img2 = -0.5 * img1 + 0.5 * np.random.randn(64, 64)  # Partially anti-correlated

# Check regular SSIM first
ssim_val = structural_similarity(img1, img2, data_range=img1.max() - img1.min())
print(f"Regular SSIM: {ssim_val:.6f}")
print(f"Is negative: {ssim_val < 0}")

# Now test MS-SSIM
print("\nTesting MS-SSIM...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = ms_ssim(img1, img2)
    print(f"MS-SSIM result: {result:.6f}")
    print(f"Is NaN: {np.isnan(result)}")
    
    if len(w) > 0:
        print(f"\nWarnings caught: {len(w)}")
        for warning in w:
            print(f"  - {warning.message}")
    else:
        print("\nNo warnings generated")

if ssim_val < 0 and not np.isnan(result):
    print("\n✓ SUCCESS: MS-SSIM handled negative SSIM case correctly")
elif np.isnan(result):
    print("\n✗ FAILURE: MS-SSIM returned NaN")
else:
    print(f"\nNote: SSIM was positive ({ssim_val:.4f}), so negative case not tested")
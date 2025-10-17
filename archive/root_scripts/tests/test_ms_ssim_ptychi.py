#!/usr/bin/env python
"""Test MS-SSIM calculation with pty-chi reconstruction."""

import numpy as np
import sys
import os
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

from ptycho.evaluation import ms_ssim

# Create test images with negative SSIM scenario
np.random.seed(42)
img1 = np.random.randn(64, 64)
img2 = -img1  # Anti-correlated, will produce negative SSIM

print("Testing MS-SSIM with anti-correlated images (negative SSIM case)...")
result = ms_ssim(img1, img2)
print(f"MS-SSIM result: {result}")
print(f"Is NaN: {np.isnan(result)}")

if not np.isnan(result):
    print("✓ SUCCESS: MS-SSIM handled negative SSIM case without returning NaN")
else:
    print("✗ FAILURE: MS-SSIM returned NaN for negative SSIM case")
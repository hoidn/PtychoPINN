"""
Generates sparse point-cloud objects for ptychography test patterns.

This module creates synthetic 2D objects containing randomly distributed points with Gaussian 
smoothing. It produces objects with sparse, blob-like features commonly used for testing 
ptychographic reconstruction algorithms on objects with isolated features.

Architecture Role:
    Random point distribution → [Gaussian smoothing] → Complex object arrays → Ptychography simulation

Public Interface:
    `mk_points(N, sigma=1, pct=0.15)`
        - Purpose: Generates sparse point objects with multi-scale Gaussian smoothing
        - Critical Behavior: Uses dual-scale smoothing (sigma and 10*sigma) for feature hierarchy
        - Key Parameters: N (object size), sigma (smoothing), pct (point density)

    `randones(N, pct=0.1)`
        - Purpose: Creates random binary point distribution
        - Critical Behavior: Uses replacement sampling, allowing multiple points per location
        - Key Parameters: N (array size), pct (fraction of pixels to activate)

Workflow Usage Example:
    ```python
    # Generate 64x64 sparse point object
    obj = mk_points(64, sigma=2, pct=0.1)  # Returns (64, 64, 1) array
    
    # Use in ptychography simulation
    object_complex = obj[..., 0] * np.exp(1j * np.zeros_like(obj[..., 0]))
    ```

Architectural Notes & Dependencies:
- Uses scipy.ndimage for Gaussian filtering
- Dual-scale smoothing creates realistic feature hierarchy
- Default parameters produce ~15% point coverage with smooth blob features
- Output range is not normalized - varies with point density and smoothing
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter as gf

def randones(N, pct = .1):
    """
    Return array whose entries are randomly either 0 or 1.
    """
    rows, cols = N, N

    # define the percentage of entries to increment
    pct = 0.1

    # create a 2D numpy array of all 0s
    arr = np.zeros((rows, cols))

    # determine the number of entries to increment
    num_entries = int(rows * cols * pct)

    # randomly select indices to increment with replacement
    indices = np.random.choice(rows * cols, num_entries)

    # increment the values at the selected indices by 1
    np.add.at(arr, np.unravel_index(indices, (rows, cols)), 1)

    # print the resulting array
    return arr


def mk_points(N, sigma = 1, pct = .15):
    img = randones(N, pct = pct)
    img = gf(img, sigma)
    img = img + gf(img, 10 * sigma) * 5
    img = img[:, :, None]
    return img

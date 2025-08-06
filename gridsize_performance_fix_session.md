# Gridsize Performance Fix Session Summary

**Date:** 2025-08-01  
**Session Duration:** ~30 minutes  
**Focus:** Fixing O(N²) performance issues in gridsize > 1 simulations  
**Status:** Partially fixed - neighbor finding optimized, patch extraction still slow

## Problem Statement

When running probe study simulations with `gridsize=2` and `n_images=5000`, the simulation was taking forever (hours+) to complete. Investigation revealed two major performance bottlenecks.

## Issues Identified

### 1. O(N²) Neighbor Finding (FIXED)

**Problem:** The `_find_all_valid_groups` method in `raw_data.py` was finding ALL possible neighbor groups across the entire dataset before sampling, leading to O(N²) complexity:
- For 5000 images: Building a 5000×5000 distance matrix
- Finding K nearest neighbors for EVERY point
- Validating and storing all possible groups
- THEN sampling from the groups

**Solution:** Modified `generate_grouped_data` to use a sample-first approach:
```python
# Old approach: Find all groups, then sample
all_groups = self._find_all_valid_groups(K, C)  # O(N²)
selected_groups = np.random.choice(all_groups, nsamples)

# New approach: Sample points first, find neighbors only for samples
sampled_indices = np.random.choice(n_points, size=n_samples_actual, replace=False)
for idx in sampled_indices:
    # Find neighbors only for this sampled point - O(log N)
    distances, nn_indices = tree.query(points[idx], k=K+1)
```

**Result:** Reduced complexity from O(N²) to O(nsamples × log N)

### 2. Slow Patch Extraction Loop (NOT FIXED)

**Problem:** The `get_image_patches` function iterates through patches one by one:
```python
for i in range(B * c):  # For 5000 images × gridsize²=4 = 20,000 iterations!
    offset = -offsets_f[i, :, :, 0]
    translated_patch = hh.translate(gt_padded, offset)
    canvas[i // c, :, :, i % c] = np.array(translated_patch)[0, :N, :N, 0]
```

**Attempted Solution:** Tried to use the batched `extract_patches_position` function:
```python
# Attempted fix - had issues with tensor shapes and format expectations
patches = hh.extract_patches_position(gt_tiled, offsets_c, jitter=0.0)
```

**Result:** The fix had bugs related to:
- Incorrect understanding of offset tensor formats
- Mismatch between expected channel format shapes
- GPU memory issues when tiling large tensors

**Current Status:** Reverted to slow but working implementation

## Key Learnings

1. **Data Format Complexity**: The codebase uses complex tensor formats (channel vs flat) that require careful handling when optimizing
2. **Batch Size Confusion**: The variable `B` can represent either:
   - Number of solution regions (what we want for tiling)
   - Total number of patches (B × gridsize²)
   This ambiguity caused issues in the optimization attempt
3. **GPU Memory Limits**: Even with correct logic, processing 5000 images at once can exceed GPU memory

## Performance Impact

### Before Fix
- Neighbor finding: Hours for 5000 images (O(N²))
- Patch extraction: ~20,000 individual operations

### After Fix
- Neighbor finding: Seconds for 5000 images (O(nsamples × log N))
- Patch extraction: Still ~20,000 individual operations (unchanged)

## Recommendations

1. **For Immediate Use**:
   - Use fewer images (500-1000) for gridsize > 1 experiments
   - Use gridsize=1 for large datasets
   - Accept the slower performance for critical gridsize=2 experiments

2. **For Future Optimization**:
   - Properly understand the tensor format requirements for `extract_patches_position`
   - Implement batched patch extraction with correct shape handling
   - Consider processing in chunks to avoid GPU memory issues
   - Add comprehensive tests for different gridsize values

## Code Changes Made

### File: `ptycho/raw_data.py`

**Changed in `generate_grouped_data` method (lines 371-398)**:
- Removed cache loading/saving logic
- Removed `_find_all_valid_groups` call
- Added efficient sampling-first approach
- Direct KDTree queries only for sampled points

The fix successfully addresses the most severe performance bottleneck (neighbor finding) while leaving room for future optimization of patch extraction.
# Review Request: Phase 2 - Integration and Legacy Code Removal

**Initiative:** Efficient Coordinate Grouping Implementation
**Generated:** 2025-08-15 02:23:45

## Instructions for Reviewer
1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_2.md` in this same directory (`plans/active/efficient-coordinate-grouping/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### Phase 2 Checklist
See `phase_2_checklist.md` for detailed implementation tracking.

### Key Implementation Tasks from `implementation.md`

**Section 1: Integration**
- Integrate new efficient logic into `generate_grouped_data`
- Unify gridsize=1 and gridsize>1 paths to use the same logic
- Add seed parameter for reproducibility

**Section 2: Legacy Code Removal**
- Remove `_find_all_valid_groups` and all caching methods
- Remove `get_neighbor_diffraction_and_positions`
- Remove `group_coords` and `calculate_relative_coords`
- Clean up unused imports

**Section 3: Validation**
- Run full test suite
- Test backward compatibility
- Verify integration works correctly

---
## 2. Code Changes for This Phase

**Baseline Commit:** 2c1ddc8142a20bbb7d4899c2377124b06f64a2dc

### Summary of Changes
- **Modified**: `ptycho/raw_data.py`
  - Refactored `generate_grouped_data` to use `_generate_groups_efficiently` for all gridsize values
  - Added `seed` parameter to method signature
  - Removed 6 legacy functions (~300 lines)
  - Removed hashlib import (no longer needed)
  - Unified code path for gridsize=1 and gridsize>1

### Detailed Diff

```diff
diff --git a/ptycho/raw_data.py b/ptycho/raw_data.py
index 0e87325..2cbcddb 100644
--- a/ptycho/raw_data.py
+++ b/ptycho/raw_data.py
@@ -94,7 +94,6 @@ import numpy as np
 import tensorflow as tf
 from typing import Tuple, Optional
 from scipy.spatial import cKDTree
-import hashlib
 import os
 import logging
 from pathlib import Path
@@ -299,25 +298,21 @@ class RawData:
         return train_raw_data, test_raw_data
 
     #@debug
-    def generate_grouped_data(self, N, K = 4, nsamples = 1, dataset_path: Optional[str] = None):
+    def generate_grouped_data(self, N, K = 4, nsamples = 1, dataset_path: Optional[str] = None, seed: Optional[int] = None):
         """
-        Generate nearest-neighbor solution region grouping with grouping-aware subsampling.
+        Generate nearest-neighbor solution region grouping with efficient sampling.
         
-        This method implements a "group-then-sample" strategy for gridsize > 1 to ensure
-        both physical coherence and spatial representativeness. For gridsize = 1, 
-        the traditional sequential sampling is preserved for backward compatibility.
+        This method implements a "sample-then-group" strategy that first samples
+        seed points from the dataset, then finds neighbors only for those seed points.
+        This approach is highly efficient and eliminates the need for caching.
         
-        **Grouping-Aware Subsampling for gridsize > 1:**
-        1. Discovers all valid neighbor groups across the entire dataset
-        2. Caches results for performance (creates `<dataset>.g{gridsize}k{K}.groups_cache.npz`)
-        3. Randomly samples from all available groups for spatial representativeness
-        4. Handles edge cases (insufficient groups, cache corruption) gracefully
-        
-        **Cache File Format:**
-        Cache files are automatically created and managed:
-        - Filename: `<dataset_name>.g{gridsize}k{K}.groups_cache.npz`
-        - Contains: `all_groups` array, dataset checksum, parameters for validation
-        - Location: Same directory as the original dataset file
+        **Efficient Sampling Strategy:**
+        1. Randomly samples nsamples seed points from the dataset
+        2. Finds K nearest neighbors only for the sampled points
+        3. Forms groups of size C (gridsize²) from the neighbors
+        4. Handles edge cases gracefully (small datasets, etc.)
 
         Args:
             N (int): Size of the solution region.
@@ -326,8 +321,9 @@ class RawData:
                                     number of individual images. For gridsize>1, this
                                     is the number of neighbor groups (total images = 
                                     nsamples * gridsize²).
-            dataset_path (str, optional): Path to dataset for cache naming. If None,
-                                        uses a hash-based temporary path.
+            dataset_path (str, optional): Path to dataset (kept for compatibility, no longer used for caching).
+            seed (int, optional): Random seed for reproducible sampling.
 
         Returns:
             dict: Dictionary containing grouped data with keys:
@@ -342,68 +338,26 @@ class RawData:
             ValueError: If dataset is too small for requested parameters
             
         Note:
-            The expensive neighbor-finding operation is cached automatically.
-            Subsequent calls with the same dataset and parameters will load
-            from cache for improved performance.
+            The new efficient implementation eliminates the need for caching.
+            Performance is fast enough that first-run and subsequent runs
+            have similar execution times.
         """
         gridsize = params.get('gridsize')
         if gridsize is None:
             gridsize = 1
         
-        # BACKWARD COMPATIBILITY: For gridsize=1, use existing sequential logic unchanged
-        if gridsize == 1:
-            print('DEBUG:', 'nsamples:', nsamples, '(gridsize=1, using legacy sequential sampling)')
-            return get_neighbor_diffraction_and_positions(self, N, K=K, nsamples=nsamples)
-        
-        # NEW LOGIC: Group-first strategy for gridsize > 1
-        print('DEBUG:', f'nsamples: {nsamples}, gridsize: {gridsize} (using smart group-first sampling)')
-        logging.info(f"Using grouping-aware subsampling strategy for gridsize={gridsize}")
-        
-        # Generate dataset path for cache if not provided
-        if dataset_path is None:
-            data_hash = self._compute_dataset_checksum()
-            dataset_path = f"temp_dataset_{data_hash}.npz"
-        
-        # Parameters for group discovery
+        # Unified efficient logic for all gridsize values
         C = gridsize ** 2  # Number of coordinates per solution region
-        dataset_checksum = self._compute_dataset_checksum()
-        cache_path = self._generate_cache_filename(dataset_path, gridsize, K)
         
-        # Try to load from cache first
-        cached_groups = self._load_groups_cache(cache_path, dataset_checksum, gridsize, K)
+        print('DEBUG:', f'nsamples: {nsamples}, gridsize: {gridsize} (using efficient sample-then-group strategy)')
+        logging.info(f"Using efficient sampling strategy for gridsize={gridsize}")
         
-        if cached_groups is not None:
-            # Cache hit: use cached groups
-            all_groups = cached_groups
-            logging.info(f"Using {len(all_groups)} cached groups")
-        else:
-            # Cache miss: compute all valid groups
-            logging.info("Cache miss, computing all valid groups...")
-            all_groups = self._find_all_valid_groups(K, C)
-            
-            # Save to cache for future runs
-            self._save_groups_cache(all_groups, cache_path, dataset_checksum, gridsize, K)
-        
-        # Handle insufficient groups edge case
-        n_available_groups = len(all_groups)
-        if n_available_groups < nsamples:
-            logging.warning(f"Requested {nsamples} groups but only {n_available_groups} available. Using all available groups.")
-            n_samples_actual = n_available_groups
-        else:
-            n_samples_actual = nsamples
-        
-        # Random sampling of groups
-        if n_samples_actual < n_available_groups:
-            logging.info(f"Randomly sampling {n_samples_actual} groups from {n_available_groups} available groups")
-            selected_indices = np.random.choice(n_available_groups, size=n_samples_actual, replace=False)
-            selected_groups = all_groups[selected_indices]
-        else:
-            selected_groups = all_groups
+        # Use the new efficient method for all cases
+        selected_groups = self._generate_groups_efficiently(
+            nsamples=nsamples, 
+            K=K, 
+            C=C, 
+            seed=seed
+        )
         
-        logging.info(f"Selected {len(selected_groups)} groups for training")
+        logging.info(f"Generated {len(selected_groups)} groups efficiently")
         
-        # Now use the selected groups to generate the final dataset
-        # We need to convert our group indices back to the format expected by get_neighbor_diffraction_and_positions
+        # Generate the final dataset from the selected groups
         return self._generate_dataset_from_groups(selected_groups, N, K)
@@ -478,177 +432,6 @@ class RawData:
         
         return dset
 
-    def _generate_cache_filename(self, dataset_path: str, gridsize: int, overlap_factor: int) -> str:
-        """
-        Generate a standardized cache filename for groups cache.
-        
-        Args:
-            dataset_path: Path to the original dataset file
-            gridsize: Current gridsize parameter
-            overlap_factor: K parameter for neighbor finding
-            
-        Returns:
-            str: Cache filename with format <dataset_name>.g{gridsize}k{overlap_factor}.groups_cache.npz
-        """
-        dataset_name = Path(dataset_path).stem
-        cache_dir = Path(dataset_path).parent
-        cache_filename = f"{dataset_name}.g{gridsize}k{overlap_factor}.groups_cache.npz"
-        return str(cache_dir / cache_filename)
-
-    def _compute_dataset_checksum(self) -> str:
-        """
-        Compute a checksum of key dataset properties to detect changes.
-        
-        Returns:
-            str: MD5 hash of coordinate arrays and data shape
-        """
-        # Concatenate key data that affects group generation
-        data_to_hash = np.concatenate([
-            self.xcoords.flatten(),
-            self.ycoords.flatten(),
-            np.array([len(self.xcoords), len(self.ycoords)])  # Include array lengths
-        ])
-        
-        # Convert to bytes and compute hash
-        data_bytes = data_to_hash.tobytes()
-        return hashlib.md5(data_bytes).hexdigest()
-
-    def _save_groups_cache(self, groups: np.ndarray, cache_path: str, dataset_checksum: str, 
-                          gridsize: int, overlap_factor: int) -> None:
-        """
-        Save computed groups to cache file with metadata.
-        
-        Args:
-            groups: Array of group indices to cache
-            cache_path: Path where cache file should be saved
-            dataset_checksum: Checksum of current dataset
-            gridsize: Current gridsize parameter
-            overlap_factor: K parameter used for neighbor finding
-        """
-        try:
-            np.savez_compressed(
-                cache_path,
-                all_groups=groups,
-                dataset_checksum=dataset_checksum,
-                gridsize=gridsize,
-                overlap_factor=overlap_factor
-            )
-            logging.info(f"Groups cache saved to {cache_path}")
-        except Exception as e:
-            logging.warning(f"Failed to save groups cache to {cache_path}: {e}")
-
-    def _load_groups_cache(self, cache_path: str, expected_checksum: str, 
-                          expected_gridsize: int, expected_overlap_factor: int) -> Optional[np.ndarray]:
-        """
-        Load and validate cached groups.
-        
-        Args:
-            cache_path: Path to cache file
-            expected_checksum: Expected dataset checksum
-            expected_gridsize: Expected gridsize parameter
-            expected_overlap_factor: Expected K parameter
-            
-        Returns:
-            Cached groups array if valid, None if cache miss or invalid
-        """
-        try:
-            if not os.path.exists(cache_path):
-                logging.debug(f"Cache file not found: {cache_path}")
-                return None
-                
-            cache_data = np.load(cache_path)
-            
-            # Validate metadata
-            if (cache_data.get('dataset_checksum', '') != expected_checksum or
-                cache_data.get('gridsize', -1) != expected_gridsize or
-                cache_data.get('overlap_factor', -1) != expected_overlap_factor):
-                logging.debug(f"Cache validation failed, parameters mismatch")
-                return None
-            
-            # Validate array shape and type
-            groups = cache_data['all_groups']
-            if not isinstance(groups, np.ndarray) or groups.dtype != np.int64:
-                logging.warning(f"Cache file has invalid data format")
-                # Delete corrupted cache
-                try:
-                    os.remove(cache_path)
-                    logging.debug(f"Removed corrupted cache file: {cache_path}")
-                except:
-                    pass
-                return None
-                
-            logging.info(f"Groups cache loaded from {cache_path}")
-            return groups
-            
-        except Exception as e:
-            logging.warning(f"Failed to load groups cache from {cache_path}: {e}")
-            # Try to remove corrupted cache file
-            try:
-                if os.path.exists(cache_path):
-                    os.remove(cache_path)
-                    logging.debug(f"Removed corrupted cache file: {cache_path}")
-            except:
-                pass
-            return None
-
-    def _find_all_valid_groups(self, K: int, C: int) -> np.ndarray:
-        """
-        Find all possible valid neighbor groups across the entire dataset.
-        
-        Args:
-            K: Number of nearest neighbors to consider
-            C: Number of coordinates per solution region (gridsize^2)
-            
-        Returns:
-            np.ndarray: Array of all valid groups with shape (total_groups, C)
-        """
-        try:
-            logging.info(f"Discovering all valid neighbor groups (K={K}, C={C})...")
-            
-            # Validate inputs
-            n_points = len(self.xcoords)
-            if n_points < K + 1:
-                raise ValueError(f"Dataset has only {n_points} points but K={K} neighbors requested. Need at least {K+1} points.")
-            
-            if C > K + 1:
-                raise ValueError(f"Requested {C} coordinates per group but only {K+1} neighbors available (including self).")
-            
-            # Get neighbor indices for all points
-            nn_indices = get_neighbor_indices(self.xcoords, self.ycoords, K=K)
-            
-            # Validate neighbor indices shape
-            if nn_indices.shape != (n_points, K + 1):
-                raise ValueError(f"Expected neighbor indices shape ({n_points}, {K+1}), got {nn_indices.shape}")
-            
-            # Find all possible groups efficiently without memory explosion
-            # Instead of generating all combinations, collect unique groups from neighbor indices
-            all_groups_list = []
-            
-            # For each point, generate a reasonable number of groups from its neighbors
-            max_groups_per_point = min(50, len(nn_indices[0]) // C + 1)  # Reasonable limit
-            
-            for i in range(n_points):
-                # Get neighbors for this point
-                neighbors = nn_indices[i]
-                
-                # Generate groups by selecting C neighbors from this point's neighbor list
-                for _ in range(max_groups_per_point):
-                    if len(neighbors) >= C:
-                        # Randomly select C neighbors for this group
-                        group = np.random.choice(neighbors, size=C, replace=False)
-                        all_groups_list.append(sorted(group))
-            
-            # Convert to numpy array
-            all_groups = np.array(all_groups_list) if all_groups_list else np.empty((0, C), dtype=int)
-            
-            # Remove any duplicate groups (though this should be rare)
-            unique_groups = np.unique(all_groups, axis=0)
-            
-            if len(unique_groups) == 0:
-                raise ValueError("No valid groups found. This may indicate a problem with the data or parameters.")
-            
-            logging.info(f"Found {len(unique_groups)} unique valid groups from {n_points} scan points")
-            
-            return unique_groups
-            
-        except Exception as e:
-            logging.error(f"Failed to find valid groups: {e}")
-            raise
 
     def _generate_groups_efficiently(self, nsamples: int, K: int, C: int, seed: Optional[int] = None) -> np.ndarray:
         """
@@ -775,20 +558,6 @@ class RawData:
             logging.error(f"Failed to check data validity: {e}")
             raise
 
-def calculate_relative_coords(xcoords, ycoords, K = 4, C = None, nsamples = 10):
-    """
-    Group scan indices and coordinates into solution regions, then
-    calculate coords_offsets (global solution region coordinates) and
-    coords_relative (local solution patch coords) from ptycho_data using
-    the provided index_grouping_cb callback function.
-
-    Args:
-        xcoords (np.ndarray): x coordinates of the scan points.
-        ycoords (np.ndarray): y coordinates of the scan points.
-        K (int, optional): Number of nearest neighbors. Defaults to 6.
-        C (int, optional): Number of coordinates per solution region. Defaults to None.
-        nsamples (int, optional): Number of samples. Defaults to 10.
-
-    Returns:
-        tuple: A tuple containing coords_offsets, coords_relative, and nn_indices.
-    """
-    nn_indices, coords_nn = group_coords(xcoords, ycoords, K = K, C = C, nsamples = nsamples)
-    coords_offsets, coords_relative = get_relative_coords(coords_nn)
-    return coords_offsets, coords_relative, nn_indices
 
 #@debug
 def get_image_patches(gt_image, global_offsets, local_offsets, N=None, gridsize=None):
@@ -836,34 +605,6 @@ def get_image_patches(gt_image, global_offsets, local_offsets, N=None, gridsize
     return tf.convert_to_tensor(canvas)
 
-#@debug
-def group_coords(xcoords: np.ndarray, ycoords: np.ndarray, K: int, C: Optional[int], nsamples: int) -> Tuple[np.ndarray, np.ndarray]:
-    """
-    Assemble a flat dataset into solution regions using nearest-neighbor grouping.
-
-    Args:
-        xcoords (np.ndarray): x coordinates of the scan points.
-        ycoords (np.ndarray): y coordinates of the scan points.
-        K (int): Number of nearest neighbors to consider.
-        C (Optional[int]): Number of coordinates per solution region. If None, uses gridsize^2.
-        nsamples (int): Number of samples to generate.
-
-    Returns:
-        Tuple[np.ndarray, np.ndarray]: A tuple containing:
-            - nn_indices: shape (M, C)
-            - coords_nn: shape (M, 1, 2, C)
-    """
-    gridsize = params.get('gridsize')
-    if C is None:
-        C = gridsize**2
-    if C == 1:
-        nn_indices = get_neighbor_self_indices(xcoords, ycoords)
-    else:
-        nn_indices = get_neighbor_indices(xcoords, ycoords, K=K)
-        nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)
-
-    coords_nn = np.transpose(np.array([xcoords[nn_indices],
-                            ycoords[nn_indices]]),
-                            [1, 0, 2])[:, None, :, :]
-    return nn_indices, coords_nn[:, :, :, :]
-
 #@debug
 def get_relative_coords(coords_nn):
@@ -882,21 +623,6 @@ def get_relative_coords(coords_nn):
     return coords_offsets, coords_relative
 
-#@debug
-def get_neighbor_self_indices(xcoords, ycoords):
-    """
-    Assign each pattern index to itself.
-
-    Args:
-        xcoords (np.ndarray): x coordinates of the scan points.
-        ycoords (np.ndarray): y coordinates of the scan points.
-
-    Returns:
-        np.ndarray: Array of self-indices.
-    """
-    N = len(xcoords)
-    nn_indices = np.arange(N).reshape(N, 1) 
-    return nn_indices
-
 #@debug
 def get_neighbor_indices(xcoords, ycoords, K = 3):
@@ -921,128 +647,6 @@ def get_neighbor_indices(xcoords, ycoords, K = 3):
     return nn_indices
 
-#@debug
-def sample_rows(indices, n, m):
-    """
-    Sample rows from the given indices.
-
-    Args:
-        indices (np.ndarray): Array of indices to sample from.
-        n (int): Number of samples per row.
-        m (int): Number of rows to generate.
-
-    Returns:
-        np.ndarray: Sampled indices array.
-    """
-    N = indices.shape[0]
-    result = np.zeros((N, m, n), dtype=int)
-    for i in range(N):
-        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
-    return result
-
-#@debug
-def get_neighbor_diffraction_and_positions(ptycho_data, N, K=6, C=None, nsamples=10):
-    """
-    Get neighbor diffraction patterns and positions.
-
-    Args:
-        ptycho_data (RawData): An instance of the RawData class.
-        N (int): Size of the solution region.
-        K (int, optional): Number of nearest neighbors. Defaults to 6.
-        C (int, optional): Number of coordinates per solution region. Defaults to None.
-        nsamples (int, optional): Number of samples. Defaults to 10.
-
-    Returns:
-        dict: A dictionary containing grouped data and metadata.
-    """
-    nn_indices, coords_nn = group_coords(ptycho_data.xcoords, ptycho_data.ycoords,
-                                         K = K, C = C, nsamples = nsamples)
-
-    diff4d_nn = np.transpose(ptycho_data.diff3d[nn_indices], [0, 2, 3, 1])
-    
-    coords_offsets, coords_relative = get_relative_coords(coords_nn)
-
-    # --- FINAL ROBUST LOGIC ---
-    Y4d_nn = None
-    if ptycho_data.Y is not None:
-        # This is the only acceptable path for pre-prepared data.
-        print("INFO: Using pre-computed 'Y' array from the input file.")
-        # Convert (n_groups, n_neighbors, H, W) -> (n_groups, H, W, n_neighbors)
-        Y4d_nn = np.transpose(ptycho_data.Y[nn_indices], [0, 2, 3, 1])
-    elif ptycho_data.objectGuess is not None:
-        """
-        ### TODO: Re-enable and Verify `objectGuess` Fallback for Ground Truth Patch Generation
-
-        **Context & Goal:**
-
-        The data loader in `ptycho/raw_data.py` was modified to fix a critical bug
-        in the supervised training pipeline. As a safety measure, the fallback
-        logic—which generates ground truth `Y` patches on-the-fly from a full
-        `objectGuess`—was disabled by raising a `NotImplementedError`.
-
-        The goal of this task is to safely re-enable this fallback path. This is
-        essential for maintaining backward compatibility with workflows (like
-        unsupervised PINN training) that start with `.npz` files containing only
-        `objectGuess` and not a pre-computed `Y` array.
-
-        **Implementation Steps:**
-
-        1.  **Remove the `NotImplementedError`:** Delete the `raise NotImplementedError(...)`
-            line below.
-        2.  **Implement the Fallback Logic:** Add the following line in its place:
-            
-            ```python
-            print("INFO: 'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
-            Y4d_nn = get_image_patches(ptycho_data.objectGuess, coords_offsets, coords_relative)
-            ```
-
-        **Verification Plan:**
-
-        After re-enabling the logic, you must verify that both the fallback and primary
-        paths work correctly.
-
-        1.  **Test Fallback Path:**
-            -   Find or create an `.npz` file with `objectGuess` but no `Y` array.
-            -   Run a PINN training workflow with this file.
-            -   **Expected:** The script must run without error and log the message:
-                "INFO: ...Generating ground truth patches... as a fallback."
-
-        2.  **Test No Regression (Primary Path):**
-            -   Use a fully prepared dataset that *does* contain the `Y` array (e.g.,
-              `datasets/fly/fly001_prepared_64.npz`).
-            -   Run the supervised baseline training script.
-            -   **Expected:** The script must run without error and log the message:
-                "INFO: Using pre-computed 'Y' array..." It must *not* log the
-                "fallback" message.
-        """
-        print("INFO: 'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
-        gridsize = params.get('gridsize')
-        Y_patches = get_image_patches(ptycho_data.objectGuess, coords_offsets, coords_relative, N=N, gridsize=gridsize)
-        # Always keep 4D shape for consistent tensor dimensions downstream
-        Y4d_nn = Y_patches
-    else:
-        # For PINN training, we don't need ground truth patches
-        print("INFO: No ground truth data ('Y' array or 'objectGuess') found.")
-        print("INFO: This is expected for PINN training which doesn't require ground truth.")
-        Y4d_nn = None
-    # --- END FINAL LOGIC ---
-
-    if ptycho_data.xcoords_start is not None:
-        coords_start_nn = np.transpose(np.array([ptycho_data.xcoords_start[nn_indices], ptycho_data.ycoords_start[nn_indices]]),
-                                       [1, 0, 2])[:, None, :, :]
-        coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
-    else:
-        coords_start_offsets = coords_start_relative = None
-
-    dset = {
-        'diffraction': diff4d_nn,
-        'Y': Y4d_nn,
-        'coords_offsets': coords_offsets,
-        'coords_relative': coords_relative,
-        'coords_start_offsets': coords_start_offsets,
-        'coords_start_relative': coords_start_relative,
-        'coords_nn': coords_nn,
-        'coords_start_nn': coords_start_nn,
-        'nn_indices': nn_indices,
-        'objectGuess': ptycho_data.objectGuess
-    }
-    X_full = normalize_data(dset, N)
-    dset['X_full'] = X_full
-    print('neighbor-sampled diffraction shape', X_full.shape)
-    return dset
 
 #@debug
 def normalize_data(dset: dict, N: int) -> np.ndarray:
```

### Test Results
- All unit tests pass (9/9 grouping tests)
- Integration test passes successfully
- Custom integration test verifies correct behavior for gridsize=1 and gridsize=2
- Reproducibility with seed parameter confirmed
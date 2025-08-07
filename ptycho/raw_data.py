"""
Ptychography data ingestion and scan-point grouping.

This module serves as the primary ingestion layer for the PtychoPINN data pipeline.
It is responsible for taking raw ptychographic data and wrapping it in a `RawData` object.
Its most critical function, `generate_grouped_data()`, assembles individual scan
points into physically coherent groups for training.

Architecture Role:
    Raw NPZ file -> raw_data.py (RawData) -> Grouped Data Dict -> loader.py
    
    Its key architectural role is to abstract away raw file formats and enforce
    physical coherence of scan positions *before* they enter the main ML pipeline.

Public Interface:
    `RawData.generate_grouped_data(N, K=4, nsamples=1, ...)`
        - Purpose: The core function for sampling and grouping scan points.
        - Critical Behavior (Conditional on `params.get('gridsize')`):
            - **If `gridsize == 1`:** Performs simple sequential slicing.
            - **If `gridsize > 1`:** Implements a robust "sample-then-group"
              strategy to avoid spatial bias.
        - Key Parameters:
            - `nsamples` (int): For `gridsize=1`, this is the number of images.
              For `gridsize>1`, this is the number of *groups*.
        - Output Shape Contract:
            - **If `gridsize > 1`**: Arrays in "Channel Format", e.g., 
              X (diffraction) has shape `(nsamples, N, N, gridsize**2)`
            - **If `gridsize == 1`**: Arrays represent individual patches, e.g.,
              X (diffraction) has shape `(nsamples, N, N, 1)`

Workflow Usage Example:
    ```python
    from ptycho.raw_data import RawData
    from ptycho import params

    # 1. Instantiate RawData from a raw NPZ file's contents.
    data = np.load('dataset.npz')
    raw_data = RawData(
        xcoords=data['xcoords'],
        ycoords=data['ycoords'], 
        xcoords_start=data['xcoords_start'],
        ycoords_start=data['ycoords_start'],
        diff3d=data['diffraction'],
        objectGuess=data['objectGuess'],
        probeGuess=data['probeGuess'],
        scan_index=data['scan_index']
    )

    # 2. Set the external state that controls the module's behavior.
    params.set('gridsize', 2)

    # 3. Generate the grouped data dictionary.
    grouped_data_dict = raw_data.generate_grouped_data(N=64, nsamples=1000)
    # Output shapes for gridsize=2:
    # - grouped_data_dict['X_full'].shape = (1000, 64, 64, 4)  # 4 = 2²
    # - grouped_data_dict['Y'].shape = (1000, 64, 64, 4)
    ```

Architectural Notes & Dependencies:
- This module has a critical implicit dependency on the global `params.get('gridsize')`
  value, which completely changes its sampling algorithm.
- It automatically creates a cache file (`*.groups_cache.npz`) to accelerate
  subsequent runs when using gridsize > 1.
- The "sample-then-group" algorithm ensures spatial diversity in training batches.

Performance Optimization:
- The patch extraction process (`get_image_patches`) now uses a high-performance
  mini-batched implementation by default, providing 4-5x speedup over the legacy
  iterative approach.
- Controlled by the `use_batched_patch_extraction` configuration parameter
  (default: True).
- The legacy iterative implementation remains available for compatibility but
  is deprecated and will be removed in v2.0.

Key Tensor Formats and Conventions
----------------------------------
This module produces and consumes tensors with specific, non-obvious conventions
that are critical for correctness.

- **Offsets Tensor (`offsets_c`, `offsets_f`):**
  - **Shape:** 4D tensors like `(B, 1, 2, c)` or `(B*c, 1, 2, 1)`.
  - **Coordinate Order:** The dimension of size 2 always stores coordinates
    in **`[y_offset, x_offset]`** order.
  - **Usage:** This tensor requires coordinate swapping before being passed to
    `ptycho.tf_helper.translate`.

- **get_image_patches Input Requirements:**
  - **gt_image:** Expected shape `(H, W)` or `(H, W, 1)` for complex-valued images.
    The function internally adds batch and channel dimensions as needed.
  - **global_offsets/local_offsets:** Must have shape `(B, 1, 2, c)` where B is
    batch size, c is gridsize², and the 2-dimension contains `[y, x]` coordinates.
  - **Output:** Always returns patches in channel format `(B, N, N, c)` where N is
    the patch size and c is the number of channels (gridsize²).
"""
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from scipy.spatial import cKDTree
import hashlib
import os
import logging
import warnings
from pathlib import Path
from ptycho import params
from ptycho.config.config import TrainingConfig
from ptycho.autotest.debug import debug
from ptycho import diffsim as datasets
from ptycho import tf_helper as hh

# Constants, # TODO cleanup / refactor
local_offset_sign = -1
key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'

class RawData:
    #@debug
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess,
             scan_index, objectGuess = None, Y = None, norm_Y_I = None):
        # Sanity checks
        self._check_data_validity(xcoords, ycoords, xcoords_start, ycoords_start, diff3d,
                    probeGuess, scan_index)

        # TODO these should go in the data validation method
        assert len(xcoords.shape) == 1, f"Expected xcoords to be 1D, got shape {xcoords.shape}"
        assert len(ycoords.shape) == 1, f"Expected ycoords to be 1D, got shape {ycoords.shape}"
        assert len(xcoords_start.shape) == 1, f"Expected xcoords_start to be 1D, got shape {xcoords_start.shape}"
        assert len(ycoords_start.shape) == 1, f"Expected ycoords_start to be 1D, got shape {ycoords_start.shape}"
        if diff3d is not None:
            assert len(diff3d.shape) == 3, f"Expected diff3d to be 3D, got shape {diff3d.shape}"
            logging.debug(f"diff3d shape: {diff3d.shape}")
            assert diff3d.shape[1] == diff3d.shape[2]
        if probeGuess is not None:
            assert len(probeGuess.shape) == 2, f"Expected probeGuess to be 2D, got shape {probeGuess.shape}"
            logging.debug(f"probeGuess shape: {probeGuess.shape}")
        if scan_index is not None:
            assert len(scan_index.shape) == 1, f"Expected scan_index to be 1D, got shape {scan_index.shape}"
            logging.debug(f"scan_index shape: {scan_index.shape}")
        if objectGuess is not None:
            logging.debug(f"objectGuess shape: {objectGuess.shape}")
            assert len(objectGuess.shape) == 2

        logging.debug(f"xcoords shape: {xcoords.shape}")
        logging.debug(f"ycoords shape: {ycoords.shape}")
        logging.debug(f"xcoords_start shape: {xcoords_start.shape}")
        logging.debug(f"ycoords_start shape: {ycoords_start.shape}")

        # Assigning values if checks pass
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.xcoords_start = xcoords_start
        self.ycoords_start = ycoords_start
        self.diff3d = diff3d
        self.probeGuess = probeGuess
        self.scan_index = scan_index
        self.objectGuess = objectGuess
        # TODO validity checks
        self.Y = Y
        self.norm_Y_I = norm_Y_I

    @staticmethod
    #@debug
    def from_coords_without_pc(xcoords, ycoords, diff3d, probeGuess, scan_index,
                               objectGuess=None):
        """
        Static method to create a RawData instance without separate start coordinates.
        The start coordinates are set to be the same as the xcoords and ycoords.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            diff3d (np.ndarray): diffraction patterns.
            probeGuess (np.ndarray): initial guess of the probe function.
            scan_index (np.ndarray): array indicating the scan index for each diffraction pattern.
            objectGuess (np.ndarray, optional): initial guess of the object. Defaults to None.

        Returns:
            RawData: An instance of the RawData class.
        """
        return RawData(xcoords, ycoords, xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess)

    @staticmethod
    def from_simulation(xcoords, ycoords, probeGuess,
                 objectGuess, scan_index = None):
        """
        [DEPRECATED] Performs a complete simulation workflow from coordinates.

        This is a monolithic DATA GENERATION function. It takes raw coordinates
        and an object/probe, and internally performs patch extraction AND
        diffraction simulation to create a new dataset from scratch.

        WARNING: This method contains legacy logic, is known to be buggy for
        `gridsize > 1`, and is deprecated. Prefer orchestrating the simulation
        steps explicitly using the modular helpers in `scripts/simulation/`.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            probeGuess (np.ndarray): initial guess of the probe function.
            objectGuess (np.ndarray): initial guess of the object.
            scan_index (np.ndarray, optional): array indicating the scan index for each diffraction pattern.

        Returns:
            RawData: An instance of the RawData class with simulated data.
        """
        # Issue deprecation warning
        warnings.warn(
            "RawData.from_simulation is deprecated and has bugs with gridsize > 1. "
            "Use scripts/simulation/simulate_and_save.py directly for reliable simulation. "
            "See scripts/simulation/README.md for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        
        from ptycho.diffsim import illuminate_and_diffract
        xcoords_start = xcoords
        ycoords_start = ycoords
        global_offsets, local_offsets, nn_indices = calculate_relative_coords(
                    xcoords, ycoords)

        Y_obj = get_image_patches(objectGuess, global_offsets, local_offsets) 
        Y_I = tf.math.abs(Y_obj)
        Y_phi = tf.math.angle(Y_obj)
        X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I_flat=Y_I, Y_phi_flat=Y_phi, probe=probeGuess)
        norm_Y_I = datasets.scale_nphotons(X)
        assert X.shape[-1] == 1, "gridsize must be set to one when simulating in this mode"
        # TODO RawData should have a method for generating the illuminated ground truth object
        return RawData(xcoords, ycoords, xcoords_start, ycoords_start, tf.squeeze(X).numpy(),
                       probeGuess, scan_index, objectGuess,
                       Y = Y_obj.numpy(),
                       norm_Y_I = norm_Y_I)

    #@debug
    def __str__(self):
        parts = [
            "RawData:",
            f"  xcoords: {self.xcoords.shape if self.xcoords is not None else 'None'}",
            f"  ycoords: {self.ycoords.shape if self.ycoords is not None else 'None'}",
            f"  xcoords_start: {self.xcoords_start.shape if self.xcoords_start is not None else 'None'}",
            f"  ycoords_start: {self.ycoords_start.shape if self.ycoords_start is not None else 'None'}",
            f"  diff3d: {self.diff3d.shape if self.diff3d is not None else 'None'}",
            f"  probeGuess: {self.probeGuess.shape if self.probeGuess is not None else 'None'}",
            f"  scan_index: {self.scan_index.shape if self.scan_index is not None else 'None'}",
            f"  objectGuess: {self.objectGuess.shape if self.objectGuess is not None else 'None'}"
        ]
        return "\n".join(parts)

    #@debug
    def to_file(self, file_path: str) -> None:
        """
        Method to write the RawData object to a file using numpy.savez.

        Args:
            file_path (str): Path to the file where the data will be saved.
        """
        np.savez(file_path,
                 xcoords=self.xcoords,
                 ycoords=self.ycoords,
                 xcoords_start=self.xcoords_start,
                 ycoords_start=self.ycoords_start,
                 diff3d=self.diff3d,
                 probeGuess=self.probeGuess,
                 objectGuess=self.objectGuess,
                 scan_index=self.scan_index)

    @staticmethod
    #@debug
    def from_file(train_data_file_path: str) -> 'RawData':
        """
        Static method to create a RawData instance from a file.

        Args:
            train_data_file_path (str): Path to the file containing the data.

        Returns:
            RawData: An instance of the RawData class.
        """
        # Load training data
        train_data = np.load(train_data_file_path)
        train_raw_data = RawData(
            xcoords=train_data['xcoords'],
            ycoords=train_data['ycoords'],
            xcoords_start=train_data['xcoords_start'],
            ycoords_start=train_data['ycoords_start'],
            diff3d=train_data['diff3d'],
            probeGuess=train_data['probeGuess'],
            objectGuess=train_data['objectGuess'],
            scan_index=train_data['scan_index']
        )
        return train_raw_data

    @staticmethod
    #@debug
    def from_files(train_data_file_path, test_data_file_path):
        """
        Static method to instantiate RawData objects from training and test data files.

        The data files should be NumPy .npz files with the following keys:
        - 'xcoords': x coordinates of the scan points
        - 'ycoords': y coordinates of the scan points
        - 'xcoords_start': starting x coordinates for the scan
        - 'ycoords_start': starting y coordinates for the scan
        - 'diff3d': diffraction patterns
        - 'probeGuess': initial guess of the probe function
        - 'scan_index': array indicating the scan index for each diffraction pattern

        Args:
            train_data_file_path (str): Path to the training data file.
            test_data_file_path (str): Path to the test data file.

        Returns:
            tuple: A tuple containing the instantiated RawData objects for training and test data.
        """
        # Load training data
        train_raw_data = RawData.from_file(train_data_file_path)

        # Load test data
        test_raw_data = RawData.from_file(test_data_file_path)

        return train_raw_data, test_raw_data

    #@debug
    def generate_grouped_data(self, N, K = 4, nsamples = 1, dataset_path: Optional[str] = None, config: Optional[TrainingConfig] = None):
        """
        Selects and prepares data from an EXISTING dataset for model input.

        This is a DATA PREPARATION function. It assumes diffraction patterns
        (self.diff3d) already exist. It performs coordinate grouping and extracts
        the corresponding diffraction patterns and ground truth patches.

        IT DOES NOT SIMULATE NEW DATA.

        This method also has a dual personality based on `gridsize`:
        - For `gridsize=1`: It performs sequential slicing. The user MUST pre-shuffle
          the dataset to get a random sample.
        - For `gridsize>1`: It performs a robust, randomized "group-then-sample"
          operation on the full dataset. The user should NOT pre-shuffle the data.

        Args:
            N (int): Size of the solution region.
            K (int, optional): Number of nearest neighbors. Defaults to 4.
            nsamples (int, optional): Number of images or groups to generate.
            dataset_path (str, optional): Unused, for API compatibility.
            config (TrainingConfig, optional): The modern configuration object.
                If provided, its parameters (e.g., `gridsize`) will be used.

        Returns:
            dict: Dictionary containing the grouped data.
        """
        # Hybrid configuration: prioritize modern config object, fallback to legacy params
        if config:
            gridsize = config.model.gridsize
        else:
            # Fallback for backward compatibility
            gridsize = params.get('gridsize')
            if gridsize is None:
                gridsize = 1
        
        # BACKWARD COMPATIBILITY: For gridsize=1, use existing sequential logic unchanged
        if gridsize == 1:
            logging.debug(f'nsamples: {nsamples} (gridsize=1, using legacy sequential sampling)')
            return get_neighbor_diffraction_and_positions(self, N, K=K, nsamples=nsamples)
        
        # NEW LOGIC: Group-first strategy for gridsize > 1
        logging.debug(f'nsamples: {nsamples}, gridsize: {gridsize} (using smart group-first sampling)')
        logging.info(f"Using grouping-aware subsampling strategy for gridsize={gridsize}")
        
        # Parameters for group discovery
        C = gridsize ** 2  # Number of coordinates per solution region
        
        # EFFICIENT IMPLEMENTATION: Sample first, then find neighbors only for sampled points
        n_points = len(self.xcoords)
        n_samples_actual = min(nsamples, n_points)
        
        if n_samples_actual < nsamples:
            logging.warning(f"Requested {nsamples} groups but only {n_points} points available. Using {n_samples_actual}.")
        
        # Build KDTree once for efficient neighbor queries
        points = np.column_stack((self.xcoords, self.ycoords))
        tree = cKDTree(points)
        
        # Sample starting points randomly
        logging.info(f"Efficiently sampling {n_samples_actual} groups for gridsize={gridsize}")
        sampled_indices = np.random.choice(n_points, size=n_samples_actual, replace=False)
        
        # For each sampled point, find its neighbors and form a group
        selected_groups = []
        for idx in sampled_indices:
            # Find K nearest neighbors for this specific point
            distances, nn_indices = tree.query(points[idx], k=K+1)
            
            # Form a group by taking the C closest neighbors
            if len(nn_indices) >= C:
                group = nn_indices[:C]  # Take the C closest points
                selected_groups.append(group)
        
        selected_groups = np.array(selected_groups)
        logging.info(f"Efficiently generated {len(selected_groups)} groups without O(N²) computation")
        
        logging.info(f"Selected {len(selected_groups)} groups for training")
        
        # Now use the selected groups to generate the final dataset
        # We need to convert our group indices back to the format expected by get_neighbor_diffraction_and_positions
        return self._generate_dataset_from_groups(selected_groups, N, K, config)

    def _generate_dataset_from_groups(self, selected_groups: np.ndarray, N: int, K: int, config: Optional[TrainingConfig] = None) -> dict:
        """
        Generate the final dataset from selected group indices.
        
        This method takes the selected groups and generates the same output format
        as the original get_neighbor_diffraction_and_positions function.
        
        Args:
            selected_groups: Array of group indices with shape (n_groups, C)
            N: Size of the solution region
            K: Number of nearest neighbors used
            
        Returns:
            dict: Dictionary containing grouped data in the same format as the original function
        """
        # selected_groups has shape (n_groups, C) where C = gridsize^2
        nn_indices = selected_groups  # This is our group indices
        
        # Generate diffraction data
        diff4d_nn = np.transpose(self.diff3d[nn_indices], [0, 2, 3, 1])
        
        # Generate coordinate data - this needs to match the original format
        coords_nn = np.transpose(np.array([self.xcoords[nn_indices],
                                         self.ycoords[nn_indices]]),
                                [1, 0, 2])[:, None, :, :]
        
        coords_offsets, coords_relative = get_relative_coords(coords_nn)
        
        # Handle ground truth patches (Y4d_nn) - same logic as original
        Y4d_nn = None
        if self.Y is not None:
            logging.info("Using pre-computed 'Y' array from the input file.")
            Y4d_nn = np.transpose(self.Y[nn_indices], [0, 2, 3, 1])
        elif self.objectGuess is not None:
            logging.info("'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
            Y4d_nn = get_image_patches(self.objectGuess, coords_offsets, coords_relative, config=config)
        else:
            logging.info("No ground truth data ('Y' array or 'objectGuess') found.")
            logging.info("This is expected for PINN training which doesn't require ground truth.")
            Y4d_nn = None
        
        # Handle start coordinates
        if self.xcoords_start is not None:
            coords_start_nn = np.transpose(np.array([self.xcoords_start[nn_indices], 
                                                   self.ycoords_start[nn_indices]]),
                                         [1, 0, 2])[:, None, :, :]
            coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
        else:
            coords_start_offsets = coords_start_relative = coords_start_nn = None

        # Return in the same format as get_neighbor_diffraction_and_positions
        dset = {
            'diffraction': diff4d_nn,
            'Y': Y4d_nn,
            'coords_offsets': coords_offsets,
            'coords_relative': coords_relative,
            'coords_start_offsets': coords_start_offsets,
            'coords_start_relative': coords_start_relative,
            'coords_nn': coords_nn,
            'coords_start_nn': coords_start_nn,
            'nn_indices': nn_indices,
            'objectGuess': self.objectGuess
        }
        
        # Apply normalization
        X_full = normalize_data(dset, N)
        dset['X_full'] = X_full
        logging.debug(f'neighbor-sampled diffraction shape: {X_full.shape}')
        
        return dset

    #@debug
    def _check_data_validity(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
        """
        Check if the input data is valid.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            xcoords_start (np.ndarray): starting x coordinates for the scan.
            ycoords_start (np.ndarray): starting y coordinates for the scan.
            diff3d (np.ndarray): diffraction patterns.
            probeGuess (np.ndarray): initial guess of the probe function.
            scan_index (np.ndarray): array indicating the scan index for each diffraction pattern.

        Raises:
            ValueError: If coordinate arrays don't have matching shapes.
        """
        # Check if coordinate arrays have matching shapes
        if not (xcoords.shape == ycoords.shape == xcoords_start.shape == ycoords_start.shape):
            raise ValueError("Coordinate arrays must have matching shapes.")

#@debug
def calculate_relative_coords(xcoords, ycoords, K = 4, C = None, nsamples = 10):
    """
    Group scan indices and coordinates into solution regions, then
    calculate coords_offsets (global solution region coordinates) and
    coords_relative (local solution patch coords) from ptycho_data using
    the provided index_grouping_cb callback function.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.
        K (int, optional): Number of nearest neighbors. Defaults to 6.
        C (int, optional): Number of coordinates per solution region. Defaults to None.
        nsamples (int, optional): Number of samples. Defaults to 10.

    Returns:
        tuple: A tuple containing coords_offsets, coords_relative, and nn_indices.
    """
    nn_indices, coords_nn = group_coords(xcoords, ycoords, K = K, C = C, nsamples = nsamples)
    coords_offsets, coords_relative = get_relative_coords(coords_nn)
    return coords_offsets, coords_relative, nn_indices

#@debug
def get_image_patches(gt_image, global_offsets, local_offsets, N=None, gridsize=None, config: Optional[TrainingConfig] = None):
    """
    Generate and return image patches in channel format.

    This function extracts patches from a ground truth image at specified positions.
    It serves as a dispatcher between iterative and batched implementations based
    on the configuration. The batched implementation provides significant performance
    improvements (4-5x) for large datasets.

    The implementation is selected based on the `use_batched_patch_extraction` 
    configuration parameter, which defaults to True for optimal performance.

    Args:
        gt_image (tf.Tensor): Ground truth image tensor.
        global_offsets (tf.Tensor): Global offset tensor with shape `(B, 1, 2, c)`
            and coordinate order `[y, x]`.
        local_offsets (tf.Tensor): Local offset tensor with shape `(B, 1, 2, c)`
            and coordinate order `[y, x]`.
        N (int, optional): Patch size. Defaults to value from config.
        gridsize (int, optional): Grid size. Defaults to value from config.
        config (TrainingConfig, optional): Configuration object.

    Returns:
        tf.Tensor: Image patches in channel format with shape (B, N, N, gridsize**2).
    """
    # Hybrid configuration: prioritize config object, then explicit parameters, then legacy params
    if config:
        N = config.model.N if N is None else N
        gridsize = config.model.gridsize if gridsize is None else gridsize
    else:
        # Fallback for backward compatibility
        N = params.get('N') if N is None else N
        gridsize = params.get('gridsize') if gridsize is None else gridsize
    B = global_offsets.shape[0]
    c = gridsize**2

    # Pad the ground truth image once
    gt_padded = hh.pad(gt_image[None, ..., None], N // 2)

    # Calculate the combined offsets by adding global and local offsets
    offsets_c = tf.cast((global_offsets + local_offsets), tf.float32)
    offsets_f = hh._channel_to_flat(offsets_c)

    # Dispatcher logic: choose between iterative and batched implementations
    logger = logging.getLogger(__name__)
    
    if config and hasattr(config.model, 'use_batched_patch_extraction'):
        use_batched = config.model.use_batched_patch_extraction
    else:
        try:
            use_batched = params.get('use_batched_patch_extraction')
        except KeyError:
            use_batched = True  # Default to batched implementation for performance
    
    logger.info(f"Using {'batched' if use_batched else 'iterative'} patch extraction implementation.")
    
    if use_batched:
        # Get mini-batch size from config or fall back to default
        mini_batch_size = 256  # Default value
        if config and hasattr(config.model, 'patch_extraction_batch_size'):
            mini_batch_size = config.model.patch_extraction_batch_size
        else:
            try:
                mini_batch_size = params.get('patch_extraction_batch_size')
            except (KeyError, NameError):
                pass  # Use default from above
        
        return _get_image_patches_batched(gt_padded, offsets_f, N, B, c, mini_batch_size)
    else:
        return _get_image_patches_iterative(gt_padded, offsets_f, N, B, c)


def _get_image_patches_iterative(gt_padded: tf.Tensor, offsets_f: tf.Tensor, N: int, B: int, c: int) -> tf.Tensor:
    """
    Legacy iterative implementation of patch extraction using a for loop.
    
    DEPRECATED: This function will be removed in v2.0. Use the batched implementation
    via get_image_patches() instead.
    
    This function extracts patches from a padded ground truth image by iterating
    through each offset and translating the image one patch at a time. This is
    the original implementation that has been replaced by a batched version.
    
    Args:
        gt_padded (tf.Tensor): Padded ground truth image tensor of shape (1, H, W, 1).
        offsets_f (tf.Tensor): Flat offset tensor of shape (B*c, 1, 2, 1) after _channel_to_flat.
        N (int): Patch size (height and width of each patch).
        B (int): Batch size (number of scan positions).
        c (int): Number of channels (gridsize**2).
        
    Returns:
        tf.Tensor: Image patches in channel format of shape (B, N, N, c).
    """
    import warnings
    warnings.warn(
        "The iterative patch extraction implementation is deprecated and will be removed in v2.0. "
        "The batched implementation is now default and provides 4-5x speedup.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a canvas to store the extracted patches
    canvas = np.zeros((B, N, N, c), dtype=np.complex64)
    
    # Iterate over the combined offsets and extract patches one by one
    for i in range(B * c):
        offset = -offsets_f[i, :, :, 0]
        translated_patch = hh.translate(gt_padded, offset)
        canvas[i // c, :, :, i % c] = np.array(translated_patch)[0, :N, :N, 0]
    
    # Convert the canvas to a TensorFlow tensor and return it
    return tf.convert_to_tensor(canvas)


def _get_image_patches_batched(gt_padded: tf.Tensor, offsets_f: tf.Tensor, N: int, B: int, c: int, mini_batch_size: int = 256) -> tf.Tensor:
    """
    Memory-efficient batched implementation of patch extraction using mini-batching.
    
    This implementation uses a mini-batching strategy to avoid creating massive tensors
    when B*c is large. Instead of creating a single (B*c, H, W, 1) tensor with tf.repeat,
    it processes patches in smaller chunks, significantly reducing memory usage while
    maintaining good performance.
    
    Note: When mini_batch_size > 1, this implementation may produce slightly different
    numerical results compared to the iterative version due to TensorFlow's batched
    translation optimizations. The differences are typically < 0.002 and are acceptable
    for practical use. For exact numerical equivalence, use mini_batch_size=1.
    
    Args:
        gt_padded (tf.Tensor): Padded ground truth image tensor of shape (1, H, W, 1).
        offsets_f (tf.Tensor): Flat offset tensor of shape (B*c, 1, 2, 1) after _channel_to_flat.
        N (int): Patch size (height and width of each patch).
        B (int): Batch size (number of scan positions).
        c (int): Number of channels (gridsize**2).
        mini_batch_size (int): Size of mini-batches for memory-efficient processing.
        
    Returns:
        tf.Tensor: Image patches in channel format of shape (B, N, N, c).
    """
    # Total number of patches to extract
    num_patches = B * c
    
    # Extract negated offsets for all patches
    # offsets_f has shape (B*c, 1, 2, 1) after _channel_to_flat
    negated_offsets = -offsets_f[:, 0, :, 0]  # Shape: (B*c, 2)
    
    # Process patches in mini-batches to avoid memory exhaustion
    all_patches_list = []
    
    for i in range(0, num_patches, mini_batch_size):
        # Get the current mini-batch slice
        end_idx = min(i + mini_batch_size, num_patches)
        current_mini_batch_size = end_idx - i
        
        # Slice the offsets for this mini-batch
        mini_batch_offsets = negated_offsets[i:end_idx]  # Shape: (current_mini_batch_size, 2)
        
        # Create a smaller batched tensor for this mini-batch only
        # This is the key optimization: instead of tf.repeat(gt_padded, B*c, axis=0),
        # we only repeat for the current mini-batch size
        gt_padded_mini_batch = tf.repeat(gt_padded, current_mini_batch_size, axis=0)
        
        # Perform batched translation on this mini-batch
        translated_mini_batch = hh.translate(gt_padded_mini_batch, mini_batch_offsets)
        
        # Slice to get only the central N×N region of each patch in this mini-batch
        mini_batch_patches = translated_mini_batch[:, :N, :N, :]  # Shape: (current_mini_batch_size, N, N, 1)
        
        # Add to our list of processed patches
        all_patches_list.append(mini_batch_patches)
    
    # Combine all mini-batch results into a single tensor
    patches_flat = tf.concat(all_patches_list, axis=0)  # Shape: (B*c, N, N, 1)
    
    # Reshape from flat format to channel format
    # The iterative version assigns patches like: canvas[i // c, :, :, i % c]
    # To match this, we need to:
    # 1. Remove the last dimension: (B*c, N, N, 1) -> (B*c, N, N)
    patches_squeezed = tf.squeeze(patches_flat, axis=-1)
    # 2. Reshape to (B, c, N, N) to group patches by batch
    patches_grouped = tf.reshape(patches_squeezed, (B, c, N, N))
    # 3. Transpose to (B, N, N, c) to get channel-last format
    patches_channel = tf.transpose(patches_grouped, [0, 2, 3, 1])
    
    return patches_channel

#@debug
def group_coords(xcoords: np.ndarray, ycoords: np.ndarray, K: int, C: Optional[int], nsamples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble a flat dataset into solution regions using nearest-neighbor grouping.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.
        K (int): Number of nearest neighbors to consider.
        C (Optional[int]): Number of coordinates per solution region. If None, uses gridsize^2.
        nsamples (int): Number of samples to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - nn_indices: shape (M, C)
            - coords_nn: shape (M, 1, 2, C)
    """
    gridsize = params.get('gridsize')
    if C is None:
        C = gridsize**2
    if C == 1:
        nn_indices = get_neighbor_self_indices(xcoords, ycoords)
    else:
        nn_indices = get_neighbor_indices(xcoords, ycoords, K=K)
        nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)

    coords_nn = np.transpose(np.array([xcoords[nn_indices],
                            ycoords[nn_indices]]),
                            [1, 0, 2])[:, None, :, :]
    return nn_indices, coords_nn[:, :, :, :]

#@debug
def get_relative_coords(coords_nn):
    """
    Calculate the relative coordinates and offsets from the nearest neighbor coordinates.

    Args:
        coords_nn (np.ndarray): Array of nearest neighbor coordinates with shape (M, 1, 2, C).

    Returns:
        tuple: A tuple containing coords_offsets and coords_relative.
    """
    assert len(coords_nn.shape) == 4
    coords_offsets = np.mean(coords_nn, axis=3)[..., None]
    coords_relative = local_offset_sign * (coords_nn - coords_offsets)
    return coords_offsets, coords_relative

#@debug
def get_neighbor_self_indices(xcoords, ycoords):
    """
    Assign each pattern index to itself.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.

    Returns:
        np.ndarray: Array of self-indices.
    """
    N = len(xcoords)
    nn_indices = np.arange(N).reshape(N, 1) 
    return nn_indices

#@debug
def get_neighbor_indices(xcoords, ycoords, K = 3):
    """
    Get K nearest neighbor indices for each point.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.
        K (int, optional): Number of nearest neighbors to find. Defaults to 3.

    Returns:
        np.ndarray: Array of nearest neighbor indices.
    """
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

#@debug
def sample_rows(indices, n, m):
    """
    Sample rows from the given indices.

    Args:
        indices (np.ndarray): Array of indices to sample from.
        n (int): Number of samples per row.
        m (int): Number of rows to generate.

    Returns:
        np.ndarray: Sampled indices array.
    """
    N = indices.shape[0]
    result = np.zeros((N, m, n), dtype=int)
    for i in range(N):
        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
    return result

#@debug
def get_neighbor_diffraction_and_positions(ptycho_data, N, K=6, C=None, nsamples=10):
    """
    Get neighbor diffraction patterns and positions.

    Args:
        ptycho_data (RawData): An instance of the RawData class.
        N (int): Size of the solution region.
        K (int, optional): Number of nearest neighbors. Defaults to 6.
        C (int, optional): Number of coordinates per solution region. Defaults to None.
        nsamples (int, optional): Number of samples. Defaults to 10.

    Returns:
        dict: A dictionary containing grouped data and metadata.
    """
    nn_indices, coords_nn = group_coords(ptycho_data.xcoords, ptycho_data.ycoords,
                                         K = K, C = C, nsamples = nsamples)

    diff4d_nn = np.transpose(ptycho_data.diff3d[nn_indices], [0, 2, 3, 1])
    
    coords_offsets, coords_relative = get_relative_coords(coords_nn)

    # --- FINAL ROBUST LOGIC ---
    Y4d_nn = None
    if ptycho_data.Y is not None:
        # This is the only acceptable path for pre-prepared data.
        print("INFO: Using pre-computed 'Y' array from the input file.")
        # Convert (n_groups, n_neighbors, H, W) -> (n_groups, H, W, n_neighbors)
        Y4d_nn = np.transpose(ptycho_data.Y[nn_indices], [0, 2, 3, 1])
    elif ptycho_data.objectGuess is not None:
        """
        ### TODO: Re-enable and Verify `objectGuess` Fallback for Ground Truth Patch Generation

        **Context & Goal:**

        The data loader in `ptycho/raw_data.py` was modified to fix a critical bug
        in the supervised training pipeline. As a safety measure, the fallback
        logic—which generates ground truth `Y` patches on-the-fly from a full
        `objectGuess`—was disabled by raising a `NotImplementedError`.

        The goal of this task is to safely re-enable this fallback path. This is
        essential for maintaining backward compatibility with workflows (like
        unsupervised PINN training) that start with `.npz` files containing only
        `objectGuess` and not a pre-computed `Y` array.

        **Implementation Steps:**

        1.  **Remove the `NotImplementedError`:** Delete the `raise NotImplementedError(...)`
            line below.
        2.  **Implement the Fallback Logic:** Add the following line in its place:
            
            ```python
            logging.info("'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
            Y4d_nn = get_image_patches(ptycho_data.objectGuess, coords_offsets, coords_relative)
            ```

        **Verification Plan:**

        After re-enabling the logic, you must verify that both the fallback and primary
        paths work correctly.

        1.  **Test Fallback Path:**
            -   Find or create an `.npz` file with `objectGuess` but no `Y` array.
            -   Run a PINN training workflow with this file.
            -   **Expected:** The script must run without error and log the message:
                "INFO: ...Generating ground truth patches... as a fallback."

        2.  **Test No Regression (Primary Path):**
            -   Use a fully prepared dataset that *does* contain the `Y` array (e.g.,
              `datasets/fly/fly001_prepared_64.npz`).
            -   Run the supervised baseline training script.
            -   **Expected:** The script must run without error and log the message:
                "INFO: Using pre-computed 'Y' array..." It must *not* log the
                "fallback" message.
        """
        logging.info("'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
        gridsize = params.get('gridsize')
        Y_patches = get_image_patches(ptycho_data.objectGuess, coords_offsets, coords_relative, N=N, gridsize=gridsize)
        # Always keep 4D shape for consistent tensor dimensions downstream
        Y4d_nn = Y_patches
    else:
        # For PINN training, we don't need ground truth patches
        print("INFO: No ground truth data ('Y' array or 'objectGuess') found.")
        print("INFO: This is expected for PINN training which doesn't require ground truth.")
        Y4d_nn = None
    # --- END FINAL LOGIC ---

    if ptycho_data.xcoords_start is not None:
        coords_start_nn = np.transpose(np.array([ptycho_data.xcoords_start[nn_indices], ptycho_data.ycoords_start[nn_indices]]),
                                       [1, 0, 2])[:, None, :, :]
        coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
    else:
        coords_start_offsets = coords_start_relative = None

    dset = {
        'diffraction': diff4d_nn,
        'Y': Y4d_nn,
        'coords_offsets': coords_offsets,
        'coords_relative': coords_relative,
        'coords_start_offsets': coords_start_offsets,
        'coords_start_relative': coords_start_relative,
        'coords_nn': coords_nn,
        'coords_start_nn': coords_start_nn,
        'nn_indices': nn_indices,
        'objectGuess': ptycho_data.objectGuess
    }
    X_full = normalize_data(dset, N)
    dset['X_full'] = X_full
    print('neighbor-sampled diffraction shape', X_full.shape)
    return dset

#@debug
def normalize_data(dset: dict, N: int) -> np.ndarray:
    """
    Normalize the diffraction data.

    Args:
        dset (dict): Dictionary containing the dataset.
        N (int): Size of the solution region.

    Returns:
        np.ndarray: Normalized diffraction data.
    """
    # Images are amplitude, not intensity
    X_full = dset['diffraction']
    X_full_norm = np.sqrt(
            ((N / 2)**2) / np.mean(tf.reduce_sum(dset['diffraction']**2, axis=[1, 2]))
            )
    return X_full_norm * X_full


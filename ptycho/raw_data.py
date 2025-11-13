"""
Core data ingestion and preprocessing module for ptychographic datasets.

This module serves as the first stage of the PtychoPINN data pipeline, responsible for
transforming raw NPZ files into structured data containers and performing critical
coordinate grouping operations for overlap-based training.

Architecture Role:
In the broader PtychoPINN architecture, this module bridges the gap between raw
experimental data files and the structured data containers needed by the machine
learning pipeline. Data flows: NPZ files → raw_data.py (RawData) → loader.py 
(PtychoDataContainer) → model-ready tensors.

Primary Components:
- `RawData`: Core data container class with validation and I/O capabilities
- `RawData.from_file()`: Static factory method for loading NPZ datasets
- `RawData.generate_grouped_data()`: Efficient coordinate grouping using "sample-then-group" strategy

Key Algorithm - Sample-Then-Group Strategy:
The coordinate grouping implementation uses an efficient "sample-then-group" approach:
1. Sample seed points from the full dataset (random or sequential)
2. Find K nearest neighbors only for sampled points (not all points)
3. Form groups of size C (gridsize²) from neighbors
4. Generate final dataset with proper coordinate transformations

Performance Characteristics:
- **10-100x faster** than cache-based approaches (no cache generation needed)
- **10-100x lower memory usage** (no storage of all possible groups)
- **Zero cache files** (eliminates disk I/O overhead)
- **Deterministic results** via optional seed parameter
- **O(nsamples * K)** complexity instead of O(n_points * K)

Public Interface:
    `RawData.generate_grouped_data(N, K, nsamples, dataset_path, seed, sequential_sampling)`
        Returns dictionary with the following structure:

        Required Keys:
        - 'diffraction': np.ndarray, shape (nsamples, N, N, gridsize²), dtype complex/float
                        Grouped diffraction patterns in channel format
        - 'coords_offsets': np.ndarray, shape (nsamples, 1, 2, 1), dtype float
                           Mean coordinates for each group (global positioning)
        - 'coords_relative': np.ndarray, shape (nsamples, 1, 2, gridsize²), dtype float
                            Relative coordinates within each group
        - 'nn_indices': np.ndarray, shape (nsamples, gridsize²), dtype int
                       Selected coordinate indices for each group
        - 'X_full': np.ndarray, shape (nsamples, N, N, gridsize²), dtype complex/float
                   Normalized diffraction data ready for model input

        Optional Keys (availability depends on input data):
        - 'Y': np.ndarray, shape (nsamples, N, N, gridsize²), dtype complex
              Ground truth object patches (if objectGuess provided)
        - 'coords_start_offsets': np.ndarray, shape (nsamples, 1, 2, 1), dtype float
                                 Start coordinate offsets (if start coords provided)
        - 'coords_start_relative': np.ndarray, shape (nsamples, 1, 2, gridsize²), dtype float
                                  Relative start coordinates (if start coords provided)
        - 'coords_nn': np.ndarray, shape (nsamples, 1, 2, gridsize²), dtype float
                      Full coordinate data for groups
        - 'coords_start_nn': np.ndarray, shape (nsamples, 1, 2, gridsize²), dtype float
                            Start coordinate data for groups (if available)
        - 'objectGuess': np.ndarray, shape (M, M), dtype complex
                        Original full object for reference (if provided)

Usage Example:
    ```python
    from ptycho.raw_data import RawData
    from ptycho import loader
    
    # Load raw experimental data
    raw_data = RawData.from_file("/path/to/data.npz")
    
    # Generate grouped data for training
    grouped_data = raw_data.generate_grouped_data(
        N=64,          # Diffraction pattern size
        K=6,           # Number of nearest neighbors
        nsamples=1000, # Number of training groups
        seed=42        # Optional: for reproducible results
    )
    
    # Access structured outputs
    diffraction = grouped_data['diffraction']  # (1000, 64, 64, 1) for gridsize=1
    coordinates = grouped_data['coords_offsets']  # (1000, 1, 2, 1)
    
    # Convert to model-ready tensors
    container = loader.load(
        cb=lambda: grouped_data,
        probeGuess=raw_data.probeGuess,
        which='train'
    )
    ```

State Dependencies:
- Depends on params.get('gridsize') for determining group size (C = gridsize²)
- Uses params.get('N') as fallback for patch size in get_image_patches()
- Caching behavior eliminated - no dependency on dataset_path for cache files

Data Contract Compliance:
Adheres to normative specs in `docs/specs/spec-ptycho-interfaces.md` and `docs/specs/spec-ptycho-core.md`.
Expected NPZ keys and dtypes:
- `xcoords (M,) float64`, `ycoords (M,) float64` — pixel coordinates on the object grid
- `diff3d (M, N, N) float32` — amplitude (sqrt of counts), not intensity
- `probeGuess (N, N) complex64` — probe in object pixel grid
- Optional: `scan_index (M,) int64` (defaults to zeros), `objectGuess (H, W) complex64`,
  `xcoords_start (M,)`, `ycoords_start (M,)` (default to `xcoords`, `ycoords`)

Primary Consumers:
- ptycho.data_preprocessing (3 imports): Uses RawData for preprocessing workflows
- ptycho.loader (1 import): Converts RawData outputs to model-ready tensors
- ptycho.workflows.components (1 import): High-level workflow orchestration
"""
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from scipy.spatial import cKDTree
import os
import logging
from pathlib import Path
from ptycho import params
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
             scan_index, objectGuess = None, Y = None, norm_Y_I = None, metadata = None):
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
            print(f"diff3d shape: {diff3d.shape}")
            assert diff3d.shape[1] == diff3d.shape[2]
        if probeGuess is not None:
            assert len(probeGuess.shape) == 2, f"Expected probeGuess to be 2D, got shape {probeGuess.shape}"
            print(f"probeGuess shape: {probeGuess.shape}")
        if scan_index is not None:
            assert len(scan_index.shape) == 1, f"Expected scan_index to be 1D, got shape {scan_index.shape}"
            print(f"scan_index shape: {scan_index.shape}")
        if objectGuess is not None:
            print(f"objectGuess shape: {objectGuess.shape}")
            assert len(objectGuess.shape) == 2

        print(f"xcoords shape: {xcoords.shape}")
        print(f"ycoords shape: {ycoords.shape}")
        print(f"xcoords_start shape: {xcoords_start.shape}")
        print(f"ycoords_start shape: {ycoords_start.shape}")

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
        self.metadata = metadata  # Store metadata from NPZ file
        self.sample_indices = None
        self.subsample_seed = None

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
        Create a RawData instance from simulation data.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            probeGuess (np.ndarray): initial guess of the probe function.
            objectGuess (np.ndarray): initial guess of the object.
            scan_index (np.ndarray, optional): array indicating the scan index for each diffraction pattern.

        Returns:
            RawData: An instance of the RawData class with simulated data.
        """
        from ptycho.diffsim import illuminate_and_diffract, scale_nphotons
        import tensorflow as tf
        
        xcoords_start = xcoords
        ycoords_start = ycoords
        
        # For gridsize=1 simulation, we handle individual coordinates directly
        # instead of complex grouping. This replaces the missing calculate_relative_coords.
        gridsize = params.get('gridsize')
        if gridsize is None:
            gridsize = 1
        if gridsize != 1:
            raise NotImplementedError(
                "from_simulation currently only supports gridsize=1. "
                "For gridsize>1, use the modern generate_grouped_data workflow instead."
            )
        
        # Create simple coordinate format for gridsize=1
        n_images = len(xcoords)
        nn_indices = np.arange(n_images)  # Each image maps to itself
        
        # Format coordinates to match expected shape: (M, 1, 2, 1) for gridsize=1
        # Each point becomes its own "group" of size 1
        coords_nn = np.zeros((n_images, 1, 2, 1))
        coords_nn[:, 0, 0, 0] = xcoords
        coords_nn[:, 0, 1, 0] = ycoords
        
        global_offsets, local_offsets = get_relative_coords(coords_nn)

        Y_obj = get_image_patches(objectGuess, global_offsets, local_offsets) 
        Y_I = tf.math.abs(Y_obj)
        Y_phi = tf.math.angle(Y_obj)
        X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
        norm_Y_I = scale_nphotons(X)
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
    def from_file(train_data_file_path: str, validate_config: bool = False, current_config = None) -> 'RawData':
        """
        Static method to create a RawData instance from a file.

        Args:
            train_data_file_path (str): Path to the file containing the data.
            validate_config (bool): Whether to validate current config against metadata
            current_config: Optional current configuration for validation

        Returns:
            RawData: An instance of the RawData class.
        """
        from ptycho.metadata import MetadataManager
        
        # Load training data with metadata support
        data_dict, metadata = MetadataManager.load_with_metadata(train_data_file_path)
        
        # Validate configuration if requested
        if validate_config and metadata and current_config:
            warnings_list = MetadataManager.validate_parameters(metadata, current_config)
            for warning in warnings_list:
                import logging
                logging.warning(f"Parameter mismatch: {warning}")
        
        # Handle legacy files that might not have all fields
        xcoords_start = data_dict.get('xcoords_start', data_dict['xcoords'])
        ycoords_start = data_dict.get('ycoords_start', data_dict['ycoords'])
        scan_index = data_dict.get('scan_index', np.zeros(len(data_dict['xcoords']), dtype=int))
        
        train_raw_data = RawData(
            xcoords=data_dict['xcoords'],
            ycoords=data_dict['ycoords'],
            xcoords_start=xcoords_start,
            ycoords_start=ycoords_start,
            diff3d=data_dict['diff3d'],
            probeGuess=data_dict['probeGuess'],
            objectGuess=data_dict.get('objectGuess'),
            scan_index=scan_index,
            metadata=metadata  # Pass metadata to RawData instance
        )
        
        # Log if metadata was loaded
        if metadata:
            import logging
            nphotons = MetadataManager.get_nphotons(metadata)
            logging.debug(f"Loaded dataset with metadata: nphotons={nphotons}")
        
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
    def generate_grouped_data(self, N, K = 4, nsamples = 1, dataset_path: Optional[str] = None, seed: Optional[int] = None, sequential_sampling: bool = False, gridsize: Optional[int] = None, enable_oversampling: bool = False, neighbor_pool_size: Optional[int] = None):
        """
        Generate nearest-neighbor solution region grouping with efficient sampling.
        
        This method implements a "sample-then-group" strategy that first samples
        seed points from the dataset, then finds neighbors only for those seed points.
        This approach is highly efficient and eliminates the need for caching.
        
        **Efficient Sampling Strategy:**
        1. Either randomly samples or sequentially selects nsamples seed points
        2. Finds K nearest neighbors only for the sampled points
        3. Forms groups of size C (gridsize²) from the neighbors
        4. Handles edge cases gracefully (small datasets, etc.)

        Args:
            N (int): Size of the solution region.
            K (int, optional): Number of nearest neighbors. Defaults to 4.
            nsamples (int, optional): Number of samples. For gridsize=1, this is the
                                    number of individual images. For gridsize>1, this
                                    is the number of neighbor groups (total images = 
                                    nsamples * gridsize²).
            dataset_path (str, optional): Path to dataset (kept for compatibility, no longer used for caching).
            seed (int, optional): Random seed for reproducible sampling.
            sequential_sampling (bool, optional): If True, uses the first nsamples points sequentially
                                                 instead of random sampling. Useful for debugging or
                                                 analyzing specific scan regions. Defaults to False.
            gridsize (int, optional): Grid size for patch grouping. If None, falls back to
                                     params.get('gridsize', 1). Explicit parameter takes precedence
                                     for better dependency injection and testing.
            enable_oversampling (bool, optional): Explicit opt-in for K choose C oversampling.
                                                 Defaults to False. When True and nsamples > n_points,
                                                 enables oversampling if neighbor_pool_size >= C.
            neighbor_pool_size (Optional[int], optional): Pool size K for oversampling. If None,
                                                         defaults to the K parameter. Must be >= C
                                                         when enable_oversampling is True.

        Returns:
            dict: Dictionary containing grouped data with keys:
                - 'diffraction': 4D array of diffraction patterns
                - 'Y': 4D array of ground truth patches (if available)
                - 'coords_offsets', 'coords_relative': Coordinate information
                - 'nn_indices': Selected neighbor indices  
                - 'X_full': Normalized diffraction data
                - Additional coordinate and metadata arrays
                
        Raises:
            ValueError: If dataset is too small for requested parameters
            
        Note:
            The new efficient implementation eliminates the need for caching.
            Performance is fast enough that first-run and subsequent runs
            have similar execution times.

        ⚠️ CRITICAL DEPENDENCY WARNING ⚠️
        This method requires params.cfg['gridsize'] to be initialized.

        Initialization requirements:
        - For training: Call update_legacy_dict(params.cfg, config) first
        - For inference: Ensure params.cfg is populated from saved model
        - For testing: Set params.cfg['gridsize'] explicitly

        Common failure scenario:
        - Symptom: Getting shape (*, 64, 64, 1) instead of (*, 64, 64, 4)
        - Cause: params.cfg['gridsize'] not set, defaults to 1
        - Fix: Ensure update_legacy_dict() called before this method

        See: docs/debugging/TROUBLESHOOTING.md#shape-mismatch-errors
        """
        # Use explicit gridsize parameter if provided, otherwise fallback to params
        if gridsize is None:
            gridsize = params.get('gridsize', 1)
        # gridsize now comes from explicit parameter, maintaining backward compatibility
        
        # Unified efficient logic for all gridsize values
        C = gridsize ** 2  # Number of coordinates per solution region
        n_points = len(self.xcoords)

        # Determine effective pool size K for oversampling
        effective_K = neighbor_pool_size if neighbor_pool_size is not None else K

        # Debug logging for method entry
        logging.info(f"[OVERSAMPLING DEBUG] generate_grouped_data called with: nsamples={nsamples}, n_points={n_points}, C={C}, K={K}")
        logging.info(f"[OVERSAMPLING DEBUG] Parameters: gridsize={gridsize}, N={N}, sequential_sampling={sequential_sampling}")
        logging.info(f"[OVERSAMPLING DEBUG] Oversampling flags: enable_oversampling={enable_oversampling}, neighbor_pool_size={neighbor_pool_size}, effective_K={effective_K}")

        # Determine if oversampling is needed
        needs_oversampling = (nsamples > n_points) and (C > 1)
        logging.info(f"[OVERSAMPLING DEBUG] Oversampling check: nsamples > n_points = {nsamples} > {n_points} = {nsamples > n_points}")
        logging.info(f"[OVERSAMPLING DEBUG] Oversampling check: C > 1 = {C} > 1 = {C > 1}")
        logging.info(f"[OVERSAMPLING DEBUG] needs_oversampling = {needs_oversampling}")

        # OVERSAMPLING-001 Guard: Require explicit opt-in and enforce preconditions
        if needs_oversampling:
            if not enable_oversampling:
                logging.error(f"[OVERSAMPLING DEBUG] Oversampling disabled (enable_oversampling={enable_oversampling})")
                raise ValueError(
                    f"Requesting {nsamples} groups but only {n_points} points available (gridsize={gridsize}, C={C}). "
                    f"K choose C oversampling is required but not enabled. "
                    f"Set enable_oversampling=True and ensure neighbor_pool_size >= {C} to proceed. "
                    f"See OVERSAMPLING-001 in docs/findings.md for details."
                )
            if effective_K < C:
                logging.error(f"[OVERSAMPLING DEBUG] neighbor_pool_size ({effective_K}) < C ({C})")
                raise ValueError(
                    f"K choose C oversampling requires neighbor_pool_size >= C (gridsize²). "
                    f"Got neighbor_pool_size={effective_K}, but C={C}. "
                    f"Increase neighbor_pool_size to at least {C}. "
                    f"See OVERSAMPLING-001 in docs/findings.md for details."
                )
            logging.info(f"[OVERSAMPLING DEBUG] Oversampling guards passed: enable_oversampling=True, effective_K={effective_K} >= C={C}")
        
        # Determine sampling strategy
        if sequential_sampling:
            # Use sequential indices (first nsamples points)
            seed_indices = np.arange(min(nsamples, len(self.xcoords)))
            print('DEBUG:', f'nsamples: {nsamples}, gridsize: {gridsize} (using sequential sampling - first {len(seed_indices)} points)')
            logging.info(f"Using sequential sampling strategy for gridsize={gridsize}")
        else:
            # Use random sampling (default)
            seed_indices = None
            strategy = "K choose C oversampling" if needs_oversampling else "efficient"
            print('DEBUG:', f'nsamples: {nsamples}, gridsize: {gridsize} (using {strategy} random sample-then-group strategy)')
            logging.info(f"Using {strategy} random sampling strategy for gridsize={gridsize}")
        
        # Automatically route to appropriate implementation
        if needs_oversampling:
            # Use K choose C oversampling when requesting more groups than available points
            logging.info(f"[OVERSAMPLING DEBUG] Taking oversampling branch: K choose C oversampling")
            logging.info(f"Automatically using K choose C oversampling: {nsamples} groups requested but only {n_points} points available (K={effective_K}, C={C})")
            selected_groups = self._generate_groups_with_oversampling(
                nsamples=nsamples,
                K=effective_K,
                C=C,
                seed=seed,
                seed_indices=seed_indices
            )
        else:
            # Use the existing efficient method for standard cases
            logging.info(f"[OVERSAMPLING DEBUG] Taking efficient branch: standard sample-then-group")
            selected_groups = self._generate_groups_efficiently(
                nsamples=nsamples, 
                K=K, 
                C=C, 
                seed=seed,
                seed_indices=seed_indices
            )
        
        logging.info(f"[OVERSAMPLING DEBUG] Generated {len(selected_groups)} groups in total")
        logging.info(f"Generated {len(selected_groups)} groups efficiently")
        
        # Generate the final dataset from the selected groups
        return self._generate_dataset_from_groups(selected_groups, N, K, gridsize)

    def _generate_dataset_from_groups(self, selected_groups: np.ndarray, N: int, K: int, gridsize: int) -> dict:
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
            print("INFO: Using pre-computed 'Y' array from the input file.")
            Y4d_nn = np.transpose(self.Y[nn_indices], [0, 2, 3, 1])
        elif self.objectGuess is not None:
            print("INFO: 'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
            Y4d_nn = get_image_patches(self.objectGuess, coords_offsets, coords_relative, N=N, gridsize=gridsize)
        else:
            print("INFO: No ground truth data ('Y' array or 'objectGuess') found.")
            print("INFO: This is expected for PINN training which doesn't require ground truth.")
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
        if getattr(self, 'sample_indices', None) is not None:
            dset['sample_indices'] = np.array(self.sample_indices, copy=True)
        
        # Apply normalization
        X_full = normalize_data(dset, N)
        dset['X_full'] = X_full
        print('neighbor-sampled diffraction shape', X_full.shape)
        
        return dset


    def _generate_groups_efficiently(self, nsamples: int, K: int, C: int, seed: Optional[int] = None, seed_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Efficiently generate coordinate groups using a "sample-then-group" strategy.
        
        This method first samples seed points from the dataset (or uses provided seed_indices),
        then finds neighbors only for those seed points, drastically reducing computation and 
        memory usage compared to the "group-then-sample" approach.
        
        Args:
            nsamples: Number of groups to generate
            K: Number of nearest neighbors to consider (including self)
            C: Number of coordinates per group (typically gridsize^2)
            seed: Random seed for reproducibility (optional)
            seed_indices: Pre-selected seed indices for sequential sampling (optional)
            
        Returns:
            np.ndarray: Array of group indices with shape (nsamples, C)
            
        Raises:
            ValueError: If K < C or if dataset is too small
        """
        try:
            # Set random seed if provided (only affects random sampling, not sequential)
            if seed is not None and seed_indices is None:
                np.random.seed(seed)
            
            n_points = len(self.xcoords)
            logging.info(f"[OVERSAMPLING DEBUG] _generate_groups_efficiently called with: nsamples={nsamples}, K={K}, C={C}")
            logging.info(f"Generating {nsamples} groups efficiently from {n_points} points (K={K}, C={C})")
            
            # Validate inputs
            if n_points < C:
                raise ValueError(f"Dataset has only {n_points} points but {C} coordinates per group requested.")
            
            if K < C:
                raise ValueError(f"K={K} must be >= C={C} (need at least C neighbors to form a group)")
            
            # Step 1: Use provided seed indices or sample them
            if seed_indices is not None:
                # Sequential sampling: use provided indices
                n_samples_actual = min(len(seed_indices), n_points)
                seed_indices = seed_indices[:n_samples_actual]
                logging.info(f"[OVERSAMPLING DEBUG] Using sequential sampling with {n_samples_actual} seed indices")
                logging.info(f"Using provided {n_samples_actual} sequential seed indices")
                # Set a fixed seed for neighbor selection to ensure determinism
                np.random.seed(0)
            else:
                # Random sampling: handle edge case where more samples requested than available points
                if nsamples > n_points:
                    logging.info(f"[OVERSAMPLING DEBUG] Capping groups: requested {nsamples} but only {n_points} points available")
                    logging.warning(f"Requested {nsamples} groups but only {n_points} points available. Using all points as seeds.")
                    n_samples_actual = n_points
                else:
                    n_samples_actual = nsamples
                    logging.info(f"[OVERSAMPLING DEBUG] Standard case: using {n_samples_actual} groups from {n_points} points")
                
                # Sample seed points randomly
                all_indices = np.arange(n_points)
                if n_samples_actual < n_points:
                    seed_indices = np.random.choice(all_indices, size=n_samples_actual, replace=False)
                    logging.info(f"[OVERSAMPLING DEBUG] Randomly sampled {n_samples_actual} seed points")
                    logging.info(f"Sampled {n_samples_actual} seed points from {n_points} total points")
                else:
                    seed_indices = all_indices
                    logging.info(f"[OVERSAMPLING DEBUG] Using all {n_points} points as seeds (no sampling needed)")
                    logging.info(f"Using all {n_points} points as seeds")
            
            # Special case for C=1 (gridsize=1): use seed indices directly without neighbor search
            if C == 1:
                # For gridsize=1, we want the seed points themselves, not their neighbors
                groups = seed_indices.reshape(-1, 1)
                logging.info(f"Using seed indices directly for C=1 (gridsize=1) - no neighbor search needed")
            else:
                # Step 2: Build KDTree for efficient neighbor search
                coords = np.column_stack([self.xcoords, self.ycoords])
                tree = cKDTree(coords)
                
                # Step 3: Find K nearest neighbors for each seed point
                seed_coords = coords[seed_indices]
                # Query K+1 neighbors (including self), then remove self
                distances, neighbor_indices = tree.query(seed_coords, k=min(K+1, n_points))
                
                # Step 4: Generate groups by selecting C coordinates from each seed's neighbors
                groups = np.zeros((n_samples_actual, C), dtype=np.int32)
                # For C > 1, select from neighbors as before
                for i in range(n_samples_actual):
                    # Get this seed's neighbors (excluding self if K+1 was queried)
                    neighbors = neighbor_indices[i]
                    if len(neighbors) > K:
                        # Remove self (first element) if we queried K+1
                        neighbors = neighbors[1:K+1]
                    else:
                        # Use all available neighbors if dataset is small
                        neighbors = neighbors[:K]
                    
                    # Ensure we have enough neighbors
                    if len(neighbors) < C:
                        # If not enough neighbors, include the seed point itself
                        available = np.concatenate([[seed_indices[i]], neighbors])
                    else:
                        available = neighbors
                    
                    # Randomly select C indices from available neighbors
                    if len(available) >= C:
                        selected = np.random.choice(available, size=C, replace=False)
                    else:
                        # If still not enough, allow replacement (edge case for very small datasets)
                        selected = np.random.choice(available, size=C, replace=True)
                    
                    groups[i] = selected
            
            logging.info(f"[OVERSAMPLING DEBUG] _generate_groups_efficiently completed: generated {n_samples_actual} groups")
            logging.info(f"Successfully generated {n_samples_actual} groups with shape {groups.shape}")
            return groups
            
        except Exception as e:
            logging.error(f"Failed to generate groups efficiently: {e}")
            raise

    def _generate_groups_with_oversampling(self, nsamples, K, C, seed=None, seed_indices=None):
        """
        Generate groups using K choose C combinations for data augmentation.
        
        This method enables creating more groups than seed points by generating
        multiple combinations from each seed's K nearest neighbors.
        
        Args:
            nsamples: Number of groups to generate (can be > number of seed points)
            K: Number of nearest neighbors to consider
            C: Number of coordinates per group (gridsize²)
            seed: Random seed for reproducibility
            seed_indices: Optional pre-selected seed indices
            
        Returns:
            np.ndarray: Array of shape (nsamples, C) containing selected indices
        """
        from itertools import combinations
        
        try:
            if seed is not None:
                np.random.seed(seed)
            
            n_points = len(self.xcoords)
            logging.info(f"[OVERSAMPLING DEBUG] _generate_groups_with_oversampling called with: nsamples={nsamples}, K={K}, C={C}")
            logging.info(f"Generating {nsamples} groups with K choose C oversampling from {n_points} points (K={K}, C={C})")
            
            # Validate inputs
            if n_points < C:
                raise ValueError(f"Dataset has only {n_points} points but {C} coordinates per group requested.")
            
            if K < C:
                raise ValueError(f"K={K} must be >= C={C} (need at least C neighbors to form a group)")
            
            # Calculate maximum combinations per seed
            max_combos_per_seed = 1 if C == 1 else len(list(combinations(range(K), C)))
            logging.info(f"[OVERSAMPLING DEBUG] Max combinations per seed: {max_combos_per_seed} (C={C}, K={K})")
            logging.info(f"Each seed point can generate up to {max_combos_per_seed} combinations")
            
            # Step 1: Determine how many seed points we need
            min_seeds_needed = max(1, (nsamples + max_combos_per_seed - 1) // max_combos_per_seed)
            n_seeds = min(min_seeds_needed * 2, n_points)  # Use 2x seeds for diversity
            logging.info(f"[OVERSAMPLING DEBUG] Calculated seed requirements: min_seeds_needed={min_seeds_needed}, using n_seeds={n_seeds}")
            
            # Step 2: Select seed points
            if seed_indices is not None and len(seed_indices) >= min_seeds_needed:
                seed_indices = seed_indices[:n_seeds]
                logging.info(f"[OVERSAMPLING DEBUG] Using provided seed_indices (first {n_seeds})")
            else:
                all_indices = np.arange(n_points)
                if n_seeds < n_points:
                    seed_indices = np.random.choice(all_indices, size=n_seeds, replace=False)
                    logging.info(f"[OVERSAMPLING DEBUG] Randomly selected {n_seeds} seed points from {n_points}")
                else:
                    seed_indices = all_indices
                    logging.info(f"[OVERSAMPLING DEBUG] Using all {n_points} points as seeds")
            
            logging.info(f"Using {len(seed_indices)} seed points to generate {nsamples} groups")
            
            # Special case for C=1 (gridsize=1)
            if C == 1:
                # For C=1, just sample with replacement if needed
                if nsamples <= len(seed_indices):
                    groups = np.random.choice(seed_indices, size=nsamples, replace=False).reshape(-1, 1)
                else:
                    groups = np.random.choice(seed_indices, size=nsamples, replace=True).reshape(-1, 1)
                logging.info(f"Generated {nsamples} groups for C=1 case")
                return groups
            
            # Step 3: Build KDTree and find neighbors
            coords = np.column_stack([self.xcoords, self.ycoords])
            tree = cKDTree(coords)
            
            # Step 4: Generate combination pool
            combination_pool = []
            seed_mapping = []  # Track which seed each combination came from
            
            for seed_idx in seed_indices:
                # Find K nearest neighbors (including self)
                seed_coord = coords[seed_idx:seed_idx+1]
                distances, neighbor_indices = tree.query(seed_coord, k=min(K+1, n_points))
                
                # Get neighbors (excluding self)
                neighbors = neighbor_indices[0]
                if len(neighbors) > K:
                    neighbors = neighbors[1:K+1]  # Exclude self
                else:
                    neighbors = neighbors[1:] if len(neighbors) > 1 else neighbors  # Handle edge case
                
                # Ensure we have enough neighbors
                if len(neighbors) < C:
                    # Include seed point if not enough neighbors
                    available = np.concatenate([[seed_idx], neighbors])
                else:
                    available = neighbors[:K]  # Use up to K neighbors
                
                # Generate all C-combinations from available points
                if len(available) >= C:
                    for combo in combinations(available, C):
                        combination_pool.append(np.array(combo))
                        seed_mapping.append(seed_idx)
                        
                        # Early stopping if we have enough combinations
                        if len(combination_pool) >= nsamples * 2:
                            break
                
                if len(combination_pool) >= nsamples * 2:
                    break
            
            # Step 5: Sample from combination pool
            total_combinations = len(combination_pool)
            logging.info(f"[OVERSAMPLING DEBUG] Generated combination pool: {total_combinations} total combinations")
            logging.info(f"Generated pool of {total_combinations} combinations")
            
            if total_combinations == 0:
                raise ValueError("No valid combinations could be generated")
            
            # Convert pool to array for efficient indexing
            combination_pool = np.array(combination_pool)
            
            # Sample combinations
            if nsamples <= total_combinations:
                # Sample without replacement for diversity
                selected_indices = np.random.choice(total_combinations, size=nsamples, replace=False)
                logging.info(f"[OVERSAMPLING DEBUG] Sampling {nsamples} from {total_combinations} combinations (without replacement)")
            else:
                # Sample with replacement if requesting more than available
                logging.info(f"[OVERSAMPLING DEBUG] Need {nsamples} groups but only {total_combinations} combinations available - using replacement")
                logging.warning(f"Requested {nsamples} groups but only {total_combinations} unique combinations available. Sampling with replacement.")
                selected_indices = np.random.choice(total_combinations, size=nsamples, replace=True)
            
            groups = combination_pool[selected_indices]
            
            logging.info(f"[OVERSAMPLING DEBUG] _generate_groups_with_oversampling completed: generated {nsamples} groups")
            logging.info(f"Successfully generated {nsamples} groups with K choose C oversampling")
            return groups
            
        except Exception as e:
            logging.error(f"Failed to generate groups with oversampling: {e}")
            raise

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
#@debug
def get_image_patches(gt_image, global_offsets, local_offsets, N=None, gridsize=None):
    """
    Generate and return image patches in channel format using a single canvas.

    Args:
        gt_image (tensor): Ground truth image tensor.
        global_offsets (tensor): Global offset tensor.
        local_offsets (tensor): Local offset tensor.
        N (int, optional): Patch size. If None, uses params.get('N').
        gridsize (int, optional): Grid size. If None, uses params.get('gridsize').

    Returns:
        tensor: Image patches in channel format.
    """
    # Use explicit parameters if provided, otherwise fall back to global params
    # This follows the project's hybrid modernization pattern
    N = N if N is not None else params.get('N')
    gridsize = gridsize if gridsize is not None else params.get('gridsize')
    B = global_offsets.shape[0]
    c = gridsize**2

    # Pad the ground truth image once
    gt_padded = hh.pad(gt_image[None, ..., None], N // 2)

    # Calculate the combined offsets by adding global and local offsets
    offsets_c = tf.cast((global_offsets + local_offsets), tf.float32)
    offsets_f = hh._channel_to_flat(offsets_c)

    # Create a canvas to store the extracted patches
    canvas = np.zeros((B, N, N, c), dtype=np.complex64)

    # Iterate over the combined offsets and extract patches one by one
    for i in range(B * c):
        offset = -offsets_f[i, :, :, 0]
        translated_patch = hh.translate(gt_padded, offset)
        canvas[i // c, :, :, i % c] = np.array(translated_patch)[0, :N, :N, 0]

    # Convert the canvas to a TensorFlow tensor and return it
    return tf.convert_to_tensor(canvas)

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

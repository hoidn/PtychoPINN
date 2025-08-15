"""
Core data ingestion and preprocessing module for ptychographic datasets.

This module serves as the first stage of the PtychoPINN data pipeline, responsible for
transforming raw NPZ files into structured data containers and performing critical
coordinate grouping operations for overlap-based training.

Primary Consumer Context:
Its primary consumers are ptycho.data_preprocessing (3 imports), ptycho.loader (1 import), 
and ptycho.workflows.components (1 import), which use it to prepare raw ptychographic 
data for model training and inference.

Key Architecture Integration:
In the broader PtychoPINN architecture, this module bridges the gap between raw
experimental data files and the structured data containers needed by the machine
learning pipeline. The data flows: NPZ files → raw_data.py (RawData) → loader.py 
(PtychoDataContainer) → model-ready tensors.

Key Components:
- `RawData`: Primary data container class with validation and I/O capabilities
  - `.generate_grouped_data()`: Core grouping method for gridsize > 1 with intelligent caching
  - `.diffraction`: Raw diffraction patterns array (amplitude, not intensity)
  - `.xcoords, .ycoords`: Scan position coordinates
  - `.objectGuess`: Full sample object for ground truth patch generation
  - `.Y`: Pre-computed ground truth patches (optional)

Public Interface:
    `group_coords(xcoords, ycoords, K, C, nsamples)`
        - Purpose: Groups scan coordinates by spatial proximity for overlap training
        - Key Parameters:
            - `K` (int): **Number of nearest neighbors for grouping**
              Controls the overlap constraint strength - larger K provides more potential
              neighbors but increases computational cost. Typical values: 4-8.
            - `C` (int): **Target coordinates per solution region** 
              Usually equals gridsize². Determines spatial coherence of training patches.
            - `nsamples` (int): **Number of training samples to generate**
              For gridsize=1: individual images. For gridsize>1: neighbor groups.
        - Returns: Tuple of grouped coordinate arrays and neighbor indices
        - Used by: RawData.generate_grouped_data(), data preprocessing workflows

    `get_neighbor_diffraction_and_positions(ptycho_data, N, K, C, nsamples)`
        - Purpose: Legacy function for generating grouped diffraction data (gridsize=1)
        - Parameters: RawData instance, solution region size, neighbor parameters
        - Returns: Dictionary with diffraction patterns, coordinates, and ground truth
        - Caching: No automatic caching (preserved for backward compatibility)

Usage Example:
    This module is typically used at the start of the PtychoPINN data loading pipeline,
    which converts raw experimental data into model-ready tensors.

    ```python
    from ptycho.raw_data import RawData
    from ptycho import loader
    
    # 1. Load raw experimental data from NPZ file
    raw_data = RawData.from_file("/path/to/experimental_data.npz")
    
    # 2. Generate grouped data for overlap-based training (gridsize > 1)
    grouped_data = raw_data.generate_grouped_data(
        N=64,  # Diffraction pattern size - must match probe dimensions
        K=6,   # Number of nearest neighbors - critical for physics constraint
        nsamples=1000  # Target number of training groups
    )
    
    # 3. Pass to loader for tensor conversion and normalization
    container = loader.load(
        cb=lambda: grouped_data,
        probeGuess=raw_data.probeGuess,
        which='train'
    )
    
    # 4. Access model-ready data
    X_data = container.X  # Normalized diffraction patterns
    Y_data = container.Y  # Ground truth patches (if available)
    ```

Integration Notes:
The K parameter in coordinate grouping is critical for physics-informed training.
Too small K limits the overlap constraint effectiveness; too large K increases
computational cost exponentially. The grouping operation is cached automatically
for gridsize > 1 using the format `<dataset>.g{gridsize}k{K}.groups_cache.npz`.

For gridsize=1, the module preserves legacy sequential sampling for backward
compatibility. For gridsize>1, it implements a "group-then-sample" strategy that
ensures both physical coherence and spatial representativeness.

Data Contract Compliance:
This module adheres to the data contracts defined in docs/data_contracts.md,
expecting NPZ files with keys: 'diffraction' (amplitude), 'objectGuess', 
'probeGuess', 'xcoords', 'ycoords'. Ground truth patches ('Y') are optional
and generated on-demand from objectGuess when not provided.
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
        from ptycho.diffsim import illuminate_and_diffract
        xcoords_start = xcoords
        ycoords_start = ycoords
        global_offsets, local_offsets, nn_indices = calculate_relative_coords(
                    xcoords, ycoords)

        Y_obj = get_image_patches(objectGuess, global_offsets, local_offsets) 
        Y_I = tf.math.abs(Y_obj)
        Y_phi = tf.math.angle(Y_obj)
        X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
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
    def generate_grouped_data(self, N, K = 4, nsamples = 1, dataset_path: Optional[str] = None, seed: Optional[int] = None):
        """
        Generate nearest-neighbor solution region grouping with efficient sampling.
        
        This method implements a "sample-then-group" strategy that first samples
        seed points from the dataset, then finds neighbors only for those seed points.
        This approach is highly efficient and eliminates the need for caching.
        
        **Efficient Sampling Strategy:**
        1. Randomly samples nsamples seed points from the dataset
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
        """
        gridsize = params.get('gridsize')
        if gridsize is None:
            gridsize = 1
        
        # Unified efficient logic for all gridsize values
        C = gridsize ** 2  # Number of coordinates per solution region
        
        print('DEBUG:', f'nsamples: {nsamples}, gridsize: {gridsize} (using efficient sample-then-group strategy)')
        logging.info(f"Using efficient sampling strategy for gridsize={gridsize}")
        
        # Use the new efficient method for all cases
        selected_groups = self._generate_groups_efficiently(
            nsamples=nsamples, 
            K=K, 
            C=C, 
            seed=seed
        )
        
        logging.info(f"Generated {len(selected_groups)} groups efficiently")
        
        # Generate the final dataset from the selected groups
        return self._generate_dataset_from_groups(selected_groups, N, K)

    def _generate_dataset_from_groups(self, selected_groups: np.ndarray, N: int, K: int) -> dict:
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
            Y4d_nn = get_image_patches(self.objectGuess, coords_offsets, coords_relative)
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
        
        # Apply normalization
        X_full = normalize_data(dset, N)
        dset['X_full'] = X_full
        print('neighbor-sampled diffraction shape', X_full.shape)
        
        return dset


    def _generate_groups_efficiently(self, nsamples: int, K: int, C: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Efficiently generate coordinate groups using a "sample-then-group" strategy.
        
        This method first samples seed points from the dataset, then finds neighbors
        only for those seed points, drastically reducing computation and memory usage
        compared to the "group-then-sample" approach.
        
        Args:
            nsamples: Number of groups to generate
            K: Number of nearest neighbors to consider (including self)
            C: Number of coordinates per group (typically gridsize^2)
            seed: Random seed for reproducibility (optional)
            
        Returns:
            np.ndarray: Array of group indices with shape (nsamples, C)
            
        Raises:
            ValueError: If K < C or if dataset is too small
        """
        try:
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
            
            n_points = len(self.xcoords)
            logging.info(f"Generating {nsamples} groups efficiently from {n_points} points (K={K}, C={C})")
            
            # Validate inputs
            if n_points < C:
                raise ValueError(f"Dataset has only {n_points} points but {C} coordinates per group requested.")
            
            if K < C:
                raise ValueError(f"K={K} must be >= C={C} (need at least C neighbors to form a group)")
            
            # Handle edge case: more samples requested than available points
            if nsamples > n_points:
                logging.warning(f"Requested {nsamples} groups but only {n_points} points available. Using all points as seeds.")
                n_samples_actual = n_points
            else:
                n_samples_actual = nsamples
            
            # Step 1: Sample seed points
            all_indices = np.arange(n_points)
            if n_samples_actual < n_points:
                seed_indices = np.random.choice(all_indices, size=n_samples_actual, replace=False)
                logging.info(f"Sampled {n_samples_actual} seed points from {n_points} total points")
            else:
                seed_indices = all_indices
                logging.info(f"Using all {n_points} points as seeds")
            
            # Step 2: Build KDTree for efficient neighbor search
            coords = np.column_stack([self.xcoords, self.ycoords])
            tree = cKDTree(coords)
            
            # Step 3: Find K nearest neighbors for each seed point
            seed_coords = coords[seed_indices]
            # Query K+1 neighbors (including self), then remove self
            distances, neighbor_indices = tree.query(seed_coords, k=min(K+1, n_points))
            
            # Step 4: Generate groups by selecting C coordinates from each seed's neighbors
            groups = np.zeros((n_samples_actual, C), dtype=np.int32)
            
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
            
            logging.info(f"Successfully generated {n_samples_actual} groups with shape {groups.shape}")
            return groups
            
        except Exception as e:
            logging.error(f"Failed to generate groups efficiently: {e}")
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


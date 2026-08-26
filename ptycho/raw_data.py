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
- `RawData.generate_grouped_data()`: Exact centered-nearest coordinate grouping

Key Algorithm - Centered-Nearest Grouping:
`RawData.generate_grouped_data` selects an exact set of unique centers (random
draw without replacement, or the first rows sequentially) and delegates index
planning to the backend-neutral centered planner in `ptycho.grouping`. Every
emitted group begins with its designated center; the remaining members are the
center's K nearest same-object candidates. Requests beyond the candidate pool
fail; grouping never oversamples or replaces.

Performance Characteristics:
- **O(nsamples * K)** planning via the centered planner (one tree per object
  partition), no cache files
- **Deterministic results** via optional seed parameter
- **Zero cache files** (eliminates disk I/O overhead)

Public Interface:
    `RawData.generate_grouped_data(N, K, nsamples, dataset_path, seed, sequential_sampling, gridsize, *, rng)`
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
- gridsize is an explicit argument on ``generate_grouped_data`` and
  ``from_simulation`` (explicit ``gridsize`` argument); group size C = gridsize².
- ``N`` and ``gridsize`` are explicit arguments on ``get_image_patches``.
- Caching behavior eliminated - no dependency on dataset_path for cache files

Data Contract Compliance:
Adheres to normative specs in `docs/specs/spec-ptycho-interfaces.md` and `docs/specs/spec-ptycho-core.md`.
Expected NPZ keys and dtypes:
- `xcoords (M,) float64`, `ycoords (M,) float64` — pixel coordinates on the object grid
- `diff3d (M, N, N) float32` — amplitude (sqrt of counts), not intensity
- `probeGuess (N, N) complex64` — probe in object pixel grid
- Optional: `scan_index (M,) int64` (defaults to zeros), `object_index (M,) int64`
  (independent-object bank membership, defaults to zeros), `objectGuess (H, W) complex64`,
  `xcoords_start (M,)`, `ycoords_start (M,)` (default to `xcoords`, `ycoords`)

Primary Consumers:
- ptycho.data_preprocessing (3 imports): Uses RawData for preprocessing workflows
- ptycho.loader (1 import): Converts RawData outputs to model-ready tensors
- ptycho.workflows.components (1 import): High-level workflow orchestration
"""
import numpy as np
from typing import Tuple, Optional
import os
import logging
from pathlib import Path
from ptycho import grouping
from ptycho.acquisition import canonicalize_identity_index

# Constants, # TODO cleanup / refactor
local_offset_sign = -1
key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'


class RawData:
    """Core data container for raw ptychographic scan data (NPZ-backed).

    Contract: docs/architecture_torch.md §Component Contracts.
    """
    #@debug
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess,
             scan_index, objectGuess = None, Y = None, norm_Y_I = None, metadata = None,
             object_index = None, probe_simulated = None,
             object_amplitude_scale = None, label = None,
             scale_contract_version = None, measurement_domain = None,
             experiment_id = None):
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
            probe_shape = tuple(probeGuess.shape)
            assert (
                len(probe_shape) == 2
                or (
                    len(probe_shape) == 3
                    and probe_shape[0] > 0
                    and probe_shape[1] == probe_shape[2]
                )
            ), (
                "Expected probeGuess to have shape (N, N) or (P, N, N), "
                f"got shape {probe_shape}"
            )
            if diff3d is not None:
                assert probe_shape[-2:] == tuple(diff3d.shape[1:]), (
                    "probeGuess spatial shape must match diff3d; "
                    f"got {probe_shape[-2:]} and {tuple(diff3d.shape[1:])}"
                )
            print(f"probeGuess shape: {probe_shape}")
        scan_index = canonicalize_identity_index(
            scan_index,
            name="scan_index",
            length=len(xcoords),
        )
        object_index = canonicalize_identity_index(
            object_index,
            name="object_index",
            length=len(xcoords),
        )
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
        self.object_index = object_index
        self.objectGuess = objectGuess
        # TODO validity checks
        self.Y = Y
        self.norm_Y_I = norm_Y_I
        self.metadata = metadata  # Store metadata from NPZ file
        self.probe_simulated = probe_simulated
        self.object_amplitude_scale = object_amplitude_scale
        self.label = label
        self.scale_contract_version = scale_contract_version
        self.measurement_domain = measurement_domain
        self.experiment_id = experiment_id
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
                 objectGuess, scan_index = None, gridsize: int = 1):
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

        Y_obj = get_image_patches(objectGuess, global_offsets, local_offsets, N=probeGuess.shape[0], gridsize=gridsize)
        Y_I = tf.math.abs(Y_obj)
        Y_phi = tf.math.angle(Y_obj)
        X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
        norm_Y_I = scale_nphotons(X)
        assert X.shape[-1] == 1, "gridsize must be set to one when simulating in this mode"
        # TODO RawData should have a method for generating the illuminated ground truth object
        return RawData(xcoords, ycoords, xcoords_start, ycoords_start, tf.squeeze(X).numpy(),
                       probeGuess, scan_index, objectGuess,
                       Y = np.asarray(Y_obj),
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
        arrays = {
            "xcoords": self.xcoords,
            "ycoords": self.ycoords,
            "xcoords_start": self.xcoords_start,
            "ycoords_start": self.ycoords_start,
            "diff3d": self.diff3d,
            "probeGuess": self.probeGuess,
            "scan_index": self.scan_index,
            "object_index": self.object_index,
        }
        if self.objectGuess is not None:
            arrays["objectGuess"] = self.objectGuess
        if self.Y is not None:
            arrays["Y"] = self.Y
        np.savez(file_path, **arrays)

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
        from ptycho.acquisition import decode_acquisition
        from ptycho.metadata import MetadataManager

        record = decode_acquisition(train_data_file_path)
        metadata = record.metadata
        
        # Validate configuration if requested
        if validate_config and metadata and current_config:
            warnings_list = MetadataManager.validate_parameters(metadata, current_config)
            for warning in warnings_list:
                import logging
                logging.warning(f"Parameter mismatch: {warning}")
        
        train_raw_data = RawData(
            xcoords=record.xcoords,
            ycoords=record.ycoords,
            xcoords_start=record.xcoords_start,
            ycoords_start=record.ycoords_start,
            diff3d=record.diff3d,
            probeGuess=record.probeGuess,
            objectGuess=record.objectGuess,
            scan_index=record.scan_index,
            Y=record.Y,
            metadata=metadata,
            object_index=record.object_index,
            probe_simulated=record.probe_simulated,
            object_amplitude_scale=record.object_amplitude_scale,
            label=record.label,
            scale_contract_version=record.scale_contract_version,
            measurement_domain=record.measurement_domain,
            experiment_id=record.experiment_id,
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
    def generate_grouped_data(self, N, K = 4, nsamples = 1, dataset_path: Optional[str] = None, seed: Optional[int] = None, sequential_sampling: bool = False, gridsize: Optional[int] = None, *, rng: Optional[np.random.Generator] = None):
        """
        Generate centered nearest-neighbor solution region grouping.

        This method selects an exact set of unique centers from the already-
        selected ``RawData`` rows and delegates index planning to the
        backend-neutral centered planner (``ptycho.grouping.plan_nearest_groups``).
        Every emitted group begins with its designated center in column zero;
        the remaining members come from that center's K nearest same-object
        candidates.  ``RawData`` retains grouped-dictionary materialization,
        normalization, and probe validation.

        Center selection:
        1. Random (default): one split-local generator draws ``nsamples`` unique
           centers without replacement and is then passed, already advanced, to
           the planner for neighbor selection.
        2. Sequential: the first ``nsamples`` rows are the centers; neighbors use
           the fixed seed-0 local generator.
        A request larger than the candidate pool fails; it is never satisfied by
        replacement or oversampling.

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
            gridsize (int): Grid size for patch grouping. Required.
            rng (np.random.Generator, optional): Caller-owned random generator. Mutually
                                                exclusive with seed.

        Returns:
            dict: Dictionary containing grouped data with keys:
                - 'diffraction': 4D array of diffraction patterns
                - 'Y': 4D array of ground truth patches (if available)
                - 'coords_offsets', 'coords_relative': Coordinate information
                - 'nn_indices': Selected neighbor indices; column zero is the
                  exact selected center of each group
                - 'X_full': Normalized diffraction data
                - Additional coordinate and metadata arrays

        Raises:
            ValueError: If more centers are requested than candidate rows, if
                seed and rng are both supplied, or if a candidate pool is too
                small for the requested grouping.

        Note:
        ``gridsize`` is required. A missing value now raises instead of silently
        defaulting to the wrong group shape.

        See: docs/debugging/TROUBLESHOOTING.md#shape-mismatch-errors
        """
        if gridsize is None:
            raise ValueError(
                "gridsize is required for generate_grouped_data; pass it "
                "explicitly (e.g. gridsize=config.model.gridsize)."
            )
        C = gridsize ** 2
        size = len(self.xcoords)
        if seed is not None and rng is not None:
            raise ValueError("seed and rng are mutually exclusive")
        if nsamples > size:
            raise ValueError(
                f"requested {nsamples} unique centers from only {size} candidates"
            )
        candidate_indices = np.arange(size, dtype=np.int64)
        if sequential_sampling:
            centers = candidate_indices[:nsamples]
            local_rng = np.random.default_rng(0)
        else:
            local_rng = rng if rng is not None else np.random.default_rng(seed)
            drawn_centers = local_rng.choice(
                candidate_indices, nsamples, replace=False
            )
            # Preserve canonical all-row order (and exact C1 arrays), while still
            # consuming the one split-local generator for random center selection.
            centers = candidate_indices if nsamples == size else drawn_centers
        source_indices = (
            self.sample_indices
            if self.sample_indices is not None
            else candidate_indices
        )
        plan = grouping.plan_nearest_groups(
            self.xcoords,
            self.ycoords,
            center_indices=centers,
            candidate_indices=candidate_indices,
            group_size=C,
            neighbor_count=K,
            object_index=self.object_index,
            experiment_id=self.experiment_id,
            source_indices=source_indices,
            rng=local_rng,
        )
        dtype = np.int64 if C == 1 else np.int32
        selected_groups = np.array(plan.neighbor_indices, dtype=dtype, copy=True)
        grouped = self._generate_dataset_from_groups(
            selected_groups, N, K, gridsize
        )
        grouped["object_index"] = np.array(
            plan.object_index, dtype=np.int64, copy=True
        )
        grouped["experiment_id"] = np.array(
            plan.experiment_id, dtype=np.int64, copy=True
        )
        return grouped

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
            Y_indexed = self.Y[nn_indices]
            # Handle case where Y has extra channel dimension and indexing adds extra dim
            # Expected input: (n_groups, C, H, W, 1) -> output: (n_groups, H, W, C)
            if Y_indexed.ndim == 5:
                Y4d_nn = np.transpose(Y_indexed, [0, 2, 3, 1, 4])[:, :, :, :, 0]
            else:
                # Standard 4D case: (n_groups, H, W, C) -> (n_groups, H, W, C)
                Y4d_nn = np.transpose(Y_indexed, [0, 2, 3, 1])
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
        gt_image (array): Ground truth image (numpy array or torch tensor).
        global_offsets (array): Global offset array.
        local_offsets (array): Local offset array.
        N (int): Patch size. Required.
        gridsize (int): Grid size. Required.

    Returns:
        np.ndarray: Image patches in channel format (B, N, N, gridsize**2),
        dtype complex64.
    """
    from ptycho_torch import pad_translate

    if N is None or gridsize is None:
        raise ValueError(
            "N and gridsize are required for get_image_patches; pass them "
            "explicitly."
        )
    B = global_offsets.shape[0]
    c = gridsize**2

    # Accept a torch tensor (torch rail may hand us one); the TF rail passes numpy.
    if hasattr(gt_image, 'detach'):
        gt_image = gt_image.detach().cpu().numpy()
    else:
        gt_image = np.asarray(gt_image)

    # Pad the ground truth image once
    gt_padded = pad_translate.pad(gt_image[None, ..., None], N // 2)

    # Calculate the combined offsets by adding global and local offsets
    offsets_c = (global_offsets + local_offsets).astype(np.float32)
    offsets_f = np.transpose(offsets_c, [0, 3, 1, 2]).reshape(-1, 1, 2, 1)

    # Create a canvas to store the extracted patches
    canvas = np.zeros((B, N, N, c), dtype=np.complex64)

    # Iterate over the combined offsets and extract patches one by one
    for i in range(B * c):
        offset = -offsets_f[i, :, :, 0]
        translated_patch = pad_translate.translate(gt_padded, offset)
        canvas[i // c, :, :, i % c] = np.asarray(translated_patch)[0, :N, :N, 0]

    return canvas

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
    # NORMALIZE-DATA-UINT16-001: cast to float64 before squaring to avoid uint16 overflow
    X_full_norm = np.float32(np.sqrt(
            ((N / 2)**2) / np.mean(np.sum(np.square(dset['diffraction'].astype(np.float64)), axis=(1, 2)))
            ))
    # Force float32 output: under numpy>=2 (NEP 50) a float64 scalar would
    # promote the product to float64, violating the float32 contract
    # (specs/data_contracts.md; enforced by ptycho_torch/data_container_bridge.py).
    return X_full_norm * X_full.astype(np.float32, copy=False)

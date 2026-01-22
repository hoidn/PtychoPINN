"""
NumPy-to-TensorFlow conversion layer for ptychographic neural networks.

This module serves as the critical NumPy→TensorFlow bridge in PtychoPINN's data pipeline,
transforming grouped NumPy arrays from ptycho.raw_data into GPU-ready TensorFlow tensors
for neural network training and inference. As the most heavily used data pipeline module
(consumed by 9+ modules), it handles the final dtype conversion, tensor shaping, and
memory layout optimization required for efficient TensorFlow computation.

Architecture Role:
    NPZ files → raw_data.py (RawData/NumPy) → loader.py (TensorFlow tensors) → model.py

Primary Components:
    - PtychoDataContainer: Core data container holding model-ready TensorFlow tensors
      including diffraction patterns (X), ground truth patches (Y_I, Y_phi), 
      coordinates, and probe functions
    - load(): Main entry point that transforms grouped data into PtychoDataContainer
      via callback mechanism, handling train/test splits and tensor conversion
    - PtychoDataset: Simple wrapper for train/test data pairs

Key Tensor Conversion Features:
    - NumPy float64/complex128 → TensorFlow float32/complex64 dtype optimization
    - Multi-channel tensor reshaping for gridsize > 1 configurations  
    - Automatic train/test data splitting with consistent tensor slicing
    - Complex tensor decomposition (amplitude/phase separation) and recomposition
    - Memory-efficient lazy evaluation via callback pattern
    - Comprehensive tensor statistics and shape validation

Public Interface:
    load(cb, probeGuess, which, create_split) -> PtychoDataContainer
        cb: Callback returning grouped data dictionary from raw_data
        probeGuess: Initial probe function as TensorFlow tensor
        which: 'train' or 'test' for data splitting
        create_split: Boolean to enable train/test splitting
        
    PtychoDataContainer.from_raw_data_without_pc(xcoords, ycoords, diff3d, ...)
        Static constructor combining raw_data grouping with tensor loading
        
    split_data(X_full, coords_nominal, coords_true, train_frac, which)
        Utility for fraction-based data splitting

Usage Example:
    # Complete pipeline: raw data → grouped data → model tensors
    from ptycho.raw_data import RawData
    from ptycho.loader import load, PtychoDataContainer
    
    # Create raw data object
    raw_data = RawData.from_coords_without_pc(
        xcoords, ycoords, diff3d, probe, scan_idx
    )
    
    # Generate grouped data and convert to tensors
    def data_callback():
        gridsize = params.get('gridsize', 1)
        return raw_data.generate_grouped_data(N=64, K=7, gridsize=gridsize)
    
    train_container = load(data_callback, probe_tensor, 'train', True)
    test_container = load(data_callback, probe_tensor, 'test', True)
    
    # Access model-ready tensors
    X_train = train_container.X        # Diffraction patterns
    Y_train = train_container.Y        # Complex ground truth
    coords = train_container.coords    # Scan coordinates
    
    # Export for later use
    train_container.to_npz("model_ready_train.npz")

TensorFlow Integration:
    - All NumPy arrays undergo dtype conversion: float64→float32, complex128→complex64
    - Tensor shapes are validated for TensorFlow compatibility and GPU efficiency
    - Multi-channel dimensions preserved for gridsize > 1 neural network architectures
    - Missing ground truth handled with properly-shaped complex dummy tensors (MAE losses should be disabled in this case)
    - Seamless integration with tf.data.Dataset and Keras model.fit() workflows
    - Primary consumers: ptycho.model, ptycho.train_pinn, ptycho.workflows.components

Coordinate Semantics:
    - Offsets use channel format `(B, 1, 2, C)` with axis order `[x, y]`
    - Channel index `c` maps to `(row, col)` via row‑major: `row=c//gridsize`, `col=c%gridsize`
"""

import numpy as np
import tensorflow as tf
from typing import Callable

from .params import params, get
from .autotest.debug import debug
from . import diffsim as datasets
from . import tf_helper as hh
from .raw_data import RawData, key_coords_offsets, key_coords_relative 

class PtychoDataset:
    @debug
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

class PtychoDataContainer:
    """
    Lazy-loading TensorFlow tensor container for model-ready ptychographic data.

    This container stores data as NumPy arrays internally and converts to TensorFlow
    tensors only on demand (lazy loading). This prevents GPU memory exhaustion when
    working with large datasets (20k+ images).

    LAZY LOADING BEHAVIOR:
        Data is stored as NumPy arrays in private attributes (e.g., _X_np).
        Accessing properties (e.g., .X) converts to TensorFlow tensor on first access
        and caches the result. For large datasets, use as_tf_dataset(batch_size)
        instead to stream data in batches without loading everything into GPU memory.

    Property Attributes (lazy TensorFlow conversion):
        X: Diffraction patterns — tf.float32, shape (B, N, N, C)
           where C = gridsize^2 for multi‑channel configurations (channel = scan position)
        Y_I: Ground truth amplitude patches — tf.float32, shape (B, N, N, C)
        Y_phi: Ground truth phase patches — tf.float32, shape (B, N, N, C)
        Y: Combined complex ground truth — tf.complex64, shape (B, N, N, C)
        coords_nominal: Scan coordinates (channel format) — tf.float32, shape (B, 1, 2, C)
        coords_true: True scan coordinates (channel format) — tf.float32, shape (B, 1, 2, C)
        probe: Probe function — tf.complex64, shape (N, N)

    NumPy Attributes (preserved from raw_data grouping):
        norm_Y_I: Normalization factors for amplitude
        YY_full: Full object reconstruction (if available)
        nn_indices: Nearest neighbor indices for patch grouping
        global_offsets: Global coordinate offsets
        local_offsets: Local coordinate offsets within patches

    The container composes complex ground truth (Y = Y_I * exp(1j · Y_phi)) lazily
    and provides comprehensive debug representations showing tensor statistics.

    See: docs/findings.md PINN-CHUNKED-001 for the OOM blocker this addresses.
    """
    @debug
    def __init__(self, X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, nn_indices, global_offsets, local_offsets, probeGuess):
        # Store as numpy, convert lazily on property access
        # Handle both tensor and array inputs
        self._X_np = X.numpy() if tf.is_tensor(X) else X
        self._Y_I_np = Y_I.numpy() if tf.is_tensor(Y_I) else Y_I
        self._Y_phi_np = Y_phi.numpy() if tf.is_tensor(Y_phi) else Y_phi
        self._coords_nominal_np = coords_nominal.numpy() if tf.is_tensor(coords_nominal) else coords_nominal
        self._coords_true_np = coords_true.numpy() if tf.is_tensor(coords_true) else coords_true
        self._probe_np = probeGuess.numpy() if tf.is_tensor(probeGuess) else probeGuess

        # Lazy cache for tensorified data
        self._tensor_cache = {}

        # These remain as-is (NumPy only attributes)
        self.norm_Y_I = norm_Y_I
        self.YY_full = YY_full
        self.nn_indices = nn_indices
        self.global_offsets = global_offsets
        self.local_offsets = local_offsets

    @property
    def X(self):
        """Diffraction patterns — tf.float32, shape (B, N, N, C).

        WARNING: Accessing this property loads the full tensor into GPU memory.
        For large datasets, use as_tf_dataset() instead.
        """
        if 'X' not in self._tensor_cache:
            self._tensor_cache['X'] = tf.convert_to_tensor(self._X_np, dtype=tf.float32)
        return self._tensor_cache['X']

    @property
    def Y_I(self):
        """Ground truth amplitude — tf.float32, shape (B, N, N, C)."""
        if 'Y_I' not in self._tensor_cache:
            self._tensor_cache['Y_I'] = tf.convert_to_tensor(self._Y_I_np, dtype=tf.float32)
        return self._tensor_cache['Y_I']

    @property
    def Y_phi(self):
        """Ground truth phase — tf.float32, shape (B, N, N, C)."""
        if 'Y_phi' not in self._tensor_cache:
            self._tensor_cache['Y_phi'] = tf.convert_to_tensor(self._Y_phi_np, dtype=tf.float32)
        return self._tensor_cache['Y_phi']

    @property
    def Y(self):
        """Combined complex ground truth — tf.complex64, shape (B, N, N, C)."""
        if 'Y' not in self._tensor_cache:
            from .tf_helper import combine_complex
            self._tensor_cache['Y'] = combine_complex(self.Y_I, self.Y_phi)
        return self._tensor_cache['Y']

    @property
    def coords_nominal(self):
        """Scan coordinates (channel format) — tf.float32, shape (B, 1, 2, C)."""
        if 'coords_nominal' not in self._tensor_cache:
            self._tensor_cache['coords_nominal'] = tf.convert_to_tensor(
                self._coords_nominal_np, dtype=tf.float32
            )
        return self._tensor_cache['coords_nominal']

    @property
    def coords(self):
        """Alias for coords_nominal (backward compatibility)."""
        return self.coords_nominal

    @property
    def coords_true(self):
        """True scan coordinates — tf.float32, shape (B, 1, 2, C)."""
        if 'coords_true' not in self._tensor_cache:
            self._tensor_cache['coords_true'] = tf.convert_to_tensor(
                self._coords_true_np, dtype=tf.float32
            )
        return self._tensor_cache['coords_true']

    @property
    def probe(self):
        """Probe function — tf.complex64, shape (N, N)."""
        if 'probe' not in self._tensor_cache:
            self._tensor_cache['probe'] = tf.convert_to_tensor(
                self._probe_np, dtype=tf.complex64
            )
        return self._tensor_cache['probe']

    @debug
    def __repr__(self):
        """Debug representation using underlying NumPy arrays (avoids GPU tensorification)."""
        repr_str = '<PtychoDataContainer'
        # Map public names to private numpy arrays (avoid triggering lazy tensor conversion)
        np_attr_map = {
            'X': '_X_np', 'Y_I': '_Y_I_np', 'Y_phi': '_Y_phi_np',
            'coords_nominal': '_coords_nominal_np', 'coords_true': '_coords_true_np',
            'probe': '_probe_np'
        }
        # Direct numpy attributes
        direct_attrs = ['norm_Y_I', 'YY_full', 'nn_indices', 'global_offsets', 'local_offsets']

        for attr_name in ['X', 'Y_I', 'Y_phi', 'coords_nominal', 'coords_true', 'probe']:
            np_name = np_attr_map.get(attr_name, attr_name)
            attr = getattr(self, np_name, None)
            if attr is not None:
                if np.iscomplexobj(attr):
                    repr_str += f' {attr_name}={attr.shape} mean_amplitude={np.mean(np.abs(attr)):.3f}'
                else:
                    repr_str += f' {attr_name}={attr.shape} mean={attr.mean():.3f}'

        for attr_name in direct_attrs:
            attr = getattr(self, attr_name, None)
            if attr is not None:
                if isinstance(attr, np.ndarray):
                    repr_str += f' {attr_name}={attr.shape}'
                else:
                    repr_str += f' {attr_name}=<scalar>'
        repr_str += '>'
        return repr_str

    def __len__(self):
        """Return number of samples in the container."""
        return len(self._X_np)

    def as_tf_dataset(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create a tf.data.Dataset for memory-efficient batched access.

        This is the preferred method for large datasets as it streams data
        in batches rather than loading everything into GPU memory.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset (default True)

        Returns:
            tf.data.Dataset yielding (inputs, outputs) tuples compatible
            with model.fit()
        """
        from . import params as p
        from . import tf_helper as hh

        n_samples = len(self._X_np)
        intensity_scale = p.get('intensity_scale')

        def generator():
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                # Convert batch to tensors
                X_batch = tf.convert_to_tensor(
                    self._X_np[batch_idx], dtype=tf.float32
                )
                coords_batch = tf.convert_to_tensor(
                    self._coords_nominal_np[batch_idx], dtype=tf.float32
                )
                Y_I_batch = tf.convert_to_tensor(
                    self._Y_I_np[batch_idx], dtype=tf.float32
                )

                # Prepare inputs: (X * intensity_scale, coords) as tuple
                inputs = (X_batch * intensity_scale, coords_batch)

                # Prepare outputs: (centered_Y_I[:,:,:,:1], X*s, (X*s)^2) as tuple
                Y_I_centered = hh.center_channels(Y_I_batch, coords_batch)[:, :, :, :1]
                X_scaled = intensity_scale * X_batch
                outputs = (Y_I_centered, X_scaled, X_scaled ** 2)

                yield inputs, outputs

        # Define output signature for tf.data.Dataset
        N = self._X_np.shape[1]
        C = self._X_np.shape[3]

        output_signature = (
            (
                tf.TensorSpec(shape=(None, N, N, C), dtype=tf.float32),  # X
                tf.TensorSpec(shape=(None, 1, 2, C), dtype=tf.float32),  # coords
            ),
            (
                tf.TensorSpec(shape=(None, N, N, 1), dtype=tf.float32),  # Y_I centered
                tf.TensorSpec(shape=(None, N, N, C), dtype=tf.float32),  # X*s
                tf.TensorSpec(shape=(None, N, N, C), dtype=tf.float32),  # (X*s)^2
            )
        )

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    @staticmethod
    @debug
    def from_raw_data_without_pc(xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess=None, N=None, K=7, nsamples=1):
        """
        Static method constructor that composes a call to RawData.from_coords_without_pc() and loader.load,
        then initializes attributes.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            diff3d (np.ndarray): diffraction patterns.
            probeGuess (np.ndarray): initial guess of the probe function.
            scan_index (np.ndarray): array indicating the scan index for each diffraction pattern.
            objectGuess (np.ndarray, optional): initial guess of the object. Defaults to None.
            N (int, optional): The size of the image. Defaults to None.
            K (int, optional): The number of nearest neighbors. Defaults to 7.
            nsamples (int, optional): The number of samples. Defaults to 1.

        Returns:
            PtychoDataContainer: An instance of the PtychoDataContainer class.
        """
        from . import params as cfg
        if N is None:
            N = cfg.get('N')
        train_raw = RawData.from_coords_without_pc(xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess)
        
        gridsize = cfg.get('gridsize', 1)
        dset_train = train_raw.generate_grouped_data(N, K=K, nsamples=nsamples, gridsize=gridsize)

        # Use loader.load() to handle the conversion to PtychoData
        return load(lambda: dset_train, probeGuess, which=None, create_split=False)

    #@debug
    def to_npz(self, file_path: str) -> None:
        """
        Write the underlying arrays to an npz file.

        Uses internal NumPy arrays directly (avoids GPU tensorification).

        Args:
            file_path (str): Path to the output npz file.
        """
        np.savez(
            file_path,
            X=self._X_np,
            Y_I=self._Y_I_np,
            Y_phi=self._Y_phi_np,
            norm_Y_I=self.norm_Y_I,
            YY_full=self.YY_full,
            coords_nominal=self._coords_nominal_np,
            coords_true=self._coords_true_np,
            nn_indices=self.nn_indices,
            global_offsets=self.global_offsets,
            local_offsets=self.local_offsets,
            probe=self._probe_np
        )

    # TODO is this deprecated, given the above method to_npz()?


@debug
def split_data(X_full, coords_nominal, coords_true, train_frac, which):
    """
    Splits the data into training and testing sets based on the specified fraction.

    Args:
        X_full (np.ndarray): The full dataset to be split.
        coords_nominal (np.ndarray): The nominal coordinates associated with the dataset.
        coords_true (np.ndarray): The true coordinates associated with the dataset.
        train_frac (float): The fraction of the dataset to be used for training.
        which (str): A string indicating whether to return the 'train' or 'test' split.

    Returns:
        tuple: A tuple containing the split data and coordinates.
    """
    n_train = int(len(X_full) * train_frac)
    if which == 'train':
        return X_full[:n_train], coords_nominal[:n_train], coords_true[:n_train]
    elif which == 'test':
        return X_full[n_train:], coords_nominal[n_train:], coords_true[n_train:]
    else:
        raise ValueError("Invalid split type specified: must be 'train' or 'test'.")

@debug
def split_tensor(tensor, frac, which='test'):
    """
    Splits a tensor into training and test portions based on the specified fraction.

    :param tensor: The tensor to split.
    :param frac: Fraction of the data to be used for training.
    :param which: Specifies whether to return the training ('train') or test ('test') portion.
    :return: The appropriate portion of the tensor based on the specified fraction and 'which' parameter.
    """
    n_train = int(len(tensor) * frac)
    return tensor[:n_train] if which == 'train' else tensor[n_train:]

@debug
def load(cb: Callable, probeGuess: tf.Tensor, which: str, create_split: bool) -> PtychoDataContainer:
    """
    Convert grouped NumPy data to a lazy-loading PtychoDataContainer.

    This function takes a callback that returns grouped data (as produced by raw_data.py)
    and creates a PtychoDataContainer that stores data as NumPy arrays internally.
    TensorFlow tensor conversion is deferred until property access (lazy loading).

    Args:
        cb: Callback function that returns grouped data dictionary from raw_data.
            The callback pattern allows lazy evaluation - data grouping only occurs
            when needed, which is crucial for memory efficiency with large datasets.
            Expected return: dict with keys 'X_full', 'Y', 'objectGuess', coordinate
            keys, and metadata from raw_data.generate_grouped_data().
        probeGuess: Initial probe function as TensorFlow complex64 tensor or NumPy array, shape (N, N)
        which: Data split selector - 'train' or 'test' (only used if create_split=True)
        create_split: If True, expects cb() to return (data_dict, train_fraction) tuple
                     and applies fraction-based splitting. If False, uses full dataset.

    Returns:
        PtychoDataContainer with NumPy arrays stored internally. TensorFlow tensors
        are created lazily on property access:
        - Diffraction data (X) -> tf.float32 on access
        - Ground truth patches (Y_I, Y_phi) -> tf.float32 on access
        - Complex ground truth (Y) -> tf.complex64 on access
        - Coordinates -> tf.float32 on access

    Notes:
        - Data is stored as NumPy arrays to prevent GPU memory exhaustion (PINN-CHUNKED-001)
        - Tensor conversion happens lazily on first property access
        - For large datasets, use container.as_tf_dataset(batch_size) instead of
          accessing .X, .Y directly to stream data in batches
        - Preserves multi-channel dimensions for gridsize > 1 configurations
    """
    train_frac = None  # Initialize to avoid possibly unbound warning

    if create_split:
        dset, train_frac = cb()
    else:
        dset = cb()

    X_full = dset['X_full']  # This is already in the correct multi-channel format.
    global_offsets = dset[key_coords_offsets]

    coords_nominal = dset[key_coords_relative]
    coords_true = dset[key_coords_relative]

    # Correctly handle splitting for both X and Y
    if create_split:
        global_offsets = split_tensor(global_offsets, train_frac, which)
        X_full_split, coords_nominal, coords_true = split_data(X_full, coords_nominal, coords_true, train_frac, which)
    else:
        X_full_split = X_full

    # Pass NumPy arrays directly — container will convert lazily on property access
    # Ensure correct dtype for NumPy storage
    X = X_full_split.astype(np.float32) if not np.issubdtype(X_full_split.dtype, np.float32) else X_full_split
    coords_nominal = coords_nominal.astype(np.float32) if not np.issubdtype(coords_nominal.dtype, np.float32) else coords_nominal
    coords_true = coords_true.astype(np.float32) if not np.issubdtype(coords_true.dtype, np.float32) else coords_true

    # Handle the Y array (ground truth patches)
    if dset['Y'] is None:
        # If Y is missing, create a placeholder with the same multi-channel shape as X.
        Y_I = np.ones(X.shape, dtype=np.float32)
        Y_phi = np.zeros(X.shape, dtype=np.float32)
        print("loader: setting dummy Y ground truth with correct channel shape.")
    else:
        Y_full = dset['Y']
        # CRITICAL: Apply the same split to Y as was applied to X
        if create_split:
            Y_split, _, _ = split_data(Y_full, coords_nominal, coords_true, train_frac, which)
        else:
            Y_split = Y_full
        # Extract amplitude and phase as NumPy arrays
        Y_I = np.abs(Y_split).astype(np.float32)
        Y_phi = np.angle(Y_split).astype(np.float32)
        print("loader: using provided ground truth patches.")

    # Final validation check (using NumPy shapes)
    if X.shape[-1] != Y_I.shape[-1]:
        raise ValueError(f"Channel mismatch between X ({X.shape[-1]}) and Y ({Y_I.shape[-1]})")

    # scale_nphotons expects a tensor, so temporarily convert X
    norm_Y_I = datasets.scale_nphotons(tf.convert_to_tensor(X, dtype=tf.float32))

    YY_full = None  # This is a placeholder

    # Create the container with NumPy arrays (lazy tensor conversion)
    container = PtychoDataContainer(X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true,
                                    dset['nn_indices'], dset['coords_offsets'], dset['coords_relative'], probeGuess)
    print('INFO:', which)
    print(container)
    return container

#@debug
def normalize_data(dset: dict, N: int) -> np.ndarray:
    # TODO this should be baked into the model pipeline. If we can
    # assume consistent normalization, we can get rid of intensity_scale
    # as a model parameter since the post normalization average L2 norm
    # will be fixed. Normalizing in the model's dataloader will make
    # things more self-contained and avoid the need for separately
    # scaling simulated datasets. While we're at it we should get rid of
    # all the unecessary multiiplying and dividing by intensity_scale.
    # As long as nphotons is a dataset-level attribute (i.e. an attribute of RawData 
    # and PtychoDataContainer), nothing is lost
    # by keeping the diffraction in normalized format everywhere except
    # before the Poisson NLL calculation in model.py.

    # Images are amplitude, not intensity
    X_full = dset['diffraction']
    X_full_norm = np.sqrt(
            ((N / 2)**2) / np.mean(tf.reduce_sum(dset['diffraction']**2, axis=[1, 2]))
            )
    #print('X NORM', X_full_norm)
    return X_full_norm * X_full

#@debug
def crop(arr2d, size):
    N, M = arr2d.shape
    return arr2d[N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2]

@debug
def get_gt_patch(offset, N, gt_image):
    from . import tf_helper as hh
    return crop(
        hh.translate(gt_image, offset),
        N // 2)

def load_xpp_npz(file_path, train_size=512):
    """
    Load ptychography data from a file and return RawData objects.

    Args:
        file_path (str, optional): Path to the data file. Defaults to the package resource 'datasets/Run1084_recon3_postPC_shrunk_3.npz'.
        train_size (int, optional): Number of data points to include in the training set. Defaults to 512.

    Returns:
        tuple: A tuple containing two RawData objects:
            - ptycho_data: RawData object containing the full dataset.
            - ptycho_data_train: RawData object containing a subset of the data for training.
    """
    # Load data from file
    data = np.load(file_path)

    # Extract required arrays from loaded data
    xcoords = data['xcoords']
    ycoords = data['ycoords']
    xcoords_start = data['xcoords_start']
    ycoords_start = data['ycoords_start']
    diff3d = data['diffraction']#np.transpose(data['diffraction'], [2, 0, 1])
    probeGuess = data['probeGuess']
    objectGuess = data['objectGuess']

    # Create scan_index array
    scan_index = np.zeros(diff3d.shape[0], dtype=int)

    # Create RawData object for the full dataset
    ptycho_data = RawData(xcoords, ycoords, xcoords_start, ycoords_start,
                                 diff3d, probeGuess, scan_index, objectGuess=objectGuess)

    # Create RawData object for the training subset
    ptycho_data_train = RawData(xcoords[:train_size], ycoords[:train_size],
                                       xcoords_start[:train_size], ycoords_start[:train_size],
                                       diff3d[:train_size], probeGuess,
                                       scan_index[:train_size], objectGuess=objectGuess)

    return ptycho_data, ptycho_data_train, data


def compute_dataset_intensity_stats(
    diff3d: np.ndarray,
    is_normalized: bool = False,
    intensity_scale: float = None,
) -> dict:
    """Compute dataset intensity statistics for normalization scale derivation.

    Per specs/spec-ptycho-core.md §Normalization Invariants:
    - Dataset-derived scale: `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`
    - This function computes E_batch[Σ_xy |Ψ|²] from raw diffraction data.

    Args:
        diff3d: Raw diffraction patterns, shape (M, N, N) or (M, N, N, C),
                containing amplitude (sqrt of counts).
        is_normalized: If True, data has been normalized by intensity_scale.
                      When True, intensity_scale must be provided to back-compute raw stats.
        intensity_scale: The scale factor used for normalization. Required when is_normalized=True.
                        Stats will be back-computed: sum(X_norm²) / s² = sum(X_raw²)

    Returns:
        dict with:
            batch_mean_sum_intensity: E_batch[Σ_xy |Ψ|²] - average total intensity per sample
            n_samples: Number of samples in the batch
            spec_reference: Citation to normative spec section
    """
    diff_arr = np.asarray(diff3d, dtype=np.float64)
    # Determine spatial axes based on shape
    ndim = diff_arr.ndim
    if ndim == 3:
        # shape (M, N, N) - sum over axes 1, 2
        sum_axes = (1, 2)
    elif ndim == 4:
        # shape (M, N, N, C) - sum over axes 1, 2, 3
        sum_axes = (1, 2, 3)
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {diff_arr.shape}")

    # diff3d is amplitude (sqrt of counts); |Ψ|² = diff3d²
    # Σ_xy |Ψ|² sums over spatial dimensions
    intensity_per_sample = np.sum(diff_arr ** 2, axis=sum_axes)  # shape (M,)

    # If data is normalized, back-compute raw stats
    if is_normalized:
        if intensity_scale is None:
            raise ValueError("intensity_scale must be provided when is_normalized=True")
        # X_norm = s * X_raw, so X_raw = X_norm / s
        # sum(X_raw²) = sum(X_norm²) / s²
        intensity_per_sample = intensity_per_sample / (intensity_scale ** 2)

    batch_mean = float(np.mean(intensity_per_sample))
    return {
        "batch_mean_sum_intensity": batch_mean,
        "n_samples": int(diff_arr.shape[0]),
        "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants",
    }

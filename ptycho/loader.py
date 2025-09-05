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
    - Missing ground truth handled with properly-shaped complex dummy tensors
    - Seamless integration with tf.data.Dataset and Keras model.fit() workflows
    - Primary consumers: ptycho.model, ptycho.train_pinn, ptycho.workflows.components
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
    TensorFlow tensor container for model-ready ptychographic data.
    
    This container holds the final NumPy->TensorFlow converted data structures
    ready for neural network training and inference. All tensor attributes use
    appropriate TensorFlow dtypes for efficient GPU computation.
    
    Tensor Attributes:
        X: Diffraction patterns - tf.float32, shape (n_images, N, N, n_channels)
           where n_channels = gridsize^2 for multi-channel configurations
        Y_I: Ground truth amplitude patches - tf.float32, shape (n_images, patch_size, patch_size, n_channels)
        Y_phi: Ground truth phase patches - tf.float32, shape (n_images, patch_size, patch_size, n_channels)
        Y: Combined complex ground truth - tf.complex64, shape (n_images, patch_size, patch_size, n_channels)
        coords_nominal: Scan coordinates - tf.float32, shape (n_images, 2)
        coords_true: True scan coordinates - tf.float32, shape (n_images, 2)
        probe: Probe function - tf.complex64, shape (N, N)
        
    NumPy Attributes (preserved from raw_data grouping):
        norm_Y_I: Normalization factors for amplitude
        YY_full: Full object reconstruction (if available)
        nn_indices: Nearest neighbor indices for patch grouping
        global_offsets: Global coordinate offsets
        local_offsets: Local coordinate offsets within patches
        
    The container automatically handles complex tensor composition (Y = Y_I * exp(1j * Y_phi))
    and provides comprehensive debug representations showing tensor statistics.
    """
    @debug
    def __init__(self, X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, nn_indices, global_offsets, local_offsets, probeGuess):
        self.X = X
        self.Y_I = Y_I
        self.Y_phi = Y_phi
        self.norm_Y_I = norm_Y_I
        self.YY_full = YY_full
        self.coords_nominal = coords_nominal
        self.coords = coords_nominal
        self.coords_true = coords_true
        self.nn_indices = nn_indices
        self.global_offsets = global_offsets
        self.local_offsets = local_offsets
        self.probe = probeGuess

        from .tf_helper import combine_complex
        self.Y = combine_complex(Y_I, Y_phi)

    @debug
    def __repr__(self):
        repr_str = '<PtychoDataContainer'
        for attr_name in ['X', 'Y_I', 'Y_phi', 'norm_Y_I', 'YY_full', 'coords_nominal', 'coords_true', 'nn_indices', 'global_offsets', 'local_offsets', 'probe']:
            attr = getattr(self, attr_name)
            if attr is not None:
                if isinstance(attr, np.ndarray):
                    if np.iscomplexobj(attr):
                        repr_str += f' {attr_name}={attr.shape} mean_amplitude={np.mean(np.abs(attr)):.3f}'
                    else:
                        repr_str += f' {attr_name}={attr.shape} mean={attr.mean():.3f}'
                else:
                    repr_str += f' {attr_name}={attr.shape}'
        repr_str += '>'
        return repr_str

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

        Args:
            file_path (str): Path to the output npz file.
        """
        np.savez(
            file_path,
            X=self.X.numpy() if tf.is_tensor(self.X) else self.X,
            Y_I=self.Y_I.numpy() if tf.is_tensor(self.Y_I) else self.Y_I,
            Y_phi=self.Y_phi.numpy() if tf.is_tensor(self.Y_phi) else self.Y_phi,
            norm_Y_I=self.norm_Y_I,
            YY_full=self.YY_full,
            coords_nominal=self.coords_nominal.numpy() if tf.is_tensor(self.coords_nominal) else self.coords_nominal,
            coords_true=self.coords_true.numpy() if tf.is_tensor(self.coords_true) else self.coords_true,
            nn_indices=self.nn_indices,
            global_offsets=self.global_offsets,
            local_offsets=self.local_offsets,
            probe=self.probe.numpy() if tf.is_tensor(self.probe) else self.probe
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
    Convert grouped NumPy data to TensorFlow tensors in a PtychoDataContainer.
    
    This is the primary NumPy->TensorFlow conversion function in the data pipeline.
    It takes a callback that returns grouped data (as produced by raw_data.py) and
    converts all relevant arrays to appropriately-typed TensorFlow tensors.
    
    Args:
        cb: Callback function that returns grouped data dictionary from raw_data.
            The callback pattern allows lazy evaluation - data grouping only occurs
            when needed, which is crucial for memory efficiency with large datasets.
            Expected return: dict with keys 'X_full', 'Y', 'objectGuess', coordinate
            keys, and metadata from raw_data.generate_grouped_data().
        probeGuess: Initial probe function as TensorFlow complex64 tensor, shape (N, N)
        which: Data split selector - 'train' or 'test' (only used if create_split=True)
        create_split: If True, expects cb() to return (data_dict, train_fraction) tuple
                     and applies fraction-based splitting. If False, uses full dataset.
    
    Returns:
        PtychoDataContainer with all arrays converted to appropriate TensorFlow dtypes:
        - Diffraction data (X) -> tf.float32
        - Ground truth patches (Y_I, Y_phi) -> tf.float32
        - Complex ground truth (Y) -> tf.complex64
        - Coordinates -> tf.float32
        
    Notes:
        - Preserves multi-channel dimensions for gridsize > 1 configurations
        - Creates dummy complex tensors if ground truth is missing
        - Validates channel consistency between X and Y tensors
        - Handles train/test splitting consistently across all tensor arrays
    """
    from . import params as cfg
    from . import probe
    
    if create_split:
        dset, train_frac = cb()
    else:
        dset = cb()
        
    gt_image = dset['objectGuess']
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

    # Convert X to a tensor, preserving its multi-channel shape
    X = tf.convert_to_tensor(X_full_split, dtype=tf.float32)
    coords_nominal = tf.convert_to_tensor(coords_nominal, dtype=tf.float32)
    coords_true = tf.convert_to_tensor(coords_true, dtype=tf.float32)

    # Handle the Y array (ground truth patches)
    if dset['Y'] is None:
        # If Y is missing, create a placeholder with the same multi-channel shape as X.
        Y = tf.ones_like(X, dtype=tf.complex64)
        print("loader: setting dummy Y ground truth with correct channel shape.")
    else:
        Y_full = dset['Y']
        # CRITICAL: Apply the same split to Y as was applied to X
        if create_split:
            Y_split, _, _ = split_data(Y_full, coords_nominal, coords_true, train_frac, which)
        else:
            Y_split = Y_full
        Y = tf.convert_to_tensor(Y_split, dtype=tf.complex64)
        print("loader: using provided ground truth patches.")

    # Final validation check
    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"Channel mismatch between X ({X.shape[-1]}) and Y ({Y.shape[-1]})")

    # Extract amplitude and phase, which will also have the correct multi-channel shape
    Y_I = tf.math.abs(Y)
    Y_phi = tf.math.angle(Y)

    norm_Y_I = datasets.scale_nphotons(X)

    YY_full = None # This is a placeholder
    
    # Create the container with correctly shaped tensors
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

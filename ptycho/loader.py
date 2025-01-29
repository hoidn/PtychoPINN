"""Generic loader for datasets with non-rectangular scan point patterns."""

import numpy as np
import tensorflow as tf
from typing import Callable, List, Optional, Union

from .params import params, get

def validate_probe_tensors(probe_list: List[tf.Tensor]) -> None:
    """Validate probe tensor compatibility.
    
    Args:
        probe_list: List of probe tensors to validate
        
    Raises:
        ValueError: If probes incompatible
    """
    if not probe_list:
        raise ValueError("Empty probe list")
        
    shape = probe_list[0].shape
    dtype = probe_list[0].dtype
    
    for i, probe in enumerate(probe_list[1:], 1):
        if probe.shape != shape:
            raise ValueError(f"Probe {i} shape {probe.shape} != {shape}")
        if probe.dtype != dtype:
            raise ValueError(f"Probe {i} dtype {probe.dtype} != {dtype}")
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
    A class to contain ptycho data attributes for easy access and manipulation.
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

class MultiPtychoDataContainer:
    def __init__(
        self,
        X: tf.Tensor,
        Y_I: tf.Tensor,
        Y_phi: tf.Tensor,
        norm_Y_I: tf.Tensor,
        YY_full: Optional[tf.Tensor],
        coords_nominal: tf.Tensor,
        coords: tf.Tensor,
        coords_true: tf.Tensor,
        nn_indices: tf.Tensor,
        global_offsets: tf.Tensor,
        local_offsets: tf.Tensor,
        probe_list: List[tf.Tensor],
        probe_indices: tf.Tensor
    ) -> None:
        """Initialize multi-probe container.

        Validates and stores probe tensors and indices.
        """
        validate_probe_tensors(probe_list)
        self.X = X
        self.Y_I = Y_I
        self.Y_phi = Y_phi
        self.norm_Y_I = norm_Y_I
        self.YY_full = YY_full
        self.coords_nominal = coords_nominal
        self.coords = coords
        self.coords_true = coords_true
        self.nn_indices = nn_indices
        self.global_offsets = global_offsets
        self.local_offsets = local_offsets
        self.probe_list = probe_list
        self.probe_indices = probe_indices

        # Combine magnitude and phase into complex tensor
        self.Y = hh.combine_complex(Y_I, Y_phi)

    def get_probe(self, index: int) -> tf.Tensor:
        """Get probe tensor by index.

        Args:
            index: Probe index to retrieve

        Returns:
            Selected probe tensor

        Raises:
            IndexError: If index invalid
        """
        if index < 0 or index >= len(self.probe_list):
            raise IndexError(f"Probe index {index} is out of range.")
        return self.probe_list[index]

    def shuffle_samples(self) -> None:
        """Shuffle samples while maintaining probe associations."""
        indices = tf.random.shuffle(tf.range(tf.shape(self.X)[0]))
        self.X = tf.gather(self.X, indices)
        self.Y_I = tf.gather(self.Y_I, indices)
        self.Y_phi = tf.gather(self.Y_phi, indices)
        self.norm_Y_I = tf.gather(self.norm_Y_I, indices)
        if self.YY_full is not None:
            self.YY_full = tf.gather(self.YY_full, indices)
        self.coords_nominal = tf.gather(self.coords_nominal, indices)
        self.coords = tf.gather(self.coords, indices)
        self.coords_true = tf.gather(self.coords_true, indices)
        self.nn_indices = tf.gather(self.nn_indices, indices)
        self.global_offsets = tf.gather(self.global_offsets, indices)
        self.local_offsets = tf.gather(self.local_offsets, indices)
        self.probe_indices = tf.gather(self.probe_indices, indices)
        # Update self.Y as well
        self.Y = tf.gather(self.Y, indices)

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
        
        dset_train = train_raw.generate_grouped_data(N, K=K, nsamples=nsamples)

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
def load(cb: Callable, probeGuess: tf.Tensor, which: str, create_split: bool, probe_indices: Optional[tf.Tensor] = None) -> Union[PtychoDataContainer, MultiPtychoDataContainer]:
    from . import params as cfg
    from . import probe
    if create_split:
        dset, train_frac = cb()
    else:
        dset = cb()
    gt_image = dset['objectGuess']
    X_full = dset['X_full'] # normalized diffraction
    global_offsets = dset['coords_offsets']
    # Define coords_nominal and coords_true before calling split_data
    coords_nominal = dset['coords_relative']
    coords_true = dset['coords_relative']
    if create_split:
        global_offsets = split_tensor(global_offsets, train_frac, which)
        X, coords_nominal, coords_true = split_data(X_full, coords_nominal, coords_true, train_frac, which)
    else:
        X = X_full
    norm_Y_I = datasets.scale_nphotons(X)
    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)

    Y = tf.ones_like(X)
    Y_I = tf.math.abs(Y)
    Y_phi = tf.math.angle(Y)

    # TODO get rid of?
    YY_full = None
    # TODO complex
    # Create appropriate container type
    if probe_indices is not None:
        # Multi-probe case
        container = MultiPtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=norm_Y_I,
            YY_full=YY_full,
            coords_nominal=coords_nominal,
            coords=coords_true,
            coords_true=coords_true,
            nn_indices=dset['nn_indices'],
            global_offsets=dset['coords_offsets'],
            local_offsets=dset['coords_relative'],
            probe_list=[probeGuess],
            probe_indices=probe_indices
        )
    else:
        # Single probe case
        container = PtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=norm_Y_I,
            YY_full=YY_full,
            coords_nominal=coords_nominal,
            coords_true=coords_true,
            nn_indices=dset['nn_indices'],
            global_offsets=dset['coords_offsets'],
            local_offsets=dset['coords_relative'],
            probe=probeGuess
        )
    print('INFO:', which)
    print(container)
    return container

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

def merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> MultiPtychoDataContainer:
    """Merge containers preserving probe associations.
    
    Args:
        containers: List of containers to merge
        shuffle: Whether to shuffle samples
        
    Returns:
        Merged container instance
    """
    # Validate containers
    if not containers:
        raise ValueError("No containers provided")
        
    # Concatenate data from all containers
    X = tf.concat([c.X for c in containers], axis=0)
    Y_I = tf.concat([c.Y_I for c in containers], axis=0)
    Y_phi = tf.concat([c.Y_phi for c in containers], axis=0)
    norm_Y_I = tf.concat([c.norm_Y_I for c in containers], axis=0)
    YY_full = None if any(c.YY_full is None for c in containers) else tf.concat([c.YY_full for c in containers], axis=0)
    coords_nominal = tf.concat([c.coords_nominal for c in containers], axis=0)
    coords = tf.concat([c.coords for c in containers], axis=0)
    coords_true = tf.concat([c.coords_true for c in containers], axis=0)
    nn_indices = tf.concat([c.nn_indices for c in containers], axis=0)
    global_offsets = tf.concat([c.global_offsets for c in containers], axis=0)
    local_offsets = tf.concat([c.local_offsets for c in containers], axis=0)
    
    # Create probe indices tensor
    probe_indices = tf.concat([tf.fill([tf.shape(c.X)[0]], i) for i, c in enumerate(containers)], axis=0)
    
    # Create merged container
    merged = MultiPtychoDataContainer(
        X=X,
        Y_I=Y_I,
        Y_phi=Y_phi,
        norm_Y_I=norm_Y_I,
        YY_full=YY_full,
        coords_nominal=coords_nominal,
        coords=coords,
        coords_true=coords_true,
        nn_indices=nn_indices,
        global_offsets=global_offsets,
        local_offsets=local_offsets,
        probe_list=[c.probe for c in containers],
        probe_indices=probe_indices
    )
    
    # Optionally shuffle the merged data
    if shuffle:
        merged.shuffle_samples()
        
    return merged

# TODO this should be a method of PtychoDataContainer
#@debug
def load(cb: Callable, probeGuess: tf.Tensor, which: str, create_split: bool) -> PtychoDataContainer:
    from . import params as cfg
    from . import probe
    if create_split:
        dset, train_frac = cb()
    else:
        dset = cb()
    gt_image = dset['objectGuess']
    X_full = dset['X_full'] # normalized diffraction
    global_offsets = dset[key_coords_offsets]
    # Define coords_nominal and coords_true before calling split_data
    coords_nominal = dset[key_coords_relative]
    coords_true = dset[key_coords_relative]
    if create_split:
        global_offsets = split_tensor(global_offsets, train_frac, which)
        X, coords_nominal, coords_true = split_data(X_full, coords_nominal, coords_true, train_frac, which)
    else:
        X = X_full
    norm_Y_I = datasets.scale_nphotons(X)
    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)
#    try:
#        Y = get_image_patches(gt_image, global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
#    except:
#        Y = tf.zeros_like(X)

    norm_Y_I = datasets.scale_nphotons(X)

    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)

    # TODO we shouldn't be nuking the ground truth
##    try:
#    if dset['Y'] is None:
#        Y = get_image_patches(gt_image,
#            global_offsets, coords_true) * probe.get_probe_mask_real(cfg.get('N'))
#        print("loader: generating ground truth patches from image and offsets")
#    else:
#        Y = dset['Y']
#        print("loader: using provided ground truth patches")
    if dset['Y'] is None:
        Y = tf.ones_like(X)
        print("loader: setting dummy Y ground truth")
    else:
        Y = dset['Y']
        print("loader: using provided ground truth patches")
    Y_I = tf.math.abs(Y)
    Y_phi = tf.math.angle(Y)

    # TODO get rid of?
    YY_full = None
    # TODO complex
    container = PtychoDataContainer(X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, dset['nn_indices'], dset['coords_offsets'], dset['coords_relative'], probeGuess)
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
    diff3d = np.transpose(data['diffraction'], [2, 0, 1])
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

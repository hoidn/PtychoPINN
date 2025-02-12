"""Generic loader for datasets with non-rectangular scan point patterns."""

import warnings
import tensorflow as tf
from typing import Callable

from .params import params, get
from .autotest.debug import debug
from . import diffsim as datasets
from . import tf_helper as hh
from .raw_data import RawData, key_coords_offsets, key_coords_relative
import numpy as np

class PtychoDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        if hasattr(train_data, 'norm_Y_I') and train_data.norm_Y_I is not None:
            self.norm_Y_I = train_data.norm_Y_I
        else:
            from . import diffsim as datasets
            import tensorflow as tf
            # Fallback: calculate norm from the train data's X
            self.norm_Y_I = datasets.scale_nphotons(tf.convert_to_tensor(train_data.X))

class PtychoDataContainer:
    """
    A class to contain ptycho data attributes for easy access and manipulation.
    """
    def __init__(self,
                 X,
                 Y_I,
                 Y_phi,
                 norm_Y_I,
                 YY_full,
                 coords_nominal,
                 coords_true,
                 nn_indices,
                 global_offsets,
                 local_offsets,
                 probeGuess,
                 probe_indices=None):
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
        self._probe = probeGuess
        self.probe_indices = probe_indices

        from .tf_helper import combine_complex
        self.Y = combine_complex(Y_I, Y_phi)

    @property
    def probe(self) -> np.ndarray:
        if self._probe is None:
            raise AttributeError("No probe available in this container.")
        return self._probe

    @probe.setter 
    def probe(self, value: np.ndarray):
        import numpy as np
        if not isinstance(value, np.ndarray):
            raise TypeError("Probe must be a numpy array.")
        if value.ndim == 2:
            # Promote 2D array to 3D with channel dimension.
            value = value[..., np.newaxis]
        elif value.ndim == 3:
            if value.shape[-1] != 1:
                raise ValueError("Invalid probe shape; expected last dimension to be 1.")
        else:
            raise ValueError("Invalid probe shape; expected a 2D or 3D array.")
        self._probe = value

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

class MultiPtychoDataContainer:
    def YY_full(self):
        """
        Access the YY_full attribute.

        Returns:
            np.ndarray: The YY_full data.

        Raises:
            AttributeError: If YY_full is not available.
        """
        if hasattr(self, 'YY_full') and self.YY_full is not None:
            return self.YY_full
        else:
            # Provide a fallback or handle the absence appropriately
            warnings.warn("YY_full is not available in multi-probe mode.")
            return None
    def __init__(self,
                 X,
                 Y_I,
                 Y_phi,
                 norm_Y_I,
                 YY_full,
                 coords_nominal,
                 coords_true,
                 nn_indices,
                 global_offsets,
                 local_offsets,
                 probe_indices,
                 probes):
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
        self.probe_indices = probe_indices
        self.probes = probes

        # Shape validation
        num_samples = X.shape[0]
        assert probe_indices.shape[0] == num_samples, "Probe indices should have the same number of samples as X"
        assert probes.shape[0] > probe_indices.max(), "Probe indices exceed the number of provided probes"
        assert len(probes.shape) == 4, "Probes should be a 4D tensor with shape [num_probes, H, W, 1]"
        assert probes.shape[-1] == 1, "Probes should have shape [num_probes, H, W, 1]"

        # Additional shape checks for other arrays
        assert Y_I.shape[0] == num_samples, "Y_I should have the same number of samples as X"
        assert Y_phi.shape[0] == num_samples, "Y_phi should have the same number of samples as X"
        assert coords_nominal.shape[0] == num_samples, "coords_nominal should have the same number of samples as X"
        assert coords_true.shape[0] == num_samples, "coords_true should have the same number of samples as X"
        assert nn_indices.shape[0] == num_samples, "nn_indices should have the same number of samples as X"
        assert global_offsets.shape[0] == num_samples, "global_offsets should have the same number of samples as X"
        assert local_offsets.shape[0] == num_samples, "local_offsets should have the same number of samples as X"

        # Validate probe indices bounds
        max_probe_index = probe_indices.max()
        min_probe_index = probe_indices.min()
        assert max_probe_index < probes.shape[0], f"Probe index {max_probe_index} is out of bounds"
        assert min_probe_index >= 0, f"Probe indices must be non-negative"
        num_samples = X.shape[0]
        assert probe_indices.shape[0] == num_samples, "Probe indices should have the same number of samples as X"
        assert len(probes.shape) == 4, "Probes should be a 4D tensor with shape [num_probes, H, W, 1]"

    @classmethod
    def from_containers(cls, containers, probes=None):
        """
        Merge multiple PtychoDataContainer instances into a MultiPtychoDataContainer.

        Args:
            containers (List[PtychoDataContainer]): List of containers to merge.
            probes (Optional[np.ndarray]): Array of probes with shape [num_probes, H, W, 1].
                If None, probes will be collected from the containers.

        Returns:
            MultiPtychoDataContainer: Merged data container.
        """
        X_list = []
        Y_I_list = []
        Y_phi_list = []
        norm_Y_I_list = []
        YY_full_list = []
        coords_nominal_list = []
        coords_true_list = []
        nn_indices_list = []
        global_offsets_list = []
        local_offsets_list = []
        probe_indices_list = []
        probes_list = []

        for idx, container in enumerate(containers):
            num_samples = container.X.shape[0]
            X_list.append(container.X)
            Y_I_list.append(container.Y_I)
            Y_phi_list.append(container.Y_phi)
            norm_Y_I_list.append(np.full(num_samples, container.norm_Y_I))
            YY_full_list.append(container.YY_full)
            coords_nominal_list.append(container.coords_nominal)
            coords_true_list.append(container.coords_true)
            nn_indices_list.append(container.nn_indices)
            global_offsets_list.append(container.global_offsets)
            local_offsets_list.append(container.local_offsets)
            # Ensure probe indices are int64
            probe_indices_list.append(np.full(num_samples, idx, dtype=np.int64))

            # Ensure container.probe has shape [H, W, 1]
            probe = container.probe
            if len(probe.shape) == 2:
                # [H, W] -> [H, W, 1]
                probe = probe[..., np.newaxis]
            elif len(probe.shape) == 3 and probe.shape[-1] == 1:
                # Correct shape
                pass
            else:
                raise ValueError("Invalid probe shape in container")

            probes_list.append(probe)

        # Handle YY_full appropriately
        if all(y is not None for y in YY_full_list):
            YY_full = np.concatenate(YY_full_list, axis=0)
        else:
            YY_full = None  # Or handle as needed
        X = np.concatenate(X_list, axis=0)
        Y_I = np.concatenate(Y_I_list, axis=0)
        Y_phi = np.concatenate(Y_phi_list, axis=0)
        norm_Y_I = np.concatenate(norm_Y_I_list, axis=0)
        coords_nominal = np.concatenate(coords_nominal_list, axis=0)
        coords_true = np.concatenate(coords_true_list, axis=0)
        nn_indices = np.concatenate(nn_indices_list, axis=0)
        global_offsets = np.concatenate(global_offsets_list, axis=0)
        local_offsets = np.concatenate(local_offsets_list, axis=0)
        probe_indices = np.concatenate(probe_indices_list, axis=0).astype(np.int64)
        probes = np.stack(probes_list, axis=0)

        return cls(
            X,
            Y_I,
            Y_phi,
            norm_Y_I,
            YY_full=None,  # Handle as needed
            coords_nominal=coords_nominal,
            coords_true=coords_true,
            nn_indices=nn_indices,
            global_offsets=global_offsets,
            local_offsets=local_offsets,
            probe_indices=probe_indices,
            probes=probes
        )

    @classmethod
    def from_single_container(cls, container):
        """
        Create a MultiPtychoDataContainer from a single PtychoDataContainer.

        Args:
            container (PtychoDataContainer): The original data container.

        Returns:
            MultiPtychoDataContainer: Converted data container.
        """
        num_samples = container.X.shape[0]
        probe_indices = np.zeros(num_samples, dtype=np.int64)
        # Ensure probe has shape [1, H, W, 1]
        probe = container.probe
        if len(probe.shape) == 2:
            probe = probe[..., np.newaxis]  # Add channel dimension
        probes = probe[np.newaxis, ...]  # Add batch dimension

        return cls(
            container.X,
            container.Y_I,
            container.Y_phi,
            container.norm_Y_I,
            container.YY_full,
            container.coords_nominal,
            container.coords_true,
            container.nn_indices,
            container.global_offsets,
            container.local_offsets,
            probe_indices,
            probes
        )


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
    # Initialize probe indices if not present
    if 'probe_indices' in dset:
        probe_indices = tf.convert_to_tensor(dset['probe_indices'], dtype=tf.int64)
    else:
        probe_indices = tf.zeros((X.shape[0],), dtype=tf.int64)
        
    container = PtychoDataContainer(X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, 
                                  dset['nn_indices'], dset['coords_offsets'], dset['coords_relative'], 
                                  probeGuess, probe_indices=probe_indices)
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

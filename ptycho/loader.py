""" 'Generic' loader for datasets with non-rectangular scan point patterns."""

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
from ptycho import diffsim as datasets
from .params import params, get
from .logging import debug

# If == 1, relative coordinates are (patch CM coordinate - solution region CM
# coordinate)
local_offset_sign = 1

key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'

class RawData:
    @debug()
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess,
                 scan_index, objectGuess = None):
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

    @staticmethod
    @debug()
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

    @debug()
    def __str__(self):
        return (f"RawData: \n"
                f"xcoords: {self.xcoords.shape} \n"
                f"ycoords: {self.ycoords.shape} \n"
                f"xcoords_start: {self.xcoords_start.shape} \n"
                f"ycoords_start: {self.ycoords_start.shape} \n"
                f"diff3d: {self.diff3d.shape} \n"
                f"probeGuess: {self.probeGuess.shape if self.probeGuess is not None else 'None'} \n"
                f"scan_index: {self.scan_index.shape} \n"
                f"objectGuess: {'Present' if self.objectGuess is not None else 'None'}")

    @debug()
    def to_file(self, file_path):
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
    @debug()
    def from_file(train_data_file_path):
        """
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
    @debug()
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

    @debug()
    def generate_grouped_data(self, N, K = 7, nsamples = 1):
        """
        Generate nearest-neighbor solution region grouping.
        """
#        np.random.seed(get('npseed'))
#        print('DEBUG:', 'setting np seed in generate_grouped_data')
        print('DEBUG:', 'nsamples:', nsamples)
        return get_neighbor_diffraction_and_positions(self, N, K=K, nsamples=nsamples)


    @debug()
    def _check_data_validity(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
        # Check if all inputs are numpy arrays
#        if not all(isinstance(arr, np.ndarray) for arr in [xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index]):
#            raise ValueError("All inputs must be numpy arrays.")

        # Check if coordinate arrays have matching shapes
        if not (xcoords.shape == ycoords.shape == xcoords_start.shape == ycoords_start.shape):
            raise ValueError("Coordinate arrays must have matching shapes.")

class PtychoDataset:
    @debug()
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data


class PtychoDataContainer:
    """
    A class to contain ptycho data attributes for easy access and manipulation.
    """
    @debug()
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

    @debug()
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
    @debug()
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
        # TODO this could be handled by a decorator
        from . import params as cfg
        if N is None:
            N = cfg.get('N')
        train_raw = RawData.from_coords_without_pc(xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess)
        
        dset_train = train_raw.generate_grouped_data(N, K=K, nsamples=nsamples)

        # Use loader.load() to handle the conversion to PtychoData
        return load(lambda: dset_train, probeGuess, which=None, create_split=False)

    # TODO currently this can only handle a single object image
    @staticmethod
    @debug()
    def from_simulation(xcoords, ycoords, xcoords_start, ycoords_start, probeGuess,
                 objectGuess, scan_index = None):
        """
        """
        from .diffsim import illuminate_and_diffract
        ptycho_data = RawData(xcoords, ycoords, xcoords_start, ycoords_start, None,
                              probeGuess, scan_index, objectGuess)

        global_offsets, local_offsets, nn_indices = calculate_relative_coords(
                    ptycho_data.xcoords, ptycho_data.ycoords)

        # TODO get rid of separate nominal and real coordinates
        _, coords_true, _ = calculate_relative_coords(ptycho_data.xcoords_start,
                                                      ptycho_data.ycoords_start)
        coords_nominal = coords_true

        Y_obj = get_image_patches(objectGuess, global_offsets, local_offsets) 
        Y_I = tf.math.abs(Y_obj)
        Y_phi = tf.math.angle(Y_obj)
        X, Y_I, Y_phi, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
        norm_Y_I = datasets.scale_nphotons(X)
        return PtychoDataContainer(X, Y_I, Y_phi, norm_Y_I, objectGuess, coords_nominal,
                                   coords_true, nn_indices, global_offsets, local_offsets, probeGuess)

####
# two functions to organize flat coordinate arrays into 'solution region' format
####
@debug()
def get_neighbor_self_indices(xcoords, ycoords):
    """
    assign each pattern index to itself
    """
    N = len(xcoords)
    nn_indices = np.arange(N).reshape(N, 1) 
    return nn_indices

@debug()
def get_neighbor_indices(xcoords, ycoords, K = 3):
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

#####

@debug()
def sample_rows(indices, n, m):
    N = indices.shape[0]
    result = np.zeros((N, m, n), dtype=int)
    for i in range(N):
        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
    return result

@debug()
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

@debug()
def crop12(arr, size):
    N, M = arr.shape[1:3]
    return arr[:, N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2, ...]

# TODO move to tf_helper, except the parts that are specific to xpp
# should be in xpp.py
from .tf_helper import complexify_function
import tensorflow as tf

@debug()
def extract_and_translate_patch_np(image, offset, patch_size):
    # Calculate the starting coordinates for cropping.
    start_x = int(offset[0]) + image.shape[0] // 2 - patch_size // 2
    start_y = int(offset[1]) + image.shape[1] // 2 - patch_size // 2

    # Crop the patch from the image.
    patch = image[start_x:start_x + patch_size, start_y:start_y + patch_size, :]

    return patch

@debug()
def unsqueeze_coords(tensor):
    """
    unsqueeze 2d coordinates to flat 4d tensor format
    """
    # TODO assert tensor dimensionality is 2
    return tensor[:, None, :, None]

from . import tf_helper as hh
@complexify_function
@debug()
def get_image_patches(gt_image, global_offsets, local_offsets):
    """
    Generate and return image patches in channel format using a single canvas.

    Args:
        gt_image (tensor): Ground truth image tensor.
        global_offsets (tensor): Global offset tensor.
        local_offsets (tensor): Local offset tensor.

    Returns:
        tensor: Image patches in channel format.
    """
    # Get necessary parameters
    gridsize = params()['gridsize']
    N = params()['N']
    B = global_offsets.shape[0]
    c = gridsize**2

    # Pad the ground truth image
    gt_padded = hh.pad(gt_image[None, ..., None], N // 2)

    # Calculate the combined offsets
    offsets_c = calculate_combined_offsets(global_offsets, local_offsets)

    offsets_f = hh._channel_to_flat(offsets_c)

    # Create a canvas to store the extracted patches
    canvas = np.array(tf.zeros((B, N, N, c)))

    # Iterate over the combined offsets and extract patches
    for i in range(B):
        # Extract the current offset
        for j in range(c):
            offset = -offsets_f[i, :, :, j]
            translated_patch = hh.translate(gt_padded, offset)
            canvas[i, :, :, j] = np.array(translated_patch)[0, :N, :N, 0]

    return tf.convert_to_tensor(canvas)

@debug()
def calculate_combined_offsets(global_offsets, local_offsets):
    """
    Calculate the combined offsets.

    Args:
        global_offsets (tensor): Global offset tensor.
        local_offsets (tensor): Local offset tensor.

    Returns:
        tensor: Combined offset tensor.
    """
    offsets_c = tf.cast((global_offsets + local_offsets), tf.float32)
    return offsets_c

# TODO move to tf_helper, except the parts that are specific to xpp
# should be in xpp.py
@debug()
def tile_gt_object(gt_image, shape):
    from . import tf_helper as hh
    gridsize = params()['gridsize']
    N = params()['N']
    B = shape[0] #* gridsize**2

    gt_repeat = tf.repeat(
        tf.repeat(gt_image[None, ...], B, axis = 0)[..., None],
        gridsize**2, axis = 3)

    gt_repeat = hh.pad(gt_repeat, N // 2)
    return gt_repeat

@debug()
def calculate_relative_coords(xcoords, ycoords, K = 6, C = None, nsamples = 10):
    """
    Group scan indices and coordinates in to solution regions, then
    calculate coords_offsets (global solution region coordinates) and
    coords_relative (local solution patch coords) from ptycho_data using
    the provided index_grouping_cb callback function.

    Args:
        ptycho_data (RawData): An instance of the RawData class containing the dataset.
        index_grouping_cb (callable): A callback function that defines how to group indices.

    Returns:
        tuple: A tuple containing coords_offsets and coords_relative.
    """
    nn_indices, coords_nn = group_coords(xcoords, ycoords, K = K, C = C, nsamples = nsamples)
    coords_offsets, coords_relative = get_relative_coords(coords_nn)
    return coords_offsets, coords_relative, nn_indices

@debug()
def group_coords(xcoords, ycoords, K = 6, C = None, nsamples = 10):
    """
    Assemble a flat dataset into solution regions using nearest-neighbor grouping.

    Assumes ptycho_data.xcoords and ptycho_data.ycoords are of shape (M).
    Returns:
        nn_indices: shape (M, C)
        coords_nn: shape (M, 1, 2, C)
    """
    gridsize = params()['gridsize']
    if C is None:
        C = gridsize**2
    if C == 1:
        nn_indices = get_neighbor_self_indices(xcoords, ycoords)
    else:
        nn_indices = get_neighbor_indices(xcoords, ycoords, K=K)
        nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)

    #diff4d_nn = np.transpose(ptycho_data.diff3d[nn_indices], [0, 2, 3, 1])
    coords_nn = np.transpose(np.array([xcoords[nn_indices],
                            ycoords[nn_indices]]),
                            [1, 0, 2])[:, None, :, :]
    return nn_indices, coords_nn[:, :, :, :]

@debug()
def get_neighbor_diffraction_and_positions(ptycho_data, N, K=6, C=None, nsamples=10):
    """
    ptycho_data: an instance of the RawData class
    """
    
    nn_indices, coords_nn = group_coords(ptycho_data.xcoords, ptycho_data.ycoords,
                                         K = K, C = C, nsamples = nsamples)

    diff4d_nn = np.transpose(ptycho_data.diff3d[nn_indices], [0, 2, 3, 1])

    # IMPORTANT: coord swap
    #coords_nn = coords_nn[:, :, ::-1, :]

    coords_offsets, coords_relative = get_relative_coords(coords_nn)

    if ptycho_data.xcoords_start is not None:
        coords_start_nn = np.transpose(np.array([ptycho_data.xcoords_start[nn_indices], ptycho_data.ycoords_start[nn_indices]]),
                                       [1, 0, 2])[:, None, :, :]
        #coords_start_nn = coords_start_nn[:, :, ::-1, :]
        coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
    else:
        coords_start_offsets = coords_start_relative = None

    dset = {
        'diffraction': diff4d_nn,
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

@complexify_function
@debug()
def get_image_patches(gt_image, global_offsets, local_offsets):
    """
    Generate and return image patches in channel format using a single canvas.

    Args:
        gt_image (tensor): Ground truth image tensor.
        global_offsets (tensor): Global offset tensor.
        local_offsets (tensor): Local offset tensor.

    Returns:
        tensor: Image patches in channel format.
    """
    # Get necessary parameters
    gridsize = params()['gridsize']
    N = params()['N']
    B = global_offsets.shape[0]
    c = gridsize**2

    # Pad the ground truth image once
    gt_padded = hh.pad(gt_image[None, ..., None], N // 2)

    # Calculate the combined offsets by adding global and local offsets
    offsets_c = tf.cast((global_offsets + local_offsets), tf.float32)
    offsets_f = hh._channel_to_flat(offsets_c)

    # Create a canvas to store the extracted patches
    canvas = np.zeros((B, N, N, c))

    # Iterate over the combined offsets and extract patches one by one
    for i in range(B * c):
        offset = -offsets_f[i, :, :, 0]
        translated_patch = hh.translate(gt_padded, offset)
        canvas[i // c, :, :, i % c] = np.array(translated_patch)[0, :N, :N, 0]

    # Convert the canvas to a TensorFlow tensor and return it
    return tf.convert_to_tensor(canvas)

@debug()
def shift_and_sum(obj_tensor, global_offsets, M=10):
    from . import tf_helper as hh
    # Extract necessary parameters
    N = params()['N']
    # Select the central part of the object tensor
    obj_tensor = obj_tensor[:, N // 2 - M // 2: N // 2 + M // 2, N // 2 - M // 2: N // 2 + M // 2, :]
    # Calculate the center of mass of global_offsets
    center_of_mass = tf.reduce_mean(tf.cast(global_offsets, tf.float32), axis=0)
    # Adjust global_offsets by subtracting the center of mass
    adjusted_offsets = tf.cast(global_offsets, tf.float32) - center_of_mass
    # Calculate dynamic padding based on maximum adjusted offset
    max_offset = tf.reduce_max(tf.abs(adjusted_offsets))
    dynamic_pad = int(tf.cast(tf.math.ceil(max_offset), tf.int32))
    print('PADDING SIZE:', dynamic_pad)
    
    # Create a canvas to store the shifted and summed object tensors
    result = tf.zeros_like(hh.pad(obj_tensor[0:1], dynamic_pad))
    
    # Iterate over the adjusted offsets and perform shift-and-sum
    for i in range(len(adjusted_offsets)):
        # Apply dynamic padding to the current object tensor
        padded_obj_tensor = hh.pad(obj_tensor[i:i+1], dynamic_pad)
        # Squeeze and cast adjusted offset to 2D float for translation
        offset_2d = tf.cast(tf.squeeze(adjusted_offsets[i]), tf.float32)
        # Translate the padded object tensor
        translated_obj = hh.translate(padded_obj_tensor, offset_2d, interpolation='bilinear')
        # Accumulate the translated object tensor
        result += translated_obj[0]
    
    # TODO: how could we support multiple scans?
    return result[0]


# TODO move to tf_helper?
@debug()
def reassemble_position(obj_tensor, global_offsets, M = 10):
    ones = tf.ones_like(obj_tensor)
    return shift_and_sum(obj_tensor, global_offsets, M = M) /\
        (1e-9 + shift_and_sum(ones, global_offsets, M = M))

@debug()
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

@debug()
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
@debug()
def load(cb, probeGuess, which=None, create_split=True, **kwargs) -> PtychoDataContainer:
    from . import params as cfg
    from . import probe
    #probeGuess = probe.get_probe(fmt = 'np')
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
#        Y_obj = get_image_patches(gt_image, global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
#    except:
#        Y_obj = tf.zeros_like(X)

    norm_Y_I = datasets.scale_nphotons(X)

    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)

#    try:
    Y_obj = get_image_patches(gt_image,
        global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
    Y_I = tf.math.abs(Y_obj)
    Y_phi = tf.math.angle(Y_obj)
#    except: 
#        Y_obj = None
#        Y_I = tf.zeros_like(X)
#        Y_phi = tf.zeros_like(X)
#    Y_I = tf.ones_like(X)
#    Y_phi = tf.ones_like(X)

    YY_full = None
    # TODO complex
    container = PtychoDataContainer(X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, dset['nn_indices'], dset['coords_offsets'], dset['coords_relative'], probeGuess)
    print('INFO:', which)
    print(container)
    return container

# Images are amplitude, not intensity
@debug()
def normalize_data(dset, N):
    X_full = dset['diffraction']
    X_full_norm = ((N / 2)**2) / np.mean(tf.reduce_sum(dset['diffraction']**2, axis=[1, 2]))
    return X_full_norm * X_full

@debug()
def crop(arr2d, size):
    N, M = arr2d.shape
    return arr2d[N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2]

@debug()
def get_gt_patch(offset, N, gt_image):
    from . import tf_helper as hh
    return crop(
        hh.translate(gt_image, offset),
        N // 2)


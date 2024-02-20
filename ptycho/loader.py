""" 'Generic' loader for datasets with non-rectangular scan point patterns."""

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
from ptycho import diffsim as datasets
from .params import params

# If == 1, relative coordinates are (patch CM coordinate - solution region CM
# coordinate)
local_offset_sign = 1

key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'

class RawData:
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess,
                 scan_index, objectGuess = None):
        # Sanity checks
        self._check_data_validity(xcoords, ycoords, xcoords_start, ycoords_start, diff3d,
                    probeGuess, scan_index)

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

    def generate_grouped_data(self, N, K = 7, nsamples = 1):
        """
        Generate nearest-neighbor solution region grouping.
        """
        return get_neighbor_diffraction_and_positions(self, N, K=K, nsamples=nsamples)


    def _check_data_validity(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
        # Check if all inputs are numpy arrays
#        if not all(isinstance(arr, np.ndarray) for arr in [xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index]):
#            raise ValueError("All inputs must be numpy arrays.")

        # Check if coordinate arrays have matching shapes
        if not (xcoords.shape == ycoords.shape == xcoords_start.shape == ycoords_start.shape):
            raise ValueError("Coordinate arrays must have matching shapes.")
        # TODO we should allow diff3d to be None
        # Add more checks as necessary, for example:
        # - Check if 'diff3d' has the expected number of dimensions
        # - Check if 'probeGuess' has a specific shape or type criteria
        # Check if 'scan_index' has the correct length
#        if len(scan_index) != diff3d.shape[0]:
#            raise ValueError("Length of scan_index array must match the number of diffraction patterns.")

class PtychoDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

class PtychoData:
    def __init__(self, X, Y_I, Y_phi, YY_full, coords_nominal, coords_true, probe, scan_index = None):
        # TODO get rid of nominal / true distinction
        from .tf_helper import combine_complex
        self.X = X
        self.Y = combine_complex(Y_I, Y_phi)
        self.Y_I = Y_I
        self.YY_full = YY_full
        self.coords_nominal = coords_nominal
        self.coords = coords_nominal
        self.coords_true = coords_true
        self.probe = probe
        self.scan_index = scan_index
    @staticmethod
    def from_loader_output(loader_output, probe):
        """
        Static method to create a PtychoData instance from loader output.

        Args:
            loader_output (dict): The output dictionary from loader.load().
            probe (any): The probe tensor.
        """
        return PtychoData(
            X=loader_output['X'],
            Y_I=loader_output['Y_I'],
            Y_phi=loader_output['Y_phi'],
            YY_full=loader_output['YY_full'],
            coords_nominal=loader_output['coords'][0],
            coords_true=loader_output['coords'][1],
            probe=probe
        )

class PtychoDataContainer:
    """
    A class to contain ptycho data attributes for easy access and manipulation.
    """
    def __init__(self, X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, nn_indices, global_offsets, local_offsets, probeGuess):
        self.X = X
        self.Y_I = Y_I
        self.Y_phi = Y_phi
        self.norm_Y_I = norm_Y_I
        self.YY_full = YY_full
        self.coords = (coords_nominal, coords_true)
        self.nn_indices = nn_indices
        self.global_offsets = global_offsets
        self.local_offsets = local_offsets
        self.probe = probeGuess

        from .tf_helper import combine_complex
        self.Y = combine_complex(Y_I, Y_phi)

    def __repr__(self):
        return f'<PtychoDataContainer X={self.X.shape}, Y_I={self.Y_I.shape}, Y_phi={self.Y_phi.shape}, ' \
               f'norm_Y_I={self.norm_Y_I.shape}, YY_full={self.YY_full}, ' \
               f'coords_nominal={self.coords[0].shape}, coords_true={self.coords[1].shape}, ' \
               f'nn_indices={self.nn_indices.shape}, global_offsets={self.global_offsets.shape}, ' \
               f'local_offsets={self.local_offsets.shape}>'
    @staticmethod
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
        train_raw = RawData.from_coords_without_pc(xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess)
        dset_train = train_raw.generate_grouped_data(N, K=K, nsamples=nsamples)

        # Use loader.load() to handle the conversion to PtychoData
        train_data = load(lambda: dset_train, which=None, create_split=False)
        intensity_scale = train_data.norm_Y_I

        return PtychoDataContainer(train_data.X, train_data.Y_I, train_data.Y_phi, intensity_scale, train_data.YY_full, train_data.coords_nominal, train_data.coords_true, train_data.nn_indices, train_data.global_offsets, train_data.local_offsets, probeGuess)

    # TODO currently this can only handle a single object image
    @staticmethod
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
def get_neighbor_self_indices(xcoords, ycoords):
    """
    assign each pattern index to itself
    """
    N = len(xcoords)
    nn_indices = np.arange(N).reshape(N, 1) 
    return nn_indices

def get_neighbor_indices(xcoords, ycoords, K = 3):
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

#####

def sample_rows(indices, n, m):
    N = indices.shape[0]
    result = np.zeros((N, m, n), dtype=int)
    for i in range(N):
        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
    return result

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

def crop12(arr, size):
    N, M = arr.shape[1:3]
    return arr[:, N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2, ...]

# TODO move to tf_helper, except the parts that are specific to xpp
# should be in xpp.py
from .tf_helper import complexify_function
@complexify_function
def get_image_patches(gt_image, global_offsets, local_offsets):
    """
    Generate and return image patches in channel format.
    """
    from . import tf_helper as hh
    gridsize = params()['gridsize']
    N = params()['N']
    B = global_offsets.shape[0]

    gt_repeat = tf.repeat(
        tf.repeat(gt_image[None, ...], B, axis = 0)[..., None],
        gridsize**2, axis = 3)

    gt_repeat = hh.pad(gt_repeat, N // 2)

    gt_repeat_f = hh._channel_to_flat(gt_repeat)

    offsets_c = tf.cast(
            (global_offsets + local_offsets),
            tf.float32)

    offsets_f = hh._channel_to_flat(offsets_c)

    gt_translated = hh.translate(tf.squeeze(gt_repeat_f)[..., None],
        -tf.squeeze(offsets_f))[:, :N, :N, :]
    gt_translated = hh._flat_to_channel(gt_translated)
    return gt_translated

# TODO move to tf_helper, except the parts that are specific to xpp
# should be in xpp.py
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

#def shift_and_sum(obj_tensor, global_offsets, M = 10):
#    canvas_pad = 100
#    from . import tf_helper as hh
#    N = params()['N']
#    offsets_2d = tf.cast(tf.squeeze(global_offsets), tf.float32)
#    obj_tensor = obj_tensor[:, N // 2 - M // 2: N // 2 + M // 2, N // 2 - M // 2: N // 2 + M // 2, :]
#    obj_tensor = hh.pad(obj_tensor, canvas_pad)
#    obj_translated = hh.translate(obj_tensor, offsets_2d, interpolation = 'bilinear')
#    return tf.reduce_sum(obj_translated, 0)

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

    # Apply dynamic padding
    obj_tensor = hh.pad(obj_tensor, dynamic_pad)

    # Squeeze and cast adjusted offsets to 2D float for translation
    offsets_2d = tf.cast(tf.squeeze(adjusted_offsets), tf.float32)

    # Translate the object tensor
    obj_translated = hh.translate(obj_tensor, offsets_2d, interpolation='bilinear')

    # Reduce and sum across the first dimension
    return tf.reduce_sum(obj_translated, axis=0)


# TODO move to tf_helper?
def reassemble_position(obj_tensor, global_offsets, M = 10):
    ones = tf.ones_like(obj_tensor)
    return shift_and_sum(obj_tensor, global_offsets, M = M) /\
        (1e-9 + shift_and_sum(ones, global_offsets, M = M))

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

    from . import params as cfg
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
    Y_obj = get_image_patches(gt_image, global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]

    norm_Y_I = datasets.scale_nphotons(X)

    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)

    Y_obj = get_image_patches(gt_image,
        global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
    Y_I = tf.math.abs(Y_obj)
    Y_phi = tf.math.angle(Y_obj)
    YY_full = None

    # TODO complex
    return PtychoDataContainer(X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, dset['nn_indices'], dset['coords_offsets'], dset['coords_relative'], probeGuess)

# Images are amplitude, not intensity
def normalize_data(dset, N):
    X_full = dset['diffraction']
    X_full_norm = ((N / 2)**2) / np.mean(tf.reduce_sum(dset['diffraction']**2, axis=[1, 2]))
    return X_full_norm * X_full

def crop(arr2d, size):
    N, M = arr2d.shape
    return arr2d[N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2]

def get_gt_patch(offset, N, gt_image):
    from . import tf_helper as hh
    return crop(
        hh.translate(gt_image, offset),
        N // 2)


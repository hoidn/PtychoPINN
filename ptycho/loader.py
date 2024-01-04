""" 'Generic' loader for datasets with non-rectangular scan point patterns."""

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

from .classes import RawData

# If == 1, relative coordinates are (patch CM coordinate - solution region CM
# coordinate)
local_offset_sign = 1

def get_neighbor_indices(xcoords, ycoords, K = 3):
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

def sample_rows(indices, n, m):
    N = indices.shape[0]
    result = np.zeros((N, m, n), dtype=int)
    for i in range(N):
        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
    return result

def get_relative_coords(coords_nn):
    assert len(coords_nn.shape) == 4
    coords_offsets = np.mean(coords_nn, axis = 3)[..., None]
    # IMPORTANT: sign
    coords_relative = local_offset_sign * (coords_nn - np.mean(coords_nn, axis = 3)[..., None])
    return coords_offsets, coords_relative

def crop12(arr, size):
    N, M = arr.shape[1:3]
    return arr[:, N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2, ...]

# TODO move to tf_helper, except the parts that are specific to xpp
# should be in xpp.py
from .tf_helper import complexify_function
@complexify_function
def get_image_patches(gt_image, global_offsets, local_offsets):
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

def get_neighbor_diffraction_and_positions(ptycho_data, N, K=6, C=None, nsamples=10):
    """
    ptycho_data: an instance of the RawData class
    """
    gridsize = params()['gridsize']
    if C is None:
        C = gridsize**2

    nn_indices = get_neighbor_indices(ptycho_data.xcoords, ptycho_data.ycoords, K=K)
    nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)

    diff4d_nn = np.transpose(ptycho_data.diff3d[nn_indices], [0, 2, 3, 1])
    coords_nn = np.transpose(np.array([ptycho_data.xcoords[nn_indices],
                            ptycho_data.ycoords[nn_indices]]),
                            [1, 0, 2])[:, None, :, :]
#    # IMPORTANT: coord swap
#    coords_nn = coords_nn[:, :, ::-1, :]

    coords_offsets, coords_relative = get_relative_coords(coords_nn)

    if ptycho_data.xcoords_start is not None:
        coords_start_nn = np.transpose(np.array([ptycho_data.xcoords_start[nn_indices], ptycho_data.ycoords_start[nn_indices]]),
                                       [1, 0, 2])[:, None, :, :]
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
        'nn_indices': nn_indices
    }
    X_full = normalize_data(dset, N)
    dset['X_full'] = X_full
    print('neighbor-sampled diffraction shape', X_full.shape)
    return dset

def shift_and_sum(obj_tensor, global_offsets, M = 10):
    canvas_pad = 100
    from . import tf_helper as hh
    N = params()['N']
    offsets_2d = tf.cast(tf.squeeze(global_offsets), tf.float32)
    obj_tensor = obj_tensor[:, N // 2 - M // 2: N // 2 + M // 2, N // 2 - M // 2: N // 2 + M // 2, :]
    obj_tensor = hh.pad(obj_tensor, canvas_pad)
    obj_translated = hh.translate(obj_tensor, offsets_2d, interpolation = 'bilinear')
    return tf.reduce_sum(obj_translated, 0)

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

def load(which, cb, **kwargs):
    from . import params as cfg
    dset, gt_image, train_frac = cb()
    X_full = dset['X_full'] # normalized diffraction
    global_offsets = split_tensor(dset[key_coords_offsets], train_frac, which)
    # Define coords_nominal and coords_true before calling split_data
    coords_nominal = dset[key_coords_relative]
    coords_true = dset[key_coords_relative]
    X, coords_nominal, coords_true = split_data(X_full, coords_nominal, coords_true, train_frac, which)

    norm_Y_I = datasets.scale_nphotons(X)

    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)

    Y_obj = get_image_patches(gt_image,
        global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
    Y_I = tf.math.abs(Y_obj)
    Y_phi = tf.math.angle(Y_obj)
    YY_full = None

    return {
        'X': X,
        'Y_I': Y_I,
        'Y_phi': Y_phi,
        'norm_Y_I': norm_Y_I,
        'YY_full': YY_full,
        'coords': (coords_nominal, coords_true),
        'nn_indices': dset['nn_indices']
    }

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

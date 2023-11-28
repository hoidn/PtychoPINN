import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import loader
from . import loader
from .datasets import scale_nphotons

train_frac = .5
N = 64
# Dimensions of the scan point grid, assuming it's square
gridh, gridw = 32, 32

np.random.seed(7)

import pkg_resources

# Use pkg_resources to get the path to the data file within the installed package
# Update the resource path to the correct top-level 'datasets' directory
data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
obj = np.load(data_file_path)
#from .utils.utils import utils
#q_grid, I_ref = pd.read_csv(utils.resource_path("path goes here"), header = None).values.T

print('raw diffraction shape', obj['diffraction'].shape)

# TODO cast to complex64?
gt_image = obj['objectGuess']

# Load data and reorder the axes
xcoords = obj['xcoords'][:gridh * gridw]
ycoords = obj['ycoords'][:gridh * gridw]
xcoords_start = obj['xcoords_start'][:gridh * gridw]
ycoords_start = obj['ycoords_start'][:gridh * gridw]
diff3d = np.transpose(obj['diffraction'][:, :, :gridh * gridw],
            [2, 0, 1])
probeGuess = obj['probeGuess']

dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
    xcoords_start, ycoords_start, K = 7, nsamples = 5)
#dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
#    xcoords_start, ycoords_start, K = 20, nsamples = 3)
#dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
#    xcoords_start, ycoords_start, K = 1, nsamples = 1, C = 1)

# Images are amplitude, not intensity
X_full = dset['diffraction']
X_full_norm = ((N / 2)**2) / np.mean(tf.reduce_sum(X_full**2, axis = [1, 2]))
X_full = X_full_norm * X_full
print('neighbor-sampled diffraction shape', X_full.shape)

# TODO refactor actual / nominal positions
key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'
#key_coords_offsets = 'coords_offsets'
#key_coords_relative = 'coords_relative'
coords_true = dset[key_coords_relative]
coords_nominal = dset[key_coords_relative]

def get_splits(which):
    """
    Returns (normalized) amplitude and phase for n generated objects
    """
    n_train = int(len(X_full) * train_frac)
    if which == 'train':
        return X_full[:n_train], coords_nominal[:n_train], coords_true[:n_train]
    elif which == 'test':
        return X_full[n_train:], coords_nominal[n_train:], coords_true[n_train:]
    else:
        raise ValueError

def split_tensor(tensor, which = 'test'):
    n_train = int(len(X_full) * train_frac)
    if which == 'train':
        return tensor[:n_train]
    elif which == 'test':
        return tensor[n_train:]
    else:
        raise ValueError

def crop(arr2d, size):
    N, M = arr2d.shape
    return arr2d[N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2]

def get_gt_patch(offset):
    from . import tf_helper as hh
    return crop(
        hh.translate(gt_image, offset),
        N // 2)

from . import params as cfg
#def load(which, **kwargs):
#    global_offsets = split_tensor(dset[key_coords_offsets], which)
#    X, coords_nominal, coords_true = get_splits(which, **kwargs)
#
#    norm_Y_I = datasets.scale_nphotons(X)
#
#    X = tf.convert_to_tensor(X)
#    coords_nominal = tf.convert_to_tensor(coords_nominal)
#    coords_true = tf.convert_to_tensor(coords_true)
#
#    # TODO save memory in get_image_patches
#    # TODO Real space ground truth is for the benefit of supervised learning
#    # reconstruction. we have to be careful, since multiplying by the probe
#    # (even just amplitude) gives away more ground truth information than what
#    # PtychoNN uses in the context of its original paper. therefore,
#    # we just multiply by the probe mask. Check that the simulated datasets
#    # have the same treatment and, if not, change it.
#    Y_obj = loader.get_image_patches(gt_image,
#        global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
#    Y_I = tf.math.abs(Y_obj)
#    Y_phi = tf.math.angle(Y_obj)
#    YY_full = None
#    return (X, Y_I, Y_phi,
#        norm_Y_I, YY_full, norm_Y_I,
#        (coords_nominal, coords_true))

def load(which, **kwargs):
    global_offsets = split_tensor(dset[key_coords_offsets], which)
    X, coords_nominal, coords_true = get_splits(which, **kwargs)

    norm_Y_I = datasets.scale_nphotons(X)

    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)

    Y_obj = loader.get_image_patches(gt_image,
        global_offsets, coords_true) * cfg.get('probe_mask')[..., 0]
    Y_I = tf.math.abs(Y_obj)
    Y_phi = tf.math.angle(Y_obj)
    YY_full = None

    return {
        'diffraction': X,
        'Y_I': Y_I,
        'Y_phi': Y_phi,
        'norm_Y_I': norm_Y_I,
        'YY_full': YY_full,
        'coords': (coords_nominal, coords_true),
        'nn_indices': dset['nn_indices']
    }

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import loader
from ptycho.diffsim import scale_nphotons
from ptycho import diffsim as datasets


train_frac = .5
N = 64
gridh, gridw = 32, 32

np.random.seed(7)

import pkg_resources

data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
obj = np.load(data_file_path)

print('raw diffraction shape', obj['diffraction'].shape)

# TODO cast to complex64?
gt_image = obj['objectGuess']

# Prepare the data for initialization of PtychoData
xcoords = obj['xcoords'][:gridh * gridw]
ycoords = obj['ycoords'][:gridh * gridw]
xcoords_start = obj['xcoords_start'][:gridh * gridw]
ycoords_start = obj['ycoords_start'][:gridh * gridw]
diff3d = np.transpose(obj['diffraction'][:, :, :gridh * gridw], [2, 0, 1])
probeGuess = obj['probeGuess']

# Initialize PtychoData with the prepared data
ptycho_data = loader.PtychoData(xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess)

dset = loader.get_neighbor_diffraction_and_positions(ptycho_data, K=7,
    nsamples=1)
#dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
#    xcoords_start, ycoords_start, K = 7, nsamples = 1)


# Images are amplitude, not intensity
X_full = dset['diffraction']
def normalize_data(X_full, N):
    X_full_norm = ((N / 2)**2) / np.mean(tf.reduce_sum(X_full**2, axis=[1, 2]))
    return X_full_norm * X_full

X_full = normalize_data(X_full, N)
print('neighbor-sampled diffraction shape', X_full.shape)

# TODO refactor actual / nominal positions
key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'
#key_coords_offsets = 'coords_offsets'
#key_coords_relative = 'coords_relative'
coords_true = dset[key_coords_relative]
coords_nominal = dset[key_coords_relative]


# Replace the call to get_splits with split_data
# This line should be removed or commented out since it's causing the error
# X, coords_nominal, coords_true = split_data(X_full, coords_nominal, coords_true, train_frac, which)

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

def load(which, **kwargs):
    global_offsets = split_tensor(dset[key_coords_offsets], which)
    # Define coords_nominal and coords_true before calling split_data
    coords_nominal = dset[key_coords_relative]
    coords_true = dset[key_coords_relative]
    X, coords_nominal, coords_true = loader.split_data(X_full, coords_nominal, coords_true, train_frac, which)

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
        'X': X,
        'Y_I': Y_I,
        'Y_phi': Y_phi,
        'norm_Y_I': norm_Y_I,
        'YY_full': YY_full,
        'coords': (coords_nominal, coords_true),
        'nn_indices': dset['nn_indices']
    }

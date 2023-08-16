import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import loader
from . import datasets
from . import params as cfg

train_frac = .6
N = 64
gridsize = 2
gridh, gridw = 32, 32

np.random.seed(7)

obj = np.load('../datasets/Run1084_recon3_postPC_shrunk.npz')
#obj = np.load('../datasets/Run1084_recon3_postPC_shrunk_lowpass.npz')

print('raw diffraction shape', obj['diffraction'].shape)

xcoords = obj['xcoords'][:gridh * gridw]
ycoords = obj['ycoords'][:gridh * gridw]

xcoords_start = obj['xcoords_start'][:gridh * gridw]
ycoords_start = obj['ycoords_start'][:gridh * gridw]

# TODO normalize the raw data
diff3d = np.transpose(obj['diffraction'][:, :, :gridh * gridw],
            [2, 0, 1])

dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
    xcoords_start, ycoords_start, K = 7, nsamples = 20)

X_full = np.sqrt(dset['diffraction'])
print('neighbor-sampled diffraction shape', X_full.shape)

#coords_true = dset['coords_start_relative']
coords_true = dset['coords_relative']
coords_nominal = dset['coords_start_relative']

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

def load(which, **kwargs):
    X, coords_nominal, coords_true = get_splits(which, **kwargs)

#    # Scale so object amplitude is of order unity
#    nphoton = np.mean(datasets.count_photons(X))
#    L2_perpix = nphoton / ((N / 2)**2)
#    norm_Y_I = tf.convert_to_tensor(np.sqrt(L2_perpix))
    norm_Y_I = datasets.scale_nphotons(X)

    X = tf.convert_to_tensor(X)
    coords_nominal = tf.convert_to_tensor(coords_nominal)
    coords_true = tf.convert_to_tensor(coords_true)
    Y_I = tf.convert_to_tensor(np.ones_like(X))
    Y_phi = tf.convert_to_tensor(np.zeros_like(X))
    YY_full = None

    return (X, Y_I, Y_phi,
        norm_Y_I, YY_full, norm_Y_I,
        (coords_nominal, coords_true))

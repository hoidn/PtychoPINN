import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import loader
from .loader import key_coords_offsets, key_coords_relative
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

# This dataset uses y, x  ordering so we swap the coordinates to match the
# program's expectation
xcoords = obj['xcoords'][:gridh * gridw]
ycoords = obj['ycoords'][:gridh * gridw]
xcoords_start = obj['xcoords_start'][:gridh * gridw]
ycoords_start = obj['ycoords_start'][:gridh * gridw]
diff3d = np.transpose(obj['diffraction'][:, :, :gridh * gridw], [2, 0, 1])
probeGuess = obj['probeGuess']
objectGuess = obj['objectGuess']

# Initialize RawData with the prepared data
scan_index = np.zeros(diff3d.shape[0], dtype=int)  # Array of zeros indicating a single scan index
ptycho_data = loader.RawData(xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index, objectGuess = objectGuess)

# half the data
train_size = 512
ptycho_data_train = loader.RawData(xcoords[:train_size], ycoords[:train_size], xcoords_start[:train_size], ycoords_start[:train_size], diff3d[:train_size], probeGuess, scan_index[:train_size], objectGuess = objectGuess)

dset = loader.get_neighbor_diffraction_and_positions(ptycho_data, N, K=7,
    nsamples=1)
#dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
#    xcoords_start, ycoords_start, K = 7, nsamples = 1)
X_full = dset['X_full']

# TODO refactor actual / nominal positions
coords_true = dset[key_coords_relative]
coords_nominal = dset[key_coords_relative]

def get_data(**kwargs):
    return dset, train_frac

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pkg_resources

from . import loader
from .loader import key_coords_offsets, key_coords_relative
from ptycho import diffsim as datasets

train_frac = .5
N = 64
gridh, gridw = 32, 32

np.random.seed(7)


def get_data(**kwargs):
    return dset, train_frac

import numpy as np
from . import loader
import pkg_resources

def load_ptycho_data(file_path, train_size=512):
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
    ptycho_data = loader.RawData(xcoords, ycoords, xcoords_start, ycoords_start,
                                 diff3d, probeGuess, scan_index, objectGuess=objectGuess)

    # Create RawData object for the training subset
    ptycho_data_train = loader.RawData(xcoords[:train_size], ycoords[:train_size],
                                       xcoords_start[:train_size], ycoords_start[:train_size],
                                       diff3d[:train_size], probeGuess,
                                       scan_index[:train_size], objectGuess=objectGuess)

    return ptycho_data, ptycho_data_train, data

data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
ptycho_data, ptycho_data_train, obj = load_ptycho_data(data_file_path)
print('raw diffraction shape', obj['diffraction'].shape)
# TODO cast to complex64?
probeGuess = obj['probeGuess']
objectGuess = obj['objectGuess']
#dset = loader.get_neighbor_diffraction_and_positions(ptycho_data, N, K=6,
#    nsamples=4)

#dset = loader.get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
#    xcoords_start, ycoords_start, K = 7, nsamples = 1)
#X_full = dset['X_full']

## TODO refactor actual / nominal positions

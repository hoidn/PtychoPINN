import numpy as np
import pkg_resources

from .loader import load_xpp_npz as load_ptycho_data

train_frac = .5
N = 64
gridh, gridw = 32, 32

np.random.seed(7)

def get_data(**kwargs):
    return dset, train_frac


data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
ptycho_data, ptycho_data_train, obj = load_ptycho_data(data_file_path)
print('raw diffraction shape', obj['diffraction'].shape)
# TODO cast to complex64?
probeGuess = obj['probeGuess']
objectGuess = obj['objectGuess']

## TODO refactor actual / nominal positions

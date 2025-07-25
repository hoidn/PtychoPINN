"""
Legacy Experimental Data Processing for 2019 Ptychographic Datasets

This module contains hardcoded data processing routines specifically designed for historical 
experimental ptychographic datasets from 2019. It provides specialized utilities for loading, 
transforming, and analyzing specific experimental data files with fixed dimensions and 
processing parameters.

Primary Functions:
  - get_full_experimental(): Cached loader for train/test splits from experimental datasets
  - reconstruct_object(): Inverse patch reconstruction from 4D object tensor data
  - cross_image(): Cross-correlation computation for shift analysis between scan positions
  - augment_inversion(): Data augmentation via spatial/phase inversion transformations
  - stack(): Array combination utility for reshaping into model-compatible formats

Architecture Integration:
  This module operates as a specialized data source adapter, providing legacy experimental 
  data in formats compatible with the modern PtychoPINN training pipeline. It bridges 
  historical experimental datasets with current model architectures through format 
  standardization and preprocessing.

Data Processing Pipeline:
  ```python
  # Load experimental dataset splits
  train_I, train_phi = get_full_experimental('train')    # Amplitude and phase arrays
  test_I, test_phi = get_full_experimental('test')       # Test set equivalents
  
  # Perform shift analysis
  correlation = cross_image(amp[0, 0], amp[1, 0])        # Inter-position correlation
  
  # Augment training data
  aug_I, aug_phi = augment_inversion(train_I, train_phi) # Spatial/phase augmentation
  ```

Experimental Dataset Specifications:
  - Source: 2019 ptychographic reconstruction experiments (20191008_39)
  - Diffraction patterns: 64x64 pixels (downsampled from original 128x128)
  - Training lines: 100 scan lines (configurable via nlines parameter)
  - Test lines: 60 scan lines (configurable via nltest parameter)
  - Full reconstruction: 544x544 pixels with 3-pixel overlap (offset_experimental)

Legacy System Dependencies:
  - Hardcoded file paths pointing to specific experimental data locations
  - Fixed dimensional parameters (N=64, train_size=272, test_size=248)
  - Dependencies on ptycho.tf_helper for inverse patch operations
  - Uses ptycho.misc caching decorators for expensive reconstruction operations

Data Format Specifications:
  Input formats:
    - Diffraction data: (n_lines, n_positions, 64, 64) amplitude arrays
    - Real space data: Complex arrays with separate amplitude/phase components
  Output formats:
    - Stacked arrays: (-1, N, N, 1) for model compatibility
    - Reconstructed objects: (1, height, width, 1) full-field images

Development Status:
  This module contains experimental code frozen at a specific point in development. 
  Functions include hardcoded parameters and file paths specific to 2019 experimental 
  campaigns. Modern workflows should use the general-purpose data loading pipeline 
  in ptycho.raw_data and ptycho.loader instead.

Integration with Modern System:
  ```python
  # Legacy experimental data integration
  from ptycho.experimental import get_full_experimental
  from ptycho.data_preprocessing import prepare_training_data
  
  # Load legacy data
  exp_I, exp_phi = get_full_experimental('train')
  
  # Convert to modern format
  training_data = prepare_training_data(exp_I, exp_phi, config)
  ```

Notes:
  - Contains module-level execution code that runs on import
  - Hardcoded paths may require modification for different environments
  - Cross-correlation functions assume grayscale image processing
  - Caching decorators improve performance for repeated data access
  - Compatible with both coordinate-based and grid-based processing modes
"""
from skimage.transform import resize
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sys

from . import tf_helper as hh

path = '.'

sys.path.append(path)
sys.path.append('PtychoNN/TF2/')

N = 64
### Read experimental diffraction data and reconstructed images

data_diffr = np.load(path+'/PtychoNN/data/20191008_39_diff.npz')['arr_0']
data_diffr.shape

data_diffr_red = np.zeros((data_diffr.shape[0],data_diffr.shape[1],64,64), float)
for i in tqdm(range(data_diffr.shape[0])):
    for j in range(data_diffr.shape[1]):
        data_diffr_red[i,j] = resize(data_diffr[i,j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
        data_diffr_red[i,j] = np.where(data_diffr_red[i,j]<3,0,data_diffr_red[i,j])

real_space = np.load(path+'/PtychoNN/data/20191008_39_amp_pha_10nm_full.npy')
amp = np.abs(real_space)
ph = np.angle(real_space)
amp.shape

### Split data and then shuffle

nlines = 100 #How many lines of data to use for training?
nltest = 60 #How many lines for the test set?
tst_strt = amp.shape[0]-nltest #Where to index from
print(tst_strt)
train_size = 272
test_size = 248

def stack(a1, a2):
    return np.array((a1, a2)).reshape((-1, N, N, 1))

def augment_inversion(Y_I_train, Y_phi_train):
    phi = stack(Y_phi_train, -Y_phi_train)
    return stack(Y_I_train, Y_I_train[:, ::-1, ::-1, :]), stack(Y_phi_train, -Y_phi_train)

def reconstruct_object(data4d, scan_grid_offset):
    """
    Given a 4d object patches, reconstruct the whole object
    """
    return hh.extract_patches_inverse(
       data4d.reshape((data4d.shape[0], data4d.shape[1], -1))[None, ...],
       N, True, gridsize = data4d.shape[0],
       offset = scan_grid_offset)

from ptycho.misc import memoize_disk_and_memory
@memoize_disk_and_memory
def get_full_experimental(which):
    """
    Returns (normalized) amplitude and phase for n generated objects
    """
    inverted_patches_I = reconstruct_object(amp, offset_experimental)
    inverted_patches_phi = reconstruct_object(ph, offset_experimental)
    print('GROUND TRUTH FULL SHAPE:', inverted_patches_I.shape)
    if which == 'train':
        YY_I = inverted_patches_I[:, :train_size, :train_size, :]
        YY_phi = inverted_patches_phi[:, :train_size, :train_size, :]
    elif which == 'test':
        YY_I = inverted_patches_I[:, -test_size:, -test_size:, :]
        YY_phi = inverted_patches_phi[:, -test_size:, -test_size:, :]
    else:
        raise ValueError
    return YY_I, YY_phi


X_train = data_diffr_red[:nlines,:].reshape(-1,N,N)[:,:,:,np.newaxis]
X_test = data_diffr_red[tst_strt:,tst_strt:].reshape(-1,N,N)[:,:,:,np.newaxis]
Y_I_train = amp[:nlines,:]#.reshape(-1,h,w)[:,:,:,np.newaxis]
Y_I_test = amp[tst_strt:,tst_strt:]#.reshape(-1,h,w)[:,:,:,np.newaxis]
Y_phi_train = ph[:nlines,:]#.reshape(-1,h,w)[:,:,:,np.newaxis]
Y_phi_test = ph[tst_strt:,tst_strt:]#.reshape(-1,h,w)[:,:,:,np.newaxis]

ntrain = X_train.shape[0]*X_train.shape[1]
ntest = X_test.shape[0]*X_test.shape[1]

print(X_train.shape, X_test.shape)


tmp1, tmp2 = Y_I_train, Y_I_test

img = np.zeros((544, 544), dtype = 'float32')[None, ..., None]
offset_experimental = 3

## Recover shift between scan points
def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = im1#np.sum(im1.astype('float'), axis=2)
    im2_gray = im2#np.sum(im2.astype('float'), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

cross = cross_image(amp[0, 0], amp[1, 0])
ref = cross_image(amp[0, 0], amp[0, 0])

cmax = lambda cross: np.array(np.where(cross.ravel()[np.argmax(cross)] == cross))

plt.imshow(cross)

cmax(cross), cmax(cross) - cmax(ref)

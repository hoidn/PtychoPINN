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

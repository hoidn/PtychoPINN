from . import datasets
from . import fourier as f
from . import params
from sklearn.utils import shuffle
import argparse
import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from ptycho import params as p
import os

now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
date_time = date_time.replace('/', '-').replace(':', '.').replace(', ', '-')
out_prefix = 'outputs/{}/'.format(date_time)

os.makedirs(out_prefix, exist_ok=True)

##if __name__ == '__main__':
#parser = argparse.ArgumentParser(
#                    #prog = 'ProgramName',
#                    #description = 'What the program does',
#                    #epilog = 'Text at the bottom of help'
#                    )
#parser.add_argument('offset', type = int)           # positional argument
##parser.add_argument('-c', '--count')      # option that takes a value
##parser.add_argument('-v', '--verbose',
##                    action='store_true')  # on/off flag
#args = parser.parse_args(sys.argv)
#
## TODO cleaner way of setting params
#offset = params.cfg['offset'] = args.offset
##else:
##    # offset between neighboring scan points, in pixels
##    offset = params.cfg['offset'] = 4

offset = params.cfg['offset']
h = w = N = params.cfg['N'] = 64
gridsize = params.cfg['gridsize'] = 2

nepochs = params.cfg['nepochs'] = 60
batch_size = params.cfg['batch_size'] = 16
train_probe = False

def normed_ff_np(arr):
    return (f.fftshift(np.absolute(f.fft2(np.array(arr)))) / np.sqrt(h * w))


matplotlib.rcParams['font.size'] = 12

# TODO parameterize
#filt = f.lowpass_g(.55, np.ones(N), sym = True)
filt = f.lowpass_g(.7, np.ones(N), sym = True)
#filt = f.lowpass_g(.9, np.ones(N), sym = True)

probe = f.gf(((np.einsum('i,j->ij', filt, filt)) > .5).astype(float), 1) + 1e-9
probe_small = probe[16:-16, 16:-16]
tprobe = (tf.convert_to_tensor(probe, tf.float32)[..., None])
tprobe_small = (tf.convert_to_tensor(probe_small, tf.float32)[..., None])

bigoffset = (gridsize - 1) * offset + N // 2
big_gridsize = 10
bigN = params.params()['bigN']
size = bigoffset * (big_gridsize - 1) + bigN

bigoffset = params.cfg['bigoffset'] = ((gridsize - 1) * offset + N // 2) // 2

# simulate data
np.random.seed(1)
X_train, Y_I_train, Y_phi_train, intensity_scale, YY_I_train_full, _  = datasets.mk_simdata(9, size, probe)
params.cfg['intensity_scale'] = intensity_scale

np.random.seed(2)
X_test, Y_I_test, Y_phi_test, _, YY_I_test_full, norm_Y_I_test = datasets.mk_simdata(3, size, probe, intensity_scale)

# TODO shuffle should be after flatten
X_train, Y_I_train, Y_phi_train = shuffle(X_train.numpy(), Y_I_train.numpy(), Y_phi_train.numpy(), random_state=0)

(Y_I_test).shape, Y_I_train.shape

print(np.linalg.norm(X_train[0]) /  np.linalg.norm(Y_I_train[0]))

# inversion symmetry
assert np.isclose(normed_ff_np(Y_I_train[0, :, :, 0]),
            tf.math.conj(normed_ff_np(Y_I_train[0, ::-1, ::-1, 0])), atol = 1e-6).all()

print('nphoton',np.log10(np.sum((X_train[:, :, :] * intensity_scale)**2, axis = (1, 2))).mean())

#i = 1
#plt.imshow(np.log(normed_ff_np
#                  (np.array(hh.combine_complex(Y_I_train, Y_phi_train))[0, :, :, 0])), cmap = 'jet')
#plt.colorbar()

#plt.imshow(np.log(X_train[0, :, :, 0]), cmap = 'jet')
#plt.colorbar()

tmp = X_train.mean(axis = (0, 3))
probe_fif = np.absolute(f.fftshift(f.ifft2(f.ifftshift(tmp))))[N // 2, :]

# variance increments of a slice down the middle
d_second_moment = (probe_fif / probe_fif.sum()) * ((np.arange(N) - N // 2)**2)
probe_sigma_guess = np.sqrt(d_second_moment.sum())


centered_indices = np.arange(N) - N // 2 + .5
x, y = np.meshgrid(centered_indices, centered_indices)
d = np.sqrt(x*x+y*y)
mu = 0.
probe_mask = (d < N // 4)
probe_guess = np.exp(-( (d-mu)**2 / ( 2.0 * probe_sigma_guess**2 ) ) )
probe_guess *= probe_mask
probe_guess *= (np.sum(tprobe) / np.sum(probe_guess))

t_probe_guess = tf.convert_to_tensor(probe_guess, tf.float32)[..., None]

if train_probe:
    params.cfg['probe'] = t_probe_guess
    params.cfg['probe.trainable'] = True
else:
    params.cfg['probe'] = tprobe
    params.cfg['probe.trainable'] = False

params.cfg['probe_mask'] = tf.convert_to_tensor(probe_mask, tf.complex64)[..., None]

from ptycho import params
from ptycho import model
#reload(hh)
#reload(model.hh)
#reload(model)

#Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

#plt.imshow(np.absolute(model.autoencoder.variables[-1]), cmap = 'jet')
#plt.colorbar()
history = model.train(nepochs, X_train, Y_I_train)#tboard_callback
b, a, reg, L2_error = model.autoencoder.predict([X_test * model.params()['intensity_scale']])

from ptycho import baselines as bl
from ptycho.params import params

bigoffset = params()['bigoffset']
bordersize = (N - bigoffset / 2) / 2
# Amount to trim from NxN reconstruction patches
borderleft = int(np.ceil(bordersize))
borderright = int(np.floor(bordersize))

# Amount to trim from the ground truth object
clipsize = (bordersize + ((gridsize - 1) * offset) // 2)
clipleft = int(np.ceil(clipsize))
clipright = int(np.floor(clipsize))
# TODO cleanup
def stitch(b, norm_Y_I_test = 1,
           nsegments = (size - bigN) // (bigoffset // 2) + 1,
           norm = True):
    if norm:
        img_recon = np.reshape((norm_Y_I_test * np.absolute(b)), (-1, nsegments,
                                                              nsegments, 64, 64, 1))
    else:
        img_recon = np.reshape((norm_Y_I_test * np.absolute(b)), (-1, nsegments,
                                                              nsegments, 64, 64, 1))
    img_recon = img_recon[:, :, :, borderleft: -borderright, borderleft: -borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched

#plt.rcParams["figure.figsize"] = (10, 10)
## ground truth
#bordersize = N // 2 - bigoffset // 4
#clipsize = (bordersize + ((gridsize - 1) * offset) // 2)
#plt.imshow(_Y_I_test_full[0, clipleft: -clipright, clipleft: -clipright], interpolation = 'none',
#          cmap = 'jet')
#vmin = np.min(_Y_I_test_full[0, clipleft: -clipright, clipleft: -clipright])
#vmax = np.max(_Y_I_test_full[0, clipleft: -clipright, clipleft: -clipright])

# reconstructed
stitched = stitch(b, norm_Y_I_test,
                  #nsegments=37,
                  norm = False)
#plt.imshow(stitched[0], interpolation = 'none', cmap = 'jet')
plt.imsave(out_prefix + 'recon.png', stitched[0][:, :, 0], cmap = 'jet')
plt.imsave(out_prefix + 'orig.png', YY_I_test_full[0, clipleft: -clipright, clipleft: -clipright, 0],
          cmap = 'jet')

with open(out_prefix + '/history.dill', 'wb') as file_pi:
    dill.dump(history.history, file_pi)

with open(out_prefix + '/params.dill', 'wb') as f:
    dill.dump(p.cfg, f)

#with open(out_prefix + '/test_data.dill', 'wb') as f:
#    dill.dump(
#        {'YY_I_test_full': YY_I_test_full,
#        'Y_I_test': Y_I_test,
#        'Y_phi_test': Y_phi_test,
#        'X_test': X_test}, f)

model.autoencoder.save('{}.h5'.format(out_prefix + 'wts'))

#with open('/trainHistoryDict', "rb") as file_pi:
#    history = pickle.load(file_pi)

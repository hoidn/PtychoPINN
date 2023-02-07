from sklearn.utils import shuffle
import numpy as np
from ptycho import params
from ptycho import datasets
from ptycho import fourier as f
import tensorflow as tf

offset = params.cfg['offset']

h = w = N = params.cfg['N'] = 64
gridsize = params.cfg['gridsize'] = 2

nepochs = params.cfg['nepochs']
batch_size = params.cfg['batch_size'] = 16
#train_probe = False

# TODO need to enforce that configs are set before this
from ptycho import probe

def normed_ff_np(arr):
    return (f.fftshift(np.absolute(f.fft2(np.array(arr)))) / np.sqrt(h * w))

bigoffset = (gridsize - 1) * offset + N // 2
# TODO move to configs
big_gridsize = 10
bigN = params.params()['bigN']
size = bigoffset * (big_gridsize - 1) + bigN

bigoffset = params.cfg['bigoffset'] = ((gridsize - 1) * offset + N // 2) // 2

# simulate data
np.random.seed(1)
(X_train, Y_I_train, Y_phi_train,
    intensity_scale, YY_I_train_full, _,
    (coords_train_nominal, coords_train_true)) =\
    datasets.mk_simdata(9, size, probe.probe, jitter_scale = 0.)
params.cfg['intensity_scale'] = intensity_scale

np.random.seed(2)
(X_test, Y_I_test, Y_phi_test,
    _, YY_I_test_full, norm_Y_I_test,
    (coords_test_nominal, coords_test_true)) =\
    datasets.mk_simdata(3, size, probe.probe, intensity_scale, jitter_scale = 0.)

# TODO shuffle should be after flatten
X_train, Y_I_train, Y_phi_train = shuffle(X_train.numpy(), Y_I_train.numpy(), Y_phi_train.numpy(), random_state=0)

(Y_I_test).shape, Y_I_train.shape

print(np.linalg.norm(X_train[0]) /  np.linalg.norm(Y_I_train[0]))

# inversion symmetry
assert np.isclose(normed_ff_np(Y_I_train[0, :, :, 0]),
            tf.math.conj(normed_ff_np(Y_I_train[0, ::-1, ::-1, 0])), atol = 1e-6).all()

print('nphoton',np.log10(np.sum((X_train[:, :, :] * intensity_scale)**2, axis = (1, 2))).mean())


if params.params()['probe.trainable']:
    params.cfg['probe'] = probe.mk_probe_guess(X_train)
else:
    params.cfg['probe'] = probe.tprobe


params.cfg['probe_mask'] = tf.convert_to_tensor(probe.probe_mask, tf.complex64)[..., None]

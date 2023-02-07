import tensorflow as tf
import numpy as np
from . import fourier as f
from . import params

N = params.cfg['N']
# TODO parameterize
#filt = f.lowpass_g(.55, np.ones(N), sym = True)
filt = f.lowpass_g(.7, np.ones(N), sym = True)
#filt = f.lowpass_g(.9, np.ones(N), sym = True)

probe = f.gf(((np.einsum('i,j->ij', filt, filt)) > .5).astype(float), 1) + 1e-9
probe_small = probe[16:-16, 16:-16]
tprobe = (tf.convert_to_tensor(probe, tf.float32)[..., None])
tprobe_small = (tf.convert_to_tensor(probe_small, tf.float32)[..., None])

centered_indices = np.arange(N) - N // 2 + .5
x, y = np.meshgrid(centered_indices, centered_indices)
d = np.sqrt(x*x+y*y)
mu = 0.

# TODO is this a good choice for the probe mask?
probe_mask = (d < N // 4)
probe_mask = tf.convert_to_tensor(probe_mask, tf.complex64)[..., None]

def mk_probe_guess(X_train):
    tmp = X_train.mean(axis = (0, 3))
    probe_fif = np.absolute(f.fftshift(f.ifft2(f.ifftshift(tmp))))[N // 2, :]

    # variance increments of a slice down the middle
    d_second_moment = (probe_fif / probe_fif.sum()) * ((np.arange(N) - N // 2)**2)
    probe_sigma_guess = np.sqrt(d_second_moment.sum())
    probe_guess = np.exp(-( (d-mu)**2 / ( 2.0 * probe_sigma_guess**2 ) ) )
    probe_guess *= probe_mask
    probe_guess *= (np.sum(tprobe) / np.sum(probe_guess))

    t_probe_guess = tf.convert_to_tensor(probe_guess, tf.float32)[..., None]
    return t_probe_guess

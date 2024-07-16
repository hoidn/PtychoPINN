import tensorflow as tf
import numpy as np
from . import fourier as f
from . import params

N = params.cfg['N']

# most common value in testing is .7. Sometimes also .55 or .9
default_probe_scale = params.cfg['default_probe_scale']

filt = f.lowpass_g(default_probe_scale, np.ones(N), sym = True)

def get_default_probe(fmt = 'tf'):
    probe_np = f.gf(((np.einsum('i,j->ij', filt, filt)) > .5).astype(float), 1) + 1e-9
    if fmt == 'np':
        return probe_np
    elif fmt == 'tf':
        return (tf.convert_to_tensor(probe_np, tf.float32)[..., None])
    else:
        raise ValueError

def get_probe(fmt = 'tf'):
    probe_tf = params.get('probe')
    assert len(probe_tf.shape) == 3
    if fmt == 'np':
        return np.array(probe_tf)[:, :, 0]
    elif fmt == 'tf':
        return probe_tf
    else:
        raise ValueError

def to_np(probe):
    assert len(probe.shape) == 3
    return np.array(probe[:, :, 0])

def get_squared_distance():
    """
    Return array of distances from the center
    """
    centered_indices = np.arange(N) - N // 2 + .5
    x, y = np.meshgrid(centered_indices, centered_indices)
    d = np.sqrt(x*x+y*y)
    return d

probe_mask_real = (get_squared_distance() < N // 4)[..., None]
# TODO adaptive probe mask?
def get_probe_mask():
    probe_mask = tf.convert_to_tensor(probe_mask_real, tf.complex64)
    return tf.convert_to_tensor(probe_mask, tf.complex64)[..., None]

def set_probe(probe):
    # TODO optimize this scaling
    mask = tf.cast(get_probe_mask(), probe.dtype)
    probe_scale = params.get('probe_scale')
    tamped_probe = mask * probe
    norm = float(probe_scale * tf.reduce_mean(tf.math.abs(tamped_probe)))
    params.set('probe', probe / norm)

def set_probe_guess(X_train = None, probe_guess = None):
    if probe_guess is None:
        mu = 0.
        tmp = X_train.mean(axis = (0, 3))
        probe_fif = np.absolute(f.fftshift(f.ifft2(f.ifftshift(tmp))))[N // 2, :]

        # variance increments of a slice down the middle
        d_second_moment = (probe_fif / probe_fif.sum()) * ((np.arange(N) - N // 2)**2)
        probe_sigma_guess = np.sqrt(d_second_moment.sum())
        probe_guess = np.exp(-( (get_squared_distance() - mu)**2 / ( 2.0 * probe_sigma_guess**2 )))[..., None]\
            + 1e-9
        probe_guess *= probe_mask_real
        probe_guess *= (np.sum(get_default_probe()) / np.sum(probe_guess))
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.float32)
    else:
        probe_guess = probe_guess[..., None]
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.complex64)

    #params.set('probe', t_probe_guess)
    set_probe(t_probe_guess)
    return t_probe_guess

# This should be more explicit
params.set('probe_mask', get_probe_mask())

# TODO this needs to be called
def set_default_probe():
    set_probe_guess(probe_guess = get_default_probe(fmt = 'np'))

import tensorflow as tf
import numpy as np
from . import fourier as f
from . import params

def get_lowpass_filter(scale, N):
    return f.lowpass_g(scale, np.ones(N), sym=True)

def get_default_probe(N, scale, fmt='tf'):
    filt = get_lowpass_filter(scale, N)
    probe_np = f.gf(((np.einsum('i,j->ij', filt, filt)) > .5).astype(float), 1) + 1e-9
    if fmt == 'np':
        return probe_np
    elif fmt == 'tf':
        return tf.convert_to_tensor(probe_np, tf.float32)[..., None]
    else:
        raise ValueError("Invalid format specified")

def get_probe(params):
    probe_tf = params.get('probe')
    assert len(probe_tf.shape) == 3
    return probe_tf

def to_np(probe):
    assert len(probe.shape) == 3
    return np.array(probe[:, :, 0])

def get_squared_distance(N):
    centered_indices = np.arange(N) - N // 2 + .5
    x, y = np.meshgrid(centered_indices, centered_indices)
    return np.sqrt(x*x+y*y)

def get_probe_mask(N):
    probe_mask_real = (get_squared_distance(N) < N // 4)[..., None]
    probe_mask = tf.convert_to_tensor(probe_mask_real, tf.complex64)
    return tf.convert_to_tensor(probe_mask, tf.complex64)[..., None]

def set_probe(probe, params):
    # This function still modifies global state
    mask = tf.cast(get_probe_mask(params.get('N')), probe.dtype)
    probe_scale = params.get('probe_scale')
    tamped_probe = mask * probe
    norm = float(probe_scale * tf.reduce_mean(tf.math.abs(tamped_probe)))
    params.set('probe', probe / norm)

def set_probe_guess(params, X_train=None, probe_guess=None):
    # This function still modifies global state
    N = params.get('N')
    if probe_guess is None:
        if X_train is not None:
            mu = 0.
            tmp = X_train.mean(axis=(0, 3))
            probe_fif = np.absolute(f.fftshift(f.ifft2(f.ifftshift(tmp))))[N // 2, :]
            d_second_moment = (probe_fif / probe_fif.sum()) * ((np.arange(N) - N // 2)**2)
            probe_sigma_guess = np.sqrt(d_second_moment.sum())
            probe_guess = np.exp(-((get_squared_distance(N) - mu)**2 / (2.0 * probe_sigma_guess**2)))[..., None] + 1e-9
            probe_mask_real = (get_squared_distance(N) < N // 4)[..., None]
            probe_guess *= probe_mask_real
            probe_guess *= (np.sum(get_default_probe(N, params.get('default_probe_scale'), 'np')) / np.sum(probe_guess))
        else:
            probe_guess = get_default_probe(N, params.get('default_probe_scale'), 'np')
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.float32)
    else:
        probe_guess = probe_guess[..., None]
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.complex64)

    set_probe(t_probe_guess, params)
    return t_probe_guess

"""Probe initialization and manipulation for ptychographic reconstruction.

Manages the scanning beam (probe) used in experiments, including creation of
idealized disk-shaped probes, automatic estimation from data, and masking.

Example:
    probe = get_default_probe(N=64, fmt='tf')  # Default disk probe
    set_probe_guess(X_train=data, probe_guess=exp_probe)  # From data
"""
import tensorflow as tf
import numpy as np
from . import fourier as f
from . import params

def get_lowpass_filter(scale, N):
    return f.lowpass_g(scale, np.ones(N), sym=True)

def get_default_probe(N, fmt='tf'):
    scale = params.cfg['default_probe_scale']
    filt = get_lowpass_filter(scale, N)
    probe_np = f.gf(((np.einsum('i,j->ij', filt, filt)) > .5).astype(float), 1) + 1e-9
    if fmt == 'np':
        return probe_np
    elif fmt == 'tf':
        return tf.convert_to_tensor(probe_np, tf.complex64)[..., None]
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

def get_probe_mask_real(N):
    return (get_squared_distance(N) < N // 4)[..., None]

def get_probe_mask(N):
    probe_mask_real = get_probe_mask_real(N)
    probe_mask = tf.convert_to_tensor(probe_mask_real, tf.complex64)
    #return tf.convert_to_tensor(probe_mask, tf.complex64)[..., None]
    return tf.convert_to_tensor(probe_mask, tf.complex64)

def set_probe(probe):
    assert len(probe.shape) == 3 or len(probe.shape) == 4
    assert probe.shape[0] == probe.shape[1]
    assert probe.shape[-1] == 1
    if len(probe.shape) == 4:
        assert probe.shape[-2] == 1
        probe = probe[:, :, :]
        print('coercing probe shape to 3d')

    # This function still modifies global state
    mask = tf.cast(get_probe_mask(params.get('N')), probe.dtype)
    probe_scale = params.get('probe_scale')
    tamped_probe = mask * probe
    norm = float(probe_scale * tf.reduce_mean(tf.math.abs(tamped_probe)))
    params.set('probe', probe / norm)

def set_probe_guess(X_train = None, probe_guess = None):
    N = params.get('N')
    if probe_guess is None:
        mu = 0.
        tmp = X_train.mean(axis = (0, 3))
        probe_fif = np.absolute(f.fftshift(f.ifft2(f.ifftshift(tmp))))[N // 2, :]

        # variance increments of a slice down the middle
        d_second_moment = (probe_fif / probe_fif.sum()) * ((np.arange(N) - N // 2)**2)
        probe_sigma_guess = np.sqrt(d_second_moment.sum())
        probe_guess = np.exp(-( (get_squared_distance(N) - mu)**2 / ( 2.0 * probe_sigma_guess**2 )))[..., None]\
            + 1e-9
        probe_guess *= get_probe_mask_real(N)
        probe_guess *= (np.sum(get_default_probe(N)) / np.sum(probe_guess))
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.float32)
    else:
        if probe_guess.ndim not in [2, 3]:
            raise ValueError("probe_guess must have 2 or 3 dimensions")
        if probe_guess.ndim == 2:
            probe_guess = probe_guess[..., None]
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.complex64)

    #params.set('probe', t_probe_guess)
    set_probe(t_probe_guess)
    return t_probe_guess

def set_default_probe():
    """
    use an idealized disk shaped probe. Only for simulated data workflows.
    """
    set_probe(get_default_probe(params.get('N'), fmt = 'tf'))

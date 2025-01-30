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
    # Ensure probe has shape [num_probes, H, W, 1]
    if len(probe.shape) == 4 and probe.shape[-1] == 1:
        # Shape is correct
        pass
    else:
        raise ValueError("Invalid probe shape")

    # Apply mask and normalize each probe individually
    mask = tf.cast(get_probe_mask(params.get('N')), probe.dtype)  # Shape: [H, W, 1]
    mask = tf.expand_dims(mask, axis=0)  # Shape: [1, H, W, 1]
    mask = tf.tile(mask, [tf.shape(probe)[0], 1, 1, 1])  # Shape: [num_probes, H, W, 1]
    masked_probes = mask * probe

    # Normalize each probe individually
    probe_scale = params.get('probe_scale')
    # Calculate norm and cast to complex dtype
    norm = tf.cast(
        probe_scale * tf.reduce_mean(tf.math.abs(masked_probes), axis=[1, 2, 3], keepdims=True),
        dtype=probe.dtype
    )
    probe_normalized = probe / norm

    params.set('probe', probe_normalized)

def set_probe_guess(X_train=None, probe_guess=None):
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
        # Convert directly to complex64 and ensure [1, H, W, 1] shape
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.complex64)
        t_probe_guess = t_probe_guess[tf.newaxis, ..., tf.newaxis]  # Add batch and channel dimensions
    else:
        # Ensure probe_guess has shape [num_probes, H, W, 1]
        if isinstance(probe_guess, np.ndarray):
            if probe_guess.ndim == 2:
                # [H, W] -> [1, H, W, 1]
                probe_guess = probe_guess[np.newaxis, ..., np.newaxis]
            elif probe_guess.ndim == 3:
                if probe_guess.shape[-1] != 1:
                    # [num_probes, H, W] -> [num_probes, H, W, 1]
                    probe_guess = probe_guess[..., np.newaxis]
            elif probe_guess.ndim == 4 and probe_guess.shape[-1] == 1:
                # Shape is already correct
                pass
            else:
                raise ValueError("Invalid shape for probe_guess")
        
        t_probe_guess = tf.convert_to_tensor(probe_guess, tf.complex64)

    set_probe(t_probe_guess)
    return t_probe_guess

def set_default_probe():
    """
    use an idealized disk shaped probe. Only for simulated data workflows.
    """
    set_probe(get_default_probe(params.get('N'), fmt = 'tf'))

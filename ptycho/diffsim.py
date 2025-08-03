"""
Diffraction pattern simulation via probe-object interaction.

This module orchestrates the complete ptychographic forward model, taking
object patches and probe functions through the physics pipeline to generate
realistic diffraction patterns. It serves as the primary simulation engine
for data generation and physics-informed model training.

Architecture Role:
    Object/Probe -> diffsim.py (orchestrator) -> Diffraction Amplitudes
    
    This module coordinates the physics simulation pipeline:
    1. Probe-object interaction (multiplication)
    2. Wave propagation (if thick object)
    3. Far-field propagation (Fourier transform)
    4. Optional noise addition

Public Interface:
    `illuminate_and_diffract(Y_I_flat, Y_phi_flat, probe, intensity_scale=None)`
        - Purpose: Simulate the complete forward ptychographic process.
        - Algorithm: Illuminates object patches with probe, propagates to
          far field, and returns diffraction amplitudes.
        - Critical: Returns amplitude (sqrt of intensity), not intensity.
        - Parameters:
            - Y_I_flat: Object amplitude patches in flat format (B, N, N, 1)
            - Y_phi_flat: Object phase patches in flat format (B, N, N, 1)
            - probe: Probe function (N, N, 1)
            - intensity_scale: Optional normalization factor

    `mk_simdata(n, size, probe, outer_offset, **kwargs)`
        - Purpose: Generate complete synthetic ptychographic datasets
        - Returns: Diffraction amplitudes, object patches, coordinates
        - Parameters: n (dataset size), size (patch dimension), probe

Workflow Usage Example:
    ```python
    import ptycho.diffsim as sim
    from ptycho import params
    import tensorflow as tf
    
    # 1. Configure simulation parameters
    params.set('nphotons', 1e8)  # Photon flux for noise modeling
    params.set('data_source', 'grf')  # Object type
    
    # 2. Generate complete synthetic dataset
    probe = tf.complex(tf.ones((64, 64)), 0.0)
    X, Y_I, Y_phi, scale, full_obj, norm, coords = sim.mk_simdata(
        n=2000, size=64, probe=probe, outer_offset=32
    )
    
    # 3. Note: X contains sqrt(intensity) values per data contract
    # To get photon counts: intensity = X ** 2
    ```

Architectural Notes:
- This module is the authoritative implementation of the forward model
  used throughout the codebase.
- The output amplitude format matches the data contract for training data.
- Noise modeling uses Poisson statistics based on params['nphotons'].
- Dependencies: ptycho.fourier, ptycho.tf_helper, ptycho.params
- Supports multiple synthetic object types via datagen modules.
"""
from skimage import draw, morphology
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import fourier as f
from . import tf_helper as hh
from . import params as p

tfk = tf.keras
tfkl = tf.keras.layers

N = 64

def observe_amplitude(amplitude):
    """
    Sample photons from wave amplitudes by drwaing from the corresponding Poisson distributions
    """
    return tf.sqrt((hh.tfd.Independent(hh.tfd.Poisson(amplitude**2))).sample())# + 0.5

def count_photons(obj):
    return tf.math.reduce_sum(obj**2, (1, 2))

def scale_nphotons(padded_obj):
    """
    Calculate the object amplitude normalization factor that gives the desired
    *expected* number of observed photons, averaged over an entire dataset.

    Returns a single scalar.
    """
    mean_photons = tf.math.reduce_mean(count_photons(padded_obj))
    norm = tf.math.sqrt(p.get('nphotons') / mean_photons)
    return norm

def diffract_obj(sample, draw_poisson = True):
    N = p.get('N')
    amplitude = hh.pad_and_diffract(sample, N, N, pad=False)[1]
    if draw_poisson:
        observed_amp = observe_amplitude(amplitude)
        return observed_amp
    else:
        return amplitude

def illuminate_and_diffract(Y_I_flat, Y_phi_flat, probe, intensity_scale = None):
    """
    Simulates diffraction for a batch of individual object patches.

    This function is a core physics engine component and operates on data in the
    "Flat Format", where each patch is an independent item in the batch. It is the
    caller's responsibility to ensure input tensors adhere to this format.

    Args:
        Y_I_flat (tf.Tensor): A tensor of object amplitude patches in Flat Format.
                             Shape: (B * gridsize**2, N, N, 1)
        Y_phi_flat (tf.Tensor): A tensor of object phase patches in Flat Format.
                               Shape: (B * gridsize**2, N, N, 1)
        probe (tf.Tensor): The probe function.
                          Shape: (N, N, 1)
        intensity_scale (float, optional): Normalization factor.

    Returns:
        Tuple[tf.Tensor, ...]: Simulated diffraction patterns, also in Flat Format.
                              Shape of X: (B * gridsize**2, N, N, 1)

    Raises:
        ValueError: If input tensors do not have a channel dimension of 1.

    See Also:
        - ptycho.tf_helper._channel_to_flat: For converting from Channel to Flat format.
        - ptycho.tf_helper._flat_to_channel: For converting from Flat to Channel format.
    """
    # ensure probe is broadcastable
    if len(probe.shape) == 2:
        assert probe.shape[0] == probe.shape[1]
        probe = probe[..., None]
    elif len(probe.shape) == 3:
        assert probe.shape[-1] == 1
    
    # After the coercion logic above, the probe tensor is guaranteed to be 3D.
    # This assertion makes that guarantee explicit and will immediately fail
    # if any unexpected shape passes through the initial checks.
    assert len(probe.shape) == 3 and probe.shape[-1] == 1, \
        f"Internal error: Probe shape must be (H, W, 1) before use, but got {probe.shape}"
    
    if intensity_scale is None:
        probe_amplitude = tf.cast(tf.abs(probe), Y_I_flat.dtype)
        intensity_scale = scale_nphotons(Y_I_flat * probe_amplitude[None, ...]).numpy()
    batch_size = p.get('batch_size')
    obj = intensity_scale * hh.combine_complex(Y_I_flat, Y_phi_flat)
    obj = obj * tf.cast(probe[None, ...], obj.dtype)
    Y_I_flat = tf.math.abs(obj)

    X = (tf.data.Dataset.from_tensor_slices(obj)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE)
               .map(diffract_obj)
               .cache())
    X = np.vstack(list(iter(X)))
    X, Y_I_flat, Y_phi_flat =\
        X / intensity_scale, Y_I_flat / intensity_scale, Y_phi_flat

    X, Y_I_flat, Y_phi_flat =\
        hh.togrid(X, Y_I_flat, Y_phi_flat)

    X, Y_I_flat, Y_phi_flat =\
        hh.grid_to_channel(X, Y_I_flat, Y_phi_flat)

    return X, Y_I_flat, Y_phi_flat, intensity_scale

def mk_rand(N):
    return int(N * np.random.uniform())

def mk_lines_img(N = 64, nlines = 10):
    image = np.zeros((N, N))
    for _ in range(nlines):
        rr, cc = draw.line(mk_rand(N), mk_rand(N), mk_rand(N), mk_rand(N))
        image[rr, cc] = 1
    res = np.zeros((N, N, 3))
    res[:, :, :] = image[..., None]
    return f.gf(res, 1) + 2 * f.gf(res, 5) + 5 * f.gf(res, 10)

def mk_noise(N = 64, nlines = 10):
    return np.random.uniform(size = N * N).reshape((N, N, 1))

from ptycho.misc import memoize_disk_and_memory

def extract_coords(size, repeats = 1, coord_type = 'offsets',
        outer_offset = None, **kwargs):
    """
    Return nominal offset coords in channel format. x and y offsets are
    stacked in the third dimension.

    offset coordinates are r - r', where
        r' is the patch center of mass
        r is the center of mass of that patch's solution region / grid,
            which contains gridsize**2 patches
    """
    x = tf.range(size, dtype = tf.float32)
    y = tf.range(size, dtype = tf.float32)
    xx, yy = tf.meshgrid(x, y)
    xx = xx[None, ..., None]
    yy = yy[None, ..., None]
    def _extract_coords(zz, fn):
        ix = fn(zz)
        ix = tf.reduce_mean(ix, axis = (1, 2))
        return tf.repeat(ix, repeats, axis = 0)[:, None, None, :]
    def outer(img):
        return hh.extract_outer(img, fmt = 'grid', outer_offset = outer_offset)
    def inner(img):
        return hh.extract_nested_patches(img, fmt = 'channel',
            outer_offset = outer_offset)
    def get_patch_offsets(coords):
        offsets_x = coords[1][0] - coords[0][0]
        offsets_y = coords[1][1] - coords[0][1]
        return tf.stack([offsets_x, offsets_y], axis = 2)[:, :, :, 0, :]
    ix = _extract_coords(xx, inner)
    iy = _extract_coords(yy, inner)
    ix_offsets = _extract_coords(xx, outer)
    iy_offsets = _extract_coords(yy, outer)
    coords = ((ix, iy), (ix_offsets, iy_offsets))
    if coord_type == 'offsets':
        return get_patch_offsets(coords)
    elif coord_type == 'global':
        return (ix, iy)
    else:
        raise ValueError

def add_position_jitter(coords, jitter_scale):
    shape = coords.shape
    jitter = jitter_scale * tf.random.normal(shape)
    return jitter + coords

def scan_and_normalize(jitter_scale = None, YY_I = None, YY_phi = None,
        **kwargs):
    """
    Inputs:
    4d tensors of full (arbitrary-sized) object phase and amplitude.

    Returns (normalized) amplitude and phase and scan point offsets.

    coords: tuple of two tuples. First gives center coords for each
    small image patch. Second gives offset coords for each solution
    region.
    """
    size = YY_I.shape[1]
    n = YY_I.shape[0]
    coords = true_coords = extract_coords(size, n, **kwargs)
    if jitter_scale is not None:
        print('simulating gaussian position jitter, scale', jitter_scale)
        true_coords = add_position_jitter(coords, jitter_scale)

    Y_I, Y_phi, _Y_I_full, norm_Y_I = hh.preprocess_objects(YY_I,
        offsets_xy = true_coords, Y_phi = YY_phi, **kwargs)
    return Y_I, Y_phi, _Y_I_full, norm_Y_I, (coords, true_coords)

import math
def dummy_phi(Y_I):
    return tf.cast(tf.constant(math.pi), tf.float32) *\
        tf.cast(tf.math.tanh( (Y_I - tf.math.reduce_max(Y_I) / 2) /
            (3 * tf.math.reduce_mean(Y_I))), tf.float32)

def sim_object_image(size, which = 'train'):
    if p.get('data_source') == 'lines':
        return mk_lines_img(2 * size, nlines = 400)[size // 2: -size // 2, size // 2: -size // 2, :1]
    elif p.get('data_source') == 'grf':
        from .datagen import grf
        return grf.mk_grf(size)
    elif p.get('data_source') == 'points':
        from .datagen import points
        return points.mk_points(size)
    elif p.get('data_source') == 'testimg':
        from .datagen import testimg
        if which == 'train':
            return testimg.get_img(size)
        elif which == 'test':
            return testimg.get_img(size, reverse = True)
        else:
            raise ValueError
    elif p.get('data_source') == 'testimg_reverse':
        from .datagen import testimg
        return testimg.get_img(size, reverse = True)
    elif p.get('data_source') == 'diagonals':
        from .datagen import diagonals
        return diagonals.mk_diags(size)
    elif p.get('data_source') == 'V':
        from .datagen import vendetta
        return vendetta.mk_vs(size)
    else:
        raise ValueError

@memoize_disk_and_memory
def mk_simdata(n, size, probe, outer_offset, intensity_scale = None,
        YY_I = None, YY_phi = None, dict_fmt = False,  which = 'train', **kwargs):
    if YY_I is None:
        YY_I = np.array([sim_object_image(size, which = which)
              for _ in range(n)])
    if p.get('set_phi') and YY_phi is None:
        YY_phi = dummy_phi(YY_I)
    Y_I, Y_phi, _, norm_Y_I, coords = scan_and_normalize(YY_I = YY_I,
        YY_phi = YY_phi, outer_offset = outer_offset, **kwargs)
    if dict_fmt:
        d = dict()
        d['I_pre_probe'] = Y_I
        d['phi_pre_probe'] = Y_phi
    X, Y_I, Y_phi, intensity_scale =\
        illuminate_and_diffract(Y_I_flat=Y_I, Y_phi_flat=Y_phi, probe=probe, intensity_scale=intensity_scale)
    if YY_phi is None:
        YY_full = hh.combine_complex(YY_I, tf.zeros_like(YY_I))
    else:
        YY_full = hh.combine_complex(YY_I, YY_phi)
    if dict_fmt:
        d['X'] = X
        d['Y_I'] = Y_I
        d['Y_phi'] = Y_phi
        d['intensity_scale'] = intensity_scale
        d['norm_Y_I'] = norm_Y_I
        d['coords'] = coords
        return d
    return X, Y_I, Y_phi, intensity_scale, YY_full,\
        norm_Y_I, coords

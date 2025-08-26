"""Core forward physics simulation engine for ptychographic reconstruction.

This module implements the differentiable forward physics model that forms the foundation 
of PtychoPINN's physics-informed neural network architecture. It simulates the complete 
ptychographic measurement process: object illumination, coherent diffraction, and 
photon detection with realistic Poisson noise characteristics.

Physics Implementation:
    The ptychographic forward model: object * probe → |FFT|² → Poisson(counts)
    Serves dual purposes: generates synthetic training data and provides physics 
    constraints for PINN training via differentiable simulation.

Architecture Role:
    - Training data generation with various object types (lines, GRF, points)
    - Physics-informed loss terms constraining network to optical principles  
    - Realistic Poisson photon noise modeling for experimental data matching

Core Functions:
    illuminate_and_diffract: Complete illumination and diffraction pipeline
    mk_simdata: Generate synthetic datasets with configurable object types
    observe_amplitude: Poisson photon noise simulation
    scale_nphotons: Photon count normalization
    sim_object_image: Synthetic object generation

Example:
    # Generate synthetic training dataset
    from ptycho import params
    params.set('nphotons', 1e6)
    params.set('data_source', 'lines')
    
    probe = np.exp(1j * np.random.uniform(0, 2*np.pi, (64, 64)))
    X, Y_I, Y_phi, intensity_scale, YY_full, norm_Y_I, coords = mk_simdata(
        n=2000, size=64, probe=probe, outer_offset=32, data_source='lines'
    )
    # X: (n, 64, 64) diffraction amplitudes, Y_I/Y_phi: object patches

Integration:
    Integrates with ptycho.model (physics losses), ptycho.loader (training data),
    ptycho.datagen.* (synthetic objects), ptycho.tf_helper (TF diffraction ops).

Note: Core physics module - preserve ptychographic forward model compatibility.
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

def illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = None):
    """
    Illuminate object with real or complex probe, then apply diffraction map.

    Returned Y_I and Y_phi are amplitude and phase *after* illumination with the
    probe.
    
    Critical invariant: Both X and Y_I are divided by intensity_scale
    to maintain consistent normalization between input diffraction patterns
    and target object patches. Breaking this symmetry has caused multiple
    historical bugs. See tests/test_scaling_regression.py for validation.
    
    Returns:
        X: Diffraction patterns normalized by intensity_scale
        Y_I: Object intensity normalized by intensity_scale  
        Y_phi: Object phase (not scaled - phase is scale-invariant)
        intensity_scale: The normalization factor for physics loss
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
        probe_amplitude = tf.cast(tf.abs(probe), Y_I.dtype)
        intensity_scale = scale_nphotons(Y_I * probe_amplitude[None, ...]).numpy()
    batch_size = p.get('batch_size')
    obj = intensity_scale * hh.combine_complex(Y_I, Y_phi)
    obj = obj * tf.cast(probe[None, ...], obj.dtype)
    Y_I = tf.math.abs(obj)

    X = (tf.data.Dataset.from_tensor_slices(obj)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE)
               .map(diffract_obj)
               .cache())
    X = np.vstack(list(iter(X)))
    
    # Assertion: intensity_scale must be valid for normalization
    assert intensity_scale > 0, f"Invalid intensity_scale: {intensity_scale}"
    assert not np.isnan(intensity_scale), f"intensity_scale is NaN"
    
    # Store original X for verification
    X_before_normalization = X.copy() if isinstance(X, np.ndarray) else X.numpy()
    
    X, Y_I, Y_phi =\
        X / intensity_scale, Y_I / intensity_scale, Y_phi
    
    # Assertion: verify both X and Y_I were scaled
    # This catches the historical bug where X scaling was removed
    assert np.allclose(X * intensity_scale, X_before_normalization, rtol=1e-6), \
        "X scaling verification failed - check line 123"

    X, Y_I, Y_phi =\
        hh.togrid(X, Y_I, Y_phi)

    X, Y_I, Y_phi =\
        hh.grid_to_channel(X, Y_I, Y_phi)

    return X, Y_I, Y_phi, intensity_scale

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
        illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
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

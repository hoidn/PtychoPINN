from skimage import draw, morphology
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import fourier as f
from . import physics
from . import tf_helper as hh
from . import params as p

tfk = tf.keras
tfkl = tf.keras.layers

# TODO
N = 64

def mk_rand(N):
    return int(N * np.random.uniform())

def mk_lines_img(N = 64, nlines = 10):
    image = np.zeros((N, N))
    for _ in range(nlines):
        rr, cc = draw.line(mk_rand(N), mk_rand(N), mk_rand(N), mk_rand(N))
        image[rr, cc] = 1
    #dilated = morphology.dilation(image, morphology.disk(radius=1))
    res = np.zeros((N, N, 3))
    #res[:, :, :] = dilated[..., None]
    res[:, :, :] = image[..., None]
    #return f.gf(res, 1) + 5 * f.gf(res, 10)
    return f.gf(res, 1) + 2 * f.gf(res, 5) + 5 * f.gf(res, 10)
    #return f.gf(res, 1) + f.gf(res, 10)

def mk_noise(N = 64, nlines = 10):
    return np.random.uniform(size = N * N).reshape((N, N, 1))

def mk_expdata(which, probe, intensity_scale = None):
    from . import experimental
    Y_I, Y_phi, _Y_I_full, norm_Y_I = experimental.preprocess_experimental(which)
    size = _Y_I_full.shape[1]
    print('shape', _Y_I_full.shape)
    coords = true_coords = extract_coords(size, 1)
    X, Y_I, Y_phi, intensity_scale =\
        physics.illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
    # TODO put this in a struct or something
    return X, Y_I, Y_phi, intensity_scale, _Y_I_full, norm_Y_I, (coords, true_coords)

def extract_coords(size, repeats = 1, coord_type = 'offsets', **kwargs):
    """
    Return nominal offset coords in channel format. x and y offsets are
    stacked in the third dimension.
    """
    # TODO enforce consistency on value of size (and repeats)
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
        return hh.extract_outer(img, fmt = 'grid')
    def inner(img):
        return hh.extract_nested_patches(img, fmt = 'channel')
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
    # TODO int jitter is just for debugging. need float eventually
#    jitter = tf.cast(jitter, tf.int32)
#    jitter = tf.cast(jitter, tf.float32)
    return jitter + coords

# TODO kwargs -> positional
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
    # x and y coordinates of the patches
    # TODO enforce consistent value of size
    size = YY_I.shape[1]
    n = YY_I.shape[0]
    coords = true_coords = extract_coords(size, n, **kwargs)
    if jitter_scale is not None:
        print('simulating gaussian position jitter, scale', jitter_scale)
        true_coords = add_position_jitter(coords, jitter_scale)

    Y_I, Y_phi, _Y_I_full, norm_Y_I = physics.preprocess_objects(YY_I,
        offsets_xy = true_coords, Y_phi = YY_phi, **kwargs)
    return Y_I, Y_phi, _Y_I_full, norm_Y_I, (coords, true_coords)

import math
def dummy_phi(Y_I):
    return tf.cast(tf.constant(math.pi), tf.float32) *\
        tf.cast(tf.math.tanh( (Y_I - tf.math.reduce_max(Y_I) / 2) /
            (3 * tf.math.reduce_mean(Y_I))), tf.float32)

def sim_object_image(size):
    if p.get('data_source') == 'lines':
        return mk_lines_img(2 * size, nlines = 400)[size // 2: -size // 2, size // 2: -size // 2, :1]
    elif p.get('data_source') == 'grf':
        from . import grf
        return grf.mk_grf(size)
    elif p.get('data_source') == 'points':
        from . import points
        return points.mk_points(size)
    elif p.get('data_source') == 'testimg':
        from . import testimg
        return testimg.get_img(size)
    elif p.get('data_source') == 'diagonals':
        from . import diagonals
        return diagonals.mk_diags(size)
    else:
        raise ValueError

def mk_simdata(n, size, probe, intensity_scale = None,
        YY_I = None, YY_phi = None, dict_fmt = False,  **kwargs):
    if YY_I is None:
        YY_I = np.array([sim_object_image(size)
              for _ in range(n)])[:, :, :]
    if p.get('set_phi'):
        YY_phi = dummy_phi(YY_I)
    # TODO two cases: n and size given, or Y_I and phi given
    # TODO there should be an option for subsampling, in case we don't want to
    # train on a dense view of each image
    Y_I, Y_phi, _Y_I_full, norm_Y_I, coords = scan_and_normalize(YY_I = YY_I,
        YY_phi = YY_phi, **kwargs)
    if dict_fmt:
        d = dict()
        d['I_pre_probe'] = Y_I
        d['phi_pre_probe'] = Y_phi
    X, Y_I, Y_phi, intensity_scale =\
        physics.illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
    if YY_phi is None:
        YY_full = hh.combine_complex(_Y_I_full, tf.zeros_like(_Y_I_full))
    else:
        YY_full = hh.combine_complex(_Y_I_full, YY_phi)
    if dict_fmt:
        d['X'] = X
        d['Y_I'] = Y_I
        d['Y_phi'] = Y_phi
        d['intensity_scale'] = intensity_scale
        d['_Y_I_full'] = _Y_I_full
        d['norm_Y_I'] = norm_Y_I
        d['coords'] = coords
        return d
    return X, Y_I, Y_phi, intensity_scale, YY_full,\
        norm_Y_I, coords

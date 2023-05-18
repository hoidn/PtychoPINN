from skimage import draw, morphology
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import fourier as f
#from . import physics
from . import tf_helper as hh
from . import params as p

tfk = tf.keras
tfkl = tf.keras.layers

# TODO
N = 64

nphotons = p.get('sim_nphotons')

def observe_amplitude(amplitude):
    """
    Sample photons from wave amplitudes by drwaing from the corresponding Poisson distributions
    """
    return tf.sqrt((hh.tfd.Independent(hh.tfd.Poisson(amplitude**2))).sample())# + 0.5

def count_photons(obj):
    return tf.math.reduce_sum(obj**2, (1, 2, 3))

def scale_nphotons(padded_obj):
    """
    Calculate the object amplitude normalization factor that gives the desired
    *expected* number of observed photons, averaged over an entire dataset.

    Returns a single scalar.
    """
    mean_photons = tf.math.reduce_mean(count_photons(padded_obj))
    norm = tf.math.sqrt(nphotons / mean_photons)
    return norm


def diffract_obj(sample, draw_poisson = True):
    # run ff diffraction
    h = p.get('h')
    w = p.get('w')
    amplitude = hh.pad_and_diffract(sample, h, w, pad=False)[1]
#     return amplitude
    # sample from Poisson observation likelihood
    if draw_poisson:
        observed_amp = observe_amplitude(amplitude)
        return observed_amp
    else:
        return amplitude

def illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = None):
    batch_size = p.get('batch_size')
    Y_I = Y_I *  probe[None, ..., None]

    if intensity_scale is None:
        intensity_scale = scale_nphotons(Y_I).numpy()

    obj = intensity_scale * hh.combine_complex(Y_I, Y_phi)
    Y_I = tf.math.abs(obj)# TODO

    # Simulate diffraction
    X = (tf.data.Dataset.from_tensor_slices(obj)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE)
               .map(diffract_obj)
               .cache())
    X = np.vstack(list(iter(X)))
    X, Y_I, Y_phi =\
        X / intensity_scale, Y_I / intensity_scale, Y_phi

    # TODO consolidate
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
    #dilated = morphology.dilation(image, morphology.disk(radius=1))
    res = np.zeros((N, N, 3))
    #res[:, :, :] = dilated[..., None]
    res[:, :, :] = image[..., None]
    #return f.gf(res, 1) + 5 * f.gf(res, 10)
    return f.gf(res, 1) + 2 * f.gf(res, 5) + 5 * f.gf(res, 10)
    #return f.gf(res, 1) + f.gf(res, 10)

def mk_noise(N = 64, nlines = 10):
    return np.random.uniform(size = N * N).reshape((N, N, 1))

# TODO why doesn't changing the probe scale parameter always change the
# memoization key?
from ptycho.misc import memoize_disk_and_memory
# TODO cleanup - for example, this function is deprecated
#@memoize_disk_and_memory
#def mk_expdata(which, probe, outer_offset, intensity_scale = None):
#    from . import experimental
#    # TODO refactor. maybe consolidate scan_and_normalize and
#    # hh.preprocess_objects
#    (YY_I, YY_phi), (Y_I, Y_phi, _, norm_Y_I) =\
#        experimental.preprocess_experimental(which, outer_offset)
#    size = YY_I.shape[1]
#    print('shape', YY_I.shape)
#    coords = true_coords = extract_coords(size, 1, outer_offset = outer_offset)
#    X, Y_I, Y_phi, intensity_scale =\
#        illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
#    # TODO put this in a struct or something
#    if YY_phi is None:
#        YY_full = hh.combine_complex(YY_I, tf.zeros_like(YY_I))
#    else:
#        YY_full = hh.combine_complex(YY_I, YY_phi)
#    return X, Y_I, Y_phi, intensity_scale, YY_full, norm_Y_I, (coords, true_coords)

def extract_coords(size, repeats = 1, coord_type = 'offsets',
        outer_offset = None, **kwargs):
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
    # TODO int jitter is just for debugging. need float eventually
#    jitter = tf.cast(jitter, tf.int32)
#    jitter = tf.cast(jitter, tf.float32)
    return jitter + coords

# TODO set separate bigoffset for train and test

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
    # TODO take outer grid offset as positional argument
    # x and y coordinates of the patches
    # TODO enforce consistent value of size
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
    elif p.get('data_source') == 'testimg_reverse':
        from . import testimg
        return testimg.get_img(size, reverse = True)
    elif p.get('data_source') == 'diagonals':
        from . import diagonals
        return diagonals.mk_diags(size)
    else:
        raise ValueError

@memoize_disk_and_memory
def mk_simdata(n, size, probe, outer_offset, intensity_scale = None,
        YY_I = None, YY_phi = None, dict_fmt = False,  **kwargs):
    if YY_I is None:
        YY_I = np.array([sim_object_image(size)
              for _ in range(n)])
    if p.get('set_phi') and YY_phi is None:
        YY_phi = dummy_phi(YY_I)
    # TODO two cases: n and size given, or Y_I and phi given
    # TODO there should be an option for subsampling, in case we don't want to
    # train on a dense view of each image
    #Y_I, Y_phi, _Y_I_full, norm_Y_I, coords = scan_and_normalize(YY_I = YY_I,
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

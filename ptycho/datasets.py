from skimage import draw, morphology
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import fourier as f
from . import physics
from . import tf_helper as hh

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
    coords = extract_coords(size, 1)
    X, Y_I, Y_phi, intensity_scale =\
        physics.illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
    # TODO put this in a struct or something
    return X, Y_I, Y_phi, intensity_scale, _Y_I_full, norm_Y_I, coords

def extract_coords(size, repeats = 1):
    """
    Return offset coords in channel format. x and y offsets are stacked in the
    third dimension.
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
#        return tf.stack([hh._channel_to_flat(offsets_x),
#            hh._channel_to_flat(offsets_y)], axis = 1)[:, :, 0, 0, 0]
    ix = _extract_coords(xx, inner)
    iy = _extract_coords(yy, inner)
    ix_offsets = _extract_coords(xx, outer)
    iy_offsets = _extract_coords(yy, outer)
    coords = ((ix, iy), (ix_offsets, iy_offsets))
    #return coords
    return get_patch_offsets(coords)

def simulate_objects(n, size):
    """
    Returns (normalized) amplitude and phase for n generated objects

    coords: tuple of two tuples. First gives center coords for each
    small image patch. Second gives offset coords for each solution
    region.
    """
#     Y_I = np.array([datasets.mk_lines_img(2 * size, nlines = 200)
#                           for _ in range(n)])[:, size // 2: -size // 2, size // 2: -size // 2, :1]
    # x and y coordinates of the patches
    # TODO enforce consistent value of size
    coords = extract_coords(size, n)

    Y_I = np.array([mk_lines_img(2 * size, nlines = 400)
          for _ in range(n)])[:, size // 2: -size // 2, size // 2: -size // 2, :1]

    Y_I, Y_phi, _Y_I_full, norm_Y_I = physics.preprocess_objects(Y_I)
    return Y_I, Y_phi, _Y_I_full, norm_Y_I, coords

def mk_simdata(n, size, probe, intensity_scale = None):
    Y_I, Y_phi, _Y_I_full, norm_Y_I, coords = simulate_objects(n, size)
    X, Y_I, Y_phi, intensity_scale =\
        physics.illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
    return X, Y_I, Y_phi, intensity_scale, _Y_I_full, norm_Y_I, coords

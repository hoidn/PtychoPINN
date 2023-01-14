from skimage import draw, morphology
from tensorflow.keras.layers import Lambda
from tensorflow.signal import fft2d, fftshift
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import fourier as f
from . import physics

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
    X, Y_I, Y_phi, intensity_scale =\
        physics.illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
    return X, Y_I, Y_phi, intensity_scale, _Y_I_full, norm_Y_I

def simulate_objects(n, size):
    """
    Returns (normalized) amplitude and phase for n generated objects
    """
#     Y_I = np.array([datasets.mk_lines_img(2 * size, nlines = 200)
#                           for _ in range(n)])[:, size // 2: -size // 2, size // 2: -size // 2, :1]
    Y_I = np.array([mk_lines_img(2 * size, nlines = 400)
          for _ in range(n)])[:, size // 2: -size // 2, size // 2: -size // 2, :1]
    return physics.preprocess_objects(Y_I)

def mk_simdata(n, size, probe, intensity_scale = None):

    Y_I, Y_phi, _Y_I_full, norm_Y_I = simulate_objects(n, size)
    X, Y_I, Y_phi, intensity_scale =\
        physics.illuminate_and_diffract(Y_I, Y_phi, probe, intensity_scale = intensity_scale)
    return X, Y_I, Y_phi, intensity_scale, _Y_I_full, norm_Y_I

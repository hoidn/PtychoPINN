import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from tensorflow.signal import fft
import tensorflow as tf
from tensorflow.keras.layers import Lambda

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from xrdc.source_separation import *
from skimage.transform import resize as sresize
from tensorflow.signal import fft2d, fftshift

from xrdc import fourier

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# from skimage.transform import resize

support_threshold = .0
@tf.function
def get_mask(input, support_threshold):
    import tensorflow as tf
    mask = tf.where(input > support_threshold, tf.ones_like(input),
                    tf.zeros_like(input))
    return mask

@tf.function
def add_support(input):
    mask = get_mask(input)
    return input * tf.cast(mask, tf.float32)

def resize(x):
    rmod = do_resize()
    rmod.compile(loss = 'mse')
    return rmod.predict(x)

def do_resize(N):
    #padx = pady = x.shape[1] // 2
    transform = tfkl.AveragePooling2D(2)
    return tfk.Sequential([
        tfk.Input(shape = (N, N, 1)),
        transform
    ])

def combine_complex(amp, phi):
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output

def pad_obj(input, h, w):
    return tfkl.ZeroPadding2D((h // 4, w // 4), name = 'padded_obj')(input)

#@tf.function
def pad_and_diffract(input, h, w, pad = True):
    """
    zero-pad the real-space object and then calculate the far field diffraction amplitude
    """
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    #sequential.add(tfk.Input(shape = (N, N, 1)))
    #sequential.add(Lambda(lambda inp: tprobe * inp))
    assert input.shape[-1] == 1
    input = (Lambda(lambda resized: (fft2d(
        #tf.squeeze # this destroys shape information so need to use slicing instead
        (tf.cast(resized, tf.complex64))[..., 0] 
        ))))(input)
    input = (Lambda(lambda X: tf.math.real(tf.math.conj(X) * X) / (h * w)))(input)
    input = (Lambda(lambda psd: 
                          tf.expand_dims(
                              tf.math.sqrt(
            fftshift(psd, (-2, -1))
                                   ), 3),
        name = 'pred_amplitude'))(input)
    return padded, input

def pd2(input, h, w, pad = True):
    """
    zero-pad the real-space object and then calculate the far field diffraction amplitude
    """
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    #sequential.add(tfk.Input(shape = (N, N, 1)))
    #sequential.add(Lambda(lambda inp: tprobe * inp))
    input = (Lambda(lambda resized: (fft2d(
        #tf.squeeze # this destroys shape information so need to use slicing instead
        (tf.cast(resized, tf.complex64)) 
        ))))(input)
    input = (Lambda(lambda X: tf.math.real(tf.math.conj(X) * X) / (h * w)))(input)
    input = (Lambda(lambda psd: 
                              tf.math.sqrt(
            fftshift(psd, (-2, -1))
                                   ),
        name = 'pred_amplitude'))(input)
    return padded, input


#prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
#                        reinterpreted_batch_ndims=1)
#
#
#encoder = tfk.Sequential([
#    tfkl.InputLayer(input_shape=input_shape),
#    tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
#    tf.keras.layers.BatchNormalization(),
#    tfkl.Conv2D(base_depth, 5, strides=1,
#                padding='same', activation=tf.nn.leaky_relu),
#    tfkl.Conv2D(base_depth, 5, strides=2,
#                padding='same', activation=tf.nn.leaky_relu),
#    tfkl.Conv2D(2 * base_depth, 5, strides=1,
#                padding='same', activation=tf.nn.leaky_relu),
#    tfkl.Conv2D(2 * base_depth, 5, strides=2,
#                padding='same', activation=tf.nn.leaky_relu),
#    tfkl.Conv2D(4 * encoded_size, 7, strides=1,
#                padding='valid', activation=tf.nn.leaky_relu),
#    tfkl.Flatten(),
#    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
#               activation=None),
#    tfpl.MultivariateNormalTriL(
#        encoded_size,
#        activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
#])


#####
## Utils for shifting and matching pieces of tensors
#####

# from scipy.spatial.distance import hamming

# def mk_coords():
#     for i in (0, 1):
#         for j in (0, 1):
#             yield (i, j)
            
# def get_pairs():
#     for i, p1 in enumerate(mk_coords()):
#         for p2 in list(mk_coords())[i:]:
#             if hamming(np.array(p1), np.array(p2)) == .5:
#                 yield (p1, p2)

# def get_range(i, j, toggle_i, toggle_j):
#     if toggle_i:
#         DX = N - offset
#         return (1 - i) * offset, (1 - i) * offset + DX, 0, offset + N
#     elif toggle_j:
#         DY = N - offset
#         return 0, offset + N, (1 - j) * offset, (1 - j) * offset + DY
#     else:
#         raise ValueError

# def toggle(coords1, coords2):
#     return (coords1[0] ^ coords2[0], coords1[1] ^ coords2[1])

# def get_overlap_pair(grid, coords1, coords2):
#     """
#     Get objects from neighboring scan points, cropping such that things line up properly.
#     """
#     i1start, i1end, j1start, j1end = get_range(*(coords1 + toggle(coords1, coords2)))
#     i2start, i2end, j2start, j2end = get_range(*(coords2 + toggle(coords1, coords2)))
#     return grid[:, coords1[0], coords1[1], i1start: i1end, j1start: j1end, :],\
#         grid[:, coords2[0], coords2[1], i2start: i2end, j2start: j2end, :]

# def get_overlap_pairs(grid):
#     pairs = list(set(get_pairs()))
#     return [get_overlap_pair(grid, p1, p2) for p1, p2 in pairs]

# def overlap_penalty(grid):
#     mae = tf.keras.losses.MeanAbsoluteError()
#     penalties = [mae(a1, a2) for a1, a2 in get_overlap_pairs(grid)]
#     return tf.math.reduce_mean(penalties)

# overlap_penalty(Y_I_test)

# pairs = list(set(get_pairs()))
# it = iter(pairs)

# a1, a2 = get_overlap_pair(Y_I_train, *next(it))
# plt.imshow((a1 - a2)[0], cmap = 'jet')

# overlap_penalty(Y_I_train)

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
from skimage.transform import resize as sresize
from tensorflow.signal import fft2d, fftshift

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

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

def pad_and_diffract(input, h, w, pad = True):
    """
    zero-pad the real-space object and then calculate the far field diffraction amplitude
    """
    padder = tfkl.ZeroPadding2D((h // 4, w // 4), name = 'padded_obj')

    if pad:
        input = padder(input)
    padded = input
    #sequential.add(tfk.Input(shape = (N, N, 1)))
    #sequential.add(Lambda(lambda inp: tprobe * inp))

    input = (Lambda(lambda resized: (fft2d(
        tf.squeeze(tf.cast(resized, tf.complex64))
        ))))(input)
    input = (Lambda(lambda X: tf.math.real(tf.math.conj(X) * X) / (h * w)))(input)
    input = (Lambda(lambda psd:
                          tf.expand_dims(
                              tf.math.sqrt(
            fftshift(psd, (-2, -1))
                                   ), 3)))(input)
    return padded, input


nphotons = 1e9

from .params import params
from . import tf_helper as hh
import tensorflow as tf
import numpy as np

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

def preprocess_objects(Y_I, Y_phi = None):
    """
    Reshapes and returns (normalized) amplitude and phase for the given real or complex objects
    """

    _Y_I_full = Y_I
    if Y_phi is None:
        Y_phi = np.zeros_like(Y_I)

    Y_I, Y_phi = \
        [hh.flatten_overlaps(imgs, fmt= 'channel') for imgs in [Y_I, Y_phi]]

    norm_Y_I = tf.math.reduce_max(Y_I, axis = (1, 2, 3))[:, None, None, None]
    Y_I /= norm_Y_I

    Y_I, Y_phi =\
        hh.channel_to_flat(Y_I, Y_phi)
    return Y_I, Y_phi, _Y_I_full, norm_Y_I

def diffract_obj(sample):
    # run ff diffraction
    h = params()['h']
    w = params()['w']
    amplitude = hh.pad_and_diffract(sample, h, w, pad=False)[1]
#     return amplitude
    # sample from Poisson observation likelihood
    observed_amp = observe_amplitude(amplitude)
    return observed_amp

def illuminate_and_diffract(Y_I_train, Y_phi_train, probe, intensity_scale = None):    
    Y_I_train = Y_I_train *  probe[None, ..., None]
    batch_size = params()['batch_size'] 
    
    if intensity_scale is None:
        intensity_scale = scale_nphotons(Y_I_train).numpy()

    obj_train = intensity_scale * hh.combine_complex(Y_I_train, Y_phi_train)
    Y_I_train = tf.math.abs(obj_train)

    # Simulate diffraction
    X_train = (tf.data.Dataset.from_tensor_slices(obj_train)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE)
               .map(diffract_obj)
               .cache())
    X_train = np.vstack(list(iter(X_train)))
    X_train, Y_I_train, Y_phi_train =\
        X_train / intensity_scale, Y_I_train / intensity_scale, Y_phi_train
    
    # TODO consolidate
    X_train, Y_I_train, Y_phi_train =\
        hh.togrid(X_train, Y_I_train, Y_phi_train)
    
    X_train, Y_I_train, Y_phi_train =\
        hh.grid_to_channel(X_train, Y_I_train, Y_phi_train)
    
    return X_train, Y_I_train, Y_phi_train, intensity_scale

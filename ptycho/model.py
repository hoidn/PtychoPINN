from datetime import datetime
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Lambda
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import math
import numpy as np
import glob
import os

from .loader import PtychoDataContainer
from . import tf_helper as hh
from . import params as cfg

import tensorflow_addons as tfa

wt_path = 'wts4.1'

gaussian_filter2d = tfa.image.gaussian_filter2d

tfk = hh.tf.keras
tfkl = hh.tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# Poisson distribution over expected diffraction intensity (i.e. photons per
# pixel)
negloglik = lambda x, rv_x: -rv_x.log_prob((x))

def interpolate(val, low, high):
    return low + (high - low) * val

def Conv_Pool_block(x0, nfilters, w1=3, w2=3, p1=2, p2=2, padding='same', data_format='channels_last'):
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = MaxPool2D((p1, p2), padding=padding, data_format=data_format)(x0)
    return x0

def Conv_Up_block(x0, nfilters, w1=3, w2=3, p1=2, p2=2, padding='same', data_format='channels_last', activation='relu'):
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation=activation, padding=padding, data_format=data_format)(x0)
    x0 = UpSampling2D((p1, p2), data_format=data_format)(x0)
    return x0

def create_encoder(input_tensor, n_filters_scale, cfg):
    N = cfg['N']
    
    if N == 64:
        filters = [n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
    elif N == 128:
        filters = [n_filters_scale * 16, n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
    elif N == 256:
        filters = [n_filters_scale * 8, n_filters_scale * 16, n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
    else:
        raise ValueError(f"Unsupported input size: {N}")
    
    x = input_tensor
    for num_filters in filters:
        x = Conv_Pool_block(x, num_filters)
    
    return x

def create_decoder_base(input_tensor, n_filters_scale, cfg):
    N = cfg['N']
    
    if N == 64:
        filters = [n_filters_scale * 64, n_filters_scale * 32]
    elif N == 128:
        filters = [n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
    elif N == 256:
        filters = [n_filters_scale * 256, n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
    else:
        raise ValueError(f"Unsupported input size: {N}")
    
    x = input_tensor
    for num_filters in filters:
        x = Conv_Up_block(x, num_filters)
    
    return x

def create_decoder_last(input_tensor, n_filters_scale, conv1, conv2, act, name, cfg):
    N = cfg['N']
    gridsize = cfg['gridsize']
    
    c_outer = 4
    x1 = conv1(input_tensor[..., :-c_outer])
    x1 = act(x1)
    x1 = tf.keras.layers.ZeroPadding2D(((N // 4), (N // 4)), name=name + '_padded')(x1)
    
    if not cfg['probe.big']:
        return x1
    
    x2 = Conv_Up_block(input_tensor[..., -c_outer:], n_filters_scale * 32)
    x2 = conv2(x2)
    x2 = tf.keras.activations.swish(x2)
    
    outputs = x1 + x2
    return outputs

def create_decoder_phase(input_tensor, n_filters_scale, gridsize, big, cfg):
    num_filters = gridsize**2 if big else 1
    conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    act = tf.keras.layers.Lambda(lambda x: math.pi * tf.keras.activations.tanh(x), name='phi')
    
    N = cfg['N']
    
    if N == 64:
        filters = [n_filters_scale * 64, n_filters_scale * 32]
    elif N == 128:
        filters = [n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
    elif N == 256:
        filters = [n_filters_scale * 256, n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
    else:
        raise ValueError(f"Unsupported input size: {N}")
    
    x = input_tensor
    for num_filters in filters:
        x = Conv_Up_block(x, num_filters)
    
    outputs = create_decoder_last(x, n_filters_scale, conv1, conv2, act=act, name='phase', cfg=cfg)
    return outputs

def create_decoder_amp(input_tensor, n_filters_scale, cfg):
    conv1 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    act = Lambda(get_amp_activation(cfg), name='amp')
    
    N = cfg['N']
    
    if N == 64:
        filters = [n_filters_scale * 64, n_filters_scale * 32]
    elif N == 128:
        filters = [n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
    elif N == 256:
        filters = [n_filters_scale * 256, n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
    else:
        raise ValueError(f"Unsupported input size: {N}")
    
    x = input_tensor
    for num_filters in filters:
        x = Conv_Up_block(x, num_filters)
    
    outputs = create_decoder_last(x, n_filters_scale, conv1, conv2, act=act, name='amp', cfg=cfg)
    return outputs

def create_autoencoder(input_tensor, n_filters_scale, gridsize, big, cfg):
    encoded = create_encoder(input_tensor, n_filters_scale, cfg)
    decoded_amp = create_decoder_amp(encoded, n_filters_scale, cfg)
    decoded_phase = create_decoder_phase(encoded, n_filters_scale, gridsize, big, cfg)
    
    return decoded_amp, decoded_phase

def get_amp_activation(cfg):
    if cfg['amp_activation'] == 'sigmoid':
        return tf.keras.activations.sigmoid
    elif cfg['amp_activation'] == 'swish':
        return tf.keras.activations.swish
    else:
        raise ValueError("Invalid amp_activation")

def compile_model(autoencoder, optimizer, realspace_weight, mae_weight, nll_weight):
    #fn_poisson_nll = lambda A_target, A_pred: negloglik(A_target**2, dist_poisson_intensity(A_pred))
    autoencoder.compile(optimizer=optimizer,
                        loss=[hh.realspace_loss,
                              'mean_absolute_error', negloglik, 'mean_absolute_error'],
                        loss_weights=[realspace_weight, mae_weight, nll_weight, 0.])

#def prepare_inputs(train_data, cfg):
#    """training inputs"""
#    return [train_data.X * cfg['intensity_scale'], train_data.coords]
#
#def prepare_outputs(train_data, cfg):
#    """training outputs"""
#    return [hh.center_channels(train_data.Y_I, train_data.coords)[:, :, :, :1],
#            (cfg['intensity_scale'] * train_data.X),
#            (cfg['intensity_scale'] * train_data.X)**2]

def prepare_inputs(train_data, cfg):
    """training inputs"""
    return [train_data.X * cfg['intensity_scale'], train_data.coords]

def prepare_outputs(train_data, cfg):
    """training outputs"""
    return [hh.center_channels(train_data.Y_I, train_data.coords)[:, :, :, :1],
            (cfg['intensity_scale'] * train_data.X),
            (cfg['intensity_scale'] * train_data.X)**2]

def train_model(autoencoder, trainset, epochs, cfg):
    assert type(trainset) == PtychoDataContainer
    coords_train = trainset.coords
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        '%s/weights.{epoch:02d}.h5' % wt_path,
        monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', period=1)

    batch_size = cfg['batch_size']
    history = autoencoder.fit(
        prepare_inputs(trainset, cfg),
        prepare_outputs(trainset, cfg),
        shuffle=True, batch_size=batch_size, verbose=1,
        epochs=epochs, validation_split=0.05,
        callbacks=[reduce_lr, earlystop])
    return history

class ProbeIllumination(tf.keras.layers.Layer):
    def __init__(self, name=None, initial_probe_guess = None, probe_mask = None):
        super(ProbeIllumination, self).__init__(name=name)
        self.w = initial_probe_guess  
        self.probe_mask = probe_mask

    def call(self, inputs):
        x = inputs[0]

        if cfg.get('probe.mask'):
            return self.w * x * self.probe_mask, (self.w * self.probe_mask)[None, ...]
        else:
            return self.w * x, (self.w)[None, ...]

#class IntensityScaler(tf.keras.layers.Layer):
#    def __init__(self, log_scale):
#        super(IntensityScaler, self).__init__()
#        self.w = log_scale
#    def call(self, inputs):
#        x, = inputs
#        return x / tf.math.exp(self.w)
#
## TODO use a bijector instead of separately defining the transform and its
## inverse
#class IntensityScaler_inv(tf.keras.layers.Layer):
#    def __init__(self, log_scale):
#        super(IntensityScaler_inv, self).__init__()
#        self.w = log_scale
#    def call(self, inputs):
#        x, = inputs
#        return tf.math.exp(self.w) * x

def _build_autoencoder(probeGuess, cfg):
    # TODO cfg should be protected from external mutation
    N = cfg['N']
    gridsize = cfg['gridsize']
    n_filters_scale = cfg['n_filters_scale']
    print('N FILTERS SCALLLLLLLE:', n_filters_scale)
    cfg['probe'] = probeGuess
    # TODO
    #probe_mask = probe.probe_mask
    probe_mask = cfg.get('probe_mask')[:, :, :, 0]
    initial_probe_guess = tf.Variable(
                initial_value=tf.cast(probeGuess, tf.complex64),
                trainable=cfg['probe.trainable'],
            )

    probe_illumination = ProbeIllumination(initial_probe_guess = initial_probe_guess, probe_mask = probe_mask)

    # TODO scaling could be done on a shot-by-shot basis, but IIRC I tried this
    # and there were issues
    log_scale_guess = np.log(cfg.get('intensity_scale'))
    log_scale = tf.Variable(
                initial_value=tf.constant(float(log_scale_guess)),
                trainable = cfg['intensity_scale.trainable'],
            )
    print('iIIIIIIIIPROOOOOOOOOOBE', cfg['intensity_scale'])



    def scale(inputs):
        x, = inputs
        res = x / tf.math.exp(log_scale)
        print('MEAEEEEEEEEEANNNNN:', tf.reduce_mean(res))
        return res

    def inv_scale(inputs):
        x, = inputs
        res = tf.math.exp(log_scale) * x
        print('iNNNNVVVVVV    MEAEEEEEEEEEANNNNN:', tf.reduce_mean(res))
        return res

    tf.keras.backend.clear_session()
    np.random.seed(2)

    files=glob.glob('%s/*' %wt_path)
    for file in files:
        os.remove(file)

    lambda_norm = Lambda(lambda x: tf.math.reduce_sum(x**2, axis = [1, 2]))
    
    input_img = Input(shape=(N, N, gridsize**2), name='input')
    input_positions = Input(shape=(1, 2, gridsize**2), name='input_positions')

    normed_input = scale([input_img])
    decoded_amp, decoded_phase = create_autoencoder(normed_input, n_filters_scale, gridsize, cfg['object.big'], cfg)

    obj = Lambda(lambda x: hh.combine_complex(x[0], x[1]), name='obj')([decoded_amp, decoded_phase])

    if cfg['object.big']:
        padded_obj_2 = Lambda(lambda x: hh.reassemble_patches(x[0], fn_reassemble_real=hh.mk_reassemble_position_real(x[1])), name='padded_obj_2')([obj, input_positions])
    else:
        padded_obj_2 = Lambda(lambda x: hh.pad_reconstruction(x), name='padded_obj_2')(obj)

    trimmed_obj = Lambda(hh.trim_reconstruction, name='trimmed_obj')(padded_obj_2)

    padded_objs_with_offsets = Lambda(lambda x: hh.extract_patches_position(x[0], x[1], 0.), name='padded_objs_with_offsets')([padded_obj_2, input_positions])

    print('SHAPPPPPE:', padded_objs_with_offsets.shape)
    print('BIGGGGGGG:', cfg['object.big'])
    padded_objs_with_offsets, probe = probe_illumination([padded_objs_with_offsets])
    flat_illuminated = padded_objs_with_offsets

    padded_objs_with_offsets, pred_diff = Lambda(lambda x: hh.pad_and_diffract(x, N, N, pad=False), name='pred_amplitude')(padded_objs_with_offsets)

    pred_diff = Lambda(lambda x: hh._flat_to_channel(x), name='pred_diff_channels')(pred_diff)

    pred_amp_scaled = inv_scale([pred_diff])

    dist_poisson_intensity = tfpl.DistributionLambda(lambda amplitude: (tfd.Independent(tfd.Poisson((amplitude**2)))))
    pred_intensity_sampled = dist_poisson_intensity(pred_amp_scaled)

    autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])

    autoencoder_no_nll = Model(inputs=[input_img, input_positions], outputs=[pred_amp_scaled])

    diffraction_to_obj = tf.keras.Model(inputs=[input_img, input_positions], outputs=[trimmed_obj])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    compile_model(autoencoder, optimizer, cfg['realspace_weight'], cfg['mae_weight'], cfg['nll_weight'])

    return {
        'autoencoder': autoencoder,
        'autoencoder_no_nll': autoencoder_no_nll,
        'diffraction_to_obj': diffraction_to_obj,
        'probe_illumination': probe_illumination,
        'log_scale': log_scale,
        'cfg': cfg
    }

def train(epochs, trainset, probeGuess, cfg):
    assert type(trainset) == PtychoDataContainer
    model_dict = _build_autoencoder(probeGuess, cfg)
    autoencoder = model_dict['autoencoder']
    history = train_model(autoencoder, trainset, epochs, cfg)
    return model_dict, history

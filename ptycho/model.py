# TODO s
# - complex convolution
# - Use tensor views:
#     https://chat.openai.com/c/e6d5e400-daf9-44b7-8ef9-d49f21a634a3
# -difference maps?
# -double -> float32
# Two lossess: a CDI loss and a ptychography loss
# Apply real space loss to both amplitude and phase of the object

from datetime import datetime
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.activations import relu, sigmoid, tanh, swish
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, InputLayer, Lambda, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import glob
import math
import numpy as np
import os
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from . import tf_helper as hh
from . import params as cfg
params = cfg.params

import tensorflow_addons as tfa
gaussian_filter2d = tfa.image.gaussian_filter2d

tfk = hh.tf.keras
tfkl = hh.tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

wt_path = 'wts4.1'
# sets the number of convolutional filters

n_filters_scale =  cfg.get('n_filters_scale')
N = cfg.get('N')
gridsize = cfg.get('gridsize')
offset = cfg.get('offset')

from . import probe
tprobe = params()['probe']
# TODO
#probe_mask = probe.probe_mask
probe_mask = cfg.get('probe_mask')[:, :, :, 0]
initial_probe_guess = tprobe
initial_probe_guess = tf.Variable(
            initial_value=tf.cast(initial_probe_guess, tf.complex64),
            trainable=params()['probe.trainable'],
        )

# TODO hyperparameters:
# TODO total variation loss
# -probe smoothing scale(?)
class ProbeIllumination(tf.keras.layers.Layer):
    def __init__(self, name = None):
        super(ProbeIllumination, self).__init__(name = name)
        self.w = initial_probe_guess
    def call(self, inputs):
        x, = inputs
        if cfg.get('probe.mask'):
            return self.w * x * probe_mask, (self.w * probe_mask)[None, ...]
        else:
            return self.w * x, (self.w)[None, ...]

# Stochastic probe
#class ProbeIllumination(tf.keras.layers.Layer):
#    def __init__(self, name = None):
#        super(ProbeIllumination, self).__init__(name = name)
#        self.w = initial_probe_guess
#        self.dist = tfd.Independent(tfd.Normal(loc = tf.math.real(self.w),
#            scale = 0.1))
#    def call(self, inputs):
#        x, = inputs
#        sample = tf.cast(self.dist.sample(), tf.complex64)
#        if cfg.get('probe.mask'):
#            return sample  * x * probe_mask, (sample * probe_mask)[None, ...]
#        else:
#            return sample * x, (sample)[None, ...]

probe_illumination = ProbeIllumination()

nphotons = cfg.get('nphotons')

# TODO scaling could be done on a shot-by-shot basis, but IIRC I tried this
# and there were issues
log_scale_guess = np.log(cfg.get('intensity_scale'))
log_scale = tf.Variable(
            initial_value=tf.constant(float(log_scale_guess)),
            trainable = params()['intensity_scale.trainable'],
        )

class IntensityScaler(tf.keras.layers.Layer):
    def __init__(self):
        super(IntensityScaler, self).__init__()
        self.w = log_scale
    def call(self, inputs):
        x, = inputs
        return x / tf.math.exp(self.w)

# TODO use a bijector instead of separately defining the transform and its
# inverse
class IntensityScaler_inv(tf.keras.layers.Layer):
    def __init__(self):
        super(IntensityScaler_inv, self).__init__()
        self.w = log_scale
    def call(self, inputs):
        x, = inputs
        return tf.math.exp(self.w) * x

def scale(inputs):
    x, = inputs
    res = x / tf.math.exp(log_scale)
    return res

def inv_scale(inputs):
    x, = inputs
    return tf.math.exp(log_scale) * x

#class LogScaler(tf.keras.layers.Layer):
#    def __init__(self):
#        super(LogScaler, self).__init__()
#    def call(self, inputs):
#        x, = inputs
#        return tf.math.log(1 + x**2) / tf.math.log(nphotons / (N**2))
#
#class LogScaler_inv(tf.keras.layers.Layer):
#    def __init__(self):
#        super(LogScaler_inv, self).__init__()
#    def call(self, inputs):
#        x, = inputs
#        return tf.math.exp((x**2) * tf.math.log(nphotons / (N**2))) - 1

tf.keras.backend.clear_session()
np.random.seed(2)

files=glob.glob('%s/*' %wt_path)
for file in files:
    os.remove(file)

lambda_norm = Lambda(lambda x: tf.math.reduce_sum(x**2, axis = [1, 2]))
input_img = Input(shape=(N, N, gridsize**2), name = 'input')
input_positions = Input(shape=(1, 2, gridsize**2), name = 'input_positions')

#logscaler = LogScaler()
#inv_logscaler = LogScaler_inv()
#normed_input = logscaler([input_img])

def Conv_Pool_block(x0,nfilters,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last'):
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = MaxPool2D((p1, p2), padding=padding, data_format=data_format)(x0)
    return x0

def Conv_Up_block(x0,nfilters,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last',
        activation = 'relu'):
    x0 = Conv2D(nfilters, (w1, w2), activation='relu', padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation=activation, padding=padding, data_format=data_format)(x0)
    x0 = UpSampling2D((p1, p2), data_format=data_format)(x0)
    return x0

def create_encoder_functional(input_tensor, n_filters_scale):
    # Blocks
    # x = Conv_Pool_block(input_tensor, n_filters_scale * 16)  # This block is commented out in the original
    x = Conv_Pool_block(input_tensor, n_filters_scale * 32)
    x = Conv_Pool_block(x, n_filters_scale * 64)
    outputs = Conv_Pool_block(x, n_filters_scale * 128)
    return outputs

def create_decoder_base_functional(input_tensor, n_filters_scale):
    # Blocks
    x = Conv_Up_block(input_tensor, n_filters_scale * 128)
    outputs = Conv_Up_block(x, n_filters_scale * 64)
    return outputs

def create_decoder_last_functional(input_tensor, n_filters_scale, conv1, conv2,
        act=tf.keras.activations.sigmoid, name=''):
    N = cfg.get('N')  # Placeholder: this should be fetched from the actual configuration
    gridsize = cfg.get('gridsize')  # Placeholder: this should be fetched from the actual configuration

    c_outer = 4
    x1 = conv1(input_tensor[..., :-c_outer])
    x1 = act(x1)
    x1 = tf.keras.layers.ZeroPadding2D(((N // 4), (N // 4)), name=name + '_padded')(x1)

    # Assuming the centermask function is similar to the one in the original class (needs to be defined)
    # x1 = centermask(x1)
    if not cfg.get('probe.big'):  # Placeholder: this should be fetched from the actual configuration
        return x1
    x2 = Conv_Up_block(input_tensor[..., -c_outer:], n_filters_scale * 32)
    x2 = conv2(x2)
    x2 = swish(x2)
    # x2 = centermask(x2)  # Applying centermask operation

    outputs = x1 + x2
    return outputs

def create_decoder_phase_functional(input_tensor, n_filters_scale, gridsize, big):
    num_filters = gridsize**2 if big else 1
    conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    # Activation function using Lambda layer
    act = tf.keras.layers.Lambda(lambda x: math.pi * tf.keras.activations.tanh(x), name='phi')
    x = create_decoder_base_functional(input_tensor, n_filters_scale)
    outputs = create_decoder_last_functional(x, n_filters_scale, conv1, conv2, act=act,
        name = 'phase')
    return outputs

def create_autoencoder_functional(input_tensor, n_filters_scale, gridsize, big):
    encoded = create_encoder_functional(input_tensor, n_filters_scale)
    decoded_amp = create_decoder_amp_functional(encoded, n_filters_scale)
    decoded_phase = create_decoder_phase_functional(encoded, n_filters_scale, gridsize, big)

    return decoded_amp, decoded_phase

def get_amp_activation():
    if cfg.get('amp_activation') == 'sigmoid':
        return lambda x: sigmoid(x)
    elif cfg.get('amp_activation') == 'swish':
        return lambda x: swish(x)
    else:
        return ValueError

def create_decoder_amp_functional(input_tensor, n_filters_scale):
    # Placeholder convolution layers and activation as defined in the original DecoderAmp class
    conv1 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    act = Lambda(get_amp_activation(), name='amp')

    x = create_decoder_base_functional(input_tensor, n_filters_scale)
    outputs = create_decoder_last_functional(x, n_filters_scale, conv1, conv2, act=act,
        name = 'amp')
    return outputs

#class PositionEncoder(Model):
#    # TODO scale tanh
#    def __init__(self, encoder):
#        super(AutoEncoder, self).__init__()
#        self.encoder = encoder
#        self.position = Lambda(lambda x:
#            tanh(
#                layers.Reshape((1, 2, gridsize**2))(
#                layers.Dense(2 * gridsize**2)(
#                layers.Flatten()(
#                layers.Dropout(0.3)(x)))), name = 'positions_enc'
#                )
#            )
#    def call(self, inputs):
#        x, xhat, y = inputs
#        encoded = tf.concat(
#            [self.encoder(x), self.encoder(xhat), self.encoder(y)])
#        encoded_pos = self.position(encoded)
#        return encoded_pos

normed_input = scale([input_img])
decoded1, decoded2 = create_autoencoder_functional(normed_input, n_filters_scale, gridsize,
    cfg.get('object.big'))

# Combine the two decoded outputs
obj = Lambda(lambda x: hh.combine_complex(x[0], x[1]), name='obj')([decoded1, decoded2])

# TODO rename
# Pad the output object
padded_obj = obj

if cfg.get('object.big'):
    # If 'object.big' is true, reassemble the patches
    padded_obj_2 = Lambda(lambda x: hh.reassemble_patches(x[0], fn_reassemble_real=hh.mk_reassemble_position_real(x[1])), name = 'padded_obj_2')([padded_obj, input_positions])
else:
    # If 'object.big' is not true, pad the reconstruction
    padded_obj_2 = Lambda(lambda x: hh.pad_reconstruction(x), name = 'padded_obj_2')(padded_obj)

# Trim the object reconstruction to N x N
trimmed_obj = Lambda(hh.trim_reconstruction, name = 'trimmed_obj')(padded_obj_2)

# Extract overlapping regions of the object
padded_objs_with_offsets = Lambda(lambda x:
    hh.extract_patches_position(x[0], x[1], 0.),
    name = 'padded_objs_with_offsets')([padded_obj_2, input_positions])

# Apply the probe illumination
padded_objs_with_offsets, probe = probe_illumination([padded_objs_with_offsets])
flat_illuminated = padded_objs_with_offsets

# Apply pad and diffract operation
padded_objs_with_offsets, pred_diff = Lambda(lambda x: hh.pad_and_diffract(x, N, N, pad=False), name = 'pred_amplitude')(padded_objs_with_offsets)

# Reshape
pred_diff = Lambda(lambda x: hh._flat_to_channel(x), name = 'pred_diff_channels')(pred_diff)

# Scale the amplitude
pred_amp_scaled = inv_scale([pred_diff])


# TODO Please pass an integer value for `reinterpreted_batch_ndims`. The current behavior corresponds to `reinterpreted_batch_ndims=tf.size(distribution.batch_shape_tensor()) - 1`.
dist_poisson_intensity = tfpl.DistributionLambda(lambda amplitude:
                                       (tfd.Independent(
                                           tfd.Poisson(
                                               (amplitude**2)))))
pred_intensity_sampled = dist_poisson_intensity(pred_amp_scaled)

# Poisson distribution over expected diffraction intensity (i.e. photons per
# pixel)
negloglik = lambda x, rv_x: -rv_x.log_prob((x))
fn_poisson_nll = lambda A_target, A_pred: negloglik(A_target**2, dist_poisson_intensity(A_pred))

autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled,
        probe])

autoencoder_no_nll = Model(inputs = [input_img, input_positions],
        outputs = [pred_amp_scaled])

# TODO this broke after encapsulating the contents of Maps
#encode_obj_to_diffraction = tf.keras.Model(inputs=[padded_obj, input_positions],
#                           outputs=[pred_diff, flat_illuminated])
diffraction_to_obj = tf.keras.Model(inputs=[input_img],
                           outputs=[obj])

mae_weight = cfg.get('mae_weight') # should normally be 0
nll_weight = cfg.get('nll_weight') # should normally be 1
# Total variation regularization on real space amplitude
realspace_weight = cfg.get('realspace_weight')#1e2
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

autoencoder.compile(optimizer= optimizer,
     #loss=[lambda target, pred: hh.total_variation(pred),
     loss=[hh.realspace_loss,
        'mean_absolute_error', negloglik, 'mean_absolute_error'],
     loss_weights = [realspace_weight, mae_weight, nll_weight, 0.])

print (autoencoder.summary())

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 profile_batch='500,520')

def prepare_inputs(X_train, coords_train):
    """training inputs"""
    return [X_train * cfg.get('intensity_scale'), coords_train]

def prepare_outputs(Y_I_train, coords_train, X_train):
    """training outputs"""
    return [hh.center_channels(Y_I_train, coords_train)[:, :, :, :1],
                (cfg.get('intensity_scale') * X_train),
                (cfg.get('intensity_scale') * X_train)**2,
               Y_I_train[:, :, :, :1]]

#def prepare_inputs(X_train, coords_train):
#    """training inputs"""
#    X_tensor = tf.convert_to_tensor(X_train * cfg.get('intensity_scale'))
#    coords_tensor = tf.convert_to_tensor(coords_train)
#    return [X_tensor, coords_tensor]
#
#def prepare_outputs(Y_I_train, coords_train, X_train):
#    """training outputs"""
#    Y_I_tensor = tf.convert_to_tensor(Y_I_train)
#    X_tensor = tf.convert_to_tensor(X_train)
#    coords_tensor = tf.convert_to_tensor(coords_train)
#    return [tf.convert_to_tensor(hh.center_channels(Y_I_tensor, coords_tensor)[:, :, :, :1]),
#            tf.convert_to_tensor(cfg.get('intensity_scale') * X_tensor),
#            tf.convert_to_tensor((cfg.get('intensity_scale') * X_tensor)**2),
#            Y_I_tensor[:, :, :, :1]]

#def train(epochs, X_train, coords_train, Y_I_train):
def train(epochs, X_train, coords_train, Y_obj_train):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoints= tf.keras.callbacks.ModelCheckpoint(
                            '%s/weights.{epoch:02d}.h5' %wt_path,
                            monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)

    batch_size = params()['batch_size']
    history=autoencoder.fit(
        prepare_inputs(X_train, coords_train),
        #prepare_outputs(Y_I_train, coords_train, X_train),
        prepare_outputs(Y_obj_train, coords_train, X_train),
        shuffle=True, batch_size=batch_size, verbose=1,
        epochs=epochs, validation_split = 0.05,
        callbacks=[reduce_lr, earlystop])
        #callbacks=[reduce_lr, earlystop, tboard_callback])
        #callbacks=[reduce_lr, earlystop, checkpoints])
    return history

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.layers import Lambda
import glob
import math
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp

from . import tf_helper as hh
from .params import params

import tensorflow_addons as tfa
gaussian_filter2d = tfa.image.gaussian_filter2d

#gft = Lambda(lambda x: gaussian_filter2d(x))

tfk = hh.tf.keras
tfkl = hh.tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

wt_path = 'wts4.1'
# sets the number of convolutional filters
n_filters_scale = 2

N = params()['N']
w = params()['w']
h = params()['h']
gridsize = params()['gridsize']
offset = params()['offset']
tprobe = params()['probe']
#intensity_scale = params()['intensity_scale']

probe_mask = params()['probe_mask']

initial_probe_guess = tprobe
initial_probe_guess = tf.Variable(
            initial_value=tf.cast(initial_probe_guess, tf.complex64),
            trainable=True,
        )

log_scale = tf.Variable(
            #initial_value=tf.constant(7.5),
            initial_value=tf.constant(7.84),
            # TODO learning rate is too slow, so the initial guess has to be
            # set quite close to the correct value
            #trainable=True,
            trainable = params()['intensity_scale.trainable'],
        )

class ProbeIllumination(tf.keras.layers.Layer):
    def __init__(self):
        super(ProbeIllumination, self).__init__()
        self.w = initial_probe_guess
    def call(self, inputs):
        x, = inputs
        return self.w * x * probe_mask

class IntensityScaler(tf.keras.layers.Layer):
    def __init__(self):
        super(IntensityScaler, self).__init__()
        self.w = log_scale
    def call(self, inputs):
        x, = inputs
        return x / tf.math.exp(self.w)

# TODO get inverse from the same class
class IntensityScaler_inv(tf.keras.layers.Layer):
    def __init__(self):
        super(IntensityScaler_inv, self).__init__()
        self.w = log_scale
    def call(self, inputs):
        x, = inputs
        return tf.math.exp(self.w) * x


tf.keras.backend.clear_session()
np.random.seed(2)

files=glob.glob('%s/*' %wt_path)
for file in files:
    os.remove(file)

lambda_norm = Lambda(lambda x: tf.math.reduce_sum(x**2, axis = [1, 2]))
input_img = Input(shape=(h, w, gridsize**2), name = 'input')

scaler = IntensityScaler()
inv_scaler = IntensityScaler_inv()
normed_input = scaler([input_img])

#x = hh.Conv_Pool_block(input_img,32,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
#x = hh.Conv_Pool_block(x,64,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
#x = hh.Conv_Pool_block(x,128,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')

x = hh.Conv_Pool_block(normed_input,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
x = hh.Conv_Pool_block(x,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
x = hh.Conv_Pool_block(x,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')

encoded=x

##Decoding arm for amplitude
#x1=hh.Conv_Up_block(encoded,128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
#x1=hh.Conv_Up_block(x1,64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')

#Decoding arm for amplitude
x1=hh.Conv_Up_block(encoded,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
x1=hh.Conv_Up_block(x1,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
#x1=hh.Conv_Up_block(x1,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
decoded1 = Conv2D(gridsize**2, (3, 3), padding='same')(x1)
decoded1 = Lambda(lambda x: sigmoid(x), name='amp')(decoded1)

##Decoding arm for phase
#x2=hh.Conv_Up_block(encoded,128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
#x2=hh.Conv_Up_block(x2,64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
##x2=Conv_Up_block(x2,32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')

#Decoding arm for phase
x2=hh.Conv_Up_block(encoded,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
x2=hh.Conv_Up_block(x2,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
#x2=hh.Conv_Up_block(x2,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
decoded2 = Conv2D(gridsize**2, (3, 3), padding='same')(x2)
decoded2 = Lambda(lambda x: math.pi * tanh(x), name='phi')(decoded2)

###Decoding arm for probe
#x3=hh.Conv_Up_block(encoded,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
#x3=hh.Conv_Up_block(x3,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
##x3=hh.Conv_Up_block(x3,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
#decoded3 = Conv2D(gridsize**2, (3, 3), padding='same')(x3)
#probe_amp = Lambda(lambda x: sigmoid(x))(decoded3)
#probe_amp_channels = tfkl.ZeroPadding2D((N // 4, N // 4), name='probe_amp')(probe_amp)
#probe_amp_flat = hh._channel_to_flat(probe_amp_channels)

obj = Lambda(lambda x: hh.combine_complex(x[0], x[1]),
                     name='obj')([decoded1, decoded2])

#padded_obj = Lambda(lambda x: x, name = 'padded_obj')(obj)
padded_obj = tfkl.ZeroPadding2D(((h // 4), (w // 4)), name = 'padded_obj')(obj)
padded_obj_2 = Lambda(lambda x:
    hh.reassemble_patches(x), name = 'padded_obj_2',
    )(padded_obj)

# Trim to N // 2 X N // 2 central region of object reconstruction
trimmed_obj = Lambda(lambda x: x[:, (offset * (gridsize - 1)) // 2: -(offset * (gridsize - 1)) // 2,
        (offset * (gridsize - 1)) // 2: -(offset * (gridsize - 1)) // 2,
        :], name = 'trimmed_obj')(padded_obj_2)

# TODO trimmed obj should really be masked by the union of all the illumination
# spots, instead of just takiing the central region
crop_center = Lambda(lambda x: x[:, N // 2: -N // 2, N // 2: -N // 2, :])
trimmed_obj_small = crop_center(ProbeIllumination()([trimmed_obj]))
#trimmed_obj_small = Lambda(
#    lambda x: crop_center(x[0] * tf.cast(x[1], tf.complex64) * x[2]))(
#        [trimmed_obj, probe_amp_flat, probe_mask])

# Extract overlapping regions of the object
padded_objs_with_offsets = Lambda(lambda x: hh.extract_nested_patches(x, fmt = 'flat'), name = 'padded_objs_with_offsets')(padded_obj_2)
#padded_objs_with_offsets = Lambda(lambda x: x[0] * tf.cast(x[1], tf.complex64) * x[2])(
#    [padded_objs_with_offsets, probe_amp_flat, probe_mask])
# Apply the probe
padded_objs_with_offsets = ProbeIllumination()([padded_objs_with_offsets])

# TODO refactor
# Diffracted amplitude
padded_objs_with_offsets, pred_diff = Lambda(lambda x:
    hh.pad_and_diffract(x, h, w, pad=False),
    name = 'pred_amplitude')(padded_objs_with_offsets)

# Reshape
pred_diff = Lambda(lambda x: hh._flat_to_channel(x), name = 'pred_diff_channels')(pred_diff)

# amplitude scaled to the correct photon count
pred_diff_scaled = inv_scaler([pred_diff])
#pred_diff_scaled = Lambda(lambda x: x[0] * x[1], name = 'scaled')([pred_diff, intensity_scale])

pred_intensity = tfpl.DistributionLambda(lambda t:
                                       (tfd.Independent(
                                           tfd.Poisson(
                                               (t**2))
                                       )))(pred_diff_scaled)


negloglik = lambda x, rv_x: -rv_x.log_prob((x))

# The first output exposes the real space object reconstruction and
# though it does not contribute to the training loss, it's used to
# calculate reconstruction errors for evaluation
autoencoder = Model([input_img], [trimmed_obj, pred_diff, pred_intensity, trimmed_obj])
#autoencoder = Model([input_img], [padded_obj, pred_diff, pred_intensity, pred_diff])

# TODO update for variable probe
#encode_obj_to_diffraction = tf.keras.Model(inputs=[padded_obj],
#                           outputs=[pred_diff])

diffraction_to_obj = tf.keras.Model(inputs=[input_img],
                           outputs=[obj])

#aux = tf.keras.Model(inputs=[input_img],
#                           outputs=[probe_amp_flat,
#                                crop_center(pred_diff), pred_diff_scaled])

#autoencoder.layers[-5].trainable_weights.extend([tprobe])

autoencoder.compile(optimizer='adam',
     loss=['mean_absolute_error', 'mean_absolute_error', negloglik, hh.total_variation_loss],
     loss_weights = [0., 0., 1., 0.0])

print (autoencoder.summary())
#plot_model(autoencoder, to_file='paper_data/str_model.png')

def train(epochs, X_train, Y_I_train):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoints= tf.keras.callbacks.ModelCheckpoint('%s/weights.{epoch:02d}.h5' %wt_path,
                                                monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=False, mode='auto', period=1)

    batch_size = params()['batch_size']
    history=autoencoder.fit([X_train * params()['intensity_scale']],
        [Y_I_train,
            X_train * params()['intensity_scale'],
            (params()['intensity_scale'] * X_train)**2,
           Y_I_train[:, :, :, :1]],
        shuffle=True, batch_size=batch_size, verbose=1,
        epochs=epochs, validation_split = 0.05,
        callbacks=[reduce_lr, earlystop])
        #callbacks=[reduce_lr, earlystop, checkpoints])
    return history
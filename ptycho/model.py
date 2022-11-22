from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.applications.vgg16 import VGG16
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
batch_size = params()['batch_size']
# TODO don't rely on this
intensity_scale = params()['intensity_scale']

# vgg = VGG16(weights='imagenet', include_top=False, input_shape=(N // 2,N // 2,3))
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(N, N, 3))
vgg.trainable = False

outputs = [vgg.get_layer('block2_conv2').output]
feat_model = Model(vgg.input, outputs)
# feat_model.trainable = False

tf.keras.backend.clear_session()
np.random.seed(2)

files=glob.glob('%s/*' %wt_path)
for file in files:
    os.remove(file)

input_img = Input(shape=(h, w, gridsize**2), name = 'input')

#x = hh.Conv_Pool_block(input_img,32,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
#x = hh.Conv_Pool_block(x,64,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
#x = hh.Conv_Pool_block(x,128,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')

x = hh.Conv_Pool_block(input_img,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
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

obj = Lambda(lambda x: hh.combine_complex(x[0], x[1]),
                     name='obj')([decoded1, decoded2])



#padded_obj = Lambda(lambda x: x, name = 'padded_obj')(obj)
padded_obj = tfkl.ZeroPadding2D(((h // 4), (w // 4)), name = 'padded_obj')(obj)
padded_obj_2 = Lambda(lambda x:
    hh.reassemble_patches(x), name = 'padded_obj_2',
    )(padded_obj)

trimmed_obj = Lambda(lambda x: x[:, (offset * (gridsize - 1)) // 2: -(offset * (gridsize - 1)) // 2,
        (offset * (gridsize - 1)) // 2: -(offset * (gridsize - 1)) // 2,
        :], name = 'trimmed_obj')(padded_obj_2)

# Extract overlapping regions of the object
padded_objs_with_offsets = Lambda(lambda x: hh.flatten_overlaps(x, fmt = 'flat'), name = 'padded_objs_with_offsets')(padded_obj_2)
# Apply the probe
padded_objs_with_offsets = Lambda(lambda x: tf.cast(tprobe, tf.complex64) * x,
                                  name = 'padded_objs_with_offsets_illuminated')(padded_objs_with_offsets)

# TODO refactor
# Diffracted amplitude
padded_objs_with_offsets, pred_diff = Lambda(lambda x:
    hh.pad_and_diffract(x, h, w, pad=False),
    name = 'pred_amplitude')(padded_objs_with_offsets)

# Reshape
pred_diff = Lambda(lambda x: hh._flat_to_channel(x), name = 'pred_diff_channels')(pred_diff)

pred_intensity = tfpl.DistributionLambda(lambda t:
                                       (tfd.Independent(
                                           tfd.Poisson(
                                               ((t * intensity_scale)**2))
                                       )))(pred_diff)

#def mul_gaussian_noise(image):
#    # image must be scaled in [0, 1]
#    with tf.name_scope('Add_gaussian_noise'):
#        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1, dtype=tf.float32)
#        noise_img = image * noise
#    return noise_img

negloglik = lambda x, rv_x: -rv_x.log_prob((x))

# The first output exposes the real space object reconstruction and
# though it does not contribute to the training loss, it's used to
# calculate reconstruction errors for evaluation
autoencoder = Model([input_img], [trimmed_obj, pred_diff, pred_intensity, pred_diff])
#autoencoder = Model([input_img], [padded_obj, pred_diff, pred_intensity, pred_diff])

encode_obj_to_diffraction = tf.keras.Model(inputs=[padded_obj],
                           outputs=[pred_diff])

diffraction_to_obj = tf.keras.Model(inputs=[input_img],
                           outputs=[obj])

autoencoder.compile(optimizer='adam',
     loss=['mean_absolute_error', 'mean_absolute_error', negloglik, hh.total_variation_loss],
     loss_weights = [0., 0., 1., 0.])

print (autoencoder.summary())
#plot_model(autoencoder, to_file='paper_data/str_model.png')

def train(epochs, X_train, Y_I_train):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoints= tf.keras.callbacks.ModelCheckpoint('%s/weights.{epoch:02d}.h5' %wt_path,
                                                monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=False, mode='auto', period=1)


    history=autoencoder.fit([X_train], [Y_I_train, X_train, (intensity_scale * X_train)**2,
                                       X_train], shuffle=True, batch_size=batch_size, verbose=1,
                               epochs=epochs, validation_split = 0.05, callbacks=[reduce_lr, earlystop, checkpoints])
    return history

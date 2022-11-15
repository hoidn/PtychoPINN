from .tf_helper import *
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras import Model


tf.keras.backend.clear_session()
np.random.seed(123)

# files=glob.glob('%s/*' %wt_path)
# for file in files:
#     os.remove(file)

h,w=64,64
nepochs=60
wt_path = 'wts4' #Where to store network weights
batch_size = 32

#Keras modules
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, UpSampling2D
from tensorflow.keras import Sequential
from tensorflow.keras import Input
import numpy as np

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
    

#checkpoints= tf.keras.callbacks.ModelCheckpoint('%s/weights.{epoch:02d}.hdf5' %wt_path, 
#                                            monitor='val_loss', verbose=1, save_best_only=True, 
#                                            save_weights_only=False, mode='auto', period=1)

def train(X_train, Y_I_train, Y_phi_train):
    tf.keras.backend.clear_session()
    input_img = Input(shape=(h, w, 1)) 

    x = Conv_Pool_block(input_img,32,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
    x = Conv_Pool_block(x,64,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
    x = Conv_Pool_block(x,128,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
    #Activations are all ReLu

    encoded=x

    #Decoding arm 1
    x1=Conv_Up_block(encoded,128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x1=Conv_Up_block(x1,64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x1=Conv_Up_block(x1,32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')

    decoded1 = Conv2D(1, (3, 3), padding='same')(x1)


    #Decoding arm 2
    x2=Conv_Up_block(encoded,128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x2=Conv_Up_block(x2,64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x2=Conv_Up_block(x2,32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')

    decoded2 = Conv2D(1, (3, 3), padding='same')(x2)
    #Put together
    autoencoder = Model(input_img, [decoded1, decoded2])

    #parallel_model = ModelMGPU(autoencoder, gpus=num_GPU)
    #parallel_model.compile(optimizer='adam', loss='mean_absolute_error')
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')

    print (autoencoder.summary())
    #plot_model(autoencoder, to_file='paper_data/str_model.png')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)

    history=autoencoder.fit(X_train, [Y_I_train, Y_phi_train], shuffle=True, batch_size=batch_size, verbose=1, 
                           epochs=nepochs, validation_split = 0.05, callbacks=[reduce_lr])
    return autoencoder

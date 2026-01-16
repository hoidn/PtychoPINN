"""Supervised learning baseline models for ptychographic reconstruction benchmarking.

This module provides traditional supervised learning approaches to ptychographic 
reconstruction that serve as performance baselines for evaluating the physics-informed 
neural network (PINN) approach. The primary implementation is a dual-output U-Net 
architecture that directly maps diffraction patterns to object amplitude and phase 
without incorporating physics constraints.

Architecture Overview:
    The baseline model uses a standard encoder-decoder architecture with two separate
    decoding branches - one for amplitude reconstruction and one for phase 
    reconstruction. Unlike the main PtychoPINN model, this approach:
    
    - Learns reconstruction purely from data without physics simulation
    - Requires ground truth amplitude and phase for supervised training
    - Uses standard convolutional layers without differentiable physics
    - Provides faster training but potentially less physically consistent results

Key Components:
    - Conv_Pool_block: Encoder blocks with conv-relu-conv-relu-maxpool pattern
    - Conv_Up_block: Decoder blocks with conv-relu-conv-relu-upsample pattern  
    - build_model: Creates dual-output U-Net with shared encoder, separate decoders
    - train: Handles model training with early stopping and learning rate scheduling

Comparison Framework Integration:
    This module integrates with the broader model comparison infrastructure through:
    - Standardized training interface compatible with comparison scripts
    - Common output formats for fair evaluation against PINN models
    - Support for the same data preprocessing and evaluation metrics
    - Integration with automated benchmarking workflows

Performance Characteristics:
    - Faster training than physics-informed approaches (no simulation overhead)
    - Requires paired training data (diffraction patterns + ground truth objects)
    - May struggle with out-of-distribution scanning positions or probe conditions
    - Provides upper bound on pure data-driven reconstruction quality

The baseline serves as a critical reference point for evaluating whether the added
complexity of physics-informed training provides meaningful improvements over
traditional supervised learning approaches.

Example:
    # Basic baseline model training
    autoencoder = build_model(X_train, Y_I_train, Y_phi_train)
    trained_model, history = train(X_train, Y_I_train, Y_phi_train, autoencoder)
    
    # Use in comparison workflow
    from ptycho.workflows.comparison import compare_models
    baseline_results = compare_models(baseline_model, pinn_model, test_data)

References:
    Based on PtychoNN implementation:
    https://github.com/mcherukara/PtychoNN/tree/master/TF2
"""
# based on https://github.com/mcherukara/PtychoNN/tree/master/TF2
# with minor changes to make comparison to PtychoPINN easier
from .tf_helper import *
from . import params
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras import Model

tf.keras.backend.clear_session()
np.random.seed(123)

# Model dimensions will be set dynamically in build_model()
wt_path = 'wts4' #Where to store network weights

n_filters_scale = params.params()['n_filters_scale']

#Keras modules
from tensorflow.keras.layers import UpSampling2D

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

def build_model(X_train, Y_I_train, Y_phi_train):
    tf.keras.backend.clear_session()
    # Baseline model is fundamentally a single-channel operator
    # Always use c=1 regardless of input data shape, as multi-channel data
    # should be flattened to independent samples before training
    c = 1  # Hardcoded: baseline model always processes single-channel data
    # Get dimensions from actual training data rather than global params
    h, w = X_train.shape[1], X_train.shape[2]
    input_img = Input(shape=(h, w, c))

    x = Conv_Pool_block(input_img,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
    x = Conv_Pool_block(x,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
    x = Conv_Pool_block(x,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2, padding='same', data_format='channels_last')
    #Activations are all ReLu

    encoded=x

    #Decoding arm 1
    x1=Conv_Up_block(encoded,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x1=Conv_Up_block(x1,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x1=Conv_Up_block(x1,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')

    decoded1 = Conv2D(c, (3, 3), padding='same')(x1)

    #Decoding arm 2
    x2=Conv_Up_block(encoded,n_filters_scale * 128,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x2=Conv_Up_block(x2,n_filters_scale * 64,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')
    x2=Conv_Up_block(x2,n_filters_scale * 32,w1=3,w2=3,p1=2,p2=2,padding='same', data_format='channels_last')

    decoded2 = Conv2D(c, (3, 3), padding='same')(x2)
    #Put together
    autoencoder = Model(input_img, [decoded1, decoded2])
    # Masked MAE creates a more apples-to-apples comparison with the main
    # model, but it doesn't seem to affect the image quality
    #autoencoder.compile(optimizer='adam', loss=masked_mae)
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    return autoencoder

def train(X_train, Y_I_train, Y_phi_train, autoencoder = None):
    if autoencoder is None:
        autoencoder = build_model(X_train, Y_I_train, Y_phi_train)

    print (autoencoder.summary())
    #plot_model(autoencoder, to_file='paper_data/str_model.png')

    # Get current values from params (not the stale global variables)
    current_nepochs = params.get('nepochs')
    current_batch_size = params.get('batch_size')
    
    print(f"Training with {current_nepochs} epochs and batch size {current_batch_size}")

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)

    #history=autoencoder.fit(X_train * params.params()['intensity_scale'],
    history=autoencoder.fit(X_train,
        [Y_I_train, Y_phi_train], shuffle=True,
        batch_size=current_batch_size, verbose=1, epochs=current_nepochs,
        validation_split = 0.05, callbacks=[reduce_lr, earlystop])
    return autoencoder, history

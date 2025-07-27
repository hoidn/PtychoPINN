"""Core physics-informed neural network architecture for ptychographic reconstruction.

This module defines the heart of the PtychoPINN system - a U-Net-based deep learning 
architecture augmented with custom physics-informed Keras layers that embed ptychographic 
forward modeling constraints directly into the neural network. This unique combination 
enables rapid, high-resolution reconstruction from scanning coherent diffraction data 
while maintaining physical consistency.

**Architecture Overview:**
The PtychoPINN model integrates three key components:
1. **Encoder-Decoder U-Net**: Learns image-to-image mapping from diffraction to object
2. **Physics Constraint Layers**: Custom layers implementing differentiable ptychography
3. **Probabilistic Loss**: Poisson noise modeling for realistic diffraction statistics

**CRITICAL: This is a core physics module with stable, validated implementations.**
The model architecture and physics simulation components should NOT be modified 
without explicit requirements and thorough validation.

Key Components:
    - ProbeIllumination: Custom layer applying complex probe illumination with smoothing
    - IntensityScaler/IntensityScaler_inv: Trainable intensity scaling for normalization  
    - U-Net Architecture: Resolution-adaptive encoder-decoder with dynamic filter scaling
    - Physics Integration: Differentiable forward model with Poisson noise simulation
    - Model Factory: create_model_with_gridsize() for explicit parameter control

**Primary Models Created:**
    - autoencoder: Main training model (diffraction -> object, amplitude, intensity)
    - diffraction_to_obj: Inference-only model (diffraction -> object reconstruction)
    - autoencoder_no_nll: Training model without negative log-likelihood output

**Core Workflows:**
    - Model Creation: Uses global config or explicit parameters via factory functions
    - Training Pipeline: Integration with PtychoDataContainer for data formatting
    - Physics Simulation: Differentiable forward model for end-to-end training
    - Inference: Direct diffraction-to-object reconstruction

Usage Example:
    ```python
    from ptycho.model import diffraction_to_obj, create_model_with_gridsize
    from ptycho.loader import PtychoDataContainer
    
    # Direct inference using global model (legacy)
    reconstruction = diffraction_to_obj.predict([diffraction_data, coordinates])
    
    # Create model with explicit parameters (modern approach)
    autoenc, inference = create_model_with_gridsize(gridsize=2, N=64)
    result = inference.predict([test_data, test_coords])
    
    # Training workflow with data container
    train_data = PtychoDataContainer(...)
    inputs = prepare_inputs(train_data)
    outputs = prepare_outputs(train_data)
    history = autoencoder.fit(inputs, outputs, epochs=50)
    ```

**Architecture Details:**
The U-Net architecture adapts filter counts based on input resolution (N):
- N=64: [32, 64, 128] -> [64, 32] filters (encoder -> decoder)  
- N=128: [16, 32, 64, 128] -> [128, 64, 32] filters
- N=256: [8, 16, 32, 64, 128] -> [256, 128, 64, 32] filters

Custom physics layers implement:
- Complex-valued probe illumination with Gaussian smoothing
- Patch extraction for multi-position scanning geometry  
- Differentiable Fourier transform for diffraction simulation
- Poisson noise modeling for realistic measurement statistics

Dependencies:
    - ptycho.tf_helper: Core TensorFlow operations and physics simulation functions
    - ptycho.loader: PtychoDataContainer for structured data handling
    - ptycho.params: Legacy global configuration system (cfg dictionary)
    - ptycho.probe: Probe initialization and processing utilities

**Global State Warning:**
The module creates model instances at import time using global configuration. 
For new code, prefer create_model_with_gridsize() to avoid global state dependencies.
"""

# TODO s
# - complex convolution
# - Use tensor views:
#     https://chat.openai.com/c/e6d5e400-daf9-44b7-8ef9-d49f21a634a3
# -difference maps?
# -double -> float32
# Apply real space loss to both amplitude and phase of the object

from datetime import datetime
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.activations import relu, sigmoid, tanh, swish, softplus
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, InputLayer, Lambda, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import glob
import math
import numpy as np
import os
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from .loader import PtychoDataContainer
from . import tf_helper as hh
from . import params as p

# Import native Gaussian filter implementation instead of tensorflow_addons
from .gaussian_filter import gaussian_filter2d, complex_gaussian_filter2d as complex_gaussian_filter2d_native

# Use the native complex gaussian filter implementation directly
complex_gaussian_filter2d = complex_gaussian_filter2d_native

tfk = hh.tf.keras
tfkl = hh.tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

wt_path = 'wts4.1'
# sets the number of convolutional filters

n_filters_scale =  p.get('n_filters_scale')
N = p.get('N')
gridsize = p.get('gridsize')
offset = p.get('offset')

from . import probe
tprobe = p.params()['probe']

probe_mask = probe.get_probe_mask(N)
#probe_mask = cfg.get('probe_mask')[:, :, :, 0]

if len(tprobe.shape) == 3:
    initial_probe_guess = tprobe[None, ...]
    #probe_mask = probe_mask[None, ...]
elif len(tprobe.shape) == 4:
    initial_probe_guess = tprobe
else:
    raise ValueError

initial_probe_guess = tf.Variable(
            initial_value=tf.cast(initial_probe_guess, tf.complex64),
            trainable=p.params()['probe.trainable'],
        )

# TODO hyperparameters:
# TODO total variation loss
# -probe smoothing scale(?)
class ProbeIllumination(tf.keras.layers.Layer):
    def __init__(self, name = None):
        super(ProbeIllumination, self).__init__(name = name)
        self.w = initial_probe_guess
        self.sigma = p.get('gaussian_smoothing_sigma')

    def call(self, inputs):
        # x is expected to have shape (batch_size, N, N, gridsize**2)
        # where N is the size of each patch and gridsize**2 is the number of patches
        x = inputs[0]
        
        # self.w has shape (1, N, N, 1) or (1, N, N, gridsize**2) if probe.big is True
        # probe_mask has shape (N, N, 1)
        
        # Apply multiplication first
        illuminated = self.w * x
        
        # Apply Gaussian smoothing only if sigma is not 0
        if self.sigma != 0:
            smoothed = complex_gaussian_filter2d(illuminated, filter_shape=(3, 3), sigma=self.sigma)
        else:
            smoothed = illuminated
        
        if p.get('probe.mask'):
            # Output shape: (batch_size, N, N, gridsize**2)
            return smoothed * tf.cast(probe_mask, tf.complex64), (self.w * tf.cast(probe_mask, tf.complex64))[None, ...]
        else:
            # Output shape: (batch_size, N, N, gridsize**2)
            return smoothed, (self.w)[None, ...]

probe_illumination = ProbeIllumination()

nphotons = p.get('nphotons')

# TODO scaling could be done on a shot-by-shot basis, but IIRC I tried this
# and there were issues
log_scale_guess = np.log(p.get('intensity_scale'))
log_scale = tf.Variable(
            initial_value=tf.constant(float(log_scale_guess)),
            trainable = p.params()['intensity_scale.trainable'],
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

tf.keras.backend.clear_session()
np.random.seed(2)

files=glob.glob('%s/*' %wt_path)
for file in files:
    os.remove(file)

lambda_norm = Lambda(lambda x: tf.math.reduce_sum(x**2, axis = [1, 2]))
input_img = Input(shape=(N, N, gridsize**2), name = 'input')
input_positions = Input(shape=(1, 2, gridsize**2), name = 'input_positions')

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

def create_encoder(input_tensor, n_filters_scale):
    N = p.get('N')
    
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

def create_decoder_base(input_tensor, n_filters_scale):
    N = p.get('N')
    
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

def get_resolution_scale_factor(N):
    """
    Calculate the resolution-dependent filter count programmatically.
    
    Args:
    N (int): The input resolution (must be a power of 2)
    
    Returns:
    int: The scale factor for the given resolution
    
    Raises:
    ValueError: If the input size is not a power of 2 or is outside the supported range
    """
    if N < 64 or N > 1024:
        raise ValueError(f"Input size {N} is outside the supported range (64 to 1024)")
    
    if not (N & (N - 1) == 0) or N == 0:
        raise ValueError(f"Input size {N} is not a power of 2")
    
    # Calculate the scale factor
    # For N=64, we want 32; for N=128, we want 16; for N=256, we want 8, etc.
    # This can be achieved by dividing 2048 by N
    return 2048 // N

def create_decoder_last(input_tensor, n_filters_scale, conv1, conv2, act=tf.keras.activations.sigmoid, name=''):
    N = p.get('N')
    gridsize = p.get('gridsize')

    scale_factor = get_resolution_scale_factor(N)
    if p.get('pad_object'):
        c_outer = 4
        x1 = conv1(input_tensor[..., :-c_outer])
        x1 = act(x1)
        x1 = tf.keras.layers.ZeroPadding2D(((N // 4), (N // 4)), name=name + '_padded')(x1)
        
        if not p.get('probe.big'):
            return x1
        
        x2 = Conv_Up_block(input_tensor[..., -c_outer:], n_filters_scale * scale_factor)
        x2 = conv2(x2)
        x2 = swish(x2)
        
        # Drop the central region of x2
        center_mask = hh.mk_centermask(x2, N, 1, kind='border')
        x2_masked = x2 * center_mask
        
        outputs = x1 + x2_masked
        return outputs

    else:
        x2 = Conv_Up_block(input_tensor, n_filters_scale * scale_factor)
        x2 = conv2(x2)
        x2 = act(x2)
        return x2


def create_decoder_phase(input_tensor, n_filters_scale, gridsize, big):
    num_filters = gridsize**2 if big else 1
    conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    act = tf.keras.layers.Lambda(lambda x: math.pi * tf.keras.activations.tanh(x), name='phi')
    
    N = p.get('N')
    
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
    
    outputs = create_decoder_last(x, n_filters_scale, conv1, conv2, act=act, name='phase')
    return outputs


def create_autoencoder(input_tensor, n_filters_scale, gridsize, big):
    encoded = create_encoder(input_tensor, n_filters_scale)
    decoded_amp = create_decoder_amp(encoded, n_filters_scale)
    decoded_phase = create_decoder_phase(encoded, n_filters_scale, gridsize, big)
    
    return decoded_amp, decoded_phase


def get_amp_activation():
    if p.get('amp_activation') == 'sigmoid':
        return lambda x: sigmoid(x)
    elif p.get('amp_activation') == 'swish':
        return lambda x: swish(x)
    elif p.get('amp_activation') == 'softplus':
        return lambda x: softplus(x)
    elif p.get('amp_activation') == 'relu':
        return lambda x: relu(x)
    else:
        return ValueError

def create_decoder_amp(input_tensor, n_filters_scale):
    # Placeholder convolution layers and activation as defined in the original DecoderAmp class
    conv1 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(1, (3, 3), padding='same')
    act = Lambda(get_amp_activation(), name='amp')

    x = create_decoder_base(input_tensor, n_filters_scale)
    outputs = create_decoder_last(x, n_filters_scale, conv1, conv2, act=act,
        name = 'amp')
    return outputs

normed_input = scale([input_img])
decoded1, decoded2 = create_autoencoder(normed_input, n_filters_scale, gridsize,
    p.get('object.big'))

# Combine the two decoded outputs
obj = Lambda(lambda x: hh.combine_complex(x[0], x[1]), name='obj')([decoded1, decoded2])

if p.get('object.big'):
    # If 'object.big' is true, reassemble the patches
    padded_obj_2 = Lambda(lambda x: hh.reassemble_patches(x[0], fn_reassemble_real=hh.mk_reassemble_position_real(x[1])), name = 'padded_obj_2')([obj, input_positions])
else:
    # If 'object.big' is not true, pad the reconstruction
    padded_obj_2 = Lambda(lambda x: hh.pad_reconstruction(x), name = 'padded_obj_2')(obj)

# TODO rename?
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
pred_diff = Lambda(lambda x: hh._flat_to_channel(x, N=N, gridsize=gridsize), name = 'pred_diff_channels')(pred_diff)

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
def negloglik(x, rv_x):
    return -rv_x.log_prob(x)
fn_poisson_nll = lambda A_target, A_pred: negloglik(A_target**2, dist_poisson_intensity(A_pred))

autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])

autoencoder_no_nll = Model(inputs = [input_img, input_positions],
        outputs = [pred_amp_scaled])

#encode_obj_to_diffraction = tf.keras.Model(inputs=[obj, input_positions],
#                           outputs=[pred_diff, flat_illuminated])
diffraction_to_obj = tf.keras.Model(inputs=[input_img, input_positions],
                           outputs=[trimmed_obj])

mae_weight = p.get('mae_weight') # should normally be 0
nll_weight = p.get('nll_weight') # should normally be 1
# Total variation regularization on real space amplitude
realspace_weight = p.get('realspace_weight')#1e2
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

def prepare_inputs(train_data: PtychoDataContainer):
    """training inputs"""
    return [train_data.X * p.get('intensity_scale'), train_data.coords]

def prepare_outputs(train_data: PtychoDataContainer):
    """training outputs"""
    return [hh.center_channels(train_data.Y_I, train_data.coords)[:, :, :, :1],
                (p.get('intensity_scale') * train_data.X),
                (p.get('intensity_scale') * train_data.X)**2]

#def train(epochs, X_train, coords_train, Y_obj_train):
def train(epochs, trainset: PtychoDataContainer):
    assert type(trainset) == PtychoDataContainer
    coords_train = trainset.coords
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoints= tf.keras.callbacks.ModelCheckpoint(
                            '%s/weights.{epoch:02d}.h5' %wt_path,
                            monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)

    batch_size = p.params()['batch_size']
    history=autoencoder.fit(
#        prepare_inputs(X_train, coords_train),
#        prepare_outputs(Y_obj_train, coords_train, X_train),
        prepare_inputs(trainset),
        prepare_outputs(trainset),
        shuffle=True, batch_size=batch_size, verbose=1,
        epochs=epochs, validation_split = 0.05,
        callbacks=[reduce_lr, earlystop])
        #callbacks=[reduce_lr, earlystop, tboard_callback])
    return history
import numpy as np

def print_model_diagnostics(model):
    """
    Prints diagnostic information for a given TensorFlow/Keras model.

    Parameters:
    - model: A TensorFlow/Keras model object.
    """
    # Print the model summary to get the architecture, layer types, output shapes, and parameter counts.
    model.summary()

    # Print input shape
    print("Model Input Shape(s):")
    for input_layer in model.inputs:
        print(input_layer.shape)

    # Print output shape
    print("Model Output Shape(s):")
    for output_layer in model.outputs:
        print(output_layer.shape)

    # Print total number of parameters
    print("Total Parameters:", model.count_params())

    # Print trainable and non-trainable parameter counts
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print("Trainable Parameters:", trainable_count)
    print("Non-trainable Parameters:", non_trainable_count)

    # If the model uses any custom layers, print their names and configurations
    print("Custom Layers (if any):")
    for layer in model.layers:
        if hasattr(layer, 'custom_objects'):
            print(f"{layer.name}: {layer.custom_objects}")

def create_model_with_gridsize(gridsize: int, N: int, **kwargs):
    """
    Create model with explicit gridsize parameter to eliminate global state dependency.
    
    Args:
        gridsize: Grid size for neighbor patch processing
        N: Image size parameter
        **kwargs: Other model configuration parameters
    
    Returns:
        Tuple of (autoencoder, diffraction_to_obj) models
    """
    # Store current global state for restoration
    original_gridsize = p.get('gridsize')
    original_N = p.get('N')
    
    try:
        # Temporarily update global state for model construction
        p.cfg.update({'gridsize': gridsize, 'N': N})
        for key, value in kwargs.items():
            p.cfg.update({key: value})
        
        # Create input layers with explicit parameters
        input_img = Input(shape=(N, N, gridsize**2), name='input')
        input_positions = Input(shape=(1, 2, gridsize**2), name='input_positions')
        
        # Create model components using explicit parameters
        normed_input = scale([input_img])
        decoded1, decoded2 = create_autoencoder(normed_input, p.get('n_filters_scale'), gridsize, p.get('object.big'))
        
        # Combine the decoded outputs
        obj = Lambda(lambda x: hh.combine_complex(x[0], x[1]), name='obj')([decoded1, decoded2])
        
        if p.get('object.big'):
            # If 'object.big' is true, reassemble the patches
            padded_obj_2 = Lambda(lambda x: hh.reassemble_patches(x[0], fn_reassemble_real=hh.mk_reassemble_position_real(x[1])), name='padded_obj_2')([obj, input_positions])
        else:
            # If 'object.big' is not true, pad the reconstruction
            padded_obj_2 = Lambda(lambda x: hh.pad_reconstruction(x), name='padded_obj_2')(obj)
        
        # Trim the object reconstruction to N x N
        trimmed_obj = Lambda(hh.trim_reconstruction, name='trimmed_obj')(padded_obj_2)
        
        # Extract overlapping regions of the object
        padded_objs_with_offsets = Lambda(lambda x:
            hh.extract_patches_position(x[0], x[1], 0.),
            name='padded_objs_with_offsets')([padded_obj_2, input_positions])
        
        # Apply the probe illumination
        padded_objs_with_offsets, probe = probe_illumination([padded_objs_with_offsets])
        flat_illuminated = padded_objs_with_offsets
        
        # Apply pad and diffract operation
        padded_objs_with_offsets, pred_diff = Lambda(lambda x: hh.pad_and_diffract(x, N, N, pad=False), name='pred_amplitude')(padded_objs_with_offsets)
        
        # Reshape with explicit parameters
        pred_diff = Lambda(lambda x: hh._flat_to_channel(x, N=N, gridsize=gridsize), name='pred_diff_channels')(pred_diff)
        
        # Scale the amplitude
        pred_amp_scaled = inv_scale([pred_diff])
        
        # Create Poisson noise layer
        from tensorflow_probability import layers as tfpl
        from tensorflow_probability import distributions as tfd
        dist_poisson_intensity = tfpl.DistributionLambda(lambda amplitude:
                                           (tfd.Independent(
                                               tfd.Poisson(
                                                   (amplitude**2)))))
        pred_intensity_sampled = dist_poisson_intensity(pred_amp_scaled)
        
        # Create models
        autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])
        diffraction_to_obj = Model(inputs=[input_img, input_positions], outputs=[trimmed_obj])
        
        return autoencoder, diffraction_to_obj
        
    finally:
        # Restore original global state
        p.cfg.update({'gridsize': original_gridsize, 'N': original_N})

def _create_models_from_global_config():
    """Create models using global configuration (for backward compatibility)."""
    gridsize = p.get('gridsize')
    N = p.get('N')
    return create_model_with_gridsize(gridsize, N)

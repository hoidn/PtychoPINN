"""Core physics-informed neural network architecture for ptychographic reconstruction.

**⚠️ CRITICAL GLOBAL STATE WARNING ⚠️**
This module suffers from a major architectural flaw: it creates model instances (autoencoder, 
diffraction_to_obj, autoencoder_no_nll) at import time using the global ptycho.params.cfg 
dictionary. This creates hidden dependencies and makes the models dependent on global state 
at import time, which is extremely problematic for testing, reproducibility, and concurrent usage.

**STRONGLY RECOMMENDED:** Use create_model_with_gridsize() factory function instead of the 
module-level model instances. This eliminates global state dependencies and provides explicit 
parameter control.

This module implements a U-Net-based physics-informed neural network that combines deep learning 
with differentiable ptychographic forward modeling. The architecture integrates encoder-decoder 
image reconstruction with custom Keras layers that enforce ptychographic physics constraints.

**Public Interface:**

Factory Functions (Recommended):
    - create_model_with_gridsize(gridsize, N, **kwargs) -> (autoencoder, diffraction_to_obj)
      Creates models with explicit parameters, avoiding global state dependencies.

Module-Level Models (Legacy - Avoid in New Code):
    - autoencoder: Main training model with 3 outputs [object, amplitude, intensity]
    - diffraction_to_obj: Inference-only model with 1 output [object]
    - autoencoder_no_nll: Training model without negative log-likelihood output

Utility Functions:
    - prepare_inputs(train_data: PtychoDataContainer) -> [scaled_diffraction, coordinates]
    - prepare_outputs(train_data: PtychoDataContainer) -> [object, amplitude, intensity]

**Model Input/Output Specifications:**

Inputs (both models):
    - diffraction: tf.float32, shape (batch_size, N, N, gridsize**2)
      Measured diffraction amplitudes (sqrt of intensity)
    - coordinates: tf.float32, shape (batch_size, 1, 2, gridsize**2) 
      Scanning probe positions in normalized coordinates

Outputs:
    - autoencoder: [object, amplitude, intensity]
      * object: tf.complex64, shape (batch_size, N, N, 1) - Complex object reconstruction
      * amplitude: tf.float32, shape (batch_size, N, N, gridsize**2) - Predicted amplitudes
      * intensity: tf.float32, shape (batch_size, N, N, gridsize**2) - Squared amplitudes
    - diffraction_to_obj: [object]
      * object: tf.complex64, shape (batch_size, N, N, 1) - Complex object reconstruction only

**Architecture Components:**

U-Net Encoder-Decoder:
    Resolution-adaptive filter scaling based on input size N:
    - N=64: [32, 64, 128] -> [64, 32] filters (encoder -> decoder)
    - N=128: [16, 32, 64, 128] -> [128, 64, 32] filters  
    - N=256: [8, 16, 32, 64, 128] -> [256, 128, 64, 32] filters

Custom Physics Layers:
    - ProbeIllumination: Complex probe multiplication with Gaussian smoothing
    - ExtractPatchesPositionLayer: Multi-position patch extraction from object
    - PadAndDiffractLayer: Differentiable Fourier transform diffraction simulation
    - IntensityScaler/IntensityScaler_inv: Trainable intensity normalization

**Usage Examples:**

Recommended approach (explicit parameters):
    ```python
    # Create models with explicit configuration
    autoenc, inference_model = create_model_with_gridsize(gridsize=2, N=64)
    
    # Inference
    object_reconstruction = inference_model.predict([diffraction_data, coordinates])
    
    # Training
    history = autoenc.fit(
        prepare_inputs(train_data),
        prepare_outputs(train_data), 
        epochs=50
    )
    ```

Legacy approach (avoid - uses global state):
    ```python
    from ptycho.model import diffraction_to_obj  # Depends on global ptycho.params.cfg
    reconstruction = diffraction_to_obj.predict([diffraction_data, coordinates])
    ```

**State Dependencies:**
- ptycho.params.cfg: Global configuration dictionary (legacy dependency)
- Global probe initialization from ptycho.probe module
- Import-time model creation with current global parameter values

**Integration Points:**
- Training: ptycho.workflows.components orchestrates complete training workflows
- Data Loading: ptycho.loader.PtychoDataContainer provides structured data interface
- Physics: ptycho.tf_helper implements core differentiable operations
- Configuration: ptycho.config provides modern dataclass-based configuration
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
from .custom_layers import (CombineComplexLayer, ExtractPatchesPositionLayer, 
                           PadReconstructionLayer, ReassemblePatchesLayer,
                           TrimReconstructionLayer, PadAndDiffractLayer,
                           FlatToChannelLayer, ScaleLayer, InvScaleLayer,
                           ActivationLayer, SquareLayer)
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
    def __init__(self, name=None, **kwargs):
        # Remove any kwargs that shouldn't be passed to parent
        kwargs.pop('dtype', None)  # Handle dtype separately if needed
        super(ProbeIllumination, self).__init__(name=name, **kwargs)
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
    
    def compute_output_shape(self, input_shape):
        # Returns two outputs - both with same shape as input
        return [input_shape, input_shape]
    
    def get_config(self):
        config = super().get_config()
        # Don't need to save w or sigma as they come from global state
        return config

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
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super(IntensityScaler, self).__init__(**kwargs)
        self.w = log_scale
    def call(self, inputs):
        x, = inputs
        return x / tf.math.exp(self.w)
    def get_config(self):
        return super().get_config()

# TODO use a bijector instead of separately defining the transform and its
# inverse
class IntensityScaler_inv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super(IntensityScaler_inv, self).__init__(**kwargs)
        self.w = log_scale
    def call(self, inputs):
        x, = inputs
        return tf.math.exp(self.w) * x
    def get_config(self):
        return super().get_config()

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

lambda_norm = Lambda(lambda x: tf.math.reduce_sum(x**2, axis = [1, 2]), output_shape=lambda s: (s[0], s[3]))
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
    # Use custom activation layer for phase
    act = ActivationLayer(activation_name='tanh', scale=math.pi, name='phi')
    
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
    # Use custom activation layer for amplitude
    try:
        amp_activation_name = p.get('amp_activation')
    except KeyError:
        amp_activation_name = 'sigmoid'
    act = ActivationLayer(activation_name=amp_activation_name, name='amp')

    x = create_decoder_base(input_tensor, n_filters_scale)
    outputs = create_decoder_last(x, n_filters_scale, conv1, conv2, act=act,
        name = 'amp')
    return outputs

normed_input = IntensityScaler(name='intensity_scaler')([input_img])

# Get N value for output shapes
N = p.params()['N']

decoded1, decoded2 = create_autoencoder(normed_input, n_filters_scale, gridsize,
    p.get('object.big'))

# Combine the two decoded outputs
obj = CombineComplexLayer(name='obj')([decoded1, decoded2])

if p.get('object.big'):
    # If 'object.big' is true, reassemble the patches
    # Calculate output shape dynamically based on padded_size
    from .params import get_padded_size
    padded_size = get_padded_size()
    padded_obj_2 = ReassemblePatchesLayer(
        dtype=tf.complex64,
        name='padded_obj_2'
    )([obj, input_positions])
else:
    # If 'object.big' is not true, pad the reconstruction
    from .params import get_padded_size
    padded_size = get_padded_size()
    padded_obj_2 = PadReconstructionLayer(
        dtype=tf.complex64,
        name='padded_obj_2'
    )(obj)

# TODO rename?
# Trim the object reconstruction to N x N
trimmed_obj = TrimReconstructionLayer(output_size=N, dtype=tf.complex64, name='trimmed_obj')(padded_obj_2)

# Extract overlapping regions of the object
# Output shape should be (batch, N, N, 1) where N is the patch size
padded_objs_with_offsets = ExtractPatchesPositionLayer(
    jitter=0.0,
    dtype=tf.complex64,
    name='padded_objs_with_offsets'
)([padded_obj_2, input_positions])

# Apply the probe illumination
padded_objs_with_offsets, probe = probe_illumination([padded_objs_with_offsets])
flat_illuminated = padded_objs_with_offsets

# Apply pad and diffract operation
pad_diffract_layer = PadAndDiffractLayer(h=N, w=N, pad=False, name='pred_amplitude')
padded_objs_with_offsets, pred_diff = pad_diffract_layer(padded_objs_with_offsets)

# Reshape
pred_diff = FlatToChannelLayer(N=N, gridsize=gridsize, name='pred_diff_channels')(pred_diff)

# Scale the amplitude
pred_amp_scaled = IntensityScaler_inv(name='intensity_scaler_inv')([pred_diff])


# TODO Please pass an integer value for `reinterpreted_batch_ndims`. The current behavior corresponds to `reinterpreted_batch_ndims=tf.size(distribution.batch_shape_tensor()) - 1`.
# In TF 2.19, using TFP distributions as model outputs is problematic
# We'll handle the distribution in the loss function instead
# For now, just use the squared amplitude as a placeholder
pred_intensity_sampled = SquareLayer(name='pred_intensity')(pred_amp_scaled)

# Create the distribution function for use in loss calculation
def create_poisson_distribution_for_loss(amplitude):
    squared = tf.square(amplitude) 
    return tfd.Independent(tfd.Poisson(squared), reinterpreted_batch_ndims=3)

# We'll use this in the loss function
dist_poisson_intensity = create_poisson_distribution_for_loss

# Poisson distribution over expected diffraction intensity (i.e. photons per
# pixel)
def negloglik(y_true, y_pred):
    """Compute Poisson negative log-likelihood using TensorFlow's built-in function"""
    # y_true is the target intensity (already squared)
    # y_pred is the predicted intensity (already squared)
    # Use TensorFlow's Poisson loss which computes: y_pred - y_true * log(y_pred)
    return tf.nn.log_poisson_loss(y_true, tf.math.log(y_pred), compute_full_loss=False)

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

# Check if XLA compilation is enabled via environment or config
use_xla_compile = os.environ.get('USE_XLA_COMPILE', '').lower() in ('1', 'true', 'yes')
# Check if parameter exists in config
try:
    use_xla_compile = use_xla_compile or p.get('use_xla_compile')
except KeyError:
    pass  # Parameter doesn't exist, keep environment value

autoencoder.compile(optimizer= optimizer,
     #loss=[lambda target, pred: hh.total_variation(pred),
     loss=[hh.realspace_loss,
        'mean_absolute_error', negloglik],
     loss_weights = [realspace_weight, mae_weight, nll_weight],
     jit_compile=use_xla_compile)

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
                            save_weights_only=False, mode='auto', save_freq='epoch')

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
        
        # Get padded size for later use
        from .params import get_padded_size
        padded_size = get_padded_size()
        
        # Create model components using explicit parameters
        normed_input = IntensityScaler(name='intensity_scaler')([input_img])
        decoded1, decoded2 = create_autoencoder(normed_input, p.get('n_filters_scale'), gridsize, p.get('object.big'))
        
        # Combine the decoded outputs
        obj = CombineComplexLayer(name='obj')([decoded1, decoded2])
        
        if p.get('object.big'):
            # If 'object.big' is true, reassemble the patches
            padded_obj_2 = ReassemblePatchesLayer(
                dtype=tf.complex64,
                name='padded_obj_2'
            )([obj, input_positions])
        else:
            # If 'object.big' is not true, pad the reconstruction
            padded_obj_2 = PadReconstructionLayer(
                dtype=tf.complex64,
                name='padded_obj_2'
            )(obj)
        
        # Trim the object reconstruction to N x N
        trimmed_obj = TrimReconstructionLayer(
            output_size=N,
            dtype=tf.complex64,
            name='trimmed_obj'
        )(padded_obj_2)
        
        # Extract overlapping regions of the object
        padded_objs_with_offsets = ExtractPatchesPositionLayer(
            jitter=0.0,
            dtype=tf.complex64,
            name='padded_objs_with_offsets'
        )([padded_obj_2, input_positions])
        
        # Apply the probe illumination
        padded_objs_with_offsets, probe = probe_illumination([padded_objs_with_offsets])
        flat_illuminated = padded_objs_with_offsets
        
        # Apply pad and diffract operation
        pad_diffract_layer = PadAndDiffractLayer(h=N, w=N, pad=False, name='pred_amplitude')
        padded_objs_with_offsets, pred_diff = pad_diffract_layer(padded_objs_with_offsets)
        
        # Reshape with explicit parameters
        pred_diff = FlatToChannelLayer(N=N, gridsize=gridsize, name='pred_diff_channels')(pred_diff)
        
        # Scale the amplitude
        pred_amp_scaled = IntensityScaler_inv(name='intensity_scaler_inv')([pred_diff])
        
        # In TF 2.19, using TFP distributions as model outputs is problematic
        # We'll handle the distribution in the loss function instead
        # For now, just use the squared amplitude as a placeholder
        pred_intensity_sampled = SquareLayer(name='pred_intensity')(pred_amp_scaled)
        
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

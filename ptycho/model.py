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

# Module-level cache for lazy-loaded singletons (REFACTOR-MODEL-SINGLETON-001)
# See docs/findings.md MODULE-SINGLETON-001 for migration guide
_lazy_cache = {}
_model_construction_done = False

wt_path = 'wts4.1'
# sets the number of convolutional filters

from . import probe


def _get_initial_probe_guess():
    """Get or create the initial_probe_guess tf.Variable lazily.

    Defers tf.Variable creation until first access to avoid import-time side effects.
    See REFACTOR-MODEL-SINGLETON-001 Phase B.
    """
    if 'initial_probe_guess' not in _lazy_cache:
        tprobe = p.params()['probe']
        N = p.get('N')
        if len(tprobe.shape) == 3:
            probe_init = tprobe[None, ...]
        elif len(tprobe.shape) == 4:
            probe_init = tprobe
        else:
            raise ValueError(f"Invalid probe shape: {tprobe.shape}")
        _lazy_cache['initial_probe_guess'] = tf.Variable(
            initial_value=tf.cast(probe_init, tf.complex64),
            trainable=p.params()['probe.trainable'],
        )
    return _lazy_cache['initial_probe_guess']

# TODO hyperparameters:
# TODO total variation loss
# -probe smoothing scale(?)
@tf.keras.utils.register_keras_serializable(package='ptycho')
class ProbeIllumination(tf.keras.layers.Layer):
    def __init__(self, name=None, initial_probe=None, N=None, **kwargs):
        # Remove any kwargs that shouldn't be passed to parent
        kwargs.pop('dtype', None)  # Handle dtype separately if needed
        super(ProbeIllumination, self).__init__(name=name, **kwargs)

        # Store N for this instance
        self._N = N if N is not None else p.get('N')

        # Generate probe_mask once at construction (efficient)
        self._probe_mask = probe.get_probe_mask(self._N)

        # Use provided probe or fall back to module-level for backward compatibility
        if initial_probe is not None:
            # Create a new tf.Variable for this instance
            if len(initial_probe.shape) == 3:
                probe_init = initial_probe[None, ...]
            elif len(initial_probe.shape) == 4:
                probe_init = initial_probe
            else:
                raise ValueError(f"Invalid probe shape: {initial_probe.shape}")
            self.w = tf.Variable(
                initial_value=tf.cast(probe_init, tf.complex64),
                trainable=p.params()['probe.trainable'],
            )
        else:
            # Backward compatibility: use lazy-loaded module-level variable
            # See REFACTOR-MODEL-SINGLETON-001 Phase B
            self.w = _get_initial_probe_guess()

        self.sigma = p.get('gaussian_smoothing_sigma')

    def call(self, inputs):
        # x is expected to have shape (batch_size, N, N, gridsize**2)
        # where N is the size of each patch and gridsize**2 is the number of patches
        x = inputs[0]

        # self.w has shape (1, N, N, 1) or (1, N, N, gridsize**2) if probe.big is True

        # Apply multiplication first
        illuminated = self.w * x

        # Apply Gaussian smoothing only if sigma is not 0
        if self.sigma != 0:
            smoothed = complex_gaussian_filter2d(illuminated, filter_shape=(3, 3), sigma=self.sigma)
        else:
            smoothed = illuminated

        if p.get('probe.mask'):
            # Use pre-computed mask (generated in __init__)
            return smoothed * tf.cast(self._probe_mask, tf.complex64), (self.w * tf.cast(self._probe_mask, tf.complex64))[None, ...]
        else:
            # Output shape: (batch_size, N, N, gridsize**2)
            return smoothed, (self.w)[None, ...]

    def compute_output_shape(self, input_shape):
        # Returns two outputs - both with same shape as input
        return [input_shape, input_shape]

    def get_config(self):
        config = super().get_config()
        config['N'] = self._N
        return config

def _get_probe_illumination():
    """Get or create the ProbeIllumination singleton lazily.

    Defers layer instantiation until first access to avoid import-time side effects.
    See REFACTOR-MODEL-SINGLETON-001 Phase B.
    """
    if 'probe_illumination' not in _lazy_cache:
        _lazy_cache['probe_illumination'] = ProbeIllumination()
    return _lazy_cache['probe_illumination']


def _get_log_scale():
    """Get or create the log_scale tf.Variable lazily.

    Defers tf.Variable creation until first access to avoid import-time side effects.
    See REFACTOR-MODEL-SINGLETON-001 Phase B.
    """
    if 'log_scale' not in _lazy_cache:
        log_scale_guess = np.log(p.get('intensity_scale'))
        _lazy_cache['log_scale'] = tf.Variable(
            initial_value=tf.constant(float(log_scale_guess)),
            trainable=p.params()['intensity_scale.trainable'],
        )
    return _lazy_cache['log_scale']

@tf.keras.utils.register_keras_serializable(package='ptycho')
class IntensityScaler(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super(IntensityScaler, self).__init__(**kwargs)
        self.w = _get_log_scale()  # Lazy loading, see REFACTOR-MODEL-SINGLETON-001
    def call(self, inputs):
        x, = inputs
        return x / tf.math.exp(self.w)
    def get_config(self):
        return super().get_config()

# TODO use a bijector instead of separately defining the transform and its
# inverse
@tf.keras.utils.register_keras_serializable(package='ptycho')
class IntensityScaler_inv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super(IntensityScaler_inv, self).__init__(**kwargs)
        self.w = _get_log_scale()  # Lazy loading, see REFACTOR-MODEL-SINGLETON-001
    def call(self, inputs):
        x, = inputs
        return tf.math.exp(self.w) * x
    def get_config(self):
        return super().get_config()

def scale(inputs):
    x, = inputs
    res = x / tf.math.exp(_get_log_scale())
    return res

def inv_scale(inputs):
    x, = inputs
    return tf.math.exp(_get_log_scale()) * x

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
    decoded_amp = create_decoder_amp(encoded, n_filters_scale, gridsize, big)
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

def create_decoder_amp(input_tensor, n_filters_scale, gridsize, big):
    """
    Decoder head that predicts object-space amplitudes.

    When object.big=True we must emit one channel per patch (C=gridsize**2)
    so the subsequent ReassemblePatchesLayer receives matching patch counts.
    """
    num_filters = gridsize**2 if big else 1

    # Placeholder convolution layers and activation as defined in the original DecoderAmp class
    conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
    conv2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')
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

# Create the distribution function for use in loss calculation
def create_poisson_distribution_for_loss(amplitude):
    squared = tf.square(amplitude)
    return tfd.Independent(tfd.Poisson(squared), reinterpreted_batch_ndims=3)

# We'll use this in the loss function
dist_poisson_intensity = create_poisson_distribution_for_loss

# Poisson distribution over expected diffraction intensity (i.e. photons per
# pixel)
def negloglik(y_true, y_pred):
    """Compute Poisson negative log-likelihood.

    Contract
    --------
    - y_true: target intensity (counts), strictly positive
    - y_pred: predicted intensity (counts), strictly positive
    - Implementations MUST ensure y_pred > 0 prior to the log operation
      (e.g., strictly-positive activation on amplitude head or epsilon ≥ 1e-12).
    """
    # y_true is the target intensity (already squared)
    # y_pred is the predicted intensity (already squared)
    # Use TensorFlow's Poisson loss which computes: y_pred - y_true * log(y_pred)
    return tf.nn.log_poisson_loss(y_true, tf.math.log(y_pred), compute_full_loss=False)


def _build_module_level_models():
    """Build module-level models on first access (lazy loading).

    This function encapsulates all the model construction code that previously
    ran at import time. It's called by __getattr__ on first access to
    autoencoder, diffraction_to_obj, or autoencoder_no_nll.

    Warning: This function modifies module-level state. It should only be
    called once per process via the _model_construction_done guard.

    See REFACTOR-MODEL-SINGLETON-001 Phase B, docs/findings.md MODULE-SINGLETON-001.
    """
    global _model_construction_done
    if _model_construction_done:
        return

    # Get current params (whatever is set when models are first accessed)
    N = p.get('N')
    gridsize = p.get('gridsize')
    n_filters_scale = p.get('n_filters_scale')

    # Create input layers
    input_img = Input(shape=(N, N, gridsize**2), name='input')
    input_positions = Input(shape=(1, 2, gridsize**2), name='input_positions')

    # === Begin model construction (moved from module level) ===
    normed_input = IntensityScaler(name='intensity_scaler')([input_img])

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
            padded_size=padded_size,
            N=N,
            gridsize=gridsize,
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

    # Trim the object reconstruction to N x N
    trimmed_obj = TrimReconstructionLayer(output_size=N, dtype=tf.complex64, name='trimmed_obj')(padded_obj_2)

    # Extract overlapping regions of the object
    # Output shape should be (batch, N, N, 1) where N is the patch size
    padded_objs_with_offsets = ExtractPatchesPositionLayer(
        jitter=0.0,
        dtype=tf.complex64,
        name='padded_objs_with_offsets'
    )([padded_obj_2, input_positions])

    # Apply the probe illumination (lazy-loaded)
    padded_objs_with_offsets, probe_tensor = _get_probe_illumination()([padded_objs_with_offsets])

    # Apply pad and diffract operation
    pad_diffract_layer = PadAndDiffractLayer(h=N, w=N, pad=False, name='pred_amplitude')
    padded_objs_with_offsets, pred_diff = pad_diffract_layer(padded_objs_with_offsets)

    # Reshape
    pred_diff = FlatToChannelLayer(N=N, gridsize=gridsize, name='pred_diff_channels')(pred_diff)

    # Scale the amplitude
    pred_amp_scaled = IntensityScaler_inv(name='intensity_scaler_inv')([pred_diff])

    # Use the squared amplitude as intensity
    pred_intensity_sampled = SquareLayer(name='pred_intensity')(pred_amp_scaled)

    # Create models
    autoencoder = Model([input_img, input_positions], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])
    autoencoder_no_nll = Model(inputs=[input_img, input_positions], outputs=[pred_amp_scaled])
    diffraction_to_obj = tf.keras.Model(inputs=[input_img, input_positions], outputs=[trimmed_obj])

    # Compile autoencoder (moved from module level)
    mae_weight = p.get('mae_weight')
    nll_weight = p.get('nll_weight')
    realspace_weight = p.get('realspace_weight')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    use_xla_compile = os.environ.get('USE_XLA_COMPILE', '').lower() in ('1', 'true', 'yes')
    try:
        use_xla_compile = use_xla_compile or p.get('use_xla_compile')
    except KeyError:
        pass

    autoencoder.compile(
        optimizer=optimizer,
        loss=[hh.realspace_loss, 'mean_absolute_error', negloglik],
        loss_weights=[realspace_weight, mae_weight, nll_weight],
        jit_compile=use_xla_compile
    )

    # Store in cache
    _lazy_cache['autoencoder'] = autoencoder
    _lazy_cache['diffraction_to_obj'] = diffraction_to_obj
    _lazy_cache['autoencoder_no_nll'] = autoencoder_no_nll

    _model_construction_done = True

def prepare_inputs(train_data: PtychoDataContainer):
    """training inputs"""
    return [train_data.X * p.get('intensity_scale'), train_data.coords]

def prepare_outputs(train_data: PtychoDataContainer):
    """training outputs"""
    return [hh.center_channels(train_data.Y_I, train_data.coords)[:, :, :, :1],
                (p.get('intensity_scale') * train_data.X),
                (p.get('intensity_scale') * train_data.X)**2]

#def train(epochs, X_train, coords_train, Y_obj_train):
def train(epochs, trainset: PtychoDataContainer, model_instance=None, use_streaming=None):
    """Train the ptychography model.

    Args:
        epochs: Number of training epochs
        trainset: Training data container
        model_instance: Optional compiled model. If None, uses module-level
                       singleton (for backward compatibility).
        use_streaming: If True, use as_tf_dataset() for memory-efficient streaming.
                      If None (default), auto-detect based on dataset size.
                      Large datasets (>10000 samples) automatically use streaming.
                      See docs/findings.md PINN-CHUNKED-001.

    Returns:
        Training history object
    """
    assert type(trainset) == PtychoDataContainer

    # Use provided model or fall back to module-level singleton for backward compatibility
    if model_instance is None:
        model_instance = autoencoder  # Backward compatible fallback

    batch_size = p.params()['batch_size']

    # Auto-detect streaming mode based on dataset size
    if use_streaming is None:
        use_streaming = len(trainset) > 10000

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
                            '%s/weights.{epoch:02d}.h5' % wt_path,
                            monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='auto', save_freq='epoch')

    if use_streaming:
        # Memory-efficient streaming for large datasets
        # Uses as_tf_dataset() to stream data in batches without loading everything into GPU memory
        print(f"Using streaming mode for {len(trainset)} samples")
        dataset = trainset.as_tf_dataset(batch_size, shuffle=True)
        # Note: validation_split not compatible with tf.data.Dataset
        # For streaming, skip validation or use separate validation dataset
        history = model_instance.fit(
            dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[reduce_lr, earlystop]
        )
    else:
        # Standard mode for smaller datasets (current behavior)
        history = model_instance.fit(
            prepare_inputs(trainset),
            prepare_outputs(trainset),
            shuffle=True, batch_size=batch_size, verbose=1,
            epochs=epochs, validation_split=0.05,
            callbacks=[reduce_lr, earlystop])

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
    original_use_xla = p.cfg.get('use_xla_translate', True)

    try:
        # Clear Keras session to avoid XLA cache conflicts from module-level model
        tf.keras.backend.clear_session()

        # Temporarily update global state for model construction
        # Disable XLA for Translation to avoid shape caching issues when N differs from import-time N
        p.cfg.update({'gridsize': gridsize, 'N': N, 'use_xla_translate': False})
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
                padded_size=padded_size,
                N=N,
                gridsize=gridsize,
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
            N=N,
            gridsize=gridsize,
            dtype=tf.complex64,
            name='padded_objs_with_offsets'
        )([padded_obj_2, input_positions])

        # Create fresh ProbeIllumination with correctly-sized probe for this N
        global_probe = p.params().get('probe')
        if global_probe is not None and global_probe.shape[0] == N:
            local_probe = global_probe
        else:
            # Generate default probe of correct size
            local_probe = probe.get_default_probe(N)
        local_probe_illumination = ProbeIllumination(
            initial_probe=local_probe,
            N=N,
            name='probe_illumination'
        )
        padded_objs_with_offsets, probe_out = local_probe_illumination([padded_objs_with_offsets])
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
        p.cfg.update({'gridsize': original_gridsize, 'N': original_N, 'use_xla_translate': original_use_xla})

def _create_models_from_global_config():
    """Create models using global configuration (for backward compatibility)."""
    gridsize = p.get('gridsize')
    N = p.get('N')
    return create_model_with_gridsize(gridsize, N)


def create_compiled_model(gridsize=None, N=None):
    """Create and compile autoencoder ready for training.

    Use this instead of the module-level singleton when gridsize
    may have changed since module import. See MODULE-SINGLETON-001
    in docs/findings.md.

    Args:
        gridsize: Grid size for model architecture. If None, reads from params.cfg.
        N: Patch size. If None, reads from params.cfg.

    Returns:
        tuple: (compiled_autoencoder, diffraction_to_obj)

    Example:
        >>> from ptycho import model, params
        >>> params.cfg['gridsize'] = 2
        >>> autoencoder, d2o = model.create_compiled_model()
        >>> assert autoencoder.input_shape[0][-1] == 4  # gridsize² channels
    """
    gridsize = gridsize if gridsize is not None else p.get('gridsize')
    N = N if N is not None else p.get('N')

    autoencoder, diffraction_to_obj = create_model_with_gridsize(gridsize, N)

    # Compile with current loss weights (mirrors module-level compilation)
    mae_weight = p.get('mae_weight')
    nll_weight = p.get('nll_weight')
    realspace_weight = p.get('realspace_weight')

    use_xla = os.environ.get('USE_XLA_COMPILE', '').lower() in ('1', 'true', 'yes')
    try:
        use_xla = use_xla or p.get('use_xla_compile')
    except KeyError:
        pass

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    autoencoder.compile(
        optimizer=optimizer,
        loss=[hh.realspace_loss, 'mean_absolute_error', negloglik],
        loss_weights=[realspace_weight, mae_weight, nll_weight],
        jit_compile=use_xla
    )

    return autoencoder, diffraction_to_obj


def __getattr__(name):
    """Lazy load module-level singletons on first access.

    Implements REFACTOR-MODEL-SINGLETON-001 fix: model construction is deferred
    until first access, allowing params.cfg to be configured before models are built.

    Emits DeprecationWarning for legacy singleton access.
    See docs/findings.md MODULE-SINGLETON-001 for migration guide.
    """
    import warnings

    if name in ('autoencoder', 'diffraction_to_obj', 'autoencoder_no_nll'):
        if name not in _lazy_cache:
            warnings.warn(
                f"Accessing deprecated module-level singleton '{name}'. "
                "Use create_compiled_model() or create_model_with_gridsize() instead. "
                "See docs/findings.md MODULE-SINGLETON-001 for migration guide.",
                DeprecationWarning,
                stacklevel=2
            )
            _build_module_level_models()
        return _lazy_cache[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

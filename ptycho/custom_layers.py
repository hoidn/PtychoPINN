"""Custom Keras layers to replace Lambda layers for proper serialization.

This module contains custom Keras layer implementations that replace Lambda layers
in the PtychoPINN model. These custom layers provide proper serialization support
for Keras 3 and enable model saving/loading without issues.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

# Register all custom layers for serialization
@tf.keras.utils.register_keras_serializable(package='ptycho')
class CombineComplexLayer(layers.Layer):
    """Combines real and imaginary parts into complex tensor."""
    
    def __init__(self, **kwargs):
        # Don't set dtype in kwargs - let it be inferred
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
        # Force output dtype to be complex
        self._compute_dtype_object = tf.complex64
    
    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """Combine real and imaginary parts.
        
        Args:
            inputs: List of [real_part, imag_part] tensors
            
        Returns:
            Complex tensor
        """
        real_part, imag_part = inputs
        # Ensure inputs are float32 for combining
        if real_part.dtype in [tf.complex64, tf.complex128]:
            real_part = tf.math.real(real_part)
        if imag_part.dtype in [tf.complex64, tf.complex128]:
            imag_part = tf.math.real(imag_part)
        
        # Cast to float32 if needed
        real_part = tf.cast(real_part, tf.float32)
        imag_part = tf.cast(imag_part, tf.float32)
        
        return tf.complex(real_part, imag_part)
    
    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        # Output shape is same as input shapes
        return input_shape[0]
    
    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package='ptycho')
class ExtractPatchesPositionLayer(layers.Layer):
    """Extract patches from object based on positions.

    Inputs
    ------
    - padded_obj: tf.complex64 or tf.float32, shape (B, M, M, 1)
    - positions: tf.float32, shape (B, 1, 2, C)  # channel-format offsets

    Output
    ------
    - patches: same dtype as input, shape (B, N, N, C)
    """
    
    def __init__(self, jitter: float = 0.0, N: Optional[int] = None,
                 gridsize: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.jitter = jitter
        self._N = N
        self._gridsize = gridsize

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """Extract patches at specified positions.
        Args:
            inputs: [padded_obj (B,M,M,1), positions (B,1,2,C)]
        Returns:
            patches (B,N,N,C)
        """
        from . import tf_helper as hh
        padded_obj, positions = inputs
        return hh.extract_patches_position(padded_obj, positions, self.jitter,
                                           N=self._N, gridsize=self._gridsize)

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        batch_size = input_shape[0][0]
        # Use stored N if available, otherwise estimate from input shape
        if self._N is not None:
            N = self._N
        else:
            N = input_shape[0][1] - 10  # Fallback: assuming padding of 5 on each side
        channels = input_shape[0][-1]
        return tf.TensorShape([batch_size, N, N, channels])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['jitter'] = self.jitter
        config['N'] = self._N
        config['gridsize'] = self._gridsize
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class PadReconstructionLayer(layers.Layer):
    """Pad reconstruction to larger size."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Pad the reconstruction.
        
        Args:
            inputs: Reconstruction tensor
            
        Returns:
            Padded reconstruction
        """
        from . import tf_helper as hh
        return hh.pad_reconstruction(inputs)
    
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        from . import params
        padded_size = params.get_padded_size()
        batch_size = input_shape[0]
        channels = input_shape[-1]
        return tf.TensorShape([batch_size, padded_size, padded_size, channels])
    
    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package='ptycho')
class ReassemblePatchesLayer(layers.Layer):
    """Reassemble patches into full object.

    Inputs: [patches (B, N, N, C), positions (B, 1, 2, C)]
    Output: (B, padded_size, padded_size, 1)
    """

    def __init__(self, padded_size: Optional[int] = None, N: Optional[int] = None,
                 gridsize: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.padded_size = padded_size
        self.N = N
        self.gridsize = gridsize

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """Reassemble patches.

        Args:
            inputs: List of [patches, positions] tensors

        Returns:
            Reassembled object
        """
        from . import tf_helper as hh
        patches, positions = inputs

        # Always use batched reassembly with a conservative batch size
        # This handles Translation layer shape variations robustly
        # The batched function automatically falls back to non-batched mode for small inputs
        batch_size = 64
        fn_reassemble = hh.mk_reassemble_position_batched_real(
            positions, batch_size=batch_size, padded_size=self.padded_size, N=self.N,
            gridsize=self.gridsize
        )

        return hh.reassemble_patches(patches, fn_reassemble_real=fn_reassemble,
                                     N=self.N, gridsize=self.gridsize)

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        from . import params
        padded_size = self.padded_size or params.get_padded_size()
        batch_size = input_shape[0][0]
        channels = input_shape[0][-1]
        return tf.TensorShape([batch_size, padded_size, padded_size, channels])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['padded_size'] = self.padded_size
        config['N'] = self.N
        config['gridsize'] = self.gridsize
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class TrimReconstructionLayer(layers.Layer):
    """Trim reconstruction to original size."""
    
    def __init__(self, output_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Trim the reconstruction.

        Args:
            inputs: Padded reconstruction tensor

        Returns:
            Trimmed reconstruction
        """
        from . import tf_helper as hh
        return hh.trim_reconstruction(inputs, N=self.output_size)
    
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        from . import params
        N = self.output_size or params.get('N')
        batch_size = input_shape[0]
        channels = input_shape[-1]
        return tf.TensorShape([batch_size, N, N, channels])
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['output_size'] = self.output_size
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class PadAndDiffractLayer(layers.Layer):
    """Apply padding and diffraction operation.

    Input: images (B, N, N, C)
    Output: (padded, amplitude) both with shape (B, h, w, C)
    """
    
    def __init__(self, h: int, w: int, pad: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.w = w
        self.pad = pad
    
    def call(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply pad and diffract operation.
        
        Args:
            inputs: Input tensor (or list containing single tensor)
            
        Returns:
            Tuple of (padded_output, diffraction_pattern)
        """
        from . import tf_helper as hh
        # Handle both tensor and list inputs
        if isinstance(inputs, list):
            inputs = inputs[0]
        return hh.pad_and_diffract(inputs, self.h, self.w, pad=self.pad)
    
    def compute_output_shape(self, input_shape) -> List[tf.TensorShape]:
        # Returns two outputs with same shape
        # Handle both single shape and list of shapes
        if isinstance(input_shape, list):
            shape = input_shape[0]
        else:
            shape = input_shape
        batch_size = shape[0]
        channels = shape[-1]
        return [tf.TensorShape([batch_size, self.h, self.w, channels]), 
                tf.TensorShape([batch_size, self.h, self.w, channels])]
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'h': self.h,
            'w': self.w,
            'pad': self.pad
        })
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class FlatToChannelLayer(layers.Layer):
    """Reshape flat tensor to channel format."""
    
    def __init__(self, N: int, gridsize: int, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.gridsize = gridsize
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reshape tensor.
        
        Args:
            inputs: Flat tensor
            
        Returns:
            Channel format tensor
        """
        from . import tf_helper as hh
        return hh._flat_to_channel(inputs, N=self.N, gridsize=self.gridsize)
    
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        batch_size = input_shape[0]
        return tf.TensorShape([batch_size, self.N, self.N, self.gridsize**2])
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'N': self.N,
            'gridsize': self.gridsize
        })
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class ScaleLayer(layers.Layer):
    """Scale tensor by learned log scale factor."""
    
    def __init__(self, log_scale_init: float = 0.0, trainable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.log_scale_init = log_scale_init
        self.trainable_scale = trainable
        
    def build(self, input_shape):
        self.log_scale = self.add_weight(
            name='log_scale',
            shape=(),
            initializer=tf.constant_initializer(self.log_scale_init),
            trainable=self.trainable_scale
        )
        super().build(input_shape)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Scale input by exp(log_scale).
        
        Args:
            inputs: Input tensor
            
        Returns:
            Scaled tensor
        """
        return inputs / tf.exp(self.log_scale)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'log_scale_init': self.log_scale_init,
            'trainable': self.trainable_scale
        })
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class InvScaleLayer(layers.Layer):
    """Inverse scale tensor by learned log scale factor."""
    
    def __init__(self, scale_layer: Optional[ScaleLayer] = None, **kwargs):
        super().__init__(**kwargs)
        self.scale_layer = scale_layer
        
    def build(self, input_shape):
        if self.scale_layer is None:
            # Create own weight if no scale layer provided
            self.log_scale = self.add_weight(
                name='log_scale',
                shape=(),
                initializer='zeros',
                trainable=True
            )
        super().build(input_shape)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Inverse scale input by exp(log_scale).
        
        Args:
            inputs: Input tensor
            
        Returns:
            Inverse scaled tensor
        """
        if self.scale_layer is not None:
            log_scale = self.scale_layer.log_scale
        else:
            log_scale = self.log_scale
        return inputs * tf.exp(log_scale)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # Note: scale_layer reference is not serialized
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class ActivationLayer(layers.Layer):
    """Custom activation layer with configurable function."""
    
    def __init__(self, activation_name: str = 'sigmoid', scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation_name
        self.scale = scale
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply activation function.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Activated tensor
        """
        if self.activation_name == 'sigmoid':
            return tf.nn.sigmoid(inputs)
        elif self.activation_name == 'tanh':
            return self.scale * tf.nn.tanh(inputs)
        elif self.activation_name == 'swish':
            return tf.nn.swish(inputs)
        elif self.activation_name == 'softplus':
            return tf.nn.softplus(inputs)
        elif self.activation_name == 'relu':
            return tf.nn.relu(inputs)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'activation_name': self.activation_name,
            'scale': self.scale
        })
        return config


@tf.keras.utils.register_keras_serializable(package='ptycho')
class SquareLayer(layers.Layer):
    """Square the input tensor."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Square the input.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Squared tensor
        """
        return tf.square(inputs)
    
    def get_config(self) -> Dict[str, Any]:
        return super().get_config()

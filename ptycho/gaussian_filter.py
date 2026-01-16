"""
Native TensorFlow implementation of Gaussian filter to replace tensorflow_addons dependency.

This module provides a drop-in replacement for tfa.image.gaussian_filter2d that uses
only core TensorFlow operations, eliminating the dependency on tensorflow_addons.
"""

import tensorflow as tf
from typing import Union, Tuple, List


def _get_gaussian_kernel_1d(size: int, sigma: float) -> tf.Tensor:
    """Create a 1D Gaussian kernel using TFA's softmax approach."""
    x = tf.range(-size // 2 + 1, size // 2 + 1)
    x = tf.cast(x**2, tf.float32)
    x = tf.nn.softmax(-x / (2.0 * (sigma**2)))
    return x


def _get_gaussian_kernel_2d(filter_shape: Tuple[int, int], sigma: Tuple[float, float]) -> tf.Tensor:
    """Create a 2D Gaussian kernel from two 1D kernels."""
    # TFA uses sigma[1] for x-direction and sigma[0] for y-direction
    kernel_x = _get_gaussian_kernel_1d(filter_shape[1], sigma[1])
    kernel_x = kernel_x[tf.newaxis, :]
    
    kernel_y = _get_gaussian_kernel_1d(filter_shape[0], sigma[0])
    kernel_y = kernel_y[:, tf.newaxis]
    
    # Create 2D kernel using matrix multiplication
    kernel_2d = tf.matmul(kernel_y, kernel_x)
    return kernel_2d


def _pad(image: tf.Tensor, filter_shape: Tuple[int, int], 
         mode: str = "REFLECT", constant_values: float = 0) -> tf.Tensor:
    """Explicitly pad a 4-D image to match TFA's padding behavior."""
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


def gaussian_filter2d(image: tf.Tensor,
                     filter_shape: Union[int, Tuple[int, int]] = 3,
                     sigma: Union[float, Tuple[float, float]] = 1.0,
                     padding: str = "REFLECT",
                     constant_values: float = 0,
                     name: str = None) -> tf.Tensor:
    """
    Perform Gaussian blur on image(s).
    
    This is a native TensorFlow implementation that matches the behavior of
    tensorflow_addons.image.gaussian_filter2d exactly.
    
    Args:
        image: Either a 2-D `Tensor` of shape `[height, width]`,
            a 3-D `Tensor` of shape `[height, width, channels]`,
            or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
        filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
            the height and width of the 2-D Gaussian filter. Can be a single
            integer to specify the same value for all spatial dimensions.
        sigma: A `float` or `tuple`/`list` of 2 floats, specifying
            the standard deviation in x and y direction of the 2-D Gaussian filter.
            Can be a single float to specify the same value for all spatial
            dimensions.
        padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
            The type of padding algorithm to use.
        constant_values: A `scalar`, the pad value to use in "CONSTANT"
            padding mode.
        name: A name for this operation (optional).
            
    Returns:
        Filtered image tensor of the same shape and dtype as input.
    """
    with tf.name_scope(name or "gaussian_filter2d"):
        # Store original shape and dtype
        original_shape = image.shape
        original_dtype = image.dtype
        original_ndims = len(image.shape)
        
        # Convert to 4D for processing
        if original_ndims == 2:
            image = tf.expand_dims(image, axis=0)
            image = tf.expand_dims(image, axis=-1)
        elif original_ndims == 3:
            image = tf.expand_dims(image, axis=0)
        elif original_ndims != 4:
            raise ValueError(f"Image must be 2D, 3D, or 4D. Got {original_ndims}D.")
        
        # Convert to float for computation if needed
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)
        
        # Normalize parameters
        if isinstance(filter_shape, int):
            filter_shape = (filter_shape, filter_shape)
        
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))
        elif isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError("sigma should be a float or a tuple/list of 2 floats")
            sigma = tuple(float(s) for s in sigma)
        
        if any(s < 0 for s in sigma):
            raise ValueError("sigma should be greater than or equal to 0.")
        
        # Get number of channels
        channels = tf.shape(image)[3]
        
        # Create Gaussian kernel
        kernel = _get_gaussian_kernel_2d(filter_shape, sigma)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        kernel = tf.cast(kernel, image.dtype)
        
        # Pad the image
        image_padded = _pad(image, filter_shape, mode=padding.upper(), 
                           constant_values=constant_values)
        
        # Apply depthwise convolution
        filtered = tf.nn.depthwise_conv2d(
            image_padded,
            kernel,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Restore original shape
        if original_ndims == 2:
            filtered = tf.squeeze(filtered, axis=[0, 3])
        elif original_ndims == 3:
            filtered = tf.squeeze(filtered, axis=0)
        
        # Restore original dtype
        return tf.cast(filtered, original_dtype)


def complex_gaussian_filter2d(input_tensor: tf.Tensor,
                             filter_shape: Union[int, Tuple[int, int]],
                             sigma: Union[float, Tuple[float, float]]) -> tf.Tensor:
    """
    Apply Gaussian filter to complex-valued tensor.
    
    This function separates the real and imaginary parts, applies Gaussian
    filtering to each, and then recombines them.
    
    Args:
        input_tensor: Complex-valued tensor
        filter_shape: Tuple of integers specifying the filter shape
        sigma: Float or tuple of floats for the Gaussian kernel standard deviation
    
    Returns:
        Complex-valued tensor after applying Gaussian filter
    """
    real_part = tf.math.real(input_tensor)
    imag_part = tf.math.imag(input_tensor)
    
    filtered_real = gaussian_filter2d(real_part, filter_shape=filter_shape, sigma=sigma)
    filtered_imag = gaussian_filter2d(imag_part, filter_shape=filter_shape, sigma=sigma)
    
    return tf.complex(filtered_real, filtered_imag)
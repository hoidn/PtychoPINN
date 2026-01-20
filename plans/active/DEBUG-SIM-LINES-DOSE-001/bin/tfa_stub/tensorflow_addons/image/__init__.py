"""TensorFlow Addons image helpers used by the legacy dose_experiments scripts."""

from __future__ import annotations

from typing import Sequence, Tuple

import tensorflow as tf

__all__ = ["translate", "gaussian_filter2d"]


def _make_separable_kernel(size: int, sigma: tf.Tensor, dtype) -> tf.Tensor:
    radius = (size - 1) / 2.0
    x = tf.linspace(-radius, radius, size)
    kernel = tf.exp(-0.5 * tf.square(x / sigma))
    kernel /= tf.reduce_sum(kernel)
    return tf.cast(kernel, dtype)


def gaussian_filter2d(
    image: tf.Tensor,
    filter_shape: Tuple[int, int] = (3, 3),
    sigma: Sequence[float] | float = 1.0,
    name: str | None = None,
) -> tf.Tensor:
    """Port of tfa.image.gaussian_filter2d with the subset we need."""
    with tf.name_scope(name or "gaussian_filter2d"):
        image = tf.convert_to_tensor(image)
        if image.shape.rank != 4:
            raise ValueError("gaussian_filter2d expects images with shape [B, H, W, C]")

        if isinstance(sigma, (tuple, list)):
            sigma_y, sigma_x = sigma
        else:
            sigma_y = sigma_x = sigma
        dtype = image.dtype if image.dtype.is_floating else tf.float32
        sigma_y = tf.cast(sigma_y, dtype)
        sigma_x = tf.cast(sigma_x, dtype)

        kernel_y = _make_separable_kernel(filter_shape[0], sigma_y, dtype)
        kernel_x = _make_separable_kernel(filter_shape[1], sigma_x, dtype)

        kernel = tf.tensordot(kernel_y, kernel_x, axes=0)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, image.shape[-1], 1])

        return tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")


# https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
def _build_translation_matrix(translations: tf.Tensor, dtype) -> tf.Tensor:
    translations = tf.reshape(translations, [-1, 2])
    dx = tf.cast(translations[:, 0], dtype)
    dy = tf.cast(translations[:, 1], dtype)
    zero = tf.zeros_like(dx)
    one = tf.ones_like(dx)
    return tf.stack([one, zero, -dx, zero, one, -dy, zero, zero], axis=1)


def translate(
    images: tf.Tensor,
    translations: tf.Tensor,
    interpolation: str = "NEAREST",
    name: str | None = None,
) -> tf.Tensor:
    """Simplified translate implementation compatible with tfa.image.translate."""
    with tf.name_scope(name or "translate"):
        images = tf.convert_to_tensor(images)
        orig_dtype = images.dtype
        images = tf.cast(images, tf.float32)
        if images.shape.rank != 4:
            raise ValueError("translate expects images with shape [B, H, W, C]")

        translations = tf.cast(translations, images.dtype)
        transforms = _build_translation_matrix(translations, images.dtype)

        output_shape = tf.shape(images)[1:3]
        translated = tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            transforms=transforms,
            output_shape=output_shape,
            interpolation=interpolation.upper(),
            fill_mode="REFLECT",
            fill_value=0.0,
        )
        if orig_dtype is not None and orig_dtype.is_floating:
            translated = tf.cast(translated, orig_dtype)
        return translated

"""Fast implementation of get_image_patches that uses batch operations."""

import numpy as np
import tensorflow as tf
from ptycho import tf_helper as hh
from ptycho import params


def get_image_patches_fast(gt_image, global_offsets, local_offsets, N=None, gridsize=None):
    """
    Fast batch implementation of get_image_patches.
    
    Instead of looping B*c times, this uses TensorFlow's batch operations
    to process all patches at once.
    """
    # Use explicit parameters if provided, otherwise fall back to global params
    N = N if N is not None else params.get('N')
    gridsize = gridsize if gridsize is not None else params.get('gridsize')
    B = global_offsets.shape[0]
    c = gridsize**2
    
    # Pad the ground truth image once
    gt_padded = hh.pad(gt_image[None, ..., None], N // 2)
    
    # Calculate the combined offsets
    offsets_c = tf.cast((global_offsets + local_offsets), tf.float32)
    
    # Reshape offsets for batch processing
    # From (B, 2, 2, c) to (B*c, 2)
    offsets_reshaped = tf.reshape(tf.transpose(offsets_c, [0, 3, 1, 2]), [-1, 2, 2])
    offsets_flat = tf.reshape(offsets_reshaped[:, :, :, None], [-1, 2])
    
    # Tile the image B*c times for batch processing
    gt_tiled = tf.tile(gt_padded, [B * c, 1, 1, 1])
    
    # Batch translate all patches at once
    translated_patches = hh.translate(gt_tiled, -offsets_flat)
    
    # Extract the center N x N region from each patch
    patches_cropped = translated_patches[:, :N, :N, 0]
    
    # Reshape to (B, N, N, c) format
    canvas = tf.reshape(patches_cropped, [B, N, N, c])
    
    return canvas
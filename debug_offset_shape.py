#!/usr/bin/env python3
"""Debug script to understand offset tensor shapes."""

import numpy as np
import tensorflow as tf

# Create a mock offset tensor as it would appear in the function
B = 2  # batch size
c = 1  # gridsize^2
N = 8  # patch size

# Create offsets_f tensor with shape (B*c, 1, 1, 2)
# This matches what the docstring says
offsets_data = np.random.randn(B * c, 1, 1, 2).astype(np.float32)
offsets_f = tf.constant(offsets_data)

print(f"offsets_f shape: {offsets_f.shape}")
print(f"offsets_f[0] shape: {offsets_f[0].shape}")

# What the iterative version does:
# offset = -offsets_f[i, :, :, 0]
i = 0
offset_iterative = -offsets_f[i, :, :, 0]
print(f"\nIterative extraction for i={i}:")
print(f"offset shape: {offset_iterative.shape}")
print(f"offset value: {offset_iterative.numpy()}")

# The translate function expects shape (batch, 2) or for single image (1, 2)
# So offset with shape (1, 1) can't be right...

# Wait, let me check if the last dimension is selecting the 2D coordinates
# Maybe the docstring is wrong and it's actually (B*c, 1, 2, 1)?
offsets_alt = np.random.randn(B * c, 1, 2, 1).astype(np.float32)
offsets_f_alt = tf.constant(offsets_alt)

print(f"\nAlternative shape (B*c, 1, 2, 1):")
print(f"offsets_f_alt shape: {offsets_f_alt.shape}")
offset_alt = -offsets_f_alt[i, :, :, 0]
print(f"offset_alt shape: {offset_alt.shape}")
print(f"offset_alt value: {offset_alt.numpy()}")

# This gives shape (1, 2) which makes sense for translate!

# So for batched version with shape (B*c, 1, 2, 1):
negated_offsets_alt = -offsets_f_alt[:, :, :, 0]  # Shape: (B*c, 1, 2)
negated_offsets_alt = tf.squeeze(negated_offsets_alt, axis=1)  # Shape: (B*c, 2)
print(f"\nBatched extraction from (B*c, 1, 2, 1):")
print(f"negated_offsets shape: {negated_offsets_alt.shape}")
print(f"negated_offsets values:\n{negated_offsets_alt.numpy()}")
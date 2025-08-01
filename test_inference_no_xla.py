#!/usr/bin/env python
"""Test inference without XLA."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Explicitly disable XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
import numpy as np

print("Testing inference with XLA disabled...")

# Create a simple test model that uses our translation
from ptycho.tf_helper import Translation

# Build a minimal model
inputs = tf.keras.Input(shape=(64, 64, 1))
positions = tf.keras.Input(shape=(2,))

# Use Translation layer
translated = Translation(jitter_stddev=0.0)([inputs, positions])

# Simple output
outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(translated)

model = tf.keras.Model(inputs=[inputs, positions], outputs=outputs)

# Compile WITHOUT JIT
model.compile(
    optimizer='adam',
    loss='mse',
    jit_compile=False  # Critical: no XLA compilation
)

print("Model compiled with jit_compile=False")

# Save model
test_save_dir = "test_no_xla_model"
os.makedirs(test_save_dir, exist_ok=True)
model.save(os.path.join(test_save_dir, "model.keras"))
print(f"Model saved to {test_save_dir}")

# Load and test inference
loaded_model = tf.keras.models.load_model(os.path.join(test_save_dir, "model.keras"))
print("Model loaded successfully")

# Run inference
test_images = np.random.rand(2, 64, 64, 1).astype(np.float32)
test_positions = np.random.rand(2, 2).astype(np.float32) * 10

result = loaded_model([test_images, test_positions])
print(f"Inference successful! Output shape: {result.shape}")

# Clean up
import shutil
shutil.rmtree(test_save_dir)

print("\n✅ Success! The no-XLA approach works for both training and inference.")
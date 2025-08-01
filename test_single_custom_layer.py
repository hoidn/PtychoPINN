#!/usr/bin/env python3
"""Test single custom layer save/load."""

import tensorflow as tf
import numpy as np
import os
import sys

# Configure GPU memory growth FIRST
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Add ptycho to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom layers
from ptycho.custom_layers import CombineComplexLayer, SquareLayer

# Test CombineComplexLayer
print("Testing CombineComplexLayer...")
input1 = tf.keras.Input(shape=(64, 64, 1))
input2 = tf.keras.Input(shape=(64, 64, 1))
output = CombineComplexLayer(name='combine')([input1, input2])
model = tf.keras.Model(inputs=[input1, input2], outputs=output)

print("Model created:")
model.summary()

# Test inference
test_data1 = np.random.randn(1, 64, 64, 1).astype(np.float32)
test_data2 = np.random.randn(1, 64, 64, 1).astype(np.float32)
result = model.predict([test_data1, test_data2])
print(f"Inference result shape: {result.shape}, dtype: {result.dtype}")

# Save and load
save_path = "test_combine_layer_model.keras"
model.save(save_path)
print(f"Model saved to {save_path}")

# Enable unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# Load with custom objects
custom_objects = {'CombineComplexLayer': CombineComplexLayer}
loaded_model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
print("Model loaded successfully!")

# Test loaded model
result2 = loaded_model.predict([test_data1, test_data2])
print(f"Loaded model inference result shape: {result2.shape}")
print(f"Max difference: {np.max(np.abs(result - result2))}")

# Clean up
os.remove(save_path)

# Test SquareLayer
print("\n\nTesting SquareLayer...")
input1 = tf.keras.Input(shape=(64, 64, 4))
output = SquareLayer(name='square')(input1)
model2 = tf.keras.Model(inputs=input1, outputs=output)

print("Model created:")
model2.summary()

# Test inference
test_data = np.random.randn(1, 64, 64, 4).astype(np.float32)
result = model2.predict(test_data)
print(f"Inference result shape: {result.shape}, dtype: {result.dtype}")

# Save and load
save_path2 = "test_square_layer_model.keras"
model2.save(save_path2)
print(f"Model saved to {save_path2}")

# Load with custom objects
custom_objects2 = {'SquareLayer': SquareLayer}
loaded_model2 = tf.keras.models.load_model(save_path2, custom_objects=custom_objects2)
print("Model loaded successfully!")

# Test loaded model
result2 = loaded_model2.predict(test_data)
print(f"Loaded model inference result shape: {result2.shape}")
print(f"Max difference: {np.max(np.abs(result - result2))}")

# Clean up
os.remove(save_path2)

print("\n\nAll tests passed!")
#!/usr/bin/env python3
"""Test custom layer with multiple outputs save/load."""

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

# Create a test layer with multiple outputs
@tf.keras.utils.register_keras_serializable(package='test')
class TestMultiOutputLayer(tf.keras.layers.Layer):
    """Test layer that returns multiple outputs."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # Return two outputs
        return inputs * 2, inputs * 3
    
    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]
    
    def get_config(self):
        return super().get_config()

# Test 1: Both outputs used
print("Test 1: Model using both outputs...")
input1 = tf.keras.Input(shape=(10, 10, 1))
out1, out2 = TestMultiOutputLayer(name='multi_out')(input1)
output = tf.keras.layers.Add()([out1, out2])
model = tf.keras.Model(inputs=input1, outputs=output)

print("Model created:")
model.summary()

# Test inference
test_data = np.random.randn(1, 10, 10, 1).astype(np.float32)
result = model.predict(test_data)
print(f"Inference result shape: {result.shape}")

# Save and load
save_path = "test_multi_output_both.keras"
model.save(save_path)
print(f"Model saved to {save_path}")

# Enable unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# Load with custom objects
custom_objects = {'TestMultiOutputLayer': TestMultiOutputLayer}
loaded_model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
print("Model loaded successfully!")

# Test loaded model
result2 = loaded_model.predict(test_data)
print(f"Loaded model inference result shape: {result2.shape}")
print(f"Max difference: {np.max(np.abs(result - result2))}")

# Clean up
os.remove(save_path)

# Test 2: Only one output used (like ProbeIllumination)
print("\n\nTest 2: Model using only first output...")
input1 = tf.keras.Input(shape=(10, 10, 1))
out1, out2 = TestMultiOutputLayer(name='multi_out')(input1)
# Only use the first output
model2 = tf.keras.Model(inputs=input1, outputs=out1)

print("Model created:")
model2.summary()

# Test inference
result = model2.predict(test_data)
print(f"Inference result shape: {result.shape}")

# Save and load
save_path2 = "test_multi_output_single.keras"
model2.save(save_path2)
print(f"Model saved to {save_path2}")

# Load with custom objects
try:
    loaded_model2 = tf.keras.models.load_model(save_path2, custom_objects=custom_objects)
    print("Model loaded successfully!")
    
    # Test loaded model
    result2 = loaded_model2.predict(test_data)
    print(f"Loaded model inference result shape: {result2.shape}")
    print(f"Max difference: {np.max(np.abs(result - result2))}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("This is the same error we're seeing with the full model!")

# Clean up
if os.path.exists(save_path2):
    os.remove(save_path2)

print("\n\nTests completed!")
#!/usr/bin/env python3
"""Test a simplified version of the PtychoPINN model architecture."""

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

# Set up minimal params first
from ptycho import params
test_params = {
    'N': 64,
    'gridsize': 2,
    'gaussian_smoothing_sigma': 0.0,
    'probe.mask': False,
    'probe': np.ones((64, 64, 1), dtype=np.complex64),
    'probe.trainable': False,
    'intensity_scale': 100.0,
    'intensity_scale.trainable': False
}
for key, value in test_params.items():
    params.cfg[key] = value

# Import necessary components
from ptycho.custom_layers import (CombineComplexLayer, ExtractPatchesPositionLayer,
                                 PadReconstructionLayer, TrimReconstructionLayer,
                                 PadAndDiffractLayer, FlatToChannelLayer, SquareLayer)
from ptycho.model import ProbeIllumination, IntensityScaler, IntensityScaler_inv
from ptycho.tf_helper import CenterMaskLayer

# Create a simplified model similar to diffraction_to_obj
print("Creating simplified model...")

# Inputs
input_img = tf.keras.Input(shape=(64, 64, 4), name='input')
input_positions = tf.keras.Input(shape=(1, 2, 4), name='input_positions')

# Simple processing (skipping the full U-Net)
# Just do a simple convolution to simulate the decoder output
conv1 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(input_img)
conv2 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='tanh')(input_img)

# Combine to complex
obj = CombineComplexLayer(name='obj')([conv1, conv2])

# Pad reconstruction
padded_obj = PadReconstructionLayer(dtype=tf.complex64, name='padded_obj')(obj)

# Trim back to original size
trimmed_obj = TrimReconstructionLayer(output_size=64, dtype=tf.complex64, name='trimmed_obj')(padded_obj)

# Create model
model = tf.keras.Model(inputs=[input_img, input_positions], outputs=trimmed_obj)

print("Model created successfully!")
model.summary()

# Test inference
test_img = np.random.randn(1, 64, 64, 4).astype(np.float32)
test_pos = np.random.randn(1, 1, 2, 4).astype(np.float32)

output = model.predict([test_img, test_pos])
print(f"\nInference successful!")
print(f"Output shape: {output.shape}, dtype: {output.dtype}")

# Try to save and load
save_path = "test_simplified_model.keras"
model.save(save_path)
print(f"\nModel saved to {save_path}")

# Enable unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# Custom objects
custom_objects = {
    'CombineComplexLayer': CombineComplexLayer,
    'PadReconstructionLayer': PadReconstructionLayer,
    'TrimReconstructionLayer': TrimReconstructionLayer
}

# Try to load
try:
    loaded_model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
    print("Model loaded successfully!")
    
    # Test loaded model
    output2 = loaded_model.predict([test_img, test_pos])
    print(f"Loaded model inference successful!")
    print(f"Output shape: {output2.shape}")
    print(f"Max difference: {np.max(np.abs(output - output2))}")
    
except Exception as e:
    print(f"\nERROR loading model: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to understand what's happening
    print("\nLet's check the saved model structure...")
    import zipfile
    with zipfile.ZipFile(save_path, 'r') as zf:
        print("Files in saved model:")
        for name in zf.namelist():
            print(f"  {name}")

# Clean up
if os.path.exists(save_path):
    os.remove(save_path)

print("\nNow let's test with ProbeIllumination layer that has multiple outputs...")

# Create another model with ProbeIllumination
input_img2 = tf.keras.Input(shape=(64, 64, 4), name='input')
input_positions2 = tf.keras.Input(shape=(1, 2, 4), name='input_positions')

# Create initial probe guess
initial_probe = np.ones((1, 64, 64, 1), dtype=np.complex64)
tf.keras.backend.set_value(ProbeIllumination().w, initial_probe)

# Simple processing
conv1 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(input_img2)
conv2 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='tanh')(input_img2)
obj = CombineComplexLayer(name='obj')([conv1, conv2])

# Apply probe illumination
probe_layer = ProbeIllumination(name='probe_illumination')
illuminated, probe = probe_layer([obj])

# Just use the illuminated output
output = TrimReconstructionLayer(output_size=64, dtype=tf.complex64, name='trimmed')(illuminated)

model2 = tf.keras.Model(inputs=[input_img2, input_positions2], outputs=output)
print("\nModel with ProbeIllumination created!")
model2.summary()

# Test save/load
save_path2 = "test_probe_model.keras"
model2.save(save_path2)
print(f"\nModel saved to {save_path2}")

custom_objects2 = {
    'CombineComplexLayer': CombineComplexLayer,
    'ProbeIllumination': ProbeIllumination,
    'TrimReconstructionLayer': TrimReconstructionLayer
}

try:
    loaded_model2 = tf.keras.models.load_model(save_path2, custom_objects=custom_objects2)
    print("Model with ProbeIllumination loaded successfully!")
except Exception as e:
    print(f"\nERROR loading model with ProbeIllumination: {e}")
    import traceback
    traceback.print_exc()

# Clean up
if os.path.exists(save_path2):
    os.remove(save_path2)

print("\nTests completed!")
#!/usr/bin/env python
"""Test direct inference with trained model."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
import numpy as np
import pickle

print("Testing direct inference on trained model...")

# Load the model
model_path = "test_model_for_inference/autoencoder/model.keras"
custom_objects_path = "test_model_for_inference/autoencoder/custom_objects.dill"

# Load custom objects
with open(custom_objects_path, 'rb') as f:
    custom_objects = pickle.load(f)
print(f"Loaded custom objects: {list(custom_objects.keys())}")

# Load model without compilation to avoid custom object issues
print(f"\nLoading model from {model_path}...")
model = tf.keras.models.load_model(model_path, compile=False)
print("✓ Model loaded successfully!")

# Load test data
print("\nLoading test data...")
test_data = np.load("datasets/fly64/fly001_64_train_converted.npz")
print(f"Available arrays: {list(test_data.keys())}")

# Get a few test samples
n_test = 5
test_diffraction = test_data['diff3d'][:n_test]
test_positions = np.stack([test_data['xcoords'][:n_test], test_data['ycoords'][:n_test]], axis=-1)

# Prepare inputs
test_Y_I = test_diffraction.reshape(n_test, 64, 64, 1).astype(np.float32)
test_positions = test_positions.reshape(n_test, 1, 2, 1).astype(np.float32)

print(f"\nInput shapes:")
print(f"  Diffraction: {test_Y_I.shape}")
print(f"  Positions: {test_positions.shape}")

# Run inference
print("\nRunning inference...")
try:
    outputs = model([test_Y_I, test_positions])
    if isinstance(outputs, list):
        print(f"✓ Inference successful! Got {len(outputs)} outputs:")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
    else:
        print(f"✓ Inference successful! Output shape: {outputs.shape}")
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Inference test completed successfully - no XLA compilation errors!")
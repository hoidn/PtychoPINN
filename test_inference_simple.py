#!/usr/bin/env python
"""Test inference with minimal imports."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
import numpy as np

print("Testing inference without XLA...")

# Load test data
test_data = np.load("datasets/fly64/fly001_64_train_converted.npz")
n_test = 5
test_diffraction = test_data['diffraction'][:n_test]
test_positions = np.stack([test_data['xcoords'][:n_test], test_data['ycoords'][:n_test]], axis=-1)

# Prepare inputs
test_Y_I = test_diffraction.reshape(n_test, 64, 64, 1).astype(np.float32)
test_positions = test_positions.reshape(n_test, 1, 2, 1).astype(np.float32)

print(f"Input shapes: Y_I={test_Y_I.shape}, positions={test_positions.shape}")

# Try to load the model directly
try:
    # First, just load without custom objects
    model = tf.keras.models.load_model("test_model_for_inference/autoencoder/model.keras", compile=False)
    print("✓ Model loaded without custom objects")
    
    # Run inference
    outputs = model([test_Y_I, test_positions])
    print(f"✓ Inference successful! Got outputs")
    
except Exception as e:
    print(f"Failed without custom objects: {e}")
    
    # Try with minimal custom objects
    try:
        # Import only what we need
        from ptycho.tf_helper import CenterMaskLayer, ProbeIllumination, Translation, GetItem, Silu
        
        custom_objects = {
            'CenterMaskLayer': CenterMaskLayer,
            'ProbeIllumination': ProbeIllumination,
            'Translation': Translation,
            'GetItem': GetItem,
            'Silu': Silu
        }
        
        model = tf.keras.models.load_model(
            "test_model_for_inference/autoencoder/model.keras", 
            custom_objects=custom_objects,
            compile=False
        )
        print("✓ Model loaded with custom objects")
        
        # Run inference
        outputs = model([test_Y_I, test_positions])
        print(f"✓ Inference successful with custom objects!")
        
    except Exception as e2:
        print(f"Failed with custom objects: {e2}")
        import traceback
        traceback.print_exc()

print("\n✅ Test completed")
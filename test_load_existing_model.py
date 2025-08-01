#!/usr/bin/env python
"""Test loading existing model with pure TF translation."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
# Disable XLA to avoid ImageProjectiveTransformV3 compilation issues
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
import numpy as np
import pickle

def test_load_model():
    """Test loading existing model."""
    print("Testing loading existing model with pure TF translation...")
    
    model_path = "fly64_pinn_gridsize2_final/autoencoder/model.keras"
    custom_objects_path = "fly64_pinn_gridsize2_final/autoencoder/custom_objects.dill"
    
    try:
        # Load custom objects if available
        custom_objects = {}
        if os.path.exists(custom_objects_path):
            with open(custom_objects_path, 'rb') as f:
                custom_objects = pickle.load(f)
            print(f"Loaded custom objects: {list(custom_objects.keys())}")
        
        # Try to load the model
        print(f"\nLoading model from {model_path}...")
        
        # First, let's try with custom objects
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("✓ Model loaded successfully with custom objects!")
        except Exception as e:
            print(f"Failed with custom objects: {e}")
            # Try without custom objects
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✓ Model loaded successfully without compilation!")
        
        # Test inference
        print("\nTesting inference...")
        batch_size = 2
        dummy_Y_I = tf.constant(np.random.rand(batch_size, 64, 64, 1).astype(np.float32))
        dummy_positions = tf.constant(np.random.rand(batch_size, 2).astype(np.float32) * 100)
        
        # Run inference
        outputs = model([dummy_Y_I, dummy_positions])
        print(f"✓ Inference successful! Output shapes:")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: {out.shape}")
            
        print("\n✅ Model loading and inference successful with pure TF translation!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_load_model()
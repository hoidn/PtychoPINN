#!/usr/bin/env python
"""Test model building with pure TF translation."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
import numpy as np
from ptycho import params
from ptycho.model import create_model

def test_model_build():
    """Test if model can be built with pure TF translation."""
    print("Testing model build with pure TF translation...")
    
    # Set up parameters
    p = params.params()
    p['N'] = 64
    p['gridsize'] = 1
    p['offset'] = 32
    p['probe_scale'] = 4.0
    p['bigN'] = 232
    p['probe'] = np.ones((64, 64), dtype=np.complex64)
    p['normY'] = 1.0
    p['normP'] = 1.0
    p['do_stitching'] = False
    
    try:
        # Create model
        print("Creating model...")
        model = create_model(positions_provided=True)
        print("✓ Model created successfully!")
        
        # Create dummy inputs
        batch_size = 2
        dummy_Y_I = tf.constant(np.random.rand(batch_size, 64, 64, 1).astype(np.float32))
        dummy_positions = tf.constant(np.random.rand(batch_size, 2).astype(np.float32) * 100)
        
        # Test forward pass
        print("\nTesting forward pass...")
        outputs = model([dummy_Y_I, dummy_positions])
        print(f"✓ Forward pass successful! Output shapes:")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: {out.shape}")
        
        # Test model compilation
        print("\nTesting model compilation...")
        model.compile(
            optimizer='adam',
            loss=['mae', 'mae', 'mae'],
            jit_compile=False  # Important: keep False to avoid XLA issues
        )
        print("✓ Model compiled successfully!")
        
        # Test if model can be saved
        print("\nTesting model saving...")
        test_dir = "test_model_save"
        os.makedirs(test_dir, exist_ok=True)
        model.save(os.path.join(test_dir, "test_model.keras"))
        print(f"✓ Model saved successfully to {test_dir}/test_model.keras")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        print("\n✅ All tests passed! Model works with pure TF translation.")
        return True
        
    except Exception as e:
        print(f"\n❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_build()
#!/usr/bin/env python3
"""Simple inference script that bypasses ModelManager."""

import tensorflow as tf
import numpy as np
import os
import sys
import dill

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

# Load parameters BEFORE importing ptycho modules
from ptycho import params
params_path = "fly64_pinn_gridsize2_final/diffraction_to_obj/params.dill"
with open(params_path, 'rb') as f:
    params_dict = dill.load(f)
    # Update global params
    for key, value in params_dict.items():
        if key != '_version':
            params.cfg[key] = value

print(f"Loaded params - gridsize: {params.cfg.get('gridsize')}")

# Now load the saved model using SavedModel format instead of Keras
model_dir = "fly64_pinn_gridsize2_final/diffraction_to_obj"

# Try to load as SavedModel
try:
    print(f"Loading model from SavedModel format: {model_dir}")
    model = tf.saved_model.load(model_dir)
    print("Successfully loaded SavedModel!")
    
    # Get the inference function
    if hasattr(model, 'signatures'):
        print(f"Available signatures: {list(model.signatures.keys())}")
        if 'serving_default' in model.signatures:
            infer = model.signatures['serving_default']
        else:
            # Try to find the first signature
            infer = list(model.signatures.values())[0]
    else:
        # Use __call__ method
        infer = model
    
    # Test inference with dummy data
    batch_size = 1
    gridsize = params.cfg.get('gridsize', 1)
    N = params.cfg.get('N', 64)
    
    # Create inputs matching the expected shape
    input_img = np.random.randn(batch_size, N, N, gridsize**2).astype(np.float32)
    input_positions = np.random.randn(batch_size, 1, 2, gridsize**2).astype(np.float32)
    
    print(f"\nTesting inference with shapes:")
    print(f"  Input image: {input_img.shape}")
    print(f"  Input positions: {input_positions.shape}")
    
    # Convert to tensors
    input_img_tensor = tf.constant(input_img)
    input_positions_tensor = tf.constant(input_positions)
    
    # Run inference
    if hasattr(infer, '__call__'):
        # If it's a function
        output = infer(input=input_img_tensor, input_positions=input_positions_tensor)
    else:
        # If it's a model
        output = infer([input_img_tensor, input_positions_tensor])
    
    print(f"\nInference successful!")
    if isinstance(output, dict):
        for key, value in output.items():
            print(f"  {key}: {value.shape}")
    else:
        print(f"  Output shape: {output.shape}")
    
except Exception as e:
    print(f"Failed to load SavedModel: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nNow trying to load the test data and run inference...")
    
    # Load actual test data
    test_data_path = "datasets/fly64/fly001_64_train_converted.npz"
    if os.path.exists(test_data_path):
        data = np.load(test_data_path)
        print(f"\nAvailable arrays in test data: {list(data.keys())}")
        
        # Extract the first few samples
        n_samples = min(5, len(data['xcoords']))
        
        # Prepare inputs based on the data structure
        if 'Y_I' in data:
            # This is the diffraction patterns
            input_img = data['Y_I'][:n_samples] * params.cfg.get('intensity_scale', 1.0)
            print(f"Using Y_I (diffraction) data with shape: {input_img.shape}")
        
        if 'xcoords' in data and 'ycoords' in data:
            # Combine coordinates
            xcoords = data['xcoords'][:n_samples]
            ycoords = data['ycoords'][:n_samples]
            # Reshape to match expected format
            coords = np.stack([xcoords, ycoords], axis=1)[:, None, :, None]
            print(f"Using coordinates with shape: {coords.shape}")
#!/usr/bin/env python3
"""Test model creation with custom layers."""

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

# Set up minimal params for testing
from ptycho import params
test_params = {
    'N': 64,
    'gridsize': 2,
    'offset': 0,
    'n_filters_scale': 2,
    'object.big': False,
    'probe.big': False,
    'pad_object': True,
    'probe.mask': False,
    'gaussian_smoothing_sigma': 0.0,
    'nphotons': 1e8,
    'intensity_scale': 100.0,
    'intensity_scale.trainable': True,
    'probe.trainable': True,
    'amp_activation': 'sigmoid',
    'use_xla_translate': False,
    'use_xla_compile': False,
    'mae_weight': 0.0,
    'nll_weight': 1.0,
    'realspace_weight': 100.0,
    'model_type': 'pinn',
    'probe': np.ones((64, 64, 1), dtype=np.complex64),
    'label': ''
}

# Update params
for key, value in test_params.items():
    params.cfg[key] = value

# Now import model after params are set
from ptycho.model import create_model_with_gridsize

try:
    print("Creating model with custom layers...")
    autoencoder, diffraction_to_obj = create_model_with_gridsize(gridsize=2, N=64)
    
    print("Model created successfully!")
    print(f"Autoencoder outputs: {[out.name for out in autoencoder.outputs]}")
    print(f"Diffraction to obj outputs: {[out.name for out in diffraction_to_obj.outputs]}")
    
    # Test inference with dummy data
    batch_size = 1
    gridsize = 2
    N = 64
    
    # Create inputs
    input_img = np.random.randn(batch_size, N, N, gridsize**2).astype(np.float32)
    input_positions = np.random.randn(batch_size, 1, 2, gridsize**2).astype(np.float32)
    
    print(f"\nTesting inference with shapes:")
    print(f"  Input image: {input_img.shape}")
    print(f"  Input positions: {input_positions.shape}")
    
    # Check model layer dtypes
    print("\nChecking layer dtypes:")
    for layer in diffraction_to_obj.layers:
        if hasattr(layer, 'dtype'):
            print(f"  {layer.name}: {layer.dtype}")
    
    # Run inference
    try:
        output = diffraction_to_obj.predict([input_img, input_positions])
    except Exception as e:
        print(f"\nPrediction error: {e}")
        print("\nLet's check the model structure more carefully...")
        # Print detailed layer info
        for i, layer in enumerate(diffraction_to_obj.layers[:20]):
            print(f"Layer {i}: {layer.name} - type: {type(layer).__name__}")
            if hasattr(layer, 'output'):
                print(f"  Output shape: {layer.output.shape}, dtype: {layer.output.dtype}")
        raise
    
    print(f"\nInference successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    
    # Try to save and load the model
    print("\nTesting model save/load...")
    test_save_path = "test_custom_layers_model"
    os.makedirs(test_save_path, exist_ok=True)
    
    # Save in Keras format
    try:
        diffraction_to_obj.save(os.path.join(test_save_path, "model.keras"))
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Try to load it back
    tf.keras.config.enable_unsafe_deserialization()
    
    # Import custom objects for loading
    from ptycho.model_manager import ModelManager
    from ptycho.tf_helper import CenterMaskLayer
    from ptycho.model import ProbeIllumination, IntensityScaler, IntensityScaler_inv
    from ptycho.custom_layers import (CombineComplexLayer, ExtractPatchesPositionLayer,
                                     PadReconstructionLayer, ReassemblePatchesLayer,
                                     TrimReconstructionLayer, PadAndDiffractLayer,
                                     FlatToChannelLayer, ScaleLayer, InvScaleLayer,
                                     ActivationLayer, SquareLayer)
    
    custom_objects = {
        'CenterMaskLayer': CenterMaskLayer,
        'ProbeIllumination': ProbeIllumination,
        'IntensityScaler': IntensityScaler,
        'IntensityScaler_inv': IntensityScaler_inv,
        'CombineComplexLayer': CombineComplexLayer,
        'ExtractPatchesPositionLayer': ExtractPatchesPositionLayer,
        'PadReconstructionLayer': PadReconstructionLayer,
        'ReassemblePatchesLayer': ReassemblePatchesLayer,
        'TrimReconstructionLayer': TrimReconstructionLayer,
        'PadAndDiffractLayer': PadAndDiffractLayer,
        'FlatToChannelLayer': FlatToChannelLayer,
        'ScaleLayer': ScaleLayer,
        'InvScaleLayer': InvScaleLayer,
        'ActivationLayer': ActivationLayer,
        'SquareLayer': SquareLayer
    }
    
    loaded_model = tf.keras.models.load_model(
        os.path.join(test_save_path, "model.keras"),
        custom_objects=custom_objects
    )
    print("Model loaded successfully!")
    
    # Test inference with loaded model
    output2 = loaded_model.predict([input_img, input_positions])
    print(f"\nLoaded model inference successful!")
    print(f"  Output shape: {output2.shape}")
    print(f"  Max difference from original: {np.max(np.abs(output - output2))}")
    
    # Clean up
    import shutil
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python3
"""Test loading the model directly without ModelManager."""

import tensorflow as tf
import os
import numpy as np
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

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# NOW import all custom objects after params are set
from ptycho.tf_helper import CenterMaskLayer, trim_reconstruction, combine_complex
from ptycho.tf_helper import reassemble_patches, mk_reassemble_position_real
from ptycho.tf_helper import pad_reconstruction, extract_patches_position, pad_and_diffract, _flat_to_channel
from ptycho.model import ProbeIllumination, IntensityScaler, IntensityScaler_inv
from ptycho.model import get_amp_activation, scale, inv_scale, negloglik
from ptycho.tf_helper import realspace_loss as hh_realspace_loss
import math

model_dir = "fly64_pinn_gridsize2_final/diffraction_to_obj"
keras_model_path = os.path.join(model_dir, "model.keras")

print(f"Attempting to load model from: {keras_model_path}")

# Define all custom objects
custom_objects = {
    'CenterMaskLayer': CenterMaskLayer,
    'ProbeIllumination': ProbeIllumination,
    'IntensityScaler': IntensityScaler,
    'IntensityScaler_inv': IntensityScaler_inv,
    'trim_reconstruction': trim_reconstruction,
    'combine_complex': combine_complex,
    'reassemble_patches': reassemble_patches,
    'mk_reassemble_position_real': mk_reassemble_position_real,
    'pad_reconstruction': pad_reconstruction,
    'extract_patches_position': extract_patches_position,
    'pad_and_diffract': pad_and_diffract,
    '_flat_to_channel': _flat_to_channel,
    'get_amp_activation': get_amp_activation,
    'scale': scale,
    'inv_scale': inv_scale,
    'negloglik': negloglik,
    'realspace_loss': hh_realspace_loss,
    'tf': tf,
    'math': math,
}

# Try loading with custom objects
try:
    model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects)
    print("Successfully loaded model!")
    print(model.summary())
    
    # Test inference with dummy data
    batch_size = 1
    input_img = np.random.randn(batch_size, 64, 64, 1).astype(np.float32)
    input_positions = np.random.randn(batch_size, 1, 2, 1).astype(np.float32)
    
    print("\nTesting inference...")
    output = model.predict([input_img, input_positions])
    print(f"Output shape: {output.shape}")
    print("Inference successful!")
    
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
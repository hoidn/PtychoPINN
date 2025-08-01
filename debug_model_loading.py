#!/usr/bin/env python3
"""Debug model loading issue."""

import tensorflow as tf
import numpy as np
import os
import sys
import json

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
    
    # Let's examine the model structure
    print("\n=== Model Structure ===")
    print(f"Model inputs: {[inp.name for inp in diffraction_to_obj.inputs]}")
    print(f"Model outputs: {[out.name for out in diffraction_to_obj.outputs]}")
    
    # Save model
    save_path = "debug_model.keras"
    diffraction_to_obj.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Let's examine what was saved
    import zipfile
    with zipfile.ZipFile(save_path, 'r') as zf:
        print("\nFiles in saved model:")
        for name in zf.namelist():
            print(f"  {name}")
        
        # Read the model config
        with zf.open('config.json', 'r') as f:
            config = json.load(f)
            
    print("\n=== Saved Model Config ===")
    # Print just the top-level structure
    print(f"Config keys: {list(config.keys())}")
    print(f"Class name: {config.get('class_name', 'Unknown')}")
    
    # Check the functional config
    if 'config' in config:
        func_config = config['config']
        print(f"\nFunctional config keys: {list(func_config.keys())}")
        
        # Check inputs
        if 'input_layers' in func_config:
            print(f"\nInput layers: {len(func_config['input_layers'])}")
            for i, inp in enumerate(func_config['input_layers']):
                print(f"  Input {i}: {inp}")
        
        # Check outputs
        if 'output_layers' in func_config:
            print(f"\nOutput layers: {len(func_config['output_layers'])}")
            for i, out in enumerate(func_config['output_layers']):
                print(f"  Output {i}: {out}")
        
        # Check layers
        if 'layers' in func_config:
            print(f"\nTotal layers: {len(func_config['layers'])}")
            # Find ProbeIllumination and PadAndDiffractLayer
            for layer in func_config['layers']:
                if 'ProbeIllumination' in str(layer) or 'PadAndDiffract' in str(layer):
                    print(f"\nMulti-output layer found: {layer.get('class_name', 'Unknown')}")
                    print(f"  Config: {layer.get('config', {}).get('name', 'Unknown')}")
                    if 'inbound_nodes' in layer:
                        print(f"  Inbound nodes: {len(layer['inbound_nodes'])}")
    
    # Now try to load with detailed error handling
    print("\n=== Attempting to Load Model ===")
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
    
    try:
        loaded_model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's try to understand what's happening
        print("\n=== Debugging Load Failure ===")
        
        # Check if it's a specific layer causing issues
        print("\nTrying to identify problematic layers...")
        
        # Manually check the connectivity
        print("\nChecking layer connectivity in config...")
        if 'config' in config and 'layers' in config['config']:
            layers = config['config']['layers']
            layer_names = {layer['config']['name']: layer for layer in layers}
            
            # Find layers with no outbound connections
            print("\nLayers with no outbound connections:")
            for layer in layers:
                layer_name = layer['config']['name']
                has_outbound = False
                for other_layer in layers:
                    if 'inbound_nodes' in other_layer:
                        for node_group in other_layer['inbound_nodes']:
                            for node in node_group:
                                if isinstance(node, list) and len(node) > 0 and node[0] == layer_name:
                                    has_outbound = True
                                    break
                if not has_outbound and layer_name not in ['trimmed_obj', 'pred_intensity']:  # These are outputs
                    print(f"  {layer_name} ({layer['class_name']})")
    
    # Clean up
    os.remove(save_path)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug completed!")
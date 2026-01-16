"""
Model lifecycle management and persistence layer for PtychoPINN.

Critical bridge between training/inference workflows and model storage, providing
comprehensive serialization and loading with complete context preservation.
Handles TensorFlow models with metadata, custom objects, parameters, and training state.

Architecture Role:
    Centralized persistence layer enabling model reuse across workflow stages. Integrates
    with training pipelines and inference workflows, managing relationships between model
    architecture, custom TensorFlow layers, and physics parameters.

Core Functionality:
    - Model serialization with context preservation
    - Architecture-aware loading with correct gridsize/N parameters  
    - Multi-model zip archives for related model pairs
    - Parameter restoration and custom object management

Key Components:
    - ModelManager: Static class for all persistence operations
    - save_model/load_model: Individual model management
    - save_multiple_models/load_multiple_models: Multi-model archives

File Format:
    Dual-format combining TensorFlow SavedModel with dill-serialized metadata in zip
    archives, ensuring compatibility and Python object preservation.

Usage Example:
    # Training - save with full context
    models = {'autoencoder': model1, 'diffraction_to_obj': model2}
    ModelManager.save_multiple_models(models, 'output/wts.h5', custom_objects, scale)
    
    # Inference - load with parameter restoration
    loaded = ModelManager.load_multiple_models('output/wts.h5', ['diffraction_to_obj'])
"""

import os
import h5py
import dill
import tempfile
import zipfile
import shutil
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional
from ptycho import params

class ModelManager:
    @staticmethod
    def save_model(model: tf.keras.Model, model_dir: str, custom_objects: Dict[str, Any], intensity_scale: float) -> None:
        """
        Save a single model along with its custom objects, parameters, and intensity scale.

        Args:
            model (tf.keras.Model): The model to save.
            model_dir (str): Directory path for saving the model.
            custom_objects (Dict[str, Any]): Dictionary of custom objects used in the model.
            intensity_scale (float): The intensity scale used in the model.
        """
        model_file = os.path.join(model_dir, "model.h5")
        custom_objects_path = os.path.join(model_dir, "custom_objects.dill")
        params_path = os.path.join(model_dir, "params.dill")
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model (Keras 3 format)
            model.save(os.path.join(model_dir, "model.keras"))
            
            # Save custom objects
            with open(custom_objects_path, 'wb') as f:
                dill.dump(custom_objects, f)
            
            # Save parameters including intensity_scale
            params_dict = params.cfg.copy()
            params_dict['intensity_scale'] = intensity_scale
            params_dict['_version'] = '1.0'  # Add version information
            with open(params_path, 'wb') as f:
                dill.dump(params_dict, f)
            
            # Save intensity_scale as an attribute in the HDF5 file
            with h5py.File(model_file, 'a') as hf:
                hf.attrs['intensity_scale'] = intensity_scale
        
        except Exception as e:
            print(f"Error saving model to {model_dir}: {str(e)}")
            raise

    @staticmethod
    def load_model(model_dir: str) -> tf.keras.Model:
        """
        Load a single model along with its custom objects, parameters, and intensity scale.
        Uses architecture-aware loading to avoid gridsize mismatch issues.

        Args:
            model_dir (str): Directory containing the model files.

        Returns:
            tf.keras.Model: The loaded model.
        """
        custom_objects_path = os.path.join(model_dir, "custom_objects.dill")
        params_path = os.path.join(model_dir, "params.dill")
        
        try:
            # Load parameters
            with open(params_path, 'rb') as f:
                loaded_params = dill.load(f)
            
            # Check version and handle any necessary migrations
            version = loaded_params.pop('_version', '1.0')
            
            # Extract gridsize and N from loaded parameters
            gridsize = loaded_params.get('gridsize')
            N = loaded_params.get('N')
            
            if gridsize is None or N is None:
                raise ValueError(f"Required parameters missing: gridsize={gridsize}, N={N}")
            
            # Update params.cfg with loaded parameters
            params.cfg.update(loaded_params)
            
            # Load custom objects
            with open(custom_objects_path, 'rb') as f:
                custom_objects = dill.load(f)
            
            # Add any missing custom objects that might not be in older saved models
            from ptycho.tf_helper import CenterMaskLayer, trim_reconstruction, combine_complex, reassemble_patches, mk_reassemble_position_real
            from ptycho.tf_helper import pad_reconstruction, extract_patches_position, pad_and_diffract, _flat_to_channel
            from ptycho.model import get_amp_activation, scale, inv_scale
            from ptycho.custom_layers import (CombineComplexLayer, ExtractPatchesPositionLayer,
                                             PadReconstructionLayer, ReassemblePatchesLayer,
                                             TrimReconstructionLayer, PadAndDiffractLayer,
                                             FlatToChannelLayer, ScaleLayer, InvScaleLayer,
                                             ActivationLayer, SquareLayer)
            import math
            
            if 'CenterMaskLayer' not in custom_objects:
                custom_objects['CenterMaskLayer'] = CenterMaskLayer
            if 'trim_reconstruction' not in custom_objects:
                custom_objects['trim_reconstruction'] = trim_reconstruction
            
            # Add all Lambda functions that might be used
            custom_objects.update({
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
                'math': math,
                'tf': tf,
                # Add custom layers
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
            })
            
            # Import model factory after parameters are loaded
            from ptycho.model import create_model_with_gridsize
            
            # Enable unsafe deserialization for Lambda layers in Keras 3
            tf.keras.config.enable_unsafe_deserialization()
            
            # Create blank models with correct architecture
            autoencoder, diffraction_to_obj = create_model_with_gridsize(gridsize, N)
            
            # Create dictionary mapping model names to blank models
            models_dict = {
                'autoencoder': autoencoder,
                'diffraction_to_obj': diffraction_to_obj
            }
            
            # Determine current model name from model_dir path
            model_name = os.path.basename(model_dir)
            
            # Select the correct blank model
            if model_name in models_dict:
                model = models_dict[model_name]
            else:
                # Default to autoencoder for backward compatibility
                model = autoencoder
            
            # Load weights into the blank model
            # Check for new Keras 3 format first
            keras_model_path = os.path.join(model_dir, "model.keras")
            h5_model_path = os.path.join(model_dir, "model.h5")
            
            if os.path.exists(keras_model_path):
                # Load from Keras 3 format
                loaded_model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects)
                # Copy weights to the blank model
                model.set_weights(loaded_model.get_weights())
            elif os.path.exists(h5_model_path) and False:  # Temporarily disable H5 loading since our H5 files are metadata only
                # Check if H5 file is valid (not just metadata)
                try:
                    import h5py
                    with h5py.File(h5_model_path, 'r') as f:
                        # Check if the file contains actual weight data
                        if len(f.keys()) > 1 or (len(f.keys()) == 1 and 'intensity_scale' not in f.attrs):
                            # Load from H5 format - just load the weights
                            model.load_weights(h5_model_path)
                        else:
                            # H5 file only contains metadata, skip to SavedModel loading
                            print(f"H5 file {h5_model_path} appears to only contain metadata, skipping...")
                            raise ValueError("H5 file is metadata only")
                except Exception as e:
                    print(f"Failed to load from H5 file: {e}")
                    # Continue to next loading method
                    pass
            elif os.path.exists(os.path.join(model_dir, "saved_model.pb")):
                # Load from SavedModel format
                # The saved model contains the full model, so we need to load it and extract weights
                try:
                    # For Keras 3, we need to use a different approach
                    # Load the SavedModel and wrap it
                    print(f"Loading SavedModel from: {model_dir}")
                    # Enable unsafe deserialization for Lambda layers
                    tf.keras.config.enable_unsafe_deserialization()
                    
                    # Try to load as a Keras model first (might work for newer saves)
                    try:
                        loaded_model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
                        # If successful, just return it
                        return loaded_model
                    except Exception as keras_load_error:
                        print(f"Failed to load as Keras model: {keras_load_error}")
                        print("Falling back to raw SavedModel loading...")
                    
                    loaded_model = tf.saved_model.load(model_dir)
                    
                    # Create a wrapper that makes the SavedModel callable like a Keras model
                    class SavedModelWrapper(tf.keras.Model):
                        def __init__(self, saved_model, blank_model):
                            super().__init__()
                            self.saved_model = saved_model
                            self.blank_model = blank_model
                            # Try to find the inference function
                            if hasattr(saved_model, 'signatures'):
                                self.inference = saved_model.signatures.get('serving_default', None)
                                if self.inference is None and len(saved_model.signatures) > 0:
                                    self.inference = list(saved_model.signatures.values())[0]
                            else:
                                self.inference = saved_model
                        
                        def call(self, inputs, training=None, mask=None):
                            # The model expects two inputs: input and input_positions
                            if isinstance(inputs, (list, tuple)):
                                if len(inputs) >= 2:
                                    # Call with keyword arguments as expected by SavedModel
                                    output_dict = self.inference(
                                        input=inputs[0],
                                        input_positions=inputs[1]
                                    )
                                else:
                                    raise ValueError(f"Expected at least 2 inputs, got {len(inputs)}")
                            else:
                                # Single input - not expected for this model
                                raise ValueError("This model expects a list of inputs [input, input_positions]")
                            
                            # Extract outputs in the expected order
                            if isinstance(output_dict, dict):
                                if 'trimmed_obj' in output_dict:
                                    return output_dict['trimmed_obj']
                                else:
                                    # Return the first output
                                    return list(output_dict.values())[0]
                            else:
                                # If output is not a dict, return as is
                                return output_dict
                        
                        @property
                        def variables(self):
                            return self.saved_model.variables
                        
                        @property
                        def trainable_variables(self):
                            return self.saved_model.trainable_variables
                    
                    # Use the wrapper instead of trying to copy weights
                    return SavedModelWrapper(loaded_model, model)
                    
                    # Old checkpoint loading code - keeping as fallback
                    checkpoint_path = os.path.join(model_dir, "variables", "variables")
                    if False and os.path.exists(checkpoint_path + ".index"):
                        print(f"Loading weights from checkpoint: {checkpoint_path}")
                        model.load_weights(checkpoint_path)
                    else:
                        # Fallback: Load the full SavedModel and copy weights
                        print(f"Loading full SavedModel from: {model_dir}")
                        # Create a wrapper model that can load SavedModel
                        loaded_model = tf.saved_model.load(model_dir)
                        
                        # Try to get the actual keras model from the loaded object
                        if hasattr(loaded_model, 'keras_api'):
                            keras_model = loaded_model.keras_api.model
                            model.set_weights(keras_model.get_weights())
                        elif hasattr(loaded_model, '__call__'):
                            # It's a concrete function, we need to extract variables
                            # Get all trainable variables
                            variables = loaded_model.variables
                            if variables:
                                # Try to match variables by name
                                loaded_weights = []
                                for var in model.variables:
                                    found = False
                                    for saved_var in variables:
                                        if var.name == saved_var.name or var.name.split('/')[-1] == saved_var.name.split('/')[-1]:
                                            loaded_weights.append(saved_var.numpy())
                                            found = True
                                            break
                                    if not found:
                                        print(f"Warning: Could not find matching variable for {var.name}")
                                        loaded_weights.append(var.numpy())
                                
                                if len(loaded_weights) == len(model.variables):
                                    model.set_weights(loaded_weights)
                                else:
                                    raise ValueError(f"Weight count mismatch: model has {len(model.variables)} variables, loaded {len(loaded_weights)}")
                        else:
                            raise ValueError("Could not extract model from SavedModel format")
                except Exception as e:
                    print(f"Failed to load from SavedModel: {e}")
                    raise
            else:
                # Fall back to old weights-only format
                model.load_weights(model_dir)
            
            return model
        
        except Exception as e:
            print(f"Error loading model from {model_dir}: {str(e)}")
            raise

    @staticmethod
    def save_multiple_models(models_dict: Dict[str, tf.keras.Model], base_path: str, custom_objects: Dict[str, Any], intensity_scale: float) -> None:
        """
        Save multiple models into a single zip archive.

        Args:
            models_dict (Dict[str, tf.keras.Model]): Dictionary of models to save.
            base_path (str): Base path for saving the zip archive.
            custom_objects (Dict[str, Any]): Dictionary of custom objects used in the models.
            intensity_scale (float): The intensity scale used in the models.
        """
        zip_path = f"{base_path}.zip"
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save manifest of included models
            manifest = {'models': list(models_dict.keys()), 'version': '1.0'}
            manifest_path = os.path.join(temp_dir, 'manifest.dill')
            with open(manifest_path, 'wb') as f:
                dill.dump(manifest, f)
            
            # Save each model to temp directory
            for model_name, model in models_dict.items():
                model_subdir = os.path.join(temp_dir, model_name)
                os.makedirs(model_subdir, exist_ok=True)
                ModelManager.save_model(model, model_subdir, custom_objects, intensity_scale)
            
            # Create zip archive
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arc_path = os.path.relpath(full_path, temp_dir)
                        zf.write(full_path, arc_path)

    @staticmethod
    def load_multiple_models(base_path: str, model_names: Optional[List[str]] = None) -> Dict[str, tf.keras.Model]:
        """
        Load multiple models from a zip archive.

        Args:
            base_path (str): Base path of the zip archive.
            model_names (Optional[List[str]]): List of model names to load. If None, loads all models.

        Returns:
            Dict[str, tf.keras.Model]: Dictionary of loaded models.
        """
        zip_path = f"{base_path}.zip"
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Model archive not found: {zip_path}")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip archive
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
            
            # Load manifest
            manifest_path = os.path.join(temp_dir, 'manifest.dill')
            with open(manifest_path, 'rb') as f:
                manifest = dill.load(f)
            
            # Determine which models to load
            available_models = manifest['models']
            if model_names is None:
                model_names = available_models
            else:
                # Validate requested models exist
                missing = set(model_names) - set(available_models)
                if missing:
                    raise ValueError(f"Requested models not found in archive: {missing}")
            
            # Load each requested model
            loaded_models = {}
            for model_name in model_names:
                model_subdir = os.path.join(temp_dir, model_name)
                loaded_models[model_name] = ModelManager.load_model(model_subdir)
            
            return loaded_models


def save(out_prefix: str) -> None:
    """Save models to a zip archive."""
    from ptycho import model
    from ptycho.model import ProbeIllumination, IntensityScaler, IntensityScaler_inv, negloglik, _get_log_scale
    from ptycho.tf_helper import Translation, CenterMaskLayer
    from ptycho.tf_helper import realspace_loss as hh_realspace_loss

    model_path = os.path.join(out_prefix, params.get('h5_path'))

    # CRITICAL FIX (INTENSITY-SCALE-SAVE-001): Get trained intensity_scale from log_scale variable
    # The log_scale tf.Variable is trained during training, but params.cfg['intensity_scale']
    # is never updated. We must save the trained value, not the initial params value.
    trained_log_scale = _get_log_scale()
    trained_intensity_scale = float(np.exp(trained_log_scale.numpy()))
    # Import custom layers
    from ptycho.custom_layers import (CombineComplexLayer, ExtractPatchesPositionLayer,
                                     PadReconstructionLayer, ReassemblePatchesLayer,
                                     TrimReconstructionLayer, PadAndDiffractLayer,
                                     FlatToChannelLayer, ScaleLayer, InvScaleLayer,
                                     ActivationLayer, SquareLayer)
    
    custom_objects = {
        'ProbeIllumination': ProbeIllumination,
        'IntensityScaler': IntensityScaler,
        'IntensityScaler_inv': IntensityScaler_inv,
        'Translation': Translation,
        'CenterMaskLayer': CenterMaskLayer,
        'negloglik': negloglik,
        'realspace_loss': hh_realspace_loss,
        # Add custom layers
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
    
    models_to_save = {
        'autoencoder': model.autoencoder,
        'diffraction_to_obj': model.diffraction_to_obj
    }

    # Use trained_intensity_scale (from log_scale variable), not params.get('intensity_scale')
    ModelManager.save_multiple_models(models_to_save, model_path, custom_objects, trained_intensity_scale)

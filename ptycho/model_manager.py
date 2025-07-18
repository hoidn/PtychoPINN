# model_manager.py

import os
import h5py
import dill
import tempfile
import zipfile
import shutil
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
            
            # Save the model
            model.save(model_dir, save_format="tf")
            
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
            
            # Import model factory after parameters are loaded
            from ptycho.model import create_model_with_gridsize
            
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
    from ptycho.model import ProbeIllumination, IntensityScaler, IntensityScaler_inv, negloglik
    from ptycho.tf_helper import Translation
    from ptycho.tf_helper import realspace_loss as hh_realspace_loss

    model_path = os.path.join(out_prefix, params.get('h5_path'))
    custom_objects = {
        'ProbeIllumination': ProbeIllumination,
        'IntensityScaler': IntensityScaler,
        'IntensityScaler_inv': IntensityScaler_inv,
        'Translation': Translation,
        'negloglik': negloglik,
        'realspace_loss': hh_realspace_loss
    }
    
    models_to_save = {
        'autoencoder': model.autoencoder,
        'diffraction_to_obj': model.diffraction_to_obj
    }
    
    ModelManager.save_multiple_models(models_to_save, model_path, custom_objects, params.get('intensity_scale'))

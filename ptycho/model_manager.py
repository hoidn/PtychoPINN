# model_manager.py

import os
import h5py
import dill
import tensorflow as tf
from typing import Dict, List, Any
from ptycho import params

class ModelManager:
    @staticmethod
    def save_model(model: tf.keras.Model, model_path: str, custom_objects: Dict[str, Any], intensity_scale: float, model_name: str) -> None:
        """
        Save a single model along with its custom objects, parameters, and intensity scale.

        Args:
            model (tf.keras.Model): The model to save.
            model_path (str): Base path for saving the model.
            custom_objects (Dict[str, Any]): Dictionary of custom objects used in the model.
            intensity_scale (float): The intensity scale used in the model.
            model_name (str): Name of the model.
        """
        model_dir = f"{model_path}_{model_name}"
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
            print(f"Error saving model {model_name}: {str(e)}")
            raise

    @staticmethod
    def load_model(model_path: str, model_name: str) -> tf.keras.Model:
        """
        Load a single model along with its custom objects, parameters, and intensity scale.

        Args:
            model_path (str): Base path for loading the model.
            model_name (str): Name of the model.

        Returns:
            tf.keras.Model: The loaded model.
        """
        model_dir = f"{model_path}_{model_name}"
        model_file = os.path.join(model_dir, "model.h5")
        custom_objects_path = os.path.join(model_dir, "custom_objects.dill")
        params_path = os.path.join(model_dir, "params.dill")
        
        try:
            
            # Load parameters
            with open(params_path, 'rb') as f:
                loaded_params = dill.load(f)
            
            # Check version and handle any necessary migrations
            version = loaded_params.pop('_version', '1.0')
            # Here you could add logic to handle different versions if needed
            
            # Update params.cfg with loaded parameters
            params.cfg.update(loaded_params)
            
            # Load custom objects
            with open(custom_objects_path, 'rb') as f:
                custom_objects = dill.load(f)
            
            # Load intensity scale
            with h5py.File(model_file, 'r') as hf:
                intensity_scale = hf.attrs['intensity_scale']
            
            # Set intensity scale in params
            params.set('intensity_scale', intensity_scale)

            # Load and return the model
            return tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
        
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise

    @staticmethod
    def save_multiple_models(models_dict: Dict[str, tf.keras.Model], base_path: str, custom_objects: Dict[str, Any], intensity_scale: float) -> None:
        """
        Save multiple models.

        Args:
            models_dict (Dict[str, tf.keras.Model]): Dictionary of models to save.
            base_path (str): Base path for saving the models.
            custom_objects (Dict[str, Any]): Dictionary of custom objects used in the models.
            intensity_scale (float): The intensity scale used in the models.
        """
        for model_name, model in models_dict.items():
            ModelManager.save_model(model, base_path, custom_objects, intensity_scale, model_name)

    @staticmethod
    def load_multiple_models(base_path: str, model_names: List[str]) -> Dict[str, tf.keras.Model]:
        """
        Load multiple models.

        Args:
            base_path (str): Base path for loading the models.
            model_names (List[str]): List of model names to load.

        Returns:
            Dict[str, tf.keras.Model]: Dictionary of loaded models.
        """
        loaded_models = {}
        for model_name in model_names:
            loaded_models[model_name] = ModelManager.load_model(base_path, model_name)
        return loaded_models


def save(out_prefix):
    from ptycho import model
    from ptycho.model import ProbeIllumination, IntensityScaler, IntensityScaler_inv, negloglik
    from ptycho.tf_helper import Translation
    from ptycho.tf_helper import realspace_loss as hh_realspace_loss

    model_path = '{}/{}'.format(out_prefix, params.get('h5_path'))
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

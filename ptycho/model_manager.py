# model_manager.py

import os
import h5py
import dill
import tensorflow as tf
from ptycho import params

class ModelManager:
    @staticmethod
    def save_model(model, model_path, custom_objects, intensity_scale, model_name):
        model_dir = f"{model_path}_{model_name}"
        model_file = os.path.join(model_dir, "model.h5")
        custom_objects_path = os.path.join(model_dir, "custom_objects.dill")
        
        # Ensure the directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model.save(model_dir, save_format="tf")
        
        # Save custom objects
        with open(custom_objects_path, 'wb') as f:
            dill.dump(custom_objects, f)
        
        # Save intensity scale
        with h5py.File(model_file, 'a') as hf:
            hf.attrs['intensity_scale'] = intensity_scale

    @staticmethod
    def load_model(model_path, model_name):
        model_dir = f"{model_path}_{model_name}"
        model_file = os.path.join(model_dir, "model.h5")
        custom_objects_path = os.path.join(model_dir, "custom_objects.dill")
        
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

    @staticmethod
    def save_multiple_models(models_dict, base_path, custom_objects, intensity_scale):
        for model_name, model in models_dict.items():
            ModelManager.save_model(model, base_path, custom_objects, intensity_scale, model_name)

    @staticmethod
    def load_multiple_models(base_path, model_names):
        loaded_models = {}
        for model_name in model_names:
            loaded_models[model_name] = ModelManager.load_model(base_path, model_name)
        return loaded_models

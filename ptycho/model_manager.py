import h5py
import dill
from tensorflow.keras.models import load_model as tf_load_model
from ptycho import params

class ModelManager:
    @staticmethod
    def save_model(model, model_path, custom_objects, intensity_scale):
        custom_objects_path = model_path + ".dill"
        model.save(model_path, save_format="tf")
        with open(custom_objects_path, 'wb') as f:
            dill.dump(custom_objects, f)
        with h5py.File(model_path, 'a') as hf:
            hf.attrs['intensity_scale'] = intensity_scale

    @staticmethod
    def load_model(model_path):
        custom_objects_path = model_path + ".dill"
        with open(custom_objects_path, 'rb') as f:
            custom_objects = dill.load(f)
        with h5py.File(model_path, 'r') as hf:
            intensity_scale = hf.attrs['intensity_scale']
        params.set('intensity_scale', intensity_scale)
        return tf_load_model(model_path, custom_objects=custom_objects)

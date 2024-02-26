import h5py
import dill
from tensorflow.keras.models import load_model as tf_load_model
from ptycho import params

class ModelManager:
    @staticmethod
    def save_model(model, model_path, custom_objects, intensity_scale):
        model.save(model_path, save_format="tf")
        with h5py.File(model_path, 'a') as f:
            f.attrs['custom_objects'] = dill.dumps(custom_objects)
            f.attrs['intensity_scale'] = intensity_scale

    @staticmethod
    def load_model(model_path):
        with h5py.File(model_path, 'r') as f:
            custom_objects = dill.loads(f.attrs['custom_objects'])
            intensity_scale = f.attrs['intensity_scale']
        params.set('intensity_scale', intensity_scale)
        return tf_load_model(model_path, custom_objects=custom_objects)

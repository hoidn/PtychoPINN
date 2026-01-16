"""
Legacy inference utilities for loading pre-trained models and performing reconstruction.

Example:
    >>> results = inference_flow("model.h5", data_container)
    >>> reconstructed_obj = results['reconstructed_obj']
"""
from ptycho.model_manager import ModelManager
from tensorflow.keras.models import Model
from ptycho import params
from ptycho.loader import PtychoDataContainer
import numpy as np

# TODO this module is for inference-only workflows. it needs to be consolidated with train_pinn

def load_pretrained_model(model_path: str) -> Model:
    """
    Load a pre-trained model from an H5 file.
    """
    model = ModelManager.load_model(model_path)
    return model

def prepare_data(data_container: PtychoDataContainer) -> tuple:
    """
    Prepare data for inference.
    """
    from ptycho import model
    X = data_container.X * model.params()['intensity_scale']
    coords_nominal = data_container.coords_nominal
    return X, coords_nominal

def perform_inference(model: Model, X: np.ndarray, coords_nominal: np.ndarray) -> dict:
    """
    Perform inference using the pre-trained model and prepared data.
    """
    from ptycho import model
    reconstructed_obj, pred_amp, reconstructed_obj_cdi = model.predict([X, coords_nominal])
    return {
        'reconstructed_obj': reconstructed_obj,
        'pred_amp': pred_amp,
        'reconstructed_obj_cdi': reconstructed_obj_cdi
    }

def inference_flow(model_path: str, data_container: PtychoDataContainer) -> dict:
    """
    The main flow for model inference, integrating the steps.
    """
    pre_trained_model = load_pretrained_model(model_path or params.get('h5_path'))
    X, coords_nominal = prepare_data(data_container)
    inference_results = perform_inference(pre_trained_model, X, coords_nominal)
    return inference_results

# Example usage
# model_path = 'path/to/model.h5'
# data_container = PtychoDataContainer(...)
# results = inference_flow(model_path, data_container)

# New alternative implementation
from ptycho.image import reassemble_patches as _reassemble_patches

def reassemble_with_config(reconstructed_obj, config, **kwargs):
    """
    Alternative implementation using new stitching module.
    Preserves existing behavior while allowing transition to new API.
    """
    try:
        return _reassemble_patches(reconstructed_obj, config, **kwargs)
    except (ValueError, TypeError) as e:
        print('Object stitching failed:', e)
        return None

import tensorflow as tf
from ptycho import params, model
from ptycho.loader import PtychoDataContainer

def load_pretrained_model(model_path):
    """
    Load a pre-trained model from an H5 file.
    """
    return tf.keras.models.load_model(model_path)

def prepare_data(data_container):
    """
    Prepare data for inference.
    """
    X = data_container.X * model.params()['intensity_scale']
    coords_nominal = data_container.coords_nominal
    return X, coords_nominal

def perform_inference(model, X, coords_nominal):
    """
    Perform inference using the pre-trained model and prepared data.
    """
    reconstructed_obj, pred_amp, reconstructed_obj_cdi = model.predict([X, coords_nominal])
    return {
        'reconstructed_obj': reconstructed_obj,
        'pred_amp': pred_amp,
        'reconstructed_obj_cdi': reconstructed_obj_cdi
    }

def inference_flow(model_path, data_container):
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

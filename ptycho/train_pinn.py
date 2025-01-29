from typing import List, Union
import tensorflow as tf
from ptycho import params
from .loader import PtychoDataContainer, MultiPtychoDataContainer
from .image import reassemble_patches

def train(train_data: Union[PtychoDataContainer, MultiPtychoDataContainer], intensity_scale=None, model_instance=None):
    from . import params as p
    # Model requires intensity_scale to be defined to set the initial
    # value of the corresponding model parameter
    if intensity_scale is None:
        intensity_scale = calculate_intensity_scale(train_data)
    p.set('intensity_scale', intensity_scale)

    from ptycho import probe
    probe.set_probe_guess(None, train_data.probe)

    from ptycho import model
    if model_instance is None:
        model_instance = model.autoencoder
    nepochs = params.cfg['nepochs']
    params.print_params()
    return model_instance, model.train(nepochs, train_data)

def train_eval(ptycho_dataset):
    ## TODO reconstructed_obj -> pred_Y or something
    model_instance, history = train(ptycho_dataset.train_data)
    results = {
        'history': history,
        'model_instance': model_instance
    }
    if ptycho_dataset.test_data is not None:
        eval_results = eval(ptycho_dataset.test_data, history, trained_model=model_instance)
        # Get config from the dataset
        config = ptycho_dataset.test_data.config if hasattr(ptycho_dataset.test_data, 'config') else params.cfg
        stitched_obj = reassemble_patches(eval_results['reconstructed_obj'], config, part='complex')
        results.update({
            'reconstructed_obj': eval_results['reconstructed_obj'],
            'pred_amp': eval_results['pred_amp'],
            'reconstructed_obj_cdi': eval_results['reconstructed_obj_cdi'],
            'stitched_obj': stitched_obj,
        })
    return results

from tensorflow.keras.models import load_model
# Enhance the existing eval function to optionally load a model for inference
def eval(test_data: Union[PtychoDataContainer, MultiPtychoDataContainer], history=None, trained_model=None, model_path=None):
    """
    Evaluate the model on test data. Optionally load a model if a path is provided.

    Parameters:
    - test_data: The test data for evaluation.
    - history: Training history, if available.
    - trained_model: An already trained model instance, if available.
    - model_path: Path to a saved model, if loading is required.

    Returns:
    - Evaluation results including reconstructed objects and prediction amplitudes.
    """
    from ptycho.data_preprocessing import reassemble

    from ptycho import probe
    probe.set_probe_guess(None, test_data.probe)
    # TODO enforce that the train and test probes are the same
    print('INFO:', 'setting probe from test data container. It MUST be consistent with the training probe')

    from ptycho import model
    if model_path is not None:
        print(f"Loading model from {model_path}")
        trained_model = load_model(model_path)
    elif trained_model is None:
        raise ValueError("Either a trained model instance or a model path must be provided.")

    reconstructed_obj, pred_amp, reconstructed_obj_cdi = trained_model.predict(
        [test_data.X * model.params()['intensity_scale'], test_data.coords_nominal]
    )
    try:
        stitched_obj = reassemble(reconstructed_obj, part='complex')
    except (ValueError, TypeError) as e:
        stitched_obj = None
        print('Object stitching failed:', e)
    return {
        'reconstructed_obj': reconstructed_obj,
        'pred_amp': pred_amp,
        'reconstructed_obj_cdi': reconstructed_obj_cdi,
        'stitched_obj': stitched_obj
    }

def calculate_intensity_scale(ptycho_data_container: Union[PtychoDataContainer, MultiPtychoDataContainer]) -> float:
    import tensorflow as tf
    import numpy as np
    from . import params as p
    def count_photons(obj):
        pcount = np.mean(tf.math.reduce_sum(obj**2, (1, 2)))
        return pcount

    def scale_nphotons(X):
        # TODO assumes X is already normalized. this should be enforced
        return tf.math.sqrt(p.get('nphotons')) / (p.get('N') / 2)

    # Calculate the intensity scale using the adapted scale_nphotons function
    intensity_scale = scale_nphotons(ptycho_data_container.X).numpy()

    return intensity_scale

# New alternative implementation
from ptycho.image import reassemble_patches as _reassemble_patches

def prepare_inputs(data: Union[PtychoDataContainer, MultiPtychoDataContainer]) -> List[tf.Tensor]:
    """Prepare model inputs including probe indices.
    
    Args:
        data: Training data container
        
    Returns:
        Model input tensors
    """
    inputs = [data.X * params.cfg.get('intensity_scale'), data.coords]
    
    # Add probe indices for multi-probe data
    if isinstance(data, MultiPtychoDataContainer):
        inputs.append(data.probe_indices)
    else:
        # For single probe, use zeros as probe indices
        inputs.append(tf.zeros([tf.shape(data.X)[0]], dtype=tf.int64))
        
    return inputs

def stitch_eval_result(reconstructed_obj, config, **kwargs):
    """
    Alternative implementation using new stitching module.
    Preserves existing behavior while allowing transition to new API.
    """
    try:
        return _reassemble_patches(reconstructed_obj, config, part='complex', **kwargs)
    except (ValueError, TypeError) as e:
        print('Object stitching failed:', e)
        return None

from .loader import PtychoDataContainer
from ptycho import params

def train(train_data: PtychoDataContainer, intensity_scale=None, model_instance=None,
          cfg = None, probeGuess = None):
    assert cfg is not None
    assert probeGuess is not None
    from . import params as p
    # Model requires intensity_scale to be defined to set the initial
    # value of the corresponding model parameter
    if intensity_scale is None:
        intensity_scale = calculate_intensity_scale(train_data)
    p.set('intensity_scale', intensity_scale)

    from ptycho import probe
    probe.set_probe_guess(None, train_data.probe)
    # TODO why can't this be in module scope?
    from ptycho import model
    if model_instance is not None:
        raise NotImplementedError

    nepochs = params.cfg['nepochs']
    model_layers, history = model.train(nepochs, train_data, probeGuess, cfg)
    return model_layers, history

def train_eval(ptycho_dataset, cfg = None, probeGuess = None):
    assert cfg is not None
    ## TODO reconstructed_obj -> pred_Y or something
    modelbuild, history = train(ptycho_dataset.train_data, cfg = cfg, probeGuess = probeGuess)
    eval_results = eval(ptycho_dataset.test_data, history, modelbuild = modelbuild, cfg = cfg)
    return {
        'history': history,
        'reconstructed_obj': eval_results['reconstructed_obj'],
        'pred_amp': eval_results['pred_amp'],
        'reconstructed_obj_cdi': eval_results['reconstructed_obj_cdi'],
        'stitched_obj': eval_results['stitched_obj'],
        'diffraction_to_obj': modelbuild['diffraction_to_obj'],
        'autoencoder': modelbuild['autoencoder'],
        'cfg': cfg
    }

from tensorflow.keras.models import load_model
# Enhance the existing eval function to optionally load a model for inference
def eval(test_data, history=None, modelbuild=None, model_path=None, cfg = None):
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
    assert cfg is not None
    if modelbuild is not None:
        trained_model = modelbuild['autoencoder']
    else:
        trained_model = None
    from ptycho.data_processing import reassemble

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
        [test_data.X * cfg['intensity_scale'], test_data.coords_nominal]
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

def calculate_intensity_scale(ptycho_data_container: PtychoDataContainer) -> float:
    import tensorflow as tf
    from . import params as p
    def count_photons(obj):
        return tf.math.reduce_sum(obj**2, (1, 2))

    def scale_nphotons(padded_obj):
        mean_photons = tf.math.reduce_mean(count_photons(padded_obj))
        norm = tf.math.sqrt(p.get('nphotons') / mean_photons)
        return norm

    # Extract the object data from the PtychoDataContainer
    obj_data = ptycho_data_container.Y_I  # Assuming Y_I contains the object data

    # Calculate the intensity scale using the adapted scale_nphotons function
    intensity_scale = scale_nphotons(obj_data).numpy()

    return intensity_scale

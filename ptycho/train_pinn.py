from ptycho.generate_data import reassemble
from ptycho import params, model

#offset = params.cfg['offset']
#N = params.cfg['N']
#gridsize = params.cfg['gridsize']
#jitter_scale = params.params()['sim_jitter_scale']
#batch_size = params.cfg['batch_size']

def train(train_data, model_instance = None):
    # training parameters
    if model_instance is None:
        model_instance = model.autoencoder
    nepochs = params.cfg['nepochs']
    return model_instance, model.train(nepochs, train_data)

def eval(test_data, history, trained_model = None):
    if trained_model is None:
        trained_model = model.autoencoder
    reconstructed_obj, pred_amp, reconstructed_obj_cdi = trained_model.predict(
        [test_data.X * model.params()['intensity_scale'], test_data.coords_nominal]
    )
    try:
        stitched_obj = reassemble(reconstructed_obj, part='complex')
    except (ValueError, TypeError) as e:
        stitched_obj = None
        print('object stitching failed:', e)
    return {
        'reconstructed_obj': reconstructed_obj,
        'pred_amp': pred_amp,
        'reconstructed_obj_cdi': reconstructed_obj_cdi,
        'stitched_obj': stitched_obj
    }

def train_eval(ptycho_dataset):
    ## TODO reconstructed_obj -> pred_Y or something
    model_instance, history = train(ptycho_dataset.train_data)
    eval_results = eval(ptycho_dataset.test_data, history, trained_model = model_instance)
    return {
        'history': history,
        'reconstructed_obj': eval_results['reconstructed_obj'],
        'pred_amp': eval_results['pred_amp'],
        'reconstructed_obj_cdi': eval_results['reconstructed_obj_cdi'],
        'stitched_obj': eval_results['stitched_obj'],
        'model_instance': model_instance,
        'dataset': ptycho_dataset.train_data
    }
from tensorflow.keras.models import load_model

# Enhance the existing eval function to optionally load a model for inference
def eval(test_data, history=None, trained_model=None, model_path=None):
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

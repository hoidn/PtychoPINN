"""Physics-informed neural network (PINN) training implementation for ptychographic reconstruction.

This module implements the PINN training workflow, which differs fundamentally from supervised 
learning by integrating physics constraints directly into the neural network loss function.
Unlike traditional supervised approaches that learn input-output mappings from ground truth 
data, the PINN approach enforces physical consistency through differentiable forward modeling 
of the ptychographic measurement process.

**Physics-Informed Training Architecture:**
The PINN training approach combines three key physics-informed loss components:
1. **Poisson Negative Log-Likelihood (NLL)**: Enforces realistic photon noise statistics
   matching experimental diffraction measurements via negloglik() loss function
2. **Real-Space Consistency**: Optional realspace_loss() constraining object reconstruction
   to be physically plausible in the sample domain
3. **Differentiable Forward Model**: Custom TensorFlow layers embed the complete 
   ptychographic measurement equation (object * probe → |FFT|² → Poisson noise)

**Key Differences from Supervised Learning:**
- **No Ground Truth Required**: Trains directly on experimental diffraction data without
  requiring known object/phase references
- **Physics-Constrained**: Loss functions enforce Maxwell equations and photon statistics
  rather than pixel-wise reconstruction accuracy
- **Self-Supervised**: The forward physics model provides supervision signals through
  differentiable simulation of the measurement process
- **Measurement Fidelity**: Optimizes for consistency with actual experimental physics
  rather than similarity to reference images

**Integration with Core Physics:**
- **ptycho.model**: Provides the U-Net + physics layers architecture with PINN loss terms
- **ptycho.diffsim**: Supplies differentiable forward model for physics constraints
- **ptycho.tf_helper**: Implements realspace_loss() and Poisson statistics functions
- **ptycho.probe**: Manages probe setup and trainable parameters for PINN optimization

Key Functions:
    train(): Main PINN training workflow with physics-informed loss configuration
    train_eval(): Complete training + evaluation pipeline with object stitching
    eval(): Inference on test data using trained PINN model
    calculate_intensity_scale(): Computes physics-consistent intensity normalization

Example:
    # Complete PINN training workflow with physics constraints (no ground truth required)
    from ptycho.raw_data import RawData
    from ptycho.loader import load
    from ptycho import params

    # Load experimental dataset (NPZ with xcoords/ycoords/diff3d/probeGuess)
    raw = RawData.from_file('experimental_data.npz')

    # Group and convert to tensors via callback
    def cb():
        return raw.generate_grouped_data(N=params.get('N'), K=7, nsamples=1024, gridsize=params.get('gridsize'))
    train_data = load(cb, raw.probeGuess, which='train', create_split=False)

    # Configure physics parameters
    params.set('nll_weight', 1.0)       # Poisson NLL physics constraint weight
    params.set('realspace_weight', 0.1) # Real-space consistency weight
    params.set('nphotons', 1e6)         # Expected photon count for Poisson model

    # Train with physics-informed loss functions
    model_instance, history = train(train_data)

    # Evaluate reconstruction quality (optional test split)
    # results = eval(test_data, trained_model=model_instance)
"""

from ptycho import params
from .loader import PtychoDataContainer
from .image import reassemble_patches

def train(train_data: PtychoDataContainer, intensity_scale=None, model_instance=None):
    from . import params as p
    # Model requires intensity_scale to be defined to set the initial
    # value of the corresponding model parameter
    if intensity_scale is None:
        intensity_scale = calculate_intensity_scale(train_data)
    p.set('intensity_scale', intensity_scale)

    from ptycho import probe
    probe.set_probe_guess(None, train_data.probe)

    from ptycho import model

    # Create fresh model with current params instead of using stale singleton
    # This fixes MODULE-SINGLETON-001: model architecture must match current gridsize
    if model_instance is None:
        model_instance, diffraction_to_obj = model.create_compiled_model()
        # Update module-level singletons so model_manager.save() saves the trained model
        # (SINGLETON-SAVE-001: save() hardcodes model.autoencoder/diffraction_to_obj)
        model.autoencoder = model_instance
        model.diffraction_to_obj = diffraction_to_obj

    nepochs = params.cfg['nepochs']
    params.print_params()
    return model_instance, model.train(nepochs, train_data, model_instance=model_instance)

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
        try:
            stitched_obj = reassemble_patches(eval_results['reconstructed_obj'], config, part='complex')
        except ValueError as e:
            print(e)
            stitched_obj = None

        results.update({
            'reconstructed_obj': eval_results['reconstructed_obj'],
            'pred_amp': eval_results['pred_amp'],
            'reconstructed_obj_cdi': eval_results['reconstructed_obj_cdi'],
            'stitched_obj': stitched_obj,
        })
    return results

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
        [test_data.X * params.get('intensity_scale'), test_data.coords_nominal]
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
    """Compute intensity scale per specs/spec-ptycho-core.md Normalization Invariants.

    Two compliant calculation modes with the following precedence:
    1) Dataset-derived (preferred): s = sqrt(nphotons / E_batch[sum_xy |X|^2])
    2) Closed-form fallback: s = sqrt(nphotons) / (N/2) when batch_mean is near zero

    Args:
        ptycho_data_container: Container with .X tensor (B, N, N, C) float32

    Returns:
        Intensity scale as Python float
    """
    import tensorflow as tf
    from . import params as p

    # Cast to float64 for numerical stability
    X = tf.cast(ptycho_data_container.X, tf.float64)

    # X shape: (B, N, N, C) - compute sum of squared amplitudes over spatial and channel dims
    # Dynamically determine reduction axes: all except batch (axis 0)
    ndims = len(X.shape)
    reduction_axes = tuple(range(1, ndims))  # (1, 2, 3) for rank-4, (1, 2) for rank-3

    # Sum |X|^2 over spatial dimensions for each sample
    sum_intensity = tf.reduce_sum(X ** 2, axis=reduction_axes)  # shape (B,)

    # E_batch[sum_xy |X|^2]
    batch_mean = tf.reduce_mean(sum_intensity)

    nphotons = tf.cast(p.get('nphotons'), tf.float64)
    N = tf.cast(p.get('N'), tf.float64)

    # Dataset-derived scale when batch_mean is sufficiently large
    if batch_mean > 1e-12:
        dataset_scale = tf.sqrt(nphotons / batch_mean)
        return float(dataset_scale.numpy())
    else:
        # Closed-form fallback: s = sqrt(nphotons) / (N/2)
        fallback_scale = tf.sqrt(nphotons) / (N / 2.0)
        return float(fallback_scale.numpy())

# New alternative implementation
from ptycho.image import reassemble_patches as _reassemble_patches

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

import logging
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ptycho.workflows.components import (
    load_data,
    run_cdi_example,
    save_outputs
)
from ptycho.config.config import TrainingConfig, InferenceConfig, ModelConfig, update_legacy_dict
from ptycho import model_manager, params, probe
from ptycho.nbutils import reconstruct_image, crop_to_non_uniform_region_with_buffer, probeshow

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.StreamHandler(sys.stdout),
                       logging.FileHandler('train_and_infer.log')
                   ])
logger = logging.getLogger(__name__)

def train_model(config: TrainingConfig):
    """Train the model using provided configuration."""
    logger.info("Starting training process...")
    
    try:
        # Load training data
        ptycho_data = load_data(str(config.train_data_file), n_images=512)
        
        # Load test data if provided
        test_data = None
        if config.test_data_file:
            test_data = load_data(str(config.test_data_file))

        # Run training
        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config)
        
        # Save model and outputs
        model_manager.save(str(config.output_dir))
        save_outputs(recon_amp, recon_phase, results, str(config.output_dir))
        
        return recon_amp, recon_phase, results
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def perform_inference(model: tf.keras.Model, test_data, K: int = 7, nsamples: int = 1):
    """Perform inference using trained model."""
    logger.info("Starting inference process...")
    
    try:
        # Set probe guess and random seeds
        probe.set_probe_guess(None, test_data.probeGuess)
        tf.random.set_seed(45)
        np.random.seed(45)

        # Generate test dataset
        gridsize = params.cfg.get('gridsize', 1)
        test_dataset = test_data.generate_grouped_data(params.cfg['N'], K=K, nsamples=nsamples, gridsize=gridsize)
        
        # Create data container
        from ptycho import loader
        test_data_container = loader.load(lambda: test_dataset, test_data.probeGuess, which=None, create_split=False)
        
        # Perform reconstruction
        obj_tensor_full, global_offsets = reconstruct_image(test_data_container, diffraction_to_obj=model)
        
        # Reassemble position
        from ptycho.tf_helper import reassemble_position
        obj_image = reassemble_position(obj_tensor_full, global_offsets, M=20)
        
        # Extract amplitude and phase
        reconstructed_amplitude = np.abs(obj_image)
        reconstructed_phase = np.angle(obj_image)
        
        # Process ePIE results
        epie_phase = crop_to_non_uniform_region_with_buffer(np.angle(test_data.objectGuess), buffer=-20)
        epie_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(test_data.objectGuess), buffer=-20)
        
        return reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

def plot_comparison(reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase):
    """Plot comparison between reconstructed and ePIE results."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot phases
    im_pinn_phase = axs[0, 0].imshow(reconstructed_phase, cmap='gray')
    axs[0, 0].set_title('PtychoPINN Phase')
    fig.colorbar(im_pinn_phase, ax=axs[0, 0])
    
    im_epie_phase = axs[0, 1].imshow(epie_phase, cmap='gray')
    axs[0, 1].set_title('ePIE Phase')
    fig.colorbar(im_epie_phase, ax=axs[0, 1])
    
    # Plot amplitudes
    im_pinn_amp = axs[1, 0].imshow(reconstructed_amplitude, cmap='viridis')
    axs[1, 0].set_title('PtychoPINN Amplitude')
    fig.colorbar(im_pinn_amp, ax=axs[1, 0])
    
    im_epie_amp = axs[1, 1].imshow(epie_amplitude, cmap='viridis')
    axs[1, 1].set_title('ePIE Amplitude')
    fig.colorbar(im_epie_amp, ax=axs[1, 1])
    
    # Remove ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def plot_probe(test_data):
    """Generate probe visualization."""
    return probeshow(test_data.probeGuess, test_data)

# Example usage in notebook:
"""
# Configuration
train_config = TrainingConfig(
    model=ModelConfig(),
    train_data_file=Path('path/to/train_data.npz'),
    test_data_file=Path('path/to/test_data.npz'),
    output_dir=Path('output_directory'),
    debug=False
)

# Update global params
update_legacy_dict(params.cfg, train_config)

# Train model
recon_amp, recon_phase, results = train_model(train_config)

# Load model for inference
model, _ = model_manager.ModelManager.load_model(train_config.output_dir)

# Load test data
test_data = load_data('path/to/test_data.npz')

# Perform inference
reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase = perform_inference(
    model, test_data, K=7, nsamples=1
)

# Plot results
fig = plot_comparison(reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase)
plt.show()

# Plot probe visualization
probe_fig = plot_probe(test_data)
plt.show()
"""

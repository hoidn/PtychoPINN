"""TensorFlow CDI orchestration: container factory, training, reassembly.

Holds the public doors (``train_cdi_model``, ``run_cdi_example``) and the
stitching/output helpers.  Collaborators that tests patch through the
``components`` facade (``resolve_generator``, ``create_ptycho_data_container``,
``probe``) are resolved late-bound via ``_components``.
"""
import logging
import os
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ptycho import loader, params, probe
from ptycho.config.config import (
    TrainingConfig,
    resolve_model_object_policy,
    update_legacy_dict,
)
from ptycho.config.legacy_state import scoped_legacy_params
from ptycho.grouping import group_from_config
from ptycho.loader import PtychoDataContainer, RawData
from ptycho.generators import registry

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho.workflows.components')

def _resolve_tensorflow_training_config(config: TrainingConfig) -> TrainingConfig:
    """Validate and materialize the public policy for TensorFlow entrypoints."""
    return replace(
        config,
        model=resolve_model_object_policy(
            config.model,
            backend="tensorflow",
            warn_deprecated=False,
        ),
    )


def create_ptycho_data_container(data: Union[RawData, PtychoDataContainer], config: TrainingConfig) -> PtychoDataContainer:
    """
    Factory function to create or return a PtychoDataContainer.

    Args:
        data (Union[RawData, PtychoDataContainer]): Input data, either RawData or PtychoDataContainer.
        config (TrainingConfig): Training configuration object.

    Returns:
        PtychoDataContainer: The resulting PtychoDataContainer.

    Raises:
        TypeError: If the input data is neither RawData nor PtychoDataContainer.
    """
    if isinstance(data, PtychoDataContainer):
        return data
    elif isinstance(data, RawData):
        # Grouping semantics (seed/oversampling/pool/count) live once in
        # ptycho.grouping.group_from_config; both mirrors delegate here.
        dataset = group_from_config(
            data,
            config,
            dataset_path=str(config.train_data_file) if config.train_data_file else None,
        )
        return loader.load(lambda: dataset, data.probeGuess, which=None, create_split=False)
    else:
        raise TypeError("data must be either RawData or PtychoDataContainer")

@scoped_legacy_params
def train_cdi_model(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig
) -> Dict[str, Any]:
    """
    Train the CDI model.

    Args:
        train_data (Union[RawData, PtychoDataContainer]): Training data.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Results dictionary containing training history.
    """
    config = _resolve_tensorflow_training_config(config)

    from ptycho.loader import PtychoDataset
    from ptycho import train_pinn
    # Convert input data to PtychoDataContainer
    train_container = create_ptycho_data_container(train_data, config)
    if test_data is not None:
        test_container = create_ptycho_data_container(test_data, config)
    else:
        test_container = None

    # Initialize probe
    probe.set_probe_guess(None, train_container.probe)

    # Ensure intensity_scale is available before model construction.
    if 'intensity_scale' not in params.cfg:
        intensity_scale = train_pinn.calculate_intensity_scale(train_container)
        params.set('intensity_scale', intensity_scale)

    # Resolve generator from config and build model
    # See ptycho/generators/README.md for adding new generators
    generator = registry.resolve_generator(config)
    logger.info(f"Using generator: {generator.name}")
    model_instance, diffraction_to_obj = generator.build_models()

    # Update module-level singletons so model_manager.save() saves the trained model
    # (SINGLETON-SAVE-001: save() hardcodes model.autoencoder/diffraction_to_obj)
    from ptycho import model
    model.autoencoder = model_instance
    model.diffraction_to_obj = diffraction_to_obj

    # Train the model
    results = train_pinn.train_eval(PtychoDataset(train_container, test_container), model_instance=model_instance)

    # Normalize history payload so downstream consumers always receive a dict.
    history_payload = results.get('history')
    normalized_history: Dict[str, Any] = {}
    if isinstance(history_payload, dict):
        normalized_history = history_payload
    elif history_payload is not None and hasattr(history_payload, 'history'):
        normalized_history = dict(history_payload.history or {})
    # Maintain legacy key expected by study runners even if Keras only reports "loss".
    if normalized_history and 'train_loss' not in normalized_history and 'loss' in normalized_history:
        normalized_history['train_loss'] = normalized_history['loss']
    results['history'] = normalized_history
    if history_payload is not None and hasattr(history_payload, 'epoch'):
        results['history_epochs'] = list(history_payload.epoch)

    results['train_container'] = train_container
    results['test_container'] = test_container
    #history = train_pinn.train(train_container)
    
    return results

def reassemble_cdi_image(
    test_data: Union[RawData, PtychoDataContainer],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    coord_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Reassemble the CDI image using the trained model.

    Args:
        test_data (Union[RawData, PtychoDataContainer]): Test data.
        config (Dict[str, Any]): Configuration dictionary.
        flip_x (bool): Whether to flip the x coordinates. Default is False.
        flip_y (bool): Whether to flip the y coordinates. Default is False.
        transpose (bool): Whether to transpose the image by swapping the 1st and 2nd dimensions. Default is False.
        M (int): Parameter for reassemble_position function. Default is 20.
        coord_scale (float): Scale factor for x and y coordinates. Default is 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        Reconstructed amplitude, reconstructed phase, and results dictionary.
    """
    # TODO use train_pinn.eval to get reconstructed diffraction amplitude
    test_container = create_ptycho_data_container(test_data, config)
    
    from ptycho import nbutils
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(test_container)
    
    # Log the shape of global_offsets
    logger.info(f"Shape of global_offsets: {global_offsets.shape}")

    # Assert that obj_tensor_full is a 4D tensor
    assert obj_tensor_full.ndim == 4, f"Expected obj_tensor_full to be a 4D tensor, but got shape {obj_tensor_full.shape}"

    # Transpose the image if requested
    if transpose:
        obj_tensor_full = np.transpose(obj_tensor_full, (0, 2, 1, 3))

    # Flip coordinates if requested
    if flip_x:
        global_offsets[:, 0, 0, :] = -global_offsets[:, 0, 0, :]
    if flip_y:
        global_offsets[:, 0, 1, :] = -global_offsets[:, 0, 1, :]
    
    # Scale coordinates
    global_offsets *= coord_scale
    
    from ptycho import tf_helper as hh
    obj_image = hh.reassemble_position(obj_tensor_full, global_offsets, M=M)
    
    recon_amp = np.absolute(obj_image)
    recon_phase = np.angle(obj_image)
    
    results = {
        "obj_tensor_full": obj_tensor_full,
        "global_offsets": global_offsets,
        "recon_amp": recon_amp,
        "recon_phase": recon_phase
    }
    
    return recon_amp, recon_phase, results

@scoped_legacy_params
def run_cdi_example(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Run the main CDI example execution flow.

    Args:
        train_data: Training data
        test_data: Optional test data
        config: Training configuration parameters
        flip_x: Whether to flip the x coordinates
        flip_y: Whether to flip the y coordinates
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function
        do_stitching: Whether to perform image stitching after training

    Returns:
        Tuple containing:
        - reconstructed amplitude (or None)
        - reconstructed phase (or None)
        - results dictionary
    """
    config = _resolve_tensorflow_training_config(config)

    # Update global params with new-style config at entry point
    update_legacy_dict(params.cfg, config)
    
    # Train the model
    train_results = train_cdi_model(train_data, test_data, config)
    
    recon_amp, recon_phase = None, None
    
    # Reassemble test image if stitching is enabled, test data is provided, and reconstructed_obj is available
    if do_stitching and test_data is not None and 'reconstructed_obj' in train_results:
        logger.info("Performing image stitching...")
        recon_amp, recon_phase, reassemble_results = reassemble_cdi_image(
            test_data, config, flip_x, flip_y, transpose, M=M
        )
        train_results.update(reassemble_results)
    else:
        logger.info("Skipping image stitching (disabled or no test data available)")
    
    return recon_amp, recon_phase, train_results


def save_outputs(amplitude: Optional[np.ndarray], phase: Optional[np.ndarray], results: Dict[str, Any], output_prefix: str) -> None:
    """Save the generated images and results."""
    os.makedirs(output_prefix, exist_ok=True)
    
    # TODO Save training history with tensorboard / mlflow
    
    # Save test results if available
    if amplitude is not None and phase is not None:
        logger.info(f"Amplitude array shape: {amplitude.shape}")
        logger.info(f"Phase array shape: {phase.shape}")
        
        # Squeeze any extra dimensions
        amplitude = np.squeeze(amplitude)
        phase = np.squeeze(phase)
        
        logger.info(f"Squeezed amplitude shape: {amplitude.shape}")
        logger.info(f"Squeezed phase shape: {phase.shape}")
        
        # Save as PNG files using plt.figure() to handle 2D arrays properly
        plt.figure(figsize=(8,8))
        plt.imshow(amplitude, cmap='gray')
        plt.colorbar()
        plt.savefig(os.path.join(output_prefix, "reconstructed_amplitude.png"))
        plt.close()
        
        plt.figure(figsize=(8,8))
        plt.imshow(phase, cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(output_prefix, "reconstructed_phase.png"))
        plt.close()
        
    logger.info(f"Outputs saved to {output_prefix}")

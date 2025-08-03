"""
Legacy inference utilities for loading pre-trained models and performing ptychographic reconstruction.

This module provides a simplified interface for running inference with pre-trained PtychoPINN models.
It handles model loading, data preparation, and reconstruction execution in a streamlined workflow.
The module serves as a bridge between saved model artifacts and the inference pipeline, abstracting
away the complexities of model initialization and data preprocessing.

Architecture Role:
    Sits between the training pipeline and end-user inference workflows:
    Saved Model (.h5) + Test Data → Model Loading → Data Preparation → Inference Execution → Reconstruction Results
    
    Key responsibilities:
    - Load pre-trained models via ModelManager
    - Prepare PtychoDataContainer inputs for inference
    - Execute model prediction with proper data scaling
    - Return structured reconstruction results
    - Provide fallback object stitching capabilities

Public Interface:
    `inference_flow(model_path, data_container)`
        - Purpose: Complete end-to-end inference workflow
        - Critical Behavior: Applies intensity scaling from global params during data preparation
        - Key Parameters: model_path (H5 file or None for params.h5_path), data_container (preprocessed data)
    
    `load_pretrained_model(model_path)`
        - Purpose: Load saved model weights and architecture via ModelManager
        - Critical Behavior: Delegates to ModelManager.load_model() for proper model restoration
        - Key Parameters: model_path (path to .h5 model file)
    
    `perform_inference(model, X, coords_nominal)`
        - Purpose: Execute model prediction on prepared inputs
        - Critical Behavior: Returns dict with 'reconstructed_obj', 'pred_amp', 'reconstructed_obj_cdi'
        - Key Parameters: model (loaded Keras model), X (scaled diffraction data), coords_nominal (scan positions)

Workflow Usage Example:
    ```python
    from ptycho.loader import PtychoDataContainer
    from ptycho.inference import inference_flow
    
    # Load test data
    data_container = PtychoDataContainer.from_file("test_data.npz")
    
    # Run complete inference pipeline
    results = inference_flow("trained_model.h5", data_container)
    
    # Extract results
    reconstructed_object = results['reconstructed_obj']
    predicted_amplitudes = results['pred_amp']
    cdi_reconstruction = results['reconstructed_obj_cdi']
    ```

Architectural Notes & Dependencies:
- ModelManager: Handles model loading/saving with proper weight restoration
- PtychoDataContainer: Provides preprocessed, model-ready diffraction data and coordinates
- Global params: Used for intensity scaling factor and fallback model paths
- ptycho.model: Direct import for accessing model parameters during data preparation
- ptycho.image: Optional dependency for modern object stitching capabilities
- State Dependency: Relies on global params configuration for intensity_scale and h5_path defaults
- Legacy Note: This module uses older parameter access patterns and is marked for consolidation with train_pinn
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

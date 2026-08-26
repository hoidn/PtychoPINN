"""TensorFlow bundle loading: the diffraction-to-object adapter and loaders.

Owns ``load_inference_bundle`` / ``load_inference_bundle_explicit`` and the
``DiffractionToObjectAdapter`` that keeps ``params.cfg['gridsize']`` aligned
with grouped inference inputs.
"""
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from ptycho import params
from ptycho.config.legacy_state import (
    isolated_archived_params_scope,
    legacy_params_scope,
    transactional_legacy_params,
)
from ptycho.model_manager import ModelManager

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho.workflows.components')

class DiffractionToObjectAdapter(tf.keras.Model):
    """
    Wrapper that keeps params.cfg['gridsize'] aligned with grouped inference inputs.

    Some exported bundles were trained with grouped data but load with lingering
    gridsize=1 in params.cfg, causing Translation to see B vs B*C tensors.
    By inspecting the diffraction input just before execution we can set the
    legacy gridsize to sqrt(channel_count) and avoid Translation crashes.
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        *,
        runtime_params: Optional[dict] = None,
    ):
        super().__init__(name=getattr(base_model, "name", "diffraction_to_obj"))
        self._model = base_model
        self._runtime_params = dict(runtime_params or {})

    def _infer_channel_count(self, diffraction_input) -> Optional[int]:
        if diffraction_input is None:
            return None

        # Try static shape first
        shape = getattr(diffraction_input, "shape", None)
        if shape is not None and shape[-1] not in (None, -1):
            return int(shape[-1])

        try:
            array_view = np.asarray(diffraction_input)
        except Exception:
            return None

        if array_view.size == 0 or array_view.ndim < 1:
            return None
        return int(array_view.shape[-1])

    def _sync_gridsize(self, maybe_inputs) -> None:
        if maybe_inputs is None:
            return

        if isinstance(maybe_inputs, (list, tuple)):
            diffraction = maybe_inputs[0]
        else:
            diffraction = maybe_inputs

        channels = self._infer_channel_count(diffraction)
        if channels is None or channels <= 0:
            return

        gridsize = int(round(math.sqrt(channels)))
        if gridsize * gridsize != channels or gridsize <= 0:
            return

        if params.cfg.get('gridsize') != gridsize:
            params.cfg['gridsize'] = gridsize

    # Named archive-restore boundary (W3.1): call/predict re-project the
    # bundle's runtime params (and input-derived gridsize) inside a scope.
    def call(self, inputs, training=False, **kwargs):
        with legacy_params_scope():
            params.cfg.update(self._runtime_params)
            self._sync_gridsize(inputs)
            return self._model(inputs, training=training, **kwargs)

    def predict(self, *args, **kwargs):
        with legacy_params_scope():
            params.cfg.update(self._runtime_params)
            input_arg = args[0] if args else kwargs.get('x')
            self._sync_gridsize(input_arg)
            return self._model.predict(*args, **kwargs)

    def __getattr__(self, item):
        underlying = super().__getattribute__("_model")
        return getattr(underlying, item)


@transactional_legacy_params
def load_inference_bundle(model_dir: Path) -> Tuple[tf.keras.Model, dict]:
    """Load a trained model bundle for inference from a directory.
    
    This is the standard, centralized function for loading a trained model for inference.
    It expects a directory from a training run containing a 'wts.h5.zip' archive with
    the 'diffraction_to_obj' inference model.
    
    Args:
        model_dir: Path to the directory containing the trained model artifacts.
                  This directory should contain 'wts.h5.zip' from a training run.
    
    Returns:
        A tuple containing:
        - model: The loaded TensorFlow/Keras model ready for inference
        - config: The configuration dictionary restored from the saved model
        
    Raises:
        ValueError: If model_dir is not a valid directory
        FileNotFoundError: If 'wts.h5.zip' is not found in the directory
        KeyError: If 'diffraction_to_obj' model is not found in the archive
        
    Example:
        >>> from pathlib import Path
        >>> from ptycho.workflows.components import load_inference_bundle
        >>> 
        >>> model_dir = Path("outputs/my_training_run")
        >>> model, config = load_inference_bundle(model_dir)
        >>> 
        >>> # Now use the model for inference
        >>> predictions = model.predict(test_data)
    """
    # Validate input directory
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")
    
    # Check for the model archive
    model_zip = model_dir / "wts.h5"
    model_zip_file = Path(f"{model_zip}.zip")
    
    if not model_zip_file.exists():
        raise FileNotFoundError(
            f"Model archive not found at: {model_zip_file}. "
            f"Expected to find 'wts.h5.zip' in the directory {model_dir}. "
            f"This file is created during training with ptycho_train."
        )
    
    logger.info(f"Loading model from: {model_dir}")
    logger.debug(f"Model archive path: {model_zip_file}")
    
    try:
        # Load multiple models from the archive
        # ModelManager expects the path without the .zip extension
        models_dict = ModelManager.load_multiple_models(
            str(model_zip),
            model_names=["diffraction_to_obj"],
        )
        
        # Get the diffraction_to_obj model which is needed for inference
        if 'diffraction_to_obj' not in models_dict:
            available_models = list(models_dict.keys())
            raise KeyError(
                f"No 'diffraction_to_obj' model found in saved models archive. "
                f"Available models: {available_models}. "
                f"The 'diffraction_to_obj' model should be created during training."
            )
        
        # ModelManager updates the global params.cfg when loading
        # Return a copy to avoid unintended modifications
        config = params.cfg.copy()
        model = DiffractionToObjectAdapter(
            models_dict['diffraction_to_obj'],
            runtime_params=config,
        )
        
        logger.info(f"Successfully loaded model from {model_dir}")
        logger.debug(f"Model configuration: {config}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {str(e)}")
        raise


def load_inference_bundle_explicit(
    model_dir: Path,
) -> Tuple[tf.keras.Model, dict]:
    """Load an archived TensorFlow model without retaining global state.

    Historical TensorFlow model reconstruction still requires the archived
    projection while the model is built. This modern seam returns that
    projection to the caller and restores the exact pre-load ``params.cfg``
    contents when reconstruction finishes.
    """
    with isolated_archived_params_scope():
        return load_inference_bundle(model_dir)



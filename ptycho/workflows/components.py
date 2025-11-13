"""High-level workflow orchestration layer for PtychoPINN pipeline integration.

This module serves as the primary orchestration layer that chains together core PtychoPINN 
modules into complete end-to-end workflows. It bridges the gap between the high-level 
scripts/command-line interfaces and low-level core library modules, providing standardized 
interfaces for data loading, configuration management, model training, and result assembly.

Architecture Role:
    The module operates at the workflow orchestration level, sitting above the core library 
    modules (model.py, diffsim.py, loader.py, etc.) and below the top-level scripts. It 
    integrates the complete PtychoPINN pipeline by:
    
    1. Configuration Management: Bridges modern dataclass-based config with legacy params
    2. Data Pipeline Integration: Orchestrates RawData → PtychoDataContainer → training
    3. Training Workflow: Chains data loading, probe initialization, and model training
    4. Reconstruction Pipeline: Coordinates inference, image reassembly, and visualization
    5. Result Management: Handles output serialization and visualization

Core Workflow Functions:
    Configuration Orchestration:
        - update_config_from_dict(): Update global config from dict (notebook workflows)
        - parse_arguments(): Auto-generate CLI parser from TrainingConfig dataclass
        - setup_configuration(): Merge YAML, CLI args, and defaults into unified config
        - load_yaml_config(): Load and validate YAML configuration files
    
    Data Pipeline Integration:
        - load_data(): Load NPZ data with coordinate transformations and validation
        - create_ptycho_data_container(): Factory for RawData → PtychoDataContainer conversion
        - load_and_prepare_data(): Legacy data loading interface (deprecated)
    
    End-to-End Workflow Orchestration:
        - run_cdi_example(): Complete training → reconstruction → visualization pipeline
        - train_cdi_model(): Orchestrate data preparation, probe setup, and model training
        - reassemble_cdi_image(): Coordinate reconstruction and image stitching workflows
        - save_outputs(): Handle result serialization and visualization generation

Integration Points:
    - Core Modules: Integrates ptycho.loader, ptycho.train_pinn, ptycho.probe, ptycho.tf_helper
    - Configuration: Bridges TrainingConfig dataclass with legacy params.cfg dictionary
    - Data Flow: Manages RawData → PtychoDataContainer → trained model → reconstruction
    - Visualization: Coordinates with matplotlib for result visualization and export

Example Usage:
    Complete end-to-end workflow orchestration:
    
    >>> from ptycho.workflows.components import (
    ...     run_cdi_example, load_data, setup_configuration, parse_arguments
    ... )
    >>> 
    >>> # Parse CLI arguments and setup unified configuration
    >>> args = parse_arguments()
    >>> config = setup_configuration(args, yaml_path=args.config)
    >>> 
    >>> # Load and validate training data
    >>> train_data = load_data(str(config.train_data_file), n_images=config.n_images)
    >>> test_data = load_data(str(config.test_data_file)) if config.test_data_file else None
    >>> 
    >>> # Execute complete pipeline: training → reconstruction → visualization
    >>> amplitude, phase, results = run_cdi_example(
    ...     train_data, test_data, config, do_stitching=True
    ... )
    >>> 
    >>> # Save results and visualizations
    >>> save_outputs(amplitude, phase, results, str(config.output_dir))

Notes:
    This module is designed to be imported by top-level scripts and provides the primary
    interface for workflow execution. It handles the complexity of integrating multiple
    core modules while providing a simple, consistent API for complete workflow execution.
"""

import argparse
import math
import yaml
import os
import numpy as np
import tensorflow as tf
from ptycho import params as p
from ptycho import probe
from ptycho.loader import RawData, PtychoDataContainer
import logging
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Any, Tuple, Literal, get_origin, get_args
from pathlib import Path
from ptycho.config.config import TrainingConfig, ModelConfig, dataclass_to_legacy_dict
from dataclasses import fields
from ptycho import loader, probe
from typing import Union, Optional, Tuple, Dict, Any
from ptycho.raw_data import RawData
from ptycho.loader import PtychoDataContainer
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho import params
from ptycho.image import reassemble_patches
from ptycho.model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dataclasses import fields
from ptycho.config.config import ModelConfig, TrainingConfig


class DiffractionToObjectAdapter(tf.keras.Model):
    """
    Wrapper that keeps params.cfg['gridsize'] aligned with grouped inference inputs.

    Some exported bundles were trained with grouped data but load with lingering
    gridsize=1 in params.cfg, causing Translation to see B vs B*C tensors.
    By inspecting the diffraction input just before execution we can set the
    legacy gridsize to sqrt(channel_count) and avoid Translation crashes.
    """

    def __init__(self, base_model: tf.keras.Model):
        super().__init__(name=getattr(base_model, "name", "diffraction_to_obj"))
        self._model = base_model

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

        if p.cfg.get('gridsize') != gridsize:
            p.cfg['gridsize'] = gridsize

    def call(self, inputs, training=False, **kwargs):
        self._sync_gridsize(inputs)
        return self._model(inputs, training=training, **kwargs)

    def predict(self, *args, **kwargs):
        input_arg = args[0] if args else kwargs.get('x')
        self._sync_gridsize(input_arg)
        return self._model.predict(*args, **kwargs)

    def __getattr__(self, item):
        underlying = super().__getattribute__("_model")
        return getattr(underlying, item)


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
        models_dict = ModelManager.load_multiple_models(str(model_zip))
        
        # Get the diffraction_to_obj model which is needed for inference
        if 'diffraction_to_obj' not in models_dict:
            available_models = list(models_dict.keys())
            raise KeyError(
                f"No 'diffraction_to_obj' model found in saved models archive. "
                f"Available models: {available_models}. "
                f"The 'diffraction_to_obj' model should be created during training."
            )
        
        model = DiffractionToObjectAdapter(models_dict['diffraction_to_obj'])
        
        # ModelManager updates the global params.cfg when loading
        # Return a copy to avoid unintended modifications
        config = params.cfg.copy()
        
        logger.info(f"Successfully loaded model from {model_dir}")
        logger.debug(f"Model configuration: {config}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {str(e)}")
        raise

def update_config_from_dict(config_updates: dict):
    """
    Updates the application's configuration from a dictionary, ideal for notebook workflows.

    Args:
        config_updates (dict): A dictionary of parameters to update.
    """
    # 1. Create a mutable dictionary from the default dataclass values
    model_defaults = {f.name: f.default for f in fields(ModelConfig)}
    training_defaults = {f.name: f.default for f in fields(TrainingConfig) if f.name != 'model'}
    
    # Merge them
    full_config_dict = {**model_defaults, **training_defaults}

    # 2. Update with the user's dictionary
    for key, value in config_updates.items():
        if key in full_config_dict:
            full_config_dict[key] = value
        else:
            # Optionally warn about unused keys
            logger.warning(f"Configuration key '{key}' is not a recognized parameter.")

    # 3. Re-construct the dataclasses
    model_args = {k: v for k, v in full_config_dict.items() if k in model_defaults}
    training_args = {k: v for k, v in full_config_dict.items() if k in training_defaults}

    # Handle required Path objects if they are not set
    if training_args.get('train_data_file') is None:
        # Assign a dummy path or handle as an error if it's essential for all workflows
        training_args['train_data_file'] = Path("dummy_path.npz")

    final_model_config = ModelConfig(**model_args)
    final_training_config = TrainingConfig(model=final_model_config, **training_args)
    
    # 4. Update the legacy global params dictionary
    update_legacy_dict(params.cfg, final_training_config)
    
    logger.info("Configuration updated programmatically for interactive session.")
    params.print_params()

def load_data(file_path, n_images=None, n_subsample=None, flip_x=False, flip_y=False, swap_xy=False, n_samples=1, coord_scale=1.0, subsample_seed=None):
    """
    Load ptychography data from a file and return RawData objects.

    Args:
        file_path (str, optional): Path to the data file. Defaults to the package resource 'datasets/Run1084_recon3_postPC_shrunk_3.npz'.
        n_images (int, optional): Number of data points to include in the training set (legacy parameter). Defaults to 512.
        n_subsample (int, optional): Number of images to subsample from the dataset before grouping. 
                                     If None, uses n_images for backward compatibility.
        flip_x (bool, optional): If True, flip the sign of x coordinates. Defaults to False.
        flip_y (bool, optional): If True, flip the sign of y coordinates. Defaults to False.
        swap_xy (bool, optional): If True, swap x and y coordinates. Defaults to False.
        n_samples (int, optional): Number of samples to generate. Defaults to 1.
        coord_scale (float, optional): Scale factor for x and y coordinates. Defaults to 1.0.
        subsample_seed (int, optional): Random seed for reproducible subsampling. If None, uses random selection.

    Returns:
        RawData: RawData object containing the dataset.
    """
    logger.info(f"Loading data from {file_path} with n_images={n_images}, n_subsample={n_subsample}")
    # Load data from file
    data = np.load(file_path)

    # Extract required arrays from loaded data
    xcoords = data['xcoords']
    ycoords = data['ycoords']
    xcoords_start = data['xcoords_start']
    ycoords_start = data['ycoords_start']
    
    # Handle flexible diffraction key and shape
    diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
    diff_data = data[diff_key]
    
    # Only transpose if the data is in (H, W, N) format, not if it's already (N, H, W)
    if diff_data.shape[0] < diff_data.shape[2]:
        # Data is in (H, W, N) format, transpose to (N, H, W)
        diff3d = np.transpose(diff_data, [2, 0, 1])
    else:
        # Data is already in (N, H, W) format
        diff3d = diff_data
    
    probeGuess = data['probeGuess']
    objectGuess = data.get('objectGuess', None)
    
    # Optional ground-truth patches. Some NPZs (e.g., Phase C patched_*.npz)
    # may include a singleton 'Y' with shape (1, N, N, 1) rather than one
    # per image. Guard against shape mismatches by degrading to None unless
    # the first axis matches the dataset size. This keeps TensorFlow loader
    # behavior consistent (it will create a placeholder when Y is missing).
    Y_patches = data['Y'] if 'Y' in data else None

    # Apply coordinate transformations
    if flip_x:
        xcoords = -xcoords
        xcoords_start = -xcoords_start
        #probeGuess = probeGuess[::-1, :]
    if flip_y:
        ycoords = -ycoords
        ycoords_start = -ycoords_start
        #probeGuess = probeGuess[:, ::-1]
    if swap_xy:
        xcoords, ycoords = ycoords, xcoords
        xcoords_start, ycoords_start = ycoords_start, xcoords_start
        #probeGuess = np.transpose(probeGuess)

    # Apply coordinate scaling
    xcoords *= coord_scale
    ycoords *= coord_scale
    xcoords_start *= coord_scale
    ycoords_start *= coord_scale

    # Create scan_index array
    scan_index = np.zeros(diff3d.shape[0], dtype=int)

    # Implement independent subsampling logic
    dataset_size = xcoords.shape[0]

    # Validate optional Y shape before any indexing with selected_indices
    if Y_patches is not None:
        try:
            if getattr(Y_patches, 'shape', None) is None or Y_patches.shape[0] != dataset_size:
                # Shape mismatch (e.g., singleton); ignore Y to avoid index errors
                import logging
                logging.getLogger(__name__).warning(
                    "Ignoring NPZ 'Y' with incompatible shape %s (expected first axis %d)",
                    getattr(Y_patches, 'shape', None), dataset_size,
                )
                Y_patches = None
        except Exception:
            # Defensive: if anything goes wrong with Y inspection, null it out
            Y_patches = None
    
    # Determine how many images to use for subsampling
    if n_subsample is not None:
        # Independent control: n_subsample controls data selection
        images_to_use = min(n_subsample, dataset_size)
        logger.info(f"Independent sampling: subsampling {images_to_use} images from {dataset_size} total")
    elif n_images is not None:
        # Legacy behavior: n_images controls subsampling
        images_to_use = min(n_images, dataset_size)
        logger.info(f"Legacy sampling: using {images_to_use} images from {dataset_size} total")
    else:
        # Default: use all data
        images_to_use = dataset_size
        logger.info(f"Using full dataset of {dataset_size} images")
    
    # Perform subsampling if needed
    if images_to_use < dataset_size:
        if subsample_seed is not None:
            # Reproducible subsampling with seed
            np.random.seed(subsample_seed)
            logger.info(f"Using seed {subsample_seed} for reproducible subsampling")
        
        # Random subsampling
        all_indices = np.arange(dataset_size)
        selected_indices = np.random.choice(all_indices, size=images_to_use, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort for consistency
        logger.info(f"Randomly subsampled {images_to_use} images")
    else:
        # Use all data
        selected_indices = np.arange(dataset_size)
    
    # Create RawData object with subsampled data
    ptycho_data = RawData(xcoords[selected_indices], ycoords[selected_indices],
                          xcoords_start[selected_indices], ycoords_start[selected_indices],
                          diff3d[selected_indices], probeGuess,
                          scan_index[selected_indices], objectGuess=objectGuess,
                          # Pass Y only when it is per-image and shape-validated
                          Y=(Y_patches[selected_indices] if Y_patches is not None else None))

    # Persist selected indices for reproducibility
    ptycho_data.sample_indices = np.array(selected_indices, copy=True)
    ptycho_data.subsample_seed = subsample_seed
    if subsample_seed is not None:
        try:
            tmp_dir = Path('tmp')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            indices_path = tmp_dir / f"subsample_seed{subsample_seed}_indices.txt"
            with indices_path.open('w', encoding='utf-8') as handle:
                for idx in ptycho_data.sample_indices:
                    handle.write(f"{int(idx)}\n")
            logger.info("Persisted subsample indices to %s", indices_path)
        except Exception as exc:
            logger.warning("Failed to persist subsample indices for seed %s: %s", subsample_seed, exc)

    return ptycho_data

def parse_arguments():
    """Parse command-line arguments based on TrainingConfig fields."""
    from ptycho.cli_args import add_logging_arguments
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--do_stitching", action='store_true', default=False,
                        help="Perform image stitching after training (default: False)")
    
    # Add logging arguments
    add_logging_arguments(parser)
    
    # Add arguments based on TrainingConfig fields
    for field in fields(TrainingConfig):
        if field.name == 'model':
            # Handle ModelConfig fields
            for model_field in fields(ModelConfig):
                # Special handling for Literal types
                if hasattr(model_field.type, "__origin__") and model_field.type.__origin__ is Literal:
                    choices = list(model_field.type.__args__)
                    parser.add_argument(
                        f"--{model_field.name}",
                        type=str,
                        choices=choices,
                        default=model_field.default,
                        help=f"Model parameter: {model_field.name}, choices: {choices}"
                    )
                else:
                    parser.add_argument(
                        f"--{model_field.name}",
                        type=model_field.type,
                        default=model_field.default,
                        help=f"Model parameter: {model_field.name}"
                    )
        else:
            # Handle path fields specially
            if field.type == Path or str(field.type).startswith("typing.Optional[pathlib.Path"):
                logger.debug(f"Field: {field.name}")
                logger.debug(f"Field type: {field.type}")
                logger.debug(f"Field default: {field.default}")
                parser.add_argument(
                    f"--{field.name}",
                    type=lambda x: (logger.debug(f"Converting path value: {x}"), Path(x) if x is not None else None)[1],
                    default=None if field.default == None else str(field.default),
                    help=f"Path for {field.name}"
                )
            else:
                # Special handling for specific parameters to provide better help text
                if field.name == 'n_groups':
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type if get_origin(field.type) is not Union else get_args(field.type)[0],
                        default=field.default,
                        help="Number of groups to generate. Always means groups regardless of gridsize. "
                             "Can exceed dataset size when using higher --neighbor_count values."
                    )
                elif field.name == 'n_images':
                    # Keep n_images for backward compatibility but mark as deprecated
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type if get_origin(field.type) is not Union else get_args(field.type)[0],
                        default=field.default,
                        help="DEPRECATED: Use --n_groups instead. Number of groups to use from the dataset."
                    )
                elif field.name == 'n_subsample':
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type if get_origin(field.type) is not Union else get_args(field.type)[0],
                        default=field.default,
                        help="Number of images to subsample from dataset before grouping (independent control). "
                             "When provided, controls data selection separately from grouping."
                    )
                elif field.name == 'subsample_seed':
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type if get_origin(field.type) is not Union else get_args(field.type)[0],
                        default=field.default,
                        help="Random seed for reproducible subsampling. "
                             "Use same seed across runs to ensure consistent data selection."
                    )
                elif field.name == 'neighbor_count':
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type,
                        default=field.default,
                        help="Number of nearest neighbors (K) for grouping. Use higher values (e.g., 7) "
                             "to enable more combinations when requesting more groups than available points."
                    )
                elif field.name == 'backend':
                    # Special handling for backend Literal type
                    if hasattr(field.type, "__origin__") and field.type.__origin__ is Literal:
                        choices = list(field.type.__args__)
                        parser.add_argument(
                            f"--{field.name}",
                            type=str,
                            choices=choices,
                            default=field.default,
                            help=f"Backend selection: {', '.join(choices)} (default: {field.default})"
                        )
                    else:
                        # Fallback if not a Literal type
                        parser.add_argument(
                            f"--{field.name}",
                            type=str,
                            default=field.default,
                            help="Backend selection for workflow orchestration"
                        )
                else:
                    # Handle Optional types
                    if get_origin(field.type) is Union:
                        # This is an Optional type (Union[T, None])
                        args = get_args(field.type)
                        actual_type = args[0] if args[0] is not type(None) else args[1]
                        parser.add_argument(
                            f"--{field.name}",
                            type=actual_type,
                            default=field.default,
                            help=f"Training parameter: {field.name}"
                        )
                    else:
                        parser.add_argument(
                            f"--{field.name}",
                            type=field.type,
                            default=field.default,
                            help=f"Training parameter: {field.name}"
                        )
    
    return parser.parse_args()

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except (yaml.YAMLError, IOError) as e:
        logger.error(f"Error loading YAML config: {e}")
        raise


#def validate_config(config: Dict[str, Any]) -> None:
#    """Validate the configuration."""
#    if 'train_data_file_path' not in config or config['train_data_file_path'] is None:
#        raise ValueError("train_data_file_path is a required parameter and must be provided")

def setup_configuration(args: argparse.Namespace, yaml_path: Optional[str]) -> TrainingConfig:
    """Set up the configuration by merging defaults, YAML file, and command-line arguments."""
    try:
        yaml_config = load_yaml_config(yaml_path) if yaml_path else {}
        args_config = vars(args)
        
        # Start with YAML config as base (if provided)
        merged_config = yaml_config.copy() if yaml_config else {}
        
        # Override with CLI arguments (CLI takes precedence over YAML)
        for key, value in args_config.items():
            if value is not None:  # Only override if CLI arg was explicitly provided
                merged_config[key] = value
        
        # Convert string paths to Path objects
        for key in ['train_data_file', 'test_data_file', 'output_dir']:
            if key in merged_config and merged_config[key] is not None:
                merged_config[key] = Path(merged_config[key])
        
        # Handle nested 'model' config from YAML
        model_config_dict = {}
        if 'model' in merged_config and isinstance(merged_config['model'], dict):
            model_config_dict = merged_config['model']
        
        # Create ModelConfig from merged values
        model_fields = {f.name for f in fields(ModelConfig)}
        model_args = {k: v for k, v in merged_config.items() if k in model_fields}
        # Override with values from nested model config if present
        model_args.update({k: v for k, v in model_config_dict.items() if k in model_fields})
        model_config = ModelConfig(**model_args)
        
        # Create TrainingConfig
        training_fields = {f.name for f in fields(TrainingConfig)}
        training_args = {k: v for k, v in merged_config.items() 
                        if k in training_fields and k != 'model'}
        
        # Log the configuration sources for debugging
        logger.debug(f"YAML config: {yaml_config}")
        logger.debug(f"CLI arguments: {args_config}")
        logger.debug(f"Merged config: {merged_config}")
        
        config = TrainingConfig(model=model_config, **training_args)
        
        # Update the global configuration
        update_legacy_dict(params.cfg, config)
        
        logger.info("Configuration setup complete")
        logger.info(f"Final configuration: {config}")
        
        return config
    except (yaml.YAMLError, IOError, ValueError) as e:
        logger.error(f"Error setting up configuration: {e}")
        raise

def load_and_prepare_data(data_file_path: str) -> Tuple[RawData, RawData, Any]:
    """
    Load and prepare the data from a single file path.

    Args:
        data_file_path (str): Path to the data file

    Returns:
        Tuple[RawData, RawData, Any]: A tuple containing the full dataset, training subset, and additional data
    """
    # TODO deprecated
    from ptycho.loader import load_xpp_npz
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    try:
        return load_xpp_npz(data_file_path)
    except Exception as e:
        logger.error(f"Error loading data from {data_file_path}: {str(e)}")
        raise

from typing import Union
from ptycho.loader import RawData, PtychoDataContainer

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
        # Use config.n_groups for nsamples - this is the interpreted value from the training script
        dataset = data.generate_grouped_data(
            config.model.N,
            K=config.neighbor_count,  # Use configurable K value
            nsamples=config.n_groups,  # Use n_groups (clearer naming)
            dataset_path=str(config.train_data_file) if config.train_data_file else None,
            sequential_sampling=config.sequential_sampling,  # Pass sequential sampling flag
            gridsize=config.model.gridsize,  # Pass gridsize explicitly (replaces global params dependency)
            enable_oversampling=config.enable_oversampling,  # Explicit opt-in for K choose C oversampling
            neighbor_pool_size=config.neighbor_pool_size  # Pool size for oversampling (if None, defaults to neighbor_count)
        )
        return loader.load(lambda: dataset, data.probeGuess, which=None, create_split=False)
    else:
        raise TypeError("data must be either RawData or PtychoDataContainer")

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

#    # Calculate intensity scale
#    intensity_scale = train_pinn.calculate_intensity_scale(train_container)

    # Train the model
    results = train_pinn.train_eval(PtychoDataset(train_container, test_container))

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

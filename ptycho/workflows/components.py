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
import logging
import math
import os
from dataclasses import fields, replace
from pathlib import Path
from types import UnionType
from typing import Any, Dict, Literal, Optional, Tuple, Union, get_args, get_origin

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from ptycho import loader, params, probe
from ptycho.config import resolve_training_config
from ptycho.config.config import (
    ModelConfig,
    TrainingConfig,
    resolve_model_object_policy,
    update_legacy_dict,
)
from ptycho.config.legacy_state import (
    configured_legacy_params,
    isolated_archived_params_scope,
    legacy_params_scope,
    scoped_legacy_params,
    transactional_legacy_params,
)
from ptycho.generators.registry import resolve_generator
from ptycho.loader import PtychoDataContainer, RawData
from ptycho.model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        models_dict = ModelManager.load_multiple_models(str(model_zip))
        
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


@configured_legacy_params
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
    xcoords_start = data['xcoords_start'] if 'xcoords_start' in data else xcoords.copy()
    ycoords_start = data['ycoords_start'] if 'ycoords_start' in data else ycoords.copy()
    
    # Handle flexible diffraction key and shape
    diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
    diff_data = data[diff_key]

    if diff_data.ndim == 4 and diff_data.shape[-1] == 1:
        diff_data = np.squeeze(diff_data, axis=-1)
    if diff_data.ndim != 3:
        raise ValueError(
            f"Expected diffraction data to have rank 3 or rank 4 with singleton channel, got {diff_data.shape}"
        )

    dataset_size = int(xcoords.shape[0])
    # Prefer coordinate-length matching over shape heuristics:
    # canonical format is (N_scans, H, W); legacy format is (H, W, N_scans).
    if diff_data.shape[0] == dataset_size:
        diff3d = diff_data
    elif diff_data.shape[-1] == dataset_size:
        diff3d = np.transpose(diff_data, [2, 0, 1])
    else:
        raise ValueError(
            f"Unable to align diffraction shape {diff_data.shape} with xcoords length {dataset_size}."
        )
    
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


PUBLIC_TRAINING_INPUT_NAMES = frozenset(
    item.name for item in fields(ModelConfig)
) | frozenset(
    item.name for item in fields(TrainingConfig) if item.name != "model"
)


def _unwrap_optional_type(annotation):
    if get_origin(annotation) not in (Union, UnionType):
        return annotation
    value_types = tuple(
        item for item in get_args(annotation) if item is not type(None)
    )
    if len(value_types) != 1:
        return annotation
    return value_types[0]


def _literal_argument_type(choices):
    choice_types = {type(choice) for choice in choices}
    if len(choice_types) != 1:
        raise TypeError(
            "public CLI Literal choices must use one primitive type"
        )
    return next(iter(choice_types))


def _public_training_argument_help(name: str, *, model_field: bool) -> str:
    if name == "n_groups":
        return (
            "Number of groups to generate. Always means groups regardless "
            "of gridsize. Can exceed dataset size when using higher "
            "--neighbor_count values."
        )
    if name == "n_images":
        return (
            "DEPRECATED: Use --n_groups instead. Number of groups to use "
            "from the dataset."
        )
    if name == "n_subsample":
        return (
            "Number of images to subsample from dataset before grouping "
            "(independent control). When provided, controls data selection "
            "separately from grouping."
        )
    if name == "subsample_seed":
        return (
            "Random seed for reproducible subsampling. Use same seed across "
            "runs to ensure consistent data selection."
        )
    if name == "neighbor_count":
        return (
            "Number of nearest neighbors (K) for grouping. Use higher "
            "values (e.g., 7) to enable more combinations when requesting "
            "more groups than available points."
        )
    if name == "backend":
        return (
            "Backend selection: tensorflow, pytorch (default: tensorflow). "
            "PyTorch backend requires torch>=2.2 (POLICY-001)."
        )
    prefix = "Model" if model_field else "Training"
    return f"{prefix} parameter: {name}"


def _add_public_training_argument(
    parser: argparse.ArgumentParser,
    config_field,
    *,
    model_field: bool,
) -> None:
    value_type = _unwrap_optional_type(config_field.type)
    options = {
        "default": argparse.SUPPRESS,
        "help": _public_training_argument_help(
            config_field.name,
            model_field=model_field,
        ),
    }

    if get_origin(value_type) is Literal:
        choices = list(get_args(value_type))
        options["choices"] = choices
        options["type"] = _literal_argument_type(choices)
    elif value_type is bool:
        options["action"] = argparse.BooleanOptionalAction
    else:
        options["type"] = value_type

    parser.add_argument(f"--{config_field.name}", **options)


def add_public_training_config_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add public training overrides without applying dataclass defaults."""
    for model_field in fields(ModelConfig):
        _add_public_training_argument(
            parser,
            model_field,
            model_field=True,
        )
    for training_field in fields(TrainingConfig):
        if training_field.name == "model":
            continue
        _add_public_training_argument(
            parser,
            training_field,
            model_field=False,
        )
    return parser


def parse_arguments():
    """Parse command-line arguments based on TrainingConfig fields."""
    from ptycho.cli_args import add_logging_arguments

    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--do_stitching", action='store_true', default=False,
                        help="Perform image stitching after training (default: False)")

    add_logging_arguments(parser)
    add_public_training_config_arguments(parser)
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
        cli_patch = {
            name: value
            for name, value in vars(args).items()
            if name in PUBLIC_TRAINING_INPUT_NAMES
        }
        config = resolve_training_config(yaml_config, cli_patch)

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
    generator = resolve_generator(config)
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

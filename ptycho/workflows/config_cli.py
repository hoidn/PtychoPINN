"""Configuration and CLI helpers for the TensorFlow workflows.

Owns the public-training argument parser, YAML/config resolution, and the
notebook ``update_config_from_dict`` bridge, plus the NPZ ``load_data`` entry.
"""
import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import fields
from types import UnionType
from typing import Annotated, Any, Dict, Literal, Optional, Union, get_args, get_origin

import numpy as np
import yaml

from ptycho.config import resolve_training_config
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.raw_data import RawData
from ptycho.acquisition import (
    decode_acquisition,
    select_acquisition,
    transform_coordinates,
)

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho.workflows.components')

def update_config_from_dict(config_updates: dict):
    """
    Updates the application's configuration from a dictionary, ideal for notebook workflows.

    Args:
        config_updates (dict): A dictionary of parameters to update.
    """
    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import configured_params_scope

    with configured_params_scope():
        model_defaults = {f.name: f.default for f in fields(ModelConfig)}
        training_defaults = {
            f.name: f.default
            for f in fields(TrainingConfig)
            if f.name != "model"
        }
        full_config_dict = {**model_defaults, **training_defaults}
        for key, value in config_updates.items():
            if key in full_config_dict:
                full_config_dict[key] = value
            else:
                logger.warning(
                    "Configuration key %r is not a recognized parameter.", key
                )
        model_args = {
            key: value
            for key, value in full_config_dict.items()
            if key in model_defaults
        }
        training_args = {
            key: value
            for key, value in full_config_dict.items()
            if key in training_defaults
        }
        if training_args.get("train_data_file") is None:
            training_args["train_data_file"] = Path("dummy_path.npz")
        final_training_config = TrainingConfig(
            model=ModelConfig(**model_args),
            **training_args,
        )
        update_legacy_dict(params.cfg, final_training_config)
        logger.info("Configuration updated programmatically for interactive session.")
        params.print_params()

def load_data(file_path, n_images=None, n_subsample=None, flip_x=False, flip_y=False, swap_xy=False, n_samples=1, coord_scale=1.0, subsample_seed=None, *, rng: Optional[np.random.Generator] = None):
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
        rng (np.random.Generator, optional): Caller-owned random generator. Mutually exclusive
                                             with subsample_seed.

    Returns:
        RawData: RawData object containing the dataset.
    """
    if subsample_seed is not None and rng is not None:
        raise ValueError("subsample_seed and rng are mutually exclusive")

    logger.info(f"Loading data from {file_path} with n_images={n_images}, n_subsample={n_subsample}")
    record = transform_coordinates(
        decode_acquisition(file_path, truth_policy="drop_incompatible"),
        flip_x=flip_x,
        flip_y=flip_y,
        swap_xy=swap_xy,
        scale=coord_scale,
    )
    dataset_size = len(record.xcoords)
    
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
    
    selection = select_acquisition(
        record,
        count=images_to_use,
        seed=subsample_seed,
        rng=rng,
    )
    selected_indices = selection.source_indices
    if selection.mode == "random_without_replacement":
        logger.info(f"Randomly subsampled {images_to_use} images")

    # Create RawData object with subsampled data
    ptycho_data = RawData(
        record.xcoords[selected_indices],
        record.ycoords[selected_indices],
        record.xcoords_start[selected_indices],
        record.ycoords_start[selected_indices],
        record.diff3d[selected_indices],
        record.probeGuess,
        record.scan_index[selected_indices],
        objectGuess=record.objectGuess,
        Y=(record.Y[selected_indices] if record.Y is not None else None),
        metadata=record.metadata,
        object_index=record.object_index[selected_indices],
        probe_simulated=record.probe_simulated,
        object_amplitude_scale=record.object_amplitude_scale,
        label=(record.label[selected_indices] if record.label is not None else None),
        scale_contract_version=record.scale_contract_version,
        measurement_domain=record.measurement_domain,
        experiment_id=record.experiment_id,
    )

    # Persist selected indices for reproducibility
    ptycho_data.sample_indices = np.array(selected_indices, copy=True)
    ptycho_data.subsample_seed = subsample_seed

    return ptycho_data


PUBLIC_TRAINING_INPUT_NAMES = frozenset(
    item.name for item in fields(ModelConfig)
) | frozenset(
    item.name for item in fields(TrainingConfig) if item.name != "model"
)

# Parse-time deprecated CLI flag aliases: old flag -> canonical config field.
_TRAINING_CLI_FLAG_ALIASES = {
    "n_groups": "training_groups",
    "n_subsample": "train_raw_selection",
}


def _unwrap_public_cli_type(annotation):
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
            continue
        if origin not in (Union, UnionType):
            return annotation
        value_types = tuple(
            item for item in get_args(annotation) if item is not type(None)
        )
        if len(value_types) == 1:
            annotation = value_types[0]
            continue
        primitive_types = {
            _unwrap_public_cli_type(item) for item in value_types
        }
        if primitive_types == {int, float}:
            return float
        return annotation


def _literal_argument_type(choices):
    choice_types = {type(choice) for choice in choices}
    if len(choice_types) != 1:
        raise TypeError(
            "public CLI Literal choices must use one primitive type"
        )
    return next(iter(choice_types))


def _public_training_argument_help(name: str, *, model_field: bool) -> str:
    if name == "training_groups":
        return (
            "Exact number of groups to generate; also the number of unique "
            "centers. Cannot exceed the selected candidate-row count."
        )
    if name == "n_images":
        return (
            "DEPRECATED: Use --training_groups instead. Number of groups to use "
            "from the dataset."
        )
    if name == "train_raw_selection":
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
            "Nearest non-center candidate count K for each group; must be at "
            "least gridsize² - 1."
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
    value_type = _unwrap_public_cli_type(config_field.type)
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
    for alias, target in _TRAINING_CLI_FLAG_ALIASES.items():
        parser.add_argument(
            f"--{alias}",
            dest=alias,
            type=int,
            default=argparse.SUPPRESS,
            help=(
                f"DEPRECATED: Use --{target} instead. "
                f"Legacy alias for the {target} field."
            ),
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
        selected_backend = cli_patch.get("backend")
        if selected_backend is None and isinstance(yaml_config, dict):
            selected_backend = yaml_config.get("backend")
        if selected_backend == "pytorch" and isinstance(yaml_config, dict):
            yaml_config = dict(yaml_config)
            yaml_config.pop("scale_contract_version", None)
            yaml_config.pop("measurement_domain", None)
        for alias, target in _TRAINING_CLI_FLAG_ALIASES.items():
            value = getattr(args, alias, None)
            if value is not None:
                warnings.warn(
                    f"--{alias} is deprecated; use --{target}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if target in cli_patch and cli_patch[target] != value:
                    raise ValueError(
                        f"--{alias} conflicts with explicit --{target} "
                        f"({value!r} vs {cli_patch[target]!r})"
                    )
                cli_patch[target] = value
        config = resolve_training_config(yaml_config, cli_patch)

        logger.info("Configuration setup complete")
        logger.info(f"Final configuration: {config}")
        
        return config
    except (yaml.YAMLError, IOError, ValueError) as e:
        logger.error(f"Error setting up configuration: {e}")
        raise

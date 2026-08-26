"""Configuration and CLI helpers for the TensorFlow workflows.

Owns the public-training argument parser, YAML/config resolution, the notebook
``update_config_from_dict`` bridge, and the NPZ ``load_data`` entry.
"""
import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import yaml

from ptycho import params
from ptycho.config import resolve_training_config
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho.config.legacy_state import configured_legacy_params
from ptycho.loader import RawData
from ptycho.acquisition import (
    decode_acquisition,
    select_acquisition,
    transform_coordinates,
)
from pydantic import TypeAdapter, ValidationError
from pydantic_settings.sources.providers.cli import CliSettingsSource

# Preserves pre-split log provenance.
logger = logging.getLogger("ptycho.workflows.components")

@configured_legacy_params
def update_config_from_dict(config_updates: dict):
    """
    Update global config from a nested dict, ideal for notebook workflows.

    config_updates must use the nested TrainingConfig structure, e.g.:
        {'model': {'N': 64}, 'nepochs': 100, 'sampling': {'training_groups': 512}}
    """
    config = resolve_training_config(config_updates, None)
    update_legacy_dict(params.cfg, config)
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


def add_public_training_config_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register TrainingConfig CLI arguments on an existing argparse parser.

    Arguments are auto-derived from the TrainingConfig model structure.
    Nested sub-config fields use dotted names, e.g. --sampling.training_groups.
    """
    CliSettingsSource(TrainingConfig, root_parser=parser, cli_parse_args=False)
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


def _get_field_annotation(cls: Any, name: str) -> Any:
    if hasattr(cls, "model_fields") and name in cls.model_fields:
        return cls.model_fields[name].annotation
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if field.name == name:
                return field.type
    return None


def _resolve_training_field_annotation(path: list[str]) -> Any:
    annotation: Any = TrainingConfig
    for part in path:
        annotation = _get_field_annotation(annotation, part)
        if annotation is None:
            break
    return annotation


def _convert_cli_value(value: Any, path: list[str]) -> Any:
    """Decode a CLI scalar only when it satisfies the destination annotation."""
    if not isinstance(value, str):
        return value
    if value.lower() in {"true", "false"}:
        decoded = value.lower() == "true"
    else:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return TypeAdapter(
            _resolve_training_field_annotation(path)
        ).validate_python(decoded)
    except ValidationError:
        return value


def _convert_cli_patch(node: Any, path: list[str]) -> Any:
    if isinstance(node, dict):
        return {
            key: _convert_cli_patch(value, [*path, key])
            for key, value in node.items()
        }
    return _convert_cli_value(node, path)


def _namespace_to_training_patch(args: argparse.Namespace) -> dict[str, Any]:
    """Extract a nested, typed TrainingConfig patch from parsed CLI arguments."""
    source = CliSettingsSource(TrainingConfig, cli_parse_args=False)
    source(parsed_args=args)
    return _convert_cli_patch(source(), [])


def setup_configuration(args: argparse.Namespace, yaml_path: Optional[str]) -> TrainingConfig:
    """Set up the configuration by merging defaults, YAML file, and command-line arguments."""
    try:
        yaml_config = load_yaml_config(yaml_path) if yaml_path else {}
        cli_patch = _namespace_to_training_patch(args)
        config = resolve_training_config(yaml_config, cli_patch)

        logger.info("Configuration setup complete")
        logger.info(f"Final configuration: {config}")

        return config
    except (yaml.YAMLError, IOError, ValueError) as e:
        logger.error(f"Error setting up configuration: {e}")
        raise

#!/usr/bin/env python

import logging
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set up file handler for debug logging
file_handler = logging.FileHandler('train_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up console handler for info logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)

from ptycho.workflows.components import (
    add_public_training_config_arguments,
    setup_configuration,
    load_data,
    save_outputs,
    logger
)
from ptycho.workflows.backend_selector import run_cdi_example_with_backend
from ptycho.config import (
    validate_runnable_training_config,
    validate_training_config_structure,
)
from ptycho.config.config import TrainingConfig
from ptycho import model_manager
import argparse


def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:
    """
    Interpret --n-images parameter based on gridsize.
    
    For gridsize=1: n_images refers to individual images (traditional behavior)
    For gridsize>1: n_images refers to number of neighbor groups
    
    Args:
        n_images: User-specified number from --n-images
        gridsize: Current gridsize setting
        
    Returns:
        tuple: (actual_n_images, interpretation_message)
    """
    if gridsize == 1:
        message = f"Parameter interpretation: --n-images={n_images} refers to individual images (gridsize=1)"
        return n_images, message
    else:
        total_patterns = n_images * gridsize * gridsize
        message = f"Parameter interpretation: --n-images={n_images} refers to neighbor groups (gridsize={gridsize}, total patterns={total_patterns})"
        return n_images, message

def interpret_sampling_parameters(config: TrainingConfig):
    """
    Interpret sampling parameters with support for independent control and oversampling.

    Priority:
    1. If n_subsample is specified: use it for data subsampling
    2. Otherwise: use n_groups for legacy behavior

    Args:
        config: Training configuration with sampling parameters

    Returns:
        tuple: (n_subsample, n_groups, enable_oversampling, neighbor_pool_size, interpretation_message)
    """
    gridsize = config.model.gridsize
    sampling = config.sampling
    enable_oversampling = sampling.enable_oversampling
    neighbor_pool_size = sampling.neighbor_pool_size

    # Case 1: Independent control with n_subsample
    if sampling.n_subsample is not None:
        n_subsample = sampling.n_subsample
        n_groups = sampling.n_groups

        if gridsize == 1:
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"using {n_groups} groups for training")
        else:
            total_from_groups = n_groups * gridsize * gridsize
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"creating {n_groups} groups (approx {total_from_groups} patterns from groups)")
            if enable_oversampling:
                K = neighbor_pool_size if neighbor_pool_size is not None else sampling.neighbor_count
                message += f" [Oversampling enabled: K={K}]"

        return n_subsample, n_groups, enable_oversampling, neighbor_pool_size, message

    # Case 2: Legacy behavior - n_groups controls both
    else:
        # For backward compatibility, n_groups controls subsampling
        if gridsize == 1:
            n_subsample = sampling.n_groups
            n_groups = sampling.n_groups
            message = f"Legacy mode: using {n_groups} groups (gridsize=1)"
        else:
            # For gridsize > 1, we need to subsample enough to create the groups
            n_subsample = sampling.n_groups  # This will be interpreted as groups by generate_grouped_data
            n_groups = sampling.n_groups
            total_patterns = n_groups * gridsize * gridsize
            message = (f"Legacy mode: --n-groups={n_groups} refers to neighbor groups "
                      f"(gridsize={gridsize}, approx {total_patterns} patterns)")
            if enable_oversampling:
                K = neighbor_pool_size if neighbor_pool_size is not None else sampling.neighbor_count
                message += f" [Oversampling enabled: K={K}]"

        return n_subsample, n_groups, enable_oversampling, neighbor_pool_size, message

def parse_arguments():
    """
    Extend the public parser with Torch runtime and optimizer flags.

    Runtime flags form an ExecutionRequest. Explicit optimizer flags form a
    canonical Torch TrainingConfig patch.
    See docs/workflows/pytorch.md §12 for flag descriptions.
    """
    from ptycho.cli_args import add_logging_arguments

    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--do_stitching", action='store_true', default=False,
                        help="Perform image stitching after training (default: False)")

    # Add logging arguments
    add_logging_arguments(parser)
    add_public_training_config_arguments(parser)

    # PyTorch-only runtime and optimizer flags (see docs/workflows/pytorch.md §12)
    parser.add_argument("--torch-accelerator", type=str,
                       choices=['auto', 'cpu', 'cuda', 'gpu', 'mps', 'tpu'],
                       default='cuda',
                       help="PyTorch accelerator for training (only applies when --backend pytorch). "
                            "Options: 'cuda' (default GPU baseline per POLICY-001), 'auto' (auto-detect with CUDA preference), "
                            "'cpu' (fallback), 'gpu', 'mps', 'tpu'. "
                            "Override with '--torch-accelerator cpu' for CPU-only runs. "
                            "See docs/workflows/pytorch.md §12 for details.")
    parser.add_argument("--torch-deterministic", action='store_true',
                       default=True,
                       help="Enable deterministic training mode for reproducibility (default: True). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-num-workers", type=int, default=0,
                       help="Number of dataloader worker processes for PyTorch training (default: 0). "
                            "Set to 0 for main process only (CPU-safe, deterministic). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-accumulate-grad-batches", type=int, default=1,
                       help="Accumulate gradients over N batches for larger effective batch size (default: 1). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-learning-rate", type=float, default=None,
                       help="Learning rate for PyTorch training (default: None, uses model default). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-scheduler", type=str, default='Default',
                       choices=['Default', 'Exponential', 'WarmupCosine', 'ReduceLROnPlateau'],
                       help="Learning rate scheduler for PyTorch training (default: 'Default'). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-plateau-factor", type=float, default=None,
                       help="ReduceLROnPlateau factor (only applies when --backend pytorch).")
    parser.add_argument("--torch-plateau-patience", type=int, default=None,
                       help="ReduceLROnPlateau patience (only applies when --backend pytorch).")
    parser.add_argument("--torch-plateau-min-lr", type=float, default=None,
                       help="ReduceLROnPlateau min lr (only applies when --backend pytorch).")
    parser.add_argument("--torch-plateau-threshold", type=float, default=None,
                       help="ReduceLROnPlateau threshold (only applies when --backend pytorch).")
    parser.add_argument("--torch-logger", type=str, default='csv',
                       choices=['csv', 'tensorboard', 'mlflow', 'none'],
                       help="Logger backend for PyTorch training (default: 'csv'). "
                            "Options: 'csv' (zero deps), 'tensorboard', 'mlflow' (requires server), 'none' (disable). "
                            "See CONFIG-LOGGER-001. Only applies when --backend pytorch.")
    parser.add_argument("--torch-recon-log-every-n-epochs", type=int, default=None,
                       help="Log intermediate reconstructions every N epochs (default: disabled). "
                            "Only applies when --torch-logger mlflow.")
    parser.add_argument("--torch-recon-log-num-patches", type=int, default=4,
                       help="Number of fixed patch indices to log (default: 4).")
    parser.add_argument("--torch-recon-log-fixed-indices", type=int, nargs='+', default=None,
                       help="Explicit patch indices to log (default: auto-select).")
    parser.add_argument("--torch-recon-log-stitch", action='store_true', default=False,
                       help="Log stitched full-resolution reconstructions (default: disabled).")
    parser.add_argument("--torch-recon-log-max-stitch-samples", type=int, default=None,
                       help="Cap on number of samples for stitched logging (default: no limit).")
    parser.add_argument("--torch-enable-checkpointing", action='store_true',
                       default=True,
                       help="Enable checkpoint saving during training (default: True). "
                            "Only applies when --backend pytorch.")
    parser.add_argument("--torch-checkpoint-save-top-k", type=int, default=1,
                       help="Save top K checkpoints (default: 1). "
                            "Only applies when --backend pytorch.")

    return parser.parse_args()


def _save_tensorflow_model_legacy(config: TrainingConfig) -> None:
    """Persist the TensorFlow bundle under its validated legacy projection.

    Remove this adapter when ``ptycho.model_manager.save`` accepts the archive
    configuration and model path explicitly.
    """
    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import (
        configured_params_scope,
        legacy_params_scope,
    )

    with legacy_params_scope():
        with configured_params_scope():
            update_legacy_dict(params.cfg, config)
            model_manager.save(str(config.output_dir))


def main() -> None:
    """Main function to orchestrate the CDI example script execution."""
    raw_argv = tuple(sys.argv[1:])
    args = parse_arguments()
    
    # Handle legacy argument name
    if hasattr(args, 'train_data_file_path'):
        args.train_data_file = args.train_data_file_path
        delattr(args, 'train_data_file_path')

    config = setup_configuration(args, args.config)
    
    # Resolve dataset-authored photon scale before validation, sampling,
    # projection, or consumption of the training data.
    from ptycho.metadata import MetadataManager

    try:
        _, metadata = MetadataManager.load_with_metadata(
            str(config.data.train_data_file)
        )
    except Exception as error:
        logger.debug(f"No metadata found or error reading metadata: {error}")
        metadata = None

    metadata_nphotons = None
    if metadata:
        if "nphotons" in metadata:
            metadata_nphotons = float(metadata["nphotons"])
        elif (
            "physics_parameters" in metadata
            and "nphotons" in metadata["physics_parameters"]
        ):
            metadata_nphotons = float(
                metadata["physics_parameters"]["nphotons"]
            )

    if metadata_nphotons is not None:
        original_nphotons = config.data.nphotons
        config = replace(
            config,
            data=replace(config.data, nphotons=metadata_nphotons),
        )
        logger.info(
            "Overriding nphotons from config "
            f"({original_nphotons:.1e}) with value from dataset metadata: "
            f"{metadata_nphotons:.1e}"
        )

    validate_training_config_structure(config)
    validate_runnable_training_config(config)

    # Interpret sampling only after the final metadata-derived record validates.
    (
        n_subsample,
        n_groups,
        enable_oversampling,
        neighbor_pool_size,
        interpretation_message,
    ) = interpret_sampling_parameters(config)
    logger.info(interpretation_message)

    if config.sampling.n_subsample is not None and config.model.gridsize > 1:
        min_required = n_groups * config.model.gridsize * config.model.gridsize
        if n_subsample < min_required:
            logger.warning(
                f"n_subsample ({n_subsample}) may be too small to create "
                f"{n_groups} groups of size {config.model.gridsize}². "
                f"Consider increasing n_subsample to at least {min_required}"
            )

    try:
        logger.info(f"Starting training with n_subsample={n_subsample}, n_groups={n_groups}, "
                   f"stitching={'enabled' if args.do_stitching else 'disabled'}")

        # Load data with new independent sampling parameters
        # Note: load_data still uses n_images parameter name internally
        ptycho_data = load_data(
            str(config.data.train_data_file),
            n_images=n_groups,  # Pass n_groups as n_images to maintain API compatibility
            n_subsample=n_subsample,
            subsample_seed=config.sampling.subsample_seed,
        )
        
        test_data = None
        if config.data.test_data_file:
            test_data = load_data(str(config.data.test_data_file))
            logger.info(f"Loaded test data from {config.data.test_data_file}")

        # Build a provenance-carrying request if the selected backend is PyTorch.
        torch_execution_request = None
        torch_factory_overrides = None
        if config.backend == 'pytorch':
            from ptycho_torch.cli.shared import (
                build_execution_request_from_args,
                build_training_config_patch_from_args,
            )

            torch_execution_request = build_execution_request_from_args(
                args,
                mode='training',
                explicit_options=raw_argv,
                lane='unified-training',
            )
            torch_factory_overrides = build_training_config_patch_from_args(
                args,
                explicit_options=raw_argv,
                lane='unified-training',
            )
            if not torch_execution_request.explicit_fields:
                logger.info("POLICY-001: No --torch-* execution flags provided. "
                           "Backend will use GPU-first defaults (auto-detects CUDA if available, else CPU). "
                           "CPU-only users should pass --torch-accelerator cpu.")

        # The shared selector owns the required CONFIG-001 projection
        # immediately before backend dispatch.
        recon_amp, recon_phase, results = run_cdi_example_with_backend(
            ptycho_data, test_data, config, do_stitching=args.do_stitching,
            torch_execution_config=torch_execution_request,
            torch_factory_overrides=torch_factory_overrides,
        )

        # TensorFlow-only persistence: only save via model_manager and save_outputs for TensorFlow backend
        # PyTorch workflows use save_torch_bundle inside the backend workflow
        if config.backend == 'tensorflow':
            _save_tensorflow_model_legacy(config)
            save_outputs(recon_amp, recon_phase, results, str(config.output_dir))
            logger.info("TensorFlow artifacts saved via model_manager and save_outputs")
        else:
            # PyTorch backend relies on internal persistence; log manifest location if available
            logger.info(f"PyTorch backend completed. Check {config.output_dir} for saved bundles.")
            if 'bundle_path' in results:
                logger.info(f"PyTorch bundle saved at: {results['bundle_path']}")
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()

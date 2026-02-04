#!/usr/bin/env python

import logging
import sys

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
    parse_arguments as base_parse_arguments,
    setup_configuration,
    load_data,
    save_outputs,
    logger
)
from ptycho.workflows.backend_selector import run_cdi_example_with_backend
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho import model_manager, params
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
    enable_oversampling = config.enable_oversampling
    neighbor_pool_size = config.neighbor_pool_size

    # Case 1: Independent control with n_subsample
    if config.n_subsample is not None:
        n_subsample = config.n_subsample
        n_groups = config.n_groups

        if gridsize == 1:
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"using {n_groups} groups for training")
        else:
            total_from_groups = n_groups * gridsize * gridsize
            message = (f"Independent sampling control: subsampling {n_subsample} images, "
                      f"creating {n_groups} groups (approx {total_from_groups} patterns from groups)")
            if enable_oversampling:
                K = neighbor_pool_size if neighbor_pool_size is not None else config.neighbor_count
                message += f" [Oversampling enabled: K={K}]"

        return n_subsample, n_groups, enable_oversampling, neighbor_pool_size, message

    # Case 2: Legacy behavior - n_groups controls both
    else:
        # For backward compatibility, n_groups controls subsampling
        if gridsize == 1:
            n_subsample = config.n_groups
            n_groups = config.n_groups
            message = f"Legacy mode: using {n_groups} groups (gridsize=1)"
        else:
            # For gridsize > 1, we need to subsample enough to create the groups
            n_subsample = config.n_groups  # This will be interpreted as groups by generate_grouped_data
            n_groups = config.n_groups
            total_patterns = n_groups * gridsize * gridsize
            message = (f"Legacy mode: --n-groups={n_groups} refers to neighbor groups "
                      f"(gridsize={gridsize}, approx {total_patterns} patterns)")
            if enable_oversampling:
                K = neighbor_pool_size if neighbor_pool_size is not None else config.neighbor_count
                message += f" [Oversampling enabled: K={K}]"

        return n_subsample, n_groups, enable_oversampling, neighbor_pool_size, message

def parse_arguments():
    """
    Custom parse_arguments extending base with PyTorch execution config flags.

    This wrapper adds PyTorch-specific execution flags on top of the standard
    TrainingConfig fields, following the pattern from scripts/inference/inference.py.
    See docs/workflows/pytorch.md §12 for flag descriptions.
    """
    # Start with a parser that has no arguments yet
    import sys
    from ptycho.cli_args import add_logging_arguments
    from ptycho.config.config import TrainingConfig, ModelConfig
    from dataclasses import fields
    from typing import get_origin, get_args, Union, Literal
    from pathlib import Path

    logger_local = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--do_stitching", action='store_true', default=False,
                        help="Perform image stitching after training (default: False)")

    # Add logging arguments
    add_logging_arguments(parser)

    # Add arguments based on TrainingConfig fields (same as base_parse_arguments)
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
                parser.add_argument(
                    f"--{field.name}",
                    type=lambda x: Path(x) if x is not None else None,
                    default=None if field.default == None else str(field.default),
                    help=f"Path for {field.name}"
                )
            elif hasattr(field.type, "__origin__") and field.type.__origin__ is Literal:
                choices = list(field.type.__args__)
                parser.add_argument(
                    f"--{field.name}",
                    type=str,
                    choices=choices,
                    default=field.default,
                    help=f"Training parameter: {field.name}, choices: {choices}"
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
                            help=f"Backend selection: {', '.join(choices)} (default: {field.default}). "
                                 f"PyTorch backend requires torch>=2.2 (POLICY-001)."
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

    # PyTorch-only execution flags (see docs/workflows/pytorch.md §12)
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
                       choices=['Default', 'ReduceLROnPlateau', 'CosineAnnealing'],
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
    parser.add_argument("--torch-logger", type=str, default='mlflow',
                       choices=['csv', 'tensorboard', 'mlflow', 'none'],
                       help="Logger backend for PyTorch training (default: 'mlflow'). "
                            "Options: 'mlflow' (requires server), 'csv' (zero deps), 'tensorboard', 'none' (disable). "
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


def apply_torch_plateau_overrides(args, argv=None) -> None:
    """Map torch-specific plateau flags onto TrainingConfig fields."""
    argv = argv or sys.argv
    if "--torch-plateau-factor" in argv and getattr(args, "torch_plateau_factor", None) is not None:
        args.plateau_factor = args.torch_plateau_factor
    if "--torch-plateau-patience" in argv and getattr(args, "torch_plateau_patience", None) is not None:
        args.plateau_patience = args.torch_plateau_patience
    if "--torch-plateau-min-lr" in argv and getattr(args, "torch_plateau_min_lr", None) is not None:
        args.plateau_min_lr = args.torch_plateau_min_lr
    if "--torch-plateau-threshold" in argv and getattr(args, "torch_plateau_threshold", None) is not None:
        args.plateau_threshold = args.torch_plateau_threshold


def main() -> None:
    """Main function to orchestrate the CDI example script execution."""
    args = parse_arguments()
    
    # Handle legacy argument name
    if hasattr(args, 'train_data_file_path'):
        args.train_data_file = args.train_data_file_path
        delattr(args, 'train_data_file_path')

    apply_torch_plateau_overrides(args)

    config = setup_configuration(args, args.config)
    
    # Interpret sampling parameters with new independent control support
    n_subsample, n_groups, enable_oversampling, neighbor_pool_size, interpretation_message = interpret_sampling_parameters(config)
    logger.info(interpretation_message)
    
    # Log warning if potentially problematic configuration
    if config.n_subsample is not None and config.model.gridsize > 1:
        min_required = n_groups * config.model.gridsize * config.model.gridsize
        if n_subsample < min_required:
            logger.warning(f"n_subsample ({n_subsample}) may be too small to create {n_groups} "
                         f"groups of size {config.model.gridsize}². Consider increasing n_subsample to at least {min_required}")
    
    # Update global params with new-style config at entry point
    update_legacy_dict(params.cfg, config)
    
    try:
        logger.info(f"Starting training with n_subsample={n_subsample}, n_groups={n_groups}, "
                   f"stitching={'enabled' if args.do_stitching else 'disabled'}")

        # Load data with new independent sampling parameters
        # Note: load_data still uses n_images parameter name internally
        ptycho_data = load_data(
            str(config.train_data_file), 
            n_images=n_groups,  # Pass n_groups as n_images to maintain API compatibility
            n_subsample=n_subsample,
            subsample_seed=config.subsample_seed
        )
        
        # Check for metadata and override nphotons if present
        from ptycho.metadata import MetadataManager
        try:
            _, metadata = MetadataManager.load_with_metadata(str(config.train_data_file))
            # Check both top-level and physics_parameters for nphotons
            metadata_nphotons = None
            if metadata:
                if 'nphotons' in metadata:
                    metadata_nphotons = float(metadata['nphotons'])
                elif 'physics_parameters' in metadata and 'nphotons' in metadata['physics_parameters']:
                    metadata_nphotons = float(metadata['physics_parameters']['nphotons'])
                
                if metadata_nphotons is not None:
                    original_nphotons = config.nphotons
                    # Update config with metadata nphotons value (create new instance for dataclass)
                    config = config.__class__(
                        **{**config.__dict__, 'nphotons': metadata_nphotons}
                    )
                    logger.info(f"Overriding nphotons from config ({original_nphotons:.1e}) with value from dataset metadata: {metadata_nphotons:.1e}")
                    # Update the legacy params dict as well
                    params.cfg['nphotons'] = metadata_nphotons
        except Exception as e:
            logger.debug(f"No metadata found or error reading metadata: {e}")
        
        test_data = None
        if config.test_data_file:
            test_data = load_data(str(config.test_data_file))
            logger.info(f"Loaded test data from {config.test_data_file}")

        # Build PyTorch execution config if backend is pytorch
        torch_execution_config = None
        if config.backend == 'pytorch':
            # Determine if user explicitly provided any --torch-* flags
            # We check against argparse defaults to detect user overrides
            torch_flags_explicitly_set = any([
                'torch_accelerator' in sys.argv or '--torch-accelerator' in sys.argv,
                'torch_deterministic' in sys.argv or '--torch-deterministic' in sys.argv,
                'torch_num_workers' in sys.argv or '--torch-num-workers' in sys.argv,
                'torch_learning_rate' in sys.argv or '--torch-learning-rate' in sys.argv,
                'torch_scheduler' in sys.argv or '--torch-scheduler' in sys.argv,
                'torch_logger' in sys.argv or '--torch-logger' in sys.argv,
                'torch_enable_checkpointing' in sys.argv or '--torch-enable-checkpointing' in sys.argv,
                'torch_checkpoint_save_top_k' in sys.argv or '--torch-checkpoint-save-top-k' in sys.argv,
                'torch_accumulate_grad_batches' in sys.argv or '--torch-accumulate-grad-batches' in sys.argv,
            ])

            if not torch_flags_explicitly_set:
                # No --torch-* flags provided: defer to backend_selector's auto-instantiated GPU defaults
                logger.info("POLICY-001: No --torch-* execution flags provided. "
                           "Backend will use GPU-first defaults (auto-detects CUDA if available, else CPU). "
                           "CPU-only users should pass --torch-accelerator cpu.")
                # Leave torch_execution_config=None to signal backend_selector to auto-instantiate
            else:
                # User provided at least one --torch-* flag: build execution config explicitly
                from ptycho_torch.cli.shared import build_execution_config_from_args

                # Map CLI flags to execution config namespace (see docs/workflows/pytorch.md §12)
                exec_args = argparse.Namespace(
                    accelerator=getattr(args, 'torch_accelerator', 'auto'),
                    deterministic=getattr(args, 'torch_deterministic', True),
                    num_workers=getattr(args, 'torch_num_workers', 0),
                    learning_rate=getattr(args, 'torch_learning_rate', None),
                    scheduler=getattr(args, 'torch_scheduler', 'Default'),
                    logger_backend=getattr(args, 'torch_logger', 'mlflow'),
                    enable_checkpointing=getattr(args, 'torch_enable_checkpointing', True),
                    checkpoint_save_top_k=getattr(args, 'torch_checkpoint_save_top_k', 1),
                    accumulate_grad_batches=getattr(args, 'torch_accumulate_grad_batches', 1),
                    checkpoint_monitor_metric='val_loss',  # Default per docs/workflows/pytorch.md
                    checkpoint_mode='min',  # Default
                    early_stop_patience=100,  # Default
                    quiet=getattr(args, 'debug', False) == False,  # Invert debug flag for quiet
                    disable_mlflow=False,  # Not applicable for training in this context
                    # Recon logging knobs
                    recon_log_every_n_epochs=getattr(args, 'torch_recon_log_every_n_epochs', None),
                    recon_log_num_patches=getattr(args, 'torch_recon_log_num_patches', 4),
                    recon_log_fixed_indices=getattr(args, 'torch_recon_log_fixed_indices', None),
                    recon_log_stitch=getattr(args, 'torch_recon_log_stitch', False),
                    recon_log_max_stitch_samples=getattr(args, 'torch_recon_log_max_stitch_samples', None),
                )

                # Build validated execution config from CLI args (POLICY-001, CONFIG-002, CONFIG-LOGGER-001)
                torch_execution_config = build_execution_config_from_args(exec_args, mode='training')
                logger.info(f"PyTorch execution config built: accelerator={torch_execution_config.accelerator}, "
                           f"num_workers={torch_execution_config.num_workers}, "
                           f"learning_rate={torch_execution_config.learning_rate}, "
                           f"logger_backend={torch_execution_config.logger_backend}")

        recon_amp, recon_phase, results = run_cdi_example_with_backend(
            ptycho_data, test_data, config, do_stitching=args.do_stitching,
            torch_execution_config=torch_execution_config
        )

        # TensorFlow-only persistence: only save via model_manager and save_outputs for TensorFlow backend
        # PyTorch workflows use save_torch_bundle inside the backend workflow
        if config.backend == 'tensorflow':
            model_manager.save(str(config.output_dir))
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

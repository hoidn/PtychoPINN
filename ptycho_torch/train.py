"""
PyTorch Training Script for PtychoPINN (INTEGRATE-PYTORCH-001)

This module provides the main training entry point for the PyTorch backend.
It supports both programmatic use via the main() function and CLI execution.

CLI Interface (Phase E2.C1):
    python -m ptycho_torch.train \\
        --train_data_file <path>       # Required: Training NPZ dataset
        --test_data_file <path>        # Optional: Validation NPZ dataset
        --output_dir <path>            # Required: Checkpoint output directory
        --max_epochs <int>             # Optional: Training epochs (default: 100)
        --n_images <int>               # Optional: Number of diffraction groups (default: 512)
        --gridsize <int>               # Optional: Grid size for grouping (default: 2)
        --batch_size <int>             # Optional: Training batch size (default: 16)
        --device <cpu|cuda>            # Optional: Compute device (default: cpu)
        --disable_mlflow               # Optional: Suppress MLflow autologging

Legacy Interface (Backward Compatible):
    python -m ptycho_torch.train --ptycho_dir <path> --config <config.json> [--disable_mlflow]

MLflow Behavior:
    By default, training uses MLflow for experiment tracking (autologging, metrics, artifacts).
    When --disable_mlflow is set:
    - mlflow.pytorch.autolog() is skipped
    - No experiment or run creation
    - Training proceeds normally without tracking overhead
    - Checkpoints are still saved via Lightning callbacks

Key Features:
- CONFIG-001 compliant: Populates params.cfg before workflow dispatch
- Lightning integration: Automatic checkpointing, early stopping, distributed training
- Configuration bridge: PyTorch configs → TensorFlow dataclasses → params.cfg

References:
    - Phase E2.C1 spec: plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md
    - Test contract: tests/torch/test_integration_workflow_torch.py
    - Config bridge: ptycho_torch/config_bridge.py
"""

#Most basic modules
import sys
import argparse
import os
import json
import random
import math
from pathlib import Path

#Typing
from dataclasses import asdict
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig

#ML libraries
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
import torch.distributed as dist

#Automation modules
#Lightning
try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.strategies import DDPStrategy
except ImportError as e:
    raise RuntimeError(
        "PyTorch backend requires Lightning. Install with: pip install -e .[torch]"
    ) from e

#MLFlow
try:
    import mlflow.pytorch
    from mlflow import MlflowClient
except ImportError as e:
    raise RuntimeError(
        "PyTorch backend requires MLflow. Install with: pip install -e .[torch]"
    ) from e

#Configs/Params
from ptycho_torch.config_params import update_existing_config

#Custom modules
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.utils import config_to_json_serializable_dict, load_config_from_json, validate_and_process_config
from ptycho_torch.train_utils import set_seed, get_training_strategy, find_learning_rate, log_parameters_mlflow, is_effectively_global_rank_zero, print_auto_logged_info
from ptycho_torch.train_utils import ModelFineTuner, PtychoDataModule

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("PtychoPINN")

#----- Helper Functions -------

def _infer_probe_size(npz_file):
    """
    Infer probe size (N) from NPZ metadata without loading full arrays.

    This function reads the probeGuess array header from an NPZ file using the
    zipfile approach, following the pattern established in ptycho_torch/dataloader.py:npz_headers().

    Args:
        npz_file (str or Path): Path to NPZ file containing probeGuess key

    Returns:
        int or None: First dimension of probeGuess shape (N), or None if probeGuess key missing

    References:
        - specs/data_contracts.md §1 — probeGuess is required key for canonical NPZ format
        - ptycho_torch/dataloader.py:29-83 — npz_headers() implementation pattern
        - INTEGRATE-PYTORCH-001-PROBE-SIZE — Probe size mismatch resolution

    Example:
        >>> N = _infer_probe_size("datasets/Run1084_recon3_postPC_shrunk_3.npz")
        >>> print(N)  # 64 (for this dataset)
    """
    import zipfile
    import numpy as np

    try:
        with zipfile.ZipFile(npz_file) as archive:
            # Search for probeGuess key in NPZ archive
            for name in archive.namelist():
                if name.startswith('probeGuess') and name.endswith('.npy'):
                    # Open the .npy file inside the archive
                    npy = archive.open(name)
                    # Read array header without loading data
                    version = np.lib.format.read_magic(npy)
                    shape, _, _ = np.lib.format._read_array_header(npy, version)
                    # Return first dimension (probe is typically N x N)
                    return shape[0]

            # probeGuess key not found - return None for fallback to default
            return None

    except (zipfile.BadZipFile, FileNotFoundError, KeyError) as e:
        # If NPZ is invalid or missing, return None
        # Caller can decide whether to use default or raise error
        return None

#----- Main -------

def main(ptycho_dir,
         config_path = None,
         existing_config = None,
         disable_mlflow = False,
         output_dir = None,
         execution_config = None):
    '''
    Main training script. Can provide dictionary of modified configs based off of dataclass attributes to overwrite

    Inputs
    --------
    ptycho_dir: Directory of ptychography files. Assumed that all diffraction pattern dimensions are equal, and the formatting is identical
                Read dataloader.py to get a sense of the formats expected
    config_path: Path to JSON configuration file (optional, for legacy interface)
    existing_config: Tuple of (DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig) if configs already instantiated
    disable_mlflow: If True, suppresses all MLflow tracking (autologging, experiment tracking, run creation).
                   Useful for CI environments without MLflow server. Default: False.
    output_dir: Optional override for checkpoint output directory. If provided, configures Lightning's default_root_dir.
    execution_config: Optional PyTorchExecutionConfig for runtime knobs (Phase C4.C3 - ADR-003).
                     If None, uses default execution config with CPU accelerator.

    Outputs
    --------
    run_id: Can be recycled to load things through MLFlow. See inference.py for details (only when disable_mlflow=False)
    '''
    #Define configs
    print('Starting training loop. Loading configs...')
    #Legacy function, if train is called by itself instead of train_full
    if not existing_config:
        try:
            config_data = load_config_from_json(config_path)
            d_config_replace, m_config_replace, t_config_replace, i_config_replace, dgen_config_replace = validate_and_process_config(config_data)
        except Exception as e:
            print(f"Failed to open/validate config because of: {e}")
        
        data_config = DataConfig()
        if d_config_replace is not None:
            update_existing_config(data_config, d_config_replace)
        
        model_config = ModelConfig()
        if m_config_replace is not None:
            update_existing_config(model_config, m_config_replace)
        
        training_config = TrainingConfig()
        if t_config_replace is not None:
            update_existing_config(training_config, t_config_replace)    

        inference_config = InferenceConfig()
        
        datagen_config = DatagenConfig()
        if dgen_config_replace is not None:
            update_existing_config(datagen_config, dgen_config_replace)
    
    #If train is called via train_full, these settings will already have been loaded
    else:
        data_config, model_config, training_config, inference_config, datagen_config = existing_config

    #Setting seed
    set_seed(42, n_devices = training_config.n_devices)

    # Data module in place of pytorch native dataloaders. Data module is a lightning class
    # Create DataModule
    print("Creating data module")
    data_module = PtychoDataModule(
        ptycho_dir,
        model_config,
        data_config,
        training_config,
        initial_remake_map=True, # Set to True to force recreation on this run
        val_split=0.05,  # Use 5% for validation (should be fine, need more training data)
        val_seed=42     # Reproducible split
    )

    #Create model
    print('Creating model...')
    model = PtychoPINN_Lightning(model_config, data_config, training_config, inference_config)
    model.training = True

    #Update LR (Phase C4.C3: Use execution_config if available, else training_config)
    base_lr = execution_config.learning_rate if execution_config else training_config.learning_rate
    updated_lr = find_learning_rate(base_lr,
                                    training_config.n_devices, training_config.batch_size)
    model.lr = updated_lr

    #Loss
    val_loss_label = model.val_loss_name

    # Configure checkpoint directory
    # If output_dir provided, use it; otherwise use current working directory parent
    checkpoint_root = Path(output_dir) if output_dir else Path(os.path.dirname(os.getcwd()))
    checkpoint_dir = checkpoint_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    #Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=val_loss_label,
        mode='min',
        save_top_k=1,
        filename='best-checkpoint',
        save_last=True  # Also save last checkpoint for recovery
    )

    # Stop early when validation loss stops improving
    early_stop_callback = EarlyStopping(
        monitor=val_loss_label,
        mode='min',
        patience=100,
        verbose=True,
        strict=True
    )
    callbacks = [
        checkpoint_callback,
        early_stop_callback
    ]

    # All training now runs single-stage
    total_training_epochs = training_config.epochs

    # Phase C4.C3: Apply execution config to Trainer (ADR-003)
    # Use execution_config if provided, otherwise create default
    if execution_config is None:
        from ptycho.config.config import PyTorchExecutionConfig
        execution_config = PyTorchExecutionConfig(
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
        )

    # Resolve accelerator (prefer execution_config, fallback to training_config logic)
    if execution_config.accelerator == 'auto':
        resolved_accelerator = 'gpu' if training_config.n_devices > 0 and torch.cuda.is_available() else 'cpu'
    else:
        resolved_accelerator = execution_config.accelerator
        # Map 'cuda' alias to 'gpu' for Lightning compatibility
        if resolved_accelerator == 'cuda':
            resolved_accelerator = 'gpu'

    # Create trainer with execution config knobs
    trainer = L.Trainer(
        max_epochs = total_training_epochs,
        default_root_dir = str(checkpoint_root),
        devices = training_config.n_devices,
        accelerator = resolved_accelerator,
        callbacks = callbacks,
        # accumulate_grad_batches=training_config.accum_steps,  # Lightning handles this
        # gradient_clip_val=training_config.gradient_clip_val,   # Lightning handles this
        strategy=get_training_strategy(training_config.n_devices),
        check_val_every_n_epoch=1,  # Validate every epoch
        enable_checkpointing=True,  # Enable checkpointing for early stopping
        enable_progress_bar=execution_config.enable_progress_bar,  # Controlled by execution config
        deterministic=execution_config.deterministic,  # Reproducibility control
    )

    #Mlflow setup
    # mlflow.set_tracking_uri("")
    print("training_config name:", training_config.experiment_name)

    if not disable_mlflow:
        mlflow.pytorch.autolog(checkpoint_monitor = val_loss_label)

    #Train the model
    # if trainer.is_global_zero:
    if is_effectively_global_rank_zero():
        if not disable_mlflow:
            mlflow.set_experiment(training_config.experiment_name)

            with mlflow.start_run() as run:
                run_ids = {}
                # Log configuration parameters
                log_parameters_mlflow(data_config, model_config, training_config, inference_config, datagen_config)
                #Set tags (relatively new)
                mlflow.set_tag("stage", "training")
                mlflow.set_tag("encoder_frozen", "False")
                mlflow.set_tag("model_name", training_config.model_name)
                if training_config.notes:
                    mlflow.set_tag("notes", training_config.notes)
                mlflow.set_tag("stage", "training")

                #Train base model
                print(f'[Rank {trainer.global_rank}] Beginning model training/final data prep...')
                trainer.fit(model, datamodule = data_module)

                print(f'[Rank {trainer.global_rank}] Done training. Printing training params...')
                print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))
                run_ids['training'] = run.info.run_id
        else:
            # MLflow disabled: train without tracking
            run_ids = {}
            print(f'[Rank {trainer.global_rank}] Beginning model training/final data prep... (MLflow disabled)')
            trainer.fit(model, datamodule = data_module)
            print(f'[Rank {trainer.global_rank}] Done training.')
            run_ids['training'] = None  # No run_id when MLflow disabled

    #Fine-tune if applicable (encoder frozen default)
    if training_config.epochs_fine_tune > 0:
        #Fine-tuning
        fine_tuner = ModelFineTuner(model, data_module, training_config)
        fine_tuning_run_id = fine_tuner.fine_tune(experiment_name = training_config.experiment_name)
        
        if is_effectively_global_rank_zero():
            run_ids['fine-tune'] = fine_tuning_run_id
    
    # if is_effectively_global_rank_zero():
    #     print(f"Training run_id: {run_ids['training']}")
    #     print(f"Fine_tune run_id: {run_ids['fine-tune']}")
        
    #     return run_ids
    # else:
    #     return None

    # CRITICAL: Final synchronization before returning
    if dist.is_initialized():
        print(f'[Rank {trainer.global_rank}] Waiting at final barrier before return...')
        dist.barrier()
        print(f'[Rank {trainer.global_rank}] ✓ Passed final barrier, about to return')

    if is_effectively_global_rank_zero():
        if run_ids.get('training'):
            print(f"Training run_id: {run_ids['training']}")
        if 'fine-tune' in run_ids:
            print(f"Fine_tune run_id: {run_ids['fine-tune']}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        return run_ids
    else:
        print(f'[Rank {trainer.global_rank}] Non-zero rank returning None')
        return None


def cli_main():
    """
    CLI entrypoint for PyTorch training workflow (Phase E2.C1).

    This function bridges the CLI interface to the existing main() function by:
    1. Parsing CLI arguments
    2. Creating PyTorch config singletons
    3. Using config_bridge adapters to convert to TensorFlow dataclasses
    4. Calling update_legacy_dict(params.cfg) to ensure CONFIG-001 compliance
    5. Delegating to main() for actual training execution

    The CLI interface mirrors the TensorFlow training script flags for consistency.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning training for ptychographic reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with new CLI interface
  python -m ptycho_torch.train --train_data_file data/train.npz --output_dir ./outputs --max_epochs 10 --device cpu --disable_mlflow

  # Train with legacy interface
  python -m ptycho_torch.train --ptycho_dir data/ --config config.json
        """
    )

    # New CLI interface flags (Phase E2.C1)
    parser.add_argument('--train_data_file', type=str,
                       help='Path to training NPZ dataset (required for new CLI interface)')
    parser.add_argument('--test_data_file', type=str,
                       help='Path to validation NPZ dataset (optional)')
    parser.add_argument('--output_dir', type=str,
                       help='Directory for checkpoint outputs (required for new CLI interface)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--n_images', type=int, default=512,
                       help='Number of diffraction groups to process (default: 512)')
    parser.add_argument('--gridsize', type=int, default=2,
                       help='Grid size for spatial grouping (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='[DEPRECATED] Use --accelerator instead. Compute device: cpu or cuda (default: cpu)')
    parser.add_argument('--disable_mlflow', action='store_true',
                       help='[DEPRECATED] Use --logger none instead. Disable all experiment tracking loggers.')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bars and verbose output')
    parser.add_argument(
        '--torch-loss-mode',
        type=str,
        default='poisson',
        choices=['poisson', 'mae'],
        help=(
            "Select the Torch backend loss pipeline. "
            "'poisson' matches the physics-weighted Poisson NLL used in TensorFlow, "
            "while 'mae' disables the physics loss and trains purely on amplitude MAE."
        )
    )

    # Execution config flags (Phase C4.C1 - ADR-003)
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'],
        help=(
            'Hardware accelerator for training: '
            'auto (auto-detect, default), cpu (CPU-only), gpu (NVIDIA GPU), '
            'cuda (alias for gpu), tpu (Google TPU), mps (Apple Silicon). '
            'Default: auto.'
        )
    )
    parser.add_argument(
        '--deterministic',
        dest='deterministic',
        action='store_true',
        default=True,
        help=(
            'Enable deterministic training for reproducibility (default: enabled). '
            'Sets torch.use_deterministic_algorithms(True) and Lightning deterministic=True. '
            'Use --no-deterministic to disable for potential performance gains.'
        )
    )
    parser.add_argument(
        '--no-deterministic',
        dest='deterministic',
        action='store_false',
        help='Disable deterministic training. May improve performance but results are non-reproducible.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        dest='num_workers',
        help=(
            'Number of DataLoader worker processes for parallel data loading (default: 0 = synchronous). '
            'Typical values: 2-8 for multi-core systems. Higher values increase data loading throughput '
            'but consume more memory. Set to 0 for single-threaded loading (safest for CI).'
        )
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        dest='learning_rate',
        help=(
            'Learning rate for Adam optimizer (default: 1e-3). '
            'Typical range: 1e-5 (slow, stable) to 1e-2 (fast, may diverge). '
            'Adjust based on convergence behavior during training.'
        )
    )

    # Logger backend flags (Phase EB3.B - ADR-003)
    parser.add_argument(
        '--logger',
        type=str,
        default='csv',
        choices=['none', 'csv', 'tensorboard', 'mlflow'],
        dest='logger_backend',
        help=(
            'Experiment tracking logger backend (default: csv). '
            'Options: none (no logging), csv (Lightning CSVLogger), '
            'tensorboard (TensorBoard via Lightning), mlflow (MLflow via Lightning). '
            'Loss metrics are logged to {output_dir}/lightning_logs/{version}/. '
            'Use --logger none if you only need progress suppression (no metrics).'
        )
    )

    # Patch statistics instrumentation (FIX-PYTORCH-FORWARD-PARITY-001 Phase A)
    parser.add_argument(
        '--log-patch-stats',
        dest='log_patch_stats',
        action='store_true',
        help=(
            'Enable per-patch amplitude statistics logging with JSON+PNG dumps. '
            'Writes torch_patch_stats.json and torch_patch_grid.png to analysis/ subdirectory. '
            'Default: disabled.'
        )
    )
    parser.add_argument(
        '--patch-stats-limit',
        dest='patch_stats_limit',
        type=int,
        default=None,
        help=(
            'Maximum number of batches to instrument for patch stats. '
            'Use small values (e.g., 2) to avoid log spam during full training. '
            'Default: None (instrument all batches when --log-patch-stats is enabled).'
        )
    )

    # Checkpoint and early stopping flags (Phase EB1.B - ADR-003)
    parser.add_argument(
        '--enable-checkpointing',
        dest='enable_checkpointing',
        action='store_true',
        default=True,
        help=(
            'Enable automatic model checkpointing during training (default: enabled). '
            'Checkpoints are saved based on monitored metric performance. '
            'Use --disable-checkpointing to turn off.'
        )
    )
    parser.add_argument(
        '--disable-checkpointing',
        dest='enable_checkpointing',
        action='store_false',
        help='Disable automatic model checkpointing during training.'
    )
    parser.add_argument(
        '--checkpoint-save-top-k',
        type=int,
        default=1,
        dest='checkpoint_save_top_k',
        help=(
            'Number of best checkpoints to keep (default: 1). '
            'Set to -1 to save all checkpoints, 0 to disable saving. '
            'Best checkpoints are determined by --checkpoint-monitor metric.'
        )
    )
    parser.add_argument(
        '--checkpoint-monitor',
        type=str,
        default='val_loss',
        dest='checkpoint_monitor_metric',
        help=(
            'Metric to monitor for checkpoint selection (default: val_loss). '
            'Common options: val_loss, train_loss, val_accuracy. '
            'Must match a metric logged by the Lightning module.'
        )
    )
    parser.add_argument(
        '--checkpoint-mode',
        type=str,
        default='min',
        choices=['min', 'max'],
        dest='checkpoint_mode',
        help=(
            'Mode for checkpoint metric optimization (default: min). '
            'Use "min" for metrics where lower is better (e.g., loss), '
            '"max" for metrics where higher is better (e.g., accuracy).'
        )
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=100,
        dest='early_stop_patience',
        help=(
            'Early stopping patience in epochs (default: 100). '
            'Training stops if monitored metric does not improve for this many epochs. '
            'Set to large value (e.g., 1000) to effectively disable early stopping.'
        )
    )

    # Optimization knobs (Phase EB2 - ADR-003)
    parser.add_argument(
        '--scheduler',
        type=str,
        default='Default',
        choices=['Default', 'Exponential', 'MultiStage', 'Adaptive'],
        dest='scheduler',
        help=(
            'Learning rate scheduler type (default: Default). '
            'Choices: Default (no scheduler), Exponential (exponential decay), '
            'MultiStage (step-wise decay), Adaptive (plateau-based reduction). '
            'Scheduler configuration must match Lightning module expectations.'
        )
    )
    parser.add_argument(
        '--accumulate-grad-batches',
        type=int,
        default=1,
        dest='accumulate_grad_batches',
        help=(
            'Number of gradient accumulation steps (default: 1 = no accumulation). '
            'Accumulation simulates larger effective batch sizes by accumulating gradients '
            'over multiple forward/backward passes before updating weights. '
            'Effective batch size = batch_size * accumulate_grad_batches. '
            'WARNING: Values >1 increase memory efficiency but may affect training dynamics. '
            'Typical values: 1-8 depending on GPU memory and batch size constraints.'
        )
    )

    # Legacy interface flags (backward compatibility)
    parser.add_argument('--ptycho_dir', type=str,
                       help='Path to ptycho directory (legacy interface)')
    parser.add_argument('--config', type=str,
                       help='Path to JSON configuration file (legacy interface)')

    args = parser.parse_args()

    # Determine which interface is being used
    legacy_interface = args.ptycho_dir is not None or args.config is not None
    new_interface = args.train_data_file is not None or args.output_dir is not None

    if legacy_interface and new_interface:
        print("ERROR: Cannot mix legacy (--ptycho_dir, --config) and new CLI flags (--train_data_file, --output_dir)")
        print("Use either the legacy interface OR the new CLI interface, not both.")
        sys.exit(1)

    if legacy_interface:
        # Legacy interface: --ptycho_dir --config
        print("Using legacy interface (--ptycho_dir --config)")
        if not args.ptycho_dir:
            print("ERROR: --ptycho_dir required for legacy interface")
            sys.exit(1)

        try:
            main(args.ptycho_dir, args.config, existing_config=None,
                 disable_mlflow=args.disable_mlflow)
        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif new_interface:
        # New CLI interface: --train_data_file --output_dir ... (Phase C4.C2 - ADR-003)
        print("Using new CLI interface with factory-based config (ADR-003)")

        # Validate required arguments
        if not args.train_data_file:
            print("ERROR: --train_data_file required for new CLI interface")
            sys.exit(1)
        if not args.output_dir:
            print("ERROR: --output_dir required for new CLI interface")
            sys.exit(1)

        # Convert paths to Path objects
        train_data_file = Path(args.train_data_file)
        test_data_file = Path(args.test_data_file) if args.test_data_file else None
        output_dir = Path(args.output_dir)

        # Validate paths using shared helper (Phase D.B - ADR-003)
        from ptycho_torch.cli.shared import validate_paths
        try:
            validate_paths(train_data_file, test_data_file, output_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        # Extract ptycho_dir from train_data_file parent directory
        # The existing main() expects a directory containing NPZ files
        ptycho_dir = train_data_file.parent

        # Phase D.B: Build execution config using shared helper (ADR-003)
        from ptycho_torch.cli.shared import build_execution_config_from_args
        try:
            execution_config = build_execution_config_from_args(args, mode='training')
        except ValueError as e:
            print(f"ERROR: Invalid execution config: {e}")
            sys.exit(1)

        # Phase C4.C2: Use config factory instead of manual config construction
        print("Creating configuration via factory (CONFIG-001 compliance)...")
        from ptycho_torch.config_factory import create_training_payload

        # Build overrides dict from CLI arguments
        overrides = {
            'n_groups': args.n_images,
            'batch_size': args.batch_size,
            'gridsize': args.gridsize,
            'max_epochs': args.max_epochs,
            'torch_loss_mode': args.torch_loss_mode,
            'log_patch_stats': args.log_patch_stats,
            'patch_stats_limit': args.patch_stats_limit,
        }
        if test_data_file:
            overrides['test_data_file'] = test_data_file

        try:
            # Call factory to create all configs and populate params.cfg
            payload = create_training_payload(
                train_data_file=train_data_file,
                output_dir=output_dir,
                overrides=overrides,
                execution_config=execution_config,
            )

            print(f"✓ Factory created configs: N={payload.pt_data_config.N}, "
                  f"gridsize={payload.pt_data_config.grid_size}, "
                  f"epochs={payload.pt_training_config.epochs}")
            print(f"✓ Execution config: accelerator={execution_config.accelerator}, "
                  f"deterministic={execution_config.deterministic}, "
                  f"learning_rate={execution_config.learning_rate}")

            # Extract configs for main()
            # Note: Factory only creates training-relevant configs; create InferenceConfig/DatagenConfig defaults
            from ptycho_torch.config_params import InferenceConfig, DatagenConfig
            existing_config = (
                payload.pt_data_config,
                payload.pt_model_config,
                payload.pt_training_config,
                InferenceConfig(),  # Not in payload, use default
                DatagenConfig(),  # Not in payload, use default
            )

        except Exception as e:
            print(f"ERROR: Configuration factory failed: {e}")
            print("Cannot proceed - factory responsible for CONFIG-001 compliance")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Call workflow-based training (Phase C4.D3 - bundle persistence)
        try:
            print(f"Starting training with {args.max_epochs} epochs...")

            # Load training data (CONFIG-001 already satisfied by factory)
            from ptycho.raw_data import RawData
            train_data = RawData.from_file(str(train_data_file))
            test_data = RawData.from_file(str(test_data_file)) if test_data_file else None

            # Route through run_cdi_example_torch for bundle persistence
            from ptycho_torch.workflows.components import run_cdi_example_torch
            amplitude, phase, results = run_cdi_example_torch(
                train_data=train_data,
                test_data=test_data,
                config=payload.tf_training_config,
                do_stitching=False  # CLI only needs training, not reconstruction
            )

            print(f"✓ Training completed successfully. Outputs saved to {output_dir}")
            print(f"✓ Model bundle saved to {output_dir}/wts.h5.zip")

        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # No interface specified
        print("ERROR: Must specify either:")
        print("  - New CLI interface: --train_data_file <path> --output_dir <path>")
        print("  - Legacy interface: --ptycho_dir <dir> --config <json>")
        parser.print_help()
        sys.exit(1)


#Define main function
if __name__ == '__main__':
    cli_main()

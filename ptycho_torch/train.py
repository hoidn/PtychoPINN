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
        --batch_size <int>             # Optional: Training batch size (default: 4)
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

#----- Main -------

def main(ptycho_dir,
         config_path = None,
         existing_config = None,
         disable_mlflow = False,
         output_dir = None):
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

    #Update LR
    updated_lr = find_learning_rate(training_config.learning_rate,
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

    # Calculate total epochs for both MultiStage and
    if training_config.stage_2_epochs > 0 or training_config.stage_3_epochs > 0:
        # Add baseline number of epochs at each stage
        total_training_epochs = training_config.stage_1_epochs + training_config.stage_2_epochs + training_config.stage_3_epochs
    else:
        total_training_epochs = training_config.epochs

    # Create trainer
    trainer = L.Trainer(
        max_epochs = total_training_epochs,
        default_root_dir = str(checkpoint_root),
        devices = training_config.n_devices,
        accelerator = 'gpu' if training_config.n_devices > 0 and torch.cuda.is_available() else 'cpu',
        callbacks = callbacks,
        # accumulate_grad_batches=training_config.accum_steps,  # Lightning handles this
        # gradient_clip_val=training_config.gradient_clip_val,   # Lightning handles this
        strategy=get_training_strategy(training_config.n_devices),
        check_val_every_n_epoch=1,  # Validate every epoch
        enable_checkpointing=True,  # Enable checkpointing for early stopping
        enable_progress_bar=True,
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
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size (default: 4)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Compute device: cpu or cuda (default: cpu)')
    parser.add_argument('--disable_mlflow', action='store_true',
                       help='Disable MLflow experiment tracking (useful for CI)')

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
        # New CLI interface: --train_data_file --output_dir ...
        print("Using new CLI interface (Phase E2.C1)")

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

        # Validate input files exist
        if not train_data_file.exists():
            print(f"ERROR: Training data file not found: {train_data_file}")
            sys.exit(1)
        if test_data_file and not test_data_file.exists():
            print(f"ERROR: Test data file not found: {test_data_file}")
            sys.exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract ptycho_dir from train_data_file parent directory
        # The existing main() expects a directory containing NPZ files
        ptycho_dir = train_data_file.parent

        # Create PyTorch config singletons
        print("Creating PyTorch configuration objects...")

        # DataConfig: Configure data pipeline parameters
        data_config = DataConfig(
            N=128,  # Default crop size - will be overridden by NPZ metadata if needed
            grid_size=(args.gridsize, args.gridsize),
            K=7,  # Default neighbor count
            nphotons=1e9,  # Use TensorFlow default to avoid divergence
        )

        # ModelConfig: Configure model architecture
        model_config = ModelConfig(
            mode='Unsupervised',  # PINN mode
            amp_activation='silu',
        )

        # TrainingConfig: Configure training hyperparameters
        training_config = TrainingConfig(
            epochs=args.max_epochs,
            batch_size=args.batch_size,
            n_devices=1 if args.device == 'cpu' else (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            experiment_name='ptychopinn_pytorch',
        )

        # InferenceConfig: Create for completeness (required by main())
        inference_config = InferenceConfig()

        # DatagenConfig: Create for completeness (required by main())
        datagen_config = DatagenConfig()

        # Bundle configs for main()
        existing_config = (data_config, model_config, training_config, inference_config, datagen_config)

        # CRITICAL: CONFIG-001 Compliance
        # Bridge PyTorch configs → TensorFlow dataclasses → params.cfg
        # This MUST happen before calling main() to ensure legacy modules see correct values
        print("Bridging PyTorch configs to TensorFlow dataclasses (CONFIG-001 compliance)...")

        from ptycho_torch.config_bridge import to_model_config, to_training_config
        from ptycho.config.config import update_legacy_dict
        import ptycho.params as params

        try:
            # Convert to TensorFlow ModelConfig
            tf_model_config = to_model_config(data_config, model_config)

            # Convert to TensorFlow TrainingConfig with required overrides
            tf_training_config = to_training_config(
                tf_model_config,
                data_config,
                model_config,
                training_config,
                overrides=dict(
                    train_data_file=train_data_file,
                    test_data_file=test_data_file,
                    output_dir=output_dir,
                    n_groups=args.n_images,
                    nphotons=1e9,  # Explicit override to match TensorFlow default
                )
            )

            # Populate params.cfg BEFORE any workflow dispatch
            update_legacy_dict(params.cfg, tf_training_config)

            print(f"✓ params.cfg populated: N={params.cfg.get('N')}, gridsize={params.cfg.get('gridsize')}, n_groups={params.cfg.get('n_groups')}")

        except Exception as e:
            print(f"ERROR: Configuration bridge failed: {e}")
            print("This is a CONFIG-001 violation - cannot proceed without valid params.cfg")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Call main() with bridged configs
        try:
            print(f"Starting training with {args.max_epochs} epochs on {args.device}...")
            main(
                str(ptycho_dir),
                config_path=None,
                existing_config=existing_config,
                disable_mlflow=args.disable_mlflow,
                output_dir=str(output_dir)
            )
            print(f"✓ Training completed successfully. Outputs saved to {output_dir}")

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

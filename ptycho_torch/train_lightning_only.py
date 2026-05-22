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

#Configs/Params
from ptycho_torch.config_params import update_existing_config

#Custom modules
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.utils import config_to_json_serializable_dict, load_config_from_json, validate_and_process_config
from ptycho_torch.train_utils import set_seed, get_training_strategy, find_learning_rate, is_effectively_global_rank_zero, PtychoDataModule, PtychoDataModuleLightning, resolve_n_devices

# NEW: Import our custom Lightning utilities
from ptycho_torch.lightning_utils import (
    ConfigLogger,
    MetadataLogger,
    create_experiment_loggers,
    print_run_summary,
    find_best_checkpoint,
    DDPSafeProgressCallback
)

from ptycho_torch.model_finetuner_modified import ModelFineTuner

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import TQDMProgressBar


# Reduce timeout to 2 minutes for debugging
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" # Optional: gives more info on hangs

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
         output_dir = None,
         execution_config = None,
         run_name = None):
    '''
    Main training script. Uses PyTorch Lightning loggers instead of MLflow.

    Inputs
    --------
    ptycho_dir: Directory of ptychography files. Assumed that all diffraction pattern dimensions are equal, and the formatting is identical
                Read dataloader.py to get a sense of the formats expected
    config_path: Path to JSON configuration file (optional, for legacy interface)
    existing_config: Tuple of (DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig) if configs already instantiated
    output_dir: Optional override for output directory. If provided, configures trainer's default_root_dir.
    execution_config: Optional PyTorchExecutionConfig for runtime knobs (Phase C4.C3 - ADR-003).
                     If None, uses default execution config with CPU accelerator.
    run_name: Optional custom name for this run. If None, uses timestamp.

    Outputs
    --------
    run_dir: Path to run directory containing checkpoints, configs, and logs
    '''
    try:
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

        #Setting global run name
        if run_name is None:
            import time
            from datetime import datetime
            os.makedirs(output_dir or "training_outputs", exist_ok=True)
            run_name_file = os.path.join(output_dir or "training_outputs", ".run_name")

            if is_effectively_global_rank_zero():
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(run_name_file, 'w') as f:
                    f.write(run_name)
                os.environ["SHARED_RUN_NAME"] = run_name
                print(f"[Rank 0] Generated run name: {run_name}")
            else:
                max_wait = 30
                for _ in range(max_wait * 10):
                    if "SHARED_RUN_NAME" in os.environ:
                        run_name = os.environ["SHARED_RUN_NAME"]
                        break
                    if os.path.exists(run_name_file):
                        with open(run_name_file, 'r') as f:
                            run_name = f.read().strip()
                        if run_name:
                            break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Non-zero rank failed to get run name from rank 0")
                print(f"[Rank {os.environ.get('RANK', '?')}] Using run name: {run_name}")

        resolve_n_devices(training_config)

        #Setting seed
        set_seed(42, n_devices = training_config.n_devices)

        # Data module in place of pytorch native dataloaders. Data module is a lightning class
        # Create DataModule
        print("Creating data module")
        data_module = PtychoDataModuleLightning(
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

        # NEW: Create experiment loggers (replaces MLflow)
        output_dir = output_dir or training_config.output_dir
        tb_logger, csv_logger = create_experiment_loggers(
            experiment_name=training_config.experiment_name,
            run_name=run_name,
            output_dir=output_dir,
        )

        # Configure checkpoint directory (under run directory)
        checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # NEW: Create custom callbacks for config and metadata logging
        config_logger = ConfigLogger(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            inference_config=inference_config,
            datagen_config=datagen_config,
        )
        
        metadata_logger = MetadataLogger(
            stage="training",
            notes=training_config.notes,
            model_name=training_config.model_name,
            encoder_frozen=False,
        )

        #Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=val_loss_label,
            mode='min',
            save_top_k=1,
            filename='best-checkpoint',
            save_last=True,
            verbose=True,  # Add verbose to see what's happening
            save_on_train_epoch_end=False,  # CRITICAL: Save on validation end, not train end
        )

        # Stop early when validation loss stops improving
        early_stop_callback = EarlyStopping(
            monitor=val_loss_label,
            mode='min',
            patience=100,
            verbose=True,
            strict=True
        )

        # Define a theme (optional)
        # progress_bar = RichProgressBar(
        #     theme=RichProgressBarTheme(
        #         description="green_yellow",
        #         progress_bar="green1",
        #         progress_bar_finished="green1",
        #         batch_progress="green_yellow",
        #         time="grey82",
        #         processing_speed="grey82",
        #         metrics="white"
        #         )
        # )

        progress_bar = TQDMProgressBar(refresh_rate=10)
        
        callbacks = [
            checkpoint_callback,
            early_stop_callback,
            config_logger,
            metadata_logger,
            progress_bar,
        ]

        # All training now runs single-stage
        total_training_epochs = training_config.epochs

        # # Resolve accelerator (prefer execution_config, fallback to training_config logic)
        # if execution_config.accelerator == 'auto':
        #     resolved_accelerator = 'gpu' if training_config.n_devices > 0 and torch.cuda.is_available() else 'cpu'
        # else:
        #     resolved_accelerator = execution_config.accelerator
        #     # Map 'cuda' alias to 'gpu' for Lightning compatibility
        #     if resolved_accelerator == 'cuda':
        #         resolved_accelerator = 'gpu'

        # Create trainer with execution config knobs and NEW loggers
        trainer = L.Trainer(
            max_epochs = total_training_epochs,
            default_root_dir = str(Path(output_dir)),
            devices = training_config.n_devices,
            accelerator = 'gpu',
            callbacks = callbacks,
            strategy=get_training_strategy(training_config.n_devices, training_config.strategy),
            check_val_every_n_epoch=1,  # Validate every epoch
            enable_checkpointing=True,  # Enable checkpointing for early stopping
            enable_progress_bar=True,  # Controlled by execution config
            deterministic=False,  # Reproducibility control
            logger=[tb_logger, csv_logger],  # NEW: Use Lightning loggers
        )

        #Train the model
        # if is_effectively_global_rank_zero():
        #     print(f'[Rank {trainer.global_rank}] Beginning model training/final data prep...')
        
        trainer.fit(model, datamodule = data_module)
        
        if is_effectively_global_rank_zero():
            print(f'[Rank {trainer.global_rank}] Done training.')
            
            # NEW: Print run summary (replaces print_auto_logged_info)
            run_dir = Path(trainer.log_dir)
            print_run_summary(run_dir)
            
            # Store run directory for return
            training_run_dir = run_dir

        #Fine-tune if applicable (encoder frozen default)
        if training_config.epochs_fine_tune > 0:
            print("Starting fine-tuning phase...")
            
            #Fine-tuning
            fine_tuner = ModelFineTuner(model, data_module, training_config)
            fine_tune_run_dir = fine_tuner.fine_tune(
                experiment_name=training_config.experiment_name,
                output_dir=output_dir,
            )
            
            if is_effectively_global_rank_zero() and fine_tune_run_dir:
                print_run_summary(fine_tune_run_dir)

        # CRITICAL: Final synchronization before returning
        if dist.is_initialized():
            print(f'[Rank {trainer.global_rank}] Waiting at final barrier before return...')
            dist.barrier()
            print(f'[Rank {trainer.global_rank}] ✓ Passed final barrier, about to return')

        if is_effectively_global_rank_zero():
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Run directory: {training_run_dir}")
            print(f"Checkpoints: {checkpoint_dir}")
            print(f"Best checkpoint: {find_best_checkpoint(training_run_dir)}")
            print(f"TensorBoard: tensorboard --logdir {training_run_dir / 'logs'}")
            print(f"{'='*60}\n")
            
            return training_run_dir
        else:
            print(f'[Rank {trainer.global_rank}] Non-zero rank returning None')
            return None
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Shutting down DDP gracefully...")
        # This is the secret sauce: 
        # It tells the torch distributed backend to release the ports and GPUs
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Force exit the process to prevent the "hang"
        sys.exit(0)

    except Exception as e:
        print(f"Training failed: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)


    #  # 2. Final synchronization
    # if torch.distributed.is_initialized():
    #     # Use a try/except to prevent core dumps on cleanup
    #     try:
    #         torch.distributed.barrier()
    #         # Destroys the process group cleanly
    #         torch.distributed.destroy_process_group()
    #     except Exception as e:
    #         print(f"Cleanup error on Rank {trainer.global_rank}: {e}")

    # # 3. ONLY NOW return or exit
    # if is_effectively_global_rank_zero():
    #     return training_run_dir
    
    # return None
    

    #Define main function
if __name__ == '__main__':
    #Parsing
    parser = argparse.ArgumentParser(description = "Run training for ptycho_torch")
    #Arguments
    parser.add_argument('--ptycho_dir', type = str, help = 'Path to ptycho directory')
    parser.add_argument('--config', type = str, default=None, help = 'Path to JSON configuration file (mandatory)')
    parser.add_argument('--output_dir', type = str, default=None, help = 'Path to output directory')
    
    #Parse
    args = parser.parse_args()

    #Assign to vars
    ptycho_dir = args.ptycho_dir
    config_path = args.config
    output_dir = args.output_dir

    print(f"Ptycho directory: {ptycho_dir}")
    print(f"Configuration file: {config_path}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        main(ptycho_dir, config_path, False, output_dir)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
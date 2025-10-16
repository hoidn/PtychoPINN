#Most basic modules
import sys
import argparse
import os
import json
import random
import math

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
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

#MLFlow
import mlflow.pytorch
from mlflow import MlflowClient

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
         existing_config = None):
    '''
    Main training script. Can provide dictionary of modified configs based off of dataclass attributes to overwrite

    Inputs
    --------
    ptycho_dir: Directory of ptychography files. Assumed that all diffraction pattern dimensions are equal, and the formatting is identical
                Read dataloader.py to get a sense of the formats expected
    *_config_replace: Dictionary updating parameters for dataclasses of the corresponding type.
                Check config_params.py for details

    Outputs
    --------
    run_id: Can be recycled to load things through MLFlow. See inference.py for details
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
    #Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=val_loss_label,
        mode='min',
        save_top_k=1,
        filename='best-checkpoint'
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
        default_root_dir = os.path.dirname(os.getcwd()),
        devices = training_config.n_devices,
        accelerator = 'gpu',
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

    mlflow.pytorch.autolog(checkpoint_monitor = val_loss_label)

    #Train the model
    # if trainer.is_global_zero:
    if is_effectively_global_rank_zero():
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
        print(f"Training run_id: {run_ids['training']}")
        if 'fine-tune' in run_ids:
            print(f"Fine_tune run_id: {run_ids['fine-tune']}")
        return run_ids
    else:
        print(f'[Rank {trainer.global_rank}] Non-zero rank returning None')
        return None



#Define main function
if __name__ == '__main__':
    #Parsing
    parser = argparse.ArgumentParser(description = "Run training for ptycho_torch")
    #Arguments
    parser.add_argument('--ptycho_dir', type = str, help = 'Path to ptycho directory')
    parser.add_argument('--config', type = str, default=None, help = 'Path to JSON configuration file (mandatory)')
    
    #Parse
    args = parser.parse_args()

    #Assign to vars
    ptycho_dir = args.ptycho_dir
    config_path = args.config

    print(f"Ptycho directory: {ptycho_dir}")
    print(f"Configuration file: {config_path}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        main(ptycho_dir, config_path, False)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

from datetime import datetime
import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from ptycho_torch.api.base_api import PtychoModel, ConfigManager, Orchestration
from typing import Optional, Dict, Any, Tuple, Union, Protocol
from ptycho_torch.config_params import TrainingConfig
from ptycho_torch.train_utils import get_training_strategy, LightningConfigSaveCallback
from lightning.pytorch.strategies import DDPStrategy

def setup_lightning_trainer(ptycho_model: PtychoModel,
                            config_manager: ConfigManager,
                            orchestration: Orchestration,
                            output_dir = 'lightning_outputs') -> L.Trainer:
    
    """
    
    """
    
    val_loss_label = ptycho_model.model.val_loss_name #Nested models because Ptycho

    #Callback setup

    checkpoint_callback = ModelCheckpoint(
        monitor = val_loss_label,
        mode = 'min',
        save_top_k = 1,
        filename = 'best-checkpoint'
    )

    early_stop_callback = EarlyStopping(
        monitor = val_loss_label,
        mode='min',
        patience=5,
        verbose = True,
        strict = True
    )

    config_dict = {
        "data_config": config_manager.data_config,
        "model_config": config_manager.model_config,
        "training_config": config_manager.training_config,
        "inference_config": config_manager.inference_config
    }

    #Setting up saving callback
    training_config = config_manager.training_config

    # Callbacks
    config_sync_callback = LightningConfigSaveCallback(
        config_map=config_dict, 
        base_output_dir=output_dir
    )

    if orchestration == 'mlflow':
        callback_list = [checkpoint_callback,
                         early_stop_callback]
        include_logger = True

    elif orchestration == 'lightning':
        callback_list = [checkpoint_callback,
                         early_stop_callback,
                         config_sync_callback]
        include_logger = False

    trainer = L.Trainer(
        max_epochs=training_config.epochs,
        devices=training_config.n_devices,
        accelerator='gpu',
        strategy= get_training_strategy(training_config.n_devices),
        callbacks=callback_list,
        enable_checkpointing=True,
        enable_progress_bar=True,
        logger=include_logger,  # No logger for clean test
    )

    return trainer, output_dir

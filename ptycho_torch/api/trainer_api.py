import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from ptycho_torch.api.base_api import PtychoModel
from typing import Optional, Dict, Any, Tuple, Union, Protocol
from ptycho_torch.config_params import TrainingConfig
from ptycho_torch.train_utils import get_training_strategy

def setup_lightning_trainer(ptycho_model: PtychoModel,
                            training_config: TrainingConfig) -> L.Trainer:
    
    val_loss_label = ptycho_model.model.val_loss_name #Nested models because Ptycho

    checkpoint_callback = ModelCheckpoint(
        monitor = val_loss_label,
        mode = 'min',
        save_top_k = 1,
        filename = 'best-checkpoint'
    )

    early_stop_callback = EarlyStopping(
        monitor = val_loss_label,
        mode='min',
        patience=100,
        verbose = True,
        strict = True
    )

    total_epochs = training_config.epochs

    #Instantiate lightning trainer
    trainer = L.Trainer(
        max_epochs = total_epochs,
        devices = training_config.n_devices,
        accelerator = 'gpu',
        callbacks = [checkpoint_callback, early_stop_callback],
        strategy = get_training_strategy(training_config.n_devices),
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        enable_progress_bar=True
    )

    return trainer

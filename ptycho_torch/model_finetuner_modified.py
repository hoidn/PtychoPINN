# Modified ModelFineTuner class for train_utils.py
# This replaces the existing ModelFineTuner to work without MLflow

import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ptycho_torch.config_params import TrainingConfig
from ptycho_torch.train_utils import get_training_strategy, is_effectively_global_rank_zero

# Import our new utilities
from ptycho_torch.lightning_utils import (
    ConfigLogger,
    MetadataLogger,
    create_experiment_loggers,
    print_run_summary,
)


class ModelFineTuner:
    """
    Handles fine-tuning/re-instantiation of training parameters, etc.
    
    Modified to use PyTorch Lightning loggers instead of MLflow.
    
    Workflow:
    1. Encoder Freeze
    2. Create new run directory with metadata indicating fine-tuning stage
    3. Creates new trainer with Lightning loggers
    """
    def __init__(self, model,
                 train_module,
                 training_config: TrainingConfig):
        
        self.model = model
        self.train_module = train_module
        self.training_config = training_config

    def fine_tune(self, 
                  experiment_name="PtychoPINN synthetic",
                  output_dir="training_outputs",
                  run_name=None):
        """
        Fine-tune the model with frozen encoder.
        
        Args:
            experiment_name: Name of experiment
            output_dir: Root output directory
            run_name: Optional custom run name (auto-generated if None)
            
        Returns:
            Path to fine-tuning run directory (rank 0 only), or None
        """

        #Freeze encoder
        self.model.freeze_encoder()

        #Calculate modified lr
        fine_tuning_lr = self.model.lr * self.training_config.fine_tune_gamma
        #Update lr
        self.model.lr = fine_tuning_lr

        # NEW: Create loggers for fine-tuning run
        if run_name is None:
            from datetime import datetime
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_finetune"
        
        tb_logger, csv_logger = create_experiment_loggers(
            experiment_name=experiment_name,
            run_name=run_name,
            output_dir=output_dir,
        )
        
        # Configure checkpoint directory
        checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # NEW: Create custom callbacks for fine-tuning
        config_logger = ConfigLogger(
            data_config=self.model.data_config,
            model_config=self.model.model_config,
            training_config=self.model.training_config,
            inference_config=self.model.inference_config,
            datagen_config=None,  # May not be available in fine-tuning
        )
        
        metadata_logger = MetadataLogger(
            stage="fine_tuning",
            notes=f"Fine-tuning with LR={fine_tuning_lr}, encoder_frozen=True",
            model_name=self.training_config.model_name,
            encoder_frozen=True,
        )

        #Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=self.model.val_loss_name,
            mode='min',
            save_top_k=1,
            filename='best-checkpoint-finetune',
            save_last=True,
        )
        
        early_stop_callback = EarlyStopping(
            monitor=self.model.val_loss_name, 
            patience=5,  # Shorter patience for fine-tuning
            mode='min',
            verbose=True,
            strict=True
        )
        
        callbacks = [
            checkpoint_callback,
            early_stop_callback,
            config_logger,
            metadata_logger,
        ]

        #Same trainer except epochs and callbacks
        fine_tune_trainer = L.Trainer(
            max_epochs = self.training_config.epochs_fine_tune,
            default_root_dir = str(Path(output_dir)),
            devices = self.training_config.n_devices,
            accelerator = 'gpu',
            callbacks = callbacks,
            accumulate_grad_batches=1,
            strategy=get_training_strategy(self.training_config.n_devices, self.training_config.strategy),
            check_val_every_n_epoch=1,  # Validate every epoch during fine-tuning
            enable_checkpointing=True,
            logger=[tb_logger, csv_logger],  # NEW: Use Lightning loggers
        )

        # Train
        if is_effectively_global_rank_zero():
            print(f'[Rank {fine_tune_trainer.global_rank if hasattr(fine_tune_trainer, "global_rank") else "N/A"}] Fine-tuning model with learning rate {fine_tuning_lr}...')
        
        fine_tune_trainer.fit(self.model, datamodule = self.train_module)

        if is_effectively_global_rank_zero():
            fine_tune_run_dir = Path(fine_tune_trainer.log_dir)
            print(f'Fine-tuning complete. Run directory: {fine_tune_run_dir}')
            return fine_tune_run_dir
        else:
            return None
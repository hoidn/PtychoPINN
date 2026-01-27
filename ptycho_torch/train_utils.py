#Most basic modules
import sys
import argparse
import os
import json
import random
import math
import warnings
import dataclasses

#Typing
from dataclasses import asdict
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig

#ML libraries
import numpy as np
import torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
import torch.distributed as dist
import mlflow.pytorch
from mlflow import MlflowClient
from torch.optim.lr_scheduler import _LRScheduler

#Automation modules
#Lightning
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.strategies import DDPStrategy

#Dataloader
from ptycho_torch.dataloader import TensorDictDataLoader, PtychoDataset, Collate_Lightning

#Custom modules
from ptycho_torch.utils import config_to_json_serializable_dict


#Helper function for mlflow
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def set_seed(seed=42, n_devices=1):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (one GPU)

    if n_devices > 1:
        torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)
        L.seed_everything(seed, workers = True) #For lightning DDP

    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed

def get_training_strategy(n_devices):
    """
    Dynamically returns training strategy based on number of GPUs. Big distinction between
    1 gpu and > 1 GPU(s)

    Args:
        n_devices: Number of GPUs being trained on
    
    """
    if n_devices <= 1:
        return 'auto'
    
    elif n_devices >= 2:
        return DDPStrategy(find_unused_parameters = False,
                           static_graph=True,
                           gradient_as_bucket_view=True,
                           process_group_backend='nccl')
    
def find_learning_rate(base_lr, n_devices, batch_size_per_gpu):
    """
    Scales LR based on effective batch size (EBS), where EBS = bs_per_gpu * n_devices
    Uses sqrt LR scaling law based on Krizhevsky, 2014, Hoffer et al., 2017)
    """
    ebs = batch_size_per_gpu * n_devices
    lr_scaled = base_lr * math.sqrt(ebs / batch_size_per_gpu)

    return lr_scaled


def compute_grad_norm(parameters, norm_type=2.0):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type) if total > 0.0 else 0.0

def log_parameters_mlflow(data_config: DataConfig,
                          model_config: ModelConfig,
                          training_config: TrainingConfig,
                          inference_config: InferenceConfig,
                          datagen_config: DatagenConfig):
    # Log configuration parameters
    print('Logging configuration parameters to MLflow...')
    # Assuming the config objects have a .get_settings() method returning a dict
    try:
        # Use specific prefixes for clarity
        # Log each config as a single JSON string parameter
        mlflow.log_param("DataConfig_params", json.dumps(config_to_json_serializable_dict(data_config)))
        mlflow.log_param("ModelConfig_params", json.dumps(config_to_json_serializable_dict(model_config)))
        mlflow.log_param("TrainingConfig_params", json.dumps(config_to_json_serializable_dict(training_config)))
        mlflow.log_param("InferenceConfig_params", json.dumps(config_to_json_serializable_dict(inference_config)))
        mlflow.log_param("DatagenConfig_params", json.dumps(config_to_json_serializable_dict(datagen_config)))
        print('Configuration parameters logged.')
    except AttributeError:
        print("Warning: Could not log config parameters. '.get_settings()' method might be missing.")
    except Exception as e:
        print(f"Warning: Error logging config parameters: {e}")

def is_effectively_global_rank_zero():
    """
    Checks if current process is global rank 0 when ddp not initialized yet
    """
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    
    return True

# Other classes

class LightningConfigSaveCallback(Callback):
    def __init__(self, config_map: dict, base_output_dir: str):
        super().__init__()
        self.config_map = config_map
        self.base_output_dir = base_output_dir
        self.run_dir = None

    def setup(self, trainer, pl_module, stage=None):
        from datetime import datetime
        
        # 1. Define the unique run directory
        self.run_dir = self.base_output_dir #os.path.join(self.base_output_dir, f"run_{timestamp}")
        config_dir = os.path.join(self.run_dir, "configs")
        checkpoint_dir = os.path.join(self.run_dir, "checkpoints")

        # 2. Rank 0 creates the directory structure
        if trainer.global_rank == 0:
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"[Rank 0] Created unique run directory: {self.run_dir}")

        # 3. Update the ModelCheckpoint callback to point to this new unique folder
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = checkpoint_dir

    def on_train_start(self, trainer, pl_module):
        # Only Rank 0 writes the JSON files
        if trainer.global_rank == 0:
            config_dir = os.path.join(self.run_dir, "configs")
            for name, cfg_instance in self.config_map.items():
                file_path = os.path.join(config_dir, f"{name}.json")
                
                # Use dataclasses.asdict for clean serialization
                cfg_dict = dataclasses.asdict(cfg_instance)
                serializable_dict = self._make_serializable(cfg_dict)
                
                with open(file_path, 'w') as f:
                    json.dump(serializable_dict, f, indent=4)

    def _make_serializable(self, d):
        """Recursively converts tensors/non-JSON types to primitives."""
        for k, v in d.items():
            if torch.is_tensor(v):
                d[k] = v.tolist() if v.numel() < 100 else f"Tensor(shape={list(v.shape)})"
            elif isinstance(v, dict):
                self._make_serializable(v)
            elif isinstance(v, (tuple, list)):
                d[k] = [x.tolist() if torch.is_tensor(x) else x for x in v]
        return d

#-----Fine-tuning------
class ModelFineTuner:
    """
    Handles fine-tuning/re-instantiation of training parameters, etc.
    Calls
    1. Encoder Freeze
    2. Adds a few additional parameters to logger, such as fine_tuning = True, encoder_frozen = True
    3. Creates new trainer
    """
    def __init__(self, model,
                 train_module,
                 training_config: TrainingConfig):
        
        self.model = model
        self.train_module = train_module
        self.training_config = training_config

    def fine_tune(self, experiment_name = "PtychoPINN synthetic"):

        #Freeze encoder
        self.model.freeze_encoder()

        #Callbacks
        callbacks = [
            EarlyStopping(
                monitor=self.model.val_loss_name, 
                patience=5,  # Shorter patience for fine-tuning
                mode='min',
                verbose=True,
                strict=True
            )
        ]
        
        #Calculate modified lr
        fine_tuning_lr = self.model.lr * self.training_config.fine_tune_gamma
        #Update lr
        self.model.lr = fine_tuning_lr

        #Same trainer except epochs
        fine_tune_trainer = L.Trainer(
            max_epochs = self.training_config.epochs_fine_tune,
            default_root_dir = os.path.dirname(os.getcwd()),
            devices = self.training_config.n_devices,  # FIXED: Use explicit n_devices instead of 'auto'
            accelerator = 'gpu',
            callbacks = callbacks,
            accumulate_grad_batches=1,
            strategy=get_training_strategy(self.training_config.n_devices),
            check_val_every_n_epoch=1,  # Validate every epoch during fine-tuning
            enable_checkpointing=True,
        )

        # Store the run_id before starting training (for proper scope)
        fine_tune_run_id = None

        #Start mlflow logging and runs for this "new run"
        #Saves a separate MLFlow run instance so we can compare fine-tuned vs non fine-tuned model
        if is_effectively_global_rank_zero():
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run() as run:
                # Store the run_id
                fine_tune_run_id = run.info.run_id
                
                #Log
                print("Logging fine-tuning configuration parameters to MLFlow...")
                try:
                    # Log each config as a single JSON string parameter
                    log_parameters_mlflow(self.model.data_config, self.model.model_config,
                                        self.model.training_config, self.model.inference_config)
                    mlflow.log_param("fine_tuning", True)
                    mlflow.log_param("encoder_frozen", True)
                    print('Fine-tuning configuration parameters logged.')
                except Exception as e:
                    print(f"Warning: Error logging config parameters: {e}")

                #Set experiment name and fine-tuning tags
                mlflow.set_tag("stage", "fine_tuning")
                mlflow.set_tag("encoder_frozen", "True")

                print(f'[Rank {fine_tune_trainer.global_rank if hasattr(fine_tune_trainer, "global_rank") else "N/A"}] Fine-tuning model with learning rate {fine_tuning_lr}...')
                fine_tune_trainer.fit(self.model, datamodule = self.train_module)

                print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        else:
            # Non-rank-0 processes still need to participate in training
            print(f'[Rank {fine_tune_trainer.global_rank if hasattr(fine_tune_trainer, "global_rank") else "N/A"}] Fine-tuning model with learning rate {fine_tuning_lr}...')
            fine_tune_trainer.fit(self.model, datamodule = self.train_module)

        # FIXED: Use the consistent rank checking function instead of trainer.is_global_zero
        return fine_tune_run_id if is_effectively_global_rank_zero() else None
    
class ModelFineTuner_Lightning:
    """
    Fine-tuning class specifically for lightning-only training. Works differently enough form mlflow-aided implementation
    I decided to keep them as separate classes. There is likely room for refactoring/class merging but dev time not priority.
    """
    def __init__(self, model, train_module, training_config, run_dir):
        self.model = model
        self.train_module = train_module
        self.training_config = training_config
        self.run_dir = run_dir  # The unique folder created in main_lightning

    def fine_tune(self):
        print(f"\n[Rank {self.model.global_rank}] Starting Fine-Tuning Stage...")

        # 1. Freeze encoder (Implementation depends on your model architecture)
        if hasattr(self.model, 'freeze_encoder'):
            self.model.freeze_encoder()
        else:
            # Generic fallback: freeze parameters starting with 'encoder'
            for name, param in self.model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False
            print("Encoder parameters frozen manually.")

        # 2. Update Learning Rate
        fine_tuning_lr = self.model.lr * self.training_config.fine_tune_gamma
        self.model.lr = fine_tuning_lr
        print(f"Fine-tuning LR set to: {fine_tuning_lr}")

        # 3. Setup Fine-tuning specific checkpointing
        ft_ckpt_path = os.path.join(self.run_dir, "finetune_checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ft_ckpt_path,
            monitor=self.model.val_loss_name,
            mode='min',
            save_top_k=1,
            filename='best-finetune-checkpoint'
        )

        callbacks = [
            EarlyStopping(monitor=self.model.val_loss_name, patience=5, mode='min', verbose=True),
            checkpoint_callback
        ]

        # 4. Create a fresh trainer for fine-tuning
        fine_tune_trainer = L.Trainer(
            max_epochs=self.training_config.epochs_fine_tune,
            devices=self.training_config.n_devices,
            accelerator='gpu',
            strategy=DDPStrategy(
                find_unused_parameters=False,
                static_graph=True,
                gradient_as_bucket_view=True,
                process_group_backend='nccl'
            ) if self.training_config.n_devices > 1 else 'auto', # Reuse the strategy from the previous trainer
            callbacks=callbacks,
            enable_checkpointing=True,
            # Use CSVLogger in a subfolder for fine-tuning logs
            logger=L.pytorch.loggers.CSVLogger(save_dir=self.run_dir, name="logs_finetune"),
        )

        # 5. Fit
        fine_tune_trainer.fit(self.model, datamodule=self.train_module)
        
        print(f"[Rank {self.model.global_rank}] Fine-tuning complete.")
        return ft_ckpt_path if self.model.global_rank == 0 else None
        
# --- Lightning Data Classes ---
class PtychoDataModule(L.LightningDataModule):
    """

    This data module class is necessary due to DDP (distributed data parallel) when multiple GPUs
    are used for training. The PtychoDataset method itself is set to only work on rank 0.
    
    The dataset is created once trainer.fit is called.
    
    """
    def __init__(self, ptycho_dir: str, model_config: ModelConfig, data_config: DataConfig,
                 training_config: TrainingConfig, initial_remake_map: bool = True,
                 val_split: float = 0.1, val_seed: int = 42,
                 memory_map_dir: str = 'data/memmap'):
        super().__init__()
        self.ptycho_dir = ptycho_dir
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.initial_remake_map = initial_remake_map # Flag for the very first creation
        self.val_split = val_split  # Fraction of data to use for validation
        self.val_seed = val_seed    # Seed for reproducible train/val split
        self.memory_map_dir = memory_map_dir

        #Self state tracking
        self._is_setup_done = False

    def prepare_data(self):
        # Called once per node on global rank 0.
        if self.training_config.orchestrator == 'Mlflow':
            print(f"[DataModule prepare_data] Global Rank: {self.trainer.global_rank if self.trainer else 'N/A'}. Creating/Verifying map.")
        elif self.training_config.orchestrator == 'Lightning':
            #Check if rank 0 setup has been done already, this will be called when fine-tuning after training
            if self._is_setup_done:
                return
            if self.initial_remake_map:
                print("[Rank 0] Creating memory map...")
                # Create dataset to generate map files
                _ = PtychoDataset(
                    ptycho_dir=self.ptycho_dir,
                    model_config=self.model_config,
                    data_config=self.data_config,
                    training_config=self.training_config,
                    remake_map=True
                )
                print("[Rank 0] Memory map created.")
    
    def setup(self, stage: str = None):

        if self.training_config.orchestrator == 'Mlflow':
            # Called on every GPU.
            # `remake_map` is True for the first "iteration" because of how Mlflow handles memory map creation
            # memory map creation happens in rank 0 "setup", not in prepare_data
            print(f"[DataModule setup] Stage: {stage}, Global Rank: {self.trainer.global_rank if self.trainer else 'N/A'}. Loading map.")
            remake_flag_for_this_setup = self.initial_remake_map
            if hasattr(self, '_setup_has_run_once') and self._is_setup_done:
                remake_flag_for_this_setup = False #Don't remake if it has run before on rank 0

        elif self.training_config.orchestrator == 'Lightning':
            remake_flag_for_this_setup = False
            if self._is_setup_done:
                print(f"[Rank {self.trainer.global_rank}] Skipping redundant data setup.")
                return
        
        if stage == "fit" or stage is None:
            if not hasattr(self, 'train_dataset'):
                print("Creating dataset...")
                full_dataset = PtychoDataset(
                    ptycho_dir=self.ptycho_dir,
                    model_config=self.model_config,
                    data_config=self.data_config,
                    training_config=self.training_config,
                    remake_map=remake_flag_for_this_setup, # Always False here, map should exist
                    data_dir = self.memory_map_dir
                )

                # Create train/validation split
                dataset_size = len(full_dataset)
                val_size = int(self.val_split * dataset_size)
                train_size = dataset_size - val_size
                
                print(f"Dataset split: Total={dataset_size}, Train={train_size}, Val={val_size}")
                
                # Use generator for reproducible split
                generator = torch.Generator().manual_seed(self.val_seed)
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size], generator=generator
                )

        self._is_setup_done = True

    def train_dataloader(self):
            return TensorDictDataLoader(
                self.train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=self.training_config.num_workers,
                collate_fn=Collate_Lightning(pin_memory_if_cuda=True), # Lightning handles device placement
                pin_memory=True, # Lightning DDP often benefits from this
                persistent_workers=True,
                prefetch_factor = 4,
            )
    def val_dataloader(self):
        return TensorDictDataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.training_config.num_workers,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor = 4,
        )
    
class PrebuiltPtychoDataModule(L.LightningDataModule):
    def __init__(self, map_path,
                 model_config, data_config, training_config):
        super().__init__()
        self.map_path = map_path
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Setup that respects rank separation"""

        from ptycho_torch.dataloader import get_current_rank, is_ddp_initialized_and_active
        
        # Only create dataset once per rank
        if self.dataset is None:
            if stage == "fit" or stage is None:
                
                # Rank-aware dataset creation
                current_rank = get_current_rank()
                is_ddp_active = is_ddp_initialized_and_active()
                
                print(f"[DataModule setup] Rank {current_rank}: Loading existing memory map")
                
                # Create dataset with NO setup logic - just load existing map
                self.dataset = PtychoDataset.from_existing_map(
                    self.map_path, 
                    self.model_config,
                    self.data_config,
                    current_rank=current_rank,
                    is_ddp_active=is_ddp_active
                )
                
                # Train/val split (all ranks do this identically)
                dataset_size = len(self.dataset)
                val_size = int(0.1 * dataset_size)
                train_size = dataset_size - val_size
                
                # Use same seed for reproducible split across ranks
                generator = torch.Generator().manual_seed(42)
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.dataset, [train_size, val_size], generator=generator
                )
                
                print(f"[DataModule setup] Rank {current_rank}: Dataset ready, "
                      f"train={train_size}, val={val_size}")

    def train_dataloader(self):
        return TensorDictDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return TensorDictDataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
# Schedulers
class MultiStageLRScheduler(_LRScheduler):
    """
    CURRENTLY UNUSED.
    Learning rate scheduler to dynamically change the normalization factor.
    
    Stage 1: Base LR (RMS normalization - stable)
    Stage 2: Reduced LR with smooth transitions (mixed loss - unstable)
    Stage 3: Further reduced LR (physics normalization - different optimization landscape)
    
    The scheduler reduces LR at stage transitions to handle the changing loss landscape,
    especially important when transitioning to physics-based normalization.
    """
    
    def __init__(self, optimizer, stage_1_epochs, stage_2_epochs, stage_3_epochs,
                 stage_3_lr_factor=0.1, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: PyTorch optimizer
            stage_1_epochs: Number of epochs for stage 1 (RMS only)
            stage_2_epochs: Number of epochs for stage 2 (transition)
            stage_3_epochs: Number of epochs for stage 3 (physics only)
            stage_3_lr_factor: LR multiplier for stage 3 (0.1 = 10% of original LR)
            lr_transition_epochs: Number of epochs to smoothly transition LR at stage boundaries
            last_epoch: Last epoch index
            verbose: Whether to print LR changes
        """
        
        # Epochs/Learning Gamma
        self.stage_1_epochs = stage_1_epochs
        self.stage_2_epochs = stage_2_epochs
        self.stage_3_epochs = stage_3_epochs
        self.stage_3_lr_factor = stage_3_lr_factor
        
        # Calculate stage boundaries
        self.stage_1_end = stage_1_epochs
        self.stage_2_end = stage_1_epochs + stage_2_epochs
        self.stage_3_end = stage_1_epochs + stage_2_epochs + stage_3_epochs
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        current_epoch = self.last_epoch
        lrs = []
        
        for base_lr in self.base_lrs:
            lr = self._calculate_lr_for_epoch(current_epoch, base_lr)
            lrs.append(lr)
        
        return lrs
    
    def _calculate_lr_for_epoch(self, epoch, base_lr):
        
        # Stage 1: Full learning rate
        if epoch < self.stage_1_end:
            return base_lr
        
        # Stage 2: Cosine transition from base_lr to base_lr * stage_3_lr_factor
        elif epoch < self.stage_2_end:
            if self.stage_2_epochs == 0:
                return base_lr * self.stage_3_lr_factor
            
            progress = (epoch - self.stage_1_end) / self.stage_2_epochs
            progress = min(1.0, progress)  # Clamp to [0, 1]
            
            # Cosine interpolation from base_lr to base_lr * stage_3_lr_factor
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr = base_lr * (cosine_factor + (1 - cosine_factor) * self.stage_3_lr_factor)
            #Skipping regular lr for now
            return base_lr * self.stage_3_lr_factor
        
        # Stage 3: Fixed reduced learning rate
        else:
            return base_lr * self.stage_3_lr_factor
    
    def get_current_stage(self):
        """Get current training stage for logging"""
        epoch = self.last_epoch
        
        if epoch < self.stage_1_end:
            return 1
        elif epoch < self.stage_2_end:
            return 2
        else:
            return 3


class AdaptiveLRScheduler(_LRScheduler):
    """
    CURRENTLY UNUSED
    Alternative scheduler that adapts LR based on physics weight during Stage 2.
    As physics weight increases, LR decreases to handle the more challenging loss landscape.
    """
    
    def __init__(self, optimizer, lightning_module, base_stage_2_lr_factor=0.5, 
                 min_stage_2_lr_factor=0.1, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: PyTorch optimizer
            lightning_module: PtychoPINN_Lightning module to get physics weight
            base_stage_2_lr_factor: Base LR factor for stage 2 start
            min_stage_2_lr_factor: Minimum LR factor when physics weight = 1.0
        """
        self.lightning_module = lightning_module
        self.base_stage_2_lr_factor = base_stage_2_lr_factor
        self.min_stage_2_lr_factor = min_stage_2_lr_factor
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """Calculate adaptive learning rate based on current training stage and physics weight"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        # Get current stage and physics weight from lightning module (fallback to single-stage defaults)
        if hasattr(self.lightning_module, 'get_current_stage_and_weight'):
            stage, physics_weight = self.lightning_module.get_current_stage_and_weight()
        else:
            stage = 1
            physics_weight = 1.0 if getattr(self.lightning_module, 'torch_loss_mode', 'poisson') == 'poisson' else 0.0
        
        lrs = []
        for base_lr in self.base_lrs:
            if stage == 1:
                # Stage 1: Full LR
                lr = base_lr
            elif stage == 2:
                # Stage 2: Adaptive LR based on physics weight
                # As physics weight increases (0->1), LR decreases
                lr_factor = self.base_stage_2_lr_factor - (self.base_stage_2_lr_factor - self.min_stage_2_lr_factor) * physics_weight
                lr = base_lr * lr_factor
            else:  # Stage 3
                # Stage 3: Minimum LR for fine-tuning
                lr = base_lr * self.min_stage_2_lr_factor
            
            lrs.append(lr)
        
        return lrs

#Most basic modules
import sys
import argparse
import os
import json
import random
import math
import warnings
import dataclasses
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
import mlflow.pytorch
from mlflow import MlflowClient
from torch.optim.lr_scheduler import _LRScheduler

#Automation modules
#Lightning
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.strategies import DDPStrategy

#Dataloader
from ptycho_torch.dataloader import (
    Collate_Lightning,
    PtychoDataset,
    TensorDictDataLoader,
    build_ptycho_loader,
)

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

def is_spawn_strategy(strategy) -> bool:
    """Check whether the given strategy uses ddp_spawn (mp.spawn-based parallelism)."""
    if isinstance(strategy, str):
        return 'spawn' in strategy
    if isinstance(strategy, DDPStrategy):
        return getattr(strategy, '_start_method', None) == 'spawn'
    return False

def get_training_strategy(strategy='auto', n_devices=None, accelerator='cuda'):
    """
    Returns the Lightning training strategy.

    If `strategy == 'auto'`, dynamically selects based on number of GPUs:
      - 1 GPU  -> 'auto'
      - 2+ GPUs -> DDPStrategy with sensible defaults
    If `strategy == 'ddp_spawn'`, returns a DDPStrategy with start_method='spawn'
    for spawn-based parallelism (required for long-running host applications).
    Otherwise, returns `strategy` unchanged so Lightning can interpret it
    (e.g. 'ddp', 'ddp_notebook', or a Strategy instance).

    Args:
        strategy: Requested strategy. Pass 'auto' to auto-select.
        n_devices: Number of devices being trained on (used only when strategy=='auto').
        accelerator: Resolved Lightning accelerator; CPU distributed runs use
            Lightning's default process-group backend instead of NCCL.
    """
    # Backward compatibility: older call sites pass only n_devices.
    if n_devices is None:
        n_devices = strategy
        strategy = 'auto'

    process_group_backend = (
        'nccl' if accelerator in {'cuda', 'gpu'} else None
    )

    if strategy == 'ddp_spawn':
        return DDPStrategy(
            find_unused_parameters=False,
            start_method='spawn',
            process_group_backend=process_group_backend,
        )

    if strategy != 'auto':
        return strategy

    if n_devices <= 1:
        return 'auto'

    return DDPStrategy(
        find_unused_parameters=False,
        process_group_backend=process_group_backend,
    )
    

def adaptive_gradient_clip_(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
    """Adaptive Gradient Clipping (AGC), operating in-place on parameter grads."""
    for p in parameters:
        if p.grad is None:
            continue
        p_norm = p.data.norm(2).clamp(min=eps)
        g_norm = p.grad.data.norm(2)
        max_norm = p_norm * clip_factor
        if g_norm > max_norm:
            p.grad.data.mul_(max_norm / g_norm)


def compute_grad_norm(parameters, norm_type=2.0):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type) if total > 0.0 else 0.0

def resolve_n_devices(training_config):
    """Resolve n_devices='auto' to actual GPU count, mutating in place."""
    if training_config.n_devices == "auto":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        training_config.n_devices = max(count, 1)
        print(f"[resolve_n_devices] auto -> {training_config.n_devices} GPU(s)")
    elif not isinstance(training_config.n_devices, int):
        raise ValueError(f"n_devices must be int or 'auto', got {training_config.n_devices!r}")



def is_effectively_global_rank_zero():
    """
    Checks if current process is global rank 0 when ddp not initialized yet
    """
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    
    return True


class EncoderFreezeCallback(Callback):
    """Single-pass fine-tuning: freezes the encoder and scales LR at a specified epoch.

    This replaces the two-Trainer fine-tuning pattern, making fine-tuning
    compatible with both ddp and ddp_spawn strategies.
    """
    def __init__(self, freeze_at_epoch: int, lr_gamma: float):
        super().__init__()
        self.freeze_at_epoch = freeze_at_epoch
        self.lr_gamma = lr_gamma
        self._frozen = False

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= self.freeze_at_epoch and not self._frozen:
            pl_module.freeze_encoder()
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.lr_gamma
            self._frozen = True
            if is_effectively_global_rank_zero():
                print(f"[EncoderFreezeCallback] Epoch {trainer.current_epoch}: "
                      f"encoder frozen, LR scaled by {self.lr_gamma}")


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
                    remake_map=True,
                    data_dir=self.memory_map_dir,
                    defer_ci_statistics=True,
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
                    data_dir = self.memory_map_dir,
                    defer_ci_statistics=True,
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
                if full_dataset.ci_contract_active:
                    self.ci_statistics = full_dataset.set_ci_statistics_from_indices(
                        self.train_dataset.indices
                    )

        self._is_setup_done = True

    def _resolve_worker_kwargs(self):
        """Returns num_workers and persistent_workers, guarded for ddp_spawn."""
        nw = self.training_config.num_workers
        if is_spawn_strategy(self.training_config.strategy):
            return dict(num_workers=0, persistent_workers=False)
        return dict(num_workers=nw, persistent_workers=nw > 0, prefetch_factor=4)

    def train_dataloader(self):
            return TensorDictDataLoader(
                self.train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
                pin_memory=True,
                **self._resolve_worker_kwargs(),
            )
    def val_dataloader(self):
        return TensorDictDataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=Collate_Lightning(pin_memory_if_cuda=True),
            pin_memory=True,
            **self._resolve_worker_kwargs(),
        )

class PrebuiltPtychoDataModule(L.LightningDataModule):
    """Load exact prebuilt train/validation maps and let Lightning shard once."""

    def __init__(
        self,
        map_path,
        model_config,
        data_config,
        training_config,
        *,
        validation_map_path=None,
        execution_config=None,
        shuffle_training=True,
        torch_training_seed=42,
        validation_fraction=0.1,
        validation_seed=42,
        drop_last_training=False,
        materialize_from=None,
    ):
        super().__init__()
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.map_path = map_path
        self.validation_map_path = validation_map_path
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.execution_config = execution_config
        self.shuffle_training = bool(shuffle_training)
        self.torch_training_seed = int(torch_training_seed)
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.drop_last_training = bool(drop_last_training)
        self.materialize_from = (
            Path(materialize_from) if materialize_from is not None else None
        )
        self.prepare_data_per_node = False
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        if self.materialize_from is None:
            return
        PtychoDataset(
            ptycho_dir=str(self.materialize_from),
            model_config=self.model_config,
            data_config=self.data_config,
            training_config=self.training_config,
            data_dir=str(self.map_path),
            remake_map=True,
            defer_ci_statistics=True,
        )

    def setup(self, stage=None):
        """Load each map once per rank and freeze statistics from train only."""

        from ptycho_torch.dataloader import (
            get_current_rank,
            is_ddp_initialized_and_active,
        )

        if self.dataset is not None or stage not in ("fit", None):
            return
        rank = get_current_rank()
        ddp = is_ddp_initialized_and_active()
        self.dataset = PtychoDataset.from_existing_map(
            self.map_path,
            self.model_config,
            self.data_config,
            current_rank=rank,
            is_ddp_active=ddp,
        )
        if self.validation_map_path is None:
            validation_size = int(self.validation_fraction * len(self.dataset))
            training_size = len(self.dataset) - validation_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                [training_size, validation_size],
                generator=torch.Generator().manual_seed(self.validation_seed),
            )
            training_indices = self.train_dataset.indices
        else:
            self.train_dataset = self.dataset
            self.val_dataset = PtychoDataset.from_existing_map(
                self.validation_map_path,
                self.model_config,
                self.data_config,
                current_rank=rank,
                is_ddp_active=ddp,
            )
            training_indices = torch.arange(len(self.dataset))

        if self.dataset.ci_contract_active:
            self.ci_statistics = self.dataset.set_ci_statistics_from_indices(
                training_indices
            )
            validation_owner = (
                self.val_dataset.dataset
                if isinstance(self.val_dataset, Subset)
                else self.val_dataset
            )
            validation_owner.data_dict["ci_statistics"] = {
                name: value.detach().clone()
                for name, value in self.ci_statistics.items()
            }

    def _loader_settings(self):
        if self.execution_config is None:
            num_workers = self.training_config.num_workers
            if is_spawn_strategy(self.training_config.strategy):
                num_workers = 0
            return {
                "num_workers": num_workers,
                "pin_memory": True,
                "persistent_workers": num_workers > 0,
                "prefetch_factor": 4 if num_workers > 0 else None,
            }
        num_workers = self.execution_config.num_workers
        if is_spawn_strategy(self.execution_config.strategy):
            num_workers = 0
        return {
            "num_workers": num_workers,
            "pin_memory": self.execution_config.pin_memory,
            "persistent_workers": (
                self.execution_config.persistent_workers
                if num_workers > 0
                else False
            ),
            "prefetch_factor": (
                self.execution_config.prefetch_factor
                if num_workers > 0
                else None
            ),
        }

    def train_dataloader(self):
        return build_ptycho_loader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=self.shuffle_training,
            seed=self.torch_training_seed,
            drop_last=self.drop_last_training,
            **self._loader_settings(),
        )

    def val_dataloader(self):
        return build_ptycho_loader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            seed=self.torch_training_seed,
            **self._loader_settings(),
        )


def build_prebuilt_ptycho_datamodule(
    ptycho_dir,
    map_path,
    model_config,
    data_config,
    training_config,
    **module_kwargs,
):
    """Return the study DataModule that materializes one PINN/ones mmap."""

    if model_config.mode != "Unsupervised":
        raise ValueError(
            "prebuilt mmap materialization supports Unsupervised training only"
        )
    if model_config.rect_s1s2_init != "ones":
        raise ValueError(
            "prebuilt mmap materialization requires rect_s1s2_init='ones'"
        )
    if training_config.orchestrator != "Lightning":
        raise ValueError(
            "prebuilt mmap materialization requires orchestrator='Lightning'"
        )

    map_path = Path(map_path)
    return PrebuiltPtychoDataModule(
        map_path,
        model_config,
        data_config,
        training_config,
        materialize_from=ptycho_dir,
        **module_kwargs,
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


class StagedFineTuner_Lightning:
    """
    Handles multi-stage fine-tuning for cross-domain transfer (synthetic -> experimental).
    Only activates when training_config.enable_staged_finetuning = True.

    Three-stage approach:
    - Stage 1: Freeze encoder, train decoder only (adapt object space)
    - Stage 2: Unfreeze top encoder, use discriminative LR (adapt high-level features)
    - Stage 3: Unfreeze all, very conservative LR (optional final refinement)
    """

    def __init__(self, model, train_module, training_config: TrainingConfig,
                 data_config: DataConfig, model_config: ModelConfig,
                 inference_config: InferenceConfig, datagen_config: DatagenConfig,
                 output_dir: str):
        from pathlib import Path
        self.model = model
        self.train_module = train_module
        self.training_config = training_config
        self.data_config = data_config
        self.model_config = model_config
        self.inference_config = inference_config
        self.datagen_config = datagen_config
        self.output_dir = Path(output_dir)
        self.current_stage = 0
        self.stage_checkpoints = {}

        from ptycho_torch.lightning_utils import (
            ConfigLogger, MetadataLogger, create_experiment_loggers, print_run_summary
        )
        self.ConfigLogger = ConfigLogger
        self.MetadataLogger = MetadataLogger
        self.create_experiment_loggers = create_experiment_loggers
        self.print_run_summary = print_run_summary

    def fine_tune(self):
        """Execute all fine-tuning stages."""
        print(f"\n{'='*60}")
        print(f"Starting Staged Fine-tuning for Cross-Domain Transfer")
        print(f"{'='*60}\n")
        self._run_stage_1()
        self._run_stage_2()
        if not self.training_config.finetune_skip_stage3:
            self._run_stage_3()
        else:
            print("\n[INFO] Skipping Stage 3 (full network fine-tuning)")
        print(f"\n{'='*60}\nStaged Fine-tuning Complete\n{'='*60}\n")
        if is_effectively_global_rank_zero():
            print("\nCheckpoints saved:")
            for stage, path in self.stage_checkpoints.items():
                print(f"  {stage}: {path}")
        return self.stage_checkpoints

    def _run_stage_1(self):
        """Stage 1: Freeze encoder, train decoder only."""
        self.current_stage = 1
        print(f"\n[STAGE 1] Decoder-Only Fine-tuning")
        print(f"Duration: {self.training_config.finetune_stage1_epochs} epochs")
        self.model.model.freeze_encoder()
        self.model.model.print_trainable_status()
        optimizer = self._create_stage1_optimizer()
        self.model.configure_optimizers = lambda: optimizer
        trainer = self._create_stage_trainer(
            max_epochs=self.training_config.finetune_stage1_epochs,
            stage_name="stage1", stage_description="Decoder-only fine-tuning"
        )
        trainer.fit(self.model, datamodule=self.train_module)
        if is_effectively_global_rank_zero():
            self.stage_checkpoints['stage1'] = trainer.checkpoint_callback.best_model_path

    def _run_stage_2(self):
        """Stage 2: Unfreeze top encoder, use discriminative LR."""
        self.current_stage = 2
        print(f"\n[STAGE 2] Partial Encoder + Decoder Fine-tuning")
        print(f"Duration: {self.training_config.finetune_stage2_epochs} epochs")
        self.model.model.freeze_encoder_bottom()
        self.model.model.unfreeze_encoder_top()
        optimizer = self._create_stage2_optimizer()
        self.model.configure_optimizers = lambda: optimizer
        trainer = self._create_stage_trainer(
            max_epochs=self.training_config.finetune_stage2_epochs,
            stage_name="stage2", stage_description="Partial encoder fine-tuning"
        )
        trainer.fit(self.model, datamodule=self.train_module)
        if is_effectively_global_rank_zero():
            self.stage_checkpoints['stage2'] = trainer.checkpoint_callback.best_model_path

    def _run_stage_3(self):
        """Stage 3: Unfreeze all, very conservative LR."""
        self.current_stage = 3
        print(f"\n[STAGE 3] Full Network Fine-tuning")
        print(f"Duration: {self.training_config.finetune_stage3_epochs} epochs")
        self.model.model.unfreeze_all()
        optimizer = self._create_stage3_optimizer()
        self.model.configure_optimizers = lambda: optimizer
        trainer = self._create_stage_trainer(
            max_epochs=self.training_config.finetune_stage3_epochs,
            stage_name="stage3", stage_description="Full network fine-tuning"
        )
        trainer.fit(self.model, datamodule=self.train_module)
        if is_effectively_global_rank_zero():
            self.stage_checkpoints['stage3'] = trainer.checkpoint_callback.best_model_path

    def _create_stage1_optimizer(self):
        """Optimizer for Stage 1: Decoder only."""
        base_lr = self.model.lr
        decoder_params = list(self.model.model.get_decoder_params())
        phase_head_params = list(self.model.model.get_phase_head_params())
        amp_head_params = list(self.model.model.get_amp_head_params())
        all_params = decoder_params + phase_head_params + amp_head_params
        trainable = [p for p in all_params if p.requires_grad]
        return torch.optim.Adam(trainable, lr=base_lr * self.training_config.finetune_stage1_lr_decoder)

    def _create_stage2_optimizer(self):
        """Optimizer for Stage 2: Discriminative LR."""
        base_lr = self.model.lr
        cfg = self.training_config
        param_groups = []
        enc_top = [p for p in self.model.model.get_encoder_top_params() if p.requires_grad]
        if enc_top:
            param_groups.append({'params': enc_top, 'lr': base_lr * cfg.finetune_stage2_lr_encoder_top})
        decoder = [p for p in self.model.model.get_decoder_params() if p.requires_grad]
        if decoder:
            param_groups.append({'params': decoder, 'lr': base_lr * cfg.finetune_stage2_lr_decoder})
        phase_head = [p for p in self.model.model.get_phase_head_params() if p.requires_grad]
        if phase_head:
            param_groups.append({'params': phase_head, 'lr': base_lr * cfg.finetune_stage2_lr_phase_head})
        amp_head = [p for p in self.model.model.get_amp_head_params() if p.requires_grad]
        if amp_head:
            param_groups.append({'params': amp_head, 'lr': base_lr * cfg.finetune_stage2_lr_decoder})
        return torch.optim.Adam(param_groups)

    def _create_stage3_optimizer(self):
        """Optimizer for Stage 3: Full network with very conservative LR."""
        base_lr = self.model.lr
        cfg = self.training_config
        param_groups = []
        enc_bot = [p for p in self.model.model.get_encoder_bottom_params() if p.requires_grad]
        if enc_bot:
            param_groups.append({'params': enc_bot, 'lr': base_lr * cfg.finetune_stage3_lr_encoder_bottom})
        enc_top = [p for p in self.model.model.get_encoder_top_params() if p.requires_grad]
        if enc_top:
            param_groups.append({'params': enc_top, 'lr': base_lr * cfg.finetune_stage3_lr_encoder_top})
        decoder = [p for p in self.model.model.get_decoder_params() if p.requires_grad]
        if decoder:
            param_groups.append({'params': decoder, 'lr': base_lr * cfg.finetune_stage3_lr_decoder})
        phase_head = [p for p in self.model.model.get_phase_head_params() if p.requires_grad]
        if phase_head:
            param_groups.append({'params': phase_head, 'lr': base_lr * cfg.finetune_stage3_lr_phase_head})
        amp_head = [p for p in self.model.model.get_amp_head_params() if p.requires_grad]
        if amp_head:
            param_groups.append({'params': amp_head, 'lr': base_lr * cfg.finetune_stage3_lr_decoder})
        return torch.optim.Adam(param_groups)

    def _create_stage_trainer(self, max_epochs, stage_name, stage_description):
        """Create Lightning trainer for a specific stage."""
        stage_dir = self.output_dir / f"finetune_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        tb_logger, csv_logger = self.create_experiment_loggers(
            experiment_name=f"{self.training_config.experiment_name}_finetune",
            run_name=f"{stage_name}_{stage_description.replace(' ', '_')}",
            output_dir=str(stage_dir)
        )
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=self.model.val_loss_name, mode='min', save_top_k=1,
            filename=f'best-{stage_name}-checkpoint', save_last=True, verbose=True,
            save_on_train_epoch_end=False,
        )
        early_stop_callback = EarlyStopping(
            monitor=self.model.val_loss_name,
            patience=self.training_config.finetune_early_stop_patience,
            mode='min', verbose=True, strict=True
        )
        trainer = L.Trainer(
            max_epochs=max_epochs,
            devices=self.training_config.n_devices,
            accelerator='gpu',
            strategy=get_training_strategy(self.training_config.strategy, self.training_config.n_devices),
            callbacks=[checkpoint_callback, early_stop_callback],
            enable_checkpointing=True,
            logger=[tb_logger, csv_logger],
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
        )
        return trainer

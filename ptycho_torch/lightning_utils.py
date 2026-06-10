"""
Lightning callbacks and utilities for config management and experiment tracking.
Replaces MLflow functionality with pure PyTorch Lightning + filesystem approach.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.utilities import rank_zero_only

from ptycho_torch.config_params import (
    DataConfig, ModelConfig, TrainingConfig, 
    InferenceConfig, DatagenConfig
)
from ptycho_torch.utils import config_to_json_serializable_dict

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("lightning.pytorch")


class ConfigLogger(Callback):
    """
    Saves all configuration dataclasses as JSON files at the start of training.
    
    Creates a 'configs/' directory in the run folder with individual JSON files
    for each config type, plus a combined full_config.json.
    """
    
    def __init__(self,
                 data_config: DataConfig,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 inference_config: InferenceConfig,
                 datagen_config: Optional[DatagenConfig] = None):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.datagen_config = datagen_config
        
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Save configs to JSON files at training start"""
        
        # Get run directory from trainer
        log_dir = Path(trainer.log_dir)
        config_dir = log_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        configs = {
            'data_config': config_to_json_serializable_dict(self.data_config),
            'model_config': config_to_json_serializable_dict(self.model_config),
            'training_config': config_to_json_serializable_dict(self.training_config),
            'inference_config': config_to_json_serializable_dict(self.inference_config),
        }
        if self.datagen_config is not None:
            configs['datagen_config'] = config_to_json_serializable_dict(self.datagen_config)
        
        # Save individual config files
        for config_name, config_dict in configs.items():
            config_path = config_dir / f"{config_name}.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved {config_name} to {config_path}")
        
        # Save combined config file
        full_config_path = config_dir / "full_config.json"
        with open(full_config_path, 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"Saved full config to {full_config_path}")


class MetadataLogger(Callback):
    """
    Logs run metadata (timestamp, stage, notes, etc.) to metadata.json.
    Updates throughout training to track stage transitions.
    """

    def __init__(self,
                 run_dir: str,
                 stage: str = "training",
                 notes: str = "",
                 model_name: str = "PtychoPINNv2",
                 encoder_frozen: bool = False,
                 dataset_dir: str = ""):
        super().__init__()
        self.run_dir = run_dir
        self.stage = stage
        self.notes = notes
        self.model_name = model_name
        self.encoder_frozen = encoder_frozen
        self.dataset_dir = dataset_dir
        self.start_time = None
        self.end_time = None
        
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Initialize metadata file at training start"""
        self.start_time = datetime.now().isoformat()

        run_dir = Path(self.run_dir)
        metadata_path = run_dir / "metadata.json"

        metadata = {
            'run_id': run_dir.name,
            'stage': self.stage,
            'model_name': self.model_name,
            'encoder_frozen': self.encoder_frozen,
            'notes': self.notes,
            'dataset_dir': str(Path(self.dataset_dir).resolve()) if self.dataset_dir else "",
            'start_time': self.start_time,
            'end_time': None,
            'status': 'running',
            'best_val_loss': None,
            'final_epoch': None,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Created metadata file at {metadata_path}")
    
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Update metadata with final stats"""
        self.end_time = datetime.now().isoformat()

        run_dir = Path(self.run_dir)
        metadata_path = run_dir / "metadata.json"
        
        # Load existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update with final info
        metadata.update({
            'end_time': self.end_time,
            'status': 'completed',
            'final_epoch': trainer.current_epoch,
        })
        
        # Try to get best validation loss from checkpoint callback
        for callback in trainer.callbacks:
            if hasattr(callback, 'best_model_score'):
                metadata['best_val_loss'] = float(callback.best_model_score)
                break
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata file at {metadata_path}")


def create_experiment_loggers(experiment_name: str,
                               run_name: Optional[str] = None,
                               output_dir: str = "training_outputs",
                               run_tag: str = "") -> Tuple[TensorBoardLogger, CSVLogger]:
    """
    Create TensorBoard and CSV loggers with consistent directory structure.

    Args:
        experiment_name: Name of experiment (e.g., "Synthetic_Runs")
        run_name: Optional custom run name. If None, auto-generated from timestamp.
        output_dir: Root output directory for all experiments
        run_tag: Optional prefix for auto-generated run names (e.g., "fno_encoder")

    Returns:
        Tuple of (TensorBoardLogger, CSVLogger)
    """
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_tag:
            run_name = f"{run_tag}_run_{timestamp}"
        else:
            run_name = f"run_{timestamp}"
    
    # Create loggers pointing to same version directory
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name=experiment_name,
        version=run_name,
        default_hp_metric=False,  # Don't create hparams metric
    )
    
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name=experiment_name,
        version=run_name,
    )
    
    print(f"Experiment: {experiment_name}")
    print(f"Run: {run_name}")
    print(f"Log directory: {tb_logger.log_dir}")
    
    return tb_logger, csv_logger


def load_configs_from_checkpoint(checkpoint_path: str) -> Tuple[DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig]:
    """
    Load configuration dataclasses from a checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory containing configs/
        
    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig)
        
    Example:
        >>> configs = load_configs_from_checkpoint("training_outputs/Synthetic_Runs/run_20260108_143022/checkpoints/best-checkpoint.ckpt")
        >>> data_config, model_config, training_config, inference_config, datagen_config = configs
    """
    checkpoint_path = Path(checkpoint_path)
    
    # If given a checkpoint file, navigate to its parent's parent (run directory)
    if checkpoint_path.is_file():
        run_dir = checkpoint_path.parent.parent
    else:
        run_dir = checkpoint_path
    
    config_dir = run_dir / "configs"
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Load individual configs
    def load_config(config_name: str, config_class):
        config_path = config_dir / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create instance with loaded values
        return config_class(**config_dict)
    
    data_config = load_config("data_config", DataConfig)
    model_config = load_config("model_config", ModelConfig)
    training_config = load_config("training_config", TrainingConfig)
    inference_config = load_config("inference_config", InferenceConfig)
    datagen_config = load_config("datagen_config", DatagenConfig)
    
    print(f"Loaded configs from {config_dir}")
    
    return data_config, model_config, training_config, inference_config, datagen_config


def load_checkpoint_with_configs(checkpoint_path: str, 
                                  model_class,
                                  device: str = 'cuda') -> Tuple[Any, Tuple]:
    """
    Load a trained model and its configs from checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        model_class: Model class to instantiate (e.g., PtychoPINN_Lightning)
        device: Device to load model onto
        
    Returns:
        Tuple of (model, configs_tuple)
        
    Example:
        >>> from ptycho_torch.model import PtychoPINN_Lightning
        >>> model, configs = load_checkpoint_with_configs(
        ...     "training_outputs/Synthetic_Runs/run_20260108_143022/checkpoints/best-checkpoint.ckpt",
        ...     PtychoPINN_Lightning
        ... )
    """
    # Load configs
    configs = load_configs_from_checkpoint(checkpoint_path)
    data_config, model_config, training_config, inference_config, datagen_config = configs
    
    # Load model from checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
        map_location=device,
    )
    
    print(f"Loaded model from {checkpoint_path}")
    
    return model, configs


def get_latest_run(experiment_name: str, output_dir: str = "training_outputs") -> Optional[Path]:
    """
    Get the most recent run directory for an experiment.
    
    Args:
        experiment_name: Name of experiment
        output_dir: Root output directory
        
    Returns:
        Path to latest run directory, or None if no runs exist
    """
    experiment_dir = Path(output_dir) / experiment_name
    
    if not experiment_dir.exists():
        return None
    
    run_dirs = [d for d in experiment_dir.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()]
    
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) and return latest
    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
    return latest_run


def print_run_summary(run_dir: Path):
    """
    Print a summary of a training run (replaces print_auto_logged_info).
    
    Args:
        run_dir: Path to run directory
    """
    print(f"\n{'='*60}")
    print(f"Run Summary: {run_dir.name}")
    print(f"{'='*60}")
    
    # Print metadata
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\nMetadata:")
        print(f"  Stage: {metadata.get('stage', 'N/A')}")
        print(f"  Model: {metadata.get('model_name', 'N/A')}")
        print(f"  Status: {metadata.get('status', 'N/A')}")
        print(f"  Start: {metadata.get('start_time', 'N/A')}")
        print(f"  End: {metadata.get('end_time', 'N/A')}")
        print(f"  Best Val Loss: {metadata.get('best_val_loss', 'N/A')}")
        print(f"  Final Epoch: {metadata.get('final_epoch', 'N/A')}")
        if metadata.get('notes'):
            print(f"  Notes: {metadata['notes']}")
    
    # Print checkpoint info
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        print(f"\nCheckpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  {ckpt.name} ({size_mb:.2f} MB)")
    
    # Print config summary
    config_dir = run_dir / "configs"
    if config_dir.exists():
        configs = list(config_dir.glob("*.json"))
        print(f"\nConfigs ({len(configs)}):")
        for config in configs:
            print(f"  {config.name}")
    
    # Print metrics summary (from CSV logger)
    metrics_file = run_dir / "logs" / "metrics.csv"
    if metrics_file.exists():
        import pandas as pd
        try:
            df = pd.read_csv(metrics_file)
            print(f"\nMetrics recorded: {', '.join(df.columns.tolist())}")
            print(f"Total steps: {len(df)}")
        except Exception as e:
            print(f"\nCould not read metrics: {e}")
    
    print(f"\nRun directory: {run_dir}")
    print(f"{'='*60}\n")


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """
    Find the best checkpoint in a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Path to best checkpoint, or None if not found
    """
    checkpoint_dir = run_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for best-checkpoint first
    best_ckpt = checkpoint_dir / "best-checkpoint.ckpt"
    if best_ckpt.exists():
        return best_ckpt
    
    # Fall back to last checkpoint
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    
    # Try to find any checkpoint
    ckpts = list(checkpoint_dir.glob("*.ckpt"))
    if ckpts:
        return ckpts[0]
    
    return None

class DDPSafeProgressCallback(Callback):
    """
    Simple epoch-based progress reporter that works reliably in DDP.
    
    Unlike the default progress bar which can freeze with multiple GPUs,
    this callback just prints epoch summaries from rank 0.
    """
    
    def __init__(self, log_every_n_epochs=1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_start_time = None
        
    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        """Print epoch start"""
        import time
        self.epoch_start_time = time.time()
        print(f"\n[Epoch {trainer.current_epoch}] Starting training...")
        sys.stdout.flush()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Pull metrics directly from the trainer's logged values
        metrics = trainer.callback_metrics
        # Filter for scalars only to avoid printing tensors
        m_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if v.numel() == 1])
        
        # Use logging instead of print
        logger.info(f"Epoch {trainer.current_epoch}: {m_str}")
        
        # Force flush (crucial for DDP)
        sys.stdout.flush()
    
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        """Print validation summary"""
        metrics = trainer.callback_metrics
        
        val_metrics = {k: v for k, v in metrics.items() if 'val' in k}
        if val_metrics:
            metric_str = ", ".join([f"{k}={float(v):.4f}" for k, v in val_metrics.items()])
            print(f"  Validation: {metric_str}")
            sys.stdout.flush()


class VerboseProgressCallback(Callback):
    """
    More verbose progress tracking for debugging DDP issues.
    Prints updates at every training/validation batch.
    """
    
    def __init__(self, log_every_n_batches=10):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0
        
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Print periodic batch updates"""
        self.batch_count += 1
        
        if self.batch_count % self.log_every_n_batches == 0:
            metrics = trainer.callback_metrics
            loss_str = f"loss={float(metrics.get('loss', 0)):.4f}" if 'loss' in metrics else ""
            print(f"  Batch {batch_idx}/{trainer.num_training_batches} | {loss_str}", end='\r')
            sys.stdout.flush()
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Clear line after epoch"""
        print()  # New line to clear the \r updates
        sys.stdout.flush()


class RankAwareProgressCallback(Callback):
    """
    Progress callback that shows what each rank is doing.
    Useful for debugging DDP synchronization issues.
    """
    
    def __init__(self):
        super().__init__()
        self.rank = None
        
    def setup(self, trainer, pl_module, stage):
        """Get rank information"""
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Each rank reports epoch start"""
        print(f"[Rank {self.rank}] Epoch {trainer.current_epoch} started")
        sys.stdout.flush()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Each rank reports epoch end"""
        print(f"[Rank {self.rank}] Epoch {trainer.current_epoch} completed")
        sys.stdout.flush()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Periodic batch updates from each rank"""
        if batch_idx % 20 == 0:
            print(f"[Rank {self.rank}] Batch {batch_idx}/{trainer.num_training_batches}")
            sys.stdout.flush()
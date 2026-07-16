#Most basic modules
import sys
import argparse
import os
import json
import random
import math
from pathlib import Path

#Typing
from dataclasses import dataclass
from typing import Any, Dict, Optional
from ptycho.config.config import PyTorchExecutionConfig
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
from ptycho_torch.scaling_contract import validate_scale_contract
from ptycho_torch.utils import config_to_json_serializable_dict, load_config_from_json, validate_and_process_config
from ptycho_torch.train_utils import set_seed, get_training_strategy, find_learning_rate, is_effectively_global_rank_zero, PtychoDataModule, PtychoDataModuleLightning

# NEW: Import our custom Lightning utilities
from ptycho_torch.lightning_utils import (
    CIStatisticsCallback,
    ConfigLogger,
    MetadataLogger,
    create_experiment_loggers,
    print_run_summary,
    find_best_checkpoint,
    DDPSafeProgressCallback
)

from ptycho_torch.model_finetuner_modified import ModelFineTuner
from ptycho_torch.training_history import build_training_history

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar


# Reduce timeout to 2 minutes for debugging
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" # Optional: gives more info on hangs

#----- Helper Functions -------


@dataclass(frozen=True)
class TrainingRunResult:
    run_dir: Path
    model: PtychoPINN_Lightning
    data_config: DataConfig
    model_config: ModelConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    datagen_config: DatagenConfig
    effective_runtime: Dict[str, Any]
    #: Per-logged-step/epoch losses, gradient norms, and output statistics
    #: parsed from the CSV logger; None when no metrics.csv was produced.
    training_history: Optional[Dict[str, Any]] = None
    milestone_checkpoints: Optional[Dict[int, Path]] = None


class _MilestoneCheckpointCallback(Callback):
    """Save exact one-based post-epoch checkpoints without affecting selection."""

    def __init__(self, dirpath: Path, epochs: tuple[int, ...]) -> None:
        super().__init__()
        if any(type(epoch) is not int or epoch <= 0 for epoch in epochs):
            raise ValueError("milestone epochs must be positive integers")
        if tuple(sorted(set(epochs))) != epochs:
            raise ValueError("milestone epochs must be strictly increasing")
        self.dirpath = Path(dirpath)
        self.epochs = epochs
        self.saved_checkpoints: Dict[int, Path] = {}

    def on_validation_end(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        external_epoch = int(trainer.current_epoch) + 1
        if external_epoch not in self.epochs or external_epoch in self.saved_checkpoints:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        checkpoint = self.dirpath / f"epoch-{external_epoch:04d}.ckpt"
        trainer.save_checkpoint(str(checkpoint))
        self.saved_checkpoints[external_epoch] = checkpoint


def _trainer_accelerator(accelerator: str) -> str:
    return "gpu" if accelerator == "cuda" else accelerator


def _torch_device_accelerator(accelerator: str) -> str:
    return "cuda" if accelerator == "gpu" else accelerator


def _effective_device_count(devices, accelerator) -> int:
    if isinstance(devices, int):
        return devices
    if accelerator in {"cuda", "gpu"}:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return max(count, 1)
    return 1


def _trainer_strategy(strategy, devices, accelerator):
    if strategy == "auto":
        return "auto"
    if accelerator in {"cuda", "gpu"}:
        return get_training_strategy(strategy, devices)
    return strategy


def _resolve_checkpoint_monitor(execution_config, model) -> str:
    configured = execution_config.checkpoint_monitor_metric
    return model.val_loss_name if configured == "val_loss" else configured


def _runtime_json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_runtime_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _runtime_json_value(item) for key, item in value.items()}
    value_type = type(value)
    return {"class": f"{value_type.__module__}.{value_type.__qualname__}"}


def _runtime_class(value) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _runtime_attributes(value, names):
    settings = {"class": _runtime_class(value)}
    for name in names:
        try:
            attribute = getattr(value, name)
        except (AttributeError, RuntimeError):
            continue
        if attribute is not None:
            settings[name] = _runtime_json_value(attribute)
    return settings


def _callback_runtime(callback):
    return _runtime_attributes(
        callback,
        (
            "monitor",
            "mode",
            "save_top_k",
            "save_last",
            "patience",
            "dirpath",
        ),
    )


def _logger_runtime(logger):
    return _runtime_attributes(
        logger,
        ("name", "version", "save_dir", "log_dir", "root_dir", "experiment_name"),
    )


def _logger_list(value):
    if value in (None, False):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _trainer_kwargs_runtime(trainer_kwargs):
    serialized = {}
    for key, value in trainer_kwargs.items():
        if key == "callbacks":
            serialized[key] = [_callback_runtime(callback) for callback in value]
        elif key == "logger":
            serialized[key] = [_logger_runtime(logger) for logger in _logger_list(value)]
        else:
            serialized[key] = _runtime_json_value(value)
    return serialized


def _strategy_runtime(strategy):
    backend = getattr(strategy, "process_group_backend", None)
    if backend is None:
        backend = getattr(strategy, "_process_group_backend", None)
    if backend is None:
        resolve_backend = getattr(strategy, "_get_process_group_backend", None)
        if callable(resolve_backend):
            backend = resolve_backend()
    if backend is None and dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
    settings = _runtime_attributes(strategy, ("root_device", "parallel_devices"))
    settings["process_group_backend"] = backend
    start_method = getattr(strategy, "_start_method", None)
    try:
        launcher = getattr(strategy, "launcher", None)
    except (AttributeError, RuntimeError):
        launcher = None
    if start_method is None and launcher is not None:
        start_method = getattr(launcher, "_start_method", None)
    if start_method is not None:
        settings["start_method"] = start_method
    if launcher is not None:
        settings["launcher"] = {"class": _runtime_class(launcher)}
    return settings


def _build_effective_runtime(
    resolved_seed,
    trainer_kwargs,
    execution_config,
    dataloader_settings=None,
    trainer=None,
):
    if trainer is None:
        raise ValueError("trainer is required to record effective runtime")
    if dataloader_settings is None:
        num_workers = execution_config.num_workers
        dataloader_settings = {
            "num_workers": num_workers,
            "pin_memory": execution_config.pin_memory,
            "persistent_workers": (
                execution_config.persistent_workers if num_workers > 0 else False
            ),
            "prefetch_factor": (
                (execution_config.prefetch_factor or 2)
                if num_workers > 0
                else None
            ),
        }
    precision_value = getattr(trainer.precision_plugin, "precision", None)
    root_device = trainer.strategy.root_device
    mps_backend = getattr(torch.backends, "mps", None)
    effective = {
        "precision": {
            "value": precision_value,
            "plugin": _runtime_class(trainer.precision_plugin),
        },
        "deterministic": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_available": bool(
                mps_backend is not None and mps_backend.is_available()
            ),
        },
        "accelerator": {
            "class": _runtime_class(trainer.accelerator),
            "device_type": root_device.type,
            "trainer_value": _runtime_json_value(trainer_kwargs.get("accelerator")),
        },
        "devices": {
            "count": trainer.num_devices,
            "ids": _runtime_json_value(trainer.device_ids),
            "trainer_value": _runtime_json_value(trainer_kwargs.get("devices")),
        },
        "strategy": {
            **_strategy_runtime(trainer.strategy),
            "trainer_value": _runtime_json_value(trainer_kwargs.get("strategy")),
        },
        "callbacks": [_callback_runtime(callback) for callback in trainer.callbacks],
        "loggers": [_logger_runtime(logger) for logger in trainer.loggers],
        "dataloader": _runtime_json_value(dataloader_settings),
    }
    requested = {
        "accelerator": execution_config.accelerator,
        "devices": execution_config.devices,
        "strategy": execution_config.strategy,
        "deterministic": execution_config.deterministic,
        "precision": execution_config.precision,
        "enable_progress_bar": execution_config.enable_progress_bar,
        "enable_checkpointing": execution_config.enable_checkpointing,
        "checkpoint_save_top_k": execution_config.checkpoint_save_top_k,
        "checkpoint_monitor_metric": execution_config.checkpoint_monitor_metric,
        "checkpoint_mode": execution_config.checkpoint_mode,
        "early_stop_patience": execution_config.early_stop_patience,
        "dataloader": {
            "num_workers": execution_config.num_workers,
            "pin_memory": execution_config.pin_memory,
            "persistent_workers": execution_config.persistent_workers,
            "prefetch_factor": execution_config.prefetch_factor,
        },
    }
    return {
        "seed": resolved_seed,
        "requested": requested,
        "effective": effective,
        "trainer_kwargs": _trainer_kwargs_runtime(trainer_kwargs),
        "dataloader": _runtime_json_value(dataloader_settings),
        "precision": precision_value,
    }

def _build_ci_statistics_callback():
    return CIStatisticsCallback()


def _resolve_seed() -> int:
    """
    Resolve the training seed from the PTYCHO_TORCH_SEED environment variable.

    Returns:
        int: The seed value from PTYCHO_TORCH_SEED if set and non-empty,
            otherwise 42.

    Raises:
        ValueError: If PTYCHO_TORCH_SEED is set to a non-integer value.
    """
    raw = os.environ.get("PTYCHO_TORCH_SEED", "")
    if not raw:
        return 42
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid PTYCHO_TORCH_SEED={raw!r}: must be an integer"
        ) from e


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
                    version = np.lib.format.read_magic(npy)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(npy)
                    elif version == (2, 0):
                        shape, _, _ = np.lib.format.read_array_header_2_0(npy)
                    else:
                        raise ValueError(f"Unsupported NPY format version: {version}")
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
         run_name = None,
         parity_scale_mode = "off",
         parity_fixed_delta = 0.0,
         parity_init_scheme = "default",
         scale_contract_version = None,
         measurement_domain = None,
         *,
         seed: Optional[int] = None,
         return_training_result: bool = False,
         milestone_epochs: tuple[int, ...] = ()):
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
                     If None, uses the default execution config.
    run_name: Optional custom name for this run. If None, uses timestamp.
    seed: Explicit training seed. If omitted, PTYCHO_TORCH_SEED (or 42) is used.
    return_training_result: Return TrainingRunResult instead of only the run directory.
    parity_scale_mode: TF-parity global intensity-scale mode passed through to
                     PtychoPINN_Lightning (see docs/plans/2026-07-08-cnn-n128-tf-parity.md
                     Task 1). Default "off" preserves current behavior exactly.
    parity_fixed_delta: Initial/frozen log-scale delta value for the parity mechanism.
    parity_init_scheme: Weight-init preset passed through to PtychoPINN_Lightning's
                     parity mechanism ("default" or "tf_glorot").

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

        from ptycho_torch.config_factory import resolve_profile_overrides
        explicit_profile = resolve_profile_overrides({
            "scale_contract_version": scale_contract_version,
            "measurement_domain": measurement_domain,
        })
        if explicit_profile is not None:
            data_config.scale_contract_version = explicit_profile[0]
            data_config.measurement_domain = explicit_profile[1]

        validate_scale_contract(data_config, model_config, training_config)

        if execution_config is None:
            execution_config = PyTorchExecutionConfig()

        # Execution config is authoritative; these mutable fields are compatibility
        # aliases consumed by the model and data module.
        training_config.n_devices = execution_config.devices
        training_config.strategy = execution_config.strategy
        training_config.device = _torch_device_accelerator(execution_config.accelerator)
        training_config.num_workers = execution_config.num_workers
        assert training_config.n_devices == execution_config.devices
        assert training_config.strategy == execution_config.strategy
        assert training_config.device == _torch_device_accelerator(
            execution_config.accelerator
        )

        # Force Lightning orchestrator — this script uses PtychoDataModuleLightning,
        # which relies on Lightning's prepare_data/setup lifecycle instead of
        # manual dist.barrier() synchronization used by the Mlflow path.
        training_config.orchestrator = 'Lightning'

        # Generate run_name before trainer creation. Under ddp_spawn, Lightning
        # spawns children inside .fit(), so anything computed here happens in the
        # parent process only — no inter-process coordination needed.
        # Under ddp (torchrun), the caller is responsible for passing the same
        # run_name to all processes (torchrun runs the same command on each rank).
        output_dir = output_dir or training_config.output_dir
        if run_name is None:
            from datetime import datetime
            import time
            run_name_file = Path(output_dir) / '.current_run_name'
            if is_effectively_global_rank_zero():
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run_name_file.parent.mkdir(parents=True, exist_ok=True)
                run_name_file.write_text(run_name)
                print(f"Generated run name: {run_name}")
            else:
                for _ in range(120):
                    if run_name_file.exists():
                        run_name = run_name_file.read_text().strip()
                        if run_name:
                            break
                    time.sleep(0.5)
                else:
                    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"[Rank {os.environ.get('RANK', '?')}] Using run name: {run_name}")

        #Setting seed
        resolved_seed = _resolve_seed() if seed is None else seed
        runtime_device_count = _effective_device_count(
            execution_config.devices,
            execution_config.accelerator,
        )
        set_seed(resolved_seed, n_devices=runtime_device_count)

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
            val_seed=42,     # Reproducible split
            execution_config=execution_config,
        )

        #Create model
        print('Creating model...')
        from ptycho_torch.config_bridge import to_model_config
        from ptycho_torch.model_spec import derive_model_spec

        model_spec = derive_model_spec(
            to_model_config(data_config, model_config),
            model_config,
            data_config,
            parity_scale_mode=parity_scale_mode,
            parity_fixed_delta=parity_fixed_delta,
            parity_init_scheme=parity_init_scheme,
        )
        model = PtychoPINN_Lightning(
            model_config,
            data_config,
            training_config,
            inference_config,
            parity_scale_mode=parity_scale_mode,
            parity_fixed_delta=parity_fixed_delta,
            parity_init_scheme=parity_init_scheme,
            model_spec=model_spec.to_payload(),
        )
        model.training = True

        #Update LR (Phase C4.C3: Use execution_config if available, else training_config)
        base_lr = execution_config.learning_rate if execution_config else training_config.learning_rate
        updated_lr = find_learning_rate(base_lr,
                                        runtime_device_count, training_config.batch_size)
        model.lr = updated_lr

        # NEW: Create experiment loggers (replaces MLflow)
        tb_logger, csv_logger = create_experiment_loggers(
            experiment_name=training_config.experiment_name,
            run_name=run_name,
            output_dir=output_dir,
        )

        # NEW: Create custom callbacks for config and metadata logging
        config_logger = ConfigLogger(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            inference_config=inference_config,
            datagen_config=datagen_config,
        )
        
        metadata_logger = MetadataLogger(
            run_dir=tb_logger.log_dir,
            stage="training",
            notes=training_config.notes,
            model_name=training_config.model_name,
            encoder_frozen=False,
        )

        callbacks = [
            _build_ci_statistics_callback(),
            config_logger,
            metadata_logger,
            LearningRateMonitor(logging_interval="epoch"),
        ]
        checkpoint_dir = None
        milestone_callback = None
        checkpoint_monitor = _resolve_checkpoint_monitor(execution_config, model)
        if execution_config.enable_checkpointing:
            checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.extend([
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    monitor=checkpoint_monitor,
                    mode=execution_config.checkpoint_mode,
                    save_top_k=execution_config.checkpoint_save_top_k,
                    filename='best-checkpoint',
                    save_last=True,
                    verbose=True,
                    save_on_train_epoch_end=False,
                ),
                EarlyStopping(
                    monitor=checkpoint_monitor,
                    mode=execution_config.checkpoint_mode,
                    patience=execution_config.early_stop_patience,
                    verbose=True,
                    strict=True,
                ),
            ])

        total_training_epochs = (
            training_config.epochs + training_config.epochs_fine_tune
        )
        if milestone_epochs:
            if not execution_config.enable_checkpointing:
                raise ValueError("milestone checkpoints require checkpointing to be enabled")
            if max(milestone_epochs) > total_training_epochs:
                raise ValueError(
                    "milestone epochs cannot exceed the configured training epochs"
                )
            milestone_callback = _MilestoneCheckpointCallback(
                checkpoint_dir / "milestones", milestone_epochs
            )
            callbacks.append(milestone_callback)

        if execution_config.enable_progress_bar:
            callbacks.append(TQDMProgressBar(refresh_rate=10))

        # Single-pass fine-tuning: if epochs_fine_tune > 0, add EncoderFreezeCallback
        # and extend total epochs. The callback freezes the encoder and scales LR
        # at the transition epoch, avoiding a second Trainer (spawn-compatible).
        if training_config.epochs_fine_tune > 0:
            from ptycho_torch.train_utils import EncoderFreezeCallback
            callbacks.append(EncoderFreezeCallback(
                freeze_at_epoch=training_config.epochs,
                lr_gamma=training_config.fine_tune_gamma,
            ))

        # Create trainer with execution config knobs and NEW loggers
        trainer_kwargs = dict(
            max_epochs = total_training_epochs,
            default_root_dir = str(Path(output_dir)),
            devices = execution_config.devices,
            accelerator = _trainer_accelerator(execution_config.accelerator),
            callbacks = callbacks,
            strategy=_trainer_strategy(
                execution_config.strategy,
                execution_config.devices,
                execution_config.accelerator,
            ),
            check_val_every_n_epoch=1,  # Validate every epoch
            enable_checkpointing=execution_config.enable_checkpointing,
            enable_progress_bar=execution_config.enable_progress_bar,
            deterministic=execution_config.deterministic,
            precision=execution_config.precision,
            logger=[tb_logger, csv_logger],  # NEW: Use Lightning loggers
        )
        trainer = L.Trainer(**trainer_kwargs)
        effective_runtime = _build_effective_runtime(
            resolved_seed,
            trainer_kwargs,
            execution_config,
            data_module.effective_dataloader_settings(),
            trainer=trainer,
        )

        #Train the model
        # if is_effectively_global_rank_zero():
        #     print(f'[Rank {trainer.global_rank}] Beginning model training/final data prep...')
        
        trainer.fit(model, datamodule = data_module)

        if milestone_callback is not None:
            missing = [
                epoch
                for epoch in milestone_epochs
                if milestone_callback.saved_checkpoints.get(epoch) is None
                or not milestone_callback.saved_checkpoints[epoch].is_file()
            ]
            if missing:
                raise RuntimeError(
                    "requested milestone checkpoints were not captured: "
                    + ", ".join(str(epoch) for epoch in missing)
                )

        # Every rank returns the same typed path; filesystem side effects remain
        # rank-zero-only below.
        training_run_dir = Path(trainer.log_dir)

        if is_effectively_global_rank_zero():
            print(f'[Rank {trainer.global_rank}] Done training.')

            # NEW: Print run summary (replaces print_auto_logged_info)
            print_run_summary(training_run_dir)
            with (training_run_dir / "effective_runtime.json").open("w") as stream:
                json.dump(effective_runtime, stream, indent=2, sort_keys=True)

        # Fine-tuning is handled by EncoderFreezeCallback when epochs_fine_tune > 0.
        # The callback was added to the trainer's callback list above (if applicable),
        # so fine-tuning happens within the single trainer.fit() call — no second
        # trainer or process group teardown needed. This is compatible with both
        # ddp and ddp_spawn strategies.

        if is_effectively_global_rank_zero():
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Run directory: {training_run_dir}")
            if execution_config.enable_checkpointing:
                print(f"Checkpoints: {checkpoint_dir}")
                print(f"Best checkpoint: {find_best_checkpoint(training_run_dir)}")
            print(f"TensorBoard: tensorboard --logdir {training_run_dir / 'logs'}")
            print(f"{'='*60}\n")

        # Lightning attaches the Trainer to the module during fit. The structured
        # handoff intentionally returns only the trained model and finalized values.
        if hasattr(model, "_trainer"):
            model._trainer = None

        if return_training_result:
            result = TrainingRunResult(
                run_dir=training_run_dir,
                model=model,
                data_config=data_config,
                model_config=model_config,
                training_config=training_config,
                inference_config=inference_config,
                datagen_config=datagen_config,
                effective_runtime=effective_runtime,
                training_history=build_training_history(
                    training_run_dir,
                    csv_logger=csv_logger,
                    model=model,
                    training_config=training_config,
                ),
                milestone_checkpoints=(
                    dict(milestone_callback.saved_checkpoints)
                    if milestone_callback is not None
                    else None
                ),
            )
            del data_module
            del trainer
            return result

        return training_run_dir

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Shutting down...")
        sys.exit(0)

    except Exception as e:
        print(f"Training failed: {e}")
        raise
    

    #Define main function
if __name__ == '__main__':
    #Parsing
    parser = argparse.ArgumentParser(description = "Run training for ptycho_torch")
    #Arguments
    parser.add_argument('--ptycho_dir', type = str, help = 'Path to ptycho directory')
    parser.add_argument('--config', type = str, default=None, help = 'Path to JSON configuration file (mandatory)')
    parser.add_argument('--output_dir', type = str, default=None, help = 'Path to output directory')
    parser.add_argument(
        '--scale-contract-version',
        choices=['ci_intensity_v2', 'legacy_v1'],
        default=None,
    )
    parser.add_argument(
        '--measurement-domain',
        choices=['count_intensity', 'normalized_amplitude'],
        default=None,
    )
    
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
        main(
            ptycho_dir,
            config_path,
            False,
            output_dir,
            scale_contract_version=args.scale_contract_version,
            measurement_domain=args.measurement_domain,
        )

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

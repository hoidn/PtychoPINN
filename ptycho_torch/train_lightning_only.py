#Most basic modules
import sys
import os
from pathlib import Path

#Typing
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional
from ptycho.config.config import PyTorchExecutionConfig
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig

import torch

#Automation modules
#Lightning
try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError as e:
    raise RuntimeError(
        "PyTorch backend requires Lightning. Install with: pip install -e .[torch]"
    ) from e

#Configs/Params
#Custom modules
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.scaling_contract import validate_scale_contract
from ptycho_torch.execution_request import resolve_runtime_execution_request
from ptycho_torch.train_utils import set_seed, get_training_strategy, find_learning_rate, is_effectively_global_rank_zero, PtychoDataModuleLightning

# NEW: Import our custom Lightning utilities
from ptycho_torch.lightning_utils import (
    CIStatisticsCallback,
    ConfigLogger,
    MetadataLogger,
    create_experiment_loggers,
    print_run_summary,
    find_best_checkpoint,
)

from ptycho_torch.training_history import build_training_history
from ptycho_torch.runtime_provenance import (
    build_effective_runtime as _build_effective_runtime,
    strategy_runtime as _strategy_runtime,  # noqa: F401 - compatibility test surface
    write_effective_runtime_json,
)

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
    Infer detector size (N) from canonical NPZ metadata without loading images.

    This function delegates NPZ layout and key handling to the canonical
    acquisition header reader.

    Args:
        npz_file (str or Path): Path to NPZ file containing probeGuess key

    Returns:
        int or None: Canonical diffraction height, or None for an invalid source.

    References:
        - specs/data_contracts.md §1 — probeGuess is required key for canonical NPZ format
        - ptycho_torch/dataloader.py:29-83 — npz_headers() implementation pattern
        - INTEGRATE-PYTORCH-001-PROBE-SIZE — Probe size mismatch resolution

    Example:
        >>> N = _infer_probe_size("datasets/Run1084_recon3_postPC_shrunk_3.npz")
        >>> print(N)  # 64 (for this dataset)
    """
    from ptycho.acquisition import inspect_probe_size

    try:
        return inspect_probe_size(npz_file)
    except (OSError, ValueError, KeyError):
        # If NPZ is invalid or missing, return None
        # Caller can decide whether to use default or raise error
        return None


#----- Main -------

def main(ptycho_dir,
         existing_config,
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
    existing_config: Already-resolved tuple of
                     (DataConfig, ModelConfig, TrainingConfig, InferenceConfig,
                     DatagenConfig)
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
        print('Starting training loop. Loading configs...')
        expected_types = (
            DataConfig,
            ModelConfig,
            TrainingConfig,
            InferenceConfig,
            DatagenConfig,
        )
        if (
            not isinstance(existing_config, tuple)
            or len(existing_config) != len(expected_types)
            or any(
                not isinstance(config, expected)
                for config, expected in zip(existing_config, expected_types)
            )
        ):
            raise TypeError(
                "existing_config must be a five-member resolved config tuple "
                "(DataConfig, ModelConfig, TrainingConfig, InferenceConfig, "
                "DatagenConfig)"
            )
        (
            data_config,
            model_config,
            training_config,
            inference_config,
            datagen_config,
        ) = tuple(replace(config) for config in existing_config)

        from ptycho_torch.config_factory import resolve_profile_overrides
        explicit_profile = resolve_profile_overrides({
            "scale_contract_version": scale_contract_version,
            "measurement_domain": measurement_domain,
        })
        if explicit_profile is not None:
            data_config = replace(
                data_config,
                scale_contract_version=explicit_profile[0],
                measurement_domain=explicit_profile[1],
            )

        if execution_config is None:
            execution_config = resolve_runtime_execution_request(
                None,
                mode="training",
            ).config
        elif not isinstance(execution_config, PyTorchExecutionConfig):
            raise TypeError(
                "execution_config must be a resolved "
                "PyTorchExecutionConfig or None"
            )
        validate_scale_contract(data_config, model_config, training_config)

        # Execution config is authoritative; project its compatibility aliases
        # into the fresh snapshot consumed by the model and data module.
        training_config = replace(
            training_config,
            n_devices=execution_config.devices,
            strategy=execution_config.strategy,
            device=_torch_device_accelerator(execution_config.accelerator),
            num_workers=execution_config.num_workers,
            orchestrator="Lightning",
        )
        assert training_config.n_devices == execution_config.devices
        assert training_config.strategy == execution_config.strategy
        assert training_config.device == _torch_device_accelerator(
            execution_config.accelerator
        )

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

        # Learning rate is owned by the resolved TrainingConfig.
        updated_lr = find_learning_rate(training_config.learning_rate,
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
        automatic_optimization = getattr(model, "automatic_optimization", True)
        clip_algorithm = training_config.gradient_clip_algorithm
        if automatic_optimization and clip_algorithm == "agc":
            raise ValueError(
                "gradient_clip_algorithm='agc' requires manual optimization; "
                "Lightning automatic optimization accepts only 'norm' or 'value'"
            )
        trainer_kwargs.update(
            accumulate_grad_batches=(
                training_config.accum_steps if automatic_optimization else 1
            ),
            gradient_clip_val=(
                training_config.gradient_clip_val
                if automatic_optimization
                else None
            ),
        )
        if automatic_optimization:
            trainer_kwargs["gradient_clip_algorithm"] = clip_algorithm
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

        if trainer.is_global_zero:
            print(f'[Rank {trainer.global_rank}] Done training.')

            # NEW: Print run summary (replaces print_auto_logged_info)
            print_run_summary(training_run_dir)
            write_effective_runtime_json(
                training_run_dir / "effective_runtime.json",
                effective_runtime,
            )

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

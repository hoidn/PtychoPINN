"""Lightning Trainer service: checkpoint selection, callbacks, and training.

Owns the serving-checkpoint-selection state machine, the Lightning callbacks,
and ``_train_with_lightning``.  Cross-module collaborators that tests patch
through the ``components`` facade are resolved late-bound via ``_components``.
"""
import dataclasses
import hashlib
import logging
from pathlib import Path
import types
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Union
import uuid

from ptycho.config.config import PyTorchExecutionConfig, TrainingConfig
from ptycho_torch.config_factory import TrainingPayload
from ptycho_torch.scaling_contract import (
    AmplitudePhysicsGainRecord,
    CI_SCALE_CONTRACT,
)
from ptycho_torch.rect_s1s2_initialization import RectS1S2InitializationRecord
from ptycho_torch.train_utils import PrebuiltPtychoDataModule, get_training_strategy
from ptycho_torch.batch_order import (
    DEFAULT_BATCH_ORDER_RECIPE,
    JULY2026_BATCH_ORDER_RECIPE,
    batch_order_provenance,
    validate_batch_order_distribution,
    validate_batch_order_loader_schedule,
    validate_batch_order_recipe,
    validate_july2026_runtime_conformance,
)
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    Callback as _LightningCallback,
    ModelCheckpoint as _LightningModelCheckpoint,
)
from ptycho_torch import model_manager, runtime_provenance, scaling_contract

from . import bundle_io, containers, dataloaders, rect_s1s2

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho_torch.workflows.components')

_CHECKPOINT_SELECTION_SCHEMA = "serving-checkpoint-selection-v1"
def _checkpoint_artifact_path(path, output_root):
    """Return one checkpoint path relative to its training artifact root."""

    if not path:
        return None
    checkpoint_path = Path(path)
    try:
        return checkpoint_path.resolve().relative_to(
            Path(output_root).resolve()
        ).as_posix()
    except ValueError as error:
        raise RuntimeError(
            f"checkpoint path {checkpoint_path} is outside output root "
            f"{output_root}"
        ) from error


def _checkpoint_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_score_value(value):
    if value is None:
        return None
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _in_memory_checkpoint_selection(
    *,
    monitor,
    mode,
    selection_token,
    recovery_path=None,
    output_root=None,
):
    return {
        "schema_version": _CHECKPOINT_SELECTION_SCHEMA,
        "selection_token": selection_token,
        "policy": "final",
        "weights_source": "in_memory",
        "monitor": monitor,
        "mode": mode,
        "selected_path": None,
        "selected_sha256": None,
        "selected_epoch": None,
        "selected_global_step": None,
        "selected_score": None,
        "recovery_path": (
            _checkpoint_artifact_path(recovery_path, output_root)
            if recovery_path and output_root is not None
            else None
        ),
    }


def _write_checkpoint_selection_atomic(path, record):
    """Publish one strict serving-weight decision without partial JSON."""

    import json
    import os
    import tempfile

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checkpoint_selection(path, *, selection_token):
    import json

    try:
        record = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "training did not publish a readable checkpoint selection record"
        ) from error
    if not isinstance(record, dict):
        raise RuntimeError("checkpoint selection record must be a JSON object")
    if record.get("schema_version") != _CHECKPOINT_SELECTION_SCHEMA:
        raise RuntimeError("checkpoint selection record has an unsupported schema")
    if record.get("selection_token") != selection_token:
        raise RuntimeError(
            "checkpoint selection record belongs to a different training invocation"
        )
    semantic_record = dict(record)
    semantic_record.pop("selection_token")
    return semantic_record


def _publish_checkpoint_selection_and_barrier(trainer, path, record):
    """Publish on global zero while the strategy is live, then release ranks."""

    if bool(getattr(trainer, "is_global_zero", False)):
        _write_checkpoint_selection_atomic(path, record)
    strategy = getattr(trainer, "strategy", None)
    barrier = getattr(strategy, "barrier", None)
    if not callable(barrier):
        raise RuntimeError(
            "Lightning strategy must expose barrier() while publishing the "
            "checkpoint selection"
        )
    barrier("serving_checkpoint_selection_published")


def _rank_shared_checkpoint_selection_token(trainer, selection_token):
    """Broadcast rank zero's invocation token through the live strategy."""

    strategy = getattr(trainer, "strategy", None)
    broadcast = getattr(strategy, "broadcast", None)
    if not callable(broadcast):
        raise RuntimeError(
            "Lightning strategy must expose broadcast() while resolving the "
            "checkpoint selection token"
        )
    rank_zero_token = (
        selection_token
        if bool(getattr(trainer, "is_global_zero", False))
        else None
    )
    shared_token = broadcast(rank_zero_token, src=0)
    if not isinstance(shared_token, str) or not shared_token:
        raise RuntimeError(
            "Lightning strategy did not broadcast a valid checkpoint "
            "selection token"
        )
    return shared_token


class _FinalModelSelectionCallback(_LightningCallback):
    """Publish an explicit final-state decision when checkpoints are disabled."""

    def __init__(self, *, selection_sink, selection_path, selection_token):
        super().__init__()
        self.selection_sink = selection_sink
        self.selection_path = Path(selection_path)
        self.selection_token = selection_token

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self.selection_token = _rank_shared_checkpoint_selection_token(
            trainer,
            self.selection_token,
        )

    def on_train_end(self, trainer, pl_module):
        self.selection_sink.clear()
        self.selection_sink.update(
            _in_memory_checkpoint_selection(
                monitor=None,
                mode=None,
                selection_token=self.selection_token,
            )
        )
        _publish_checkpoint_selection_and_barrier(
            trainer,
            self.selection_path,
            self.selection_sink,
        )


class _MilestoneCheckpointCallback(_LightningCallback):
    """Save requested one-based epoch checkpoints without changing selection."""

    def __init__(self, dirpath: Path, epochs: tuple[int, ...]):
        super().__init__()
        if any(type(epoch) is not int or epoch <= 0 for epoch in epochs):
            raise ValueError("milestone epochs must be positive integers")
        if tuple(sorted(set(epochs))) != epochs:
            raise ValueError("milestone epochs must be strictly increasing")
        self.dirpath = Path(dirpath)
        self.epochs = epochs
        self.saved_checkpoints: Dict[int, Path] = {}

    def _save_epoch(self, trainer):
        epoch = int(trainer.current_epoch) + 1
        if epoch not in self.epochs or epoch in self.saved_checkpoints:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        checkpoint = self.dirpath / f"epoch-{epoch:04d}.ckpt"
        trainer.save_checkpoint(str(checkpoint))
        self.saved_checkpoints[epoch] = checkpoint

    def on_validation_end(self, trainer, pl_module):
        del pl_module
        if not trainer.sanity_checking:
            self._save_epoch(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        self._save_epoch(trainer)


class _LossHistoryCallback(_LightningCallback):
    """Collect the dynamic train/validation loss metrics for compatibility."""

    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    @staticmethod
    def _find_loss_metric(metrics, prefix):
        for key in metrics:
            if prefix in key and "loss" in key:
                return float(metrics[key])
        return None

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        value = self._find_loss_metric(trainer.callback_metrics, "train")
        if value is not None:
            self.train_loss.append(value)

    def on_validation_epoch_end(self, trainer, pl_module):
        del pl_module
        value = self._find_loss_metric(trainer.callback_metrics, "val")
        if value is not None:
            self.val_loss.append(value)


class _TrainingSummaryCallback(_LightningCallback):
    """Publish initialization identity while the distributed group is live."""

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.record = None

    def set_record(self, record):
        self.record = RectS1S2InitializationRecord.from_mapping(record)

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        if self.record is None:
            raise RuntimeError(
                "rect_s1s2 initialization record must be set before fit"
            )
        rect_s1s2._publish_training_summary_and_barrier(
            trainer,
            self.path,
            self.record,
        )


class _ServingModelCheckpointMixin:
    """Save the true-final recovery checkpoint, then restore serving weights."""

    def __init__(
        self,
        *,
        selection_sink,
        output_root,
        selection_path=None,
        selection_token=None,
        **kwargs,
    ):
        self.selection_sink = selection_sink
        self.output_root = Path(output_root)
        self.selection_path = Path(
            selection_path
            if selection_path is not None
            else self.output_root / "checkpoint_selection.json"
        )
        self.selection_token = selection_token or uuid.uuid4().hex
        super().__init__(**kwargs)

    def state_dict(self):
        state = super().state_dict()
        state["serving_checkpoint_selection"] = dict(self.selection_sink)
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        selection = state.pop("serving_checkpoint_selection", None)
        if selection:
            self.selection_sink.clear()
            self.selection_sink.update(selection)
        return super().load_state_dict(state)

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self.selection_token = _rank_shared_checkpoint_selection_token(
            trainer,
            self.selection_token,
        )

    def on_train_end(self, trainer, pl_module):
        # ModelCheckpoint owns last.ckpt. It must capture the true final state
        # before this callback changes the in-memory module to serving weights.
        super().on_train_end(trainer, pl_module)

        if self.save_top_k == 0:
            self.selection_sink.clear()
            self.selection_sink.update(
                _in_memory_checkpoint_selection(
                    monitor=self.monitor,
                    mode=self.mode,
                    selection_token=self.selection_token,
                    recovery_path=self.last_model_path,
                    output_root=self.output_root,
                )
            )
            _publish_checkpoint_selection_and_barrier(
                trainer,
                self.selection_path,
                self.selection_sink,
            )
            return

        strategy = getattr(trainer, "strategy", None)
        barrier = getattr(strategy, "barrier", None)
        load_checkpoint = getattr(strategy, "load_checkpoint", None)
        load_model_state_dict = getattr(strategy, "load_model_state_dict", None)
        if not all(
            callable(value)
            for value in (barrier, load_checkpoint, load_model_state_dict)
        ):
            raise RuntimeError(
                "Lightning strategy must expose barrier(), load_checkpoint(), "
                "and load_model_state_dict() for serving checkpoint selection"
            )

        barrier("serving_checkpoint_written")
        selected_path = Path(self.best_model_path) if self.best_model_path else None
        if selected_path is None or not selected_path.is_file():
            raise RuntimeError(
                "selected best checkpoint does not exist; refusing to bundle "
                "undeclared final weights"
            )
        checkpoint = load_checkpoint(selected_path)
        load_model_state_dict(checkpoint, strict=True)
        barrier("serving_checkpoint_restored")

        self.selection_sink.clear()
        self.selection_sink.update(
            {
                "schema_version": _CHECKPOINT_SELECTION_SCHEMA,
                "selection_token": self.selection_token,
                "policy": "best",
                "weights_source": "checkpoint",
                "monitor": self.monitor,
                "mode": self.mode,
                "selected_path": _checkpoint_artifact_path(
                    selected_path,
                    self.output_root,
                ),
                "selected_sha256": _checkpoint_file_sha256(selected_path),
                "selected_epoch": int(checkpoint["epoch"]),
                "selected_global_step": int(checkpoint["global_step"]),
                "selected_score": _checkpoint_score_value(
                    self.best_model_score
                ),
                "recovery_path": _checkpoint_artifact_path(
                    self.last_model_path,
                    self.output_root,
                ),
            }
        )
        _publish_checkpoint_selection_and_barrier(
            trainer,
            self.selection_path,
            self.selection_sink,
        )


class _ServingModelCheckpoint(
    _ServingModelCheckpointMixin,
    _LightningModelCheckpoint,
):
    """ModelCheckpoint whose postcondition is the declared serving state."""


def _resolve_checkpoint_monitor(execution_config, model, *, has_validation=True):
    configured = execution_config.checkpoint_monitor_metric
    if configured == "val_loss":
        return model.val_loss_name if has_validation else model.loss_name
    if configured == "train_loss":
        return model.loss_name
    if not has_validation and "val_" in configured:
        return configured.replace("val_", "train_")
    return configured


def _validate_training_execution_input(
    execution_config: Optional[Any],
    resolved_payload: Optional[TrainingPayload],
) -> None:
    """Validate unresolved workflow input before any legacy-state mutation."""
    if resolved_payload is not None:
        if execution_config is not None:
            raise TypeError(
                "execution_config must be omitted when resolved_payload owns "
                "the resolved PyTorchExecutionConfig"
            )
        return

    from ptycho_torch.execution_request import normalize_execution_input

    normalize_execution_input(execution_config, mode="training")


def _is_global_zero(trainer) -> bool:
    """True on the rank that owns global-side-effect writes."""
    return bool(getattr(trainer, "is_global_zero", True))


@dataclasses.dataclass(frozen=True)
class _PreparedTrainingData:
    """Explicit product of the data-preparation seam."""

    kind: str
    data_module: Optional[PrebuiltPtychoDataModule]
    train_loader: Any
    val_loader: Any
    dataset_size: Optional[int]
    effective_seed: int
    batch_order_recipe: str
    resolved_scale_contract: Any
    datagen_config: Any
    pt_configs: Tuple[Any, Any, Any, Any]
    execution_config: Any


def _prepare_training_data(
    payload: TrainingPayload,
    train_container: Any,
    test_container: Optional['PtychoDataContainerTorch'],
    torch_training_seed: Optional[int],
    batch_order_recipe: str,
    datagen_config: Optional[Any],
) -> _PreparedTrainingData:
    """Resolve seed, configs, and the RAM/mmap data source exactly once."""
    config = payload.tf_training_config

    logger.info("_train_with_lightning orchestrating Lightning training")
    logger.info(f"Training config: nepochs={config.nepochs}, training_groups={config.training_groups}")

    pt_data_config = payload.pt_data_config
    pt_model_config = payload.pt_model_config
    pt_training_config = payload.pt_training_config
    execution_config = payload.execution_config

    # Seed before constructing any module parameters; the same stream is
    # forwarded to the dataloader boundary below.
    effective_torch_training_seed = dataloaders._resolve_torch_training_seed(
        config,
        torch_training_seed,
    )
    batch_order_recipe = validate_batch_order_recipe(batch_order_recipe)
    validate_batch_order_loader_schedule(
        batch_order_recipe,
        has_validation_loader=(
            test_container is not None
            or isinstance(train_container, PrebuiltPtychoDataModule)
        ),
    )
    if batch_order_recipe == JULY2026_BATCH_ORDER_RECIPE:
        validate_july2026_runtime_conformance()
    L.seed_everything(effective_torch_training_seed)

    resolved_scale_contract = scaling_contract.validate_scale_contract(
        pt_data_config,
        pt_model_config,
        pt_training_config,
    )

    pt_inference_config = payload.pt_inference_config
    from ptycho_torch.config_params import DatagenConfig

    if datagen_config is None:
        datagen_config = DatagenConfig()
    elif not isinstance(datagen_config, DatagenConfig):
        raise TypeError("datagen_config must be a DatagenConfig or None")

    # B2.3: Build dataloaders via helper
    data_product = (
        train_container
        if isinstance(train_container, PrebuiltPtychoDataModule)
        else dataloaders._build_lightning_dataloaders(
            train_container,
            test_container,
            config,
            payload=payload,
            torch_training_seed=effective_torch_training_seed,
            batch_order_recipe=batch_order_recipe,
        )
    )

    # DDP-style launchers use the resolved datamodule; other routes return a
    # regular train/validation loader tuple.
    if isinstance(data_product, PrebuiltPtychoDataModule):
        data_module = data_product
        train_loader, val_loader = None, None
        kind = "datamodule"
    else:
        data_module = None
        train_loader, val_loader = data_product
        kind = "loaders"

    # DATA-SUP-001: Supervised mode requires labeled data
    if pt_model_config.mode == 'Supervised':
        # Inspect first batch to verify label keys exist
        try:
            if kind == "datamodule":
                data_module.setup("fit")
                supervised_loader = data_module.train_dataloader()
            else:
                supervised_loader = train_loader
            first_batch = next(iter(supervised_loader))
            batch_dict = first_batch[0]  # Extract tensor dict from batch tuple
            if 'label_amp' not in batch_dict or 'label_phase' not in batch_dict:
                raise RuntimeError(
                    f"Supervised mode (model_type='supervised') requires labeled datasets with "
                    f"'label_amp' and 'label_phase' keys, but training data lacks these fields. "
                    f"Either: (1) Use a labeled NPZ dataset (see ptycho_torch/notebooks/create_supervised_datasets.ipynb), "
                    f"or (2) Switch to PINN mode (--model_type pinn) for self-supervised physics-based training. "
                    f"See DATA-SUP-001 in docs/findings.md for details."
                )
        except StopIteration:
            raise RuntimeError(
                f"Training dataloader is empty. Check dataset path and training_groups configuration."
            )

    dataset_size = (
        len(getattr(train_loader, "dataset", train_container))
        if kind == "loaders"
        else None
    )

    return _PreparedTrainingData(
        kind=kind,
        data_module=data_module,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_size=dataset_size,
        effective_seed=effective_torch_training_seed,
        batch_order_recipe=batch_order_recipe,
        resolved_scale_contract=resolved_scale_contract,
        datagen_config=datagen_config,
        pt_configs=(pt_data_config, pt_model_config, pt_training_config, pt_inference_config),
        execution_config=execution_config,
    )


def _construct_application(payload, pt_configs, ci_statistics):
    """Build the Lightning module once from the sealed structural identity."""
    pt_data_config, _, pt_training_config, pt_inference_config = pt_configs

    from ptycho_torch.application_factory import build_ptychopinn_application

    model = build_ptychopinn_application(
        payload.model_spec,
        pt_data_config,
        pt_training_config,
        pt_inference_config,
    )

    # Save hyperparameters so checkpoint can reconstruct module without external state
    model.save_hyperparameters()

    if ci_statistics is not None:
        model.register_ci_statistics(ci_statistics)

    return model


@dataclasses.dataclass(frozen=True)
class _AssembledTrainer:
    """Explicit product of the trainer-assembly seam."""

    trainer: Any
    trainer_kwargs: Dict[str, Any]
    lightning_logger: Any
    loss_history_cb: Any
    checkpoint_selection_callback: Any
    milestone_callback: Optional[Any]
    milestone_epochs: tuple[int, ...]
    # ponytail: intentionally shared mutable sink — Lightning callbacks write
    # into it during fit; do not freeze
    checkpoint_selection: Dict[str, Any]
    checkpoint_selection_path: Path
    dataloader_settings: Mapping[str, Any]
    rect_s1s2_initialization: Any
    training_summary_path: Path
    output_dir: Path
    total_training_epochs: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataloader_settings",
            types.MappingProxyType(self.dataloader_settings),
        )


def _build_callbacks(prepared, config, model, test_container, milestone_epochs):
    """Build loss/summary/CI/config/metadata and checkpoint-selection callbacks."""
    pt_data_config, pt_model_config, pt_training_config, pt_inference_config = prepared.pt_configs
    execution_config = prepared.execution_config

    # B2.5: Configure Trainer with settings from config
    output_dir = Path(config.output_dir)
    training_summary_path = output_dir / "training_summary.json"

    loss_history_cb = _LossHistoryCallback()
    training_summary_cb = _TrainingSummaryCallback(training_summary_path)

    from ptycho_torch.lightning_utils import (
        CIStatisticsCallback,
        ConfigLogger,
        MetadataLogger,
    )

    callbacks: list = [loss_history_cb, training_summary_cb]
    if prepared.kind == "datamodule":
        callbacks.append(CIStatisticsCallback())
    callbacks.append(
        ConfigLogger(
            data_config=pt_data_config,
            model_config=pt_model_config,
            training_config=pt_training_config,
            inference_config=pt_inference_config,
            datagen_config=prepared.datagen_config,
            run_dir=output_dir,
        )
    )
    callbacks.append(
        MetadataLogger(
            run_dir=output_dir,
            notes=pt_training_config.notes,
            model_name=pt_training_config.model_name,
        )
    )
    total_training_epochs = (
        pt_training_config.epochs + pt_training_config.epochs_fine_tune
    )
    if pt_training_config.epochs_fine_tune > 0:
        from ptycho_torch.train_utils import EncoderFreezeCallback

        callbacks.append(
            EncoderFreezeCallback(
                freeze_at_epoch=pt_training_config.epochs,
                lr_gamma=pt_training_config.fine_tune_gamma,
            )
        )
    checkpoint_selection: dict[str, Any] = {}
    checkpoint_selection_path = output_dir / "checkpoint_selection.json"
    checkpoint_selection_token = uuid.uuid4().hex
    if execution_config.enable_checkpointing:
        from lightning.pytorch.callbacks import EarlyStopping

        has_validation = test_container is not None or prepared.kind == "datamodule"

        monitor_metric = _resolve_checkpoint_monitor(
            execution_config,
            model,
            has_validation=has_validation,
        )

        if has_validation:
            metric_short_name = monitor_metric.replace('_loss', '')
            filename_template = f'epoch={{epoch:02d}}-{metric_short_name}={{{monitor_metric}:.4f}}'
        else:
            filename_template = 'epoch={epoch:02d}'

        checkpoint_selection_callback = _ServingModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename=filename_template,
            monitor=monitor_metric,
            mode=execution_config.checkpoint_mode,
            save_top_k=execution_config.checkpoint_save_top_k,
            save_last=True,  # Always keep last checkpoint for recovery
            verbose=False,
            selection_sink=checkpoint_selection,
            output_root=output_dir,
            selection_path=checkpoint_selection_path,
            selection_token=checkpoint_selection_token,
        )
        callbacks.append(checkpoint_selection_callback)

        if has_validation:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                mode=execution_config.checkpoint_mode,
                patience=execution_config.early_stop_patience,
                verbose=False,
            )
            callbacks.append(early_stop_callback)
    else:
        checkpoint_selection_callback = _FinalModelSelectionCallback(
            selection_sink=checkpoint_selection,
            selection_path=checkpoint_selection_path,
            selection_token=checkpoint_selection_token,
        )
        callbacks.append(checkpoint_selection_callback)

    milestone_callback = None
    if milestone_epochs:
        if not execution_config.enable_checkpointing:
            raise ValueError(
                "milestone checkpoints require checkpointing to be enabled"
            )
        if max(milestone_epochs) > total_training_epochs:
            raise ValueError(
                "milestone epochs cannot exceed the configured training epochs"
            )
        milestone_callback = _MilestoneCheckpointCallback(
            output_dir / "checkpoints" / "milestones",
            milestone_epochs,
        )
        callbacks.append(milestone_callback)

    return (
        callbacks,
        total_training_epochs,
        training_summary_cb,
        checkpoint_selection_callback,
        milestone_callback,
        checkpoint_selection,
        checkpoint_selection_path,
        loss_history_cb,
        training_summary_path,
        output_dir,
    )


def _build_recon_logging_callback(execution_config, callbacks):
    """Append the MLflow-gated recon logging callback when opted in."""
    if (execution_config.logger_backend == 'mlflow'
            and execution_config.recon_log_every_n_epochs is not None):
        from ptycho_torch.workflows.recon_logging import PtychoReconLoggingCallback
        recon_cb = PtychoReconLoggingCallback(
            every_n_epochs=execution_config.recon_log_every_n_epochs,
            num_patches=execution_config.recon_log_num_patches,
            fixed_indices=execution_config.recon_log_fixed_indices,
            log_stitch=execution_config.recon_log_stitch,
            max_stitch_samples=execution_config.recon_log_max_stitch_samples,
        )
        callbacks.append(recon_cb)
        logger.info("Enabled recon logging callback (every %d epochs, %d patches, stitch=%s)",
                     execution_config.recon_log_every_n_epochs,
                     execution_config.recon_log_num_patches,
                     execution_config.recon_log_stitch)


def _build_trainer(callbacks, total_training_epochs, prepared, config, model, output_dir):
    """Assemble the Lightning logger, kwargs, and Trainer."""
    pt_training_config = prepared.pt_configs[2]
    execution_config = prepared.execution_config

    # Instantiate logger based on execution config
    lightning_logger = False  # Default: no logger
    if execution_config.logger_backend is not None:
        try:
            if execution_config.logger_backend == 'csv':
                from lightning.pytorch.loggers import CSVLogger
                lightning_logger = CSVLogger(
                    save_dir=str(output_dir),
                    name='lightning_logs',
                )
                logger.info(f"Enabled CSVLogger: metrics saved to {output_dir}/lightning_logs/")
            elif execution_config.logger_backend == 'tensorboard':
                from lightning.pytorch.loggers import TensorBoardLogger
                lightning_logger = TensorBoardLogger(
                    save_dir=str(output_dir),
                    name='lightning_logs',
                )
                logger.info(f"Enabled TensorBoardLogger: run `tensorboard --logdir={output_dir}/lightning_logs/`")
            elif execution_config.logger_backend == 'mlflow':
                from lightning.pytorch.loggers import MLFlowLogger
                lightning_logger = MLFlowLogger(
                    experiment_name=getattr(config, 'experiment_name', 'PtychoPINN'),
                    tracking_uri=str(output_dir / 'mlruns'),
                )
                logger.info(f"Enabled MLFlowLogger: tracking URI={output_dir}/mlruns")
            else:
                logger.warning(
                    f"Unknown logger_backend '{execution_config.logger_backend}'. "
                    f"Falling back to logger=False. Supported: 'csv', 'tensorboard', 'mlflow'."
                )
        except ImportError as e:
            logger.warning(
                f"Failed to import Lightning logger '{execution_config.logger_backend}': {e}. "
                f"Metrics logging disabled. Install the required package to enable logging."
            )
            lightning_logger = False
    else:
        logger.info("Logger disabled (logger_backend=None). Loss metrics will not be saved to disk.")

    # Not dead: touching ``log_dir`` forces the logger to resolve its lazy
    # version directory, which build_training_history reads metrics.csv from.
    if lightning_logger is not False:
        _ = getattr(lightning_logger, "log_dir", None)


    automatic_optimization = getattr(model, "automatic_optimization", True)
    effective_accum_steps = pt_training_config.accum_steps
    effective_clip_val = pt_training_config.gradient_clip_val
    effective_clip_algorithm = pt_training_config.gradient_clip_algorithm
    if not automatic_optimization and effective_clip_val:
        logger.info(
            "Manual optimization enabled; disabling Lightning Trainer gradient_clip_val "
            "and relying on model-level gradient clipping."
        )
    if automatic_optimization and effective_clip_algorithm == "agc":
        raise ValueError(
            "gradient_clip_algorithm='agc' requires manual optimization; "
            "Lightning automatic optimization accepts only 'norm' or 'value'"
        )

    trainer_kwargs = dict(
        max_epochs=total_training_epochs,
        accelerator=execution_config.accelerator,
        strategy=get_training_strategy(
            execution_config.strategy,
            execution_config.devices,
            accelerator=execution_config.accelerator,
        ),
        deterministic=execution_config.deterministic,
        gradient_clip_val=(
            effective_clip_val if automatic_optimization else None
        ),
        accumulate_grad_batches=(
            effective_accum_steps if automatic_optimization else 1
        ),
        enable_progress_bar=execution_config.enable_progress_bar,
        enable_checkpointing=execution_config.enable_checkpointing,
        callbacks=callbacks,
        devices=execution_config.devices,
        precision=execution_config.precision,
        log_every_n_steps=1,
        default_root_dir=str(output_dir),
        logger=lightning_logger,
    )
    if automatic_optimization:
        trainer_kwargs["gradient_clip_algorithm"] = effective_clip_algorithm
    trainer = L.Trainer(**trainer_kwargs)

    return trainer, trainer_kwargs, lightning_logger


def _init_rect_s1s2(trainer, training_summary_cb, prepared, model):
    """Validate batch order, build dataloader settings, and init rect_s1s2."""
    execution_config = prepared.execution_config
    data_module = prepared.data_module
    train_loader = prepared.train_loader
    pt_model_config = prepared.pt_configs[1]

    accelerator_connector = getattr(trainer, "_accelerator_connector", None)
    validate_batch_order_distribution(
        prepared.batch_order_recipe,
        is_distributed=bool(
            getattr(accelerator_connector, "is_distributed", False)
        ),
    )
    dataloader_settings = rect_s1s2._effective_dataloader_settings(
        data_module,
        train_loader,
        execution_config,
    )

    rect_s1s2_mode = getattr(pt_model_config, "rect_s1s2_init", "ones")
    rect_s1s2_initialization = rect_s1s2._initialize_rect_s1s2(
        model,
        mode=rect_s1s2_mode,
        training_loader=rect_s1s2._rect_s1s2_training_loader(
            data_module,
            train_loader,
            rect_s1s2_mode,
        ),
    )
    logger.info(
        "rect_s1s2 initialization: %s",
        rect_s1s2_initialization,
    )
    training_summary_cb.set_record(rect_s1s2_initialization)

    return dataloader_settings, rect_s1s2_initialization


def _assemble_trainer(prepared, config, model, test_container, milestone_epochs):
    """Build callbacks, logger, and Trainer, then initialize rect_s1s2."""
    execution_config = prepared.execution_config

    (
        callbacks,
        total_training_epochs,
        training_summary_cb,
        checkpoint_selection_callback,
        milestone_callback,
        checkpoint_selection,
        checkpoint_selection_path,
        loss_history_cb,
        training_summary_path,
        output_dir,
    ) = _build_callbacks(prepared, config, model, test_container, milestone_epochs)

    _build_recon_logging_callback(execution_config, callbacks)

    trainer, trainer_kwargs, lightning_logger = _build_trainer(
        callbacks,
        total_training_epochs,
        prepared,
        config,
        model,
        output_dir,
    )

    dataloader_settings, rect_s1s2_initialization = _init_rect_s1s2(
        trainer,
        training_summary_cb,
        prepared,
        model,
    )

    return _AssembledTrainer(
        trainer=trainer,
        trainer_kwargs=trainer_kwargs,
        lightning_logger=lightning_logger,
        loss_history_cb=loss_history_cb,
        checkpoint_selection_callback=checkpoint_selection_callback,
        milestone_callback=milestone_callback,
        milestone_epochs=milestone_epochs,
        checkpoint_selection=checkpoint_selection,
        checkpoint_selection_path=checkpoint_selection_path,
        dataloader_settings=dataloader_settings,
        rect_s1s2_initialization=rect_s1s2_initialization,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        total_training_epochs=total_training_epochs,
    )


@dataclasses.dataclass(frozen=True)
class _FitResult:
    """Explicit product of the fit-and-collect seam."""

    history: Dict[str, Any]
    training_history: Any
    effective_runtime: Dict[str, Any]
    checkpoint_selection: Dict[str, Any]
    selected_checkpoint: Optional[Any]
    milestone_checkpoints: Optional[Dict[int, Any]]
    training_sampling: Any
    dataloader_settings: Dict[str, Any]


def _resolve_selected_checkpoint(state, trainer, execution_config, output_dir, effective_runtime):
    """Resolve the selected checkpoint from the shared selection sink."""
    checkpoint_selection_callback = state.checkpoint_selection_callback
    checkpoint_selection_token = checkpoint_selection_callback.selection_token
    if (
        not execution_config.enable_checkpointing
        and _is_global_zero(trainer)
    ):
        _write_checkpoint_selection_atomic(
            state.checkpoint_selection_path,
            _in_memory_checkpoint_selection(
                monitor=None,
                mode=None,
                selection_token=checkpoint_selection_token,
            ),
        )
    checkpoint_selection = state.checkpoint_selection
    checkpoint_selection.clear()
    checkpoint_selection.update(
        _read_checkpoint_selection(
            state.checkpoint_selection_path,
            selection_token=checkpoint_selection_token,
        )
    )
    effective_runtime["checkpoint_selection"] = dict(checkpoint_selection)

    selected_path = checkpoint_selection.get("selected_path")
    selected_checkpoint = (
        output_dir / selected_path if selected_path is not None else None
    )
    return selected_checkpoint




def _fit_and_collect(state, prepared, model, train_container, test_container):
    """Run the fit cycle and collect history, provenance, and selection."""
    trainer = state.trainer
    execution_config = prepared.execution_config
    output_dir = state.output_dir

    # B2.6: Execute training cycle
    logger.info(
        "Starting Lightning training: %s epochs",
        state.total_training_epochs,
    )
    if prepared.kind == "datamodule":
        fit_kwargs = {"datamodule": prepared.data_module}
    else:
        fit_kwargs = {
            "train_dataloaders": prepared.train_loader,
            "val_dataloaders": prepared.val_loader,
        }
    try:
        trainer.fit(model, **fit_kwargs)
    except Exception as e:
        logger.error(f"Lightning training failed: {e}")
        raise RuntimeError(f"Lightning training failed. See logs for details.") from e

    milestone_checkpoints = None
    if state.milestone_callback is not None:
        milestone_checkpoints = {
            epoch: output_dir
            / "checkpoints"
            / "milestones"
            / f"epoch-{epoch:04d}.ckpt"
            for epoch in state.milestone_epochs
        }
        missing = [
            epoch
            for epoch in state.milestone_epochs
            if not milestone_checkpoints[epoch].is_file()
        ]
        if missing:
            raise RuntimeError(
                "requested milestone checkpoints were not captured: "
                + ", ".join(str(epoch) for epoch in missing)
            )
    if prepared.kind == "datamodule":
        data_module = prepared.data_module
        if data_module.train_dataset is None:
            data_module.setup("fit")
        training_dataset = (
            data_module.train_dataset
            if data_module.train_dataset is not None
            else train_container
        )
        dataset_size = len(training_dataset)
    else:
        dataset_size = prepared.dataset_size

    training_sampling = batch_order_provenance(
        recipe=prepared.batch_order_recipe,
        seed=prepared.effective_seed,
        dataset_size=dataset_size,
        reference_epochs=(
            3 if prepared.batch_order_recipe == JULY2026_BATCH_ORDER_RECIPE else 0
        ),
    )
    effective_settings = {
        **state.dataloader_settings,
        "batch_order": training_sampling,
    }
    effective_runtime = runtime_provenance.build_effective_runtime(
        prepared.effective_seed,
        state.trainer_kwargs,
        execution_config,
        effective_settings,
        trainer=trainer,
    )

    selected_checkpoint = _resolve_selected_checkpoint(
        state,
        trainer,
        execution_config,
        output_dir,
        effective_runtime,
    )

    if _is_global_zero(trainer):
        runtime_provenance.write_effective_runtime_json(
            output_dir / "effective_runtime.json",
            effective_runtime,
        )

    # Extract loss history from the custom callback
    history = {
        "train_loss": state.loss_history_cb.train_loss,
        "val_loss": (
            state.loss_history_cb.val_loss
            if test_container is not None or prepared.kind == "datamodule"
            else None
        ),
    }
    from ptycho_torch.training_history import build_training_history

    training_history = build_training_history(
        output_dir,
        csv_logger=(
            state.lightning_logger
            if execution_config.logger_backend == "csv"
            else None
        ),
        model=model,
        training_config=prepared.pt_configs[2],
    )

    return _FitResult(
        history=history,
        training_history=training_history,
        effective_runtime=effective_runtime,
        checkpoint_selection=dict(state.checkpoint_selection),
        selected_checkpoint=selected_checkpoint,
        milestone_checkpoints=milestone_checkpoints,
        training_sampling=training_sampling,
        dataloader_settings=effective_settings,
    )


class _PersistBundleResult(NamedTuple):
    bundle_path: Optional[Path]
    should_persist: bool


def _persist_bundle(
    state,
    fit,
    model,
    config,
    persist_bundle,
    intensity_scale,
    amplitude_physics_gain_record,
    rescaled_source_sha256=None,
):
    """Persist the strict bundle and detach the module from its Trainer."""
    output_dir = state.output_dir
    trainer = state.trainer

    bundle_path = None
    should_persist = _is_global_zero(trainer)
    if persist_bundle and should_persist:
        archive_path = output_dir / "wts.h5"
        model_manager.save_torch_bundle(
            models_dict={
                "diffraction_to_obj": model,
                "autoencoder": model,
            },
            base_path=str(archive_path),
            config=config,
            intensity_scale=intensity_scale,
        )
        bundle_path = archive_path.with_suffix(".h5.zip")
        if bundle_path.is_file():
            bundle_io._persist_bundle_scaling_metadata(
                bundle_path,
                model,
                amplitude_physics_gain_record=amplitude_physics_gain_record,
                checkpoint_selection=fit.checkpoint_selection,
                training_sampling=fit.training_sampling,
                rescaled_source_sha256=rescaled_source_sha256,
            )
        elif amplitude_physics_gain_record is not None:
            raise RuntimeError(
                "Cannot persist amplitude_physics_gain_record because the "
                "training bundle or resolved model metadata is unavailable."
            )

    model._trainer = None

    logger.info("Lightning training complete")

    return _PersistBundleResult(bundle_path=bundle_path, should_persist=should_persist)


def _train_with_lightning(
    payload: TrainingPayload,
    train_container: Union[
        'PtychoDataContainerTorch', 'PtychoDataset', 'PrebuiltPtychoDataModule',
    ],
    test_container: Optional['PtychoDataContainerTorch'] = None,
    *,
    torch_training_seed: Optional[int] = None,
    batch_order_recipe: str = DEFAULT_BATCH_ORDER_RECIPE,
    datagen_config: Optional[Any] = None,
    milestone_epochs: tuple[int, ...] = (),
    persist_bundle: bool = False,
    intensity_scale: Optional[float] = None,
    amplitude_physics_gain_record: Optional[AmplitudePhysicsGainRecord] = None,
    rescaled_source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate Lightning training across five behavioral seams.

    Consumes one resolved payload and either the RAM rail or a prebuilt mmap
    DataModule; constructs the module/callbacks/Trainer, restores the declared
    serving checkpoint, publishes history and sidecars, and optionally saves a
    strict bundle.
    """
    prepared = _prepare_training_data(
        payload,
        train_container,
        test_container,
        torch_training_seed,
        batch_order_recipe,
        datagen_config,
    )

    ci_statistics = None
    if (
        prepared.resolved_scale_contract is not None
        and prepared.resolved_scale_contract.version == CI_SCALE_CONTRACT
        and prepared.kind == "loaders"
    ):
        ci_statistics = containers._get_finalized_ci_statistics(train_container)

    model = _construct_application(payload, prepared.pt_configs, ci_statistics)

    state = _assemble_trainer(
        prepared, payload.tf_training_config, model, test_container, milestone_epochs,
    )
    fit = _fit_and_collect(state, prepared, model, train_container, test_container)
    bundle = _persist_bundle(
        state, fit, model, payload.tf_training_config, persist_bundle,
        intensity_scale, amplitude_physics_gain_record,
        rescaled_source_sha256=rescaled_source_sha256,
    )

    return {
        "history": fit.history,
        "train_container": train_container,
        "test_container": test_container,
        "rect_s1s2_initialization": state.rect_s1s2_initialization,
        "training_summary_path": state.training_summary_path,
        "execution_config": prepared.execution_config,
        "effective_runtime": fit.effective_runtime,
        "checkpoint_selection": fit.checkpoint_selection,
        "training_sampling": fit.training_sampling,
        "run_dir": state.output_dir,
        "selected_checkpoint": fit.selected_checkpoint,
        "training_history": fit.training_history,
        "milestone_checkpoints": fit.milestone_checkpoints,
        "bundle_path": bundle.bundle_path,
        "should_persist": bundle.should_persist,
        "models": {
            "diffraction_to_obj": model,
            "autoencoder": model,
        },
    }

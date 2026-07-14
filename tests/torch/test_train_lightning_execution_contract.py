import json
import math
from dataclasses import fields
from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset

from ptycho.config.config import PyTorchExecutionConfig
from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
import ptycho_torch.train_lightning_only as train_module
from ptycho_torch import train_utils
from ptycho_torch import reassembly
from ptycho_torch.lightning_utils import MetadataLogger


class _TinyMonitoredModule(L.LightningModule):
    val_loss_name = "poisson_val"

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, _batch_idx):
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(self.layer(inputs), targets)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(self.layer(inputs), targets)
        self.log(self.val_loss_name, loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _tiny_loader():
    inputs = torch.ones(2, 1)
    targets = torch.zeros(2, 1)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def test_execution_config_accepts_explicit_default_contract():
    config = PyTorchExecutionConfig(
        accelerator="cpu",
        devices=1,
        precision="32-true",
    )

    assert config.devices == 1
    assert config.precision == "32-true"


@pytest.mark.parametrize("devices", [1, 3, "auto"])
def test_execution_config_accepts_supported_devices(devices):
    config = PyTorchExecutionConfig(accelerator="cpu", devices=devices)

    assert config.devices == devices


@pytest.mark.parametrize("devices", [0, -1, "gpu", "1", 1.5, True, None])
def test_execution_config_rejects_unsupported_devices(devices):
    with pytest.raises(ValueError, match="devices"):
        PyTorchExecutionConfig(accelerator="cpu", devices=devices)


@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_execution_config_accepts_supported_precision(precision):
    config = PyTorchExecutionConfig(accelerator="cpu", precision=precision)

    assert config.precision == precision


@pytest.mark.parametrize(
    "precision",
    ["32", "16", "bf16", "16-true", "32-mixed", "fp12", None, ["32-true"]],
)
def test_execution_config_rejects_unsupported_precision(precision):
    with pytest.raises(ValueError, match="precision"):
        PyTorchExecutionConfig(accelerator="cpu", precision=precision)


def test_execution_config_rejects_tpu_without_torch_xla_contract():
    with pytest.raises(ValueError, match="Torch-XLA.*TPU.*unsupported"):
        PyTorchExecutionConfig(accelerator="tpu")


def _configs(tmp_path):
    training_config = TrainingConfig(
        device="cpu",
        strategy="auto",
        n_devices=1,
        epochs=1,
        output_dir=str(tmp_path / "output"),
    )
    return (
        DataConfig(),
        ModelConfig(),
        training_config,
        InferenceConfig(),
        DatagenConfig(),
    )


@pytest.mark.parametrize("loader_method", ["train_dataloader", "val_dataloader"])
def test_lightning_data_module_uses_execution_loader_settings(
    monkeypatch,
    loader_method,
):
    captured = []
    monkeypatch.setattr(
        train_utils,
        "TensorDictDataLoader",
        lambda dataset, **kwargs: captured.append((dataset, kwargs)) or object(),
    )
    training_config = TrainingConfig(strategy="auto", num_workers=99)
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        num_workers=3,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=5,
    )
    data_module = train_utils.PtychoDataModuleLightning(
        "unused",
        ModelConfig(),
        DataConfig(),
        training_config,
        execution_config=execution_config,
    )
    data_module.train_dataset = object()
    data_module.val_dataset = object()

    getattr(data_module, loader_method)()

    assert len(captured) == 1
    _, loader_kwargs = captured[0]
    assert loader_kwargs["num_workers"] == 3
    assert loader_kwargs["pin_memory"] is False
    assert loader_kwargs["persistent_workers"] is False
    assert loader_kwargs["prefetch_factor"] == 5
    assert loader_kwargs["collate_fn"].pin_memory_if_cuda is False


@pytest.mark.parametrize("loader_method", ["train_dataloader", "val_dataloader"])
def test_lightning_data_module_normalizes_zero_worker_settings(
    monkeypatch,
    loader_method,
):
    captured = []
    monkeypatch.setattr(
        train_utils,
        "TensorDictDataLoader",
        lambda dataset, **kwargs: captured.append((dataset, kwargs)) or object(),
    )
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=7,
    )
    data_module = train_utils.PtychoDataModuleLightning(
        "unused",
        ModelConfig(),
        DataConfig(),
        TrainingConfig(strategy="auto", num_workers=99),
        execution_config=execution_config,
    )
    data_module.train_dataset = object()
    data_module.val_dataset = object()

    getattr(data_module, loader_method)()

    _, loader_kwargs = captured[0]
    assert loader_kwargs["num_workers"] == 0
    assert loader_kwargs["persistent_workers"] is False
    assert "prefetch_factor" not in loader_kwargs
    assert data_module.effective_dataloader_settings() == {
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": None,
    }


def _install_training_fakes(monkeypatch, run_dir):
    captured = {
        "checkpoint_kwargs": [],
        "early_stop_kwargs": [],
        "progress_count": 0,
    }

    class FakeLogger:
        log_dir = str(run_dir)
        name = "fake"
        version = "version"
        save_dir = str(run_dir.parent)

    class FakeDataModule:
        def __init__(self, _ptycho_dir, _model, _data, training, **_kwargs):
            captured["data_training_config"] = training
            captured["data_module_kwargs"] = _kwargs

        def effective_dataloader_settings(self):
            execution_config = captured["data_module_kwargs"]["execution_config"]
            num_workers = execution_config.num_workers
            return {
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

    class FakeModel:
        loss_name = "train_loss"
        val_loss_name = "poisson_val"

        def __init__(self, _model, _data, training, _inference, **_kwargs):
            self.training_config = training
            self.training = False
            self.lr = None
            self._trainer = None

    class FakeTrainer:
        global_rank = 0

        def __init__(self, **kwargs):
            class FakeAccelerator:
                pass

            class FakePrecisionPlugin:
                precision = kwargs["precision"]

            class FakeStrategy:
                root_device = torch.device(
                    "cuda" if kwargs["accelerator"] == "gpu" else kwargs["accelerator"]
                )
                parallel_devices = [root_device]
                _process_group_backend = (
                    "nccl"
                    if kwargs["accelerator"] == "gpu" and kwargs["strategy"] != "auto"
                    else None
                )

            captured["trainer_kwargs"] = kwargs
            captured["trainer_constructed"] = True
            self.log_dir = str(run_dir)
            self.callbacks = kwargs["callbacks"]
            self.loggers = kwargs["logger"]
            self.accelerator = FakeAccelerator()
            self.precision_plugin = FakePrecisionPlugin()
            self.strategy = FakeStrategy()
            self.num_devices = kwargs["devices"] if isinstance(kwargs["devices"], int) else 1
            self.device_ids = list(range(self.num_devices))

        def fit(self, model, datamodule):
            captured["fit"] = (model, datamodule)
            model._trainer = self

    class FakeModelCheckpoint:
        def __init__(self, **kwargs):
            captured["checkpoint_kwargs"].append(kwargs)

    class FakeEarlyStopping:
        def __init__(self, **kwargs):
            captured["early_stop_kwargs"].append(kwargs)

    class FakeProgressBar:
        def __init__(self, **_kwargs):
            captured["progress_count"] += 1

    monkeypatch.setattr(train_module, "validate_scale_contract", lambda *_args: None)
    monkeypatch.setattr(train_module, "PtychoDataModuleLightning", FakeDataModule)
    monkeypatch.setattr(train_module, "PtychoPINN_Lightning", FakeModel)
    monkeypatch.setattr(train_module.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_module, "ModelCheckpoint", FakeModelCheckpoint)
    monkeypatch.setattr(train_module, "EarlyStopping", FakeEarlyStopping)
    monkeypatch.setattr(train_module, "TQDMProgressBar", FakeProgressBar)
    monkeypatch.setattr(
        train_module,
        "create_experiment_loggers",
        lambda **_kwargs: (FakeLogger(), FakeLogger()),
    )
    monkeypatch.setattr(train_module, "ConfigLogger", lambda **_kwargs: object())
    monkeypatch.setattr(train_module, "MetadataLogger", lambda **_kwargs: object())
    monkeypatch.setattr(train_module, "_build_ci_statistics_callback", object)
    monkeypatch.setattr(train_module, "find_learning_rate", lambda lr, *_args: lr)
    monkeypatch.setattr(train_module, "print_run_summary", lambda *_args: None)
    monkeypatch.setattr(train_module, "find_best_checkpoint", lambda *_args: None)
    monkeypatch.setattr(train_module, "is_effectively_global_rank_zero", lambda: True)
    return captured


def test_main_resolves_default_checkpoint_monitor_to_model_metric(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured = _install_training_fakes(monkeypatch, run_dir)
    monkeypatch.setattr(train_module, "set_seed", lambda *_args, **_kwargs: None)

    train_module.main(
        tmp_path / "data",
        existing_config=_configs(tmp_path),
        output_dir=tmp_path / "output",
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            checkpoint_mode="max",
            checkpoint_save_top_k=2,
            early_stop_patience=7,
        ),
        run_name="default-monitor",
        seed=11,
    )

    assert {
        key: captured["checkpoint_kwargs"][0][key]
        for key in ("monitor", "mode", "save_top_k")
    } == {
        "monitor": "poisson_val",
        "mode": "max",
        "save_top_k": 2,
    }
    assert {
        key: captured["early_stop_kwargs"][0][key]
        for key in ("monitor", "mode", "patience")
    } == {
        "monitor": "poisson_val",
        "mode": "max",
        "patience": 7,
    }


def test_default_monitor_completes_real_lightning_validation(tmp_path):
    model = _TinyMonitoredModule()
    execution_config = PyTorchExecutionConfig(accelerator="cpu")
    monitor = train_module._resolve_checkpoint_monitor(execution_config, model)
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        monitor=monitor,
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode="min",
        patience=1,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[checkpoint, early_stopping],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=_tiny_loader(), val_dataloaders=_tiny_loader())

    assert checkpoint.monitor == model.val_loss_name
    assert checkpoint.best_model_score is not None


def test_milestone_callback_captures_exact_post_epoch_checkpoints_without_best_drift(
    tmp_path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="poisson_val",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
    )
    milestones = train_module._MilestoneCheckpointCallback(
        checkpoint_dir / "milestones", (5, 20, 40, 80)
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=80,
        callbacks=[best, milestones],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        _TinyMonitoredModule(),
        train_dataloaders=_tiny_loader(),
        val_dataloaders=_tiny_loader(),
    )

    assert train_module.find_best_checkpoint(tmp_path) == checkpoint_dir / (
        "best-checkpoint.ckpt"
    )
    assert best.monitor == "poisson_val"
    assert best.mode == "min"
    assert best.save_top_k == 1
    assert tuple(milestones.saved_checkpoints) == (5, 20, 40, 80)
    for external_epoch, checkpoint in milestones.saved_checkpoints.items():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["epoch"] == external_epoch - 1


def _fit_tiny_best_checkpoint(root: Path, *, capture_milestone: bool):
    L.seed_everything(314159, workers=True)
    checkpoint_dir = root / "checkpoints"
    best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="poisson_val",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
    )
    callbacks = [best]
    if capture_milestone:
        callbacks.append(
            train_module._MilestoneCheckpointCallback(
                checkpoint_dir / "milestones", (5,)
            )
        )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=5,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        _TinyMonitoredModule(),
        train_dataloaders=_tiny_loader(),
        val_dataloaders=_tiny_loader(),
    )
    selected = train_module.find_best_checkpoint(root)
    return best.best_model_score, torch.load(
        selected, map_location="cpu", weights_only=False
    )


def test_milestone_capture_does_not_drift_selected_best_score_or_model_payload(
    tmp_path,
):
    absent_score, absent_payload = _fit_tiny_best_checkpoint(
        tmp_path / "absent", capture_milestone=False
    )
    enabled_score, enabled_payload = _fit_tiny_best_checkpoint(
        tmp_path / "enabled", capture_milestone=True
    )

    torch.testing.assert_close(enabled_score, absent_score, rtol=0.0, atol=0.0)
    assert enabled_payload["epoch"] == absent_payload["epoch"]
    assert enabled_payload["global_step"] == absent_payload["global_step"]
    assert enabled_payload["state_dict"].keys() == absent_payload["state_dict"].keys()
    for name, absent_value in absent_payload["state_dict"].items():
        torch.testing.assert_close(
            enabled_payload["state_dict"][name], absent_value, rtol=0.0, atol=0.0
        )


def test_effective_runtime_uses_actual_cpu_trainer_resolution(tmp_path):
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        monitor="poisson_val",
        mode="min",
        save_top_k=2,
    )
    early_stopping = EarlyStopping(
        monitor="poisson_val",
        mode="min",
        patience=4,
    )
    logger = CSVLogger(tmp_path, name="runtime", version="actual")
    trainer_kwargs = {
        "max_epochs": 0,
        "default_root_dir": str(tmp_path),
        "callbacks": [checkpoint, early_stopping],
        "logger": logger,
        "devices": "auto",
        "accelerator": "cpu",
        "strategy": "auto",
        "deterministic": True,
        "enable_progress_bar": False,
        "enable_checkpointing": True,
        "precision": "16-mixed",
    }
    trainer = L.Trainer(**trainer_kwargs, enable_model_summary=False)
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        devices="auto",
        precision="16-mixed",
        checkpoint_save_top_k=2,
        early_stop_patience=4,
    )

    runtime = train_module._build_effective_runtime(
        11,
        trainer_kwargs,
        execution_config,
        dataloader_settings={
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        },
        trainer=trainer,
    )

    assert set(runtime["trainer_kwargs"]) == set(trainer_kwargs)
    assert runtime["requested"]["precision"] == "16-mixed"
    assert runtime["requested"]["deterministic"] is True
    assert runtime["requested"]["enable_progress_bar"] is False
    assert runtime["requested"]["enable_checkpointing"] is True
    assert runtime["requested"]["checkpoint_save_top_k"] == 2
    assert runtime["requested"]["checkpoint_monitor_metric"] == "val_loss"
    assert runtime["requested"]["checkpoint_mode"] == "min"
    assert runtime["requested"]["early_stop_patience"] == 4
    assert runtime["effective"]["precision"]["value"] == trainer.precision_plugin.precision
    assert runtime["effective"]["precision"]["value"] == "bf16-mixed"
    assert runtime["effective"]["deterministic"] == {
        "algorithms_enabled": True,
        "warn_only": False,
    }
    assert runtime["effective"]["environment"]["cuda_available"] is torch.cuda.is_available()
    assert runtime["effective"]["strategy"]["class"].endswith(
        ".SingleDeviceStrategy"
    )
    assert reassembly.resolve_inference_precision_for_device(
        "16-mixed", trainer.strategy.root_device
    ) == "bf16-mixed"
    assert runtime["effective"]["accelerator"]["device_type"] == trainer.strategy.root_device.type
    assert runtime["effective"]["accelerator"]["trainer_value"] == "cpu"
    assert runtime["effective"]["devices"]["count"] == trainer.num_devices
    assert runtime["effective"]["devices"]["ids"] == trainer.device_ids
    assert runtime["effective"]["devices"]["trainer_value"] == "auto"
    assert runtime["effective"]["strategy"]["root_device"] == str(trainer.strategy.root_device)
    assert runtime["effective"]["strategy"]["trainer_value"] == "auto"
    checkpoint_runtime = next(
        callback
        for callback in runtime["effective"]["callbacks"]
        if callback["class"].endswith(".ModelCheckpoint")
    )
    early_stopping_runtime = next(
        callback
        for callback in runtime["effective"]["callbacks"]
        if callback["class"].endswith(".EarlyStopping")
    )
    assert checkpoint_runtime["monitor"] == checkpoint.monitor
    assert checkpoint_runtime["save_top_k"] == checkpoint.save_top_k
    assert early_stopping_runtime["patience"] == early_stopping.patience
    assert runtime["effective"]["loggers"][0]["name"] == logger.name
    assert runtime["effective"]["loggers"][0]["version"] == logger.version


@pytest.mark.parametrize("accelerator", ["cpu", "mps"])
def test_auto_device_count_is_accelerator_aware(monkeypatch, accelerator):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert train_module._effective_device_count("auto", accelerator) == 1


@pytest.mark.parametrize("accelerator", ["cpu", "mps"])
def test_auto_strategy_does_not_force_cuda_backend(monkeypatch, accelerator):
    monkeypatch.setattr(
        train_module,
        "get_training_strategy",
        lambda *_args, **_kwargs: pytest.fail("CPU/MPS auto must stay with Lightning"),
    )

    assert train_module._trainer_strategy("auto", 2, accelerator) == "auto"


@pytest.mark.parametrize(
    ("device_type", "expected_backend"),
    [("cpu", "gloo"), ("cuda", "nccl")],
)
def test_strategy_runtime_resolves_backend_without_process_group(
    device_type,
    expected_backend,
):
    strategy = DDPStrategy(
        parallel_devices=[torch.device(device_type), torch.device(device_type)]
    )

    runtime = train_module._strategy_runtime(strategy)

    assert runtime["process_group_backend"] == expected_backend


@pytest.mark.parametrize(
    ("strategy_name", "start_method", "launcher_suffix"),
    [
        ("ddp", "popen", "._SubprocessScriptLauncher"),
        ("ddp_spawn", "spawn", "._MultiProcessingLauncher"),
    ],
)
def test_strategy_runtime_distinguishes_ddp_launchers(
    strategy_name,
    start_method,
    launcher_suffix,
):
    trainer = L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=strategy_name,
        max_epochs=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    runtime = train_module._strategy_runtime(trainer.strategy)

    assert runtime["start_method"] == start_method
    assert runtime["launcher"]["class"].endswith(launcher_suffix)


def test_strategy_runtime_handles_missing_launcher_internals():
    class MinimalStrategy:
        root_device = torch.device("cpu")
        parallel_devices = [root_device]

    runtime = train_module._strategy_runtime(MinimalStrategy())

    assert "start_method" not in runtime
    assert "launcher" not in runtime


def test_real_cpu_one_device_ddp_runtime_passes_effective_validation(tmp_path):
    trainer_kwargs = {
        "max_epochs": 0,
        "default_root_dir": str(tmp_path),
        "callbacks": [],
        "logger": False,
        "devices": 1,
        "accelerator": "cpu",
        "strategy": "ddp",
        "deterministic": False,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "precision": "32-true",
    }
    trainer = L.Trainer(**trainer_kwargs, enable_model_summary=False)
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        devices=1,
        strategy="ddp",
        deterministic=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        precision="32-true",
    )
    runtime = train_module._build_effective_runtime(
        19,
        trainer_kwargs,
        execution_config,
        dataloader_settings={
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        },
        trainer=trainer,
    )

    assert runtime["effective"]["strategy"]["class"].endswith(".DDPStrategy")
    assert runtime["effective"]["strategy"]["parallel_devices"] == ["cpu"]


def test_main_builds_runtime_after_trainer_construction(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured = _install_training_fakes(monkeypatch, run_dir)
    monkeypatch.setattr(train_module, "set_seed", lambda *_args, **_kwargs: None)
    original = train_module._build_effective_runtime

    def assert_constructed(*args, **kwargs):
        assert captured.get("trainer_constructed") is True
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "_build_effective_runtime", assert_constructed)

    train_module.main(
        tmp_path / "data",
        existing_config=_configs(tmp_path),
        output_dir=tmp_path / "output",
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            enable_checkpointing=False,
        ),
        run_name="construction-order",
        seed=11,
    )


def test_returned_training_config_uses_torch_cuda_alias(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_training_fakes(monkeypatch, run_dir)
    monkeypatch.setattr(train_module, "set_seed", lambda *_args, **_kwargs: None)

    result = train_module.main(
        tmp_path / "data",
        existing_config=_configs(tmp_path),
        output_dir=tmp_path / "output",
        execution_config=PyTorchExecutionConfig(
            accelerator="gpu",
            enable_checkpointing=False,
        ),
        run_name="cuda-alias",
        seed=11,
        return_training_result=True,
    )

    assert result.training_config.device == "cuda"
    assert torch.device(result.training_config.device).type == "cuda"


def test_metadata_logger_handles_checkpointing_without_best_model(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = _TinyMonitoredModule()
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor=model.val_loss_name,
        mode="min",
        save_top_k=0,
    )
    metadata = MetadataLogger(run_dir=str(run_dir))
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[checkpoint, metadata],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=_tiny_loader(), val_dataloaders=_tiny_loader())

    payload = json.loads((run_dir / "metadata.json").read_text())
    assert checkpoint.best_model_score is None
    assert payload["status"] == "completed"
    assert payload["best_val_loss"] is None


def test_training_result_run_dir_is_path_on_nonzero_rank(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_training_fakes(monkeypatch, run_dir)
    monkeypatch.setattr(train_module, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "is_effectively_global_rank_zero", lambda: False)

    result = train_module.main(
        tmp_path / "data",
        existing_config=_configs(tmp_path),
        output_dir=tmp_path / "output",
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            enable_checkpointing=False,
        ),
        run_name="rank-one",
        seed=11,
        return_training_result=True,
    )

    assert isinstance(result.run_dir, Path)
    assert result.run_dir == run_dir
    assert not (run_dir / "effective_runtime.json").exists()


@pytest.mark.parametrize(
    ("enable_checkpointing", "enable_progress_bar", "expected_optional_callbacks"),
    [(False, False, 0), (True, True, 3)],
)
def test_main_honors_effective_execution_contract(
    tmp_path,
    monkeypatch,
    enable_checkpointing,
    enable_progress_bar,
    expected_optional_callbacks,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured = _install_training_fakes(monkeypatch, run_dir)
    seed_calls = []
    monkeypatch.setattr(
        train_module,
        "set_seed",
        lambda seed, n_devices: seed_calls.append((seed, n_devices)),
    )
    monkeypatch.setattr(
        train_module,
        "_resolve_seed",
        lambda: pytest.fail("explicit seed must bypass PTYCHO_TORCH_SEED"),
    )
    configs = _configs(tmp_path)
    execution_config = PyTorchExecutionConfig(
        accelerator="cuda",
        devices=2,
        strategy="ddp",
        deterministic="warn",
        enable_progress_bar=enable_progress_bar,
        enable_checkpointing=enable_checkpointing,
        checkpoint_monitor_metric="configured_val_loss",
        checkpoint_save_top_k=2,
        checkpoint_mode="max",
        early_stop_patience=7,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=5,
        precision="bf16-mixed",
    )

    returned = train_module.main(
        tmp_path / "data",
        existing_config=configs,
        output_dir=tmp_path / "output",
        execution_config=execution_config,
        run_name="contract-test",
        seed=11,
    )

    trainer_kwargs = captured["trainer_kwargs"]
    expected_trainer_kwargs = {
        "devices": 2,
        "accelerator": "gpu",
        "strategy": "ddp",
        "deterministic": "warn",
        "enable_progress_bar": enable_progress_bar,
        "enable_checkpointing": enable_checkpointing,
        "precision": "bf16-mixed",
    }
    assert {key: trainer_kwargs[key] for key in expected_trainer_kwargs} == expected_trainer_kwargs
    assert seed_calls == [(11, 2)]

    training_config = configs[2]
    assert training_config.n_devices == execution_config.devices
    assert training_config.strategy == execution_config.strategy
    assert training_config.device == execution_config.accelerator
    assert training_config.num_workers == execution_config.num_workers
    assert captured["data_training_config"] is training_config

    assert len(captured["checkpoint_kwargs"]) == int(enable_checkpointing)
    assert len(captured["early_stop_kwargs"]) == int(enable_checkpointing)
    if enable_checkpointing:
        assert {
            key: captured["checkpoint_kwargs"][0][key]
            for key in ("monitor", "mode", "save_top_k")
        } == {
            "monitor": "configured_val_loss",
            "mode": "max",
            "save_top_k": 2,
        }
        assert {
            key: captured["early_stop_kwargs"][0][key]
            for key in ("monitor", "mode", "patience")
        } == {
            "monitor": "configured_val_loss",
            "mode": "max",
            "patience": 7,
        }
    assert captured["data_module_kwargs"]["execution_config"] is execution_config
    assert captured["progress_count"] == int(enable_progress_bar)
    callback_count = 4 + expected_optional_callbacks
    assert len(trainer_kwargs["callbacks"]) == callback_count
    assert any(
        type(callback).__name__ == "LearningRateMonitor"
        for callback in trainer_kwargs["callbacks"]
    )

    effective_runtime = json.loads((run_dir / "effective_runtime.json").read_text())
    assert effective_runtime["seed"] == 11
    assert set(effective_runtime["trainer_kwargs"]) == set(trainer_kwargs)
    assert effective_runtime["requested"]["precision"] == "bf16-mixed"
    assert effective_runtime["effective"]["precision"]["value"] == "bf16-mixed"
    assert effective_runtime["dataloader"] == {
        "num_workers": 3,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 5,
    }
    assert effective_runtime["precision"] == "bf16-mixed"
    assert returned == run_dir


def test_main_returns_opt_in_training_result_without_trainer(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured = _install_training_fakes(monkeypatch, run_dir)
    seed_calls = []
    monkeypatch.setattr(
        train_module,
        "set_seed",
        lambda seed, n_devices: seed_calls.append((seed, n_devices)),
    )
    monkeypatch.setenv("PTYCHO_TORCH_SEED", "23")
    configs = _configs(tmp_path)
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        enable_progress_bar=False,
        enable_checkpointing=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=7,
        precision="32-true",
    )

    result = train_module.main(
        tmp_path / "data",
        existing_config=configs,
        output_dir=tmp_path / "output",
        execution_config=execution_config,
        run_name="result-test",
        return_training_result=True,
    )

    assert isinstance(result, train_module.TrainingRunResult)
    assert result.run_dir == run_dir
    assert result.model is captured["fit"][0]
    assert result.model._trainer is None
    assert (
        result.data_config,
        result.model_config,
        result.training_config,
        result.inference_config,
        result.datagen_config,
    ) == configs
    assert result.effective_runtime == json.loads(
        (run_dir / "effective_runtime.json").read_text()
    )
    assert result.effective_runtime["dataloader"] == {
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": None,
    }
    assert seed_calls == [(23, 1)]
    assert "trainer" not in {field.name for field in fields(result)}
    assert "data_module" not in {field.name for field in fields(result)}


def test_training_result_exposes_training_history_from_csv_logs(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_training_fakes(monkeypatch, run_dir)
    monkeypatch.setattr(train_module, "set_seed", lambda *_args, **_kwargs: None)
    (run_dir / "metrics.csv").write_text(
        "epoch,step,train_loss_epoch,grad_norm_preclip_step\n"
        "0,0,,3.5\n"
        "0,1,0.75,2.5\n"
        "1,3,0.5,\n",
        encoding="utf-8",
    )

    result = train_module.main(
        tmp_path / "data",
        existing_config=_configs(tmp_path),
        output_dir=tmp_path / "output",
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            enable_checkpointing=False,
        ),
        run_name="history-test",
        seed=11,
        return_training_result=True,
    )

    history = result.training_history
    assert history["schema_version"] == "training_history_v1"
    assert history["source"] == "lightning_csv_logger"
    assert history["train_loss_name"] == "train_loss"
    assert history["val_loss_name"] == "poisson_val"
    assert history["gradient_clip_val"] is None
    assert history["series"]["train_loss_epoch"] == {
        "step": [1, 3],
        "epoch": [0, 1],
        "value": [0.75, 0.5],
    }
    assert history["series"]["grad_norm_preclip_step"] == {
        "step": [0, 1],
        "epoch": [0, 0],
        "value": [3.5, 2.5],
    }


def test_training_result_history_is_explicitly_absent_without_metrics_csv(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_training_fakes(monkeypatch, run_dir)
    monkeypatch.setattr(train_module, "set_seed", lambda *_args, **_kwargs: None)

    result = train_module.main(
        tmp_path / "data",
        existing_config=_configs(tmp_path),
        output_dir=tmp_path / "output",
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            enable_checkpointing=False,
        ),
        run_name="no-history-test",
        seed=11,
        return_training_result=True,
    )

    assert result.training_history is None
    # Backward-compatible optional field: absent history defaults to None.
    history_field = next(
        field
        for field in fields(train_module.TrainingRunResult)
        if field.name == "training_history"
    )
    assert history_field.default is None


def test_build_training_history_parses_real_lightning_csv(tmp_path):
    from ptycho_torch.training_history import build_training_history

    model = _TinyMonitoredModule()
    logger = CSVLogger(tmp_path, name="history", version="real")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=_tiny_loader(), val_dataloaders=_tiny_loader())

    history = build_training_history(
        Path(logger.log_dir),
        csv_logger=logger,
        model=model,
        training_config=None,
    )
    assert history is not None
    assert history["val_loss_name"] == "poisson_val"
    losses = history["series"]["train_loss_epoch"]["value"]
    assert losses
    assert all(math.isfinite(value) for value in losses)
    val_losses = history["series"]["poisson_val"]["value"]
    assert val_losses
    assert all(math.isfinite(value) for value in val_losses)

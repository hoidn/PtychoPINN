"""Size-gate and seam-contract tests for ``_train_with_lightning``.

Phase 2 decomposes the 641-line coordinator into five behavioral seams. These
tests lock the two structural guarantees the decomposition must preserve:

* the coordinator stays a straight-line orchestrator (<= 100 source lines);
* the RAM-vs-mmap discrimination happens exactly once in the data seam; and
* seeding precedes module construction.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptycho_torch.workflows import lightning_service


def _execution_request(**values):
    from ptycho_torch.execution_request import ExecutionRequest

    return ExecutionRequest(
        values=values,
        explicit_fields=frozenset(values),
    )


def _build_payload(tmp_path, **execution_values):
    """Resolve a TrainingPayload from a canonical TrainingConfig."""
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.config_factory import resolve_training_payload

    train_npz = tmp_path / "train.npz"
    np.savez(train_npz, probeGuess=np.ones((64, 64), dtype=np.complex64))
    config = TrainingConfig(
        model=ModelConfig(N=64),
        train_data_file=train_npz,
        output_dir=tmp_path,
        training_groups=10,
        nphotons=1e9,
        nepochs=1,
    )
    execution_config = (
        _execution_request(**execution_values) if execution_values else None
    )
    return resolve_training_payload(
        train_data_file=train_npz,
        output_dir=tmp_path,
        overrides={"training_groups": 10, "nphotons": 1e9},
        execution_config=execution_config,
        training_baseline=config,
    )


def test_coordinator_size_gate():
    """The coordinator stays a straight-line orchestrator under 100 lines."""
    source = inspect.getsource(lightning_service._train_with_lightning)
    assert len(source.splitlines()) <= 100


def test_prepare_training_data_keeps_datamodule(tmp_path, monkeypatch):
    """A selected PrebuiltPtychoDataModule is reused, never rebuilt."""
    from ptycho_torch.batch_order import DEFAULT_BATCH_ORDER_RECIPE
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    payload = _build_payload(tmp_path)
    data_module = PrebuiltPtychoDataModule(
        "unused-map",
        payload.pt_model_config,
        payload.pt_data_config,
        payload.pt_training_config,
        execution_config=payload.execution_config,
    )
    monkeypatch.setattr(
        "ptycho_torch.workflows.dataloaders._build_lightning_dataloaders",
        lambda *_args, **_kwargs: pytest.fail(
            "a selected DataModule must not be rebuilt"
        ),
    )

    product = lightning_service._prepare_training_data(
        payload, data_module, None, None, DEFAULT_BATCH_ORDER_RECIPE, None,
    )

    assert product.kind == "datamodule"
    assert product.data_module is data_module
    assert product.train_loader is None
    assert product.val_loader is None
    assert product.dataset_size is None


def test_prepare_training_data_builds_loaders_for_container(tmp_path, monkeypatch):
    """A RAM/mmap container is routed through the loader builder."""
    from ptycho_torch.batch_order import DEFAULT_BATCH_ORDER_RECIPE

    payload = _build_payload(tmp_path)
    container = {"X": np.zeros((2, 16, 16, 1), dtype=np.float32)}
    fake_train = SimpleNamespace(dataset=list(range(5)))
    fake_val = SimpleNamespace()
    monkeypatch.setattr(
        "ptycho_torch.workflows.dataloaders._build_lightning_dataloaders",
        lambda *_args, **_kwargs: (fake_train, fake_val),
    )

    product = lightning_service._prepare_training_data(
        payload, container, None, None, DEFAULT_BATCH_ORDER_RECIPE, None,
    )

    assert product.kind == "loaders"
    assert product.data_module is None
    assert product.train_loader is fake_train
    assert product.val_loader is fake_val
    assert product.dataset_size == 5


def test_seed_runs_before_application_construction(tmp_path, monkeypatch):
    """Seeding precedes module construction in the coordinator call order."""
    from ptycho_torch.train_utils import PrebuiltPtychoDataModule

    payload = _build_payload(
        tmp_path,
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        num_workers=0,
        enable_checkpointing=False,
    )
    data_module = PrebuiltPtychoDataModule(
        "unused-map",
        payload.pt_model_config,
        payload.pt_data_config,
        payload.pt_training_config,
        execution_config=payload.execution_config,
    )
    data_module.train_dataset = torch.utils.data.TensorDataset(torch.arange(24))

    calls = []

    monkeypatch.setattr(
        "lightning.pytorch.seed_everything",
        lambda *_args, **_kwargs: calls.append("seed"),
    )
    monkeypatch.setattr(
        "ptycho_torch.workflows.dataloaders._build_lightning_dataloaders",
        lambda *_args, **_kwargs: pytest.fail(
            "a selected DataModule must not be rebuilt"
        ),
    )

    class StubModel:
        automatic_optimization = True
        val_loss_name = "val_loss"

        def save_hyperparameters(self):
            pass

    def spy_build(*_args, **_kwargs):
        calls.append("build")
        return StubModel()

    monkeypatch.setattr(
        "ptycho_torch.application_factory.build_ptychopinn_application", spy_build
    )

    monkeypatch.setattr(
        "ptycho_torch.workflows.rect_s1s2._initialize_rect_s1s2",
        lambda *_args, **_kwargs: {
            "schema_version": "rect-s1s2-initialization-v2",
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        },
    )
    monkeypatch.setattr(
        "ptycho_torch.runtime_provenance.build_effective_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "ptycho_torch.runtime_provenance.write_effective_runtime_json",
        lambda *_args, **_kwargs: None,
    )

    class StubTrainer:
        is_global_zero = True
        _accelerator_connector = SimpleNamespace(is_distributed=False)

        def fit(self, model, **kwargs):
            model._trainer = self

    monkeypatch.setattr(
        "lightning.pytorch.Trainer", lambda **_kwargs: StubTrainer()
    )

    lightning_service._train_with_lightning(payload, data_module, None)

    assert "seed" in calls
    assert "build" in calls
    assert calls.index("seed") < calls.index("build")


def test_assembled_trainer_dataloader_settings_immutable():
    """Mutation of _AssembledTrainer.dataloader_settings raises TypeError."""
    from ptycho_torch.workflows.lightning_service import _AssembledTrainer

    assembled = _AssembledTrainer(
        trainer=None,
        trainer_kwargs={},
        lightning_logger=False,
        loss_history_cb=None,
        checkpoint_selection_callback=None,
        milestone_callback=None,
        milestone_epochs=(),
        checkpoint_selection={},
        checkpoint_selection_path=None,
        dataloader_settings={"batch_size": 4},
        rect_s1s2_initialization={},
        training_summary_path=None,
        output_dir=None,
        total_training_epochs=1,
    )
    with pytest.raises(TypeError):
        assembled.dataloader_settings["batch_order"] = "x"

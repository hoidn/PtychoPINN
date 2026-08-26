"""Serving bundles select declared weights without corrupting recovery state."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch


class _FakeCheckpointBase:
    """Small ModelCheckpoint stand-in that saves true-final state first."""

    def __init__(
        self,
        *,
        save_top_k,
        monitor="val_loss",
        mode="min",
        **_kwargs,
    ):
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_model_path = ""
        self.last_model_path = ""
        self.best_model_score = None

    def on_train_end(self, trainer, pl_module):
        trainer.events.append("save_true_last")


class _FakeStrategy:
    def __init__(self, events, checkpoint):
        self.events = events
        self.checkpoint = checkpoint
        self.loaded_state = None

    def barrier(self, name=None):
        self.events.append(("barrier", name))

    def load_checkpoint(self, path):
        self.events.append(("load_checkpoint", str(path)))
        return self.checkpoint

    def load_model_state_dict(self, checkpoint, strict=True):
        self.events.append(("load_model_state_dict", strict))
        self.loaded_state = checkpoint["state_dict"]


def _callback_type():
    from ptycho_torch.workflows.components import _ServingModelCheckpointMixin

    class Callback(_ServingModelCheckpointMixin, _FakeCheckpointBase):
        pass

    return Callback


def test_serving_checkpoint_saves_true_last_before_strict_best_restore(tmp_path):
    best = tmp_path / "checkpoints" / "epoch=03.ckpt"
    last = tmp_path / "checkpoints" / "last.ckpt"
    best.parent.mkdir()
    best.write_bytes(b"best checkpoint")
    last.write_bytes(b"true final checkpoint")
    sink = {}
    callback = _callback_type()(
        selection_sink=sink,
        output_root=tmp_path,
        save_top_k=1,
        monitor="poisson_val_Amp_loss",
        mode="min",
    )
    callback.best_model_path = str(best)
    callback.last_model_path = str(last)
    callback.best_model_score = torch.tensor(12.5)
    events = []
    strategy = _FakeStrategy(
        events,
        {"state_dict": {"weight": torch.tensor([3.0])}, "epoch": 3, "global_step": 40},
    )
    trainer = SimpleNamespace(events=events, strategy=strategy)

    callback.on_train_end(trainer, object())

    assert events[0] == "save_true_last"
    assert events[1:] == [
        ("barrier", "serving_checkpoint_written"),
        ("load_checkpoint", str(best)),
        ("load_model_state_dict", True),
        ("barrier", "serving_checkpoint_restored"),
        ("barrier", "serving_checkpoint_selection_published"),
    ]
    assert strategy.loaded_state["weight"].item() == 3.0
    assert sink["schema_version"] == "serving-checkpoint-selection-v1"
    assert sink["policy"] == "best"
    assert sink["weights_source"] == "checkpoint"
    assert sink["monitor"] == "poisson_val_Amp_loss"
    assert sink["mode"] == "min"
    assert sink["selected_path"] == "checkpoints/epoch=03.ckpt"
    assert sink["selected_epoch"] == 3
    assert sink["selected_global_step"] == 40
    assert sink["selected_score"] == 12.5
    assert sink["recovery_path"] == "checkpoints/last.ckpt"
    assert len(sink["selected_sha256"]) == 64


def test_serving_checkpoint_top_k_zero_declares_final_without_reload(tmp_path):
    sink = {}
    callback = _callback_type()(
        selection_sink=sink,
        output_root=tmp_path,
        save_top_k=0,
    )
    callback.last_model_path = str(tmp_path / "checkpoints" / "last.ckpt")
    events = []
    trainer = SimpleNamespace(
        events=events,
        strategy=_FakeStrategy(events, checkpoint={}),
    )

    callback.on_train_end(trainer, object())

    assert events == [
        "save_true_last",
        ("barrier", "serving_checkpoint_selection_published"),
    ]
    assert sink["policy"] == "final"
    assert sink["weights_source"] == "in_memory"
    assert sink["selected_path"] is None
    assert sink["recovery_path"] == "checkpoints/last.ckpt"


def test_serving_checkpoint_fails_closed_when_declared_best_is_missing(tmp_path):
    callback = _callback_type()(
        selection_sink={},
        output_root=tmp_path,
        save_top_k=1,
    )
    callback.best_model_path = str(tmp_path / "missing.ckpt")
    events = []
    trainer = SimpleNamespace(
        events=events,
        strategy=_FakeStrategy(events, checkpoint={}),
    )

    with pytest.raises(RuntimeError, match="selected best checkpoint.*does not exist"):
        callback.on_train_end(trainer, object())

    assert events == [
        "save_true_last",
        ("barrier", "serving_checkpoint_written"),
    ]


def test_checkpoint_selection_token_is_transport_only(tmp_path):
    from ptycho_torch.workflows.components import (
        _read_checkpoint_selection,
        _write_checkpoint_selection_atomic,
    )

    path = tmp_path / "checkpoint_selection.json"
    raw = {
        "schema_version": "serving-checkpoint-selection-v1",
        "selection_token": "invocation-token",
        "policy": "best",
    }
    _write_checkpoint_selection_atomic(path, raw)

    assert "selection_token" in path.read_text()
    assert _read_checkpoint_selection(
        path,
        selection_token="invocation-token",
    ) == {
        "schema_version": "serving-checkpoint-selection-v1",
        "policy": "best",
    }
    with pytest.raises(RuntimeError, match="different training invocation"):
        _read_checkpoint_selection(path, selection_token="stale-token")


def test_milestones_capture_exact_post_epoch_files(tmp_path):
    from ptycho_torch.workflows.components import _MilestoneCheckpointCallback

    callback = _MilestoneCheckpointCallback(tmp_path, (1, 3))
    saved = []
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        save_checkpoint=lambda path: (
            Path(path).write_bytes(b"checkpoint"),
            saved.append(Path(path)),
        ),
    )

    callback.on_validation_end(trainer, object())
    callback.on_train_epoch_end(trainer, object())
    trainer.current_epoch = 1
    callback.on_train_epoch_end(trainer, object())
    trainer.current_epoch = 2
    callback.on_train_epoch_end(trainer, object())

    assert saved == [tmp_path / "epoch-0001.ckpt", tmp_path / "epoch-0003.ckpt"]
    assert callback.saved_checkpoints == {1: saved[0], 3: saved[1]}


def test_non_global_rank_training_result_does_not_publish_bundle(
    tmp_path,
    monkeypatch,
):
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.workflows import components

    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        train_data_file=tmp_path / "train.npz",
        test_data_file=tmp_path / "test.npz",
        output_dir=tmp_path,
    )
    monkeypatch.setattr(
        components,
        "train_cdi_model_torch",
        lambda *_args, **_kwargs: {
            "models": {"autoencoder": object(), "diffraction_to_obj": object()},
            "should_persist": False,
        },
    )
    monkeypatch.setattr(
        components,
        "save_torch_bundle",
        lambda **_kwargs: pytest.fail("non-global rank attempted bundle publication"),
    )

    _amplitude, _phase, result = components.run_cdi_example_torch(
        train_data=object(),
        test_data=None,
        config=config,
    )

    assert result["should_persist"] is False
    assert "bundle_path" not in result


def test_real_lightning_keeps_true_last_but_returns_best_serving_state(tmp_path):
    import lightning.pytorch as L
    from torch.utils.data import DataLoader, TensorDataset

    from ptycho_torch.workflows.components import _ServingModelCheckpoint

    class WorseningModule(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def training_step(self, _batch, _batch_idx):
            with torch.no_grad():
                self.weight.fill_(float(self.current_epoch + 1))
            return self.weight * 0.0

        def validation_step(self, _batch, _batch_idx):
            self.log("val_loss", self.weight, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    sink = {}
    callback = _ServingModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="epoch={epoch:02d}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        selection_sink=sink,
        output_root=tmp_path,
    )
    module = WorseningModule()
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=2)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        callbacks=[callback],
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    best = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)
    last = torch.load(callback.last_model_path, map_location="cpu", weights_only=False)
    assert best["state_dict"]["weight"].item() == pytest.approx(1.0)
    assert last["state_dict"]["weight"].item() == pytest.approx(2.0)
    assert module.weight.item() == pytest.approx(1.0)
    assert sink["policy"] == "best"
    assert sink["selected_epoch"] == 0
    assert sink["selected_path"] == "checkpoints/epoch=epoch=00.ckpt"
    assert sink["recovery_path"] == "checkpoints/last.ckpt"


@pytest.mark.parametrize(
    ("configured", "has_validation", "expected"),
    [
        ("val_loss", True, "poisson_val_Amp_loss"),
        ("val_loss", False, "poisson_train_Amp_loss"),
        ("train_loss", True, "poisson_train_Amp_loss"),
        ("train_loss", False, "poisson_train_Amp_loss"),
        ("custom_val_score", False, "custom_train_score"),
    ],
)
def test_checkpoint_monitor_aliases_match_logged_model_metrics(
    configured,
    has_validation,
    expected,
):
    from ptycho_torch.workflows.components import _resolve_checkpoint_monitor

    execution = SimpleNamespace(checkpoint_monitor_metric=configured)
    model = SimpleNamespace(
        loss_name="poisson_train_Amp_loss",
        val_loss_name="poisson_val_Amp_loss",
    )

    assert (
        _resolve_checkpoint_monitor(
            execution,
            model,
            has_validation=has_validation,
        )
        == expected
    )


def test_real_lightning_without_validation_selects_dynamic_training_loss(tmp_path):
    import lightning.pytorch as L
    from torch.utils.data import DataLoader, TensorDataset

    from ptycho_torch.workflows.components import (
        _ServingModelCheckpoint,
        _resolve_checkpoint_monitor,
    )

    class TrainingOnlyModule(L.LightningModule):
        loss_name = "poisson_train_Amp_loss"
        val_loss_name = "poisson_val_Amp_loss"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def training_step(self, _batch, _batch_idx):
            loss = (self.weight - 1.0).square()
            self.log(self.loss_name, loss, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    module = TrainingOnlyModule()
    monitor = _resolve_checkpoint_monitor(
        SimpleNamespace(checkpoint_monitor_metric="val_loss"),
        module,
        has_validation=False,
    )
    sink = {}
    callback = _ServingModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="epoch={epoch:02d}",
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
        selection_sink=sink,
        output_root=tmp_path,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        callbacks=[callback],
        enable_progress_bar=False,
    )
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=2)

    trainer.fit(module, train_dataloaders=loader)

    assert monitor == module.loss_name
    assert Path(callback.best_model_path).is_file()
    assert sink["monitor"] == module.loss_name
    assert sink["policy"] == "best"


def test_real_lightning_ddp_shares_selection_token_and_only_rank_zero_publishes(
    tmp_path,
):
    """Every DDP rank validates one rank-zero checkpoint-selection record."""

    script = tmp_path / "ddp_checkpoint_selection.py"
    output_dir = tmp_path / "artifacts"
    script.write_text(
        textwrap.dedent(
            """
            import json
            from pathlib import Path
            import sys
            import uuid

            import lightning.pytorch as L
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            sys.path.insert(0, sys.argv[2])
            from ptycho_torch.workflows import components


            class TinyModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor(0.0))

                def training_step(self, _batch, _batch_idx):
                    return self.weight * 0.0

                def validation_step(self, _batch, _batch_idx):
                    self.log("val_loss", self.weight, on_epoch=True, sync_dist=True)

                def configure_optimizers(self):
                    return torch.optim.SGD(self.parameters(), lr=0.0)


            class RankEvidence(L.Callback):
                def __init__(self, selection_callback, selection_path, output_dir):
                    super().__init__()
                    self.selection_callback = selection_callback
                    self.selection_path = selection_path
                    self.output_dir = output_dir

                def on_fit_end(self, trainer, _pl_module):
                    try:
                        selection = components._read_checkpoint_selection(
                            self.selection_path,
                            selection_token=self.selection_callback.selection_token,
                        )
                        policy = selection["policy"]
                        validation_error = None
                    except RuntimeError as error:
                        policy = None
                        validation_error = str(error)
                    rank = trainer.global_rank
                    (self.output_dir / f"rank-{rank}.json").write_text(
                        json.dumps(
                            {
                                "selection_token": (
                                    self.selection_callback.selection_token
                                ),
                                "policy": policy,
                                "should_persist": trainer.is_global_zero,
                                "validation_error": validation_error,
                            }
                        )
                    )


            def main():
                output_dir = Path(sys.argv[1])
                output_dir.mkdir(parents=True, exist_ok=True)
                selection_path = output_dir / "checkpoint_selection.json"
                selection_sink = {}
                callback = components._ServingModelCheckpoint(
                    dirpath=output_dir / "checkpoints",
                    filename="epoch={epoch:02d}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    selection_sink=selection_sink,
                    output_root=output_dir,
                    selection_path=selection_path,
                    selection_token=uuid.uuid4().hex,
                )

                write_selection = components._write_checkpoint_selection_atomic

                def tracked_write(path, record):
                    rank = torch.distributed.get_rank()
                    (output_dir / f"published-by-rank-{rank}").touch()
                    write_selection(path, record)

                components._write_checkpoint_selection_atomic = tracked_write
                trainer = L.Trainer(
                    accelerator="cpu",
                    devices=2,
                    strategy="ddp",
                    max_epochs=1,
                    logger=False,
                    callbacks=[
                        callback,
                        RankEvidence(callback, selection_path, output_dir),
                    ],
                    enable_checkpointing=True,
                    enable_progress_bar=False,
                    num_sanity_val_steps=0,
                )
                loader = DataLoader(
                    TensorDataset(torch.ones(2, 1)),
                    batch_size=1,
                )
                trainer.fit(
                    TinyModule(),
                    train_dataloaders=loader,
                    val_dataloaders=loader,
                )


            if __name__ == "__main__":
                main()
            """
        )
    )

    source_root = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, str(script), str(output_dir), str(source_root)],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    rank_records = [
        json.loads((output_dir / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    published_markers = sorted(
        path.name for path in output_dir.glob("published-by-rank-*")
    )
    published_selection = json.loads(
        (output_dir / "checkpoint_selection.json").read_text()
    )

    assert rank_records[0]["selection_token"] == rank_records[1]["selection_token"]
    assert published_selection["selection_token"] == rank_records[0]["selection_token"]
    assert [record["policy"] for record in rank_records] == ["best", "best"]
    assert [record["validation_error"] for record in rank_records] == [None, None]
    assert [record["should_persist"] for record in rank_records] == [True, False]
    assert published_markers == ["published-by-rank-0"]


@pytest.mark.integration
def test_shared_service_cpu_ddp_spawn_returns_parent_artifacts(tmp_path):
    """The shared service survives spawn and reopens child-owned run state."""

    script = tmp_path / "shared_service_spawn.py"
    output_dir = tmp_path / "artifacts"
    script.write_text(
        textwrap.dedent(
            """
            import json
            from pathlib import Path
            import sys

            import lightning.pytorch as L
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            sys.path.insert(0, sys.argv[2])
            from ptycho.config.config import PyTorchExecutionConfig
            from ptycho_torch import application_factory
            from ptycho_torch.config_factory import (
                create_training_payload_from_resolved_configs,
            )
            from ptycho_torch.config_params import (
                DataConfig,
                InferenceConfig,
                ModelConfig,
                TrainingConfig,
            )
            from ptycho_torch.train_utils import PrebuiltPtychoDataModule
            from ptycho_torch.workflows import components


            class TinyModule(L.LightningModule):
                automatic_optimization = True
                loss_name = "train_loss"
                val_loss_name = "val_loss"

                def __init__(
                    self,
                    model_spec,
                    data_config,
                    training_config,
                    inference_config,
                ):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor(0.0))
                    self._model_spec = model_spec
                    self.model_config = model_spec.to_model_config()
                    self.data_config = data_config
                    self.training_config = training_config
                    self.inference_config = inference_config

                def get_ci_statistics(self):
                    return None

                def training_step(self, _batch, _batch_idx):
                    loss = (self.weight - 1.0).square()
                    self.log("train_loss", loss, on_epoch=True, sync_dist=True)
                    return loss

                def validation_step(self, _batch, _batch_idx):
                    self.log(
                        "val_loss",
                        self.weight.square(),
                        on_epoch=True,
                        sync_dist=True,
                    )

                def configure_optimizers(self):
                    return torch.optim.SGD(self.parameters(), lr=0.1)


            class SpawnDataModule(PrebuiltPtychoDataModule):
                def prepare_data(self):
                    Path(self.map_path).mkdir(parents=True, exist_ok=True)

                def setup(self, stage=None):
                    if self.dataset is not None or stage not in ("fit", None):
                        return
                    self.dataset = TensorDataset(torch.arange(8).float().unsqueeze(1))
                    self.train_dataset = self.dataset
                    self.val_dataset = TensorDataset(torch.arange(2).float().unsqueeze(1))

                def train_dataloader(self):
                    return DataLoader(self.train_dataset, batch_size=2)

                def val_dataloader(self):
                    return DataLoader(self.val_dataset, batch_size=2)


            def main():
                output_dir = Path(sys.argv[1])
                data = DataConfig(
                    N=64,
                    C=1,
                    K=1,
                    n_subsample=1,
                    grid_size=(1, 1),
                    scale_contract_version="legacy_v1",
                    measurement_domain="normalized_amplitude",
                )
                model = ModelConfig(
                    C_model=1,
                    C_forward=1,
                    object_big=False,
                    cbam_encoder=False,
                    rect_s1s2_trainable=False,
                )
                training = TrainingConfig(
                    device="cpu",
                    strategy="ddp_spawn",
                    n_devices=2,
                    orchestrator="Lightning",
                    epochs=2,
                    batch_size=2,
                    num_workers=0,
                )
                inference = InferenceConfig()
                execution = PyTorchExecutionConfig(
                    accelerator="cpu",
                    devices=2,
                    strategy="ddp_spawn",
                    num_workers=0,
                    enable_progress_bar=False,
                    enable_checkpointing=True,
                    logger_backend="csv",
                )
                payload = create_training_payload_from_resolved_configs(
                    data,
                    model,
                    training,
                    inference,
                    execution,
                    train_data_file=output_dir / "train.npz",
                    output_dir=output_dir,
                    n_groups=8,
                )
                application_factory.build_ptychopinn_application = TinyModule

                def save_bundle(*, models_dict, base_path, **_kwargs):
                    selection = json.loads(
                        (output_dir / "checkpoint_selection.json").read_text()
                    )
                    checkpoint = torch.load(
                        output_dir / selection["selected_path"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    current = models_dict["diffraction_to_obj"].state_dict()
                    assert all(
                        torch.equal(current[name].cpu(), value.cpu())
                        for name, value in checkpoint["state_dict"].items()
                    )
                    Path(f"{base_path}.zip").write_bytes(b"selected weights observed")

                components.save_torch_bundle = save_bundle
                components._persist_bundle_scaling_metadata = lambda *_args, **_kwargs: None
                data_module = SpawnDataModule(
                    output_dir / "map",
                    model,
                    data,
                    training,
                    execution_config=execution,
                )
                result = components._train_with_lightning(
                    data_module,
                    None,
                    payload.tf_training_config,
                    resolved_payload=payload,
                    milestone_epochs=(1, 2),
                    persist_bundle=True,
                )
                evidence = {
                    "parent_dataset_open": data_module.train_dataset is not None,
                    "trainer_detached": result["models"]["diffraction_to_obj"]._trainer is None,
                    "selected_checkpoint": result["selected_checkpoint"].is_file(),
                    "milestones": all(
                        path.is_file()
                        for path in result["milestone_checkpoints"].values()
                    ),
                    "history": result["training_history"]["schema_version"],
                    "bundle": result["bundle_path"].is_file(),
                    "artifacts": all(
                        (output_dir / path).is_file()
                        for path in (
                            "configs/full_config.json",
                            "metadata.json",
                            "training_summary.json",
                            "checkpoint_selection.json",
                            "effective_runtime.json",
                        )
                    ),
                }
                (output_dir / "evidence.json").write_text(json.dumps(evidence))


            if __name__ == "__main__":
                main()
            """
        )
    )

    source_root = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, str(script), str(output_dir), str(source_root)],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((output_dir / "evidence.json").read_text()) == {
        "parent_dataset_open": True,
        "trainer_detached": True,
        "selected_checkpoint": True,
        "milestones": True,
        "history": "training_history_v1",
        "bundle": True,
        "artifacts": True,
    }

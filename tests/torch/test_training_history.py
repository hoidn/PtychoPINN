"""Regression tests for shared Lightning CSV history publication."""

from pathlib import Path


def test_csv_history_path_does_not_create_a_parent_experiment():
    from ptycho_torch.training_history import _metrics_csv_path

    class Logger:
        log_dir = "/tmp/run/lightning_logs/version_0"

        @property
        def experiment(self):
            raise AssertionError("history parsing must not reopen the CSV experiment")

    assert _metrics_csv_path(Path("/tmp/run"), Logger()) == Path(
        "/tmp/run/lightning_logs/version_0/metrics.csv"
    )


def test_build_training_history_parses_real_lightning_csv(tmp_path):
    import math

    import lightning.pytorch as L
    import torch
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, TensorDataset

    from ptycho_torch.training_history import build_training_history

    class TinyModule(L.LightningModule):
        loss_name = "poisson_train_loss"
        val_loss_name = "poisson_val_loss"

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, _batch_idx):
            inputs, targets = batch
            loss = torch.nn.functional.mse_loss(self.layer(inputs), targets)
            self.log(self.loss_name, loss, on_epoch=True)
            return loss

        def validation_step(self, batch, _batch_idx):
            inputs, targets = batch
            loss = torch.nn.functional.mse_loss(self.layer(inputs), targets)
            self.log(self.val_loss_name, loss, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)),
        batch_size=2,
    )
    model = TinyModule()
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
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    history = build_training_history(
        Path(logger.log_dir),
        csv_logger=logger,
        model=model,
    )

    assert history["schema_version"] == "training_history_v1"
    assert history["train_loss_name"] == model.loss_name
    assert history["val_loss_name"] == model.val_loss_name
    for name in (f"{model.loss_name}_epoch", model.val_loss_name):
        values = history["series"][name]["value"]
        assert values and all(math.isfinite(value) for value in values)


def test_build_training_history_is_absent_without_metrics_csv(tmp_path):
    from ptycho_torch.training_history import build_training_history

    assert build_training_history(tmp_path) is None

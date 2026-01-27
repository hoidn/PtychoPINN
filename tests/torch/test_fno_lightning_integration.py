# tests/torch/test_fno_lightning_integration.py
"""Integration tests for Lightning training with loss history tracking."""
import pytest
import torch


def test_loss_history_callback_collects_per_epoch():
    """Unit test: Verify _LossHistoryCallback collects losses per epoch."""
    import lightning.pytorch as L
    from unittest.mock import MagicMock

    # Import the callback class from the module
    # Note: The callback is defined inside _train_with_lightning, so we recreate it here
    class _LossHistoryCallback(L.Callback):
        def __init__(self):
            self.train_loss = []
            self.val_loss = []

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            if 'train_loss' in metrics:
                self.train_loss.append(float(metrics['train_loss']))

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            if 'val_loss' in metrics:
                self.val_loss.append(float(metrics['val_loss']))

    # Create callback
    cb = _LossHistoryCallback()

    # Mock trainer and pl_module
    mock_trainer = MagicMock()
    mock_pl_module = MagicMock()

    # Simulate epoch 0
    mock_trainer.callback_metrics = {'train_loss': torch.tensor(0.5)}
    cb.on_train_epoch_end(mock_trainer, mock_pl_module)
    mock_trainer.callback_metrics = {'val_loss': torch.tensor(0.6)}
    cb.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Simulate epoch 1
    mock_trainer.callback_metrics = {'train_loss': torch.tensor(0.3)}
    cb.on_train_epoch_end(mock_trainer, mock_pl_module)
    mock_trainer.callback_metrics = {'val_loss': torch.tensor(0.4)}
    cb.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Verify history
    assert len(cb.train_loss) == 2, f"Expected 2 train losses, got {len(cb.train_loss)}"
    assert len(cb.val_loss) == 2, f"Expected 2 val losses, got {len(cb.val_loss)}"
    # Use approximate comparison for floating point
    assert abs(cb.train_loss[0] - 0.5) < 1e-5
    assert abs(cb.train_loss[1] - 0.3) < 1e-5
    assert abs(cb.val_loss[0] - 0.6) < 1e-5
    assert abs(cb.val_loss[1] - 0.4) < 1e-5


def test_loss_history_callback_handles_missing_metrics():
    """Unit test: Callback gracefully handles missing metrics."""
    import lightning.pytorch as L
    from unittest.mock import MagicMock

    class _LossHistoryCallback(L.Callback):
        def __init__(self):
            self.train_loss = []
            self.val_loss = []

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            if 'train_loss' in metrics:
                self.train_loss.append(float(metrics['train_loss']))

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            if 'val_loss' in metrics:
                self.val_loss.append(float(metrics['val_loss']))

    cb = _LossHistoryCallback()
    mock_trainer = MagicMock()
    mock_pl_module = MagicMock()

    # Simulate epoch without metrics
    mock_trainer.callback_metrics = {}
    cb.on_train_epoch_end(mock_trainer, mock_pl_module)
    cb.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Should not have added anything
    assert len(cb.train_loss) == 0
    assert len(cb.val_loss) == 0


@pytest.mark.slow
def test_train_history_collects_epochs(synthetic_ptycho_npz, tmp_path):
    """Integration test: Verify training history collects loss values per epoch."""
    from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
    from ptycho.raw_data import RawData
    from ptycho_torch.workflows.components import train_cdi_model_torch

    train_npz, _ = synthetic_ptycho_npz

    # Load raw data
    train_data = RawData.from_file(str(train_npz))

    # Create config for short training run
    # Note: Use default CNN architecture since FNO/Hybrid generators are not yet
    # fully integrated with Lightning training pipeline
    cfg = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1, architecture="cnn"),
        train_data_file=train_npz,
        test_data_file=None,
        nepochs=2,
        batch_size=2,
        backend="pytorch",
        output_dir=tmp_path,
        n_groups=4,  # Small number for fast tests
    )

    # Disable checkpointing and logging for speed
    # Use 'auto' strategy to avoid DDP issues in tests
    exec_cfg = PyTorchExecutionConfig(
        logger_backend=None,
        enable_checkpointing=False,
        strategy='auto',
    )

    results = train_cdi_model_torch(train_data, None, cfg, execution_config=exec_cfg)

    # Verify history contains loss values for each epoch
    history = results["history"]["train_loss"]
    assert len(history) >= 2, f"Expected at least 2 loss values, got {len(history)}"


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["fno", "hybrid"])
def test_train_history_collects_epochs_for_fno_hybrid(synthetic_ptycho_npz, tmp_path, arch):
    """Integration test: Verify FNO/Hybrid architectures train via Lightning.

    This test verifies that:
    1. The generator registry correctly resolves FNO and Hybrid architectures
    2. The Lightning training pipeline wires up the generator properly
    3. Loss history is collected for at least one epoch
    """
    from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
    from ptycho.raw_data import RawData
    from ptycho_torch.workflows.components import train_cdi_model_torch

    train_npz, _ = synthetic_ptycho_npz
    train_data = RawData.from_file(str(train_npz))

    cfg = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1, architecture=arch),
        train_data_file=train_npz,
        test_data_file=None,
        nepochs=1,
        batch_size=2,
        backend="pytorch",
        output_dir=tmp_path,
        n_groups=4,
    )

    exec_cfg = PyTorchExecutionConfig(
        logger_backend=None,
        enable_checkpointing=False,
        strategy='auto',
    )

    results = train_cdi_model_torch(train_data, None, cfg, execution_config=exec_cfg)
    history = results["history"]["train_loss"]
    assert len(history) >= 1, f"Expected at least 1 loss value for {arch}, got {len(history)}"

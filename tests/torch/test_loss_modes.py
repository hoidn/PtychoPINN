import torch
import pytest
from pathlib import Path

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from ptycho_torch.model import PtychoPINN_Lightning


class _LoggingCaptureModule(PtychoPINN_Lightning):
    """Subclass that captures metric names for assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_names = []

    def log(self, name, *args, **kwargs):
        self.logged_names.append(name)
        return name

    def compute_loss(self, batch):
        loss = super().compute_loss(batch)
        self.log(f"{self.loss_name}_step")
        self.log(f"{self.loss_name}_epoch")
        return loss


def _build_stub_module(
    torch_loss_mode: str,
    *,
    torch_mae_pred_l2_match_target: bool = False,
) -> _LoggingCaptureModule:
    """Create a Lightning module with a deterministic stubbed model."""
    model_cfg = ModelConfig(mode='Unsupervised', loss_function='Poisson')
    data_cfg = DataConfig(N=64, grid_size=(1, 1))
    train_cfg = TrainingConfig(
        epochs=1,
        batch_size=2,
        n_devices=1,
        n_groups=2,
        torch_loss_mode=torch_loss_mode,
        torch_mae_pred_l2_match_target=torch_mae_pred_l2_match_target,
    )
    infer_cfg = InferenceConfig()
    module = _LoggingCaptureModule(model_cfg, data_cfg, train_cfg, infer_cfg)

    def fake_forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids=None):
        pred = x.clone()
        amp = torch.abs(x)
        phase = torch.zeros_like(x)
        return pred, amp, phase

    module.forward = fake_forward.__get__(module, _LoggingCaptureModule)
    return module


def _make_stub_batch(batch_size=2, height=64, width=64):
    tensor_dict = {
        'images': torch.ones((batch_size, 1, height, width), dtype=torch.float32),
        'coords_relative': torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
        'rms_scaling_constant': torch.ones((batch_size, 1, 1, 1), dtype=torch.float32),
        'physics_scaling_constant': torch.ones((batch_size, 1, 1, 1), dtype=torch.float32),
        'experiment_id': torch.zeros(batch_size, dtype=torch.long),
    }
    probe = torch.ones((batch_size, 1, height, width), dtype=torch.complex64)
    scaling = torch.ones(batch_size, dtype=torch.float32)
    return tensor_dict, probe, scaling


@pytest.mark.torch
def test_poisson_loss_mode_logs_poisson_metrics():
    module = _build_stub_module('poisson')
    batch = _make_stub_batch()
    module.compute_loss(batch)

    logged = set(module.logged_names)
    assert 'poisson_train_loss_step' in logged and 'poisson_train_loss_epoch' in logged
    assert 'mae_train_loss_step' not in logged and 'mae_train_loss_epoch' not in logged


@pytest.mark.torch
def test_mae_loss_mode_logs_mae_metrics_only():
    module = _build_stub_module('mae')
    batch = _make_stub_batch()
    module.compute_loss(batch)

    logged = set(module.logged_names)
    assert 'mae_train_loss_step' in logged and 'mae_train_loss_epoch' in logged
    assert 'poisson_train_loss_step' not in logged and 'poisson_train_loss_epoch' not in logged
    assert module.model_config.loss_function == 'MAE'


@pytest.mark.torch
def test_mae_pred_l2_match_target_scales_prediction_per_sample():
    batch = _make_stub_batch()

    module_default = _build_stub_module('mae', torch_mae_pred_l2_match_target=False)
    module_matched = _build_stub_module('mae', torch_mae_pred_l2_match_target=True)

    def fake_forward_scale_by_two(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids=None):
        pred = 2.0 * x
        amp = torch.abs(pred)
        phase = torch.zeros_like(x)
        return pred, amp, phase

    module_default.forward = fake_forward_scale_by_two.__get__(module_default, _LoggingCaptureModule)
    module_matched.forward = fake_forward_scale_by_two.__get__(module_matched, _LoggingCaptureModule)

    loss_default = module_default.compute_loss(batch)
    loss_matched = module_matched.compute_loss(batch)

    assert loss_default > 0
    assert torch.isclose(loss_matched, torch.tensor(0.0), atol=1e-6, rtol=0.0)

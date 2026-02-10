from ptycho_torch.config_params import TrainingConfig


def test_training_config_has_grad_norm_flags():
    cfg = TrainingConfig()
    assert hasattr(cfg, "log_grad_norm")
    assert hasattr(cfg, "grad_norm_log_freq")

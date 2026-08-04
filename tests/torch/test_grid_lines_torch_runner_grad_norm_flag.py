from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, setup_torch_configs


def test_runner_config_supports_grad_norm_logging():
    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="fno",
    )
    assert hasattr(cfg, "log_grad_norm")
    assert hasattr(cfg, "grad_norm_log_freq")


def test_runner_config_keeps_grad_clip_owned_by_training_config():
    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="fno",
        gradient_clip_val=50.0,
    )
    training_config, execution_config = setup_torch_configs(cfg)
    assert training_config.gradient_clip.val == 50.0
    assert training_config.intensity_scale_trainable is False
    assert "gradient_clip_val" not in execution_config.values

from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, setup_torch_configs


def test_runner_config_supports_grad_norm_logging():
    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="hybrid",
    )
    assert hasattr(cfg, "log_grad_norm")
    assert hasattr(cfg, "grad_norm_log_freq")


def test_runner_config_propagates_grad_clip_to_training_config():
    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="hybrid",
        gradient_clip_val=50.0,
    )
    training_config, execution_config = setup_torch_configs(cfg)
    assert training_config.gradient_clip_val == 50.0
    assert execution_config.gradient_clip_val == 50.0

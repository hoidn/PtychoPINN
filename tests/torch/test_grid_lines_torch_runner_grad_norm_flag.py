from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig


def test_runner_config_supports_grad_norm_logging():
    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="hybrid",
    )
    assert hasattr(cfg, "log_grad_norm")
    assert hasattr(cfg, "grad_norm_log_freq")

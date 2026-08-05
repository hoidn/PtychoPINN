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


def test_runner_builds_nested_sampling_optimizer_and_scheduler_configs():
    from ptycho_torch.config_factory import build_training_factory_overrides

    cfg = TorchRunnerConfig(
        train_npz="/tmp/train.npz",
        test_npz="/tmp/test.npz",
        output_dir="/tmp/out",
        architecture="ffno",
        seed=19,
        optimizer="adamw",
        weight_decay=0.012,
        momentum=0.37,
        adam_beta1=0.81,
        adam_beta2=0.93,
        scheduler="ReduceLROnPlateau",
        plateau_factor=0.25,
        plateau_patience=5,
        plateau_min_lr=1e-5,
        plateau_threshold=1e-3,
    )

    training_config, _ = setup_torch_configs(cfg)
    overrides = build_training_factory_overrides(training_config)

    assert overrides["subsample_seed"] == 19
    assert overrides["optimizer"] == "adamw"
    assert overrides["weight_decay"] == 0.012
    assert overrides["momentum"] == 0.37
    assert overrides["adam_beta1"] == 0.81
    assert overrides["adam_beta2"] == 0.93
    assert overrides["scheduler"] == "ReduceLROnPlateau"
    assert overrides["plateau_factor"] == 0.25
    assert overrides["plateau_patience"] == 5
    assert overrides["plateau_min_lr"] == 1e-5
    assert overrides["plateau_threshold"] == 1e-3

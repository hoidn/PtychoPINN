"""Focused contracts for the shared Torch CLI boundary."""

import argparse
import warnings

import pytest


_EXPECTED_RUNTIME_OPTIONS_BY_LANE = {
    "native-training": {
        "--accelerator": {"accelerator"},
        "--device": {"accelerator"},
        "--deterministic": {"deterministic"},
        "--no-deterministic": {"deterministic"},
        "--num-workers": {"num_workers"},
        "--logger": {"logger_backend"},
        "--quiet": {"enable_progress_bar"},
        "--disable_mlflow": {
            "logger_backend",
            "enable_progress_bar",
        },
        "--enable-checkpointing": {"enable_checkpointing"},
        "--disable-checkpointing": {"enable_checkpointing"},
        "--checkpoint-save-top-k": {"checkpoint_save_top_k"},
        "--checkpoint-monitor": {"checkpoint_monitor_metric"},
        "--checkpoint-mode": {"checkpoint_mode"},
        "--early-stop-patience": {"early_stop_patience"},
    },
    "native-inference": {
        "--accelerator": {"accelerator"},
        "--device": {"accelerator"},
        "--num-workers": {"num_workers"},
        "--quiet": {"enable_progress_bar"},
        "--inference-batch-size": {"inference_batch_size"},
    },
    "unified-training": {
        "--torch-accelerator": {"accelerator"},
        "--torch-deterministic": {"deterministic"},
        "--torch-num-workers": {"num_workers"},
        "--torch-logger": {"logger_backend"},
        "--quiet": {"enable_progress_bar"},
        "--torch-enable-checkpointing": {"enable_checkpointing"},
        "--torch-checkpoint-save-top-k": {"checkpoint_save_top_k"},
        "--torch-recon-log-every-n-epochs": {
            "recon_log_every_n_epochs"
        },
        "--torch-recon-log-num-patches": {"recon_log_num_patches"},
        "--torch-recon-log-fixed-indices": {
            "recon_log_fixed_indices"
        },
        "--torch-recon-log-stitch": {"recon_log_stitch"},
        "--torch-recon-log-max-stitch-samples": {
            "recon_log_max_stitch_samples"
        },
    },
    "unified-inference": {
        "--torch-accelerator": {"accelerator"},
        "--torch-num-workers": {"num_workers"},
        "--torch-inference-batch-size": {"inference_batch_size"},
    },
}


@pytest.mark.parametrize(
    "lane",
    sorted(_EXPECTED_RUNTIME_OPTIONS_BY_LANE),
)
def test_runtime_option_registry_is_exact_and_lane_aware(lane):
    from ptycho_torch.cli.shared import canonicalize_execution_options

    expected = _EXPECTED_RUNTIME_OPTIONS_BY_LANE[lane]
    all_options = {
        option
        for lane_options in _EXPECTED_RUNTIME_OPTIONS_BY_LANE.values()
        for option in lane_options
    }
    for option in sorted(all_options):
        fields, sources = canonicalize_execution_options(
            {f"{option}=explicit-value"},
            lane=lane,
        )
        assert fields == expected.get(option, set())
        assert sources == ({option} if option in expected else set())


@pytest.mark.parametrize(
    "option",
    [
        "--learning-rate",
        "--scheduler",
        "--accumulate-grad-batches",
        "--torch-learning-rate",
        "--torch-scheduler",
        "--torch-accumulate-grad-batches",
        "--gradient_clip_val",
        "--optimizer",
        "--plateau_factor",
        "--torch-plateau-factor",
    ],
)
def test_optimizer_options_are_not_runtime_options(option):
    from ptycho_torch.cli.shared import canonicalize_execution_options

    for lane in ("native-training", "unified-training"):
        assert canonicalize_execution_options({option}, lane=lane) == (
            set(),
            set(),
        )


@pytest.mark.parametrize(
    ("lane", "mode", "option", "destination"),
    [
        ("native-training", "training", "--num-workers", "num_workers"),
        ("native-inference", "inference", "--num-workers", "num_workers"),
        (
            "unified-training",
            "training",
            "--torch-num-workers",
            "torch_num_workers",
        ),
        (
            "unified-inference",
            "inference",
            "--torch-num-workers",
            "torch_num_workers",
        ),
    ],
)
def test_runtime_request_distinguishes_omitted_and_explicit_defaults(
    lane,
    mode,
    option,
    destination,
):
    from ptycho_torch.cli.shared import build_execution_request_from_args

    args = argparse.Namespace(**{destination: 0})
    omitted = build_execution_request_from_args(
        args,
        mode=mode,
        explicit_options=(),
        lane=lane,
    )
    explicit = build_execution_request_from_args(
        args,
        mode=mode,
        explicit_options={f"{option}=0"},
        lane=lane,
    )

    assert omitted.values["num_workers"] == 0
    assert explicit.values["num_workers"] == 0
    assert "num_workers" not in omitted.explicit_fields
    assert explicit.explicit_fields == frozenset({"num_workers"})


@pytest.mark.parametrize(
    ("lane", "logger_option", "logger_destination"),
    [
        ("native-training", "--logger", "logger_backend"),
        ("unified-training", "--torch-logger", "torch_logger"),
    ],
)
def test_runtime_request_normalizes_logger_none_and_quiet(
    lane,
    logger_option,
    logger_destination,
):
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(**{logger_destination: "none", "quiet": True}),
        mode="training",
        explicit_options={f"{logger_option}=none", "--quiet"},
        lane=lane,
    )

    assert request.values["logger_backend"] is None
    assert request.values["enable_progress_bar"] is False
    assert request.explicit_fields == frozenset(
        {"logger_backend", "enable_progress_bar"}
    )


def test_unified_inference_does_not_bind_unexposed_quiet_option():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(quiet=True),
        mode="inference",
        explicit_options={"--quiet"},
        lane="unified-inference",
    )

    assert request.values["enable_progress_bar"] is True
    assert "enable_progress_bar" not in request.explicit_fields


def test_runtime_request_freezes_reconstruction_indices():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    indices = [1, 3, 5]
    request = build_execution_request_from_args(
        argparse.Namespace(torch_recon_log_fixed_indices=indices),
        mode="training",
        explicit_options={"--torch-recon-log-fixed-indices=1"},
        lane="unified-training",
    )
    indices.append(7)

    assert request.values["recon_log_fixed_indices"] == (1, 3, 5)
    assert request.as_dict()["recon_log_fixed_indices"] == [1, 3, 5]


def test_runtime_request_defers_warnings_and_capability_observation(
    monkeypatch,
):
    import torch

    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.cli.shared import build_execution_request_from_args

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "request construction must defer effects and capabilities"
        )

    monkeypatch.setattr(warnings, "warn", fail_if_called)
    monkeypatch.setattr(PyTorchExecutionConfig, "__init__", fail_if_called)
    monkeypatch.setattr(torch.cuda, "is_available", fail_if_called)
    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)

    request = build_execution_request_from_args(
        argparse.Namespace(
            accelerator="auto",
            device="cuda",
            deterministic=True,
            num_workers=2,
            disable_mlflow=True,
        ),
        mode="training",
        explicit_options={
            "--device=cuda",
            "--deterministic",
            "--num-workers=2",
            "--disable_mlflow",
        },
        lane="native-training",
    )

    assert request.values["accelerator"] == "gpu"
    assert request.values["logger_backend"] is None
    assert request.values["enable_progress_bar"] is False
    assert [notice.category for notice in request.notices] == [
        DeprecationWarning,
        DeprecationWarning,
        UserWarning,
    ]


def test_canonical_accelerator_wins_deprecated_device_conflict():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(accelerator="cpu", device="cuda"),
        mode="training",
        explicit_options=("--accelerator=cpu", "--device=cuda"),
        lane="native-training",
    )

    assert request.values["accelerator"] == "cpu"
    assert request.explicit_fields == frozenset({"accelerator"})
    assert len(request.notices) == 1
    assert "Ignoring --device value" in request.notices[0].message


def test_optimizer_options_never_enter_runtime_request():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(
            accelerator="cpu",
            learning_rate=0.002,
            scheduler="Adaptive",
            accumulate_grad_batches=3,
        ),
        mode="training",
        explicit_options=(
            "--accelerator=cpu",
            "--learning-rate=0.002",
            "--scheduler=Adaptive",
            "--accumulate-grad-batches=3",
        ),
        lane="native-training",
    )

    assert request.explicit_fields == frozenset({"accelerator"})
    assert set(request.values).isdisjoint(
        {"learning_rate", "scheduler", "accum_steps"}
    )


def test_native_training_patch_contains_only_explicit_optimizer_values():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        learning_rate=0.002,
        scheduler="Adaptive",
        accumulate_grad_batches=3,
    )

    assert build_training_config_patch_from_args(
        args,
        explicit_options=(),
        lane="native-training",
    ) == {}
    assert build_training_config_patch_from_args(
        args,
        explicit_options=(
            "--learning-rate=0.002",
            "--scheduler",
            "Adaptive",
            "--accumulate-grad-batches=3",
        ),
        lane="native-training",
    ) == {
        "learning_rate": 0.002,
        "scheduler": "Adaptive",
        "accum_steps": 3,
    }


def test_unified_training_patch_has_exact_sixteen_canonical_fields():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    values = {
        "gradient_clip_val": 0.5,
        "gradient_clip_algorithm": "norm",
        "optimizer": "adamw",
        "momentum": 0.8,
        "weight_decay": 0.01,
        "adam_beta1": 0.85,
        "adam_beta2": 0.98,
        "scheduler": "WarmupCosine",
        "lr_warmup_epochs": 2,
        "lr_min_ratio": 0.2,
        "plateau_factor": 0.4,
        "plateau_patience": 4,
        "plateau_min_lr": 1e-5,
        "plateau_threshold": 1e-3,
        "torch_learning_rate": 0.002,
        "torch_accumulate_grad_batches": 3,
    }
    options = tuple(
        f"--{name}" for name in values if not name.startswith("torch_")
    ) + (
        "--torch-learning-rate=0.002",
        "--torch-accumulate-grad-batches",
        "3",
    )

    patch = build_training_config_patch_from_args(
        argparse.Namespace(**values),
        explicit_options=options,
        lane="unified-training",
    )

    assert patch == {
        "learning_rate": 0.002,
        "accum_steps": 3,
        "gradient_clip_val": 0.5,
        "gradient_clip_algorithm": "norm",
        "optimizer": "adamw",
        "momentum": 0.8,
        "weight_decay": 0.01,
        "adam_beta1": 0.85,
        "adam_beta2": 0.98,
        "scheduler": "WarmupCosine",
        "lr_warmup_epochs": 2,
        "lr_min_ratio": 0.2,
        "plateau_factor": 0.4,
        "plateau_patience": 4,
        "plateau_min_lr": 1e-5,
        "plateau_threshold": 1e-3,
    }


def test_unified_canonical_optimizer_spelling_wins_torch_alias():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        scheduler="WarmupCosine",
        torch_scheduler="Exponential",
        plateau_factor=0.4,
        torch_plateau_factor=0.25,
    )

    patch = build_training_config_patch_from_args(
        args,
        explicit_options=(
            "--torch-scheduler=Exponential",
            "--scheduler=WarmupCosine",
            "--torch-plateau-factor=0.25",
            "--plateau_factor=0.4",
        ),
        lane="unified-training",
    )

    assert patch == {
        "scheduler": "WarmupCosine",
        "plateau_factor": 0.4,
    }


def test_unified_torch_optimizer_aliases_map_to_canonical_fields():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        torch_scheduler="Exponential",
        torch_plateau_factor=0.25,
    )

    assert build_training_config_patch_from_args(
        args,
        explicit_options=(
            "--torch-scheduler=Exponential",
            "--torch-plateau-factor=0.25",
        ),
        lane="unified-training",
    ) == {
        "scheduler": "Exponential",
        "plateau_factor": 0.25,
    }


def test_unified_omitted_optimizer_defaults_do_not_enter_patch():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        scheduler="Default",
        torch_learning_rate=None,
        torch_accumulate_grad_batches=1,
        torch_plateau_factor=None,
    )

    assert build_training_config_patch_from_args(
        args,
        explicit_options=(),
        lane="unified-training",
    ) == {}


def test_training_patch_rejects_inference_lane():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    with pytest.raises(ValueError, match="training CLI lane"):
        build_training_config_patch_from_args(
            argparse.Namespace(),
            explicit_options=(),
            lane="native-inference",
        )


def test_obsolete_effectful_cli_helpers_are_not_exported():
    import ptycho_torch.cli.shared as shared

    assert not hasattr(shared, "resolve_accelerator")
    assert not hasattr(shared, "build_execution_config_from_args")


def test_validate_paths_creates_output_directory(tmp_path):
    from ptycho_torch.cli.shared import validate_paths

    train_file = tmp_path / "train.npz"
    train_file.touch()
    output_dir = tmp_path / "nested" / "outputs"

    validate_paths(train_file, None, output_dir)

    assert output_dir.is_dir()


@pytest.mark.parametrize(
    ("missing_kind", "match"),
    [
        ("train", "Training data file not found"),
        ("test", "Test data file not found"),
    ],
)
def test_validate_paths_rejects_missing_inputs(
    tmp_path,
    missing_kind,
    match,
):
    from ptycho_torch.cli.shared import validate_paths

    train_file = tmp_path / "train.npz"
    test_file = tmp_path / "test.npz"
    if missing_kind != "train":
        train_file.touch()
    if missing_kind != "test":
        test_file.touch()

    with pytest.raises(FileNotFoundError, match=match):
        validate_paths(train_file, test_file, tmp_path / "outputs")

"""Contracts for the resolved PyTorch runtime carrier."""

from __future__ import annotations

from dataclasses import fields

import pytest


EXPECTED_RUNTIME_FIELDS = (
    "accelerator",
    "devices",
    "strategy",
    "deterministic",
    "precision",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "enable_progress_bar",
    "enable_checkpointing",
    "checkpoint_save_top_k",
    "checkpoint_monitor_metric",
    "checkpoint_mode",
    "early_stop_patience",
    "logger_backend",
    "recon_log_every_n_epochs",
    "recon_log_num_patches",
    "recon_log_fixed_indices",
    "recon_log_stitch",
    "recon_log_max_stitch_samples",
    "inference_batch_size",
    "middle_trim",
    "pad_eval",
)


def test_execution_config_is_exact_resolved_runtime_carrier() -> None:
    from ptycho.config.config import PyTorchExecutionConfig

    config = PyTorchExecutionConfig()

    assert tuple(item.name for item in fields(PyTorchExecutionConfig)) == (
        EXPECTED_RUNTIME_FIELDS
    )
    assert config.accelerator == "cpu"
    assert config.devices == 1
    assert not hasattr(config, "_explicit_structural_aliases")


def test_execution_config_construction_never_observes_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from ptycho.config.config import PyTorchExecutionConfig

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("resolved carrier construction observed hardware")

    monkeypatch.setattr(torch.cuda, "is_available", fail_if_called)
    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)

    config = PyTorchExecutionConfig(accelerator="cpu", devices=1)

    assert config.accelerator == "cpu"
    assert config.devices == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"accelerator": "auto"}, "accelerator"),
        ({"accelerator": "tpu"}, "Torch-XLA"),
        ({"devices": "auto"}, "devices"),
        ({"devices": 0}, "devices"),
        (
            {"num_workers": 0, "persistent_workers": True},
            "persistent_workers",
        ),
        ({"logger_backend": "none"}, "logger_backend"),
    ],
)
def test_execution_config_rejects_unresolved_or_invalid_runtime_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from ptycho.config.config import PyTorchExecutionConfig

    with pytest.raises(ValueError, match=message):
        PyTorchExecutionConfig(**kwargs)


@pytest.mark.parametrize(
    "logger_backend",
    ["csv", "tensorboard", "mlflow", None],
)
def test_execution_config_accepts_canonical_logger_backends(
    logger_backend: str | None,
) -> None:
    from ptycho.config.config import PyTorchExecutionConfig

    config = PyTorchExecutionConfig(logger_backend=logger_backend)

    assert config.logger_backend == logger_backend


def test_execution_config_preserves_positional_runtime_prefix() -> None:
    from ptycho.config.config import PyTorchExecutionConfig

    config = PyTorchExecutionConfig("cuda", 2, "ddp", False)

    assert (
        config.accelerator,
        config.devices,
        config.strategy,
        config.deterministic,
    ) == ("cuda", 2, "ddp", False)

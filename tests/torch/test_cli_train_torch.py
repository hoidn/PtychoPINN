"""Native Torch CLI delegation to the public dataset training function."""

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def minimal_train_args(tmp_path):
    train_file = tmp_path / "train.npz"
    train_file.touch()
    return [
        "--train_data_file", str(train_file),
        "--output_dir", str(tmp_path / "outputs"),
        "--n_images", "64",
        "--max_epochs", "2",
    ]


def _run(monkeypatch, argv, tmp_path):
    from ptycho_torch.execution_request import ExecutionRequest
    import ptycho_torch.train as training

    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with patch.object(
        training,
        "train",
        return_value=tmp_path / "outputs" / "wts.h5.zip",
    ) as public_train:
        training.cli_main()
    public_train.assert_called_once()
    assert isinstance(public_train.call_args.kwargs["execution_config"], ExecutionRequest)
    return public_train.call_args


def test_native_cli_delegates_once_with_ci_default(
    minimal_train_args, monkeypatch, tmp_path
):
    call = _run(monkeypatch, minimal_train_args, tmp_path)

    dataset, output_dir, settings = call.args
    assert Path(dataset).name == "train.npz"
    assert Path(output_dir).name == "outputs"
    assert settings["training_groups"] == 64
    assert settings["epochs"] == 2
    assert "nphotons" not in settings
    assert "neighbor_count" not in settings
    assert call.kwargs["profile"] == "ci"


def test_native_cli_forwards_authored_measurement_and_neighbor_values(
    minimal_train_args, monkeypatch, tmp_path
):
    call = _run(
        monkeypatch,
        [*minimal_train_args, "--nphotons", "2500000", "--neighbor-count", "7"],
        tmp_path,
    )

    settings = call.args[2]
    assert settings["nphotons"] == pytest.approx(2_500_000)
    assert settings["neighbor_count"] == 7


def test_native_cli_complete_legacy_pair_selects_custom_profile(
    minimal_train_args, monkeypatch, tmp_path
):
    call = _run(
        monkeypatch,
        [
            *minimal_train_args,
            "--scale-contract-version", "legacy_v1",
            "--measurement-domain", "normalized_amplitude",
        ],
        tmp_path,
    )

    assert call.kwargs["profile"] is None


def test_native_cli_preserves_resolver_error_message(
    minimal_train_args, monkeypatch, capsys
):
    import ptycho_torch.train as training

    argv = [*minimal_train_args, "--scale-contract-version", "legacy_v1"]
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with patch.object(
        training,
        "train",
        side_effect=ValueError(
            "scale_contract_version and measurement_domain must be supplied together"
        ),
    ), pytest.raises(SystemExit) as error:
        training.cli_main()

    assert error.value.code != 0
    assert "scale_contract_version and measurement_domain" in capsys.readouterr().out


def test_native_cli_preserves_execution_and_training_suppliedness(
    minimal_train_args, monkeypatch, tmp_path
):
    call = _run(
        monkeypatch,
        [
            *minimal_train_args,
            "--accelerator=cpu",
            "--no-deterministic",
            "--num-workers=2",
            "--learning-rate", "0.002",
            "--scheduler=ReduceLROnPlateau",
            "--accumulate-grad-batches", "3",
            "--logger=none",
            "--quiet",
            "--disable-checkpointing",
            "--checkpoint-save-top-k=0",
            "--checkpoint-monitor", "train_loss",
            "--checkpoint-mode=max",
            "--early-stop-patience", "9",
        ],
        tmp_path,
    )

    request = call.kwargs["execution_config"]
    assert request.values["accelerator"] == "cpu"
    assert request.values["deterministic"] is False
    assert request.values["num_workers"] == 2
    assert request.values["logger_backend"] is None
    assert request.values["enable_progress_bar"] is False
    assert request.values["enable_checkpointing"] is False
    assert request.values["checkpoint_save_top_k"] == 0
    assert request.values["checkpoint_monitor_metric"] == "train_loss"
    assert request.values["checkpoint_mode"] == "max"
    assert request.values["early_stop_patience"] == 9
    assert {name: call.args[2][name] for name in (
        "learning_rate", "scheduler", "accum_steps"
    )} == {
        "learning_rate": 0.002,
        "scheduler": "ReduceLROnPlateau",
        "accum_steps": 3,
    }


def test_native_training_import_is_torch_and_legacy_state_free():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, ptycho_torch.train; "
            "assert 'ptycho.params' not in sys.modules; "
            "assert 'ptycho.config.legacy_state' not in sys.modules; "
            "assert 'tensorflow' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

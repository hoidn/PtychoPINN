"""CLI-parse and import-safety tests for scripts/studies/tf_reference_cnn_runner.py.

CPU-only: does not train (no TF/CDI run is exercised). Mirrors the CLI-parse
test style used for scripts/studies/varpro_probe_ablation_runner.py in
tests/torch/test_varpro_probe_ablation_runner.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "studies"))
import tf_reference_cnn_runner as runner  # noqa: E402


def _base_argv(**overrides):
    argv = {
        "--train_data_file": "dummy_train.npz",
        "--test_data_file": "dummy_test.npz",
        "--output_dir": "dummy_out",
    }
    flat = []
    for flag, value in argv.items():
        flat += [flag, value]
    for flag, value in overrides.items():
        flat += [flag, str(value)]
    return flat


def test_module_imports_without_executing_training():
    # Importing the module (done at collection time, above) must not invoke
    # argparse or training -- guarded by `if __name__ == "__main__": main()`.
    assert hasattr(runner, "main")
    assert hasattr(runner, "parse_args")


def test_parse_args_accepts_documented_flags(monkeypatch):
    argv = ["prog"] + _base_argv(
        **{
            "--N": 128,
            "--gridsize": 1,
            "--nepochs": 25,
            "--batch_size": 8,
            "--n_groups": 512,
            "--nphotons": 1768920.0,
            "--intensity_scale_trainable": 1,
        }
    )
    monkeypatch.setattr(sys, "argv", argv)
    cli = runner.parse_args()

    assert cli.train_data_file == "dummy_train.npz"
    assert cli.test_data_file == "dummy_test.npz"
    assert cli.N == 128
    assert cli.gridsize == 1
    assert cli.nepochs == 25
    assert cli.batch_size == 8
    assert cli.n_groups == 512
    assert cli.nphotons == 1768920.0
    assert cli.intensity_scale_trainable == 1
    assert cli.output_dir == "dummy_out"


def test_parse_args_defaults_match_e4_recipe(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"] + _base_argv())
    cli = runner.parse_args()

    assert cli.N == 128
    assert cli.gridsize == 1
    assert cli.nepochs == 25
    assert cli.batch_size == 8
    assert cli.n_groups == 512
    assert cli.nphotons == 1768920.0
    assert cli.intensity_scale_trainable == 1


@pytest.mark.parametrize("bad_value", ["2", "-1", "true"])
def test_parse_args_rejects_invalid_intensity_scale_trainable(monkeypatch, bad_value):
    argv = ["prog"] + _base_argv(**{"--intensity_scale_trainable": bad_value})
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_parse_args_requires_train_data_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--test_data_file", "dummy_test.npz", "--output_dir", "dummy_out"],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_save_trained_model_records_artifact_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr("ptycho.model_manager.save", lambda out_prefix: None)
    expected_artifact = runner.params.get("h5_path") + ".zip"

    model_saved, model_artifact, model_save_error = runner.save_trained_model(tmp_path)

    assert model_saved is True
    assert model_artifact == expected_artifact
    assert model_save_error is None


def test_save_trained_model_reports_failure_without_raising(monkeypatch, tmp_path):
    def _boom(out_prefix):
        raise RuntimeError("disk full")

    monkeypatch.setattr("ptycho.model_manager.save", _boom)

    model_saved, model_artifact, model_save_error = runner.save_trained_model(tmp_path)

    assert model_saved is False
    assert model_artifact is None
    assert model_save_error == "disk full"

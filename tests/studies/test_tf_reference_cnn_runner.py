"""CLI-parse and import-safety tests for scripts/studies/tf_reference_cnn_runner.py.

CPU-only: does not train (no TF/CDI run is exercised). Mirrors the CLI-parse
test style used for scripts/studies/varpro_probe_ablation_runner.py in
tests/torch/test_varpro_probe_ablation_runner.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "studies"))
import tf_reference_cnn_runner as runner  # noqa: E402
from ptycho import params
from ptycho.config import resolve_training_config


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
            "--training_groups": 512,
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
    assert cli.training_groups == 512
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
    assert cli.training_groups == 512
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
    expected_artifact = "wts.h5.zip"

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


def test_tensorflow_reference_leaf_projects_owner_and_restores_ambient_state(
    monkeypatch, tmp_path
):
    config = resolve_training_config(
        None,
        {
            "N": 128,
            "gridsize": 1,
            "batch_size": 8,
            "training_groups": 16,
            "nphotons": 1234.0,
            "intensity_scale_trainable": False,
            "train_data_file": "train.npz",
            "test_data_file": "test.npz",
        },
    )
    results = {"reconstructed_obj": object(), "test_container": object()}
    observed = {}

    def fake_run(train_data, test_data, owner, *, do_stitching):
        observed.update(
            N=params.cfg["N"],
            gridsize=params.cfg["gridsize"],
            batch_size=params.cfg["batch_size"],
            nphotons=params.cfg["nphotons"],
            trainable=params.cfg["intensity_scale.trainable"],
        )
        params.cfg["intensity_scale"] = 4.0
        return "amp", "phase", results

    def fake_save(output_dir):
        assert params.cfg["intensity_scale"] == 4.0
        return True, "wts.h5.zip", None

    monkeypatch.setattr(runner, "run_cdi_example_with_backend", fake_run)
    monkeypatch.setattr(runner, "save_trained_model", fake_save)

    before = {
        "sentinel": object(),
        "N": 999,
        "batch_size": 256,
        "intensity_scale.trainable": "poison",
    }
    previous = dict(params.cfg)
    previous_sealed = params._sealed
    try:
        params.cfg.clear()
        params.cfg.update(before)
        params.seal()

        amp, phase, actual_results, diagnostics = (
            runner.run_tensorflow_reference_leaf(
                object(),
                object(),
                config,
                tmp_path,
            )
        )

        assert (amp, phase, actual_results) == ("amp", "phase", results)
        assert diagnostics == {
            "log_scale_init": np.log(4.0),
            "log_scale_final": None,
            "model_saved": True,
            "model_artifact": "wts.h5.zip",
            "model_save_error": None,
        }
        assert observed == {
            "N": 128,
            "gridsize": 1,
            "batch_size": 8,
            "nphotons": 1234.0,
            "trainable": False,
        }
        assert params.cfg == before
        assert params._sealed is True
    finally:
        params.cfg.clear()
        params.cfg.update(previous)
        params.seal() if previous_sealed else params.unseal()


def test_tensorflow_reference_leaf_restores_ambient_state_on_failure(
    monkeypatch, tmp_path
):
    config = resolve_training_config(
        None,
        {
            "N": 128,
            "gridsize": 1,
            "batch_size": 8,
            "training_groups": 16,
            "train_data_file": "train.npz",
            "test_data_file": "test.npz",
        },
    )
    monkeypatch.setattr(
        runner,
        "run_cdi_example_with_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("training failure")
        ),
    )

    before = {"sentinel": "failure", "N": 999}
    previous = dict(params.cfg)
    previous_sealed = params._sealed
    try:
        params.cfg.clear()
        params.cfg.update(before)
        params.seal()

        with pytest.raises(RuntimeError, match="training failure"):
            runner.run_tensorflow_reference_leaf(
                object(),
                object(),
                config,
                tmp_path,
            )

        assert params.cfg == before
        assert params._sealed is True
    finally:
        params.cfg.clear()
        params.cfg.update(previous)
        params.seal() if previous_sealed else params.unseal()

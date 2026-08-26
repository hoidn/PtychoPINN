"""Unified CLI direct Torch delegation and bounded TensorFlow behavior."""

import argparse
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _config(
    tmp_path,
    *,
    backend="pytorch",
    nphotons=1e9,
    neighbor_count=4,
    subsample_seed=None,
):
    from ptycho.config import ModelConfig, TrainingConfig

    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    train.touch()
    test.touch()
    return TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        train_data_file=train,
        test_data_file=test,
        backend=backend,
        nepochs=1,
        training_groups=3,
        train_raw_selection=None,
        nphotons=nphotons,
        neighbor_count=neighbor_count,
        subsample_seed=subsample_seed,
        output_dir=tmp_path / "out",
    )


def _args(**values):
    defaults = {"config": None, "do_stitching": False}
    defaults.update(values)
    return argparse.Namespace(**defaults)


def _patch_resolution(monkeypatch, training_script, config, *, raw_yaml=None):
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(training_script, "parse_arguments", lambda argv=None: _args())
    monkeypatch.setattr(
        "ptycho.workflows.config_cli.setup_configuration", lambda *_args: config
    )
    monkeypatch.setattr(
        "ptycho.workflows.config_cli.load_yaml_config",
        lambda _path: {} if raw_yaml is None else raw_yaml,
    )
    monkeypatch.setattr(training_script, "_metadata_photon_count", lambda _path: None)
    monkeypatch.setattr("ptycho.config.validate_training_config_structure", lambda _c: None)
    monkeypatch.setattr("ptycho.config.validate_runnable_training_config", lambda _c: None)


def test_unified_torch_delegates_once_with_ci_default(monkeypatch, tmp_path):
    from ptycho_torch.execution_request import ExecutionRequest
    from scripts.training import train as training_script

    config = _config(tmp_path)
    _patch_resolution(monkeypatch, training_script, config)
    public_train = MagicMock(return_value=config.output_dir / "wts.h5.zip")
    monkeypatch.setattr("ptycho_torch.train.train", public_train)

    result = training_script.main(["--backend", "pytorch"])

    assert result == config.output_dir / "wts.h5.zip"
    public_train.assert_called_once()
    dataset, output_dir, settings = public_train.call_args.args
    assert dataset == config.train_data_file
    assert output_dir == config.output_dir
    assert "nphotons" not in settings
    assert "neighbor_count" not in settings
    assert public_train.call_args.kwargs["profile"] == "ci"
    assert isinstance(
        public_train.call_args.kwargs["execution_config"], ExecutionRequest
    )


@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_unified_torch_forwards_only_authored_nphotons_and_neighbors(
    source, monkeypatch, tmp_path
):
    from scripts.training import train as training_script

    config = _config(tmp_path, nphotons=2.5e6, neighbor_count=7)
    args = _args()
    raw_yaml = None
    argv = ["--backend", "pytorch"]
    if source == "cli":
        args.nphotons = 2.5e6
        args.neighbor_count = 7
        argv += ["--nphotons", "2500000", "--neighbor_count", "7"]
    else:
        args.config = "run.yaml"
        raw_yaml = {"nphotons": 2.5e6, "neighbor_count": 7}
        argv += ["--config", "run.yaml"]
    _patch_resolution(monkeypatch, training_script, config, raw_yaml=raw_yaml)
    monkeypatch.setattr(training_script, "parse_arguments", lambda _argv=None: args)
    public_train = MagicMock(return_value=config.output_dir / "wts.h5.zip")
    monkeypatch.setattr("ptycho_torch.train.train", public_train)

    training_script.main(argv)

    settings = public_train.call_args.args[2]
    assert settings["nphotons"] == pytest.approx(2.5e6)
    assert settings["neighbor_count"] == 7


def test_unified_complete_legacy_pair_selects_custom_profile(monkeypatch, tmp_path):
    from scripts.training import train as training_script

    config = _config(tmp_path)
    _patch_resolution(monkeypatch, training_script, config)
    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda _argv=None: _args(
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        ),
    )
    public_train = MagicMock(return_value=config.output_dir / "wts.h5.zip")
    monkeypatch.setattr("ptycho_torch.train.train", public_train)

    training_script.main(["--backend", "pytorch"])

    assert public_train.call_args.kwargs["profile"] is None


def test_unified_yaml_complete_legacy_pair_reaches_public_torch_door(
    monkeypatch, tmp_path
):
    from scripts.training import train as training_script

    config = _config(tmp_path)
    raw_yaml = {
        "backend": "pytorch",
        "train_data_file": str(config.train_data_file),
        "test_data_file": str(config.test_data_file),
        "output_dir": str(config.output_dir),
        "nepochs": 1,
        "training_groups": 3,
        "scale_contract_version": "legacy_v1",
        "measurement_domain": "normalized_amplitude",
    }
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(training_script, "_metadata_photon_count", lambda _path: None)
    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda _argv=None: _args(config="run.yaml"),
    )
    monkeypatch.setattr(
        "ptycho.workflows.config_cli.load_yaml_config",
        lambda _path: raw_yaml,
    )
    public_train = MagicMock(return_value=config.output_dir / "wts.h5.zip")
    monkeypatch.setattr("ptycho_torch.train.train", public_train)

    training_script.main(["--config", "run.yaml"])

    assert public_train.call_args.kwargs["profile"] is None
    settings = public_train.call_args.args[2]
    assert settings["scale_contract_version"] == "legacy_v1"
    assert settings["measurement_domain"] == "normalized_amplitude"


def test_unified_yaml_partial_legacy_pair_reaches_public_torch_validation(
    monkeypatch, tmp_path
):
    from scripts.training import train as training_script

    config = _config(tmp_path)
    raw_yaml = {
        "backend": "pytorch",
        "train_data_file": str(config.train_data_file),
        "test_data_file": str(config.test_data_file),
        "output_dir": str(config.output_dir),
        "nepochs": 1,
        "training_groups": 3,
        "scale_contract_version": "legacy_v1",
    }
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(training_script, "_metadata_photon_count", lambda _path: None)
    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda _argv=None: _args(config="run.yaml"),
    )
    monkeypatch.setattr(
        "ptycho.workflows.config_cli.load_yaml_config",
        lambda _path: raw_yaml,
    )
    public_train = MagicMock(
        side_effect=ValueError("legacy_v1 requires normalized_amplitude")
    )
    monkeypatch.setattr("ptycho_torch.train.train", public_train)

    with pytest.raises(ValueError, match="legacy_v1 requires normalized_amplitude"):
        training_script.main(["--config", "run.yaml"])

    public_train.assert_called_once()


def test_tensorflow_yaml_rejects_torch_only_measurement_fields(monkeypatch, tmp_path):
    from ptycho.workflows.config_cli import setup_configuration

    raw_yaml = {
        "backend": "tensorflow",
        "train_data_file": str(tmp_path / "train.npz"),
        "scale_contract_version": "legacy_v1",
        "measurement_domain": "normalized_amplitude",
    }
    monkeypatch.setattr(
        "ptycho.workflows.config_cli.load_yaml_config",
        lambda _path: raw_yaml,
    )

    with pytest.raises(ValueError, match="unknown root fields"):
        setup_configuration(_args(config="run.yaml"), "run.yaml")


def test_unified_resolver_error_is_not_rewritten(monkeypatch, tmp_path):
    from scripts.training import train as training_script

    config = _config(tmp_path)
    _patch_resolution(monkeypatch, training_script, config)
    monkeypatch.setattr(
        "ptycho_torch.train.train",
        MagicMock(side_effect=ValueError("neighbor_count must be at least 3")),
    )

    with pytest.raises(ValueError, match="neighbor_count must be at least 3"):
        training_script.main(["--backend", "pytorch"])


def test_unified_tensorflow_preserves_direct_legacy_workflow(monkeypatch, tmp_path):
    from ptycho import params
    from scripts.training import train as training_script

    config = _config(tmp_path, backend="tensorflow", subsample_seed=41)
    _patch_resolution(monkeypatch, training_script, config)
    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda _argv=None: _args(do_stitching=True),
    )
    events = []
    original = dict(params.cfg)
    cfg_object = params.cfg
    params.cfg["review_ambient_marker"] = "outside"
    params.cfg["N"] = 128
    ambient = dict(params.cfg)

    class Raw:
        probeGuess = np.ones((64, 64), dtype=np.complex64)

        def __init__(self, split):
            self.split = split

        def generate_grouped_data(self, **kwargs):
            events.append(
                (
                    f"group:{self.split}",
                    kwargs["nsamples"],
                    kwargs["seed"],
                    params.cfg["N"],
                )
            )
            return {"nn_indices": np.zeros((kwargs["nsamples"], 1), dtype=np.int32)}

    def load_data(path, **kwargs):
        split = "train" if Path(path) == Path(config.train_data_file) else "validation"
        events.append(
            (f"load:{split}", kwargs.get("n_subsample"), params.cfg["N"])
        )
        return Raw(split)

    monkeypatch.setattr("ptycho.workflows.config_cli.load_data", load_data)
    monkeypatch.setattr("ptycho.loader.load", lambda callback, *_args, **_kwargs: callback())
    backend_results = {"trained": True}
    monkeypatch.setattr(
        "ptycho.workflows.workflow_orchestration.run_cdi_example",
        lambda *_args, **kwargs: (
            events.append(("dispatch", kwargs["do_stitching"])) or None,
            None,
            backend_results,
        ),
    )
    saved = MagicMock()
    outputs = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append(
            ("outputs", params.cfg["N"])
        )
    )
    monkeypatch.setattr("ptycho.model_manager.save", saved)
    monkeypatch.setattr("ptycho.workflows.workflow_orchestration.save_outputs", outputs)

    try:
        result = training_script.main(
            ["--backend", "tensorflow", "--do_stitching"]
        )

        assert result == {"trained": True, "backend": "tensorflow"}
        assert events[:4] == [
            ("load:train", 3, 64),
            ("load:validation", None, 64),
            ("group:train", 3, 41, 64),
            ("group:validation", 3, 41, 64),
        ]
        assert events[4] == ("dispatch", True)
        assert events[5] == ("outputs", 128)
        saved.assert_called_once_with(str(config.output_dir))
        outputs.assert_called_once()
        assert outputs.call_args.args[2]["backend"] == "tensorflow"
        assert params.cfg is cfg_object
        assert dict(params.cfg) == ambient
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_metadata_photons_resolve_before_validation(monkeypatch, tmp_path):
    from scripts.training import train as training_script

    config = _config(tmp_path, nphotons=1.0)
    _patch_resolution(monkeypatch, training_script, config)
    monkeypatch.setattr(training_script, "_metadata_photon_count", lambda _path: 9.0)
    seen = []
    monkeypatch.setattr(
        "ptycho.config.validate_training_config_structure",
        lambda candidate: seen.append(candidate.nphotons),
    )
    monkeypatch.setattr(
        "ptycho.config.validate_runnable_training_config",
        lambda candidate: seen.append(candidate.nphotons),
    )
    monkeypatch.setattr(
        "ptycho_torch.train.train", MagicMock(return_value=config.output_dir / "wts.h5.zip")
    )

    training_script.main(["--backend", "pytorch"])

    assert seen == [9.0, 9.0]


def test_unified_torch_parse_and_main_are_legacy_and_tensorflow_free():
    code = r'''
import sys
from pathlib import Path

import scripts.training.train as training_script
from ptycho.config import ModelConfig, TrainingConfig
import ptycho.workflows.config_cli as config_cli
import ptycho.config as public_config
import ptycho_torch.train as torch_train

config = TrainingConfig(
    model=ModelConfig(N=64, gridsize=1),
    train_data_file=Path("train.npz"),
    test_data_file=Path("test.npz"),
    backend="pytorch",
    nepochs=1,
    training_groups=3,
    output_dir=Path("out"),
)
training_script._configure_logging = lambda: None
training_script._metadata_photon_count = lambda _path: None
config_cli.setup_configuration = lambda *_args: config
config_cli.load_yaml_config = lambda _path: {}
public_config.validate_training_config_structure = lambda _config: None
public_config.validate_runnable_training_config = lambda _config: None
torch_train.train = lambda *_args, **_kwargs: Path("out/wts.h5.zip")

assert training_script.main(["--backend", "pytorch"]) == Path("out/wts.h5.zip")
for forbidden in (
    "tensorflow",
    "ptycho.params",
    "ptycho.loader",
    "ptycho.config.legacy_state",
):
    assert forbidden not in sys.modules, forbidden
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

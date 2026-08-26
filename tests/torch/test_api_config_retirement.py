"""Focused contracts for retiring loose legacy API configuration loading."""

from copy import deepcopy
from collections import UserDict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ptycho.params as params
from ptycho.config.config import update_legacy_dict
from ptycho.config.legacy_state import legacy_params_scope
from ptycho_torch.api import api_helper
from ptycho_torch.api import base_api
from ptycho_torch.api.base_api import ConfigManager, PtychoModel
from ptycho_torch.config_bridge import to_model_config, to_training_config
from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.mark.parametrize(
    "config_class",
    [DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig],
)
def test_parse_config_copies_exact_resolved_owner(config_class):
    supplied = config_class()

    parsed = ConfigManager._parse_config(supplied, config_class)

    assert type(parsed) is config_class
    assert parsed == supplied
    assert parsed is not supplied


def test_parse_config_constructs_default_for_none():
    parsed = ConfigManager._parse_config(None, DataConfig)

    assert type(parsed) is DataConfig
    assert parsed == DataConfig()


@pytest.mark.parametrize("mapping", [{}, UserDict()])
def test_parse_config_rejects_ownerless_mapping_with_migration_guidance(mapping):
    with pytest.raises(TypeError, match=r"resolved DataConfig.*config_factory"):
        ConfigManager._parse_config(mapping, DataConfig)


def test_parse_config_rejects_nonexact_record_types():
    class DataConfigSubclass(DataConfig):
        pass

    with pytest.raises(TypeError, match=r"exact DataConfig"):
        ConfigManager._parse_config(DataConfigSubclass(), DataConfig)


def test_from_lightning_json_delegates_to_versioned_loader(monkeypatch):
    configs = (
        DataConfig(),
        ModelConfig(),
        TrainingConfig(),
        InferenceConfig(),
        DatagenConfig(),
    )
    observed = []

    def fake_loader(path):
        observed.append(path)
        return configs

    monkeypatch.setattr(
        "ptycho_torch.lightning_utils.load_configs_from_checkpoint",
        fake_loader,
    )

    manager = ConfigManager._from_lightning_json("/tmp/versioned-run")

    assert observed == ["/tmp/versioned-run"]
    assert all(
        actual is expected
        for actual, expected in zip(manager.to_tuple(), configs, strict=True)
    )


def test_from_lightning_json_fails_closed(monkeypatch):
    def reject_invalid_artifact(path):
        raise ValueError(f"invalid versioned artifact: {path}")

    monkeypatch.setattr(
        "ptycho_torch.lightning_utils.load_configs_from_checkpoint",
        reject_invalid_artifact,
    )

    with pytest.raises(ValueError, match="invalid versioned artifact"):
        ConfigManager._from_lightning_json("/tmp/invalid-run")


def test_loose_configuration_entry_points_are_removed():
    for name in ("_from_json", "_flexible_load", "_from_mlflow", "update"):
        assert not hasattr(ConfigManager, name)

    for name in (
        "config_from_json",
        "load_configs_from_local_dir",
        "update_manager_with_json",
    ):
        assert not hasattr(api_helper, name)


def _model_unpickled_with_retired_rect_s1s2_mode():
    model_config = ModelConfig()
    model_config.rect_s1s2_init = "data"
    return SimpleNamespace(model_config=model_config)


def test_mlflow_utils_public_loaders_are_importable():
    from ptycho_torch.api import mlflow_utils

    assert callable(mlflow_utils.load_model_from_mlflow)
    assert callable(mlflow_utils.load_model_and_configs)


@pytest.mark.parametrize(
    "loader_name",
    [
        "api_helper.load_with_mlflow",
        "PtychoModel.load_from_mlflow",
        "mlflow_utils.load_model_from_mlflow",
        "mlflow_utils.load_model_and_configs",
    ],
)
def test_mlflow_whole_model_loaders_revalidate_unpickled_model_config(
    loader_name,
    monkeypatch,
):
    loaded_model = _model_unpickled_with_retired_rect_s1s2_mode()
    if loader_name.startswith("mlflow_utils"):
        from ptycho_torch.api import mlflow_utils

        mlflow_module = mlflow_utils
    elif loader_name.startswith("PtychoModel"):
        mlflow_module = base_api
    else:
        mlflow_module = api_helper
    monkeypatch.setattr(
        mlflow_module.mlflow.pytorch,
        "load_model",
        lambda _model_uri: loaded_model,
    )
    monkeypatch.setattr(
        mlflow_module.mlflow,
        "set_tracking_uri",
        lambda _tracking_uri: None,
    )

    if loader_name == "mlflow_utils.load_model_and_configs":
        monkeypatch.setattr(
            mlflow_utils,
            "get_run_id_from_model_version",
            lambda **_kwargs: "run-123",
        )

        def unexpected_config_lookup(*_args, **_kwargs):
            raise AssertionError("validation must precede MLflow config lookup")

        monkeypatch.setattr(
            mlflow_utils,
            "load_configs_from_run",
            unexpected_config_lookup,
        )

    with pytest.raises(
        ValueError,
        match=r"data.*unsupported.*ones.*dose_closure.*historical code or retraining",
    ):
        if loader_name == "api_helper.load_with_mlflow":
            api_helper.load_with_mlflow(
                run_id="run-123",
                mlflow_tracking_uri="/tmp/mlruns",
            )
        elif loader_name == "PtychoModel.load_from_mlflow":
            PtychoModel.load_from_mlflow(
                run_id="run-123",
                mlflow_tracking_uri="/tmp/mlruns",
            )
        elif loader_name == "mlflow_utils.load_model_from_mlflow":
            mlflow_utils.load_model_from_mlflow(
                run_id="run-123",
                model_class=object,
            )
        else:
            mlflow_utils.load_model_and_configs(
                model_name="ptychopinn",
                version=1,
            )


def test_save_pytorch_projects_resolved_owners_without_reading_legacy_global(
    tmp_path,
):
    data_config = DataConfig(
        N=128,
        neighbor_count=7,
        gridsize=2,
        nphotons=2.5e6,
    )
    model_config = ModelConfig(
        architecture="fno",
        fno_modes=9,
        fno_width=24,
    )
    training_config = TrainingConfig(
        train_data_file=str(tmp_path / "train.npz"),
        test_data_file=str(tmp_path / "test.npz"),
        output_dir=str(tmp_path / "training"),
        training_groups=23,
        epochs=7,
        batch_size=3,
    )
    records_before = (
        deepcopy(data_config),
        deepcopy(model_config),
        deepcopy(training_config),
    )
    model = PtychoModel(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
    )
    model_records_before = (
        deepcopy(model.data_config),
        deepcopy(model.model_config),
        deepcopy(model.training_config),
    )

    tf_model_config = to_model_config(data_config, model_config)
    tf_training_config = to_training_config(
        tf_model_config,
        data_config,
        model_config,
        training_config,
        overrides={
            "train_data_file": training_config.train_data_file,
            "test_data_file": training_config.test_data_file,
            "output_dir": training_config.output_dir,
            "training_groups": training_config.training_groups,
            "nphotons": data_config.nphotons,
        },
    )
    expected_projection = {}
    update_legacy_dict(expected_projection, tf_training_config)
    expected_snapshot = json.loads(
        json.dumps(expected_projection, default=str)
    )

    checkpoint = tmp_path / "source" / "best-checkpoint.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint-bytes")
    bundle = tmp_path / "bundle"
    poisoned_global = {"N": -999, "ambient_poison": object()}

    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update(poisoned_global)
        manifest_path = model.save_pytorch(
            str(bundle),
            checkpoint_path=str(checkpoint),
        )
        assert params.cfg == poisoned_global

    manifest = json.loads(manifest_path.read_text())

    assert list(manifest) == [
        "backend",
        "checkpoint",
        "params_cfg_snapshot",
        "version",
        "notes",
    ]
    assert manifest == {
        "backend": "pytorch",
        "checkpoint": checkpoint.name,
        "params_cfg_snapshot": expected_snapshot,
        "version": "1.0",
        "notes": "Minimal PyTorch persistence shim (Phase R reactivation)",
    }
    assert checkpoint.read_bytes() == b"checkpoint-bytes"
    assert not (bundle / checkpoint.name).exists()
    assert data_config == records_before[0]
    assert model_config == records_before[1]
    assert training_config == records_before[2]
    assert model.data_config == model_records_before[0]
    assert model.model_config == model_records_before[1]
    assert model.training_config == model_records_before[2]


def test_save_pytorch_preserves_missing_checkpoint_error(tmp_path):
    model = PtychoModel()
    missing_checkpoint = tmp_path / "missing.ckpt"

    with legacy_params_scope():
        params.cfg.clear()
        params.cfg["ambient_poison"] = True
        with pytest.raises(
            FileNotFoundError,
            match=f"Checkpoint not found: {missing_checkpoint}",
        ):
            model.save_pytorch(
                str(tmp_path / "bundle"),
                checkpoint_path=str(missing_checkpoint),
            )


@pytest.mark.parametrize(
    "filename",
    [
        "example_train.py",
        "example_train_lightning.py",
        "example_train_predict_in_memory.py",
        "example_predict.py",
        "example_use.py",
        "usage_example.ipynb",
    ],
)
def test_broken_compatibility_only_config_examples_are_retired(filename):
    api_dir = Path(__file__).parents[2] / "ptycho_torch" / "api"

    assert not (api_dir / filename).exists()


def test_obsolete_learned_gauge_api_is_retired():
    from ptycho_torch.model import PtychoPINN_Lightning

    retired_methods = (
        "calibrate_" + "rect_s1s2",
        "_loss_target_" + "intensity",
    )
    assert all(
        not hasattr(PtychoPINN_Lightning, method)
        for method in retired_methods
    )

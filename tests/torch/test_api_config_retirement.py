"""Focused contracts for retiring loose legacy API configuration loading."""

from collections import UserDict
from pathlib import Path

import pytest

from ptycho_torch.api import api_helper
from ptycho_torch.api.base_api import ConfigManager
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

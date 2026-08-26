"""TF-canonical rename contract (Wave B of fno-stable refactoring plan).

One name per config quantity end-to-end:
  TrainingConfig.n_groups   -> training_groups
  TrainingConfig.n_subsample -> train_raw_selection
  InferenceConfig.n_groups  -> inference_groups
  InferenceConfig.n_subsample -> inference_raw_selection
  n_images (deprecated) is kept as a parse-time alias for the group counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptycho.config.config import InferenceConfig, ModelConfig, TrainingConfig
from ptycho.config.resolution import (
    resolve_inference_config,
    resolve_training_config,
)


def test_training_config_accepts_new_names_and_drops_old():
    config = TrainingConfig(model=ModelConfig(), training_groups=7, train_raw_selection=20)
    assert config.training_groups == 7
    assert config.train_raw_selection == 20
    assert not hasattr(config, "n_groups")
    assert not hasattr(config, "n_subsample")


def test_inference_config_accepts_new_names_and_drops_old():
    config = InferenceConfig(
        model=ModelConfig(),
        model_path=Path("models/m"),
        test_data_file=Path("data/t.npz"),
        inference_groups=7,
        inference_raw_selection=20,
    )
    assert config.inference_groups == 7
    assert config.inference_raw_selection == 20
    assert not hasattr(config, "n_groups")
    assert not hasattr(config, "n_subsample")


def test_n_images_alias_maps_to_training_groups():
    with pytest.warns(DeprecationWarning, match="n_images"):
        config = TrainingConfig(model=ModelConfig(), n_images=8)
    assert config.training_groups == 8


def test_n_images_alias_maps_to_inference_groups():
    with pytest.warns(DeprecationWarning, match="n_images"):
        config = InferenceConfig(
            model=ModelConfig(),
            model_path=Path("models/m"),
            test_data_file=Path("data/t.npz"),
            n_images=8,
        )
    assert config.inference_groups == 8


def test_resolve_training_config_uses_new_names():
    config = resolve_training_config(
        {"model": {}, "training_groups": 9, "train_raw_selection": 11},
        {},
    )
    assert config.training_groups == 9
    assert config.train_raw_selection == 11


def test_resolve_training_config_n_images_alias_warns_and_resolves():
    with pytest.warns(DeprecationWarning, match="n_images"):
        config = resolve_training_config({"model": {}, "n_images": 8}, {})
    assert config.training_groups == 8


def test_resolve_inference_config_uses_new_names():
    config = resolve_inference_config(
        {
            "model": {},
            "model_path": "models/m",
            "test_data_file": "data/t.npz",
            "inference_groups": 9,
            "inference_raw_selection": 11,
        },
        {},
    )
    assert config.inference_groups == 9
    assert config.inference_raw_selection == 11


def test_resolve_inference_config_n_images_alias_warns_and_resolves():
    with pytest.warns(DeprecationWarning, match="n_images"):
        config = resolve_inference_config(
            {
                "model": {},
                "model_path": "models/m",
                "test_data_file": "data/t.npz",
                "n_images": 8,
            },
            {},
        )
    assert config.inference_groups == 8

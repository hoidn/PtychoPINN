"""Rename-elimination contract: one name per torch config quantity.

Wave A renamed ``DataConfig.K`` -> ``neighbor_count`` and
``TrainingConfig.n_groups`` -> ``training_groups``, and dropped
``DataConfig.groups_per_center`` from the persisted wire + dataclass. These
tests pin the post-rename spelling at the dataclass, resolver, and wire-encode
boundaries so a stale alias cannot silently reappear.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.config_resolution import (
    normalize_inference_patch,
    normalize_training_patch,
)
from ptycho_torch.model_spec import derive_model_spec


def _field_names(config_type) -> set[str]:
    return {item.name for item in fields(config_type)}


def test_dataclass_fields_use_new_spellings():
    data_names = _field_names(DataConfig)
    assert "neighbor_count" in data_names
    assert "K" not in data_names
    assert "groups_per_center" not in data_names

    training_names = _field_names(TrainingConfig)
    assert "training_groups" in training_names
    assert "n_groups" not in training_names


def test_training_patch_accepts_neighbor_count_rejects_K():
    normalized = normalize_training_patch({"neighbor_count": 7})
    assert normalized.values["neighbor_count"] == 7

    with pytest.raises(ValueError, match="unknown training input field"):
        normalize_training_patch({"K": 7})


def test_training_patch_accepts_training_groups_rejects_n_groups():
    normalized = normalize_training_patch({"training_groups": 4})
    assert normalized.values["training_groups"] == 4

    with pytest.raises(ValueError, match="unknown training input field"):
        normalize_training_patch({"n_groups": 4})


def test_inference_patch_accepts_new_spellings():
    normalized = normalize_inference_patch(
        {"neighbor_count": 3, "inference_groups": 2}
    )
    assert normalized.values["neighbor_count"] == 3
    assert normalized.values["inference_groups"] == 2

    # Legacy spelling is a permanently fenced alias (H2): it must normalize
    # to the canonical key, not survive verbatim.
    legacy = normalize_inference_patch({"training_groups": 2})
    assert legacy.values["inference_groups"] == 2
    assert "training_groups" not in legacy.values

    with pytest.raises(ValueError, match="unknown inference input field"):
        normalize_inference_patch({"K": 3})


def test_encode_artifact_identity_emits_v4_spellings():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V4_VERSION,
        encode_artifact_identity,
    )
    from ptycho_torch.config_bridge import to_model_config

    data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
    model = ModelConfig(object_big=False, probe_big=False, probe_mask=False)
    training = TrainingConfig(device="cpu", torch_loss_mode="poisson")
    inference = InferenceConfig()
    spec = derive_model_spec(to_model_config(data, model), model, data)
    payload = encode_artifact_identity(spec, data, training, inference)
    assert payload["schema_version"] == ARTIFACT_SCHEMA_V4_VERSION
    assert "neighbor_count" in payload["data_config"]
    assert "K" not in payload["data_config"]
    assert "groups_per_center" not in payload["data_config"]
    assert "training_groups" in payload["training_config"]
    assert "n_groups" not in payload["training_config"]

"""Focused frozen-v1 to public-object-policy v2 model identity contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest


def _coherent_configs(*, object_layout="single_patch", training_canvas="independent"):
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho_torch.config_params import DataConfig, ModelConfig

    data = DataConfig(N=64, C=1, grid_size=(1, 1), probe_scale=4.0)
    model = ModelConfig(
        C_model=1,
        C_forward=1,
        object_layout=object_layout,
        training_canvas=training_canvas,
        training_patch_weighting="probe",
        object_big=None,
        amp_activation="silu",
    )
    canonical = CanonicalModelConfig(
        N=64,
        gridsize=1,
        object_layout=object_layout,
        training_canvas=training_canvas,
        training_patch_weighting="probe",
        object_big=None,
        amp_activation="swish",
    )
    return canonical, data, model


def _legacy_v1_payload(spec):
    from ptycho_torch.model_spec import (
        MODEL_SPEC_V1_MODEL_FIELDS,
        MODEL_SPEC_V1_VERSION,
    )

    model = spec.to_model_config()
    legacy = {
        name: getattr(model, name)
        for name in MODEL_SPEC_V1_MODEL_FIELDS
    }
    return {
        "schema_version": MODEL_SPEC_V1_VERSION,
        "model_config": legacy,
        "parity_scale_mode": spec.parity_scale_mode,
        "parity_fixed_delta": spec.parity_fixed_delta,
        "parity_init_scheme": spec.parity_init_scheme,
    }


def test_current_model_spec_v2_owns_public_axes_not_object_big():
    from ptycho_torch.model_spec import (
        CURRENT_MODEL_SPEC_VERSION,
        derive_model_spec,
    )

    canonical, data, model = _coherent_configs()
    spec = derive_model_spec(canonical, model, data)
    payload = spec.to_payload()

    assert CURRENT_MODEL_SPEC_VERSION == "torch-model-spec-portable-v2"
    assert payload["schema_version"] == "torch-model-spec-portable-v2"
    assert "object_big" not in payload["model_config"]
    assert payload["model_config"]["object_layout"] == "single_patch"
    assert payload["model_config"]["training_canvas"] == "independent"
    assert payload["model_config"]["training_patch_weighting"] == "probe"
    assert spec.to_model_config().object_big is False


def test_frozen_model_spec_v1_deterministically_upgrades_to_v2():
    from ptycho_torch.model_spec import ModelSpec, derive_model_spec

    canonical, data, model = _coherent_configs()
    current = derive_model_spec(canonical, model, data)
    legacy_payload = _legacy_v1_payload(current)

    upgraded = ModelSpec.from_payload(legacy_payload)

    assert upgraded.schema_version == "torch-model-spec-portable-v2"
    assert upgraded.to_payload()["schema_version"] == "torch-model-spec-portable-v2"
    assert upgraded.to_model_config() == current.to_model_config()


def test_frozen_model_spec_v1_field_set_is_exact():
    from ptycho_torch.model_spec import ModelSpec, derive_model_spec

    canonical, data, model = _coherent_configs()
    payload = _legacy_v1_payload(derive_model_spec(canonical, model, data))
    payload["model_config"].pop("object_big")
    with pytest.raises(ValueError, match="v1.*missing=.*object_big"):
        ModelSpec.from_payload(payload)

    payload = _legacy_v1_payload(derive_model_spec(canonical, model, data))
    payload["model_config"]["future_default"] = True
    with pytest.raises(ValueError, match="v1.*unknown=.*future_default"):
        ModelSpec.from_payload(payload)


def test_model_spec_v2_rejects_unsupported_public_axis_pair():
    from ptycho_torch.model_spec import ModelSpec, derive_model_spec

    canonical, data, model = _coherent_configs()
    payload = derive_model_spec(canonical, model, data).to_payload()
    payload["model_config"]["training_canvas"] = "relative_overlap"

    with pytest.raises(ValueError, match="unsupported.*pair"):
        ModelSpec.from_payload(payload)


def test_v1_upgrade_preserves_parity_and_tensor_mask():
    import torch

    from ptycho_torch.model_spec import ModelSpec, derive_model_spec

    canonical, data, model = _coherent_configs()
    model = replace(
        model,
        probe_mask=True,
        probe_mask_tensor=torch.arange(16, dtype=torch.float32).reshape(4, 4),
    )
    canonical = replace(canonical, probe_mask=True)
    current = derive_model_spec(
        canonical,
        model,
        data,
        parity_scale_mode="fixed",
        parity_fixed_delta=1.25,
        parity_init_scheme="tf_glorot",
    )

    upgraded = ModelSpec.from_payload(_legacy_v1_payload(current))

    assert upgraded.parity_scale_mode == "fixed"
    assert upgraded.parity_fixed_delta == 1.25
    assert upgraded.parity_init_scheme == "tf_glorot"
    assert torch.equal(
        upgraded.to_model_config().probe_mask_tensor,
        model.probe_mask_tensor,
    )

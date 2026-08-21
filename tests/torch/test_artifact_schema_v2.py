"""Focused outer torch-artifact-v2 and immutable v1 decode contracts."""

from __future__ import annotations

from copy import deepcopy


def _identity_sections():
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ptycho_torch.model_spec import derive_model_spec

    data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
    model = ModelConfig(
        object_layout="single_patch",
        training_canvas="independent",
        training_patch_weighting="uniform",
        object_big=None,
        amp_activation="silu",
    )
    canonical = CanonicalModelConfig(
        N=64,
        gridsize=1,
        object_layout="single_patch",
        training_canvas="independent",
        training_patch_weighting="uniform",
        object_big=None,
        amp_activation="swish",
    )
    return (
        derive_model_spec(canonical, model, data),
        data,
        TrainingConfig(torch_loss_mode="poisson"),
        InferenceConfig(),
    )


def _v1_model_spec_payload(spec):
    from ptycho_torch.model_spec import (
        MODEL_SPEC_V1_MODEL_FIELDS,
        MODEL_SPEC_V1_VERSION,
    )

    model = spec.to_model_config()
    legacy = {}
    for name in MODEL_SPEC_V1_MODEL_FIELDS:
        if name in ("C_model", "C_forward"):
            legacy[name] = 1  # derived at decode; dropped by the upgrade
        else:
            legacy[name] = getattr(model, name)
    return {
        "schema_version": MODEL_SPEC_V1_VERSION,
        "model_config": legacy,
        "parity_scale_mode": spec.parity_scale_mode,
        "parity_fixed_delta": spec.parity_fixed_delta,
        "parity_init_scheme": spec.parity_init_scheme,
    }

def _v1_data_config_payload(data):
    from ptycho_torch.artifact_schema import PORTABLE_V1_DATA_FIELDS

    channels = data.gridsize * data.gridsize
    legacy = {
        name: getattr(data, name)
        for name in PORTABLE_V1_DATA_FIELDS
        if name not in ("C", "grid_size", "n_subsample")
    }
    legacy["C"] = channels
    legacy["grid_size"] = (data.gridsize, data.gridsize)
    legacy["n_subsample"] = data.n_raw_frames_selected
    return legacy


def test_new_artifact_identity_is_v2_with_nested_model_spec_v2():
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        decode_artifact_identity,
        encode_artifact_identity,
    )

    spec, data, training, inference = _identity_sections()
    payload = encode_artifact_identity(spec, data, training, inference)

    assert CURRENT_ARTIFACT_SCHEMA_VERSION == "torch-artifact-portable-v3"
    assert payload["schema_version"] == "torch-artifact-portable-v3"
    assert payload["model_spec"]["schema_version"] == (
        "torch-model-spec-portable-v3"
    )
    decoded = decode_artifact_identity(payload)
    assert decoded.model_spec.to_model_config() == spec.to_model_config()


def test_outer_artifact_v1_with_nested_model_spec_v1_upgrades_to_v2():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        decode_artifact_identity,
        encode_artifact_identity,
    )

    spec, data, training, inference = _identity_sections()
    payload = encode_artifact_identity(spec, data, training, inference)
    payload["schema_version"] = ARTIFACT_SCHEMA_V1_VERSION
    payload["model_spec"] = _v1_model_spec_payload(spec)
    payload["data_config"] = _v1_data_config_payload(data)
    decoded = decode_artifact_identity(payload)

    assert decoded.model_spec.schema_version == "torch-model-spec-portable-v3"
    assert decoded.model_spec.to_model_config() == spec.to_model_config()


def test_bundle_manifest_accepts_v1_and_v2_without_changing_container_or_roles():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        REQUIRED_BUNDLE_ROLES,
        TORCH_BUNDLE_VERSION,
        validate_torch_bundle_manifest,
    )

    assert TORCH_BUNDLE_VERSION == "2.0-pytorch"
    assert REQUIRED_BUNDLE_ROLES == {"autoencoder", "diffraction_to_obj"}
    for schema in (ARTIFACT_SCHEMA_V1_VERSION, CURRENT_ARTIFACT_SCHEMA_VERSION):
        manifest = {
            "version": TORCH_BUNDLE_VERSION,
            "backend": "pytorch",
            "artifact_schema_version": schema,
            "models": ["autoencoder", "diffraction_to_obj"],
        }
        assert validate_torch_bundle_manifest(manifest) == schema


def test_artifact_v2_rejects_compatibility_alias_contradiction_before_return():
    from ptycho_torch.artifact_schema import (
        decode_artifact_identity,
        encode_artifact_identity,
    )

    spec, data, training, inference = _identity_sections()
    payload = encode_artifact_identity(spec, data, training, inference)
    payload["model_spec"]["model_config"]["object_big"] = True

    try:
        decode_artifact_identity(payload)
    except ValueError as exc:
        assert "object_big" in str(exc)
    else:
        raise AssertionError("contradictory v2 object_big alias was accepted")

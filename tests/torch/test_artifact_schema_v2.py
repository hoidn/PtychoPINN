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
    values = {}
    for name in MODEL_SPEC_V1_MODEL_FIELDS:
        if name == "object_big":
            values[name] = model.object_layout == "grouped_patches"
        elif name in ("C_model", "C_forward"):
            values[name] = 1  # derived at decode; dropped by the v1->v3 upgrade
        else:
            values[name] = getattr(model, name)
    return {
        "schema_version": MODEL_SPEC_V1_VERSION,
        "model_config": values,
        "parity_scale_mode": spec.parity_scale_mode,
        "parity_fixed_delta": spec.parity_fixed_delta,
        "parity_init_scheme": spec.parity_init_scheme,
    }


def test_new_artifact_identity_is_current_era_with_nested_model_spec_v3():
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        decode_artifact_identity,
        encode_artifact_identity,
    )

    spec, data, training, inference = _identity_sections()
    payload = encode_artifact_identity(spec, data, training, inference)

    assert CURRENT_ARTIFACT_SCHEMA_VERSION == "torch-artifact-v4"
    assert payload["schema_version"] == "torch-artifact-v4"
    assert payload["model_spec"]["schema_version"] == "torch-model-spec-v3"
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
    legacy_data = dict(payload["data_config"])
    legacy_data["grid_size"] = (legacy_data.pop("gridsize"),) * 2
    legacy_data["C"] = legacy_data["grid_size"][0] * legacy_data["grid_size"][1]
    legacy_data["n_subsample"] = legacy_data.pop("n_raw_frames_selected")
    legacy_data["K"] = legacy_data.pop("neighbor_count")
    payload["data_config"] = legacy_data
    legacy_training = dict(payload["training_config"])
    legacy_training["n_groups"] = legacy_training.pop("training_groups")
    payload["training_config"] = legacy_training

    decoded = decode_artifact_identity(payload)

    assert decoded.model_spec.schema_version == "torch-model-spec-v3"
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


def test_artifact_v4_roundtrip_preserves_identity_and_declares_v4_fields():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V4_VERSION,
        decode_artifact_identity,
        encode_artifact_identity,
    )

    spec, data, training, inference = _identity_sections()
    payload = encode_artifact_identity(spec, data, training, inference)

    assert payload["schema_version"] == ARTIFACT_SCHEMA_V4_VERSION
    assert payload["model_spec"]["schema_version"] == "torch-model-spec-v3"
    # v3 data section declares the derived/renamed fields, not the stored ones.
    assert payload["data_config"]["gridsize"] == data.gridsize
    assert payload["data_config"]["n_raw_frames_selected"] == data.n_raw_frames_selected
    assert "C" not in payload["data_config"]
    assert "grid_size" not in payload["data_config"]
    assert "n_subsample" not in payload["data_config"]
    # v4 data section: renamed neighbor count, no vestigial runtime knob.
    assert payload["data_config"]["neighbor_count"] == data.neighbor_count
    assert "K" not in payload["data_config"]
    assert "groups_per_center" not in payload["data_config"]
    # v4 training section: honest groups spelling.
    assert payload["training_config"]["training_groups"] == training.training_groups
    assert "n_groups" not in payload["training_config"]
    # v3 model spec drops the stored C-family.
    assert "C_model" not in payload["model_spec"]["model_config"]
    assert "C_forward" not in payload["model_spec"]["model_config"]

    decoded = decode_artifact_identity(payload)
    assert decoded.model_spec.to_model_config() == spec.to_model_config()
    assert decoded.data_config == data
    assert decoded.training_config == training
    assert decoded.inference_config == inference


def test_artifact_v4_roundtrip_with_tensor_and_parity_identity():
    import torch

    from ptycho_torch.artifact_schema import (
        decode_artifact_identity,
        encode_artifact_identity,
    )

    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ptycho_torch.model_spec import derive_model_spec
    from ptycho.config.config import ModelConfig as CanonicalModelConfig

    data = DataConfig(N=64, gridsize=2, probe_scale=4.0)
    model = ModelConfig(
                object_layout="single_patch",
        training_canvas="independent",
        training_patch_weighting="uniform",
        object_big=None,
        amp_activation="silu",
        probe_mask=True,
        probe_mask_tensor=torch.arange(16, dtype=torch.float32).reshape(4, 4),
    )
    canonical = CanonicalModelConfig(
        N=64,
        gridsize=2,
        object_layout="single_patch",
        training_canvas="independent",
        training_patch_weighting="uniform",
        object_big=None,
        amp_activation="swish",
        probe_mask=True,
    )
    spec = derive_model_spec(
        canonical,
        model,
        data,
        parity_scale_mode="fixed",
        parity_fixed_delta=1.25,
        parity_init_scheme="tf_glorot",
    )
    training = TrainingConfig(torch_loss_mode="poisson")
    inference = InferenceConfig()

    payload = encode_artifact_identity(spec, data, training, inference)
    decoded = decode_artifact_identity(payload)

    assert decoded.model_spec.parity_scale_mode == "fixed"
    assert decoded.model_spec.parity_fixed_delta == 1.25
    assert torch.equal(
        decoded.model_spec.to_model_config().probe_mask_tensor,
        model.probe_mask_tensor,
    )
    assert decoded.data_config == data


def test_artifact_v4_rejects_unknown_model_spec_field_on_decode():
    import pytest

    from ptycho_torch.artifact_schema import (
        decode_artifact_identity,
        encode_artifact_identity,
    )

    spec, data, training, inference = _identity_sections()
    payload = encode_artifact_identity(spec, data, training, inference)
    payload["model_spec"]["model_config"]["future_default"] = True
    with pytest.raises(ValueError, match="v3.*unknown=.*future_default"):
        decode_artifact_identity(payload)


def test_v3_era_tuples_alias_identical_sections_and_data_stays_literal():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_V2_DATA_FIELDS,
        ARTIFACT_V2_INFERENCE_FIELDS,
        ARTIFACT_V2_TRAINING_FIELDS,
        ARTIFACT_V3_DATA_FIELDS,
        ARTIFACT_V3_INFERENCE_FIELDS,
        ARTIFACT_V3_TRAINING_FIELDS,
    )

    # Training/inference are identical to v2, so they alias the frozen v1/v2
    # literals (W2.1 dedup).
    assert isinstance(ARTIFACT_V3_TRAINING_FIELDS, tuple)
    assert isinstance(ARTIFACT_V3_INFERENCE_FIELDS, tuple)
    assert ARTIFACT_V3_TRAINING_FIELDS == ARTIFACT_V2_TRAINING_FIELDS
    assert ARTIFACT_V3_INFERENCE_FIELDS == ARTIFACT_V2_INFERENCE_FIELDS
    # Data drops stored C/grid_size/n_subsample and gains the v3 fields; it
    # stays a literal because it encodes the v2->v3 rename history.
    assert isinstance(ARTIFACT_V3_DATA_FIELDS, tuple)
    assert set(ARTIFACT_V3_DATA_FIELDS) == (
        set(ARTIFACT_V2_DATA_FIELDS) - {"C", "grid_size", "n_subsample"}
    ) | {"gridsize", "n_raw_frames_selected", "groups_per_center"}


def test_v4_training_fields_keep_rename_history():
    from ptycho_torch.artifact_schema import ARTIFACT_V4_TRAINING_FIELDS

    # v4 renamed n_groups -> training_groups. A wrong alias to an earlier era
    # (which still spells n_groups) must fail here at unit level.
    assert "training_groups" in ARTIFACT_V4_TRAINING_FIELDS
    assert "n_groups" not in ARTIFACT_V4_TRAINING_FIELDS

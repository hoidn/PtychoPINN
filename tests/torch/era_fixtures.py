"""Per-era ``wts.h5.zip`` bundle builders for the era x load-path matrix.

Each builder writes a REAL bundle into ``tmp_path`` and returns the bundle
directory (the parent of ``wts.h5.zip``).  The container layouts are
constructed literally — never by calling the live save path — so the fixtures
pin the wire format the way the frozen era tuples pin the field sets:

* ``dill_era``          — pre-JSON container: ``manifest.dill`` + per-model
                          ``params.dill`` (dill-encoded), no versioned torch
                          metadata member.  The only supported route for these
                          is ``python scripts/migrate_legacy_bundle.py``.
* ``portable_v1``       — JSON manifest container whose sealed identity
                          payload declares ``torch-artifact-portable-v1``
                          (sections filtered to the frozen v1 field tuples).
* ``portable_v2_json``  — current era: JSON manifest + sealed
                          ``torch-artifact-portable-v2`` identity payload.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import dill
import torch

from ptycho_torch.artifact_schema import encode_artifact_identity
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)

_BUNDLE_VERSION = "2.0-pytorch"
_ROLES = ("autoencoder", "diffraction_to_obj")
_METADATA_MEMBER = "torch_scaling_metadata.pt"

_V1 = "torch-artifact-portable-v1"
_V2 = "torch-artifact-portable-v2"
_V3 = "torch-artifact-portable-v3"
_V4 = "torch-artifact-portable-v4"


def _tiny_model():
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config = DataConfig(N=64, gridsize=1)
    model_config = ModelConfig(
        object_big=False,
        probe_big=False,
        n_filters_scale=1,
        physics_forward_mode="rectangular_scaled",
        cnn_output_mode="real_imag",
        rect_s1s2_trainable=False,
    )
    model = PtychoPINN_Lightning(
        model_config,
        data_config,
        TrainingConfig(device="cpu", torch_loss_mode="poisson"),
        InferenceConfig(),
    )
    model.register_ci_statistics(
        {
            "rms_input_scale": torch.tensor([0.375]),
            "mean_measured_intensity": torch.tensor([9.0]),
        }
    )
    return model


def _params_snapshot() -> dict:
    from ptycho.config.config import (
        ModelConfig as CanonicalModelConfig,
        TrainingConfig as CanonicalTrainingConfig,
        dataclass_to_legacy_dict,
    )

    config = CanonicalTrainingConfig(
        model=CanonicalModelConfig(N=64, gridsize=1)
    )
    snapshot = dataclass_to_legacy_dict(config)
    snapshot["intensity_scale"] = 1.0
    snapshot["_version"] = _BUNDLE_VERSION
    return snapshot


def _identity_payload(model) -> dict:
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.model_spec import derive_model_spec

    statistics = model.get_ci_statistics()
    serialized = {
        name: value.detach().cpu().reshape(-1).tolist()
        for name, value in statistics.items()
    }
    model_spec = derive_model_spec(
        to_model_config(model.data_config, model.model_config),
        model.model_config,
        model.data_config,
    )
    return encode_artifact_identity(
        model_spec,
        model.data_config,
        model.training_config,
        model.inference_config,
        ci_statistics=serialized,
    )


def _state_dict_bytes(model) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()

def _downgrade_data_section(section: dict) -> dict:
    """Project a live (v4) data section onto the frozen legacy wire shape."""
    gridsize = section["gridsize"]
    legacy = {
        name: value
        for name, value in section.items()
        if name not in ("gridsize", "n_raw_frames_selected", "neighbor_count")
    }
    legacy["C"] = gridsize * gridsize
    legacy["grid_size"] = (gridsize, gridsize)
    legacy["n_subsample"] = section["n_raw_frames_selected"]
    legacy["K"] = section["neighbor_count"]
    return legacy


def _downgrade_spec_payload(spec_payload: dict, *, to_version: str, channels: int) -> dict:
    """Project a live (spec-v3) ModelSpec payload onto the v1/v2 wire shape."""
    model_fields = dict(spec_payload["model_config"])
    model_fields["C_model"] = channels
    model_fields["C_forward"] = channels
    if to_version == "torch-model-spec-portable-v1":
        grouped = model_fields.pop("object_layout") == "grouped_patches"
        model_fields.pop("training_canvas")
        model_fields["object_big"] = grouped
    return {
        **spec_payload,
        "schema_version": to_version,
        "model_config": model_fields,
    }


def _payload_as_legacy(payload: dict, era: str) -> dict:
    """Project a current identity payload down to the frozen v1/v2 wire shape."""
    channels = payload["data_config"]["gridsize"] ** 2
    downgraded = dict(payload)
    downgraded["schema_version"] = era
    downgraded["data_config"] = _downgrade_data_section(payload["data_config"])
    training = dict(payload["training_config"])
    training["n_groups"] = training.pop("training_groups")
    downgraded["training_config"] = training
    downgraded["model_spec"] = _downgrade_spec_payload(
        payload["model_spec"],
        to_version=(
            "torch-model-spec-portable-v1"
            if era == _V1
            else "torch-model-spec-portable-v2"
        ),
        channels=channels,
    )
    return downgraded


def _downgrade_to_v3(payload: dict) -> dict:
    """Project a live (v4) identity payload down to the frozen v3 wire shape."""
    downgraded = dict(payload)
    downgraded["schema_version"] = _V3
    data = dict(payload["data_config"])
    data["K"] = data.pop("neighbor_count")
    training = dict(payload["training_config"])
    training["n_groups"] = training.pop("training_groups")
    downgraded["data_config"] = data
    downgraded["training_config"] = training
    return downgraded


def build_bundle(tmp_path: Path, era: str) -> Path:
    """Write a real per-era bundle; return the bundle directory."""
    bundle_dir = tmp_path / era
    bundle_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bundle_dir / "wts.h5.zip"

    params = _params_snapshot()

    if era == "dill_era":
        # Real metadata-free bundles carry the fixed legacy CNN architecture;
        # use the vendored reconstruction recipe so the state_dict is faithful.
        # Loaded by file path: `tests/scripts` shadows the `scripts` package.
        import importlib.util

        script_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "migrate_legacy_bundle.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_migrate_legacy_bundle_fixture", script_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        legacy_model = module._create_legacy_model(
            int(params["gridsize"]), int(params["N"]), dict(params)
        )
        weights = _state_dict_bytes(legacy_model)
        manifest = {"models": list(_ROLES), "version": _BUNDLE_VERSION}
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.dill", dill.dumps(manifest))
            for role in _ROLES:
                archive.writestr(f"{role}/model.pth", weights)
                archive.writestr(f"{role}/params.dill", dill.dumps(dict(params)))
        return bundle_dir
    if era == "dill_sealed_v1":
        # A dill-era bundle that still carries a sealed v1 identity in
        # torch_scaling_metadata.pt. The legacy migrator preserves that sealed
        # identity verbatim, so the migration door must re-encode it to v4 in
        # the same pass.
        model = _tiny_model()
        weights = _state_dict_bytes(model)
        payload = _payload_as_legacy(_identity_payload(model), _V1)
        metadata_buffer = io.BytesIO()
        torch.save(payload, metadata_buffer)
        manifest = {"models": list(_ROLES), "version": _BUNDLE_VERSION}
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.dill", dill.dumps(manifest))
            for role in _ROLES:
                archive.writestr(f"{role}/model.pth", weights)
                archive.writestr(f"{role}/params.dill", dill.dumps(dict(params)))
            archive.writestr(_METADATA_MEMBER, metadata_buffer.getvalue())
        return bundle_dir


    model = _tiny_model()
    weights = _state_dict_bytes(model)

    if era not in {"portable_v1", "portable_v2_json", "portable_v3", "portable_v4"}:
        raise ValueError(f"unknown bundle era {era!r}")

    schema = {
        "portable_v1": _V1,
        "portable_v2_json": _V2,
        "portable_v3": _V3,
        "portable_v4": _V4,
    }[era]
    payload = _identity_payload(model)
    if era in ("portable_v1", "portable_v2_json"):
        payload = _payload_as_legacy(payload, schema)
    elif era == "portable_v3":
        payload = _downgrade_to_v3(payload)
    metadata_buffer = io.BytesIO()
    torch.save(payload, metadata_buffer)

    manifest = {
        "models": list(_ROLES),
        "version": _BUNDLE_VERSION,
        "manifest_version": "torch-manifest-v1",
        "backend": "pytorch",
        "artifact_schema_version": schema,
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for role in _ROLES:
            archive.writestr(f"{role}/model.pth", weights)
            archive.writestr(f"{role}/params.json", json.dumps(dict(params)))
        archive.writestr(_METADATA_MEMBER, metadata_buffer.getvalue())
    return bundle_dir

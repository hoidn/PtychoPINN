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

from ptycho_torch.artifact_schema import (
    PORTABLE_V1_DATA_FIELDS,
    PORTABLE_V1_INFERENCE_FIELDS,
    PORTABLE_V1_TRAINING_FIELDS,
    encode_artifact_identity,
)
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


def _tiny_model():
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config = DataConfig(N=64, C=1, grid_size=(1, 1))
    model_config = ModelConfig(
        C_model=1,
        C_forward=1,
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


def _payload_as_v1(payload: dict) -> dict:
    """Project a current identity payload down to the frozen v1 field tuples."""
    downgraded = dict(payload)
    downgraded["schema_version"] = _V1
    for section, fields in (
        ("data_config", PORTABLE_V1_DATA_FIELDS),
        ("training_config", PORTABLE_V1_TRAINING_FIELDS),
        ("inference_config", PORTABLE_V1_INFERENCE_FIELDS),
    ):
        downgraded[section] = {
            name: payload[section][name]
            for name in fields
            if name in payload[section]
        }
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

    model = _tiny_model()
    weights = _state_dict_bytes(model)

    if era not in {"portable_v1", "portable_v2_json"}:
        raise ValueError(f"unknown bundle era {era!r}")

    schema = _V1 if era == "portable_v1" else _V2
    payload = _identity_payload(model)
    if era == "portable_v1":
        payload = _payload_as_v1(payload)
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

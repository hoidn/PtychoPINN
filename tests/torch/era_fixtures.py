"""Fixture factory for the bundle-era x load-path compatibility matrix.

Each builder returns a ``tmp_path``-rooted bundle directory (or a wire payload
dict) for one artifact era. The frozen pre-migration v1/v2 fixtures under
``tests/fixtures/config`` are read-only inputs; everything else is synthesized
in-memory with the smallest model each loader accepts.

Construction routes per era:

- ``v3``           -> real ``save_torch_bundle`` + ``_persist_bundle_scaling_metadata``
- ``v1`` / ``v2``  -> frozen fixture payload stamped into the wts.h5.zip layout
- ``ci-entrypoints-v1`` -> transitional template (real weights + ci-entrypoints metadata)
- ``metadata-free-legacy`` -> deleted-writer dill shape (manifest.dill + params.dill)
- ``unfaithful-legacy`` -> v1 payload with data ``C`` mutated to ``gridsize**2 + 3``

Nothing here edits production modules; the only production imports are the same
writers/loaders the compatibility matrix is meant to pin.
"""

from __future__ import annotations

import copy
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import dill
import torch

from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model_spec import derive_model_spec

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "config"

_FROZEN_V1 = "pydantic_pre_migration_torch_artifact_v1.json"
_FROZEN_V2 = "pydantic_pre_migration_torch_artifact_v2.json"


# ---------------------------------------------------------------------------
# Wire payloads
# ---------------------------------------------------------------------------

def _load_frozen(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def v1_payload() -> dict:
    """The frozen ``torch-artifact-v1`` wire payload (read-only source)."""
    return copy.deepcopy(_load_frozen(_FROZEN_V1))


def v2_payload() -> dict:
    """The frozen ``torch-artifact-v2`` wire payload (read-only source)."""
    return copy.deepcopy(_load_frozen(_FROZEN_V2))


def unfaithful_v1_payload() -> dict:
    """A v1 payload whose stored data ``C`` disagrees with ``gridsize**2``."""
    payload = v1_payload()
    grid_h, grid_w = payload["data_config"]["grid_size"]
    assert grid_h == grid_w, "frozen fixture is square"
    payload["data_config"]["C"] = grid_h * grid_h + 3
    return payload


def _identity_parts():
    """Smallest real Lightning construction (mirrors test_artifact_schema.py)."""
    data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
    model = ModelConfig(
        object_big=False,
        probe_big=False,
        probe_mask=False,
        probe_mask_tensor=None,
    )
    training = TrainingConfig(device="cpu", torch_loss_mode="poisson")
    inference = InferenceConfig()
    spec = derive_model_spec(to_model_config(data, model), model, data)
    return spec, data, training, inference


def v5_payload() -> dict:
    """A current-era ``torch-artifact-v5`` payload from the live encoder."""
    from ptycho_torch.artifact_schema import encode_artifact_identity

    spec, data, training, inference = _identity_parts()
    return encode_artifact_identity(spec, data, training, inference)


def v4_payload() -> dict:
    """A synthesized C1 ``torch-artifact-v4`` payload (pre-centered wire).

    The v4 era stored the retired K-choose-C policy fields in the data
    section and had no ``grouping_contract`` marker. Synthesized from the
    current dataclasses with those spellings restored so the v4->v5 upgrade
    path is exercised end to end.
    """
    from ptycho_torch.artifact_schema import (
        ARTIFACT_V4_DATA_FIELDS,
        ARTIFACT_V4_INFERENCE_FIELDS,
        ARTIFACT_V4_TRAINING_FIELDS,
    )

    spec, data, training, inference = _identity_parts()
    data_values = asdict(data)
    data_values["K_quadrant"] = 30
    data_values["neighbor_function"] = "Nearest"
    data_values["min_neighbor_distance"] = 0.0
    data_values["max_neighbor_distance"] = 3.0
    data_values["scan_pattern"] = "Isotropic"
    data_wire = {name: data_values[name] for name in ARTIFACT_V4_DATA_FIELDS}
    training_wire = {
        name: asdict(training)[name] for name in ARTIFACT_V4_TRAINING_FIELDS
    }
    inference_wire = {
        name: asdict(inference)[name] for name in ARTIFACT_V4_INFERENCE_FIELDS
    }
    return {
        "backend": "pytorch",
        "schema_version": "torch-artifact-v4",
        "model_spec": spec.to_payload(),
        "data_config": data_wire,
        "training_config": training_wire,
        "inference_config": inference_wire,
        "ci_statistics": None,
    }


def v3_c4_payload() -> dict:
    """A C>1 (gridsize=2) variant of the synthesized v3 payload."""
    payload = v3_payload()
    payload["data_config"]["gridsize"] = 2
    return payload


def v4_c4_payload() -> dict:
    """A C>1 (gridsize=2) variant of the synthesized v4 payload."""
    payload = v4_payload()
    payload["data_config"]["gridsize"] = 2
    return payload


def v1_c1_payload() -> dict:
    """A C1 variant of the frozen v1 payload (single-channel upgrade path)."""
    payload = v1_payload()
    payload["data_config"]["grid_size"] = [1, 1]
    payload["data_config"]["C"] = 1
    payload["model_spec"]["model_config"]["C_model"] = 1
    payload["model_spec"]["model_config"]["C_forward"] = 1
    return payload


def v2_c1_payload() -> dict:
    """A C1 variant of the frozen v2 payload (single-channel upgrade path)."""
    payload = v2_payload()
    payload["data_config"]["grid_size"] = [1, 1]
    payload["data_config"]["C"] = 1
    payload["model_spec"]["model_config"]["C_model"] = 1
    payload["model_spec"]["model_config"]["C_forward"] = 1
    return payload


def v3_payload() -> dict:
    """A frozen ``torch-artifact-v3`` wire payload (pre-rename field shapes).

    The v3 era stored ``K``/``groups_per_center`` in the data section and
    ``n_groups`` in the training section. Synthesized from the current-era
    dataclasses with those spellings restored so the v3->v4 upgrade path is
    exercised end to end.
    """
    from ptycho_torch.artifact_schema import (
        ARTIFACT_V3_DATA_FIELDS,
        ARTIFACT_V3_INFERENCE_FIELDS,
        ARTIFACT_V3_TRAINING_FIELDS,
    )

    spec, data, training, inference = _identity_parts()
    data_values = asdict(data)
    data_values["K"] = 7
    data_values["groups_per_center"] = 3
    data_values["K_quadrant"] = 30
    data_values["neighbor_function"] = "Nearest"
    data_values["min_neighbor_distance"] = 0.0
    data_values["max_neighbor_distance"] = 3.0
    data_values["scan_pattern"] = "Isotropic"
    data_wire = {name: data_values[name] for name in ARTIFACT_V3_DATA_FIELDS}
    training_values = asdict(training)
    training_values["n_groups"] = 4
    training_wire = {
        name: training_values[name] for name in ARTIFACT_V3_TRAINING_FIELDS
    }
    inference_wire = {
        name: asdict(inference)[name] for name in ARTIFACT_V3_INFERENCE_FIELDS
    }
    return {
        "backend": "pytorch",
        "schema_version": "torch-artifact-v3",
        "model_spec": spec.to_payload(),
        "data_config": data_wire,
        "training_config": training_wire,
        "inference_config": inference_wire,
        "ci_statistics": None,
    }


# ---------------------------------------------------------------------------
# Bundle builders (return a directory containing wts.h5.zip)
# ---------------------------------------------------------------------------

def _stamp_identity_bundle(
    tmp_path: Path, schema_version: str, payload: dict, name: str
) -> Path:
    """Stamp a sealed-identity payload into the versioned-JSON bundle layout.

    The loaders that accept synthesized v1/v2 run at the metadata-decode level
    (no model weights); the migrator's metadata-present branch likewise never
    reconstructs, so ``model.pth`` is intentionally absent.
    """
    grid_h, grid_w = payload["data_config"]["grid_size"]
    params = {
        "_version": "2.0-pytorch",
        "N": payload["data_config"]["N"],
        "gridsize": grid_h,
        "C": payload["data_config"]["C"],
    }
    manifest = {
        "models": ["autoencoder", "diffraction_to_obj"],
        "version": "2.0-pytorch",
        "manifest_version": "torch-manifest-v1",
        "backend": "pytorch",
        "artifact_schema_version": schema_version,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    members = {
        "manifest.json": json.dumps(manifest).encode("utf-8"),
        "torch_scaling_metadata.pt": buffer.getvalue(),
        "autoencoder/params.json": json.dumps(params).encode("utf-8"),
        "diffraction_to_obj/params.json": json.dumps(params).encode("utf-8"),
    }
    bundle_dir = tmp_path / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member, content in members.items():
            zf.writestr(member, content)
    return bundle_dir


def v1_bundle(tmp_path: Path) -> Path:
    return _stamp_identity_bundle(
        tmp_path, "torch-artifact-v1", v1_payload(), "era-v1"
    )


def v2_bundle(tmp_path: Path) -> Path:
    return _stamp_identity_bundle(
        tmp_path, "torch-artifact-v2", v2_payload(), "era-v2"
    )


def unfaithful_bundle(tmp_path: Path) -> Path:
    return _stamp_identity_bundle(
        tmp_path, "torch-artifact-v1", unfaithful_v1_payload(), "era-unfaithful"
    )

def mismatched_v2_bundle(tmp_path: Path) -> Path:
    """Manifest declares v2 but the sealed metadata payload declares v1.

    Pins the manifest/metadata agreement gate for every versioned era
    (regression: the gate's era set once omitted v2, silently skipping the
    agreement check for v2 bundles).
    """
    return _stamp_identity_bundle(
        tmp_path, "torch-artifact-v2", v1_payload(), "era-mismatch"
    )



def v5_bundle(tmp_path: Path) -> Path:
    """A real current-era (v5) bundle: save + persist scaling metadata + reload."""
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_manager import save_torch_bundle
    from ptycho_torch.workflows.bundle_io import _persist_bundle_scaling_metadata

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    bundle_dir = tmp_path / "era-v5"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    base_path = bundle_dir / "wts.h5"
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(base_path),
        CanonicalTrainingConfig(
            model=CanonicalModelConfig(N=64, gridsize=1),
            output_dir=bundle_dir,
        ),
    )
    _persist_bundle_scaling_metadata(base_path.with_suffix(".h5.zip"), model)
    return bundle_dir


def v4_bundle(tmp_path: Path) -> Path:
    """A frozen v4-era bundle: stamped v4 identity, metadata-decode level."""
    payload = v4_payload()
    grid = payload["data_config"]["gridsize"]
    params = {
        "_version": "2.0-pytorch",
        "N": payload["data_config"]["N"],
        "gridsize": grid,
        "C": grid * grid,
    }
    manifest = {
        "models": ["autoencoder", "diffraction_to_obj"],
        "version": "2.0-pytorch",
        "manifest_version": "torch-manifest-v1",
        "backend": "pytorch",
        "artifact_schema_version": "torch-artifact-v4",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    members = {
        "manifest.json": json.dumps(manifest).encode("utf-8"),
        "torch_scaling_metadata.pt": buffer.getvalue(),
        "autoencoder/params.json": json.dumps(params).encode("utf-8"),
        "diffraction_to_obj/params.json": json.dumps(params).encode("utf-8"),
    }
    bundle_dir = tmp_path / "era-v4"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member, content in members.items():
            zf.writestr(member, content)
    return bundle_dir


def v1_c1_bundle(tmp_path: Path) -> Path:
    return _stamp_identity_bundle(
        tmp_path, "torch-artifact-v1", v1_c1_payload(), "era-v1-c1"
    )


def v3_c4_bundle(tmp_path: Path) -> Path:
    """A C>1 (gridsize=2) v3-era bundle: stamped v3 identity, no weights."""
    payload = v3_c4_payload()
    grid = payload["data_config"]["gridsize"]
    params = {
        "_version": "2.0-pytorch",
        "N": payload["data_config"]["N"],
        "gridsize": grid,
        "C": grid * grid,
    }
    manifest = {
        "models": ["autoencoder", "diffraction_to_obj"],
        "version": "2.0-pytorch",
        "manifest_version": "torch-manifest-v1",
        "backend": "pytorch",
        "artifact_schema_version": "torch-artifact-v3",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    members = {
        "manifest.json": json.dumps(manifest).encode("utf-8"),
        "torch_scaling_metadata.pt": buffer.getvalue(),
        "autoencoder/params.json": json.dumps(params).encode("utf-8"),
        "diffraction_to_obj/params.json": json.dumps(params).encode("utf-8"),
    }
    bundle_dir = tmp_path / "era-v3-c4"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member, content in members.items():
            zf.writestr(member, content)
    return bundle_dir


def v4_c4_bundle(tmp_path: Path) -> Path:
    """A C>1 (gridsize=2) v4-era bundle: stamped v4 identity, no weights."""
    payload = v4_c4_payload()
    grid = payload["data_config"]["gridsize"]
    params = {
        "_version": "2.0-pytorch",
        "N": payload["data_config"]["N"],
        "gridsize": grid,
        "C": grid * grid,
    }
    manifest = {
        "models": ["autoencoder", "diffraction_to_obj"],
        "version": "2.0-pytorch",
        "manifest_version": "torch-manifest-v1",
        "backend": "pytorch",
        "artifact_schema_version": "torch-artifact-v4",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    members = {
        "manifest.json": json.dumps(manifest).encode("utf-8"),
        "torch_scaling_metadata.pt": buffer.getvalue(),
        "autoencoder/params.json": json.dumps(params).encode("utf-8"),
        "diffraction_to_obj/params.json": json.dumps(params).encode("utf-8"),
    }
    bundle_dir = tmp_path / "era-v4-c4"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member, content in members.items():
            zf.writestr(member, content)
    return bundle_dir


def v2_c1_bundle(tmp_path: Path) -> Path:
    return _stamp_identity_bundle(
        tmp_path, "torch-artifact-v2", v2_c1_payload(), "era-v2-c1"
    )


def v3_bundle(tmp_path: Path) -> Path:
    """A frozen v3-era bundle: stamped v3 identity, metadata-decode level."""
    payload = v3_payload()
    grid = payload["data_config"]["gridsize"]
    params = {
        "_version": "2.0-pytorch",
        "N": payload["data_config"]["N"],
        "gridsize": grid,
        "C": grid * grid,
    }
    manifest = {
        "models": ["autoencoder", "diffraction_to_obj"],
        "version": "2.0-pytorch",
        "manifest_version": "torch-manifest-v1",
        "backend": "pytorch",
        "artifact_schema_version": "torch-artifact-v3",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    members = {
        "manifest.json": json.dumps(manifest).encode("utf-8"),
        "torch_scaling_metadata.pt": buffer.getvalue(),
        "autoencoder/params.json": json.dumps(params).encode("utf-8"),
        "diffraction_to_obj/params.json": json.dumps(params).encode("utf-8"),
    }
    bundle_dir = tmp_path / "era-v3"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for member, content in members.items():
            zf.writestr(member, content)
    return bundle_dir


def ci_entrypoints_bundle(tmp_path: Path) -> Path:
    """The transitional ci-entrypoints-v1 template (real weights + metadata)."""
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_manager import save_torch_bundle

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    bundle_dir = tmp_path / "era-ci-entrypoints"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    base_path = bundle_dir / "wts.h5"
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(base_path),
        CanonicalTrainingConfig(
            model=CanonicalModelConfig(N=64, gridsize=1),
            output_dir=bundle_dir,
        ),
    )
    transitional = {
        "schema_version": "ci-entrypoints-v1",
        "data_config": asdict(data),
        "model_config": asdict(model.model_config),
        "training_config": asdict(training),
        "inference_config": asdict(inference),
        "ci_statistics": None,
    }
    buffer = io.BytesIO()
    torch.save(transitional, buffer)
    with zipfile.ZipFile(
        base_path.with_suffix(".h5.zip"), "a", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("torch_scaling_metadata.pt", buffer.getvalue())
    return bundle_dir


def _legacy_shaped_sections(data, model_config) -> tuple[dict, dict]:
    """Project current-era dataclass sections onto the LEGACY (v1) shapes.

    Real ci-entrypoints-v1 bundles were written by pre-v3 code whose asdict
    carried C/grid_size/n_subsample and C_model/C_forward. Built from the era
    tuples so drift fails loudly here instead of silently testing the wrong
    branch.
    """
    from ptycho_torch.artifact_schema import ARTIFACT_V1_DATA_FIELDS
    from ptycho_torch.model_spec import MODEL_SPEC_V1_MODEL_FIELDS

    current_data = asdict(data)
    gridsize = current_data["gridsize"]
    legacy_values = {
        "C": gridsize * gridsize,
        "grid_size": (gridsize, gridsize),
        "n_subsample": current_data["n_raw_frames_selected"],
        "K": current_data["neighbor_count"],
        "K_quadrant": 30,
        "neighbor_function": "Nearest",
        "min_neighbor_distance": 0.0,
        "max_neighbor_distance": 3.0,
        "scan_pattern": "Isotropic",
    }
    legacy_data = {
        name: legacy_values[name] if name in legacy_values else current_data[name]
        for name in ARTIFACT_V1_DATA_FIELDS
    }
    current_model = asdict(model_config)
    legacy_model_values = {
        "C_model": gridsize * gridsize,
        "C_forward": gridsize * gridsize,
    }
    legacy_model = {
        name: (
            legacy_model_values[name]
            if name in legacy_model_values
            else current_model[name]
        )
        for name in MODEL_SPEC_V1_MODEL_FIELDS
    }
    assert set(legacy_data) == set(ARTIFACT_V1_DATA_FIELDS)
    assert set(legacy_model) == set(MODEL_SPEC_V1_MODEL_FIELDS)
    return legacy_data, legacy_model


def _stamp_transitional_metadata(bundle_dir: Path, transitional: dict) -> Path:
    buffer = io.BytesIO()
    torch.save(transitional, buffer)
    with zipfile.ZipFile(
        bundle_dir / "wts.h5.zip", "a", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("torch_scaling_metadata.pt", buffer.getvalue())
    return bundle_dir


def ci_entrypoints_legacy_shaped_bundle(tmp_path: Path) -> Path:
    """A shape-faithful ci-entrypoints-v1 bundle (LEGACY section shapes).

    Exercises upgrade_unversioned_sections' legacy-shaped branch including
    the era="unversioned" channel-faithfulness validation.
    """
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_manager import save_torch_bundle

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    bundle_dir = tmp_path / "era-ci-entrypoints-legacy-shaped"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(bundle_dir / "wts.h5"),
        CanonicalTrainingConfig(
            model=CanonicalModelConfig(N=64, gridsize=1),
            output_dir=bundle_dir,
        ),
    )
    legacy_data, legacy_model = _legacy_shaped_sections(data, model.model_config)
    return _stamp_transitional_metadata(
        bundle_dir,
        {
            "schema_version": "ci-entrypoints-v1",
            "data_config": legacy_data,
            "model_config": legacy_model,
            "training_config": asdict(training),
            "inference_config": asdict(inference),
            "ci_statistics": None,
        },
    )


def unfaithful_unversioned_bundle(tmp_path: Path) -> Path:
    """Legacy-shaped ci-entrypoints metadata whose stored C disagrees."""
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_manager import save_torch_bundle

    spec, data, training, inference = _identity_parts()
    model = build_ptychopinn_application(spec, data, training, inference)
    bundle_dir = tmp_path / "era-unfaithful-unversioned"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(bundle_dir / "wts.h5"),
        CanonicalTrainingConfig(
            model=CanonicalModelConfig(N=64, gridsize=1),
            output_dir=bundle_dir,
        ),
    )
    legacy_data, legacy_model = _legacy_shaped_sections(data, model.model_config)
    legacy_data["C"] = legacy_data["C"] + 2
    return _stamp_transitional_metadata(
        bundle_dir,
        {
            "schema_version": "ci-entrypoints-v1",
            "data_config": legacy_data,
            "model_config": legacy_model,
            "training_config": asdict(training),
            "inference_config": asdict(inference),
            "ci_statistics": None,
        },
    )


def _legacy_model(gridsize: int, N: int, params_dict: dict):
    """Mirror the metadata-free-era cnn reconstruction (current-era configs).

    ``scripts/migrate_legacy_bundle.create_torch_model_with_gridsize`` is the
    production twin; it is broken pre-fix (retired ``C``/``grid_size``/``C_model``
    kwargs), so the fixture builds the identical model with the current-era
    dataclass fields. Channel count derives from ``gridsize`` exactly as the
    post-fix migrator will derive it.
    """
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config = DataConfig(
        N=N,
        gridsize=gridsize,
        neighbor_count=params_dict.get("neighbor_count", 6),
        nphotons=params_dict.get("nphotons", 1e5),
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
    )
    model_config = ModelConfig(
        architecture="cnn",
        n_filters_scale=int(params_dict.get("n_filters_scale", 2)),
        amp_activation=params_dict.get("amp_activation", "silu"),
        mode="Unsupervised",
        cnn_output_mode="amp_phase",
        use_shared_decoder=False,
        batch_norm=False,
        edge_pad=10,
        decoder_last_c_outer_fraction=0.125,
        decoder_last_amp_channels=1,
        use_legacy_decoder_channel_override=False,
        eca_encoder=False,
        cbam_encoder=True,
        cbam_bottleneck=False,
        cbam_decoder=False,
        eca_decoder=False,
        spatial_decoder=False,
        decoder_spatial_kernel=7,
        object_big=params_dict.get("object.big", False),
        probe_big=params_dict.get("probe.big", True),
        physics_forward_mode="amplitude",
        training_patch_weighting="central_mask",
        loss_function=params_dict.get("loss_function", "Poisson"),
        intensity_scale=params_dict.get("intensity_scale", 1.0),
        intensity_scale_trainable=params_dict.get(
            "intensity_scale.trainable", False
        ),
    )
    training_config = TrainingConfig(
        epochs=params_dict.get("nepochs", 50),
        batch_size=params_dict.get("batch_size", 16),
        learning_rate=params_dict.get("learning_rate", 1e-3),
        torch_loss_mode="poisson",
    )
    inference_config = InferenceConfig(batch_size=params_dict.get("batch_size", 1000))
    return PtychoPINN_Lightning(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )


def metadata_free_bundle(tmp_path: Path) -> Path:
    """The deleted-writer dill shape: manifest.dill + params.dill, no identity."""
    from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
    from ptycho.config.config import ModelConfig as CanonicalModelConfig
    from ptycho.config.config import dataclass_to_legacy_dict

    config = CanonicalTrainingConfig(
        model=CanonicalModelConfig(N=64, gridsize=1), nphotons=1e9
    )
    archived = dataclass_to_legacy_dict(config)
    archived["intensity_scale"] = 1.0
    archived["_version"] = "2.0-pytorch"
    model = _legacy_model(1, 64, archived)

    source = tmp_path / "era-metadata-free"
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source / "wts.h5.zip", "w") as zf:
        zf.writestr(
            "manifest.dill",
            dill.dumps(
                {
                    "models": ["autoencoder", "diffraction_to_obj"],
                    "version": "2.0-pytorch",
                }
            ),
        )
        for model_name in ("autoencoder", "diffraction_to_obj"):
            zf.writestr(f"{model_name}/params.dill", dill.dumps(archived))
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            zf.writestr(f"{model_name}/model.pth", buffer.getvalue())
    return source


# ---------------------------------------------------------------------------
# Read helper (the real metadata decode from a stamped bundle)
# ---------------------------------------------------------------------------

def read_bundle_metadata(bundle_dir: Path) -> dict:
    """Read the sealed ``torch_scaling_metadata.pt`` payload out of a bundle."""
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "r") as zf:
        return torch.load(
            io.BytesIO(zf.read("torch_scaling_metadata.pt")),
            map_location="cpu",
            weights_only=True,
        )

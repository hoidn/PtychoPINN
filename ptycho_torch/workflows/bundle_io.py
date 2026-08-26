"""Bundle persistence, decoding, and strict reconstruction for ``wts.h5.zip``.

Owns the bundle metadata decode/encode slab, ``load_inference_bundle_torch``,
and the scaling-metadata persistence. The ``components`` facade re-exports this
slab so the spec-pinned module path and public names are unchanged.
"""
import io
import logging
from contextlib import contextmanager
from pathlib import Path
import zipfile
from typing import Any, Dict, Optional, Tuple, Union

from ptycho import params
from ptycho.config.legacy_state import transactional_legacy_params
from ptycho_torch.model_manager import (
    _read_torch_bundle_manifest_and_params,
    _reconstruct_torch_bundle_explicit,
)
from ptycho_torch.scaling_contract import (
    AmplitudePhysicsGainRecord,
    CI_SCALE_CONTRACT,
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
    amplitude_physics_gain_record_from_json,
    amplitude_physics_gain_record_to_json,
    ci_scaling_active,
    resolve_scale_contract,
    validate_amplitude_physics_gain,
)

# Preserves pre-split log provenance.
logger = logging.getLogger("ptycho_torch.workflows.components")

_BUNDLE_SCALING_METADATA = "torch_scaling_metadata.pt"
_BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD = "amplitude_physics_gain_record.json"

def _persist_bundle_scaling_metadata(
    archive_path: Path,
    model,
    *,
    amplitude_physics_gain_record: Optional[
        AmplitudePhysicsGainRecord
    ] = None,
) -> None:
    """Append the torch config and frozen CI statistics needed for strict reload."""
    statistics = model.get_ci_statistics()
    profile = resolve_scale_contract(
        model.data_config.scale_contract_version,
        model.data_config.measurement_domain,
    )
    ci_bundle = ci_scaling_active(model.model_config) and profile.version == CI_SCALE_CONTRACT
    if ci_bundle and statistics is None:
        raise ValueError(
            "Cannot persist a CI bundle without frozen training ci_statistics."
        )
    serialized_statistics = None
    if statistics is not None:
        serialized_statistics = {
            name: value.detach().cpu().reshape(-1).tolist()
            for name, value in statistics.items()
        }
    import torch

    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        TORCH_ARTIFACT_BACKEND,
        encode_artifact_identity,
        validate_torch_bundle_manifest,
    )
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.model_spec import derive_model_spec

    model_spec = getattr(model, "_model_spec", None)
    if model_spec is None:
        model_spec = derive_model_spec(
            to_model_config(model.data_config, model.model_config),
            model.model_config,
            model.data_config,
            parity_scale_mode=getattr(model, "parity_scale_mode", "off"),
            parity_fixed_delta=float(model.hparams.get("parity_fixed_delta", 0.0)),
            parity_init_scheme=model.hparams.get("parity_init_scheme", "default"),
        )
    payload = encode_artifact_identity(
        model_spec,
        model.data_config,
        model.training_config,
        model.inference_config,
        ci_statistics=serialized_statistics,
    )
    sidecar_json = (
        amplitude_physics_gain_record_to_json(amplitude_physics_gain_record)
        if amplitude_physics_gain_record is not None
        else None
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)

    import dill
    import os
    import tempfile

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename
            not in {
                "manifest.dill",
                _BUNDLE_SCALING_METADATA,
                _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
            }
        }
        manifest = dill.loads(archive.read("manifest.dill"))
    validate_torch_bundle_manifest(manifest)
    manifest.update(
        backend=TORCH_ARTIFACT_BACKEND,
        artifact_schema_version=CURRENT_ARTIFACT_SCHEMA_VERSION,
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=archive_path.name,
        suffix=".tmp",
        dir=archive_path.parent,
    )
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary_name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.dill", dill.dumps(manifest))
            for name, content in members.items():
                archive.writestr(name, content)
            archive.writestr(_BUNDLE_SCALING_METADATA, buffer.getvalue())
            if sidecar_json is not None:
                archive.writestr(
                    _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
                    sidecar_json.encode("utf-8"),
                )
        os.replace(temporary_name, archive_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_bundle_scaling_metadata(archive_path: Path):
    import torch

    if not archive_path.is_file():
        return None
    with zipfile.ZipFile(archive_path, "r") as archive:
        if _BUNDLE_SCALING_METADATA not in archive.namelist():
            return None
        return torch.load(
            io.BytesIO(archive.read(_BUNDLE_SCALING_METADATA)),
            map_location="cpu",
            weights_only=False,
        )


def _read_bundle_amplitude_physics_gain_record(
    archive_path: Path,
) -> Optional[AmplitudePhysicsGainRecord]:
    if not archive_path.is_file():
        return None
    with zipfile.ZipFile(archive_path, "r") as archive:
        if _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD not in archive.namelist():
            return None
        encoded = archive.read(_BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD)
    return amplitude_physics_gain_record_from_json(encoded)


def _decode_bundle_metadata(metadata):
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        decode_artifact_identity,
        upgrade_unversioned_sections,
    )

    schema = metadata.get("schema_version") if isinstance(metadata, dict) else None
    if schema in {
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
    }:
        return decode_artifact_identity(metadata)
    if schema == "ci-entrypoints-v1":
        return upgrade_unversioned_sections(
            data_config=metadata["data_config"],
            model_config=metadata["model_config"],
            training_config=metadata["training_config"],
            inference_config=metadata["inference_config"],
            ci_statistics=metadata.get("ci_statistics"),
        )
    raise ValueError(
        f"unsupported wts.h5.zip Torch metadata schema {schema!r}"
    )


def _strictly_reconstruct_bundle_model(archive_path: Path, identity, model_name: str):
    import torch
    from ptycho_torch.application_factory import build_ptychopinn_application

    model = build_ptychopinn_application(
        identity.model_spec,
        identity.data_config,
        identity.training_config,
        identity.inference_config,
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        try:
            state_dict = torch.load(
                io.BytesIO(archive.read(f"{model_name}/model.pth")),
                map_location="cpu",
                weights_only=False,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Bundle weights are missing for model '{model_name}'. Regenerate "
                "the bundle from a successful training result."
            ) from exc
    if not isinstance(state_dict, dict) or "_sentinel" in state_dict:
        raise RuntimeError(
            f"Bundle weights for model '{model_name}' are not a trained state_dict. "
            "Regenerate the bundle from a successful training result."
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Bundle architecture-era incompatibility for model '{model_name}': "
            "strict physics/model weight "
            "loading failed. Do not use strict=False; regenerate this bundle with "
            f"the current architecture. Original error: {exc}"
        ) from exc

    profile = resolve_scale_contract(
        identity.data_config.scale_contract_version,
        identity.data_config.measurement_domain,
    )
    ci_bundle = (
        ci_scaling_active(model.model_config)
        and profile.version == CI_SCALE_CONTRACT
    )
    statistics = identity.ci_statistics
    if ci_bundle and statistics is None:
        raise ValueError(
            "CI bundle is missing frozen training ci_statistics; regenerate the bundle."
        )
    if statistics is not None:
        model.register_ci_statistics(statistics)
    return model


def _reconstruct_inference_bundle_explicit(
    archive_path: Path,
    zip_path: Path,
    *,
    manifest: dict,
    params_dict: dict,
    identity: Optional[Any],
    explicit_profile: Optional[Tuple[str, str]],
    model_name: str,
) -> Tuple[Dict[str, Any], dict, Optional[Any]]:
    """Reconstruct a decoded bundle without consulting or mutating params.cfg."""
    decoded_params = dict(params_dict)
    available_models = manifest["models"]

    if identity is None:
        required_profile = (
            LEGACY_SCALE_CONTRACT,
            NORMALIZED_AMPLITUDE,
        )
        if explicit_profile != required_profile:
            raise ValueError(
                "This metadata-free bundle is provenance-known legacy. Supply both "
                "scale_contract_version='legacy_v1' and "
                "measurement_domain='normalized_amplitude'."
            )
        models_dict, _ = _reconstruct_torch_bundle_explicit(
            str(archive_path),
            manifest=manifest,
            params_dict=params_dict,
            model_name=model_name,
        )
        for loaded_model in models_dict.values():
            loaded_model.data_config.scale_contract_version = (
                LEGACY_SCALE_CONTRACT
            )
            loaded_model.data_config.measurement_domain = NORMALIZED_AMPLITUDE
        decoded_params["scale_contract_version"] = LEGACY_SCALE_CONTRACT
        decoded_params["measurement_domain"] = NORMALIZED_AMPLITUDE
        return models_dict, decoded_params, None

    persisted_profile = resolve_scale_contract(
        identity.data_config.scale_contract_version,
        identity.data_config.measurement_domain,
    )
    if explicit_profile is not None and explicit_profile != (
        persisted_profile.version,
        persisted_profile.measurement_domain,
    ):
        raise ValueError(
            "Explicit bundle profile overrides contradict persisted metadata."
        )
    models_dict = {
        archived_model_name: _strictly_reconstruct_bundle_model(
            zip_path,
            identity,
            archived_model_name,
        )
        for archived_model_name in available_models
    }
    decoded_params["scale_contract_version"] = persisted_profile.version
    decoded_params["measurement_domain"] = (
        persisted_profile.measurement_domain
    )
    decoded_params["ci_statistics"] = identity.ci_statistics
    return models_dict, decoded_params, identity


@contextmanager
def _pinned_bundle_snapshot(zip_path: Path):
    """Yield an immutable private snapshot of one archive generation."""
    from collections import Counter
    import shutil
    import tempfile

    if not zip_path.is_file():
        raise FileNotFoundError(f"Model archive not found: {zip_path}")

    with tempfile.TemporaryDirectory(
        prefix="ptycho-torch-bundle-snapshot-"
    ) as temporary_directory:
        snapshot_zip_path = Path(temporary_directory) / zip_path.name
        with zip_path.open("rb") as source, snapshot_zip_path.open("wb") as target:
            shutil.copyfileobj(source, target)

        with zipfile.ZipFile(snapshot_zip_path, "r") as archive:
            counts = Counter(info.filename for info in archive.infolist())
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                "Torch bundle contains duplicate archive member(s): "
                + ", ".join(duplicates)
            )

        yield snapshot_zip_path.with_suffix(""), snapshot_zip_path


def _decode_pinned_inference_bundle(
    archive_path: Path,
    zip_path: Path,
    *,
    model_name: str,
    explicit_profile: Optional[Tuple[str, str]],
) -> Tuple[Any, dict, Optional[AmplitudePhysicsGainRecord]]:
    """Decode and reconstruct all members from one private snapshot."""
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        validate_torch_bundle_manifest,
    )

    manifest, params_dict = _read_torch_bundle_manifest_and_params(
        str(archive_path)
    )
    manifest_era = validate_torch_bundle_manifest(manifest)
    metadata = _read_bundle_scaling_metadata(zip_path)
    amplitude_physics_gain_record = (
        _read_bundle_amplitude_physics_gain_record(zip_path)
    )
    if metadata is None:
        known_legacy = (
            params_dict.get("_version") == "2.0-pytorch"
            and "scale_contract_version" not in params_dict
            and "measurement_domain" not in params_dict
        )
        if not known_legacy:
            raise ValueError(
                "wts.h5.zip has no versioned Torch metadata and is not the "
                "declared metadata-free legacy_v1 era"
            )
        identity = None
    else:
        metadata_schema = metadata.get("schema_version")
        if (
            manifest_era in {
                ARTIFACT_SCHEMA_V1_VERSION,
                CURRENT_ARTIFACT_SCHEMA_VERSION,
            }
            and metadata_schema != manifest_era
        ):
            raise ValueError(
                "wts.h5.zip root manifest and metadata schemas disagree: "
                f"manifest={manifest_era!r}, declares {metadata_schema!r}"
            )
        if (
            manifest_era == "metadata-free-legacy"
            and metadata_schema != "ci-entrypoints-v1"
        ):
            raise ValueError(
                "wts.h5.zip legacy root supports only transitional "
                f"ci-entrypoints-v1 metadata, found {metadata_schema!r}"
            )
        identity = _decode_bundle_metadata(metadata)

    if amplitude_physics_gain_record is not None:
        if identity is None:
            raise ValueError(
                "amplitude physics gain sidecar requires persisted ModelSpec "
                "metadata for its scalar join"
            )
        model_gain = validate_amplitude_physics_gain(
            identity.model_spec.to_model_config()
        )
        if amplitude_physics_gain_record.value != model_gain:
            raise ValueError(
                "amplitude physics gain record disagrees with persisted "
                "ModelSpec amplitude_physics_gain"
            )

    models_dict, params_dict, _ = _reconstruct_inference_bundle_explicit(
        archive_path,
        zip_path,
        manifest=manifest,
        params_dict=params_dict,
        identity=identity,
        explicit_profile=explicit_profile,
        model_name=model_name,
    )
    return models_dict, params_dict, amplitude_physics_gain_record


@transactional_legacy_params
def load_inference_bundle_torch(
    bundle_dir: Union[str, Path],
    model_name: str = "diffraction_to_obj",
    *,
    scale_contract_version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
) -> Tuple[Any, dict]:
    """Strictly load a trained PyTorch bundle from a pinned snapshot."""
    archive_path = Path(bundle_dir) / "wts.h5"
    zip_path = archive_path.with_suffix(".h5.zip")
    logger.info("Loading PyTorch inference bundle from %s.zip", archive_path)

    from ptycho_torch.config_factory import resolve_profile_overrides

    explicit_profile = resolve_profile_overrides(
        {
            "scale_contract_version": scale_contract_version,
            "measurement_domain": measurement_domain,
        }
    )
    with _pinned_bundle_snapshot(zip_path) as (
        pinned_archive_path,
        pinned_zip_path,
    ):
        models_dict, params_dict, amplitude_physics_gain_record = (
            _decode_pinned_inference_bundle(
                pinned_archive_path,
                pinned_zip_path,
                model_name=model_name,
                explicit_profile=explicit_profile,
            )
        )

    params.cfg.update(params_dict)
    returned_params = dict(params_dict)
    if amplitude_physics_gain_record is not None:
        returned_params["amplitude_physics_gain_record"] = (
            amplitude_physics_gain_record
        )

    logger.info(
        "Inference bundle loaded successfully. Models: %s, Params keys: %s...",
        list(models_dict),
        list(params_dict)[:5],
    )
    return models_dict, returned_params

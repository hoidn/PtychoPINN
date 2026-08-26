"""Bundle persistence, decoding, and strict reconstruction for ``wts.h5.zip``.

Owns the versioned JSON manifest and ``torch.load`` serialization sites in the
workflow layer (Ratchet B pins their per-file counts here).  The ``components``
facade re-exports this slab so the spec-pinned module path and public names are
unchanged.
"""
from contextlib import contextmanager
import io
import json
import logging
from pathlib import Path
import zipfile
from typing import Any, Dict, Optional, Tuple, Union

from ptycho_torch.scaling_contract import (
    AmplitudePhysicsGainRecord,
    CI_SCALE_CONTRACT,
    amplitude_physics_gain_record_from_json,
    amplitude_physics_gain_record_to_json,
    ci_scaling_active,
    resolve_scale_contract,
    validate_amplitude_physics_gain,
)
from ptycho_torch import model_manager

# Preserves pre-split log provenance: records stay on the components facade logger.
logger = logging.getLogger('ptycho_torch.workflows.components')

_BUNDLE_SCALING_METADATA = "torch_scaling_metadata.pt"
_BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD = "amplitude_physics_gain_record.json"
def _persist_bundle_scaling_metadata(
    archive_path: Path,
    model,
    *,
    amplitude_physics_gain_record: Optional[
        AmplitudePhysicsGainRecord
    ] = None,
    checkpoint_selection: Optional[Dict[str, Any]] = None,
    training_sampling: Optional[Dict[str, Any]] = None,
    rescaled_source_sha256: Optional[str] = None,
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
        TORCH_MANIFEST_MEMBER,
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
    sidecar_json = (
        amplitude_physics_gain_record_to_json(amplitude_physics_gain_record)
        if amplitude_physics_gain_record is not None
        else None
    )
    payload = encode_artifact_identity(
        model_spec,
        model.data_config,
        model.training_config,
        model.inference_config,
        ci_statistics=serialized_statistics,
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    import os
    import tempfile

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename
            not in {
                TORCH_MANIFEST_MEMBER,
                _BUNDLE_SCALING_METADATA,
                _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
            }
        }
        manifest = json.loads(archive.read(TORCH_MANIFEST_MEMBER))
    validate_torch_bundle_manifest(manifest)
    manifest.update(
        backend=TORCH_ARTIFACT_BACKEND,
        artifact_schema_version=CURRENT_ARTIFACT_SCHEMA_VERSION,
    )
    if checkpoint_selection is not None:
        manifest["checkpoint_selection"] = dict(checkpoint_selection)
    if training_sampling is not None:
        manifest["training_sampling"] = dict(training_sampling)
    if rescaled_source_sha256 is not None:
        manifest["rescaled_source_sha256"] = rescaled_source_sha256
    validate_torch_bundle_manifest(manifest)
    handle, temporary_name = tempfile.mkstemp(
        prefix=archive_path.name,
        suffix=".tmp",
        dir=archive_path.parent,
    )
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary_name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(TORCH_MANIFEST_MEMBER, json.dumps(manifest))
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
            weights_only=True,
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
        RUNTIME_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
        decode_artifact_identity,
        ensure_supported_artifact_schema_version,
        upgrade_unversioned_sections,
    )

    schema = metadata.get("schema_version") if isinstance(metadata, dict) else None
    if schema == "ci-entrypoints-v1":
        return upgrade_unversioned_sections(
            data_config=metadata["data_config"],
            model_config=metadata["model_config"],
            training_config=metadata["training_config"],
            inference_config=metadata["inference_config"],
            ci_statistics=metadata.get("ci_statistics"),
        )
    ensure_supported_artifact_schema_version(
        schema,
        context="wts.h5.zip Torch metadata",
        supported_versions=RUNTIME_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    )
    return decode_artifact_identity(metadata)


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
                weights_only=True,
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
    identity: Any,
    explicit_profile: Optional[Tuple[str, str]],
    model_name: str,
) -> Tuple[Dict[str, Any], dict, Any]:
    """Reconstruct a decoded bundle without consulting or mutating params.cfg."""
    decoded_params = dict(params_dict)
    available_models = manifest["models"]

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
    """Yield an immutable private snapshot of one opened archive generation."""
    from collections import Counter
    import shutil
    import tempfile

    if not zip_path.is_file():
        raise FileNotFoundError(f"Model archive not found: {zip_path}")

    with tempfile.TemporaryDirectory(
        prefix="ptycho-torch-bundle-snapshot-"
    ) as temporary_directory:
        snapshot_zip_path = Path(temporary_directory) / zip_path.name
        # Opening the live path once pins its inode even if another process
        # atomically replaces the directory entry while the copy is running.
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
    """Decode and reconstruct every bundle member from one private snapshot."""
    from ptycho_torch.artifact_schema import (
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
        validate_torch_bundle_manifest,
    )

    manifest, params_dict = model_manager._read_torch_bundle_manifest_and_params(
        str(archive_path)
    )
    manifest_era = validate_torch_bundle_manifest(manifest)
    if "rescaled_source_sha256" in manifest:
        params_dict = dict(params_dict)
        params_dict["rescaled_source_sha256"] = manifest[
            "rescaled_source_sha256"
        ]
    metadata = _read_bundle_scaling_metadata(zip_path)
    amplitude_physics_gain_record = (
        _read_bundle_amplitude_physics_gain_record(zip_path)
    )
    if metadata is None:
        raise ValueError(
            "wts.h5.zip has no sealed identity metadata. Run "
            "`python -m ptycho_torch.migrate_bundle` to migrate this pre-JSON or "
            "metadata-free legacy bundle to the versioned JSON manifest era."
        )
    metadata_schema = metadata.get("schema_version")
    if (
        manifest_era in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS
        and metadata_schema != manifest_era
    ):
        raise ValueError(
            "wts.h5.zip root manifest and metadata schemas disagree: "
            f"manifest={manifest_era!r}, "
            f"declares {metadata_schema!r}"
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


def load_inference_bundle_torch(
    bundle_dir: Union[str, Path],
    model_name: str = 'diffraction_to_obj',
    *,
    scale_contract_version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
) -> Tuple[Any, dict]:
    """
    Load a trained PyTorch model bundle for inference from a directory.

    This function provides API parity with ptycho.workflows.components.load_inference_bundle,
    enabling transparent backend selection for model loading.

    Decoding and model reconstruction are explicit and global-free. This
    legacy/API wrapper commits the validated archive projection only after
    every declared model role has loaded successfully.

    Args:
        bundle_dir: Path to the directory containing the wts.h5.zip archive.
                   Expects TorchModelManager archive format matching TensorFlow baseline.
        model_name: Name of model to load from dual-model bundle (default: 'diffraction_to_obj')
        scale_contract_version: Optional explicit profile; when supplied together
            with ``measurement_domain``, must match the bundle's sealed identity.
        measurement_domain: Optional explicit profile companion of
            ``scale_contract_version``.

    Returns:
        Tuple containing:
        - models_dict: Dictionary with loaded model
        - params_dict: Configuration dictionary restored from saved bundle

    Raises:
        ValueError: If bundle archive not found or params.json incomplete, or an
            explicit profile contradicts the sealed identity.
        FileNotFoundError: If bundle_dir does not contain wts.h5.zip

    Example:
        >>> models_dict, params_dict = load_inference_bundle_torch("outputs/run_001")
        >>> # params.cfg now restored with training-time N, gridsize, nphotons
        >>> # Use models_dict['diffraction_to_obj'] for inference

    Lifecycle: authored config -> resolved payload -> sealed identity ->
    restored identity (four invocation points: the training service, loader
    construction, module construction, and the inference kernel; see
    docs/workflows/pytorch.md).  Hop count: CLI -> load_inference_bundle_torch
    -> _decode_pinned_inference_bundle -> _reconstruct_inference_bundle_explicit
    -> _strictly_reconstruct_bundle_model (kernel), 4 hops.
    """
    from ptycho import params
    from ptycho.config.legacy_state import archived_params_scope

    with archived_params_scope({}):
        # Normalize bundle_dir to string for Path compatibility
        bundle_dir_str = str(bundle_dir)

        # Build archive path following TensorFlow convention (wts.h5.zip in bundle_dir)
        # TensorFlow baseline: load_inference_bundle expects model_dir containing wts.h5.zip
        # PyTorch mirrors this: bundle_dir/wts.h5.zip.
        archive_path = Path(bundle_dir_str) / "wts.h5"
        zip_path = archive_path.with_suffix(".h5.zip")

        logger.info(f"Loading PyTorch inference bundle from {archive_path}.zip")

        from ptycho_torch.config_factory import resolve_profile_overrides
        explicit_profile = resolve_profile_overrides({
            "scale_contract_version": scale_contract_version,
            "measurement_domain": measurement_domain,
        })
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

        logger.info(f"Inference bundle loaded successfully. Models: {list(models_dict.keys())}, Params keys: {list(params_dict.keys())[:5]}...")

        # Return (models_dict, params_dict) matching TensorFlow baseline signature
        # models_dict already contains both models per implementation
        return models_dict, returned_params

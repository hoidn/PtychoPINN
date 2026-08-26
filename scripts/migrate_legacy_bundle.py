"""Legacy dill-hosting migration implementation for ``ptycho_torch.migrate_bundle``.

This module is the sole supported route for recovering PyTorch bundles that
predate the versioned JSON manifest + sealed-identity contract.  It reads a
legacy archive (``manifest.dill`` + ``params.dill``, or the JSON-manifest-but-unsealed
form emitted by ``save_torch_bundle`` alone), classifies its metadata era, and
rewrites it as a JSON-manifest bundle carrying a sealed identity payload:

* ``torch-artifact-v1`` / ``torch-artifact-v2`` / ``torch-artifact-v3`` /
  ``ci-entrypoints-v1`` bundles already carry an identity payload; their
  manifest and params members are re-serialized as JSON and the identity is
  preserved verbatim.
* metadata-free bundles reconstruct the model the way the removed legacy load
  path did, seal a fresh current (``torch-artifact-v5``) identity payload from
  the reconstructed configuration, and stamp it into ``torch_scaling_metadata.pt``.

The reconstruction helpers below are vendored copies of the deleted
``ptycho_torch.model_manager`` legacy loaders; they live only here so pre-spec
bundles remain recoverable.  The canonical offline CLI is
``python -m ptycho_torch.migrate_bundle``; this module stays outside
``ptycho_torch`` because it imports ``dill`` to read pre-JSON archives.
"""

import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import dill
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptycho_torch.artifact_schema import (
    CURRENT_ARTIFACT_SCHEMA_VERSION,
    TORCH_ARTIFACT_BACKEND,
    TORCH_MANIFEST_MEMBER,
    TORCH_MANIFEST_JSON_VERSION,
    TORCH_PARAMS_MEMBER,
    decode_artifact_identity,
    encode_artifact_identity,
    ensure_supported_artifact_schema_version,
    validate_torch_bundle_manifest,
)
from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.model_spec import derive_model_spec

_BUNDLE_SCALING_METADATA = "torch_scaling_metadata.pt"
_CI_ENTRYPOINTS_SCHEMA = "ci-entrypoints-v1"
_LEGACY_MANIFEST_MEMBER = "manifest.dill"
_LEGACY_PARAMS_MEMBER = "params.dill"

# Retired centered-grouping policy names: never re-emitted into migrated
# params.json members (the current DataConfig rejects them as unknown fields).
_RETIRED_PARAM_NAMES = frozenset(
    {
        "enable_oversampling",
        "neighbor_pool_size",
        "neighbor_function",
        "K_quadrant",
        "min_neighbor_distance",
        "max_neighbor_distance",
        "scan_pattern",
        "require_complete_group_coverage",
    }
)


# ---------------------------------------------------------------------------
# Vendored legacy reconstruction (deleted from ptycho_torch.model_manager)
# ---------------------------------------------------------------------------

def create_torch_model_with_gridsize(gridsize, N, params_dict=None):
    """Vendored legacy model builder: reconstruct the metadata-free-era model."""
    from ptycho_torch.config_params import (
        ModelConfig,
        DataConfig,
        TrainingConfig,
        InferenceConfig,
    )

    params_dict = params_dict or {}

    # Derive the channel count from the legacy C/grid_size identity and require
    # it to agree with the gridsize used to reconstruct the model.
    gridsize = int(gridsize)
    derived_channels = gridsize * gridsize
    stored_c = params_dict.get('C', derived_channels)
    if int(stored_c) != derived_channels:
        raise ValueError(
            "legacy bundle channel identity is unfaithful: stored C="
            f"{stored_c} conflicts with derived gridsize**2={derived_channels}"
        )
    if derived_channels != 1:
        raise ValueError(
            "metadata-free legacy bundles derive C="
            f"{derived_channels} from gridsize={gridsize}; the centered-nearest "
            "grouping contract supports exactly one derived channel; retrain "
            "under torch-artifact-v5"
        )
    if params_dict.get('model_type', 'pinn') != 'pinn':
        raise ValueError(
            "metadata-free legacy_v1 wts.h5.zip supports only the declared "
            "unsupervised CNN model_type='pinn' era"
        )

    data_config = DataConfig(
        N=N,
        gridsize=gridsize,
        neighbor_count=params_dict.get('neighbor_count', 6),
        nphotons=params_dict.get('nphotons', 1e5),
        scale_contract_version='legacy_v1',
        measurement_domain='normalized_amplitude',
    )

    model_config = ModelConfig(
        architecture='cnn',
        n_filters_scale=int(params_dict.get('n_filters_scale', 2)),
        amp_activation=params_dict.get('amp_activation', 'silu'),
        mode='Unsupervised',
        cnn_output_mode='amp_phase',
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
        object_big=params_dict.get('object.big', False),
        probe_big=params_dict.get('probe.big', True),
        physics_forward_mode='amplitude',
        training_patch_weighting='central_mask',
        loss_function=params_dict.get('loss_function', 'Poisson'),
        intensity_scale=params_dict.get('intensity_scale', 1.0),
        intensity_scale_trainable=params_dict.get(
            'intensity_scale.trainable', False
        ),
    )

    training_config = TrainingConfig(
        epochs=params_dict.get('nepochs', 50),
        batch_size=params_dict.get('batch_size', 16),
        learning_rate=params_dict.get('learning_rate', 1e-3),
        torch_loss_mode='poisson',
    )

    inference_config = InferenceConfig(
        batch_size=params_dict.get('batch_size', 1000),
    )

    from ptycho_torch.model import PtychoPINN_Lightning
    return PtychoPINN_Lightning(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )


def _reconstruct_legacy_models(zip_path, manifest, params_dict):
    """Vendored legacy reconstruction: build + strictly load every role."""
    available_models = manifest["models"]
    required_fields = ["N", "gridsize"]
    missing = [field for field in required_fields if field not in params_dict]
    if missing:
        raise ValueError(
            f"params.dill missing required fields: {missing}. "
            "Cannot reconstruct model architecture."
        )

    expected_models = {"autoencoder", "diffraction_to_obj"}
    if set(available_models) != expected_models:
        raise RuntimeError(
            "Torch bundle does not contain the required dual-model weights "
            f"{sorted(expected_models)}; found {sorted(available_models)}. "
            "Regenerate the bundle from a successful training result."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)

        gridsize = params_dict["gridsize"]
        N = params_dict["N"]

        models_dict = {}
        for model_name in available_models:
            model = create_torch_model_with_gridsize(gridsize, N, params_dict)

            model_path = os.path.join(temp_dir, model_name, "model.pth")
            if not os.path.isfile(model_path):
                raise RuntimeError(
                    f"Bundle weights are missing for model '{model_name}'. "
                    "Regenerate the bundle from a successful training result."
                )
            state_dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state_dict, dict) or "_sentinel" in state_dict:
                raise RuntimeError(
                    f"Bundle weights for model '{model_name}' are not a trained "
                    "state_dict. Regenerate the bundle from a successful training result."
                )
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Bundle architecture-era incompatibility for model "
                    f"'{model_name}': strict weight loading failed. Do not use "
                    "strict=False; regenerate this bundle with the current architecture. "
                    f"Original error: {exc}"
                ) from exc

            models_dict[model_name] = model

        return models_dict


# ---------------------------------------------------------------------------
# Era classification and identity sealing
# ---------------------------------------------------------------------------

def _read_manifest_and_params(archive, names):
    """Read manifest + params from a legacy or JSON bundle (JSON-first)."""
    if TORCH_MANIFEST_MEMBER in names:
        manifest = json.loads(archive.read(TORCH_MANIFEST_MEMBER))
    else:
        manifest = dill.loads(archive.read(_LEGACY_MANIFEST_MEMBER))
    available_models = manifest["models"]
    if not available_models:
        raise ValueError("Bundle manifest contains no models.")
    json_params = f"{available_models[0]}/{TORCH_PARAMS_MEMBER}"
    dill_params = f"{available_models[0]}/{_LEGACY_PARAMS_MEMBER}"
    if json_params in names:
        params_dict = json.loads(archive.read(json_params))
    else:
        params_dict = dill.loads(archive.read(dill_params))
    return manifest, params_dict


def _read_scaling_metadata(archive, names):
    if _BUNDLE_SCALING_METADATA not in names:
        return None
    return torch.load(
        io.BytesIO(archive.read(_BUNDLE_SCALING_METADATA)),
        map_location="cpu",
        weights_only=True,
    )


def _seal_identity(model):
    """Derive + encode a fresh current identity from a reconstructed legacy model."""
    model_spec = derive_model_spec(
        to_model_config(model.data_config, model.model_config),
        model.model_config,
        model.data_config,
        parity_scale_mode=getattr(model, "parity_scale_mode", "off"),
        parity_fixed_delta=float(model.hparams.get("parity_fixed_delta", 0.0)),
        parity_init_scheme=model.hparams.get("parity_init_scheme", "default"),
    )
    return encode_artifact_identity(
        model_spec,
        model.data_config,
        model.training_config,
        model.inference_config,
        ci_statistics=None,
    )


def migrate_bundle(source_dir, out_dir):
    """Migrate ``<source_dir>/wts.h5.zip`` and write ``<out_dir>/wts.h5.zip``."""
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    source_zip = source_dir / "wts.h5.zip"
    if not source_zip.is_file():
        raise FileNotFoundError(f"Legacy bundle not found: {source_zip}")

    with zipfile.ZipFile(source_zip, "r") as archive:
        names = set(archive.namelist())
        manifest, params_dict = _read_manifest_and_params(archive, names)
        params_dict = {
            name: value
            for name, value in params_dict.items()
            if name not in _RETIRED_PARAM_NAMES
        }
        metadata = _read_scaling_metadata(archive, names)
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename
            not in {
                TORCH_MANIFEST_MEMBER,
                _LEGACY_MANIFEST_MEMBER,
                _BUNDLE_SCALING_METADATA,
            }
            and not info.filename.endswith(f"/{TORCH_PARAMS_MEMBER}")
            and not info.filename.endswith(f"/{_LEGACY_PARAMS_MEMBER}")
        }

    manifest_era = validate_torch_bundle_manifest(manifest)

    manifest_out = dict(manifest)
    if metadata is None:
        # Metadata-free legacy: reconstruct, seal a fresh current identity, and
        # promote the manifest to the current schema era.
        models_dict = _reconstruct_legacy_models(source_zip, manifest, params_dict)
        first_model = models_dict[manifest["models"][0]]
        identity_payload = _seal_identity(first_model)
        manifest_out["backend"] = TORCH_ARTIFACT_BACKEND
        manifest_out["artifact_schema_version"] = CURRENT_ARTIFACT_SCHEMA_VERSION
        manifest_out["manifest_version"] = TORCH_MANIFEST_JSON_VERSION
    else:
        metadata_schema = metadata.get("schema_version")
        if metadata_schema == _CI_ENTRYPOINTS_SCHEMA:
            if manifest_era != "metadata-free-legacy":
                raise ValueError(
                    "wts.h5.zip legacy root supports only transitional "
                    f"ci-entrypoints-v1 metadata, found {metadata_schema!r}"
                )
            # The transitional era is preserved verbatim; the runtime upgrades it.
            identity_payload = metadata
        else:
            ensure_supported_artifact_schema_version(
                metadata_schema, context="wts.h5.zip Torch metadata"
            )
            if manifest_era != metadata_schema:
                raise ValueError(
                    "wts.h5.zip root manifest and metadata schemas disagree: "
                    f"manifest={manifest_era!r}, declares {metadata_schema!r}"
                )
            # Validate the sealed identity so an unfaithful legacy channel
            # identity is rejected rather than re-emitted, then re-seal it under
            # the current era so the output loads on the v3-v5 runtime path.
            decoded = decode_artifact_identity(metadata)
            identity_payload = encode_artifact_identity(
                decoded.model_spec,
                decoded.data_config,
                decoded.training_config,
                decoded.inference_config,
                ci_statistics=decoded.ci_statistics,
            )
            manifest_out["artifact_schema_version"] = CURRENT_ARTIFACT_SCHEMA_VERSION

    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "wts.h5.zip"
    handle, temporary_name = tempfile.mkstemp(
        prefix=out_zip.name, suffix=".tmp", dir=out_dir
    )
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary_name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(TORCH_MANIFEST_MEMBER, json.dumps(manifest_out))
            for model_name in manifest["models"]:
                archive.writestr(
                    f"{model_name}/{TORCH_PARAMS_MEMBER}",
                    json.dumps(params_dict),
                )
            for name, content in members.items():
                archive.writestr(name, content)
            buffer = io.BytesIO()
            torch.save(identity_payload, buffer)
            archive.writestr(_BUNDLE_SCALING_METADATA, buffer.getvalue())
        os.replace(temporary_name, out_zip)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return out_zip


def main() -> None:
    raise SystemExit(
        "scripts/migrate_legacy_bundle.py is a library module; run the "
        "offline migration CLI via `python -m ptycho_torch.migrate_bundle "
        "SOURCE_DIR OUT_DIR` instead."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Offline migration for pre-JSON PyTorch ``wts.h5.zip`` bundles.

This script is the sole supported route for recovering PyTorch bundles that
predate the versioned JSON manifest contract (spec §4.6).  It reads a legacy
archive (``manifest.dill`` + per-model ``params.dill``), classifies its
metadata era, and rewrites it as a JSON-manifest bundle:

* Bundles already carrying a sealed identity payload
  (``torch-artifact-portable-v1`` / ``torch-artifact-portable-v2`` /
  transitional ``ci-entrypoints-v1``) have their manifest and params members
  re-serialized as JSON; the identity payload is preserved verbatim.
* Metadata-free bundles reconstruct the model the way the retired in-process
  legacy load path did (declared ``legacy_v1`` / ``normalized_amplitude``
  profile), then seal a fresh current (``torch-artifact-portable-v2``)
  identity payload into ``torch_scaling_metadata.pt``.

The reconstruction helpers below are vendored copies of the retired
``ptycho_torch.model_manager`` legacy loaders; they live only here so pre-spec
bundles remain recoverable.  This module stays outside ``ptycho_torch``
because it imports ``dill`` to read pre-JSON archives.

Usage:
    python scripts/migrate_legacy_bundle.py SOURCE_DIR OUT_DIR

where SOURCE_DIR contains the legacy ``wts.h5.zip`` and OUT_DIR receives the
migrated ``wts.h5.zip``.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import dill
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptycho_torch.artifact_schema import (  # noqa: E402
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    TORCH_ARTIFACT_BACKEND,
    TORCH_BUNDLE_VERSION,
    TORCH_MANIFEST_JSON_VERSION,
    TORCH_MANIFEST_MEMBER,
    TORCH_PARAMS_MEMBER,
)

_METADATA_MEMBER = "torch_scaling_metadata.pt"
_CI_ENTRYPOINTS_SCHEMA = "ci-entrypoints-v1"
_LEGACY_MANIFEST_MEMBER = "manifest.dill"
_LEGACY_PARAMS_SUFFIX = "/params.dill"


# ---------------------------------------------------------------------------
# Vendored legacy reconstruction (retired from ptycho_torch.model_manager)
# ---------------------------------------------------------------------------

def _create_legacy_model(gridsize: int, N: int, params_dict: dict):
    """Vendored ``create_torch_model_with_gridsize`` for metadata-free bundles."""
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ptycho_torch.model import PtychoPINN_Lightning

    channels = int(params_dict.get("C", gridsize * gridsize))
    if channels != gridsize * gridsize:
        raise ValueError(
            "legacy params channel identity is unfaithful: stored C="
            f"{channels} conflicts with gridsize={gridsize}"
        )
    if params_dict.get("model_type", "pinn") != "pinn":
        raise ValueError(
            "metadata-free legacy_v1 wts.h5.zip supports only the declared "
            "unsupervised CNN model_type='pinn' era"
        )
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
    inference_config = InferenceConfig(
        batch_size=params_dict.get("batch_size", 1000),
    )
    return PtychoPINN_Lightning(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )


def _reconstruct_legacy_model(zip_path: Path, role: str, params_dict: dict):
    """Vendored strict state-dict restore from the retired explicit loader."""
    required = [field for field in ("N", "gridsize") if field not in params_dict]
    if required:
        raise ValueError(
            f"params.dill missing required fields: {required}. "
            "Cannot reconstruct model architecture."
        )
    model = _create_legacy_model(
        int(params_dict["gridsize"]), int(params_dict["N"]), params_dict
    )
    with zipfile.ZipFile(zip_path, "r") as archive:
        state_dict = torch.load(
            io.BytesIO(archive.read(f"{role}/model.pth")),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(state_dict, dict) or "_sentinel" in state_dict:
        raise RuntimeError(
            f"Bundle weights for model '{role}' are not a trained state_dict."
        )
    model.load_state_dict(state_dict, strict=True)
    return model


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _jsonify(value):
    """Best-effort projection of dill-era params values onto JSON types."""
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def migrate_bundle(source_dir: Path, out_dir: Path) -> Path:
    """Migrate one legacy bundle; returns the migrated wts.h5.zip path."""
    source_zip = Path(source_dir) / "wts.h5.zip"
    if not source_zip.is_file():
        raise FileNotFoundError(f"No wts.h5.zip in {source_dir}")

    with zipfile.ZipFile(source_zip, "r") as archive:
        names = set(archive.namelist())
        members = {name: archive.read(name) for name in names}

    if TORCH_MANIFEST_MEMBER in names:
        raise ValueError(
            f"{source_zip} already carries {TORCH_MANIFEST_MEMBER}; nothing to migrate."
        )
    if _LEGACY_MANIFEST_MEMBER not in names:
        raise ValueError(
            f"{source_zip} has no {_LEGACY_MANIFEST_MEMBER}; not a legacy Torch bundle."
        )

    manifest = dill.loads(members[_LEGACY_MANIFEST_MEMBER])
    if manifest.get("version") != TORCH_BUNDLE_VERSION:
        raise ValueError(
            f"unsupported legacy manifest version {manifest.get('version')!r}"
        )
    roles = list(manifest.get("models", ()))
    if not roles:
        raise ValueError("legacy manifest contains no models")

    params_dict = dill.loads(members[f"{roles[0]}{_LEGACY_PARAMS_SUFFIX}"])
    params_json = json.dumps(_jsonify(params_dict))

    metadata_bytes = members.get(_METADATA_MEMBER)
    sealed_schema = None
    if metadata_bytes is not None:
        metadata = torch.load(
            io.BytesIO(metadata_bytes), map_location="cpu", weights_only=True
        )
        sealed_schema = metadata.get("schema_version")
        if (
            sealed_schema not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS
            and sealed_schema != _CI_ENTRYPOINTS_SCHEMA
        ):
            raise ValueError(
                f"unsupported sealed metadata schema {sealed_schema!r}"
            )

    new_manifest = {
        "models": roles,
        "version": TORCH_BUNDLE_VERSION,
        "manifest_version": TORCH_MANIFEST_JSON_VERSION,
        "backend": TORCH_ARTIFACT_BACKEND,
    }
    if sealed_schema in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        new_manifest["artifact_schema_version"] = sealed_schema

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "wts.h5.zip"

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(TORCH_MANIFEST_MEMBER, json.dumps(new_manifest))
        for name, content in members.items():
            if name == _LEGACY_MANIFEST_MEMBER or name.endswith(
                _LEGACY_PARAMS_SUFFIX
            ):
                continue
            archive.writestr(name, content)
        for role in roles:
            archive.writestr(f"{role}/{TORCH_PARAMS_MEMBER}", params_json)

    if metadata_bytes is None:
        # Metadata-free bundle: reconstruct via the vendored legacy loader and
        # seal a fresh current identity the way the training path does.
        from ptycho_torch.workflows.bundle_io import (
            _persist_bundle_scaling_metadata,
        )

        model = _reconstruct_legacy_model(source_zip, roles[0], params_dict)
        _persist_bundle_scaling_metadata(out_zip, model)
        # The retired restore path declared this profile on load; persist it.
        declared = {str(key): _jsonify(item) for key, item in params_dict.items()}
        declared["scale_contract_version"] = "legacy_v1"
        declared["measurement_domain"] = "normalized_amplitude"
        declared_json = json.dumps(declared)
        with zipfile.ZipFile(out_zip, "r") as archive:
            sealed_members = {
                name: archive.read(name) for name in archive.namelist()
            }
        for role in roles:
            sealed_members[f"{role}/{TORCH_PARAMS_MEMBER}"] = (
                declared_json.encode("utf-8")
            )
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sealed_members.items():
                archive.writestr(name, content)

    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a pre-JSON PyTorch wts.h5.zip bundle to the "
        "versioned JSON manifest era."
    )
    parser.add_argument("source_dir", type=Path, help="directory holding the legacy wts.h5.zip")
    parser.add_argument("out_dir", type=Path, help="directory receiving the migrated wts.h5.zip")
    args = parser.parse_args()
    out_zip = migrate_bundle(args.source_dir, args.out_dir)
    print(f"migrated bundle written to {out_zip}")


if __name__ == "__main__":
    main()

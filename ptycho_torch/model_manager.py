"""
PyTorch model persistence layer for PtychoPINN.

This module provides wts.h5.zip-compatible archive creation and loading functions
for PyTorch models, maintaining format parity with the TensorFlow ModelManager
implementation in ptycho/model_manager.py.

Critical Design Requirements:
- Archive format matches the TensorFlow baseline layout (root manifest + per-model
  subdirs), but PyTorch archives use a versioned JSON manifest (`manifest.json`) and
  per-model `params.json` (spec §4.6).
- Legacy projection snapshots MUST come from dataclass_to_legacy_dict (CONFIG-001)
- Dual-model bundle support MUST be maintained (spec §4.6 requirement)

Architecture Role:
    Persistence shim bridging PyTorch training (Lightning-based) and the reconstructor
    contract defined in specs/ptychodus_api_spec.md §4.5-4.6. Enables cross-backend
    compatibility by preserving TensorFlow archive schema.

Core Functionality:
    - save_torch_bundle: Create wts.h5.zip archives from trained PyTorch models
    - _read_torch_bundle_manifest_and_params: JSON-only bundle identity/params reader

File Format:
    Identical to TensorFlow except model.pth (state_dict) replaces model.keras,
    with version='2.0-pytorch' tag for backend detection.

Usage Example:
    # Training - save dual-model bundle
    models = {'autoencoder': model1, 'diffraction_to_obj': model2}
    save_torch_bundle(models, 'output/wts.h5', config)

    # Inference - load through the strict sealed-identity path
    from ptycho_torch.workflows.components import load_inference_bundle_torch
    models, params_dict = load_inference_bundle_torch('output')

References:
- TensorFlow baseline: ptycho/model_manager.py:346-378 (save_multiple_models)
- Persistence contract: specs/ptychodus_api_spec.md §4.6 (PyTorch archives)
- Spec contract: specs/ptychodus_api_spec.md:192-202
"""

import os
import json
import tempfile
import zipfile
from typing import Dict, Any, Tuple, Optional

# PyTorch is now a mandatory dependency (Phase F3.1/F3.2)
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for ptycho_torch modules. "
        "Install PyTorch >= 2.2 with: pip install torch>=2.2"
    ) from e


def save_torch_bundle(
    models_dict: Dict[str, Any],
    base_path: str,
    config: Any,
    intensity_scale: Optional[float] = None
) -> None:
    """
    Save PyTorch models to wts.h5.zip-compatible archive with dual-model structure.

    Creates a zip archive matching the TensorFlow ModelManager.save_multiple_models
    format, enabling cross-backend compatibility and satisfying the reconstructor
    persistence contract (spec §4.6).

    Archive Structure:
        {base_path}.zip/
        ├── manifest.json  # {'models': [...], 'version': '2.0-pytorch', ...}
        ├── autoencoder/
        │   ├── model.pth  # PyTorch state_dict
        │   └── params.json  # Config-derived legacy projection (CONFIG-001)
        └── diffraction_to_obj/
            ├── model.pth
            └── params.json

    Args:
        models_dict: Dictionary mapping model names to nn.Module instances.
                     MUST contain 'autoencoder' and 'diffraction_to_obj' keys
                     (dual-model bundle requirement per spec §4.6).
        base_path: Base path for output archive (will create {base_path}.zip).
        config: TrainingConfig instance used to generate the legacy projection via
                dataclass_to_legacy_dict (CONFIG-001 bridge from Phase B).
        intensity_scale: Optional learned intensity normalization factor. ``None``
                        uses the versioned codec default of ``1.0``.

    Raises:
        ValueError: If models_dict is empty or missing required model names.
        RuntimeError: If PyTorch is unavailable and models are not sentinel dicts.

    Example:
        >>> from ptycho.config.config import TrainingConfig, ModelConfig
        >>> config = TrainingConfig(model=ModelConfig(N=64, gridsize=2), ...)
        >>> models = {'autoencoder': ae_model, 'diffraction_to_obj': recon_model}
        >>> save_torch_bundle(models, 'output/wts.h5', config)
        >>> # Creates output/wts.h5.zip with dual-model structure
    """
    from ptycho.config.config import dataclass_to_legacy_dict

    # Validate inputs
    if not models_dict:
        raise ValueError("models_dict must contain at least one model")

    # The public bundle contract requires both logical roles.
    expected_keys = {'autoencoder', 'diffraction_to_obj'}
    if set(models_dict.keys()) != expected_keys:
        raise ValueError(
            f"Incomplete dual-model bundle: found {set(models_dict.keys())}, "
            f"expected {expected_keys}."
        )

    # Build the legacy projection from the caller-owned resolved config only.
    params_snapshot = dataclass_to_legacy_dict(config)

    # Learned state is supplied explicitly; absence has a codec-owned default.
    params_snapshot['intensity_scale'] = (
        1.0 if intensity_scale is None else intensity_scale
    )

    # Add version tag for backend detection
    params_snapshot['_version'] = '2.0-pytorch'

    # Prepare output path
    zip_path = f"{base_path}.zip"
    os.makedirs(os.path.dirname(zip_path) if os.path.dirname(zip_path) else '.', exist_ok=True)

    # Create archive in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save manifest (versioned JSON era for PyTorch archives, spec §4.6)
        from ptycho_torch.artifact_schema import (
            TORCH_MANIFEST_JSON_VERSION,
            TORCH_MANIFEST_MEMBER,
            TORCH_PARAMS_MEMBER,
        )

        manifest = {
            'models': list(models_dict.keys()),
            'version': '2.0-pytorch',
            'manifest_version': TORCH_MANIFEST_JSON_VERSION,
            'backend': 'pytorch',
        }
        manifest_path = os.path.join(temp_dir, TORCH_MANIFEST_MEMBER)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Save each model
        for model_name, model in models_dict.items():
            model_dir = os.path.join(temp_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save model weights (PyTorch is mandatory, no fallback)
            model_path = os.path.join(model_dir, 'model.pth')
            if not isinstance(model, nn.Module):
                raise RuntimeError(
                    f"Cannot save model '{model_name}': expected a trained nn.Module, "
                    f"got {type(model)}. Sentinel/fresh model placeholders are not "
                    "valid persistence inputs."
                )
            torch.save(model.state_dict(), model_path)

            # Save params snapshot (CONFIG-001 critical; JSON-native projection)
            params_path = os.path.join(model_dir, TORCH_PARAMS_MEMBER)
            with open(params_path, 'w') as f:
                json.dump(params_snapshot.copy(), f)

        # Create zip archive
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arc_path = os.path.relpath(full_path, temp_dir)
                    zf.write(full_path, arc_path)


def _read_torch_bundle_manifest_and_params(base_path: str) -> Tuple[dict, dict]:
    """Read bundle identity and archived flat state without constructing a model."""
    zip_path = f"{base_path}.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model archive not found: {zip_path}")

    from ptycho_torch.artifact_schema import (
        TORCH_MANIFEST_MEMBER,
        TORCH_PARAMS_MEMBER,
        validate_torch_bundle_manifest,
    )

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        try:
            manifest = json.loads(archive.read(TORCH_MANIFEST_MEMBER))
            available_models = manifest["models"]
            if not available_models:
                raise ValueError("Bundle manifest contains no models.")
            params_member = f"{available_models[0]}/{TORCH_PARAMS_MEMBER}"
            params_dict = json.loads(archive.read(params_member))
        except KeyError as exc:
            if "manifest.dill" in names or any(
                name.endswith("/params.dill") for name in names
            ):
                raise ValueError(
                    "wts.h5.zip is a legacy pre-JSON bundle; run "
                    "`python -m ptycho_torch.migrate_bundle` to migrate it to the "
                    "versioned JSON manifest era."
                ) from exc
            raise ValueError(
                f"Malformed Torch bundle: missing required archive member {exc}."
            ) from exc

    validate_torch_bundle_manifest(manifest)
    if params_dict.get("_version") != "2.0-pytorch":
        raise ValueError(
            "unsupported wts.h5.zip params version "
            f"{params_dict.get('_version')!r}; expected '2.0-pytorch'"
        )
    return manifest, params_dict

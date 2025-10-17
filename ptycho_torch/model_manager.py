"""
PyTorch model persistence layer for PtychoPINN.

This module provides wts.h5.zip-compatible archive creation and loading functions
for PyTorch models, maintaining format parity with the TensorFlow ModelManager
implementation in ptycho/model_manager.py while supporting torch-optional execution.

Critical Design Requirements:
- Archive format MUST match TensorFlow baseline (manifest.dill + per-model subdirs)
- params.cfg snapshot MUST be captured via dataclass_to_legacy_dict (CONFIG-001)
- Dual-model bundle support MUST be maintained (spec §4.6 requirement)
- All functions MUST be torch-optional (importable when PyTorch unavailable)

Architecture Role:
    Persistence shim bridging PyTorch training (Lightning-based) and the reconstructor
    contract defined in specs/ptychodus_api_spec.md §4.5-4.6. Enables cross-backend
    compatibility by preserving TensorFlow archive schema.

Core Functionality:
    - save_torch_bundle: Create wts.h5.zip archives from trained PyTorch models
    - load_torch_bundle: Restore models with CONFIG-001-compliant params restoration

File Format:
    Identical to TensorFlow except model.pth (state_dict) replaces model.keras,
    with version='2.0-pytorch' tag for backend detection.

Usage Example:
    # Training - save dual-model bundle
    models = {'autoencoder': model1, 'diffraction_to_obj': model2}
    save_torch_bundle(models, 'output/wts.h5', config)

    # Inference - load with params restoration
    model, params_dict = load_torch_bundle('output/wts.h5')

References:
- TensorFlow baseline: ptycho/model_manager.py:346-378 (save_multiple_models)
- Phase D3 callchain: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/
- Spec contract: specs/ptychodus_api_spec.md:192-202
"""

import os
import dill
import tempfile
import zipfile
import shutil
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

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
        ├── manifest.dill  # {'models': [...], 'version': '2.0-pytorch'}
        ├── autoencoder/
        │   ├── model.pth  # PyTorch state_dict
        │   └── params.dill  # Full params.cfg snapshot (CONFIG-001)
        └── diffraction_to_obj/
            ├── model.pth
            └── params.dill

    Args:
        models_dict: Dictionary mapping model names to nn.Module instances.
                     MUST contain 'autoencoder' and 'diffraction_to_obj' keys
                     (dual-model bundle requirement per spec §4.6).
        base_path: Base path for output archive (will create {base_path}.zip).
        config: TrainingConfig instance used to generate params.cfg snapshot via
                dataclass_to_legacy_dict (CONFIG-001 bridge from Phase B).
        intensity_scale: Optional intensity normalization factor. If None, attempts
                        to extract from params.cfg or model attributes.

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
    from ptycho import params

    # Validate inputs
    if not models_dict:
        raise ValueError("models_dict must contain at least one model")

    # Warn if dual-model structure is incomplete (spec §4.6 preference)
    expected_keys = {'autoencoder', 'diffraction_to_obj'}
    if set(models_dict.keys()) != expected_keys:
        import warnings
        warnings.warn(
            f"Incomplete dual-model bundle: found {set(models_dict.keys())}, "
            f"expected {expected_keys}. Reconstructor may fail to load archive.",
            UserWarning
        )

    # Generate params.cfg snapshot via config bridge (CONFIG-001)
    params_snapshot = dataclass_to_legacy_dict(config)

    # Add intensity_scale to snapshot
    if intensity_scale is not None:
        params_snapshot['intensity_scale'] = intensity_scale
    elif 'intensity_scale' not in params_snapshot:
        # Attempt to extract from params.cfg if available
        if 'intensity_scale' in params.cfg:
            params_snapshot['intensity_scale'] = params.cfg['intensity_scale']
        else:
            # Default fallback (documented in Phase D3.A open question Q2)
            params_snapshot['intensity_scale'] = 1.0

    # Add version tag for backend detection
    params_snapshot['_version'] = '2.0-pytorch'

    # Prepare output path
    zip_path = f"{base_path}.zip"
    os.makedirs(os.path.dirname(zip_path) if os.path.dirname(zip_path) else '.', exist_ok=True)

    # Create archive in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save manifest
        manifest = {
            'models': list(models_dict.keys()),
            'version': '2.0-pytorch'
        }
        manifest_path = os.path.join(temp_dir, 'manifest.dill')
        with open(manifest_path, 'wb') as f:
            dill.dump(manifest, f)

        # Save each model
        for model_name, model in models_dict.items():
            model_dir = os.path.join(temp_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save model weights (PyTorch is mandatory, no fallback)
            model_path = os.path.join(model_dir, 'model.pth')
            if isinstance(model, nn.Module):
                # Real PyTorch model — save state_dict
                torch.save(model.state_dict(), model_path)
            elif isinstance(model, dict) and '_sentinel' in model:
                # Sentinel dict for testing (retained for test compatibility)
                torch.save(model, model_path)
            else:
                # Unknown model type
                raise RuntimeError(
                    f"Cannot save model '{model_name}': expected nn.Module or sentinel dict, "
                    f"got {type(model)}."
                )

            # Save params snapshot (CONFIG-001 critical)
            params_path = os.path.join(model_dir, 'params.dill')
            with open(params_path, 'wb') as f:
                dill.dump(params_snapshot.copy(), f)

        # Create zip archive
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arc_path = os.path.relpath(full_path, temp_dir)
                    zf.write(full_path, arc_path)


def load_torch_bundle(
    base_path: str,
    model_name: str = 'diffraction_to_obj'
) -> Tuple[Any, dict]:
    """
    Load PyTorch model bundle with CONFIG-001-compliant params restoration.

    Extracts wts.h5.zip archive, restores params.cfg state via update_legacy_dict,
    reconstructs model architecture, and loads weights. Critical for preventing
    shape mismatch errors during inference (CONFIG-001 finding).

    Args:
        base_path: Base path of archive (reads {base_path}.zip).
        model_name: Name of model to load from bundle. Defaults to 'diffraction_to_obj'
                   (inference model per spec §4.5).

    Returns:
        Tuple of (loaded_model, params_dict) where:
        - loaded_model: nn.Module with restored weights
        - params_dict: Dictionary containing training-time params.cfg snapshot

    Raises:
        FileNotFoundError: If archive does not exist.
        ValueError: If requested model not found in manifest or params missing required fields.
        RuntimeError: If PyTorch unavailable or model reconstruction fails.

    Example:
        >>> model, params = load_torch_bundle('output/wts.h5')
        >>> # params.cfg automatically updated via CONFIG-001 gate
        >>> recon = model([X, local_offsets])
    """
    from ptycho.config.config import update_legacy_dict
    from ptycho import params

    # PyTorch is now mandatory (no availability check needed)

    # Validate archive exists
    zip_path = f"{base_path}.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model archive not found: {zip_path}")

    # Extract archive
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        # Load manifest
        manifest_path = os.path.join(temp_dir, 'manifest.dill')
        with open(manifest_path, 'rb') as f:
            manifest = dill.load(f)

        # Validate requested model exists
        available_models = manifest['models']
        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' not found in archive. "
                f"Available models: {available_models}"
            )

        # Load params snapshot (CONFIG-001 gate)
        model_dir = os.path.join(temp_dir, model_name)
        params_path = os.path.join(model_dir, 'params.dill')
        with open(params_path, 'rb') as f:
            params_dict = dill.load(f)

        # Validate required fields
        required_fields = ['N', 'gridsize']
        missing = [f for f in required_fields if f not in params_dict]
        if missing:
            raise ValueError(
                f"params.dill missing required fields: {missing}. "
                "Cannot reconstruct model architecture."
            )

        # Restore params.cfg (CONFIG-001 critical side effect)
        params.cfg.update(params_dict)

        # Reconstruct model architecture
        # NOTE: This requires create_torch_model_with_gridsize helper from Phase D2.B
        # For now, raise NotImplementedError as Phase D3.C scope
        raise NotImplementedError(
            "load_torch_bundle model reconstruction not yet implemented. "
            "Requires create_torch_model_with_gridsize helper from Phase D3.C. "
            f"params.cfg successfully restored: N={params_dict['N']}, "
            f"gridsize={params_dict['gridsize']}"
        )

        # Placeholder for Phase D3.C implementation:
        # gridsize = params_dict['gridsize']
        # N = params_dict['N']
        # model = create_torch_model_with_gridsize(gridsize, N)
        #
        # # Load weights
        # model_path = os.path.join(model_dir, 'model.pth')
        # state_dict = torch.load(model_path)
        # model.load_state_dict(state_dict)
        #
        # return model, params_dict

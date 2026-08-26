"""Strict-path bundle persistence: intensity scale and legacy profile round-trip.

The metadata-free legacy restore path was deleted; these bundles now recover
through ``scripts/migrate_legacy_bundle.py``, which reconstructs the legacy
model, seals a versioned identity, and promotes the manifest to JSON.
"""

import io
import zipfile
from pathlib import Path

import dill
import torch

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.config.config import dataclass_to_legacy_dict
from ptycho_torch.scaling_contract import (
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)
from ptycho_torch.workflows.components import load_inference_bundle_torch
from scripts.migrate_legacy_bundle import (
    create_torch_model_with_gridsize,
    migrate_bundle,
)


def _write_metadata_free_bundle(tmp_path: Path, archived: dict, model) -> Path:
    """Emit the deleted writer's dill shape: manifest + params.dill, no identity."""
    archived = dict(archived)
    archived["_version"] = "2.0-pytorch"
    source = tmp_path / "legacy"
    source.mkdir()
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


def test_bundle_persists_intensity_scale(tmp_path: Path):
    config = TrainingConfig(model=ModelConfig(N=64, gridsize=1), nphotons=1e9)
    archived = dataclass_to_legacy_dict(config)
    archived["intensity_scale"] = 123.0
    model = create_torch_model_with_gridsize(1, 64, archived)
    source = _write_metadata_free_bundle(tmp_path, archived, model)

    migrate_bundle(source, tmp_path / "migrated")

    models_loaded, params = load_inference_bundle_torch(tmp_path / "migrated")
    assert params["intensity_scale"] == 123.0
    assert models_loaded["diffraction_to_obj"].model_config.intensity_scale == 123.0


def test_metadata_free_legacy_bundle_migrates_to_legacy_profile(tmp_path: Path):
    config = TrainingConfig(model=ModelConfig(N=64, gridsize=1), nphotons=1e9)
    archived = dataclass_to_legacy_dict(config)
    archived["intensity_scale"] = 1.0
    model = create_torch_model_with_gridsize(1, 64, archived)
    source = _write_metadata_free_bundle(tmp_path, archived, model)

    migrate_bundle(source, tmp_path / "migrated")

    models, params = load_inference_bundle_torch(tmp_path / "migrated")

    assert params["scale_contract_version"] == LEGACY_SCALE_CONTRACT
    assert params["measurement_domain"] == NORMALIZED_AMPLITUDE
    for loaded_model in models.values():
        assert loaded_model.data_config.scale_contract_version == LEGACY_SCALE_CONTRACT
        assert loaded_model.data_config.measurement_domain == NORMALIZED_AMPLITUDE

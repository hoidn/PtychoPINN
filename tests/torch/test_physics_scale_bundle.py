from pathlib import Path

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.config.config import dataclass_to_legacy_dict
from ptycho import params as legacy_params
from ptycho_torch.model_manager import (
    create_torch_model_with_gridsize,
    load_torch_bundle,
    save_torch_bundle,
)
from ptycho_torch.scaling_contract import (
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)
from ptycho_torch.workflows.components import load_inference_bundle_torch


def test_bundle_persists_intensity_scale(tmp_path: Path):
    config = TrainingConfig(model=ModelConfig(N=64, gridsize=1), nphotons=1e9)
    archived = dataclass_to_legacy_dict(config)
    archived["intensity_scale"] = 123.0
    model = create_torch_model_with_gridsize(1, 64, archived)
    models = {"autoencoder": model, "diffraction_to_obj": model}
    base_path = tmp_path / "wts.h5"

    save_torch_bundle(models, str(base_path), config, intensity_scale=123.0)

    models_loaded, params = load_torch_bundle(str(base_path))
    assert params["intensity_scale"] == 123.0
    assert models_loaded["diffraction_to_obj"].model_config.intensity_scale == 123.0


def test_metadata_free_legacy_bundle_loads_with_explicit_profile(tmp_path: Path):
    config = TrainingConfig(model=ModelConfig(N=64, gridsize=1), nphotons=1e9)
    archived = dataclass_to_legacy_dict(config)
    archived["intensity_scale"] = 1.0
    model = create_torch_model_with_gridsize(1, 64, archived)
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(tmp_path / "wts.h5"),
        config,
        intensity_scale=1.0,
    )

    legacy_params.cfg.clear()
    models, params = load_inference_bundle_torch(
        tmp_path,
        scale_contract_version=LEGACY_SCALE_CONTRACT,
        measurement_domain=NORMALIZED_AMPLITUDE,
    )

    assert params["scale_contract_version"] == LEGACY_SCALE_CONTRACT
    assert params["measurement_domain"] == NORMALIZED_AMPLITUDE
    assert legacy_params.cfg["scale_contract_version"] == LEGACY_SCALE_CONTRACT
    for loaded_model in models.values():
        assert loaded_model.data_config.scale_contract_version == LEGACY_SCALE_CONTRACT
        assert loaded_model.data_config.measurement_domain == NORMALIZED_AMPLITUDE

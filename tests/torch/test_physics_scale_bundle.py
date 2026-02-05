from pathlib import Path

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle


def test_bundle_persists_intensity_scale(tmp_path: Path):
    config = TrainingConfig(model=ModelConfig(N=64, gridsize=1), nphotons=1e9)
    models = {
        "autoencoder": {"_sentinel": True},
        "diffraction_to_obj": {"_sentinel": True},
    }
    base_path = tmp_path / "wts.h5"

    save_torch_bundle(models, str(base_path), config, intensity_scale=123.0)

    models_loaded, params = load_torch_bundle(str(base_path))
    assert params["intensity_scale"] == 123.0
    assert models_loaded["diffraction_to_obj"].model_config.intensity_scale == 123.0

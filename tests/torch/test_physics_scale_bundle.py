from pathlib import Path

import pytest

from ptycho.config.config import TrainingConfig, ModelConfig, DataConfig
from ptycho_torch.config_params import (
    DataConfig as PTDataConfig,
    InferenceConfig as PTInferenceConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
)
from ptycho_torch.model_manager import save_torch_bundle
from ptycho_torch.workflows.bundle_io import _persist_bundle_scaling_metadata
from ptycho_torch.workflows.components import load_inference_bundle_torch


def _tiny_model(intensity_scale: float):
    from ptycho_torch.model import PtychoPINN_Lightning

    return PtychoPINN_Lightning(
        PTModelConfig(
            C_model=1,
            C_forward=1,
            object_big=False,
            probe_big=False,
            n_filters_scale=1,
            physics_forward_mode="rectangular_scaled",
            cnn_output_mode="real_imag",
            rect_s1s2_trainable=False,
            intensity_scale=intensity_scale,
        ),
        PTDataConfig(N=64, C=1, grid_size=(1, 1)),
        PTTrainingConfig(device="cpu", torch_loss_mode="poisson"),
        PTInferenceConfig(),
    )


def _save_sealed_bundle(tmp_path: Path, model, intensity_scale: float):
    import torch

    model.register_ci_statistics(
        {
            "rms_input_scale": torch.tensor([0.375]),
            "mean_measured_intensity": torch.tensor([9.0]),
        }
    )
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1), data=DataConfig(nphotons=1e9)
    )
    base_path = tmp_path / "wts.h5"
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(base_path),
        config,
        intensity_scale=intensity_scale,
    )
    _persist_bundle_scaling_metadata(base_path.with_suffix(".h5.zip"), model)


def test_bundle_persists_intensity_scale(tmp_path: Path):
    model = _tiny_model(intensity_scale=123.0)
    _save_sealed_bundle(tmp_path, model, intensity_scale=123.0)

    models_loaded, params = load_inference_bundle_torch(tmp_path)

    assert params["intensity_scale"] == 123.0
    assert (
        models_loaded["diffraction_to_obj"].model_config.intensity_scale == 123.0
    )


def test_unsealed_bundle_fails_loudly_naming_migration(tmp_path: Path):
    """The retired metadata-free restore is gone: unsealed bundles raise."""
    model = _tiny_model(intensity_scale=1.0)
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1), data=DataConfig(nphotons=1e9)
    )
    save_torch_bundle(
        {"autoencoder": model, "diffraction_to_obj": model},
        str(tmp_path / "wts.h5"),
        config,
        intensity_scale=1.0,
    )

    with pytest.raises(ValueError, match="migrate_legacy_bundle"):
        load_inference_bundle_torch(tmp_path)

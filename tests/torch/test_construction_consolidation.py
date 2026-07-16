import pytest

from ptycho.config.config import ModelConfig as CanonicalModelConfig
from ptycho.config.config import TrainingConfig as CanonicalTrainingConfig
from ptycho_torch.application_factory import build_ptychopinn_application
from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.generators.registry import _REGISTRY, resolve_generator
from ptycho_torch.model_spec import derive_model_spec


@pytest.mark.parametrize("architecture", sorted(_REGISTRY))
def test_registry_and_model_spec_construction_have_one_state_signature(architecture):
    image_size = 128 if architecture == "neuralop_uno" else 64
    data = DataConfig(N=image_size, C=1, grid_size=(1, 1), probe_scale=4.0)
    model = ModelConfig(
        architecture=architecture,
        C_model=1,
        C_forward=1,
        object_big=False,
        probe_big=False,
        n_filters_scale=1,
        fno_width=4,
        fno_modes=2,
        fno_blocks=3,
        fno_cnn_blocks=1,
    )
    training = TrainingConfig(device="cpu", torch_loss_mode="poisson")
    inference = InferenceConfig()
    canonical_model = to_model_config(data, model)
    canonical_training = CanonicalTrainingConfig(
        model=CanonicalModelConfig(
            N=image_size,
            gridsize=1,
            architecture=architecture,
        )
    )
    pt_configs = {
        "data_config": data,
        "model_config": model,
        "training_config": training,
        "inference_config": inference,
    }

    registry_model = resolve_generator(canonical_training).build_model(pt_configs)
    spec_model = build_ptychopinn_application(
        derive_model_spec(canonical_model, model, data),
        data,
        training,
        inference,
    )

    registry_signature = {
        name: tuple(value.shape) for name, value in registry_model.state_dict().items()
    }
    spec_signature = {
        name: tuple(value.shape) for name, value in spec_model.state_dict().items()
    }
    assert registry_signature == spec_signature


@pytest.mark.parametrize("architecture", sorted(_REGISTRY))
def test_registry_wrappers_delegate_to_the_single_application_factory(
    architecture,
    monkeypatch,
):
    sentinel = object()
    captured = []

    def fake_builder(pt_configs):
        captured.append(pt_configs)
        return sentinel

    monkeypatch.setattr(
        "ptycho_torch.application_factory.build_ptychopinn_from_configs",
        fake_builder,
        raising=False,
    )
    canonical = CanonicalTrainingConfig(
        model=CanonicalModelConfig(architecture=architecture)
    )
    pt_configs = {"identity": architecture}

    result = resolve_generator(canonical).build_model(pt_configs)

    assert result is sentinel
    assert captured == [pt_configs]

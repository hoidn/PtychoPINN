# tests/torch/test_generator_registry.py
import pytest
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho_torch.generators.registry import resolve_generator


def test_resolve_generator_cnn():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    gen = resolve_generator(cfg)
    assert gen.name == 'cnn'


def test_resolve_generator_unknown_raises():
    cfg = TrainingConfig(model=ModelConfig(architecture='unknown'))
    with pytest.raises(ValueError):
        resolve_generator(cfg)


def test_resolve_generator_fno_vanilla():
    cfg = TrainingConfig(model=ModelConfig(architecture='fno_vanilla'))
    gen = resolve_generator(cfg)
    assert gen.name == 'fno_vanilla'


def test_resolve_generator_hybrid_resnet():
    cfg = TrainingConfig(model=ModelConfig(architecture='hybrid_resnet'))
    gen = resolve_generator(cfg)
    assert gen.name == 'hybrid_resnet'


def test_resolve_generator_ffno():
    cfg = TrainingConfig(model=ModelConfig(architecture='ffno'))
    gen = resolve_generator(cfg)
    assert gen.name == 'ffno'


def test_ffno_generator_builds_lightning_model():
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig as PTInferenceConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )
    from ptycho_torch.model import PtychoPINN_Lightning

    cfg = TrainingConfig(model=ModelConfig(architecture='ffno', N=64, gridsize=1))
    gen = resolve_generator(cfg)

    pt_configs = {
        "data_config": DataConfig(N=64, C=1),
        "model_config": PTModelConfig(
            architecture='ffno',
            fno_width=32,
            fno_blocks=4,
            fno_cnn_blocks=2,
        ),
        "training_config": PTTrainingConfig(),
        "inference_config": PTInferenceConfig(),
    }

    model = gen.build_model(pt_configs)
    assert isinstance(model, PtychoPINN_Lightning)
    assert model.model.generator.__class__.__name__ == "FfnoGeneratorModule"

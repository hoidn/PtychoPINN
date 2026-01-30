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

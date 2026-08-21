# tests/test_generator_registry.py
import pytest
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.generators.registry import resolve_generator


def test_resolve_generator_cnn():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    gen = resolve_generator(cfg)
    assert gen.name == 'cnn'


def test_training_config_rejects_unknown_generator_architecture():
    with pytest.raises(ValueError, match="architecture"):
        TrainingConfig(model=ModelConfig(architecture='unknown'))

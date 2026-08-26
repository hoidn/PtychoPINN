# tests/test_model_config_architecture.py
import pytest
from ptycho.config.config import ModelConfig, validate_model_config


def test_model_config_architecture_default_ok():
    cfg = ModelConfig()
    validate_model_config(cfg)


def test_model_config_architecture_invalid_raises():
    cfg = ModelConfig(architecture="not-a-real-arch")
    with pytest.raises(ValueError):
        validate_model_config(cfg)

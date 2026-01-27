"""
Generator registry for TensorFlow PINN architectures.

This module provides a central registry for generator implementations,
enabling architecture selection via configuration.

Usage:
    from ptycho.config.config import TrainingConfig, ModelConfig
    from ptycho.generators.registry import resolve_generator

    config = TrainingConfig(model=ModelConfig(architecture='cnn'))
    generator = resolve_generator(config)
    model, diffraction_to_obj = generator.build_models()
"""
from ptycho.generators.cnn import CnnGenerator

_REGISTRY = {
    'cnn': CnnGenerator,
}


def resolve_generator(config):
    """
    Resolve the generator class for the given configuration.

    Args:
        config: TrainingConfig or InferenceConfig with model.architecture field

    Returns:
        Generator instance configured for the specified architecture

    Raises:
        ValueError: If architecture is not registered
    """
    arch = config.model.architecture
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[arch](config)

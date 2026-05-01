"""
Generator registry for PyTorch PINN architectures.

This module provides a central registry for generator implementations,
enabling architecture selection via configuration.

Usage:
    from ptycho.config.config import TrainingConfig, ModelConfig
    from ptycho_torch.generators.registry import resolve_generator

    config = TrainingConfig(model=ModelConfig(architecture='cnn'))
    generator = resolve_generator(config)
    model = generator.build_model(pt_configs)

Supported architectures:
    - 'cnn': CNN-based U-Net generator (default)
    - 'fno': Cascaded FNO → CNN generator (Arch A)
    - 'hybrid': Hybrid U-NO generator (Arch B)
"""
from ptycho_torch.generators.cnn import CnnGenerator
from ptycho_torch.generators.ffno import FfnoGenerator
from ptycho_torch.generators.fno import FnoGenerator, HybridGenerator, StableHybridGenerator
from ptycho_torch.generators.fno_vanilla import FnoVanillaGenerator
from ptycho_torch.generators.hybrid_resnet_ffno_bottleneck import HybridResnetFfnoBottleneckGenerator
from ptycho_torch.generators.hybrid_resnet import HybridResnetGenerator
from ptycho_torch.generators.neuralop_uno import NeuralopUnoGenerator
from ptycho_torch.generators.spectral_resnet_bottleneck_linear_decoder import (
    SpectralResnetBottleneckLinearDecoderGenerator,
)
from ptycho_torch.generators.spectral_resnet_bottleneck import SpectralResnetBottleneckGenerator

_REGISTRY = {
    'cnn': CnnGenerator,
    'ffno': FfnoGenerator,
    'fno': FnoGenerator,
    'hybrid': HybridGenerator,
    'stable_hybrid': StableHybridGenerator,
    'fno_vanilla': FnoVanillaGenerator,
    'neuralop_uno': NeuralopUnoGenerator,
    'hybrid_resnet': HybridResnetGenerator,
    'spectral_resnet_bottleneck_net': SpectralResnetBottleneckGenerator,
    'spectral_resnet_bottleneck_linear_decoder': SpectralResnetBottleneckLinearDecoderGenerator,
    'hybrid_resnet_ffno_bottleneck': HybridResnetFfnoBottleneckGenerator,
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

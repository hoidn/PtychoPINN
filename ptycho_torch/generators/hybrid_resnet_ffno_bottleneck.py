"""Hybrid ResNet shell with an FFNO-close bottleneck replacement."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from ptycho_torch.generators.ffno_bottleneck import SharedFactorizedFfnoBottleneck
from ptycho_torch.generators.hybrid_resnet import HybridResnetGeneratorModule


class HybridResnetFfnoBottleneckGeneratorModule(HybridResnetGeneratorModule):
    """Hybrid ResNet shell that swaps the local ResNet bottleneck for FFNO-close blocks."""

    def __init__(
        self,
        *,
        ffno_bottleneck_blocks: int = 6,
        ffno_bottleneck_modes: int = 12,
        ffno_bottleneck_share_spectral_weights: bool = True,
        ffno_bottleneck_mlp_ratio: float = 2.0,
        ffno_bottleneck_gate_init: float = 0.1,
        ffno_bottleneck_norm: str = "instance",
        ffno_bottleneck_local_conv_kernel_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resnet = SharedFactorizedFfnoBottleneck(
            int(self.bottleneck_channels),
            n_blocks=ffno_bottleneck_blocks,
            modes=ffno_bottleneck_modes,
            share_spectral_weights=ffno_bottleneck_share_spectral_weights,
            mlp_ratio=ffno_bottleneck_mlp_ratio,
            gate_init=ffno_bottleneck_gate_init,
            norm=ffno_bottleneck_norm,
            local_conv_kernel_size=ffno_bottleneck_local_conv_kernel_size,
        )


class HybridResnetFfnoBottleneckGenerator:
    """Generator registry wrapper for hybrid_resnet_ffno_bottleneck."""

    name = "hybrid_resnet_ffno_bottleneck"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> "nn.Module":
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)

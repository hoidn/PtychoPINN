"""Spectral ResNet bottleneck shell with a lighter bilinear + 1x1 decoder."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ptycho_torch.generators.spectral_resnet_bottleneck import (
    SpectralResnetBottleneckGeneratorModule,
)


class BilinearProjectionUpsampler(nn.Module):
    """Upsample by 2x with bilinear interpolation followed by a 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class SpectralResnetBottleneckLinearDecoderGeneratorModule(
    SpectralResnetBottleneckGeneratorModule
):
    """Spectral ResNet bottleneck shell that replaces CycleGAN upsamplers with bilinear + 1x1 stages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upsample_layers = nn.ModuleList(
            [
                BilinearProjectionUpsampler(self.decoder_widths[index], self.decoder_widths[index + 1])
                for index in range(self.hybrid_downsample_steps)
            ]
        )


class SpectralResnetBottleneckLinearDecoderGenerator:
    """Generator registry wrapper for spectral_resnet_bottleneck_linear_decoder."""

    name = "spectral_resnet_bottleneck_linear_decoder"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> "nn.Module":
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)

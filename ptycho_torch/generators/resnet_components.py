"""CycleGAN-style ResNet components used by hybrid_resnet generator."""
from typing import Optional

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """CycleGAN-style ResNet block with GELU activations and a residual gate."""

    def __init__(
        self,
        channels: int,
        layerscale_init: float = 0.1,
        shared_layerscale: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
        )
        self._shared_layerscale_ref: Optional[list[nn.Parameter]] = None
        if shared_layerscale is None:
            self._layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        else:
            self.register_parameter("_layerscale", None)
            self._shared_layerscale_ref = [shared_layerscale]

    @property
    def layerscale(self) -> nn.Parameter:
        if self._shared_layerscale_ref is not None:
            return self._shared_layerscale_ref[0]
        return self._layerscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layerscale * self.block(x)


class ResnetBottleneck(nn.Module):
    """Stack of ResNet blocks at constant resolution with one shared residual gate."""

    def __init__(self, channels: int, n_blocks: int = 6, layerscale_init: float = 0.1):
        super().__init__()
        self.layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        self.blocks = nn.Sequential(
            *[
                ResnetBlock(channels, shared_layerscale=self.layerscale)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CycleGanUpsampler(nn.Module):
    """CycleGAN upsampling block (ConvTranspose2d + InstanceNorm + GELU)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

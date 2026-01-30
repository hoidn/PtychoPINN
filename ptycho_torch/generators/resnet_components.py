"""CycleGAN-style ResNet components used by hybrid_resnet generator."""
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """CycleGAN-style ResNet block with reflection padding + InstanceNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetBottleneck(nn.Module):
    """Stack of ResNet blocks at constant resolution."""

    def __init__(self, channels: int, n_blocks: int = 6):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResnetBlock(channels) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CycleGanUpsampler(nn.Module):
    """CycleGAN upsampling block (ConvTranspose2d + InstanceNorm + ReLU)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

from __future__ import annotations

import torch
import torch.nn as nn


class SharedMeasurementEncoder(nn.Module):
    """Shared anisotropic encoder that lifts one measurement image to a latent field."""

    def __init__(self, latent_channels: int, base_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=(5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
        )
        self.time_downsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=(2, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.project = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.time_downsample(x)
        x = torch.nn.functional.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        return self.project(x)


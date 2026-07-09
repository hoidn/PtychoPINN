"""Shared spectral primitives independent of the resnet generator family."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FactorizedSpectralConv2d(nn.Module):
    """Shape-preserving factorized spectral operator using separate axis FFTs."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if self.modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}.")
        scale = 1.0 / math.sqrt(self.channels)
        self.weights_x = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, self.modes, dtype=torch.cfloat)
        )
        self.weights_y = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        modes_h = min(self.modes, height // 2 + 1)
        modes_w = min(self.modes, width // 2 + 1)

        x_ft_w = torch.fft.rfft(x, dim=-1)
        out_ft_w = torch.zeros(
            batch,
            self.channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft_w[:, :, :, :modes_w] = torch.einsum(
            "bihm,iom->bohm",
            x_ft_w[:, :, :, :modes_w],
            self.weights_x[:, :, :modes_w],
        )
        out_w = torch.fft.irfft(out_ft_w, n=width, dim=-1)

        x_ft_h = torch.fft.rfft(x, dim=-2)
        out_ft_h = torch.zeros(
            batch,
            self.channels,
            height // 2 + 1,
            width,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft_h[:, :, :modes_h, :] = torch.einsum(
            "bimw,iom->bomw",
            x_ft_h[:, :, :modes_h, :],
            self.weights_y[:, :, :modes_h],
        )
        out_h = torch.fft.irfft(out_ft_h, n=height, dim=-2)
        return out_w + out_h

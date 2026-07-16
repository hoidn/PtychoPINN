"""CDI-compatible FFNO generator for the PyTorch PINN path."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from ptycho_torch.generators.ffno_bottleneck import build_no_refiner_ffno_stack
from ptycho_torch.generators.fno import SpatialLifter


class _LocalResidualRefiner(nn.Module):
    """Small residual local-conv refiner used after the FFNO stack."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class FfnoGeneratorModule(nn.Module):
    """Constant-resolution FFNO stack that preserves the CDI real/imag contract."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 2,
        hidden_channels: int = 32,
        n_blocks: int = 4,
        modes: int = 12,
        cnn_blocks: int = 2,
        C: int = 4,
        input_transform: str = "none",
        output_mode: str = "real_imag",
    ):
        super().__init__()
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}.")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}.")
        if cnn_blocks < 0:
            raise ValueError(f"cnn_blocks must be >= 0, got {cnn_blocks}.")

        self.C = int(C)
        self.output_mode = output_mode
        self.lifter = SpatialLifter(
            in_channels * self.C,
            hidden_channels,
            input_transform=input_transform,
        )
        self.ffno_stack = build_no_refiner_ffno_stack(
            hidden_channels,
            n_blocks=n_blocks,
            modes=modes,
            share_spectral_weights=True,
            mlp_ratio=2.0,
            gate_init=0.1,
            norm="instance",
        )
        self.blocks = self.ffno_stack.blocks
        self.refiners = nn.ModuleList(
            [_LocalResidualRefiner(hidden_channels) for _ in range(int(cnn_blocks))]
        )
        if self.output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(hidden_channels, self.C, kernel_size=1)
            self.output_phase = nn.Conv2d(hidden_channels, self.C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(
                hidden_channels,
                out_channels * self.C,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        x = self.lifter(x)
        x = self.ffno_stack(x)
        for block in self.refiners:
            x = block(x)
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase
        x = self.output_proj(x)
        x = x.view(batch, 2, self.C, height, width)
        x = x.permute(0, 3, 4, 2, 1)
        return x


class FfnoGenerator:
    """Generator-registry wrapper for the CDI FFNO path."""

    name = "ffno"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> nn.Module:
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)

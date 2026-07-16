"""Constant-resolution FNO generator (baseline)."""
import math
from typing import Dict, Any

import torch
import torch.nn as nn

from ptycho_torch.generators.fno import SpatialLifter, PtychoBlock


class FnoVanillaGeneratorModule(nn.Module):
    """Constant-resolution FNO stack with 1x1 output projection."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        hidden_channels: int = 32,
        n_blocks: int = 4,
        modes: int = 12,
        C: int = 4,
        input_transform: str = "none",
        output_mode: str = "real_imag",
    ):
        super().__init__()
        self.C = C
        self.output_mode = output_mode
        self.lifter = SpatialLifter(
            in_channels * C,
            hidden_channels,
            input_transform=input_transform,
        )
        self.blocks = nn.ModuleList(
            [PtychoBlock(hidden_channels, modes=modes) for _ in range(n_blocks)]
        )
        if self.output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(hidden_channels, C, kernel_size=1)
            self.output_phase = nn.Conv2d(hidden_channels, C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(hidden_channels, out_channels * C, kernel_size=1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.lifter(x)
        for block in self.blocks:
            x = block(x)
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase
        x = self.output_proj(x)
        x = x.view(B, 2, self.C, H, W)
        x = x.permute(0, 3, 4, 2, 1)
        return x


class FnoVanillaGenerator:
    """Generator registry wrapper for constant-resolution FNO baseline."""

    name = "fno_vanilla"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> "nn.Module":
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)

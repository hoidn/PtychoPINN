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
        """Build the Lightning module for training."""
        from ptycho_torch.model import PtychoPINN_Lightning

        data_config = pt_configs["data_config"]
        model_config = pt_configs["model_config"]
        training_config = pt_configs["training_config"]
        inference_config = pt_configs["inference_config"]

        C = getattr(data_config, "C", 4)
        fno_width = getattr(model_config, "fno_width", 32)
        fno_blocks = getattr(model_config, "fno_blocks", 4)
        fno_modes = getattr(model_config, "fno_modes", 12)
        input_transform = getattr(model_config, "fno_input_transform", "none")
        output_mode = getattr(model_config, "generator_output_mode", "real_imag")
        generator_mode = "amp_phase" if output_mode == "amp_phase" else "real_imag"

        core = FnoVanillaGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=fno_width,
            n_blocks=fno_blocks,
            modes=fno_modes,
            C=C,
            input_transform=input_transform,
            output_mode=generator_mode,
        )

        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )

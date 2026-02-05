"""Hybrid ResNet-6 generator (FNO encoder + CycleGAN decoder)."""
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from ptycho_torch.generators.fno import SpatialLifter, PtychoBlock
from ptycho_torch.generators.resnet_components import ResnetBottleneck, CycleGanUpsampler


class HybridResnetGeneratorModule(nn.Module):
    """FNO encoder -> ResNet-6 bottleneck -> CycleGAN upsamplers."""

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
        max_hidden_channels: Optional[int] = None,
        resnet_blocks: int = 6,
        resnet_width: Optional[int] = None,
    ):
        super().__init__()
        if n_blocks < 3:
            raise ValueError(
                "hybrid_resnet requires fno_blocks >= 3 for two downsample steps "
                f"(got fno_blocks={n_blocks})."
            )
        self.C = C
        self.output_mode = output_mode

        # Lifter
        self.lifter = SpatialLifter(
            in_channels * C,
            hidden_channels,
            input_transform=input_transform,
        )

        # Encoder: downsample twice (N -> N/4), remaining blocks stay at N/4
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = hidden_channels
        n_downsample = 2
        for i in range(n_blocks):
            self.encoder_blocks.append(PtychoBlock(ch, modes=modes))
            if i < n_downsample:
                next_ch = ch * 2
                if max_hidden_channels is not None:
                    next_ch = min(next_ch, max_hidden_channels)
                self.downsample.append(nn.Conv2d(ch, next_ch, kernel_size=2, stride=2))
                ch = next_ch

        # Adapter for ResNet bottleneck width
        if resnet_width is not None:
            if resnet_width <= 0:
                raise ValueError(
                    f"resnet_width must be positive when set, got {resnet_width}."
                )
            if resnet_width % 4 != 0:
                raise ValueError(
                    "resnet_width must be divisible by 4 so the CycleGAN upsamplers "
                    f"produce integer channel sizes (got {resnet_width})."
                )
        target_width = ch if resnet_width is None else resnet_width
        self.adapter = nn.Identity()
        if ch != target_width:
            self.adapter = nn.Conv2d(ch, target_width, kernel_size=1)

        self.resnet = ResnetBottleneck(target_width, n_blocks=resnet_blocks)

        # CycleGAN upsamplers (N/4 -> N)
        self.up1 = CycleGanUpsampler(target_width, target_width // 2)
        self.up2 = CycleGanUpsampler(target_width // 2, target_width // 4)
        out_ch = target_width // 4

        if self.output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(out_ch, C, kernel_size=1)
            self.output_phase = nn.Conv2d(out_ch, C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(out_ch, out_channels * C, kernel_size=1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.lifter(x)
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        x = self.adapter(x)
        x = self.resnet(x)
        x = self.up1(x)
        x = self.up2(x)
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase
        x = self.output_proj(x)
        x = x.view(B, 2, self.C, H, W).permute(0, 3, 4, 2, 1)
        return x


class HybridResnetGenerator:
    """Generator registry wrapper for hybrid_resnet architecture."""

    name = "hybrid_resnet"

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
        max_hidden_channels = getattr(model_config, "max_hidden_channels", None)

        resnet_width = getattr(model_config, "resnet_width", None)
        core = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=fno_width,
            n_blocks=fno_blocks,
            modes=fno_modes,
            C=C,
            input_transform=input_transform,
            output_mode=generator_mode,
            max_hidden_channels=max_hidden_channels,
            resnet_width=resnet_width,
        )

        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )

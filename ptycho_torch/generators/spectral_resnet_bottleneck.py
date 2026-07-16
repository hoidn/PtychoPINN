"""Spectral ResNet bottleneck generator family."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import (
    AvgPoolConvDownsample,
    BlurPoolConvDownsample,
    HybridResnetEncoderBlock,
    StrideConvDownsample,
)
from ptycho_torch.generators.resnet_components import CycleGanUpsampler
from ptycho_torch.generators.spectral_layers import FactorizedSpectralConv2d


def build_resnet_local_conv_body(channels: int) -> nn.Sequential:
    """Raw two-conv local body matching ResnetBlock internals without nested residual."""

    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(channels, channels, kernel_size=3, padding=0),
        nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
        nn.GELU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(channels, channels, kernel_size=3, padding=0),
        nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
    )


class SpectralResnetBlock(nn.Module):
    """ResNet local body plus shared spectral residual branch with one outer residual."""

    def __init__(
        self,
        *,
        channels: int,
        shared_spectral: nn.Module,
        local_scale: Optional[nn.Parameter] = None,
        spectral_gate: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.local_conv_body = build_resnet_local_conv_body(channels)
        self.shared_spectral = shared_spectral
        self._shared_local_scale_ref: Optional[list[nn.Parameter]] = None
        self._shared_spectral_gate_ref: Optional[list[nn.Parameter]] = None
        if local_scale is None:
            self.local_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("local_scale", None)
            self._shared_local_scale_ref = [local_scale]
        if spectral_gate is None:
            self.spectral_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("spectral_gate", None)
            self._shared_spectral_gate_ref = [spectral_gate]

    @property
    def effective_local_scale(self) -> nn.Parameter:
        if self._shared_local_scale_ref is not None:
            return self._shared_local_scale_ref[0]
        return self.local_scale

    @property
    def effective_spectral_gate(self) -> nn.Parameter:
        if self._shared_spectral_gate_ref is not None:
            return self._shared_spectral_gate_ref[0]
        return self.spectral_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            + self.effective_local_scale * self.local_conv_body(x)
            + self.effective_spectral_gate * self.shared_spectral(x)
        )


class SharedSpectralResnetBottleneck(nn.Module):
    """Stack of local-residual blocks with a shared factorized spectral operator."""

    def __init__(
        self,
        channels: int,
        *,
        n_blocks: int = 6,
        modes: int = 12,
        share_spectral_weights: bool = True,
        gate_init: float = 0.1,
        gate_mode: str = "shared",
        layerscale_init: float = 0.1,
    ):
        super().__init__()
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}.")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}.")
        if gate_mode not in {"shared", "per_block"}:
            raise ValueError(
                f"gate_mode must be one of ['per_block', 'shared'], got {gate_mode!r}."
            )

        self.layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        self.shared_spectral = FactorizedSpectralConv2d(channels=channels, modes=modes)
        self.blocks = nn.ModuleList()
        self.spectral_gates = nn.ParameterList()
        shared_gate = nn.Parameter(torch.tensor(float(gate_init))) if gate_mode == "shared" else None
        if shared_gate is not None:
            self.spectral_gates.append(shared_gate)
        for _ in range(n_blocks):
            shared_spectral = self.shared_spectral if share_spectral_weights else FactorizedSpectralConv2d(
                channels=channels,
                modes=modes,
            )
            gate = shared_gate
            if gate_mode == "per_block":
                gate = nn.Parameter(torch.tensor(float(gate_init)))
                self.spectral_gates.append(gate)
            self.blocks.append(
                SpectralResnetBlock(
                    channels=channels,
                    shared_spectral=shared_spectral,
                    local_scale=self.layerscale,
                    spectral_gate=gate,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SpectralResnetBottleneckGeneratorModule(nn.Module):
    """Hybrid ResNet shell with a shared spectral ResNet bottleneck."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 2,
        hidden_channels: int = 32,
        n_blocks: int = 4,
        modes: int = 12,
        C: int = 4,
        input_transform: str = "none",
        output_mode: str = "real_imag",
        max_hidden_channels: Optional[int] = None,
        resnet_width: Optional[int] = None,
        resnet_blocks: int = 6,
        hybrid_downsample_steps: int = 2,
        hybrid_downsample_op: str = "stride_conv",
        hybrid_encoder_conv_hidden_scale: float = 1.0,
        hybrid_encoder_spectral_hidden_scale: float = 1.0,
        hybrid_encoder_conv_hidden_channels: Optional[int] = None,
        hybrid_encoder_spectral_hidden_channels: Optional[int] = None,
        spectral_bottleneck_blocks: int = 6,
        spectral_bottleneck_modes: int = 12,
        spectral_bottleneck_share_weights: bool = True,
        spectral_bottleneck_gate_init: float = 0.1,
        spectral_bottleneck_gate_mode: str = "shared",
    ):
        super().__init__()
        if hybrid_downsample_steps not in (1, 2):
            raise ValueError(
                f"hybrid_downsample_steps must be in [1, 2] for current implementation (got {hybrid_downsample_steps})."
            )
        if hybrid_downsample_op not in {"stride_conv", "avgpool_conv", "blurpool_conv"}:
            raise ValueError(f"Unsupported downsample op: {hybrid_downsample_op!r}")
        if resnet_blocks <= 0:
            raise ValueError(f"resnet_blocks must be positive, got {resnet_blocks}.")
        if n_blocks < 3:
            raise ValueError(
                "spectral_resnet_bottleneck_net requires fno_blocks >= 3 "
                f"(got fno_blocks={n_blocks})."
            )
        self.C = int(C)
        self.output_mode = output_mode
        self.hybrid_downsample_steps = int(hybrid_downsample_steps)

        self.lifter = SpatialLifter(in_channels * self.C, hidden_channels, input_transform=input_transform)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = hidden_channels
        for index in range(n_blocks):
            conv_hidden = hybrid_encoder_conv_hidden_channels
            if conv_hidden is None:
                conv_hidden = max(1, int(round(channels * float(hybrid_encoder_conv_hidden_scale))))
            spectral_hidden = hybrid_encoder_spectral_hidden_channels
            if spectral_hidden is None:
                spectral_hidden = max(1, int(round(channels * float(hybrid_encoder_spectral_hidden_scale))))
            self.encoder_blocks.append(
                HybridResnetEncoderBlock(
                    channels,
                    modes=modes,
                    conv_hidden_channels=int(conv_hidden),
                    spectral_hidden_channels=int(spectral_hidden),
                )
            )
            if index < self.hybrid_downsample_steps:
                next_channels = channels * 2
                if max_hidden_channels is not None:
                    next_channels = min(next_channels, max_hidden_channels)
                self.downsample_layers.append(
                    self._build_downsample_layer(hybrid_downsample_op, channels, next_channels)
                )
                channels = next_channels

        if resnet_width is not None:
            if resnet_width <= 0:
                raise ValueError(f"resnet_width must be positive when set, got {resnet_width}.")
            if resnet_width % 4 != 0:
                raise ValueError(
                    "resnet_width must be divisible by 4 so the CycleGAN upsamplers produce integer channel sizes "
                    f"(got {resnet_width})."
                )
        target_width = channels if resnet_width is None else int(resnet_width)
        self.bottleneck_channels = int(target_width)
        self.adapter = nn.Identity()
        if target_width != channels:
            self.adapter = nn.Conv2d(channels, target_width, kernel_size=1)
        self.resnet = SharedSpectralResnetBottleneck(
            target_width,
            n_blocks=spectral_bottleneck_blocks,
            modes=spectral_bottleneck_modes,
            share_spectral_weights=spectral_bottleneck_share_weights,
            gate_init=spectral_bottleneck_gate_init,
            gate_mode=spectral_bottleneck_gate_mode,
        )

        upsample_widths = [target_width]
        for _ in range(self.hybrid_downsample_steps):
            upsample_widths.append(max(hidden_channels, upsample_widths[-1] // 2))
        self.decoder_widths = list(upsample_widths)
        self.upsample_layers = nn.ModuleList(
            [
                CycleGanUpsampler(upsample_widths[index], upsample_widths[index + 1])
                for index in range(self.hybrid_downsample_steps)
            ]
        )
        self.output_proj = nn.Conv2d(upsample_widths[-1], out_channels * self.C, kernel_size=1)
        self.output_amp = nn.Conv2d(upsample_widths[-1], self.C, kernel_size=1)
        self.output_phase = nn.Conv2d(upsample_widths[-1], self.C, kernel_size=1)

    @staticmethod
    def _build_downsample_layer(downsample_op: str, in_channels: int, out_channels: int) -> nn.Module:
        if downsample_op == "stride_conv":
            return StrideConvDownsample(in_channels, out_channels)
        if downsample_op == "avgpool_conv":
            return AvgPoolConvDownsample(in_channels, out_channels)
        if downsample_op == "blurpool_conv":
            return BlurPoolConvDownsample(in_channels, out_channels)
        raise ValueError(f"Unsupported downsample op: {downsample_op!r}")

    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        x = self.lifter(x)
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = self.adapter(x)
        x = self.resnet(x)
        for upsample in self.upsample_layers:
            x = upsample(x)
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase
        x = self.output_proj(x)
        return x.view(batch, 2, self.C, height, width).permute(0, 3, 4, 2, 1)


class SpectralResnetBottleneckGenerator:
    """Generator registry wrapper for spectral_resnet_bottleneck_net."""

    name = "spectral_resnet_bottleneck_net"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> "nn.Module":
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)

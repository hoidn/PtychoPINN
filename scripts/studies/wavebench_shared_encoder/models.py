from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ptycho_torch.generators.ffno_bottleneck import build_no_refiner_ffno_stack
from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import (
    AvgPoolConvDownsample,
    BlurPoolConvDownsample,
    HybridResnetEncoderBlock,
    StrideConvDownsample,
)
from ptycho_torch.generators.resnet_components import CycleGanUpsampler, ResnetBottleneck
from ptycho_torch.generators.spectral_resnet_bottleneck import SharedSpectralResnetBottleneck

from scripts.studies.wavebench_shared_encoder.encoder import SharedMeasurementEncoder


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = torch.nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class LocalUnetBody(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, width: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        self.up1 = Up(width * 8, width * 4)
        self.up2 = Up(width * 4, width * 2)
        self.up3 = Up(width * 2, width)
        self.out = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FourierLayer2d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = nn.Conv2d(channels, channels, kernel_size=1)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


def concat_coordinates(x: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, device=x.device),
        torch.linspace(-1.0, 1.0, steps=width, device=x.device),
        indexing="ij",
    )
    coords = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    return torch.cat([x, coords], dim=1)


class FnoReconstructionBody(nn.Module):
    def __init__(self, in_channels: int, hidden_width: int = 32, modes: int = 12, n_blocks: int = 4):
        super().__init__()
        self.lifter = nn.Sequential(
            Conv1x1(in_channels + 2, hidden_width),
            nn.GELU(),
            Conv1x1(hidden_width, hidden_width),
        )
        self.hidden_layers = nn.Sequential(
            *[FourierLayer2d(hidden_width, modes) for _ in range(n_blocks)]
        )
        self.projector = nn.Sequential(
            Conv1x1(hidden_width, hidden_width),
            nn.GELU(),
            Conv1x1(hidden_width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = concat_coordinates(x)
        x = self.lifter(x)
        x = self.hidden_layers(x)
        return self.projector(x)


class _LocalResidualRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class FfnoReconstructionBody(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, modes: int = 12, n_blocks: int = 4):
        super().__init__()
        self.lifter = SpatialLifter(in_channels, hidden_channels)
        self.ffno_stack = build_no_refiner_ffno_stack(
            hidden_channels,
            n_blocks=n_blocks,
            modes=modes,
            share_spectral_weights=True,
            mlp_ratio=2.0,
            gate_init=0.1,
            norm="instance",
        )
        self.refiners = nn.Sequential(_LocalResidualRefiner(hidden_channels), _LocalResidualRefiner(hidden_channels))
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        x = self.ffno_stack(x)
        x = self.refiners(x)
        return self.output(x)


class HybridShellBody(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_builder: Callable[[int], nn.Module],
        *,
        hidden_channels: int = 32,
        modes: int = 12,
        encoder_blocks: int = 4,
        downsample_steps: int = 2,
        downsample_op: str = "stride_conv",
    ):
        super().__init__()
        self.lifter = SpatialLifter(in_channels, hidden_channels)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        channels = hidden_channels
        for index in range(encoder_blocks):
            self.encoder_blocks.append(HybridResnetEncoderBlock(channels, modes=modes))
            if index < downsample_steps:
                next_channels = channels * 2
                self.downsample_layers.append(_build_downsample(downsample_op, channels, next_channels))
                channels = next_channels
        self.bottleneck = bottleneck_builder(channels)
        decoder_widths = [channels]
        for _ in range(downsample_steps):
            decoder_widths.append(decoder_widths[-1] // 2)
        self.upsample_layers = nn.ModuleList(
            CycleGanUpsampler(decoder_widths[index], decoder_widths[index + 1])
            for index in range(downsample_steps)
        )
        self.output = nn.Conv2d(decoder_widths[-1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = self.bottleneck(x)
        for upsample in self.upsample_layers:
            x = upsample(x)
        return self.output(x)


def _build_downsample(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    if kind == "stride_conv":
        return StrideConvDownsample(in_channels, out_channels)
    if kind == "avgpool_conv":
        return AvgPoolConvDownsample(in_channels, out_channels)
    if kind == "blurpool_conv":
        return BlurPoolConvDownsample(in_channels, out_channels)
    raise ValueError(f"unsupported downsample kind: {kind}")


class SharedEncoderReconstructionModel(nn.Module):
    def __init__(self, encoder: nn.Module, body: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.body(latent)


def build_shared_encoder_row(*, row: str, latent_channels: int) -> nn.Module:
    encoder = SharedMeasurementEncoder(latent_channels=latent_channels)
    row = str(row)
    if row == "cnn":
        body = LocalUnetBody(in_channels=latent_channels)
    elif row == "hybrid_resnet":
        body = HybridShellBody(
            in_channels=latent_channels,
            bottleneck_builder=lambda channels: ResnetBottleneck(channels, n_blocks=6),
        )
    elif row == "spectral_resnet_bottleneck_net":
        body = HybridShellBody(
            in_channels=latent_channels,
            bottleneck_builder=lambda channels: SharedSpectralResnetBottleneck(
                channels,
                n_blocks=6,
                modes=12,
            ),
        )
    elif row == "fno":
        body = FnoReconstructionBody(in_channels=latent_channels)
    elif row == "ffno":
        body = FfnoReconstructionBody(in_channels=latent_channels)
    else:
        raise ValueError(f"Unknown shared-encoder row: {row}")
    return SharedEncoderReconstructionModel(encoder=encoder, body=body)


def profile_model(model: nn.Module) -> dict[str, int]:
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    body_parameters = sum(parameter.numel() for parameter in model.body.parameters())
    return {
        "encoder_parameters": int(encoder_parameters),
        "body_parameters": int(body_parameters),
        "total_parameters": int(encoder_parameters + body_parameters),
    }

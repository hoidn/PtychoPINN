"""Tiny supervised model adapters for the PDEBench SWE smoke gate."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import (
    AvgPoolConvDownsample,
    HybridResnetEncoderBlock,
)
from ptycho_torch.generators.resnet_components import CycleGanUpsampler, ResnetBottleneck


class ModelBuildBlocker(RuntimeError):
    """Controlled blocker for optional baseline dependencies."""

    def __init__(self, model: str, reason: str, message: str):
        super().__init__(message)
        self.model = model
        self.reason = reason

    def to_payload(self, *, run_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "reason": self.reason,
            "message": str(self),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return payload


def _import_neuralop_fno():
    from neuralop.models import FNO

    return FNO


class PadCropWrapper(nn.Module):
    """Pad spatial dimensions for internal divisibility, then crop output back."""

    def __init__(self, module: nn.Module, multiple: int = 1):
        super().__init__()
        self.module = module
        self.multiple = max(1, int(multiple))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        padded_height = ((height + self.multiple - 1) // self.multiple) * self.multiple
        padded_width = ((width + self.multiple - 1) // self.multiple) * self.multiple
        x_padded = F.pad(x, (0, padded_width - width, 0, padded_height - height))
        y = self.module(x_padded)
        return y[..., :height, :width]


class HybridResnetSweModel(nn.Module):
    """Supervised real-channel PDE adapter using Hybrid ResNet components."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        fno_modes: int,
        fno_blocks: int,
        resnet_blocks: int,
        downsample_steps: int,
    ):
        super().__init__()
        self.lifter = SpatialLifter(in_channels, hidden_channels)
        channels = hidden_channels
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for index in range(fno_blocks):
            self.encoder_blocks.append(HybridResnetEncoderBlock(channels, modes=fno_modes))
            if index < downsample_steps:
                self.downsample_layers.append(AvgPoolConvDownsample(channels, channels * 2))
                channels *= 2
        self.resnet = ResnetBottleneck(channels, n_blocks=resnet_blocks)
        upsample_widths = [channels]
        for _ in range(downsample_steps):
            upsample_widths.append(max(hidden_channels, upsample_widths[-1] // 2))
        self.upsample_layers = nn.ModuleList(
            [
                CycleGanUpsampler(upsample_widths[index], upsample_widths[index + 1])
                for index in range(downsample_steps)
            ]
        )
        self.output = nn.Conv2d(upsample_widths[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = self.resnet(x)
        for upsample in self.upsample_layers:
            x = upsample(x)
        return self.output(x)


class SmallUNet(nn.Module):
    """Compact U-Net baseline for CPU/GPU smoke tests."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.out = nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc1(x)
        y = self.mid(self.down(skip))
        y = self.up(y)
        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([y, skip], dim=1))


def build_model(
    model_name: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_shape: tuple[int, int],
    smoke_config: Mapping[str, Any],
) -> nn.Module:
    """Build a tiny one-step supervised model with `model(x)` semantics."""
    name = model_name.strip().lower()
    hidden_channels = int(smoke_config.get("hidden_channels", 8))
    fno_modes = int(smoke_config.get("fno_modes", 4))
    fno_blocks = int(smoke_config.get("fno_blocks", 3))
    downsample_steps = int(smoke_config.get("hybrid_downsample_steps", 1))
    resnet_blocks = int(smoke_config.get("hybrid_resnet_blocks", 1))

    if name == "hybrid_resnet":
        multiple = 2 ** max(0, downsample_steps)
        return PadCropWrapper(
            HybridResnetSweModel(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                fno_modes=fno_modes,
                fno_blocks=max(1, fno_blocks),
                resnet_blocks=max(1, resnet_blocks),
                downsample_steps=max(0, downsample_steps),
            ),
            multiple=multiple,
        )
    if name == "unet":
        return PadCropWrapper(SmallUNet(in_channels, out_channels, hidden_channels), multiple=2)
    if name == "fno":
        try:
            fno_cls = _import_neuralop_fno()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ModelBuildBlocker(
                "fno",
                "model_dependency_unavailable",
                f"neuralop.models.FNO is unavailable: {exc}",
            ) from exc
        modes = min(fno_modes, int(spatial_shape[0]), int(spatial_shape[1]))
        return fno_cls(
            n_modes=(modes, modes),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=max(1, fno_blocks),
        )
    raise ValueError(f"unknown SWE smoke model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def describe_model(model: nn.Module, *, model_name: str, smoke_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "class_name": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "smoke_config": dict(smoke_config),
    }

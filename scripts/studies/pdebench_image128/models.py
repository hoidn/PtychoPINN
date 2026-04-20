"""Supervised model profiles for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import AvgPoolConvDownsample, HybridResnetEncoderBlock
from ptycho_torch.generators.resnet_components import CycleGanUpsampler, ResnetBottleneck
from scripts.studies.pdebench_image128.run_config import ModelProfile


class ModelBuildBlocker(RuntimeError):
    """Controlled blocker for optional model dependencies."""

    def __init__(self, model: str, reason: str, message: str):
        super().__init__(message)
        self.model = model
        self.reason = reason

    def to_payload(self, *, run_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "reason": self.reason, "message": str(self)}
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
        y = self.module(F.pad(x, (0, padded_width - width, 0, padded_height - height)))
        return y[..., :height, :width]


class HybridResnetImageModel(nn.Module):
    """Supervised real-channel adapter around the Hybrid ResNet body."""

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
            [CycleGanUpsampler(upsample_widths[index], upsample_widths[index + 1]) for index in range(downsample_steps)]
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
    """Compact U-Net for readiness-only smoke checks."""

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


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StrongUNet(nn.Module):
    """PDEBench-style U-Net profile with init_features=32 by default."""

    def __init__(self, in_channels: int, out_channels: int, init_features: int = 32):
        super().__init__()
        features = int(init_features)
        self.encoder1 = UNetBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNetBlock(features * 8, features * 16)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNetBlock(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNetBlock(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNetBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNetBlock(features * 2, features)
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


def build_model_from_profile(
    profile: ModelProfile,
    *,
    in_channels: int,
    out_channels: int,
    spatial_shape: tuple[int, int],
) -> nn.Module:
    config = profile.to_model_config()
    if profile.base_model == "hybrid_resnet":
        downsample_steps = int(config.get("hybrid_downsample_steps", 2))
        return PadCropWrapper(
            HybridResnetImageModel(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=int(config.get("hidden_channels", 32)),
                fno_modes=int(config.get("fno_modes", 12)),
                fno_blocks=int(config.get("fno_blocks", 4)),
                resnet_blocks=int(config.get("hybrid_resnet_blocks", 6)),
                downsample_steps=downsample_steps,
            ),
            multiple=2 ** max(0, downsample_steps),
        )
    if profile.base_model == "unet_tiny":
        return PadCropWrapper(SmallUNet(in_channels, out_channels, int(config.get("hidden_channels", 16))), multiple=2)
    if profile.base_model == "unet_strong":
        return PadCropWrapper(StrongUNet(in_channels, out_channels, int(config.get("unet_init_features", 32))), multiple=16)
    if profile.base_model == "fno":
        try:
            fno_cls = _import_neuralop_fno()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ModelBuildBlocker(
                profile.profile_id,
                "model_dependency_unavailable",
                f"neuralop.models.FNO is unavailable: {exc}",
            ) from exc
        modes = min(int(config.get("fno_modes", 12)), int(spatial_shape[0]), int(spatial_shape[1]))
        return fno_cls(
            n_modes=(modes, modes),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 32)),
            n_layers=int(config.get("fno_blocks", 4)),
        )
    raise ValueError(f"unknown model profile base_model: {profile.base_model}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def describe_model(model: nn.Module, *, profile: ModelProfile) -> dict[str, Any]:
    return {
        "schema_version": "pdebench_image128_model_profile_v1",
        "profile_id": profile.profile_id,
        "base_model": profile.base_model,
        "class_name": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "profile_config": profile.to_model_config(),
        "strong_baseline": bool(profile.strong_baseline),
        "evidence_scope": profile.evidence_scope,
    }


def assert_strong_baseline_profile(profile: ModelProfile) -> None:
    if not profile.strong_baseline:
        raise ValueError(f"{profile.profile_id} is readiness-only and cannot satisfy the strong-baseline gate")

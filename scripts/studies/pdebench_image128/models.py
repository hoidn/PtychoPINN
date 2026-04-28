"""Supervised model profiles for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import HybridResnetEncoderBlock, StrideConvDownsample
from ptycho_torch.generators.resnet_components import CycleGanUpsampler, ResnetBottleneck
from ptycho_torch.generators.ffno_bottleneck import SharedFactorizedFfnoBottleneck
from ptycho_torch.generators.spectral_resnet_bottleneck import SharedSpectralResnetBottleneck
from scripts.studies.pdebench_image128.author_ffno_adapter import (
    AuthorFfnoAdapterBuildError,
    AuthorFfnoCnsModel,
)
from scripts.studies.pdebench_image128.gnot_adapter import GnotAdapterBuildError, GnotCnsModel
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


class InterpConvUpsampler(nn.Module):
    """Resize-convolution upsampler for artifact study variants."""

    def __init__(self, in_channels: int, out_channels: int, *, mode: str = "bilinear"):
        super().__init__()
        self.mode = str(mode)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"bilinear", "bicubic"}:
            x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        return self.proj(x)


class PixelShuffleUpsampler(nn.Module):
    """Sub-pixel convolution upsampler for artifact study variants."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_hybrid_upsampler(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    if kind == "cyclegan_transpose":
        return CycleGanUpsampler(in_channels, out_channels)
    if kind == "interp_bilinear_conv":
        return InterpConvUpsampler(in_channels, out_channels, mode="bilinear")
    if kind == "interp_nearest_conv":
        return InterpConvUpsampler(in_channels, out_channels, mode="nearest")
    if kind == "pixelshuffle":
        return PixelShuffleUpsampler(in_channels, out_channels)
    raise ValueError(f"unknown hybrid upsampler: {kind}")


class _SharedPdebenchHybridShell(nn.Module):
    """Shared supervised shell for PDEBench image-suite bottleneck variants."""

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
        upsampler: str = "cyclegan_transpose",
        skip_connections: bool = False,
        hybrid_skip_style: str = "add",
        bottleneck_builder: Callable[[int], nn.Module],
    ):
        super().__init__()
        if hybrid_skip_style not in {"add", "concat", "gated_add"}:
            raise ValueError(f"hybrid_skip_style must be one of add|concat|gated_add (got {hybrid_skip_style!r}).")
        self.lifter = SpatialLifter(in_channels, hidden_channels)
        self.skip_connections = bool(skip_connections)
        self.hybrid_skip_style = str(hybrid_skip_style)
        channels = hidden_channels
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.stage_metadata: list[dict[str, int]] = [{"resolution_divisor": 1, "channels": channels}]
        for index in range(fno_blocks):
            self.encoder_blocks.append(HybridResnetEncoderBlock(channels, modes=fno_modes))
            if index < downsample_steps:
                self.downsample_layers.append(StrideConvDownsample(channels, channels * 2))
                channels *= 2
                self.stage_metadata.append(
                    {
                        "resolution_divisor": 2 ** len(self.downsample_layers),
                        "channels": channels,
                    }
                )
        self.resnet = bottleneck_builder(channels)
        upsample_widths = [channels]
        for _ in range(downsample_steps):
            upsample_widths.append(max(hidden_channels, upsample_widths[-1] // 2))
        self.upsample_layers = nn.ModuleList(
            [
                _make_hybrid_upsampler(upsampler, upsample_widths[index], upsample_widths[index + 1])
                for index in range(downsample_steps)
            ]
        )

        self.encoder_tap_plan, metadata_skip_plan = self._derive_skip_topology_from_stage_metadata(
            self.stage_metadata,
            downsample_steps=downsample_steps,
        )
        self.skip_fusion_plan: list[dict[str, int | str]] = []
        self.skip_fusion_projections = nn.ModuleDict()
        self.skip_fusion_gates = nn.ParameterDict()
        if self.skip_connections:
            for index, plan in enumerate(metadata_skip_plan):
                key = str(plan["key"])
                decoder_channels = upsample_widths[index + 1]
                skip_channels = int(plan["channels"])
                projection_in_channels = skip_channels
                if self.hybrid_skip_style == "concat":
                    projection_in_channels += decoder_channels
                self.skip_fusion_plan.append(
                    {
                        "key": key,
                        "resolution_divisor": int(plan["resolution_divisor"]),
                        "skip_channels": skip_channels,
                        "decoder_channels": decoder_channels,
                    }
                )
                self.skip_fusion_projections[key] = nn.Conv2d(
                    projection_in_channels,
                    decoder_channels,
                    kernel_size=1,
                )
                if self.hybrid_skip_style == "gated_add":
                    self.skip_fusion_gates[key] = nn.Parameter(torch.tensor(0.1))
        self.output = nn.Conv2d(upsample_widths[-1], out_channels, kernel_size=1)

    @staticmethod
    def _derive_skip_topology_from_stage_metadata(
        stage_metadata: list[dict[str, int]],
        *,
        downsample_steps: int,
    ) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
        if downsample_steps < 0:
            raise ValueError(f"downsample_steps must be non-negative, got {downsample_steps}.")
        if len(stage_metadata) < downsample_steps + 1:
            raise ValueError(
                "stage_metadata must include one entry per downsample stage plus bottleneck "
                f"(len={len(stage_metadata)}, downsample_steps={downsample_steps})."
            )
        tap_stages = stage_metadata[:downsample_steps]
        encoder_tap_plan: list[dict[str, int | str]] = []
        for stage in tap_stages:
            divisor = int(stage["resolution_divisor"])
            encoder_tap_plan.append(
                {
                    "key": f"d{divisor}",
                    "resolution_divisor": divisor,
                    "channels": int(stage["channels"]),
                }
            )
        return encoder_tap_plan, list(reversed(encoder_tap_plan))

    def _apply_skip_fusion(self, decoder_x: torch.Tensor, skip_x: torch.Tensor, fusion_key: str) -> torch.Tensor:
        if self.hybrid_skip_style == "add":
            return decoder_x + self.skip_fusion_projections[fusion_key](skip_x)
        if self.hybrid_skip_style == "concat":
            return self.skip_fusion_projections[fusion_key](torch.cat([decoder_x, skip_x], dim=1))
        if self.hybrid_skip_style == "gated_add":
            gate = self.skip_fusion_gates[fusion_key]
            return decoder_x + gate * self.skip_fusion_projections[fusion_key](skip_x)
        raise RuntimeError(f"Unknown hybrid_skip_style: {self.hybrid_skip_style!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        encoder_taps: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.downsample_layers):
                tap = self.encoder_tap_plan[index]
                encoder_taps[str(tap["key"])] = x
                x = self.downsample_layers[index](x)
        x = self.resnet(x)
        for index, upsample in enumerate(self.upsample_layers):
            x = upsample(x)
            if self.skip_connections and index < len(self.skip_fusion_plan):
                plan = self.skip_fusion_plan[index]
                key = str(plan["key"])
                skip_tensor = encoder_taps.get(key)
                if skip_tensor is not None:
                    x = self._apply_skip_fusion(x, skip_tensor, key)
        return self.output(x)


class HybridResnetImageModel(_SharedPdebenchHybridShell):
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
        upsampler: str = "cyclegan_transpose",
        skip_connections: bool = False,
        hybrid_skip_style: str = "add",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            fno_modes=fno_modes,
            fno_blocks=fno_blocks,
            resnet_blocks=resnet_blocks,
            downsample_steps=downsample_steps,
            upsampler=upsampler,
            skip_connections=skip_connections,
            hybrid_skip_style=hybrid_skip_style,
            bottleneck_builder=lambda channels: ResnetBottleneck(channels, n_blocks=resnet_blocks),
        )


class SpectralResnetBottleneckImageModel(_SharedPdebenchHybridShell):
    """Supervised real-channel adapter around the spectral ResNet bottleneck body."""

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
        spectral_bottleneck_blocks: int,
        spectral_bottleneck_modes: int,
        spectral_bottleneck_share_weights: bool,
        spectral_bottleneck_gate_init: float,
        spectral_bottleneck_gate_mode: str,
        upsampler: str = "cyclegan_transpose",
        skip_connections: bool = False,
        hybrid_skip_style: str = "add",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            fno_modes=fno_modes,
            fno_blocks=fno_blocks,
            resnet_blocks=resnet_blocks,
            downsample_steps=downsample_steps,
            upsampler=upsampler,
            skip_connections=skip_connections,
            hybrid_skip_style=hybrid_skip_style,
            bottleneck_builder=lambda channels: SharedSpectralResnetBottleneck(
                channels,
                n_blocks=spectral_bottleneck_blocks,
                modes=spectral_bottleneck_modes,
                share_spectral_weights=spectral_bottleneck_share_weights,
                gate_init=spectral_bottleneck_gate_init,
                gate_mode=spectral_bottleneck_gate_mode,
                layerscale_init=0.1,
            ),
        )


class FfnoBottleneckImageModel(_SharedPdebenchHybridShell):
    """Supervised real-channel adapter around the FFNO-close bottleneck body."""

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
        ffno_bottleneck_blocks: int,
        ffno_bottleneck_modes: int,
        ffno_bottleneck_share_weights: bool,
        ffno_bottleneck_mlp_ratio: float,
        ffno_bottleneck_gate_init: float,
        ffno_bottleneck_norm: str,
        ffno_bottleneck_local_conv: bool,
        ffno_bottleneck_local_conv_kernel_size: int | None,
        upsampler: str = "cyclegan_transpose",
        skip_connections: bool = False,
        hybrid_skip_style: str = "add",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            fno_modes=fno_modes,
            fno_blocks=fno_blocks,
            resnet_blocks=resnet_blocks,
            downsample_steps=downsample_steps,
            upsampler=upsampler,
            skip_connections=skip_connections,
            hybrid_skip_style=hybrid_skip_style,
            bottleneck_builder=lambda channels: SharedFactorizedFfnoBottleneck(
                channels,
                n_blocks=ffno_bottleneck_blocks,
                modes=ffno_bottleneck_modes,
                share_spectral_weights=ffno_bottleneck_share_weights,
                mlp_ratio=ffno_bottleneck_mlp_ratio,
                gate_init=ffno_bottleneck_gate_init,
                norm=ffno_bottleneck_norm,
                local_conv_kernel_size=(
                    int(ffno_bottleneck_local_conv_kernel_size)
                    if ffno_bottleneck_local_conv and ffno_bottleneck_local_conv_kernel_size is not None
                    else None
                ),
            ),
        )


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
    task_metadata: dict[str, Any] | None = None,
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
                upsampler=str(config.get("hybrid_upsampler", "cyclegan_transpose")),
                skip_connections=bool(config.get("hybrid_skip_connections", False)),
                hybrid_skip_style=str(config.get("hybrid_skip_style", "add")),
            ),
            multiple=2 ** max(0, downsample_steps),
        )
    if profile.base_model == "spectral_resnet_bottleneck_net":
        downsample_steps = int(config.get("hybrid_downsample_steps", 2))
        return PadCropWrapper(
            SpectralResnetBottleneckImageModel(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=int(config.get("hidden_channels", 32)),
                fno_modes=int(config.get("fno_modes", 12)),
                fno_blocks=int(config.get("fno_blocks", 4)),
                resnet_blocks=int(config.get("hybrid_resnet_blocks", 6)),
                downsample_steps=downsample_steps,
                spectral_bottleneck_blocks=int(config.get("spectral_bottleneck_blocks", 6)),
                spectral_bottleneck_modes=int(config.get("spectral_bottleneck_modes", 12)),
                spectral_bottleneck_share_weights=bool(config.get("spectral_bottleneck_share_weights", True)),
                spectral_bottleneck_gate_init=float(config.get("spectral_bottleneck_gate_init", 0.1)),
                spectral_bottleneck_gate_mode=str(config.get("spectral_bottleneck_gate_mode", "shared")),
                upsampler=str(config.get("hybrid_upsampler", "cyclegan_transpose")),
                skip_connections=bool(config.get("hybrid_skip_connections", False)),
                hybrid_skip_style=str(config.get("hybrid_skip_style", "add")),
            ),
            multiple=2 ** max(0, downsample_steps),
        )
    if profile.base_model == "ffno_bottleneck_net":
        downsample_steps = int(config.get("hybrid_downsample_steps", 2))
        return PadCropWrapper(
            FfnoBottleneckImageModel(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=int(config.get("hidden_channels", 32)),
                fno_modes=int(config.get("fno_modes", 12)),
                fno_blocks=int(config.get("fno_blocks", 4)),
                resnet_blocks=int(config.get("hybrid_resnet_blocks", 6)),
                downsample_steps=downsample_steps,
                ffno_bottleneck_blocks=int(config.get("ffno_bottleneck_blocks", 6)),
                ffno_bottleneck_modes=int(config.get("ffno_bottleneck_modes", 12)),
                ffno_bottleneck_share_weights=bool(config.get("ffno_bottleneck_share_weights", True)),
                ffno_bottleneck_mlp_ratio=float(config.get("ffno_bottleneck_mlp_ratio", 2.0)),
                ffno_bottleneck_gate_init=float(config.get("ffno_bottleneck_gate_init", 0.1)),
                ffno_bottleneck_norm=str(config.get("ffno_bottleneck_norm", "instance")),
                ffno_bottleneck_local_conv=bool(config.get("ffno_bottleneck_local_conv", False)),
                ffno_bottleneck_local_conv_kernel_size=(
                    int(config["ffno_bottleneck_local_conv_kernel_size"])
                    if config.get("ffno_bottleneck_local_conv_kernel_size") is not None
                    else None
                ),
                upsampler=str(config.get("hybrid_upsampler", "cyclegan_transpose")),
                skip_connections=bool(config.get("hybrid_skip_connections", False)),
                hybrid_skip_style=str(config.get("hybrid_skip_style", "add")),
            ),
            multiple=2 ** max(0, downsample_steps),
        )
    if profile.base_model == "gnot_cns_net":
        if task_metadata is None:
            raise ModelBuildBlocker(
                profile.profile_id,
                "missing_task_metadata",
                "GNOT CNS model requires task_metadata from the runner.",
            )
        try:
            return GnotCnsModel(
                in_channels=in_channels,
                out_channels=out_channels,
                spatial_shape=spatial_shape,
                task_metadata=task_metadata,
                n_hidden=int(config.get("gnot_hidden", config.get("hidden_channels", 64))),
                n_layers=int(config.get("gnot_layers", 3)),
                n_head=int(config.get("gnot_heads", 1)),
                n_experts=int(config.get("gnot_experts", 1)),
                n_inner=int(config.get("gnot_inner_multiplier", 4)),
                mlp_layers=int(config.get("gnot_mlp_layers", 3)),
                attn_type=str(config.get("gnot_attn_type", "linear")),
            )
        except GnotAdapterBuildError as exc:
            raise ModelBuildBlocker(
                profile.profile_id,
                exc.reason,
                str(exc),
            ) from exc
    if profile.base_model == "author_ffno_cns_net":
        if task_metadata is None:
            raise ModelBuildBlocker(
                profile.profile_id,
                "missing_task_metadata",
                "Author FFNO CNS model requires task_metadata from the runner.",
            )
        try:
            return AuthorFfnoCnsModel(
                in_channels=in_channels,
                out_channels=out_channels,
                spatial_shape=spatial_shape,
                task_metadata=task_metadata,
                modes=int(config.get("author_ffno_modes", config.get("fno_modes", 16))),
                width=int(config.get("author_ffno_width", config.get("hidden_channels", 64))),
                n_layers=int(config.get("author_ffno_layers", config.get("fno_blocks", 24))),
                share_weight=bool(config.get("author_ffno_share_weight", True)),
                factor=int(config.get("author_ffno_factor", 4)),
                ff_weight_norm=bool(config.get("author_ffno_ff_weight_norm", True)),
                n_ff_layers=int(config.get("author_ffno_n_ff_layers", 2)),
                gain=float(config.get("author_ffno_gain", 0.1)),
                dropout=float(config.get("author_ffno_dropout", 0.0)),
                in_dropout=float(config.get("author_ffno_in_dropout", 0.0)),
                layer_norm=bool(config.get("author_ffno_layer_norm", False)),
                use_position=bool(config.get("author_ffno_use_position", True)),
                mode=str(config.get("author_ffno_mode", "full")),
            )
        except AuthorFfnoAdapterBuildError as exc:
            raise ModelBuildBlocker(
                profile.profile_id,
                exc.reason,
                str(exc),
            ) from exc
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


def _external_source_provenance(model: nn.Module) -> dict[str, Any] | None:
    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        provenance = getattr(current, "external_source_provenance", None)
        if isinstance(provenance, dict):
            return provenance
        next_model = getattr(current, "module", None)
        if not isinstance(next_model, nn.Module):
            break
        current = next_model
    return None


def describe_model(model: nn.Module, *, profile: ModelProfile) -> dict[str, Any]:
    payload = {
        "schema_version": "pdebench_image128_model_profile_v1",
        "profile_id": profile.profile_id,
        "base_model": profile.base_model,
        "class_name": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "profile_config": profile.to_model_config(),
        "strong_baseline": bool(profile.strong_baseline),
        "evidence_scope": profile.evidence_scope,
    }
    provenance = _external_source_provenance(model)
    if provenance is not None:
        payload["external_source_provenance"] = provenance
    return payload


def assert_strong_baseline_profile(profile: ModelProfile) -> None:
    if not profile.strong_baseline:
        raise ValueError(f"{profile.profile_id} is readiness-only and cannot satisfy the strong-baseline gate")

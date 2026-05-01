"""Hybrid ResNet-6 generator (FNO encoder + CycleGAN decoder)."""
import math
from typing import Dict, Any, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import (
    SpatialLifter,
    HAS_NEURALOPERATOR,
    _FallbackSpectralConv2d,
)
from ptycho_torch.generators.resnet_components import ResnetBottleneck, CycleGanUpsampler

try:
    from neuraloperator.layers.spectral_convolution import SpectralConv
except Exception:  # pragma: no cover - fallback exercised when dependency missing
    SpectralConv = None


class StrideConvDownsample(nn.Module):
    """Stride-2 convolution downsample."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AvgPoolConvDownsample(nn.Module):
    """Average-pool followed by 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


class BlurPoolConvDownsample(nn.Module):
    """Low-pass blurpool downsample followed by 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
        self.register_buffer("blur_kernel", kernel)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.blur_kernel.expand(channels, 1, 3, 3)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel, stride=2, groups=channels)
        return self.proj(x)


class HybridResnetEncoderBlock(nn.Module):
    """Hybrid encoder block with optional branch-capacity decoupling."""

    def __init__(
        self,
        channels: int,
        modes: int,
        conv_hidden_channels: Optional[int] = None,
        spectral_hidden_channels: Optional[int] = None,
    ):
        super().__init__()

        conv_hidden = channels if conv_hidden_channels is None else int(conv_hidden_channels)
        spectral_hidden = channels if spectral_hidden_channels is None else int(spectral_hidden_channels)

        self.spectral_in = nn.Identity()
        self.spectral_out = nn.Identity()
        if spectral_hidden != channels:
            self.spectral_in = nn.Conv2d(channels, spectral_hidden, kernel_size=1)
            self.spectral_out = nn.Conv2d(spectral_hidden, channels, kernel_size=1)

        if HAS_NEURALOPERATOR and SpectralConv is not None:
            self.spectral = SpectralConv(spectral_hidden, spectral_hidden, n_modes=(modes, modes))
        else:
            self.spectral = _FallbackSpectralConv2d(spectral_hidden, spectral_hidden, modes)

        self.conv_in = nn.Identity()
        self.conv_out = nn.Identity()
        if conv_hidden != channels:
            self.conv_in = nn.Conv2d(channels, conv_hidden, kernel_size=1)
            self.conv_out = nn.Conv2d(conv_hidden, channels, kernel_size=1)
        self.local_conv = nn.Conv2d(
            conv_hidden,
            conv_hidden,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectral = self.spectral_out(self.spectral(self.spectral_in(x)))
        conv = self.conv_out(self.local_conv(self.conv_in(x)))
        return x + self.act(spectral + conv)


class HybridResnetGeneratorModule(nn.Module):
    """FNO encoder -> ResNet bottleneck -> CycleGAN upsamplers."""

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
        skip_connections: bool = False,
        hybrid_downsample_steps: int = 2,
        hybrid_downsample_op: Literal["stride_conv", "avgpool_conv", "blurpool_conv"] = "stride_conv",
        hybrid_encoder_conv_hidden_scale: float = 1.0,
        hybrid_encoder_spectral_hidden_scale: float = 1.0,
        hybrid_encoder_conv_hidden_channels: Optional[int] = None,
        hybrid_encoder_spectral_hidden_channels: Optional[int] = None,
        hybrid_skip_style: Literal["add", "concat", "gated_add"] = "add",
        bottleneck_layerscale_mode: Literal["learned", "fixed"] = "learned",
        bottleneck_layerscale_value: Optional[float] = None,
    ):
        super().__init__()
        if hybrid_downsample_steps not in (1, 2):
            raise ValueError(
                "hybrid_downsample_steps must be in [1, 2] "
                f"for current hybrid_resnet implementation (got {hybrid_downsample_steps})."
            )
        if hybrid_downsample_op not in {"stride_conv", "avgpool_conv", "blurpool_conv"}:
            raise ValueError(
                "hybrid_downsample_op must be one of: stride_conv, avgpool_conv, blurpool_conv "
                f"(got {hybrid_downsample_op!r})."
            )
        if hybrid_skip_style not in {"add", "concat", "gated_add"}:
            raise ValueError(
                f"hybrid_skip_style must be one of add|concat|gated_add (got {hybrid_skip_style!r})."
            )
        if (
            not math.isfinite(float(hybrid_encoder_conv_hidden_scale))
            or float(hybrid_encoder_conv_hidden_scale) <= 0.0
        ):
            raise ValueError(
                "hybrid_encoder_conv_hidden_scale must be finite and > 0 "
                f"(got {hybrid_encoder_conv_hidden_scale})."
            )
        if (
            not math.isfinite(float(hybrid_encoder_spectral_hidden_scale))
            or float(hybrid_encoder_spectral_hidden_scale) <= 0.0
        ):
            raise ValueError(
                "hybrid_encoder_spectral_hidden_scale must be finite and > 0 "
                f"(got {hybrid_encoder_spectral_hidden_scale})."
            )
        if hybrid_encoder_conv_hidden_channels is not None and hybrid_encoder_conv_hidden_channels <= 0:
            raise ValueError(
                "hybrid_encoder_conv_hidden_channels must be positive when set, "
                f"got {hybrid_encoder_conv_hidden_channels}."
            )
        if (
            hybrid_encoder_spectral_hidden_channels is not None
            and hybrid_encoder_spectral_hidden_channels <= 0
        ):
            raise ValueError(
                "hybrid_encoder_spectral_hidden_channels must be positive when set, "
                f"got {hybrid_encoder_spectral_hidden_channels}."
            )
        if resnet_blocks <= 0:
            raise ValueError(f"resnet_blocks must be positive, got {resnet_blocks}.")
        if bottleneck_layerscale_mode not in {"learned", "fixed"}:
            raise ValueError(
                "bottleneck_layerscale_mode must be one of ['fixed', 'learned'] "
                f"(got {bottleneck_layerscale_mode!r})."
            )
        if bottleneck_layerscale_mode == "learned" and bottleneck_layerscale_value is not None:
            raise ValueError(
                "bottleneck_layerscale_value must be omitted when "
                "bottleneck_layerscale_mode='learned'."
            )
        if bottleneck_layerscale_mode == "fixed":
            if bottleneck_layerscale_value is None:
                raise ValueError(
                    "bottleneck_layerscale_value must be provided when "
                    "bottleneck_layerscale_mode='fixed'."
                )
            if (
                not math.isfinite(float(bottleneck_layerscale_value))
                or float(bottleneck_layerscale_value) <= 0.0
            ):
                raise ValueError(
                    "bottleneck_layerscale_value must be finite and > 0 when fixed "
                    f"(got {bottleneck_layerscale_value})."
                )
        if n_blocks < 3:
            raise ValueError(
                "hybrid_resnet requires fno_blocks >= 3 for current topology "
                f"(got fno_blocks={n_blocks})."
            )

        self.C = C
        self.output_mode = output_mode
        self.skip_connections = bool(skip_connections)
        self.hybrid_skip_style = hybrid_skip_style
        self.hybrid_downsample_steps = int(hybrid_downsample_steps)
        self.hybrid_encoder_conv_hidden_scale = float(hybrid_encoder_conv_hidden_scale)
        self.hybrid_encoder_spectral_hidden_scale = float(hybrid_encoder_spectral_hidden_scale)
        self.bottleneck_layerscale_mode = str(bottleneck_layerscale_mode)
        self.bottleneck_layerscale_value = (
            None if bottleneck_layerscale_value is None else float(bottleneck_layerscale_value)
        )

        self.lifter = SpatialLifter(
            in_channels * C,
            hidden_channels,
            input_transform=input_transform,
        )

        # Encoder: derive stage topology from downsample schedule.
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.downsample = self.downsample_layers  # compatibility alias
        ch = hidden_channels
        self.encoder_stage_channels: list[int] = []
        self.encoder_conv_hidden_resolved_per_block: list[int] = []
        self.encoder_spectral_hidden_resolved_per_block: list[int] = []
        self.stage_metadata: list[dict[str, int]] = [{"resolution_divisor": 1, "channels": ch}]
        for i in range(n_blocks):
            self.encoder_stage_channels.append(ch)
            conv_hidden_channels = hybrid_encoder_conv_hidden_channels
            if conv_hidden_channels is None:
                conv_hidden_channels = self._resolve_hidden_width(
                    ch,
                    self.hybrid_encoder_conv_hidden_scale,
                )
            spectral_hidden_channels = hybrid_encoder_spectral_hidden_channels
            if spectral_hidden_channels is None:
                spectral_hidden_channels = self._resolve_hidden_width(
                    ch,
                    self.hybrid_encoder_spectral_hidden_scale,
                )
            self.encoder_conv_hidden_resolved_per_block.append(int(conv_hidden_channels))
            self.encoder_spectral_hidden_resolved_per_block.append(int(spectral_hidden_channels))
            self.encoder_blocks.append(
                HybridResnetEncoderBlock(
                    ch,
                    modes=modes,
                    conv_hidden_channels=int(conv_hidden_channels),
                    spectral_hidden_channels=int(spectral_hidden_channels),
                )
            )
            if i < self.hybrid_downsample_steps:
                next_ch = ch * 2
                if max_hidden_channels is not None:
                    next_ch = min(next_ch, max_hidden_channels)
                self.downsample_layers.append(self._build_downsample_layer(hybrid_downsample_op, ch, next_ch))
                ch = next_ch
                self.stage_metadata.append(
                    {
                        "resolution_divisor": 2 ** len(self.downsample_layers),
                        "channels": ch,
                    }
                )

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
        self.bottleneck_channels = int(target_width)
        self.adapter = nn.Identity()
        if ch != target_width:
            self.adapter = nn.Conv2d(ch, target_width, kernel_size=1)

        self.resnet = ResnetBottleneck(
            target_width,
            n_blocks=resnet_blocks,
            layerscale_mode=self.bottleneck_layerscale_mode,
            layerscale_value=self.bottleneck_layerscale_value,
        )

        decoder_widths = [target_width]
        for _ in range(self.hybrid_downsample_steps):
            decoder_widths.append(decoder_widths[-1] // 2)

        self.upsample_layers = nn.ModuleList(
            [
                CycleGanUpsampler(decoder_widths[idx], decoder_widths[idx + 1])
                for idx in range(self.hybrid_downsample_steps)
            ]
        )
        out_ch = decoder_widths[-1]

        self.encoder_tap_plan, metadata_skip_plan = self._derive_skip_topology_from_stage_metadata(
            self.stage_metadata,
            downsample_steps=self.hybrid_downsample_steps,
        )

        self.skip_fusion_plan: list[dict[str, int | str]] = []
        self.skip_fusion_projections = nn.ModuleDict()
        self.skip_fusion_gates = nn.ParameterDict()
        if self.skip_connections:
            for idx, plan in enumerate(metadata_skip_plan):
                resolution_divisor = int(plan["resolution_divisor"])
                key = str(plan["key"])
                decoder_channels = decoder_widths[idx + 1]
                self.skip_fusion_plan.append(
                    {
                        "key": key,
                        "resolution_divisor": resolution_divisor,
                        "decoder_channels": decoder_channels,
                    }
                )
                self.skip_fusion_projections[key] = nn.LazyConv2d(decoder_channels, kernel_size=1)
                if self.hybrid_skip_style == "gated_add":
                    self.skip_fusion_gates[key] = nn.Parameter(torch.tensor(0.1))

        if self.output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(out_ch, C, kernel_size=1)
            self.output_phase = nn.Conv2d(out_ch, C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(out_ch, out_channels * C, kernel_size=1)

    @staticmethod
    def _build_downsample_layer(
        downsample_op: str,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        if downsample_op == "stride_conv":
            return StrideConvDownsample(in_channels, out_channels)
        if downsample_op == "avgpool_conv":
            return AvgPoolConvDownsample(in_channels, out_channels)
        if downsample_op == "blurpool_conv":
            return BlurPoolConvDownsample(in_channels, out_channels)
        raise ValueError(f"Unsupported downsample op: {downsample_op!r}")

    @staticmethod
    def _resolve_hidden_width(stage_channels: int, scale: float) -> int:
        return max(1, int(round(int(stage_channels) * float(scale))))

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
                }
            )
        skip_fusion_plan = list(reversed(encoder_tap_plan))
        return encoder_tap_plan, skip_fusion_plan

    def _apply_skip_fusion(
        self,
        decoder_x: torch.Tensor,
        skip_x: torch.Tensor,
        fusion_key: str,
    ) -> torch.Tensor:
        if self.hybrid_skip_style == "add":
            return decoder_x + self.skip_fusion_projections[fusion_key](skip_x)
        if self.hybrid_skip_style == "concat":
            merged = torch.cat([decoder_x, skip_x], dim=1)
            return self.skip_fusion_projections[fusion_key](merged)
        if self.hybrid_skip_style == "gated_add":
            gate = self.skip_fusion_gates[fusion_key]
            return decoder_x + gate * self.skip_fusion_projections[fusion_key](skip_x)
        raise RuntimeError(f"Unknown hybrid_skip_style: {self.hybrid_skip_style!r}")

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.lifter(x)

        encoder_taps: dict[str, torch.Tensor] = {}
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.downsample_layers):
                tap = self.encoder_tap_plan[i]
                encoder_taps[str(tap["key"])] = x
                x = self.downsample_layers[i](x)

        x = self.adapter(x)
        x = self.resnet(x)

        for idx, upsample in enumerate(self.upsample_layers):
            x = upsample(x)
            if self.skip_connections and idx < len(self.skip_fusion_plan):
                plan = self.skip_fusion_plan[idx]
                key = str(plan["key"])
                skip_tensor = encoder_taps.get(key)
                if skip_tensor is not None:
                    x = self._apply_skip_fusion(x, skip_tensor, key)

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
            resnet_blocks=getattr(model_config, "hybrid_resnet_blocks", 6),
            skip_connections=getattr(model_config, "hybrid_skip_connections", False),
            hybrid_downsample_steps=getattr(model_config, "hybrid_downsample_steps", 2),
            hybrid_downsample_op=getattr(model_config, "hybrid_downsample_op", "stride_conv"),
            hybrid_encoder_conv_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_scale",
                1.0,
            ),
            hybrid_encoder_spectral_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_scale",
                1.0,
            ),
            hybrid_encoder_conv_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_channels",
                None,
            ),
            hybrid_encoder_spectral_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_channels",
                None,
            ),
            hybrid_skip_style=getattr(model_config, "hybrid_skip_style", "add"),
            bottleneck_layerscale_mode=getattr(
                model_config,
                "hybrid_resnet_bottleneck_layerscale_mode",
                "learned",
            ),
            bottleneck_layerscale_value=getattr(
                model_config,
                "hybrid_resnet_bottleneck_layerscale_value",
                None,
            ),
        )

        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )

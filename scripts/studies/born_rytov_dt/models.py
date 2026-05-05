"""Task-local BRDT model adapters.

Reuses existing model bodies (Hybrid ResNet, FNO vanilla, U-Net) only
through ordinary real-channel ``model(x) -> q_pred`` adapters. These
adapters are deliberately NOT registered in
``ptycho_torch.generators.registry`` because the registry is tied to
the CDI/PtychoPINN output and stitching contract; BRDT must remain
task-local.

All adapters share the contract:

- input ``x``: ``(B, C_in, 128, 128)`` real-channel born_init_image
  representation (real, optional imag, optional confidence/mask),
- output: ``(B, 1, 128, 128)`` real-channel ``q_pred`` (in normalized or
  physical q units, decided by the run config / lightning module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.run_config import (
    SUPPORTED_ARCHITECTURES,
    get_default_arch_kwargs,
)


class AdapterBuildError(RuntimeError):
    """Controlled blocker for optional adapter dependencies."""

    def __init__(self, model: str, reason: str, message: str):
        super().__init__(message)
        self.model = model
        self.reason = reason

    def to_payload(self) -> Dict[str, Any]:
        return {"model": self.model, "reason": self.reason, "message": str(self)}


class _PadCropWrapper(nn.Module):
    """Pad input spatial dims to a multiple, run module, crop output back."""

    def __init__(self, module: nn.Module, multiple: int = 1):
        super().__init__()
        self.module = module
        self.multiple = max(1, int(multiple))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = ((h + self.multiple - 1) // self.multiple) * self.multiple
        pw = ((w + self.multiple - 1) // self.multiple) * self.multiple
        y = self.module(F.pad(x, (0, pw - w, 0, ph - h)))
        return y[..., :h, :w]


class _BRDTUNet(nn.Module):
    """Compact U-Net body suitable for the bounded BRDT preflight."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv2d(h, h * 2, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(h * 2, h * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up = nn.ConvTranspose2d(h * 2, h, kernel_size=2, stride=2)
        self.out = nn.Conv2d(h * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc(x)
        y = self.up(self.mid(self.down(skip)))
        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([y, skip], dim=1))


class _BRDTFFNO(nn.Module):
    """Task-local FFNO body: lifter -> shared factorized FFNO blocks -> 1x1 head.

    Reuses ``ptycho_torch.generators.ffno_bottleneck.SharedFactorizedFfnoBottleneck``
    behind a BRDT-local wrapper so the BRDT row produces ordinary
    real-channel ``q_pred`` output without registering FFNO in the CDI
    generator registry. The architecture identity is intentionally
    distinct from ``fno_vanilla`` (which uses ``neuralop.models.FNO``).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        fno_modes: int = 8,
        fno_blocks: int = 4,
        share_spectral_weights: bool = False,
        mlp_ratio: float = 2.0,
        cnn_blocks: int = 2,
    ):
        super().__init__()
        from ptycho_torch.generators.ffno_bottleneck import (
            SharedFactorizedFfnoBottleneck,
        )
        from ptycho_torch.generators.fno import SpatialLifter

        self.lifter = SpatialLifter(int(in_channels), int(hidden_channels))
        self.bottleneck = SharedFactorizedFfnoBottleneck(
            int(hidden_channels),
            n_blocks=max(1, int(fno_blocks)),
            modes=max(1, int(fno_modes)),
            share_spectral_weights=bool(share_spectral_weights),
            mlp_ratio=float(mlp_ratio),
            norm="instance",
        )
        if int(cnn_blocks) > 0:
            refiners: list[nn.Module] = []
            for _ in range(int(cnn_blocks)):
                refiners.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(hidden_channels),
                            int(hidden_channels),
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            int(hidden_channels),
                            int(hidden_channels),
                            kernel_size=3,
                            padding=1,
                        ),
                    )
                )
            self.refiners = nn.ModuleList(refiners)
        else:
            self.refiners = nn.ModuleList()
        self.output_proj = nn.Conv2d(
            int(hidden_channels), int(out_channels), kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        x = self.bottleneck(x)
        for refiner in self.refiners:
            x = x + refiner(x)
        return self.output_proj(x)


class _BRDTHybridResnet(nn.Module):
    """Hybrid ResNet body reused through a task-local adapter."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        fno_modes: int = 8,
        fno_blocks: int = 2,
        resnet_blocks: int = 2,
        downsample_steps: int = 1,
    ):
        super().__init__()
        from ptycho_torch.generators.fno import SpatialLifter
        from ptycho_torch.generators.hybrid_resnet import (
            AvgPoolConvDownsample,
            HybridResnetEncoderBlock,
        )
        from ptycho_torch.generators.resnet_components import (
            CycleGanUpsampler,
            ResnetBottleneck,
        )

        self.lifter = SpatialLifter(in_channels, hidden_channels)
        channels = hidden_channels
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for index in range(max(1, int(fno_blocks))):
            self.encoder_blocks.append(HybridResnetEncoderBlock(channels, modes=fno_modes))
            if index < downsample_steps:
                self.downsample_layers.append(AvgPoolConvDownsample(channels, channels * 2))
                channels *= 2
        self.bottleneck = ResnetBottleneck(channels, n_blocks=max(1, int(resnet_blocks)))
        actual_down = len(self.downsample_layers)
        widths = [channels]
        for _ in range(actual_down):
            widths.append(max(hidden_channels, widths[-1] // 2))
        self.upsample_layers = nn.ModuleList(
            [CycleGanUpsampler(widths[i], widths[i + 1]) for i in range(actual_down)]
        )
        self.output = nn.Conv2d(widths[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = self.bottleneck(x)
        for layer in self.upsample_layers:
            x = layer(x)
        return self.output(x)


def _build_fno_vanilla(
    *,
    in_channels: int,
    out_channels: int,
    grid_size: int,
    hidden_channels: int,
    fno_modes: int,
    fno_blocks: int,
) -> nn.Module:
    """Build a vanilla FNO body using the ``neuralop`` package.

    The optional dependency is gated behind ``AdapterBuildError`` so a
    missing ``neuralop`` install surfaces as a row-level blocker rather
    than a silent omission.
    """
    try:  # pragma: no cover - environment dependent
        from neuralop.models import FNO
    except Exception as exc:
        raise AdapterBuildError(
            model="fno_vanilla",
            reason="neuralop_unavailable",
            message=f"neuralop.models.FNO is unavailable: {exc}",
        ) from exc
    modes = max(1, min(int(fno_modes), int(grid_size)))
    return FNO(
        n_modes=(modes, modes),
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        hidden_channels=int(hidden_channels),
        n_layers=max(1, int(fno_blocks)),
    )


@dataclass(frozen=True)
class AdapterInfo:
    """Identification payload for a constructed adapter."""

    architecture: str
    in_channels: int
    out_channels: int
    grid_size: int
    parameter_count: int
    arch_kwargs: Dict[str, Any]


class BRDTModelAdapter(nn.Module):
    """Ordinary real-channel ``model(x) -> q_pred`` adapter for BRDT.

    The output is a single-channel real tensor in either normalized or
    physical q units (decided by the run config / loss wrapper). The
    adapter never calls the operator directly; the lightning module
    handles unnormalize-before-physics routing.
    """

    def __init__(
        self,
        *,
        architecture: str,
        in_channels: int,
        out_channels: int = 1,
        grid_size: int = dc.LOCKED_GRID_SIZE,
        arch_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"unsupported architecture={architecture!r}; "
                f"allowed: {SUPPORTED_ARCHITECTURES}"
            )
        if architecture == "classical_born_backprop":
            raise ValueError(
                "classical_born_backprop is not a neural adapter; use "
                "scripts.studies.born_rytov_dt.classical for the reference path"
            )
        self.architecture = architecture
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.grid_size = int(grid_size)
        self.arch_kwargs: Dict[str, Any] = {
            **get_default_arch_kwargs(architecture),
            **(arch_kwargs or {}),
        }

        if architecture == "unet":
            body: nn.Module = _PadCropWrapper(
                _BRDTUNet(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    hidden_channels=int(self.arch_kwargs.get("hidden_channels", 16)),
                ),
                multiple=2,
            )
        elif architecture == "fno_vanilla":
            body = _build_fno_vanilla(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                grid_size=self.grid_size,
                hidden_channels=int(self.arch_kwargs.get("hidden_channels", 16)),
                fno_modes=int(self.arch_kwargs.get("fno_modes", 8)),
                fno_blocks=int(self.arch_kwargs.get("fno_blocks", 4)),
            )
        elif architecture == "ffno":
            try:
                body = _BRDTFFNO(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    hidden_channels=int(self.arch_kwargs.get("hidden_channels", 16)),
                    fno_modes=int(self.arch_kwargs.get("fno_modes", 8)),
                    fno_blocks=int(self.arch_kwargs.get("fno_blocks", 4)),
                    share_spectral_weights=bool(
                        self.arch_kwargs.get("share_spectral_weights", False)
                    ),
                    mlp_ratio=float(self.arch_kwargs.get("mlp_ratio", 2.0)),
                    cnn_blocks=int(self.arch_kwargs.get("cnn_blocks", 2)),
                )
            except ImportError as exc:
                raise AdapterBuildError(
                    model="ffno",
                    reason="ffno_components_unavailable",
                    message=(
                        "ptycho_torch FFNO components are unavailable: "
                        f"{exc}"
                    ),
                ) from exc
        elif architecture == "hybrid_resnet":
            downsample_steps = int(self.arch_kwargs.get("downsample_steps", 1))
            body = _PadCropWrapper(
                _BRDTHybridResnet(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    hidden_channels=int(self.arch_kwargs.get("hidden_channels", 16)),
                    fno_modes=int(self.arch_kwargs.get("fno_modes", 8)),
                    fno_blocks=int(self.arch_kwargs.get("fno_blocks", 2)),
                    resnet_blocks=int(self.arch_kwargs.get("resnet_blocks", 2)),
                    downsample_steps=downsample_steps,
                ),
                multiple=2 ** max(0, downsample_steps),
            )
        else:  # pragma: no cover - guarded above
            raise ValueError(f"unsupported architecture: {architecture}")
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be 4-D (B, C, H, W); got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x has {x.shape[1]} channels; adapter built for in_channels={self.in_channels}"
            )
        if x.shape[2] != self.grid_size or x.shape[3] != self.grid_size:
            raise ValueError(
                f"x spatial shape {tuple(x.shape[2:])} != "
                f"({self.grid_size}, {self.grid_size})"
            )
        return self.body(x)

    def info(self) -> AdapterInfo:
        params = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return AdapterInfo(
            architecture=self.architecture,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            grid_size=self.grid_size,
            parameter_count=params,
            arch_kwargs=dict(self.arch_kwargs),
        )


def build_neural_adapter(
    architecture: str,
    *,
    in_channels: int,
    out_channels: int = 1,
    grid_size: int = dc.LOCKED_GRID_SIZE,
    arch_kwargs: Optional[Dict[str, Any]] = None,
) -> BRDTModelAdapter:
    """Top-level factory; raises ``AdapterBuildError`` for missing optional deps."""
    return BRDTModelAdapter(
        architecture=architecture,
        in_channels=in_channels,
        out_channels=out_channels,
        grid_size=grid_size,
        arch_kwargs=arch_kwargs,
    )

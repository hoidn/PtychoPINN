"""CycleGAN-style ResNet components used by hybrid_resnet generator."""
import math
from typing import Optional, Literal

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """CycleGAN-style ResNet block with GELU activations and a residual gate."""

    def __init__(
        self,
        channels: int,
        layerscale_init: float = 0.1,
        shared_layerscale: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
        )
        self._shared_layerscale_ref: Optional[list[nn.Parameter]] = None
        if shared_layerscale is None:
            self._layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        else:
            self.register_parameter("_layerscale", None)
            self._shared_layerscale_ref = [shared_layerscale]

    @property
    def layerscale(self) -> nn.Parameter:
        if self._shared_layerscale_ref is not None:
            return self._shared_layerscale_ref[0]
        return self._layerscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layerscale * self.block(x)


class ResnetBottleneck(nn.Module):
    """Stack of ResNet blocks at constant resolution with one shared residual gate."""

    def __init__(
        self,
        channels: int,
        n_blocks: int = 6,
        layerscale_init: float = 0.1,
        *,
        layerscale_mode: Literal["learned", "fixed"] = "learned",
        layerscale_value: Optional[float] = None,
    ):
        super().__init__()
        if layerscale_mode not in {"learned", "fixed"}:
            raise ValueError(
                f"layerscale_mode must be 'learned' or 'fixed' (got {layerscale_mode!r})."
            )
        if layerscale_mode == "learned":
            if layerscale_value is not None:
                raise ValueError(
                    "layerscale_value must be omitted when layerscale_mode='learned'."
                )
            self.layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        else:
            if layerscale_value is None:
                raise ValueError(
                    "layerscale_value must be provided when layerscale_mode='fixed'."
                )
            fixed_value = float(layerscale_value)
            if not torch.isfinite(torch.tensor(fixed_value)):
                raise ValueError(
                    f"layerscale_value must be finite when fixed (got {layerscale_value!r})."
                )
            if fixed_value <= 0.0:
                raise ValueError(
                    f"layerscale_value must be > 0 when fixed (got {layerscale_value!r})."
                )
            self.layerscale = nn.Parameter(
                torch.tensor(fixed_value),
                requires_grad=False,
            )
        self.blocks = nn.Sequential(
            *[
                ResnetBlock(channels, shared_layerscale=self.layerscale)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvNextBottleneckBlock(nn.Module):
    """ConvNeXt-style block at constant resolution.

    Topology per block (channels-first throughout):
        depthwise 7x7 -> LayerNorm (channels-first) -> 1x1 expand 4x ->
        GELU -> 1x1 project -> residual with LayerScale gamma.

    The LayerScale parameter ``layerscale`` follows the SRU-Net convention used by
    :class:`ResnetBlock` (init 0.1, learnable) so that block family is the only axis
    swapped relative to the SRU-Net + PINN baseline. The canonical tiny-ConvNeXt
    LayerScale init (1e-6) is intentionally not used by this first row.
    """

    def __init__(
        self,
        channels: int,
        layerscale_init: float = 0.1,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        shared_layerscale: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        if not math.isfinite(float(layerscale_init)) or float(layerscale_init) <= 0.0:
            raise ValueError(
                "layerscale_init must be finite and > 0 "
                f"(got {layerscale_init})."
            )
        if not math.isfinite(float(mlp_ratio)) or float(mlp_ratio) <= 0.0:
            raise ValueError(
                f"mlp_ratio must be finite and > 0 (got {mlp_ratio})."
            )
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd int (got {kernel_size})."
            )
        hidden = int(round(channels * float(mlp_ratio)))
        if hidden < 1:
            raise ValueError(
                "ConvNeXt expansion produced hidden width < 1 "
                f"(channels={channels}, mlp_ratio={mlp_ratio})."
            )
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=channels,
            padding_mode="reflect",
        )
        # InstanceNorm2d with affine=True acts as a per-channel norm operating on
        # channels-first tensors, matching the SRU-Net normalization family used
        # by ResnetBlock so the only structural change is the block topology.
        self.norm = nn.InstanceNorm2d(channels, affine=True, eps=1e-5)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)

        self._shared_layerscale_ref: Optional[list[nn.Parameter]] = None
        if shared_layerscale is None:
            self._layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        else:
            self.register_parameter("_layerscale", None)
            self._shared_layerscale_ref = [shared_layerscale]

    @property
    def layerscale(self) -> nn.Parameter:
        if self._shared_layerscale_ref is not None:
            return self._shared_layerscale_ref[0]
        return self._layerscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.depthwise(x)
        h = self.norm(h)
        h = self.expand(h)
        h = self.act(h)
        h = self.project(h)
        return residual + self.layerscale * h


class ConvNextBottleneck(nn.Module):
    """Stack of :class:`ConvNextBottleneckBlock` at constant resolution.

    Width and block count match the SRU-Net ResnetBottleneck contract so the
    bottleneck is exchangeable with :class:`ResnetBottleneck` while only swapping
    the per-block topology.
    """

    def __init__(
        self,
        channels: int,
        n_blocks: int = 6,
        layerscale_init: float = 0.1,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        *,
        layerscale_mode: Literal["learned", "fixed"] = "learned",
        layerscale_value: Optional[float] = None,
    ):
        super().__init__()
        if int(n_blocks) <= 0:
            raise ValueError(f"n_blocks must be positive (got {n_blocks}).")
        if layerscale_mode not in {"learned", "fixed"}:
            raise ValueError(
                "layerscale_mode must be 'learned' or 'fixed' "
                f"(got {layerscale_mode!r})."
            )
        if layerscale_mode == "learned":
            if layerscale_value is not None:
                raise ValueError(
                    "layerscale_value must be omitted when layerscale_mode='learned'."
                )
            self.layerscale = nn.Parameter(torch.tensor(float(layerscale_init)))
        else:
            if layerscale_value is None:
                raise ValueError(
                    "layerscale_value must be provided when layerscale_mode='fixed'."
                )
            fixed_value = float(layerscale_value)
            if not math.isfinite(fixed_value) or fixed_value <= 0.0:
                raise ValueError(
                    "layerscale_value must be finite and > 0 when fixed "
                    f"(got {layerscale_value!r})."
                )
            self.layerscale = nn.Parameter(
                torch.tensor(fixed_value),
                requires_grad=False,
            )
        self.blocks = nn.Sequential(
            *[
                ConvNextBottleneckBlock(
                    channels,
                    mlp_ratio=mlp_ratio,
                    kernel_size=kernel_size,
                    shared_layerscale=self.layerscale,
                )
                for _ in range(int(n_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CycleGanUpsampler(nn.Module):
    """CycleGAN upsampling block (ConvTranspose2d + InstanceNorm + GELU)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

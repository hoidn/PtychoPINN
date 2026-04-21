"""FFNO-close bottleneck generator pieces for PDEBench experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from ptycho_torch.generators.spectral_resnet_bottleneck import FactorizedSpectralConv2d


def _build_norm(kind: str, channels: int) -> nn.Module:
    kind = str(kind)
    if kind == "identity":
        return nn.Identity()
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True, eps=1e-5)
    if kind == "layer":
        return nn.GroupNorm(1, channels, affine=True, eps=1e-5)
    raise ValueError(f"unsupported FFNO bottleneck norm: {kind!r}")


class FactorizedFfnoBlock(nn.Module):
    """Shape-preserving FFNO-close residual block."""

    def __init__(
        self,
        *,
        channels: int,
        shared_spectral: nn.Module,
        mlp_ratio: float = 2.0,
        gate: nn.Parameter | None = None,
        norm: str = "instance",
    ):
        super().__init__()
        hidden = max(channels, int(round(channels * float(mlp_ratio))))
        self.shared_spectral = shared_spectral
        self.norm = _build_norm(norm, channels)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)
        if gate is None:
            self.gate = nn.Parameter(torch.tensor(0.1))
            self._shared_gate_ref: list[nn.Parameter] | None = None
        else:
            self.register_parameter("gate", None)
            self._shared_gate_ref = [gate]

    @property
    def effective_gate(self) -> nn.Parameter:
        if self._shared_gate_ref is not None:
            return self._shared_gate_ref[0]
        return self.gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.shared_spectral(x)
        update = self.norm(update)
        update = self.expand(update)
        update = self.act(update)
        update = self.project(update)
        return x + self.effective_gate * update


class SharedFactorizedFfnoBottleneck(nn.Module):
    """Stack of FFNO-close bottleneck blocks with optional shared spectral weights."""

    def __init__(
        self,
        channels: int,
        *,
        n_blocks: int = 6,
        modes: int = 12,
        share_spectral_weights: bool = True,
        mlp_ratio: float = 2.0,
        gate_init: float = 0.1,
        norm: str = "instance",
    ):
        super().__init__()
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}.")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}.")
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        shared_spectral = FactorizedSpectralConv2d(channels=channels, modes=modes)
        shared_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.shared_spectral = shared_spectral
        self.shared_gate = shared_gate
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            spectral = shared_spectral if share_spectral_weights else FactorizedSpectralConv2d(channels=channels, modes=modes)
            gate = shared_gate if share_spectral_weights else nn.Parameter(torch.tensor(float(gate_init)))
            self.blocks.append(
                FactorizedFfnoBlock(
                    channels=channels,
                    shared_spectral=spectral,
                    mlp_ratio=mlp_ratio,
                    gate=gate,
                    norm=norm,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FfnoBottleneckGeneratorModule(nn.Module):
    """Minimal generator-style shell for FFNO-close bottleneck unit tests."""

    def __init__(
        self,
        *,
        channels: int,
        n_blocks: int = 6,
        modes: int = 12,
        share_spectral_weights: bool = True,
        mlp_ratio: float = 2.0,
        gate_init: float = 0.1,
        norm: str = "instance",
    ):
        super().__init__()
        self.bottleneck = SharedFactorizedFfnoBottleneck(
            channels,
            n_blocks=n_blocks,
            modes=modes,
            share_spectral_weights=share_spectral_weights,
            mlp_ratio=mlp_ratio,
            gate_init=gate_init,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(x)

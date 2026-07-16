"""
FNO generators for PyTorch PINN architecture.

Implements:
- Arch A (Cascaded FNO → CNN, architecture='fno'): FNO coarse + CNN refiner

Design decisions per docs/plans/2026-01-27-modular-generator-implementation.md:
- Lifter: 2×3x3 convs with GELU (spatially aware, precedes Fourier layers)
- PtychoBlock: spectral conv + 3x3 local conv, wrapped by outer residual
- Block form: y = x + GELU(Spectral(x) + Conv3x3(x))

Dependencies:
    - neuraloperator: For SpectralConv2d and FNO building blocks
    - torch: PyTorch backend

See also:
    - ptycho_torch/generators/README.md for adding new generators
    - docs/plans/2026-01-27-modular-generator-implementation.md for architecture decisions
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Check if neuraloperator is available
try:
    from neuraloperator.layers.spectral_convolution import SpectralConv
    HAS_NEURALOPERATOR = True
except ImportError:
    HAS_NEURALOPERATOR = False


class SpatialLifter(nn.Module):
    """Lightweight spatial lifter before Fourier layers.

    Two 3x3 convs with GELU between, preserving spatial dimensions.
    This provides spatially-aware lifting before any spectral operations.

    Architecture:
        Conv2d(in, hidden, 3x3, pad=same) → GELU → Conv2d(hidden, out, 3x3, pad=same)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        input_transform: str = "none",
    ):
        super().__init__()
        hidden = hidden_channels or out_channels
        self.input_transform = InputTransform(input_transform, channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_transform(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class InputTransform(nn.Module):
    """Optional input dynamic-range transform for FNO lifters."""

    def __init__(self, mode: str = "none", channels: int = 1):
        super().__init__()
        self.mode = mode
        self.norm = None
        if mode == "instancenorm":
            self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if self.mode == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0))
        if self.mode == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        if self.mode == "instancenorm":
            return self.norm(x)
        raise ValueError(f"Unknown input transform: {self.mode}")


class PtychoBlock(nn.Module):
    """Spectral + local convolution block with outer residual.

    Architecture:
        y = x + GELU(SpectralConv(x) + Conv3x3(x))

    The outer residual provides high-frequency bypass.
    The 3x3 local path carries spatial gradients that 1x1 cannot.
    """

    def __init__(self, channels: int, modes: int = 12):
        super().__init__()
        self.channels = channels
        self.modes = modes

        # Spectral convolution (2D)
        if HAS_NEURALOPERATOR:
            self.spectral = SpectralConv(channels, channels, n_modes=(modes, modes))
        else:
            # Fallback: use FFT-based spectral conv
            self.spectral = _FallbackSpectralConv2d(channels, channels, modes)

        # Local 3x3 conv
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        spectral_out = self.spectral(x)
        local_out = self.local_conv(x)
        return x + self.act(spectral_out + local_out)


class StablePtychoBlock(nn.Module):
    """Spectral + local convolution block with LayerScale-gated residual.

    Architecture:
        update = InstanceNorm(GELU(SpectralConv(x) + Conv3x3(x)))
        y = x + layerscale * update

    Uses InstanceNorm (affine weights=1, bias=0) for centered updates, and a
    per-channel LayerScale parameter (init ~1e-3) to gate residual magnitude.
    At initialization the block is near-identity, preventing gradient explosions
    in deep FNO stacks while still allowing residual learning (unlike the
    zero-gamma approach which trapped weights on the identity manifold;
    see Finding STABLE-GAMMA-001).

    Contract: input/output shape (B, C, H, W).

    See also:
        - docs/strategy/mainstrategy.md §1.A (Norm-Last residual)
        - docs/plans/2026-01-29-layerscale-stable-hybrid.md Task 1
    """

    def __init__(self, channels: int, modes: int = 12, layerscale_init: float = 1e-3):
        super().__init__()
        self.channels = channels
        self.modes = modes

        # Spectral convolution (2D)
        if HAS_NEURALOPERATOR:
            self.spectral = SpectralConv(channels, channels, n_modes=(modes, modes))
        else:
            self.spectral = _FallbackSpectralConv2d(channels, channels, modes)

        # Local 3x3 conv
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.act = nn.GELU()

        # InstanceNorm with default affine (weight=1, bias=0)
        self.norm = nn.InstanceNorm2d(channels, affine=True, eps=1e-5)

        # LayerScale: per-channel learnable gate, small init for near-identity start
        self.layerscale = nn.Parameter(torch.full((channels,), layerscale_init))
        self._layerscale_shape = (1, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        update = self.act(self.spectral(x) + self.local_conv(x))
        update = self.norm(update)
        ls = self.layerscale.view(*self._layerscale_shape)
        return x + ls * update


class _FallbackSpectralConv2d(nn.Module):
    """Fallback spectral conv when neuraloperator is not available.

    Simple FFT-based spectral convolution for development/testing.
    Handles varying spatial dimensions by capping modes to input size.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable spectral weights
        scale = 1 / (in_channels ** 0.5)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Cap modes to actual spatial dimensions (handles downsampled inputs)
        modes_h = min(self.modes, H)
        modes_w = min(self.modes, W // 2 + 1)

        # Multiply in frequency domain (truncated to modes)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes_h, :modes_w] = torch.einsum(
            "bcxy,coxy->boxy",
            x_ft[:, :, :modes_h, :modes_w],
            self.weights[:, :, :modes_h, :modes_w]
        )

        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(H, W))


class CascadedFNOGenerator(nn.Module):
    """Cascaded FNO → CNN generator (Arch A).

    Architecture:
        - FNO stage: Produces coarse patch
        - CNN refiner: Produces final patch

    Output: Real/imag patches (B, N, N, C, 2)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        hidden_channels: int = 32,
        fno_blocks: int = 4,
        cnn_blocks: int = 2,
        modes: int = 12,
        C: int = 4,
        input_transform: str = "none",
        output_mode: str = "real_imag",
    ):
        super().__init__()
        self.C = C
        self.output_mode = output_mode

        # Lifter
        self.lifter = SpatialLifter(
            in_channels * C,
            hidden_channels,
            input_transform=input_transform,
        )

        # FNO stage
        self.fno_blocks = nn.ModuleList([
            PtychoBlock(hidden_channels, modes=modes)
            for _ in range(fno_blocks)
        ])

        # Intermediate projection
        self.intermediate = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)

        # CNN refiner stage
        self.cnn_refiner = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in range(cnn_blocks)
        ])

        # Output projection
        if self.output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(hidden_channels, C, kernel_size=1)
            self.output_phase = nn.Conv2d(hidden_channels, C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(hidden_channels, out_channels * C, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Lift
        x = self.lifter(x)

        # FNO stage (coarse)
        for block in self.fno_blocks:
            x = block(x)

        # Intermediate
        x = self.intermediate(x)

        # CNN refiner (fine)
        x = self.cnn_refiner(x)

        # Output
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase

        x = self.output_proj(x)
        # Reshape to (B, N, N, C, 2)
        x = x.view(B, 2, self.C, H, W)
        x = x.permute(0, 3, 4, 2, 1)
        return x


class FnoGenerator:
    """Generator class for Cascaded FNO architecture (Arch A).

    Implements the generator interface expected by the registry.
    """
    name = 'fno'

    def __init__(self, config):
        """Initialize the FNO generator.

        Args:
            config: TrainingConfig or InferenceConfig with model settings
        """
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> 'nn.Module':
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)

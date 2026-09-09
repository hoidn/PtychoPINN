"""Matched-compute global-attention generator for CDI."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import InputTransform


def _rotate_pairs(value: torch.Tensor) -> torch.Tensor:
    paired = value.reshape(*value.shape[:-1], -1, 2)
    return torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)


def _rope_tables(grid_size: int, head_dim: int, registers: int) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 4:
        raise ValueError("attention head dimension must be divisible by 4 for 2D rotary positions")
    axis_dim = head_dim // 2
    inv_frequency = 1.0 / (
        10_000 ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
    )
    y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    angles = torch.cat(
        (
            torch.repeat_interleave(y.flatten()[:, None] * inv_frequency, 2, dim=-1),
            torch.repeat_interleave(x.flatten()[:, None] * inv_frequency, 2, dim=-1),
        ),
        dim=-1,
    )
    angles = F.pad(angles, (0, 0, registers, 0))
    return angles.cos()[None, None], angles.sin()[None, None]


class GlobalAttention(nn.Module):
    """Bias-free QKV attention with per-head QK normalization and 2D RoPE."""

    def __init__(self, width: int, heads: int, grid_size: int, registers: int):
        super().__init__()
        if width % heads:
            raise ValueError("ViT width must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = width // heads
        cosine, sine = _rope_tables(grid_size, self.head_dim, registers)
        self.register_buffer("rope_cos", cosine, persistent=False)
        self.register_buffer("rope_sin", sine, persistent=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.projection = nn.Linear(width, width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        cosine = self.rope_cos.to(dtype=query.dtype)
        sine = self.rope_sin.to(dtype=query.dtype)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = query * cosine + _rotate_pairs(query) * sine
        key = key * cosine + _rotate_pairs(key) * sine
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, count, width)
        return self.projection(attended)


class VitBlock(nn.Module):
    """Pre-norm ViT block with LayerScale on attention and MLP residuals."""

    def __init__(self, width: int, heads: int, grid_size: int, registers: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.attention = GlobalAttention(width, heads, grid_size, registers)
        self.scale_attention = nn.Parameter(torch.full((width,), 1e-5))
        self.norm2 = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.scale_mlp = nn.Parameter(torch.full((width,), 1e-5))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.scale_attention * self.attention(self.norm1(tokens))
        return tokens + self.scale_mlp * self.mlp(self.norm2(tokens))


class VitGeneratorModule(nn.Module):
    """Plain ViT-5-style baseline fixed to the N=128, C=1 comparison contract."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 2,
        patch_size: int = 4,
        width: int = 256,
        depth: int = 12,
        heads: int = 4,
        C: int = 1,
        input_transform: str = "none",
        output_mode: str = "real_imag",
    ):
        super().__init__()
        if C != 1:
            raise ValueError(f"vit only supports the locked C=1 CDI contract; got C={C}")
        if in_channels != 1 or out_channels != 2:
            raise ValueError("vit requires one input channel and two output components")
        if patch_size <= 0 or 128 % patch_size:
            raise ValueError("vit patch_size must be a positive divisor of 128")
        if depth <= 0:
            raise ValueError("vit depth must be positive")
        if heads <= 0 or width % heads or (width // heads) % 4:
            raise ValueError("vit width must be divisible by heads with head dimension divisible by 4")
        if output_mode not in {"real_imag", "amp_phase"}:
            raise ValueError("vit output_mode must be 'real_imag' or 'amp_phase'")

        self.patch_size = patch_size
        self.num_heads = heads
        self.num_register_tokens = 4
        self.image_size = 128
        self.grid_size = self.image_size // self.patch_size
        self.output_mode = output_mode
        self.input_transform = InputTransform(input_transform, channels=in_channels)
        self.patch_embedding = nn.Conv2d(
            in_channels,
            width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        token_count = self.grid_size**2 + self.num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.num_register_tokens, width)
        )
        self.absolute_position = nn.Parameter(torch.zeros(1, token_count, width))
        self.blocks = nn.ModuleList(
            [
                VitBlock(
                    width,
                    self.num_heads,
                    self.grid_size,
                    self.num_register_tokens,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(width)
        self.amplitude_decoder = nn.Conv2d(
            width,
            self.patch_size**2,
            kernel_size=1,
        )
        self.phase_decoder = nn.Conv2d(
            width,
            self.patch_size**2,
            kernel_size=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(self.patch_size)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.absolute_position, std=0.02)

    def forward(self, diffraction: torch.Tensor):
        if diffraction.ndim != 4 or tuple(diffraction.shape[1:]) != (1, 128, 128):
            raise ValueError(
                "vit expects input shape (B, 1, 128, 128); "
                f"got {tuple(diffraction.shape)}"
            )
        spatial = self.patch_embedding(self.input_transform(diffraction))
        batch, width, height, width_tokens = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2)
        registers = self.register_tokens.expand(batch, -1, -1)
        tokens = torch.cat((registers, tokens), dim=1) + self.absolute_position
        for block in self.blocks:
            tokens = block(tokens)
        spatial = self.norm(tokens[:, self.num_register_tokens :])
        spatial = spatial.transpose(1, 2).reshape(batch, width, height, width_tokens)
        first = self.pixel_shuffle(self.amplitude_decoder(spatial))
        second = self.pixel_shuffle(self.phase_decoder(spatial))
        if self.output_mode == "amp_phase":
            return torch.sigmoid(first), math.pi * torch.tanh(second)
        return torch.stack((first, second), dim=-1).permute(0, 2, 3, 1, 4)

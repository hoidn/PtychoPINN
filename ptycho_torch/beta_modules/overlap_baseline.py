"""Overlap Baseline: Pairwise-Offset Gated Mixing + PixelShuffle Decoder.

Minimal baseline for latent-space overlap fusion. Weight-shared FNO+CNN
encoder (interleaved, same as PC-CCNF) processes each patch independently,
a Fourier-encoded pairwise offset gate mixes features across overlapping
patches at the bottleneck, and a PixelShuffle+ConvNeXt decoder with
encoder skip injection reconstructs each patch.

Compared to the full PC-CCNF (latent_model.py), this model has:
- Same FNO+CNN encoder (shared architecture for fair comparison)
- No spatial canvas or canvas scaling
- No cross-attention fusion — only a small pairwise MLP for overlap mixing
- No multi-resolution feature volume
- Per-patch PixelShuffle decoder (no canvas crop step)
"""

import copy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
from ptycho_torch.model import (
    ForwardModel, IntensityScalerModule, CombineComplexRectangular,
)
from ptycho_torch.model_attention import CBAM
from ptycho_torch.beta_modules.latent_model import (
    FourierPositionalEncoding,
    FNOCNNEncoder,
    LayerNorm2d,
    PixelShuffleUpsampleStage,
)

import math


class OverlapHybridEncoding(nn.Module):
    """Hybrid positional encoding combining overlap fractions with Fourier features.

    Per-axis features (all use |delta| for symmetric gates):
      - max(0, 1-|delta|)   : linear overlap fraction
      - max(0, 1-|delta|)^2 : quadratic taper
      - sin/cos(pi*|delta|*f) for f in {8, 16} : high-frequency Fourier
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('freqs', torch.tensor([8.0, 16.0]))

    @property
    def output_dim(self):
        return 12  # 2 axes x (2 overlap + 2 sin + 2 cos)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta: (..., 2) pairwise offsets already normalized by N
        Returns:
            (..., 12) hybrid features
        """
        abs_delta = delta.abs()  # (..., 2)

        ov = (1.0 - abs_delta).clamp(min=0.0)  # (..., 2)
        ov_sq = ov * ov  # (..., 2)

        # (..., 2, 1) * (F,) -> (..., 2, F)
        x = abs_delta.unsqueeze(-1) * self.freqs * math.pi
        s = torch.sin(x)  # (..., 2, F)
        c = torch.cos(x)  # (..., 2, F)

        # Per-axis concat: [ov, ov_sq, sin_f1, sin_f2, cos_f1, cos_f2] for each axis
        # Then flatten across axes
        fourier = torch.cat([s, c], dim=-1)  # (..., 2, 2F)
        fourier = fourier.reshape(*delta.shape[:-1], -1)  # (..., 4F)

        overlap = torch.cat([ov, ov_sq], dim=-1)  # (..., 4)

        return torch.cat([overlap, fourier], dim=-1)  # (..., 4 + 4F) = (..., 12)



class PairwiseOffsetGate(nn.Module):
    """Pairwise overlap mixing via Fourier-encoded offset gates.

    For each pair of patches (i, j), computes their normalized relative
    offset, Fourier-encodes it, and projects to a per-channel-group gate.
    The gated features are combined via weighted sum across all patches.

    This is C-agnostic: the same learned gate function applies to any
    number of overlapping patches at training or inference time.
    """
    def __init__(self, bottleneck_channels: int, num_bands: int = 4,
                 num_groups: int = 16, encoding: str = 'fourier'):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.num_groups = min(num_groups, bottleneck_channels)
        self.channels_per_group = bottleneck_channels // self.num_groups
        self.encoding = encoding

        if encoding == 'hybrid':
            self.enc = OverlapHybridEncoding()
            enc_dim = self.enc.output_dim
            hidden = max(num_groups, 16)
            self.gate_proj = nn.Sequential(
                nn.Linear(enc_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.num_groups),
            )
        else:
            self.enc = FourierPositionalEncoding(
                num_bands=num_bands, max_freq_log2=num_bands - 1
            )
            self.gate_proj = nn.Linear(self.enc.output_dim, self.num_groups)

    def forward(self, z: torch.Tensor, coords: torch.Tensor,
                N: int) -> torch.Tensor:
        """
        Args:
            z: (B, C, D, H, W) per-patch bottleneck features
            coords: (B, C, 1, 2) relative coordinates in pixel units
            N: diffraction pattern size for normalization
        Returns:
            z_mixed: (B, C, D, H, W) overlap-mixed features
        """
        B, C, D, H, W = z.shape
        coords_flat = coords.squeeze(2)  # (B, C, 2)

        # Pairwise offsets: delta_ij = coord_i - coord_j, normalized by N
        # (B, C, 1, 2) - (B, 1, C, 2) -> (B, C, C, 2)
        delta = (coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)) / N

        # Encode offsets and project to group gates
        gamma = self.enc(delta)
        gate_groups = torch.sigmoid(self.gate_proj(gamma))  # (B, C, C, G)

        # Expand groups to full channel dim: (B, C, C, G) -> (B, C, C, D)
        gate = gate_groups.repeat_interleave(self.channels_per_group, dim=-1)
        if gate.shape[-1] < D:
            gate = F.pad(gate, (0, D - gate.shape[-1]), value=1.0)

        # Weighted mixing: z'_i = sum_j(g_ij * z_j) / sum_j(g_ij)
        # z: (B, C, D, H, W) -> (B, C, D, H*W)
        z_flat = z.reshape(B, C, D, H * W)

        # gate: (B, C_i, C_j, D) @ z: (B, C_j, D, HW) -> (B, C_i, D, HW)
        # Weighted sum over j for each i
        # gate[b, i, j, d] * z[b, j, d, hw] -> sum over j
        gate_sum = gate.sum(dim=2, keepdim=True).clamp(min=1e-6)  # (B, C, 1, D)
        gate_norm = gate / gate_sum  # (B, C, C, D)

        # Einstein summation: mix features across C dimension
        z_mixed = torch.einsum('bijd,bjdp->bidp', gate_norm, z_flat)
        return z_mixed.reshape(B, C, D, H, W)


class PerPatchPixelShuffleDecoder(nn.Module):
    """Per-patch PixelShuffle+ConvNeXt decoder with encoder skip injection.

    Adapted from CNNCanvasDecoder (latent_model.py) with the canvas crop
    step removed. Takes per-patch bottleneck features directly and upsamples
    to full resolution via PixelShuffle stages with ConvNeXt refinement.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 bottleneck_channels: int, H_enc: int,
                 encoder_skip_info: Optional[list] = None):
        super().__init__()
        self.N = data_config.N
        self.H_enc = H_enc

        F_in = bottleneck_channels
        base_ch = getattr(model_config, 'cnn_decoder_base_ch', 128)
        n_stages = int(math.log2(self.N / H_enc))

        assert self.N == H_enc * (2 ** n_stages), (
            f"N={self.N} must be H_enc * 2^k for integer k"
        )

        self.stem = nn.Sequential(
            nn.Conv2d(F_in, base_ch, kernel_size=1),
            LayerNorm2d(base_ch),
            nn.GELU(),
        )

        channels = [base_ch]
        ch = base_ch
        for s in range(n_stages):
            ch_out = max(16, ch // 2) if s > 0 else ch
            channels.append(ch_out)
            ch = ch_out

        skip_channels_per_stage = [0] * n_stages
        self._skip_indices = [None] * n_stages
        if encoder_skip_info is not None:
            for s in range(n_stages):
                target_res = H_enc * (2 ** (s + 1))
                for k in range(len(encoder_skip_info) - 1, -1, -1):
                    if encoder_skip_info[k][0] == target_res:
                        skip_channels_per_stage[s] = encoder_skip_info[k][1]
                        self._skip_indices[s] = k
                        break

        self.stages = nn.ModuleList()
        for s in range(n_stages):
            self.stages.append(
                PixelShuffleUpsampleStage(channels[s], channels[s + 1],
                                          skip_channels=skip_channels_per_stage[s])
            )

        last_ch = channels[-1]
        self.head_real = nn.Conv2d(last_ch, 1, kernel_size=3, padding=1)
        self.head_imag = nn.Conv2d(last_ch, 1, kernel_size=3, padding=1)

        nn.init.zeros_(self.head_real.bias)
        nn.init.zeros_(self.head_imag.bias)

    def forward(self, z: torch.Tensor,
                encoder_skips: Optional[list] = None) -> tuple:
        """
        Args:
            z: (B*C, D, H_enc, W_enc) per-patch bottleneck features
            encoder_skips: list of (B*C, ch_k, res_k, res_k) encoder skips
        Returns:
            x_real: (B*C, 1, N, N) real component
            x_imag: (B*C, 1, N, N) imaginary component
        """
        stage_skips = [None] * len(self.stages)
        if encoder_skips is not None:
            for s, skip_idx in enumerate(self._skip_indices):
                if skip_idx is not None:
                    stage_skips[s] = encoder_skips[skip_idx]

        x = self.stem(z)
        for i, stage in enumerate(self.stages):
            x = stage(x, skip=stage_skips[i])

        x_real = 0.2 + torch.tanh(self.head_real(x))
        x_imag = 1.2 * torch.tanh(self.head_imag(x))

        return x_real, x_imag


class AutoencoderOverlapBaseline(nn.Module):
    """Weight-shared FNO+CNN encoder + pairwise offset gate + PixelShuffle decoder.

    The encoder processes each patch independently (weight-shared).
    The pairwise offset gate mixes features across overlapping patches
    in the bottleneck. A PixelShuffle+ConvNeXt decoder reconstructs each
    patch from the mixed features with encoder skip injection.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        # Weight-shared FNO+CNN encoder (interleaved)
        encoder_config = copy.deepcopy(model_config)
        object.__setattr__(encoder_config, 'C_model', 1)
        object.__setattr__(encoder_config, 'object_big', False)
        object.__setattr__(encoder_config, 'fno_interleave', True)
        self.encoder = FNOCNNEncoder(encoder_config, data_config)

        # Probe encoder output shapes for decoder construction
        N = data_config.N
        with torch.no_grad():
            dummy = torch.zeros(1, 1, N, N)
            dummy_out, dummy_skips = self.encoder(dummy)
        H_enc = dummy_out.shape[2]
        encoder_skip_info = [(s.shape[2], s.shape[1]) for s in dummy_skips]

        # Optional CBAM at bottleneck
        bottleneck_channels = self.encoder.filters[-1]
        if model_config.cbam_bottleneck:
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity()

        # Pairwise offset gate for overlap mixing
        num_bands = getattr(model_config, 'overlap_fourier_bands', 4)
        num_groups = getattr(model_config, 'overlap_gate_groups', 16)
        encoding = getattr(model_config, 'overlap_encoding', 'fourier')
        self.overlap_gate = PairwiseOffsetGate(
            bottleneck_channels, num_bands=num_bands, num_groups=num_groups,
            encoding=encoding
        )

        # PixelShuffle decoder with encoder skip injection
        self.decoder = PerPatchPixelShuffleDecoder(
            model_config, data_config,
            bottleneck_channels=bottleneck_channels,
            H_enc=H_enc,
            encoder_skip_info=encoder_skip_info,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                probe: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: (B, C, N, N) diffraction patterns (already intensity-scaled)
            coords: (B, C, 1, 2) relative coordinates (pixel units)
            probe: unused, accepted for interface compatibility
        Returns:
            complex_out: (B, C, N, N) complex64 object patches
            x_real: (B, C, N, N) real component
            x_imag: (B, C, N, N) imaginary component
        """
        B, C, N, _ = x.shape

        # 1. Weight-shared FNO+CNN encoding
        x_flat = x.reshape(B * C, 1, N, N)
        z_flat, encoder_skips = self.encoder(x_flat)  # (B*C, D, H, W)
        z_flat = self.bottleneck_cbam(z_flat)
        D, H, W = z_flat.shape[1], z_flat.shape[2], z_flat.shape[3]
        z = z_flat.reshape(B, C, D, H, W)

        # 2. Pairwise offset gate mixing
        z_mixed = self.overlap_gate(z, coords, N)

        # 3. Per-patch PixelShuffle decoding
        z_dec = z_mixed.reshape(B * C, D, H, W)
        x_real, x_imag = self.decoder(z_dec, encoder_skips=encoder_skips)

        x_real = x_real.reshape(B, C, N, N)
        x_imag = x_imag.reshape(B, C, N, N)

        complex_out = x_real + 1j * x_imag

        return complex_out, x_real, x_imag


class PtychoPINN_OverlapBaseline(nn.Module):
    """Full PtychoPINN module using the overlap baseline architecture."""
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        self.autoencoder = AutoencoderOverlapBaseline(model_config, data_config)
        self.forward_model = ForwardModel(model_config, data_config)

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor,
                experiment_ids=None, fine_tune=False):
        x = self.scaler.scale(x, input_scale_factor)

        x_combined, x_real, x_imag = self.autoencoder(x, positions, probe)

        x_out = self.forward_model.forward(
            x_combined, x,
            positions, probe / self.probe_scale, output_scale_factor,
            experiment_ids=experiment_ids,
            fine_tune=fine_tune
        )

        return x_out, x_real, x_imag

    def forward_predict(self, x, positions, probe, input_scale_factor):
        x = self.scaler.scale(x, input_scale_factor)
        x_combined, _, _ = self.autoencoder(x, positions, probe)
        return x_combined

    def get_encoder_bottom_params(self):
        encoder = self.autoencoder.encoder
        if not hasattr(encoder, 'blocks') or len(encoder.blocks) == 0:
            return []
        split_idx = len(encoder.blocks) // 2
        params = []
        for block in encoder.blocks[:split_idx]:
            params.extend(block.parameters())
        return params

    def get_encoder_top_params(self):
        encoder = self.autoencoder.encoder
        if not hasattr(encoder, 'blocks') or len(encoder.blocks) == 0:
            return []
        split_idx = len(encoder.blocks) // 2
        params = []
        for block in encoder.blocks[split_idx:]:
            params.extend(block.parameters())
        return params

    def freeze_encoder(self):
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

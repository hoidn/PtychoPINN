"""Input-Conditioned Overlap Model: Spatial + Probe Conditioning.

Extends the loss-only overlap baseline with auxiliary input channels that
provide geometric and illumination context to the encoder:

1. Spatial encoding (2 channels per patch = 2C): Per-pixel sinusoidal
   offset from the group centroid — sin(2*pi*x/N), sin(2*pi*y/N).
   Encodes each pixel's position relative to the scan center so the
   network can learn geometry-aware overlap weighting.

2. Probe intensity (1 shared channel): Normalized sum |P_p|^2 over
   incoherent modes. Provides the illumination map so the network
   can learn to weight pixels by measurement importance.

Total encoder input: (B, 3C+1, N, N).
Decoder output unchanged: (B, C, N, N) complex object patches.
"""

import copy
import math
from typing import Optional

import torch
from torch import nn

from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
from ptycho_torch.model import ForwardModel, IntensityScalerModule
from ptycho_torch.model_attention import CBAM
from ptycho_torch.beta_modules.latent_model import FNOCNNEncoder
from ptycho_torch.beta_modules.loss_overlap_baseline import MultiChannelPixelShuffleDecoder


class SpatialEncoder(nn.Module):
    """Per-pixel sinusoidal offset maps from patch-level relative coordinates.

    For each of the C patches, computes the offset of every pixel from the
    group centroid (arithmetic mean of scan positions), then encodes as
    sin(2*pi*offset/N). Produces 2 channels per patch (x and y).
    """
    def __init__(self, N: int):
        super().__init__()
        self.N = N
        grid_1d = torch.arange(N, dtype=torch.float32) - (N - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())

    def forward(self, coords_relative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_relative: (B, C, 1, 2) offsets from group centroid.
        Returns:
            (B, 2C, N, N) sinusoidal spatial encoding.
        """
        dx = coords_relative[:, :, 0, 0]  # (B, C)
        dy = coords_relative[:, :, 0, 1]  # (B, C)

        x_offset = dx[:, :, None, None] + self.grid_x  # (B, C, N, N)
        y_offset = dy[:, :, None, None] + self.grid_y  # (B, C, N, N)

        u = 2.0 * math.pi * x_offset / self.N
        v = 2.0 * math.pi * y_offset / self.N

        return torch.cat([torch.sin(u), torch.sin(v)], dim=1)  # (B, 2C, N, N)


class FNOCNNEncoderConditioned(FNOCNNEncoder):
    """FNOCNNEncoder with lifting layer sized for conditioned input.

    Lifting: Conv2d(3C+1, fno_width, 1) to accept C diffraction channels,
    2C spatial encoding channels, and 1 probe intensity channel.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(model_config, data_config)
        C = data_config.C
        n_input = 3 * C + 1
        fno_width = model_config.fno_width
        self.lifting = nn.Conv2d(n_input, fno_width, kernel_size=1)
        self.filters[0] = n_input


class AutoencoderConditionedOverlap(nn.Module):
    """Multi-channel encoder with spatial + probe intensity conditioning.

    Input is augmented from (B, C, N, N) to (B, 3C+1, N, N) before the
    encoder. The decoder reconstructs C complex object patches as before.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        N = data_config.N
        C = data_config.C
        n_input = 3 * C + 1

        self.spatial_encoder = SpatialEncoder(N)

        encoder_config = copy.deepcopy(model_config)
        object.__setattr__(encoder_config, 'fno_interleave', True)
        self.encoder = FNOCNNEncoderConditioned(encoder_config, data_config)

        with torch.no_grad():
            dummy = torch.zeros(1, n_input, N, N)
            dummy_out, dummy_skips = self.encoder(dummy)
        H_enc = dummy_out.shape[2]
        encoder_skip_info = [(s.shape[2], s.shape[1]) for s in dummy_skips]

        bottleneck_channels = self.encoder.filters[-1]
        if model_config.cbam_bottleneck:
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity()

        self.decoder = MultiChannelPixelShuffleDecoder(
            model_config, data_config,
            bottleneck_channels=bottleneck_channels,
            H_enc=H_enc,
            encoder_skip_info=encoder_skip_info,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                probe_intensity: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, N, N) diffraction patterns (already intensity-scaled).
            coords: (B, C, 1, 2) relative coordinates from group centroid.
            probe_intensity: (B, 1, N, N) normalized probe intensity map.
        Returns:
            complex_out: (B, C, N, N) complex64 object patches.
            x_real: (B, C, N, N) real component.
            x_imag: (B, C, N, N) imaginary component.
        """
        spatial = self.spatial_encoder(coords)  # (B, 2C, N, N)
        encoder_input = torch.cat([x, spatial, probe_intensity], dim=1)  # (B, 3C+1, N, N)

        z, encoder_skips = self.encoder(encoder_input)
        z = self.bottleneck_cbam(z)
        x_real, x_imag = self.decoder(z, encoder_skips=encoder_skips)
        complex_out = x_real + 1j * x_imag
        return complex_out, x_real, x_imag


class PtychoPINN_ConditionedOverlap(nn.Module):
    """Full PtychoPINN module with input-conditioned overlap model."""
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        self.autoencoder = AutoencoderConditionedOverlap(model_config, data_config)
        self.forward_model = ForwardModel(model_config, data_config)

    def _compute_probe_intensity(self, probe: torch.Tensor) -> torch.Tensor:
        """Fallback: compute normalized probe intensity from raw probe tensor.

        Args:
            probe: (B, C, P, N, N) complex64.
        Returns:
            (B, 1, N, N) float32, normalized to [0, 1].
        """
        intensity = torch.sum(torch.abs(probe) ** 2, dim=2)  # (B, C, N, N)
        intensity = intensity[:, 0:1, :, :]  # (B, 1, N, N)
        imax = intensity.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        return intensity / imax

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor,
                experiment_ids=None, fine_tune=False, probe_intensity=None):
        x = self.scaler.scale(x, input_scale_factor)

        if probe_intensity is None:
            probe_intensity = self._compute_probe_intensity(probe)

        x_combined, x_real, x_imag = self.autoencoder(x, positions, probe_intensity)

        x_out = self.forward_model.forward(
            x_combined, x,
            positions, probe / self.probe_scale, output_scale_factor,
            experiment_ids=experiment_ids,
            fine_tune=fine_tune
        )

        return x_out, x_real, x_imag

    def forward_predict(self, x, positions, probe, input_scale_factor,
                        probe_intensity=None):
        x = self.scaler.scale(x, input_scale_factor)
        if probe_intensity is None:
            probe_intensity = self._compute_probe_intensity(probe)
        x_combined, _, _ = self.autoencoder(x, positions, probe_intensity)
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

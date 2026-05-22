"""Input-Conditioned Overlap Model: Spatial + Probe Conditioning.

Extends the loss-only overlap baseline with auxiliary input channels that
provide geometric and illumination context to the encoder:

1. Spatial encoding (2 channels per patch = 2C):
   - 'sinusoidal': sin(2*pi*offset/N) for x and y.  Range [-1, 1].
   - 'linear':     offset/N for x and y.  Range ~[-0.5, 0.5].
   Controlled by ModelConfig.coord_encoding.

2. Probe conditioning (1 or 2 channels):
   - 'intensity' (1 ch): Normalized |P|^2.  Range [0, 1].
   - 'cartesian' (2 ch): Re(P)/|P|_max and Im(P)/|P|_max.  Range [-1, 1].
   - 'polar'     (2 ch): |P|/|P|_max [0, 1] and angle(P)/pi [-1, 1].
   Controlled by ModelConfig.probe_encoding.

Total encoder input: (B, 3C + n_probe_ch, N, N)
  where n_probe_ch = 1 (intensity) or 2 (cartesian/polar).
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
    """Per-pixel offset maps from patch-level relative coordinates.

    For each of the C patches, computes the offset of every pixel from the
    group centroid (arithmetic mean of scan positions), then encodes as:
      - 'sinusoidal': sin(2*pi*offset/N).  Range [-1, 1].
      - 'linear':     offset/N.            Range ~[-0.5, 0.5] for typical offsets.
    Produces 2 channels per patch (x and y).
    """
    def __init__(self, N: int, coord_encoding: str = 'sinusoidal'):
        super().__init__()
        self.N = N
        self.coord_encoding = coord_encoding
        grid_1d = torch.arange(N, dtype=torch.float32) - (N - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())

    def forward(self, coords_relative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_relative: (B, C, 1, 2) offsets from group centroid.
        Returns:
            (B, 2C, N, N) spatial encoding.
        """
        dx = coords_relative[:, :, 0, 0]  # (B, C)
        dy = coords_relative[:, :, 0, 1]  # (B, C)

        x_offset = dx[:, :, None, None] + self.grid_x  # (B, C, N, N)
        y_offset = dy[:, :, None, None] + self.grid_y  # (B, C, N, N)

        if self.coord_encoding == 'sinusoidal':
            u = 2.0 * math.pi * x_offset / self.N
            v = 2.0 * math.pi * y_offset / self.N
            return torch.cat([torch.sin(u), torch.sin(v)], dim=1)
        else:
            u = x_offset / self.N
            v = y_offset / self.N
            return torch.cat([u, v], dim=1)  # (B, 2C, N, N)


class FNOCNNEncoderConditioned(FNOCNNEncoder):
    """FNOCNNEncoder with lifting layer sized for conditioned input.

    Lifting: Conv2d(n_input, fno_width, 1) where n_input = C + 2C + n_probe_ch.
    n_probe_ch is 1 for intensity encoding, 2 for cartesian/polar.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(model_config, data_config)
        C = data_config.C
        n_probe_ch = 2 if model_config.probe_encoding in ('cartesian', 'polar') else 1
        n_input = 3 * C + n_probe_ch
        fno_width = model_config.fno_width
        self.lifting = nn.Conv2d(n_input, fno_width, kernel_size=1)
        self.filters[0] = n_input


class AutoencoderConditionedOverlap(nn.Module):
    """Multi-channel encoder with spatial + probe conditioning.

    Input is augmented from (B, C, N, N) to (B, 3C + n_probe_ch, N, N)
    before the encoder. n_probe_ch is 1 (intensity) or 2 (cartesian/polar).
    The decoder reconstructs C complex object patches as before.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        N = data_config.N
        C = data_config.C
        n_probe_ch = 2 if model_config.probe_encoding in ('cartesian', 'polar') else 1
        n_input = 3 * C + n_probe_ch

        self.spatial_encoder = SpatialEncoder(N, coord_encoding=model_config.coord_encoding)

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
                probe_encoding: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, N, N) diffraction patterns (already intensity-scaled).
            coords: (B, C, 1, 2) relative coordinates from group centroid.
            probe_encoding: (B, n_probe_ch, N, N) probe conditioning channels.
        Returns:
            complex_out: (B, C, N, N) complex64 object patches.
            x_real: (B, C, N, N) real component.
            x_imag: (B, C, N, N) imaginary component.
        """
        spatial = self.spatial_encoder(coords)  # (B, 2C, N, N)
        encoder_input = torch.cat([x, spatial, probe_encoding], dim=1)

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
        self.probe_encoding = model_config.probe_encoding

        self.autoencoder = AutoencoderConditionedOverlap(model_config, data_config)
        self.forward_model = ForwardModel(model_config, data_config)

    def _compute_probe_encoding(self, probe: torch.Tensor) -> torch.Tensor:
        """Compute probe conditioning channels from raw probe tensor.

        Args:
            probe: (B, C, P, N, N) complex64, or (B, C, P, N, N, 2) real view
                   when DataParallel converts complex tensors.
        Returns:
            (B, n_probe_ch, N, N) float32.
            n_probe_ch is 1 for 'intensity', 2 for 'cartesian'/'polar'.
        """
        # DataParallel may convert complex to real view (..., 2)
        if not probe.is_complex() and probe.shape[-1] == 2:
            probe = torch.view_as_complex(probe.contiguous())

        if self.probe_encoding == 'intensity':
            intensity = torch.sum(torch.abs(probe) ** 2, dim=2)  # (B, C, N, N)
            intensity = intensity[:, 0:1, :, :]  # (B, 1, N, N)
            imax = intensity.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            return intensity / imax

        P_dom = probe[:, 0, 0]  # (B, N, N) complex — dominant mode
        P_max = torch.abs(P_dom).amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)

        if self.probe_encoding == 'cartesian':
            re = P_dom.real / P_max  # (B, N, N), range [-1, 1]
            im = P_dom.imag / P_max  # (B, N, N), range [-1, 1]
            return torch.stack([re, im], dim=1)  # (B, 2, N, N)
        else:  # polar
            amp = torch.abs(P_dom) / P_max  # (B, N, N), range [0, 1]
            phase = torch.angle(P_dom) / math.pi  # (B, N, N), range [-1, 1]
            print(amp.dtype, phase.dtype)
            return torch.stack([amp, phase], dim=1)  # (B, 2, N, N)

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor,
                experiment_ids=None, fine_tune=False, probe_intensity=None):
        x = self.scaler.scale(x, input_scale_factor)

        if self.probe_encoding != 'intensity' or probe_intensity is None:
            probe_encoding = self._compute_probe_encoding(probe)
            
        else:
            probe_encoding = probe_intensity

        x_combined, x_real, x_imag = self.autoencoder(x, positions, probe_encoding)

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
        if self.probe_encoding != 'intensity' or probe_intensity is None:
            probe_encoding = self._compute_probe_encoding(probe)
            print(f"Probe encoding shape: {probe_encoding.shape}")
        else:
            probe_encoding = probe_intensity
        x_combined, _, _ = self.autoencoder(x, positions, probe_encoding)
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

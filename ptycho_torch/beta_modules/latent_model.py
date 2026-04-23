"""PC-CCNF Architecture: Position-Conditioned Shared Canvas + Neural Field Decoder.

Hybrid architecture combining latent-space canvas assembly (from PC-SCD)
with a coordinate-conditioned neural field decoder (from CCNF) for
ptychographic reconstruction with explicit positional conditioning
and structural phase consistency.
"""

import math
import copy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
from ptycho_torch.model import Encoder, ForwardModel, IntensityScalerModule
from ptycho_torch.model_attention import CBAM
import ptycho_torch.helper as hh


class FourierPositionalEncoding(nn.Module):
    """Fourier feature encoding for spatial coordinates.

    Maps 2D coordinates to high-dimensional features using sinusoidal functions
    at multiple frequency bands. Supports progressive frequency annealing.
    """
    def __init__(self, num_bands: int = 10, max_freq_log2: Optional[int] = None):
        super().__init__()
        self.num_bands = num_bands
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1
        freqs = 2.0 ** torch.linspace(0, max_freq_log2, num_bands)
        self.register_buffer('freqs', freqs)
        self.active_bands = num_bands

    def set_active_bands(self, n: int):
        self.active_bands = min(n, self.num_bands)

    @property
    def output_dim(self):
        return 2 * 2 * self.num_bands  # 2 coords × (sin + cos) × num_bands

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (..., 2) spatial coordinates
        Returns:
            (..., 4 * num_bands) Fourier features
        """
        freqs = self.freqs[:self.active_bands]
        # coords: (..., 2), freqs: (L,) -> (..., 2, L)
        x = coords.unsqueeze(-1) * freqs * math.pi
        # (..., 2, L) -> (..., 2*L) for sin and cos each
        features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        # Flatten last two dims: (..., 2, 2*L) -> (..., 4*L)
        features = features.reshape(*coords.shape[:-1], -1)
        if self.active_bands < self.num_bands:
            pad_size = 4 * (self.num_bands - self.active_bands)
            features = F.pad(features, (0, pad_size), value=0.0)
        return features


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation conditioned on spatial coordinates.

    Encodes patch positions via Fourier features, then produces per-patch
    scale (gamma) and shift (beta) vectors to modulate latent feature maps.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.pos_encoder = FourierPositionalEncoding(
            num_bands=model_config.fourier_bands_film
        )
        pos_dim = self.pos_encoder.output_dim
        latent_channels = model_config.n_filters_scale * 128

        self.mlp = nn.Sequential(
            nn.Linear(pos_dim, model_config.film_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_config.film_dim, 2 * latent_channels)
        )
        self.latent_channels = latent_channels

    def forward(self, latent: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, C, D, H, W) per-patch latent feature maps
            coords: (B, C, 2) relative coordinates per patch
        Returns:
            (B, C, D, H, W) conditioned latent
        """
        pos_features = self.pos_encoder(coords)       # (B, C, pos_dim)
        gamma_beta = self.mlp(pos_features)            # (B, C, 2*D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)      # each (B, C, D)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)      # (B, C, D, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return latent * torch.sigmoid(gamma) + beta


class LatentCanvasAssembler(nn.Module):
    """Assembles per-patch latents onto a shared spatial canvas.

    Translates each patch's latent feature map to its correct position
    on a larger canvas using the existing Translation() infrastructure,
    then averages overlapping contributions.
    """
    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        M = hh.get_padded_size(data_config, model_config)
        self.canvas_size = math.ceil(M / 8)
        self.spatial_size = 8  # encoder output is 8x8 for N=64

    def forward(self, latents: torch.Tensor, coords: torch.Tensor,
                probe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            latents: (B, C, D, 8, 8) position-conditioned latent feature maps
            coords: (B, C, 1, 2) relative coordinates
            probe: (B, C, P, N, N) complex probe (optional, for weighting)
        Returns:
            canvas: (B, D, M_lat, M_lat) assembled latent canvas
        """
        B, C, D, H, W = latents.shape
        M_lat = self.canvas_size
        device = latents.device

        latent_coords = coords / self.spatial_size  # scale to latent resolution

        pad_total = M_lat - H
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        canvas_num = torch.zeros(B, D, M_lat, M_lat, device=device)
        canvas_den = torch.zeros(B, 1, M_lat, M_lat, device=device)

        if probe is not None:
            probe_weights = torch.sum(torch.abs(probe) ** 2, dim=2)  # (B, C, N, N)
            probe_weights = F.adaptive_avg_pool2d(
                probe_weights.flatten(0, 1), (H, W)
            ).reshape(B, C, H, W)
        else:
            probe_weights = torch.ones(B, C, H, W, device=device)

        for i in range(C):
            lat_i = latents[:, i]  # (B, D, H, W)
            w_i = probe_weights[:, i]  # (B, H, W)
            off_i = latent_coords[:, i]  # (B, 1, 2)

            lat_i_padded = F.pad(lat_i, (pad_left, pad_right, pad_left, pad_right))
            w_i_padded = F.pad(w_i, (pad_left, pad_right, pad_left, pad_right))

            # Translation expects (N, H, W) and (N, 1, 2), returns (N, 1, H, W)
            # Reshape D into batch dim: (B, D, M, M) -> (B*D, M, M)
            lat_flat = lat_i_padded.reshape(B * D, M_lat, M_lat)
            # Repeat each batch element's offset D times: (B, 1, 2) -> (B*D, 1, 2)
            off_expanded = off_i.repeat_interleave(D, dim=0)
            translated = hh.Translation(lat_flat, off_expanded, 0.0)  # (B*D, 1, M, M)
            translated = translated.squeeze(1).reshape(B, D, M_lat, M_lat)

            # w_i_padded is already (B, M_lat, M_lat)
            w_translated = hh.Translation(w_i_padded, off_i, 0.0).squeeze(1)  # (B, M_lat, M_lat)

            canvas_num += translated * w_translated.unsqueeze(1)
            canvas_den += w_translated.unsqueeze(1)

        canvas = canvas_num / torch.clamp(canvas_den, min=1e-6)
        return canvas


class MultiResolutionFeatureVolume(nn.Module):
    """Extracts features at multiple receptive field scales from the latent canvas."""
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        in_channels = model_config.n_filters_scale * 128
        out_channels = model_config.feature_volume_channels

        self.level0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.level1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.level2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)

    @property
    def output_channels(self):
        return 3 * self.level0.out_channels

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            canvas: (B, D, M_lat, M_lat) latent canvas
        Returns:
            (B, 3*out_channels, M_lat, M_lat) multi-resolution features
        """
        return torch.cat([
            self.level0(canvas),
            self.level1(canvas),
            self.level2(canvas),
        ], dim=1)


class NeuralFieldDecoder(nn.Module):
    """Coordinate-conditioned MLP decoder with Fourier positional encoding.

    Evaluates f(r, z_canvas) -> (amplitude, phase) at each query coordinate
    by sampling the feature volume and concatenating with Fourier-encoded positions.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.N = data_config.N

        self.coord_encoder = FourierPositionalEncoding(
            num_bands=model_config.fourier_bands_coord
        )
        coord_dim = self.coord_encoder.output_dim
        feat_dim = 3 * model_config.feature_volume_channels
        input_dim = coord_dim + feat_dim

        layers = []
        hidden_dims = model_config.neural_field_hidden
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU(inplace=True))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # amplitude, phase
        self.mlp = nn.Sequential(*layers)

        M = hh.get_padded_size(data_config, model_config)
        self.canvas_size = math.ceil(M / 8)
        self.spatial_size = 8

    def forward(self, feature_volume: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_volume: (B, F, M_lat, M_lat) multi-resolution feature volume
            coords: (B, C, 1, 2) relative coordinates
        Returns:
            (B, C, N, N) complex64 object patches
        """
        B = feature_volume.shape[0]
        C = coords.shape[1]
        N = self.N
        M_lat = self.canvas_size
        device = feature_volume.device

        # Local grid coordinates centered at origin, in pixel units
        linspace = torch.linspace(-(N - 1) / 2, (N - 1) / 2, N, device=device)
        gy, gx = torch.meshgrid(linspace, linspace, indexing='ij')
        grid_local = torch.stack([gx, gy], dim=-1)  # (N, N, 2)
        grid_local = grid_local.reshape(1, 1, N * N, 2)  # (1, 1, N², 2)

        # Offset to canvas coordinates (in pixel units, then normalize for grid_sample)
        offsets = coords.squeeze(2)  # (B, C, 2)
        offsets = offsets.unsqueeze(2)  # (B, C, 1, 2)
        r_canvas = grid_local + offsets  # (B, C, N², 2) broadcast

        # Normalize to [-1, 1] for F.grid_sample (canvas is M_lat × M_lat, in latent pixel units)
        r_canvas_latent = r_canvas / self.spatial_size
        r_normalized = r_canvas_latent / (M_lat / 2)  # map to [-1, 1] range approximately
        # Flip x,y for grid_sample (expects [x, y] but grid_sample indexes [y, x])
        r_sample = r_normalized[..., [1, 0]]

        # Sample features for all patches at once
        r_sample_flat = r_sample.reshape(B, C * N * N, 1, 2)  # (B, C*N², 1, 2)
        # grid_sample expects (B, C_feat, H, W) input, (B, H_out, W_out, 2) grid
        sampled = F.grid_sample(
            feature_volume, r_sample_flat,
            mode='bilinear', align_corners=False, padding_mode='zeros'
        )  # (B, F, C*N², 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, C*N², F)

        # Fourier encode the query coordinates (in pixel units for meaningful frequencies)
        r_for_encoding = r_canvas.reshape(B, C * N * N, 2)  # (B, C*N², 2)
        coord_features = self.coord_encoder(r_for_encoding)   # (B, C*N², coord_dim)

        # Concatenate and run MLP
        mlp_input = torch.cat([coord_features, sampled], dim=-1)  # (B, C*N², input_dim)
        output = self.mlp(mlp_input)  # (B, C*N², 2)

        # Apply activations
        amplitude = torch.sigmoid(output[..., 0])  # [0, 1]
        phase = 1.2 * torch.tanh(output[..., 1])   # [-1.2, 1.2]

        # Combine to complex and reshape
        amplitude = amplitude.reshape(B, C, N, N)
        phase = phase.reshape(B, C, N, N)
        result = amplitude.to(torch.complex64) + 1j * phase.to(torch.complex64)

        return result, amplitude, phase


class AutoencoderCCNF(nn.Module):
    """Position-Conditioned Shared Canvas + Neural Field Decoder.

    Replaces the standard Autoencoder with:
    1. Weight-shared CNN encoder (per-patch, single-channel input)
    2. FiLM positional conditioning at the bottleneck
    3. Latent-space canvas assembly via Translation()
    4. Multi-resolution feature volume extraction
    5. Neural field MLP decoder with Fourier coordinate encoding
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        encoder_config = copy.deepcopy(model_config)
        object.__setattr__(encoder_config, 'C_model', 1)
        object.__setattr__(encoder_config, 'object_big', False)
        self.encoder = Encoder(encoder_config, data_config)

        if model_config.cbam_bottleneck:
            bottleneck_channels = self.encoder.filters[-1]
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity()

        self.film = FiLMConditioning(model_config)
        self.canvas_assembler = LatentCanvasAssembler(data_config, model_config)
        self.feature_volume = MultiResolutionFeatureVolume(model_config)
        self.decoder = NeuralFieldDecoder(model_config, data_config)

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                probe: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: (B, C, N, N) diffraction patterns (already intensity-scaled)
            coords: (B, C, 1, 2) relative coordinates
            probe: (B, C, P, N, N) complex probe (optional)
        Returns:
            complex_out: (B, C, N, N) complex64 object patches
            amplitude: (B, C, N, N) amplitude component
            phase: (B, C, N, N) phase component
        """
        B, C, N, _ = x.shape

        # 1. Weight-shared encoding: process each patch independently
        x_flat = x.reshape(B * C, 1, N, N)
        z_flat, _skips = self.encoder(x_flat)  # (B*C, D, 8, 8)
        z_flat = self.bottleneck_cbam(z_flat)
        D = z_flat.shape[1]
        z = z_flat.reshape(B, C, D, z_flat.shape[2], z_flat.shape[3])  # (B, C, D, 8, 8)

        # 2. FiLM positional conditioning
        coords_2d = coords.squeeze(2)  # (B, C, 2)
        z = self.film(z, coords_2d)    # (B, C, D, 8, 8)

        # 3. Assemble latent canvas
        canvas = self.canvas_assembler(z, coords, probe)  # (B, D, M_lat, M_lat)

        # 4. Multi-resolution feature extraction
        feat_vol = self.feature_volume(canvas)  # (B, F, M_lat, M_lat)

        # 5. Neural field decoding
        complex_out, amplitude, phase = self.decoder(feat_vol, coords)

        return complex_out, amplitude, phase


class PtychoPINN_CCNF(nn.Module):
    """Full PtychoPINN module using the PC-CCNF architecture."""
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        self.autoencoder = AutoencoderCCNF(model_config, data_config)
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

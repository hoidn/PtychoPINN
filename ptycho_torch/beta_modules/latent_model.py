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
from ptycho_torch.model import Encoder, ForwardModel, IntensityScalerModule, ConvPoolBlock
from ptycho_torch.model_attention import CBAM


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


def compute_canvas_scale(coords: torch.Tensor, N: int, M_lat: int,
                         margin: int = 2) -> torch.Tensor:
    """Compute per-batch, per-axis scale factors for dynamic canvas mapping.

    Maps the coordinate bounding box plus patch extent to fill the latent
    canvas, with independent scaling for x and y axes.

    Args:
        coords: (B, C, 1, 2) relative coordinates in pixel units
        N: diffraction pattern size (e.g., 64)
        M_lat: canvas size in canvas pixels (e.g., 16)
        margin: canvas margin in pixels per side (default 2 total)
    Returns:
        scale: (B, 1, 2) canvas pixels per real pixel, per axis
    """
    coords_2d = coords.squeeze(2)  # (B, C, 2)
    bbox = coords_2d.max(dim=1).values - coords_2d.min(dim=1).values  # (B, 2)
    total_extent = (bbox + N).clamp(min=float(N))  # (B, 2) in pixels
    scale = (M_lat - margin) / total_extent  # (B, 2)
    return scale.unsqueeze(1)  # (B, 1, 2)


class GeometryTaggedAttentionFusion(nn.Module):
    """Geometry-tagged cross-attention fusion for assembling per-patch latents.

    Bilinearly splats each patch's feature map onto a shared canvas, attaches
    local geometry offset tags, and fuses contributions via multi-head
    cross-attention. Uses dynamic per-batch, per-axis scaling to map
    coordinates to the canvas.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        D = model_config.n_filters_scale * 128
        M_lat = model_config.latent_canvas_size
        self.D = D
        self.canvas_size = M_lat
        self.N = data_config.N
        self.n_heads = model_config.fusion_heads

        self.tag_encoder = FourierPositionalEncoding(
            num_bands=model_config.fusion_tag_bands
        )
        tag_dim = self.tag_encoder.output_dim

        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D + tag_dim, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)
        self.layer_norm = nn.LayerNorm(D)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        half = (M_lat - 1) / 2.0
        gy, gx = torch.meshgrid(
            torch.arange(M_lat, dtype=torch.float32) - half,
            torch.arange(M_lat, dtype=torch.float32) - half,
            indexing='ij'
        )
        self.register_buffer('canvas_grid', torch.stack([gx, gy], dim=-1))

    def forward(self, latents: torch.Tensor, coords: torch.Tensor,
                scale: torch.Tensor,
                probe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            latents: (B, C, D, H, W) per-patch latent feature maps
            coords: (B, C, 1, 2) relative coordinates per patch (pixel units)
            scale: (B, 1, 2) per-axis canvas scale (canvas pixels per real pixel)
            probe: (B, C, P, N, N) complex probe (optional, unused)
        Returns:
            (B, D, M_lat, M_lat) fused latent canvas
        """
        B, C, D, H, W = latents.shape
        M_lat = self.canvas_size
        N = self.N
        S = M_lat * M_lat

        patch_centers = coords.squeeze(2) * scale  # (B, C, 2) in canvas pixels

        # Bilinear splatting: sample each patch's features at every canvas position
        canvas_positions = self.canvas_grid.reshape(1, 1, S, 2)  # (1, 1, S, 2)
        centers = patch_centers.unsqueeze(2)  # (B, C, 1, 2)
        local_coords = canvas_positions - centers  # (B, C, S, 2) in canvas pixels

        # Normalize for grid_sample on H×W encoder output:
        # encoder covers N real pixels = N*scale canvas pixels per axis
        scale_norm = (N * scale).unsqueeze(2)  # (B, 1, 1, 2) broadcasts to (B, C, S, 2)
        norm_local = local_coords * 2.0 / scale_norm
        sample_grid = norm_local.reshape(B * C, M_lat, M_lat, 2)

        latents_flat = latents.reshape(B * C, D, H, W)
        splatted = F.grid_sample(
            latents_flat, sample_grid,
            mode='bilinear', align_corners=False, padding_mode='zeros'
        )  # (B*C, D, M_lat, M_lat)
        splatted = splatted.reshape(B, C, D, M_lat, M_lat)

        # Reshape for attention: (B, S, C, D)
        features = splatted.permute(0, 3, 4, 1, 2).reshape(B, S, C, D)

        # Geometry tags: Fourier encode local offsets
        tags = self.tag_encoder(local_coords)  # (B, C, S, tag_dim)
        tags = tags.permute(0, 2, 1, 3)  # (B, S, C, tag_dim)

        # Cross-attention
        d_head = D // self.n_heads

        q = self.q_proj(features.mean(dim=2))  # (B, S, D)
        k = self.k_proj(torch.cat([features, tags], dim=-1))  # (B, S, C, D)
        v = self.v_proj(features)  # (B, S, C, D)

        q_residual = q  # save for residual connection

        q = q.reshape(B, S, self.n_heads, d_head).transpose(1, 2)  # (B, heads, S, d_head)
        k = k.reshape(B, S, C, self.n_heads, d_head).permute(0, 3, 1, 2, 4)  # (B, heads, S, C, d_head)
        v = v.reshape(B, S, C, self.n_heads, d_head).permute(0, 3, 1, 2, 4)  # (B, heads, S, C, d_head)

        attn_logits = (q.unsqueeze(3) * k).sum(dim=-1) / math.sqrt(d_head)  # (B, heads, S, C)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, heads, S, C)

        out = (attn_weights.unsqueeze(-1) * v).sum(dim=3)  # (B, heads, S, d_head)
        out = out.transpose(1, 2).reshape(B, S, D)  # (B, S, D)

        out = self.layer_norm(self.out_proj(out) + q_residual)

        canvas = out.reshape(B, M_lat, M_lat, D).permute(0, 3, 1, 2)
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

    Evaluates f(r, z_canvas) -> (x_real, x_imag) at each query coordinate
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
        layers.append(nn.Linear(prev_dim, 2))  # real, imaginary
        self.mlp = nn.Sequential(*layers)

        self.canvas_size = model_config.latent_canvas_size
        self.chunk_size = getattr(model_config, 'nf_chunk_size', 4096)

    def forward(self, feature_volume: torch.Tensor,
                coords: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_volume: (B, F, M_lat, M_lat) multi-resolution feature volume
            coords: (B, C, 1, 2) relative coordinates (pixel units)
            scale: (B, 1, 2) per-axis canvas scale (canvas pixels per real pixel)
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

        # Offset to canvas coordinates (in pixel units)
        offsets = coords.squeeze(2)  # (B, C, 2)
        offsets = offsets.unsqueeze(2)  # (B, C, 1, 2)
        r_canvas = grid_local + offsets  # (B, C, N², 2) broadcast, pixel units

        # Dynamic scaling: pixel coords → canvas coords → normalized [-1, 1]
        scale_expanded = scale.unsqueeze(2)  # (B, 1, 1, 2)
        r_canvas_scaled = r_canvas * scale_expanded  # (B, C, N², 2) canvas pixels
        r_normalized = r_canvas_scaled / (M_lat / 2)

        Q = C * N * N
        r_flat = r_normalized.reshape(B, Q, 2)
        r_canvas_flat = r_canvas.reshape(B, Q, 2)

        chunk_size = self.chunk_size
        outputs = []

        for start in range(0, Q, chunk_size):
            end = min(start + chunk_size, Q)
            r_chunk = r_flat[:, start:end, :]
            rc_chunk = r_canvas_flat[:, start:end, :]

            sampled = F.grid_sample(
                feature_volume,
                r_chunk.unsqueeze(2),
                mode='bilinear', align_corners=False, padding_mode='zeros'
            ).squeeze(-1).permute(0, 2, 1)

            coord_features = self.coord_encoder(rc_chunk)
            mlp_input = torch.cat([coord_features, sampled], dim=-1)
            outputs.append(self.mlp(mlp_input))

        output = torch.cat(outputs, dim=1)  # (B, Q, 2)

        x_real = 0.2 + torch.tanh(output[..., 0])
        x_imag = 1.2 * torch.tanh(output[..., 1])

        x_real = x_real.reshape(B, C, N, N)
        x_imag = x_imag.reshape(B, C, N, N)
        result = x_real.to(torch.complex64) + 1j * x_imag.to(torch.complex64)

        return result, x_real, x_imag


class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm for (B, C, H, W) tensors."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt V1 block with depthwise conv, inverted bottleneck, and layer scale."""
    def __init__(self, channels: int, expansion_factor: int = 4,
                 layer_scale_init: float = 1e-6):
        super().__init__()
        expanded = channels * expansion_factor
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7,
                                padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, expanded, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(expanded, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(1, channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale * x
        return x + residual


class PixelShuffleUpsampleStage(nn.Module):
    """2x spatial upsampling via PixelShuffle followed by ConvNeXt refinement."""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.upsample_conv = nn.Conv2d(ch_in, 4 * ch_out, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.refine = ConvNeXtBlock(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)
        return x


class CNNCanvasDecoder(nn.Module):
    """CNN-based decoder for the PC-CCNF architecture.

    Extracts per-patch crops from the feature volume using grid_sample,
    then upsamples through ConvNeXt blocks with PixelShuffle to reconstruct
    complex-valued object patches.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.N = data_config.N
        self.canvas_size = model_config.latent_canvas_size
        self.crop_size = getattr(model_config, 'cnn_decoder_crop_size', 8)

        F_in = 3 * model_config.feature_volume_channels
        base_ch = getattr(model_config, 'cnn_decoder_base_ch', 128)
        n_stages = int(math.log2(self.N / self.crop_size))

        assert self.crop_size <= self.canvas_size, (
            f"crop_size={self.crop_size} must be <= "
            f"latent_canvas_size={self.canvas_size}"
        )
        assert self.N == self.crop_size * (2 ** n_stages), (
            f"N={self.N} must be crop_size * 2^k for integer k"
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

        self.stages = nn.ModuleList()
        for s in range(n_stages):
            self.stages.append(
                PixelShuffleUpsampleStage(channels[s], channels[s + 1])
            )

        last_ch = channels[-1]
        self.head_real = nn.Conv2d(last_ch, 1, kernel_size=3, padding=1)
        self.head_imag = nn.Conv2d(last_ch, 1, kernel_size=3, padding=1)

        nn.init.zeros_(self.head_real.bias)
        nn.init.zeros_(self.head_imag.bias)

    def _build_sample_grid(self, feature_volume: torch.Tensor,
                           coords: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
        B, F, M_lat, _ = feature_volume.shape
        C = coords.shape[1]
        S = self.crop_size
        N = self.N
        device = feature_volume.device

        patch_centers = coords.squeeze(2) * scale  # (B, C, 2)

        half = (N - 1) / 2.0
        lin = torch.linspace(-half, half, S, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing='ij')
        local_grid = torch.stack([gx, gy], dim=-1)  # (S, S, 2) pixel units

        local_grid = local_grid.reshape(1, 1, S, S, 2)
        centers = patch_centers.reshape(B, C, 1, 1, 2)
        scale_exp = scale.reshape(B, 1, 1, 1, 2)

        sample_grid = local_grid * scale_exp + centers
        sample_grid = sample_grid / (M_lat / 2.0)
        return sample_grid.reshape(B * C, S, S, 2)

    def _decode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        x = self.stem(crops)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, feature_volume: torch.Tensor,
                coords: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        B = feature_volume.shape[0]
        C = coords.shape[1]
        N = self.N
        M_lat = self.canvas_size
        F_ch = feature_volume.shape[1]

        sample_grid = self._build_sample_grid(feature_volume, coords, scale)

        fv_expanded = feature_volume.unsqueeze(1).expand(-1, C, -1, -1, -1)
        fv_flat = fv_expanded.reshape(B * C, F_ch, M_lat, M_lat)

        crops = F.grid_sample(
            fv_flat, sample_grid,
            mode='bilinear', align_corners=False, padding_mode='zeros'
        )

        x = self._decode_crops(crops)

        x_real = 0.2 + torch.tanh(self.head_real(x))
        x_imag = 1.2 * torch.tanh(self.head_imag(x))

        x_real = x_real.reshape(B, C, N, N)
        x_imag = x_imag.reshape(B, C, N, N)
        result = x_real.to(torch.complex64) + 1j * x_imag.to(torch.complex64)

        return result, x_real, x_imag


class SpectralConv2d(nn.Module):
    """Learnable spectral convolution via truncated Fourier modes.

    Multiplies the input's Fourier transform by a learnable complex weight
    tensor, truncated to the lowest `modes` frequencies in each dimension.
    Uses rfft2 (real FFT) so the frequency tensor has shape (H, W//2+1).
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        x_ft = torch.fft.rfft2(x, norm='ortho')

        w = torch.view_as_complex(self.weight)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Positive ky, positive kx corner
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :self.modes, :self.modes], w
        )

        # Negative ky (wrapped), positive kx corner
        out_ft[:, :, -self.modes:, :self.modes] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, -self.modes:, :self.modes], w
        )

        return torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')


class FNOBlock(nn.Module):
    """Single Fourier Neural Operator block.

    Combines a global spectral path (SpectralConv2d) with a local pointwise
    bypass (1x1 conv), InstanceNorm, GELU activation, and residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes)
        self.local_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.GELU()
        self.residual = (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.spectral_conv(x) + self.local_conv(x)
        out = self.norm(out)
        out = self.activation(out)
        if self.residual:
            out = out + identity
        return out


class FNOCNNEncoder(nn.Module):
    """Hybrid FNO+CNN encoder for ptychographic diffraction patterns.

    FNO blocks at full spatial resolution provide global context (Friedel pairs,
    radial ring structure, speckle correlations). CNN blocks then compress
    spatially to the 8x8 bottleneck required by the latent canvas.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        N = data_config.N
        nfs = model_config.n_filters_scale
        fno_width = model_config.fno_width
        fno_modes = model_config.fno_modes
        n_fno_blocks = model_config.fno_blocks

        max_modes = N // 2 + 1
        if fno_modes > max_modes:
            raise ValueError(
                f"fno_modes={fno_modes} exceeds max for N={N} "
                f"(rfft2 gives {max_modes} frequency bins)"
            )

        self.lifting = nn.Conv2d(1, fno_width, kernel_size=1)

        self.fno_blocks = nn.ModuleList([
            FNOBlock(fno_width, fno_width, fno_modes)
            for _ in range(n_fno_blocks)
        ])

        if N < 16 or (N & (N - 1)) != 0:
            raise ValueError(f"N must be a power of 2 >= 16, got N={N}")
        n_cnn_pools = int(math.log2(N / 8))
        n_cnn_blocks = n_cnn_pools - 1
        cnn_filter_list = [fno_width] + [
            nfs * (128 >> i) for i in range(n_cnn_blocks - 1, -1, -1)
        ]

        self.cnn_blocks = nn.ModuleList()
        for i in range(len(cnn_filter_list) - 1):
            self.cnn_blocks.append(
                ConvPoolBlock(
                    in_channels=cnn_filter_list[i],
                    out_channels=cnn_filter_list[i + 1],
                    model_config=model_config,
                    batch_norm=model_config.batch_norm
                )
            )

        self.final_pool = nn.MaxPool2d(kernel_size=2)

        self.blocks = nn.ModuleList(
            list(self.fno_blocks) + list(self.cnn_blocks)
        )

        self.filters = [1] + [fno_width] * n_fno_blocks + cnn_filter_list[1:]

    def forward(self, x: torch.Tensor):
        skips = []

        x = self.lifting(x)

        for fno_block in self.fno_blocks:
            x = fno_block(x)
            skips.append(x)

        for cnn_block in self.cnn_blocks:
            x = cnn_block.forward_conv(x)
            skips.append(x)
            x = cnn_block.forward_pool(x)

        x = self.final_pool(x)

        return x, skips


class AutoencoderCCNF(nn.Module):
    """Position-Conditioned Shared Canvas + Neural Field Decoder.

    Replaces the standard Autoencoder with:
    1. Weight-shared encoder (per-patch, single-channel input; CNN or FNO+CNN)
    2. Geometry-tagged cross-attention fusion onto shared canvas
    3. Multi-resolution feature volume extraction
    4. Neural field MLP decoder with Fourier coordinate encoding
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        encoder_config = copy.deepcopy(model_config)
        object.__setattr__(encoder_config, 'C_model', 1)
        object.__setattr__(encoder_config, 'object_big', False)

        if getattr(model_config, 'encoder_type', 'cnn') == 'fno_cnn':
            self.encoder = FNOCNNEncoder(encoder_config, data_config)
        else:
            self.encoder = Encoder(encoder_config, data_config)

        if model_config.cbam_bottleneck:
            bottleneck_channels = self.encoder.filters[-1]
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity()

        self.fusion = GeometryTaggedAttentionFusion(model_config, data_config)
        self.feature_volume = MultiResolutionFeatureVolume(model_config)

        if getattr(model_config, 'ccnf_decoder_type', 'neural_field') == 'cnn':
            self.decoder = CNNCanvasDecoder(model_config, data_config)
        else:
            self.decoder = NeuralFieldDecoder(model_config, data_config)

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                probe: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: (B, C, N, N) diffraction patterns (already intensity-scaled)
            coords: (B, C, 1, 2) relative coordinates (pixel units)
            probe: (B, C, P, N, N) complex probe (optional)
        Returns:
            complex_out: (B, C, N, N) complex64 object patches
            x_real: (B, C, N, N) real component
            x_imag: (B, C, N, N) imaginary component
        """
        B, C, N, _ = x.shape

        # Compute per-batch, per-axis canvas scale
        M_lat = self.model_config.latent_canvas_size
        scale = compute_canvas_scale(coords, N, M_lat)

        # 1. Weight-shared encoding: process each patch independently
        x_flat = x.reshape(B * C, 1, N, N)
        z_flat, _skips = self.encoder(x_flat)  # (B*C, D, 8, 8)
        z_flat = self.bottleneck_cbam(z_flat)
        D = z_flat.shape[1]
        z = z_flat.reshape(B, C, D, z_flat.shape[2], z_flat.shape[3])  # (B, C, D, 8, 8)

        # 2. Geometry-tagged cross-attention fusion
        canvas = self.fusion(z, coords, scale)  # (B, D, M_lat, M_lat)

        # 3. Multi-resolution feature extraction
        feat_vol = self.feature_volume(canvas)  # (B, F, M_lat, M_lat)

        # 4. Neural field decoding
        complex_out, x_real, x_imag = self.decoder(feat_vol, coords, scale)

        return complex_out, x_real, x_imag


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

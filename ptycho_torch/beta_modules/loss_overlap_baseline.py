"""Loss-Only Overlap Baseline: Multi-Channel FNO+CNN Encoder + PixelShuffle Decoder.

Minimal baseline for loss-enforced overlap consistency. All C overlapping
diffraction patches enter the encoder as channels (B, C, N, N), enabling
implicit information mixing through shared convolutions. The bottleneck
is a pure identity (no pairwise gate, no cross-attention). A PixelShuffle
decoder with encoder skip injection reconstructs all C patches jointly.

Overlap consistency is enforced exclusively by the physics loss in the
ForwardModel, which reassembles patches, re-extracts them, and penalizes
diffraction prediction error.

Compared to overlap_baseline.py:
- Encoder takes (B, C, N, N) instead of (B*C, 1, N, N)
- No PairwiseOffsetGate at the bottleneck
- Decoder outputs C channels instead of 1
- Overlap mixing is implicit (convolutions see all C channels)
"""

import copy
from typing import Optional

import torch
from torch import nn

from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
from ptycho_torch.model import ForwardModel, IntensityScalerModule
from ptycho_torch.model_attention import CBAM
from ptycho_torch.beta_modules.latent_model import FNOCNNEncoder, LayerNorm2d
from ptycho_torch.beta_modules.overlap_baseline import PerPatchPixelShuffleDecoder


class FNOCNNEncoderMultiChannel(FNOCNNEncoder):
    """FNOCNNEncoder with multi-channel lifting layer.

    Replaces the single-channel lifting Conv2d(1, fno_width, 1) with
    Conv2d(C, fno_width, 1) so that C overlapping patches are processed
    as input channels with implicit mixing through shared convolutions.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(model_config, data_config)
        C = data_config.C
        fno_width = model_config.fno_width
        self.lifting = nn.Conv2d(C, fno_width, kernel_size=1)
        self.filters[0] = C


class MultiChannelPixelShuffleDecoder(PerPatchPixelShuffleDecoder):
    """PixelShuffle decoder with C-channel output heads.

    Replaces Conv2d(last_ch, 1, ...) with Conv2d(last_ch, C, ...) so the
    decoder outputs all C patch reconstructions jointly.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 bottleneck_channels: int, H_enc: int,
                 encoder_skip_info: Optional[list] = None):
        super().__init__(model_config, data_config, bottleneck_channels, H_enc,
                         encoder_skip_info=encoder_skip_info)
        C = data_config.C
        last_ch = self.head_real[0].in_channels
        self.head_real = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(last_ch, C, kernel_size=3, padding=1),
        )
        self.head_imag = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(last_ch, C, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.head_real[-1].bias)
        nn.init.zeros_(self.head_imag[-1].bias)


class AutoencoderLossOverlapBaseline(nn.Module):
    """Multi-channel FNO+CNN encoder + identity bottleneck + PixelShuffle decoder.

    All C patches are processed jointly through the encoder as channels.
    No explicit overlap mixing — overlap consistency comes from the physics
    loss in the ForwardModel.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        encoder_config = copy.deepcopy(model_config)
        object.__setattr__(encoder_config, 'fno_interleave', True)
        self.encoder = FNOCNNEncoderMultiChannel(encoder_config, data_config)

        N = data_config.N
        C = data_config.C
        with torch.no_grad():
            dummy = torch.zeros(1, C, N, N)
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
                probe: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: (B, C, N, N) diffraction patterns (already intensity-scaled)
            coords: (B, C, 1, 2) relative coordinates (unused, interface compat)
            probe: unused, accepted for interface compatibility
        Returns:
            complex_out: (B, C, N, N) complex64 object patches
            x_real: (B, C, N, N) real component
            x_imag: (B, C, N, N) imaginary component
        """
        z, encoder_skips = self.encoder(x)
        z = self.bottleneck_cbam(z)
        x_real, x_imag = self.decoder(z, encoder_skips=encoder_skips)
        complex_out = x_real + 1j * x_imag
        return complex_out, x_real, x_imag


class PtychoPINN_LossOverlapBaseline(nn.Module):
    """Full PtychoPINN module using the loss-only overlap baseline."""
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        self.autoencoder = AutoencoderLossOverlapBaseline(model_config, data_config)
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

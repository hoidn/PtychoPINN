"""Patterson Correlation Feature Model for PtychoPINN.

Encodes ptychographic overlap constraints directly in the input representation
via Patterson cross-products and probe overlap maps, then processes through a
standard CNN encoder with a ConvNeXt + PixelShuffle decoder.
"""

import math
import copy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
from ptycho_torch.model import (
    Encoder, ForwardModel, IntensityScalerModule,
    CombineComplexRectangular,
)
from ptycho_torch.model_attention import CBAM
from ptycho_torch.beta_modules.patterson_featurizer import PattersonFeaturizer
from ptycho_torch.beta_modules.latent_model import (
    LayerNorm2d, ConvNeXtBlock, PixelShuffleUpsampleStage,
)


class ConvNeXtDecoder(nn.Module):
    """ConvNeXt + PixelShuffle decoder with encoder skip connections.

    Upsamples from encoder bottleneck to full N x N resolution via
    PixelShuffle stages with ConvNeXt refinement blocks. Produces
    separate real and imaginary output heads.

    Control decoder capacity via cnn_decoder_base_ch (channel width)
    and cnn_decoder_n_refine (ConvNeXt blocks per stage).
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 encoder_skip_info: Optional[list] = None):
        super().__init__()
        self.N = data_config.N
        N = data_config.N
        C = model_config.C_model
        nfs = model_config.n_filters_scale
        bottleneck_ch = nfs * 128
        base_ch = getattr(model_config, 'cnn_decoder_base_ch', 128)
        n_refine = getattr(model_config, 'cnn_decoder_n_refine', 1)

        bottleneck_spatial = 8
        n_stages = int(math.log2(N / bottleneck_spatial))

        self.stem = nn.Sequential(
            nn.Conv2d(bottleneck_ch, base_ch, kernel_size=1),
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
                target_res = bottleneck_spatial * (2 ** (s + 1))
                for k in range(len(encoder_skip_info) - 1, -1, -1):
                    if encoder_skip_info[k][0] == target_res:
                        skip_channels_per_stage[s] = encoder_skip_info[k][1]
                        self._skip_indices[s] = k
                        break

        self.stages = nn.ModuleList()
        self.extra_refine = nn.ModuleList()
        for s in range(n_stages):
            self.stages.append(
                PixelShuffleUpsampleStage(
                    channels[s], channels[s + 1],
                    skip_channels=skip_channels_per_stage[s],
                )
            )
            extra = nn.Sequential(*[
                ConvNeXtBlock(channels[s + 1])
                for _ in range(n_refine - 1)
            ]) if n_refine > 1 else nn.Identity()
            self.extra_refine.append(extra)

        last_ch = channels[-1]
        self.head_real = nn.Conv2d(last_ch, C, kernel_size=3, padding=1)
        self.head_imag = nn.Conv2d(last_ch, C, kernel_size=3, padding=1)

        nn.init.zeros_(self.head_real.bias)
        nn.init.zeros_(self.head_imag.bias)

    def forward(self, bottleneck: torch.Tensor,
                encoder_skips: Optional[list] = None) -> tuple:
        """
        Args:
            bottleneck: (B, D, 8, 8) encoder bottleneck
            encoder_skips: list of (B, ch, H, W) skip tensors from encoder
        Returns:
            (complex_out, x_real, x_imag) each (B, C, N, N)
        """
        x = self.stem(bottleneck)

        for i, stage in enumerate(self.stages):
            skip = None
            if encoder_skips is not None and self._skip_indices[i] is not None:
                skip = encoder_skips[self._skip_indices[i]]
            x = stage(x, skip=skip)
            x = self.extra_refine[i](x)

        x_real = torch.tanh(self.head_real(x))
        x_imag = torch.tanh(self.head_imag(x))

        result = x_real.to(torch.complex64) + 1j * x_imag.to(torch.complex64)
        return result, x_real, x_imag


class PattersonAutoencoder(nn.Module):
    """Encoder-decoder with Patterson feature lifting convolution.

    A 3x3 lifting conv reduces 4*C Patterson feature channels to C before
    the standard CNN encoder. The ConvNeXt + PixelShuffle decoder upsamples
    from the 8x8 bottleneck back to N x N with encoder skip connections.
    """
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        C = model_config.C_model

        self.lifting = nn.Sequential(
            nn.Conv2d(4 * C, C, kernel_size=3, padding=1),
            nn.InstanceNorm2d(C),
            nn.GELU(),
        )

        self.encoder = Encoder(model_config, data_config)

        if model_config.cbam_bottleneck:
            bottleneck_channels = self.encoder.filters[-1]
            self.bottleneck_cbam = CBAM(gate_channels=bottleneck_channels)
        else:
            self.bottleneck_cbam = nn.Identity()

        # Probe encoder skip info for resolution matching
        N = data_config.N
        with torch.no_grad():
            dummy = torch.zeros(1, C, N, N)
            _, dummy_skips = self.encoder(dummy)
            encoder_skip_info = [
                (s.shape[2], s.shape[1]) for s in dummy_skips
            ]

        self.decoder = ConvNeXtDecoder(
            model_config, data_config,
            encoder_skip_info=encoder_skip_info,
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, 4*C, N, N) Patterson feature tensor
        Returns:
            (complex_out, x_real, x_imag) each (B, C, N, N)
        """
        x = self.lifting(x)
        bottleneck, skips = self.encoder(x)
        bottleneck = self.bottleneck_cbam(bottleneck)
        return self.decoder(bottleneck, encoder_skips=skips)


class PtychoPINN_Patterson(nn.Module):
    """Full PtychoPINN module using Patterson correlation input features."""
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.featurizer = PattersonFeaturizer()
        self.scaler = IntensityScalerModule(model_config)
        self.probe_scale = data_config.probe_scale

        self.autoencoder = PattersonAutoencoder(model_config, data_config)
        self.combine_complex = CombineComplexRectangular()
        self.forward_model = ForwardModel(model_config, data_config)

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor,
                experiment_ids=None, fine_tune=False):
        features = self.featurizer(x, probe)
        features = self.scaler.scale(features, input_scale_factor)

        x_combined, x_real, x_imag = self.autoencoder(features)

        x_out = self.forward_model.forward(
            x_combined, x,
            positions, probe / self.probe_scale, output_scale_factor,
            experiment_ids=experiment_ids,
            fine_tune=fine_tune,
        )

        return x_out, x_real, x_imag

    def forward_predict(self, x, positions, probe, input_scale_factor):
        features = self.featurizer(x, probe)
        features = self.scaler.scale(features, input_scale_factor)
        x_combined, _, _ = self.autoencoder(features)
        return x_combined

    def get_encoder_bottom_params(self):
        encoder = self.autoencoder.encoder
        if not hasattr(encoder, 'blocks') or len(encoder.blocks) == 0:
            return []
        split_idx = len(encoder.blocks) // 2
        params = list(self.autoencoder.lifting.parameters())
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
        for param in self.autoencoder.lifting.parameters():
            param.requires_grad = False
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

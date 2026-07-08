"""Spectral ResNet bottleneck shell with a lighter bilinear + 1x1 decoder."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ptycho_torch.generators.spectral_resnet_bottleneck import (
    SpectralResnetBottleneckGeneratorModule,
)


class BilinearProjectionUpsampler(nn.Module):
    """Upsample by 2x with bilinear interpolation followed by a 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class SpectralResnetBottleneckLinearDecoderGeneratorModule(
    SpectralResnetBottleneckGeneratorModule
):
    """Spectral ResNet bottleneck shell that replaces CycleGAN upsamplers with bilinear + 1x1 stages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upsample_layers = nn.ModuleList(
            [
                BilinearProjectionUpsampler(self.decoder_widths[index], self.decoder_widths[index + 1])
                for index in range(self.hybrid_downsample_steps)
            ]
        )


class SpectralResnetBottleneckLinearDecoderGenerator:
    """Generator registry wrapper for spectral_resnet_bottleneck_linear_decoder."""

    name = "spectral_resnet_bottleneck_linear_decoder"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> "nn.Module":
        from ptycho_torch.model import PtychoPINN_Lightning

        data_config = pt_configs["data_config"]
        model_config = pt_configs["model_config"]
        training_config = pt_configs["training_config"]
        inference_config = pt_configs["inference_config"]
        output_mode = getattr(model_config, "generator_output_mode", "real_imag")
        generator_mode = "amp_phase" if output_mode == "amp_phase" else "real_imag"

        core = SpectralResnetBottleneckLinearDecoderGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=getattr(model_config, "fno_width", 32),
            n_blocks=getattr(model_config, "fno_blocks", 4),
            modes=getattr(model_config, "fno_modes", 12),
            C=getattr(data_config, "C", 4),
            input_transform=getattr(model_config, "fno_input_transform", "none"),
            output_mode=generator_mode,
            max_hidden_channels=getattr(model_config, "max_hidden_channels", None),
            resnet_width=getattr(model_config, "resnet_width", None),
            resnet_blocks=getattr(model_config, "hybrid_resnet_blocks", 6),
            hybrid_downsample_steps=getattr(model_config, "hybrid_downsample_steps", 2),
            hybrid_downsample_op=getattr(model_config, "hybrid_downsample_op", "stride_conv"),
            hybrid_encoder_conv_hidden_scale=getattr(model_config, "hybrid_encoder_conv_hidden_scale", 1.0),
            hybrid_encoder_spectral_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_scale",
                1.0,
            ),
            hybrid_encoder_conv_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_channels",
                None,
            ),
            hybrid_encoder_spectral_hidden_channels=getattr(
                model_config,
                "hybrid_encoder_spectral_hidden_channels",
                None,
            ),
            spectral_bottleneck_blocks=getattr(model_config, "spectral_bottleneck_blocks", 6),
            spectral_bottleneck_modes=getattr(model_config, "spectral_bottleneck_modes", 12),
            spectral_bottleneck_share_weights=getattr(
                model_config,
                "spectral_bottleneck_share_weights",
                True,
            ),
            spectral_bottleneck_gate_init=getattr(model_config, "spectral_bottleneck_gate_init", 0.1),
            spectral_bottleneck_gate_mode=getattr(model_config, "spectral_bottleneck_gate_mode", "shared"),
        )

        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )

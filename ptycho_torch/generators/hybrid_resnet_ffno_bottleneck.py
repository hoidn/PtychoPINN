"""Hybrid ResNet shell with an FFNO-close bottleneck replacement."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from ptycho_torch.generators.ffno_bottleneck import SharedFactorizedFfnoBottleneck
from ptycho_torch.generators.hybrid_resnet import HybridResnetGeneratorModule


class HybridResnetFfnoBottleneckGeneratorModule(HybridResnetGeneratorModule):
    """Hybrid ResNet shell that swaps the local ResNet bottleneck for FFNO-close blocks."""

    def __init__(
        self,
        *,
        ffno_bottleneck_blocks: int = 6,
        ffno_bottleneck_modes: int = 12,
        ffno_bottleneck_share_spectral_weights: bool = True,
        ffno_bottleneck_mlp_ratio: float = 2.0,
        ffno_bottleneck_gate_init: float = 0.1,
        ffno_bottleneck_norm: str = "instance",
        ffno_bottleneck_local_conv_kernel_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resnet = SharedFactorizedFfnoBottleneck(
            int(self.bottleneck_channels),
            n_blocks=ffno_bottleneck_blocks,
            modes=ffno_bottleneck_modes,
            share_spectral_weights=ffno_bottleneck_share_spectral_weights,
            mlp_ratio=ffno_bottleneck_mlp_ratio,
            gate_init=ffno_bottleneck_gate_init,
            norm=ffno_bottleneck_norm,
            local_conv_kernel_size=ffno_bottleneck_local_conv_kernel_size,
        )


class HybridResnetFfnoBottleneckGenerator:
    """Generator registry wrapper for hybrid_resnet_ffno_bottleneck."""

    name = "hybrid_resnet_ffno_bottleneck"

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

        core = HybridResnetFfnoBottleneckGeneratorModule(
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
            skip_connections=getattr(model_config, "hybrid_skip_connections", False),
            hybrid_downsample_steps=getattr(model_config, "hybrid_downsample_steps", 2),
            hybrid_downsample_op=getattr(model_config, "hybrid_downsample_op", "stride_conv"),
            hybrid_encoder_conv_hidden_scale=getattr(
                model_config,
                "hybrid_encoder_conv_hidden_scale",
                1.0,
            ),
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
            hybrid_skip_style=getattr(model_config, "hybrid_skip_style", "add"),
            ffno_bottleneck_blocks=getattr(model_config, "spectral_bottleneck_blocks", 6),
            ffno_bottleneck_modes=getattr(model_config, "spectral_bottleneck_modes", 12),
            ffno_bottleneck_share_spectral_weights=getattr(
                model_config,
                "spectral_bottleneck_share_weights",
                True,
            ),
            ffno_bottleneck_mlp_ratio=2.0,
            ffno_bottleneck_gate_init=getattr(model_config, "spectral_bottleneck_gate_init", 0.1),
            ffno_bottleneck_norm="instance",
            ffno_bottleneck_local_conv_kernel_size=None,
        )

        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )

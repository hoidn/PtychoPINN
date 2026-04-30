"""External NeuralOperator U-NO adapter for the locked Lines128 CDI path."""

from __future__ import annotations

import importlib
from typing import Any, Dict

import torch
import torch.nn as nn


_LOCKED_IMAGE_SIZE = 128
_LOCKED_UNO_KWARGS = {
    "in_channels": 1,
    "out_channels": 2,
    "hidden_channels": 32,
    "lifting_channels": 128,
    "projection_channels": 128,
    "n_layers": 4,
    "uno_out_channels": [32, 64, 64, 32],
    "uno_n_modes": [[12, 12], [12, 12], [12, 12], [12, 12]],
    "uno_scalings": [[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]],
    "positional_embedding": "grid",
    "channel_mlp_skip": "linear",
}


def _load_uno_class():
    try:
        neuralop_models = importlib.import_module("neuralop.models")
    except Exception as exc:
        raise RuntimeError(
            "neuralop_uno requires neuraloperator==2.0.0 with importable "
            "neuralop.models.UNO."
        ) from exc

    uno_cls = getattr(neuralop_models, "UNO", None)
    if uno_cls is None:
        raise RuntimeError(
            "neuralop_uno requires neuralop.models.UNO from neuraloperator==2.0.0; "
            "the installed package is missing that API surface."
        )
    return uno_cls


class NeuralopUnoGeneratorModule(nn.Module):
    """Wrap external `neuralop.models.UNO` behind the CDI real/imag contract."""

    def __init__(
        self,
        *,
        C: int = 1,
        output_mode: str = "real_imag",
    ):
        super().__init__()
        if int(C) != 1:
            raise ValueError(
                f"neuralop_uno only supports the locked C=1 CDI contract; got C={C}."
            )
        if output_mode != "real_imag":
            raise ValueError(
                "neuralop_uno only supports generator_output_mode='real_imag'."
            )

        uno_cls = _load_uno_class()
        self.uno = uno_cls(**_LOCKED_UNO_KWARGS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"neuralop_uno expects input with shape (B, 1, H, W), got {tuple(x.shape)}."
            )
        batch, channels, height, width = x.shape
        if channels != 1:
            raise ValueError(
                f"neuralop_uno expects a single diffraction channel, got {channels}."
            )
        if height != _LOCKED_IMAGE_SIZE or width != _LOCKED_IMAGE_SIZE:
            raise ValueError(
                "neuralop_uno only supports the locked Lines128 image size "
                f"(128x128); got {height}x{width}."
            )

        raw = self.uno(x)
        expected = (batch, 2, height, width)
        if tuple(raw.shape) != expected:
            raise RuntimeError(
                "neuralop_uno expected raw UNO output shape "
                f"{expected} but received {tuple(raw.shape)}."
            )
        return raw.permute(0, 2, 3, 1).unsqueeze(-2).contiguous()


class NeuralopUnoGenerator:
    """Generator-registry wrapper for the locked external U-NO CDI path."""

    name = "neuralop_uno"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> nn.Module:
        from ptycho_torch.model import PtychoPINN_Lightning

        data_config = pt_configs["data_config"]
        model_config = pt_configs["model_config"]
        training_config = pt_configs["training_config"]
        inference_config = pt_configs["inference_config"]

        if int(getattr(data_config, "N", _LOCKED_IMAGE_SIZE)) != _LOCKED_IMAGE_SIZE:
            raise ValueError(
                "neuralop_uno only supports the locked Lines128 CDI contract "
                f"(N=128); got N={getattr(data_config, 'N', None)}."
            )
        if tuple(getattr(data_config, "grid_size", (1, 1))) != (1, 1):
            raise ValueError(
                "neuralop_uno only supports the locked gridsize=1 CDI contract; "
                f"got grid_size={getattr(data_config, 'grid_size', None)}."
            )

        output_mode = getattr(model_config, "generator_output_mode", "real_imag")
        core = NeuralopUnoGeneratorModule(
            C=getattr(data_config, "C", 1),
            output_mode=output_mode,
        )

        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )

"""Thin adapter that hosts NeuralOperator U-NO under the CNS image-suite contract."""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn


class NeuralopUnoAdapterBuildError(RuntimeError):
    """Raised when the external NeuralOperator U-NO surface is unavailable."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


def load_neuralop_uno_dependencies():
    try:
        neuralop_module = importlib.import_module("neuralop")
        neuralop_models = importlib.import_module("neuralop.models")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise NeuralopUnoAdapterBuildError(
            "model_dependency_unavailable",
            f"neuralop import failed for the U-NO adapter: {exc}",
        ) from exc

    uno_cls = getattr(neuralop_models, "UNO", None)
    if uno_cls is None:
        raise NeuralopUnoAdapterBuildError(
            "model_dependency_unavailable",
            "neuralop.models.UNO is unavailable in the installed neuraloperator package.",
        )

    return uno_cls, {
        "external_repo": "https://github.com/neuraloperator/neuraloperator",
        "external_distribution": "neuraloperator",
        "external_module": "neuralop.models.UNO",
        "external_version": getattr(neuralop_module, "__version__", None),
        "module_file": getattr(neuralop_module, "__file__", None),
        "torch_version": torch.__version__,
    }


class NeuralopUnoCnsModel(nn.Module):
    """Wrap `neuralop.models.UNO` behind a `B,C,H,W -> B,C,H,W` supervised interface."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, int],
        task_metadata: dict[str, Any] | None = None,
        hidden_channels: int = 32,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        n_layers: int = 4,
        modes: int = 12,
        positional_embedding: str = "grid",
        channel_mlp_skip: str = "linear",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        task_id = None if task_metadata is None else str(task_metadata.get("task_id"))
        if task_id not in (None, "2d_cfd_cns"):
            raise ValueError("NeuralopUnoCnsModel only supports task_metadata['task_id'] == '2d_cfd_cns'")

        uno_out_channels = [int(hidden_channels), int(hidden_channels * 2), int(hidden_channels * 2), int(hidden_channels)]
        uno_n_modes = [[int(modes), int(modes)] for _ in range(int(n_layers))]
        uno_scalings = [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]
        if int(n_layers) != 4:
            raise ValueError("NeuralopUnoCnsModel currently supports the frozen 4-layer U-NO contract only")

        uno_cls, provenance = load_neuralop_uno_dependencies()
        self.external_source_provenance = {
            **provenance,
            "local_contract_adaptation": {
                "task_id": "2d_cfd_cns",
                "input_layout": "B,C,H,W",
                "output_layout": "B,C,H,W",
                "positional_embedding": str(positional_embedding),
                "channel_mlp_skip": str(channel_mlp_skip),
                "external_model_config": {
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "hidden_channels": int(hidden_channels),
                    "lifting_channels": int(lifting_channels),
                    "projection_channels": int(projection_channels),
                    "n_layers": int(n_layers),
                    "uno_out_channels": uno_out_channels,
                    "uno_n_modes": uno_n_modes,
                    "uno_scalings": uno_scalings,
                },
            },
        }
        self.model = uno_cls(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=int(hidden_channels),
            lifting_channels=int(lifting_channels),
            projection_channels=int(projection_channels),
            positional_embedding=str(positional_embedding),
            n_layers=int(n_layers),
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            channel_mlp_skip=str(channel_mlp_skip),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"NeuralopUnoCnsModel expected {self.in_channels} input channels, got {channels}")
        if (int(height), int(width)) != self.spatial_shape:
            raise ValueError(
                f"NeuralopUnoCnsModel expected spatial_shape={self.spatial_shape}, got {(int(height), int(width))}"
            )
        output = self.model(x)
        expected = (batch_size, self.out_channels, int(height), int(width))
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"NeuralopUnoCnsModel expected UNO output shape {expected}, got {tuple(output.shape)}"
            )
        return output

"""PyTorch backend for PtychoPINN with lazy public compatibility exports."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any


try:
    import torch as _torch
except ImportError as exc:
    raise RuntimeError(
        "PyTorch is required for ptycho_torch package. "
        "Install PyTorch >= 2.2 with: pip install torch>=2.2"
    ) from exc

TORCH_AVAILABLE = True
del _torch

_LAZY_EXPORTS = {
    "to_model_config": ("ptycho_torch.config_bridge", "to_model_config"),
    "to_training_config": ("ptycho_torch.config_bridge", "to_training_config"),
    "to_inference_config": ("ptycho_torch.config_bridge", "to_inference_config"),
    "RawDataTorch": ("ptycho_torch.raw_data_bridge", "RawDataTorch"),
    "PtychoDataContainerTorch": (
        "ptycho_torch.data_container_bridge",
        "PtychoDataContainerTorch",
    ),
    "MemmapDatasetBridge": ("ptycho_torch.memmap_bridge", "MemmapDatasetBridge"),
    "run_cdi_example_torch": (
        "ptycho_torch.workflows.components",
        "run_cdi_example_torch",
    ),
    "train_cdi_model_torch": (
        "ptycho_torch.workflows.components",
        "train_cdi_model_torch",
    ),
    "load_inference_bundle_torch": (
        "ptycho_torch.workflows.components",
        "load_inference_bundle_torch",
    ),
}

__all__ = [
    "to_model_config",
    "to_training_config",
    "to_inference_config",
    "RawDataTorch",
    "PtychoDataContainerTorch",
    "MemmapDatasetBridge",
    "run_cdi_example_torch",
    "train_cdi_model_torch",
    "load_inference_bundle_torch",
    "TORCH_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attribute_name = target
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value

    qualified_name = f"{__name__}.{name}"
    if find_spec(qualified_name) is not None:
        module = import_module(qualified_name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

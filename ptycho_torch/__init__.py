"""
PyTorch backend for PtychoPINN.

This package provides PyTorch-compatible implementations and adapters for the
PtychoPINN ptychography reconstruction framework, maintaining parity with the
TensorFlow backend while enabling PyTorch-specific optimizations.

Public Exports:
---------------
- config_bridge: Configuration translation utilities (PyTorch to TensorFlow dataclasses)
- raw_data_bridge: Torch-optional RawDataTorch adapter
- data_container_bridge: Torch-optional PtychoDataContainerTorch for model-ready tensors
- workflows: Orchestration layer (run_cdi_example_torch, train_cdi_model_torch, etc.)
"""

# PyTorch is now a mandatory dependency (Phase F3.1/F3.2)
# Removed TORCH_AVAILABLE guard per phase_f_torch_mandatory.md F3.2
try:
    import torch
    TORCH_AVAILABLE = True  # Preserved for backward compatibility
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for ptycho_torch package. "
        "Install PyTorch >= 2.2 with: pip install torch>=2.2"
    ) from e

# Export config bridge (PyTorch is now mandatory)
from ptycho_torch.config_bridge import (
    to_model_config,
    to_training_config,
    to_inference_config
)

# Always export raw_data_bridge (torch-optional per Phase C.C1)
from ptycho_torch.raw_data_bridge import RawDataTorch

# Always export data_container_bridge (torch-optional per Phase C.C2)
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

# Always export memmap_bridge (torch-optional per Phase C.C3)
from ptycho_torch.memmap_bridge import MemmapDatasetBridge

# Always export workflows module (torch-optional per Phase D2.A)
from ptycho_torch.workflows.components import (
    run_cdi_example_torch,
    train_cdi_model_torch,
    load_inference_bundle_torch,
)

__all__ = [
    'to_model_config',
    'to_training_config',
    'to_inference_config',
    'RawDataTorch',
    'PtychoDataContainerTorch',
    'MemmapDatasetBridge',
    'run_cdi_example_torch',
    'train_cdi_model_torch',
    'load_inference_bundle_torch',
    'TORCH_AVAILABLE',
]

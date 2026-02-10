"""
PyTorch workflow orchestration module.

This package provides PyTorch equivalents of the TensorFlow workflow orchestration
functions defined in ptycho.workflows.components, maintaining API parity to enable
transparent backend selection from Ptychodus.

Module Structure:
- components.py: Core entry points (run_cdi_example_torch, train_cdi_model_torch, etc.)

Exports:
All workflow functions are re-exported from this package for convenient imports.
"""

# Torch-optional: This file must be importable even when torch unavailable
# All torch-specific imports are guarded inside components.py

from .components import (
    run_cdi_example_torch,
    train_cdi_model_torch,
    load_inference_bundle_torch,
)

__all__ = [
    "run_cdi_example_torch",
    "train_cdi_model_torch",
    "load_inference_bundle_torch",
]

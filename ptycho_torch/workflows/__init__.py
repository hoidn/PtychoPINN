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

# Torch-optional: importing this package without torch raises the actionable
# RuntimeError from ptycho_torch/__init__.py's package-level guard.

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

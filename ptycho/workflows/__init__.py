"""TensorFlow workflow orchestration module.

This package provides the TensorFlow orchestration facade defined in
``ptycho.workflows.components``, maintaining API parity with the PyTorch
twin facade under ``ptycho_torch.workflows``.

Module Structure:
- components.py: Core entry points (run_cdi_example, train_cdi_model, etc.)

Exports:
All workflow functions are re-exported from this package for convenient imports.
"""

# Import the facade first so that submodule-first imports (e.g. a cold
# ``import ptycho.workflows.workflow_orchestration``) resolve the components
# facade before any relocated slab re-imports it late-bound.  This mirrors the
# torch package's guard in ``ptycho_torch/workflows/__init__.py``.

from .components import (
    run_cdi_example,
    train_cdi_model,
    load_inference_bundle,
)

__all__ = [
    "run_cdi_example",
    "train_cdi_model",
    "load_inference_bundle",
]

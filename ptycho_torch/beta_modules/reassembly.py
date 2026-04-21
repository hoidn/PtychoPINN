"""Backward-compatibility shim. All functionality moved to ptycho_torch.reassembly."""
from ptycho_torch.reassembly import (
    VarProScaler,
    VectorizedWeightedAccumulator,
    reconstruct_image_barycentric as reconstruct_image_barycentric_weighted,
    detect_swap_probe_reference,
    equalize_by_ratio,
)

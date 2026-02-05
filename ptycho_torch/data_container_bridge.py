"""
Torch-optional PtychoDataContainerTorch adapter for PyTorch/TensorFlow data pipeline parity.

This module provides a lightweight container that mirrors TensorFlow's `PtychoDataContainer` API
while maintaining torch-optional behavior, enabling PyTorch workflows to consume the same
grouped data dictionary structure produced by RawData.generate_grouped_data().

Architecture Role:
------------------
This adapter completes the NumPy → tensor conversion pipeline for PyTorch:
    NPZ files → RawDataTorch → grouped dict → PtychoDataContainerTorch → model tensors

The container accepts the grouped data dictionary (from RawDataTorch.generate_grouped_data())
and exposes TensorFlow-compatible tensor attributes, using PyTorch tensors when available
or falling back to NumPy arrays when PyTorch is not installed.

Key Design Decisions:
---------------------
1. **Torch-Optional**: Module is importable without PyTorch; uses NumPy fallback when unavailable
2. **API Parity**: Exposes identical attributes to TensorFlow PtychoDataContainer (X, Y, Y_I, Y_phi, etc.)
3. **Delegation Pattern**: Reuses grouped data from RawDataTorch (mirrors Phase C.C1 strategy)
4. **DATA-001 Compliance**: Enforces complex64 dtype for Y patches with explicit validation
5. **Dtype Consistency**: Matches TensorFlow tensor dtypes (float32, complex64, int32, float64)

Public Interface:
-----------------
    PtychoDataContainerTorch(grouped_data, probe)
        Constructor accepting grouped data dict + probe tensor.

    Attributes (matching TensorFlow PtychoDataContainer):
        X: (n_images, N, N, gridsize²) float32/torch.float32 — diffraction patterns
        Y: (n_images, N, N, gridsize²) complex64/torch.complex64 — combined ground truth
        Y_I: (n_images, N, N, gridsize²) float32/torch.float32 — amplitude patches
        Y_phi: (n_images, N, N, gridsize²) float32/torch.float32 — phase patches
        coords_nominal: (n_images, 1, 2, gridsize²) float32/torch.float32 — scan coordinates
        coords_true: (n_images, 1, 2, gridsize²) float32/torch.float32 — true coordinates (alias for coords_nominal)
        coords: (alias for coords_nominal)
        probe: (N, N) complex64/torch.complex64 — probe function
        nn_indices: (n_images, gridsize²) int32/torch.int32 — nearest neighbor indices
        global_offsets: (n_images, 1, 2, 1) float64/torch.float64 — global coordinate offsets
        local_offsets: (n_images, 1, 2, gridsize²) float64/torch.float64 — local offsets
        norm_Y_I: Optional normalization factor (preserved from TensorFlow)
        YY_full: Optional full object reconstruction (preserved from TensorFlow)

Usage Example:
--------------
    from ptycho_torch.raw_data_bridge import RawDataTorch
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
    from ptycho.config.config import TrainingConfig, ModelConfig
    import numpy as np

    # Create configuration
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        n_groups=10,
        neighbor_count=4,
        nphotons=1e9
    )

    # Create adapter
    raw_data = RawDataTorch(xcoords, ycoords, diff3d, probe, scan_index, obj, config)

    # Generate grouped data
    grouped = raw_data.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2)

    # Create container
    container = PtychoDataContainerTorch(grouped, probe)

    # Access tensors (PyTorch when available, NumPy otherwise)
    X = container.X  # shape (10, 64, 64, 4)
    Y = container.Y  # shape (10, 64, 64, 4), dtype complex64
    coords = container.coords_nominal  # shape (10, 1, 2, 4)

Contract Compliance:
--------------------
- Data Contract: specs/data_contracts.md §3 (PtychoDataContainer attributes)
- API Spec: specs/ptychodus_api_spec.md §4.4 (container tensor requirements)
- Phase C Plan: plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md (C.C2)
- Test Contract: tests/torch/test_data_pipeline.py::TestDataContainerParity

Source References:
------------------
- TensorFlow PtychoDataContainer: ptycho/loader.py:93-138
- Grouped Data Contract: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md
- Evidence Requirements: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T075914Z/data_container_requirements.md
- Test Blueprint: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/test_blueprint.md

Findings Applied:
-----------------
- DATA-001: Y patches MUST be complex64 (not float64)
- CONFIG-001: params.cfg initialization handled upstream by RawDataTorch
- NORMALIZATION-001: Preserve normalization from grouped data (no photon scaling)
"""

import numpy as np
from typing import Dict, Any, Optional, Union

# PyTorch is now a mandatory dependency (Phase F3.1/F3.2)
try:
    import torch
    TensorType = Union[torch.Tensor, np.ndarray]
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for ptycho_torch modules. "
        "Install PyTorch >= 2.2 with: pip install torch>=2.2"
    ) from e


class PtychoDataContainerTorch:
    """
    Torch-optional container for model-ready ptychographic data.

    This container mirrors the TensorFlow `PtychoDataContainer` API while providing
    torch-optional behavior. When PyTorch is available, tensors are returned as
    torch.Tensor objects; otherwise, NumPy arrays are used.

    All tensor attributes maintain the same shapes and dtypes as the TensorFlow
    implementation to ensure compatibility with existing workflows.

    Attributes:
        X: Diffraction patterns (amplitude, not intensity)
        Y: Combined complex ground truth patches
        Y_I: Amplitude component of ground truth
        Y_phi: Phase component of ground truth
        coords_nominal: Nominal scan coordinates
        coords_true: True scan coordinates (alias for coords_nominal)
        coords: Convenience alias for coords_nominal
        probe: Probe function
        nn_indices: Nearest neighbor indices from grouping
        global_offsets: Global coordinate offsets
        local_offsets: Local coordinate offsets per channel
        norm_Y_I: Optional normalization factors
        YY_full: Optional full object reconstruction

    Critical Requirements (DATA-001):
        - Y patches MUST be complex64 (torch.complex64 when torch available)
        - Silent dtype conversion to float64 is forbidden and will raise ValueError
    """

    def __init__(self, grouped_data: Dict[str, np.ndarray], probe: np.ndarray):
        """
        Create PtychoDataContainerTorch from grouped data dictionary.

        Args:
            grouped_data: Dictionary from RawDataTorch.generate_grouped_data() with keys:
                - 'X_full': (nsamples, N, N, gridsize²) float32 — normalized diffraction
                - 'Y': (nsamples, N, N, gridsize²) complex64 — ground truth patches
                - 'coords_relative': (nsamples, 1, 2, gridsize²) float32 — coordinates
                - 'coords_offsets': (nsamples, 1, 2, 1) float64 — global offsets
                - 'nn_indices': (nsamples, gridsize²) int32 — neighbor indices
                - Additional optional keys
            probe: Probe function, shape (N, N), dtype complex64

        Raises:
            ValueError: If Y patches are not complex64 (DATA-001 violation)
            ValueError: If required keys are missing from grouped_data
            TypeError: If dtypes do not match expected contract

        Note:
            This constructor does NOT modify params.cfg. Configuration initialization
            must be handled upstream by RawDataTorch (per CONFIG-001).
        """
        # Validate required keys are present
        required_keys = ['X_full', 'coords_relative', 'coords_offsets', 'nn_indices']
        missing_keys = [k for k in required_keys if k not in grouped_data]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in grouped_data: {missing_keys}. "
                f"Expected keys from RawDataTorch.generate_grouped_data(): {required_keys}"
            )

        # Extract and validate diffraction data (X_full → X)
        X_np = grouped_data['X_full']
        if X_np.dtype != np.float32:
            raise TypeError(
                f"X_full dtype must be float32, got {X_np.dtype}. "
                f"Check data_contracts.md normalization requirements."
            )

        # Extract and validate ground truth (Y)
        if grouped_data['Y'] is not None:
            Y_raw = grouped_data['Y']
            # Convert TensorFlow tensors to NumPy if needed
            if hasattr(Y_raw, 'numpy'):  # TensorFlow tensor
                Y_np = Y_raw.numpy()
            else:
                Y_np = np.asarray(Y_raw)

            # CRITICAL DATA-001 validation: Y MUST be complex64
            if Y_np.dtype != np.complex64:
                raise ValueError(
                    f"DATA-001 violation: Y patches MUST be complex64, got {Y_np.dtype}. "
                    f"Historical bug: silent float64 conversion caused major training failure. "
                    f"See docs/findings.md:DATA-001 and specs/data_contracts.md:19"
                )
        else:
            # Create dummy complex tensor matching X shape (per TensorFlow loader.py:310-313)
            Y_np = np.ones_like(X_np, dtype=np.complex64)
            print("PtychoDataContainerTorch: setting dummy Y ground truth with correct channel shape.")

        # Validate probe dtype
        probe_np = np.asarray(probe, dtype=np.complex64)
        if probe_np.dtype != np.complex64:
            raise TypeError(
                f"probe dtype must be complex64, got {probe_np.dtype}"
            )

        # Convert to torch tensors (PyTorch is mandatory, no NumPy fallback)
        # PyTorch tensor conversion with explicit dtype specifications
        self.X = torch.from_numpy(X_np).to(torch.float32)
        self.Y = torch.from_numpy(Y_np).to(torch.complex64)
        self.Y_I = torch.abs(self.Y).to(torch.float32)
        self.Y_phi = torch.angle(self.Y).to(torch.float32)
        self.probe = torch.from_numpy(probe_np).to(torch.complex64)

        # Coordinates and offsets
        self.coords_nominal = torch.from_numpy(
            grouped_data['coords_relative']
        ).to(torch.float32)
        self.coords_relative = self.coords_nominal  # Explicit alias for relative offsets
        self.coords_true = self.coords_nominal  # Alias per TensorFlow loader.py:295
        self.global_offsets = torch.from_numpy(
            grouped_data['coords_offsets']
        ).to(torch.float64)  # Keep float64 per TF baseline

        # nn_indices (int32)
        self.nn_indices = torch.from_numpy(
            grouped_data['nn_indices']
        ).to(torch.int32)

        # local_offsets (same as coords_relative per TF loader.py:338)
        self.local_offsets = torch.from_numpy(
            grouped_data['coords_relative']
        ).to(torch.float64)  # Keep float64 per TF baseline

        # Convenience alias (per TensorFlow loader.py:129)
        self.coords = self.coords_nominal

        # Optional attributes (preserved from TensorFlow PtychoDataContainer)
        self.norm_Y_I = grouped_data.get('norm_Y_I', None)
        self.YY_full = grouped_data.get('objectGuess', None)

    def __repr__(self) -> str:
        """
        Debug representation showing tensor shapes and dtypes.

        Mirrors TensorFlow PtychoDataContainer.__repr__ format for parity debugging.
        """
        repr_str = '<PtychoDataContainerTorch'
        for attr_name in ['X', 'Y_I', 'Y_phi', 'coords_nominal', 'probe',
                          'nn_indices', 'global_offsets', 'local_offsets']:
            attr = getattr(self, attr_name, None)
            if attr is not None:
                # Get shape (works for both torch.Tensor and np.ndarray)
                shape = tuple(attr.shape) if hasattr(attr, 'shape') else None

                # Get dtype (all attributes are now torch.Tensor, no fallback)
                if hasattr(attr, 'dtype'):
                    dtype = str(attr.dtype).replace('torch.', '')
                else:
                    dtype = 'unknown'

                repr_str += f' {attr_name}={shape}/{dtype}'

        repr_str += '>'
        return repr_str

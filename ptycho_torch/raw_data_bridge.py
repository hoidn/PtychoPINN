"""
Torch-optional RawDataTorch adapter bridging PyTorch and TensorFlow data pipelines.

This module provides a lightweight wrapper around `ptycho.raw_data.RawData` that enables
PyTorch workflows to reuse the proven TensorFlow grouping implementation while maintaining
torch-optional behavior for testing and fallback scenarios.

Architecture Role:
-----------------
This adapter sits at the boundary between PyTorch-specific code and the legacy TensorFlow
data pipeline. It ensures configuration state is correctly initialized via the config bridge
before delegating to TensorFlow's RawData implementation.

Key Design Decisions:
---------------------
1. **Delegation over Reimplementation**: Wraps `ptycho.raw_data.RawData` instead of
   duplicating grouping logic (maintaining parity per INTEGRATE-PYTORCH-001 Phase C.C1).

2. **Torch-Optional**: Module is importable without PyTorch installed; only imports torch
   for future tensor conversion utilities (currently not used).

3. **Configuration Bridge Enforcement**: Constructor accepts dataclass configs and calls
   `update_legacy_dict()` internally, preventing CONFIG-001 shape mismatch bugs.

4. **NumPy-First Output**: Returns NumPy arrays (matching TensorFlow RawData behavior)
   to maintain compatibility with existing test fixtures and downstream code.

Public Interface:
-----------------
    RawDataTorch(xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess, config)
        Wrapper constructor accepting NPZ-like arrays plus optional TrainingConfig.

    generate_grouped_data(N, K, nsamples, seed, sequential_sampling, gridsize)
        Delegates to `ptycho.raw_data.RawData.generate_grouped_data()`.
        Returns dict with keys per specs/data_contracts.md §2:
        - 'diffraction': (nsamples, N, N, gridsize²) float32
        - 'X_full': (nsamples, N, N, gridsize²) float32
        - 'coords_offsets': (nsamples, 1, 2, 1) float32
        - 'coords_relative': (nsamples, 1, 2, gridsize²) float32
        - 'nn_indices': (nsamples, gridsize²) int32
        - 'Y': (nsamples, N, N, gridsize²) complex64 (if objectGuess provided)

Usage Example:
--------------
    from ptycho_torch.raw_data_bridge import RawDataTorch
    from ptycho.config.config import TrainingConfig, ModelConfig

    # Create configuration
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        n_groups=512,
        neighbor_count=4,
        nphotons=1e9
    )

    # Create adapter (auto-initializes params.cfg)
    raw_data = RawDataTorch(
        xcoords=coords_x,
        ycoords=coords_y,
        diff3d=diffraction_patterns,
        probeGuess=probe,
        scan_index=indices,
        objectGuess=obj,
        config=config
    )

    # Generate grouped data (delegates to TensorFlow)
    grouped = raw_data.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2)

    # Access outputs
    assert grouped['diffraction'].shape == (10, 64, 64, 4)
    assert grouped['diffraction'].dtype == np.float32

Contract Compliance:
--------------------
- Data Contract: specs/data_contracts.md §1 (NPZ schema), §2 (grouped data dict)
- API Spec: specs/ptychodus_api_spec.md §4.3 (data ingestion)
- Phase C Plan: plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md (C.C1)
- Test Contract: tests/torch/test_data_pipeline.py::TestRawDataTorchAdapter

Source References:
------------------
- TensorFlow RawData: ptycho/raw_data.py:126-363
- Config Bridge: ptycho_torch/config_bridge.py (Phase B.B3 implementation)
- Data Contract: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md
- Test Blueprint: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/test_blueprint.md

Findings Applied:
-----------------
- CONFIG-001: Mandatory `update_legacy_dict()` before data operations
- DATA-001: Preserve complex64 dtype for Y patches
- NORMALIZATION-001: Do not apply photon scaling to data (handled by config)
"""

import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

# Torch-optional import guard (per test_blueprint.md §1.C and CLAUDE.md:57-59)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RawDataTorch:
    """
    Torch-optional wrapper for ptycho.raw_data.RawData.

    This class provides a PyTorch-compatible interface to the TensorFlow data pipeline
    while maintaining backward compatibility and torch-optional behavior.

    Attributes:
        _tf_raw_data: Underlying TensorFlow RawData instance (delegation target)
        _config: Optional TrainingConfig/InferenceConfig for params initialization

    Critical Gotchas (per CLAUDE.md:76-93):
    ----------------------------------------
    1. MUST call update_legacy_dict() before generate_grouped_data()
    2. Failure causes shape mismatches (e.g., (*, 64, 64, 1) instead of (*, 64, 64, 4))
    3. This adapter handles initialization automatically in constructor
    """

    def __init__(
        self,
        xcoords: np.ndarray,
        ycoords: np.ndarray,
        diff3d: np.ndarray,
        probeGuess: np.ndarray,
        scan_index: np.ndarray,
        objectGuess: Optional[np.ndarray] = None,
        config: Optional[Any] = None
    ):
        """
        Create RawDataTorch adapter from NPZ-like arrays.

        Args:
            xcoords: X-coordinates of scan positions, shape (n_images,), dtype float64
            ycoords: Y-coordinates of scan positions, shape (n_images,), dtype float64
            diff3d: Diffraction patterns (amplitude), shape (n_images, N, N), dtype float32
            probeGuess: Probe function, shape (N, N), dtype complex64
            scan_index: Scan point indices, shape (n_images,), dtype int
            objectGuess: Full object (optional), shape (M, M), dtype complex64
            config: Optional TrainingConfig/InferenceConfig for params initialization

        Raises:
            ImportError: If ptycho module unavailable (installation issue)
            ValueError: If data arrays have incompatible shapes

        Note:
            If config is provided, this constructor calls update_legacy_dict(params.cfg, config)
            automatically, preventing CONFIG-001 shape mismatch bugs. If config is None,
            assumes params.cfg is already initialized (e.g., from loaded model).
        """
        # Import TensorFlow dependencies
        from ptycho.raw_data import RawData
        from ptycho.config.config import update_legacy_dict
        from ptycho import params as p

        # Initialize params.cfg if config provided (CRITICAL per CONFIG-001)
        if config is not None:
            update_legacy_dict(p.cfg, config)

        # Delegate to TensorFlow RawData via factory method
        # (avoids need for separate start coordinates)
        self._tf_raw_data = RawData.from_coords_without_pc(
            xcoords=xcoords,
            ycoords=ycoords,
            diff3d=diff3d,
            probeGuess=probeGuess,
            scan_index=scan_index,
            objectGuess=objectGuess
        )

        # Store config reference for potential future use
        self._config = config

    def generate_grouped_data(
        self,
        N: int,
        K: int = 4,
        nsamples: int = 1,
        seed: Optional[int] = None,
        sequential_sampling: bool = False,
        gridsize: Optional[int] = None,
        dataset_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate nearest-neighbor grouped data by delegating to TensorFlow RawData.

        This method wraps `ptycho.raw_data.RawData.generate_grouped_data()` without
        modification, ensuring 100% parity with TensorFlow behavior.

        Args:
            N: Size of solution region (diffraction pattern crop size)
            K: Number of nearest neighbors (default 4)
            nsamples: Number of groups to generate (default 1)
            seed: Optional random seed for reproducibility
            sequential_sampling: If True, use first N points instead of random (default False)
            gridsize: Explicit gridsize override (if None, uses params.cfg['gridsize'])
            dataset_path: Optional path for caching (forwarded to TensorFlow, currently ignored for caching)

        Returns:
            Dictionary containing grouped data with keys (per data_contracts.md §2):
                - 'diffraction': (nsamples, N, N, gridsize²) float32
                - 'X_full': (nsamples, N, N, gridsize²) float32 (normalized)
                - 'coords_offsets': (nsamples, 1, 2, 1) float32
                - 'coords_relative': (nsamples, 1, 2, gridsize²) float32
                - 'nn_indices': (nsamples, gridsize²) int32
                - 'Y': (nsamples, N, N, gridsize²) complex64 (if objectGuess provided)
                - 'objectGuess': (M, M) complex64 (if provided)
                - Additional optional keys for start coordinates

        Raises:
            ValueError: If dataset too small for requested parameters
            RuntimeError: If params.cfg not initialized (CONFIG-001)

        Example:
            grouped = raw_data.generate_grouped_data(
                N=64, K=4, nsamples=10, gridsize=2, seed=42
            )
            assert grouped['diffraction'].shape == (10, 64, 64, 4)
            assert grouped['diffraction'].dtype == np.float32
            assert grouped['nn_indices'].shape == (10, 4)

        Source: ptycho/raw_data.py:365-486 (TensorFlow implementation)
        Contract: specs/data_contracts.md:58-176 (grouped data structure)
        """
        # Direct delegation to TensorFlow RawData (maintaining parity)
        return self._tf_raw_data.generate_grouped_data(
            N=N,
            K=K,
            nsamples=nsamples,
            dataset_path=dataset_path,  # Forwarded to TensorFlow (currently ignored for caching)
            seed=seed,
            sequential_sampling=sequential_sampling,
            gridsize=gridsize
        )

    @property
    def probeGuess(self) -> np.ndarray:
        """Access underlying probe guess from TensorFlow RawData."""
        return self._tf_raw_data.probeGuess

    @property
    def objectGuess(self) -> Optional[np.ndarray]:
        """Access underlying object guess from TensorFlow RawData."""
        return self._tf_raw_data.objectGuess

    @property
    def xcoords(self) -> np.ndarray:
        """Access x-coordinates from TensorFlow RawData."""
        return self._tf_raw_data.xcoords

    @property
    def ycoords(self) -> np.ndarray:
        """Access y-coordinates from TensorFlow RawData."""
        return self._tf_raw_data.ycoords

    @property
    def diff3d(self) -> np.ndarray:
        """Access diffraction patterns from TensorFlow RawData."""
        return self._tf_raw_data.diff3d

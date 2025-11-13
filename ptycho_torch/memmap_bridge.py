"""
Memory-mapped dataset bridge delegating to RawDataTorch.

Phase C.C3 Goal: Connect NPZ files with memory-mapped loading to RawDataTorch
adapter while preserving grouping semantics and cache reuse.

Design Principles:
1. Delegation over reimplementation (reuse RawDataTorch grouping)
2. Config bridge integration (dataclass configs, update_legacy_dict)
3. Torch-optional (graceful fallback to NumPy)
4. Minimal surface area (simple adapter, not full PyTorch Dataset)

Source Contracts:
- specs/data_contracts.md:7-70 (NPZ schema, grouped-data dict)
- specs/ptychodus_api_spec.md:164-215 (data ingestion requirements)
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T082035Z/memmap_bridge_analysis.md

Evidence Traceability:
- Phase C.C3 requirement per plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:46
- Gap #1 (duplicate grouping) addressed via RawDataTorch delegation
- Gap #2 (config bridge bypass) fixed via constructor config parameter
- Gap #3 (output format mismatch) resolved by returning grouped-data dict

Author: Ralph (engineer loop #37)
Date: 2025-10-17
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

# PyTorch is now a mandatory dependency (Phase F3.1/F3.2)
# This module does not directly use torch but downstream consumers require it
try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for ptycho_torch modules. "
        "Install PyTorch >= 2.2 with: pip install torch>=2.2"
    ) from e


class MemmapDatasetBridge:
    """
    Lightweight bridge connecting memory-mapped NPZ files to RawDataTorch.

    Delegates grouping logic to RawDataTorch (which delegates to TensorFlow RawData),
    ensuring parity with existing TensorFlow data pipeline while supporting large
    datasets through memory-mapped array access.

    Args:
        npz_path: Path to NPZ file containing ptychography dataset
        config: TrainingConfig or InferenceConfig instance (triggers config bridge)
        memmap_dir: Directory for memory-mapped cache (currently unused, preserved for API)

    Example:
        >>> from ptycho.config.config import TrainingConfig, ModelConfig
        >>> config = TrainingConfig(
        ...     model=ModelConfig(N=64, gridsize=2),
        ...     n_groups=100,
        ...     neighbor_count=4,
        ...     nphotons=1e9
        ... )
        >>> bridge = MemmapDatasetBridge("dataset.npz", config)
        >>> grouped = bridge.get_grouped_data(N=64, K=4, nsamples=100, gridsize=2)
        >>> print(grouped.keys())
        dict_keys(['diffraction', 'X_full', 'coords_offsets', 'coords_relative', 'nn_indices', ...])

    Contract:
        - Input NPZ must conform to specs/data_contracts.md:7-70
        - Output grouped dict matches RawData.generate_grouped_data() schema
        - Config bridge called automatically (satisfies CONFIG-001 finding)
        - Cache reuse inherited from TensorFlow RawData (.groups_cache.npz)
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        config,  # TrainingConfig or InferenceConfig
        memmap_dir: str = "data/memmap"
    ):
        """
        Initialize memory-mapped dataset bridge with RawDataTorch delegation.

        Loads NPZ data (with optional memory mapping), instantiates RawDataTorch,
        and ensures params.cfg synchronization via config bridge.

        Args:
            npz_path: Path to NPZ file
            config: TrainingConfig or InferenceConfig (used for config bridge)
            memmap_dir: Cache directory hint (preserved for API compatibility,
                       actual cache managed by TensorFlow RawData)
        """
        from ptycho_torch.raw_data_bridge import RawDataTorch

        self.npz_path = Path(npz_path)
        self.config = config
        self.memmap_dir = memmap_dir

        # Load NPZ data with memory mapping for large datasets
        # (mmap_mode='r' provides read-only memory-mapped access)
        self._npz_data = np.load(self.npz_path, mmap_mode='r')

        # Extract required arrays (per specs/data_contracts.md:13-21)
        # Note: Using .astype() forces materialization for small datasets,
        # but for large datasets the memory-mapped arrays remain lazy
        self.xcoords = self._get_array('xcoords', np.float64)
        self.ycoords = self._get_array('ycoords', np.float64)

        # DATA-001 compliance: Accept both canonical 'diffraction' and legacy 'diff3d' keys
        # Spec: specs/data_contracts.md:207 - canonical key is 'diffraction'
        # Pattern: scripts/run_tike_reconstruction.py:165-169, generate_patches_tool.py:66-68
        # Historical context: Phase E training blocked by KeyError (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001)
        if 'diffraction' in self._npz_data:
            self.diff3d = self._get_array('diffraction', np.float32)
        elif 'diff3d' in self._npz_data:
            self.diff3d = self._get_array('diff3d', np.float32)
        else:
            raise KeyError(
                f"Required diffraction data missing from NPZ file {self.npz_path}. "
                f"Need either 'diffraction' (canonical, per DATA-001) or 'diff3d' (legacy). "
                f"Available keys: {list(self._npz_data.keys())}. "
                f"See specs/data_contracts.md:207 for schema."
            )

        self.probeGuess = self._get_array('probeGuess', np.complex64)
        self.objectGuess = self._get_array('objectGuess', np.complex64)
        self.scan_index = self._get_array('scan_index', np.int32, optional=True)

        # Create scan_index if missing (per data_contracts.md:21)
        if self.scan_index is None:
            self.scan_index = np.arange(len(self.xcoords), dtype=np.int32)

        # Instantiate RawDataTorch adapter (delegates to TensorFlow RawData)
        # Config passed automatically triggers update_legacy_dict (CONFIG-001)
        self.raw_data_torch = RawDataTorch(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            diff3d=self.diff3d,
            probeGuess=self.probeGuess,
            scan_index=self.scan_index,
            objectGuess=self.objectGuess,
            config=config  # Triggers config bridge (ptycho_torch/raw_data_bridge.py:109)
        )

    def _get_array(
        self,
        key: str,
        dtype: np.dtype,
        optional: bool = False
    ) -> Optional[np.ndarray]:
        """
        Extract and cast array from NPZ file.

        Args:
            key: NPZ key name
            dtype: Target NumPy dtype
            optional: If True, return None for missing keys instead of raising

        Returns:
            NumPy array cast to specified dtype, or None if optional and missing

        Raises:
            KeyError: If required key missing from NPZ
        """
        if key not in self._npz_data:
            if optional:
                return None
            raise KeyError(
                f"Required key '{key}' missing from NPZ file {self.npz_path}. "
                f"Available keys: {list(self._npz_data.keys())}. "
                f"See specs/data_contracts.md:13-21 for required schema."
            )

        arr = self._npz_data[key]

        # Cast to target dtype (forces materialization for memory-mapped arrays)
        # For large arrays, consider lazy casting or chunked processing if memory becomes an issue
        if arr.dtype != dtype:
            arr = arr.astype(dtype)

        return arr

    def get_grouped_data(
        self,
        N: int,
        K: int,
        nsamples: int,
        gridsize: int,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate grouped data by delegating to RawDataTorch.

        This is the primary method for obtaining grouped data compatible with
        PtychoDataContainerTorch. All grouping logic is delegated to RawDataTorch,
        which further delegates to TensorFlow RawData.generate_grouped_data().

        Args:
            N: Crop size for diffraction patterns (e.g., 64, 128)
            K: Number of nearest neighbors to consider for grouping
            nsamples: Number of grouped samples to generate
            gridsize: Group size (e.g., 2 for 2x2 patches = 4 images per group)
            seed: Random seed for reproducibility (optional)

        Returns:
            Grouped data dictionary with keys:
            - 'diffraction': (nsamples, N, N, gridsize²) float32
            - 'X_full': (nsamples, N, N, gridsize²) float32
            - 'coords_offsets': (nsamples, 1, 2, 1) float64
            - 'coords_relative': (nsamples, 1, 2, gridsize²) float32
            - 'local_offsets': (nsamples, 1, 2, gridsize²) float32
            - 'nn_indices': (nsamples, gridsize²) int32
            - And other keys per specs/data_contracts.md:110-176

        Example:
            >>> grouped = bridge.get_grouped_data(N=64, K=4, nsamples=100, gridsize=2, seed=42)
            >>> print(grouped['diffraction'].shape)
            (100, 64, 64, 4)

        Contract:
            - Output dict matches TensorFlow RawData.generate_grouped_data() exactly
            - Delegation ensures grouping parity (no duplicate logic)
            - Cache reuse inherited from underlying RawData (see .groups_cache.npz)
        """
        # Delegate to RawDataTorch adapter (which delegates to TensorFlow RawData)
        # This satisfies Phase C.C3 requirement: "delegate grouping to RawDataTorch"
        return self.raw_data_torch.generate_grouped_data(
            N=N,
            K=K,
            nsamples=nsamples,
            gridsize=gridsize,
            seed=seed
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MemmapDatasetBridge("
            f"npz_path={self.npz_path}, "
            f"n_points={len(self.xcoords)}, "
            f"diff_shape={self.diff3d.shape})"
        )

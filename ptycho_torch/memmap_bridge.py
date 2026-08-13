"""Historical RAM materializer retained until the Phase 2 loader convergence."""

from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

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
    Compatibility bridge that materializes an NPZ through RawDataTorch in RAM.

    Args:
        npz_path: Path to NPZ file containing ptychography dataset
        config: TrainingConfig or InferenceConfig instance (triggers config bridge)
        memmap_dir: Unused compatibility argument

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
        - This class does not provide memory mapping or cache ownership
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        config,  # TrainingConfig or InferenceConfig
        memmap_dir: str = "data/memmap"
    ):
        """
        Initialize memory-mapped dataset bridge with RawDataTorch delegation.

        Load the NPZ into a RawDataTorch compatibility adapter.

        Args:
            npz_path: Path to NPZ file
            config: TrainingConfig or InferenceConfig (used for config bridge)
            memmap_dir: Unused compatibility argument
        """
        from ptycho.acquisition import decode_acquisition
        from ptycho_torch.raw_data_bridge import RawDataTorch

        self.npz_path = Path(npz_path)
        self.config = config
        self.memmap_dir = memmap_dir

        record = decode_acquisition(self.npz_path)
        record = replace(
            record,
            xcoords=np.asarray(record.xcoords, dtype=np.float64),
            ycoords=np.asarray(record.ycoords, dtype=np.float64),
            diff3d=np.asarray(record.diff3d, dtype=np.float32),
            probeGuess=np.asarray(record.probeGuess, dtype=np.complex64),
            scan_index=np.asarray(record.scan_index, dtype=np.int32),
            objectGuess=(
                None
                if record.objectGuess is None
                else np.asarray(record.objectGuess, dtype=np.complex64)
            ),
        )
        self.xcoords = record.xcoords
        self.ycoords = record.ycoords
        self.diff3d = record.diff3d
        self.probeGuess = record.probeGuess
        self.objectGuess = record.objectGuess
        self.scan_index = record.scan_index
        self.raw_data_torch = RawDataTorch.from_acquisition(record, config=config)

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

        Group through the RawDataTorch compatibility adapter.

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

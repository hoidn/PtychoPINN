"""Torch tensor container for canonical ``RawData`` grouped output."""

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
    """Model-ready Torch view of a canonical grouped-data dictionary.

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
            grouped_data: Dictionary from RawData.generate_grouped_data() with keys:
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

        This constructor does not read or modify ``params.cfg``.
        """
        # Validate required keys are present
        required_keys = ['X_full', 'coords_relative', 'coords_offsets', 'nn_indices']
        missing_keys = [k for k in required_keys if k not in grouped_data]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in grouped_data: {missing_keys}. "
                f"Expected keys from RawData.generate_grouped_data(): {required_keys}"
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
                    f"See specs/data_contracts.md for the public data contract."
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

        sample_count = int(self.X.shape[0])
        for name in ("object_index", "experiment_id"):
            values = grouped_data.get(name)
            if values is None:
                values = np.zeros(sample_count, dtype=np.int64)
            values = np.asarray(values)
            if values.shape != (sample_count,):
                raise ValueError(
                    f"{name} must have shape ({sample_count},), got {values.shape}."
                )
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{name} must contain integers, got {values.dtype}.")
            setattr(
                self,
                name,
                torch.from_numpy(values.astype(np.int64, copy=False)),
            )

        raw_grouped = grouped_data.get("diffraction")
        self.raw_grouped_diffraction = (
            None
            if raw_grouped is None
            else np.ascontiguousarray(raw_grouped, dtype=np.float32)
        )

        # Coordinates and offsets
        # Verbatim pass-through (matches TF ptycho/loader.py per
        # ptychodus_api_spec.md:172; see docs/findings.md TORCH-REASSEMBLY-SIGN-001).
        coords_relative = grouped_data['coords_relative']
        self.coords_nominal = torch.from_numpy(
            coords_relative
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
            coords_relative
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

"""Framework-neutral records for acquisition data crossing backend boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class AcquisitionRecord:
    """The NumPy acquisition state needed to reconstruct a ``RawData`` adapter.

    This record deliberately contains no loading, grouping, tensor conversion, or
    backend behavior. Those operations remain owned by their existing adapters.
    """

    xcoords: np.ndarray
    ycoords: np.ndarray
    xcoords_start: Optional[np.ndarray]
    ycoords_start: Optional[np.ndarray]
    diff3d: Optional[np.ndarray]
    probeGuess: Optional[np.ndarray]
    scan_index: Optional[np.ndarray]
    objectGuess: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    norm_Y_I: Any = None
    metadata: Any = None
    sample_indices: Optional[np.ndarray] = None
    subsample_seed: Optional[int] = None

    @classmethod
    def from_raw_data(cls, raw_data: Any) -> "AcquisitionRecord":
        """Snapshot the acquisition fields consumed by the RawData bridge."""

        return cls(
            xcoords=raw_data.xcoords,
            ycoords=raw_data.ycoords,
            xcoords_start=raw_data.xcoords_start,
            ycoords_start=raw_data.ycoords_start,
            diff3d=raw_data.diff3d,
            probeGuess=raw_data.probeGuess,
            scan_index=raw_data.scan_index,
            objectGuess=raw_data.objectGuess,
            Y=getattr(raw_data, "Y", None),
            norm_Y_I=getattr(raw_data, "norm_Y_I", None),
            metadata=getattr(raw_data, "metadata", None),
            sample_indices=getattr(raw_data, "sample_indices", None),
            subsample_seed=getattr(raw_data, "subsample_seed", None),
        )

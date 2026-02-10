from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_DP_KEYS = ("dp",)
REQUIRED_PARA_KEYS = ("object", "probe", "probe_position_x_m", "probe_position_y_m")


@dataclass(frozen=True)
class PtychoViTHdf5Pair:
    dp_hdf5: Path
    para_hdf5: Path
    object_name: str


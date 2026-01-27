"""Grid-based lines workflow orchestration.

Skeleton module for running probe prep → grid simulation → training → inference →
metrics for the deprecated ptycho_lines workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np

from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p


@dataclass
class GridLinesConfig:
    N: int
    gridsize: int
    output_dir: Path
    probe_npz: Path
    size: int = 392
    offset: int = 4
    outer_offset_train: int = 8
    outer_offset_test: int = 20
    nimgs_train: int = 2
    nimgs_test: int = 2
    nphotons: float = 1e9
    nepochs: int = 60
    batch_size: int = 16
    nll_weight: float = 0.0
    mae_weight: float = 1.0
    realspace_weight: float = 0.0
    probe_smoothing_sigma: float = 0.5


def run_grid_lines_workflow(cfg: GridLinesConfig) -> Dict[str, Any]:
    """Orchestrate probe prep → sim → train → infer → stitch → metrics."""
    raise NotImplementedError

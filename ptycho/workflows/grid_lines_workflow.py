"""Grid-based lines workflow orchestration.

Skeleton module for running probe prep → grid simulation → training → inference →
metrics for the deprecated ptycho_lines workflow.

Data contracts: see specs/data_contracts.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np

from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from scripts.tools.prepare_data_tool import interpolate_array, smooth_complex_array


@dataclass
class GridLinesConfig:
    """Configuration for grid-based lines workflow.

    See docs/plans/2026-01-27-grid-lines-workflow.md for parameter details.
    """

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


# ---------------------------------------------------------------------------
# Probe Extraction + Upscaling Helpers (Task 2)
# ---------------------------------------------------------------------------


def load_probe_guess(npz_path: Path) -> np.ndarray:
    """Load probeGuess from NPZ file."""
    data = np.load(npz_path)
    if "probeGuess" not in data:
        raise KeyError("probeGuess missing from probe npz")
    return data["probeGuess"]


def scale_probe(probe: np.ndarray, target_N: int, smoothing_sigma: float) -> np.ndarray:
    """Resize probe to target_N and optionally smooth.

    Uses interpolate_array for cubic spline interpolation on real/imag parts,
    then smooth_complex_array for Gaussian smoothing on amp/phase.
    """
    if probe.shape[0] != probe.shape[1]:
        raise ValueError("probe must be square")
    if probe.shape[0] != target_N:
        zoom_factor = target_N / probe.shape[0]
        probe = interpolate_array(probe, zoom_factor)
    if smoothing_sigma and smoothing_sigma > 0:
        probe = smooth_complex_array(probe, smoothing_sigma)
    return probe.astype(np.complex64)


def run_grid_lines_workflow(cfg: GridLinesConfig) -> Dict[str, Any]:
    """Orchestrate probe prep → sim → train → infer → stitch → metrics."""
    raise NotImplementedError

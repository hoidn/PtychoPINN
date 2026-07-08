"""Weak-scattering phantom generators for the BRDT smoke dataset.

The phantom roster is intentionally small and non-CDI: overlapping
ellipses, soft cell-like blobs, and sparse fine inclusions. All
generators take a per-sample seed and return refractive-index maps in
the weak-scattering envelope; ``dataset_contract.refractive_index_to_q``
converts these to the physical scattering potential consumed by the
operator.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .dataset_contract import (
    DELTA_N_MAX,
    DELTA_N_MIN,
    LOCKED_GRID_SIZE,
    LOCKED_MEDIUM_RI,
)


def _support_margin(grid: int, margin: float = 0.12) -> Tuple[int, int]:
    lo = int(math.floor(grid * margin))
    hi = int(math.ceil(grid * (1.0 - margin)))
    return lo, hi


def _sample_delta_n(rng: np.random.Generator) -> float:
    return float(rng.uniform(DELTA_N_MIN, DELTA_N_MAX))


def _clip_to_weak_scattering(n_field: np.ndarray, n_m: float) -> np.ndarray:
    """Clip deviation from the medium index to the weak-scattering envelope.

    Phantom families accumulate per-object perturbations additively, which
    can drive ``|n - n_m|`` beyond ``DELTA_N_MAX``. The candidate-lane
    contract (see ``born_rytov_dt_candidate_lane_design.md``) requires the
    final field to stay within ``delta_n in [-DELTA_N_MAX, +DELTA_N_MAX]``
    so the Born approximation remains valid.
    """
    delta = n_field - float(n_m)
    np.clip(delta, -DELTA_N_MAX, DELTA_N_MAX, out=delta)
    return delta + float(n_m)


def overlapping_ellipses(
    seed: int,
    grid: int = LOCKED_GRID_SIZE,
    n_m: float = LOCKED_MEDIUM_RI,
    count_range: Tuple[int, int] = (3, 7),
) -> np.ndarray:
    """Random overlapping ellipses with weak refractive-index contrast."""
    rng = np.random.default_rng(int(seed))
    n_field = np.full((grid, grid), float(n_m), dtype=np.float64)
    j_grid, i_grid = np.meshgrid(np.arange(grid), np.arange(grid), indexing="xy")
    lo, hi = _support_margin(grid)
    n_ellipses = int(rng.integers(count_range[0], count_range[1] + 1))
    for _ in range(n_ellipses):
        cx = float(rng.uniform(lo, hi))
        cz = float(rng.uniform(lo, hi))
        a = float(rng.uniform(grid * 0.06, grid * 0.18))
        b = float(rng.uniform(grid * 0.06, grid * 0.18))
        theta = float(rng.uniform(0.0, math.pi))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = j_grid - cx
        dz = i_grid - cz
        xr = cos_t * dx + sin_t * dz
        zr = -sin_t * dx + cos_t * dz
        inside = (xr * xr) / (a * a) + (zr * zr) / (b * b) <= 1.0
        delta = _sample_delta_n(rng) * (1.0 if rng.uniform() > 0.3 else -1.0)
        n_field[inside] += delta
    return _clip_to_weak_scattering(n_field, n_m)


def soft_blobs(
    seed: int,
    grid: int = LOCKED_GRID_SIZE,
    n_m: float = LOCKED_MEDIUM_RI,
    count_range: Tuple[int, int] = (2, 5),
) -> np.ndarray:
    """Soft cell-like Gaussian blobs."""
    rng = np.random.default_rng(int(seed) + 1009)
    n_field = np.full((grid, grid), float(n_m), dtype=np.float64)
    j_grid, i_grid = np.meshgrid(np.arange(grid), np.arange(grid), indexing="xy")
    lo, hi = _support_margin(grid)
    n_blobs = int(rng.integers(count_range[0], count_range[1] + 1))
    for _ in range(n_blobs):
        cx = float(rng.uniform(lo, hi))
        cz = float(rng.uniform(lo, hi))
        sigma = float(rng.uniform(grid * 0.05, grid * 0.12))
        amp = _sample_delta_n(rng)
        dx = j_grid - cx
        dz = i_grid - cz
        n_field += amp * np.exp(-(dx * dx + dz * dz) / (2.0 * sigma * sigma))
    return _clip_to_weak_scattering(n_field, n_m)


def sparse_inclusions(
    seed: int,
    grid: int = LOCKED_GRID_SIZE,
    n_m: float = LOCKED_MEDIUM_RI,
    count_range: Tuple[int, int] = (4, 9),
) -> np.ndarray:
    """Sparse fine inclusions: small disks at random positions."""
    rng = np.random.default_rng(int(seed) + 2017)
    n_field = np.full((grid, grid), float(n_m), dtype=np.float64)
    j_grid, i_grid = np.meshgrid(np.arange(grid), np.arange(grid), indexing="xy")
    lo, hi = _support_margin(grid, margin=0.18)
    n_inc = int(rng.integers(count_range[0], count_range[1] + 1))
    for _ in range(n_inc):
        cx = float(rng.uniform(lo, hi))
        cz = float(rng.uniform(lo, hi))
        r = float(rng.uniform(grid * 0.015, grid * 0.04))
        amp = _sample_delta_n(rng) * (1.0 if rng.uniform() > 0.5 else -1.0)
        dx = j_grid - cx
        dz = i_grid - cz
        n_field[dx * dx + dz * dz <= r * r] += amp
    return _clip_to_weak_scattering(n_field, n_m)


def generate_refractive_index(
    family: str,
    seed: int,
    grid: int = LOCKED_GRID_SIZE,
    n_m: float = LOCKED_MEDIUM_RI,
) -> np.ndarray:
    """Dispatch by phantom-family name."""
    if family == "overlapping_ellipses":
        return overlapping_ellipses(seed=seed, grid=grid, n_m=n_m)
    if family == "soft_blobs":
        return soft_blobs(seed=seed, grid=grid, n_m=n_m)
    if family == "sparse_inclusions":
        return sparse_inclusions(seed=seed, grid=grid, n_m=n_m)
    raise ValueError(f"Unknown phantom family: {family!r}")

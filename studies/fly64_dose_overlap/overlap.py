"""
Phase D overlap metrics pipeline for fly64 dose/overlap study.

This module implements overlap-driven sampling and reporting per specs/overlap_metrics.md:
  1. Loading Phase C train/test NPZs for each dose
  2. Deterministic image subsampling via s_img
  3. Group formation (gs=1: single-image groups; gs=2: KNN groups with allowed duplication)
  4. Computing three disc-overlap metrics (group-based, image-based, group↔group COM)
  5. Writing output NPZs with measured overlap metrics and sampling parameters

References:
- specs/overlap_metrics.md (normative spec for Metric 1/2/3 and API)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md §Phase D
- docs/GRIDSIZE_N_GROUPS_GUIDE.md (unified n_groups semantics)
- specs/data_contracts.md:207 (DATA-001 NPZ contract)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

GEOMETRY_ACCEPTANCE_CAP = 0.10
GEOMETRY_ACCEPTANCE_EPS = 1e-6


@dataclass
class OverlapMetrics:
    """
    Overlap metrics for a dataset split per specs/overlap_metrics.md.

    Attributes:
        metrics_version: Semantic version string (e.g., "1.0")
        gridsize: 1 or 2
        s_img: Image subsampling fraction (0 < s_img <= 1]
        n_groups: Number of groups
        neighbor_count: K for neighbor-based averages (default 6)
        probe_diameter_px: Nominal probe diameter in pixels
        rng_seed_subsample: RNG seed for deterministic subsampling
        metric_1_group_based_avg: Metric 1 average (gs=2 only; None for gs=1)
        metric_2_image_based_avg: Metric 2 average (global image-based)
        metric_3_group_to_group_avg: Metric 3 average (group↔group COM)
        n_images_total: Total images before subsampling
        n_images_subsampled: Images after s_img subsampling
        n_unique_images: Unique images by exact (x,y) equality
        n_groups_actual: Actual number of groups produced
        geometry_acceptance_bound: Bounding-box-limited max acceptance (≤10%)
        effective_min_acceptance: Guarded acceptance floor (ε ≤ value ≤ 10%)
    """
    metrics_version: str
    gridsize: int
    s_img: float
    n_groups: int
    neighbor_count: int
    probe_diameter_px: float
    rng_seed_subsample: int
    metric_1_group_based_avg: Optional[float]
    metric_2_image_based_avg: float
    metric_3_group_to_group_avg: float
    n_images_total: int
    n_images_subsampled: int
    n_unique_images: int
    n_groups_actual: int
    geometry_acceptance_bound: float
    effective_min_acceptance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Omit None values for cleanliness
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class GeometryAcceptance:
    """Diagnostics for geometry-based acceptance bounds."""

    bounding_box_area: float
    disc_area: float
    theoretical_bound: float
    geometry_acceptance_bound: float
    effective_min_acceptance: float
    span_x: float
    span_y: float
    n_positions: int


def compute_geometry_acceptance(
    coords: np.ndarray,
    probe_diameter_px: float,
    *,
    acceptance_cap: float = GEOMETRY_ACCEPTANCE_CAP,
    epsilon: float = GEOMETRY_ACCEPTANCE_EPS,
) -> GeometryAcceptance:
    """
    Compute the theoretical geometry acceptance bound for a set of coordinates.

    The bound is defined as:
        bound = min((bounding_box_area / (n_positions * disc_area)), acceptance_cap)
    where disc_area is derived from the probe diameter. The effective minimum acceptance
    guards against zero by clamping the bound to >= epsilon.
    """
    n_positions = int(len(coords))
    if n_positions == 0:
        disc_area = np.pi * (max(probe_diameter_px, epsilon) / 2.0) ** 2
        return GeometryAcceptance(
            bounding_box_area=0.0,
            disc_area=disc_area,
            theoretical_bound=0.0,
            geometry_acceptance_bound=0.0,
            effective_min_acceptance=epsilon,
            span_x=0.0,
            span_y=0.0,
            n_positions=0,
        )

    span_x = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    span_y = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    bounding_box_area = max(span_x * span_y, epsilon)
    disc_area = np.pi * (max(probe_diameter_px, epsilon) / 2.0) ** 2

    theoretical_bound = bounding_box_area / max(n_positions * disc_area, epsilon)
    geometry_acceptance_bound = float(min(theoretical_bound, acceptance_cap))
    effective_min_acceptance = float(max(geometry_acceptance_bound, epsilon))

    return GeometryAcceptance(
        bounding_box_area=bounding_box_area,
        disc_area=disc_area,
        theoretical_bound=theoretical_bound,
        geometry_acceptance_bound=geometry_acceptance_bound,
        effective_min_acceptance=effective_min_acceptance,
        span_x=span_x,
        span_y=span_y,
        n_positions=n_positions,
    )


def disc_overlap_area(d: float, diameter: float) -> float:
    """
    Compute overlap area between two axis-aligned discs.

    For two discs with common diameter D and centers separated by distance d:
    - If d >= D, overlap area = 0
    - Else, overlap area = 2 R^2 arccos(d / (2R)) - (d/2) sqrt(4 R^2 - d^2)

    Args:
        d: Distance between disc centers (pixels)
        diameter: Disc diameter (pixels)

    Returns:
        Overlap area in square pixels

    References:
        - specs/overlap_metrics.md §2D Overlap Definition
    """
    if d >= diameter:
        return 0.0

    R = diameter / 2.0
    # Avoid numerical issues when d is very close to 0
    if d < 1e-10:
        # Full overlap
        return np.pi * R ** 2

    # Standard formula
    term1 = 2.0 * R ** 2 * np.arccos(d / (2.0 * R))
    term2 = (d / 2.0) * np.sqrt(4.0 * R ** 2 - d ** 2)
    return term1 - term2


def disc_overlap_fraction(d: float, diameter: float) -> float:
    """
    Compute normalized overlap fraction between two discs.

    Normalized by the area of a single disc: f_overlap = A_overlap / (π R^2)

    Args:
        d: Distance between disc centers (pixels)
        diameter: Disc diameter (pixels)

    Returns:
        Overlap fraction in [0, 1]

    References:
        - specs/overlap_metrics.md §2D Overlap Definition

    Examples:
        >>> # d=0 (perfect overlap) → 1.0
        >>> abs(disc_overlap_fraction(0.0, 10.0) - 1.0) < 1e-6
        True
        >>> # d=R (half-diameter) → ~0.391...
        >>> 0.39 < disc_overlap_fraction(5.0, 10.0) < 0.40
        True
        >>> # d>=D (no overlap) → 0.0
        >>> disc_overlap_fraction(10.0, 10.0)
        0.0
    """
    if d >= diameter:
        return 0.0

    R = diameter / 2.0
    area_disc = np.pi * R ** 2
    area_overlap = disc_overlap_area(d, diameter)
    return area_overlap / area_disc


def subsample_images(
    coords: np.ndarray,
    s_img: float,
    rng_seed: int,
) -> np.ndarray:
    """
    Deterministically subsample images by fraction s_img.

    Args:
        coords: Image coordinates, shape (N, 2)
        s_img: Subsampling fraction (0 < s_img <= 1]
        rng_seed: RNG seed for reproducibility

    Returns:
        Boolean mask, shape (N,), indicating retained images

    References:
        - specs/overlap_metrics.md §Parameters (s_img)
    """
    if not (0.0 < s_img <= 1.0):
        raise ValueError(f"s_img must be in (0, 1], got {s_img}")

    n_total = len(coords)
    n_keep = max(1, int(np.round(s_img * n_total)))

    rng = np.random.default_rng(rng_seed)
    indices = rng.choice(n_total, size=n_keep, replace=False)

    mask = np.zeros(n_total, dtype=bool)
    mask[indices] = True
    return mask


def filter_dataset_by_mask(
    data: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Filter dataset arrays by boolean mask along first axis.

    Handles scalar and 0-D array metadata by broadcasting to mask length.

    Args:
        data: Dictionary of arrays (from np.load)
        mask: Boolean mask, shape (N,)

    Returns:
        Filtered dictionary with same keys, reduced first dimension for arrays
        matching mask length. Scalar/0-D arrays are broadcast to mask length.

    Raises:
        ValueError: If mask length doesn't match first dimension for non-scalar arrays

    Example:
        >>> data = {'diffraction': np.zeros((3, 64, 64)), 'xcoords': np.array([0, 1, 2])}
        >>> mask = np.array([True, False, True])
        >>> filtered = filter_dataset_by_mask(data, mask)
        >>> filtered['diffraction'].shape
        (2, 64, 64)
        >>> filtered['xcoords'].tolist()
        [0.0, 2.0]

        >>> # Scalar metadata is broadcast
        >>> data_with_scalar = {'coords': np.array([0, 1, 2]), 'dose': 1000.0}
        >>> filtered = filter_dataset_by_mask(data_with_scalar, mask)
        >>> filtered['dose']  # Broadcast to mask length
        array([1000., 1000.])
    """
    filtered = {}
    n_expected = len(mask)
    n_keep = int(mask.sum())

    for key, arr in data.items():
        # Skip non-array metadata
        if not isinstance(arr, np.ndarray):
            filtered[key] = arr
            continue

        # Handle scalar (0-D) arrays by broadcasting
        if arr.ndim == 0:
            filtered[key] = np.full(n_keep, arr.item())
            continue

        # Filter along first axis if it matches mask length
        if len(arr) == n_expected:
            filtered[key] = arr[mask]
        else:
            # Preserve arrays with different first dimension (e.g., probeGuess, objectGuess)
            filtered[key] = arr

    return filtered


def form_groups_gs1(coords: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Form groups for gridsize=1: one image per group.

    Args:
        coords: Subsampled coordinates, shape (N, 2)
        n_groups: Number of groups (should equal N for gs=1)

    Returns:
        Group assignments, shape (N,), values in [0, n_groups)

    Notes:
        For gs=1, n_groups typically equals the number of images.
        Each group contains exactly one image (no KNN grouping).

    References:
        - specs/overlap_metrics.md §Parameters (n_groups for gs=1)
        - docs/GRIDSIZE_N_GROUPS_GUIDE.md (unified n_groups semantics)
    """
    n = len(coords)
    if n_groups != n:
        # Allow flexibility but warn
        pass
    # Simply assign sequential group IDs
    return np.arange(n)


def form_groups_gs2(coords: np.ndarray, n_groups: int, gridsize: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Form groups for gridsize=2 via KNN sampling with allowed duplication.

    Each group contains gridsize^2 images. Groups are formed by:
    1. Randomly sampling n_groups seed positions
    2. For each seed, finding its (gridsize^2 - 1) nearest neighbors
    3. Duplication across groups is allowed per existing behavior

    Args:
        coords: Subsampled coordinates, shape (N, 2)
        n_groups: Number of groups to produce
        gridsize: Grid size (default 2)

    Returns:
        Tuple of (coords_expanded, group_assignments):
            coords_expanded: Coordinates with duplicates, shape (n_groups * gridsize^2, 2)
            group_assignments: Group IDs, shape (n_groups * gridsize^2, ), values in [0, n_groups)

    References:
        - specs/overlap_metrics.md §Metrics (grouping for gs=2)
        - ptycho/raw_data.py (KNN grouping implementation)
    """
    n = len(coords)
    c = gridsize ** 2

    if n < c:
        raise ValueError(f"Need at least {c} images for gridsize={gridsize}, got {n}")

    # Sample n_groups seeds (with replacement if n_groups > n)
    rng = np.random.default_rng(42)  # Fixed seed for grouping reproducibility
    seed_indices = rng.choice(n, size=n_groups, replace=(n_groups > n))

    # Build KNN index
    nbrs = NearestNeighbors(n_neighbors=c, algorithm='auto').fit(coords)

    # For each seed, get nearest neighbors and build expanded coords + groups
    coords_expanded = []
    group_assignments = []

    for group_id, seed_idx in enumerate(seed_indices):
        seed_coord = coords[seed_idx:seed_idx+1]
        distances, indices = nbrs.kneighbors(seed_coord)
        # indices shape: (1, c)
        member_indices = indices[0]

        # Add coordinates for this group's members
        coords_expanded.append(coords[member_indices])
        # Add group IDs for this group
        group_assignments.extend([group_id] * len(member_indices))

    coords_expanded = np.vstack(coords_expanded)
    group_assignments = np.array(group_assignments)

    return coords_expanded, group_assignments


def compute_metric_1_group_based(
    coords: np.ndarray,
    group_assignments: np.ndarray,
    neighbor_count: int,
    probe_diameter_px: float,
) -> float:
    """
    Compute Metric 1: Group-based overlap (gs=2 only).

    For each sample within a group, compute mean overlap to its neighbor_count
    group neighbors (seed-to-neighbors only, not all pairwise).
    Average across all samples/groups.

    Args:
        coords: All coordinates (including duplicates if gs=2), shape (M, 2)
        group_assignments: Group ID for each coordinate, shape (M,)
        neighbor_count: K for neighbor-based averages
        probe_diameter_px: Probe diameter in pixels

    Returns:
        Global average of per-sample mean overlap fractions

    References:
        - specs/overlap_metrics.md §Metrics (Metric 1)
    """
    unique_groups = np.unique(group_assignments)
    sample_means = []

    for group_id in unique_groups:
        group_mask = (group_assignments == group_id)
        group_coords = coords[group_mask]

        if len(group_coords) <= 1:
            # Single-member group: no neighbors
            continue

        # For each sample (seed) in the group
        for i, seed_coord in enumerate(group_coords):
            # Compute distances to other group members
            other_coords = np.delete(group_coords, i, axis=0)
            if len(other_coords) == 0:
                continue

            distances = np.linalg.norm(other_coords - seed_coord, axis=1)

            # Take up to neighbor_count nearest neighbors
            k = min(neighbor_count, len(distances))
            nearest_dists = np.partition(distances, k-1)[:k]

            # Compute mean overlap to these neighbors
            overlaps = [disc_overlap_fraction(d, probe_diameter_px) for d in nearest_dists]
            sample_means.append(np.mean(overlaps))

    return float(np.mean(sample_means)) if sample_means else 0.0


def compute_metric_2_image_based(
    coords: np.ndarray,
    neighbor_count: int,
    probe_diameter_px: float,
) -> float:
    """
    Compute Metric 2: Image-based global overlap with deduplication.

    Deduplicate images by exact (x, y) equality, then for each unique image,
    compute mean overlap to its neighbor_count nearest neighbors in the global set.
    Average across all unique images.

    Args:
        coords: All coordinates (may contain duplicates), shape (M, 2)
        neighbor_count: K for neighbor-based averages
        probe_diameter_px: Probe diameter in pixels

    Returns:
        Global average of per-image mean overlap fractions

    References:
        - specs/overlap_metrics.md §Metrics (Metric 2)
    """
    # Deduplicate by exact (x, y) equality
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)

    if len(unique_coords) <= 1:
        return 0.0

    image_means = []
    for i, image_coord in enumerate(unique_coords):
        # Compute distances to all other unique images
        other_coords = np.delete(unique_coords, i, axis=0)
        distances = np.linalg.norm(other_coords - image_coord, axis=1)

        # Take up to neighbor_count nearest neighbors
        k = min(neighbor_count, len(distances))
        nearest_dists = np.partition(distances, k-1)[:k]

        # Compute mean overlap
        overlaps = [disc_overlap_fraction(d, probe_diameter_px) for d in nearest_dists]
        image_means.append(np.mean(overlaps))

    return float(np.mean(image_means))


def compute_metric_3_group_to_group(
    coords: np.ndarray,
    group_assignments: np.ndarray,
    probe_diameter_px: float,
) -> float:
    """
    Compute Metric 3: Group↔Group COM-based overlap.

    Compute center-of-mass (COM) for each group, then for each group find
    all other groups whose COM distance d < probe_diameter_px.
    Compute mean overlap to this neighbor set, average across groups.

    Args:
        coords: All coordinates (including duplicates if gs=2), shape (M, 2)
        group_assignments: Group ID for each coordinate, shape (M,)
        probe_diameter_px: Probe diameter in pixels

    Returns:
        Global average of per-group mean overlap fractions

    References:
        - specs/overlap_metrics.md §Metrics (Metric 3)
    """
    unique_groups = np.unique(group_assignments)

    if len(unique_groups) <= 1:
        return 0.0

    # Compute COM for each group
    group_coms = []
    for group_id in unique_groups:
        group_mask = (group_assignments == group_id)
        group_coords = coords[group_mask]
        com = group_coords.mean(axis=0)
        group_coms.append(com)

    group_coms = np.array(group_coms)

    # For each group, find overlapping neighbors
    group_means = []
    for i, com in enumerate(group_coms):
        # Compute distances to all other group COMs
        other_coms = np.delete(group_coms, i, axis=0)
        distances = np.linalg.norm(other_coms - com, axis=1)

        # Find neighbors with d < probe_diameter_px
        overlapping_mask = (distances < probe_diameter_px)
        overlapping_dists = distances[overlapping_mask]

        if len(overlapping_dists) == 0:
            # No overlapping neighbors → contribution is 0
            group_means.append(0.0)
        else:
            # Compute mean overlap to overlapping neighbors
            overlaps = [disc_overlap_fraction(d, probe_diameter_px) for d in overlapping_dists]
            group_means.append(np.mean(overlaps))

    return float(np.mean(group_means))


def compute_overlap_metrics(
    coords: np.ndarray,
    gridsize: int,
    s_img: float,
    n_groups: int,
    neighbor_count: int = 6,
    probe_diameter_px: Optional[float] = None,
    rng_seed_subsample: Optional[int] = None,
    *,
    coords_are_subsampled: bool = False,
) -> OverlapMetrics:
    """
    Compute overlap metrics per specs/overlap_metrics.md.

    This is the primary Python API for overlap metrics computation.

    Args:
        coords: All image coordinates (before subsampling), shape (N, 2)
        gridsize: 1 or 2
        s_img: Image subsampling fraction (0 < s_img <= 1]
        n_groups: Number of groups to produce
        neighbor_count: K for neighbor-based averages (default 6)
        probe_diameter_px: Probe diameter in pixels (default: 0.6 * N if N available)
        rng_seed_subsample: RNG seed for subsampling (default: 42)

    Returns:
        OverlapMetrics instance with all computed metrics

    Raises:
        ValueError: If parameters are invalid or degenerate

    References:
        - specs/overlap_metrics.md §API and CLI (Python API)

    Examples:
        >>> coords = np.random.rand(100, 2) * 100  # 100 images in 100x100 px region
        >>> metrics = compute_overlap_metrics(
        ...     coords, gridsize=2, s_img=0.8, n_groups=50,
        ...     neighbor_count=6, probe_diameter_px=38.4, rng_seed_subsample=123
        ... )
        >>> assert metrics.metric_1_group_based_avg is not None  # gs=2
        >>> assert 0.0 <= metrics.metric_2_image_based_avg <= 1.0
    """
    if gridsize not in [1, 2]:
        raise ValueError(f"gridsize must be 1 or 2, got {gridsize}")

    if not (0.0 < s_img <= 1.0):
        raise ValueError(f"s_img must be in (0, 1], got {s_img}")

    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")

    # Default probe_diameter_px if not provided
    if probe_diameter_px is None:
        # Fallback: 0.6 * N (assuming N=64 for this study)
        probe_diameter_px = 0.6 * 64

    # Default RNG seed
    if rng_seed_subsample is None:
        rng_seed_subsample = 42

    n_images_total = len(coords)

    # Subsample images unless coords have already been filtered upstream
    if coords_are_subsampled:
        coords_sub = coords
        n_images_subsampled = len(coords_sub)
    else:
        subsample_mask = subsample_images(coords, s_img, rng_seed_subsample)
        coords_sub = coords[subsample_mask]
        n_images_subsampled = len(coords_sub)

    if n_images_subsampled == 0:
        raise ValueError("Subsampling resulted in zero images; increase s_img")

    # Form groups
    if gridsize == 1:
        group_assignments = form_groups_gs1(coords_sub, n_groups)
        coords_with_groups = coords_sub  # No duplication for gs=1
    else:  # gridsize == 2
        coords_with_groups, group_assignments = form_groups_gs2(coords_sub, n_groups, gridsize=2)
        # coords_with_groups now has duplicates to match group_assignments length

    # Deduplicate for counting
    unique_coords = np.unique(coords_sub, axis=0)
    n_unique_images = len(unique_coords)
    n_groups_actual = len(np.unique(group_assignments))

    # Compute Metric 1 (gs=2 only)
    if gridsize == 2:
        metric_1_group_based_avg = compute_metric_1_group_based(
            coords_with_groups, group_assignments, neighbor_count, probe_diameter_px
        )
    else:
        metric_1_group_based_avg = None  # Not applicable for gs=1

    # Compute Metric 2 (global image-based)
    metric_2_image_based_avg = compute_metric_2_image_based(
        coords_sub, neighbor_count, probe_diameter_px
    )

    # Compute Metric 3 (group↔group COM)
    metric_3_group_to_group_avg = compute_metric_3_group_to_group(
        coords_with_groups, group_assignments, probe_diameter_px
    )

    geometry_stats = compute_geometry_acceptance(
        coords_sub,
        probe_diameter_px,
        acceptance_cap=GEOMETRY_ACCEPTANCE_CAP,
        epsilon=GEOMETRY_ACCEPTANCE_EPS,
    )

    return OverlapMetrics(
        metrics_version="1.0",
        gridsize=gridsize,
        s_img=s_img,
        n_groups=n_groups,
        neighbor_count=neighbor_count,
        probe_diameter_px=probe_diameter_px,
        rng_seed_subsample=rng_seed_subsample,
        metric_1_group_based_avg=metric_1_group_based_avg,
        metric_2_image_based_avg=metric_2_image_based_avg,
        metric_3_group_to_group_avg=metric_3_group_to_group_avg,
        n_images_total=n_images_total,
        n_images_subsampled=n_images_subsampled,
        n_unique_images=n_unique_images,
        n_groups_actual=n_groups_actual,
        geometry_acceptance_bound=geometry_stats.geometry_acceptance_bound,
        effective_min_acceptance=geometry_stats.effective_min_acceptance,
    )


def generate_overlap_views(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    gridsize: int,
    s_img: float,
    n_groups: int,
    neighbor_count: int = 6,
    probe_diameter_px: Optional[float] = None,
    rng_seed_subsample: int = 42,
) -> Dict[str, Any]:
    """
    Generate overlap views from Phase C train/test NPZs per specs/overlap_metrics.md.

    This function:
    1. Loads train and test NPZs
    2. Computes overlap metrics for both splits
    3. Writes filtered NPZs with metadata
    4. Returns metrics bundle

    Args:
        train_path: Path to Phase C train NPZ
        test_path: Path to Phase C test NPZ
        output_dir: Directory for output NPZs and metrics
        gridsize: 1 or 2
        s_img: Image subsampling fraction
        n_groups: Number of groups
        neighbor_count: K for neighbor-based averages (default 6)
        probe_diameter_px: Probe diameter in pixels (default: 0.6 * 64)
        rng_seed_subsample: RNG seed for subsampling (default 42)

    Returns:
        Dictionary with keys:
            'train_metrics': OverlapMetrics for train split
            'test_metrics': OverlapMetrics for test split
            'train_output': Path to train NPZ
            'test_output': Path to test NPZ
            'metrics_bundle_path': Path to metrics_bundle.json

    Raises:
        ValueError: If validation fails or parameters invalid
        FileNotFoundError: If input NPZs don't exist

    References:
        - specs/overlap_metrics.md §Outputs
    """
    if probe_diameter_px is None:
        probe_diameter_px = 0.6 * 64  # Default for this study

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating overlap views (gridsize={gridsize})")
    print(f"s_img={s_img}, n_groups={n_groups}, K={neighbor_count}, D={probe_diameter_px:.1f}px")
    print(f"{'='*60}\n")

    results = {}

    for split_name, split_path in [('train', train_path), ('test', test_path)]:
        print(f"[{split_name.upper()}] Processing {split_path.name}...")

        # Load dataset
        with np.load(split_path, allow_pickle=True) as data:
            data_dict = {k: data[k] for k in data.keys()}

        # Extract coordinates and apply subsampling mask for output NPZs
        coords = np.stack([data_dict['xcoords'], data_dict['ycoords']], axis=1)

        # Compute subsample mask once so NPZs and metrics are aligned
        subsample_mask = subsample_images(coords, s_img, rng_seed_subsample)

        # Filter the dataset by mask along the first axis where applicable
        filtered_dict = filter_dataset_by_mask(data_dict, subsample_mask)
        coords_sub = np.stack([filtered_dict['xcoords'], filtered_dict['ycoords']], axis=1)

        # Compute metrics
        print(f"  Computing overlap metrics...")
        metrics = compute_overlap_metrics(
            coords=coords_sub,
            gridsize=gridsize,
            s_img=s_img,
            n_groups=n_groups,
            neighbor_count=neighbor_count,
            probe_diameter_px=probe_diameter_px,
            rng_seed_subsample=rng_seed_subsample,
            coords_are_subsampled=True,
        )

        print(f"  Metrics:")
        if metrics.metric_1_group_based_avg is not None:
            print(f"    Metric 1 (group-based): {metrics.metric_1_group_based_avg:.4f}")
        else:
            print(f"    Metric 1 (group-based): N/A (gs={gridsize})")
        print(f"    Metric 2 (image-based): {metrics.metric_2_image_based_avg:.4f}")
        print(f"    Metric 3 (group↔group): {metrics.metric_3_group_to_group_avg:.4f}")
        print(f"    Images: {metrics.n_images_subsampled}/{metrics.n_images_total} (unique: {metrics.n_unique_images})")
        print(f"    Groups: {metrics.n_groups_actual}")
        print(
            "    Geometry acceptance: "
            f"bound={metrics.geometry_acceptance_bound:.4f}, "
            f"effective_floor={metrics.effective_min_acceptance:.4f}"
        )

        # Write the filtered dataset NPZ for this split
        output_path = output_dir / f"{split_name}.npz"
        print(f"  Writing output NPZ: {output_path}")

        # Add metadata to filtered payload
        filtered_dict['_metadata'] = json.dumps({
            'gridsize': gridsize,
            's_img': s_img,
            'n_groups': n_groups,
            'neighbor_count': neighbor_count,
            'probe_diameter_px': probe_diameter_px,
            'rng_seed_subsample': rng_seed_subsample,
            'metrics_version': metrics.metrics_version,
            'geometry_acceptance_bound': metrics.geometry_acceptance_bound,
            'effective_min_acceptance': metrics.effective_min_acceptance,
        })

        np.savez_compressed(output_path, **filtered_dict)

        # Write per-split metrics JSON
        metrics_json_path = output_dir / f"{split_name}_metrics.json"
        print(f"  Writing metrics JSON: {metrics_json_path}")
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        results[f'{split_name}_metrics'] = metrics
        results[f'{split_name}_metrics_path'] = metrics_json_path
        results[f'{split_name}_output'] = output_path
        print(f"  ✓ {split_name.upper()} complete\n")

    # Create metrics bundle (train + test aggregated)
    metrics_bundle = {
        'train': results['train_metrics'].to_dict(),
        'test': results['test_metrics'].to_dict(),
    }
    metrics_bundle_path = output_dir / 'metrics_bundle.json'
    print(f"Writing metrics bundle: {metrics_bundle_path}")
    with open(metrics_bundle_path, 'w') as f:
        json.dump(metrics_bundle, f, indent=2)

    results['metrics_bundle_path'] = metrics_bundle_path

    print(f"{'='*60}")
    print(f"Overlap view generation complete")
    print(f"{'='*60}\n")

    return results


def main():
    """
    CLI entry point for generating overlap views per specs/overlap_metrics.md.

    Usage:
        python -m studies.fly64_dose_overlap.overlap \\
            --phase-c-root data/phase_c/dose_1000 \\
            --output-root tmp/phase_d_overlap \\
            --artifact-root plans/active/.../reports/<timestamp>/phase_d_overlap_metrics \\
            --gridsize 2 \\
            --s-img 0.8 \\
            --n-groups 512 \\
            --neighbor-count 6 \\
            --probe-diameter-px 38.4 \\
            --rng-seed-subsample 456
    """
    parser = argparse.ArgumentParser(
        description="Generate overlap views from Phase C datasets (Phase D) per specs/overlap_metrics.md"
    )
    parser.add_argument(
        '--phase-c-root',
        type=Path,
        required=True,
        help='Directory containing Phase C train/test NPZs (e.g., data/phase_c/dose_1000)',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Output directory for filtered NPZs and metrics',
    )
    parser.add_argument(
        '--artifact-root',
        type=Path,
        help='Optional reports hub directory for copying metrics (e.g., plans/active/.../reports/<timestamp>)',
    )
    parser.add_argument(
        '--gridsize',
        type=int,
        required=True,
        choices=[1, 2],
        help='Gridsize (1 or 2)',
    )
    parser.add_argument(
        '--s-img',
        type=float,
        required=True,
        help='Image subsampling fraction (0 < s_img <= 1]',
    )
    parser.add_argument(
        '--n-groups',
        type=int,
        required=True,
        help='Number of groups to produce',
    )
    parser.add_argument(
        '--neighbor-count',
        type=int,
        default=6,
        help='K for neighbor-based averages (default: 6)',
    )
    parser.add_argument(
        '--probe-diameter-px',
        type=float,
        default=None,
        help='Probe diameter in pixels (default: 0.6 * 64 = 38.4)',
    )
    parser.add_argument(
        '--rng-seed-subsample',
        type=int,
        default=42,
        help='RNG seed for deterministic subsampling (default: 42)',
    )
    args = parser.parse_args()

    # Find Phase C train/test NPZs
    train_candidates = list(args.phase_c_root.glob("*train.npz"))
    test_candidates = list(args.phase_c_root.glob("*test.npz"))

    if not train_candidates or not test_candidates:
        print(f"ERROR: Missing train/test NPZs in {args.phase_c_root}", file=sys.stderr)
        sys.exit(1)

    train_path = train_candidates[0]
    test_path = test_candidates[0]

    print("=" * 80)
    print("Phase D: Overlap Metrics Generation per specs/overlap_metrics.md")
    print("=" * 80)
    print(f"Phase C root:        {args.phase_c_root}")
    print(f"Output root:         {args.output_root}")
    print(f"Train NPZ:           {train_path.name}")
    print(f"Test NPZ:            {test_path.name}")
    print(f"Gridsize:            {args.gridsize}")
    print(f"s_img:               {args.s_img}")
    print(f"n_groups:            {args.n_groups}")
    print(f"neighbor_count:      {args.neighbor_count}")
    print(f"probe_diameter_px:   {args.probe_diameter_px or '0.6 * 64 (default)'}")
    print(f"rng_seed_subsample:  {args.rng_seed_subsample}")
    print("=" * 80)
    print()

    try:
        results = generate_overlap_views(
            train_path=train_path,
            test_path=test_path,
            output_dir=args.output_root,
            gridsize=args.gridsize,
            s_img=args.s_img,
            n_groups=args.n_groups,
            neighbor_count=args.neighbor_count,
            probe_diameter_px=args.probe_diameter_px,
            rng_seed_subsample=args.rng_seed_subsample,
        )

        # Copy metrics to artifact root if specified
        if args.artifact_root:
            artifact_dir = args.artifact_root
            artifact_dir.mkdir(parents=True, exist_ok=True)

            import shutil
            for key in ['train_metrics_path', 'test_metrics_path', 'metrics_bundle_path']:
                src = results[key]
                dst = artifact_dir / src.name
                shutil.copy2(src, dst)
                print(f"Copied {src.name} to {dst}")

        print("\n" + "=" * 80)
        print("Overlap view generation completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

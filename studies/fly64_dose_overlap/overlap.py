"""
Phase D overlap filtering pipeline for fly64 dose/overlap study.

This module implements the dense/sparse overlap view generation via:
  1. Loading Phase C train/test NPZs for each dose
  2. Computing pairwise spacing metrics from scan coordinates
  3. Filtering to dense/sparse selections based on StudyDesign thresholds
  4. Validating filtered views against DATA-001 + spacing constraints
  5. Writing output NPZs with metadata and spacing metrics

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md §Phase D
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md
- docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151 (spacing formula S ≈ (1 - f_overlap) × N)
- docs/SAMPLING_USER_GUIDE.md:112-140 (K-choose-C oversampling)
- specs/data_contracts.md:207 (DATA-001 NPZ contract)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from studies.fly64_dose_overlap.design import get_study_design, StudyDesign
from studies.fly64_dose_overlap.validation import validate_dataset_contract


@dataclass
class SpacingMetrics:
    """
    Statistics for inter-position spacing in a dataset.

    Attributes:
        min_spacing: Minimum pairwise distance (pixels)
        max_spacing: Maximum pairwise distance (pixels)
        mean_spacing: Mean pairwise distance (pixels)
        median_spacing: Median pairwise distance (pixels)
        threshold: Required minimum spacing for this view (pixels)
        n_positions: Number of scan positions
        n_accepted: Number of positions meeting threshold
        n_rejected: Number of positions below threshold
        acceptance_rate: Fraction of positions accepted
        geometry_acceptance_bound: Theoretical max acceptance based on bounding box (optional)
        effective_min_acceptance: Effective minimum acceptance rate used (optional)
    """
    min_spacing: float
    max_spacing: float
    mean_spacing: float
    median_spacing: float
    threshold: float
    n_positions: int
    n_accepted: int
    n_rejected: int
    acceptance_rate: float
    geometry_acceptance_bound: float | None = None
    effective_min_acceptance: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            'min_spacing': float(self.min_spacing),
            'max_spacing': float(self.max_spacing),
            'mean_spacing': float(self.mean_spacing),
            'median_spacing': float(self.median_spacing),
            'threshold': float(self.threshold),
            'n_positions': int(self.n_positions),
            'n_accepted': int(self.n_accepted),
            'n_rejected': int(self.n_rejected),
            'acceptance_rate': float(self.acceptance_rate),
        }
        if self.geometry_acceptance_bound is not None:
            result['geometry_acceptance_bound'] = float(self.geometry_acceptance_bound)
        if self.effective_min_acceptance is not None:
            result['effective_min_acceptance'] = float(self.effective_min_acceptance)
        return result


def compute_spacing_matrix(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise spacing matrix and per-position minimum spacing.

    Args:
        coords: Scan coordinates, shape (N, 2) for N positions (x, y)

    Returns:
        distances: Full pairwise distance matrix, shape (N, N)
        min_spacing_by_point: Minimum spacing to any other point, shape (N,)

    References:
        - docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-147 (spacing definition)
        - scipy.spatial.distance.pdist (condensed distance computation)

    Example:
        >>> coords = np.array([[0, 0], [10, 0], [0, 10]])
        >>> distances, min_spacings = compute_spacing_matrix(coords)
        >>> distances.shape
        (3, 3)
        >>> np.allclose(min_spacings, [10, 10, 10])
        True
    """
    if len(coords) == 0:
        return np.array([]), np.array([])

    if len(coords) == 1:
        # Single point: no spacing to compute
        return np.zeros((1, 1)), np.array([np.inf])

    # Compute condensed distance matrix (upper triangle only)
    condensed_distances = pdist(coords)

    # Convert to full square matrix
    distances = squareform(condensed_distances)

    # For each point, find minimum distance to any other point
    # Set diagonal to inf to exclude self-distances
    distances_masked = distances.copy()
    np.fill_diagonal(distances_masked, np.inf)
    min_spacing_by_point = distances_masked.min(axis=1)

    return distances, min_spacing_by_point


def build_acceptance_mask(
    min_spacing_by_point: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Build boolean mask accepting positions with spacing ≥ threshold.

    Args:
        min_spacing_by_point: Per-position minimum spacing, shape (N,)
        threshold: Minimum required spacing (pixels)

    Returns:
        mask: Boolean array, shape (N,), True for accepted positions

    References:
        - docs/GRIDSIZE_N_GROUPS_GUIDE.md:146-151 (threshold enforcement)

    Example:
        >>> min_spacings = np.array([50, 30, 100])
        >>> mask = build_acceptance_mask(min_spacings, threshold=40.0)
        >>> mask.tolist()
        [True, False, True]
    """
    return min_spacing_by_point >= threshold


def filter_dataset_by_mask(
    data: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Filter dataset arrays by boolean mask along first axis.

    Args:
        data: Dictionary of arrays (from np.load)
        mask: Boolean mask, shape (N,)

    Returns:
        Filtered dictionary with same keys, reduced first dimension

    Raises:
        ValueError: If mask length doesn't match first dimension

    Example:
        >>> data = {'diffraction': np.zeros((3, 64, 64)), 'xcoords': np.array([0, 1, 2])}
        >>> mask = np.array([True, False, True])
        >>> filtered = filter_dataset_by_mask(data, mask)
        >>> filtered['diffraction'].shape
        (2, 64, 64)
        >>> filtered['xcoords'].tolist()
        [0.0, 2.0]
    """
    filtered = {}
    n_expected = len(mask)

    for key, arr in data.items():
        # Skip non-array metadata
        if not isinstance(arr, np.ndarray):
            filtered[key] = arr
            continue

        # Filter along first axis if it matches mask length
        if len(arr) == n_expected:
            filtered[key] = arr[mask]
        else:
            # Preserve arrays with different first dimension (e.g., probeGuess, objectGuess)
            filtered[key] = arr

    return filtered


def greedy_min_spacing_selection(
    coords: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Greedily select positions satisfying minimum spacing constraint.

    Algorithm:
    1. Sort positions by (y, x) to ensure deterministic ordering
    2. Start with empty selection
    3. For each candidate position:
       - If selection is empty, accept it
       - Else, compute distances to all already-selected positions
       - If min distance >= threshold, accept it
    4. Return boolean mask of accepted positions

    This produces a deterministic subset that maximizes coverage while
    respecting spacing constraints. Not globally optimal but efficient
    and stable.

    Args:
        coords: Scan coordinates, shape (N, 2) for N positions (x, y)
        threshold: Minimum required spacing (pixels)

    Returns:
        mask: Boolean array, shape (N,), True for accepted positions

    References:
        - docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151 (spacing formula)
        - input.md:8-9 (Phase D sparse downsampling rescue)
        - plans/.../phase_d_sparse_downsampling_fix/plan.md:D7.2

    Example:
        >>> coords = np.array([[0, 0], [50, 0], [25, 0], [100, 0]])
        >>> mask = greedy_min_spacing_selection(coords, threshold=60.0)
        >>> # Should select positions with spacing ≥60
        >>> accepted_coords = coords[mask]
        >>> # Deterministic: sorted by (y, x) before greedy pass
    """
    if len(coords) == 0:
        return np.array([], dtype=bool)

    if len(coords) == 1:
        # Single position: always accept (no spacing constraint)
        return np.array([True])

    # Sort positions by (y, x) for deterministic ordering
    n = len(coords)
    sort_order = np.lexsort((coords[:, 0], coords[:, 1]))
    sorted_coords = coords[sort_order]

    # Greedy selection
    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []

    for i in range(n):
        if len(selected_indices) == 0:
            # Accept first position
            selected_mask[i] = True
            selected_indices.append(i)
        else:
            # Compute distances to already-selected positions
            candidate = sorted_coords[i]
            selected_coords = sorted_coords[selected_indices]
            distances = np.linalg.norm(selected_coords - candidate, axis=1)
            min_dist = distances.min()

            if min_dist >= threshold:
                selected_mask[i] = True
                selected_indices.append(i)

    # Un-sort mask to match original coordinate order
    unsorted_mask = np.zeros(n, dtype=bool)
    unsorted_mask[sort_order] = selected_mask

    return unsorted_mask


def compute_spacing_metrics(
    coords: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> SpacingMetrics:
    """
    Compute spacing statistics for a set of coordinates.

    Args:
        coords: Scan coordinates, shape (N, 2)
        threshold: Required minimum spacing (pixels)
        mask: Optional boolean mask to identify accepted positions

    Returns:
        SpacingMetrics instance with statistics

    References:
        - docs/GRIDSIZE_N_GROUPS_GUIDE.md:151 (acceptance rate logging)
    """
    distances, min_spacing_by_point = compute_spacing_matrix(coords)

    if mask is None:
        mask = build_acceptance_mask(min_spacing_by_point, threshold)

    # Compute statistics
    if len(distances) > 0:
        # Use condensed form for aggregate stats (avoid double-counting)
        condensed = pdist(coords)
        min_spacing = float(condensed.min()) if len(condensed) > 0 else 0.0
        max_spacing = float(condensed.max()) if len(condensed) > 0 else 0.0
        mean_spacing = float(condensed.mean()) if len(condensed) > 0 else 0.0
        median_spacing = float(np.median(condensed)) if len(condensed) > 0 else 0.0
    else:
        min_spacing = max_spacing = mean_spacing = median_spacing = 0.0

    n_positions = len(coords)
    n_accepted = int(mask.sum())
    n_rejected = n_positions - n_accepted
    acceptance_rate = n_accepted / n_positions if n_positions > 0 else 0.0

    return SpacingMetrics(
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        mean_spacing=mean_spacing,
        median_spacing=median_spacing,
        threshold=threshold,
        n_positions=n_positions,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        acceptance_rate=acceptance_rate,
    )


def compute_geometry_aware_acceptance_floor(
    coords: np.ndarray,
    threshold: float,
    conservative_factor: float = 0.5,
) -> float:
    """
    Compute geometry-aware acceptance bound based on bounding box area and circle packing.

    For dense views where the spacing threshold is large relative to the scan
    region, the hard-coded 10% MIN_ACCEPTANCE_RATE may be geometrically impossible.
    This function computes the theoretical maximum acceptance rate based on the
    split's bounding box area using circle packing constraints, then clamps to ≤0.10.

    Algorithm:
    1. Compute bounding box (xmin, xmax, ymin, ymax) from coordinates
    2. Calculate split area = (xmax - xmin) * (ymax - ymin)
    3. Each position requires approximately π × (threshold/2)² area (circle packing)
    4. Theoretical max slots = split_area / (π × (threshold/2)²)
    5. Theoretical max acceptance = max_slots / n_positions
    6. Clamp result to ≤0.10 to enforce a reasonable upper bound

    Args:
        coords: Scan coordinates, shape (N, 2) for N positions (x, y)
        threshold: Minimum required spacing (pixels)
        conservative_factor: Unused (kept for API compatibility)

    Returns:
        Geometry-aware acceptance bound (fraction, not percentage), clamped to [0, 0.10]

    References:
        - input.md:2 — geometry-aware acceptance bound requirement with circle packing
        - fix_plan.md:43 — Phase G dense blocker (ACCEPTANCE-001)
        - docs/findings.md:17 (ACCEPTANCE-001) — bounding-box acceptance bound formula

    Example:
        >>> coords = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
        >>> bound = compute_geometry_aware_acceptance_floor(coords, threshold=50.0)
        >>> # Bounding box: 100x100 = 10,000 px²
        >>> # Circle area per position: π × (50/2)² ≈ 1963.5 px²
        >>> # Theoretical max slots: 10,000 / 1963.5 ≈ 5.09
        >>> # Theoretical acceptance: 5.09 / 4 ≈ 1.27 → clamped to 0.10
        >>> assert bound == 0.10
    """
    if len(coords) == 0:
        return 0.0

    if len(coords) == 1:
        # Single position: clamp to 0.10
        return 0.10

    # Compute bounding box
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Compute split area
    split_area = (x_max - x_min) * (y_max - y_min)

    # Each position requires approximately π × (threshold/2)² area (circle packing constraint)
    radius = threshold / 2.0
    area_per_position = np.pi * (radius ** 2)

    # Theoretical maximum number of positions that can fit
    theoretical_max_slots = split_area / area_per_position if area_per_position > 0 else 0.0

    # Theoretical maximum acceptance rate
    n_positions = len(coords)
    theoretical_max_acceptance = theoretical_max_slots / n_positions if n_positions > 0 else 0.0

    # Clamp to [0, 0.10] per ACCEPTANCE-001
    return max(0.0, min(0.10, theoretical_max_acceptance))


def generate_overlap_views(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    view: str,
    design: StudyDesign | None = None,
) -> Dict[str, Any]:
    """
    Generate dense or sparse overlap view from Phase C train/test NPZs.

    This function:
    1. Loads train and test NPZs
    2. Computes spacing matrices and filters to positions meeting threshold
    3. Validates filtered datasets against DATA-001 + spacing constraints
    4. Writes filtered NPZs with metadata
    5. Returns spacing metrics for both splits

    Args:
        train_path: Path to Phase C train NPZ
        test_path: Path to Phase C test NPZ
        output_dir: Directory for output NPZs and metrics
        view: Overlap view name ('dense' or 'sparse')
        design: StudyDesign instance (default: get_study_design())

    Returns:
        Dictionary with keys:
            'train_metrics': SpacingMetrics for train split
            'test_metrics': SpacingMetrics for test split
            'train_output': Path to filtered train NPZ
            'test_output': Path to filtered test NPZ

    Raises:
        ValueError: If validation fails or spacing threshold violated
        FileNotFoundError: If input NPZs don't exist

    References:
        - CONFIG-001: Pure utility, no params.cfg access
        - DATA-001: Validator enforces canonical keys/dtypes
        - OVERSAMPLING-001: Preserved for downstream Phase E grouping
    """
    if design is None:
        design = get_study_design()

    # Validate view
    if view not in design.spacing_thresholds:
        raise ValueError(
            f"Unknown view '{view}'. Expected one of: {list(design.spacing_thresholds.keys())}"
        )

    threshold = design.spacing_thresholds[view]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating {view} overlap view")
    print(f"Threshold: {threshold:.2f} px (f_overlap={design.overlap_views[view]})")
    print(f"{'='*60}\n")

    results = {}

    for split_name, split_path in [('train', train_path), ('test', test_path)]:
        print(f"[{split_name.upper()}] Processing {split_path.name}...")

        # Load dataset (allow_pickle=True for _metadata field)
        with np.load(split_path, allow_pickle=True) as data:
            data_dict = {k: data[k] for k in data.keys()}

        # Extract coordinates
        coords = np.stack([data_dict['xcoords'], data_dict['ycoords']], axis=1)

        # Compute spacing and build acceptance mask
        print(f"  Computing spacing matrix for {len(coords)} positions...")
        distances, min_spacing_by_point = compute_spacing_matrix(coords)
        mask = build_acceptance_mask(min_spacing_by_point, threshold)

        # Compute metrics before filtering
        metrics = compute_spacing_metrics(coords, threshold, mask)
        print(f"  Spacing metrics:")
        print(f"    Min: {metrics.min_spacing:.2f} px")
        print(f"    Max: {metrics.max_spacing:.2f} px")
        print(f"    Mean: {metrics.mean_spacing:.2f} px")
        print(f"    Median: {metrics.median_spacing:.2f} px")
        print(f"    Acceptance: {metrics.n_accepted}/{metrics.n_positions} ({metrics.acceptance_rate:.1%})")

        # Guard: require minimum acceptance rate to avoid degenerate datasets
        # Compute geometry-aware acceptance bound based on split bounding box (ACCEPTANCE-001)
        geometry_bound = compute_geometry_aware_acceptance_floor(coords, threshold)
        # Epsilon guard: prevent zero-position datasets with a tiny lower bound
        # Use 80% of geometry bound (greedy packing efficiency) or 0.05% minimum
        epsilon = 0.0005
        min_acceptance_rate = max(epsilon, 0.80 * geometry_bound)

        print(f"  Geometry acceptance bound: {geometry_bound:.4f} ({geometry_bound*100:.2f}%)")
        print(f"  Effective minimum acceptance: {min_acceptance_rate:.4f} ({min_acceptance_rate*100:.2f}%)")

        selection_strategy = 'direct'  # Track whether greedy fallback was used

        if metrics.acceptance_rate < min_acceptance_rate:
            # Attempt greedy spacing-aware downsampling
            print(f"  ⚠ Initial acceptance rate {metrics.acceptance_rate:.1%} < {min_acceptance_rate:.1%}")
            print(f"  Attempting greedy spacing selection with threshold={threshold:.2f} px...")

            greedy_mask = greedy_min_spacing_selection(coords, threshold)
            greedy_metrics = compute_spacing_metrics(coords, threshold, greedy_mask)

            print(f"  Greedy selection result:")
            print(f"    Accepted: {greedy_metrics.n_accepted}/{greedy_metrics.n_positions} ({greedy_metrics.acceptance_rate:.1%})")

            if greedy_metrics.acceptance_rate >= min_acceptance_rate:
                # Greedy selection succeeded
                print(f"    ✓ Greedy selection meets minimum threshold")
                mask = greedy_mask
                metrics = greedy_metrics
                selection_strategy = 'greedy'
            else:
                # Even greedy selection insufficient
                raise ValueError(
                    f"Insufficient positions meet spacing threshold for {view} view in {split_name} split "
                    f"even after greedy downsampling. "
                    f"Direct acceptance: {metrics.acceptance_rate:.1%}, "
                    f"Greedy acceptance: {greedy_metrics.acceptance_rate:.1%}, "
                    f"both < minimum {min_acceptance_rate:.1%} (geometry acceptance bound: {geometry_bound:.1%}). "
                    f"Min spacing: {metrics.min_spacing:.2f} px < threshold: {threshold:.2f} px. "
                    f"Consider regenerating Phase C with wider scan spacing or relaxing overlap fraction."
                )

        # Attach geometry-aware metadata to metrics object (ACCEPTANCE-001)
        metrics.geometry_acceptance_bound = geometry_bound
        metrics.effective_min_acceptance = min_acceptance_rate

        # Filter dataset
        print(f"  Filtering to accepted positions...")
        filtered_data = filter_dataset_by_mask(data_dict, mask)

        # Add metadata (ACCEPTANCE-001)
        filtered_data['_metadata'] = json.dumps({
            'overlap_view': view,
            'spacing_threshold': float(threshold),
            'source_file': str(split_path),
            'n_accepted': int(metrics.n_accepted),
            'n_rejected': int(metrics.n_rejected),
            'acceptance_rate': float(metrics.acceptance_rate),
            'selection_strategy': selection_strategy,
            'geometry_acceptance_bound': float(geometry_bound),
            'effective_min_acceptance': float(min_acceptance_rate),
        })

        # Validate filtered dataset
        print(f"  Validating DATA-001 compliance + spacing constraint...")
        validate_dataset_contract(
            data=filtered_data,
            view=view,
            gridsize=1,  # Phase C datasets are gridsize=1
            neighbor_count=design.neighbor_count,
            design=design,
        )
        print(f"    ✓ Validation passed")

        # Write output NPZ
        output_path = output_dir / f"{view}_{split_name}.npz"
        print(f"  Writing filtered NPZ: {output_path}")
        np.savez_compressed(output_path, **filtered_data)

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
    print(f"{view.capitalize()} view generation complete")
    print(f"{'='*60}\n")

    return results


def main():
    """
    CLI entry point for generating dense/sparse overlap views.

    Usage:
        python -m studies.fly64_dose_overlap.overlap \\
            --phase-c-root data/studies/fly64_dose_overlap \\
            --output-root data/studies/fly64_dose_overlap_views
    """
    parser = argparse.ArgumentParser(
        description="Generate dense/sparse overlap views from Phase C datasets (Phase D)"
    )
    parser.add_argument(
        '--phase-c-root',
        type=Path,
        required=True,
        help='Root directory containing Phase C dose_* subdirectories',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Root directory for overlap view outputs',
    )
    parser.add_argument(
        '--doses',
        type=float,
        nargs='*',
        help='Specific doses to process (default: all from StudyDesign)',
    )
    parser.add_argument(
        '--views',
        type=str,
        nargs='*',
        choices=['dense', 'sparse'],
        help='Specific views to generate (default: both)',
    )
    parser.add_argument(
        '--artifact-root',
        type=Path,
        help='Optional root directory for copying metrics to reports hub (e.g., plans/active/.../reports/<timestamp>)',
    )
    args = parser.parse_args()

    # Load study design
    design = get_study_design()

    # Determine doses to process
    doses = args.doses if args.doses else design.dose_list
    views = args.views if args.views else list(design.overlap_views.keys())

    print("=" * 80)
    print("Phase D: Dense/Sparse Overlap View Generation")
    print("=" * 80)
    print(f"Phase C root: {args.phase_c_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Doses:        {doses}")
    print(f"Views:        {views}")
    print("=" * 80)
    print()

    # Verify Phase C root exists
    if not args.phase_c_root.exists():
        print(f"ERROR: Phase C root not found: {args.phase_c_root}", file=sys.stderr)
        sys.exit(1)

    # Process each dose × view combination
    manifest = {}
    for dose in doses:
        dose_dir = args.phase_c_root / f"dose_{int(dose)}"
        if not dose_dir.exists():
            print(f"WARNING: Skipping dose={dose} (directory not found: {dose_dir})", file=sys.stderr)
            continue

        # Find Phase C train/test NPZs
        # Expected pattern: dose_*/patched_train.npz, dose_*/patched_test.npz
        train_candidates = list(dose_dir.glob("*train.npz"))
        test_candidates = list(dose_dir.glob("*test.npz"))

        if not train_candidates or not test_candidates:
            print(f"WARNING: Skipping dose={dose} (missing train/test NPZs)", file=sys.stderr)
            continue

        train_path = train_candidates[0]
        test_path = test_candidates[0]

        dose_manifest = {'dose': dose, 'views': {}}

        for view in views:
            try:
                output_dir = args.output_root / f"dose_{int(dose)}" / view

                results = generate_overlap_views(
                    train_path=train_path,
                    test_path=test_path,
                    output_dir=output_dir,
                    view=view,
                    design=design,
                )

                # Copy metrics to artifact root if specified
                if args.artifact_root:
                    artifact_metrics_dir = args.artifact_root / 'metrics' / f"dose_{int(dose)}"
                    artifact_metrics_dir.mkdir(parents=True, exist_ok=True)

                    # Copy metrics bundle to reports hub
                    import shutil
                    artifact_bundle_path = artifact_metrics_dir / f"{view}.json"
                    shutil.copy2(results['metrics_bundle_path'], artifact_bundle_path)

                    print(f"  Copied metrics bundle to artifact root:")
                    print(f"    {artifact_bundle_path}")

                dose_manifest['views'][view] = {
                    'train': str(results['train_output']),
                    'test': str(results['test_output']),
                    'train_metrics': str(results['train_metrics_path']),
                    'test_metrics': str(results['test_metrics_path']),
                    'metrics_bundle': str(results['metrics_bundle_path']),
                }

            except Exception as e:
                print(f"\nERROR generating view={view} for dose={dose}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)

        manifest[f"dose_{int(dose)}"] = dose_manifest

    # Write manifest
    manifest_path = args.output_root / 'overlap_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("All overlap views generated successfully!")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()

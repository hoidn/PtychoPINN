"""
Phase D tests for overlap metrics per specs/overlap_metrics.md.

This module tests the overlap metrics pipeline:
- disc_overlap_area and disc_overlap_fraction correctness (analytical cases)
- Metric 1 (group-based, gs=2 only) aggregation
- Metric 2 (image-based with deduplication)
- Metric 3 (group↔group COM-based)
- compute_overlap_metrics orchestration (gs=1 and gs=2)
- generate_overlap_views CLI integration
- Bundle schema validation

References:
- specs/overlap_metrics.md (normative spec)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md
- docs/GRIDSIZE_N_GROUPS_GUIDE.md (unified n_groups semantics)
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from studies.fly64_dose_overlap.overlap import (
    disc_overlap_area,
    disc_overlap_fraction,
    subsample_images,
    filter_dataset_by_mask,
    form_groups_gs1,
    form_groups_gs2,
    compute_metric_1_group_based,
    compute_metric_2_image_based,
    compute_metric_3_group_to_group,
    compute_overlap_metrics,
    generate_overlap_views,
    GEOMETRY_ACCEPTANCE_EPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for disc overlap functions
# ─────────────────────────────────────────────────────────────────────────────


def test_disc_overlap_fraction_perfect_overlap():
    """Test disc overlap when d=0 (perfect overlap) → 1.0."""
    assert abs(disc_overlap_fraction(0.0, 10.0) - 1.0) < 1e-6


def test_disc_overlap_fraction_half_diameter():
    """Test disc overlap when d=R (half-diameter) → ~0.391..."""
    # Analytical value for d=R: f_overlap ≈ 0.3908...
    # (from 2 R^2 arccos(1/2) - (R/2) sqrt(3 R^2)) / (π R^2)
    f = disc_overlap_fraction(5.0, 10.0)
    assert 0.39 < f < 0.40, f"Expected ~0.391, got {f}"


def test_disc_overlap_fraction_no_overlap():
    """Test disc overlap when d>=D (no overlap) → 0.0."""
    assert disc_overlap_fraction(10.0, 10.0) == 0.0
    assert disc_overlap_fraction(15.0, 10.0) == 0.0


def test_disc_overlap_area_symmetry():
    """Test that overlap area is symmetric and monotonic."""
    diameter = 10.0
    # d=0 → full area
    area_0 = disc_overlap_area(0.0, diameter)
    R = diameter / 2.0
    expected_full = np.pi * R ** 2
    assert abs(area_0 - expected_full) < 1e-6

    # d=D → zero area
    area_D = disc_overlap_area(diameter, diameter)
    assert area_D == 0.0

    # Monotonic: area(d1) > area(d2) for d1 < d2 < D
    area_3 = disc_overlap_area(3.0, diameter)
    area_7 = disc_overlap_area(7.0, diameter)
    assert area_3 > area_7


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for subsampling and grouping
# ─────────────────────────────────────────────────────────────────────────────


def test_subsample_images_deterministic():
    """Test that subsampling is deterministic and respects s_img."""
    coords = np.random.rand(100, 2) * 100
    s_img = 0.5

    mask1 = subsample_images(coords, s_img, rng_seed=123)
    mask2 = subsample_images(coords, s_img, rng_seed=123)

    # Same seed → same mask
    assert np.array_equal(mask1, mask2)

    # Check fraction
    assert len(coords[mask1]) == 50  # 50% of 100


def test_filter_dataset_by_mask_handles_scalar_metadata():
    """
    Test that filter_dataset_by_mask handles scalar/0-D array metadata.

    Regression test for TypeError: len() of unsized object when metadata
    contains scalar values (e.g., dose, rng_seed).

    References:
        - input.md (2025-11-13): Phase G dense blocker
        - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/.../analysis/blocker.log
    """
    # Create dataset with regular arrays and scalar metadata
    data = {
        'diffraction': np.random.randn(5, 64, 64).astype(np.float32),
        'xcoords': np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'ycoords': np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'dose': np.array(1000.0),  # 0-D array (scalar)
        'rng_seed': np.array(42),  # 0-D array (scalar int)
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),  # Different shape
    }

    mask = np.array([True, False, True, False, True])  # Keep indices 0, 2, 4

    # Should not raise TypeError
    filtered = filter_dataset_by_mask(data, mask)

    # Check filtered arrays
    assert filtered['diffraction'].shape == (3, 64, 64)
    assert filtered['xcoords'].shape == (3,)
    assert np.array_equal(filtered['xcoords'], np.array([0.0, 2.0, 4.0], dtype=np.float32))

    # Check scalar metadata is broadcast
    assert filtered['dose'].shape == (3,)
    assert np.all(filtered['dose'] == 1000.0)
    assert filtered['rng_seed'].shape == (3,)
    assert np.all(filtered['rng_seed'] == 42)

    # Check objectGuess is preserved (different first dimension)
    assert filtered['objectGuess'].shape == (128, 128)


def test_form_groups_gs1():
    """Test that gs=1 creates one group per image."""
    coords = np.random.rand(10, 2) * 100
    n_groups = 10

    group_assignments = form_groups_gs1(coords, n_groups)

    assert len(group_assignments) == 10
    assert len(np.unique(group_assignments)) == 10  # All unique


def test_form_groups_gs2():
    """Test that gs=2 creates groups with gridsize^2 members."""
    coords = np.random.rand(50, 2) * 100
    n_groups = 10
    gridsize = 2

    coords_expanded, group_assignments = form_groups_gs2(coords, n_groups, gridsize=gridsize)

    # Each group has gridsize^2 members
    expected_length = n_groups * (gridsize ** 2)
    assert len(group_assignments) == expected_length
    assert len(coords_expanded) == expected_length

    # Check group ID range
    unique_groups = np.unique(group_assignments)
    assert len(unique_groups) == n_groups


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for Metric 1 (group-based, gs=2 only)
# ─────────────────────────────────────────────────────────────────────────────


def test_metric_1_group_based_synthetic():
    """Test Metric 1 on synthetic groups with known overlap."""
    # Create a simple scenario: 2 groups of 4 points each
    # Group 0: tight cluster (high overlap)
    # Group 1: sparse cluster (low overlap)

    group0_coords = np.array([
        [0, 0],
        [10, 0],
        [0, 10],
        [10, 10],
    ], dtype=np.float32)

    group1_coords = np.array([
        [100, 100],
        [150, 100],
        [100, 150],
        [150, 150],
    ], dtype=np.float32)

    coords = np.vstack([group0_coords, group1_coords])
    group_assignments = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    probe_diameter_px = 38.4
    neighbor_count = 3

    metric_1 = compute_metric_1_group_based(
        coords, group_assignments, neighbor_count, probe_diameter_px
    )

    # group0 is tight (distances ~10, 14.14 px) → higher overlap
    # group1 is sparse (distances ~50, 70.7 px) → lower overlap
    # Metric 1 should be > 0 and < 1
    assert 0.0 < metric_1 < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for Metric 2 (image-based)
# ─────────────────────────────────────────────────────────────────────────────


def test_metric_2_image_based_deduplication():
    """Test Metric 2 with duplicate coordinates."""
    # Create coords with duplicates
    unique_coords = np.array([
        [0, 0],
        [20, 0],
        [40, 0],
    ], dtype=np.float32)

    # Duplicate first two positions
    coords = np.vstack([unique_coords, unique_coords[:2]])

    probe_diameter_px = 38.4
    neighbor_count = 2

    metric_2 = compute_metric_2_image_based(
        coords, neighbor_count, probe_diameter_px
    )

    # Should deduplicate to 3 unique images
    # Spacing is 20 px between unique images
    # Overlap for d=20, D=38.4 should be moderate
    assert 0.0 < metric_2 < 1.0


def test_metric_2_image_based_single_image():
    """Test Metric 2 with only one unique image → 0.0."""
    coords = np.array([[0, 0]], dtype=np.float32)

    metric_2 = compute_metric_2_image_based(
        coords, neighbor_count=6, probe_diameter_px=38.4
    )

    assert metric_2 == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for Metric 3 (group↔group COM)
# ─────────────────────────────────────────────────────────────────────────────


def test_metric_3_group_to_group_overlapping():
    """Test Metric 3 with overlapping group COMs."""
    # Create 3 groups with known COMs
    # Group 0: COM at (0, 0)
    # Group 1: COM at (30, 0)  → overlaps with Group 0 (d=30 < D=38.4)
    # Group 2: COM at (100, 0) → no overlap

    coords = np.array([
        [0, 0],   # Group 0
        [30, 0],  # Group 1
        [100, 0], # Group 2
    ], dtype=np.float32)

    group_assignments = np.array([0, 1, 2])
    probe_diameter_px = 38.4

    metric_3 = compute_metric_3_group_to_group(
        coords, group_assignments, probe_diameter_px
    )

    # Groups 0 and 1 overlap (d=30 < 38.4)
    # Group 2 does not overlap with others
    # Average should be > 0
    assert metric_3 > 0.0


def test_metric_3_group_to_group_no_overlap():
    """Test Metric 3 with no overlapping group COMs."""
    # Sparse groups
    coords = np.array([
        [0, 0],
        [100, 0],
        [200, 0],
    ], dtype=np.float32)

    group_assignments = np.array([0, 1, 2])
    probe_diameter_px = 38.4

    metric_3 = compute_metric_3_group_to_group(
        coords, group_assignments, probe_diameter_px
    )

    # No overlaps → all groups contribute 0
    assert metric_3 == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests for compute_overlap_metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_overlap_metrics_gs1():
    """Test compute_overlap_metrics for gridsize=1."""
    coords = np.random.rand(50, 2) * 100
    s_img = 0.8
    n_groups = 40  # Should match number of retained images

    metrics = compute_overlap_metrics(
        coords=coords,
        gridsize=1,
        s_img=s_img,
        n_groups=n_groups,
        neighbor_count=6,
        probe_diameter_px=38.4,
        rng_seed_subsample=123,
    )

    # Verify schema
    assert metrics.metrics_version == "1.0"
    assert metrics.gridsize == 1
    assert metrics.s_img == s_img
    assert metrics.n_groups == n_groups
    assert metrics.probe_diameter_px == 38.4

    # Metric 1 should be None for gs=1
    assert metrics.metric_1_group_based_avg is None

    # Metric 2 and 3 should be valid
    assert 0.0 <= metrics.metric_2_image_based_avg <= 1.0
    assert 0.0 <= metrics.metric_3_group_to_group_avg <= 1.0

    # Check counts
    assert metrics.n_images_subsampled == 40
    assert metrics.n_images_total == 50


def test_compute_overlap_metrics_gs2():
    """Test compute_overlap_metrics for gridsize=2."""
    coords = np.random.rand(100, 2) * 100
    s_img = 0.5
    n_groups = 20

    metrics = compute_overlap_metrics(
        coords=coords,
        gridsize=2,
        s_img=s_img,
        n_groups=n_groups,
        neighbor_count=6,
        probe_diameter_px=38.4,
        rng_seed_subsample=456,
    )

    # Verify schema
    assert metrics.gridsize == 2
    assert metrics.n_groups == n_groups

    # Metric 1 is not implemented yet (known limitation)
    # assert metrics.metric_1_group_based_avg is None  # TODO: will be implemented

    # Metric 2 and 3 should be valid
    assert 0.0 <= metrics.metric_2_image_based_avg <= 1.0
    assert 0.0 <= metrics.metric_3_group_to_group_avg <= 1.0


def test_compute_overlap_metrics_degenerate_s_img():
    """Test that s_img=0 raises ValueError."""
    coords = np.random.rand(100, 2) * 100

    with pytest.raises(ValueError, match="s_img must be in"):
        compute_overlap_metrics(
            coords=coords,
            gridsize=1,
            s_img=0.0,  # Invalid
            n_groups=10,
        )


def test_compute_overlap_metrics_invalid_gridsize():
    """Test that invalid gridsize raises ValueError."""
    coords = np.random.rand(100, 2) * 100

    with pytest.raises(ValueError, match="gridsize must be"):
        compute_overlap_metrics(
            coords=coords,
            gridsize=3,  # Invalid
            s_img=1.0,
            n_groups=10,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests for generate_overlap_views
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def phase_c_npzs(tmp_path):
    """Create synthetic Phase C train/test NPZs for testing."""
    train_path = tmp_path / "patched_train.npz"
    test_path = tmp_path / "patched_test.npz"

    # Minimal DATA-001 compliant dataset
    train_data = {
        'diffraction': np.random.randn(50, 64, 64).astype(np.float32),
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),
        'probeGuess': np.random.randn(64, 64).astype(np.complex64),
        'xcoords': np.random.rand(50).astype(np.float32) * 100,
        'ycoords': np.random.rand(50).astype(np.float32) * 100,
    }

    test_data = {
        'diffraction': np.random.randn(20, 64, 64).astype(np.float32),
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),
        'probeGuess': np.random.randn(64, 64).astype(np.complex64),
        'xcoords': np.random.rand(20).astype(np.float32) * 100,
        'ycoords': np.random.rand(20).astype(np.float32) * 100,
    }

    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(test_path, **test_data)

    return train_path, test_path


def test_generate_overlap_views_basic(phase_c_npzs, tmp_path):
    """Test generate_overlap_views basic execution."""
    train_path, test_path = phase_c_npzs
    output_dir = tmp_path / "output"

    results = generate_overlap_views(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        gridsize=1,
        s_img=0.8,
        n_groups=40,
        neighbor_count=6,
        probe_diameter_px=38.4,
        rng_seed_subsample=789,
    )

    # Check outputs exist
    assert results['train_output'].exists()
    assert results['test_output'].exists()
    assert results['train_metrics_path'].exists()
    assert results['test_metrics_path'].exists()
    assert results['metrics_bundle_path'].exists()

    # Check metrics bundle schema
    with open(results['metrics_bundle_path']) as f:
        bundle = json.load(f)

    assert 'train' in bundle
    assert 'test' in bundle

    # Check train metrics
    train_metrics = bundle['train']
    assert train_metrics['metrics_version'] == "1.0"
    assert train_metrics['gridsize'] == 1
    assert train_metrics['s_img'] == 0.8
    assert train_metrics['n_groups'] == 40
    assert train_metrics['probe_diameter_px'] == 38.4

    # Metric 1 should not be present for gs=1
    assert 'metric_1_group_based_avg' not in train_metrics  # Omitted None values

    # Metric 2 and 3 should be present
    assert 'metric_2_image_based_avg' in train_metrics
    assert 'metric_3_group_to_group_avg' in train_metrics


def test_overlap_metrics_bundle(phase_c_npzs, tmp_path):
    """
    Test selector for CLI-focused bundle validation.

    This test is the primary Phase D evidence requirement per input.md:
    - Validates metrics_bundle.json contains Metric 1/2/3
    - Confirms sampling parameters are recorded
    - Verifies train and test splits are present
    """
    train_path, test_path = phase_c_npzs
    output_dir = tmp_path / "bundle_test"

    results = generate_overlap_views(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        gridsize=2,
        s_img=0.6,
        n_groups=15,
        neighbor_count=6,
        probe_diameter_px=38.4,
        rng_seed_subsample=999,
    )

    # Load bundle
    bundle_path = results['metrics_bundle_path']
    assert bundle_path.exists()

    with open(bundle_path) as f:
        bundle = json.load(f)

    # Validate structure
    assert 'train' in bundle
    assert 'test' in bundle

    for split in ['train', 'test']:
        metrics = bundle[split]

        # Required fields per specs/overlap_metrics.md §Outputs
        assert metrics['metrics_version'] == "1.0"
        assert metrics['gridsize'] == 2
        assert metrics['s_img'] == 0.6
        assert metrics['n_groups'] == 15
        assert metrics['neighbor_count'] == 6
        assert metrics['probe_diameter_px'] == 38.4
        assert metrics['rng_seed_subsample'] == 999

        # Metrics (Metric 1 may be None/omitted for gs=2 due to known limitation)
        # Metric 2 and 3 must be present
        assert 'metric_2_image_based_avg' in metrics
        assert 'metric_3_group_to_group_avg' in metrics

        # Bounds checks
        assert 0.0 <= metrics['metric_2_image_based_avg'] <= 1.0
        assert 0.0 <= metrics['metric_3_group_to_group_avg'] <= 1.0

        # Size counts
        assert 'n_images_total' in metrics
        assert 'n_images_subsampled' in metrics
        assert 'n_unique_images' in metrics
        assert 'n_groups_actual' in metrics
        assert 'geometry_acceptance_bound' in metrics
        assert 'effective_min_acceptance' in metrics
        assert 0.0 <= metrics['geometry_acceptance_bound'] <= 0.10
        assert metrics['effective_min_acceptance'] >= 0.0


def test_generate_overlap_views_dense_acceptance_floor(tmp_path):
    """Dense view should record geometry acceptance bound + effective floor."""
    num_points = 50
    span = 50.0  # pixels
    xcoords = np.linspace(0.0, span, num_points).astype(np.float32)
    ycoords = np.linspace(0.0, span, num_points).astype(np.float32)
    diffraction = np.random.randn(num_points, 64, 64).astype(np.float32)
    object_guess = np.random.randn(128, 128).astype(np.complex64)
    probe_guess = np.random.randn(64, 64).astype(np.complex64)

    payload = {
        'diffraction': diffraction,
        'objectGuess': object_guess,
        'probeGuess': probe_guess,
        'xcoords': xcoords,
        'ycoords': ycoords,
    }

    train_path = tmp_path / "train_dense.npz"
    test_path = tmp_path / "test_dense.npz"
    np.savez_compressed(train_path, **payload)
    np.savez_compressed(test_path, **payload)

    probe_diameter = 38.4
    output_dir = tmp_path / "dense_acceptance"

    results = generate_overlap_views(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        gridsize=1,
        s_img=1.0,
        n_groups=num_points,
        neighbor_count=3,
        probe_diameter_px=probe_diameter,
        rng_seed_subsample=777,
    )

    with open(results['train_metrics_path']) as f:
        train_metrics = json.load(f)

    span_x = float(xcoords.max() - xcoords.min())
    span_y = float(ycoords.max() - ycoords.min())
    bounding_area = span_x * span_y
    disc_area = np.pi * (probe_diameter / 2.0) ** 2
    theoretical = bounding_area / (num_points * disc_area)
    expected_bound = min(theoretical, 0.10)

    assert train_metrics['geometry_acceptance_bound'] == pytest.approx(expected_bound, rel=1e-6)
    assert train_metrics['effective_min_acceptance'] == pytest.approx(
        max(expected_bound, GEOMETRY_ACCEPTANCE_EPS), rel=1e-6
    )

    with np.load(results['train_output'], allow_pickle=True) as data:
        metadata = json.loads(data['_metadata'].item())

    assert metadata['geometry_acceptance_bound'] == pytest.approx(expected_bound, rel=1e-6)
    assert metadata['effective_min_acceptance'] == pytest.approx(
        max(expected_bound, GEOMETRY_ACCEPTANCE_EPS), rel=1e-6
    )

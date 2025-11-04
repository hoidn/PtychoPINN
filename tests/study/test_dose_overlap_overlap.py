"""
Phase D tests for dense/sparse overlap view generation.

This module tests the overlap filtering pipeline:
- compute_spacing_matrix correctness
- build_acceptance_mask threshold enforcement
- generate_overlap_views orchestration
- Spacing threshold validation (RED → GREEN workflow)

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/plan.md
- docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151
"""

import json
from pathlib import Path
from typing import Dict
import tempfile

import numpy as np
import pytest

from studies.fly64_dose_overlap.design import get_study_design, StudyDesign
from studies.fly64_dose_overlap.overlap import (
    compute_spacing_matrix,
    build_acceptance_mask,
    filter_dataset_by_mask,
    compute_spacing_metrics,
    generate_overlap_views,
)
from studies.fly64_dose_overlap.validation import validate_dataset_contract


# Fixtures for synthetic test data


@pytest.fixture
def study_design() -> StudyDesign:
    """Canonical study design for test parametrization."""
    return get_study_design()


@pytest.fixture
def synthetic_coords_dense() -> np.ndarray:
    """
    Synthetic coordinates with dense spacing (< dense threshold).

    Designed to violate dense threshold (38.4 px) to trigger RED failure.
    Grid spacing: 30 px → min spacing 30 px < 38.4 px.
    """
    # 3x3 grid with 30 px spacing
    x = np.array([0, 30, 60, 0, 30, 60, 0, 30, 60], dtype=np.float32)
    y = np.array([0, 0, 0, 30, 30, 30, 60, 60, 60], dtype=np.float32)
    return np.stack([x, y], axis=1)  # (9, 2)


@pytest.fixture
def synthetic_coords_sparse() -> np.ndarray:
    """
    Synthetic coordinates with sparse spacing (> sparse threshold).

    Spacing: 110 px > sparse threshold (102.4 px) → should pass.
    """
    x = np.array([0, 110, 220], dtype=np.float32)
    y = np.array([0, 0, 0], dtype=np.float32)
    return np.stack([x, y], axis=1)  # (3, 2)


@pytest.fixture
def synthetic_dataset(synthetic_coords_dense: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Minimal DATA-001 compliant dataset for testing.

    Uses dense coordinates (will violate dense threshold).
    """
    n = len(synthetic_coords_dense)
    return {
        'diffraction': np.random.randn(n, 64, 64).astype(np.float32),
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),
        'probeGuess': np.random.randn(64, 64).astype(np.complex64),
        'xcoords': synthetic_coords_dense[:, 0],
        'ycoords': synthetic_coords_dense[:, 1],
    }


# Unit tests for helper functions


def test_compute_spacing_matrix_basic():
    """Test compute_spacing_matrix on simple 3-point layout."""
    # Equilateral triangle: (0, 0), (10, 0), (5, 8.66)
    coords = np.array([[0, 0], [10, 0], [5, 8.66]], dtype=np.float32)

    distances, min_spacings = compute_spacing_matrix(coords)

    # Check shapes
    assert distances.shape == (3, 3)
    assert min_spacings.shape == (3,)

    # Check symmetry
    assert np.allclose(distances, distances.T)

    # Check diagonal is zero
    assert np.allclose(np.diag(distances), 0)

    # Check min spacings are approximately 10 (triangle side length)
    assert np.allclose(min_spacings, 10.0, atol=0.1)


def test_compute_spacing_matrix_empty():
    """Test edge case: empty coordinates."""
    coords = np.array([]).reshape(0, 2)
    distances, min_spacings = compute_spacing_matrix(coords)

    assert distances.shape == (0,)
    assert min_spacings.shape == (0,)


def test_compute_spacing_matrix_single():
    """Test edge case: single coordinate."""
    coords = np.array([[0, 0]], dtype=np.float32)
    distances, min_spacings = compute_spacing_matrix(coords)

    assert distances.shape == (1, 1)
    assert distances[0, 0] == 0
    assert min_spacings[0] == np.inf  # No neighbors


@pytest.mark.parametrize("view,expected_pass", [
    ("sparse", True),   # sparse threshold = 102.4 px < 110 px spacing
    ("dense", False),   # dense threshold = 38.4 px > 30 px spacing
])
def test_spacing_filter_parametrized(
    synthetic_coords_dense: np.ndarray,
    synthetic_coords_sparse: np.ndarray,
    study_design: StudyDesign,
    view: str,
    expected_pass: bool,
):
    """
    Parametrized test for spacing threshold enforcement.

    This test validates that:
    - Dense coordinates (30 px spacing) violate dense threshold (38.4 px)
    - Sparse coordinates (110 px spacing) pass sparse threshold (102.4 px)

    RED expectation: dense view should fail validation.
    GREEN expectation: After implementation, both views validate correctly.
    """
    # Use appropriate coords for view
    coords = synthetic_coords_sparse if view == "sparse" else synthetic_coords_dense
    threshold = study_design.spacing_thresholds[view]

    # Compute spacing
    distances, min_spacings = compute_spacing_matrix(coords)
    mask = build_acceptance_mask(min_spacings, threshold)

    # Compute actual min spacing
    min_actual = min_spacings[min_spacings != np.inf].min() if len(min_spacings) > 0 else np.inf

    if expected_pass:
        # Should pass: min spacing >= threshold
        assert min_actual >= threshold, (
            f"Expected {view} view to pass but min_spacing={min_actual:.2f} < threshold={threshold:.2f}"
        )
        assert mask.sum() == len(coords), "All positions should be accepted"
    else:
        # Should fail: min spacing < threshold
        assert min_actual < threshold, (
            f"Expected {view} view to fail but min_spacing={min_actual:.2f} >= threshold={threshold:.2f}"
        )
        assert mask.sum() < len(coords), "Some positions should be rejected"


def test_build_acceptance_mask():
    """Test acceptance mask generation with known spacings."""
    min_spacings = np.array([50, 30, 100, 40])
    threshold = 40.0

    mask = build_acceptance_mask(min_spacings, threshold)

    expected = np.array([True, False, True, True])
    assert np.array_equal(mask, expected)


def test_filter_dataset_by_mask(synthetic_dataset: Dict[str, np.ndarray]):
    """Test dataset filtering preserves DATA-001 structure."""
    n = len(synthetic_dataset['xcoords'])
    mask = np.array([True, False, True] + [False] * (n - 3))

    filtered = filter_dataset_by_mask(synthetic_dataset, mask)

    # Check filtered arrays
    assert filtered['diffraction'].shape[0] == 2
    assert len(filtered['xcoords']) == 2
    assert len(filtered['ycoords']) == 2

    # Check preserved arrays (no first-axis filtering)
    assert filtered['objectGuess'].shape == synthetic_dataset['objectGuess'].shape
    assert filtered['probeGuess'].shape == synthetic_dataset['probeGuess'].shape


def test_compute_spacing_metrics(synthetic_coords_dense: np.ndarray, study_design: StudyDesign):
    """Test spacing metrics computation."""
    threshold = study_design.spacing_thresholds['dense']

    metrics = compute_spacing_metrics(synthetic_coords_dense, threshold)

    # Check fields present
    assert metrics.min_spacing > 0
    assert metrics.max_spacing > metrics.min_spacing
    assert metrics.mean_spacing > 0
    assert metrics.median_spacing > 0
    assert metrics.threshold == threshold
    assert metrics.n_positions == len(synthetic_coords_dense)
    assert metrics.n_accepted + metrics.n_rejected == metrics.n_positions
    assert 0 <= metrics.acceptance_rate <= 1.0

    # For 30 px spacing grid, min spacing should be 30 px
    assert np.isclose(metrics.min_spacing, 30.0, atol=1.0)


# Integration test for generate_overlap_views


def test_generate_overlap_views_paths(tmp_path: Path, study_design: StudyDesign):
    """
    Integration test for generate_overlap_views orchestration.

    RED expectation: This test will fail when validation enforces spacing threshold
    on dense coordinates that violate the threshold.

    GREEN expectation: After implementation correctly filters positions,
    this test passes with accepted positions only.

    Selector: pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_paths -vv
    """
    # Create synthetic train/test NPZs with DENSE coordinates (will violate dense threshold)
    coords_dense = np.array([[0, 0], [30, 0], [60, 0], [0, 30], [30, 30], [60, 30]], dtype=np.float32)
    n = len(coords_dense)

    train_data = {
        'diffraction': np.random.randn(n, 64, 64).astype(np.float32),
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),
        'probeGuess': np.random.randn(64, 64).astype(np.complex64),
        'xcoords': coords_dense[:, 0],
        'ycoords': coords_dense[:, 1],
    }
    test_data = train_data.copy()  # Simplified: reuse same coords

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(test_path, **test_data)

    output_dir = tmp_path / "dense_view"

    # Attempt to generate dense view
    # RED: This should raise ValueError due to spacing < threshold
    # The RED test confirms the guard works - coordinates are too dense
    with pytest.raises(ValueError, match="Insufficient positions meet spacing threshold"):
        results = generate_overlap_views(
            train_path=train_path,
            test_path=test_path,
            output_dir=output_dir,
            view='dense',
            design=study_design,
        )

    # GREEN assertion: Test with coordinates that DO meet the threshold
    # Use wider spacing (50 px > dense threshold of 38.4 px)
    coords_acceptable = np.array([[0, 0], [50, 0], [100, 0], [0, 50], [50, 50], [100, 50]], dtype=np.float32)
    n_acceptable = len(coords_acceptable)

    train_data_acceptable = {
        'diffraction': np.random.randn(n_acceptable, 64, 64).astype(np.float32),
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),
        'probeGuess': np.random.randn(64, 64).astype(np.complex64),
        'xcoords': coords_acceptable[:, 0],
        'ycoords': coords_acceptable[:, 1],
    }
    test_data_acceptable = train_data_acceptable.copy()

    train_path_acceptable = tmp_path / "train_acceptable.npz"
    test_path_acceptable = tmp_path / "test_acceptable.npz"
    np.savez_compressed(train_path_acceptable, **train_data_acceptable)
    np.savez_compressed(test_path_acceptable, **test_data_acceptable)

    output_dir_green = tmp_path / "dense_view_green"

    # This should succeed
    results = generate_overlap_views(
        train_path=train_path_acceptable,
        test_path=test_path_acceptable,
        output_dir=output_dir_green,
        view='dense',
        design=study_design,
    )

    # Check outputs exist
    assert results['train_output'].exists()
    assert results['test_output'].exists()

    # Check metrics
    assert results['train_metrics'].n_accepted >= 1  # At least some positions accepted
    assert results['train_metrics'].acceptance_rate > 0.0

    # Validate filtered datasets
    with np.load(results['train_output']) as data:
        filtered_train = {k: data[k] for k in data.keys()}

    validate_dataset_contract(
        data=filtered_train,
        view='dense',
        gridsize=1,
        neighbor_count=study_design.neighbor_count,
        design=study_design,
    )


def test_generate_overlap_views_metrics_manifest(
    tmp_path: Path, study_design: StudyDesign
):
    """
    Test that generate_overlap_views returns metrics file paths in results.

    Phase D requirement (D2): metrics must be stored under
    reports/.../metrics/<dose>/<view>.json, and the function should return
    the paths so CLI consumers can trace evidence.

    RED → GREEN workflow:
    - RED: Assert 'train_metrics_path' and 'test_metrics_path' keys are missing
    - GREEN: After implementation, assert paths exist and point to valid JSON files

    References:
    - input.md:16 — D2 calls for metrics stored under reports/.../metrics/<dose>/<view>.json
    - studies/fly64_dose_overlap/overlap.py:304 — generate_overlap_views return value
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T041900Z/phase_d_metrics_alignment/plan.md
    """
    # Setup: Create synthetic datasets that meet dense threshold
    coords = np.array(
        [[0, 0], [50, 0], [100, 0], [0, 50], [50, 50], [100, 50]], dtype=np.float32
    )  # 50 px spacing > dense threshold (38.4 px)
    n = len(coords)

    dataset = {
        'diffraction': np.random.randn(n, 64, 64).astype(np.float32),
        'objectGuess': np.random.randn(128, 128).astype(np.complex64),
        'probeGuess': np.random.randn(64, 64).astype(np.complex64),
        'xcoords': coords[:, 0],
        'ycoords': coords[:, 1],
    }

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    np.savez_compressed(train_path, **dataset)
    np.savez_compressed(test_path, **dataset)

    output_dir = tmp_path / "dense_view"

    # Execute
    results = generate_overlap_views(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        view='dense',
        design=study_design,
    )

    # GREEN assertions: Validate metrics paths are returned
    assert 'train_metrics_path' in results
    assert 'test_metrics_path' in results
    assert results['train_metrics_path'].exists()
    assert results['test_metrics_path'].exists()

    # Validate metrics bundle path (Phase D requirement D2)
    assert 'metrics_bundle_path' in results
    assert results['metrics_bundle_path'].exists()

    # Validate JSON structure for individual metrics
    with open(results['train_metrics_path']) as f:
        train_metrics_json = json.load(f)
        assert 'min_spacing' in train_metrics_json
        assert 'threshold' in train_metrics_json
        assert 'acceptance_rate' in train_metrics_json

    with open(results['test_metrics_path']) as f:
        test_metrics_json = json.load(f)
        assert 'min_spacing' in test_metrics_json
        assert 'threshold' in test_metrics_json
        assert 'acceptance_rate' in test_metrics_json

    # Validate metrics bundle contains both train and test entries
    with open(results['metrics_bundle_path']) as f:
        bundle = json.load(f)
        assert 'train' in bundle
        assert 'test' in bundle
        assert 'min_spacing' in bundle['train']
        assert 'acceptance_rate' in bundle['train']
        assert 'min_spacing' in bundle['test']
        assert 'acceptance_rate' in bundle['test']

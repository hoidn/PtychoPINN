"""
Tests for the fly64 dose/overlap study design constants.

Validates that Phase A design parameters are correctly encoded and satisfy
all documented constraints from the implementation plan.

Test tier: Unit
Test strategy: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md
"""

import pytest
from studies.fly64_dose_overlap.design import get_study_design, StudyDesign


def test_study_design_constants():
    """
    RED → GREEN TDD test for Phase A design constants.

    Validates that get_study_design() returns a StudyDesign with:
    - Correct dose list [1e3, 1e4, 1e5]
    - Gridsizes {1, 2}
    - neighbor_count=7 for K-choose-C (K ≥ C=4)
    - Patch size N=128 pixels
    - Overlap views: dense=0.7, sparse=0.2
    - Derived spacing thresholds S ≈ (1 - f_group) × N
    - Train/test split axis 'y'
    - Fixed RNG seeds for simulation/grouping/subsampling
    - MS-SSIM sigma=1.0, phase emphasis enabled

    References:
    - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:24-30
    - docs/GRIDSIZE_N_GROUPS_GUIDE.md:142-151 (spacing rule)
    - docs/SAMPLING_USER_GUIDE.md:116-119 (K ≥ C constraint)
    """
    design = get_study_design()

    # Validate type
    assert isinstance(design, StudyDesign), "get_study_design must return StudyDesign instance"

    # Dose sweep
    assert design.dose_list == [1e3, 1e4, 1e5], \
        f"Expected dose_list=[1e3, 1e4, 1e5], got {design.dose_list}"

    # Grouping
    assert design.gridsizes == [1, 2], \
        f"Expected gridsizes=[1, 2], got {design.gridsizes}"
    assert design.neighbor_count == 7, \
        f"Expected neighbor_count=7, got {design.neighbor_count}"

    # K ≥ C constraint for K-choose-C oversampling
    max_gridsize = max(design.gridsizes)
    C = max_gridsize ** 2  # C=4 for gridsize=2
    assert design.neighbor_count >= C, \
        f"neighbor_count={design.neighbor_count} must be ≥ C={C} (gridsize²)"

    # Patch geometry
    assert design.patch_size_pixels == 128, \
        f"Expected patch_size_pixels=128, got {design.patch_size_pixels}"

    # Overlap views
    assert 'dense' in design.overlap_views, "overlap_views must include 'dense'"
    assert 'sparse' in design.overlap_views, "overlap_views must include 'sparse'"
    assert design.overlap_views['dense'] == 0.7, \
        f"Expected dense overlap=0.7, got {design.overlap_views['dense']}"
    assert design.overlap_views['sparse'] == 0.2, \
        f"Expected sparse overlap=0.2, got {design.overlap_views['sparse']}"

    # Derived spacing thresholds S ≈ (1 - f_group) × N
    assert 'dense' in design.spacing_thresholds, "spacing_thresholds must include 'dense'"
    assert 'sparse' in design.spacing_thresholds, "spacing_thresholds must include 'sparse'"

    # Dense: S ≈ (1 - 0.7) × 128 = 38.4 pixels
    expected_dense_spacing = (1.0 - 0.7) * 128
    assert design.spacing_thresholds['dense'] == pytest.approx(expected_dense_spacing), \
        f"Expected dense spacing ≈ {expected_dense_spacing}, got {design.spacing_thresholds['dense']}"

    # Sparse: S ≈ (1 - 0.2) × 128 = 102.4 pixels
    expected_sparse_spacing = (1.0 - 0.2) * 128
    assert design.spacing_thresholds['sparse'] == pytest.approx(expected_sparse_spacing), \
        f"Expected sparse spacing ≈ {expected_sparse_spacing}, got {design.spacing_thresholds['sparse']}"

    # Train/test split
    assert design.train_test_split_axis == 'y', \
        f"Expected split_axis='y', got {design.train_test_split_axis}"

    # RNG seeds
    assert 'simulation' in design.rng_seeds, "rng_seeds must include 'simulation'"
    assert 'grouping' in design.rng_seeds, "rng_seeds must include 'grouping'"
    assert 'subsampling' in design.rng_seeds, "rng_seeds must include 'subsampling'"
    assert design.rng_seeds['simulation'] == 42, \
        f"Expected simulation seed=42, got {design.rng_seeds['simulation']}"
    assert design.rng_seeds['grouping'] == 123, \
        f"Expected grouping seed=123, got {design.rng_seeds['grouping']}"
    assert design.rng_seeds['subsampling'] == 456, \
        f"Expected subsampling seed=456, got {design.rng_seeds['subsampling']}"

    # Metrics config
    assert design.metrics_config['ms_ssim_sigma'] == 1.0, \
        f"Expected ms_ssim_sigma=1.0, got {design.metrics_config['ms_ssim_sigma']}"
    assert design.metrics_config['emphasize_phase'] is True, \
        f"Expected emphasize_phase=True, got {design.metrics_config['emphasize_phase']}"
    assert design.metrics_config['report_amplitude'] is True, \
        f"Expected report_amplitude=True, got {design.metrics_config['report_amplitude']}"


def test_study_design_validation():
    """
    Test that design.validate() raises on constraint violations.

    Validates constraint checking for:
    - K ≥ C (neighbor_count ≥ gridsize²)
    - Overlap fractions in [0, 1)
    - Spacing thresholds positive
    - Dose list non-empty and positive
    """
    # Valid design should not raise
    design = get_study_design()
    design.validate()  # Should not raise

    # Invalid K < C
    bad_design = StudyDesign(neighbor_count=3, gridsizes=[1, 2])
    with pytest.raises(ValueError, match="neighbor_count.*must be ≥ C"):
        bad_design.validate()

    # Invalid overlap fraction
    bad_design = StudyDesign(overlap_views={'dense': 1.5, 'sparse': 0.2})
    with pytest.raises(ValueError, match="overlap_views.*must be in"):
        bad_design.validate()

    # Empty dose list
    bad_design = StudyDesign(dose_list=[])
    with pytest.raises(ValueError, match="dose_list must not be empty"):
        bad_design.validate()

    # Negative dose
    bad_design = StudyDesign(dose_list=[1e3, -1e4, 1e5])
    with pytest.raises(ValueError, match="All doses must be positive"):
        bad_design.validate()


def test_study_design_to_dict():
    """Test that to_dict() exports all design parameters correctly."""
    design = get_study_design()
    d = design.to_dict()

    # Check all keys present
    expected_keys = {
        'dose_list', 'gridsizes', 'neighbor_count', 'patch_size_pixels',
        'overlap_views', 'spacing_thresholds', 'train_test_split_axis',
        'rng_seeds', 'metrics_config'
    }
    assert set(d.keys()) == expected_keys, \
        f"to_dict() missing keys: {expected_keys - set(d.keys())}"

    # Spot check values
    assert d['dose_list'] == [1e3, 1e4, 1e5]
    assert d['gridsizes'] == [1, 2]
    assert d['neighbor_count'] == 7

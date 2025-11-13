"""
Phase B test coverage for dataset contract validation (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001).

Tests validate_dataset_contract() against DATA-001, spacing thresholds, and
oversampling preconditions using small in-memory fixtures (no production datasets).

Test Strategy:
- Use native pytest (no unittest.TestCase mixins) per project guidance
- Small synthetic fixtures for pass/fail scenarios
- Actionable ValueError messages for debugging

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:48-60
- docs/findings.md:DATA-001, CONFIG-001, OVERSAMPLING-001
"""

import numpy as np
import pytest

from studies.fly64_dose_overlap.validation import validate_dataset_contract
from studies.fly64_dose_overlap.design import get_study_design


@pytest.fixture
def valid_dataset():
    """
    Minimal valid NPZ dataset satisfying DATA-001 contract.

    Returns:
        Dictionary mimicking np.load() output with required keys/dtypes.
    """
    n_images = 10
    H, W = 64, 64  # diffraction patch size
    M = 128  # object size (larger than probe per contract guidance)

    return {
        'diffraction': np.random.rand(n_images, H, W).astype(np.float32),  # amplitude
        'objectGuess': np.random.rand(M, M).astype(np.complex64),
        'probeGuess': np.random.rand(H, W).astype(np.complex64),
        # Coordinates in a grid layout with sufficient spacing for 'sparse' view
        # sparse threshold ≈ 102.4 px per StudyDesign
        'xcoords': np.arange(n_images, dtype=np.float32) * 110.0,
        'ycoords': np.arange(n_images, dtype=np.float32) * 110.0,
    }


@pytest.fixture
def design():
    """Study design constants."""
    return get_study_design()


def test_validate_dataset_contract_happy_path(valid_dataset, design):
    """
    Happy path: valid dataset passes all checks without view-specific validation.

    Validates:
    - Required keys present
    - Correct dtypes (float32 amplitude, complex64 object/probe, float coords)
    - Shape consistency (n_images alignment)
    """
    # Should not raise
    validate_dataset_contract(valid_dataset, view=None, gridsize=1, design=design)


def test_validate_dataset_contract_missing_key(valid_dataset, design):
    """
    Fail case: missing required key raises ValueError with actionable message.

    DATA-001 enforcement.
    """
    del valid_dataset['diffraction']
    with pytest.raises(ValueError, match=r"Missing required NPZ keys.*diffraction"):
        validate_dataset_contract(valid_dataset, design=design)


def test_validate_dataset_contract_wrong_dtype_diffraction(valid_dataset, design):
    """
    Fail case: diffraction as complex (intensity) instead of float (amplitude).

    DATA-001 requires amplitude representation.
    """
    # Replace with complex array (wrong)
    valid_dataset['diffraction'] = valid_dataset['diffraction'].astype(np.complex64)
    with pytest.raises(ValueError, match=r"diffraction must be float.*amplitude"):
        validate_dataset_contract(valid_dataset, design=design)


def test_validate_dataset_contract_wrong_dtype_object(valid_dataset, design):
    """
    Fail case: objectGuess as float instead of complex.
    """
    valid_dataset['objectGuess'] = np.abs(valid_dataset['objectGuess']).astype(np.float32)
    with pytest.raises(ValueError, match=r"objectGuess must be complex"):
        validate_dataset_contract(valid_dataset, design=design)


def test_validate_dataset_contract_shape_mismatch(valid_dataset, design):
    """
    Fail case: xcoords length does not match diffraction first axis.
    """
    # Truncate xcoords
    valid_dataset['xcoords'] = valid_dataset['xcoords'][:5]
    with pytest.raises(ValueError, match=r"xcoords length.*!= diffraction first axis"):
        validate_dataset_contract(valid_dataset, design=design)


def test_validate_dataset_contract_spacing_dense(valid_dataset, design):
    """
    Pass case: coordinates satisfy dense spacing threshold (S ≈ 38.4 px).

    Dense overlap requires tighter spacing than sparse.
    """
    # Modify coordinates for dense packing: 40px spacing > 38.4px threshold
    n = len(valid_dataset['xcoords'])
    valid_dataset['xcoords'] = np.arange(n, dtype=np.float32) * 40.0
    valid_dataset['ycoords'] = np.arange(n, dtype=np.float32) * 40.0

    # Should pass for dense view
    validate_dataset_contract(valid_dataset, view='dense', gridsize=1, design=design)


def test_validate_dataset_contract_spacing_violation(valid_dataset, design):
    """
    Spacing thresholds were removed in Phase D: validator should not raise.
    See docs/GRIDSIZE_N_GROUPS_GUIDE.md (spacing acceptance gates retired).
    """
    # Pack coordinates too tightly: 20px < 38.4px threshold for dense
    n = len(valid_dataset['xcoords'])
    valid_dataset['xcoords'] = np.arange(n, dtype=np.float32) * 20.0
    valid_dataset['ycoords'] = np.arange(n, dtype=np.float32) * 20.0

    try:
        validate_dataset_contract(valid_dataset, view='dense', gridsize=1, design=design)
    except ValueError as exc:
        pytest.fail(f"Spacing gates are no longer enforced; unexpected ValueError: {exc}")


def test_validate_dataset_contract_oversampling_precondition_pass(valid_dataset, design):
    """
    Pass case: neighbor_count=7 ≥ C=4 for gridsize=2.

    OVERSAMPLING-001 compliance.
    """
    # Should pass when K ≥ C
    validate_dataset_contract(
        valid_dataset,
        view=None,
        gridsize=2,
        neighbor_count=7,
        design=design,
    )


def test_validate_dataset_contract_oversampling_precondition_fail(valid_dataset, design):
    """
    Fail case: neighbor_count=3 < C=4 for gridsize=2.

    OVERSAMPLING-001 violation.
    """
    with pytest.raises(ValueError, match=r"neighbor_count=3 < C=4.*OVERSAMPLING-001"):
        validate_dataset_contract(
            valid_dataset,
            view=None,
            gridsize=2,
            neighbor_count=3,
            design=design,
        )


def test_validate_dataset_contract_oversampling_missing_neighbor_count(valid_dataset, design):
    """
    Fail case: gridsize>1 but neighbor_count not provided.
    """
    with pytest.raises(ValueError, match=r"neighbor_count required when gridsize=2"):
        validate_dataset_contract(
            valid_dataset,
            view=None,
            gridsize=2,
            neighbor_count=None,
            design=design,
        )


def test_validate_dataset_contract_unknown_view(valid_dataset, design):
    """
    Unknown 'view' values are tolerated for backward compatibility.
    """
    try:
        validate_dataset_contract(
            valid_dataset,
            view='invalid',
            gridsize=1,
            design=design,
        )
    except ValueError as exc:
        pytest.fail(f"'view' parameter is deprecated; unexpected ValueError: {exc}")

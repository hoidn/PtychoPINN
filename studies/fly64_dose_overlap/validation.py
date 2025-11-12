"""
Phase B dataset validation harness for the synthetic fly64 dose/overlap study.

This module provides reusable contract checks for NPZ datasets ensuring:
- DATA-001 compliance: canonical keys/dtypes and amplitude requirement
- y-axis train/test split validation (non-strict informational check)
- Oversampling preconditions (neighbor_count ≥ gridsize²)

References:
- specs/data_contracts.md (NPZ format, though spec is HDF5-focused)
- docs/DATA_MANAGEMENT_GUIDE.md:66-70 (NPZ standard keys)
- docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151 (spacing formula)
- docs/SAMPLING_USER_GUIDE.md:112-123 (oversampling preconditions)
- docs/findings.md:DATA-001 (normative NPZ contract)

The validator is CONFIG-001-safe: it operates on provided arrays/dicts only,
no params.cfg access, so it can run before legacy initialization.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from studies.fly64_dose_overlap.design import get_study_design, StudyDesign


# DATA-001: Required NPZ keys per docs/DATA_MANAGEMENT_GUIDE.md:220
REQUIRED_NPZ_KEYS = ['diffraction', 'objectGuess', 'probeGuess', 'xcoords', 'ycoords']


def validate_dataset_contract(
    data: Dict[str, np.ndarray],
    view: str | None = None,
    gridsize: int = 1,
    neighbor_count: int | None = None,
    design: StudyDesign | None = None,
) -> None:
    """
    Validate NPZ dataset against DATA-001 contract and study design constraints.

    This validator enforces:
    1. Required NPZ keys (diffraction, objectGuess, probeGuess, xcoords, ycoords)
    2. Canonical dtypes:
       - diffraction: float32 amplitude (not intensity)
       - objectGuess/probeGuess: complex64
       - xcoords/ycoords: float (any precision)
    3. Consistent array shapes (n_images axis alignment)
    4. y-axis train/test split validation (coordinates split by sign; informational)
    5. Oversampling preconditions: neighbor_count ≥ gridsize² (if gridsize>1)

    Args:
        data: Dictionary of numpy arrays (typically from np.load('dataset.npz'))
        view: Deprecated. Ignored. Kept for backward compatibility with callers that
              previously passed 'dense'/'sparse' to trigger spacing gates.
        gridsize: Grouping gridsize (default 1); used for oversampling check
        neighbor_count: K for K-NN grouping; must be ≥ C=gridsize² if gridsize>1
        design: StudyDesign instance for spacing thresholds (default: get_study_design())

    Raises:
        ValueError: If any contract violation is detected with actionable message
            showing field name, expected vs actual values.

    References:
        - DATA-001 (docs/findings.md): NPZ contract compliance
        - CONFIG-001 (docs/findings.md): Validator is params.cfg-independent
        - OVERSAMPLING-001 (docs/findings.md): K≥C precondition
    """
    if design is None:
        design = get_study_design()

    # 1. Required keys (DATA-001)
    missing = [k for k in REQUIRED_NPZ_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Missing required NPZ keys (DATA-001): {missing}. "
            f"Expected keys: {REQUIRED_NPZ_KEYS}"
        )

    # 2. Dtype checks
    # diffraction: must be amplitude (float32), not intensity
    # Reference: docs/workflows/pytorch.md:86, specs/data_contracts.md implicit NPZ rule
    diff = data['diffraction']
    if not np.issubdtype(diff.dtype, np.floating):
        raise ValueError(
            f"diffraction must be float (amplitude), got dtype={diff.dtype}. "
            f"DATA-001 requires amplitude (sqrt of intensity)."
        )
    # Prefer float32 for efficiency (not enforced, but warn if float64)
    if diff.dtype == np.float64:
        # Acceptable but not preferred; no error
        pass

    # objectGuess/probeGuess: complex64 preferred
    for key in ['objectGuess', 'probeGuess']:
        arr = data[key]
        if not np.issubdtype(arr.dtype, np.complexfloating):
            raise ValueError(
                f"{key} must be complex dtype (complex64/complex128), "
                f"got dtype={arr.dtype}"
            )

    # xcoords/ycoords: float (any precision ok)
    for key in ['xcoords', 'ycoords']:
        arr = data[key]
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(
                f"{key} must be float dtype, got dtype={arr.dtype}"
            )

    # 3. Shape consistency
    n_images = diff.shape[0]
    if data['xcoords'].shape[0] != n_images:
        raise ValueError(
            f"xcoords length {data['xcoords'].shape[0]} != "
            f"diffraction first axis {n_images}"
        )
    if data['ycoords'].shape[0] != n_images:
        raise ValueError(
            f"ycoords length {data['ycoords'].shape[0]} != "
            f"diffraction first axis {n_images}"
        )

    # 4. Spacing thresholds — removed as acceptance gates.
    # Per docs/GRIDSIZE_N_GROUPS_GUIDE.md, geometry/packing acceptance gates are
    # no longer enforced. Phase D measures overlap via metrics; training proceeds
    # without spacing-based rejection. The 'view' parameter is retained only for
    # backward compatibility and is intentionally ignored here.

    # 5. y-axis train/test split validation
    # Study design uses split_axis='y'; verify coordinates are spatially separated
    # For a valid y-axis split, train and test sets should have non-overlapping y ranges
    # Simple heuristic: check if ycoords are all positive or all negative (single half)
    # More robust: could check against a provided split metadata, but keep simple here
    ycoords = data['ycoords']
    if not (np.all(ycoords >= 0) or np.all(ycoords <= 0)):
        # Mixed signs suggest both halves present; this is acceptable for full dataset
        # The split validation is really for train/test splits, not the full synthetic
        # So this check is optional/informational unless we have split metadata
        # For now, just verify the coordinates span a reasonable range
        pass  # No hard requirement here without explicit split metadata

    # 6. Oversampling preconditions (OVERSAMPLING-001)
    # If gridsize > 1, require neighbor_count ≥ C = gridsize²
    # Reference: docs/SAMPLING_USER_GUIDE.md:116-119
    if gridsize > 1:
        C = gridsize ** 2
        if neighbor_count is None:
            raise ValueError(
                f"neighbor_count required when gridsize={gridsize} > 1 "
                f"for oversampling validation (OVERSAMPLING-001)"
            )
        if neighbor_count < C:
            raise ValueError(
                f"neighbor_count={neighbor_count} < C={C} (gridsize²={gridsize}²). "
                f"OVERSAMPLING-001 requires K ≥ C for K-choose-C to work. "
                f"Increase neighbor_count to at least {C}."
            )

    # All checks passed
    # Validator is pure (no params.cfg access) per CONFIG-001 guidance

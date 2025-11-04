"""
Phase A design constants for the synthetic fly64 dose/overlap study.

This module provides the canonical source of truth for study parameters,
eliminating duplicate constants across scripts. All values are derived from
the implementation plan (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001).

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
- docs/GRIDSIZE_N_GROUPS_GUIDE.md §Inter-Group Overlap Control
- docs/SAMPLING_USER_GUIDE.md §Oversampling (K choose C)
- specs/data_contracts.md §2 (NPZ format requirements)
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class StudyDesign:
    """
    Complete design specification for the fly64 dose/overlap study.

    Attributes:
        dose_list: Photon dose levels to sweep (in photons per exposure)
        gridsizes: Gridsize configurations {1: single-image groups, 2: 4-image groups}
        neighbor_count: K for K-NN grouping and K-choose-C oversampling (gs=2 only)
        patch_size_pixels: Nominal patch size N for spacing calculations
        overlap_views: Overlap fraction targets for dense/sparse views
        spacing_thresholds: Derived minimum center spacing S (pixels) for each view
        train_test_split_axis: Spatial axis for train/test separation ('y')
        rng_seeds: Fixed random seeds for reproducibility
        metrics_config: MS-SSIM and other metric parameters
    """

    # Dose sweep parameters
    dose_list: List[float] = field(default_factory=lambda: [1e3, 1e4, 1e5])

    # Grouping parameters
    gridsizes: List[int] = field(default_factory=lambda: [1, 2])
    neighbor_count: int = 7  # K for K-NN; must be ≥ C=gridsize² for gs=2

    # Patch geometry
    patch_size_pixels: int = 128  # N: nominal patch size from fly64 reconstructions

    # Inter-group overlap control
    overlap_views: Dict[str, float] = field(default_factory=lambda: {
        'dense': 0.7,   # 70% overlap between group centers
        'sparse': 0.2,  # 20% overlap between group centers
    })

    # Derived spacing thresholds S ≈ (1 - f_group) × N
    spacing_thresholds: Dict[str, float] = field(init=False)

    # Train/test split
    train_test_split_axis: str = 'y'

    # Reproducibility seeds
    rng_seeds: Dict[str, int] = field(default_factory=lambda: {
        'simulation': 42,       # For diffraction pattern generation
        'grouping': 123,        # For K-NN group formation
        'subsampling': 456,     # For n_subsample selection
    })

    # Metrics configuration
    metrics_config: Dict[str, Any] = field(default_factory=lambda: {
        'ms_ssim_sigma': 1.0,
        'emphasize_phase': True,
        'report_amplitude': True,
        'frc_threshold': 0.5,
    })

    def __post_init__(self):
        """
        Compute derived spacing thresholds from overlap targets.

        Rule: S ≈ (1 − f_group) × N
        where S is min center spacing (pixels), f_group is overlap fraction,
        N is patch size.

        Reference: docs/GRIDSIZE_N_GROUPS_GUIDE.md:142-151
        """
        self.spacing_thresholds = {
            view: (1.0 - f_overlap) * self.patch_size_pixels
            for view, f_overlap in self.overlap_views.items()
        }

    def validate(self) -> None:
        """
        Validate design constraints and raise if any are violated.

        Checks:
        - K ≥ C for K-choose-C oversampling (K=neighbor_count, C=gridsize²)
        - Overlap fractions in [0, 1)
        - Spacing thresholds positive
        - Dose list non-empty and positive

        Raises:
            ValueError: If any constraint is violated
        """
        # K-choose-C constraint (docs/SAMPLING_USER_GUIDE.md:116-119)
        max_gridsize = max(self.gridsizes)
        C = max_gridsize ** 2
        if self.neighbor_count < C:
            raise ValueError(
                f"neighbor_count={self.neighbor_count} must be ≥ C={C} "
                f"(gridsize²={max_gridsize}²) for K-choose-C oversampling"
            )

        # Overlap fractions in valid range
        for view, f_overlap in self.overlap_views.items():
            if not (0.0 <= f_overlap < 1.0):
                raise ValueError(
                    f"overlap_views['{view}']={f_overlap} must be in [0, 1)"
                )

        # Spacing thresholds positive
        for view, S in self.spacing_thresholds.items():
            if S <= 0:
                raise ValueError(
                    f"spacing_thresholds['{view}']={S} must be positive"
                )

        # Dose list non-empty and positive
        if not self.dose_list:
            raise ValueError("dose_list must not be empty")
        if any(dose <= 0 for dose in self.dose_list):
            raise ValueError("All doses must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Export design as a dictionary for serialization."""
        return {
            'dose_list': self.dose_list,
            'gridsizes': self.gridsizes,
            'neighbor_count': self.neighbor_count,
            'patch_size_pixels': self.patch_size_pixels,
            'overlap_views': self.overlap_views,
            'spacing_thresholds': self.spacing_thresholds,
            'train_test_split_axis': self.train_test_split_axis,
            'rng_seeds': self.rng_seeds,
            'metrics_config': self.metrics_config,
        }


def get_study_design() -> StudyDesign:
    """
    Return the canonical Phase A study design.

    This is the single source of truth for all study parameters.
    Validation is performed automatically on construction.

    Returns:
        StudyDesign instance with validated Phase A constants

    Raises:
        ValueError: If design constraints are violated
    """
    design = StudyDesign()
    design.validate()
    return design

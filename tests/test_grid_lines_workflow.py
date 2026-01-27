"""Unit tests for ptycho.workflows.grid_lines_workflow.

Test strategy: plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md
"""

import numpy as np
import pytest
from pathlib import Path

from ptycho.workflows.grid_lines_workflow import (
    GridLinesConfig,
    scale_probe,
    dataset_out_dir,
)


class TestProbeHelpers:
    """Tests for probe extraction and scaling helpers (Task 2)."""

    def test_scale_probe_resizes_and_smooths(self):
        """scale_probe should resize 4x4 to 8x8 and preserve complex dtype."""
        probe = (np.ones((4, 4)) + 1j * np.ones((4, 4))).astype(np.complex64)
        scaled = scale_probe(probe, target_N=8, smoothing_sigma=0.5)
        assert scaled.shape == (8, 8)
        assert scaled.dtype == np.complex64

    def test_scale_probe_no_resize_when_same_size(self):
        """scale_probe should not resize if already target size."""
        probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        scaled = scale_probe(probe, target_N=8, smoothing_sigma=0.0)
        assert scaled.shape == (8, 8)
        # No smoothing with sigma=0, should be similar
        np.testing.assert_array_almost_equal(scaled, probe)

    def test_scale_probe_rejects_non_square(self):
        """scale_probe should raise for non-square probes."""
        probe = (np.ones((4, 6)) + 1j * np.ones((4, 6))).astype(np.complex64)
        with pytest.raises(ValueError, match="probe must be square"):
            scale_probe(probe, target_N=8, smoothing_sigma=0.5)


class TestDatasetPersistence:
    """Tests for simulation and dataset persistence helpers (Task 3)."""

    def test_dataset_out_dir_layout(self, tmp_path: Path):
        """dataset_out_dir should produce correct path hierarchy."""
        cfg = GridLinesConfig(
            N=64, gridsize=2, output_dir=tmp_path, probe_npz=Path("probe.npz")
        )
        assert dataset_out_dir(cfg) == tmp_path / "datasets" / "N64" / "gs2"

    def test_dataset_out_dir_gridsize1(self, tmp_path: Path):
        """dataset_out_dir should handle gridsize=1."""
        cfg = GridLinesConfig(
            N=128, gridsize=1, output_dir=tmp_path, probe_npz=Path("probe.npz")
        )
        assert dataset_out_dir(cfg) == tmp_path / "datasets" / "N128" / "gs1"

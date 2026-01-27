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
    stitch_predictions,
    save_recon_artifact,
    save_comparison_png_dynamic,
)
from ptycho import params as p


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


class TestStitching:
    """Tests for stitching helper (Task 4)."""

    def test_stitch_predictions_gridsize1(self):
        """stitch_predictions should handle gridsize=1."""
        # Setup params
        p.set("N", 64)
        p.set("gridsize", 1)
        p.set("outer_offset_test", 20)
        p.set("nimgs_test", 4)

        # Create mock predictions: (4 images, 64x64, 1 channel)
        preds = np.random.randn(4, 64, 64, 1) + 1j * np.random.randn(4, 64, 64, 1)
        stitched = stitch_predictions(preds, norm_Y_I=1.0, part="amp")

        # Should produce output with last dim = 1
        assert stitched.shape[-1] == 1
        assert stitched.ndim == 4

    def test_stitch_predictions_gridsize2(self):
        """stitch_predictions should handle gridsize=2."""
        # Setup params
        p.set("N", 64)
        p.set("gridsize", 2)
        p.set("outer_offset_test", 20)
        p.set("nimgs_test", 4)

        # Create mock predictions: (4 images, 64x64, 4 channels for 2x2 grid)
        preds = np.random.randn(4, 64, 64, 4) + 1j * np.random.randn(4, 64, 64, 4)
        stitched = stitch_predictions(preds, norm_Y_I=1.0, part="amp")

        # Should produce output with last dim = 1
        assert stitched.shape[-1] == 1
        assert stitched.ndim == 4

    def test_stitch_predictions_phase(self):
        """stitch_predictions should extract phase correctly."""
        p.set("N", 64)
        p.set("gridsize", 1)
        p.set("outer_offset_test", 20)
        p.set("nimgs_test", 2)

        preds = np.exp(1j * np.pi / 4) * np.ones((2, 64, 64, 1))
        stitched = stitch_predictions(preds, norm_Y_I=1.0, part="phase")

        # Phase should be close to pi/4
        assert stitched.shape[-1] == 1
        # Values should be approximately pi/4 (0.785...)
        assert np.allclose(stitched, np.pi / 4, atol=0.01)


class TestReconArtifacts:
    """Tests for recon artifact helpers."""

    def test_save_recon_artifact_writes_npz(self, tmp_path: Path):
        """save_recon_artifact should write recon.npz with expected keys."""
        recon = (np.ones((4, 4)) + 1j * np.ones((4, 4))).astype(np.complex64)
        path = save_recon_artifact(tmp_path, "pinn", recon)
        assert path.exists()
        with np.load(path) as data:
            assert "YY_pred" in data
            assert "amp" in data
            assert "phase" in data
            assert data["YY_pred"].shape == (4, 4)

    def test_save_comparison_png_dynamic(self, tmp_path: Path):
        """save_comparison_png_dynamic should create a comparison PNG."""
        gt_amp = np.ones((4, 4))
        gt_phase = np.zeros((4, 4))
        recons = {"pinn": {"amp": np.zeros((4, 4)), "phase": np.zeros((4, 4))}}
        out = save_comparison_png_dynamic(
            tmp_path,
            gt_amp,
            gt_phase,
            recons,
            order=("pinn",),
        )
        assert out.exists()

    def test_save_comparison_png_skips_missing(self, tmp_path: Path):
        """save_comparison_png_dynamic should skip missing labels."""
        gt_amp = np.ones((4, 4))
        gt_phase = np.zeros((4, 4))
        recons = {"baseline": {"amp": np.zeros((4, 4)), "phase": np.zeros((4, 4))}}
        out = save_comparison_png_dynamic(
            tmp_path,
            gt_amp,
            gt_phase,
            recons,
            order=("pinn", "baseline"),
        )
        assert out.exists()

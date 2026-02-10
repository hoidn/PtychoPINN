"""Unit tests for ptycho.workflows.grid_lines_workflow.

Test strategy: plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md
"""

import numpy as np
import pytest
from pathlib import Path

from ptycho.workflows.grid_lines_workflow import (
    GridLinesConfig,
    scale_probe,
    apply_probe_mask,
    run_grid_lines_workflow,
    save_split_npz,
    dataset_out_dir,
    stitch_predictions,
    save_recon_artifact,
    save_comparison_png_dynamic,
    _should_share_colorbar,
)
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho import params as p


class TestProbeHelpers:
    """Tests for probe extraction and scaling helpers (Task 2)."""

    def test_scale_probe_resizes_and_smooths(self):
        """scale_probe should resize 4x4 to 8x8 and preserve complex dtype."""
        probe = (np.ones((4, 4)) + 1j * np.ones((4, 4))).astype(np.complex64)
        scaled = scale_probe(
            probe,
            target_N=8,
            smoothing_sigma=0.5,
            scale_mode="interpolate",
        )
        assert scaled.shape == (8, 8)
        assert scaled.dtype == np.complex64

    def test_scale_probe_no_resize_when_same_size(self):
        """scale_probe should not resize if already target size."""
        probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        scaled = scale_probe(
            probe,
            target_N=8,
            smoothing_sigma=0.0,
            scale_mode="interpolate",
        )
        assert scaled.shape == (8, 8)
        # No smoothing with sigma=0, should be similar
        np.testing.assert_array_almost_equal(scaled, probe)

    def test_scale_probe_rejects_non_square(self):
        """scale_probe should raise for non-square probes."""
        probe = (np.ones((4, 6)) + 1j * np.ones((4, 6))).astype(np.complex64)
        with pytest.raises(ValueError, match="probe must be square"):
            scale_probe(
                probe,
                target_N=8,
                smoothing_sigma=0.5,
                scale_mode="interpolate",
            )

    def test_scale_probe_pad_extrapolate_pads_amplitude_and_extrapolates_phase(self):
        """pad_extrapolate should edge-pad amplitude and extrapolate phase."""
        amplitude = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        phase = np.zeros_like(amplitude)
        probe = (amplitude * np.exp(1j * phase)).astype(np.complex64)

        scaled = scale_probe(
            probe,
            target_N=4,
            smoothing_sigma=0.0,
            scale_mode="pad_extrapolate",
        )

        expected_amp = np.pad(amplitude, pad_width=1, mode="edge")
        np.testing.assert_allclose(np.abs(scaled), expected_amp, atol=1e-6)
        assert np.allclose(np.angle(scaled), 0.0, atol=1e-5)

    def test_scale_probe_pad_extrapolate_rejects_downscale(self):
        """pad_extrapolate should reject target sizes smaller than probe."""
        probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        with pytest.raises(
            ValueError,
            match="pad_extrapolate requires target_N >= probe size",
        ):
            scale_probe(
                probe,
                target_N=4,
                smoothing_sigma=0.0,
                scale_mode="pad_extrapolate",
            )

    def test_apply_probe_mask_centered_disk(self):
        """apply_probe_mask should zero outside centered disk."""
        probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        masked = apply_probe_mask(probe, diameter=4)
        assert masked.shape == (8, 8)
        assert masked.dtype == np.complex64
        assert masked[4, 4] != 0
        assert masked[0, 0] == 0

    def test_run_grid_lines_workflow_applies_mask(self, monkeypatch, tmp_path: Path):
        """run_grid_lines_workflow should apply probe mask before simulation."""
        captured = {}

        class _StopWorkflow(Exception):
            pass

        def fake_sim(cfg, probe_np):
            captured["probe"] = probe_np
            raise _StopWorkflow()

        monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", fake_sim)

        probe_path = tmp_path / "probe.npz"
        probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        np.savez(probe_path, probeGuess=probe)

        cfg = GridLinesConfig(
            N=8,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=probe_path,
            probe_mask_diameter=4,
        )

        with pytest.raises(_StopWorkflow):
            run_grid_lines_workflow(cfg)

        assert captured["probe"][0, 0] == 0
        assert captured["probe"][4, 4] != 0

    def test_grid_lines_uses_ideal_disk_probe(self, monkeypatch, tmp_path: Path):
        captured = {}

        class DummyModel:
            def save(self, *args, **kwargs):
                return None

        def fake_sim(cfg, probe_np):
            captured["probe"] = probe_np
            return {
                "train": {
                    "container": object(),
                    "X": np.zeros((1, 1, 1, 1)),
                    "Y_I": np.zeros((1, 1, 1, 1)),
                    "Y_phi": np.zeros((1, 1, 1, 1)),
                },
                "test": {
                    "X": np.zeros((1, 1, 1, 1)),
                    "coords_nominal": np.zeros((1, 2)),
                    "norm_Y_I": 1.0,
                    "YY_ground_truth": np.ones((1, 1, 1, 1), dtype=np.complex64),
                },
            }

        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.simulate_grid_data",
            fake_sim,
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.configure_legacy_params",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.save_split_npz",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.train_pinn_model",
            lambda *args, **kwargs: (DummyModel(), {}),
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.save_pinn_model",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.train_baseline_model",
            lambda *args, **kwargs: (DummyModel(), {}),
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.run_pinn_inference",
            lambda *args, **kwargs: np.zeros((1, 1, 1, 1), dtype=np.complex64),
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.run_baseline_inference",
            lambda *args, **kwargs: np.zeros((1, 1, 1, 1), dtype=np.complex64),
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.stitch_predictions",
            lambda *args, **kwargs: np.zeros((1, 1, 1, 1)),
        )
        monkeypatch.setattr(
            "ptycho.evaluation.eval_reconstruction",
            lambda *args, **kwargs: {},
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.save_recon_artifact",
            lambda *args, **kwargs: tmp_path / "recon.npz",
        )
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.save_comparison_png_dynamic",
            lambda *args, **kwargs: tmp_path / "compare.png",
        )

        cfg = GridLinesConfig(
            N=8,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=tmp_path / "probe.npz",
            probe_source="ideal_disk",
            probe_smoothing_sigma=0.0,
        )

        dummy_probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        np.savez(cfg.probe_npz, probeGuess=dummy_probe)

        run_grid_lines_workflow(cfg)

        probe = captured["probe"]
        assert probe.shape == (8, 8)
        assert probe.dtype == np.complex64
        assert probe[4, 4] != 0
        assert np.abs(probe[0, 0]) < 1e-3


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

    def test_metadata_includes_probe_source(self, monkeypatch, tmp_path: Path):
        captured = {}

        def fake_save_with_metadata(path, payload, metadata):
            captured["metadata"] = metadata

        monkeypatch.setattr(
            "ptycho.metadata.MetadataManager.save_with_metadata",
            fake_save_with_metadata,
        )

        cfg = GridLinesConfig(
            N=8,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=tmp_path / "probe.npz",
            probe_source="ideal_disk",
        )

        config = TrainingConfig(
            model=ModelConfig(N=8, gridsize=1, object_big=False),
            nphotons=1e9,
            nepochs=1,
            batch_size=1,
            nll_weight=0.0,
            mae_weight=1.0,
            realspace_weight=0.0,
        )

        data = {
            "X": np.zeros((1, 1, 1, 1)),
            "Y_I": np.zeros((1, 1, 1, 1)),
            "Y_phi": np.zeros((1, 1, 1, 1)),
            "coords_nominal": np.zeros((1, 2)),
            "coords_true": np.zeros((1, 2)),
            "YY_full": np.zeros((1, 1, 1, 1), dtype=np.complex64),
        }

        save_split_npz(cfg, "train", data, config)

        assert captured["metadata"]["additional_parameters"]["probe_source"] == "ideal_disk"

    def test_metadata_includes_coords_type(self, monkeypatch, tmp_path: Path):
        captured = {}

        def fake_save_with_metadata(path, payload, metadata):
            captured["metadata"] = metadata

        monkeypatch.setattr(
            "ptycho.metadata.MetadataManager.save_with_metadata",
            fake_save_with_metadata,
        )

        cfg = GridLinesConfig(
            N=8,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=tmp_path / "probe.npz",
            probe_source="ideal_disk",
        )

        config = TrainingConfig(
            model=ModelConfig(N=8, gridsize=1, object_big=False),
            nphotons=1e9,
            nepochs=1,
            batch_size=1,
            nll_weight=0.0,
            mae_weight=1.0,
            realspace_weight=0.0,
        )

        data = {
            "X": np.zeros((1, 1, 1, 1)),
            "Y_I": np.zeros((1, 1, 1, 1)),
            "Y_phi": np.zeros((1, 1, 1, 1)),
            "coords_nominal": np.zeros((1, 2)),
            "coords_true": np.zeros((1, 2)),
            "YY_full": np.zeros((1, 1, 1, 1), dtype=np.complex64),
        }

        save_split_npz(cfg, "train", data, config)

        assert captured["metadata"]["additional_parameters"]["coords_type"] == "relative"

    def test_metadata_includes_probe_mask_diameter(self, monkeypatch, tmp_path: Path):
        """Metadata should include probe_mask_diameter when set."""
        from ptycho.workflows.grid_lines_workflow import run_grid_lines_workflow
        from ptycho.config.config import TrainingConfig, ModelConfig

        captured = {}

        class _StopWorkflow(Exception):
            pass

        def fake_save_with_metadata(path, payload, metadata):
            captured["metadata"] = metadata
            raise _StopWorkflow()

        def fake_sim(cfg, probe_np):
            dummy = np.zeros((1, cfg.N, cfg.N, 1), dtype=np.float32)
            coords = np.zeros((1, 2), dtype=np.float32)
            yy_full = np.zeros((cfg.N, cfg.N), dtype=np.complex64)
            return {
                "train": {
                    "X": dummy,
                    "Y_I": dummy,
                    "Y_phi": dummy,
                    "coords_nominal": coords,
                    "coords_true": coords,
                    "YY_full": yy_full,
                },
                "test": {
                    "X": dummy,
                    "Y_I": dummy,
                    "Y_phi": dummy,
                    "coords_nominal": coords,
                    "coords_true": coords,
                    "YY_full": yy_full,
                    "YY_ground_truth": yy_full,
                    "norm_Y_I": 1.0,
                },
            }

        def fake_config(cfg, probe_np):
            return TrainingConfig(
                model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize, object_big=False)
            )

        monkeypatch.setattr(
            "ptycho.metadata.MetadataManager.save_with_metadata",
            fake_save_with_metadata,
        )
        monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", fake_sim)
        monkeypatch.setattr(
            "ptycho.workflows.grid_lines_workflow.configure_legacy_params", fake_config
        )

        probe_path = tmp_path / "probe.npz"
        probe = (np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64)
        np.savez(probe_path, probeGuess=probe)

        cfg = GridLinesConfig(
            N=8,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=probe_path,
            probe_mask_diameter=4,
        )

        with pytest.raises(_StopWorkflow):
            run_grid_lines_workflow(cfg)

        assert captured["metadata"]["additional_parameters"]["probe_mask_diameter"] == 4


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


class TestColorbarSharing:
    """Tests for colorbar sharing decisions."""

    def test_share_colorbar_when_ranges_match(self):
        arr1 = np.array([[0.0, 1.0]])
        arr2 = np.array([[0.0, 1.0]])
        assert _should_share_colorbar([arr1, arr2]) is True

    def test_do_not_share_when_ranges_differ(self):
        arr1 = np.array([[0.0, 1.0]])
        arr2 = np.array([[0.1, 1.0]])
        assert _should_share_colorbar([arr1, arr2]) is False

    def test_do_not_share_when_nan_present(self):
        arr1 = np.array([[np.nan, np.nan]])
        arr2 = np.array([[0.0, 1.0]])
        assert _should_share_colorbar([arr1, arr2]) is False

    def test_share_when_single_subplot(self):
        arr1 = np.array([[0.0, 1.0]])
        assert _should_share_colorbar([arr1]) is True

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


class TestGridLinesCLI:
    def test_grid_lines_cli_probe_mask_diameter(self, tmp_path: Path):
        from scripts.studies import grid_lines_workflow as cli

        args = cli.parse_args([
            "--N", "64",
            "--gridsize", "1",
            "--output-dir", str(tmp_path),
            "--probe-mask-diameter", "64",
        ])
        cfg = cli.build_config(args)
        assert cfg.probe_mask_diameter == 64

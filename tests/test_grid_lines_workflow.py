"""Unit tests for ptycho.workflows.grid_lines_workflow.

Test strategy: plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md
"""

import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

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

    def test_main_writes_cli_invocation_artifacts(self, tmp_path: Path, monkeypatch):
        import json
        from scripts.studies import grid_lines_workflow as cli

        called = {"run": False}

        def fake_run_grid_lines_workflow(cfg):
            called["run"] = True
            return {"metrics": {}}

        monkeypatch.setattr(cli, "run_grid_lines_workflow", fake_run_grid_lines_workflow)

        cli.main(
            [
                "--N",
                "64",
                "--gridsize",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )

        assert called["run"] is True
        inv_json = tmp_path / "invocation.json"
        inv_sh = tmp_path / "invocation.sh"
        assert inv_json.exists()
        assert inv_sh.exists()
        payload = json.loads(inv_json.read_text())
        assert "grid_lines_workflow.py" in payload["command"]
        assert "--output-dir" in payload["argv"]


def test_build_grid_lines_datasets_writes_train_test_npz(monkeypatch, tmp_path: Path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets

    def fake_sim(cfg, probe_np):
        _ = probe_np
        gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
        return {"train": {}, "test": {"YY_ground_truth": gt}}

    def fake_cfg(cfg, probe_np):
        _ = (cfg, probe_np)
        return object()

    def fake_save(cfg, split, data, config):
        _ = (data, config)
        out = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{split}.npz"
        np.savez(path, ok=np.array([1], dtype=np.int8))
        return path

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", fake_sim)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.configure_legacy_params", fake_cfg)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.save_split_npz", fake_save)

    cfg = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe.npz")
    np.savez(
        cfg.probe_npz,
        probeGuess=(np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64),
    )
    result = build_grid_lines_datasets(cfg)
    assert Path(result["train_npz"]).exists()
    assert Path(result["test_npz"]).exists()
    assert Path(result["gt_recon"]).exists()


def test_build_grid_lines_datasets_uses_shared_canonical_gt(monkeypatch, tmp_path: Path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets

    def fake_sim(cfg, probe_np):
        _ = (cfg, probe_np)
        gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
        return {"train": {}, "test": {"YY_ground_truth": gt}}

    def fake_cfg(cfg, probe_np):
        _ = (cfg, probe_np)
        return object()

    def fake_save(cfg, split, data, config):
        _ = (data, config)
        out = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{split}.npz"
        np.savez(path, ok=np.array([1], dtype=np.int8))
        return path

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", fake_sim)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.configure_legacy_params", fake_cfg)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.save_split_npz", fake_save)

    cfg128 = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe128.npz")
    cfg256 = GridLinesConfig(N=256, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe256.npz")
    np.savez(
        cfg128.probe_npz,
        probeGuess=(np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64),
    )
    np.savez(
        cfg256.probe_npz,
        probeGuess=(np.ones((256, 256)) + 1j * np.ones((256, 256))).astype(np.complex64),
    )
    out128 = build_grid_lines_datasets(cfg128, dataset_tag="N128", canonical_gt_label="gt")
    out256 = build_grid_lines_datasets(cfg256, dataset_tag="N256", canonical_gt_label="gt")
    assert out128["gt_recon"] == out256["gt_recon"]
    assert out128["gt_recon"].endswith("recons/gt/recon.npz")


def test_build_grid_lines_datasets_by_n_builds_each_required_n(monkeypatch, tmp_path: Path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets_by_n

    called = []

    def fake_build(cfg, dataset_tag=None, canonical_gt_label="gt"):
        called.append((cfg.N, dataset_tag, canonical_gt_label))
        return {
            "train_npz": "train.npz",
            "test_npz": "test.npz",
            "gt_recon": f"recons/gt_{dataset_tag}/recon.npz",
            "tag": dataset_tag,
        }

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets", fake_build)
    base_cfg = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe.npz")
    out = build_grid_lines_datasets_by_n(base_cfg, required_ns=[256, 128, 256])
    assert sorted(out.keys()) == [128, 256]
    assert called == [(128, "N128", "gt"), (256, "N256", "gt")]


def test_build_grid_lines_datasets_by_n_resets_backend_state(monkeypatch, tmp_path: Path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets_by_n

    reset_calls = []

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow._reset_backend_state",
        lambda: reset_calls.append("reset"),
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets",
        lambda cfg, dataset_tag=None, canonical_gt_label="gt": {
            "train_npz": "train.npz",
            "test_npz": "test.npz",
            "gt_recon": "recons/gt/recon.npz",
            "tag": dataset_tag,
        },
    )

    base_cfg = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe.npz")
    build_grid_lines_datasets_by_n(base_cfg, required_ns=[256, 128, 256])
    assert reset_calls == ["reset", "reset"]


def test_build_grid_lines_datasets_persists_nonconstant_scan_positions(monkeypatch, tmp_path: Path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets

    coords_offsets = np.zeros((3, 1, 2, 1), dtype=np.float32)
    coords_offsets[:, 0, 0, 0] = np.array([0.0, 0.25, 0.5], dtype=np.float32)
    coords_offsets[:, 0, 1, 0] = np.array([0.0, -0.25, -0.5], dtype=np.float32)

    def fake_sim(cfg, probe_np):
        _ = (cfg, probe_np)
        gt = (np.ones((16, 16)) + 1j * np.ones((16, 16))).astype(np.complex64)
        split = {
            "X": np.ones((3, 8, 8, 1), dtype=np.float32),
            "Y_I": np.ones((3, 8, 8, 1), dtype=np.float32),
            "Y_phi": np.zeros((3, 8, 8, 1), dtype=np.float32),
            "coords_nominal": np.zeros((3, 1, 2, 1), dtype=np.float32),
            "coords_true": np.zeros((3, 1, 2, 1), dtype=np.float32),
            "coords_offsets": coords_offsets,
            "YY_full": np.ones((16, 16), dtype=np.complex64),
        }
        return {
            "train": dict(split),
            "test": {
                **split,
                "YY_ground_truth": gt,
                "norm_Y_I": np.array([1.0], dtype=np.float32),
            },
        }

    def fake_cfg(cfg, probe_np):
        _ = (cfg, probe_np)
        return TrainingConfig(
            model=ModelConfig(N=8, gridsize=1, object_big=False),
            nphotons=1e9,
            nepochs=1,
            batch_size=1,
            nll_weight=0.0,
            mae_weight=1.0,
            realspace_weight=0.0,
        )

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.simulate_grid_data", fake_sim)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.configure_legacy_params", fake_cfg)

    probe_path = tmp_path / "probe.npz"
    np.savez(
        probe_path,
        probeGuess=(np.ones((8, 8)) + 1j * np.ones((8, 8))).astype(np.complex64),
    )
    cfg = GridLinesConfig(N=8, gridsize=1, output_dir=tmp_path, probe_npz=probe_path)
    out = build_grid_lines_datasets(cfg)

    with np.load(out["train_npz"], allow_pickle=True) as train_data:
        assert "coords_offsets" in train_data.files
        pos = np.asarray(train_data["coords_offsets"])
        assert np.unique(pos).size > 1


def test_simulate_grid_data_derives_global_scan_positions_when_container_offsets_degenerate(
    monkeypatch,
    tmp_path: Path,
):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, simulate_grid_data

    train_x = np.ones((3, 8, 8, 1), dtype=np.float32)
    test_x = np.ones((2, 8, 8, 1), dtype=np.float32)
    zeros_train = np.zeros((3, 1, 2, 1), dtype=np.float32)
    zeros_test = np.zeros((2, 1, 2, 1), dtype=np.float32)
    gt = (np.ones((16, 16)) + 1j * np.ones((16, 16))).astype(np.complex64)

    dataset = SimpleNamespace(
        train_data=SimpleNamespace(
            coords_nominal=zeros_train,
            coords_true=zeros_train,
            global_offsets=None,
            YY_full=np.ones((16, 16), dtype=np.complex64),
        ),
        test_data=SimpleNamespace(
            coords_nominal=zeros_test,
            coords_true=zeros_test,
            global_offsets=None,
            YY_full=np.ones((16, 16), dtype=np.complex64),
        ),
    )

    def fake_generate_data():
        return (
            train_x,
            np.ones_like(train_x),
            np.zeros_like(train_x),
            test_x,
            np.ones_like(test_x),
            np.zeros_like(test_x),
            gt,
            dataset,
            np.ones((16, 16), dtype=np.complex64),
            np.array([1.0], dtype=np.float32),
        )

    def fake_extract_coords(size, repeats=1, coord_type="offsets", outer_offset=None, **kwargs):
        _ = (size, repeats, coord_type, outer_offset, kwargs)
        if outer_offset == 8:
            ix = np.array([0.0, 1.0, 2.0], dtype=np.float32)[:, None, None, None]
            iy = np.array([10.0, 11.0, 12.0], dtype=np.float32)[:, None, None, None]
            return ix, iy
        ix = np.array([5.0, 6.0], dtype=np.float32)[:, None, None, None]
        iy = np.array([15.0, 16.0], dtype=np.float32)[:, None, None, None]
        return ix, iy

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.configure_legacy_params", lambda *args, **kwargs: None)
    monkeypatch.setattr("ptycho.data_preprocessing.generate_data", fake_generate_data)
    monkeypatch.setattr("ptycho.diffsim.extract_coords", fake_extract_coords)
    p.set("intensity_scale", 1.0)

    cfg = GridLinesConfig(
        N=8,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=tmp_path / "probe.npz",
        nimgs_train=3,
        nimgs_test=2,
    )
    out = simulate_grid_data(cfg, probe_np=np.ones((8, 8), dtype=np.complex64))

    train_offsets = np.asarray(out["train"]["coords_offsets"])
    test_offsets = np.asarray(out["test"]["coords_offsets"])
    assert train_offsets.shape == (3, 1, 2, 1)
    assert test_offsets.shape == (2, 1, 2, 1)
    assert np.unique(train_offsets).size > 1
    assert np.unique(test_offsets).size > 1

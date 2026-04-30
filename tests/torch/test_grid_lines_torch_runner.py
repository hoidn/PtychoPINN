# tests/torch/test_grid_lines_torch_runner.py
"""Smoke tests for the Torch grid-lines runner."""
import json
import math
import subprocess
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.metadata import MetadataManager

from scripts.studies.grid_lines_torch_runner import (
    TorchRunnerConfig,
    _build_paper_row_payload,
    compute_metrics,
    load_cached_dataset,
    _resolve_position_crop_border,
    _select_coords_relative,
    _choose_position_backend,
    _harmonize_prediction_shape,
    _reassemble_position_batched,
    _reassemble_position_shift_sum,
    _reassemble_with_coords_offsets,
    setup_torch_configs,
    run_grid_lines_torch,
    run_torch_training,
)


@pytest.fixture
def synthetic_npz(tmp_path):
    """Create synthetic NPZ files for testing."""
    N = 64
    n_samples = 4
    gridsize = 1

    # Create synthetic data matching expected contract
    data = {
        'diffraction': np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        'Y_I': np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        'Y_phi': np.random.rand(n_samples, N, N, gridsize**2).astype(np.float32),
        'coords_nominal': np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        'coords_true': np.random.rand(n_samples * gridsize**2, 2).astype(np.float32),
        'YY_full': np.random.rand(1, N * 2, N * 2).astype(np.complex64),
        'YY_ground_truth': np.random.rand(1, N * 2, N * 2).astype(np.complex64),
    }

    # Compute correct YY_ground_truth shape for stitching params
    outer_offset_test = 20
    bordersize = (N - outer_offset_test / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))
    tile_size = N - (borderleft + borderright)
    data["YY_ground_truth"] = np.random.rand(tile_size, tile_size, 1).astype(np.complex64)
    data["norm_Y_I"] = np.array(1.0, dtype=np.float32)

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"

    cfg_for_meta = TrainingConfig(model=ModelConfig(N=N, gridsize=gridsize))
    metadata = MetadataManager.create_metadata(
        cfg_for_meta,
        script_name="test_fixture",
        nimgs_test=n_samples,
        outer_offset_test=outer_offset_test,
    )
    MetadataManager.save_with_metadata(str(train_path), data, metadata)
    MetadataManager.save_with_metadata(str(test_path), data, metadata)

    return train_path, test_path


class TestTorchRunnerConfig:
    """Tests for TorchRunnerConfig dataclass."""

    def test_config_creation(self, tmp_path):
        """Test config can be created with required fields."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
        )
        assert cfg.architecture == "fno"
        assert cfg.epochs == 50  # default
        assert cfg.seed == 42  # default

    def test_default_loss_mode_is_mae(self, tmp_path):
        """Test torch_loss_mode defaults to MAE."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
        )
        assert cfg.torch_loss_mode == "mae"

    def test_generator_output_mode_override(self, tmp_path):
        """Generator output mode should be configurable."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
            generator_output_mode="amp_phase_logits",
        )
        assert cfg.generator_output_mode == "amp_phase_logits"

    def test_position_strategy_defaults_are_stable(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
        )
        assert cfg.position_reassembly_backend == "auto"
        assert cfg.position_reassembly_batch_size == 64
        assert cfg.position_crop_border is None

    def test_position_crop_auto_default_resolves_to_quarter_patch(self):
        assert _resolve_position_crop_border(128, 128, None) == 32
        assert _resolve_position_crop_border(256, 256, None) == 64

    def test_position_crop_explicit_zero_disables_crop(self):
        assert _resolve_position_crop_border(128, 128, 0) == 0

    def test_position_crop_clamps_to_nonempty_patch(self):
        assert _resolve_position_crop_border(128, 128, 99) == 63

    @pytest.mark.parametrize("backend", ["auto", "shift_sum", "batched"])
    def test_position_strategy_accepts_supported_backends(self, tmp_path, backend):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
            position_reassembly_backend=backend,
        )
        assert cfg.position_reassembly_backend == backend

    def test_position_strategy_rejects_unknown_backend(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
            position_reassembly_backend="invalid",
        )
        with pytest.raises(ValueError, match="position_reassembly_backend"):
            run_grid_lines_torch(cfg)


class TestLoadCachedDataset:
    """Tests for load_cached_dataset function."""

    def test_load_valid_npz(self, synthetic_npz):
        """Test loading valid NPZ with required keys."""
        train_path, _ = synthetic_npz
        data = load_cached_dataset(train_path)

        assert 'diffraction' in data
        assert 'Y_I' in data
        assert 'Y_phi' in data
        assert 'coords_nominal' in data

    def test_load_missing_key_raises(self, tmp_path):
        """Test loading NPZ missing required keys raises KeyError."""
        bad_path = tmp_path / "bad.npz"
        np.savez(bad_path, other_data=np.zeros(10))

        with pytest.raises(KeyError, match="Missing required key"):
            load_cached_dataset(bad_path)


class TestCoordsRelativeSelection:
    def test_coords_relative_wins_over_nominal(self):
        coords_rel = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        coords_nominal = np.array([[[[9.0, 9.0], [9.0, 9.0]]]], dtype=np.float32)
        data = {"coords_relative": coords_rel, "coords_nominal": coords_nominal}
        metadata = {"additional_parameters": {"coords_type": "nominal"}}
        selected = _select_coords_relative(data, metadata, n_samples=1, channels=2)
        assert np.allclose(selected, coords_rel)

    def test_coords_type_nominal_normalizes(self):
        coords_nominal = np.array([[[[1.0, 3.0], [2.0, 0.0]]]], dtype=np.float32)
        data = {"coords_nominal": coords_nominal}
        metadata = {"additional_parameters": {"coords_type": "nominal"}}
        selected = _select_coords_relative(data, metadata, n_samples=1, channels=2)
        mean = coords_nominal.mean(axis=3, keepdims=True)
        expected = -(coords_nominal - mean)
        assert np.allclose(selected, expected)

    def test_coords_type_relative_passthrough(self):
        coords_nominal = np.array([[[[1.0, 3.0], [2.0, 0.0]]]], dtype=np.float32)
        data = {"coords_nominal": coords_nominal}
        metadata = {"additional_parameters": {"coords_type": "relative"}}
        selected = _select_coords_relative(data, metadata, n_samples=1, channels=2)
        assert np.allclose(selected, coords_nominal)


class TestSetupTorchConfigs:
    """Tests for setup_torch_configs function."""

    def test_creates_training_config(self, tmp_path):
        """Test config setup creates valid TrainingConfig."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
            epochs=100,
            batch_size=32,
        )

        training_config, execution_config = setup_torch_configs(cfg)

        assert training_config.model.architecture == "fno"
        assert training_config.nepochs == 100
        assert training_config.batch_size == 32
        assert training_config.backend == "pytorch"

    def test_creates_execution_config(self, tmp_path):
        """Test config setup creates valid PyTorchExecutionConfig."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="hybrid",
            learning_rate=0.001,
        )

        _, execution_config = setup_torch_configs(cfg)

        assert execution_config.learning_rate == 0.001
        assert execution_config.deterministic is True

    def test_setup_configs_threads_seed_into_subsample_seed(self, tmp_path):
        """Study seed should become the effective subsample/lightning seed."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="hybrid_resnet",
            seed=7,
        )

        training_config, _ = setup_torch_configs(cfg)

        assert training_config.subsample_seed == 7

    def test_fno_input_transform_passed(self, tmp_path):
        """FNO input transform should be forwarded to ModelConfig."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
            fno_input_transform="sqrt",
        )

        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.fno_input_transform == "sqrt"

    def test_setup_configs_threads_scheduler_fields(self, tmp_path):
        """Test that scheduler fields propagate through setup_torch_configs."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="stable_hybrid",
            learning_rate=5e-4,
            scheduler='WarmupCosine',
            lr_warmup_epochs=5,
            lr_min_ratio=0.05,
        )
        training_cfg, exec_cfg = setup_torch_configs(cfg)
        assert training_cfg.scheduler == 'WarmupCosine'
        assert training_cfg.lr_warmup_epochs == 5
        assert training_cfg.lr_min_ratio == 0.05

    def test_setup_configs_threads_plateau_params(self, tmp_path):
        """Test that ReduceLROnPlateau params propagate through setup_torch_configs."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="hybrid",
            scheduler="ReduceLROnPlateau",
            plateau_factor=0.25,
            plateau_patience=5,
            plateau_min_lr=1e-5,
            plateau_threshold=1e-3,
        )
        training_cfg, _ = setup_torch_configs(cfg)
        assert training_cfg.scheduler == "ReduceLROnPlateau"
        assert training_cfg.plateau_factor == 0.25
        assert training_cfg.plateau_patience == 5
        assert training_cfg.plateau_min_lr == 1e-5
        assert training_cfg.plateau_threshold == 1e-3


class TestRunGridLinesTorchScaffold:
    """Smoke tests for the main runner function (scaffold mode)."""

    def test_runner_emits_metrics_json(self, synthetic_npz, tmp_path):
        """Test runner creates metrics.json in output directory."""
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="fno",
            epochs=1,
        )

        # Mock the training to return scaffold results
        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {'train_loss': [], 'val_loss': []},
                'generator': 'fno',
                'scaffold': True,
            }

            # Mock inference to return dummy predictions
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)

                # Mock metrics computation
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1, 'ssim': 0.9}

                    result = run_grid_lines_torch(cfg)

        # Verify output structure
        assert result['architecture'] == 'fno'
        assert 'metrics' in result
        assert 'run_dir' in result

        # Verify metrics.json was created
        metrics_path = Path(result['run_dir']) / "metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            saved_metrics = json.load(f)
        assert 'mse' in saved_metrics or 'ssim' in saved_metrics

    def test_runner_writes_recon_artifact(self, synthetic_npz, tmp_path):
        """Runner should persist recon artifact for visualization."""
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="fno",
            epochs=1,
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {},
                'generator': 'fno',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1}
                    result = run_grid_lines_torch(cfg)

        recon_path = output_dir / "recons" / "pinn_fno" / "recon.npz"
        assert recon_path.exists()
        assert "recon_path" in result

    def test_runner_reports_model_params_and_inference_time(self, synthetic_npz, tmp_path):
        """Runner should report model params and inference time."""
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="fno",
            epochs=1,
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {},
                'generator': 'fno',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1}
                    result = run_grid_lines_torch(cfg)

        assert 'model_params' in result
        assert isinstance(result['model_params'], int)
        assert 'inference_time_s' in result
        assert isinstance(result['inference_time_s'], float)

    def test_runner_emits_paper_row_payload(self, synthetic_npz, tmp_path):
        """Runner should emit paper-row metadata for bundle collation."""
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="hybrid_resnet",
            epochs=2,
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {'train_loss': [0.4, 0.2]},
                'generator': 'hybrid_resnet',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {
                        'mae': [0.1, 0.2],
                        'mse': [0.01, 0.02],
                        'psnr': [70.0, 65.0],
                        'ssim': [0.9, 0.8],
                        'ms_ssim': [0.85, 0.75],
                        'frc50': [64, 48],
                    }
                    result = run_grid_lines_torch(cfg)

        payload = result["paper_row_payload"]
        assert payload["model_label"] == "Hybrid ResNet + PINN"
        assert payload["architecture_id"] == "hybrid_resnet"
        assert payload["training_procedure"] == "pinn"
        assert payload["epoch_budget"] == 2
        assert payload["final_completed_epoch"] == 2
        assert payload["final_train_loss"] == 0.2
        assert payload["row_status"] == "paper_grade"

    def test_runner_writes_randomness_contract(self, synthetic_npz, tmp_path):
        """Runner should publish the effective randomness contract per run."""
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="hybrid_resnet",
            epochs=1,
            seed=11,
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {},
                'generator': 'hybrid_resnet',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1}
                    result = run_grid_lines_torch(cfg)

        contract_path = Path(result['run_dir']) / "randomness_contract.json"
        assert contract_path.exists()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert contract == {
            "requested_seed": 11,
            "effective_subsample_seed": 11,
            "effective_lightning_seed": 11,
        }

    def test_runner_writes_config_and_json_stable_metrics(self, synthetic_npz, tmp_path):
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="hybrid_resnet",
            epochs=1,
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {'train_loss': [0.2], 'val_loss': [0.05]},
                'generator': 'hybrid_resnet',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {
                        'mae': np.array([0.1, 0.2], dtype=np.float32),
                        'mse': np.array([0.01, 0.02], dtype=np.float32),
                        'psnr': np.array([70.0, 65.0], dtype=np.float32),
                        'ssim': np.array([0.9, 0.8], dtype=np.float32),
                        'ms_ssim': np.array([0.85, 0.75], dtype=np.float32),
                        'frc50': np.array([64, 48], dtype=np.float32),
                    }
                    result = run_grid_lines_torch(cfg)

        run_dir = Path(result['run_dir'])
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        assert config_path.exists()
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics_payload["mae"] == [0.10000000149011612, 0.20000000298023224]
        assert isinstance(metrics_payload["frc50"], list)

    def test_build_paper_row_payload_uses_emitted_validation_loss_when_present(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            epochs=2,
            N=128,
        )

        payload = _build_paper_row_payload(
            cfg,
            metrics={"mae": [0.1, 0.2]},
            history={"train_loss": [0.4, 0.2], "val_loss": [0.3, 0.05]},
            model_params=123,
            train_wall_time_sec=1.0,
            inference_time_s=0.5,
        )

        assert payload["validation_loss"] == {"status": "emitted", "value": 0.05}

    def test_runner_creates_run_directory_structure(self, synthetic_npz, tmp_path):
        """Test runner creates proper directory structure."""
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="hybrid",
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {},
                'generator': 'hybrid',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(4, 64, 64, 1).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {}

                    result = run_grid_lines_torch(cfg)

        run_dir = Path(result['run_dir'])
        assert run_dir.name == "pinn_hybrid"
        assert run_dir.parent.name == "runs"

    def test_metrics_stitch_predictions_to_ground_truth(self, synthetic_npz, tmp_path, monkeypatch):
        """Runner should stitch predictions before eval_reconstruction."""
        train_path, test_path = synthetic_npz

        cfg = TrainingConfig(model=ModelConfig(N=64, gridsize=1))
        metadata = MetadataManager.create_metadata(
            cfg,
            script_name="test_grid_lines_torch_runner",
            nimgs_test=1,
            outer_offset_test=20,
        )
        data = dict(np.load(test_path, allow_pickle=True))
        N = 64
        outer_offset_test = 20
        bordersize = (N - outer_offset_test / 2) / 2
        borderleft = int(np.ceil(bordersize))
        borderright = int(np.floor(bordersize))
        tile_size = N - (borderleft + borderright)
        data["YY_ground_truth"] = np.random.rand(tile_size, tile_size, 1).astype(np.complex64)
        data["norm_Y_I"] = np.array(1.0, dtype=np.float32)
        MetadataManager.save_with_metadata(str(test_path), data, metadata)

        called = {"ok": False}

        def fake_eval(stitched_obj, ground_truth_obj, label="", **kwargs):
            _ = kwargs
            if stitched_obj.ndim == 4:
                stitched_obj = stitched_obj[0]
            assert stitched_obj.shape[0] == ground_truth_obj.shape[0]
            assert stitched_obj.shape[1] == ground_truth_obj.shape[1]
            called["ok"] = True
            return {"mse": 0.0}

        monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
            epochs=1,
        )

        def fake_train(*args, **kwargs):
            return {
                "model": None,
                "history": {"train_loss": [], "val_loss": []},
                "generator": "fno",
                "scaffold": True,
            }

        def fake_infer(*args, **kwargs):
            return np.random.rand(1, 64, 64, 1, 2).astype(np.float32)

        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_training",
            fake_train,
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_inference",
            fake_infer,
        )

        run_grid_lines_torch(cfg)
        assert called["ok"] is True

    def test_compute_metrics_accepts_2d_complex_inputs(self, monkeypatch):
        """compute_metrics should normalize 2D complex arrays to (H,W,1)."""
        captured = {}

        def fake_eval(pred, gt, label="", **kwargs):
            _ = kwargs
            captured["pred_shape"] = tuple(np.asarray(pred).shape)
            captured["gt_shape"] = tuple(np.asarray(gt).shape)
            captured["label"] = label
            return {"mse": 0.0}

        monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

        pred = np.ones((64, 64), dtype=np.complex64)
        gt = np.ones((64, 64), dtype=np.complex64)
        out = compute_metrics(pred, gt, label="pinn_hybrid_resnet")

        assert out["mse"] == 0.0
        assert captured["pred_shape"] == (64, 64, 1)
        assert captured["gt_shape"] == (64, 64, 1)
        assert captured["label"] == "pinn_hybrid_resnet"

    def test_compute_metrics_does_not_pass_single_image_frc_kwargs(self, monkeypatch):
        captured = {}

        def fake_eval(pred, gt, label="", **kwargs):
            captured["pred_shape"] = tuple(np.asarray(pred).shape)
            captured["gt_shape"] = tuple(np.asarray(gt).shape)
            captured["label"] = label
            captured["kwargs"] = dict(kwargs)
            return {"mse": 0.0}

        monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

        pred = np.ones((64, 64), dtype=np.complex64)
        gt = np.ones((64, 64), dtype=np.complex64)
        out = compute_metrics(pred, gt, label="pinn_hybrid_resnet")

        assert out["mse"] == 0.0
        assert captured["pred_shape"] == (64, 64, 1)
        assert captured["gt_shape"] == (64, 64, 1)
        assert not any(key.startswith("single_image_frc") for key in captured["kwargs"])

    def test_compute_metrics_normalizes_channel_first_real_imag_ground_truth(self, monkeypatch):
        captured = {}

        def fake_eval(pred, gt, label="", **kwargs):
            _ = kwargs
            captured["pred_shape"] = tuple(np.asarray(pred).shape)
            captured["gt_shape"] = tuple(np.asarray(gt).shape)
            captured["gt_dtype"] = np.asarray(gt).dtype
            captured["label"] = label
            return {"mse": 0.0}

        monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

        pred = np.ones((48, 50), dtype=np.complex64)
        gt = np.stack(
            [
                np.ones((48, 50), dtype=np.float32),
                np.zeros((48, 50), dtype=np.float32),
            ],
            axis=0,
        )
        out = compute_metrics(pred, gt, label="pinn_hybrid_resnet")

        assert out["mse"] == 0.0
        assert captured["pred_shape"] == (48, 50, 1)
        assert captured["gt_shape"] == (48, 50, 1)
        assert captured["label"] == "pinn_hybrid_resnet"

    def test_compute_metrics_normalizes_channel_first_complex_ground_truth(self, monkeypatch):
        captured = {}

        def fake_eval(pred, gt, label="", **kwargs):
            _ = kwargs
            captured["pred_shape"] = tuple(np.asarray(pred).shape)
            captured["gt_shape"] = tuple(np.asarray(gt).shape)
            captured["gt_dtype"] = np.asarray(gt).dtype
            captured["label"] = label
            return {"mse": 0.0}

        monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

        pred = np.ones((32, 34), dtype=np.complex64)
        gt = np.stack(
            [
                np.ones((32, 34), dtype=np.complex64),
                np.full((32, 34), 2.0 + 0.0j, dtype=np.complex64),
            ],
            axis=0,
        )
        out = compute_metrics(pred, gt, label="pinn_hybrid_resnet")

        assert out["mse"] == 0.0
        assert captured["pred_shape"] == (32, 34, 1)
        assert captured["gt_shape"] == (32, 34, 1)
        assert captured["gt_dtype"] == np.complex64
        assert captured["label"] == "pinn_hybrid_resnet"

    def test_position_reassembly_mode_uses_coords_offsets(self, synthetic_npz, tmp_path, monkeypatch):
        """Position mode should use coords_offsets-based reassembly."""
        train_path, test_path = synthetic_npz

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
            epochs=1,
            reassembly_mode="position",
        )

        called = {"position": False}

        def fake_reassemble(obj_tensor, global_offsets, M=20):
            called["position"] = True
            assert global_offsets.shape[-1] == 2
            return np.ones((64, 64), dtype=np.complex64)

        def fake_load_with_metadata(npz_path):
            _ = npz_path
            data = {
                "diffraction": np.ones((4, 64, 64, 1), dtype=np.float32),
                "Y_I": np.ones((4, 64, 64, 1), dtype=np.float32),
                "Y_phi": np.zeros((4, 64, 64, 1), dtype=np.float32),
                "coords_nominal": np.zeros((4, 1, 2, 1), dtype=np.float32),
                "coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32),
                "YY_full": np.ones((64, 64), dtype=np.complex64),
            }
            return data, {"additional_parameters": {}}

        monkeypatch.setattr("ptycho.tf_helper.reassemble_position", fake_reassemble)
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.load_cached_dataset_with_metadata",
            fake_load_with_metadata,
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_training",
            lambda *args, **kwargs: {"model": None, "history": {}},
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_inference",
            lambda *args, **kwargs: np.random.rand(4, 64, 64, 1, 2).astype(np.float32),
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.compute_metrics",
            lambda predictions, ground_truth, label, **kwargs: {"mse": 0.0},
        )

        run_grid_lines_torch(cfg)
        assert called["position"] is True

    def test_position_reassembly_aligns_prediction_to_ground_truth_shape(self, synthetic_npz, tmp_path, monkeypatch):
        """Position mode should harmonize non-square GT and square reassembly outputs."""
        train_path, test_path = synthetic_npz
        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
            epochs=1,
            reassembly_mode="position",
        )

        captured = {}

        def fake_reassemble(obj_tensor, global_offsets, M=20):
            _ = obj_tensor
            _ = global_offsets
            _ = M
            return np.ones((464, 464), dtype=np.complex64)

        def fake_load_with_metadata(npz_path):
            _ = npz_path
            data = {
                "diffraction": np.ones((4, 64, 64, 1), dtype=np.float32),
                "Y_I": np.ones((4, 64, 64, 1), dtype=np.float32),
                "Y_phi": np.zeros((4, 64, 64, 1), dtype=np.float32),
                "coords_nominal": np.zeros((4, 1, 2, 1), dtype=np.float32),
                "coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32),
                "YY_full": np.ones((462, 461), dtype=np.complex64),
                "YY_ground_truth": np.ones((462, 461), dtype=np.complex64),
            }
            return data, {"additional_parameters": {}}

        def fake_compute_metrics(predictions, ground_truth, label, **kwargs):
            captured["pred_shape"] = tuple(np.asarray(predictions).shape)
            captured["gt_shape"] = tuple(np.asarray(ground_truth).shape)
            _ = label
            _ = kwargs
            return {"mse": 0.0}

        monkeypatch.setattr("ptycho.tf_helper.reassemble_position", fake_reassemble)
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.load_cached_dataset_with_metadata",
            fake_load_with_metadata,
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_training",
            lambda *args, **kwargs: {"model": None, "history": {}},
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_inference",
            lambda *args, **kwargs: np.random.rand(4, 64, 64, 1, 2).astype(np.float32),
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.compute_metrics",
            fake_compute_metrics,
        )

        run_grid_lines_torch(cfg)

        assert captured["gt_shape"] == (462, 461)
        assert captured["pred_shape"] == (462, 461)

    def test_position_reassembly_mode_requires_coords_offsets(self):
        """Position mode should fail fast when coords_offsets are absent."""
        pred = np.ones((2, 64, 64, 1), dtype=np.complex64)
        with pytest.raises(ValueError, match="coords_offsets"):
            _reassemble_with_coords_offsets(pred, {"diffraction": np.ones((2, 64, 64, 1), dtype=np.float32)})

    def test_position_reassembly_handles_channel_first_predictions(self, monkeypatch):
        """Channel-first predictions should be normalized before position reassembly."""
        captured = {}

        def fake_reassemble(obj_tensor, global_offsets, M=20):
            captured["obj_shape"] = tuple(np.asarray(obj_tensor).shape)
            captured["offsets_shape"] = tuple(np.asarray(global_offsets).shape)
            captured["M"] = int(M)
            return np.ones((64, 64), dtype=np.complex64)

        monkeypatch.setattr("ptycho.tf_helper.reassemble_position", fake_reassemble)

        pred_channel_first = np.ones((4, 1, 64, 64), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float64)}
        _ = _reassemble_with_coords_offsets(
            pred_channel_first,
            test_data,
            M=64,
            position_crop_border=0,
        )

        assert captured["obj_shape"] == (4, 64, 64, 1)
        assert captured["offsets_shape"] == (4, 1, 1, 2)
        assert captured["M"] == 64

    def test_position_backend_shift_sum_calls_reassemble_position(self, monkeypatch):
        called = {"shift": False}

        def fake_reassemble_position(obj_tensor, global_offsets, M=20):
            called["shift"] = True
            assert int(M) == 64
            assert np.asarray(obj_tensor).shape == (4, 64, 64, 1)
            assert np.asarray(global_offsets).shape == (4, 1, 1, 2)
            return np.ones((64, 64), dtype=np.complex64)

        monkeypatch.setattr("ptycho.tf_helper.reassemble_position", fake_reassemble_position)

        pred = np.ones((4, 64, 64, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=64,
            backend="shift_sum",
            batch_size=64,
            position_crop_border=0,
        )

        assert called["shift"] is True
        assert out.shape == (64, 64)

    def test_position_backend_batched_calls_chunked_reassemble_position(self, monkeypatch):
        captured = {}

        def fake_reassemble_position(obj_tensor, global_offsets, M=20, chunk_size=128):
            captured["patches_shape"] = tuple(np.asarray(obj_tensor).shape)
            captured["offsets_shape"] = tuple(np.asarray(global_offsets).shape)
            captured["M"] = int(M)
            captured["chunk_size"] = int(chunk_size)
            return np.ones((M, M), dtype=np.complex64)

        monkeypatch.setattr("ptycho.tf_helper.reassemble_position", fake_reassemble_position)

        pred = np.ones((4, 64, 64, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=64,
            backend="batched",
            batch_size=32,
            position_crop_border=0,
        )

        assert captured["patches_shape"] == (4, 64, 64, 1)
        assert captured["offsets_shape"] == (4, 1, 1, 2)
        assert captured["M"] == 64
        assert captured["chunk_size"] == 32
        assert out.shape == (64, 64)

    def test_position_backend_batched_matches_shift_sum_on_external_offsets(self):
        """Batched position reassembly should match shift-sum for external offsets."""
        from ptycho import params as p

        # Keep legacy globals deterministic for helper internals.
        p.set("N", 128)
        p.set("gridsize", 1)
        p.set("use_xla_translate", True)

        rng = np.random.default_rng(0)
        patches = (
            rng.standard_normal((8, 128, 128, 1))
            + 1j * rng.standard_normal((8, 128, 128, 1))
        ).astype(np.complex64)
        offsets_b12c = rng.uniform(-150, 150, size=(8, 1, 2, 1)).astype(np.float64)
        offsets_b112 = np.transpose(offsets_b12c, (0, 1, 3, 2))

        shift_sum = _reassemble_position_shift_sum(patches, offsets_b112, M=128)
        batched = _reassemble_position_batched(
            patches,
            offsets_b12c,
            M=128,
            batch_size=16,
        )

        batched_aligned = _harmonize_prediction_shape(batched, shift_sum)
        mae = float(np.mean(np.abs(np.asarray(shift_sum) - np.asarray(batched_aligned))))
        assert mae < 1e-3

    def test_auto_backend_prefers_shift_sum_for_large_position_jobs(self):
        pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}
        backend = _choose_position_backend(pred, test_data, configured="auto")
        assert backend == "shift_sum"

    def test_auto_backend_prefers_shift_sum_for_small_jobs(self):
        pred = np.ones((64, 64, 64, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((64, 1, 2, 1), dtype=np.float32)}
        backend = _choose_position_backend(pred, test_data, configured="auto")
        assert backend == "shift_sum"

    def test_explicit_batched_backend_overrides_auto_preference(self):
        pred = np.ones((64, 64, 64, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((64, 1, 2, 1), dtype=np.float32)}
        backend = _choose_position_backend(pred, test_data, configured="batched")
        assert backend == "batched"

    @pytest.mark.parametrize("backend", ["auto", "shift_sum"])
    def test_shift_sum_oom_falls_back_to_batched(self, monkeypatch, backend):
        import tensorflow as tf

        def raise_resource_exhausted(*args, **kwargs):
            _ = (args, kwargs)
            raise tf.errors.ResourceExhaustedError(node_def=None, op=None, message="OOM")

        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
            raise_resource_exhausted,
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
            lambda patches, offsets_b12c, M, batch_size: np.ones((M, M), dtype=np.complex64),
        )

        pred = np.ones((4, 64, 64, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        runtime_contract = {}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=64,
            backend=backend,
            batch_size=32,
            position_crop_border=0,
            runtime_contract_out=runtime_contract,
        )
        assert out.shape == (64, 64)
        assert runtime_contract["requested_reassembly_backend"] == backend
        assert runtime_contract["resolved_reassembly_backend"] == "batched"
        assert runtime_contract["fallback_used"] is True

    def test_shift_sum_oom_raises_when_fallback_disabled(self, monkeypatch):
        import tensorflow as tf

        def raise_resource_exhausted(*args, **kwargs):
            _ = (args, kwargs)
            raise tf.errors.ResourceExhaustedError(node_def=None, op=None, message="OOM")

        def fail_if_batched_called(*args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("batched fallback should not be called when disabled")

        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
            raise_resource_exhausted,
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_batched",
            fail_if_batched_called,
        )

        pred = np.ones((4, 64, 64, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        runtime_contract = {}
        with pytest.raises(tf.errors.ResourceExhaustedError):
            _reassemble_with_coords_offsets(
                pred,
                test_data,
                M=64,
                backend="shift_sum",
                batch_size=32,
                allow_oom_fallback=False,
                runtime_contract_out=runtime_contract,
            )
        assert runtime_contract["fallback_used"] is False
        assert runtime_contract["resolved_reassembly_backend"] == "shift_sum"

    def test_position_reassembly_does_not_pretrim_patch_tensor(self, monkeypatch):
        captured = {}

        def fake_shift_sum(patches, offsets_b112, M):
            captured["patch_shape"] = tuple(np.asarray(patches).shape)
            captured["offsets_shape"] = tuple(np.asarray(offsets_b112).shape)
            captured["M"] = int(M)
            return np.ones((int(M), int(M)), dtype=np.complex64)

        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
            fake_shift_sum,
        )

        pred = np.ones((4, 128, 128, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=128,
            backend="shift_sum",
            batch_size=64,
            position_crop_border=None,
        )

        assert out.shape == (64, 64)
        assert captured["patch_shape"] == (4, 128, 128, 1)
        assert captured["offsets_shape"] == (4, 1, 1, 2)
        assert captured["M"] == 64

    def test_position_crop_border_zero_keeps_full_window(self, monkeypatch):
        captured = {}

        def fake_shift_sum(patches, offsets_b112, M):
            captured["patch_shape"] = tuple(np.asarray(patches).shape)
            captured["M"] = int(M)
            return np.ones((int(M), int(M)), dtype=np.complex64)

        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
            fake_shift_sum,
        )

        pred = np.ones((4, 128, 128, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=128,
            backend="shift_sum",
            batch_size=64,
            position_crop_border=0,
        )
        assert out.shape == (128, 128)
        assert captured["patch_shape"] == (4, 128, 128, 1)
        assert captured["M"] == 128

    def test_position_crop_border_positive_reduces_effective_m_not_tensor_shape(self, monkeypatch):
        captured = {}

        def fake_shift_sum(patches, offsets_b112, M):
            captured["patch_shape"] = tuple(np.asarray(patches).shape)
            captured["M"] = int(M)
            return np.ones((int(M), int(M)), dtype=np.complex64)

        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
            fake_shift_sum,
        )

        pred = np.ones((4, 128, 128, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=128,
            backend="shift_sum",
            batch_size=64,
            position_crop_border=16,
        )
        assert out.shape == (96, 96)
        assert captured["patch_shape"] == (4, 128, 128, 1)
        assert captured["M"] == 96

    def test_runtime_contract_reports_requested_and_effective_m(self, monkeypatch):
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner._reassemble_position_shift_sum",
            lambda patches, offsets_b112, M: np.ones((int(M), int(M)), dtype=np.complex64),
        )

        pred = np.ones((4, 128, 128, 1), dtype=np.complex64)
        test_data = {"coords_offsets": np.zeros((4, 1, 2, 1), dtype=np.float32)}
        runtime_contract = {}
        out = _reassemble_with_coords_offsets(
            pred,
            test_data,
            M=128,
            backend="shift_sum",
            batch_size=64,
            position_crop_border=None,
            runtime_contract_out=runtime_contract,
        )
        assert out.shape == (64, 64)
        assert runtime_contract["position_crop_border_configured"] is None
        assert runtime_contract["position_crop_border_resolved"] == 32
        assert runtime_contract["position_patch_shape_forwarded"] == [4, 128, 128, 1]
        assert runtime_contract["position_reassembly_M_requested"] == 128
        assert runtime_contract["position_reassembly_M_effective"] == 64

    def test_grid_lines_mode_keeps_existing_stitching_path(self, synthetic_npz, tmp_path, monkeypatch):
        """grid_lines mode should still use the stitch helper path."""
        train_path, test_path = synthetic_npz
        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
            epochs=1,
            reassembly_mode="grid_lines",
        )
        called = {"stitch": False}

        def fake_stitch(pred_complex, cfg_obj, metadata, norm_Y_I):
            called["stitch"] = True
            return np.ones((64, 64), dtype=np.complex64)

        monkeypatch.setattr("scripts.studies.grid_lines_torch_runner._stitch_for_metrics", fake_stitch)
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_training",
            lambda *args, **kwargs: {"model": None, "history": {}},
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.run_torch_inference",
            lambda *args, **kwargs: np.random.rand(4, 64, 64, 1, 2).astype(np.float32),
        )
        monkeypatch.setattr(
            "scripts.studies.grid_lines_torch_runner.compute_metrics",
            lambda predictions, ground_truth, label, **kwargs: {"mse": 0.0},
        )

        run_grid_lines_torch(cfg)
        assert called["stitch"] is True


class TestGradientClipAlgorithm:
    """Tests for gradient_clip_algorithm config propagation."""

    def test_gradient_clip_algorithm_forwarded(self, tmp_path):
        """Test that gradient_clip_algorithm propagates through setup_torch_configs.

        Task ID: FNO-STABILITY-OVERHAUL-001 Task 1.4
        """
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
            gradient_clip_algorithm="agc",
        )

        training_config, execution_config = setup_torch_configs(cfg)

        assert training_config.gradient_clip_algorithm == "agc", \
            "TrainingConfig should inherit gradient_clip_algorithm from runner config"

    def test_gradient_clip_algorithm_default(self, tmp_path):
        """Test that gradient_clip_algorithm defaults to 'norm'."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "output",
            architecture="fno",
        )

        training_config, _ = setup_torch_configs(cfg)

        assert training_config.gradient_clip_algorithm == "norm"


class TestChannelGridsizeAlignment:
    """Tests for channel/gridsize alignment in runner configuration."""

    def test_runner_sets_gridsize_from_config(self, tmp_path):
        """Test that setup_torch_configs propagates gridsize to model config."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid",
            gridsize=2,
        )
        training_config, _ = setup_torch_configs(cfg)
        # Expect gridsize to be propagated correctly
        assert training_config.model.gridsize == 2

    def test_runner_sets_torch_loss_mode(self, tmp_path):
        """Test that setup_torch_configs propagates torch_loss_mode."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid",
            torch_loss_mode="mae",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.torch_loss_mode == "mae"

    def test_runner_sets_torch_mae_pred_l2_match_target(self, tmp_path):
        """Test that setup_torch_configs propagates torch_mae_pred_l2_match_target."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid",
            torch_mae_pred_l2_match_target=True,
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.torch_mae_pred_l2_match_target is True

    def test_runner_sets_training_output_dir_from_cfg(self, tmp_path):
        """TrainingConfig.output_dir should follow the runner output_dir."""
        output_dir = tmp_path / "stage_c_run"
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=output_dir,
            architecture="hybrid_resnet",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.output_dir == output_dir

    def test_runner_sets_probe_mask_controls(self, tmp_path):
        """Test that setup_torch_configs propagates probe mask controls."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid",
            probe_mask=True,
            probe_mask_sigma=0.0,
            probe_mask_diameter=0.75,
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.probe_mask is True
        assert training_config.model.probe_mask_sigma == 0.0
        assert training_config.model.probe_mask_diameter == 0.75

    def test_runner_accepts_stable_hybrid(self, tmp_path):
        """Test that setup_torch_configs accepts 'stable_hybrid' architecture.

        Task ID: FNO-STABILITY-OVERHAUL-001 Task 2.3
        """
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="stable_hybrid",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.architecture == "stable_hybrid"

    def test_runner_accepts_hybrid_resnet(self, tmp_path):
        """Test that setup_torch_configs accepts 'hybrid_resnet' architecture."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.architecture == "hybrid_resnet"

    def test_runner_accepts_spectral_resnet_bottleneck_net(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="spectral_resnet_bottleneck_net",
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert training_config.model.architecture == "spectral_resnet_bottleneck_net"
        assert execution_config.spectral_bottleneck_blocks == 6
        assert execution_config.spectral_bottleneck_modes == 12

    def test_runner_accepts_fno_vanilla(self, tmp_path):
        """Test that setup_torch_configs accepts 'fno_vanilla' architecture."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="fno_vanilla",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.architecture == "fno_vanilla"

    def test_runner_accepts_ffno(self, tmp_path):
        """Test that setup_torch_configs accepts 'ffno' architecture."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="ffno",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.architecture == "ffno"

    def test_runner_accepts_neuralop_uno(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="neuralop_uno",
            N=128,
            gridsize=1,
            generator_output_mode="real_imag",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.architecture == "neuralop_uno"

    def test_runner_accepts_supervised_training_procedure(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="ffno",
            training_procedure="supervised",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.model_type == "supervised"

    def test_runner_accepts_neuralop_uno_for_supervised_training(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="neuralop_uno",
            training_procedure="supervised",
            N=128,
            gridsize=1,
            generator_output_mode="real_imag",
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.architecture == "neuralop_uno"
        assert training_config.model.model_type == "supervised"

    def test_runner_rejects_neuralop_uno_non_lines128_size(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="neuralop_uno",
            N=64,
        )
        with pytest.raises(ValueError, match="N=128"):
            setup_torch_configs(cfg)

    def test_runner_rejects_neuralop_uno_non_unit_gridsize(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="neuralop_uno",
            N=128,
            gridsize=2,
        )
        with pytest.raises(ValueError, match="gridsize=1"):
            setup_torch_configs(cfg)

    def test_runner_rejects_neuralop_uno_non_real_imag_output(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="neuralop_uno",
            N=128,
            generator_output_mode="amp_phase",
        )
        with pytest.raises(ValueError, match="real_imag"):
            setup_torch_configs(cfg)

    def test_runner_rejects_hybrid_resnet_shallow_blocks(self, tmp_path):
        """hybrid_resnet should reject fno_blocks < 3 with a clear error."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            fno_blocks=2,
        )
        with pytest.raises(ValueError, match="fno-blocks"):
            setup_torch_configs(cfg)

    def test_runner_rejects_invalid_resnet_width(self, tmp_path):
        """hybrid_resnet should reject resnet_width not divisible by 4."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            resnet_width=255,
        )
        with pytest.raises(ValueError, match="divisible by 4"):
            setup_torch_configs(cfg)

    def test_runner_passes_resnet_width(self, tmp_path):
        """resnet_width should propagate into ModelConfig for hybrid_resnet."""
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            resnet_width=256,
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.resnet_width == 256

    def test_runner_accepts_capped_channels(self):
        """TorchRunnerConfig max_hidden_channels propagates to ModelConfig."""
        from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, setup_torch_configs
        from pathlib import Path
        cfg = TorchRunnerConfig(
            train_npz=Path("/tmp/fake_train.npz"),
            test_npz=Path("/tmp/fake_test.npz"),
            output_dir=Path("/tmp/fake_out"),
            architecture="hybrid",
            max_hidden_channels=512,
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.model.max_hidden_channels == 512

    def test_runner_accepts_optimizer(self, tmp_path):
        """Test that setup_torch_configs copies optimizer fields from CLI.

        Task ID: FNO-STABILITY-OVERHAUL-001 Phase 8 Task 1
        """
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="stable_hybrid",
            optimizer="adamw",
            weight_decay=0.01,
            momentum=0.9,
            adam_beta1=0.9,
            adam_beta2=0.999,
        )
        training_config, _ = setup_torch_configs(cfg)
        assert training_config.optimizer == "adamw"
        assert training_config.weight_decay == 0.01
        assert training_config.momentum == 0.9
        assert training_config.adam_beta1 == 0.9
        assert training_config.adam_beta2 == 0.999

    def test_runner_torch_only_downsample_steps_stays_out_of_model_config(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_downsample_steps=1,
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.hybrid_downsample_steps == 1
        assert not hasattr(training_config.model, "hybrid_downsample_steps")

    def test_runner_torch_only_downsample_op_stays_out_of_model_config(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_downsample_op="avgpool_conv",
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.hybrid_downsample_op == "avgpool_conv"
        assert not hasattr(training_config.model, "hybrid_downsample_op")

    def test_runner_rejects_invalid_hybrid_downsample_steps(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_downsample_steps=0,
        )
        with pytest.raises(ValueError, match="hybrid_downsample_steps"):
            setup_torch_configs(cfg)

    def test_runner_rejects_invalid_hybrid_downsample_op(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_downsample_op="bad_op",
        )
        with pytest.raises(ValueError, match="hybrid_downsample_op"):
            setup_torch_configs(cfg)

    def test_runner_torch_only_hybrid_encoder_conv_hidden_scale(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_encoder_conv_hidden_scale=0.5,
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.hybrid_encoder_conv_hidden_scale == 0.5
        assert not hasattr(training_config.model, "hybrid_encoder_conv_hidden_scale")

    def test_runner_torch_only_hybrid_encoder_spectral_hidden_scale(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_encoder_spectral_hidden_scale=2.0,
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.hybrid_encoder_spectral_hidden_scale == 2.0
        assert not hasattr(training_config.model, "hybrid_encoder_spectral_hidden_scale")

    def test_runner_torch_only_hybrid_resnet_blocks(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_resnet_blocks=8,
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.hybrid_resnet_blocks == 8
        assert not hasattr(training_config.model, "hybrid_resnet_blocks")

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, math.inf, math.nan])
    def test_runner_rejects_invalid_hybrid_encoder_conv_hidden_scale(self, tmp_path, bad_scale):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_encoder_conv_hidden_scale=bad_scale,
        )
        with pytest.raises(ValueError, match="hybrid_encoder_conv_hidden_scale"):
            setup_torch_configs(cfg)

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, math.inf, math.nan])
    def test_runner_rejects_invalid_hybrid_encoder_spectral_hidden_scale(self, tmp_path, bad_scale):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_encoder_spectral_hidden_scale=bad_scale,
        )
        with pytest.raises(ValueError, match="hybrid_encoder_spectral_hidden_scale"):
            setup_torch_configs(cfg)

    def test_runner_rejects_invalid_hybrid_resnet_blocks(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_resnet_blocks=0,
        )
        with pytest.raises(ValueError, match="hybrid_resnet_blocks"):
            setup_torch_configs(cfg)

    def test_runner_torch_only_spectral_bottleneck_fields_stay_out_of_model_config(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="spectral_resnet_bottleneck_net",
            spectral_bottleneck_blocks=8,
            spectral_bottleneck_modes=10,
            spectral_bottleneck_share_weights=False,
            spectral_bottleneck_gate_init=0.2,
            spectral_bottleneck_gate_mode="per_block",
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.spectral_bottleneck_blocks == 8
        assert execution_config.spectral_bottleneck_modes == 10
        assert execution_config.spectral_bottleneck_share_weights is False
        assert execution_config.spectral_bottleneck_gate_init == 0.2
        assert execution_config.spectral_bottleneck_gate_mode == "per_block"
        assert not hasattr(training_config.model, "spectral_bottleneck_blocks")

    def test_runner_rejects_invalid_spectral_bottleneck_blocks(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="spectral_resnet_bottleneck_net",
            spectral_bottleneck_blocks=0,
        )
        with pytest.raises(ValueError, match="spectral_bottleneck_blocks"):
            setup_torch_configs(cfg)

    def test_runner_rejects_invalid_spectral_bottleneck_modes(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="spectral_resnet_bottleneck_net",
            spectral_bottleneck_modes=0,
        )
        with pytest.raises(ValueError, match="spectral_bottleneck_modes"):
            setup_torch_configs(cfg)

    def test_runner_rejects_invalid_spectral_bottleneck_gate_mode(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="spectral_resnet_bottleneck_net",
            spectral_bottleneck_gate_mode="bad_mode",
        )
        with pytest.raises(ValueError, match="spectral_bottleneck_gate_mode"):
            setup_torch_configs(cfg)

    def test_runner_torch_only_skip_style_stays_out_of_model_config(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_skip_style="concat",
        )
        training_config, execution_config = setup_torch_configs(cfg)
        assert execution_config.hybrid_skip_style == "concat"
        assert not hasattr(training_config.model, "hybrid_skip_style")

    def test_runner_rejects_invalid_skip_style(self, tmp_path):
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=tmp_path / "test.npz",
            output_dir=tmp_path / "out",
            architecture="hybrid_resnet",
            hybrid_skip_style="bad_style",
        )
        with pytest.raises(ValueError, match="hybrid_skip_style"):
            setup_torch_configs(cfg)

    def test_runner_channels_derived_from_gridsize(self, synthetic_npz, tmp_path):
        """Test that channel count C = gridsize^2 is derived correctly.

        For FNO/Hybrid architectures, the number of channels should match
        gridsize squared to handle multi-patch stitching.
        """
        train_path, test_path = synthetic_npz
        for gridsize in [1, 2]:
            cfg = TorchRunnerConfig(
                train_npz=train_path,
                test_npz=test_path,
                output_dir=tmp_path / f"out_{gridsize}",
                architecture="fno",
                gridsize=gridsize,
            )
            training_config, _ = setup_torch_configs(cfg)
            assert training_config.model.gridsize == gridsize, f"gridsize mismatch for gridsize={gridsize}"

    def test_create_training_payload_sets_channels_from_gridsize(self, synthetic_ptycho_npz, tmp_path):
        """Test that config_factory derives C/grid_size from gridsize overrides."""
        from ptycho_torch.config_factory import create_training_payload

        train_npz, _ = synthetic_ptycho_npz
        payload = create_training_payload(
            train_data_file=train_npz,
            output_dir=tmp_path,
            overrides={"n_groups": 4, "gridsize": 1},
        )
        assert payload.pt_data_config.grid_size == (1, 1)
        assert payload.pt_data_config.C == 1
        assert payload.pt_model_config.C_forward == 1
        assert payload.pt_model_config.C_model == 1


class TestArchitecturePropagation:
    """Tests for architecture propagation into torch factory overrides."""

    def test_training_payload_receives_architecture(self, monkeypatch, tmp_path):
        from pathlib import Path
        from ptycho_torch.workflows import components
        from ptycho.config.config import TrainingConfig, ModelConfig

        cfg = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1, architecture="hybrid"),
            train_data_file=Path("/tmp/dummy_train.npz"),
            output_dir=tmp_path,
            backend="pytorch",
            n_groups=4,
        )
        called = {"arch": None}

        def spy_create_payload(*args, **kwargs):
            called["arch"] = kwargs["overrides"].get("architecture")
            raise RuntimeError("stop")

        monkeypatch.setattr("ptycho_torch.config_factory.create_training_payload", spy_create_payload)
        with pytest.raises(RuntimeError, match="stop"):
            components._train_with_lightning(train_container=object(), test_container=None, config=cfg)
        assert called["arch"] == "hybrid"

    def test_workflow_forwards_downsample_steps_and_downsample_op_to_factory(self, monkeypatch, tmp_path):
        from pathlib import Path
        from ptycho_torch.workflows import components
        from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig

        cfg = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1, architecture="hybrid_resnet"),
            train_data_file=Path("/tmp/dummy_train.npz"),
            output_dir=tmp_path,
            backend="pytorch",
            n_groups=4,
        )
        exec_cfg = PyTorchExecutionConfig(
            hybrid_downsample_steps=1,
            hybrid_downsample_op="avgpool_conv",
        )
        captured = {}

        def spy_create_payload(*args, **kwargs):
            captured["overrides"] = kwargs["overrides"]
            raise RuntimeError("stop")

        monkeypatch.setattr("ptycho_torch.config_factory.create_training_payload", spy_create_payload)
        with pytest.raises(RuntimeError, match="stop"):
            components._train_with_lightning(
                train_container=object(),
                test_container=None,
                config=cfg,
                execution_config=exec_cfg,
            )
        assert captured["overrides"]["hybrid_downsample_steps"] == 1
        assert captured["overrides"]["hybrid_downsample_op"] == "avgpool_conv"

    def test_workflow_forwards_hybrid_encoder_conv_hidden_scale_and_hybrid_encoder_spectral_hidden_scale_and_hybrid_resnet_blocks_and_factory(
        self, monkeypatch, tmp_path
    ):
        from pathlib import Path
        from ptycho_torch.workflows import components
        from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig

        cfg = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1, architecture="hybrid_resnet"),
            train_data_file=Path("/tmp/dummy_train.npz"),
            output_dir=tmp_path,
            backend="pytorch",
            n_groups=4,
        )
        exec_cfg = PyTorchExecutionConfig(
            hybrid_encoder_conv_hidden_scale=0.5,
            hybrid_encoder_spectral_hidden_scale=2.0,
            hybrid_resnet_blocks=8,
        )
        captured = {}

        def spy_create_payload(*args, **kwargs):
            captured["overrides"] = kwargs["overrides"]
            raise RuntimeError("stop")

        monkeypatch.setattr("ptycho_torch.config_factory.create_training_payload", spy_create_payload)
        with pytest.raises(RuntimeError, match="stop"):
            components._train_with_lightning(
                train_container=object(),
                test_container=None,
                config=cfg,
                execution_config=exec_cfg,
            )
        assert captured["overrides"]["hybrid_encoder_conv_hidden_scale"] == 0.5
        assert captured["overrides"]["hybrid_encoder_spectral_hidden_scale"] == 2.0
        assert captured["overrides"]["hybrid_resnet_blocks"] == 8

    def test_workflow_forwards_skip_style_to_factory(self, monkeypatch, tmp_path):
        from pathlib import Path
        from ptycho_torch.workflows import components
        from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig

        cfg = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1, architecture="hybrid_resnet"),
            train_data_file=Path("/tmp/dummy_train.npz"),
            output_dir=tmp_path,
            backend="pytorch",
            n_groups=4,
        )
        exec_cfg = PyTorchExecutionConfig(hybrid_skip_style="gated_add")
        captured = {}

        def spy_create_payload(*args, **kwargs):
            captured["overrides"] = kwargs["overrides"]
            raise RuntimeError("stop")

        monkeypatch.setattr("ptycho_torch.config_factory.create_training_payload", spy_create_payload)
        with pytest.raises(RuntimeError, match="stop"):
            components._train_with_lightning(
                train_container=object(),
                test_container=None,
                config=cfg,
                execution_config=exec_cfg,
            )
        assert captured["overrides"]["hybrid_skip_style"] == "gated_add"


class TestForwardSignatureEnforcement:
    """Tests for FNO/Hybrid forward signature enforcement."""

    def test_fno_inference_uses_forward_predict(self, synthetic_npz, tmp_path):
        """Test that FNO/Hybrid inference calls forward_predict with full signature."""
        from scripts.studies.grid_lines_torch_runner import run_torch_inference
        import torch

        train_path, test_path = synthetic_npz
        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
        )

        class SpyModel:
            """Model that tracks call arguments."""
            def eval(self):
                return self

            def __init__(self):
                self.calls = []

            def forward_predict(self, x, positions, probe, input_scale_factor):
                self.calls.append((x, positions, probe, input_scale_factor))
                batch_size = x.shape[0]
                return torch.zeros(batch_size, x.shape[2], x.shape[3], dtype=torch.complex64)

        # Load test data
        test_data = dict(np.load(test_path, allow_pickle=True))

        # Run inference with spy model
        spy_model = SpyModel()
        _ = run_torch_inference(spy_model, test_data, cfg)

        assert spy_model.calls, "forward_predict was never called"
        for x, positions, probe, input_scale_factor in spy_model.calls:
            assert x.ndim == 4
            assert positions.ndim == 4
            assert probe is not None
            assert input_scale_factor.shape[-3:] == (1, 1, 1)

    def test_inference_explicitly_moves_model_to_device(self, synthetic_npz, tmp_path):
        """Inference must call model.to(device) before running forward_predict."""
        from scripts.studies.grid_lines_torch_runner import run_torch_inference
        import torch

        _, test_path = synthetic_npz
        cfg = TorchRunnerConfig(
            train_npz=tmp_path / "train.npz",
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
        )

        class SpyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
                self.to_calls = []

            def eval(self):
                return self

            def to(self, device):
                self.to_calls.append(str(device))
                return super().to(device)

            def forward_predict(self, x, positions, probe, input_scale_factor):
                _ = (positions, probe, input_scale_factor)
                return torch.zeros(
                    x.shape[0],
                    x.shape[2],
                    x.shape[3],
                    dtype=torch.complex64,
                    device=x.device,
                )

        test_data = dict(np.load(test_path, allow_pickle=True))
        model = SpyModel()
        _ = run_torch_inference(model, test_data, cfg)

        assert model.to_calls, "run_torch_inference must call model.to(device) explicitly"
        assert any(
            ("cpu" in call) or ("cuda" in call) or ("mps" in call)
            for call in model.to_calls
        )


class TestOutputContractConversion:
    """Tests for output contract conversion (real/imag to complex)."""

    def test_to_complex_patches_basic(self):
        """Test that to_complex_patches converts real/imag to complex correctly."""
        from scripts.studies.grid_lines_torch_runner import to_complex_patches

        # Create test input: (B, H, W, C, 2) where last dim is [real, imag]
        real_imag = np.zeros((2, 4, 4, 1, 2), dtype=np.float32)
        real_imag[..., 0] = 1.0  # Real part
        real_imag[..., 1] = 2.0  # Imaginary part

        result = to_complex_patches(real_imag)

        # Check output shape (should drop the last dimension)
        assert result.shape == (2, 4, 4, 1), f"Expected shape (2, 4, 4, 1), got {result.shape}"
        # Check complex values
        assert result.dtype == np.complex64 or result.dtype == np.complex128
        np.testing.assert_array_almost_equal(result.real, 1.0)
        np.testing.assert_array_almost_equal(result.imag, 2.0)

    def test_to_complex_patches_preserves_values(self):
        """Test that to_complex_patches preserves real and imaginary values."""
        from scripts.studies.grid_lines_torch_runner import to_complex_patches

        # Random input
        real_part = np.random.rand(3, 8, 8, 2).astype(np.float32)
        imag_part = np.random.rand(3, 8, 8, 2).astype(np.float32)
        real_imag = np.stack([real_part, imag_part], axis=-1)

        result = to_complex_patches(real_imag)

        np.testing.assert_array_almost_equal(result.real, real_part)
        np.testing.assert_array_almost_equal(result.imag, imag_part)

    def test_runner_returns_predictions_complex(self, synthetic_npz, tmp_path):
        """Test that run_grid_lines_torch returns predictions_complex key.

        The runner should convert model outputs (real/imag format) to complex
        patches for downstream physics consistency checks.
        """
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="fno",
            epochs=1,
        )

        # Mock training to return scaffold results (model=None to skip state_dict save)
        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,  # None skips checkpoint save
                'history': {'train_loss': [], 'val_loss': []},
                'generator': 'fno',
                'scaffold': True,
            }

            # Mock inference to return real/imag output
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                # Simulate model output: (B, H, W, C, 2) real/imag format
                mock_output = np.random.rand(4, 64, 64, 1, 2).astype(np.float32)
                mock_infer.return_value = mock_output

                # Mock metrics computation
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1}

                    result = run_grid_lines_torch(cfg)

        # Verify predictions_complex is in result
        assert 'predictions_complex' in result, (
            "run_grid_lines_torch should return 'predictions_complex' key "
            "with complex-valued predictions"
        )
        # Verify it's actually complex
        assert np.iscomplexobj(result['predictions_complex']), (
            "predictions_complex should be complex-valued"
        )


class TestTorchTrainingPath:
    """Tests for PyTorch training path usage."""

    def test_runner_uses_lightning_training(self, synthetic_npz, tmp_path, monkeypatch):
        """Test that runner delegates training to Lightning workflow."""
        from unittest.mock import MagicMock
        train_path, test_path = synthetic_npz

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="fno",
        )

        train_data = load_cached_dataset(train_path)
        test_data = load_cached_dataset(test_path)

        called = {"train": False}

        def fake_train(train_container, test_container, config, execution_config=None, overrides=None):
            called["train"] = True
            assert "X" in train_container
            assert "coords_nominal" in train_container
            assert train_container["coords_nominal"].ndim == 4
            return {
                "history": {"train_loss": []},
                "models": {"diffraction_to_obj": MagicMock()},
            }

        monkeypatch.setattr("ptycho_torch.workflows.components._train_with_lightning", fake_train)
        result = run_torch_training(cfg, train_data, test_data)
        assert called["train"] is True
        assert "models" in result

    def test_runner_bridges_supervised_labels_from_grid_lines_targets(self, synthetic_npz, tmp_path, monkeypatch):
        """Supervised study runs should bridge Y_I/Y_phi into label_amp/label_phase."""
        from unittest.mock import MagicMock

        train_path, test_path = synthetic_npz

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=tmp_path,
            architecture="ffno",
            training_procedure="supervised",
        )

        train_data = load_cached_dataset(train_path)
        test_data = load_cached_dataset(test_path)

        captured = {}

        def fake_train(train_container, test_container, config, execution_config=None, overrides=None):
            captured["train_container"] = train_container
            captured["test_container"] = test_container
            captured["model_type"] = config.model.model_type
            return {
                "history": {"train_loss": []},
                "models": {"diffraction_to_obj": MagicMock()},
            }

        monkeypatch.setattr("ptycho_torch.workflows.components._train_with_lightning", fake_train)

        run_torch_training(cfg, train_data, test_data)

        assert captured["model_type"] == "supervised"
        assert "label_amp" in captured["train_container"]
        assert "label_phase" in captured["train_container"]
        assert "label_amp" in captured["test_container"]
        assert "label_phase" in captured["test_container"]
        np.testing.assert_allclose(captured["train_container"]["label_amp"], train_data["Y_I"])
        np.testing.assert_allclose(captured["train_container"]["label_phase"], train_data["Y_phi"])


def test_main_writes_cli_invocation_artifacts(tmp_path, monkeypatch):
    import ptycho_torch
    from scripts.studies import grid_lines_torch_runner as runner

    called = {"run": False}
    fake_ptycho_init = tmp_path / "fake_ptycho_torch" / "__init__.py"
    fake_ptycho_init.parent.mkdir(parents=True, exist_ok=True)
    fake_ptycho_init.write_text("MARKER = 'runner-test'\n", encoding="utf-8")

    def fake_run_grid_lines_torch(cfg, **kwargs):
        called["run"] = True
        assert "invocation_argv" in kwargs
        assert "invocation_extra" in kwargs
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        runner._write_runner_invocation_artifacts(
            cfg,
            argv=kwargs["invocation_argv"],
            extra=kwargs["invocation_extra"],
        )
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)
    monkeypatch.setattr(ptycho_torch, "__file__", str(fake_ptycho_init))
    monkeypatch.setenv("PYTHONPATH", "/tmp/session_repo")

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "hybrid_resnet",
            "--epochs",
            "1",
        ]
    )

    assert called["run"] is True
    inv_json = out_dir / "runs" / "pinn_hybrid_resnet" / "invocation.json"
    inv_sh = out_dir / "runs" / "pinn_hybrid_resnet" / "invocation.sh"
    assert inv_json.exists()
    assert inv_sh.exists()
    payload = json.loads(inv_json.read_text())
    assert "grid_lines_torch_runner.py" in payload["command"]
    assert "--architecture" in payload["argv"]
    assert payload["extra"]["runtime_provenance"]["pythonpath"] == "/tmp/session_repo"
    assert payload["extra"]["runtime_provenance"]["ptycho_torch_file"] == str(fake_ptycho_init)


def test_library_run_writes_invocation_artifacts(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    class FakeModel:
        def parameters(self):
            return []

        def state_dict(self):
            return {"weights": np.array([1.0], dtype=np.float32)}

    dataset = {
        "YY_ground_truth": np.ones((4, 4), dtype=np.complex64),
    }

    monkeypatch.setattr(runner, "load_cached_dataset_with_metadata", lambda path: (dataset, None))
    monkeypatch.setattr(
        runner,
        "run_torch_training",
        lambda cfg, train_data, test_data, train_metadata=None, test_metadata=None: {
            "history": {"train_loss": [0.1]},
            "model": FakeModel(),
        },
    )
    monkeypatch.setattr(
        runner,
        "run_torch_inference",
        lambda model, test_data, cfg, metadata=None: np.ones((4, 4), dtype=np.complex64),
    )
    monkeypatch.setattr(
        runner,
        "compute_metrics",
        lambda pred, gt, label: {"mae": [0.1, 0.2], "mse": [0.01, 0.02]},
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.save_recon_artifact",
        lambda output_dir, model_id, recon: output_dir / "recons" / model_id / "recon.npz",
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="hybrid_resnet",
        epochs=1,
    )
    cfg.train_npz.write_bytes(b"stub")
    cfg.test_npz.write_bytes(b"stub")

    runner.run_grid_lines_torch(cfg)

    inv_json = tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.json"
    inv_sh = tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.sh"
    assert inv_json.exists()
    assert inv_sh.exists()
    payload = json.loads(inv_json.read_text())
    assert payload["script"] == "scripts/studies/grid_lines_torch_runner.py"
    assert payload["parsed_args"]["architecture"] == "hybrid_resnet"
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["finished_at_utc"]


def test_build_paper_row_payload_uses_supervised_ffno_label(tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="ffno",
        training_procedure="supervised",
        epochs=1,
        N=128,
    )
    payload = _build_paper_row_payload(
        cfg,
        metrics={"mae": [0.1, 0.2]},
        history={"train_loss": [0.3]},
        model_params=123,
        train_wall_time_sec=1.5,
        inference_time_s=0.4,
    )
    assert payload["model_label"] == "FFNO + supervised"
    assert payload["training_procedure"] == "supervised"


def test_build_paper_row_payload_uses_neuralop_uno_labels(tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="neuralop_uno",
        training_procedure="supervised",
        epochs=1,
        N=128,
    )
    payload = _build_paper_row_payload(
        cfg,
        metrics={"mae": [0.1, 0.2]},
        history={"train_loss": [0.3]},
        model_params=123,
        train_wall_time_sec=1.5,
        inference_time_s=0.4,
    )
    assert payload["model_label"] == "U-NO + supervised"
    assert payload["architecture_id"] == "neuralop_uno"
    assert payload["training_procedure"] == "supervised"


def test_script_path_execution_bootstraps_repo_imports(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/studies/grid_lines_torch_runner.py",
            "--train-npz",
            str(tmp_path / "missing_train.npz"),
            "--test-npz",
            str(tmp_path / "missing_test.npz"),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "hybrid_resnet",
            "--epochs",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "scripts.studies.invocation_logging" not in result.stderr
    assert (out_dir / "runs" / "pinn_hybrid_resnet" / "invocation.json").exists()


def test_main_defaults_torch_logger_to_csv(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    captured = {"cfg": None}

    def fake_run_grid_lines_torch(cfg, **kwargs):
        captured["cfg"] = cfg
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "fno",
            "--epochs",
            "1",
        ]
    )

    assert captured["cfg"] is not None
    assert captured["cfg"].logger_backend == "csv"


def test_main_hybrid_resnet_defaults_bias_local_branch(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    captured = {"cfg": None}

    def fake_run_grid_lines_torch(cfg, **kwargs):
        captured["cfg"] = cfg
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "hybrid_resnet",
            "--epochs",
            "1",
        ]
    )

    assert captured["cfg"] is not None
    assert captured["cfg"].hybrid_encoder_conv_hidden_scale == 2.0
    assert captured["cfg"].hybrid_encoder_spectral_hidden_scale == 1.0


def test_main_accepts_spectral_resnet_bottleneck_flags(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    captured = {"cfg": None}

    def fake_run_grid_lines_torch(cfg, **kwargs):
        captured["cfg"] = cfg
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "spectral_resnet_bottleneck_net",
            "--epochs",
            "1",
            "--spectral-bottleneck-blocks",
            "8",
            "--spectral-bottleneck-modes",
            "10",
            "--spectral-bottleneck-share-weights",
            "--spectral-bottleneck-gate-init",
            "0.2",
            "--spectral-bottleneck-gate-mode",
            "per_block",
        ]
    )

    assert captured["cfg"] is not None
    assert captured["cfg"].architecture == "spectral_resnet_bottleneck_net"
    assert captured["cfg"].spectral_bottleneck_blocks == 8
    assert captured["cfg"].spectral_bottleneck_modes == 10
    assert captured["cfg"].spectral_bottleneck_share_weights is True
    assert captured["cfg"].spectral_bottleneck_gate_init == 0.2
    assert captured["cfg"].spectral_bottleneck_gate_mode == "per_block"


def test_main_maps_torch_logger_none_to_disabled(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    captured = {"cfg": None}

    def fake_run_grid_lines_torch(cfg, **kwargs):
        captured["cfg"] = cfg
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "fno",
            "--epochs",
            "1",
            "--torch-logger",
            "none",
        ]
    )

    assert captured["cfg"] is not None
    assert captured["cfg"].logger_backend is None


def test_main_does_not_expose_single_image_frc_config(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_torch_runner as runner

    captured = {"cfg": None}

    def fake_run_grid_lines_torch(cfg, **kwargs):
        captured["cfg"] = cfg
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {}}

    monkeypatch.setattr(runner, "run_grid_lines_torch", fake_run_grid_lines_torch)

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    runner.main(
        [
            "--train-npz",
            str(train_npz),
            "--test-npz",
            str(test_npz),
            "--output-dir",
            str(out_dir),
            "--architecture",
            "fno",
            "--epochs",
            "1",
        ]
    )
    assert captured["cfg"] is not None
    assert not hasattr(captured["cfg"], "single_image_frc")
    assert not hasattr(captured["cfg"], "single_image_frc_split_mode")
    assert not hasattr(captured["cfg"], "single_image_frc_rng_seed")


def test_main_rejects_removed_single_image_frc_flag(tmp_path):
    from scripts.studies import grid_lines_torch_runner as runner

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--train-npz",
                str(train_npz),
                "--test-npz",
                str(test_npz),
                "--output-dir",
                str(out_dir),
                "--architecture",
                "fno",
                "--epochs",
                "1",
                "--no-single-image-frc",
            ]
        )


def test_main_rejects_removed_single_image_frc_split_mode(tmp_path):
    from scripts.studies import grid_lines_torch_runner as runner

    out_dir = tmp_path / "output"
    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    train_npz.write_bytes(b"stub")
    test_npz.write_bytes(b"stub")

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--train-npz",
                str(train_npz),
                "--test-npz",
                str(test_npz),
                "--output-dir",
                str(out_dir),
                "--architecture",
                "fno",
                "--epochs",
                "1",
                "--single-image-frc-split-mode",
                "binomial",
            ]
        )

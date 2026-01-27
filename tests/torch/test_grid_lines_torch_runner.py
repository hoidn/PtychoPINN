# tests/torch/test_grid_lines_torch_runner.py
"""Smoke tests for the Torch grid-lines runner."""
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.studies.grid_lines_torch_runner import (
    TorchRunnerConfig,
    load_cached_dataset,
    setup_torch_configs,
    run_grid_lines_torch,
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

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"

    np.savez(train_path, **data)
    np.savez(test_path, **data)

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
                mock_infer.return_value = np.random.rand(1, 64, 64).astype(np.complex64)

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
                mock_infer.return_value = np.random.rand(1, 64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {}

                    result = run_grid_lines_torch(cfg)

        run_dir = Path(result['run_dir'])
        assert run_dir.name == "pinn_hybrid"
        assert run_dir.parent.name == "runs"

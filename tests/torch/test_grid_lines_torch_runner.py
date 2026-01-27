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
            # Channel count should be gridsize squared
            expected_C = gridsize * gridsize
            assert training_config.model.gridsize == gridsize, f"gridsize mismatch for gridsize={gridsize}"
            # Note: TrainingConfig doesn't expose C directly, but downstream
            # consumers derive it from gridsize. This test ensures gridsize is correct.

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

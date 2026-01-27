# tests/test_workflow_generator_integration.py
"""Tests for generator registry integration with workflows."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ptycho.config.config import TrainingConfig, ModelConfig


class TestTFWorkflowGeneratorIntegration:
    """Tests for TensorFlow workflow generator integration."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal TrainingConfig for testing."""
        model_config = ModelConfig(N=64, gridsize=1, architecture='cnn')
        return TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy.npz"),
            n_groups=10,
            nepochs=1,
        )

    def test_train_cdi_model_calls_resolve_generator(self, minimal_config, monkeypatch):
        """Verify train_cdi_model uses generator registry."""
        # Mock the generator
        mock_generator = MagicMock()
        mock_generator.name = 'cnn'
        mock_generator.build_models.return_value = (MagicMock(), MagicMock())

        # Patch resolve_generator
        with patch('ptycho.workflows.components.resolve_generator', return_value=mock_generator) as mock_resolve:
            # Patch train_pinn at the source module (it's imported inside the function)
            with patch('ptycho.train_pinn.train_eval') as mock_train_eval:
                mock_train_eval.return_value = {'history': {}}

                # Patch create_ptycho_data_container
                with patch('ptycho.workflows.components.create_ptycho_data_container'):
                    # Patch probe.set_probe_guess
                    with patch('ptycho.workflows.components.probe'):
                        from ptycho.workflows.components import train_cdi_model
                        from ptycho.raw_data import RawData

                        # Create minimal mock data
                        mock_data = MagicMock(spec=RawData)

                        # Call the function
                        try:
                            train_cdi_model(mock_data, None, minimal_config)
                        except Exception:
                            pass  # May fail due to mocking, but we just want to verify resolve_generator was called

                        # Verify resolve_generator was called with config
                        mock_resolve.assert_called_once_with(minimal_config)


class TestTorchWorkflowGeneratorIntegration:
    """Tests for PyTorch workflow generator integration."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal TrainingConfig for testing."""
        model_config = ModelConfig(N=64, gridsize=1, architecture='cnn')
        return TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy.npz"),
            n_groups=10,
            nepochs=1,
        )

    def test_train_with_lightning_calls_resolve_generator(self, minimal_config, monkeypatch):
        """Verify _train_with_lightning uses generator registry."""
        torch = pytest.importorskip("torch")
        lightning = pytest.importorskip("lightning")

        # Mock the generator
        mock_generator = MagicMock()
        mock_generator.name = 'cnn'
        mock_model = MagicMock()
        mock_model.automatic_optimization = True
        mock_model.val_loss_name = 'val_loss'
        mock_generator.build_model.return_value = mock_model

        # Patch resolve_generator
        with patch('ptycho_torch.workflows.components.resolve_generator', return_value=mock_generator) as mock_resolve:
            # Patch _build_lightning_dataloaders to return mock loaders
            with patch('ptycho_torch.workflows.components._build_lightning_dataloaders') as mock_build_loaders:
                mock_loader = MagicMock()
                mock_build_loaders.return_value = (mock_loader, None)

                # Patch Lightning Trainer in the workflows module where it's used
                with patch.object(lightning.pytorch, 'Trainer') as mock_trainer_class:
                    mock_trainer = MagicMock()
                    mock_trainer.callback_metrics = {}
                    mock_trainer_class.return_value = mock_trainer

                    # Patch config_factory
                    with patch('ptycho_torch.workflows.components.create_training_payload') as mock_factory:
                        mock_payload = MagicMock()
                        mock_payload.pt_data_config = MagicMock()
                        mock_payload.pt_model_config = MagicMock()
                        mock_payload.pt_model_config.mode = 'Unsupervised'
                        mock_payload.pt_model_config.loss_function = 'Poisson'
                        mock_payload.pt_training_config = MagicMock()
                        mock_payload.pt_inference_config = MagicMock()
                        mock_factory.return_value = mock_payload

                        # Create mock container
                        mock_container = MagicMock()
                        mock_container.X = torch.randn(10, 64, 64)

                        from ptycho_torch.workflows.components import _train_with_lightning

                        try:
                            _train_with_lightning(mock_container, None, minimal_config)
                        except Exception:
                            pass  # May fail, but we want to verify resolve_generator was called

                        # Verify resolve_generator was called with config
                        mock_resolve.assert_called_once_with(minimal_config)

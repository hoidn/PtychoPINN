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
            training_groups=10,
            nepochs=1,
        )

    def test_train_cdi_model_calls_resolve_generator(self, minimal_config, monkeypatch):
        """Verify train_cdi_model uses generator registry."""
        # Mock the generator
        mock_generator = MagicMock()
        mock_generator.name = 'cnn'
        mock_generator.build_models.return_value = (MagicMock(), MagicMock())

        # Patch resolve_generator
        with patch('ptycho.generators.registry.resolve_generator', return_value=mock_generator) as mock_resolve:
            # Patch train_pinn at the source module (it's imported inside the function)
            with patch('ptycho.train_pinn.train_eval') as mock_train_eval:
                mock_train_eval.return_value = {'history': {}}

                # Patch create_ptycho_data_container
                with patch('ptycho.workflows.workflow_orchestration.create_ptycho_data_container'):
                    # Patch probe.set_probe_guess
                    with patch('ptycho.workflows.workflow_orchestration.probe'):
                        from ptycho.workflows.components import train_cdi_model
                        from ptycho.raw_data import RawData

                        # Create minimal mock data
                        mock_data = MagicMock(spec=RawData)

                        # Call the function
                        try:
                            train_cdi_model(mock_data, None, minimal_config)
                        except Exception:
                            pass  # May fail due to mocking, but we just want to verify resolve_generator was called

                        mock_resolve.assert_called_once()
                        resolved_config = mock_resolve.call_args.args[0]
                        assert resolved_config.model.architecture == "cnn"

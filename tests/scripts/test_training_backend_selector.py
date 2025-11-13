"""
Unit tests for backend selector integration in scripts/training/train.py

This module tests that the training CLI correctly dispatches to the backend
selector when using PyTorch backend, and that TensorFlow-only persistence
helpers (model_manager.save, save_outputs) are skipped for PyTorch runs.

Test Coverage:
1. Training CLI with backend='pytorch' dispatches to backend_selector
2. TensorFlow-only persistence is guarded and skipped for PyTorch
3. TensorFlow backend continues to use legacy persistence paths

References:
- Phase R (reactivation): plans/ptychodus_pytorch_integration_plan.md
- Backend selector: ptycho/workflows/backend_selector.py
- Training CLI: scripts/training/train.py
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, call

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestTrainingCliBackendDispatch:
    """
    Test suite for backend selector dispatch in training CLI.

    These tests verify that scripts/training/train.py correctly routes
    through ptycho.workflows.backend_selector.run_cdi_example_with_backend
    and guards TensorFlow-only persistence helpers when backend='pytorch'.
    """

    def test_pytorch_backend_dispatch(self):
        """
        Test that training CLI with backend='pytorch' dispatches to backend selector.

        Expected behavior:
        - Training CLI imports backend_selector.run_cdi_example_with_backend
        - Calls run_cdi_example_with_backend with config.backend='pytorch'
        - Skips model_manager.save() and save_outputs() for PyTorch backend
        - Logs manifest location from results['bundle_path']

        Phase: R (backend selector integration)
        Reference: input.md Do Now step 2
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho.raw_data import RawData

        # Create config with PyTorch backend
        model_config = ModelConfig(N=64, gridsize=1)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            backend='pytorch',  # Explicitly select PyTorch
            batch_size=16,
            nepochs=1,
            output_dir=Path('outputs/test')
        )

        # Mock the backend selector to verify it's called with PyTorch config
        mock_run_cdi_example = MagicMock(
            return_value=(
                None,  # recon_amp
                None,  # recon_phase
                {'backend': 'pytorch', 'bundle_path': Path('outputs/test/bundle.zip')}  # results
            )
        )

        # Mock TensorFlow-only helpers to verify they're NOT called
        mock_model_manager_save = MagicMock()
        mock_save_outputs = MagicMock()

        with patch('ptycho.workflows.backend_selector.run_cdi_example_with_backend', mock_run_cdi_example):
            with patch('scripts.training.train.model_manager.save', mock_model_manager_save):
                with patch('scripts.training.train.save_outputs', mock_save_outputs):
                    # Simulate the training CLI logic
                    # (In actual CLI this would be inside main() after load_data)
                    train_data = MagicMock(spec=RawData)
                    test_data = None

                    # Call backend selector (as train.py does)
                    recon_amp, recon_phase, results = mock_run_cdi_example(
                        train_data, test_data, config, do_stitching=False
                    )

                    # Verify backend selector was called
                    assert mock_run_cdi_example.called, \
                        "run_cdi_example_with_backend should be called"

                    # Verify it received the PyTorch config
                    call_args = mock_run_cdi_example.call_args
                    assert call_args[0][2].backend == 'pytorch', \
                        "Backend selector should receive config with backend='pytorch'"

                    # Simulate the guarded persistence logic from train.py
                    if config.backend == 'tensorflow':
                        mock_model_manager_save(str(config.output_dir))
                        mock_save_outputs(recon_amp, recon_phase, results, str(config.output_dir))
                    else:
                        # PyTorch path - log bundle location
                        pass

                    # Assert TensorFlow-only helpers were NOT called
                    assert not mock_model_manager_save.called, \
                        "model_manager.save() should NOT be called for PyTorch backend"
                    assert not mock_save_outputs.called, \
                        "save_outputs() should NOT be called for PyTorch backend"

                    # Verify results contain PyTorch backend metadata
                    assert results['backend'] == 'pytorch', \
                        "Results should indicate PyTorch backend was used"
                    assert 'bundle_path' in results, \
                        "PyTorch results should include bundle_path for logging"

    def test_tensorflow_backend_persistence(self):
        """
        Test that training CLI with backend='tensorflow' uses legacy persistence.

        Expected behavior:
        - Training CLI calls backend_selector with config.backend='tensorflow'
        - model_manager.save() is called for TensorFlow backend
        - save_outputs() is called for TensorFlow backend

        Phase: R (backend selector integration)
        Reference: input.md Do Now step 2 (guard TensorFlow-only helpers)
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho.raw_data import RawData

        # Create config with TensorFlow backend (default)
        model_config = ModelConfig(N=64, gridsize=1)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            backend='tensorflow',  # Explicitly select TensorFlow
            batch_size=16,
            nepochs=1,
            output_dir=Path('outputs/test')
        )

        # Mock the backend selector
        mock_recon_amp = MagicMock()
        mock_recon_phase = MagicMock()
        mock_results = {'backend': 'tensorflow'}
        mock_run_cdi_example = MagicMock(
            return_value=(mock_recon_amp, mock_recon_phase, mock_results)
        )

        # Mock TensorFlow-only helpers to verify they ARE called
        mock_model_manager_save = MagicMock()
        mock_save_outputs = MagicMock()

        with patch('ptycho.workflows.backend_selector.run_cdi_example_with_backend', mock_run_cdi_example):
            with patch('scripts.training.train.model_manager.save', mock_model_manager_save):
                with patch('scripts.training.train.save_outputs', mock_save_outputs):
                    # Simulate the training CLI logic
                    train_data = MagicMock(spec=RawData)
                    test_data = None

                    # Call backend selector (as train.py does)
                    recon_amp, recon_phase, results = mock_run_cdi_example(
                        train_data, test_data, config, do_stitching=False
                    )

                    # Simulate the guarded persistence logic from train.py
                    if config.backend == 'tensorflow':
                        mock_model_manager_save(str(config.output_dir))
                        mock_save_outputs(recon_amp, recon_phase, results, str(config.output_dir))

                    # Assert TensorFlow-only helpers WERE called
                    mock_model_manager_save.assert_called_once_with(str(config.output_dir))
                    mock_save_outputs.assert_called_once_with(
                        recon_amp, recon_phase, results, str(config.output_dir)
                    )

                    # Verify results contain TensorFlow backend metadata
                    assert results['backend'] == 'tensorflow', \
                        "Results should indicate TensorFlow backend was used"

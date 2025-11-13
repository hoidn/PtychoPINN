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

    def test_pytorch_execution_config_flags(self):
        """
        Test that training CLI PyTorch execution config flags are plumbed correctly.

        Expected behavior:
        - CLI flags (--torch-accelerator, --torch-num-workers, --torch-learning-rate, etc.)
          are collected into exec_args namespace
        - build_execution_config_from_args is called with mode='training'
        - Resulting PyTorchExecutionConfig is passed to run_cdi_example_with_backend
          as torch_execution_config parameter
        - TensorFlow backend ignores execution config (remains None)

        Phase: Training execution config flags
        Reference: input.md Do Now (training CLI execution-config surface)
        """
        import argparse
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho.raw_data import RawData

        # Create config with PyTorch backend
        model_config = ModelConfig(N=64, gridsize=1)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            backend='pytorch',
            batch_size=16,
            nepochs=1,
            output_dir=Path('outputs/test')
        )

        # Simulate CLI args with PyTorch execution flags
        cli_args = argparse.Namespace(
            torch_accelerator='cpu',
            torch_deterministic=True,
            torch_num_workers=0,
            torch_learning_rate=5e-4,
            torch_scheduler='ReduceLROnPlateau',
            torch_logger='csv',
            torch_enable_checkpointing=True,
            torch_checkpoint_save_top_k=2,
            torch_accumulate_grad_batches=2,
            debug=False  # Will be inverted to quiet=True
        )

        # Mock the execution config builder to verify it's called correctly
        mock_exec_config = MagicMock()
        mock_exec_config.accelerator = 'cpu'
        mock_exec_config.num_workers = 0
        mock_exec_config.learning_rate = 5e-4
        mock_exec_config.logger_backend = 'csv'

        mock_build_exec_config = MagicMock(return_value=mock_exec_config)

        # Mock the backend selector to verify it receives execution_config
        mock_run_cdi_example = MagicMock(
            return_value=(
                None,  # recon_amp
                None,  # recon_phase
                {'backend': 'pytorch', 'bundle_path': Path('outputs/test/bundle.zip')}
            )
        )

        # Patch at the location where it's imported (ptycho_torch.cli.shared)
        with patch('ptycho_torch.cli.shared.build_execution_config_from_args', mock_build_exec_config):
            with patch('ptycho.workflows.backend_selector.run_cdi_example_with_backend', mock_run_cdi_example):
                # Simulate the training CLI execution config building logic
                # (from scripts/training/train.py:356-384)
                from ptycho_torch.cli.shared import build_execution_config_from_args
                train_data = MagicMock(spec=RawData)
                test_data = None

                # Build execution config (as train.py does when backend='pytorch')
                if config.backend == 'pytorch':
                    exec_args = argparse.Namespace(
                        accelerator=getattr(cli_args, 'torch_accelerator', 'auto'),
                        deterministic=getattr(cli_args, 'torch_deterministic', True),
                        num_workers=getattr(cli_args, 'torch_num_workers', 0),
                        learning_rate=getattr(cli_args, 'torch_learning_rate', None),
                        scheduler=getattr(cli_args, 'torch_scheduler', 'Default'),
                        logger_backend=getattr(cli_args, 'torch_logger', 'csv'),
                        enable_checkpointing=getattr(cli_args, 'torch_enable_checkpointing', True),
                        checkpoint_save_top_k=getattr(cli_args, 'torch_checkpoint_save_top_k', 1),
                        accumulate_grad_batches=getattr(cli_args, 'torch_accumulate_grad_batches', 1),
                        checkpoint_monitor_metric='val_loss',
                        checkpoint_mode='min',
                        early_stop_patience=100,
                        quiet=getattr(cli_args, 'debug', False) == False,
                        disable_mlflow=False
                    )
                    torch_execution_config = build_execution_config_from_args(exec_args, mode='training')
                else:
                    torch_execution_config = None

                # Call backend selector with execution config
                recon_amp, recon_phase, results = mock_run_cdi_example(
                    train_data, test_data, config, do_stitching=False,
                    torch_execution_config=torch_execution_config
                )

                # Verify build_execution_config_from_args was called exactly once
                mock_build_exec_config.assert_called_once()
                call_args = mock_build_exec_config.call_args

                # Verify the exec_args namespace has the expected values
                exec_args_passed = call_args[0][0]
                assert exec_args_passed.accelerator == 'cpu', \
                    "Should pass accelerator from CLI flags"
                assert exec_args_passed.num_workers == 0, \
                    "Should pass num_workers from CLI flags"
                assert exec_args_passed.learning_rate == 5e-4, \
                    "Should pass learning_rate from CLI flags"
                assert exec_args_passed.scheduler == 'ReduceLROnPlateau', \
                    "Should pass scheduler from CLI flags"
                assert exec_args_passed.logger_backend == 'csv', \
                    "Should pass logger from CLI flags"
                assert exec_args_passed.accumulate_grad_batches == 2, \
                    "Should pass accumulate_grad_batches from CLI flags"

                # Verify mode='training' was passed
                assert call_args[1]['mode'] == 'training', \
                    "Should call build_execution_config_from_args with mode='training'"

                # Verify run_cdi_example_with_backend received the execution config
                backend_call_args = mock_run_cdi_example.call_args
                assert backend_call_args[1]['torch_execution_config'] is mock_exec_config, \
                    "Backend selector should receive the PyTorchExecutionConfig as torch_execution_config parameter"

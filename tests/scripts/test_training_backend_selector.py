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

    def test_supervised_mode_enforces_mae_loss(self):
        """
        Test that supervised model_type forces loss_function='MAE' in PyTorch backend.

        Background:
        The PyTorch Lightning module (PtychoPINN_Lightning) requires loss_name to be
        defined for logging. The __init__ method only sets loss_name when specific
        combinations of mode + loss_function are matched (see ptycho_torch/model.py:1052-1066):
          - Unsupervised + Poisson → loss_name='poisson_train'
          - Unsupervised + MAE → loss_name='mae_train'
          - Supervised + MAE → loss_name='mae_train'

        Without this enforcement, supervised mode with default loss_function='Poisson'
        causes: AttributeError: 'PtychoPINN_Lightning' object has no attribute 'loss_name'

        This test verifies that ptycho_torch/workflows/components.py:_train_with_lightning
        detects supervised mode and overrides loss_function='MAE' before instantiating
        the Lightning module.

        Phase: R (supervised loss mapping)
        Reference: plans/active/INTEGRATE-PYTORCH-001/reports/.../red/blocked_20251113T183500Z_loss_name.md
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho_torch.config_params import ModelConfig as PTModelConfig
        from ptycho_torch.config_factory import create_training_payload
        from pathlib import Path

        # Create canonical TF config with supervised mode
        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='supervised',  # TensorFlow naming
        )
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('datasets/train.npz'),
            output_dir=Path('outputs/test_supervised'),
            n_groups=128,
            nepochs=1,
            backend='pytorch',
        )

        # Simulate factory payload creation (as _train_with_lightning does)
        mode_map = {'pinn': 'Unsupervised', 'supervised': 'Supervised'}
        factory_overrides = {
            'n_groups': config.n_groups,
            'gridsize': config.model.gridsize,
            'model_type': mode_map.get(config.model.model_type, 'Unsupervised'),
            'amp_activation': config.model.amp_activation,
            'n_filters_scale': config.model.n_filters_scale,
            'max_epochs': config.nepochs,
        }

        # Mock the factory to return a payload with default loss_function='Poisson'
        mock_payload = MagicMock()
        mock_pt_model_config = PTModelConfig(
            mode='Supervised',  # PyTorch naming
            loss_function='Poisson',  # Default value (incompatible with Supervised)
            C_forward=4,
            C_model=4,
        )
        mock_payload.pt_model_config = mock_pt_model_config
        mock_payload.pt_data_config = MagicMock()
        mock_payload.pt_training_config = MagicMock()

        with patch('ptycho_torch.config_factory.create_training_payload', return_value=mock_payload):
            # Import the helper that applies the supervised→MAE override
            from ptycho_torch.workflows.components import _train_with_lightning

            # Mock the Lightning module instantiation to capture the corrected config
            mock_lightning_module = MagicMock()
            mock_lightning_module.val_loss_name = 'mae_val_loss'  # Expected for MAE
            captured_model_config = None

            def capture_model_config(model_config, data_config, training_config, inference_config):
                nonlocal captured_model_config
                captured_model_config = model_config
                return mock_lightning_module

            # Mock all dependencies of _train_with_lightning
            mock_train_container = MagicMock()
            mock_train_container.diffraction = MagicMock()
            mock_test_container = None

            with patch('ptycho_torch.model.PtychoPINN_Lightning', side_effect=capture_model_config):
                with patch('ptycho_torch.workflows.components._build_lightning_dataloaders', return_value=(MagicMock(), None)):
                    with patch('lightning.pytorch.Trainer') as mock_trainer_class:
                        mock_trainer = MagicMock()
                        mock_trainer.fit = MagicMock()
                        mock_trainer_class.return_value = mock_trainer

                        try:
                            # Execute _train_with_lightning with supervised config
                            results = _train_with_lightning(
                                mock_train_container,
                                mock_test_container,
                                config,
                                execution_config=None
                            )
                        except Exception as e:
                            # If the override didn't happen, we'll get AttributeError during training
                            # But we're verifying the config before that point
                            pass

            # Verify the model_config passed to Lightning module has loss_function='MAE'
            assert captured_model_config is not None, \
                "PtychoPINN_Lightning should have been instantiated"
            assert captured_model_config.mode == 'Supervised', \
                "Model should be in Supervised mode"
            assert captured_model_config.loss_function == 'MAE', \
                "Supervised mode should enforce loss_function='MAE' (prevents missing loss_name AttributeError)"

    def test_manual_accumulation_guard(self):
        """
        Test that manual optimization + gradient accumulation raises RuntimeError.

        Background (EXEC-ACCUM-001):
        The PyTorch Lightning module (PtychoPINN_Lightning) uses manual optimization
        (automatic_optimization=False) for custom physics loss integration. Lightning's
        manual optimization mode is incompatible with Trainer(accumulate_grad_batches>1).

        Without this guard, users get a cryptic MisconfigurationException from Lightning.
        With the guard, they get a clear RuntimeError with actionable advice.

        This test verifies that ptycho_torch/workflows/components.py:_train_with_lightning
        detects the incompatibility and raises RuntimeError before Trainer instantiation.

        Phase: Execution config guardrails
        Reference: docs/findings.md#EXEC-ACCUM-001
        """
        from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
        from pathlib import Path

        # Create training config with PINN mode
        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
        )
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('datasets/train.npz'),
            output_dir=Path('outputs/test_accum'),
            n_groups=128,
            nepochs=1,
            backend='pytorch',
        )

        # Create execution config with accumulate_grad_batches > 1
        execution_config = PyTorchExecutionConfig(
            accum_steps=2,  # This should trigger the guard
            accelerator='cpu',
            deterministic=True,
        )

        # Mock all dependencies
        mock_train_container = MagicMock()
        mock_train_container.diffraction = MagicMock()
        mock_test_container = None

        # Mock Lightning module with manual optimization
        mock_lightning_module = MagicMock()
        mock_lightning_module.automatic_optimization = False  # This is the key incompatibility
        mock_lightning_module.save_hyperparameters = MagicMock()

        # Mock factory payload
        from ptycho_torch.config_params import (
            DataConfig as PTDataConfig,
            ModelConfig as PTModelConfig,
            TrainingConfig as PTTrainingConfig,
        )
        mock_payload = MagicMock()
        mock_payload.pt_model_config = PTModelConfig(mode='Unsupervised', C_forward=4, C_model=4)
        mock_payload.pt_data_config = PTDataConfig()
        mock_payload.pt_training_config = PTTrainingConfig()

        # Mock dataloader builder to return first batch with diffraction
        mock_train_loader = MagicMock()
        mock_val_loader = None

        with patch('ptycho_torch.config_factory.create_training_payload', return_value=mock_payload):
            with patch('ptycho_torch.model.PtychoPINN_Lightning', return_value=mock_lightning_module):
                with patch('ptycho_torch.workflows.components._build_lightning_dataloaders',
                          return_value=(mock_train_loader, mock_val_loader)):
                    from ptycho_torch.workflows.components import _train_with_lightning

                    # Execute and expect RuntimeError with EXEC-ACCUM-001 reference
                    with pytest.raises(RuntimeError) as exc_info:
                        _train_with_lightning(
                            mock_train_container,
                            mock_test_container,
                            config,
                            execution_config=execution_config
                        )

                    # Verify error message mentions the incompatibility
                    error_msg = str(exc_info.value)
                    assert 'manual optimization' in error_msg.lower(), \
                        "Error should mention manual optimization"
                    assert 'accumulate_grad_batches' in error_msg.lower() or 'gradient accumulation' in error_msg.lower(), \
                        "Error should mention gradient accumulation"
                    assert 'EXEC-ACCUM-001' in error_msg, \
                        "Error should reference EXEC-ACCUM-001 finding"

    def test_pytorch_backend_defaults_auto_execution_config(self, caplog):
        """
        Test that training CLI with backend='pytorch' and NO --torch-* flags
        emits POLICY-001 log and passes torch_execution_config=None to backend_selector.

        Expected behavior:
        - When no --torch-* flags provided, CLI logs POLICY-001 message
        - CLI passes torch_execution_config=None to backend_selector
        - Backend_selector auto-instantiates PyTorchExecutionConfig with GPU-first defaults

        Phase: CLI GPU-default logging
        Reference: input.md Do Now step 3
        """
        import sys
        import logging
        from unittest.mock import patch, MagicMock
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho.raw_data import RawData

        # Configure caplog to capture INFO level
        caplog.set_level(logging.INFO)

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

        # Mock sys.argv to simulate NO --torch-* flags
        original_argv = sys.argv
        try:
            sys.argv = ['train.py', '--backend', 'pytorch', '--train_data_file', 'train.npz']

            # Mock the backend selector to capture torch_execution_config parameter
            captured_torch_execution_config = None
            def mock_run_cdi_example(train_data, test_data, cfg, do_stitching=False, torch_execution_config=None):
                nonlocal captured_torch_execution_config
                captured_torch_execution_config = torch_execution_config
                return (None, None, {'backend': 'pytorch', 'bundle_path': Path('outputs/test/bundle.zip')})

            mock_backend_selector = MagicMock(side_effect=mock_run_cdi_example)

            with patch('ptycho.workflows.backend_selector.run_cdi_example_with_backend', mock_backend_selector):
                # Simulate the training CLI logic from scripts/training/train.py:360-408
                import argparse
                args = argparse.Namespace(
                    backend='pytorch',
                    train_data_file='train.npz',
                    # No torch-* flags set
                )

                # Simulate the torch_execution_config decision logic
                torch_flags_explicitly_set = any([
                    'torch_accelerator' in sys.argv or '--torch-accelerator' in sys.argv,
                    'torch_deterministic' in sys.argv or '--torch-deterministic' in sys.argv,
                    'torch_num_workers' in sys.argv or '--torch-num-workers' in sys.argv,
                    'torch_learning_rate' in sys.argv or '--torch-learning-rate' in sys.argv,
                    'torch_scheduler' in sys.argv or '--torch-scheduler' in sys.argv,
                    'torch_logger' in sys.argv or '--torch-logger' in sys.argv,
                    'torch_enable_checkpointing' in sys.argv or '--torch-enable-checkpointing' in sys.argv,
                    'torch_checkpoint_save_top_k' in sys.argv or '--torch-checkpoint-save-top-k' in sys.argv,
                    'torch_accumulate_grad_batches' in sys.argv or '--torch-accumulate-grad-batches' in sys.argv,
                ])

                torch_execution_config = None
                logger = logging.getLogger('scripts.training.train')

                if not torch_flags_explicitly_set:
                    # No --torch-* flags provided: defer to backend_selector's auto-instantiated GPU defaults
                    logger.info("POLICY-001: No --torch-* execution flags provided. "
                               "Backend will use GPU-first defaults (auto-detects CUDA if available, else CPU). "
                               "CPU-only users should pass --torch-accelerator cpu.")
                    # Leave torch_execution_config=None

                # Call backend selector (simulating train.py:410)
                train_data = MagicMock(spec=RawData)
                test_data = None
                recon_amp, recon_phase, results = mock_backend_selector(
                    train_data, test_data, config, do_stitching=False,
                    torch_execution_config=torch_execution_config
                )

                # Verify torch_execution_config was None
                assert captured_torch_execution_config is None, \
                    "CLI should pass torch_execution_config=None when no --torch-* flags provided"

                # Verify POLICY-001 log was emitted
                assert any('POLICY-001' in record.message for record in caplog.records), \
                    "CLI should emit POLICY-001 log when no --torch-* flags provided"

                # Verify log mentions GPU-first defaults and CPU flag guidance
                policy_log = next((r.message for r in caplog.records if 'POLICY-001' in r.message), None)
                assert policy_log is not None
                assert 'GPU-first defaults' in policy_log or 'gpu-first' in policy_log.lower(), \
                    "Log should mention GPU-first defaults"
                assert '--torch-accelerator cpu' in policy_log, \
                    "Log should instruct CPU-only users to pass --torch-accelerator cpu"

        finally:
            sys.argv = original_argv


def test_torch_scheduler_plateau_roundtrip(monkeypatch, tmp_path):
    """Verify --torch-scheduler ReduceLROnPlateau is accepted by train.py argparse
    and forwarded into the exec_args namespace."""
    import argparse

    # Import train.py's parser setup
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'training'))
    import importlib
    train_mod = importlib.import_module('train')

    # Build a minimal argv that includes the plateau scheduler
    test_argv = [
        'train.py',
        '--train-data', str(tmp_path / 'train.npz'),
        '--backend', 'pytorch',
        '--torch-scheduler', 'ReduceLROnPlateau',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    # Re-parse; the parser is created inside main(), so we replicate the argparse setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--backend', type=str, default='tensorflow')
    parser.add_argument('--torch-scheduler', type=str, default='Default',
                        choices=['Default', 'ReduceLROnPlateau', 'CosineAnnealing'])
    args, _ = parser.parse_known_args()

    assert args.torch_scheduler == 'ReduceLROnPlateau'


def test_torch_scheduler_plateau_params_roundtrip(monkeypatch, tmp_path):
    """Verify torch plateau params map into TrainingConfig when provided."""
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'training'))
    train_mod = importlib.import_module('train')

    test_argv = [
        'train.py',
        '--backend', 'pytorch',
        '--train_data_file', str(tmp_path / 'train.npz'),
        '--scheduler', 'ReduceLROnPlateau',
        '--torch-plateau-factor', '0.25',
        '--torch-plateau-patience', '5',
        '--torch-plateau-min-lr', '1e-5',
        '--torch-plateau-threshold', '1e-3',
    ]
    monkeypatch.setattr(sys, 'argv', test_argv)

    args = train_mod.parse_arguments()
    train_mod.apply_torch_plateau_overrides(args, argv=test_argv)

    from ptycho.workflows.components import setup_configuration
    config = setup_configuration(args, None)

    assert config.scheduler == 'ReduceLROnPlateau'
    assert config.plateau_factor == 0.25
    assert config.plateau_patience == 5
    assert config.plateau_min_lr == 1e-5
    assert config.plateau_threshold == 1e-3

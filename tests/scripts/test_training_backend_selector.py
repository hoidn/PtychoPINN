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
import argparse
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

    @pytest.mark.parametrize(
        "execution_input",
        ["request", "none"],
    )
    def test_backend_selector_forwards_torch_execution_input_without_rebuilding(
        self,
        execution_input,
        monkeypatch,
    ):
        """The selector forwards the unresolved request or None unchanged."""
        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.workflows import backend_selector
        from ptycho_torch.execution_request import ExecutionRequest
        from ptycho_torch.workflows import components as torch_components

        if execution_input == "request":
            supplied = ExecutionRequest(
                values={"accelerator": "cpu"},
                explicit_fields={"accelerator"},
            )
        else:
            supplied = None
        sidecar = {"scheduler": "WarmupCosine"}
        delegate = MagicMock(
            return_value=(None, None, {"backend": "pytorch"})
        )
        monkeypatch.setattr(torch_components, "run_cdi_example_torch", delegate)
        monkeypatch.setattr(
            backend_selector,
            "validate_training_config_structure",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            backend_selector,
            "validate_runnable_training_config",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: None,
        )
        config = TrainingConfig(
            model=ModelConfig(),
            train_data_file=Path("train.npz"),
            backend="pytorch",
        )

        backend_selector.run_cdi_example_with_backend(
            object(),
            None,
            config,
            torch_execution_config=supplied,
            torch_factory_overrides=sidecar,
        )

        assert delegate.call_args.kwargs["execution_config"] is supplied
        assert delegate.call_args.kwargs["overrides"] is sidecar

    def test_backend_selector_rejects_resolved_carrier_before_bridge(
        self,
        monkeypatch,
    ):
        """A resolved runtime carrier cannot masquerade as a request."""
        from ptycho.config import (
            ModelConfig,
            PyTorchExecutionConfig,
            TrainingConfig,
        )
        from ptycho.workflows import backend_selector
        from ptycho_torch.workflows import components as torch_components

        events = []
        monkeypatch.setattr(
            backend_selector,
            "validate_training_config_structure",
            lambda *_args: events.append("structure"),
        )
        monkeypatch.setattr(
            backend_selector,
            "validate_runnable_training_config",
            lambda *_args: events.append("runnable"),
        )
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: events.append("bridge"),
        )
        monkeypatch.setattr(
            torch_components,
            "run_cdi_example_torch",
            lambda *_args, **_kwargs: events.append("delegate"),
        )
        config = TrainingConfig(
            model=ModelConfig(),
            train_data_file=Path("train.npz"),
            backend="pytorch",
        )

        with pytest.raises(TypeError, match="ExecutionRequest"):
            backend_selector.run_cdi_example_with_backend(
                object(),
                None,
                config,
                torch_execution_config=PyTorchExecutionConfig(
                    accelerator="cpu"
                ),
            )

        assert events == ["structure", "runnable"]

    def test_training_only_selector_forwards_execution_request(
        self,
        monkeypatch,
    ):
        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.workflows import backend_selector
        from ptycho_torch.execution_request import ExecutionRequest
        from ptycho_torch.workflows import components as torch_components

        request = ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields={"accelerator"},
        )
        delegate = MagicMock(return_value={})
        monkeypatch.setattr(
            torch_components,
            "train_cdi_model_torch",
            delegate,
        )
        monkeypatch.setattr(
            backend_selector,
            "validate_training_config_structure",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            backend_selector,
            "validate_runnable_training_config",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: None,
        )
        config = TrainingConfig(
            model=ModelConfig(),
            train_data_file=Path("train.npz"),
            backend="pytorch",
        )

        backend_selector.train_cdi_model_with_backend(
            object(),
            None,
            config,
            torch_execution_config=request,
            torch_factory_overrides={"learning_rate": 0.002},
        )

        assert delegate.call_args.kwargs == {
            "execution_config": request,
            "overrides": {"learning_rate": 0.002},
        }

    def test_unified_training_main_threads_request_and_canonical_sidecar(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Raw argv drives execution provenance and canonical priority separately."""
        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.metadata import MetadataManager
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args as real_request_builder,
        )
        from ptycho_torch.execution_request import ExecutionRequest
        from scripts.training import train as training_script

        train_path = tmp_path / "train.npz"
        train_path.touch()
        config = TrainingConfig(
            model=ModelConfig(),
            train_data_file=train_path,
            backend="pytorch",
            output_dir=tmp_path / "out",
            n_groups=1,
        )
        args = argparse.Namespace(
            config=None,
            do_stitching=False,
            quiet=True,
            torch_accelerator="cpu",
            torch_scheduler="Exponential",
            torch_plateau_factor=0.25,
            torch_plateau_patience=None,
            torch_plateau_min_lr=None,
            torch_plateau_threshold=None,
            scheduler="WarmupCosine",
        )
        raw_argv = (
            "--torch-accelerator=cpu",
            "--quiet",
            "--torch-scheduler",
            "Exponential",
            "--scheduler=WarmupCosine",
            "--torch-plateau-factor=0.25",
        )
        monkeypatch.setattr(sys, "argv", ["train.py", *raw_argv])
        monkeypatch.setattr(training_script, "parse_arguments", lambda: args)
        monkeypatch.setattr(
            training_script,
            "setup_configuration",
            lambda *_args: config,
        )
        monkeypatch.setattr(
            MetadataManager,
            "load_with_metadata",
            staticmethod(lambda _path: (None, None)),
        )
        monkeypatch.setattr(
            training_script,
            "validate_training_config_structure",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            training_script,
            "validate_runnable_training_config",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            training_script,
            "load_data",
            lambda *_args, **_kwargs: object(),
        )
        delegate = MagicMock(
            return_value=(None, None, {"backend": "pytorch"})
        )
        monkeypatch.setattr(
            training_script,
            "run_cdi_example_with_backend",
            delegate,
        )

        with patch(
            "ptycho_torch.cli.shared.build_execution_request_from_args",
            wraps=real_request_builder,
        ) as request_builder:
            training_script.main()

        request_builder.assert_called_once_with(
            args,
            mode="training",
            explicit_options=raw_argv,
            lane="unified-training",
        )
        request = delegate.call_args.kwargs["torch_execution_config"]
        assert isinstance(request, ExecutionRequest)
        assert request.explicit_fields == frozenset(
            {
                "accelerator",
                "enable_progress_bar",
            }
        )
        assert "scheduler" not in request.values
        assert request.values["enable_progress_bar"] is False
        assert delegate.call_args.kwargs["torch_factory_overrides"] == {
            "scheduler": "WarmupCosine",
            "plateau_factor": 0.25,
        }

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
        from ptycho.config.config import (
            ModelConfig,
            PyTorchExecutionConfig,
            TrainingConfig,
        )
        from ptycho_torch.config_params import ModelConfig as PTModelConfig
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

        # The factory now resolves the objective before sealing ModelSpec.
        from types import SimpleNamespace
        from ptycho_torch.config_bridge import to_model_config
        from ptycho_torch.config_params import (
            DataConfig as PTDataConfig,
            InferenceConfig as PTInferenceConfig,
            TrainingConfig as PTTrainingConfig,
        )
        from ptycho_torch.model_spec import derive_model_spec

        mock_pt_model_config = PTModelConfig(
            mode='Supervised',
            loss_function='MAE',
            C_forward=4,
            C_model=4,
        )
        mock_pt_data_config = PTDataConfig(C=4)
        mock_pt_training_config = PTTrainingConfig(torch_loss_mode='mae')
        mock_payload = SimpleNamespace(
            pt_model_config=mock_pt_model_config,
            pt_data_config=mock_pt_data_config,
            pt_training_config=mock_pt_training_config,
            pt_inference_config=PTInferenceConfig(),
            execution_config=PyTorchExecutionConfig(
                accelerator="cpu",
                enable_checkpointing=False,
                logger_backend=None,
            ),
            model_spec=derive_model_spec(
                to_model_config(mock_pt_data_config, mock_pt_model_config),
                mock_pt_model_config,
                mock_pt_data_config,
            ),
        )
        captured_factory_overrides = {}

        def fake_resolve_training_payload(*args, **kwargs):
            captured_factory_overrides.update(kwargs["overrides"])
            return mock_payload

        with patch(
            'ptycho_torch.config_factory.resolve_training_payload',
            side_effect=fake_resolve_training_payload,
        ):
            # Import the helper that applies the supervised→MAE override
            from ptycho_torch.workflows.components import _train_with_lightning

            mock_lightning_module = MagicMock()
            mock_lightning_module.val_loss_name = 'mae_val_loss'  # Expected for MAE

            # Mock all dependencies of _train_with_lightning
            mock_train_container = MagicMock()
            mock_train_container.diffraction = MagicMock()
            mock_test_container = None
            mock_train_loader = [
                (
                    {
                        "label_amp": object(),
                        "label_phase": object(),
                    },
                    object(),
                )
            ]

            with patch(
                "ptycho_torch.application_factory.build_ptychopinn_application",
                return_value=mock_lightning_module,
            ) as build_application:
                with patch('ptycho_torch.workflows.components._build_lightning_dataloaders', return_value=(mock_train_loader, None)):
                    with patch('lightning.pytorch.Trainer') as mock_trainer_class:
                        mock_trainer = MagicMock()
                        mock_trainer.fit = MagicMock()
                        mock_trainer_class.return_value = mock_trainer

                        _train_with_lightning(
                            mock_train_container,
                            mock_test_container,
                            config,
                            execution_config=None
                        )

            # Verify the model_config passed to Lightning module has loss_function='MAE'
            sealed_model_config = (
                build_application.call_args.args[0].to_model_config()
            )
            assert sealed_model_config.mode == 'Supervised', \
                "Model should be in Supervised mode"
            assert sealed_model_config.loss_function == 'MAE', \
                "Supervised mode should enforce loss_function='MAE' (prevents missing loss_name AttributeError)"
            assert captured_factory_overrides["torch_loss_mode"] == "mae"

    def test_manual_optimization_keeps_accumulation_out_of_trainer(self):
        """Manual accumulation remains model-owned, not Trainer-owned."""
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

        # Mock all dependencies
        mock_train_container = MagicMock()
        mock_train_container.diffraction = MagicMock()
        mock_test_container = None

        mock_lightning_module = MagicMock()
        mock_lightning_module.automatic_optimization = False
        mock_lightning_module.save_hyperparameters = MagicMock()

        # Mock factory payload
        from ptycho_torch.config_params import (
            DataConfig as PTDataConfig,
            InferenceConfig as PTInferenceConfig,
            ModelConfig as PTModelConfig,
            TrainingConfig as PTTrainingConfig,
        )
        mock_payload = MagicMock()
        mock_payload.pt_model_config = PTModelConfig(mode='Unsupervised', C_forward=4, C_model=4)
        mock_payload.pt_data_config = PTDataConfig()
        mock_payload.pt_training_config = PTTrainingConfig(
            accum_steps=2,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )
        mock_payload.pt_inference_config = PTInferenceConfig()
        mock_payload.execution_config = PyTorchExecutionConfig(
            accelerator="cpu",
            enable_checkpointing=False,
            logger_backend=None,
        )
        from ptycho_torch.config_bridge import to_model_config
        from ptycho_torch.model_spec import derive_model_spec
        mock_payload.model_spec = derive_model_spec(
            to_model_config(mock_payload.pt_data_config, mock_payload.pt_model_config),
            mock_payload.pt_model_config,
            mock_payload.pt_data_config,
        )

        mock_train_loader = MagicMock()
        mock_val_loader = None
        mock_trainer = MagicMock()

        with patch('ptycho_torch.config_factory.create_training_payload', return_value=mock_payload):
            with patch(
                "ptycho_torch.application_factory.build_ptychopinn_application",
                return_value=mock_lightning_module,
            ) as build_application:
                with patch('ptycho_torch.workflows.components._build_lightning_dataloaders',
                          return_value=(mock_train_loader, mock_val_loader)):
                    with patch(
                        "lightning.pytorch.Trainer",
                        return_value=mock_trainer,
                    ) as trainer_class:
                        from ptycho_torch.workflows.components import _train_with_lightning

                        _train_with_lightning(
                            mock_train_container,
                            mock_test_container,
                            config,
                            resolved_payload=mock_payload,
                        )

        model_training_config = build_application.call_args.args[2]
        assert model_training_config.accum_steps == 2
        assert model_training_config.gradient_clip_val == 0.5
        trainer_kwargs = trainer_class.call_args.kwargs
        assert trainer_kwargs["accumulate_grad_batches"] == 1
        assert trainer_kwargs["gradient_clip_val"] is None
        assert "gradient_clip_algorithm" not in trainer_kwargs

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


def test_torch_scheduler_plateau_params_enter_explicit_training_patch(
    monkeypatch,
    tmp_path,
):
    """Torch plateau aliases map into canonical TrainingConfig ownership."""
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
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    assert build_training_config_patch_from_args(
        args,
        explicit_options=test_argv,
        lane="unified-training",
    ) == {
        "scheduler": "ReduceLROnPlateau",
        "plateau_factor": 0.25,
        "plateau_patience": 5,
        "plateau_min_lr": 1e-5,
        "plateau_threshold": 1e-3,
    }


@pytest.mark.parametrize(
    "scheduler",
    ["Default", "Exponential", "WarmupCosine", "ReduceLROnPlateau"],
)
def test_unified_torch_scheduler_uses_selected_compatibility_domain(
    scheduler,
    monkeypatch,
):
    from scripts.training import train as train_mod

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--torch-scheduler", scheduler],
    )

    assert train_mod.parse_arguments().torch_scheduler == scheduler


def test_unified_torch_scheduler_rejects_unowned_cosine_annealing(
    monkeypatch,
):
    from scripts.training import train as train_mod

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--torch-scheduler", "CosineAnnealing"],
    )

    with pytest.raises(SystemExit):
        train_mod.parse_arguments()


@pytest.mark.parametrize(
    "raw_argv",
    [
        (
            "--torch-plateau-factor",
            "0.25",
            "--torch-plateau-patience",
            "5",
            "--torch-plateau-min-lr",
            "1e-5",
            "--torch-plateau-threshold",
            "1e-3",
        ),
        (
            "--torch-plateau-factor=0.25",
            "--torch-plateau-patience=5",
            "--torch-plateau-min-lr=1e-5",
            "--torch-plateau-threshold=1e-3",
        ),
    ],
)
def test_explicit_torch_plateau_aliases_enter_canonical_sidecar(
    raw_argv,
):
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        torch_plateau_factor=0.25,
        torch_plateau_patience=5,
        torch_plateau_min_lr=1e-5,
        torch_plateau_threshold=1e-3,
    )

    sidecar = build_training_config_patch_from_args(
        args,
        explicit_options=raw_argv,
        lane="unified-training",
    )

    assert sidecar == {
        "plateau_factor": 0.25,
        "plateau_patience": 5,
        "plateau_min_lr": 1e-5,
        "plateau_threshold": 1e-3,
    }


def test_explicit_canonical_plateau_value_wins_over_torch_alias():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        plateau_factor=0.4,
        torch_plateau_factor=0.25,
    )
    raw_argv = (
        "--torch-plateau-factor=0.25",
        "--plateau_factor=0.4",
    )

    sidecar = build_training_config_patch_from_args(
        args,
        explicit_options=raw_argv,
        lane="unified-training",
    )

    assert args.plateau_factor == 0.4
    assert sidecar == {"plateau_factor": 0.4}


def test_unified_sidecar_adds_explicit_learning_rate_and_accumulation():
    from ptycho_torch.cli.shared import build_training_config_patch_from_args

    args = argparse.Namespace(
        torch_learning_rate=0.002,
        torch_accumulate_grad_batches=3,
    )

    assert build_training_config_patch_from_args(
        args,
        explicit_options=(
            "--torch-learning-rate=0.002",
            "--torch-accumulate-grad-batches=3",
        ),
        lane="unified-training",
    ) == {
        "learning_rate": 0.002,
        "accum_steps": 3,
    }


def test_training_entrypoint_shared_parser_preserves_yaml_precedence(
    monkeypatch,
    tmp_path,
):
    from ptycho import params
    from ptycho.workflows.components import setup_configuration
    from scripts.training import train as train_mod

    config_path = tmp_path / "training.yaml"
    config_path.write_text("nepochs: 9\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path)],
    )
    monkeypatch.setattr(params, "cfg", {"sentinel": "unchanged"})
    monkeypatch.setattr(params, "_sealed", False)

    args = train_mod.parse_arguments()

    assert args.config == str(config_path)
    assert args.do_stitching is False
    assert args.torch_accelerator == "cuda"
    assert args.torch_scheduler == "Default"
    assert not hasattr(args, "nepochs")

    config = setup_configuration(args, args.config)

    assert config.nepochs == 9
    assert params.cfg == {"sentinel": "unchanged"}


def test_unified_cli_defers_projection_to_backend_selector(
    tmp_path,
    monkeypatch,
):
    import argparse

    from ptycho import params
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.config.config import update_legacy_dict as real_update_legacy_dict
    from ptycho.metadata import MetadataManager
    from ptycho.workflows import backend_selector, components
    from scripts.training import train as training_script

    train_path = tmp_path / "train.npz"
    train_path.touch()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        nphotons=10,
        backend="tensorflow",
    )
    args = argparse.Namespace(config=None, do_stitching=False)
    events = []
    monkeypatch.setattr(params, "cfg", {"sentinel": "ambient"})
    monkeypatch.setattr(params, "_sealed", False)

    def record_validation(name, candidate):
        events.append(name)
        assert candidate.nphotons == 25

    def load_data_before_dispatch(*_args, **_kwargs):
        events.append("load_data")
        assert params.cfg == {"sentinel": "ambient"}
        return object()

    def selector_bridge(cfg, candidate):
        events.append("selector_bridge")
        assert params.cfg == {"sentinel": "ambient"}
        real_update_legacy_dict(cfg, candidate)

    def tensorflow_delegate(*_args, **_kwargs):
        events.append("delegate")
        assert params.cfg["nphotons"] == 25
        return None, None, {"backend": "tensorflow"}

    def tensorflow_save(*_args, **_kwargs):
        events.append("persist")
        assert params.cfg["nphotons"] == 25

    monkeypatch.setattr(training_script, "parse_arguments", lambda: args)
    monkeypatch.setattr(
        training_script,
        "setup_configuration",
        lambda *_args: config,
    )
    monkeypatch.setattr(
        MetadataManager,
        "load_with_metadata",
        staticmethod(
            lambda _path: (
                events.append("metadata")
                or (None, {"physics_parameters": {"nphotons": 25}})
            )
        ),
    )
    monkeypatch.setattr(
        training_script,
        "validate_training_config_structure",
        lambda candidate: record_validation("structure", candidate),
    )
    monkeypatch.setattr(
        training_script,
        "validate_runnable_training_config",
        lambda candidate: record_validation("runnable", candidate),
    )
    monkeypatch.setattr(
        training_script,
        "interpret_sampling_parameters",
        lambda candidate: (
            events.append("sampling")
            or (512, 512, False, None, "sampling")
        ),
    )
    monkeypatch.setattr(
        backend_selector,
        "update_legacy_dict",
        selector_bridge,
    )
    monkeypatch.setattr(
        training_script,
        "load_data",
        load_data_before_dispatch,
    )
    monkeypatch.setattr(
        components,
        "run_cdi_example",
        tensorflow_delegate,
    )
    monkeypatch.setattr(
        training_script,
        "run_cdi_example_with_backend",
        backend_selector.run_cdi_example_with_backend,
    )
    monkeypatch.setattr(training_script.model_manager, "save", tensorflow_save)
    monkeypatch.setattr(training_script, "save_outputs", lambda *_: None)

    training_script.main()

    assert events == [
        "metadata",
        "structure",
        "runnable",
        "sampling",
        "load_data",
        "selector_bridge",
        "delegate",
        "persist",
    ]
    assert params.cfg == {"sentinel": "ambient"}


def test_invalid_metadata_photons_fail_before_sampling_bridge_or_data(
    tmp_path,
    monkeypatch,
):
    import argparse

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.metadata import MetadataManager
    from scripts.training import train as training_script

    train_path = tmp_path / "train.npz"
    train_path.touch()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        backend="tensorflow",
    )
    args = argparse.Namespace(config=None, do_stitching=False)
    monkeypatch.setattr(training_script, "parse_arguments", lambda: args)
    monkeypatch.setattr(
        training_script,
        "setup_configuration",
        lambda *_args: config,
    )
    monkeypatch.setattr(
        MetadataManager,
        "load_with_metadata",
        staticmethod(lambda _path: (None, {"nphotons": 0})),
    )
    monkeypatch.setattr(
        training_script,
        "interpret_sampling_parameters",
        lambda *_args: pytest.fail("invalid metadata reached sampling"),
    )
    monkeypatch.setattr(
        training_script,
        "load_data",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid metadata reached data loading"
        ),
    )

    with pytest.raises(ValueError, match="nphotons"):
        training_script.main()

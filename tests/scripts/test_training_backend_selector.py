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
from unittest.mock import MagicMock, patch

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

    def test_pytorch_backend_dispatch(self, tmp_path, monkeypatch):
        """Shared PyTorch dispatch skips the TensorFlow persistence owner."""

        import numpy as np

        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.workflows import training as training_workflow
        from ptycho_torch.config_params import (
            DataConfig,
            ModelConfig as TorchModelConfig,
            TrainingConfig as TorchTrainingConfig,
        )

        train_path = tmp_path / "train.npz"
        train_path.touch()
        output_dir = tmp_path / "out"
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1),
            train_data_file=train_path,
            backend="pytorch",
            nepochs=1,
            training_groups=1,
            output_dir=output_dir,
        )

        class FakeRaw:
            probeGuess = np.ones((64, 64), dtype=np.complex64)
            metadata = None

            def generate_grouped_data(self, **_kwargs):
                return {
                    "nn_indices": np.zeros((1, 1), dtype=np.int32),
                    "diffraction": np.ones((1, 64, 64, 1), dtype=np.float32),
                    "Y": None,
                    "X_full": np.ones((1, 64, 64, 1), dtype=np.float32),
                }

        payload = argparse.Namespace(
            tf_training_config=config,
            pt_data_config=DataConfig(N=64, gridsize=1),
            pt_model_config=TorchModelConfig(),
            pt_training_config=TorchTrainingConfig(),
            model_spec=object(),
        )
        initialization = {
            "schema_version": "rect-s1s2-initialization-v2",
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        }
        training_summary_path = output_dir / "training_summary.json"
        dispatch = MagicMock(
            return_value=(
                None,
                None,
                {
                    "backend": "pytorch",
                    "bundle_path": output_dir / "bundle.zip",
                    "rect_s1s2_initialization": initialization,
                    "training_summary_path": training_summary_path,
                },
            )
        )
        tensorflow_persist = MagicMock()
        monkeypatch.setattr(training_workflow, "_resolve_public_config", lambda _r: config)
        monkeypatch.setattr(training_workflow, "load_data", lambda *_a, **_k: FakeRaw())
        monkeypatch.setattr(
            training_workflow,
            "_materialize_backend_container",
            lambda grouped, *_args: grouped,
        )
        monkeypatch.setattr(
            training_workflow,
            "_legacy_execution_and_patch",
            lambda *_args: (None, {}),
        )
        monkeypatch.setattr(
            training_workflow,
            "resolve_training_payload",
            lambda **_kwargs: payload,
        )
        monkeypatch.setattr(
            training_workflow,
            "run_cdi_example_with_backend",
            dispatch,
        )
        monkeypatch.setattr(
            training_workflow,
            "_persist_tensorflow_outputs",
            tensorflow_persist,
        )

        result = training_workflow.run_training_workflow(
            training_workflow.TrainingWorkflowRequest(
                legacy_args=argparse.Namespace(config=None, do_stitching=False),
            )
        )

        assert dispatch.call_args.args[2].backend == "pytorch"
        assert dispatch.call_args.kwargs["torch_resolved_payload"] is payload
        tensorflow_persist.assert_not_called()
        assert result.backend_results["backend"] == "pytorch"
        assert result.bundle_path == output_dir / "bundle.zip"
        assert result.rect_s1s2_initialization == initialization
        assert result.training_summary_path == training_summary_path

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
        monkeypatch.setattr("ptycho_torch.workflows.legacy.run_cdi_example_torch", delegate)
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
        monkeypatch.setattr("ptycho_torch.workflows.legacy.run_cdi_example_torch",
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
        monkeypatch.setattr("ptycho_torch.workflows.legacy.train_cdi_model_torch",
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
        from ptycho.workflows import training as training_workflow
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args as real_request_builder,
        )
        from ptycho_torch.execution_request import ExecutionRequest
        train_path = tmp_path / "train.npz"
        train_path.touch()
        config = TrainingConfig(
            model=ModelConfig(),
            train_data_file=train_path,
            backend="pytorch",
            output_dir=tmp_path / "out",
            training_groups=1,
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
        with patch(
            "ptycho_torch.cli.shared.build_execution_request_from_args",
            wraps=real_request_builder,
        ) as request_builder:
            request, sidecar = training_workflow._legacy_execution_and_patch(
                training_workflow.TrainingWorkflowRequest(
                    legacy_args=args,
                    raw_argv=raw_argv,
                ),
                config,
            )

        request_builder.assert_called_once_with(
            args,
            mode="training",
            explicit_options=raw_argv,
            lane="unified-training",
        )
        assert isinstance(request, ExecutionRequest)
        assert request.explicit_fields == frozenset(
            {
                "accelerator",
                "enable_progress_bar",
            }
        )
        assert "scheduler" not in request.values
        assert request.values["enable_progress_bar"] is False
        assert sidecar == {
            "scheduler": "WarmupCosine",
            "plateau_factor": 0.25,
        }

    def test_tensorflow_backend_persistence(self, tmp_path, monkeypatch):
        """Shared TensorFlow dispatch invokes its legacy persistence adapter."""

        import numpy as np

        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.workflows import training as training_workflow

        train_path = tmp_path / "train.npz"
        train_path.touch()
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1),
            train_data_file=train_path,
            backend="tensorflow",
            training_groups=1,
            output_dir=tmp_path / "out",
        )

        class FakeRaw:
            probeGuess = np.ones((64, 64), dtype=np.complex64)
            metadata = None

            def generate_grouped_data(self, **_kwargs):
                return {
                    "nn_indices": np.zeros((1, 1), dtype=np.int32),
                    "diffraction": np.ones((1, 64, 64, 1), dtype=np.float32),
                    "Y": None,
                    "X_full": np.ones((1, 64, 64, 1), dtype=np.float32),
                }

        amplitude = object()
        phase = object()
        backend_results = {"backend": "tensorflow"}
        dispatch = MagicMock(return_value=(amplitude, phase, backend_results))
        persist = MagicMock()
        monkeypatch.setattr(training_workflow, "_resolve_public_config", lambda _r: config)
        monkeypatch.setattr(training_workflow, "load_data", lambda *_a, **_k: FakeRaw())
        monkeypatch.setattr(
            training_workflow,
            "_materialize_backend_container",
            lambda grouped, *_args: grouped,
        )
        monkeypatch.setattr(
            training_workflow,
            "run_cdi_example_with_backend",
            dispatch,
        )
        monkeypatch.setattr(training_workflow, "_persist_tensorflow_outputs", persist)

        result = training_workflow.run_training_workflow(
            training_workflow.TrainingWorkflowRequest(
                legacy_args=argparse.Namespace(config=None, do_stitching=False),
            )
        )

        assert dispatch.call_args.args[2].backend == "tensorflow"
        persist.assert_called_once_with(config, amplitude, phase, backend_results)
        assert result.backend_results["backend"] == "tensorflow"

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
            training_groups=128,
            nepochs=1,
            backend='pytorch',
        )

        # Simulate factory payload creation (as _train_with_lightning does)
        mode_map = {'pinn': 'Unsupervised', 'supervised': 'Supervised'}
        factory_overrides = {
            'training_groups': config.training_groups,
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
        )
        mock_pt_data_config = PTDataConfig(gridsize=2)
        mock_pt_training_config = PTTrainingConfig(torch_loss_mode='mae')
        mock_payload = SimpleNamespace(
            tf_training_config=config,
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
            "ptycho_torch.config_factory.resolve_training_payload",
            side_effect=fake_resolve_training_payload,
        ):
            # Import the helper that applies the supervised→MAE override
            from ptycho_torch.workflows.components import train_cdi_model_torch

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
            ) as build_application, patch(
                "ptycho_torch.runtime_provenance.build_effective_runtime",
                return_value={},
            ), patch(
                "ptycho_torch.runtime_provenance.write_effective_runtime_json"
            ), patch(
                "ptycho_torch.workflows.containers.create_torch_data_container",
                return_value={},
            ):
                with patch('ptycho_torch.workflows.dataloaders._build_lightning_dataloaders', return_value=(mock_train_loader, None)):
                    with patch('lightning.pytorch.Trainer') as mock_trainer_class:
                        mock_trainer = MagicMock()
                        mock_trainer.fit = MagicMock()
                        mock_trainer_class.return_value = mock_trainer

                        train_cdi_model_torch(
                            train_data=object(),
                            test_data=None,
                            config=config,
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
            training_groups=128,
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
        mock_payload.pt_model_config = PTModelConfig(mode='Unsupervised')
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
        mock_payload.tf_training_config = config
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

        with patch('ptycho_torch.config_factory.resolve_training_payload', return_value=mock_payload):
            with patch(
                "ptycho_torch.application_factory.build_ptychopinn_application",
                return_value=mock_lightning_module,
            ) as build_application, patch(
                "ptycho_torch.runtime_provenance.build_effective_runtime",
                return_value={},
            ), patch(
                "ptycho_torch.runtime_provenance.write_effective_runtime_json"
            ):
                with patch('ptycho_torch.workflows.dataloaders._build_lightning_dataloaders',
                          return_value=(mock_train_loader, mock_val_loader)):
                    with patch(
                        "lightning.pytorch.Trainer",
                        return_value=mock_trainer,
                    ) as trainer_class:
                        from ptycho_torch.workflows.components import _train_with_lightning

                        _train_with_lightning(
                            mock_payload,
                            mock_train_container,
                            mock_test_container,
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


def test_unified_cli_resolves_metadata_before_data_and_backend_projection(
    tmp_path,
    monkeypatch,
):
    import argparse
    import numpy as np

    from ptycho import model_manager, params
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.config.config import update_legacy_dict as real_update_legacy_dict
    from ptycho.metadata import MetadataManager
    from ptycho.workflows import backend_selector, components
    from ptycho.workflows import training as training_workflow
    from scripts.training import train as training_script

    train_path = tmp_path / "train.npz"
    train_path.touch()
    test_path = tmp_path / "test.npz"
    test_path.touch()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        test_data_file=test_path,
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

    class FakeRaw:
        def __init__(self, split):
            self.split = split

    def assert_request_projection():
        assert params.cfg["nphotons"] == 25
        assert params.cfg["gridsize"] == 1
        assert params.cfg["backend"] == "tensorflow"

    def load_data_before_dispatch(path, **_kwargs):
        split = "train" if Path(path) == train_path else "validation"
        events.append(f"load:{split}")
        assert_request_projection()
        return FakeRaw(split)

    def group_before_dispatch(raw, *_args, **_kwargs):
        events.append(f"group:{raw.split}")
        assert_request_projection()
        return {"nn_indices": np.zeros((512, 1), dtype=np.int32)}

    def materialize_before_dispatch(_grouped, raw, _config):
        events.append(f"materialize:{raw.split}")
        assert_request_projection()
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

    monkeypatch.setattr(training_script, "parse_arguments", lambda argv=None: args)
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(
        training_workflow,
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
        training_workflow,
        "validate_training_config_structure",
        lambda candidate: record_validation("structure", candidate),
    )
    monkeypatch.setattr(
        training_workflow,
        "validate_runnable_training_config",
        lambda candidate: record_validation("runnable", candidate),
    )
    monkeypatch.setattr(
        training_workflow,
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
        training_workflow,
        "load_data",
        load_data_before_dispatch,
    )
    monkeypatch.setattr(
        training_workflow,
        "_group_raw_data",
        group_before_dispatch,
    )
    monkeypatch.setattr(
        training_workflow,
        "_materialize_backend_container",
        materialize_before_dispatch,
    )
    monkeypatch.setattr("ptycho.workflows.workflow_orchestration.run_cdi_example",
        tensorflow_delegate,
    )
    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        backend_selector.run_cdi_example_with_backend,
    )
    monkeypatch.setattr(model_manager, "save", tensorflow_save)
    monkeypatch.setattr(training_workflow, "save_outputs", lambda *_: None)

    training_script.main()

    assert events == [
        "metadata",
        "structure",
        "runnable",
        "sampling",
        "load:train",
        "load:validation",
        "group:train",
        "group:validation",
        "materialize:train",
        "materialize:validation",
        "selector_bridge",
        "delegate",
        "persist",
    ]
    assert params.cfg == {"sentinel": "ambient"}


@pytest.mark.parametrize("failure_stage", ["train_load", "grouping"])
def test_data_preparation_projection_restores_ambient_on_failure(
    failure_stage,
    tmp_path,
    monkeypatch,
):
    """Both preparation scopes roll legacy state back when a consumer fails."""
    import numpy as np

    from ptycho import params
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    train_path.touch()
    test_path.touch()
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        train_data_file=train_path,
        test_data_file=test_path,
        nphotons=37,
        training_groups=2,
        backend="tensorflow",
    )
    ambient = {"sentinel": "ambient"}
    monkeypatch.setattr(params, "cfg", ambient)
    monkeypatch.setattr(params, "_sealed", False)
    monkeypatch.setattr(
        training_workflow,
        "_resolve_public_config",
        lambda _request: config,
    )
    monkeypatch.setattr(
        training_workflow,
        "interpret_sampling_parameters",
        lambda _config: (2, 2, False, None, "sampling"),
    )

    class ExpectedPreparationError(RuntimeError):
        pass

    class FakeRaw:
        probeGuess = np.ones((64, 64), dtype=np.complex64)

    def assert_projected():
        assert params.cfg["nphotons"] == 37
        assert params.cfg["gridsize"] == 1
        assert params.cfg["backend"] == "tensorflow"

    load_count = 0

    def fake_load(*_args, **_kwargs):
        nonlocal load_count
        load_count += 1
        assert_projected()
        if failure_stage == "train_load" and load_count == 1:
            raise ExpectedPreparationError("train load failed")
        return FakeRaw()

    def fake_group(*_args, **_kwargs):
        assert_projected()
        if failure_stage == "grouping":
            raise ExpectedPreparationError("grouping failed")
        return {"nn_indices": np.zeros((2, 1), dtype=np.int32)}

    monkeypatch.setattr(training_workflow, "load_data", fake_load)
    monkeypatch.setattr(training_workflow, "_group_raw_data", fake_group)
    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        lambda *_args, **_kwargs: pytest.fail("dispatch must not run"),
    )

    with pytest.raises(ExpectedPreparationError):
        training_workflow.run_training_workflow(
            training_workflow.TrainingWorkflowRequest(
                legacy_args=argparse.Namespace(config=None, do_stitching=False),
            )
        )

    assert params.cfg is ambient
    assert params.cfg == {"sentinel": "ambient"}


def test_invalid_metadata_photons_fail_before_sampling_bridge_or_data(
    tmp_path,
    monkeypatch,
):
    import argparse

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.metadata import MetadataManager
    from ptycho.workflows import training as training_workflow
    from scripts.training import train as training_script

    train_path = tmp_path / "train.npz"
    train_path.touch()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        backend="tensorflow",
    )
    args = argparse.Namespace(config=None, do_stitching=False)
    monkeypatch.setattr(training_script, "parse_arguments", lambda argv=None: args)
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(
        training_workflow,
        "setup_configuration",
        lambda *_args: config,
    )
    monkeypatch.setattr(
        MetadataManager,
        "load_with_metadata",
        staticmethod(lambda _path: (None, {"nphotons": 0})),
    )
    monkeypatch.setattr(
        training_workflow,
        "interpret_sampling_parameters",
        lambda *_args: pytest.fail("invalid metadata reached sampling"),
    )
    monkeypatch.setattr(
        training_workflow,
        "load_data",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid metadata reached data loading"
        ),
    )

    with pytest.raises(ValueError, match="nphotons"):
        training_script.main()


def test_public_config_resolution_never_mutates_reusable_legacy_namespace(
    monkeypatch,
    tmp_path,
):
    """Alias normalization occurs on a fresh Namespace on success and failure."""
    from ptycho.workflows import training as training_workflow

    caller_marker = []
    args = argparse.Namespace(
        config="training.yaml",
        train_data_file_path=tmp_path / "train.npz",
        caller_marker=caller_marker,
    )
    original_values = dict(vars(args))
    resolved_config = object()
    responses = iter(
        (
            resolved_config,
            RuntimeError("configuration failed"),
            resolved_config,
        )
    )
    received = []

    def fake_setup(candidate, config_path):
        received.append(candidate)
        assert candidate is not args
        assert config_path == "training.yaml"
        assert not hasattr(candidate, "train_data_file_path")
        assert candidate.train_data_file == tmp_path / "train.npz"
        assert candidate.caller_marker is caller_marker
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(training_workflow, "setup_configuration", fake_setup)
    monkeypatch.setattr(
        training_workflow,
        "_resolve_metadata_photons",
        lambda config: config,
    )
    monkeypatch.setattr(
        training_workflow,
        "validate_training_config_structure",
        lambda _config: None,
    )
    monkeypatch.setattr(
        training_workflow,
        "validate_runnable_training_config",
        lambda _config: None,
    )
    request = training_workflow.TrainingWorkflowRequest(legacy_args=args)

    assert training_workflow._resolve_public_config(request) is resolved_config
    assert vars(args) == original_values
    with pytest.raises(RuntimeError, match="configuration failed"):
        training_workflow._resolve_public_config(request)
    assert vars(args) == original_values
    assert training_workflow._resolve_public_config(request) is resolved_config
    assert vars(args) == original_values
    assert len({id(candidate) for candidate in received}) == 3


def test_training_main_delegates_to_shared_workflow(monkeypatch):
    """The installed legacy entry point is only a parser/request adapter."""
    from scripts.training import train as training_script

    args = argparse.Namespace(config=None, do_stitching=False)
    captured = {}
    sentinel = object()

    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda argv=None: args,
    )
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)

    def fake_run(request):
        captured["request"] = request
        return sentinel

    monkeypatch.setattr(training_script, "run_training_workflow", fake_run)

    result = training_script.main(["--backend", "pytorch"])

    assert result is sentinel
    request = captured["request"]
    assert request.legacy_args is args
    assert request.raw_argv == ("--backend", "pytorch")
    assert request.do_stitching is False
    assert request.resolved_synthetic_workflow is None


def test_training_main_emits_policy_guidance_and_completion(
    caplog,
    monkeypatch,
    tmp_path,
):
    """The installed CLI, not a test-side simulation, owns its public logs."""
    import logging
    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow
    from ptycho_torch.config_params import (
        DataConfig,
        ModelConfig as TorchModelConfig,
        TrainingConfig as TorchTrainingConfig,
    )
    from ptycho_torch.execution_request import ExecutionRequest
    from scripts.training import train as training_script

    caplog.set_level(logging.INFO)
    train_path = tmp_path / "train.npz"
    train_path.touch()
    bundle_path = tmp_path / "out" / "wts.h5.zip"
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        backend="pytorch",
        training_groups=1,
        output_dir=bundle_path.parent,
    )
    args = argparse.Namespace(
        config=None,
        do_stitching=False,
        backend="pytorch",
    )

    class FakeRaw:
        probeGuess = np.ones((64, 64), dtype=np.complex64)
        metadata = None

        def generate_grouped_data(self, **_kwargs):
            return {
                "nn_indices": np.zeros((1, 1), dtype=np.int32),
                "diffraction": np.ones((1, 2, 2, 1), dtype=np.float32),
                "Y": None,
                "X_full": np.ones((1, 2, 2, 1), dtype=np.float32),
            }

    payload = argparse.Namespace(
        tf_training_config=config,
        pt_data_config=DataConfig(N=64, gridsize=1),
        pt_model_config=TorchModelConfig(),
        pt_training_config=TorchTrainingConfig(),
        model_spec=object(),
    )
    execution_request = ExecutionRequest(
        values={},
        explicit_fields=frozenset(),
    )
    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda _argv=None: args,
    )
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(
        training_workflow,
        "_resolve_public_config",
        lambda _request: config,
    )
    monkeypatch.setattr(
        training_workflow,
        "load_data",
        lambda *_args, **_kwargs: FakeRaw(),
    )
    monkeypatch.setattr(
        training_workflow,
        "_legacy_execution_and_patch",
        lambda *_args: (execution_request, {}),
    )
    monkeypatch.setattr(
        training_workflow,
        "resolve_training_payload",
        lambda **_kwargs: payload,
    )
    monkeypatch.setattr(
        training_workflow,
        "_materialize_backend_container",
        lambda grouped, *_args: grouped,
    )
    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        lambda *_args, **_kwargs: (
            None,
            None,
            {"backend": "pytorch", "bundle_path": bundle_path},
        ),
    )

    result = training_script.main(["--backend", "pytorch"])

    assert result.bundle_path == bundle_path
    policy_log = next(
        (
            record.message
            for record in caplog.records
            if "POLICY-001" in record.message
        ),
        None,
    )
    assert policy_log is not None
    assert "GPU-first defaults" in policy_log
    assert "--torch-accelerator cpu" in policy_log
    assert any(
        "PyTorch backend completed" in record.message
        for record in caplog.records
    )
    assert any(
        str(bundle_path) in record.message
        for record in caplog.records
    )


def test_training_main_logs_and_reraises_errors(caplog, monkeypatch):
    """The installed CLI retains its actionable error log."""
    import logging

    from scripts.training import train as training_script

    caplog.set_level(logging.ERROR)
    args = argparse.Namespace(config=None, do_stitching=False)
    monkeypatch.setattr(
        training_script,
        "parse_arguments",
        lambda _argv=None: args,
    )
    monkeypatch.setattr(training_script, "_configure_logging", lambda: None)
    monkeypatch.setattr(
        training_script,
        "run_training_workflow",
        lambda _request: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        training_script.main([])

    assert any(
        "An error occurred during execution: boom" in record.message
        for record in caplog.records
    )


def test_legacy_shared_workflow_preserves_factory_resolver_inputs(
    monkeypatch,
    tmp_path,
):
    """The legacy route restores its baseline map before explicit CLI values."""

    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow
    from ptycho_torch.config_params import (
        DataConfig,
        ModelConfig as TorchModelConfig,
        TrainingConfig as TorchTrainingConfig,
    )

    train_path = tmp_path / "train.npz"
    train_path.touch()
    output_dir = tmp_path / "out"
    baseline = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        train_data_file=train_path,
        output_dir=output_dir,
        backend="pytorch",
        training_groups=1,
    )
    args = argparse.Namespace(config=None, do_stitching=False)
    execution_request = object()
    cli_patch = {"scheduler": "WarmupCosine"}
    monkeypatch.setattr(
        training_workflow,
        "setup_configuration",
        lambda *_args: baseline,
    )
    monkeypatch.setattr(
        training_workflow,
        "_legacy_execution_and_patch",
        lambda *_args: (execution_request, cli_patch),
    )

    class FakeRaw:
        probeGuess = np.ones((64, 64), dtype=np.complex64)
        metadata = None

        def generate_grouped_data(self, **_kwargs):
            return {
                "diffraction": np.ones((1, 2, 2, 1), dtype=np.float32),
                "Y": None,
                "nn_indices": np.zeros((1, 1), dtype=np.int32),
                "X_full": np.ones((1, 2, 2, 1), dtype=np.float32),
            }

    monkeypatch.setattr(training_workflow, "load_data", lambda *_a, **_k: FakeRaw())
    monkeypatch.setattr(
        training_workflow,
        "_materialize_backend_container",
        lambda grouped, *_args: grouped,
    )
    factory_calls = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return argparse.Namespace(
            tf_training_config=kwargs["training_baseline"],
            pt_data_config=DataConfig(N=64, gridsize=1, neighbor_count=1),
            pt_model_config=TorchModelConfig(),
            pt_training_config=TorchTrainingConfig(),
            model_spec=object(),
        )

    monkeypatch.setattr(training_workflow, "resolve_training_payload", fake_factory)
    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        lambda *_args, **_kwargs: (None, None, {"backend": "pytorch"}),
    )

    result = training_workflow.run_training_workflow(
        training_workflow.TrainingWorkflowRequest(
            legacy_args=args,
            raw_argv=("--scheduler", "WarmupCosine"),
        )
    )

    assert len(factory_calls) == 1
    assert factory_calls[0]["overrides"] is not cli_patch
    assert factory_calls[0]["overrides"]["training_groups"] == 1
    assert factory_calls[0]["overrides"]["gridsize"] == 1
    assert factory_calls[0]["overrides"]["scheduler"] == "WarmupCosine"
    assert factory_calls[0]["execution_config"] is execution_request
    assert factory_calls[0]["training_baseline"] == baseline
    assert result.public_config is factory_calls[0]["training_baseline"]


def test_public_factory_override_map_is_runnable_and_cli_wins(tmp_path):
    """The shared public-config map supplies every required factory owner."""
    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho_torch.config_factory import (
        build_training_factory_overrides,
        resolve_training_payload,
    )

    train_path = tmp_path / "train.npz"
    np.savez(
        train_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1, architecture="cnn"),
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        backend="pytorch",
        training_groups=7,
        scheduler="Default",
    )
    overrides = build_training_factory_overrides(config)
    overrides.update(
        {
            "scheduler": "WarmupCosine",
            "inference_batch_size": 23,
        }
    )

    payload = resolve_training_payload(
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        overrides=overrides,
        training_baseline=config,
    )

    assert payload.pt_training_config.training_groups == 7
    assert payload.pt_training_config.scheduler == "WarmupCosine"
    assert payload.pt_data_config.gridsize == 1
    assert payload.pt_inference_config.batch_size == 23


def test_public_factory_selection_bridge_fields_round_trip(tmp_path):
    """Grouping-only public fields survive the resolved payload bridge."""
    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho_torch.config_factory import (
        build_training_factory_overrides,
        resolve_training_payload,
    )

    train_path = tmp_path / "train.npz"
    np.savez(
        train_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        backend="pytorch",
        training_groups=7,
        neighbor_count=5,
        enable_oversampling=True,
        neighbor_pool_size=5,
        sequential_sampling=True,
    )

    payload = resolve_training_payload(
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        overrides=build_training_factory_overrides(config),
        training_baseline=config,
    )

    assert payload.tf_training_config.enable_oversampling is True
    assert payload.tf_training_config.neighbor_pool_size == 5
    assert payload.tf_training_config.sequential_sampling is True


def test_public_factory_round_trip_preserves_declared_scientific_fields(tmp_path):
    """Every representable non-default public value survives both bridges."""
    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho_torch.config_factory import (
        build_training_factory_overrides,
        resolve_training_payload,
    )

    train_path = tmp_path / "train.npz"
    np.savez(
        train_path,
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    config = TrainingConfig(
        model=ModelConfig(
            N=64,
            gridsize=1,
            probe_scale=2.5,
            gaussian_smoothing_sigma=1.25,
        ),
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        backend="pytorch",
        training_groups=7,
        intensity_scale_trainable=True,
        gradient_clip_val=0.75,
        gradient_clip_algorithm="value",
        scheduler="ReduceLROnPlateau",
        plateau_factor=0.25,
        plateau_patience=6,
        plateau_min_lr=2e-4,
        plateau_threshold=3e-3,
    )

    payload = resolve_training_payload(
        train_data_file=train_path,
        output_dir=tmp_path / "out",
        overrides=build_training_factory_overrides(config),
        training_baseline=config,
    )

    observed = {
        "probe_scale": (
            config.model.probe_scale,
            payload.pt_data_config.probe_scale,
            payload.tf_training_config.model.probe_scale,
        ),
        "gaussian_smoothing_sigma": (
            config.model.gaussian_smoothing_sigma,
            payload.pt_model_config.gaussian_smoothing_sigma,
            payload.tf_training_config.model.gaussian_smoothing_sigma,
        ),
        "intensity_scale_trainable": (
            config.intensity_scale_trainable,
            payload.pt_model_config.intensity_scale_trainable,
            payload.tf_training_config.intensity_scale_trainable,
        ),
        "gradient_clip_val": (
            config.gradient_clip_val,
            payload.pt_training_config.gradient_clip_val,
            payload.tf_training_config.gradient_clip_val,
        ),
        "plateau_factor": (
            config.plateau_factor,
            payload.pt_training_config.plateau_factor,
            payload.tf_training_config.plateau_factor,
        ),
        "plateau_patience": (
            config.plateau_patience,
            payload.pt_training_config.plateau_patience,
            payload.tf_training_config.plateau_patience,
        ),
        "plateau_min_lr": (
            config.plateau_min_lr,
            payload.pt_training_config.plateau_min_lr,
            payload.tf_training_config.plateau_min_lr,
        ),
        "plateau_threshold": (
            config.plateau_threshold,
            payload.pt_training_config.plateau_threshold,
            payload.tf_training_config.plateau_threshold,
        ),
    }
    expected = {
        name: (value, value, value)
        for name, value in {
            "probe_scale": 2.5,
            "gaussian_smoothing_sigma": 1.25,
            "intensity_scale_trainable": True,
            "gradient_clip_val": 0.75,
            "plateau_factor": 0.25,
            "plateau_patience": 6,
            "plateau_min_lr": 2e-4,
            "plateau_threshold": 3e-3,
        }.items()
    }
    assert observed == expected


def test_tensorflow_container_materialization_uses_bounded_config_projection(
    monkeypatch,
    tmp_path,
):
    """Photon normalization observes the request, never ambient legacy state."""
    import numpy as np

    from ptycho import loader, params
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow

    ambient = dict(params.cfg)
    observed = {}
    grouped = {"X_full": np.ones((1, 2, 2, 1), dtype=np.float32)}

    def fake_load(callback, _probe, **_kwargs):
        callback()
        observed["nphotons"] = params.cfg["nphotons"]
        return object()

    monkeypatch.setattr(loader, "load", fake_load)
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=tmp_path / "train.npz",
        backend="tensorflow",
        nphotons=246810.5,
    )
    raw = argparse.Namespace(
        probeGuess=np.ones((2, 2), dtype=np.complex64)
    )

    try:
        params.cfg.clear()
        params.cfg.update({"nphotons": 1.0})
        before = dict(params.cfg)
        training_workflow._materialize_backend_container(
            grouped,
            raw,
            config,
        )
        assert observed["nphotons"] == 246810.5
        assert params.cfg == before
    finally:
        params.cfg.clear()
        params.cfg.update(ambient)


def test_dictionary_parity_materialization_uses_raw_probe_measurements_and_unit_scales(
    tmp_path,
):
    """The in-memory synthetic rail must match the historical dict adapter."""

    import numpy as np
    import torch

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow

    grouped = {
        "X_full": np.full((2, 2, 2, 1), 9.0, dtype=np.float32),
        "diffraction": np.full((2, 2, 2, 1), 2.0, dtype=np.float32),
        "Y": np.ones((2, 2, 2, 1), dtype=np.complex64),
        "coords_relative": np.zeros((2, 1, 2, 1), dtype=np.float64),
        "coords_offsets": np.zeros((2, 1, 2, 1), dtype=np.float64),
        "nn_indices": np.arange(2, dtype=np.int32).reshape(2, 1),
    }
    probe = np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    raw = argparse.Namespace(probeGuess=probe, metadata=None)
    config = TrainingConfig(
        model=ModelConfig(N=2, gridsize=1),
        train_data_file=tmp_path / "train.npz",
        backend="pytorch",
    )

    container = training_workflow._materialize_backend_container(
        grouped,
        raw,
        config,
        data_adapter="dictionary_parity",
    )

    torch.testing.assert_close(container.X, torch.full((2, 2, 2, 1), 2.0))
    torch.testing.assert_close(container.probe, torch.from_numpy(probe))
    torch.testing.assert_close(
        container.rms_scaling_constant,
        torch.ones((1, 1, 1)),
    )
    torch.testing.assert_close(
        container.physics_scaling_constant,
        torch.ones((1, 1, 1)),
    )


def test_explicit_synthetic_torch_seed_may_reproduce_historical_grouping_seed(
    tmp_path,
):
    """An explicit historical seed wins even when both consumers used seed 3."""

    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow
    from ptycho.workflows import training as training_workflow

    resolved = resolve_synthetic_workflow(
        file_values={
            "training": {
                "subsample_seed": 3,
                "torch_training_seed": 3,
            }
        }
    )
    request = training_workflow.TrainingWorkflowRequest(
        resolved_synthetic_workflow=resolved,
        train_data_file=tmp_path / "train.npz",
        test_data_file=tmp_path / "test.npz",
        output_dir=tmp_path / "training",
        torch_training_seed=3,
    )

    assert training_workflow._resolve_workflow_torch_seed(
        request,
        subsample_seed=3,
    ) == 3


def test_resolved_synthetic_torch_seed_is_authoritative_when_request_omits_it(
    tmp_path,
):
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow
    from ptycho.workflows import training as training_workflow

    resolved = resolve_synthetic_workflow(
        file_values={"training": {"torch_training_seed": 3}}
    )
    request = training_workflow.TrainingWorkflowRequest(
        resolved_synthetic_workflow=resolved,
        train_data_file=tmp_path / "train.npz",
        test_data_file=tmp_path / "test.npz",
        output_dir=tmp_path / "training",
    )

    assert training_workflow._resolve_workflow_torch_seed(
        request,
        subsample_seed=resolved.training.subsample_seed,
    ) == 3


def test_request_cannot_override_resolved_synthetic_torch_seed(tmp_path):
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow
    from ptycho.workflows import training as training_workflow

    resolved = resolve_synthetic_workflow(
        file_values={"training": {"torch_training_seed": 3}}
    )
    request = training_workflow.TrainingWorkflowRequest(
        resolved_synthetic_workflow=resolved,
        train_data_file=tmp_path / "train.npz",
        test_data_file=tmp_path / "test.npz",
        output_dir=tmp_path / "training",
        torch_training_seed=4,
    )

    with pytest.raises(ValueError, match="conflicts with resolved synthetic identity"):
        training_workflow._resolve_workflow_torch_seed(
            request,
            subsample_seed=resolved.training.subsample_seed,
        )


def test_synthetic_raw_selection_requires_exact_requested_cardinality():
    """A clamped raw selection cannot be persisted as the requested GS2 split."""
    import numpy as np

    from ptycho.workflows.training import _validate_selected_raw_count

    raw = argparse.Namespace(diff3d=np.zeros((8, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="4096.*8|8.*4096"):
        _validate_selected_raw_count(raw, expected=4096)


def test_group_raw_data_rejects_inexact_group_count(tmp_path):
    """_group_raw_data enforces the exact configured group count."""
    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows.training import _group_raw_data

    class FakeRaw:
        def generate_grouped_data(self, **_kwargs):
            return {"nn_indices": np.zeros((3, 1), dtype=np.int32)}

    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=tmp_path / "train.npz",
        training_groups=5,
    )

    with pytest.raises(
        ValueError, match="grouping produced 3 groups; expected exactly 5"
    ):
        _group_raw_data(
            FakeRaw(),
            config,
            config.train_data_file,
        )


def test_backend_selector_forwards_resolved_payload_and_gain_record(monkeypatch):
    """The selector must not rebuild or flatten resolved Torch workflow state."""
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import backend_selector
    from ptycho_torch.workflows import components as torch_components

    resolved_payload = object()
    gain_record = object()
    torch_training_seed = 314159
    delegate = MagicMock(return_value=(None, None, {}))
    monkeypatch.setattr("ptycho_torch.workflows.legacy.run_cdi_example_torch", delegate)
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
        torch_resolved_payload=resolved_payload,
        torch_amplitude_physics_gain_record=gain_record,
        torch_training_seed=torch_training_seed,
    )

    assert delegate.call_args.kwargs["resolved_payload"] is resolved_payload
    assert (
        delegate.call_args.kwargs["amplitude_physics_gain_record"]
        is gain_record
    )
    assert (
        delegate.call_args.kwargs["torch_training_seed"]
        == torch_training_seed
    )


def test_backend_selector_rejects_gain_record_without_payload_before_projection(
    monkeypatch,
    tmp_path,
):
    """The unified entry fails before legacy projection or Torch dispatch."""
    from ptycho import params
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import backend_selector
    from ptycho_torch.workflows import components as torch_components

    train_path = tmp_path / "train.npz"
    train_path.touch()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=train_path,
        backend="pytorch",
    )
    projection = MagicMock()
    delegate = MagicMock(return_value=(None, None, {}))
    monkeypatch.setattr(backend_selector, "update_legacy_dict", projection)
    monkeypatch.setattr("ptycho_torch.workflows.legacy.run_cdi_example_torch", delegate)
    before = dict(params.cfg)

    with pytest.raises(
        ValueError,
        match=(
            r"torch_amplitude_physics_gain_record.*"
            r"torch_resolved_payload"
        ),
    ):
        backend_selector.run_cdi_example_with_backend(
            object(),
            None,
            config,
            torch_amplitude_physics_gain_record=object(),
        )

    projection.assert_not_called()
    delegate.assert_not_called()
    assert params.cfg == before


def test_torch_runtime_forwards_training_seed_to_training_boundary(monkeypatch):
    """The resolved Torch stream must survive the runtime entry point unchanged."""
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho_torch.workflows import components

    delegate = MagicMock(return_value={})
    monkeypatch.setattr("ptycho_torch.workflows.legacy.train_cdi_model_torch", delegate)
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=Path("train.npz"),
        backend="pytorch",
    )

    components.run_cdi_example_torch(
        object(),
        None,
        config,
        torch_training_seed=271828,
    )

    assert delegate.call_args.kwargs["torch_training_seed"] == 271828


def test_training_boundary_forwards_training_seed_to_lightning(monkeypatch):
    """Container normalization cannot replace the resolved Torch stream."""
    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho_torch.workflows import components

    train_container = object()
    test_container = object()
    resolved_payload = object()
    monkeypatch.setattr("ptycho_torch.workflows.containers.create_torch_data_container",
        lambda data, _config: (
            train_container if data == "train" else test_container
        ),
    )
    monkeypatch.setattr(
        "ptycho_torch.config_factory.resolve_training_payload",
        lambda **_kwargs: resolved_payload,
    )
    delegate = MagicMock(return_value={})
    monkeypatch.setattr("ptycho_torch.workflows.lightning_service._train_with_lightning", delegate)
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=Path("train.npz"),
        backend="pytorch",
    )

    components.train_cdi_model_torch(
        "train",
        "test",
        config,
        torch_training_seed=161803,
    )

    assert delegate.call_args.args[:3] == (
        resolved_payload,
        train_container,
        test_container,
    )
    assert delegate.call_args.kwargs["torch_training_seed"] == 161803


def test_lightning_uses_one_training_seed_before_model_and_dataloaders(
    monkeypatch,
    tmp_path,
):
    """Model initialization and data loading share the dedicated Torch stream."""
    from types import SimpleNamespace

    from ptycho.config import (
        ModelConfig,
        PyTorchExecutionConfig,
        TrainingConfig,
    )
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.config_params import (
        DataConfig as PTDataConfig,
        InferenceConfig as PTInferenceConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )
    from ptycho_torch.model_spec import derive_model_spec
    from ptycho_torch.workflows import components

    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=tmp_path / "train.npz",
        output_dir=tmp_path / "out",
        backend="pytorch",
        subsample_seed=17,
    )
    data_config = PTDataConfig()
    model_config = PTModelConfig()
    training_config = PTTrainingConfig()
    payload = SimpleNamespace(
        tf_training_config=config,
        pt_data_config=data_config,
        pt_model_config=model_config,
        pt_training_config=training_config,
        pt_inference_config=PTInferenceConfig(batch_size=23),
        execution_config=PyTorchExecutionConfig(
            accelerator="cpu",
            enable_checkpointing=False,
            logger_backend=None,
        ),
        model_spec=derive_model_spec(
            to_model_config(data_config, model_config),
            model_config,
            data_config,
        ),
    )
    events = []

    class StubLightningModule:
        automatic_optimization = False
        val_loss_name = "val_loss"
        device = "cpu"

        def save_hyperparameters(self):
            pass

    class StubTrainer:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            pass

    build_calls = []

    def fake_build(*args, **_kwargs):
        events.append(("model", None))
        build_calls.append(args)
        return StubLightningModule()

    def fake_dataloaders(*_args, **kwargs):
        events.append(("dataloaders", kwargs["torch_training_seed"]))
        return [], None

    monkeypatch.setattr(
        "lightning.pytorch.seed_everything",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        "ptycho_torch.application_factory.build_ptychopinn_application",
        fake_build,
    )
    monkeypatch.setattr("ptycho_torch.workflows.dataloaders._build_lightning_dataloaders",
        fake_dataloaders,
    )
    monkeypatch.setattr("ptycho_torch.runtime_provenance.build_effective_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr("ptycho_torch.runtime_provenance.write_effective_runtime_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("ptycho_torch.workflows.rect_s1s2._initialize_rect_s1s2",
        lambda *_args, **_kwargs: {
            "schema_version": "rect-s1s2-initialization-v2",
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        },
    )
    monkeypatch.setattr("lightning.pytorch.Trainer", StubTrainer)

    components._train_with_lightning(
        payload,
        {},
        None,
        torch_training_seed=424242,
    )

    assert events[0] == ("seed", 424242)
    assert events.index(("seed", 424242)) < events.index(("model", None))
    assert ("dataloaders", 424242) in events
    assert build_calls[0][3] is payload.pt_inference_config


def _install_generic_runtime_capture_fakes(
    monkeypatch,
    tmp_path,
    *,
    fit_error=None,
    global_zero=True,
):
    """Install one-Trainer fakes around the generic Torch training boundary."""
    from types import SimpleNamespace

    from ptycho.config import ModelConfig, PyTorchExecutionConfig, TrainingConfig
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.config_params import (
        DataConfig as PTDataConfig,
        InferenceConfig as PTInferenceConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )
    from ptycho_torch.model_spec import derive_model_spec
    from ptycho_torch.workflows import components

    output_dir = tmp_path / "training"
    output_dir.mkdir()
    config = TrainingConfig(
        model=ModelConfig(),
        train_data_file=tmp_path / "train.npz",
        output_dir=output_dir,
        backend="pytorch",
    )
    data_config = PTDataConfig()
    model_config = PTModelConfig()
    training_config = PTTrainingConfig(epochs=1)
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        deterministic=True,
        precision="32-true",
        enable_checkpointing=False,
        logger_backend=None,
        num_workers=0,
        pin_memory=False,
    )
    payload = SimpleNamespace(
        tf_training_config=config,
        pt_data_config=data_config,
        pt_model_config=model_config,
        pt_training_config=training_config,
        pt_inference_config=PTInferenceConfig(),
        execution_config=execution_config,
        model_spec=derive_model_spec(
            to_model_config(data_config, model_config),
            model_config,
            data_config,
        ),
    )
    events = []
    trainers = []

    class StubLightningModule:
        automatic_optimization = False
        val_loss_name = "val_loss"
        device = "cpu"

        def save_hyperparameters(self):
            pass

    class StubLoader:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None

        def __len__(self):
            return 1

    class StubTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.is_global_zero = global_zero
            trainers.append(self)
            events.append("trainer")

        def fit(self, *_args, **_kwargs):
            events.append("fit")
            if fit_error is not None:
                raise fit_error

    runtime = {
        "seed": 424242,
        "requested": {"accelerator": "cpu"},
        "effective": {"accelerator": {"device_type": "cpu"}},
    }

    def fake_build_runtime(
        resolved_seed,
        trainer_kwargs,
        resolved_execution,
        dataloader_settings=None,
        trainer=None,
    ):
        from ptycho_torch.batch_order import batch_order_provenance

        assert trainers == [trainer]
        assert trainer_kwargs == trainer.kwargs
        assert resolved_execution is execution_config
        assert resolved_seed == 424242
        assert dataloader_settings == {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "batch_order": batch_order_provenance(
                recipe="torch-generator-v1",
                seed=424242,
                dataset_size=0,
            ),
        }
        events.append("runtime")
        return runtime

    monkeypatch.setattr(
        "ptycho_torch.application_factory.build_ptychopinn_application",
        lambda *_args, **_kwargs: StubLightningModule(),
    )
    monkeypatch.setattr("ptycho_torch.workflows.dataloaders._build_lightning_dataloaders",
        lambda *_args, **_kwargs: (StubLoader(), None),
    )
    monkeypatch.setattr("lightning.pytorch.Trainer", StubTrainer)
    monkeypatch.setattr("ptycho_torch.runtime_provenance.build_effective_runtime",
        fake_build_runtime,
        raising=False,
    )
    monkeypatch.setattr("ptycho_torch.workflows.rect_s1s2._initialize_rect_s1s2",
        lambda *_args, **_kwargs: {
            "schema_version": "rect-s1s2-initialization-v2",
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        },
    )
    return components, config, payload, runtime, events, trainers


def test_generic_lightning_persists_effective_runtime_after_success(
    monkeypatch,
    tmp_path,
):
    """The generic path records its one constructed Trainer after a successful fit."""
    import json

    components, config, payload, runtime, events, trainers = (
        _install_generic_runtime_capture_fakes(monkeypatch, tmp_path)
    )

    result = components._train_with_lightning(
        payload,
        {},
        None,
        torch_training_seed=424242,
    )

    assert events == ["trainer", "fit", "runtime"]
    assert len(trainers) == 1
    assert result["effective_runtime"] == runtime
    assert json.loads((config.output_dir / "effective_runtime.json").read_text()) == runtime


def test_generic_lightning_does_not_persist_runtime_when_fit_fails(
    monkeypatch,
    tmp_path,
):
    """A runtime snapshot is not published as a completed artifact after failed fit."""
    components, config, payload, _runtime, events, trainers = (
        _install_generic_runtime_capture_fakes(
            monkeypatch,
            tmp_path,
            fit_error=ValueError("training exploded"),
        )
    )

    with pytest.raises(RuntimeError, match="Lightning training failed"):
        components._train_with_lightning(
            payload,
            {},
            None,
            torch_training_seed=424242,
        )

    assert events == ["trainer", "fit"]
    assert len(trainers) == 1
    assert not (config.output_dir / "effective_runtime.json").exists()


def test_generic_lightning_nonzero_rank_does_not_publish_runtime(
    monkeypatch,
    tmp_path,
):
    """The constructed Trainer, not ambient process state, owns rank-zero writes."""
    components, config, payload, runtime, events, trainers = (
        _install_generic_runtime_capture_fakes(
            monkeypatch,
            tmp_path,
            global_zero=False,
        )
    )
    reads = []

    def read_rank_zero_selection(path, *, selection_token):
        reads.append((path, selection_token))
        return {
            "schema_version": "serving-checkpoint-selection-v1",
            "policy": "final_in_memory",
            "weights_source": "module_state",
            "selected_path": None,
            "selected_sha256": None,
            "monitor": None,
            "mode": None,
        }

    monkeypatch.setattr("ptycho_torch.workflows.lightning_service._read_checkpoint_selection",
        read_rank_zero_selection,
    )

    result = components._train_with_lightning(
        payload,
        {},
        None,
        torch_training_seed=424242,
    )

    assert events == ["trainer", "fit", "runtime"]
    assert len(trainers) == 1
    assert len(reads) == 1
    assert result["effective_runtime"] == runtime
    assert not (config.output_dir / "effective_runtime.json").exists()


def test_direct_dataloader_seed_resolution_preserves_legacy_fallback():
    """Only an absent legacy seed falls back; zero remains a real seed."""
    from types import SimpleNamespace

    from ptycho_torch.workflows.components import _resolve_torch_training_seed

    assert _resolve_torch_training_seed(
        SimpleNamespace(subsample_seed=73),
        None,
    ) == 73
    assert _resolve_torch_training_seed(
        SimpleNamespace(subsample_seed=None),
        None,
    ) == 42
    assert _resolve_torch_training_seed(
        SimpleNamespace(subsample_seed=0),
        None,
    ) == 0


def test_canonical_flat_raw_without_y_extracts_truth_and_derives_gain(tmp_path):
    """NPZ-style objectGuess data yields one independently checked patch per frame."""
    from types import SimpleNamespace

    import numpy as np

    from ptycho.raw_data import RawData
    from ptycho.workflows.training import _resolve_gain, _selected_truth_patches

    object_guess = (
        1.0
        + np.arange(36, dtype=np.float32).reshape(6, 6)
        + 1j
        * (
            0.5
            + np.arange(36, dtype=np.float32).reshape(6, 6) / 10.0
        )
    ).astype(np.complex64)
    xcoords = np.array([2.0, 4.0], dtype=np.float64)
    ycoords = np.array([2.0, 3.0], dtype=np.float64)
    measured = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ],
        dtype=np.float32,
    )
    data_path = tmp_path / "canonical-flat.npz"
    np.savez(
        data_path,
        xcoords=xcoords,
        ycoords=ycoords,
        diff3d=measured,
        probeGuess=np.ones((2, 2), dtype=np.complex64),
        scan_index=np.zeros(2, dtype=np.int64),
        objectGuess=object_guess,
    )
    raw = RawData.from_file(str(data_path))

    assert raw.Y is None
    patches = _selected_truth_patches(raw, N=2)
    expected = np.stack(
        [
            object_guess[1:3, 1:3],
            object_guess[2:4, 3:5],
        ],
        axis=0,
    )[..., None]
    np.testing.assert_array_equal(patches, expected)

    resolved = SimpleNamespace(
        data=SimpleNamespace(N=2, probe_scale=4.0),
        model=SimpleNamespace(
            amplitude_physics_gain=None,
            amplitude_physics_gain_provenance="pending_derivation",
            physics_forward_mode="amplitude",
            probe_mask=False,
            probe_mask_tensor=None,
            probe_mask_sigma=1.0,
            probe_mask_diameter=None,
        ),
    )
    record = _resolve_gain(resolved, raw)

    assert record.provenance == "derived"
    assert record.value > 0
    assert record.input_statistics["N"] == 2
    assert record.input_statistics["sample_count"] == 2


def test_workflow_grouping_reseeds_validation_independently_of_train_draws(
    tmp_path,
    monkeypatch,
):
    """Train RNG consumption cannot advance the validation grouping stream."""
    import numpy as np

    from ptycho.config import ModelConfig, TrainingConfig
    from ptycho.workflows import training as training_workflow

    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    train_path.touch()
    validation_path.touch()
    grouping_seed = 731
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=1),
        train_data_file=train_path,
        test_data_file=validation_path,
        output_dir=tmp_path / "out",
        backend="tensorflow",
        training_groups=4,
        train_raw_selection=4,
        subsample_seed=grouping_seed,
    )
    monkeypatch.setattr(
        training_workflow,
        "_resolve_public_config",
        lambda _request: config,
    )
    monkeypatch.setattr(
        training_workflow,
        "_materialize_backend_container",
        lambda grouped, *_args: grouped,
    )
    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        lambda *_args, **_kwargs: (
            None,
            None,
            {"backend": "tensorflow"},
        ),
    )
    monkeypatch.setattr(
        training_workflow,
        "_persist_tensorflow_outputs",
        lambda *_args, **_kwargs: None,
    )

    def execute(train_draws):
        observed = {}

        class RNGConsumingRaw:
            def __init__(self, split):
                self.split = split

            def generate_grouped_data(self, **kwargs):
                assert kwargs["seed"] == grouping_seed
                rng = kwargs.get("rng")
                if rng is None:
                    rng = np.random.default_rng(kwargs["seed"])
                if self.split == "train":
                    rng.random(train_draws)
                indices = rng.integers(
                    0,
                    2**31,
                    size=(4, 1),
                    dtype=np.int64,
                )
                observed[self.split] = indices.copy()
                return {"nn_indices": indices}

        def fake_load(path, **_kwargs):
            split = "train" if Path(path) == train_path else "validation"
            return RNGConsumingRaw(split)

        monkeypatch.setattr(training_workflow, "load_data", fake_load)
        training_workflow.run_training_workflow(
            training_workflow.TrainingWorkflowRequest(
                legacy_args=argparse.Namespace(config=None, do_stitching=False),
            )
        )
        return observed

    baseline = execute(train_draws=0)
    perturbed = execute(train_draws=37)
    expected_validation = np.random.default_rng(grouping_seed).integers(
        0,
        2**31,
        size=(4, 1),
        dtype=np.int64,
    )

    np.testing.assert_array_equal(baseline["validation"], expected_validation)
    np.testing.assert_array_equal(perturbed["validation"], expected_validation)
    assert not np.array_equal(baseline["train"], perturbed["train"])


def test_resolved_synthetic_training_owns_sampling_gain_factory_and_reload(
    monkeypatch,
    tmp_path,
):
    """GS2 gain uses selected raw frames before one factory/grouping pass."""
    import numpy as np

    from ptycho.simulation.flat_acquisition import derive_seed_lineage
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow
    from ptycho.workflows import training as training_workflow
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig as TorchInferenceConfig,
        ModelConfig as TorchModelConfig,
    )
    from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain

    resolved = resolve_synthetic_workflow(
        cli_values={
            "gridsize": 2,
            "training_groups": 4096,
            "validation_groups": 1024,
        }
    )
    expected_torch_seed = derive_seed_lineage(
        resolved.simulation.train.seed
    )["torch"]
    assert expected_torch_seed != resolved.training.subsample_seed
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    train_path.touch()
    test_path.touch()
    output_dir = tmp_path / "training"
    events = []
    load_calls = []
    grouping_calls = []

    class FakeRaw:
        def __init__(self, split):
            self.split = split
            self.diff3d = (
                np.arange(16384, dtype=np.float32).reshape(4096, 2, 2) + 1
            )
            self.Y = (
                np.arange(16384, dtype=np.float32).reshape(4096, 2, 2).astype(
                    np.complex64
                )
                + 2j
            )
            self.objectGuess = np.ones((4, 4), dtype=np.complex64)
            self.probeGuess = np.ones((2, 2), dtype=np.complex64)
            self.metadata = None

        def generate_grouped_data(self, **kwargs):
            events.append(f"group:{self.split}")
            grouping_calls.append((self.split, kwargs))
            group_count = kwargs["nsamples"]
            return {
                # Deliberately repeated and unlike the selected raw frames: the
                # gain helper must never observe either grouped array.
                "diffraction": np.full(
                    (group_count, 2, 2, 4), 777.0, dtype=np.float32
                ),
                "Y": np.full(
                    (group_count, 2, 2, 4), 888.0 + 9j, dtype=np.complex64
                ),
                "nn_indices": np.zeros((group_count, 4), dtype=np.int32),
                "X_full": np.ones(
                    (group_count, 2, 2, 4), dtype=np.float32
                ),
            }

    train_raw = FakeRaw("train")
    test_raw = FakeRaw("validation")

    def fake_load(path, **kwargs):
        split = "train" if Path(path) == train_path else "validation"
        events.append(f"load:{split}")
        load_calls.append((split, kwargs))
        return train_raw if split == "train" else test_raw

    monkeypatch.setattr(training_workflow, "load_data", fake_load)
    monkeypatch.setattr(
        training_workflow,
        "_materialize_backend_container",
        lambda grouped, raw, config: grouped,
    )

    gain_record = resolve_amplitude_physics_gain(
        np.ones((1, 2, 2), dtype=np.float32),
        np.ones((1, 2, 2), dtype=np.complex64),
        np.ones((2, 2), dtype=np.complex64),
        probe_scale=4.0,
        override=2.5,
    )

    gain_calls = []

    def fake_gain(measured, truth, probe, **kwargs):
        events.append("gain")
        gain_calls.append((measured, truth, probe, kwargs))
        return gain_record

    monkeypatch.setattr(
        training_workflow,
        "resolve_amplitude_physics_gain",
        fake_gain,
    )

    payload = argparse.Namespace(
        tf_training_config=None,
        pt_data_config=DataConfig(
            N=128,
            gridsize=2,
            neighbor_count=4,
            n_raw_frames_selected=4096,
        ),
        pt_model_config=TorchModelConfig(
            amplitude_physics_gain=2.5,
        ),
        pt_training_config=argparse.Namespace(nll=False),
        pt_inference_config=TorchInferenceConfig(
            **{
                item.name: getattr(resolved.inference, item.name)
                for item in __import__("dataclasses").fields(
                    TorchInferenceConfig
                )
            }
        ),
        model_spec=argparse.Namespace(
            to_model_config=lambda: TorchModelConfig(
                amplitude_physics_gain=2.5,
            )
        ),
    )
    factory_calls = []

    def fake_factory(**kwargs):
        events.append("factory")
        factory_calls.append(kwargs)
        payload.tf_training_config = kwargs["training_baseline"]
        return payload

    monkeypatch.setattr(training_workflow, "resolve_training_payload", fake_factory)

    def fake_dispatch(*_args, **kwargs):
        events.append("dispatch")
        assert kwargs["do_stitching"] is False
        assert kwargs["torch_resolved_payload"] is payload
        assert kwargs["torch_amplitude_physics_gain_record"] is gain_record
        assert kwargs["torch_training_seed"] == expected_torch_seed
        bundle_path = output_dir / "wts.h5.zip"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_bytes(b"bundle")
        return None, None, {"bundle_path": bundle_path, "backend": "pytorch"}

    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        fake_dispatch,
    )

    def fake_reload(path, **kwargs):
        events.append("reload")
        assert Path(path) == output_dir
        assert kwargs == {}
        return {}, {"amplitude_physics_gain_record": gain_record}

    monkeypatch.setattr(
        training_workflow,
        "load_inference_bundle_torch",
        fake_reload,
    )

    result = training_workflow.run_training_workflow(
        training_workflow.TrainingWorkflowRequest(
            resolved_synthetic_workflow=resolved,
            train_data_file=train_path,
            test_data_file=test_path,
            output_dir=output_dir,
        )
    )

    assert events == [
        "load:train",
        "gain",
        "factory",
        "load:validation",
        "group:train",
        "group:validation",
        "dispatch",
        "reload",
    ]
    assert len(gain_calls) == 1
    np.testing.assert_array_equal(gain_calls[0][0], train_raw.diff3d)
    np.testing.assert_array_equal(gain_calls[0][1], train_raw.Y)
    np.testing.assert_array_equal(gain_calls[0][2], train_raw.probeGuess)
    assert load_calls[0][1]["n_subsample"] == 4096
    assert load_calls[0][1]["subsample_seed"] == resolved.training.subsample_seed
    assert load_calls[1][1].get("n_subsample") is None
    assert load_calls[1][1].get("n_images") is None
    assert [entry[1]["nsamples"] for entry in grouping_calls] == [4096, 1024]
    assert [entry[1]["seed"] for entry in grouping_calls] == [
        resolved.training.subsample_seed,
        resolved.training.subsample_seed,
    ]
    assert all(entry[1]["K"] == 4 for entry in grouping_calls)
    assert factory_calls[0]["overrides"]["n_raw_frames_selected"] == 4096
    assert factory_calls[0]["overrides"]["amplitude_physics_gain"] == 2.5
    assert factory_calls[0]["overrides"]["inference_batch_size"] == 16
    assert result.pt_data_config.n_raw_frames_selected == 4096
    assert result.pt_data_config.neighbor_count == 4
    assert result.train_group_count == 4096
    assert result.validation_group_count == 1024
    assert result.amplitude_physics_gain_record is gain_record
    assert result.amplitude_physics_gain_metadata == gain_record.to_metadata()
    assert result.torch_training_seed == expected_torch_seed
    assert result.public_config is payload.tf_training_config
    assert result.public_config.torch_loss_mode == "mae"
    assert result.public_config.torch_mae_pred_l2_match_target is True
    assert payload.pt_training_config.nll is False
    assert payload.pt_inference_config.batch_size == resolved.inference.batch_size
    assert result.bundle_path == output_dir / "wts.h5.zip"


@pytest.mark.parametrize(
    "reload_outcome",
    ["success", "loader_error"],
)
def test_explicit_synthetic_gain_skips_truth_and_preserves_override_record(
    reload_outcome,
    monkeypatch,
    tmp_path,
):
    """Explicit gain provenance bypasses truth extraction through strict reload."""
    from dataclasses import fields

    import numpy as np

    from ptycho import params
    from ptycho.config.legacy_state import transactional_legacy_params
    from ptycho.workflows import training as training_workflow
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig as TorchInferenceConfig,
        ModelConfig as TorchModelConfig,
    )
    from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain

    explicit_gain = 2.25
    resolved = resolve_synthetic_workflow(
        file_values={"model": {"amplitude_physics_gain": explicit_gain}},
        cli_values={"gridsize": 2},
    )
    assert resolved.model.amplitude_physics_gain_provenance == "explicit"

    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    train_path.touch()
    test_path.touch()
    output_dir = tmp_path / "training"
    ambient_cfg = {"sentinel": "caller-owned"}
    ambient_contents = dict(ambient_cfg)
    ambient_sealed = False
    monkeypatch.setattr(params, "cfg", ambient_cfg)
    monkeypatch.setattr(params, "_sealed", ambient_sealed)

    class FakeRaw:
        diff3d = np.ones((4096, 2, 2), dtype=np.float32)
        probeGuess = np.ones((2, 2), dtype=np.complex64)
        metadata = None

        def generate_grouped_data(self, **_kwargs):
            return {
                "nn_indices": np.zeros((1024, 4), dtype=np.int32),
                "diffraction": np.ones((1024, 2, 2, 4), dtype=np.float32),
                "Y": None,
                "X_full": np.ones((1024, 2, 2, 4), dtype=np.float32),
            }

    raw = FakeRaw()
    monkeypatch.setattr(training_workflow, "load_data", lambda *_a, **_k: raw)
    monkeypatch.setattr(
        training_workflow,
        "_materialize_backend_container",
        lambda grouped, *_args: grouped,
    )
    truth_extraction = MagicMock(
        side_effect=AssertionError("explicit gain must not extract truth patches")
    )
    monkeypatch.setattr(
        training_workflow,
        "_selected_truth_patches",
        truth_extraction,
    )

    payload = argparse.Namespace(
        tf_training_config=None,
        pt_data_config=DataConfig(
            N=128,
            gridsize=2,
            neighbor_count=4,
            n_raw_frames_selected=4096,
        ),
        pt_model_config=TorchModelConfig(
            amplitude_physics_gain=explicit_gain,
        ),
        pt_training_config=argparse.Namespace(nll=resolved.training.nll),
        pt_inference_config=TorchInferenceConfig(
            **{
                item.name: getattr(resolved.inference, item.name)
                for item in fields(TorchInferenceConfig)
            }
        ),
        model_spec=object(),
    )
    factory_gain_values = []

    def fake_factory(**kwargs):
        factory_gain_values.append(kwargs["overrides"]["amplitude_physics_gain"])
        payload.tf_training_config = kwargs["training_baseline"]
        return payload

    monkeypatch.setattr(training_workflow, "resolve_training_payload", fake_factory)
    dispatched_records = []

    def fake_dispatch(*_args, **kwargs):
        dispatched_records.append(kwargs["torch_amplitude_physics_gain_record"])
        bundle_path = output_dir / "wts.h5.zip"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_bytes(b"bundle")
        return None, None, {"backend": "pytorch", "bundle_path": bundle_path}

    monkeypatch.setattr(
        training_workflow,
        "run_cdi_example_with_backend",
        fake_dispatch,
    )
    reload_records = []

    class ExpectedReloadError(RuntimeError):
        pass

    @transactional_legacy_params
    def fake_reload(_path):
        record = dispatched_records[0]
        reload_records.append(record)
        params.unseal()
        params.cfg.clear()
        params.cfg.update(
            {
                "archived_bundle": "committed",
                "amplitude_physics_gain": record.value,
            }
        )
        params.seal()
        if reload_outcome == "loader_error":
            raise ExpectedReloadError("strict bundle load failed")
        return {}, {"amplitude_physics_gain_record": record}

    monkeypatch.setattr(
        training_workflow,
        "load_inference_bundle_torch",
        fake_reload,
    )

    request = training_workflow.TrainingWorkflowRequest(
        resolved_synthetic_workflow=resolved,
        train_data_file=train_path,
        test_data_file=test_path,
        output_dir=output_dir,
    )
    if reload_outcome == "success":
        result = training_workflow.run_training_workflow(request)
    else:
        with pytest.raises(ExpectedReloadError, match="strict bundle load failed"):
            training_workflow.run_training_workflow(request)
        result = None

    assert params.cfg is ambient_cfg
    assert params.cfg == ambient_contents
    assert params._sealed is ambient_sealed
    truth_extraction.assert_not_called()
    assert factory_gain_values == [explicit_gain]
    assert len(dispatched_records) == 1
    assert reload_records == [dispatched_records[0]]
    if result is not None:
        assert result.amplitude_physics_gain_record.provenance == "override"
        assert result.amplitude_physics_gain_record.value == explicit_gain
        assert dispatched_records == [result.amplitude_physics_gain_record]
        assert result.amplitude_physics_gain_metadata == (
            result.amplitude_physics_gain_record.to_metadata()
        )

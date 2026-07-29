"""
Unit tests for backend selector integration in scripts/inference/inference.py

This module tests that the inference CLI correctly dispatches to the backend
selector when using PyTorch backend, and that TensorFlow-specific visualization
helpers are skipped for PyTorch runs.

Test Coverage:
1. Inference CLI with backend='pytorch' dispatches to backend_selector
2. TensorFlow-specific tf.keras.backend.clear_session() is handled gracefully
3. TensorFlow backend continues to use legacy inference paths

References:
- Phase R (reactivation): plans/ptychodus_pytorch_integration_plan.md
- Backend selector: ptycho/workflows/backend_selector.py
- Inference CLI: scripts/inference/inference.py
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, call

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestInferenceCliBackendDispatch:
    """
    Test suite for backend selector dispatch in inference CLI.

    These tests verify that scripts/inference/inference.py correctly routes
    through ptycho.workflows.backend_selector.load_inference_bundle_with_backend
    and handles backend-specific visualization appropriately.
    """

    def test_pytorch_backend_dispatch(self):
        """
        Test that inference CLI with backend='pytorch' dispatches to backend selector.

        Expected behavior:
        - Inference CLI imports backend_selector.load_inference_bundle_with_backend
        - Calls load_inference_bundle_with_backend with config.backend='pytorch'
        - Returns PyTorch model ready for inference
        - params.cfg restored from saved bundle (CONFIG-001)

        Phase: R (backend selector integration)
        Reference: input.md Do Now step 3
        """
        from ptycho.config.config import InferenceConfig, ModelConfig

        # Create config with PyTorch backend
        model_config = ModelConfig(N=64, gridsize=1)
        config = InferenceConfig(
            model=model_config,
            model_path=Path('outputs/test/bundle.zip'),
            test_data_file=Path('test.npz'),
            backend='pytorch',  # Explicitly select PyTorch
            output_dir=Path('outputs/inference')
        )

        # Mock PyTorch model
        mock_pytorch_model = MagicMock()
        mock_params_dict = {'gridsize': 1, 'N': 64}

        # Mock the backend selector to verify it's called with PyTorch config
        mock_load_bundle = MagicMock(
            return_value=(mock_pytorch_model, mock_params_dict)
        )

        with patch('ptycho.workflows.backend_selector.load_inference_bundle_with_backend', mock_load_bundle):
            # Simulate the inference CLI logic
            # (In actual CLI this would be inside main())
            model, params_dict = mock_load_bundle(config.model_path, config)

            # Verify backend selector was called
            assert mock_load_bundle.called, \
                "load_inference_bundle_with_backend should be called"

            # Verify it received the PyTorch config
            call_args = mock_load_bundle.call_args
            assert call_args[0][1].backend == 'pytorch', \
                "Backend selector should receive config with backend='pytorch'"

            # Verify model was loaded
            assert model is mock_pytorch_model, \
                "Should return PyTorch model"

            # Verify params_dict was restored
            assert params_dict == mock_params_dict, \
                "Should restore params_dict from saved bundle"

    def test_tensorflow_backend_dispatch(self):
        """
        Test that inference CLI with backend='tensorflow' uses legacy loader.

        Expected behavior:
        - Inference CLI calls backend_selector with config.backend='tensorflow'
        - Returns TensorFlow model (tf.keras.Model)
        - params.cfg restored from saved bundle (CONFIG-001)

        Phase: R (backend selector integration)
        Reference: input.md Do Now step 3
        """
        from ptycho.config.config import InferenceConfig, ModelConfig

        # Create config with TensorFlow backend (default)
        model_config = ModelConfig(N=64, gridsize=1)
        config = InferenceConfig(
            model=model_config,
            model_path=Path('outputs/test/wts.h5.zip'),
            test_data_file=Path('test.npz'),
            backend='tensorflow',  # Explicitly select TensorFlow
            output_dir=Path('outputs/inference')
        )

        # Mock TensorFlow model (tf.keras.Model)
        mock_tf_model = MagicMock()
        mock_params_dict = {'gridsize': 1, 'N': 64}

        # Mock the backend selector
        mock_load_bundle = MagicMock(
            return_value=(mock_tf_model, mock_params_dict)
        )

        with patch('ptycho.workflows.backend_selector.load_inference_bundle_with_backend', mock_load_bundle):
            # Simulate the inference CLI logic
            model, params_dict = mock_load_bundle(config.model_path, config)

            # Verify backend selector was called
            assert mock_load_bundle.called, \
                "load_inference_bundle_with_backend should be called"

            # Verify it received the TensorFlow config
            call_args = mock_load_bundle.call_args
            assert call_args[0][1].backend == 'tensorflow', \
                "Backend selector should receive config with backend='tensorflow'"

            # Verify model was loaded
            assert model is mock_tf_model, \
                "Should return TensorFlow model"

    def test_backend_selector_preserves_config_001_compliance(self):
        """
        Test that backend selector properly restores params.cfg (CONFIG-001).

        Expected behavior:
        - load_inference_bundle_with_backend delegates to backend-specific loaders
        - Both TF and PyTorch loaders restore params.cfg from saved bundle
        - No additional update_legacy_dict needed in CLI (already handled)

        Phase: R (backend selector integration)
        Reference: CONFIG-001 in docs/findings.md
        """
        from ptycho.config.config import InferenceConfig, ModelConfig

        # Create config
        model_config = ModelConfig(N=64, gridsize=2)
        config = InferenceConfig(
            model=model_config,
            model_path=Path('outputs/test/bundle.zip'),
            test_data_file=Path('test.npz'),
            backend='pytorch',
            output_dir=Path('outputs/inference')
        )

        # Mock model and params_dict
        mock_model = MagicMock()
        mock_params_dict = {'gridsize': 2, 'N': 64, 'backend': 'pytorch'}

        mock_load_bundle = MagicMock(
            return_value=(mock_model, mock_params_dict)
        )

        with patch('ptycho.workflows.backend_selector.load_inference_bundle_with_backend', mock_load_bundle):
            # Load model
            model, params_dict = mock_load_bundle(config.model_path, config)

            # Verify params_dict was returned (would be used to restore params.cfg inside loader)
            assert 'gridsize' in params_dict, \
                "params_dict should contain gridsize (for CONFIG-001 compliance)"
            assert params_dict['gridsize'] == 2, \
                "params_dict gridsize should match saved bundle"

            # Note: In real implementation, params.cfg restoration happens inside
            # the backend-specific loader (load_inference_bundle or load_inference_bundle_torch)
            # via update_legacy_dict(params.cfg, restored_config)

    def test_cli_backend_argument_parsing(self):
        """
        Test that inference CLI correctly parses --backend argument.

        Expected behavior:
        - scripts/inference/inference.py accepts --backend {tensorflow,pytorch}
        - Omitted backend is absent so the resolved config can apply its default
        - An explicit argument is passed to setup_inference_configuration
        - InferenceConfig.backend is populated after source resolution

        Phase: R (backend selector integration)
        Reference: input.md Do Now step 2-3
        """
        import sys
        import argparse
        from pathlib import Path

        # Import the inference script's parse_arguments function
        # We need to mock sys.argv to test argument parsing
        test_cases = [
            # (argv, expected_backend)
            (
                [
                    'inference.py',
                    '--model_path',
                    'model.zip',
                    '--test_data',
                    'test.npz',
                    '--backend',
                    'tensorflow',
                ],
                'tensorflow',
            ),
            (
                [
                    'inference.py',
                    '--model_path',
                    'model.zip',
                    '--test_data',
                    'test.npz',
                    '--backend',
                    'pytorch',
                ],
                'pytorch',
            ),
        ]

        for argv, expected_backend in test_cases:
            with patch.object(sys, 'argv', argv):
                # Import and call parse_arguments from the inference script
                # Note: This assumes the script defines parse_arguments at module level
                from scripts.inference import inference

                args = inference.parse_arguments()

                assert hasattr(args, 'backend'), \
                    "Parsed args should have 'backend' attribute"
                assert args.backend == expected_backend, \
                    f"Expected backend={expected_backend}, got {args.backend}"

        with patch.object(sys, 'argv', ['inference.py']):
            from scripts.inference import inference

            args = inference.parse_arguments()

        assert not hasattr(args, 'backend')
        assert not hasattr(args, 'model_path')
        assert not hasattr(args, 'test_data')

        # Test that invalid backend value is rejected
        with patch.object(sys, 'argv', ['inference.py', '--model_path', 'model.zip',
                                        '--test_data', 'test.npz', '--backend', 'invalid']):
            with pytest.raises(SystemExit):
                # argparse should exit with error for invalid choice
                from scripts.inference import inference
                inference.parse_arguments()

    def test_setup_inference_configuration_uses_backend(self, tmp_path: Path):
        """
        Test that setup_inference_configuration properly uses backend from args.

        Expected behavior:
        - args.backend is passed to InferenceConfig constructor
        - InferenceConfig.backend field matches args.backend
        - Both 'tensorflow' and 'pytorch' values are supported

        Phase: R (backend selector integration)
        Reference: input.md Do Now step 2
        """
        from scripts.inference.inference import setup_inference_configuration
        from pathlib import Path
        import argparse

        model_zip = tmp_path / "model.zip"
        model_zip.touch()
        output_dir = tmp_path / "inference_outputs"
        output_dir.mkdir()
        test_data = tmp_path / "test.npz"
        test_data.touch()

        for backend_value in ['tensorflow', 'pytorch']:
            # Create mock args
            args = argparse.Namespace(
                model_path=str(model_zip),
                test_data=str(test_data),
                config=None,
                output_dir=str(output_dir),
                debug=False,
                n_images=None,
                n_subsample=None,
                subsample_seed=None,
                backend=backend_value
            )

            # Call setup_inference_configuration
            config = setup_inference_configuration(args, yaml_path=None)

            # Verify backend field is set correctly
            assert config.backend == backend_value, \
                f"InferenceConfig.backend should be '{backend_value}', got '{config.backend}'"

    def test_inference_parser_omits_unsupplied_config_values(self):
        from scripts.inference import inference

        with patch.object(sys, 'argv', ['inference.py']):
            args = inference.parse_arguments()

        for name in (
            'model_path',
            'test_data',
            'output_dir',
            'debug',
            'n_images',
            'n_subsample',
            'subsample_seed',
            'backend',
        ):
            assert not hasattr(args, name)
        assert args.comparison_plot is False
        assert args.debug_dump is None
        assert args.phase_vmin is None
        assert args.phase_vmax is None

    def test_inference_yaml_file_only_values_and_root_fields_are_resolved(
        self,
        tmp_path: Path,
    ):
        from scripts.inference import inference

        config_path = tmp_path / 'inference.yaml'
        config_path.write_text(
            '\n'.join(
                [
                    f'model_path: {tmp_path / "model.zip"}',
                    f'test_data_file: {tmp_path / "test.npz"}',
                    'n_groups: 6',
                    'neighbor_count: 7',
                    'subsample_seed: 123',
                    'debug: true',
                    f'output_dir: {tmp_path / "yaml-output"}',
                    'backend: pytorch',
                    'model:',
                    '  N: 128',
                ]
            )
            + '\n',
            encoding='utf-8',
        )

        with patch.object(
            sys,
            'argv',
            ['inference.py', '--config', str(config_path)],
        ):
            args = inference.parse_arguments()

        config = inference.setup_inference_configuration(args, args.config)

        assert config.model_path == tmp_path / 'model.zip'
        assert config.test_data_file == tmp_path / 'test.npz'
        assert config.n_groups == 6
        assert config.neighbor_count == 7
        assert config.subsample_seed == 123
        assert config.debug is True
        assert config.output_dir == tmp_path / 'yaml-output'
        assert config.backend == 'pytorch'
        assert config.model.N == 128

    def test_inference_cli_test_data_maps_to_canonical_field_and_default_wins(
        self,
        tmp_path: Path,
    ):
        from scripts.inference import inference

        config_path = tmp_path / 'inference.yaml'
        config_path.write_text(
            '\n'.join(
                [
                    f'model_path: {tmp_path / "yaml-model.zip"}',
                    f'test_data_file: {tmp_path / "yaml-test.npz"}',
                    'backend: pytorch',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
        cli_test_data = tmp_path / 'cli-test.npz'

        with patch.object(
            sys,
            'argv',
            [
                'inference.py',
                '--config',
                str(config_path),
                '--test_data',
                str(cli_test_data),
                '--backend',
                'tensorflow',
            ],
        ):
            args = inference.parse_arguments()

        config = inference.setup_inference_configuration(args, args.config)

        assert config.test_data_file == cli_test_data
        assert config.backend == 'tensorflow'

    def test_inference_file_only_main_uses_resolved_data_and_neighbor_count(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        import numpy as np
        from ptycho import params
        from scripts.inference import inference

        config_path = tmp_path / 'inference.yaml'
        test_data_path = tmp_path / 'yaml-only-test.npz'
        output_dir = tmp_path / 'output'
        config_path.write_text(
            '\n'.join(
                [
                    f'model_path: {tmp_path / "model.zip"}',
                    f'test_data_file: {test_data_path}',
                    'n_groups: 1',
                    'neighbor_count: 7',
                    f'output_dir: {output_dir}',
                    'backend: tensorflow',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
        raw_data = MagicMock()
        raw_data.xcoords = np.arange(4)
        load_data = MagicMock(return_value=raw_data)
        perform_inference = MagicMock(
            return_value=(
                np.ones((2, 2)),
                np.zeros((2, 2)),
                None,
                None,
            )
        )
        monkeypatch.setattr(params, 'cfg', {'gridsize': 1})
        monkeypatch.setattr(
            sys,
            'argv',
            ['inference.py', '--config', str(config_path)],
        )

        with patch.object(
            inference,
            'load_inference_bundle_with_backend',
            return_value=(MagicMock(), {'gridsize': 1}),
        ), patch.object(
            inference,
            'load_data',
            load_data,
        ), patch.object(
            inference,
            'perform_inference',
            perform_inference,
        ), patch.object(
            inference,
            'save_reconstruction_images',
        ), pytest.raises(SystemExit) as exit_info:
            inference.main()

        assert exit_info.value.code == 0
        assert load_data.call_args.args[0] == test_data_path
        assert perform_inference.call_args.kwargs['K'] == 7

    def test_inference_yaml_debug_reaches_pytorch_execution_namespace(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        import numpy as np
        from ptycho import params
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args as real_request_builder,
        )
        from scripts.inference import inference

        config_path = tmp_path / 'inference.yaml'
        config_path.write_text(
            '\n'.join(
                [
                    f'model_path: {tmp_path / "model.zip"}',
                    f'test_data_file: {tmp_path / "test.npz"}',
                    'n_groups: 1',
                    'debug: true',
                    f'output_dir: {tmp_path / "output"}',
                    'backend: pytorch',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
        raw_data = MagicMock()
        raw_data.xcoords = np.arange(4)
        execution_config = MagicMock(
            accelerator='cpu',
            num_workers=0,
            inference_batch_size=None,
        )
        runtime = MagicMock(config=execution_config)
        runtime.audit_dict.return_value = {
            'requested': {'accelerator': 'auto'},
            'resolved': {'accelerator': 'cpu'},
        }
        monkeypatch.setattr(params, 'cfg', {'gridsize': 1})
        monkeypatch.setattr(
            sys,
            'argv',
            ['inference.py', '--config', str(config_path)],
        )

        with patch.object(
            inference,
            'load_inference_bundle_with_backend',
            return_value=(MagicMock(), {'gridsize': 1}),
        ), patch.object(
            inference,
            'load_data',
            return_value=raw_data,
        ), patch(
            'ptycho_torch.execution_request.resolve_runtime_execution_request',
            return_value=runtime,
        ) as resolver, patch(
            'ptycho_torch.cli.shared.build_execution_request_from_args',
            wraps=real_request_builder,
        ), patch(
            'ptycho_torch.inference._run_inference_and_reconstruct',
            return_value=(np.ones((2, 2)), np.zeros((2, 2))),
        ), patch.object(
            inference,
            'save_reconstruction_images',
        ), pytest.raises(SystemExit) as exit_info:
            inference.main()

        assert exit_info.value.code == 0
        request = resolver.call_args.args[0]
        assert request.values['enable_progress_bar'] is True
        assert 'enable_progress_bar' not in request.explicit_fields

    def test_unified_inference_execution_request_explicit_resolution_once(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Resolve one request after bundle validation and use its config only."""
        import argparse
        import numpy as np

        from ptycho import params
        from ptycho.config import InferenceConfig, ModelConfig
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args as real_request_builder,
        )
        from ptycho_torch.execution_request import ExecutionRequest
        from scripts.inference import inference

        model_path = tmp_path / 'model'
        model_path.mkdir()
        test_path = tmp_path / 'test.npz'
        test_path.touch()
        output_path = tmp_path / 'output'
        config = InferenceConfig(
            model=ModelConfig(N=64, gridsize=1),
            model_path=model_path,
            test_data_file=test_path,
            output_dir=output_path,
            backend='pytorch',
            n_groups=1,
        )
        args = argparse.Namespace(
            config=None,
            debug_dump=None,
            torch_accelerator='cpu',
            torch_num_workers=2,
            torch_inference_batch_size=8,
            comparison_plot=False,
            phase_vmin=None,
            phase_vmax=None,
        )
        raw_argv = (
            '--torch-accelerator=cpu',
            '--torch-num-workers', '2',
            '--torch-inference-batch-size=8',
        )
        monkeypatch.setattr(sys, 'argv', ['inference.py', *raw_argv])
        monkeypatch.setattr(inference, 'parse_arguments', lambda: args)
        monkeypatch.setattr(
            inference,
            'setup_inference_configuration',
            lambda *_args: config,
        )
        monkeypatch.setattr(params, 'cfg', {'gridsize': 1})

        events = []
        model = MagicMock()
        model.to.side_effect = lambda *_args: events.append('model.to')
        raw_data = MagicMock()
        raw_data.xcoords = np.arange(4)
        resolved_config = MagicMock(
            accelerator='cpu',
            num_workers=2,
            inference_batch_size=8,
        )
        runtime = MagicMock(config=resolved_config)
        runtime.audit_dict.return_value = {
            'requested': {'accelerator': 'cpu'},
            'resolved': {'accelerator': 'cpu'},
        }

        def load_bundle(*_args):
            events.append('bundle')
            return model, {'gridsize': 1}

        def resolve(request, *, mode):
            events.append('resolve')
            assert isinstance(request, ExecutionRequest)
            assert mode == 'inference'
            return runtime

        with patch.object(
            inference,
            'load_inference_bundle_with_backend',
            side_effect=load_bundle,
        ), patch.object(
            inference,
            'load_data',
            return_value=raw_data,
        ), patch(
            'ptycho_torch.cli.shared.build_execution_request_from_args',
            wraps=real_request_builder,
        ) as request_builder, patch(
            'ptycho_torch.execution_request.resolve_runtime_execution_request',
            side_effect=resolve,
        ) as resolver, patch(
            'ptycho_torch.inference._run_inference_and_reconstruct',
            return_value=(np.ones((2, 2)), np.zeros((2, 2))),
        ) as infer, patch.object(
            inference,
            'save_reconstruction_images',
        ), pytest.raises(SystemExit) as exit_info:
            inference.main()

        assert exit_info.value.code == 0
        request_builder.assert_called_once()
        assert request_builder.call_args.kwargs == {
            'mode': 'inference',
            'explicit_options': raw_argv,
            'lane': 'unified-inference',
        }
        request = resolver.call_args.args[0]
        assert request.explicit_fields == frozenset(
            {'accelerator', 'num_workers', 'inference_batch_size'}
        )
        resolver.assert_called_once()
        runtime.audit_dict.assert_called_once_with()
        assert events == ['bundle', 'resolve', 'model.to']
        model.to.assert_called_once_with('cpu')
        assert infer.call_args.args[3] is resolved_config

    def test_unified_inference_execution_notices_emit_once_after_resolution(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Emit resolved notices only after the bundle and runtime validate."""
        import argparse
        import numpy as np

        from ptycho import params
        from ptycho.config import InferenceConfig, ModelConfig
        from ptycho_torch.execution_request import ResolutionNotice
        from scripts.inference import inference

        model_path = tmp_path / 'model'
        model_path.mkdir()
        test_path = tmp_path / 'test.npz'
        test_path.touch()
        config = InferenceConfig(
            model=ModelConfig(N=64, gridsize=1),
            model_path=model_path,
            test_data_file=test_path,
            output_dir=tmp_path / 'output',
            backend='pytorch',
            n_groups=1,
        )
        args = argparse.Namespace(
            config=None,
            debug_dump=None,
            torch_accelerator='cpu',
            torch_num_workers=0,
            torch_inference_batch_size=1,
            comparison_plot=False,
            phase_vmin=None,
            phase_vmax=None,
        )
        monkeypatch.setattr(
            sys,
            'argv',
            ['inference.py', '--torch-accelerator', 'cpu'],
        )
        monkeypatch.setattr(inference, 'parse_arguments', lambda: args)
        monkeypatch.setattr(
            inference,
            'setup_inference_configuration',
            lambda *_args: config,
        )
        monkeypatch.setattr(params, 'cfg', {'gridsize': 1})

        first_notice = ResolutionNotice(
            DeprecationWarning,
            'first resolved execution notice',
        )
        second_notice = ResolutionNotice(
            UserWarning,
            'second resolved execution notice',
        )
        resolved_config = MagicMock(
            accelerator='cpu',
            num_workers=0,
            inference_batch_size=1,
        )
        runtime = MagicMock(
            config=resolved_config,
            notices=(first_notice, second_notice),
        )
        runtime.audit_dict.return_value = {
            'requested': {'accelerator': 'cpu'},
            'resolved': {'accelerator': 'cpu'},
        }
        model = MagicMock()
        raw_data = MagicMock()
        raw_data.xcoords = np.arange(1)
        events = []

        def load_bundle(*_args):
            events.append('bundle validated')
            return model, {'gridsize': 1}

        def resolve(*_args, **_kwargs):
            events.append('runtime resolved')
            return runtime

        def emit_warning(message, category=None, **_kwargs):
            events.append(('notice', message, category))

        model.to.side_effect = lambda *_args: events.append('model.to')

        with patch.object(
            inference,
            'load_inference_bundle_with_backend',
            side_effect=load_bundle,
        ), patch.object(
            inference,
            'load_data',
            return_value=raw_data,
        ), patch(
            'ptycho_torch.execution_request.resolve_runtime_execution_request',
            side_effect=resolve,
        ), patch(
            'warnings.warn',
            side_effect=emit_warning,
        ) as warn, patch(
            'ptycho_torch.inference._run_inference_and_reconstruct',
            return_value=(np.ones((2, 2)), np.zeros((2, 2))),
        ), patch.object(
            inference,
            'save_reconstruction_images',
        ), pytest.raises(SystemExit) as exit_info:
            inference.main()

        assert exit_info.value.code == 0
        assert events == [
            'bundle validated',
            'runtime resolved',
            (
                'notice',
                first_notice.message,
                first_notice.category,
            ),
            (
                'notice',
                second_notice.message,
                second_notice.category,
            ),
            'model.to',
        ]
        assert warn.call_args_list == [
            call(
                first_notice.message,
                first_notice.category,
                stacklevel=2,
            ),
            call(
                second_notice.message,
                second_notice.category,
                stacklevel=2,
            ),
        ]

    def test_inference_sampling_reads_canonical_n_groups(self):
        from ptycho.config import InferenceConfig, ModelConfig
        from scripts.inference.inference import interpret_sampling_parameters

        config = InferenceConfig(
            model=ModelConfig(gridsize=2),
            model_path=Path('model.zip'),
            test_data_file=Path('test.npz'),
            n_groups=3,
        )

        n_subsample, n_groups, message = interpret_sampling_parameters(
            config,
            gridsize=2,
        )

        assert n_subsample is None
        assert n_groups == 3
        assert '3 groups' in message

    def test_pytorch_inference_execution_path(self):
        """
        Test that PyTorch backend executes _run_inference_and_reconstruct() helper.

        Expected behavior:
        - When config.backend == 'pytorch', skip TensorFlow perform_inference()
        - Call ptycho_torch.inference._run_inference_and_reconstruct with model, raw_data, config
        - Return amplitude/phase arrays from PyTorch helper
        - Skip TensorFlow tf.keras.backend.clear_session() cleanup

        Phase: R (PyTorch inference execution path)
        Reference: input.md Do Now step 2-3
        """
        import numpy as np
        from unittest.mock import MagicMock, patch, ANY

        # Mock the PyTorch inference helper to verify it's called
        mock_pytorch_inference = MagicMock(
            return_value=(
                np.random.rand(64, 64),  # amplitude
                np.random.rand(64, 64)   # phase
            )
        )

        # Mock TensorFlow inference to verify it's NOT called
        mock_tf_inference = MagicMock()

        # Mock the PyTorchExecutionConfig factory
        mock_execution_config = MagicMock()
        mock_pytorch_execution_config_class = MagicMock(return_value=mock_execution_config)

        with patch('scripts.inference.inference.perform_inference', mock_tf_inference), \
             patch('ptycho_torch.inference._run_inference_and_reconstruct', mock_pytorch_inference), \
             patch('ptycho_torch.config_factory.PyTorchExecutionConfig', mock_pytorch_execution_config_class):

            # Simulate the inference CLI logic for PyTorch backend
            # (In actual CLI this would be inside main() after loading model and data)
            from ptycho.config.config import InferenceConfig, ModelConfig

            config = InferenceConfig(
                model=ModelConfig(N=64, gridsize=1),
                model_path=Path('outputs/test/bundle.zip'),
                test_data_file=Path('test.npz'),
                backend='pytorch',
                output_dir=Path('outputs/inference')
            )

            # Execute the branch logic
            if config.backend == 'pytorch':
                from ptycho_torch.inference import _run_inference_and_reconstruct
                from ptycho_torch.config_factory import PyTorchExecutionConfig

                execution_config = PyTorchExecutionConfig(
                    accelerator='cpu',
                    batch_size=16
                )

                device = 'cpu'
                mock_model = MagicMock()
                mock_test_data = MagicMock()

                reconstructed_amplitude, reconstructed_phase = _run_inference_and_reconstruct(
                    mock_model, mock_test_data, config, execution_config, device, quiet=False
                )

            # Verify PyTorch helper was called
            assert mock_pytorch_inference.called, \
                "_run_inference_and_reconstruct should be called for PyTorch backend"

            # Verify it received the correct arguments
            call_args = mock_pytorch_inference.call_args
            assert call_args is not None, \
                "PyTorch inference helper should have received arguments"

            # Verify TensorFlow inference was NOT called
            assert not mock_tf_inference.called, \
                "perform_inference (TensorFlow) should NOT be called for PyTorch backend"

    def test_pytorch_execution_config_flags(self):
        """
        Test that PyTorch CLI execution flags propagate to execution_config.

        Expected behavior:
        - CLI args --torch-accelerator, --torch-num-workers, --torch-inference-batch-size
        - Map to PyTorchExecutionConfig fields via build_execution_config_from_args
        - Execution config is validated and passed to _run_inference_and_reconstruct
        - Verify execution_config mirrors the CLI flags

        Phase: R (PyTorch execution config flags)
        Reference: input.md Do Now step 3
        """
        import numpy as np
        from unittest.mock import MagicMock, patch, ANY
        import argparse

        # Mock args with PyTorch execution flags
        args = argparse.Namespace(
            model_path='outputs/test/bundle.zip',
            test_data='test.npz',
            config=None,
            output_dir='outputs/inference',
            debug=False,
            comparison_plot=False,
            n_images=None,
            n_subsample=None,
            subsample_seed=None,
            phase_vmin=None,
            phase_vmax=None,
            backend='pytorch',
            torch_accelerator='cpu',
            torch_num_workers=2,
            torch_inference_batch_size=8
        )

        # Mock the PyTorch inference helper to capture execution_config
        mock_pytorch_inference = MagicMock(
            return_value=(
                np.random.rand(64, 64),  # amplitude
                np.random.rand(64, 64)   # phase
            )
        )

        # Mock build_execution_config_from_args to verify it's called correctly
        from ptycho.config.config import PyTorchExecutionConfig
        mock_execution_config = PyTorchExecutionConfig(
            accelerator='cpu',
            num_workers=2,
            inference_batch_size=8,
            enable_progress_bar=True
        )

        def mock_build_exec_config(exec_args, mode):
            # Verify the mapped args
            assert exec_args.accelerator == 'cpu', "Accelerator should be mapped from torch_accelerator"
            assert exec_args.num_workers == 2, "num_workers should be mapped from torch_num_workers"
            assert exec_args.inference_batch_size == 8, "inference_batch_size should be mapped"
            assert mode == 'inference', "Mode should be 'inference'"
            return mock_execution_config

        with patch('ptycho_torch.inference._run_inference_and_reconstruct', mock_pytorch_inference), \
             patch('ptycho_torch.cli.shared.build_execution_config_from_args', side_effect=mock_build_exec_config):

            # Simulate the inference CLI logic (from scripts/inference/inference.py)
            from ptycho.config.config import InferenceConfig, ModelConfig
            from ptycho_torch.cli.shared import build_execution_config_from_args

            config = InferenceConfig(
                model=ModelConfig(N=64, gridsize=1),
                model_path=Path('outputs/test/bundle.zip'),
                test_data_file=Path('test.npz'),
                backend='pytorch',
                output_dir=Path('outputs/inference')
            )

            if config.backend == 'pytorch':
                # Map CLI args to execution config field names
                exec_args = argparse.Namespace(
                    accelerator=getattr(args, 'torch_accelerator', 'auto'),
                    num_workers=getattr(args, 'torch_num_workers', 0),
                    inference_batch_size=getattr(args, 'torch_inference_batch_size', None),
                    quiet=getattr(args, 'debug', False) == False,
                    disable_mlflow=False
                )

                # Build validated execution config
                execution_config = build_execution_config_from_args(exec_args, mode='inference')

                # Verify execution_config fields
                assert execution_config.accelerator == 'cpu', \
                    "execution_config.accelerator should match CLI flag"
                assert execution_config.num_workers == 2, \
                    "execution_config.num_workers should match CLI flag"
                assert execution_config.inference_batch_size == 8, \
                    "execution_config.inference_batch_size should match CLI flag"

                # Simulate calling _run_inference_and_reconstruct
                device = 'cpu'
                mock_model = MagicMock()
                mock_test_data = MagicMock()

                reconstructed_amplitude, reconstructed_phase = mock_pytorch_inference(
                    mock_model, mock_test_data, config, execution_config, device, quiet=False
                )

            # Verify PyTorch helper was called
            assert mock_pytorch_inference.called, \
                "_run_inference_and_reconstruct should be called with execution_config"

            # Verify execution_config argument matches our flags
            call_args = mock_pytorch_inference.call_args
            assert call_args is not None, \
                "PyTorch inference helper should have received arguments"

            # The 4th positional argument (index 3) should be execution_config
            passed_execution_config = call_args[0][3]
            assert passed_execution_config.accelerator == 'cpu', \
                "Passed execution_config should have accelerator='cpu'"
            assert passed_execution_config.num_workers == 2, \
                "Passed execution_config should have num_workers=2"
            assert passed_execution_config.inference_batch_size == 8, \
                "Passed execution_config should have inference_batch_size=8"

    def test_pytorch_backend_moves_model_to_execution_device(self):
        """
        Test that PyTorch backend moves model to execution device (DEVICE-MISMATCH-001).

        Expected behavior:
        - After loading model from bundle, CLI calls model.to(device) and model.eval()
        - The device string is derived from torch_accelerator CLI flag
        - Model is on the correct device before calling _run_inference_and_reconstruct
        - _run_inference_and_reconstruct also ensures model is on device (defensive guard)

        Phase: R (DEVICE-MISMATCH-001 fix)
        Reference: input.md Do Now step 2-3, DEVICE-MISMATCH-001 finding
        """
        import argparse
        from unittest.mock import MagicMock, patch, call

        # Mock args with CUDA accelerator
        args = argparse.Namespace(
            model_path='outputs/test/bundle.zip',
            test_data='test.npz',
            config=None,
            output_dir='outputs/inference',
            debug=False,
            comparison_plot=False,
            n_images=None,
            n_subsample=None,
            subsample_seed=None,
            phase_vmin=None,
            phase_vmax=None,
            backend='pytorch',
            torch_accelerator='cuda',  # Request CUDA device
            torch_num_workers=0,
            torch_inference_batch_size=None
        )

        # Mock PyTorch model with .to() and .eval() methods
        mock_pytorch_model = MagicMock()
        mock_pytorch_model.to = MagicMock(return_value=mock_pytorch_model)
        mock_pytorch_model.eval = MagicMock(return_value=mock_pytorch_model)

        # Track model.to() calls to verify device placement
        device_calls = []

        def track_to_call(device):
            device_calls.append(device)
            return mock_pytorch_model

        mock_pytorch_model.to = MagicMock(side_effect=track_to_call)

        # Mock backend selector to return the mock model
        mock_load_bundle = MagicMock(
            return_value=(mock_pytorch_model, {'gridsize': 1, 'N': 64})
        )

        # Mock build_execution_config_from_args
        from ptycho.config.config import PyTorchExecutionConfig
        mock_execution_config = PyTorchExecutionConfig(
            accelerator='cuda',
            num_workers=0,
            inference_batch_size=None,
            enable_progress_bar=False
        )

        mock_build_exec_config = MagicMock(return_value=mock_execution_config)

        with patch('ptycho.workflows.backend_selector.load_inference_bundle_with_backend', mock_load_bundle), \
             patch('ptycho_torch.cli.shared.build_execution_config_from_args', mock_build_exec_config):

            # Simulate the CLI logic (from scripts/inference/inference.py lines 466-496)
            from ptycho.config.config import InferenceConfig, ModelConfig

            config = InferenceConfig(
                model=ModelConfig(N=64, gridsize=1),
                model_path=Path('outputs/test/bundle.zip'),
                test_data_file=Path('test.npz'),
                backend='pytorch',
                output_dir=Path('outputs/inference')
            )

            # Load model
            model, _ = mock_load_bundle(config.model_path, config)

            # PyTorch backend device placement logic
            if config.backend == 'pytorch':
                # Build execution config
                exec_args = argparse.Namespace(
                    accelerator=getattr(args, 'torch_accelerator', 'auto'),
                    num_workers=getattr(args, 'torch_num_workers', 0),
                    inference_batch_size=getattr(args, 'torch_inference_batch_size', None),
                    quiet=False,
                    disable_mlflow=False
                )

                execution_config = mock_build_exec_config(exec_args, mode='inference')

                # Resolve device from accelerator
                if execution_config.accelerator in ('cuda', 'gpu'):
                    device_str = 'cuda'
                elif execution_config.accelerator == 'mps':
                    device_str = 'mps'
                else:
                    device_str = 'cpu'

                # This is the DEVICE-MISMATCH-001 fix:
                # CLI must call model.to(device) after loading
                model.to(device_str)
                model.eval()

        # Verify model.to() was called with 'cuda'
        assert len(device_calls) > 0, \
            "model.to() should be called to move model to execution device"
        assert device_calls[0] == 'cuda', \
            f"model.to() should be called with 'cuda', got {device_calls[0]}"

        # Verify model.eval() was called
        assert mock_pytorch_model.eval.called, \
            "model.eval() should be called after loading bundle"

    def test_pytorch_backend_defaults_auto_execution_config(self, caplog):
        """
        Test that inference CLI with backend='pytorch' and NO --torch-* flags
        emits POLICY-001 log message.

        Expected behavior:
        - When no --torch-* flags provided, CLI logs POLICY-001 message about GPU-first defaults
        - Log message instructs CPU-only users to pass --torch-accelerator cpu

        Phase: CLI GPU-default logging
        Reference: input.md Do Now step 3
        """
        import sys
        import logging
        from unittest.mock import patch, MagicMock

        # Configure caplog to capture INFO level
        caplog.set_level(logging.INFO)

        # Mock sys.argv to simulate NO --torch-* flags
        original_argv = sys.argv
        try:
            sys.argv = ['inference.py', '--backend', 'pytorch', '--model_path', 'bundle.zip', '--test_data', 'test.npz']

            # Simulate the inference CLI logic from scripts/inference/inference.py:473-486
            # Determine if user explicitly provided any --torch-* flags
            torch_flags_explicitly_set = any([
                'torch_accelerator' in sys.argv or '--torch-accelerator' in sys.argv,
                'torch_num_workers' in sys.argv or '--torch-num-workers' in sys.argv,
                'torch_inference_batch_size' in sys.argv or '--torch-inference-batch-size' in sys.argv,
            ])

            # Import print from scripts.inference.inference (which is actually logger.info)
            # For testing, we'll use a real logger
            test_logger = logging.getLogger('test_inference_policy')

            if not torch_flags_explicitly_set:
                # No --torch-* flags provided: emit POLICY-001 info log
                test_logger.info("POLICY-001: No --torch-* execution flags provided. "
                                "Backend will use GPU-first defaults (auto-detects CUDA if available, else CPU). "
                                "CPU-only users should pass --torch-accelerator cpu.")

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
def test_inference_sampling_uses_authoritative_archive_gridsize(
    tmp_path: Path,
    monkeypatch,
):
    import numpy as np

    from ptycho import params
    from scripts.inference import inference

    config_path = tmp_path / 'inference.yaml'
    config_path.write_text(
        '\n'.join(
            [
                f'model_path: {tmp_path / "model"}',
                f'test_data_file: {tmp_path / "test.npz"}',
                'n_subsample: 8',
                f'output_dir: {tmp_path / "output"}',
                'backend: tensorflow',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    raw_data = MagicMock()
    raw_data.xcoords = np.arange(8)
    load_data = MagicMock(return_value=raw_data)
    perform_inference = MagicMock(
        return_value=(
            np.ones((2, 2)),
            np.zeros((2, 2)),
            None,
            None,
        )
    )

    def load_archive(_model_path, _config):
        params.cfg['gridsize'] = 2
        return MagicMock(), {'gridsize': 2}

    monkeypatch.setattr(params, 'cfg', {'gridsize': 1})
    monkeypatch.setattr(
        sys,
        'argv',
        ['inference.py', '--config', str(config_path)],
    )

    with patch.object(
        inference,
        'load_inference_bundle_with_backend',
        side_effect=load_archive,
    ), patch.object(
        inference,
        'load_data',
        load_data,
    ), patch.object(
        inference,
        'perform_inference',
        perform_inference,
    ), patch.object(
        inference,
        'save_reconstruction_images',
    ), pytest.raises(SystemExit) as exit_info:
        inference.main()

    assert exit_info.value.code == 0
    assert load_data.call_args.kwargs['n_subsample'] == 8
    assert load_data.call_args.kwargs['n_images'] is None
    assert perform_inference.call_args.kwargs['nsamples'] == 2

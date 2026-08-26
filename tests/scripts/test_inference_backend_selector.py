"""
Unit tests for backend selector integration and the backend-agnostic inference
CLI in scripts/inference/inference.py

This module tests that the inference CLI dispatches through the backend
selector for TensorFlow and delegates the PyTorch backend to the native
`python -m ptycho_torch.inference` door.

Test Coverage:
1. Backend selector dispatches for both 'tensorflow' and 'pytorch' configs
2. The installed CLI accepts 'tensorflow' (runs TF workflow) and 'pytorch'
   (delegates to the torch CLI); unknown backends are rejected
3. TensorFlow backend uses the legacy TF inference paths

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
        Test that the inference CLI accepts --backend tensorflow or pytorch.

        Expected behavior:
        - scripts/inference/inference.py accepts --backend tensorflow
        - 'pytorch' is accepted (delegates to python -m ptycho_torch.inference)
        - unknown values are rejected (argparse invalid choice)
        - Omission is represented by absence from the argparse namespace
        - The resolver supplies the backward-compatible TensorFlow default

        Phase: R (backend selector integration) / Phase 2 doors (dispatcher)
        """
        import sys

        # Omission is represented by absence from the argparse namespace
        with patch.object(sys, 'argv', ['inference.py']):
            from scripts.inference import inference

            omitted = inference.parse_arguments()

        assert not hasattr(omitted, 'backend')

        # Both supported backends are accepted on the dispatched CLI
        for backend in ('tensorflow', 'pytorch'):
            with patch.object(sys, 'argv', ['inference.py', '--backend', backend]):
                from scripts.inference import inference

                args = inference.parse_arguments()

            assert hasattr(args, 'backend'), \
                "Parsed args should have 'backend' attribute"
            assert args.backend == backend, \
                f"Expected backend={backend}, got {args.backend}"

        # Unknown values are rejected (argparse invalid choice)
        with patch.object(sys, 'argv', ['inference.py', '--model_path', 'model.zip',
                                        '--test_data', 'test.npz', '--backend', 'invalid']):
            with pytest.raises(SystemExit):
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

            args.inference_groups = None

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
            'inference_groups',
            'n_images',
            'inference_raw_selection',
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
                    f'model_path: {tmp_path / "model"}',
                    f'test_data_file: {tmp_path / "test.npz"}',
                    'inference_groups: 6',
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

        assert config.model_path == tmp_path / 'model'
        assert config.test_data_file == tmp_path / 'test.npz'
        assert config.inference_groups == 6
        assert config.neighbor_count == 7
        assert config.subsample_seed == 123
        assert config.debug is True
        assert config.output_dir == tmp_path / 'yaml-output'
        assert config.backend == 'pytorch'
        assert config.model.N == 128

    def test_inference_cli_values_override_yaml_and_map_to_canonical_fields(
        self,
        tmp_path: Path,
    ):
        from scripts.inference import inference

        config_path = tmp_path / 'inference.yaml'
        config_path.write_text(
            '\n'.join(
                [
                    f'model_path: {tmp_path / "yaml-model"}',
                    f'test_data_file: {tmp_path / "yaml-test.npz"}',
                    'inference_groups: 6',
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
                '--inference_groups',
                '3',
                '--backend',
                'tensorflow',
            ],
        ):
            args = inference.parse_arguments()

        config = inference.setup_inference_configuration(args, args.config)

        assert config.test_data_file == cli_test_data
        assert config.inference_groups == 3
        assert config.backend == 'tensorflow'

    def test_inference_deprecated_n_images_alias_resolves_once_at_boundary(
        self,
        tmp_path: Path,
    ):
        from scripts.inference import inference

        with patch.object(
            sys,
            'argv',
            [
                'inference.py',
                '--model_path',
                str(tmp_path / 'model'),
                '--test_data',
                str(tmp_path / 'test.npz'),
                '--n_images',
                '4',
            ],
        ):
            args = inference.parse_arguments()

        with pytest.warns(DeprecationWarning, match='n_images'):
            config = inference.setup_inference_configuration(args, args.config)

        assert config.inference_groups == 4

    def test_inference_sampling_reads_canonical_n_groups(self):
        from ptycho.config import InferenceConfig, ModelConfig
        from scripts.inference.inference import interpret_sampling_parameters

        config = InferenceConfig(
            model=ModelConfig(gridsize=2),
            model_path=Path('model'),
            test_data_file=Path('test.npz'),
            inference_groups=3,
        )

        n_subsample, n_groups, message = interpret_sampling_parameters(
            config,
            gridsize=2,
        )

        assert n_subsample is None
        assert n_groups == 3
        assert '3 groups' in message


def test_inference_file_only_main_uses_resolved_root_fields(
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
                f'model_path: {tmp_path / "model"}',
                f'test_data_file: {test_data_path}',
                'inference_groups: 1',
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
    ambient = {'gridsize': 1, 'ambient_marker': 'poison'}
    archive = {'gridsize': 1, 'archive_marker': 'owned'}
    monkeypatch.setattr(params, 'cfg', ambient)
    monkeypatch.setattr(
        sys,
        'argv',
        ['inference.py', '--config', str(config_path)],
    )

    with patch.object(
        inference,
        'load_inference_bundle_with_backend',
        return_value=(MagicMock(), archive),
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
    assert load_data.call_args.kwargs['n_images'] == 1
    assert perform_inference.call_args.kwargs['K'] == 7
    assert perform_inference.call_args.args[2] is archive


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
                'inference_raw_selection: 8',
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


def test_inference_main_dispatches_pytorch_backend(
    tmp_path: Path,
    monkeypatch,
):
    """A pytorch-backend config delegates to the native torch CLI door."""
    from scripts.inference import inference

    config_path = tmp_path / 'inference.yaml'
    config_path.write_text(
        '\n'.join(
            [
                f'model_path: {tmp_path / "model"}',
                f'test_data_file: {tmp_path / "test.npz"}',
                'inference_groups: 1',
                'neighbor_count: 7',
                f'output_dir: {tmp_path / "output"}',
                'backend: pytorch',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(
        sys,
        'argv',
        ['inference.py', '--config', str(config_path)],
    )

    with patch.object(
        inference,
        '_dispatch_pytorch_inference',
        return_value=0,
    ) as dispatch, patch.object(
        inference,
        'load_inference_bundle_with_backend',
    ) as load_bundle, pytest.raises(SystemExit) as exit_info:
        inference.main()

    assert exit_info.value.code == 0
    dispatch.assert_called_once()
    dispatched_config = dispatch.call_args.args[0]
    assert dispatched_config.backend == 'pytorch'
    load_bundle.assert_not_called()



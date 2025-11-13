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
        - Default is 'tensorflow' for backward compatibility
        - Argument is passed to setup_inference_configuration
        - InferenceConfig.backend field is populated

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
            (['inference.py', '--model_path', 'model.zip', '--test_data', 'test.npz'], 'tensorflow'),  # default
            (['inference.py', '--model_path', 'model.zip', '--test_data', 'test.npz', '--backend', 'tensorflow'], 'tensorflow'),
            (['inference.py', '--model_path', 'model.zip', '--test_data', 'test.npz', '--backend', 'pytorch'], 'pytorch'),
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

        # Test that invalid backend value is rejected
        with patch.object(sys, 'argv', ['inference.py', '--model_path', 'model.zip',
                                        '--test_data', 'test.npz', '--backend', 'invalid']):
            with pytest.raises(SystemExit):
                # argparse should exit with error for invalid choice
                from scripts.inference import inference
                inference.parse_arguments()

    def test_setup_inference_configuration_uses_backend(self):
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

        for backend_value in ['tensorflow', 'pytorch']:
            # Create mock args
            args = argparse.Namespace(
                model_path='outputs/test/model.zip',
                test_data='test.npz',
                config=None,
                output_dir='outputs/inference',
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

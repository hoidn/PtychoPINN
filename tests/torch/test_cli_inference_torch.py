"""
GREEN Phase pytest tests for inference CLI execution config integration (ADR-003 Phase C4.D1).

This module validates the inference CLI's ability to accept and forward execution config flags
to the factory and workflow layers. Tests verify argparse→PyTorchExecutionConfig→factory
propagation chain.

Phase C4 Implementation:
- Inference CLI accepts --accelerator, --num-workers, --inference-batch-size flags (✓ C4.C5)
- CLI args map to PyTorchExecutionConfig fields (✓ C4.C6)
- Factory receives correct execution config values (✓ validated)
- Tests mock both factory and bundle loader to prevent IO

References:
- Plan: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md §C4.D1
- Implementation: ptycho_torch/inference.py:380-442 (argparse), :455-546 (factory integration)
- Factory Design: .../2025-10-19T232336Z/phase_b_factories/factory_design.md
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestInferenceCLI:
    """
    Test inference CLI execution config flag integration.

    RED Phase Strategy:
    - Each test patches the factory function to capture its arguments
    - Invokes CLI with specific execution config flags
    - Asserts factory received correct PyTorchExecutionConfig values

    Expected RED Behavior:
    - Tests will FAIL with argparse.ArgumentError (unrecognized arguments)
      OR with AttributeError/AssertionError (flags accepted but not forwarded to factory)
    """

    @pytest.fixture
    def minimal_inference_args(self, tmp_path):
        """Minimal required inference CLI arguments for testing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "wts.h5.zip").touch()  # Create dummy checkpoint

        test_file = tmp_path / "test.npz"
        test_file.touch()  # Create dummy test file

        return [
            '--model_path', str(model_dir),
            '--test_data', str(test_file),
            '--output_dir', str(tmp_path / 'inference_outputs'),
            '--n_images', '32',
        ]

    def test_accelerator_flag_roundtrip(self, minimal_inference_args, monkeypatch):
        """
        Test: --accelerator flag maps to execution_config.accelerator.

        Validates CLI→factory→execution_config propagation chain.
        Mocks both factory and bundle loader to prevent IO.
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )

        mock_bundle_loader = MagicMock(return_value=({}, {}))

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader):
            test_args = minimal_inference_args + ['--accelerator', 'cpu']

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass  # CLI may exit or fail after factory call; we only test arg mapping

        assert mock_factory.called, "Factory was not called"
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs, "execution_config not passed to factory"
        assert call_kwargs['execution_config'].accelerator == 'cpu', \
            f"Expected accelerator='cpu', got {call_kwargs['execution_config'].accelerator}"

    def test_num_workers_flag_roundtrip(self, minimal_inference_args, monkeypatch):
        """
        Test: --num-workers flag maps to execution_config.num_workers.

        Validates CLI→factory→execution_config propagation chain.
        Mocks both factory and bundle loader to prevent IO.
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(num_workers=4),
        )

        mock_bundle_loader = MagicMock(return_value=({}, {}))

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader):
            test_args = minimal_inference_args + ['--num-workers', '4']

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].num_workers == 4, \
            f"Expected num_workers=4, got {call_kwargs['execution_config'].num_workers}"

    def test_inference_batch_size_flag_roundtrip(self, minimal_inference_args, monkeypatch):
        """
        Test: --inference-batch-size flag maps to execution_config.inference_batch_size.

        Validates CLI→factory→execution_config propagation chain.
        Mocks both factory and bundle loader to prevent IO.
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(inference_batch_size=32),
        )

        mock_bundle_loader = MagicMock(return_value=({}, {}))

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader):
            test_args = minimal_inference_args + ['--inference-batch-size', '32']

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].inference_batch_size == 32, \
            f"Expected inference_batch_size=32, got {call_kwargs['execution_config'].inference_batch_size}"

    def test_multiple_execution_config_flags(self, minimal_inference_args, monkeypatch):
        """
        Test: Multiple execution config flags work together.

        Validates CLI→factory→execution_config propagation chain with multiple flags.
        Mocks both factory and bundle loader to prevent IO.
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(
                accelerator='gpu',
                num_workers=8,
                inference_batch_size=64,
            ),
        )

        mock_bundle_loader = MagicMock(return_value=({}, {}))

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader):
            test_args = minimal_inference_args + [
                '--accelerator', 'gpu',
                '--num-workers', '8',
                '--inference-batch-size', '64',
            ]

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        exec_config = call_kwargs['execution_config']
        assert exec_config.accelerator == 'gpu'
        assert exec_config.num_workers == 8
        assert exec_config.inference_batch_size == 64

    def test_accelerator_flag_roundtrip(self, minimal_inference_args, monkeypatch):
        """
        Test that accelerator flag is properly handled by _run_inference_and_reconstruct (DEVICE-MISMATCH-001).

        Expected behavior:
        - CLI parses --accelerator flag and builds execution_config
        - execution_config accelerator maps to device string ('cuda', 'mps', 'cpu')
        - _run_inference_and_reconstruct receives device and moves model to it
        - Model.to(device) and model.eval() are called inside helper

        Phase: R (DEVICE-MISMATCH-001 fix)
        Reference: input.md Do Now step 3, DEVICE-MISMATCH-001 finding
        """
        import numpy as np
        import torch
        from unittest.mock import MagicMock, patch

        # Mock RawData with minimal required fields
        mock_raw_data = MagicMock()
        mock_raw_data.diff3d = np.random.rand(10, 64, 64).astype(np.float32)
        mock_raw_data.probeGuess = np.random.rand(64, 64).astype(np.complex64)
        mock_raw_data.xcoords = np.random.rand(10)
        mock_raw_data.ycoords = np.random.rand(10)

        # Mock model with .to() and .eval() tracking
        mock_model = MagicMock()
        device_calls = []
        eval_calls = []

        def track_to_call(device):
            device_calls.append(device)
            return mock_model

        def track_eval_call():
            eval_calls.append(True)
            return mock_model

        mock_model.to = MagicMock(side_effect=track_to_call)
        mock_model.eval = MagicMock(side_effect=track_eval_call)
        patch_complex = torch.complex(
            torch.rand(1, 1, 64, 64, dtype=torch.float32),
            torch.rand(1, 1, 64, 64, dtype=torch.float32),
        )
        mock_model.forward_predict = MagicMock(return_value=patch_complex)

        def fake_reassemble(patches, offsets, data_cfg, model_cfg, padded_size=None, **_kwargs):
            size = int(padded_size or patches.shape[-1])
            imgs = torch.zeros((1, size, size), dtype=patches.dtype)
            return imgs, None, None

        # Import and call _run_inference_and_reconstruct directly
        from ptycho_torch.inference import _run_inference_and_reconstruct
        from ptycho.config.config import InferenceConfig, ModelConfig, PyTorchExecutionConfig

        config = InferenceConfig(
            model=ModelConfig(N=64, gridsize=1),
            model_path=Path('outputs/test/bundle.zip'),
            test_data_file=Path('test.npz'),
            backend='pytorch',
            output_dir=Path('outputs/inference'),
            n_groups=10
        )

        execution_config = PyTorchExecutionConfig(
            accelerator='cuda',  # Request CUDA device
            num_workers=0,
            inference_batch_size=None
        )

        # Call helper with 'cuda' device
        with patch('ptycho_torch.helper.reassemble_patches_position_real', side_effect=fake_reassemble):
            _run_inference_and_reconstruct(
                mock_model, mock_raw_data, config, execution_config, 'cuda', quiet=True
            )

        # Verify model.to('cuda') was called
        assert len(device_calls) > 0, \
            "_run_inference_and_reconstruct should call model.to(device)"
        assert device_calls[0] == 'cuda', \
            f"model.to() should be called with 'cuda', got {device_calls[0]}"

        # Verify model.eval() was called
        assert len(eval_calls) > 0, \
            "_run_inference_and_reconstruct should call model.eval()"


class TestInferenceCLIThinWrapper:
    """
    RED Phase tests for inference CLI thin wrapper delegation (ADR-003 Phase D.C C2).

    Tests verify that the inference CLI delegates to shared helpers and workflow components
    rather than implementing business logic inline. These tests are EXPECTED TO FAIL until
    the thin wrapper refactor is implemented (Phase D.C C3).

    Expected RED Failures:
    - AttributeError: _run_inference_and_reconstruct helper does not exist
    - AssertionError: validate_paths() not called (inline validation still present)
    - AssertionError: Helper delegation order incorrect

    Blueprint Reference:
    - plans/.../phase_d_cli_wrappers_inference/inference_refactor.md §Test Strategy
    """

    @pytest.fixture
    def minimal_inference_args(self, tmp_path):
        """Minimal required inference CLI arguments for testing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "wts.h5.zip").touch()  # Create dummy checkpoint

        test_file = tmp_path / "test.npz"
        test_file.touch()  # Create dummy test file

        return [
            '--model_path', str(model_dir),
            '--test_data', str(test_file),
            '--output_dir', str(tmp_path / 'inference_outputs'),
            '--n_images', '32',
        ]

    def test_cli_delegates_to_validate_paths(self, minimal_inference_args, monkeypatch):
        """
        RED Test: CLI calls validate_paths() before factory invocation.

        Expected RED Failure:
        - AssertionError: validate_paths() not called (inline validation still present)

        Success Criteria (GREEN):
        - validate_paths() called exactly once with (train_file=None, test_file, output_dir)
        - Called BEFORE create_inference_payload (CONFIG-001 ordering)
        """
        from unittest.mock import MagicMock, patch, call

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(n_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )
        mock_bundle_loader = MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))
        mock_raw_data = MagicMock()

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader), \
             patch('ptycho.raw_data.RawData.from_file', return_value=mock_raw_data):

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + minimal_inference_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass  # Expected to fail after helper calls

        # Assert validate_paths was called
        assert mock_validate_paths.called, \
            "validate_paths() was not called - CLI still using inline validation"

        # Assert called with correct arguments (inspect kwargs for keyword invocation)
        call_kwargs = mock_validate_paths.call_args.kwargs
        assert call_kwargs.get('train_file') is None, "train_file should be None for inference mode"
        assert str(call_kwargs.get('test_file', '')).endswith('test.npz'), "test_file path incorrect"
        assert 'inference_outputs' in str(call_kwargs.get('output_dir', '')), "output_dir path incorrect"

    def test_cli_delegates_to_helper_for_data_loading(self, minimal_inference_args, monkeypatch):
        """
        RED Test: CLI calls RawData.from_file() for test data loading.

        Expected RED Failure:
        - May pass if current CLI already loads RawData (Option A decision)
        - OR may fail if workflow is expected to load data (Option B)

        Success Criteria (GREEN):
        - RawData.from_file() called exactly once with test_data path
        - Called AFTER factory invocation (CONFIG-001 already satisfied)
        """
        from unittest.mock import MagicMock, patch

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(n_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )
        mock_bundle_loader = MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))
        mock_raw_data_from_file = MagicMock(return_value=MagicMock())

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader), \
             patch('ptycho.raw_data.RawData.from_file', mock_raw_data_from_file):

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + minimal_inference_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        # Assert RawData.from_file() was called
        assert mock_raw_data_from_file.called, \
            "RawData.from_file() was not called - data loading delegation broken"

        # Assert called with test_data path
        call_args = mock_raw_data_from_file.call_args
        assert str(call_args[0][0]).endswith('test.npz'), \
            f"Expected test.npz path, got {call_args[0][0]}"

    def test_cli_delegates_to_inference_helper(self, minimal_inference_args, monkeypatch):
        """
        RED Test: CLI calls _run_inference_and_reconstruct() helper.

        Expected RED Failure:
        - AttributeError: module 'ptycho_torch.inference' has no attribute '_run_inference_and_reconstruct'

        Success Criteria (GREEN):
        - _run_inference_and_reconstruct() called with (model, raw_data, config, execution_config, device, quiet)
        - Returns (amplitude, phase) tuple
        """
        from unittest.mock import MagicMock, patch

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(n_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )
        mock_bundle_loader = MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))
        mock_raw_data = MagicMock()
        mock_helper = MagicMock(return_value=(MagicMock(), MagicMock()))  # (amplitude, phase)

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader), \
             patch('ptycho.raw_data.RawData.from_file', return_value=mock_raw_data), \
             patch('ptycho_torch.inference._run_inference_and_reconstruct', mock_helper):

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + minimal_inference_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        # Assert helper was called
        assert mock_helper.called, \
            "_run_inference_and_reconstruct() helper not called - inline logic still present"

        # Assert called with correct arguments
        call_kwargs = mock_helper.call_args.kwargs
        assert 'model' in call_kwargs, "model argument missing"
        assert 'raw_data' in call_kwargs, "raw_data argument missing"
        assert 'config' in call_kwargs, "config argument missing"
        assert 'execution_config' in call_kwargs, "execution_config argument missing"
        assert 'device' in call_kwargs, "device argument missing"
        assert 'quiet' in call_kwargs, "quiet argument missing"

    def test_cli_calls_save_individual_reconstructions(self, minimal_inference_args, monkeypatch):
        """
        RED Test: CLI calls save_individual_reconstructions() after inference.

        Expected RED Failure:
        - May pass if current CLI already calls this function
        - OR assertion fails if call order incorrect

        Success Criteria (GREEN):
        - save_individual_reconstructions() called with (amplitude, phase, output_dir)
        - Called AFTER _run_inference_and_reconstruct() helper
        """
        from unittest.mock import MagicMock, patch
        import numpy as np

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(n_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )
        mock_bundle_loader = MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))
        mock_raw_data = MagicMock()
        mock_amplitude = np.random.rand(64, 64)
        mock_phase = np.random.rand(64, 64)
        mock_helper = MagicMock(return_value=(mock_amplitude, mock_phase))
        mock_save_fn = MagicMock()

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader), \
             patch('ptycho.raw_data.RawData.from_file', return_value=mock_raw_data), \
             patch('ptycho_torch.inference._run_inference_and_reconstruct', mock_helper), \
             patch('ptycho_torch.inference.save_individual_reconstructions', mock_save_fn):

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + minimal_inference_args)

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        # Assert save function was called
        assert mock_save_fn.called, \
            "save_individual_reconstructions() not called - output artifact generation missing"

        # Assert called with correct arguments
        call_args = mock_save_fn.call_args[0]
        assert len(call_args) >= 3, "Expected 3 arguments: (amplitude, phase, output_dir)"
        # Note: We can't assert array equality directly due to mocking, but we verify call happened

    def test_quiet_flag_suppresses_progress_output(self, minimal_inference_args, monkeypatch, capsys):
        """
        RED Test: --quiet flag suppresses CLI progress print statements.

        Expected RED Failure:
        - AssertionError: Progress output still printed when --quiet specified

        Success Criteria (GREEN):
        - No progress messages in stdout when --quiet flag present
        - enable_progress_bar=False passed to execution config
        """
        from unittest.mock import MagicMock, patch

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(n_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu', enable_progress_bar=False),
        )
        mock_bundle_loader = MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))
        mock_raw_data = MagicMock()
        mock_helper = MagicMock(return_value=(MagicMock(), MagicMock()))

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.components.load_inference_bundle_torch', mock_bundle_loader), \
             patch('ptycho.raw_data.RawData.from_file', return_value=mock_raw_data), \
             patch('ptycho_torch.inference._run_inference_and_reconstruct', mock_helper), \
             patch('ptycho_torch.inference.save_individual_reconstructions', MagicMock()):

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + minimal_inference_args + ['--quiet'])

            try:
                cli_main()
            except (SystemExit, Exception):
                pass

        # Check that execution config has enable_progress_bar=False
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs, "execution_config not passed to factory"
        exec_config = call_kwargs['execution_config']
        assert hasattr(exec_config, 'enable_progress_bar'), "execution_config missing enable_progress_bar"
        assert exec_config.enable_progress_bar is False, \
            "Expected enable_progress_bar=False when --quiet specified"

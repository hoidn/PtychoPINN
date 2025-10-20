"""
RED Phase pytest scaffolds for inference CLI execution config integration (ADR-003 Phase C4.B2).

This module tests the inference CLI's ability to accept and forward execution config flags
to the factory and workflow layers. Tests are expected to FAIL in RED phase because
the CLI implementation does not yet exist.

Phase C4 Requirements:
- Inference CLI accepts --accelerator, --num-workers, --inference-batch-size flags
- CLI args map to PyTorchExecutionConfig fields
- Factory receives correct execution config values
- Workflow helpers receive execution config from factory payload

References:
- Plan: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md §C4.B2
- Argparse Schema: .../phase_c4_cli_integration/argparse_schema.md
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
        RED Test: --accelerator flag maps to execution_config.accelerator.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --accelerator cpu
        OR
        - AssertionError: execution_config.accelerator != 'cpu'
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory):
            test_args = minimal_inference_args + ['--accelerator', 'cpu']

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called, "Factory was not called"
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs, "execution_config not passed to factory"
        assert call_kwargs['execution_config'].accelerator == 'cpu', \
            f"Expected accelerator='cpu', got {call_kwargs['execution_config'].accelerator}"

    def test_num_workers_flag_roundtrip(self, minimal_inference_args, monkeypatch):
        """
        RED Test: --num-workers flag maps to execution_config.num_workers.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --num-workers 4
        OR
        - AssertionError: execution_config.num_workers != 4
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            execution_config=MagicMock(num_workers=4),
        )

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory):
            test_args = minimal_inference_args + ['--num-workers', '4']

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].num_workers == 4, \
            f"Expected num_workers=4, got {call_kwargs['execution_config'].num_workers}"

    def test_inference_batch_size_flag_roundtrip(self, minimal_inference_args, monkeypatch):
        """
        RED Test: --inference-batch-size flag maps to execution_config.inference_batch_size.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --inference-batch-size 32
        OR
        - AssertionError: execution_config.inference_batch_size != 32
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            execution_config=MagicMock(inference_batch_size=32),
        )

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory):
            test_args = minimal_inference_args + ['--inference-batch-size', '32']

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].inference_batch_size == 32, \
            f"Expected inference_batch_size=32, got {call_kwargs['execution_config'].inference_batch_size}"

    def test_multiple_execution_config_flags(self, minimal_inference_args, monkeypatch):
        """
        RED Test: Multiple execution config flags work together.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments (any of the new flags)
        OR
        - AssertionError: execution_config fields do not match expected values
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(),
            execution_config=MagicMock(
                accelerator='gpu',
                num_workers=8,
                inference_batch_size=64,
            ),
        )

        with patch('ptycho_torch.config_factory.create_inference_payload', mock_factory):
            test_args = minimal_inference_args + [
                '--accelerator', 'gpu',
                '--num-workers', '8',
                '--inference-batch-size', '64',
            ]

            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        exec_config = call_kwargs['execution_config']
        assert exec_config.accelerator == 'gpu'
        assert exec_config.num_workers == 8
        assert exec_config.inference_batch_size == 64


# RED Phase Note:
# These tests are EXPECTED TO FAIL because:
# 1. The CLI argparse definition does not yet include the new flags
# 2. The CLI does not yet instantiate PyTorchExecutionConfig from parsed args
# 3. The CLI does not yet pass execution_config to create_inference_payload()
#
# Phase C4.C implementation will:
# 1. Add argparse arguments for --accelerator, --num-workers, --inference-batch-size
# 2. Instantiate PyTorchExecutionConfig from args
# 3. Pass execution_config to factory
# 4. Turn these RED tests GREEN

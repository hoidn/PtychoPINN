"""
RED Phase pytest scaffolds for training CLI execution config integration (ADR-003 Phase C4.B1).

This module tests the training CLI's ability to accept and forward execution config flags
to the factory and workflow layers. Tests are expected to FAIL in RED phase because
the CLI implementation does not yet exist.

Phase C4 Requirements:
- Training CLI accepts --accelerator, --deterministic, --num-workers, --learning-rate flags
- CLI args map to PyTorchExecutionConfig fields
- Factory receives correct execution config values
- Workflow helpers receive execution config from factory payload

References:
- Plan: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md §C4.B1
- Argparse Schema: .../phase_c4_cli_integration/argparse_schema.md
- Factory Design: .../2025-10-19T232336Z/phase_b_factories/factory_design.md
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestExecutionConfigCLI:
    """
    Test training CLI execution config flag integration.

    RED Phase Strategy:
    - Each test patches the factory function to capture its arguments
    - Invokes CLI with specific execution config flags
    - Asserts factory received correct PyTorchExecutionConfig values

    Expected RED Behavior:
    - Tests will FAIL with argparse.ArgumentError (unrecognized arguments)
      OR with AttributeError/AssertionError (flags accepted but not forwarded to factory)
    """

    @pytest.fixture
    def minimal_train_args(self, tmp_path):
        """Minimal required training CLI arguments for testing."""
        train_file = tmp_path / "train.npz"
        train_file.touch()  # Create dummy file
        return [
            '--train_data_file', str(train_file),
            '--output_dir', str(tmp_path / 'outputs'),
            '--n_images', '64',
            '--max_epochs', '2',
        ]

    def test_accelerator_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        RED Test: --accelerator flag maps to execution_config.accelerator.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --accelerator cpu
        OR
        - AssertionError: execution_config.accelerator != 'cpu'
        """
        # Patch factory to capture execution_config argument
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            data_config=MagicMock(),
            pt_model_config=MagicMock(),
            pt_training_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),  # Expected value
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            # Simulate CLI invocation with --accelerator cpu
            test_args = minimal_train_args + ['--accelerator', 'cpu']

            # Import and invoke CLI main (will fail in RED phase)
            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass  # CLI may exit; catch to inspect mock calls

        # Assert factory was called with execution_config containing accelerator='cpu'
        assert mock_factory.called, "Factory was not called"
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs, "execution_config not passed to factory"
        assert call_kwargs['execution_config'].accelerator == 'cpu', \
            f"Expected accelerator='cpu', got {call_kwargs['execution_config'].accelerator}"

    def test_deterministic_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        RED Test: --deterministic flag maps to execution_config.deterministic=True.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --deterministic
        OR
        - AssertionError: execution_config.deterministic != True
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(deterministic=True),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--deterministic']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].deterministic is True, \
            "Expected deterministic=True"

    def test_no_deterministic_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        RED Test: --no-deterministic flag maps to execution_config.deterministic=False.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --no-deterministic
        OR
        - AssertionError: execution_config.deterministic != False
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(deterministic=False),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--no-deterministic']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].deterministic is False, \
            "Expected deterministic=False with --no-deterministic"

    def test_num_workers_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        RED Test: --num-workers flag maps to execution_config.num_workers.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --num-workers 4
        OR
        - AssertionError: execution_config.num_workers != 4
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(num_workers=4),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--num-workers', '4']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].num_workers == 4, \
            f"Expected num_workers=4, got {call_kwargs['execution_config'].num_workers}"

    def test_learning_rate_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        RED Test: --learning-rate flag maps to execution_config.learning_rate.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --learning-rate 5e-4
        OR
        - AssertionError: execution_config.learning_rate != 5e-4
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(learning_rate=5e-4),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--learning-rate', '5e-4']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert abs(call_kwargs['execution_config'].learning_rate - 5e-4) < 1e-10, \
            f"Expected learning_rate=5e-4, got {call_kwargs['execution_config'].learning_rate}"

    def test_multiple_execution_config_flags(self, minimal_train_args, monkeypatch):
        """
        RED Test: Multiple execution config flags work together.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments (any of the new flags)
        OR
        - AssertionError: execution_config fields do not match expected values
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(
                accelerator='gpu',
                deterministic=False,
                num_workers=8,
                learning_rate=1e-3,
            ),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + [
                '--accelerator', 'gpu',
                '--no-deterministic',
                '--num-workers', '8',
                '--learning-rate', '1e-3',
            ]

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        exec_config = call_kwargs['execution_config']
        assert exec_config.accelerator == 'gpu'
        assert exec_config.deterministic is False
        assert exec_config.num_workers == 8
        assert abs(exec_config.learning_rate - 1e-3) < 1e-10


# RED Phase Note:
# These tests are EXPECTED TO FAIL because:
# 1. The CLI argparse definition does not yet include the new flags
# 2. The CLI does not yet instantiate PyTorchExecutionConfig from parsed args
# 3. The CLI does not yet pass execution_config to create_training_payload()
#
# Phase C4.C implementation will:
# 1. Add argparse arguments for --accelerator, --deterministic, --num-workers, --learning-rate
# 2. Instantiate PyTorchExecutionConfig from args
# 3. Pass execution_config to factory
# 4. Turn these RED tests GREEN

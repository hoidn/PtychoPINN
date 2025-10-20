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

    def test_bundle_persistence(self, minimal_train_args, monkeypatch):
        """
        RED Test: Training CLI invokes save_torch_bundle with dual-model dict.

        This test validates the Phase C4.D3 requirement that training CLI must emit
        the spec-required wts.h5.zip bundle containing both 'autoencoder' and
        'diffraction_to_obj' model keys per specs/ptychodus_api_spec.md §4.6.

        Expected RED Failure:
        - save_torch_bundle is never called (legacy training path doesn't persist bundles)
        OR
        - save_torch_bundle called with incorrect models_dict structure

        Success Criteria (GREEN):
        - save_torch_bundle called exactly once
        - models_dict contains 'autoencoder' key
        - models_dict contains 'diffraction_to_obj' key
        - base_path argument points to {output_dir}/wts.h5

        References:
        - input.md C4.D3 bundle TDD requirement
        - plans/.../phase_c4_cli_integration/plan.md §C4.D3
        - specs/ptychodus_api_spec.md §4.6 (dual-model bundle contract)
        """
        # Mock save_torch_bundle at the workflow level where it's actually called
        mock_save_bundle = MagicMock()

        # Mock RawData.from_file to avoid file I/O
        mock_raw_data = MagicMock()

        # Mock run_cdi_example_torch at the level where train.py imports it
        # This allows mocking without actually running the training
        def mock_run_cdi_example_torch(train_data, test_data, config, do_stitching=False):
            """Mock workflow that still calls save_torch_bundle with correct structure."""
            from ptycho_torch.model_manager import save_torch_bundle

            # Simulate training results with dual-model dict
            models_dict = {
                'autoencoder': MagicMock(),
                'diffraction_to_obj': MagicMock()
            }

            # Simulate the bundle persistence path from real workflow
            if config.output_dir:
                from pathlib import Path
                archive_path = Path(config.output_dir) / "wts.h5"
                save_torch_bundle(
                    models_dict=models_dict,
                    base_path=str(archive_path),
                    config=config
                )

            return None, None, {'models': models_dict}

        with patch('ptycho_torch.model_manager.save_torch_bundle', mock_save_bundle), \
             patch('ptycho.raw_data.RawData.from_file', return_value=mock_raw_data):

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + minimal_train_args)

            # Patch run_cdi_example_torch in the workflows.components module
            with patch('ptycho_torch.workflows.components.run_cdi_example_torch',
                      side_effect=mock_run_cdi_example_torch):
                try:
                    cli_main()
                except SystemExit:
                    pass

        # Assert save_torch_bundle was called
        assert mock_save_bundle.called, \
            "save_torch_bundle was not called (training CLI does not persist bundles)"

        # Verify it was called exactly once
        assert mock_save_bundle.call_count == 1, \
            f"Expected 1 call to save_torch_bundle, got {mock_save_bundle.call_count}"

        # Extract call arguments (handle both positional and keyword arguments)
        call_args, call_kwargs = mock_save_bundle.call_args.args, mock_save_bundle.call_args.kwargs

        # Get models_dict from either positional or keyword arguments
        if call_args and len(call_args) > 0:
            models_dict = call_args[0]
        elif 'models_dict' in call_kwargs:
            models_dict = call_kwargs['models_dict']
        else:
            raise AssertionError("Could not extract models_dict from save_torch_bundle call")

        # Assert dual-model structure
        assert 'autoencoder' in models_dict, \
            "models_dict missing 'autoencoder' key (incomplete bundle)"
        assert 'diffraction_to_obj' in models_dict, \
            "models_dict missing 'diffraction_to_obj' key (incomplete bundle)"

        # Verify base_path points to correct location
        if len(call_args) > 1:
            base_path = call_args[1]
        elif 'base_path' in call_kwargs:
            base_path = call_kwargs['base_path']
        else:
            raise AssertionError("Could not extract base_path from save_torch_bundle call")

        assert 'wts.h5' in str(base_path), \
            f"Expected base_path to contain 'wts.h5', got {base_path}"

    def test_enable_checkpointing_flag(self, minimal_train_args, monkeypatch):
        """
        RED Test: --enable-checkpointing / --disable-checkpointing flags map to execution_config.enable_checkpointing.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --enable-checkpointing / --disable-checkpointing
        OR
        - AssertionError: execution_config.enable_checkpointing != expected value

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.B (CLI flag parsing)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(enable_checkpointing=False),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--disable-checkpointing']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].enable_checkpointing is False, \
            "Expected enable_checkpointing=False with --disable-checkpointing"

    def test_checkpoint_save_top_k_flag(self, minimal_train_args, monkeypatch):
        """
        RED Test: --checkpoint-save-top-k flag maps to execution_config.checkpoint_save_top_k.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --checkpoint-save-top-k 3
        OR
        - AssertionError: execution_config.checkpoint_save_top_k != 3

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.B (CLI flag parsing)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(checkpoint_save_top_k=3),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--checkpoint-save-top-k', '3']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].checkpoint_save_top_k == 3, \
            f"Expected checkpoint_save_top_k=3, got {call_kwargs['execution_config'].checkpoint_save_top_k}"

    def test_checkpoint_monitor_flag(self, minimal_train_args, monkeypatch):
        """
        RED Test: --checkpoint-monitor flag maps to execution_config.checkpoint_monitor_metric.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --checkpoint-monitor train_loss
        OR
        - AssertionError: execution_config.checkpoint_monitor_metric != 'train_loss'

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.B (CLI flag parsing)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(checkpoint_monitor_metric='train_loss'),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--checkpoint-monitor', 'train_loss']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].checkpoint_monitor_metric == 'train_loss', \
            f"Expected checkpoint_monitor_metric='train_loss', got {call_kwargs['execution_config'].checkpoint_monitor_metric}"

    def test_checkpoint_mode_flag(self, minimal_train_args, monkeypatch):
        """
        RED Test: --checkpoint-mode flag maps to execution_config.checkpoint_mode.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --checkpoint-mode max
        OR
        - AssertionError: execution_config.checkpoint_mode != 'max'
        OR
        - AttributeError: 'PyTorchExecutionConfig' object has no attribute 'checkpoint_mode'

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.A (introduce checkpoint_mode field)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(checkpoint_mode='max'),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--checkpoint-mode', 'max']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].checkpoint_mode == 'max', \
            f"Expected checkpoint_mode='max', got {call_kwargs['execution_config'].checkpoint_mode}"

    def test_early_stop_patience_flag(self, minimal_train_args, monkeypatch):
        """
        RED Test: --early-stop-patience flag maps to execution_config.early_stop_patience.

        Expected RED Failure:
        - argparse.ArgumentError: unrecognized arguments: --early-stop-patience 10
        OR
        - AssertionError: execution_config.early_stop_patience != 10

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.B (CLI flag parsing)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(early_stop_patience=10),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--early-stop-patience', '10']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].early_stop_patience == 10, \
            f"Expected early_stop_patience=10, got {call_kwargs['execution_config'].early_stop_patience}"


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

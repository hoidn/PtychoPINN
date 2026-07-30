"""
GREEN coverage for native Torch training CLI configuration and workflow integration.

This module verifies that native CLI arguments form an unresolved
``ExecutionRequest`` and a separate Torch training patch, reach
``create_training_payload``, and forward the resulting explicit payload through
``resolved_payload`` to the shared workflow. It also covers model-bundle
persistence, checkpoint and logger controls, and patch-stat instrumentation.

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
    Regression coverage for native training CLI configuration routing.

    Tests invoke ``cli_main`` across execution, optimization, checkpoint, and
    logger flags, then inspect the unresolved factory request or the explicit
    payload forwarded to the shared workflow.
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
        Regression intent: --accelerator maps to execution_config.accelerator.

        The CLI must preserve the explicit value in the factory request.
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

            # Import and invoke the current CLI entry point.
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
        assert call_kwargs['execution_config'].values['accelerator'] == 'cpu'

    def test_deterministic_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --deterministic maps to deterministic=True.

        The CLI must preserve explicit boolean presence in the factory request.
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
        assert call_kwargs['execution_config'].values['deterministic'] is True, \
            "Expected deterministic=True"

    def test_no_deterministic_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --no-deterministic maps to deterministic=False.

        The CLI must preserve explicit boolean presence in the factory request.
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
        assert call_kwargs['execution_config'].values['deterministic'] is False, \
            "Expected deterministic=False with --no-deterministic"

    def test_num_workers_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --num-workers maps to execution_config.num_workers.

        The CLI must preserve the explicit worker count in the factory request.
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
        assert call_kwargs['execution_config'].values['num_workers'] == 4

    def test_learning_rate_flag_roundtrip(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --learning-rate maps to the TrainingConfig patch.

        Optimization values remain separate from the execution request.
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
        assert call_kwargs['overrides']['learning_rate'] == pytest.approx(5e-4)
        assert 'learning_rate' not in call_kwargs['execution_config'].values

    def test_multiple_execution_config_flags(self, minimal_train_args, monkeypatch):
        """
        Regression intent: multiple execution flags retain all explicit values.

        The combined factory request must not drop or overwrite a flag.
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
        request = call_kwargs['execution_config']
        assert request.values['accelerator'] == 'gpu'
        assert request.values['deterministic'] is False
        assert request.values['num_workers'] == 8
        assert 'learning_rate' not in request.values
        assert call_kwargs['overrides']['learning_rate'] == pytest.approx(1e-3)

    def test_native_training_execution_request_preserves_explicit_options(
        self,
        minimal_train_args,
        monkeypatch,
    ):
        """The native CLI passes raw suppliedness and a request to its factory."""
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args as real_request_builder,
        )
        from ptycho_torch.execution_request import ExecutionRequest
        from ptycho_torch.train import cli_main

        argv = minimal_train_args + [
            '--accelerator=cpu',
            '--device', 'cuda',
            '--deterministic',
            '--no-deterministic',
            '--num-workers=2',
            '--learning-rate', '0.002',
            '--scheduler=ReduceLROnPlateau',
            '--accumulate-grad-batches', '3',
            '--logger=none',
            '--quiet',
            '--disable_mlflow',
            '--enable-checkpointing',
            '--disable-checkpointing',
            '--checkpoint-save-top-k=0',
            '--checkpoint-monitor', 'train_loss',
            '--checkpoint-mode=max',
            '--early-stop-patience', '9',
        ]
        monkeypatch.setattr('sys.argv', ['train.py', *argv])

        with patch(
            'ptycho_torch.cli.shared.build_execution_request_from_args',
            wraps=real_request_builder,
        ) as request_builder, patch(
            'ptycho_torch.config_factory.create_training_payload',
            side_effect=RuntimeError('stop after request capture'),
        ) as factory, pytest.raises(SystemExit):
            cli_main()

        request_builder.assert_called_once()
        assert request_builder.call_args.kwargs == {
            'mode': 'training',
            'explicit_options': tuple(argv),
            'lane': 'native-training',
        }
        request = factory.call_args.kwargs['execution_config']
        assert isinstance(request, ExecutionRequest)
        assert request.explicit_fields == frozenset(
            {
                'accelerator',
                'deterministic',
                'num_workers',
                'logger_backend',
                'enable_progress_bar',
                'enable_checkpointing',
                'checkpoint_save_top_k',
                'checkpoint_monitor_metric',
                'checkpoint_mode',
                'early_stop_patience',
            }
        )
        assert request.values['accelerator'] == 'cpu'
        assert request.values['deterministic'] is False
        assert request.values['num_workers'] == 2
        assert set(request.values).isdisjoint(
            {'learning_rate', 'scheduler', 'accum_steps'}
        )
        assert request.values['logger_backend'] is None
        assert request.values['enable_progress_bar'] is False
        assert request.values['enable_checkpointing'] is False
        assert request.values['checkpoint_save_top_k'] == 0
        assert request.values['checkpoint_monitor_metric'] == 'train_loss'
        assert request.values['checkpoint_mode'] == 'max'
        assert request.values['early_stop_patience'] == 9
        overrides = factory.call_args.kwargs['overrides']
        assert {
            key: overrides[key]
            for key in ('learning_rate', 'scheduler', 'accum_steps')
        } == {
            'learning_rate': 0.002,
            'scheduler': 'ReduceLROnPlateau',
            'accum_steps': 3,
        }

    def test_native_training_execution_explicit_checkpoint_help_is_current(
        self,
        monkeypatch,
        capsys,
    ):
        """The native help must not advertise the rejected save-all spelling."""
        from ptycho_torch.train import cli_main

        monkeypatch.setattr('sys.argv', ['train.py', '--help'])
        with pytest.raises(SystemExit) as exit_info:
            cli_main()

        assert exit_info.value.code == 0
        assert '-1 to save all' not in capsys.readouterr().out

    def test_bundle_persistence(self, minimal_train_args, monkeypatch):
        """
        Regression intent: the training CLI persists the dual-model bundle.

        This test validates the Phase C4.D3 requirement that training CLI must emit
        the spec-required wts.h5.zip bundle containing both 'autoencoder' and
        'diffraction_to_obj' model keys per specs/ptychodus_api_spec.md §4.6.

        Verified behavior:
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

        # Keep mmap ingestion at the CLI boundary without constructing real maps.
        mock_train_dataset = MagicMock()

        # Import the workflow before patching the provider so its module-level
        # save_torch_bundle binding cannot retain this test's mock afterward.
        import ptycho_torch.workflows.components

        # Mock run_cdi_example_torch at the level where train.py imports it
        # This allows mocking without actually running the training
        def mock_run_cdi_example_torch(
            train_data,
            test_data,
            config,
            do_stitching=False,
            execution_config=None,
            overrides=None,
            resolved_payload=None,
        ):
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
             patch(
                 'ptycho_torch.cli.mmap_ingestion.build_cli_mmap_dataset',
                 side_effect=[mock_train_dataset],
             ):

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

        expected_base_path = Path(minimal_train_args[3]) / 'wts.h5'
        assert Path(base_path) == expected_base_path, \
            f"Expected base_path {expected_base_path}, got {base_path}"

    def test_enable_checkpointing_flag(self, minimal_train_args, monkeypatch):
        """
        Regression intent: checkpoint toggles map to enable_checkpointing.

        The CLI must preserve the explicitly selected boolean value.

        References:
        - input.md EB1.E (checkpoint controls)
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
        assert call_kwargs['execution_config'].values['enable_checkpointing'] is False, \
            "Expected enable_checkpointing=False with --disable-checkpointing"

    def test_checkpoint_save_top_k_flag(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --checkpoint-save-top-k maps to its execution field.

        The CLI must preserve the explicit checkpoint retention count.

        References:
        - input.md EB1.E (checkpoint controls)
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
        assert call_kwargs['execution_config'].values['checkpoint_save_top_k'] == 3

    def test_checkpoint_monitor_flag(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --checkpoint-monitor maps to its execution field.

        The CLI must preserve the explicit monitor metric.

        References:
        - input.md EB1.E (checkpoint controls)
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
        assert call_kwargs['execution_config'].values['checkpoint_monitor_metric'] == 'train_loss'

    def test_checkpoint_mode_flag(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --checkpoint-mode maps to checkpoint_mode.

        The CLI must preserve the explicit optimization direction.

        References:
        - input.md EB1.E (checkpoint controls)
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
        assert call_kwargs['execution_config'].values['checkpoint_mode'] == 'max'

    def test_early_stop_patience_flag(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --early-stop-patience maps to its execution field.

        The CLI must preserve the explicit patience value.

        References:
        - input.md EB1.E (checkpoint controls)
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
        assert call_kwargs['execution_config'].values['early_stop_patience'] == 10

    @pytest.mark.parametrize(
        "scheduler",
        [
            "Default",
            "Exponential",
            "MultiStage",
            "Adaptive",
            "WarmupCosine",
            "ReduceLROnPlateau",
        ],
    )
    def test_scheduler_flag_roundtrip(
        self,
        minimal_train_args,
        monkeypatch,
        scheduler,
    ):
        """
        Regression intent: --scheduler maps to the TrainingConfig patch.

        Optimization values remain separate from the execution request.

        References:
        - input.md EB2.A3 (scheduler/accumulation controls)
        - plans/.../phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md §EB2.A (CLI flag parsing)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(scheduler=scheduler),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--scheduler', scheduler]

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs['overrides']['scheduler'] == scheduler
        assert 'scheduler' not in call_kwargs['execution_config'].values

    def test_accumulate_grad_batches_roundtrip(self, minimal_train_args, monkeypatch):
        """
        Regression intent: accumulation maps to the TrainingConfig patch.

        Optimization values remain separate from the execution request.

        References:
        - input.md EB2.A3 (scheduler/accumulation controls)
        - plans/.../phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md §EB2.A (CLI flag parsing)
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(accum_steps=4),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--accumulate-grad-batches', '4']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs['overrides']['accum_steps'] == 4
        assert 'accum_steps' not in call_kwargs['execution_config'].values

    def test_logger_backend_csv_default(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --logger csv selects the CSV execution backend.

        The CLI must preserve the explicit logger selection.

        References:
        - input.md EB3.B1 (logger controls)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md §Q1
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(logger_backend='csv'),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--logger', 'csv']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].values['logger_backend'] == 'csv'

    def test_logger_backend_tensorboard(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --logger tensorboard selects that execution backend.

        The CLI must preserve the explicit logger selection.

        References:
        - input.md EB3.B1 (logger controls)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md §Q2
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(logger_backend='tensorboard'),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--logger', 'tensorboard']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].values['logger_backend'] == 'tensorboard'

    def test_logger_backend_none(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --logger none disables the execution logger.

        The CLI must preserve the explicit no-logger selection as None.

        References:
        - input.md EB3.B1 (logger controls)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(logger_backend=None),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--logger', 'none']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        assert call_kwargs['execution_config'].values['logger_backend'] is None

    def test_disable_mlflow_deprecation_warning(self, minimal_train_args, monkeypatch):
        """
        Regression intent: --disable_mlflow warns and maps to --logger none.

        The deprecated spelling must preserve behavior and emit its notice.

        References:
        - input.md EB3.B1 (logger controls)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md §Q3
        """
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_training_config=MagicMock(),
            execution_config=MagicMock(logger_backend=None),
        )

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory):
            test_args = minimal_train_args + ['--disable_mlflow']

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        # Verify logger_backend was set to None
        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        assert 'execution_config' in call_kwargs
        request = call_kwargs['execution_config']
        assert request.values['logger_backend'] is None
        assert any(
            notice.category is DeprecationWarning
            and '--disable_mlflow' in notice.message
            for notice in request.notices
        )


# Current integration status:
# The CLI accepts these flags, builds an ExecutionRequest and training patch,
# resolves them through create_training_payload, and forwards the explicit
# payload to the shared workflow via resolved_payload.


class TestPatchStatsCLI:
    """
    Test patch stats instrumentation CLI integration (FIX-PYTORCH-FORWARD-PARITY-001 Phase A).
    """

    @pytest.fixture
    def minimal_train_args(self, tmp_path):
        """Minimal required training CLI arguments for testing."""
        import numpy as np

        # Seed for deterministic non-zero variance (Phase C3 regression guard)
        np.random.seed(12345)

        train_file = tmp_path / "train.npz"
        np.savez(
            train_file,
            diff3d=np.random.rand(10, 64, 64).astype(np.float32),
            xcoords=np.random.rand(10).astype(np.float32),
            ycoords=np.random.rand(10).astype(np.float32),
            probeGuess=np.random.rand(64, 64).astype(np.complex64),
            objectGuess=np.random.rand(200, 200).astype(np.complex64),
        )

        return [
            '--train_data_file', str(train_file),
            '--output_dir', str(tmp_path / 'outputs'),
            '--n_images', '8',
            '--max_epochs', '1',
            '--batch_size', '16',
            '--gridsize', '2',
        ]

    def test_patch_stats_dump(self, minimal_train_args, monkeypatch, tmp_path):
        """
        Test that --log-patch-stats produces JSON and PNG artifacts.

        This is the selector required by input.md Phase A Do Now:
        pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump

        Expected behavior:
        - CLI accepts --log-patch-stats and --patch-stats-limit flags
        - After training, <output_dir>/analysis/ contains:
          - torch_patch_stats.json
          - torch_patch_grid.png
        """
        from ptycho_torch.train import cli_main

        output_dir = tmp_path / 'outputs'
        test_args = minimal_train_args + [
            '--log-patch-stats',
            '--patch-stats-limit', '2',
            '--accelerator', 'cpu',
            '--quiet',
        ]

        monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

        # Run training
        try:
            exit_code = cli_main()
        except SystemExit as e:
            exit_code = e.code

        # Assert training completed
        assert exit_code == 0 or exit_code is None, \
            f"Training CLI failed with exit code {exit_code}"

        # Assert artifacts exist
        analysis_dir = output_dir / 'analysis'
        json_path = analysis_dir / 'torch_patch_stats.json'
        png_path = analysis_dir / 'torch_patch_grid.png'

        assert json_path.exists(), \
            f"Missing torch_patch_stats.json at {json_path}"
        assert png_path.exists(), \
            f"Missing torch_patch_grid.png at {png_path}"

        # Verify JSON structure and variance guard (Phase C3)
        import json
        with open(json_path) as f:
            stats = json.load(f)

        assert isinstance(stats, list), "Expected list of batch stats"
        assert len(stats) > 0, "Expected at least one batch logged"

        # Phase C3 regression guard: assert non-zero variance for gridsize>=2
        # Threshold rationale: analysis/phase_c2_pytorch_only_metrics.txt shows
        # gridsize=2 baseline has patch.var_zero_mean=8.97e9, while gridsize=1
        # collapsed to 0.0. Guard only applies to gridsize>=2 configurations.
        # References: POLICY-001 (PyTorch mandatory), CONFIG-001 (config bridge)
        first_batch = stats[0]

        # Assert patch variance is non-zero (guards against zero-variance regression)
        assert 'var_zero_mean' in first_batch, "Missing var_zero_mean in batch stats"
        patch_var = first_batch['var_zero_mean']
        assert patch_var > 1e-6, \
            f"Patch variance too low (var_zero_mean={patch_var:.2e}), expected > 1e-6 for gridsize>=2"

        # Assert global mean is non-zero (guards against output collapse)
        assert 'global_mean' in first_batch, "Missing global_mean in batch stats"
        global_mean = first_batch['global_mean']
        assert abs(global_mean) > 1e-9, \
            f"Global mean is zero (global_mean={global_mean:.2e}), indicates output collapse"

        # Phase C3c: Run inference CLI to verify forward-path parity (gridsize>=2)
        # Rationale: analysis/phase_c2_pytorch_only_metrics.txt and
        # docs/specs/spec-ptycho-workflow.md require that inference patches retain
        # variance when gridsize>=2, ensuring forward reassembly produces structured
        # patches before stitching (POLICY-001/CONFIG-001).
        from ptycho_torch.inference import cli_main as inference_cli_main

        # Extract train data path from minimal_train_args
        train_data_idx = minimal_train_args.index('--train_data_file')
        test_data_path = minimal_train_args[train_data_idx + 1]

        inference_args = [
            '--model_path', str(output_dir),
            '--test_data', test_data_path,
            '--output_dir', str(tmp_path / 'outputs_infer'),
            '--n_images', '8',
            '--log-patch-stats',
            '--patch-stats-limit', '2',
            '--accelerator', 'cpu',
            '--quiet',
        ]

        monkeypatch.setattr('sys.argv', ['inference.py'] + inference_args)

        # Run inference
        try:
            exit_code = inference_cli_main()
        except SystemExit as e:
            exit_code = e.code

        # Assert inference completed
        assert exit_code == 0 or exit_code is None, \
            f"Inference CLI failed with exit code {exit_code}"

        # Assert inference artifacts exist
        # Note: inference CLI uses same artifact names as training (torch_patch_stats.json)
        inference_output_dir = tmp_path / 'outputs_infer'
        inference_analysis_dir = inference_output_dir / 'analysis'
        inference_json_path = inference_analysis_dir / 'torch_patch_stats.json'
        inference_png_path = inference_analysis_dir / 'torch_patch_grid.png'

        assert inference_json_path.exists(), \
            f"Missing torch_patch_stats.json at {inference_json_path} " \
            f"(POLICY-001/CONFIG-001: inference instrumentation must emit stats for gridsize>=2)"
        assert inference_png_path.exists(), \
            f"Missing torch_patch_grid.png at {inference_png_path}"

        # Verify inference JSON structure and variance guard (Phase C3c/C3d)
        import json
        with open(inference_json_path) as f:
            inference_stats = json.load(f)

        assert isinstance(inference_stats, list), "Expected list of batch stats from inference"
        assert len(inference_stats) > 0, "Expected at least one batch logged during inference"

        # Phase C3d regression guard: assert non-zero variance for inference path
        # Threshold rationale (same as training): analysis/phase_c2_pytorch_only_metrics.txt
        # shows that gridsize=2 baseline retains patch.var_zero_mean=8.97e9, while gridsize=1
        # collapsed to 0.0. Inference forward path must preserve variance for gridsize>=2.
        # References: POLICY-001 (PyTorch mandatory), CONFIG-001 (config bridge),
        # docs/specs/spec-ptycho-workflow.md (forward reassembly parity requirement)
        inference_first_batch = inference_stats[0]

        assert 'var_zero_mean' in inference_first_batch, \
            "Missing var_zero_mean in inference batch stats"
        inference_patch_var = inference_first_batch['var_zero_mean']
        assert inference_patch_var > 1e-6, \
            f"Inference patch variance too low (var_zero_mean={inference_patch_var:.2e}), " \
            f"expected > 1e-6 for gridsize>=2 (forward-path parity requirement)"

        assert 'global_mean' in inference_first_batch, \
            "Missing global_mean in inference batch stats"
        inference_global_mean = inference_first_batch['global_mean']
        assert abs(inference_global_mean) > 1e-9, \
            f"Inference global mean is zero (global_mean={inference_global_mean:.2e}), " \
            f"indicates inference output collapse (violates forward-path parity)"

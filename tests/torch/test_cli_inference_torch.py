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
             patch('ptycho_torch.workflows.bundle_io.load_inference_bundle_torch', mock_bundle_loader):
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
        assert call_kwargs['execution_config'].values['accelerator'] == 'cpu'

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
             patch('ptycho_torch.workflows.bundle_io.load_inference_bundle_torch', mock_bundle_loader):
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
        assert call_kwargs['execution_config'].values['num_workers'] == 4

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
             patch('ptycho_torch.workflows.bundle_io.load_inference_bundle_torch', mock_bundle_loader):
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
        assert call_kwargs['execution_config'].values['inference_batch_size'] == 32

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
             patch('ptycho_torch.workflows.bundle_io.load_inference_bundle_torch', mock_bundle_loader):
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
        request = call_kwargs['execution_config']
        assert request.values['accelerator'] == 'gpu'
        assert request.values['num_workers'] == 8
        assert request.values['inference_batch_size'] == 64

    def test_native_inference_execution_request_preserves_explicit_options(
        self,
        minimal_inference_args,
        monkeypatch,
    ):
        """The native inference request reaches the payload factory unchanged."""
        from ptycho_torch.cli.shared import (
            build_execution_request_from_args as real_request_builder,
        )
        from ptycho_torch.execution_request import ExecutionRequest
        from ptycho_torch.inference import cli_main

        argv = minimal_inference_args + [
            '--accelerator=cpu',
            '--device', 'cuda',
            '--num-workers=2',
            '--inference-batch-size', '8',
            '--quiet',
        ]
        monkeypatch.setattr('sys.argv', ['inference.py', *argv])

        with patch(
            'ptycho_torch.cli.shared.build_execution_request_from_args',
            wraps=real_request_builder,
        ) as request_builder, patch(
            'ptycho_torch.config_factory.create_inference_payload',
            side_effect=RuntimeError('stop after request capture'),
        ) as factory, pytest.raises(RuntimeError, match='Failed to create'):
            cli_main()

        request_builder.assert_called_once()
        assert request_builder.call_args.kwargs == {
            'mode': 'inference',
            'explicit_options': tuple(argv),
            'lane': 'native-inference',
        }
        request = factory.call_args.kwargs['execution_config']
        assert isinstance(request, ExecutionRequest)
        assert request.explicit_fields == frozenset(
            {
                'accelerator',
                'num_workers',
                'inference_batch_size',
                'enable_progress_bar',
            }
        )
        assert request.values['accelerator'] == 'cpu'
        assert request.values['num_workers'] == 2
        assert request.values['inference_batch_size'] == 8
        assert request.values['enable_progress_bar'] is False


class TestInferenceCLIThinWrapper:
    """
    RED Phase tests for inference CLI thin wrapper delegation (ADR-003 Phase D.C C2).

    Tests verify that the inference CLI delegates to shared helpers and workflow components
    rather than implementing business logic inline. These tests are EXPECTED TO FAIL until
    the thin wrapper refactor is implemented (Phase D.C C3).

    Expected RED Failures:
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
            tf_inference_config=MagicMock(inference_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu'),
        )
        mock_bundle_loader = MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))
        mock_raw_data = MagicMock()

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.workflows.bundle_io.load_inference_bundle_torch', mock_bundle_loader), \
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

    def test_cli_calls_save_individual_reconstructions(self, minimal_inference_args, monkeypatch):
        """The CLI saves the barycentric kernel's amplitude/phase to the output dir."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        import numpy as np

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(inference_groups=32),
            pt_inference_config=MagicMock(log_patch_stats=False),
            execution_config=MagicMock(
                accelerator='cpu', num_workers=0, inference_batch_size=None
            ),
        )
        mock_amplitude = np.random.rand(64, 64)
        mock_phase = np.random.rand(64, 64)
        mock_helper = MagicMock(
            return_value=SimpleNamespace(amplitude=mock_amplitude, phase=mock_phase)
        )
        mock_save_fn = MagicMock()

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.inference.reconstruct', mock_helper), \
             patch('ptycho_torch.inference.save_individual_reconstructions', mock_save_fn):
            from ptycho_torch.inference import cli_main
            monkeypatch.setattr('sys.argv', ['inference.py'] + minimal_inference_args)
            cli_main()

        # Assert save function was called with (amplitude, phase, output_dir)
        assert mock_save_fn.called, \
            "save_individual_reconstructions() not called - output artifact generation missing"
        call_args = mock_save_fn.call_args[0]
        assert len(call_args) >= 3, "Expected 3 arguments: (amplitude, phase, output_dir)"

    def test_quiet_flag_suppresses_progress_output(self, minimal_inference_args, monkeypatch, capsys):
        """--quiet passes enable_progress_bar=False through the execution request."""
        from unittest.mock import MagicMock, patch

        mock_validate_paths = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock(
            tf_inference_config=MagicMock(inference_groups=32),
            pt_data_config=MagicMock(),
            execution_config=MagicMock(accelerator='cpu', enable_progress_bar=False),
        )

        with patch('ptycho_torch.cli.shared.validate_paths', mock_validate_paths), \
             patch('ptycho_torch.config_factory.create_inference_payload', mock_factory), \
             patch('ptycho_torch.inference.reconstruct', MagicMock()), \
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
        request = call_kwargs['execution_config']
        assert request.values['enable_progress_bar'] is False, \
            "Expected enable_progress_bar=False when --quiet specified"


class TestCorruptionRetainedBoundaries:
    """Phase 4 Task 4: each deleted check leaves a retained boundary that still
    rejects the same corruption."""

    def test_corruption_channel_join_rejected_by_retained_decode(self):
        """The inference-side C-join is gone; decode still rejects a broken join."""
        from ptycho.config.config import ModelConfig as CanonicalModelConfig
        from ptycho_torch.artifact_schema import (
            decode_artifact_identity,
            encode_artifact_identity,
        )
        from ptycho_torch.config_params import (
            DataConfig,
            InferenceConfig,
            ModelConfig,
            TrainingConfig,
        )
        from ptycho_torch.model_spec import derive_model_spec

        data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
        model = ModelConfig(
            object_layout="single_patch",
            training_canvas="independent",
            training_patch_weighting="uniform",
            object_big=None,
            amp_activation="silu",
        )
        canonical = CanonicalModelConfig(
            N=64,
            gridsize=1,
            object_layout="single_patch",
            training_canvas="independent",
            training_patch_weighting="uniform",
            object_big=None,
            amp_activation="swish",
        )
        spec = derive_model_spec(canonical, model, data)
        payload = encode_artifact_identity(
            spec, data, TrainingConfig(torch_loss_mode="poisson"), InferenceConfig()
        )
        payload["data_config"]["C"] = 2
        with pytest.raises(ValueError, match="field set is not exact"):
            decode_artifact_identity(payload)

    def test_corruption_scale_contract_rejected_by_retained_validation(self):
        """The inference-side scale-contract check is gone; construction still rejects a broken pair."""
        from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
        from ptycho_torch.scaling_contract import validate_scale_contract

        data = DataConfig(
            N=64,
            gridsize=1,
            scale_contract_version="ci_intensity_v2",
            measurement_domain="normalized_amplitude",
        )
        model = ModelConfig(
            physics_forward_mode="rectangular_scaled"
        )
        with pytest.raises(ValueError, match="scale contract"):
            validate_scale_contract(
                data, model, TrainingConfig(torch_loss_mode="poisson")
            )

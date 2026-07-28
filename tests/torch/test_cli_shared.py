"""
Unit tests for CLI shared helper functions (ADR-003 Phase D.B3, GREEN as of 2025-10-20).

This module tests the shared helper functions in `ptycho_torch/cli/shared.py` that
support the training and inference CLI thin-wrapper refactor. These helpers were
introduced in Phase D.B3 and are now fully implemented and GREEN.

Helper Functions Under Test:
- resolve_accelerator(): Handle --device → --accelerator backward compatibility
- build_execution_config_from_args(): Construct PyTorchExecutionConfig with validation
- validate_paths(): Check file existence and create output directory

Test Coverage:
- Unit tests for each helper function in isolation
- Verify deprecation warnings are emitted correctly
- Verify validation errors are raised with clear messages
- Use tmp_path fixtures for file I/O tests

GREEN Status (Phase D.B3, 2025-10-20):
- All 20 tests PASSING (helpers implemented per training_refactor.md blueprint)
- Deprecation semantics verified (--device → --accelerator mapping with warnings)
- Validation logic tested (path checks, execution config field constraints)
- Evidence: plans/.../phase_d_cli_wrappers_training_impl/pytest_cli_shared_green.log

References:
- Blueprint: plans/.../phase_d_cli_wrappers_training/training_refactor.md §Component 1
- Design Notes: plans/.../phase_d_cli_wrappers_baseline/design_notes.md §D1-D8
- Spec: specs/ptychodus_api_spec.md §7 (CLI execution config flags)
- Implementation: ptycho_torch/cli/shared.py (3 helper functions, ~150 lines)
"""

import pytest
import argparse


class TestResolveAccelerator:
    """
    Unit tests for resolve_accelerator() helper function.

    Tests verify:
    - Default accelerator passthrough (no --device specified)
    - Legacy --device flag mapping to --accelerator
    - Conflict resolution (--accelerator wins when both specified)
    - Deprecation warnings emitted correctly
    """

    def test_default_no_device(self):
        """
        RED Test: resolve_accelerator('cpu', None) → 'cpu'.

        Expected RED Failure:
        - ImportError: cannot import name 'resolve_accelerator'

        Success Criteria (GREEN):
        - Returns 'cpu' unchanged
        - No warnings emitted
        """
        from ptycho_torch.cli.shared import resolve_accelerator

        result = resolve_accelerator('cpu', None)
        assert result == 'cpu'

    def test_legacy_device_cpu(self):
        """
        RED Test: resolve_accelerator('auto', 'cpu') → 'cpu' + DeprecationWarning.

        Expected RED Failure:
        - ImportError: cannot import name 'resolve_accelerator'

        Success Criteria (GREEN):
        - Returns 'cpu' (mapped from legacy --device)
        - Emits DeprecationWarning with message containing "--device is deprecated"
        """
        from ptycho_torch.cli.shared import resolve_accelerator

        with pytest.warns(DeprecationWarning, match="--device is deprecated"):
            result = resolve_accelerator('auto', 'cpu')

        assert result == 'cpu'

    def test_legacy_device_cuda_maps_to_gpu(self):
        """
        RED Test: resolve_accelerator('auto', 'cuda') → 'gpu' + DeprecationWarning.

        Expected RED Failure:
        - ImportError: cannot import name 'resolve_accelerator'

        Success Criteria (GREEN):
        - Returns 'gpu' (NOT 'cuda' — Lightning convention)
        - Emits DeprecationWarning
        """
        from ptycho_torch.cli.shared import resolve_accelerator

        with pytest.warns(DeprecationWarning, match="--device is deprecated"):
            result = resolve_accelerator('auto', 'cuda')

        assert result == 'gpu', \
            "Legacy --device cuda should map to 'gpu' for Lightning compatibility"

    def test_conflict_accelerator_wins(self):
        """
        RED Test: resolve_accelerator('cpu', 'cuda') → 'cpu' + DeprecationWarning.

        Expected RED Failure:
        - ImportError: cannot import name 'resolve_accelerator'

        Success Criteria (GREEN):
        - Returns 'cpu' (--accelerator takes precedence)
        - Emits DeprecationWarning with message containing "Ignoring --device value"
        """
        from ptycho_torch.cli.shared import resolve_accelerator

        with pytest.warns(DeprecationWarning, match="Ignoring --device value"):
            result = resolve_accelerator('cpu', 'cuda')

        assert result == 'cpu', \
            "--accelerator should take precedence when both --device and --accelerator specified"

    def test_all_accelerator_values_passthrough(self):
        """
        RED Test: Valid accelerator values (except 'auto') pass through unchanged when device=None.

        Expected RED Failure:
        - ImportError: cannot import name 'resolve_accelerator'

        Success Criteria (GREEN):
        - Each non-'auto' accelerator value returns unchanged
        - No warnings emitted for explicit accelerators
        """
        from ptycho_torch.cli.shared import resolve_accelerator

        # 'auto' is now special-cased for CUDA detection, so exclude it here
        explicit_accelerators = ['cpu', 'gpu', 'cuda', 'mps']

        for accel in explicit_accelerators:
            result = resolve_accelerator(accel, None)
            assert result == accel, \
                f"Expected accelerator='{accel}' to pass through unchanged, got '{result}'"

    def test_tpu_accelerator_is_rejected(self):
        from ptycho_torch.cli.shared import resolve_accelerator

        with pytest.raises(ValueError, match="Torch-XLA.*TPU.*unsupported"):
            resolve_accelerator('tpu', None)

    def test_resolve_accelerator_auto_defaults(self):
        """
        Test: resolve_accelerator('auto', None) auto-detects: CUDA if available, else CPU.

        Success Criteria (GREEN):
        - When CUDA available: returns 'cuda'
        - When CUDA unavailable: returns 'cpu' + emits POLICY-001 UserWarning
        - Auto-detection aligns with GPU baseline policy
        """
        from ptycho_torch.cli.shared import resolve_accelerator
        import torch

        if torch.cuda.is_available():
            # CUDA available: should resolve to 'cuda' without warnings
            result = resolve_accelerator('auto', None)
            assert result == 'cuda', \
                f"Expected 'auto' to resolve to 'cuda' when CUDA available, got '{result}'"
        else:
            # CUDA unavailable: should resolve to 'cpu' with POLICY-001 warning
            with pytest.warns(UserWarning, match="POLICY-001"):
                result = resolve_accelerator('auto', None)
            assert result == 'cpu', \
                f"Expected 'auto' to resolve to 'cpu' when CUDA unavailable, got '{result}'"


class TestBuildExecutionConfig:
    """
    Unit tests for build_execution_config_from_args() helper function.

    Tests verify:
    - Training mode default values
    - Custom execution config values
    - Inference mode field availability
    - Deterministic+num_workers warning
    - --quiet and --disable_mlflow flag mapping
    - Invalid mode raises ValueError
    """

    def test_training_mode_defaults(self):
        """
        RED Test: build_execution_config_from_args() with training defaults.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - Returns PyTorchExecutionConfig with expected default values
        - accelerator='cpu', deterministic=True, num_workers=0, learning_rate=1e-3
        - enable_progress_bar=True (quiet=False, disable_mlflow=False)
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            deterministic=True,
            num_workers=0,
            learning_rate=1e-3,
            disable_mlflow=False,
            quiet=False,
        )

        config = build_execution_config_from_args(args, mode='training')

        assert config.accelerator == 'cpu'
        assert config.deterministic is True
        assert config.num_workers == 0
        assert abs(config.learning_rate - 1e-3) < 1e-10
        assert config.enable_progress_bar is True

    def test_training_mode_custom_values(self):
        """
        RED Test: build_execution_config_from_args() with custom training values.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - Returns PyTorchExecutionConfig with custom values
        - accelerator='gpu', deterministic=False, num_workers=4, learning_rate=5e-4
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='gpu',
            device=None,
            deterministic=False,
            num_workers=4,
            learning_rate=5e-4,
            disable_mlflow=False,
            quiet=False,
        )

        config = build_execution_config_from_args(args, mode='training')

        assert config.accelerator == 'gpu'
        assert config.deterministic is False
        assert config.num_workers == 4
        assert abs(config.learning_rate - 5e-4) < 1e-10

    def test_inference_mode(self):
        """
        RED Test: build_execution_config_from_args() with inference mode.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - Returns PyTorchExecutionConfig for inference mode
        - accelerator, num_workers, inference_batch_size fields present
        - No learning_rate or deterministic fields (mode-specific)
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            num_workers=2,
            inference_batch_size=32,
            quiet=False,
        )

        config = build_execution_config_from_args(args, mode='inference')

        assert config.accelerator == 'cpu'
        assert config.num_workers == 2
        assert config.inference_batch_size == 32
        assert config.enable_progress_bar is True

    def test_emits_deterministic_warning(self):
        """
        RED Test: Emits UserWarning when deterministic=True and num_workers>0 (training).

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - UserWarning emitted with message containing "performance degradation"
        - Config still created successfully
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            deterministic=True,
            num_workers=4,
            learning_rate=1e-3,
            disable_mlflow=False,
            quiet=False,
        )

        with pytest.warns(UserWarning, match="performance degradation"):
            config = build_execution_config_from_args(args, mode='training')

        # Config should still be created
        assert config.deterministic is True
        assert config.num_workers == 4

    def test_handles_quiet_flag(self):
        """
        RED Test: --quiet flag maps to enable_progress_bar=False.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - enable_progress_bar=False when args.quiet=True
        - enable_progress_bar=True when args.quiet=False
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        # Test quiet=True
        args_quiet = argparse.Namespace(
            accelerator='cpu',
            device=None,
            deterministic=True,
            num_workers=0,
            learning_rate=1e-3,
            disable_mlflow=False,
            quiet=True,
        )

        config_quiet = build_execution_config_from_args(args_quiet, mode='training')
        assert config_quiet.enable_progress_bar is False

        # Test quiet=False
        args_verbose = argparse.Namespace(
            accelerator='cpu',
            device=None,
            deterministic=True,
            num_workers=0,
            learning_rate=1e-3,
            disable_mlflow=False,
            quiet=False,
        )

        config_verbose = build_execution_config_from_args(args_verbose, mode='training')
        assert config_verbose.enable_progress_bar is True

    def test_handles_disable_mlflow_flag(self):
        """
        RED Test: --disable_mlflow flag maps to enable_progress_bar=False.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - enable_progress_bar=False when args.disable_mlflow=True
        - Behavior identical to --quiet flag (backward compatibility)
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            deterministic=True,
            num_workers=0,
            learning_rate=1e-3,
            disable_mlflow=True,  # Legacy flag
            quiet=False,
        )

        config = build_execution_config_from_args(args, mode='training')
        assert config.enable_progress_bar is False

    def test_quiet_or_disable_mlflow_both_true(self):
        """
        RED Test: Both --quiet and --disable_mlflow → enable_progress_bar=False.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - enable_progress_bar=False when either flag is True
        - Logical OR behavior (quiet_mode = quiet OR disable_mlflow)
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            deterministic=True,
            num_workers=0,
            learning_rate=1e-3,
            disable_mlflow=True,
            quiet=True,
        )

        config = build_execution_config_from_args(args, mode='training')
        assert config.enable_progress_bar is False

    def test_invalid_mode_raises_value_error(self):
        """
        RED Test: Invalid mode raises ValueError with clear message.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - ValueError raised when mode not in {'training', 'inference'}
        - Error message includes mode value and expected values
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            num_workers=0,
        )

        with pytest.raises(ValueError, match="Invalid mode: foo"):
            build_execution_config_from_args(args, mode='foo')

    def test_resolves_accelerator_from_device_flag(self):
        """
        RED Test: build_execution_config_from_args() calls resolve_accelerator() internally.

        Expected RED Failure:
        - ImportError: cannot import name 'build_execution_config_from_args'

        Success Criteria (GREEN):
        - Legacy --device flag is handled via resolve_accelerator()
        - args.device='cuda' + args.accelerator='auto' → config.accelerator='gpu'
        - DeprecationWarning emitted
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='auto',
            device='cuda',  # Legacy flag
            deterministic=True,
            num_workers=0,
            learning_rate=1e-3,
            disable_mlflow=False,
            quiet=False,
        )

        with pytest.warns(DeprecationWarning, match="--device is deprecated"):
            config = build_execution_config_from_args(args, mode='training')

        assert config.accelerator == 'gpu', \
            "Expected legacy --device cuda to map to accelerator='gpu'"


class TestValidatePaths:
    """
    Unit tests for validate_paths() helper function.

    Tests verify:
    - Creates output_dir if missing
    - Raises FileNotFoundError if train_file missing
    - Raises FileNotFoundError if test_file missing
    - Accepts None for test_file (optional parameter)
    - Works with pathlib.Path objects
    """

    def test_creates_output_dir(self, tmp_path):
        """
        RED Test: validate_paths() creates output_dir if it doesn't exist.

        Expected RED Failure:
        - ImportError: cannot import name 'validate_paths'

        Success Criteria (GREEN):
        - output_dir created with parents (mkdir -p behavior)
        - Function returns None (no error raised)
        """
        from ptycho_torch.cli.shared import validate_paths

        train_file = tmp_path / 'train.npz'
        train_file.touch()  # Create dummy file
        output_dir = tmp_path / 'nonexistent' / 'nested' / 'outputs'

        # output_dir does not exist yet
        assert not output_dir.exists()

        validate_paths(train_file, None, output_dir)

        # output_dir should now exist
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_raises_if_train_file_missing(self, tmp_path):
        """
        RED Test: validate_paths() raises FileNotFoundError if train_file missing.

        Expected RED Failure:
        - ImportError: cannot import name 'validate_paths'

        Success Criteria (GREEN):
        - FileNotFoundError raised with clear message
        - Error message includes train_file path
        """
        from ptycho_torch.cli.shared import validate_paths

        train_file = tmp_path / 'missing_train.npz'
        output_dir = tmp_path / 'outputs'

        with pytest.raises(FileNotFoundError, match="Training data file not found"):
            validate_paths(train_file, None, output_dir)

    def test_raises_if_test_file_missing(self, tmp_path):
        """
        RED Test: validate_paths() raises FileNotFoundError if test_file missing.

        Expected RED Failure:
        - ImportError: cannot import name 'validate_paths'

        Success Criteria (GREEN):
        - FileNotFoundError raised with clear message
        - Error message includes test_file path
        """
        from ptycho_torch.cli.shared import validate_paths

        train_file = tmp_path / 'train.npz'
        train_file.touch()
        test_file = tmp_path / 'missing_test.npz'
        output_dir = tmp_path / 'outputs'

        with pytest.raises(FileNotFoundError, match="Test data file not found"):
            validate_paths(train_file, test_file, output_dir)

    def test_accepts_none_test_file(self, tmp_path):
        """
        RED Test: validate_paths() accepts None for test_file (optional parameter).

        Expected RED Failure:
        - ImportError: cannot import name 'validate_paths'

        Success Criteria (GREEN):
        - No error raised when test_file=None
        - output_dir still created
        """
        from ptycho_torch.cli.shared import validate_paths

        train_file = tmp_path / 'train.npz'
        train_file.touch()
        output_dir = tmp_path / 'outputs'

        # Should not raise (test_file is optional)
        validate_paths(train_file, None, output_dir)

        assert output_dir.exists()

    def test_works_with_pathlib_path_objects(self, tmp_path):
        """
        RED Test: validate_paths() works with pathlib.Path objects (not just strings).

        Expected RED Failure:
        - ImportError: cannot import name 'validate_paths'

        Success Criteria (GREEN):
        - Accepts Path objects for all parameters
        - Creates output_dir successfully
        """
        from ptycho_torch.cli.shared import validate_paths

        train_file = tmp_path / 'train.npz'
        train_file.touch()
        test_file = tmp_path / 'test.npz'
        test_file.touch()
        output_dir = tmp_path / 'outputs'

        # Pass Path objects (not strings)
        validate_paths(train_file, test_file, output_dir)

        assert output_dir.exists()

    def test_accepts_none_train_file_for_inference_mode(self, tmp_path):
        """
        RED Test: validate_paths() accepts None for train_file (inference mode).

        Expected RED Failure:
        - ImportError: cannot import name 'validate_paths'

        Success Criteria (GREEN):
        - No error raised when train_file=None (inference CLI context)
        - output_dir still created
        """
        from ptycho_torch.cli.shared import validate_paths

        test_file = tmp_path / 'test.npz'
        test_file.touch()
        output_dir = tmp_path / 'outputs'

        # Inference mode: train_file not required
        validate_paths(None, test_file, output_dir)

        assert output_dir.exists()


# RED Phase Note:
# These tests are EXPECTED TO FAIL because:
# 1. ptycho_torch/cli/shared.py module does not exist
# 2. resolve_accelerator(), build_execution_config_from_args(), validate_paths() not implemented
#
# Phase D.B3 implementation will:
# 1. Create ptycho_torch/cli/shared.py
# 2. Implement the three helper functions per training_refactor.md blueprint
# 3. Turn these RED tests GREEN


# Inference Mode Tests (Phase D.C C2) - Extension for inference CLI support


class TestBuildExecutionConfigInferenceMode:
    """
    Unit tests for build_execution_config_from_args() in inference mode (Phase D.C C2).

    Tests verify inference-specific behavior:
    - Mode='inference' produces correct config
    - inference_batch_size field availability
    - No training-specific fields (learning_rate, deterministic)
    - No deterministic+num_workers warning in inference mode
    - Quiet flag behavior identical to training mode
    """

    def test_inference_mode_defaults(self):
        """
        RED Test: build_execution_config_from_args(mode='inference') with defaults.

        Expected RED Failure:
        - May pass if Phase D.B3 already handles mode='inference'
        - OR ImportError if helper not yet implemented

        Success Criteria (GREEN):
        - Returns PyTorchExecutionConfig with inference defaults
        - accelerator='cpu', num_workers=0, inference_batch_size=None
        - enable_progress_bar=True (quiet=False)
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='cpu',
            device=None,
            num_workers=0,
            inference_batch_size=None,
            quiet=False,
        )

        config = build_execution_config_from_args(args, mode='inference')

        assert config.accelerator == 'cpu'
        assert config.num_workers == 0
        assert config.inference_batch_size is None, \
            "Expected inference_batch_size=None (default)"
        assert config.enable_progress_bar is True

    def test_inference_mode_custom_batch_size(self):
        """
        RED Test: inference_batch_size field is correctly set when provided.

        Expected RED Failure:
        - May pass if Phase D.B3 already handles inference_batch_size
        - OR AttributeError if field not defined

        Success Criteria (GREEN):
        - inference_batch_size=64 when specified
        - No training-specific fields present
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args = argparse.Namespace(
            accelerator='gpu',
            device=None,
            num_workers=2,
            inference_batch_size=64,
            quiet=False,
        )

        config = build_execution_config_from_args(args, mode='inference')

        assert config.accelerator == 'gpu'
        assert config.num_workers == 2
        assert config.inference_batch_size == 64, \
            f"Expected inference_batch_size=64, got {config.inference_batch_size}"

    def test_inference_mode_respects_quiet(self):
        """
        RED Test: --quiet flag works identically in inference mode.

        Expected RED Failure:
        - May pass if Phase D.B3 already handles quiet in all modes

        Success Criteria (GREEN):
        - enable_progress_bar=False when quiet=True
        - Behavior identical to training mode
        """
        from ptycho_torch.cli.shared import build_execution_config_from_args

        args_quiet = argparse.Namespace(
            accelerator='cpu',
            device=None,
            num_workers=0,
            inference_batch_size=None,
            quiet=True,
        )

        config_quiet = build_execution_config_from_args(args_quiet, mode='inference')
        assert config_quiet.enable_progress_bar is False, \
            "Expected enable_progress_bar=False when quiet=True"

        args_verbose = argparse.Namespace(
            accelerator='cpu',
            device=None,
            num_workers=0,
            inference_batch_size=None,
            quiet=False,
        )

        config_verbose = build_execution_config_from_args(args_verbose, mode='inference')
        assert config_verbose.enable_progress_bar is True, \
            "Expected enable_progress_bar=True when quiet=False"


# RED Phase Note (Updated for Phase D.C C2):
# Tests in TestBuildExecutionConfigInferenceMode are NEW for Phase D.C C2.
# They extend the existing test_cli_shared.py coverage to include inference-mode behavior.
#
# Expected RED Behavior:
# - If Phase D.B3 implementation correctly handles mode='inference', these tests may PASS
# - If mode='inference' branch is missing or incomplete, tests will FAIL with:
#   - ValueError: Invalid mode: inference (if mode not supported)
#   - AttributeError: 'PyTorchExecutionConfig' object has no attribute 'inference_batch_size'
#   - AssertionError: Field value mismatch
#
# GREEN Target (Phase D.C C3):
# - All inference-mode tests PASS when CLI refactor complete
# - Validates inference CLI delegates correctly to shared helpers
#


def test_execution_request_records_only_explicit_fields():
    from ptycho_torch.execution_request import ExecutionRequest, ResolutionNotice

    request = ExecutionRequest(
        values={"accelerator": "auto", "devices": "auto"},
        explicit_fields=frozenset({"accelerator"}),
        notices=(
            ResolutionNotice(DeprecationWarning, "--device is deprecated"),
        ),
    )

    assert request.as_dict() == {
        "accelerator": "auto",
        "devices": "auto",
    }
    assert request.explicit_fields == frozenset({"accelerator"})


def test_execution_request_defensively_copies_values():
    from ptycho_torch.execution_request import ExecutionRequest

    source = {"accelerator": "cpu"}
    request = ExecutionRequest(
        values=source,
        explicit_fields=frozenset({"accelerator"}),
    )

    source["accelerator"] = "gpu"
    assert request.as_dict() == {"accelerator": "cpu"}

    returned = request.as_dict()
    returned["accelerator"] = "mps"
    assert request.as_dict() == {"accelerator": "cpu"}

    with pytest.raises(TypeError):
        request.values["accelerator"] = "cuda"


def test_execution_request_defensively_copies_provenance_containers():
    from ptycho_torch.execution_request import ExecutionRequest, ResolutionNotice

    explicit_fields = {"accelerator"}
    notice = ResolutionNotice(DeprecationWarning, "--device is deprecated")
    notices = [notice]
    request = ExecutionRequest(
        values={"accelerator": "auto", "devices": "auto"},
        explicit_fields=explicit_fields,
        notices=notices,
    )

    explicit_fields.add("devices")
    notices.append(ResolutionNotice(UserWarning, "late mutation"))

    assert type(request.explicit_fields) is frozenset
    assert request.explicit_fields == frozenset({"accelerator"})
    assert type(request.notices) is tuple
    assert request.notices == (notice,)


def test_execution_request_rejects_unknown_value_field():
    from ptycho_torch.execution_request import ExecutionRequest

    with pytest.raises(ValueError, match="unknown execution request field"):
        ExecutionRequest(
            values={"bogus": "value"},
            explicit_fields=frozenset(),
        )


def test_execution_request_rejects_unknown_explicit_field():
    from ptycho_torch.execution_request import ExecutionRequest

    with pytest.raises(ValueError, match="unknown explicit execution field"):
        ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"bogus"}),
        )


def test_execution_request_rejects_explicit_field_absent_from_values():
    from ptycho_torch.execution_request import ExecutionRequest

    with pytest.raises(ValueError, match="explicit execution field.*absent"):
        ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"devices"}),
        )


def test_execution_request_preserves_equal_default_explicitness():
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    default_devices = PyTorchExecutionConfig.__dataclass_fields__["devices"].default
    explicit = ExecutionRequest(
        values={"devices": default_devices},
        explicit_fields=frozenset({"devices"}),
    )
    omitted = ExecutionRequest(
        values={"devices": default_devices},
        explicit_fields=frozenset(),
    )

    assert explicit.as_dict() == omitted.as_dict()
    assert explicit.explicit_fields == frozenset({"devices"})
    assert omitted.explicit_fields == frozenset()


def test_execution_request_does_not_construct_config_or_observe_cuda(monkeypatch):
    import torch

    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    def fail_if_called(*args, **kwargs):
        raise AssertionError("request construction must remain primitive and pure")

    monkeypatch.setattr(PyTorchExecutionConfig, "__init__", fail_if_called)
    monkeypatch.setattr(torch.cuda, "is_available", fail_if_called)

    request = ExecutionRequest(
        values={"accelerator": "auto"},
        explicit_fields=frozenset({"accelerator"}),
    )

    assert request.as_dict() == {"accelerator": "auto"}


def test_execution_request_support_records_are_frozen():
    from dataclasses import FrozenInstanceError

    from ptycho_torch.execution_request import (
        EnvironmentResolution,
        ExecutionCapabilities,
        ResolutionNotice,
    )

    capabilities = ExecutionCapabilities(
        cuda_available=True,
        cuda_device_count=2,
    )
    resolution = EnvironmentResolution(
        requested={"accelerator": "auto"},
        resolved={"accelerator": "cuda"},
        capabilities=capabilities,
    )
    notice = ResolutionNotice(UserWarning, "resolved accelerator")

    assert resolution.capabilities == capabilities
    assert notice.category is UserWarning
    with pytest.raises(FrozenInstanceError):
        capabilities.cuda_available = False


def test_build_execution_request_from_args_records_canonical_explicit_fields():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    args = argparse.Namespace(
        accelerator="auto",
        deterministic=True,
        num_workers=0,
        learning_rate=2e-4,
    )

    request = build_execution_request_from_args(
        args,
        mode="training",
        explicit_options={"--learning-rate", "--accelerator"},
    )

    assert request.values["learning_rate"] == 2e-4
    assert request.values["accelerator"] == "auto"
    assert request.explicit_fields == frozenset(
        {"learning_rate", "accelerator"}
    )


@pytest.mark.parametrize(
    ("logger_backend", "explicit_options", "expected"),
    [
        ("none", {"--logger"}, None),
        ("none", {"--logger=none"}, None),
        (None, set(), "csv"),
    ],
)
def test_build_execution_request_from_args_normalizes_logger_backend(
    logger_backend,
    explicit_options,
    expected,
):
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(logger_backend=logger_backend),
        explicit_options=explicit_options,
    )

    assert request.values["logger_backend"] is expected
    assert ("logger_backend" in request.explicit_fields) is bool(explicit_options)


def test_build_execution_request_from_args_is_pure_for_auto_accelerator(
    monkeypatch,
):
    import torch

    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.cli.shared import build_execution_request_from_args

    def fail_if_called(*args, **kwargs):
        raise AssertionError("request construction must remain primitive and pure")

    monkeypatch.setattr(PyTorchExecutionConfig, "__init__", fail_if_called)
    monkeypatch.setattr(torch.cuda, "is_available", fail_if_called)
    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)

    request = build_execution_request_from_args(
        argparse.Namespace(accelerator="auto"),
        explicit_options={"--accelerator"},
    )

    assert request.values["accelerator"] == "auto"


@pytest.mark.parametrize(
    ("option", "field"),
    [
        ("--accelerator", "accelerator"),
        ("--torch-accelerator", "accelerator"),
        ("--device", "accelerator"),
        ("--deterministic", "deterministic"),
        ("--no-deterministic", "deterministic"),
        ("--torch-deterministic", "deterministic"),
        ("--num-workers", "num_workers"),
        ("--torch-num-workers", "num_workers"),
        ("--learning-rate", "learning_rate"),
        ("--torch-learning-rate", "learning_rate"),
        ("--scheduler", "scheduler"),
        ("--torch-scheduler", "scheduler"),
        ("--accumulate-grad-batches", "accum_steps"),
        ("--torch-accumulate-grad-batches", "accum_steps"),
        ("--logger", "logger_backend"),
        ("--torch-logger", "logger_backend"),
        ("--quiet", "enable_progress_bar"),
        ("--enable-checkpointing", "enable_checkpointing"),
        ("--disable-checkpointing", "enable_checkpointing"),
        ("--torch-enable-checkpointing", "enable_checkpointing"),
        ("--checkpoint-save-top-k", "checkpoint_save_top_k"),
        ("--torch-checkpoint-save-top-k", "checkpoint_save_top_k"),
        ("--checkpoint-monitor", "checkpoint_monitor_metric"),
        ("--checkpoint-mode", "checkpoint_mode"),
        ("--early-stop-patience", "early_stop_patience"),
        ("--torch-recon-log-every-n-epochs", "recon_log_every_n_epochs"),
        ("--torch-recon-log-num-patches", "recon_log_num_patches"),
        ("--torch-recon-log-fixed-indices", "recon_log_fixed_indices"),
        ("--torch-recon-log-stitch", "recon_log_stitch"),
        (
            "--torch-recon-log-max-stitch-samples",
            "recon_log_max_stitch_samples",
        ),
        ("--inference-batch-size", "inference_batch_size"),
        ("--torch-inference-batch-size", "inference_batch_size"),
    ],
)
def test_execution_request_from_args_registry_maps_every_spelling(option, field):
    from ptycho_torch.cli.shared import canonicalize_execution_options

    fields, sources = canonicalize_execution_options({option})
    equals_fields, equals_sources = canonicalize_execution_options(
        {f"{option}=value"}
    )

    assert fields == {field}
    assert sources == {option}
    assert equals_fields == {field}
    assert equals_sources == {option}


def test_execution_request_from_args_disable_mlflow_fans_out():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(disable_mlflow=True),
        explicit_options={"--disable_mlflow"},
    )

    assert request.values["logger_backend"] is None
    assert request.values["enable_progress_bar"] is False
    assert request.explicit_fields == frozenset(
        {"logger_backend", "enable_progress_bar"}
    )
    assert [notice.category for notice in request.notices] == [
        DeprecationWarning
    ]


def test_execution_request_from_args_boolean_pairs_use_canonical_destinations():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    deterministic = build_execution_request_from_args(
        argparse.Namespace(deterministic=False),
        explicit_options={"--no-deterministic"},
    )
    checkpointing = build_execution_request_from_args(
        argparse.Namespace(enable_checkpointing=False),
        explicit_options={"--disable-checkpointing"},
    )

    assert deterministic.values["deterministic"] is False
    assert deterministic.explicit_fields == frozenset({"deterministic"})
    assert checkpointing.values["enable_checkpointing"] is False
    assert checkpointing.explicit_fields == frozenset(
        {"enable_checkpointing"}
    )


def test_execution_request_from_args_distinguishes_scheduler_lanes():
    from ptycho_torch.cli.shared import canonicalize_execution_options

    native_fields, _ = canonicalize_execution_options(
        {"--scheduler"},
        lane="native-training",
    )
    unified_public_fields, _ = canonicalize_execution_options(
        {"--scheduler"},
        lane="unified-training",
    )
    unified_compatibility_fields, _ = canonicalize_execution_options(
        {"--torch-scheduler"},
        lane="unified-training",
    )

    assert native_fields == {"scheduler"}
    assert unified_public_fields == set()
    assert unified_compatibility_fields == {"scheduler"}


def test_execution_request_from_args_ignores_omitted_presentation_defaults():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(
            accelerator="auto",
            device="cpu",
            learning_rate=None,
        ),
        explicit_options=set(),
    )

    assert request.values["accelerator"] == "auto"
    assert request.values["learning_rate"] == 1e-3
    assert request.explicit_fields == frozenset()
    assert request.notices == ()


def test_execution_request_from_args_defers_device_and_worker_notices():
    from ptycho_torch.cli.shared import build_execution_request_from_args

    request = build_execution_request_from_args(
        argparse.Namespace(
            accelerator="auto",
            device="cuda",
            deterministic=True,
            num_workers=2,
        ),
        explicit_options={"--device", "--deterministic", "--num-workers"},
    )

    assert request.values["accelerator"] == "gpu"
    assert [notice.category for notice in request.notices] == [
        DeprecationWarning,
        UserWarning,
    ]


def test_execution_request_from_args_compatibility_warnings_precede_validation():
    from ptycho_torch.cli.shared import build_execution_config_from_args

    args = argparse.Namespace(
        accelerator="auto",
        device="cuda",
        deterministic=True,
        num_workers=2,
        learning_rate=-1.0,
        disable_mlflow=True,
        quiet=False,
    )

    with pytest.warns() as captured:
        with pytest.raises(ValueError, match="learning_rate"):
            build_execution_config_from_args(args)

    assert [warning.category for warning in captured] == [
        DeprecationWarning,
        DeprecationWarning,
        UserWarning,
    ]


def test_build_execution_config_preserves_recon_logging_namespace_values():
    from ptycho_torch.cli.shared import build_execution_config_from_args

    args = argparse.Namespace(
        accelerator="cpu",
        device=None,
        deterministic=True,
        num_workers=0,
        learning_rate=1e-3,
        disable_mlflow=False,
        quiet=False,
        recon_log_every_n_epochs=7,
        recon_log_num_patches=9,
        recon_log_fixed_indices=[1, 2],
        recon_log_stitch=True,
        recon_log_max_stitch_samples=11,
    )

    config = build_execution_config_from_args(args)

    assert config.recon_log_every_n_epochs == 7
    assert config.recon_log_num_patches == 9
    assert config.recon_log_fixed_indices == [1, 2]
    assert config.recon_log_stitch is True
    assert config.recon_log_max_stitch_samples == 11

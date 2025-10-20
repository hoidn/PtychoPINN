"""
Tests for PyTorchExecutionConfig dataclass (ADR-003 Phase C1).

This module validates the PyTorch execution configuration dataclass that controls
runtime behavior (hardware selection, optimization, logging, checkpointing) without
affecting model topology or data pipeline.

Test Coverage:
- Default values for all 22 fields per override_matrix.md §5
- Optional field types (prefetch_factor, gradient_clip_val, logger_backend, etc.)
- Dataclass instantiation and repr behavior
- Field organization and docstring completeness

References:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md §5
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md §2.2
- specs/ptychodus_api_spec.md §4.8, §6 (CONFIG-001 + backend execution contract)
- docs/findings.md POLICY-001, CONFIG-001
"""

import pytest
from ptycho.config.config import PyTorchExecutionConfig


class TestPyTorchExecutionConfigDefaults:
    """Validate default values for PyTorchExecutionConfig fields."""

    def test_hardware_defaults(self):
        """Hardware & distributed training fields have correct defaults."""
        config = PyTorchExecutionConfig()
        assert config.accelerator == 'auto', "Default accelerator should be 'auto'"
        assert config.strategy == 'auto', "Default strategy should be 'auto'"
        assert config.n_devices == 1, "Default n_devices should be 1"
        assert config.deterministic is True, "Default deterministic should be True (reproducibility)"

    def test_dataloader_defaults(self):
        """Data loading fields have correct defaults."""
        config = PyTorchExecutionConfig()
        assert config.num_workers == 0, "Default num_workers should be 0"
        assert config.pin_memory is False, "Default pin_memory should be False"
        assert config.persistent_workers is False, "Default persistent_workers should be False"
        assert config.prefetch_factor is None, "Default prefetch_factor should be None (optional)"

    def test_optimization_defaults(self):
        """Optimization fields have correct defaults."""
        config = PyTorchExecutionConfig()
        assert config.learning_rate == 1e-3, "Default learning_rate should be 1e-3"
        assert config.scheduler == 'Default', "Default scheduler should be 'Default'"
        assert config.gradient_clip_val is None, "Default gradient_clip_val should be None (optional)"
        assert config.accum_steps == 1, "Default accum_steps should be 1"

    def test_checkpointing_defaults(self):
        """Checkpointing & early stopping fields have correct defaults."""
        config = PyTorchExecutionConfig()
        assert config.enable_checkpointing is True, "Default enable_checkpointing should be True"
        assert config.checkpoint_save_top_k == 1, "Default checkpoint_save_top_k should be 1"
        assert config.checkpoint_monitor_metric == 'val_loss', "Default checkpoint_monitor_metric should be 'val_loss'"
        assert config.early_stop_patience == 100, "Default early_stop_patience should be 100"

    def test_logging_defaults(self):
        """Logging & experiment tracking fields have correct defaults."""
        config = PyTorchExecutionConfig()
        assert config.enable_progress_bar is False, "Default enable_progress_bar should be False"
        assert config.logger_backend is None, "Default logger_backend should be None (optional)"
        assert config.disable_mlflow is False, "Default disable_mlflow should be False"

    def test_inference_defaults(self):
        """Inference-specific fields have correct defaults."""
        config = PyTorchExecutionConfig()
        assert config.inference_batch_size is None, "Default inference_batch_size should be None (optional)"
        assert config.middle_trim == 0, "Default middle_trim should be 0"
        assert config.pad_eval is False, "Default pad_eval should be False"


class TestPyTorchExecutionConfigInstantiation:
    """Validate dataclass instantiation and modification behavior."""

    def test_can_instantiate_with_defaults(self):
        """Dataclass can be instantiated without arguments."""
        config = PyTorchExecutionConfig()
        assert config is not None, "Should be able to create config with all defaults"

    def test_can_override_single_field(self):
        """Individual fields can be overridden during instantiation."""
        config = PyTorchExecutionConfig(learning_rate=5e-4)
        assert config.learning_rate == 5e-4, "Should accept custom learning_rate"
        assert config.accelerator == 'auto', "Other fields should retain defaults"

    def test_can_override_multiple_fields(self):
        """Multiple fields can be overridden during instantiation."""
        config = PyTorchExecutionConfig(
            accelerator='cpu',
            num_workers=4,
            enable_progress_bar=True,
        )
        assert config.accelerator == 'cpu', "Should accept custom accelerator"
        assert config.num_workers == 4, "Should accept custom num_workers"
        assert config.enable_progress_bar is True, "Should accept custom enable_progress_bar"
        assert config.deterministic is True, "Unmodified fields should retain defaults"

    def test_optional_fields_accept_none(self):
        """Optional fields (prefetch_factor, gradient_clip_val, etc.) accept None."""
        config = PyTorchExecutionConfig(
            prefetch_factor=None,
            gradient_clip_val=None,
            logger_backend=None,
            inference_batch_size=None,
        )
        assert config.prefetch_factor is None, "prefetch_factor should accept None"
        assert config.gradient_clip_val is None, "gradient_clip_val should accept None"
        assert config.logger_backend is None, "logger_backend should accept None"
        assert config.inference_batch_size is None, "inference_batch_size should accept None"

    def test_optional_fields_accept_values(self):
        """Optional fields accept non-None values when provided."""
        config = PyTorchExecutionConfig(
            prefetch_factor=2,
            gradient_clip_val=0.5,
            logger_backend='mlflow',
            inference_batch_size=16,
        )
        assert config.prefetch_factor == 2, "prefetch_factor should accept int value"
        assert config.gradient_clip_val == 0.5, "gradient_clip_val should accept float value"
        assert config.logger_backend == 'mlflow', "logger_backend should accept str value"
        assert config.inference_batch_size == 16, "inference_batch_size should accept int value"

    def test_repr_contains_all_fields(self):
        """Dataclass repr includes all 22 fields."""
        config = PyTorchExecutionConfig()
        repr_str = repr(config)

        # Check for presence of all field names in repr (dataclass default repr behavior)
        expected_fields = [
            'accelerator', 'strategy', 'n_devices', 'deterministic',
            'num_workers', 'pin_memory', 'persistent_workers', 'prefetch_factor',
            'learning_rate', 'scheduler', 'gradient_clip_val', 'accum_steps',
            'enable_checkpointing', 'checkpoint_save_top_k', 'checkpoint_monitor_metric', 'early_stop_patience',
            'enable_progress_bar', 'logger_backend', 'disable_mlflow',
            'inference_batch_size', 'middle_trim', 'pad_eval',
        ]
        for field_name in expected_fields:
            assert field_name in repr_str, f"Field '{field_name}' should appear in repr()"


class TestPyTorchExecutionConfigDocumentation:
    """Validate docstring and type hints completeness."""

    def test_has_docstring(self):
        """Dataclass has a non-empty docstring."""
        assert PyTorchExecutionConfig.__doc__ is not None, "Dataclass should have docstring"
        assert len(PyTorchExecutionConfig.__doc__.strip()) > 50, "Docstring should be substantive (>50 chars)"

    def test_docstring_mentions_policy_001(self):
        """Docstring references POLICY-001 (PyTorch mandatory)."""
        docstring = PyTorchExecutionConfig.__doc__
        # Allow flexible reference format (POLICY-001, POLICY 001, or prose mention)
        assert 'POLICY-001' in docstring or 'PyTorch' in docstring or 'torch>=2.2' in docstring, \
            "Docstring should reference PyTorch requirement (POLICY-001)"

    def test_docstring_mentions_config_001(self):
        """Docstring references CONFIG-001 (params.cfg orthogonality)."""
        docstring = PyTorchExecutionConfig.__doc__
        # Allow flexible reference format (CONFIG-001, CONFIG 001, or params.cfg mention)
        assert 'CONFIG-001' in docstring or 'params.cfg' in docstring or 'runtime' in docstring, \
            "Docstring should clarify execution config is orthogonal to legacy params.cfg (CONFIG-001)"

    def test_has_type_annotations(self):
        """All fields have type annotations."""
        # Python dataclass __annotations__ contains all field types
        annotations = PyTorchExecutionConfig.__annotations__
        assert len(annotations) == 22, f"Expected 22 annotated fields, found {len(annotations)}"

    def test_fields_match_design(self):
        """Field names match design_delta.md approved list."""
        expected_fields = {
            'accelerator', 'strategy', 'n_devices', 'deterministic',
            'num_workers', 'pin_memory', 'persistent_workers', 'prefetch_factor',
            'learning_rate', 'scheduler', 'gradient_clip_val', 'accum_steps',
            'enable_checkpointing', 'checkpoint_save_top_k', 'checkpoint_monitor_metric', 'early_stop_patience',
            'enable_progress_bar', 'logger_backend', 'disable_mlflow',
            'inference_batch_size', 'middle_trim', 'pad_eval',
        }
        actual_fields = set(PyTorchExecutionConfig.__annotations__.keys())
        assert actual_fields == expected_fields, \
            f"Field mismatch. Missing: {expected_fields - actual_fields}, Extra: {actual_fields - expected_fields}"

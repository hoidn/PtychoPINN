"""
Test suite for ptycho_torch.api deprecation warnings per ADR-003 Phase E.C1.

Validates that legacy API entry points emit DeprecationWarning with migration
guidance steering users toward factory-driven workflows documented in
docs/workflows/pytorch.md.

Reference:
- SPEC: specs/ptychodus_api_spec.md (no explicit deprecation contract, but
  follows CLI logger deprecation semantics at §4.9)
- ARCH: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/
        phase_e_governance_adr_addendum/adr_addendum.md:295-334
  (defers legacy API decision to Phase E.C1)
- WORKFLOW: docs/workflows/pytorch.md:188-196 flags ptycho_torch/api as
            deprecated surface needing migration instructions

Test Strategy:
- RED Phase: Import legacy modules expecting DeprecationWarning via
  warnings.catch_warnings context manager (stacklevel=2 ensures caller sees
  accurate origin).
- GREEN Phase: After warn_legacy_api_import implementation, validate warning
  message content includes migration keywords (ptycho_train_torch, config_factory).
- No behavior changes: legacy modules remain functional; only messaging added.
"""

import sys
import warnings
import pytest


class TestLegacyAPIDeprecation:
    """
    Validate DeprecationWarning emission for ptycho_torch.api legacy entry points.

    Per input.md guidance:
    - Use warnings.warn(..., DeprecationWarning, stacklevel=2) for accurate stack origin
    - Ensure tests clear sys.modules or use importlib.reload for reliable warning capture
    - Keep warning message consistent across modules (centralized in __init__.py)
    """

    def test_example_train_import_emits_deprecation_warning(self):
        """
        Test that importing ptycho_torch.api package emits DeprecationWarning.

        RED Expectation:
        - AssertionError or no warning captured (warn_legacy_api_import not yet implemented)

        GREEN Expectation:
        - Exactly one DeprecationWarning captured
        - Warning message contains migration guidance keywords:
          * 'ptycho_train_torch' or 'CLI'
          * 'config_factory' or 'factory'
          * 'deprecated' or 'legacy'

        Evidence Parameter Validation:
        - Source: ptycho_torch/api/__init__.py (legacy API package init)
        - Mechanism: import triggers __init__.py module-level warn_legacy_api_import() call
        - Validation: stacklevel=2 ensures warning points to test frame, not __init__.py

        Note: We import the package (not example_train.py) to avoid executing
        hardcoded training logic that expects GPU and specific data paths.
        """
        # Clear module cache to ensure fresh import triggers warning
        if 'ptycho_torch.api' in sys.modules:
            del sys.modules['ptycho_torch.api']
        # Also clear submodules that might be cached
        for key in list(sys.modules.keys()):
            if key.startswith('ptycho_torch.api.'):
                del sys.modules[key]

        # Capture warnings during import
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")  # Ensure DeprecationWarning not filtered

            # Import legacy API package (should trigger warning at __init__.py)
            import ptycho_torch.api  # noqa: F401

            # Validate exactly one DeprecationWarning emitted (filter out distutils noise)
            deprecation_warnings = [
                w for w in caught_warnings
                if issubclass(w.category, DeprecationWarning)
                and 'ptycho_torch.api' in str(w.message).lower()
            ]
            assert len(deprecation_warnings) == 1, (
                f"Expected exactly 1 ptycho_torch.api DeprecationWarning, got {len(deprecation_warnings)}. "
                f"All captured warnings: {[(w.category.__name__, str(w.message)) for w in caught_warnings]}"
            )

            # Validate warning message contains migration keywords
            warning_message = str(deprecation_warnings[0].message).lower()
            assert 'deprecated' in warning_message or 'legacy' in warning_message, (
                f"Warning message missing 'deprecated' or 'legacy': {warning_message}"
            )
            assert 'ptycho_train_torch' in warning_message or 'cli' in warning_message, (
                f"Warning message missing CLI entry point guidance: {warning_message}"
            )
            assert 'config_factory' in warning_message or 'factory' in warning_message, (
                f"Warning message missing factory workflow guidance: {warning_message}"
            )

    def test_api_package_import_is_idempotent(self):
        """
        Test that repeated imports of ptycho_torch.api emit warning only once.

        Validates that DeprecationWarning is not suppressed by Python's warning
        filter after first emission, ensuring users see the message on first use.

        Strategy: Import package twice without clearing sys.modules between imports.
        Expected: Warning fires on first import, not on second (Python default behavior).
        """
        # Clear module cache for clean slate
        if 'ptycho_torch.api' in sys.modules:
            del sys.modules['ptycho_torch.api']
        for key in list(sys.modules.keys()):
            if key.startswith('ptycho_torch.api.'):
                del sys.modules[key]

        # First import - should emit warning
        with warnings.catch_warnings(record=True) as caught_warnings_first:
            warnings.simplefilter("always")
            import ptycho_torch.api  # noqa: F401
            deprecation_warnings_first = [w for w in caught_warnings_first if issubclass(w.category, DeprecationWarning)]

        # Second import (module already loaded) - may or may not emit depending on implementation
        with warnings.catch_warnings(record=True) as caught_warnings_second:
            warnings.simplefilter("always")
            import ptycho_torch.api  # noqa: F401, F811
            deprecation_warnings_second = [w for w in caught_warnings_second if issubclass(w.category, DeprecationWarning)]

        # Filter for ptycho_torch.api warnings only (ignore distutils noise)
        api_warnings_first = [
            w for w in deprecation_warnings_first
            if 'ptycho_torch.api' in str(w.message).lower()
        ]
        api_warnings_second = [
            w for w in deprecation_warnings_second
            if 'ptycho_torch.api' in str(w.message).lower()
        ]

        # Validate first import emitted warning (or AssertionError if not implemented)
        assert len(api_warnings_first) >= 1, (
            f"Expected at least 1 ptycho_torch.api DeprecationWarning on first import, got {len(api_warnings_first)}"
        )

        # Second import should not re-emit (module already loaded in sys.modules)
        assert len(api_warnings_second) == 0, (
            f"Expected 0 ptycho_torch.api DeprecationWarning on second import (module cached), got {len(api_warnings_second)}"
        )

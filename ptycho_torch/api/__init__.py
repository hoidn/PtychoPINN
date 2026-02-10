"""
Legacy API surface for ptycho_torch (DEPRECATED).

This module provided an earlier workflow interface that predates the
factory-driven design documented in ADR-003. New code should use:

  - CLI entry points: ptycho_train_torch, ptycho_infer_torch
  - Programmatic API: ptycho_torch.config_factory functions
  - Workflows: ptycho_torch.workflows.components

See docs/workflows/pytorch.md for migration examples.

Phase E.C1 Deprecation Strategy (ADR-003):
- Emit DeprecationWarning on first import (module-level, stacklevel=2)
- Leave behavior unchanged (no breaking changes)
- Centralize warning text here to ensure consistency across api/ submodules

Reference:
- ARCH: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/
        phase_e_governance_adr_addendum/adr_addendum.md:295-334
- SPEC: specs/ptychodus_api_spec.md:300-307 (CLI logger deprecation semantics)
- WORKFLOW: docs/workflows/pytorch.md:188-196
"""

import warnings


def _warn_legacy_api_import():
    """
    Emit DeprecationWarning steering users to factory-driven workflows.

    Called at module init (ptycho_torch.api.__init__.py) to ensure warning
    fires once per Python session on first import.

    Stacklevel=2 ensures the warning points to the caller's import statement,
    not this function's frame.

    Migration Guidance:
    - Training CLI: Use `ptycho_train_torch` instead of custom scripts
    - Inference CLI: Use `ptycho_infer_torch` for reconstruction
    - Programmatic: Use `ptycho_torch.config_factory.create_training_payload()`
    - Workflows: Import from `ptycho_torch.workflows.components`

    Per input.md guidance:
    - Keep warning message consistent (centralized here, not per submodule)
    - No behavior changes (only messaging)
    - Avoid hardcoding filesystem paths (reference CLI entry points instead)

    Evidence:
    - SPEC: specs/ptychodus_api_spec.md:300-307 documents CLI deprecation pattern
    - WORKFLOW: docs/workflows/pytorch.md sections 12-13 documents factory API
    """
    warnings.warn(
        "ptycho_torch.api is deprecated and will be removed in a future release. "
        "The legacy API predates the factory-driven configuration design (ADR-003). "
        "Please migrate to the standardized workflows:\n"
        "  - Training CLI: ptycho_train_torch (see --help for options)\n"
        "  - Inference CLI: ptycho_infer_torch\n"
        "  - Programmatic API: ptycho_torch.config_factory functions\n"
        "    (create_training_payload, create_inference_payload)\n"
        "  - Workflow components: ptycho_torch.workflows.components\n"
        "For migration examples, see docs/workflows/pytorch.md sections 12-13.",
        DeprecationWarning,
        stacklevel=2
    )


# Emit warning on first import
_warn_legacy_api_import()

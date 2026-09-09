"""
Legacy API surface for ptycho_torch (DEPRECATED).

This module provided an earlier workflow interface that predates the
public training/reconstruction design in specs/ptychodus_api_spec.md §4.8. New code should use:

  - CLI entry points: python -m ptycho_torch.train, python -m ptycho_torch.inference
  - Programmatic API: ptycho_torch.train.train, ptycho_torch.inference.reconstruct

See docs/workflows/pytorch.md for migration examples.

Phase E.C1 Deprecation Strategy (config-factory contract; docs/specs/spec-ptycho-config-bridge.md):
- Emit DeprecationWarning on first import (module-level, stacklevel=2)
- Leave behavior unchanged (no breaking changes)
- Centralize warning text here to ensure consistency across api/ submodules

Reference:
- ARCH: docs/findings.md (see git history for the originating plan)
        phase_e_governance_adr_addendum/adr_addendum.md:295-334
- SPEC: specs/ptychodus_api_spec.md §4.8 (public training and reconstruction)
- WORKFLOW: docs/workflows/pytorch.md
"""

import warnings


def _warn_legacy_api_import():
    """
    Emit DeprecationWarning steering users to public train/reconstruct workflows.

    Called at module init (ptycho_torch.api.__init__.py) to ensure warning
    fires once per Python session on first import.

    Stacklevel=2 ensures the warning points to the caller's import statement,
    not this function's frame.

    Migration Guidance:
    - Training CLI: Use `python -m ptycho_torch.train`
    - Inference CLI: Use `python -m ptycho_torch.inference`
    - Programmatic: Use `ptycho_torch.train.train` and
      `ptycho_torch.inference.reconstruct`

    Per input.md guidance:
    - Keep warning message consistent (centralized here, not per submodule)
    - No behavior changes (only messaging)
    - Avoid hardcoding filesystem paths (reference CLI entry points instead)

    Evidence:
    - SPEC: specs/ptychodus_api_spec.md §4.8 owns the public doors
    - WORKFLOW: docs/workflows/pytorch.md documents their use
    """
    warnings.warn(
        "ptycho_torch.api is deprecated and will be removed in a future release. "
        "The legacy API predates the public training/reconstruction API (specs/ptychodus_api_spec.md). "
        "Please migrate to the standardized workflows:\n"
        "  - Training CLI: python -m ptycho_torch.train (see --help for options)\n"
        "  - Inference CLI: python -m ptycho_torch.inference\n"
        "  - Programmatic API: ptycho_torch.train.train and\n"
        "    ptycho_torch.inference.reconstruct\n"
        "For migration examples, see docs/workflows/pytorch.md.",
        DeprecationWarning,
        stacklevel=2
    )


# Emit warning on first import
_warn_legacy_api_import()

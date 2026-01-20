"""Compatibility shim that adds the legacy update_params helper."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def _load_real_module() -> ModuleType:
    repo_root = Path(
        os.environ.get("DOSE_REAL_REPO", "/home/ollie/Documents/PtychoPINN")
    )
    real_path = repo_root / "ptycho" / "workflows" / "components.py"
    spec = importlib.util.spec_from_file_location("_legacy_components_impl", real_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load real components module at {real_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REAL = _load_real_module()

# Re-export every public symbol from the real module.
for _name in dir(_REAL):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_REAL, _name)


def update_params(mapping):
    """Legacy helper used by the old dose_experiments simulation CLI."""
    from ptycho import params as legacy_params  # Imported lazily to avoid cycles

    legacy_params.cfg.update(mapping)

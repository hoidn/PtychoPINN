"""Compatibility wrapper for legacy imports."""

from __future__ import annotations

import warnings

from ptycho.cache import memoize_raw_data

warnings.warn(
    "scripts.simulation.cache_utils is deprecated; use ptycho.cache",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["memoize_raw_data"]


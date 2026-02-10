"""Parity debug helpers (env-gated).

Use PTYCHO_DEBUG_PARITY=1 to enable logging.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

_LOGGER = logging.getLogger("ptycho.debug_parity")


def parity_debug_enabled() -> bool:
    """Return True when parity debug logging is enabled."""
    value = os.getenv("PTYCHO_DEBUG_PARITY", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _to_numpy(array: Any) -> Optional[np.ndarray]:
    if array is None:
        return None
    # Torch tensor
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        try:
            array = array.cpu()
        except Exception:
            pass
    # TF tensor or numpy-compatible
    if hasattr(array, "numpy"):
        try:
            array = array.numpy()
        except Exception:
            pass
    return np.asarray(array)


def summarize_array(array: Any) -> Dict[str, Any]:
    arr = _to_numpy(array)
    if arr is None:
        return {"shape": None, "dtype": None, "min": None, "max": None, "mean": None, "std": None}

    if arr.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def summarize_offsets(offsets: Any) -> Dict[str, Any]:
    arr = _to_numpy(offsets)
    if arr is None:
        return {"shape": None, "dtype": None, "unique_count": None, "std": None}

    if arr.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "unique_count": 0,
            "std": None,
        }

    if arr.ndim >= 2:
        flat = arr.reshape(-1, arr.shape[-1])
    else:
        flat = arr.reshape(-1, 1)

    try:
        unique_count = int(np.unique(flat, axis=0).shape[0])
    except Exception:
        unique_count = None

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "unique_count": unique_count,
        "std": float(arr.std()),
    }


def _log_stats(label: str, stats: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    logger = logger or _LOGGER
    root_logger = logging.getLogger()
    message = f"[PARITY] {label}: {stats}"
    if not logger.handlers and not root_logger.handlers:
        print(message)
        return
    if root_logger.getEffectiveLevel() > logging.INFO:
        print(message)
        return
    logger.info(message)


def log_array_stats(label: str, array: Any, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    if not parity_debug_enabled():
        return None
    stats = summarize_array(array)
    _log_stats(label, stats, logger)
    return stats


def log_offsets_stats(label: str, offsets: Any, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    if not parity_debug_enabled():
        return None
    stats = summarize_offsets(offsets)
    _log_stats(label, stats, logger)
    return stats

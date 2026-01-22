"""Disk-backed memoization helpers for caching RawData payloads."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ptycho.raw_data import RawData

__all__ = ["memoize_raw_data"]


def _hash_numpy(array: np.ndarray) -> str:
    """Return a stable hash for a NumPy array regardless of contiguity."""
    array = np.ascontiguousarray(array)
    payload = {
        "dtype": str(array.dtype),
        "shape": array.shape,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_for_hash(value: Any) -> Any:
    """Normalize common argument types so hashing is deterministic."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": _hash_numpy(value)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(val) for key, val in value.items()}
    return value


def _hash_payload(payload: dict) -> str:
    normalized = _normalize_for_hash(payload)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def memoize_raw_data(
    *,
    default_cache_dir: Path,
    cache_prefix: str,
    exclude_keys: Iterable[str] | None = None,
) -> Callable[[Callable[..., RawData]], Callable[..., RawData]]:
    """Cache RawData-returning functions to disk using a hashed argument key."""
    excluded = set(exclude_keys or [])
    excluded.update({"use_cache", "cache_dir"})

    def decorator(func: Callable[..., RawData]) -> Callable[..., RawData]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> RawData:
            bound = signature.bind_partial(*args, **kwargs)
            arguments = dict(bound.arguments)
            use_cache = bool(arguments.pop("use_cache", True))
            cache_dir = arguments.pop("cache_dir", None)
            for key in excluded:
                arguments.pop(key, None)

            cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir
            cache_key = _hash_payload(arguments)
            cache_path = cache_root / f"{cache_prefix}_{cache_key}.npz"

            if use_cache:
                cache_root.mkdir(parents=True, exist_ok=True)
                if cache_path.exists():
                    logging.info("Loading cached %s: %s", cache_prefix, cache_path)
                    return RawData.from_file(str(cache_path))

            result = func(*bound.args, **bound.kwargs)
            if not isinstance(result, RawData):
                raise TypeError("memoize_raw_data expects RawData return type")

            if use_cache:
                logging.info("Caching %s: %s", cache_prefix, cache_path)
                result.to_file(str(cache_path))

            return result

        return wrapper

    return decorator

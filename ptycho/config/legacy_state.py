"""Lifecycle containment for the legacy process-local parameter dictionary."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import os
import threading
from typing import Any, Callable, Iterator, Mapping, ParamSpec, TypeVar

from ptycho import params


P = ParamSpec("P")
R = TypeVar("R")


_state_lock = threading.Lock()
_owner_identity: tuple[int, int | None] | None = None


@dataclass
class _Frame:
    snapshot: dict
    sealed: bool
    archive: bool = False


_frames: list[_Frame] = []


def _reset_after_fork() -> None:
    global _state_lock, _owner_identity, _frames

    _state_lock = threading.Lock()
    _owner_identity = None
    _frames = []


os.register_at_fork(after_in_child=_reset_after_fork)


def _execution_identity() -> tuple[int, int | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), id(task) if task is not None else None


def _push_frame(*, archive: bool = False) -> None:
    global _owner_identity

    identity = _execution_identity()
    with _state_lock:
        if _owner_identity is not None and _owner_identity != identity:
            raise RuntimeError("legacy params scope is already active in another execution context")
        _owner_identity = identity
        _frames.append(_Frame(dict(params.cfg), params._sealed, archive))


def _restore_frame(frame: _Frame) -> None:
    params.cfg.clear()
    params.cfg.update(frame.snapshot)
    if frame.sealed:
        params.seal()
    else:
        params.unseal()


@contextmanager
def legacy_params_scope() -> Iterator[dict]:
    """Restore ``params.cfg`` in place when a bounded operation finishes."""
    global _owner_identity

    _push_frame()
    try:
        yield params.cfg
    finally:
        with _state_lock:
            frame = _frames.pop()
            _restore_frame(frame)
            if not _frames:
                _owner_identity = None


@contextmanager
def archived_params_scope(values: Mapping[str, Any]) -> Iterator[dict]:
    """Apply archived state transactionally and commit it after successful load."""
    global _owner_identity

    _push_frame(archive=True)
    try:
        params.cfg.update(values)
        yield params.cfg
    except BaseException:
        with _state_lock:
            frame = _frames.pop()
            _restore_frame(frame)
            if not _frames:
                _owner_identity = None
        raise
    else:
        with _state_lock:
            _frames.pop()
            if not any(frame.archive for frame in _frames):
                committed = dict(params.cfg)
                sealed = params._sealed
                for frame in _frames:
                    frame.snapshot = dict(committed)
                    frame.sealed = sealed
            if not _frames:
                _owner_identity = None


def scoped_legacy_params(function: Callable[P, R]) -> Callable[P, R]:
    """Run a supported entrypoint with temporary legacy state containment."""
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        identity = _execution_identity()
        with _state_lock:
            nested_in_parent = _owner_identity == identity
        if nested_in_parent:
            return function(*args, **kwargs)
        with legacy_params_scope():
            return function(*args, **kwargs)

    return wrapper


@contextmanager
def configured_params_scope() -> Iterator[dict]:
    """Commit a configuration update locally, rolling it back on failure."""
    global _owner_identity

    _push_frame()
    try:
        yield params.cfg
    except BaseException:
        with _state_lock:
            frame = _frames.pop()
            _restore_frame(frame)
            if not _frames:
                _owner_identity = None
        raise
    else:
        with _state_lock:
            _frames.pop()
            if not _frames:
                _owner_identity = None


def configured_legacy_params(function: Callable[P, R]) -> Callable[P, R]:
    """Apply a persistent config bridge bounded by any surrounding workflow."""
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with configured_params_scope():
            return function(*args, **kwargs)

    return wrapper


def transactional_legacy_params(function: Callable[P, R]) -> Callable[P, R]:
    """Commit legacy state only when a supported load entrypoint succeeds."""
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with archived_params_scope({}):
            return function(*args, **kwargs)

    return wrapper

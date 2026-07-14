"""Torch-free runtime error types and staging helpers.

Extracted from :mod:`runtime_planning` / :mod:`runtime_execution` so that
dry-run-safe consumers (the reference-qualification spec layer) can import
the typed errors without transitively importing Torch through the
configuration module. The historical import paths keep working: both source
modules re-export these names.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class StudyRequestError(ValueError):
    """A user-facing request/validation error (CLI exit code 2)."""


class RuntimeExecutionError(RuntimeError):
    """A canonical-run failure attributed to one named execution stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Attribute any escaping exception to the named execution stage."""
    try:
        yield
    except RuntimeExecutionError:
        raise
    except Exception as error:
        raise RuntimeExecutionError(name, f"{type(error).__name__}: {error}") from error


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

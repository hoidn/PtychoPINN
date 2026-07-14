"""Race-safe immutable file publication helpers for study evidence."""

from __future__ import annotations

import ctypes
import errno
import os
import tempfile
from pathlib import Path

from .runtime_errors import RuntimeExecutionError

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def atomic_rename_directory_no_replace(
    source: Path, destination: Path, *, stage: str
) -> None:
    """Atomically publish a directory without replacing a competing path."""
    source = Path(source)
    destination = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeExecutionError(
            stage, "renameat2(RENAME_NOREPLACE) is unavailable; refusing publication"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RuntimeExecutionError(
            stage,
            f"refusing to overwrite concurrently published artifact at {destination}",
        )
    raise RuntimeExecutionError(
        stage,
        "no-replace directory publication failed for "
        f"{destination}: {os.strerror(error_number)}",
    )


def atomic_write_bytes_no_clobber(path: Path, data: bytes, *, stage: str) -> None:
    """Atomically create ``path`` without replacing existing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise RuntimeExecutionError(
                stage, f"refusing to overwrite existing artifact at {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

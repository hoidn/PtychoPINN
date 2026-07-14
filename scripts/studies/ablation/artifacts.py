"""Reproducible fingerprints and atomic artifacts for Torch ablation runs."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .manifest import FrozenDict


class ArtifactError(Exception):
    """Base class for ablation artifact contract failures."""


class FingerprintError(ArtifactError, ValueError):
    """A fingerprint input cannot be represented canonically."""


class ArtifactPathError(ArtifactError, ValueError):
    """An artifact path is unsafe or not canonical."""


class CompletedAttemptError(ArtifactError):
    """An operation would overwrite a completed attempt."""


class IncompleteAttemptError(ArtifactError):
    """An interrupted attempt must be preserved and restarted."""


class CompletionRefusedError(ArtifactError):
    """A completed attempt cannot be reused as requested."""


class CompletionMismatchError(CompletionRefusedError):
    """Completion fingerprints do not match the requested run."""


class CorruptCompletionError(CompletionRefusedError):
    """A completion record or one of its artifacts is corrupt."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strictly JSON-native finite data in canonical compact form."""
    _validate_json_native(value, path="$", error_type=FingerprintError)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_json_native(
    value: Any, *, path: str, error_type: type[ArtifactError]
) -> None:
    value_type = type(value)
    if value is None or value_type in (bool, str, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise error_type(f"{path} must contain only finite JSON-native numbers")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_native(item, path=f"{path}[{index}]", error_type=error_type)
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise error_type(f"{path} JSON objects must have string keys")
            _validate_json_native(
                item, path=f"{path}.{key}", error_type=error_type
            )
        return
    raise error_type(
        f"{path} must contain only JSON-native values; got {value_type.__name__}"
    )


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise FingerprintError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise FingerprintError(f"{field_name} must be an integer, not bool")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    digest = _require_string(value, field_name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FingerprintError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _require_normalized_relative_path(value: object, field_name: str) -> str:
    path_text = _require_string(value, field_name)
    if "\x00" in path_text:
        raise FingerprintError(f"{field_name} must not contain NUL")
    if "\\" in path_text:
        raise FingerprintError(f"{field_name} must be a normalized relative POSIX path")
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or path_text != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise FingerprintError(f"{field_name} must be a normalized relative path")
    return path_text


def _normalize_typed_json(value: Any, *, frozen_context: bool = False) -> Any:
    """Thaw only tuples owned by this module's immutable JSON representation."""
    if type(value) is FrozenDict:
        return {
            key: _normalize_typed_json(item, frozen_context=True)
            for key, item in value.items()
        }
    if type(value) is tuple:
        if not frozen_context:
            raise FingerprintError("raw tuple is not a JSON-native value")
        return [
            _normalize_typed_json(item, frozen_context=True) for item in value
        ]
    if type(value) is dict:
        return {
            key: _normalize_typed_json(item, frozen_context=False)
            for key, item in value.items()
        }
    if type(value) is list:
        return [
            _normalize_typed_json(item, frozen_context=False) for item in value
        ]
    return value


@dataclass(frozen=True)
class UntrackedSource:
    """Identity of one relevant untracked source used by a dirty run."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_normalized_relative_path(self.path, "untracked source path")
        _require_sha256(self.sha256, "untracked source sha256")


@dataclass(frozen=True)
class GitIdentity:
    """Git commit and sufficient clean/dirty checkout evidence."""

    commit: str
    clean: bool
    tracked_patch_sha256: str | None = None
    untracked_sources: tuple[UntrackedSource, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.commit, "git commit")
        if type(self.clean) is not bool:
            raise FingerprintError("git clean must be bool, not an integer")
        if type(self.untracked_sources) is not tuple or any(
            type(item) is not UntrackedSource for item in self.untracked_sources
        ):
            raise FingerprintError(
                "git untracked_sources must be a tuple of UntrackedSource values"
            )
        ordered = tuple(sorted(self.untracked_sources, key=lambda item: item.path))
        if len({item.path for item in ordered}) != len(ordered):
            raise FingerprintError("duplicate relevant untracked source path")
        object.__setattr__(self, "untracked_sources", ordered)
        if self.clean:
            if self.tracked_patch_sha256 is not None or ordered:
                raise FingerprintError("clean git identity cannot contain dirty evidence")
        else:
            if self.tracked_patch_sha256 is None:
                raise FingerprintError("dirty git identity requires tracked patch SHA-256")
            _require_sha256(self.tracked_patch_sha256, "tracked patch sha256")

    @property
    def claim_grade(self) -> bool:
        """Only a clean checkout is eligible for claim-grade output."""
        return self.clean

    def _json_value(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "commit": self.commit,
            "tracked_patch_sha256": self.tracked_patch_sha256,
            "untracked_sources": [
                {"path": item.path, "sha256": item.sha256}
                for item in self.untracked_sources
            ],
        }


@dataclass(frozen=True)
class TrainingFingerprintInput:
    """Complete immutable input snapshot for a training fingerprint."""

    schema_version: int
    manifest_sha256: str
    logical_run_id: str
    resolved_configs: dict[str, Any] | FrozenDict
    seed: int
    git: GitIdentity
    environment_digest: str
    content_sha256s: dict[str, str] | FrozenDict
    _canonical_payload: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        version = _require_int(self.schema_version, "schema_version")
        if version <= 0:
            raise FingerprintError("schema_version must be positive")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_string(self.logical_run_id, "logical_run_id")
        _require_int(self.seed, "seed")
        if type(self.git) is not GitIdentity:
            raise FingerprintError("git must be a GitIdentity")
        _require_sha256(self.environment_digest, "environment_digest")
        if type(self.resolved_configs) not in (dict, FrozenDict):
            raise FingerprintError("resolved_configs must be a JSON-native object")
        if type(self.content_sha256s) not in (dict, FrozenDict) or not self.content_sha256s:
            raise FingerprintError("content_sha256s must be a non-empty object")
        resolved_configs = _normalize_typed_json(self.resolved_configs)
        content_sha256s = _normalize_typed_json(self.content_sha256s)
        for name, digest in content_sha256s.items():
            _require_string(name, "content_sha256s key")
            _require_sha256(digest, f"content_sha256s.{name}")

        payload = {
            "content_sha256s": content_sha256s,
            "environment_digest": self.environment_digest,
            "git": self.git._json_value(),
            "logical_run_id": self.logical_run_id,
            "manifest_sha256": self.manifest_sha256,
            "resolved_configs": resolved_configs,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }
        canonical = canonical_json_bytes(payload)
        object.__setattr__(self, "_canonical_payload", canonical)
        object.__setattr__(self, "resolved_configs", FrozenDict(resolved_configs))
        object.__setattr__(self, "content_sha256s", FrozenDict(content_sha256s))

    @property
    def claim_grade(self) -> bool:
        """Whether this input is eligible to produce claim-grade artifacts."""
        return self.git.claim_grade


@dataclass(frozen=True)
class InferenceFingerprintInput:
    """Training identity plus the exact checkpoint selected for inference."""

    training: TrainingFingerprintInput
    selected_checkpoint_sha256: str
    _canonical_payload: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.training) is not TrainingFingerprintInput:
            raise FingerprintError("training must be a TrainingFingerprintInput")
        _require_sha256(
            self.selected_checkpoint_sha256, "selected_checkpoint_sha256"
        )
        object.__setattr__(
            self,
            "_canonical_payload",
            canonical_json_bytes(
                {
                    "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
                    "training": json.loads(self.training._canonical_payload),
                }
            ),
        )


def training_fingerprint(value: TrainingFingerprintInput) -> str:
    """Return the SHA-256 training fingerprint."""
    if type(value) is not TrainingFingerprintInput:
        raise FingerprintError("training fingerprint input has the wrong type")
    return hashlib.sha256(value._canonical_payload).hexdigest()


def inference_fingerprint(value: InferenceFingerprintInput) -> str:
    """Return the SHA-256 inference fingerprint."""
    if type(value) is not InferenceFingerprintInput:
        raise FingerprintError("inference fingerprint input has the wrong type")
    return hashlib.sha256(value._canonical_payload).hexdigest()


@dataclass(frozen=True)
class FingerprintPair:
    """Expected training and inference identities for one attempt."""

    training: str
    inference: str

    def __post_init__(self) -> None:
        _require_sha256(self.training, "training fingerprint")
        _require_sha256(self.inference, "inference fingerprint")


@dataclass(frozen=True)
class CompletionArtifact:
    """Immutable identity of one required artifact."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class CompletionRecord:
    """Validated versioned completion record."""

    schema_version: int
    training_fingerprint: str
    inference_fingerprint: str
    artifacts: tuple[CompletionArtifact, ...]


class PrepareOutcome(str, Enum):
    """Typed result of preparing a logical run for execution or resume."""

    REUSABLE = "reusable"
    ALLOCATED = "allocated"


@dataclass(frozen=True)
class PreparedAttempt:
    """A reusable completion or a newly allocated attempt directory."""

    outcome: PrepareOutcome
    attempt: Path
    completion: CompletionRecord | None = None


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=int(getattr(value, "st_dev", 0)),
            inode=int(getattr(value, "st_ino", 0)),
            size=int(value.st_size),
            mtime_ns=int(
                getattr(value, "st_mtime_ns", round(value.st_mtime * 1_000_000_000))
            ),
            ctime_ns=int(
                getattr(value, "st_ctime_ns", round(value.st_ctime * 1_000_000_000))
            ),
        )


_ATTEMPT_RE = re.compile(r"^attempt-([1-9][0-9]*)$")
_ARCHIVE_RE = re.compile(r"^attempt-([1-9][0-9]*)-")
_COMPLETION_NAME = "completion.json"
_COMPLETION_TEMP_NAME = "completion.json.tmp"
_ATTEMPT_LOCK_NAME = ".artifacts.lock"
_COMPLETION_FIELDS = {
    "schema_version",
    "training_fingerprint",
    "inference_fingerprint",
    "artifacts",
}
_ARTIFACT_FIELDS = {"path", "sha256", "size"}


def allocate_attempt(run_root: Path) -> Path:
    """Concurrency-safely allocate the next ``attempt-N`` directory."""
    root = _ensure_run_root(run_root)
    candidate_number = _largest_attempt_number(root) + 1
    while True:
        candidate = root / f"attempt-{candidate_number}"
        try:
            candidate.mkdir()
        except FileExistsError:
            candidate_number += 1
            continue
        except OSError as exc:
            raise ArtifactError(f"cannot allocate attempt directory {candidate}: {exc}") from exc
        _fsync_directory(root)
        return candidate


class AttemptSession:
    """Cross-process serialized artifact operations for one attempt."""

    def __init__(self, attempt: Path) -> None:
        self.attempt = _require_attempt_directory(attempt)
        self._attempt_descriptor: int | None = None
        self._lock_descriptor: int | None = None

    def __enter__(self) -> AttemptSession:
        if self._lock_descriptor is not None:
            raise ArtifactError("attempt session is already entered")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            attempt_descriptor = os.open(self.attempt, directory_flags)
            _validate_directory_path(
                self.attempt, attempt_descriptor, "attempt directory"
            )
            fcntl.flock(attempt_descriptor, fcntl.LOCK_EX)
            _validate_directory_path(
                self.attempt, attempt_descriptor, "attempt directory"
            )
            lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            lock_descriptor = os.open(
                _ATTEMPT_LOCK_NAME,
                lock_flags,
                0o600,
                dir_fd=attempt_descriptor,
            )
            if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise ArtifactPathError("attempt lock must be a regular file")
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            _validate_lock_path(attempt_descriptor, lock_descriptor)
            _validate_directory_path(
                self.attempt, attempt_descriptor, "attempt directory"
            )
        except OSError as exc:
            if "lock_descriptor" in locals():
                os.close(lock_descriptor)
            if "attempt_descriptor" in locals():
                os.close(attempt_descriptor)
            raise ArtifactPathError(
                f"cannot acquire path-safe attempt lock for {self.attempt}: {exc}"
            ) from exc
        except BaseException:
            if "lock_descriptor" in locals():
                os.close(lock_descriptor)
            if "attempt_descriptor" in locals():
                os.close(attempt_descriptor)
            raise
        self._attempt_descriptor = attempt_descriptor
        self._lock_descriptor = lock_descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        descriptor = self._lock_descriptor
        attempt_descriptor = self._attempt_descriptor
        self._lock_descriptor = None
        self._attempt_descriptor = None
        cleanup_error: BaseException | None = None

        def cleanup(operation: Any) -> None:
            nonlocal cleanup_error
            try:
                operation()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc

        if descriptor is not None:
            cleanup(lambda: fcntl.flock(descriptor, fcntl.LOCK_UN))
            cleanup(lambda: os.close(descriptor))
        if attempt_descriptor is not None:
            cleanup(lambda: fcntl.flock(attempt_descriptor, fcntl.LOCK_UN))
            cleanup(lambda: os.close(attempt_descriptor))
        body_exception = len(_args) >= 2 and _args[1] is not None
        if cleanup_error is not None and not body_exception:
            raise cleanup_error

    def write_artifact_atomic(self, relative_path: str, payload: bytes) -> Path:
        """Write an artifact while retaining this session's attempt lock."""
        self._require_entered()
        return _write_artifact_atomic_locked(self, relative_path, payload)

    def complete_attempt(
        self,
        fingerprints: FingerprintPair,
        required_artifacts: tuple[str, ...] | list[str],
    ) -> CompletionRecord:
        """Publish completion while retaining this session's attempt lock."""
        self._require_entered()
        return _complete_attempt_locked(self, fingerprints, required_artifacts)

    def validate_completion(
        self, expected: FingerprintPair | None = None
    ) -> CompletionRecord:
        """Validate completion while retaining this session's attempt lock."""
        self._require_entered()
        return _validate_completion_locked(self, expected)

    def _require_entered(self) -> None:
        if self._lock_descriptor is None or self._attempt_descriptor is None:
            raise ArtifactError("attempt session must be entered before use")

    @property
    def _attempt_fd(self) -> int:
        self._require_entered()
        assert self._attempt_descriptor is not None
        return self._attempt_descriptor


def _validate_lock_path(attempt_descriptor: int, lock_descriptor: int) -> None:
    try:
        opened = os.fstat(lock_descriptor)
        current = os.stat(
            _ATTEMPT_LOCK_NAME,
            dir_fd=attempt_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ArtifactError(
            "attempt lock path changed during acquisition; retry the operation"
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (int(opened.st_dev), int(opened.st_ino))
        != (int(current.st_dev), int(current.st_ino))
    ):
        raise ArtifactError(
            "attempt lock path changed during acquisition; retry the operation"
        )


def write_artifact_atomic(attempt: Path, relative_path: str, payload: bytes) -> Path:
    """Atomically write one artifact under the attempt lock."""
    with AttemptSession(attempt) as session:
        return session.write_artifact_atomic(relative_path, payload)


def _write_artifact_atomic_locked(
    session: AttemptSession, relative_path: str, payload: bytes
) -> Path:
    _refuse_completed_or_interrupted_locked(session, allow_interrupted=False)
    if type(payload) is not bytes:
        raise ArtifactError("atomic artifact payload must be bytes")
    normalized = _normalize_artifact_path(relative_path)
    relative = PurePosixPath(normalized)
    with _open_parent_descriptor(session, relative, create_parents=True) as parent:
        temporary_name = f".{relative.name}.tmp-{secrets.token_hex(12)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=parent.descriptor)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _refuse_completed_or_interrupted_locked(
                session, allow_interrupted=False
            )
            _validate_parent_descriptor(session, parent)
            try:
                target = os.stat(
                    relative.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                target = None
            if target is not None and stat.S_ISLNK(target.st_mode):
                raise ArtifactPathError(
                    f"artifact target must not be a symlink: {relative_path}"
                )
            os.replace(
                temporary_name,
                relative.name,
                src_dir_fd=parent.descriptor,
                dst_dir_fd=parent.descriptor,
            )
            _fsync_directory_descriptor(parent.descriptor, parent.path)
            _validate_parent_descriptor(session, parent)
        except BaseException:
            try:
                os.unlink(temporary_name, dir_fd=parent.descriptor)
            except FileNotFoundError:
                pass
            raise
        return parent.path / relative.name


def complete_attempt(
    attempt: Path,
    fingerprints: FingerprintPair,
    required_artifacts: tuple[str, ...] | list[str],
) -> CompletionRecord:
    """Hash artifacts and publish completion under the attempt lock."""
    with AttemptSession(attempt) as session:
        return session.complete_attempt(fingerprints, required_artifacts)


def _complete_attempt_locked(
    session: AttemptSession,
    fingerprints: FingerprintPair,
    required_artifacts: tuple[str, ...] | list[str],
) -> CompletionRecord:
    _require_fingerprint_pair(fingerprints)
    _refuse_completed_or_interrupted_locked(session, allow_interrupted=False)
    if type(required_artifacts) not in (tuple, list) or not required_artifacts:
        raise ArtifactPathError("required artifacts must be a non-empty tuple or list")
    normalized = [_normalize_artifact_path(item) for item in required_artifacts]
    if len(set(normalized)) != len(normalized):
        raise ArtifactPathError("required artifact paths must be unique")

    artifacts = tuple(
        _hash_required_artifact(session, relative_path)
        for relative_path in sorted(normalized)
    )
    record = CompletionRecord(
        schema_version=1,
        training_fingerprint=fingerprints.training,
        inference_fingerprint=fingerprints.inference,
        artifacts=artifacts,
    )
    _publish_completion(session, record)
    try:
        return _validate_completion_locked(session, fingerprints)
    except BaseException:
        try:
            os.unlink(_COMPLETION_NAME, dir_fd=session._attempt_fd)
        except FileNotFoundError:
            pass
        _fsync_directory_descriptor(session._attempt_fd, session.attempt)
        raise


def validate_completion(
    attempt: Path, expected: FingerprintPair | None = None
) -> CompletionRecord:
    """Strictly validate completion and artifacts under the attempt lock."""
    with AttemptSession(attempt) as session:
        return session.validate_completion(expected)


def _validate_completion_locked(
    session: AttemptSession, expected: FingerprintPair | None = None
) -> CompletionRecord:
    attempt_root = session.attempt
    if expected is not None:
        _require_fingerprint_pair(expected)
    completion_path = attempt_root / _COMPLETION_NAME
    if completion_path.is_symlink():
        raise CorruptCompletionError(
            "completion record must not be a symlink; use --rerun to archive the attempt"
        )
    try:
        raw = _read_stable_entry(
            session,
            session._attempt_fd,
            _COMPLETION_NAME,
            completion_path,
            parent=None,
        )
    except (OSError, ArtifactPathError, CorruptCompletionError) as exc:
        raise CorruptCompletionError(
            f"cannot validate completion record; use --rerun to archive the attempt: {exc}"
        ) from exc
    record = _parse_completion(raw)
    if expected is not None and (
        record.training_fingerprint != expected.training
        or record.inference_fingerprint != expected.inference
    ):
        raise CompletionMismatchError(
            "completed attempt fingerprints do not match; pass --rerun to archive it "
            "and allocate a new attempt"
        )
    _validate_record_artifacts(session, record)
    return record


def prepare_attempt(
    run_root: Path, expected: FingerprintPair, *, rerun: bool = False
) -> PreparedAttempt:
    """Reuse a valid completion, restart an incomplete run, or force a rerun."""
    _require_fingerprint_pair(expected)
    root = _ensure_run_root(run_root)
    attempts = _active_attempts(root)
    if rerun:
        for attempt in attempts:
            with AttemptSession(attempt) as session:
                if _entry_exists(session._attempt_fd, _COMPLETION_NAME):
                    _archive_attempt_locked(root, session)
        return PreparedAttempt(PrepareOutcome.ALLOCATED, allocate_attempt(root))
    if not attempts:
        return PreparedAttempt(PrepareOutcome.ALLOCATED, allocate_attempt(root))

    latest = attempts[-1]
    if not _path_entry_exists(latest / _COMPLETION_NAME):
        return PreparedAttempt(PrepareOutcome.ALLOCATED, allocate_attempt(root))
    completion = validate_completion(latest, expected)
    return PreparedAttempt(PrepareOutcome.REUSABLE, latest, completion)


def _ensure_run_root(run_root: Path) -> Path:
    if not isinstance(run_root, Path):
        raise ArtifactPathError("run root must be a pathlib.Path")
    try:
        if run_root.is_symlink():
            raise ArtifactPathError(f"run root must not be a symlink: {run_root}")
        run_root.mkdir(parents=True, exist_ok=True)
        if not run_root.is_dir() or run_root.is_symlink():
            raise ArtifactPathError(f"run root must be a real directory: {run_root}")
        return run_root.resolve()
    except OSError as exc:
        raise ArtifactPathError(f"cannot prepare run root {run_root}: {exc}") from exc


def _require_attempt_directory(attempt: Path) -> Path:
    if not isinstance(attempt, Path):
        raise ArtifactPathError("attempt must be a pathlib.Path")
    try:
        if attempt.is_symlink() or not attempt.is_dir():
            raise ArtifactPathError(f"attempt must be a real directory: {attempt}")
        return attempt.resolve()
    except OSError as exc:
        raise ArtifactPathError(f"cannot inspect attempt directory {attempt}: {exc}") from exc


def _require_fingerprint_pair(value: object) -> FingerprintPair:
    if type(value) is not FingerprintPair:
        raise FingerprintError("expected fingerprints must be a FingerprintPair")
    return value


def _normalize_artifact_path(value: object) -> str:
    try:
        path_text = _require_normalized_relative_path(value, "artifact path")
    except FingerprintError as exc:
        raise ArtifactPathError(str(exc)) from exc
    if PurePosixPath(path_text).name in {_COMPLETION_NAME, _COMPLETION_TEMP_NAME}:
        raise ArtifactPathError("required artifacts must not name completion records")
    if path_text == _ATTEMPT_LOCK_NAME:
        raise ArtifactPathError("required artifact path is reserved for the attempt lock")
    return path_text


@dataclass(frozen=True)
class _OpenParent:
    descriptor: int
    path: Path
    parts: tuple[str, ...]


@contextmanager
def _open_parent_descriptor(
    session: AttemptSession,
    relative: PurePosixPath,
    *,
    create_parents: bool,
) -> Any:
    descriptor = os.dup(session._attempt_fd)
    path = session.attempt
    parts: list[str] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for part in relative.parts[:-1]:
            created = False
            if create_parents:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
            try:
                child = os.open(part, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise ArtifactPathError(
                    "artifact parent must be an existing no-follow directory, not a "
                    f"symlink: {path / part}"
                ) from exc
            if created:
                try:
                    _fsync_directory_descriptor(child, path / part)
                    _fsync_directory_descriptor(descriptor, path)
                except BaseException:
                    os.close(child)
                    raise
            os.close(descriptor)
            descriptor = child
            path = path / part
            parts.append(part)
        parent = _OpenParent(descriptor, path, tuple(parts))
        _validate_parent_descriptor(session, parent)
        yield parent
    finally:
        os.close(descriptor)


def _validate_parent_descriptor(
    session: AttemptSession, parent: _OpenParent
) -> None:
    _validate_directory_path(
        session.attempt, session._attempt_fd, "attempt directory"
    )
    descriptor = os.dup(session._attempt_fd)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for part in parent.parts:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        expected = _directory_identity(os.fstat(parent.descriptor))
        current = _directory_identity(os.fstat(descriptor))
    except OSError as exc:
        raise CorruptCompletionError(
            f"artifact parent changed or was replaced: {parent.path}"
        ) from exc
    finally:
        os.close(descriptor)
    if expected != current:
        raise CorruptCompletionError(
            f"artifact parent changed or was replaced: {parent.path}"
        )


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise ArtifactPathError("expected a real directory")
    return int(value.st_dev), int(value.st_ino)


def _validate_directory_path(path: Path, descriptor: int, label: str) -> None:
    try:
        current = path.lstat()
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise CorruptCompletionError(f"{label} changed or was replaced: {path}") from exc
    if stat.S_ISLNK(current.st_mode) or (
        _directory_identity(current) != _directory_identity(opened)
    ):
        raise CorruptCompletionError(f"{label} changed or was replaced: {path}")


def _hash_required_artifact(
    session: AttemptSession, relative_path: str
) -> CompletionArtifact:
    normalized = _normalize_artifact_path(relative_path)
    relative = PurePosixPath(normalized)
    with _open_parent_descriptor(
        session, relative, create_parents=False
    ) as parent:
        path = parent.path / relative.name
        digest, size = _hash_stable_entry(
            session,
            parent,
            relative.name,
            path,
            invoke_artifact_hook=True,
        )
    return CompletionArtifact(path=relative_path, sha256=digest, size=size)


def _hash_stable_entry(
    session: AttemptSession,
    parent: _OpenParent,
    name: str,
    path: Path,
    *,
    invoke_artifact_hook: bool,
) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    except OSError as exc:
        if path.is_symlink():
            raise ArtifactPathError(f"required artifact must not be a symlink: {path}") from exc
        raise CorruptCompletionError(f"required artifact is missing or unreadable: {path}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = _FileIdentity.from_stat(os.fstat(handle.fileno()))
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ArtifactPathError(f"required artifact must be a regular file: {path}")
        digest = _hash_open_file(handle)
        if invoke_artifact_hook:
            _after_artifact_hash(path)
        _ensure_file_identity_at(parent.descriptor, name, path, handle, before)
        _validate_parent_descriptor(session, parent)
    return digest, before.size


def _hash_open_file(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _after_artifact_hash(_path: Path) -> None:
    """Race-test hook called after hashing and before path identity validation."""


def _ensure_file_identity_at(
    parent_descriptor: int,
    name: str,
    path: Path,
    handle: BinaryIO,
    before: _FileIdentity,
) -> None:
    try:
        after = _FileIdentity.from_stat(os.fstat(handle.fileno()))
        current_stat = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        current = _FileIdentity.from_stat(current_stat)
    except OSError as exc:
        raise CorruptCompletionError(
            f"required artifact changed or was replaced during validation: {path}"
        ) from exc
    if stat.S_ISLNK(current_stat.st_mode) or before != after or before != current:
        raise CorruptCompletionError(
            f"required artifact changed or was replaced during validation: {path}"
        )


def _read_stable_entry(
    session: AttemptSession,
    parent_descriptor: int,
    name: str,
    path: Path,
    *,
    parent: _OpenParent | None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    with os.fdopen(descriptor, "rb") as handle:
        before = _FileIdentity.from_stat(os.fstat(handle.fileno()))
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ArtifactPathError(f"record must be a regular file: {path}")
        payload = handle.read()
        _ensure_file_identity_at(parent_descriptor, name, path, handle, before)
        if parent is None:
            _validate_directory_path(
                session.attempt, session._attempt_fd, "completion parent"
            )
        else:
            _validate_parent_descriptor(session, parent)
    return payload


def _validate_record_artifacts(
    session: AttemptSession, record: CompletionRecord
) -> None:
    for artifact in record.artifacts:
        try:
            observed = _hash_required_artifact(session, artifact.path)
        except (ArtifactPathError, CorruptCompletionError) as exc:
            raise CorruptCompletionError(
                f"completion artifact cannot be validated; use --rerun: {exc}"
            ) from exc
        if observed.sha256 != artifact.sha256 or observed.size != artifact.size:
            raise CorruptCompletionError(
                f"completion artifact hash or size mismatch for {artifact.path}; "
                "use --rerun to archive the attempt"
            )


def _record_json_value(record: CompletionRecord) -> dict[str, Any]:
    return {
        "artifacts": [
            {"path": item.path, "sha256": item.sha256, "size": item.size}
            for item in record.artifacts
        ],
        "inference_fingerprint": record.inference_fingerprint,
        "schema_version": record.schema_version,
        "training_fingerprint": record.training_fingerprint,
    }


def _publish_completion(session: AttemptSession, record: CompletionRecord) -> None:
    _refuse_completed_or_interrupted_locked(session, allow_interrupted=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(
            _COMPLETION_TEMP_NAME,
            flags,
            0o644,
            dir_fd=session._attempt_fd,
        )
    except FileExistsError as exc:
        raise IncompleteAttemptError(
            "interrupted completion temp exists; preserve this attempt and allocate a new one"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(_record_json_value(record)))
            handle.flush()
            os.fsync(handle.fileno())
        _before_completion_publish(session.attempt)
        _validate_directory_path(
            session.attempt, session._attempt_fd, "completion parent"
        )
        _at_completion_publish(session.attempt)
        _validate_directory_path(
            session.attempt, session._attempt_fd, "completion parent"
        )
        try:
            os.link(
                _COMPLETION_TEMP_NAME,
                _COMPLETION_NAME,
                src_dir_fd=session._attempt_fd,
                dst_dir_fd=session._attempt_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise CompletedAttemptError(
                "completed attempt cannot be overwritten"
            ) from exc
        os.unlink(_COMPLETION_TEMP_NAME, dir_fd=session._attempt_fd)
        _fsync_directory_descriptor(session._attempt_fd, session.attempt)
        try:
            _validate_directory_path(
                session.attempt, session._attempt_fd, "completion parent"
            )
        except BaseException:
            os.unlink(_COMPLETION_NAME, dir_fd=session._attempt_fd)
            _fsync_directory_descriptor(session._attempt_fd, session.attempt)
            raise
    except BaseException:
        try:
            os.unlink(_COMPLETION_TEMP_NAME, dir_fd=session._attempt_fd)
        except FileNotFoundError:
            pass
        raise


def _before_completion_publish(_attempt: Path) -> None:
    """Race-test hook before completion publication."""


def _at_completion_publish(_attempt: Path) -> None:
    """Race-test hook at the atomic no-clobber publication point."""


def _parse_completion(payload: bytes) -> CompletionRecord:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
        _validate_json_native(value, path="$", error_type=CorruptCompletionError)
        if type(value) is not dict or set(value) != _COMPLETION_FIELDS:
            raise ValueError("completion record has missing or unknown fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("completion schema_version must be integer 1")
        training = _completion_sha(value["training_fingerprint"], "training_fingerprint")
        inference = _completion_sha(
            value["inference_fingerprint"], "inference_fingerprint"
        )
        artifact_values = value["artifacts"]
        if type(artifact_values) is not list or not artifact_values:
            raise ValueError("completion artifacts must be a non-empty list")
        artifacts: list[CompletionArtifact] = []
        seen: set[str] = set()
        for item in artifact_values:
            if type(item) is not dict or set(item) != _ARTIFACT_FIELDS:
                raise ValueError("completion artifact has missing or unknown fields")
            path = _normalize_artifact_path(item["path"])
            if path in seen:
                raise ValueError("completion artifact paths must be unique")
            seen.add(path)
            digest = _completion_sha(item["sha256"], f"artifact {path} sha256")
            size = item["size"]
            if type(size) is not int or size < 0:
                raise ValueError(f"artifact {path} size must be a non-negative integer")
            artifacts.append(CompletionArtifact(path, digest, size))
    except (UnicodeError, json.JSONDecodeError, ValueError, ArtifactPathError, CorruptCompletionError) as exc:
        raise CorruptCompletionError(
            f"completion record is malformed or violates its closed schema; use --rerun: {exc}"
        ) from exc
    return CompletionRecord(1, training, inference, tuple(artifacts))


def _completion_sha(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _refuse_completed_or_interrupted_locked(
    session: AttemptSession, *, allow_interrupted: bool
) -> None:
    _validate_directory_path(
        session.attempt, session._attempt_fd, "attempt directory"
    )
    if _entry_exists(session._attempt_fd, _COMPLETION_NAME):
        raise CompletedAttemptError("completed attempt cannot be overwritten")
    if not allow_interrupted and _entry_exists(
        session._attempt_fd, _COMPLETION_TEMP_NAME
    ):
        raise IncompleteAttemptError(
            "interrupted completion temp exists; preserve this attempt and allocate a new one"
        )


def _entry_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _active_attempts(root: Path) -> list[Path]:
    attempts: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _ATTEMPT_RE.fullmatch(child.name)
        if match is not None:
            if child.is_symlink() or not child.is_dir():
                raise ArtifactPathError(f"attempt entry must be a real directory: {child}")
            attempts.append((int(match.group(1)), child))
    return [path for _, path in sorted(attempts)]


def _largest_attempt_number(root: Path) -> int:
    numbers = [
        int(match.group(1))
        for child in root.iterdir()
        if (match := _ATTEMPT_RE.fullmatch(child.name)) is not None
    ]
    archive = root / "archive"
    if archive.exists():
        if archive.is_symlink() or not archive.is_dir():
            raise ArtifactPathError("attempt archive must be a real directory")
        numbers.extend(
            int(match.group(1))
            for child in archive.iterdir()
            if (match := _ARCHIVE_RE.match(child.name)) is not None
        )
    return max(numbers, default=0)


def _archive_attempt_locked(root: Path, session: AttemptSession) -> Path:
    session._require_entered()
    attempt = session.attempt
    archive = root / "archive"
    try:
        archive.mkdir(exist_ok=True)
    except OSError as exc:
        raise ArtifactError(f"cannot create attempt archive: {exc}") from exc
    if archive.is_symlink() or not archive.is_dir():
        raise ArtifactPathError("attempt archive must be a real directory")

    suffix = 1
    while True:
        disambiguator = "" if suffix == 1 else f"-{suffix}"
        destination = archive / f"{attempt.name}-archived{disambiguator}"
        lock = archive / f".{destination.name}.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        try:
            if _path_entry_exists(destination):
                suffix += 1
                continue
            os.rename(attempt, destination)
            _fsync_directory(archive)
            _fsync_directory(root)
            return destination
        finally:
            try:
                lock.rmdir()
            except FileNotFoundError:
                pass


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactError(
            f"cannot open directory for durability fsync {path}: {exc}"
        ) from exc
    try:
        _fsync_directory_descriptor(descriptor, path)
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int, path: Path) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno in unsupported:
            return
        raise ArtifactError(
            f"cannot fsync directory for durability {path}: {exc}"
        ) from exc

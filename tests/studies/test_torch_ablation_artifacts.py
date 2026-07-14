from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.studies.ablation.artifacts import (
    AttemptSession,
    ArtifactError,
    ArtifactPathError,
    CompletedAttemptError,
    CompletionMismatchError,
    CorruptCompletionError,
    FingerprintError,
    FingerprintPair,
    GitIdentity,
    InferenceFingerprintInput,
    PrepareOutcome,
    TrainingFingerprintInput,
    UntrackedSource,
    allocate_attempt,
    canonical_json_bytes,
    complete_attempt,
    inference_fingerprint,
    prepare_attempt,
    training_fingerprint,
    validate_completion,
    write_artifact_atomic,
)


def _hold_then_write_artifact(
    attempt: Path,
    entered: Any,
    release: Any,
) -> None:
    with AttemptSession(attempt) as session:
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test did not release held attempt session")
        session.write_artifact_atomic("metrics.json", b"serialized")


def _hold_attempt_session(
    attempt: Path,
    entered: Any,
    release: Any,
) -> None:
    with AttemptSession(attempt):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test did not release held attempt session")


def _complete_in_process(
    attempt: Path,
    training: str,
    inference: str,
    started: Any,
) -> None:
    started.set()
    complete_attempt(
        attempt,
        FingerprintPair(training=training, inference=inference),
        ("metrics.json",),
    )


def _complete_existing_artifact_in_process(
    attempt: Path,
    training: str,
    inference: str,
) -> None:
    complete_attempt(
        attempt,
        FingerprintPair(training=training, inference=inference),
        ("metrics.json",),
    )


def _rerun_in_process(
    run_root: Path,
    training: str,
    inference: str,
    started: Any,
) -> None:
    started.set()
    prepare_attempt(
        run_root,
        FingerprintPair(training=training, inference=inference),
        rerun=True,
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _training_input() -> TrainingFingerprintInput:
    return TrainingFingerprintInput(
        schema_version=1,
        manifest_sha256=_digest("manifest"),
        logical_run_id="cnn/seed-17",
        resolved_configs={
            "data": {"N": 64, "shuffle": False},
            "model": {"architecture": "cnn"},
            "training": {"learning_rate": 0.001},
        },
        seed=17,
        git=GitIdentity(commit="a" * 40, clean=True),
        environment_digest=_digest("environment"),
        content_sha256s={
            "dataset.train": _digest("train"),
            "dataset.test": _digest("test"),
            "provenance": _digest("provenance"),
            "probe": _digest("probe"),
        },
    )


def test_canonical_json_bytes_are_compact_utf8_and_order_independent() -> None:
    left = {"z": [3, {"beta": "β", "alpha": None}], "a": True}
    right = {"a": True, "z": [3, {"alpha": None, "beta": "β"}]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == (
        '{"a":true,"z":[3,{"alpha":null,"beta":"β"}]}'.encode("utf-8")
    )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {1: "non-string key"},
        Path("custom/path"),
        ("tuple",),
        object(),
    ],
)
def test_canonical_json_rejects_non_native_or_non_finite_values(value: Any) -> None:
    with pytest.raises(FingerprintError, match="JSON-native|finite|string keys"):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, schema_version=2),
        lambda value: replace(value, manifest_sha256=_digest("other manifest")),
        lambda value: replace(value, logical_run_id="fno/seed-17"),
        lambda value: replace(
            value,
            resolved_configs={**value.resolved_configs, "data": {"N": 128}},
        ),
        lambda value: replace(value, seed=18),
        lambda value: replace(value, git=replace(value.git, commit="b" * 40)),
        lambda value: replace(
            value,
            git=GitIdentity(
                commit=value.git.commit,
                clean=False,
                tracked_patch_sha256=_digest("patch"),
            ),
        ),
        lambda value: replace(
            value, environment_digest=_digest("other environment")
        ),
        lambda value: replace(
            value,
            content_sha256s={
                **value.content_sha256s,
                "probe": _digest("other probe"),
            },
        ),
    ],
)
def test_training_fingerprint_is_sensitive_to_each_governing_field(change: Any) -> None:
    original = _training_input()

    assert training_fingerprint(change(original)) != training_fingerprint(original)


def test_training_fingerprint_covers_every_resolved_config_and_content_digest() -> None:
    original = _training_input()
    expected = training_fingerprint(original)

    for namespace, config in original.resolved_configs.items():
        changed_configs = dict(original.resolved_configs)
        changed_configs[namespace] = {**config, "sensitivity_marker": namespace}
        assert training_fingerprint(
            replace(original, resolved_configs=changed_configs)
        ) != expected
    for name in original.content_sha256s:
        changed_hashes = dict(original.content_sha256s)
        changed_hashes[name] = _digest(f"changed {name}")
        assert training_fingerprint(
            replace(original, content_sha256s=changed_hashes)
        ) != expected


def test_inference_fingerprint_adds_selected_checkpoint_identity() -> None:
    training = _training_input()
    first = InferenceFingerprintInput(training, _digest("checkpoint one"))
    second = replace(first, selected_checkpoint_sha256=_digest("checkpoint two"))

    assert inference_fingerprint(first) != inference_fingerprint(second)
    assert inference_fingerprint(first) != training_fingerprint(training)


def test_inference_fingerprint_retains_all_training_identity() -> None:
    original = _training_input()
    changed = replace(original, seed=original.seed + 1)

    assert inference_fingerprint(
        InferenceFingerprintInput(original, _digest("checkpoint"))
    ) != inference_fingerprint(
        InferenceFingerprintInput(changed, _digest("checkpoint"))
    )


def test_training_fingerprint_typed_integer_fields_reject_bool() -> None:
    with pytest.raises(FingerprintError, match="schema_version"):
        replace(_training_input(), schema_version=True)
    with pytest.raises(FingerprintError, match="seed"):
        replace(_training_input(), seed=False)


def test_git_clean_field_rejects_integer_and_evidence_contradictions() -> None:
    with pytest.raises(FingerprintError, match="clean"):
        GitIdentity(commit="a" * 40, clean=1)  # type: ignore[arg-type]
    with pytest.raises(FingerprintError, match="clean.*evidence|evidence.*clean"):
        GitIdentity(
            commit="a" * 40,
            clean=True,
            tracked_patch_sha256=_digest("patch"),
        )
    with pytest.raises(FingerprintError, match="dirty.*tracked patch"):
        GitIdentity(commit="a" * 40, clean=False)


def test_clean_and_dirty_git_evidence_controls_claim_grade() -> None:
    clean = GitIdentity(commit="a" * 40, clean=True)
    dirty = GitIdentity(
        commit="a" * 40,
        clean=False,
        tracked_patch_sha256=_digest("patch"),
        untracked_sources=(
            UntrackedSource("scripts/local_probe.py", _digest("local probe")),
            UntrackedSource("configs/smoke.json", _digest("smoke config")),
        ),
    )

    assert clean.claim_grade is True
    assert dirty.claim_grade is False
    assert _training_input().claim_grade is True
    assert replace(_training_input(), git=dirty).claim_grade is False
    assert training_fingerprint(replace(_training_input(), git=dirty)) != (
        training_fingerprint(_training_input())
    )


def test_dirty_fingerprint_covers_patch_and_every_untracked_source_identity() -> None:
    dirty = GitIdentity(
        commit="a" * 40,
        clean=False,
        tracked_patch_sha256=_digest("patch"),
        untracked_sources=(
            UntrackedSource("scripts/a.py", _digest("source a")),
            UntrackedSource("scripts/b.py", _digest("source b")),
        ),
    )
    original = replace(_training_input(), git=dirty)
    expected = training_fingerprint(original)

    changed_patch = replace(
        original,
        git=replace(dirty, tracked_patch_sha256=_digest("changed patch")),
    )
    assert training_fingerprint(changed_patch) != expected
    for index, source in enumerate(dirty.untracked_sources):
        changed_sources = list(dirty.untracked_sources)
        changed_sources[index] = replace(source, sha256=_digest(f"changed {source.path}"))
        changed = replace(original, git=replace(dirty, untracked_sources=tuple(changed_sources)))
        assert training_fingerprint(changed) != expected


def test_training_fingerprint_inputs_are_recursively_immutable_snapshots() -> None:
    value = _training_input()
    expected = training_fingerprint(value)

    with pytest.raises(TypeError):
        value.resolved_configs["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        value.resolved_configs["data"]["N"] = 128  # type: ignore[index]
    with pytest.raises(TypeError):
        value.content_sha256s["probe"] = _digest("changed")  # type: ignore[index]
    assert training_fingerprint(value) == expected


@pytest.mark.parametrize(
    "resolved_configs",
    [
        {"data": {"shape": (64, 64)}},
        {"data": [{"shape": (64, 64)}]},
        {"data": {"nested": [{"modes": (8, 8)}]}},
    ],
)
def test_training_fingerprint_rejects_raw_nested_tuples(
    resolved_configs: dict[str, Any],
) -> None:
    with pytest.raises(FingerprintError, match="JSON-native|tuple"):
        replace(_training_input(), resolved_configs=resolved_configs)


def test_training_fingerprint_can_reuse_genuine_frozen_list_snapshots() -> None:
    original = replace(
        _training_input(),
        resolved_configs={"data": {"shape": [64, 64], "axes": [[0, 1]]}},
    )

    reused = replace(original, seed=original.seed + 1)

    assert training_fingerprint(reused) != training_fingerprint(original)


def test_dirty_evidence_rejects_duplicate_or_unsafe_untracked_paths() -> None:
    source = UntrackedSource("scripts/local.py", _digest("local"))
    with pytest.raises(FingerprintError, match="duplicate"):
        GitIdentity(
            commit="a" * 40,
            clean=False,
            tracked_patch_sha256=_digest("patch"),
            untracked_sources=(source, source),
        )
    with pytest.raises(FingerprintError, match="relative|normalized"):
        UntrackedSource("../outside.py", _digest("outside"))


def _fingerprints(label: str = "expected") -> FingerprintPair:
    return FingerprintPair(
        training=_digest(f"{label} training"),
        inference=_digest(f"{label} inference"),
    )


def _completed_attempt(
    run_root: Path, fingerprints: FingerprintPair | None = None
) -> Path:
    attempt = allocate_attempt(run_root)
    write_artifact_atomic(attempt, "metrics.json", b'{"loss":0.25}')
    write_artifact_atomic(attempt, "checkpoints/best.ckpt", b"checkpoint")
    complete_attempt(
        attempt,
        fingerprints or _fingerprints(),
        ("metrics.json", "checkpoints/best.ckpt"),
    )
    return attempt


def test_atomic_completion_records_artifacts_and_validated_resume(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt = _completed_attempt(run_root)

    completion_path = attempt / "completion.json"
    payload = json.loads(completion_path.read_bytes())
    assert set(payload) == {
        "schema_version",
        "training_fingerprint",
        "inference_fingerprint",
        "artifacts",
    }
    assert payload["schema_version"] == 1
    assert payload["artifacts"] == [
        {
            "path": "checkpoints/best.ckpt",
            "sha256": _digest("checkpoint"),
            "size": 10,
        },
        {
            "path": "metrics.json",
            "sha256": hashlib.sha256(b'{"loss":0.25}').hexdigest(),
            "size": 13,
        },
    ]
    assert not (attempt / "completion.json.tmp").exists()

    prepared = prepare_attempt(run_root, _fingerprints())
    assert prepared.outcome is PrepareOutcome.REUSABLE
    assert prepared.attempt == attempt
    assert prepared.completion is not None
    assert tuple(item.path for item in prepared.completion.artifacts) == (
        "checkpoints/best.ckpt",
        "metrics.json",
    )


def test_attempt_session_serializes_writer_and_completion_across_processes(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    attempt = allocate_attempt(tmp_path / "run")
    expected = _fingerprints()
    entered = context.Event()
    release = context.Event()
    completion_started = context.Event()
    writer = context.Process(
        target=_hold_then_write_artifact,
        args=(attempt, entered, release),
    )
    completer = context.Process(
        target=_complete_in_process,
        args=(attempt, expected.training, expected.inference, completion_started),
    )
    writer.start()
    assert entered.wait(10)
    lock_descriptor = os.open(attempt / ".artifacts.lock", os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(lock_descriptor)
    completer.start()
    assert completion_started.wait(10)

    completer.join(timeout=0.25)
    assert completer.is_alive()
    assert not (attempt / "completion.json").exists()
    release.set()
    writer.join(timeout=10)
    completer.join(timeout=10)

    assert writer.exitcode == 0
    assert completer.exitcode == 0
    assert (attempt / "metrics.json").read_bytes() == b"serialized"
    record = validate_completion(attempt, expected)
    assert record.artifacts[0].sha256 == _digest("serialized")


def test_attempt_session_releases_lock_on_exception(tmp_path: Path) -> None:
    attempt = allocate_attempt(tmp_path / "run")

    with pytest.raises(RuntimeError, match="injected"):
        with AttemptSession(attempt):
            raise RuntimeError("injected")

    with AttemptSession(attempt) as session:
        session.write_artifact_atomic("after-error.bin", b"released")
    assert (attempt / "after-error.bin").read_bytes() == b"released"


def test_attempt_session_close_failure_still_closes_attempt_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    session = AttemptSession(allocate_attempt(tmp_path / "run"))
    session.__enter__()
    lock_descriptor = session._lock_descriptor
    attempt_descriptor = session._attempt_descriptor
    assert lock_descriptor is not None
    assert attempt_descriptor is not None
    real_close = artifacts.os.close
    close_attempts: list[int] = []

    def fail_lock_close(descriptor: int) -> None:
        close_attempts.append(descriptor)
        if descriptor == lock_descriptor:
            raise OSError(errno.EIO, "injected lock close failure")
        real_close(descriptor)

    monkeypatch.setattr(artifacts.os, "close", fail_lock_close)
    try:
        with pytest.raises(OSError, match="injected lock close failure"):
            session.__exit__(None, None, None)
    finally:
        monkeypatch.setattr(artifacts.os, "close", real_close)
        real_close(lock_descriptor)

    assert attempt_descriptor in close_attempts
    with pytest.raises(OSError):
        os.fstat(attempt_descriptor)


def test_replacing_lock_path_cannot_create_concurrent_attempt_sessions(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    attempt = allocate_attempt(tmp_path / "run")
    first_entered = context.Event()
    first_release = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    first = context.Process(
        target=_hold_attempt_session,
        args=(attempt, first_entered, first_release),
    )
    second = context.Process(
        target=_hold_attempt_session,
        args=(attempt, second_entered, second_release),
    )
    first.start()
    assert first_entered.wait(10)
    lock_path = attempt / ".artifacts.lock"
    displaced_lock = attempt / ".displaced-artifacts.lock"
    os.rename(lock_path, displaced_lock)
    lock_path.write_bytes(b"replacement lock inode")
    second.start()

    second.join(timeout=0.25)
    second_was_blocked = second.is_alive() and not second_entered.is_set()
    first_release.set()
    assert second_entered.wait(10)
    second_release.set()
    first.join(timeout=10)
    second.join(timeout=10)
    if first.is_alive():
        first.terminate()
        first.join(timeout=10)
    if second.is_alive():
        second.terminate()
        second.join(timeout=10)

    assert second_was_blocked
    assert first.exitcode == 0
    assert second.exitcode == 0


def test_attempt_lock_record_cannot_be_required_artifact(tmp_path: Path) -> None:
    attempt = allocate_attempt(tmp_path / "run")
    with AttemptSession(attempt):
        pass

    with pytest.raises(ArtifactPathError, match="reserved|lock"):
        complete_attempt(attempt, _fingerprints(), (".artifacts.lock",))


def test_interrupted_completion_temp_is_incomplete_and_preserved(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    first = allocate_attempt(run_root)
    interrupted = b'{"interrupted":true}'
    (first / "completion.json.tmp").write_bytes(interrupted)

    prepared = prepare_attempt(run_root, _fingerprints())

    assert prepared.outcome is PrepareOutcome.ALLOCATED
    assert prepared.attempt.name == "attempt-2"
    assert (first / "completion.json.tmp").read_bytes() == interrupted
    assert not (first / "completion.json").exists()


def test_incomplete_attempt_restarts_in_new_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    first = allocate_attempt(run_root)
    write_artifact_atomic(first, "partial.log", b"still running")

    prepared = prepare_attempt(run_root, _fingerprints())

    assert prepared.outcome is PrepareOutcome.ALLOCATED
    assert prepared.attempt.name == "attempt-2"
    assert (first / "partial.log").read_bytes() == b"still running"


def test_completed_fingerprint_mismatch_refuses_with_rerun_remedy(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt = _completed_attempt(run_root)
    original = (attempt / "completion.json").read_bytes()

    with pytest.raises(CompletionMismatchError, match="--rerun"):
        prepare_attempt(run_root, _fingerprints("different"))

    assert (attempt / "completion.json").read_bytes() == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: b"not-json",
        lambda payload: canonical_json_bytes({**payload, "unknown": True}),
        lambda payload: canonical_json_bytes(
            {**payload, "artifacts": [{**payload["artifacts"][0], "unknown": 1}]}
        ),
        lambda payload: canonical_json_bytes({**payload, "schema_version": True}),
        lambda payload: canonical_json_bytes(
            {
                **payload,
                "artifacts": [
                    {**payload["artifacts"][0], "size": True},
                    *payload["artifacts"][1:],
                ],
            }
        ),
    ],
)
def test_malformed_or_unknown_completion_refuses_resume(
    tmp_path: Path, mutate: Any
) -> None:
    run_root = tmp_path / "run"
    attempt = _completed_attempt(run_root)
    completion = attempt / "completion.json"
    payload = json.loads(completion.read_bytes())
    completion.write_bytes(mutate(payload))

    with pytest.raises(CorruptCompletionError, match="--rerun"):
        prepare_attempt(run_root, _fingerprints())


@pytest.mark.parametrize("damage", ["missing", "changed", "wrong_size"])
def test_missing_changed_or_inconsistent_artifact_refuses_resume(
    tmp_path: Path, damage: str
) -> None:
    run_root = tmp_path / "run"
    attempt = _completed_attempt(run_root)
    metrics = attempt / "metrics.json"
    if damage == "missing":
        metrics.unlink()
    elif damage == "changed":
        metrics.write_bytes(b'{"loss":9.99}')
    else:
        metrics.write_bytes(b"short")

    with pytest.raises(CorruptCompletionError, match="artifact.*--rerun|--rerun.*artifact"):
        prepare_attempt(run_root, _fingerprints())


def test_rerun_archives_completed_attempt_and_never_overwrites_archive(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    first = _completed_attempt(run_root)
    first_bytes = (first / "completion.json").read_bytes()

    rerun = prepare_attempt(run_root, _fingerprints("different"), rerun=True)

    assert rerun.outcome is PrepareOutcome.ALLOCATED
    assert rerun.attempt.name == "attempt-2"
    archives = sorted((run_root / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "completion.json").read_bytes() == first_bytes
    write_artifact_atomic(rerun.attempt, "metrics.json", b"new")
    complete_attempt(rerun.attempt, _fingerprints("different"), ("metrics.json",))

    next_rerun = prepare_attempt(
        run_root, _fingerprints("third"), rerun=True
    )
    assert next_rerun.attempt.name == "attempt-3"
    archives = sorted((run_root / "archive").iterdir())
    assert len(archives) == 2
    assert any((archive / "completion.json").read_bytes() == first_bytes for archive in archives)


def test_rerun_waits_for_completion_publication_and_archives_valid_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    context = multiprocessing.get_context("fork")
    run_root = tmp_path / "run"
    attempt = allocate_attempt(run_root)
    write_artifact_atomic(attempt, "metrics.json", b"serialized completion")
    expected = _fingerprints()
    post_validation = context.Event()
    release_completion = context.Event()
    rerun_started = context.Event()
    hash_count = 0

    def pause_during_post_validation(_path: Path) -> None:
        nonlocal hash_count
        hash_count += 1
        if hash_count == 2:
            post_validation.set()
            if not release_completion.wait(10):
                raise RuntimeError("test did not release completion validation")

    monkeypatch.setattr(
        artifacts, "_after_artifact_hash", pause_during_post_validation
    )
    completer = context.Process(
        target=_complete_existing_artifact_in_process,
        args=(attempt, expected.training, expected.inference),
    )
    rerunner = context.Process(
        target=_rerun_in_process,
        args=(run_root, expected.training, expected.inference, rerun_started),
    )
    completer.start()
    assert post_validation.wait(10)
    completion_bytes = (attempt / "completion.json").read_bytes()
    rerunner.start()
    assert rerun_started.wait(10)

    rerunner.join(timeout=0.25)
    rerun_waited = rerunner.is_alive()
    archive_before_release = run_root / "archive"
    archive_was_absent = not archive_before_release.exists()
    release_completion.set()
    completer.join(timeout=10)
    rerunner.join(timeout=10)
    if completer.is_alive():
        completer.terminate()
        completer.join(timeout=10)
    if rerunner.is_alive():
        rerunner.terminate()
        rerunner.join(timeout=10)

    assert rerun_waited
    assert archive_was_absent
    assert completer.exitcode == 0
    assert rerunner.exitcode == 0
    archives = list((run_root / "archive").iterdir())
    assert len(archives) == 1
    archived = archives[0]
    assert (archived / "completion.json").read_bytes() == completion_bytes
    record = validate_completion(archived, expected)
    assert record.artifacts[0].sha256 == _digest("serialized completion")
    assert (run_root / "attempt-2").is_dir()


def test_rerun_archives_malformed_completion_without_overwriting_it(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt = _completed_attempt(run_root)
    malformed = b"malformed-completion"
    (attempt / "completion.json").write_bytes(malformed)

    prepared = prepare_attempt(run_root, _fingerprints(), rerun=True)

    assert prepared.outcome is PrepareOutcome.ALLOCATED
    archives = list((run_root / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "completion.json").read_bytes() == malformed


def test_rerun_archives_completion_symlink_without_following_target(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt = allocate_attempt(run_root)
    outside = tmp_path / "outside-completion.json"
    outside.write_bytes(b"external target")
    completion = attempt / "completion.json"
    completion.symlink_to(outside)

    prepared = prepare_attempt(run_root, _fingerprints(), rerun=True)

    assert prepared.outcome is PrepareOutcome.ALLOCATED
    archives = list((run_root / "archive").iterdir())
    assert len(archives) == 1
    archived_completion = archives[0] / "completion.json"
    assert archived_completion.is_symlink()
    assert os.readlink(archived_completion) == str(outside)
    assert outside.read_bytes() == b"external target"


def test_rerun_archives_unreadable_completion_without_reading_it(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    attempt = allocate_attempt(run_root)
    completion = attempt / "completion.json"
    completion.write_bytes(b"unreadable corrupt record")
    completion.chmod(0)

    prepared = prepare_attempt(run_root, _fingerprints(), rerun=True)

    assert prepared.outcome is PrepareOutcome.ALLOCATED
    archives = list((run_root / "archive").iterdir())
    assert len(archives) == 1
    archived_completion = archives[0] / "completion.json"
    assert stat.S_IMODE(archived_completion.lstat().st_mode) == 0


@pytest.mark.parametrize(
    "paths",
    [
        ("../outside.bin",),
        ("artifact.bin", "artifact.bin"),
        ("completion.json",),
        ("completion.json.tmp",),
        ("nested/../artifact.bin",),
        ("/absolute.bin",),
        ("nul\x00artifact.bin",),
    ],
)
def test_completion_rejects_traversal_duplicates_and_record_self_reference(
    tmp_path: Path, paths: tuple[str, ...]
) -> None:
    attempt = allocate_attempt(tmp_path / "run")
    (attempt / "artifact.bin").write_bytes(b"artifact")

    with pytest.raises(ArtifactPathError):
        complete_attempt(attempt, _fingerprints(), paths)

    assert not (attempt / "completion.json").exists()


def test_atomic_writer_rejects_nul_artifact_path_as_typed_error(
    tmp_path: Path,
) -> None:
    attempt = allocate_attempt(tmp_path / "run")

    with pytest.raises(ArtifactPathError, match="NUL|normalized"):
        write_artifact_atomic(attempt, "nested/nul\x00artifact.bin", b"payload")


def test_nested_artifact_creation_fsyncs_each_new_directory_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    attempt = allocate_attempt(tmp_path / "run")
    real_fsync = artifacts.os.fsync
    synced_directories: list[tuple[int, int]] = []

    def record_fsync(descriptor: int) -> None:
        identity = os.fstat(descriptor)
        if stat.S_ISDIR(identity.st_mode):
            synced_directories.append((int(identity.st_dev), int(identity.st_ino)))
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", record_fsync)
    write_artifact_atomic(attempt, "level-one/level-two/artifact.bin", b"data")

    def directory_identity(path: Path) -> tuple[int, int]:
        value = path.stat()
        return int(value.st_dev), int(value.st_ino)

    attempt_id = directory_identity(attempt)
    level_one_id = directory_identity(attempt / "level-one")
    level_two_id = directory_identity(attempt / "level-one/level-two")
    assert synced_directories[:4] == [
        level_one_id,
        attempt_id,
        level_two_id,
        level_one_id,
    ]
    assert synced_directories[-1] == level_two_id


def test_nested_directory_fsync_failure_aborts_before_artifact_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    attempt = allocate_attempt(tmp_path / "run")
    real_fsync = artifacts.os.fsync
    failed = False

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed
        identity = os.fstat(descriptor)
        if stat.S_ISDIR(identity.st_mode) and not failed:
            failed = True
            raise OSError(errno.EIO, "injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(ArtifactError, match="fsync|durability"):
        write_artifact_atomic(attempt, "nested/artifact.bin", b"data")

    assert failed
    assert not (attempt / "nested/artifact.bin").exists()
    assert not list(attempt.rglob("*.tmp-*"))


def test_required_artifact_and_parent_symlinks_are_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    attempt = allocate_attempt(run_root)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (attempt / "linked.bin").symlink_to(outside)

    with pytest.raises(ArtifactPathError, match="symlink"):
        complete_attempt(attempt, _fingerprints(), ("linked.bin",))
    (attempt / "linked-parent").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ArtifactPathError, match="symlink"):
        write_artifact_atomic(attempt, "linked-parent/escape.bin", b"escape")


def test_artifact_replacement_during_hashing_prevents_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    attempt = allocate_attempt(tmp_path / "run")
    write_artifact_atomic(attempt, "artifact.bin", b"original")

    def replace_after_hash(path: Path) -> None:
        replacement = path.with_name("replacement.bin")
        replacement.write_bytes(b"replaced")
        os.replace(replacement, path)

    monkeypatch.setattr(artifacts, "_after_artifact_hash", replace_after_hash)
    with pytest.raises(CorruptCompletionError, match="replaced|changed"):
        complete_attempt(attempt, _fingerprints(), ("artifact.bin",))

    assert not (attempt / "completion.json").exists()


def test_nested_artifact_parent_replacement_prevents_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    attempt = allocate_attempt(tmp_path / "run")
    artifact = write_artifact_atomic(attempt, "nested/artifact.bin", b"same inode")
    replacement_parent = attempt / "replacement-nested"
    replacement_parent.mkdir()
    os.link(artifact, replacement_parent / artifact.name)
    displaced_parent = attempt / "displaced-nested"
    replaced = False

    def replace_parent_after_hash(path: Path) -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        os.rename(path.parent, displaced_parent)
        os.rename(replacement_parent, path.parent)

    monkeypatch.setattr(artifacts, "_after_artifact_hash", replace_parent_after_hash)
    with pytest.raises(CorruptCompletionError, match="parent|replaced|changed"):
        complete_attempt(attempt, _fingerprints(), ("nested/artifact.bin",))

    assert artifact.stat().st_ino == (displaced_parent / artifact.name).stat().st_ino
    assert not (attempt / "completion.json").exists()


def test_attempt_directory_replacement_prevents_completion_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    run_root = tmp_path / "run"
    attempt = allocate_attempt(run_root)
    artifact = write_artifact_atomic(attempt, "artifact.bin", b"same inode")
    replacement_attempt = run_root / "replacement-attempt"
    replacement_attempt.mkdir()
    os.link(artifact, replacement_attempt / artifact.name)
    displaced_attempt = run_root / "displaced-attempt"
    replaced = False

    def replace_attempt_after_hash(path: Path) -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        os.rename(attempt, displaced_attempt)
        os.rename(replacement_attempt, attempt)

    monkeypatch.setattr(artifacts, "_after_artifact_hash", replace_attempt_after_hash)
    with pytest.raises(CorruptCompletionError, match="parent|attempt|replaced|changed"):
        complete_attempt(attempt, _fingerprints(), ("artifact.bin",))

    assert artifact.stat().st_ino == (displaced_attempt / artifact.name).stat().st_ino
    assert not (attempt / "completion.json").exists()
    assert not (displaced_attempt / "completion.json").exists()


def test_completion_parent_replacement_during_publication_leaves_no_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    run_root = tmp_path / "run"
    attempt = allocate_attempt(run_root)
    artifact = write_artifact_atomic(attempt, "artifact.bin", b"same inode")
    replacement_attempt = run_root / "replacement-attempt"
    replacement_attempt.mkdir()
    os.link(artifact, replacement_attempt / artifact.name)
    displaced_attempt = run_root / "displaced-attempt"

    def replace_completion_parent(_attempt: Path) -> None:
        os.rename(attempt, displaced_attempt)
        os.rename(replacement_attempt, attempt)

    monkeypatch.setattr(
        artifacts,
        "_before_completion_publish",
        replace_completion_parent,
        raising=False,
    )
    with pytest.raises(CorruptCompletionError, match="parent|attempt|replaced|changed"):
        complete_attempt(attempt, _fingerprints(), ("artifact.bin",))

    assert not (attempt / "completion.json").exists()
    assert not (displaced_attempt / "completion.json").exists()


def test_parent_replacement_during_post_validation_removes_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    attempt = allocate_attempt(tmp_path / "run")
    artifact = write_artifact_atomic(attempt, "nested/artifact.bin", b"same inode")
    replacement_parent = attempt / "replacement-nested"
    replacement_parent.mkdir()
    os.link(artifact, replacement_parent / artifact.name)
    displaced_parent = attempt / "displaced-nested"
    hash_count = 0

    def replace_parent_during_post_validation(path: Path) -> None:
        nonlocal hash_count
        hash_count += 1
        if hash_count == 2:
            os.rename(path.parent, displaced_parent)
            os.rename(replacement_parent, path.parent)

    monkeypatch.setattr(
        artifacts, "_after_artifact_hash", replace_parent_during_post_validation
    )
    with pytest.raises(CorruptCompletionError, match="parent|replaced|changed"):
        complete_attempt(attempt, _fingerprints(), ("nested/artifact.bin",))

    assert hash_count == 2
    assert not (attempt / "completion.json").exists()


def test_concurrent_attempt_allocation_uses_exclusive_directories(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(lambda _: allocate_attempt(run_root), range(16)))

    assert len(set(attempts)) == 16
    assert {path.name for path in attempts} == {
        f"attempt-{number}" for number in range(1, 17)
    }


def test_completed_attempt_is_never_overwritten(tmp_path: Path) -> None:
    attempt = _completed_attempt(tmp_path / "run")
    completion = attempt / "completion.json"
    original = completion.read_bytes()

    with pytest.raises(CompletedAttemptError, match="completed|overwrite"):
        complete_attempt(attempt, _fingerprints(), ("metrics.json",))
    with pytest.raises(CompletedAttemptError, match="completed|overwrite"):
        write_artifact_atomic(attempt, "late.bin", b"late")

    assert completion.read_bytes() == original


def test_completion_publication_atomically_refuses_competing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.studies.ablation.artifacts as artifacts

    attempt = allocate_attempt(tmp_path / "run")
    write_artifact_atomic(attempt, "metrics.json", b"metrics")
    competing = b'{"competing":true}'

    def publish_competitor(_attempt: Path) -> None:
        (_attempt / "completion.json").write_bytes(competing)

    monkeypatch.setattr(
        artifacts,
        "_at_completion_publish",
        publish_competitor,
        raising=False,
    )
    with pytest.raises(CompletedAttemptError, match="completed|overwrite|exists"):
        complete_attempt(attempt, _fingerprints(), ("metrics.json",))

    assert (attempt / "completion.json").read_bytes() == competing
    assert not (attempt / "completion.json.tmp").exists()

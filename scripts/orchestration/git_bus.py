from __future__ import annotations

import subprocess
import os
from typing import Iterable, Optional
import shutil
from datetime import datetime
from pathlib import Path
import time
try:
    import fcntl  # POSIX file locking
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore
import random


def _run(cmd: Iterable[str], timeout: Optional[int] = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), timeout=timeout, check=check, capture_output=True, text=True)


def _run_with_lock_retry(cmd: Iterable[str], *, timeout: Optional[int] = None, retries: int = 10, log_print=None) -> subprocess.CompletedProcess:
    """Run a git command with backoff when the index lock is busy.

    Concurrent orchestrators in the same repo can temporarily hold
    `.git/index.lock`. Rather than failing immediately, back off with
    jitter and retry a few times. Never force-remove the lock here.
    """
    attempt = 0
    last_cp: Optional[subprocess.CompletedProcess] = None
    base = 0.4
    while attempt <= retries:
        cp = _run(cmd, timeout=timeout)
        last_cp = cp
        err = (cp.stderr or "")
        if cp.returncode == 0:
            return cp
        if "index.lock" in err:
            delay = min(5.0, base * (2 ** attempt)) + random.random() * 0.25
            if log_print:
                log_print(f"[git_bus] index.lock busy; retrying in {delay:.2f}s (attempt {attempt+1}/{retries})…")
            time.sleep(delay)
            attempt += 1
            continue
        # Not a lock-related error; return immediately
        return cp
    return last_cp if last_cp is not None else _run(cmd, timeout=timeout)


class _GitMutex:
    """A coarse repo-wide mutex around Git operations.

    Prevents concurrent orchestrator processes from running index-mutating
    commands at the same time, which commonly triggers `.git/index.lock`
    contention. Uses POSIX `flock` when available; otherwise acts as a
    best-effort no-op (still relying on `_run_with_lock_retry`).
    """

    def __init__(self, lock_path: str = ".git/orchestrator.lock", timeout: float = 30.0, log_print=None) -> None:
        self.lock_path = lock_path
        self.timeout = timeout
        self.log_print = log_print
        self._fh: Optional[object] = None

    def __enter__(self):
        if fcntl is None:
            # Non-POSIX: no-op
            return self
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        self._fh = open(self.lock_path, "w")
        start = time.time()
        attempt = 0
        while True:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                # record holder pid for forensics
                try:
                    self._fh.seek(0)
                    self._fh.truncate()
                    self._fh.write(f"pid={os.getpid()}\n")
                    self._fh.flush()
                except Exception:
                    pass
                return self
            except BlockingIOError:
                attempt += 1
                if (time.time() - start) > self.timeout:
                    if self.log_print:
                        self.log_print(f"[git_bus] git mutex acquire timeout after {self.timeout:.1f}s; proceeding without lock (will retry on index.lock)")
                    return self
                # short jittered backoff while waiting for peer
                time.sleep(min(0.5, 0.05 * (2 ** min(attempt, 4))))

    def __exit__(self, exc_type, exc, tb):
        if fcntl is None:
            return False
        try:
            if self._fh is not None:
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
                finally:
                    self._fh.close()
        except Exception:
            pass
        return False


def git_lock(log_print=None, timeout: float = 30.0) -> _GitMutex:
    """Return a context manager that acquires a coarse Git mutex.

    Usage:
        with git_lock(logger):
            safe_pull(logger, ...)
            add([...]); commit("msg"); push_with_rebase(...)
    """
    return _GitMutex(timeout=timeout, log_print=log_print)


def _rebase_in_progress() -> bool:
    return os.path.isdir(os.path.join('.git', 'rebase-merge')) or os.path.isdir(os.path.join('.git', 'rebase-apply'))


def _abort_rebase(log_print) -> None:
    """Attempt to abort any in-progress rebase and hard-clean stale state.

    In practice, Git can leave .git/rebase-merge or .git/rebase-apply
    directories behind if a prior process crashed. When that happens,
    a subsequent `git pull --rebase` or `git rebase` will fail with
    "rebase-merge directory exists". We first try a normal abort, and
    if the state persists we move the stale directories out to a
    timestamped backup under tmp/git-rebase-backups/ so the operation
    can proceed while preserving forensics.
    """
    try:
        _run(["git", "rebase", "--abort"])
        log_print("Aborted in-progress rebase.")
    except Exception:
        # Ignore and proceed to hard cleanup if needed
        pass

    # If rebase state still present, forcibly relocate it out of .git
    if _rebase_in_progress():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_root = Path("tmp") / "git-rebase-backups" / ts
        try:
            backup_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create backup directory, fall back to direct removal
            backup_root = None

        for dname in ("rebase-merge", "rebase-apply"):
            src = Path(".git") / dname
            if src.exists():
                try:
                    if backup_root is not None:
                        dst = backup_root / dname
                        shutil.move(str(src), str(dst))
                        log_print(f"[git_bus] Moved stale {src} to {dst}.")
                    else:
                        shutil.rmtree(src)
                        log_print(f"[git_bus] Removed stale {src} (no backup).")
                except Exception as e:
                    # As a last resort, try to unlink tree contents
                    try:
                        shutil.rmtree(src, ignore_errors=True)
                        log_print(f"[git_bus] Force-removed stale {src} after error: {e}.")
                    except Exception:
                        # Leave a breadcrumb but continue; subsequent operations will report failure
                        log_print(f"[git_bus] WARNING: Could not clear stale {src}: {e}")


def safe_pull(log_print, remote: Optional[str] = None, branch: Optional[str] = None) -> bool:
    """
    Attempt to update the current branch with a rebase pull.

    Returns True on success. On failure, logs details and attempts a
    non-rebase pull as a recovery path. Detects and fast-fails on
    untracked-file overwrite collisions so callers can surface clear
    remediation guidance instead of silently stalling.
    """
    if _rebase_in_progress():
        _abort_rebase(log_print)
    try:
        # Prefer explicit fetch+rebase when target specified to bypass
        # branch.<name>.merge multi-branch configs that break `git pull --rebase`.
        if remote and branch:
            # Ensure no rebase is in progress
            if _rebase_in_progress():
                _abort_rebase(log_print)
            # Fetch latest for the specific branch
            fcp = _run_with_lock_retry(["git", "fetch", remote, branch], timeout=60, log_print=log_print)
            if fcp.stdout:
                log_print(fcp.stdout.rstrip())
            if fcp.stderr:
                log_print(fcp.stderr.rstrip())
            # Attempt a rebase onto the fetched remote branch with autostash
            rcp = _run_with_lock_retry(["git", "-c", "rebase.autoStash=true", "rebase", f"{remote}/{branch}"], timeout=180, log_print=log_print)
            if rcp.stdout:
                log_print(rcp.stdout.rstrip())
            if rcp.stderr:
                log_print(rcp.stderr.rstrip())
            # Detect untracked-file collisions or similar hard blockers
            err = (rcp.stderr or "").lower()
            if (
                "untracked working tree files would be overwritten" in err
                or "would be overwritten by merge" in err
                or "would be overwritten by checkout" in err
            ):
                log_print(
                    "ERROR: Rebase blocked by untracked-file collisions. "
                    "Move/remove conflicting files and retry."
                )
                return False
            if rcp.returncode == 0:
                return True
            # If rebase failed due to local modifications that weren't autostashed, try manual one-shot autostash
            if (
                "you have unstaged changes" in err
                or "please commit or stash them" in err
                or "your local changes to the following files would be overwritten" in err
            ):
                try:
                    log_print("[git_bus] Rebase blocked; attempting manual autostash…")
                    _run_with_lock_retry(["git", "stash", "push", "-u", "-m", "orchestrator-safe_pull-autostash"], timeout=45, log_print=log_print)
                    rcp2 = _run_with_lock_retry(["git", "rebase", f"{remote}/{branch}"], timeout=180, log_print=log_print)
                    if rcp2.stdout:
                        log_print(rcp2.stdout.rstrip())
                    if rcp2.stderr:
                        log_print(rcp2.stderr.rstrip())
                    pop = _run_with_lock_retry(["git", "stash", "pop"], timeout=45, log_print=log_print)
                    if pop.stdout:
                        log_print(pop.stdout.rstrip())
                    if pop.stderr:
                        log_print(pop.stderr.rstrip())
                    return rcp2.returncode == 0
                except Exception as e2:
                    log_print(f"[git_bus] Manual autostash rebase failed: {e2}")
            # Final fallthrough for explicit remote/branch path
            return False
        # Legacy path: rely on upstream config
        cmd = ["git", "pull", "--rebase"]
        cp = _run_with_lock_retry(cmd, timeout=60, log_print=log_print)
        if cp.stdout:
            log_print(cp.stdout.rstrip())
        if cp.stderr:
            log_print(cp.stderr.rstrip())
        # Fast-fail when untracked files would be overwritten
        err = (cp.stderr or "").lower()
        if (
            "untracked working tree files would be overwritten" in err
            or "would be overwritten by merge" in err
            or "would be overwritten by checkout" in err
        ):
            log_print(
                "ERROR: Pull blocked by untracked-file collisions. "
                "Move/remove conflicting files and retry."
            )
            return False
        # If pull failed due to local modifications, attempt one-shot autostash
        if cp.returncode != 0 and (
            "cannot pull with rebase: you have unstaged changes" in err
            or "please commit or stash them" in err
            or "your local changes to the following files would be overwritten" in err
        ):
            try:
                log_print("[git_bus] Detected local modifications blocking rebase; attempting autostash…")
                _run_with_lock_retry(["git", "stash", "push", "-u", "-m", "orchestrator-safe_pull-autostash"], timeout=45, log_print=log_print)
                cmd2 = ["git", "pull", "--rebase"]
                if remote and branch:
                    cmd2 += [remote, branch]
                cp2 = _run_with_lock_retry(cmd2, timeout=90, log_print=log_print)
                if cp2.stdout:
                    log_print(cp2.stdout.rstrip())
                if cp2.stderr:
                    log_print(cp2.stderr.rstrip())
                # Try to restore working changes
                pop = _run_with_lock_retry(["git", "stash", "pop"], timeout=45, log_print=log_print)
                if pop.stdout:
                    log_print(pop.stdout.rstrip())
                if pop.stderr:
                    # Conflicts here are acceptable; surface but don't fail pull outcome
                    log_print(pop.stderr.rstrip())
                return cp2.returncode == 0
            except Exception as e2:
                log_print(f"[git_bus] Autostash pull failed: {e2}")
                # Fall through to recovery path
        return cp.returncode == 0
    except Exception as e:
        log_print(f"git pull --rebase failed or timed out: {e}")

    # Recovery path
    _abort_rebase(log_print)
    if remote and branch:
        # Fetch and attempt fast-forward merge
        fcp = _run_with_lock_retry(["git", "fetch", remote, branch], timeout=60, log_print=log_print)
        if fcp.stdout:
            log_print(fcp.stdout.rstrip())
        if fcp.stderr:
            log_print(fcp.stderr.rstrip())
        mcp = _run_with_lock_retry(["git", "merge", "--ff-only", f"{remote}/{branch}"], timeout=120, log_print=log_print)
        if mcp.stdout:
            log_print(mcp.stdout.rstrip())
        if mcp.stderr:
            log_print(mcp.stderr.rstrip())
        return mcp.returncode == 0
    else:
        cp2 = _run_with_lock_retry(["git", "pull", "--no-rebase"], timeout=60, log_print=log_print)
        if cp2.stdout:
            log_print(cp2.stdout.rstrip())
        if cp2.stderr:
            log_print(cp2.stderr.rstrip())
        err2 = (cp2.stderr or "").lower()
        if (
            "untracked working tree files would be overwritten" in err2
            or "would be overwritten by merge" in err2
            or "would be overwritten by checkout" in err2
        ):
            log_print(
                "ERROR: Pull blocked by untracked-file collisions. "
                "Move/remove conflicting files and retry."
            )
            return False
        return cp2.returncode == 0


def add(paths: Iterable[str]) -> None:
    # Add may also fail on index.lock; retry briefly
    _run_with_lock_retry(["git", "add", *paths], timeout=60)


def commit(message: str) -> bool:
    cp = _run_with_lock_retry(["git", "commit", "-m", message], timeout=120)
    return cp.returncode == 0


def push(log_print) -> None:
    cp = _run(["git", "push"])  # network op; index lock not expected
    if cp.stdout:
        log_print(cp.stdout.rstrip())
    if cp.stderr:
        log_print(cp.stderr.rstrip())


def push_to(branch: str, log_print, remote: str = "origin") -> None:
    cp = _run(["git", "push", remote, f"HEAD:{branch}"])  # index lock not expected
    if cp.stdout:
        log_print(cp.stdout.rstrip())
    if cp.stderr:
        log_print(cp.stderr.rstrip())

def push_with_rebase(branch: str, log_print, remote: str = "origin") -> bool:
    """
    Try to push HEAD to the given remote branch. If rejected, attempt
    to reconcile by pulling/rebasing and retry the push. Returns True
    on success, False otherwise.
    """
    cp = _run(["git", "push", remote, f"HEAD:{branch}"])  # index lock not expected
    if cp.stdout:
        log_print(cp.stdout.rstrip())
    if cp.returncode == 0:
        if cp.stderr:
            log_print(cp.stderr.rstrip())
        return True
    if cp.stderr:
        log_print(cp.stderr.rstrip())
    # Attempt to reconcile and retry (explicit remote/branch to avoid config pitfalls)
    safe_pull(log_print, remote, branch)
    cp2 = _run(["git", "push", remote, f"HEAD:{branch}"])
    if cp2.stdout:
        log_print(cp2.stdout.rstrip())
    if cp2.stderr:
        log_print(cp2.stderr.rstrip())
    return cp2.returncode == 0


def short_head() -> str:
    cp = _run(["git", "rev-parse", "--short", "HEAD"]) 
    return (cp.stdout or "").strip() or "unknown"


def current_branch() -> str:
    cp = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) 
    return (cp.stdout or "").strip()


def assert_on_branch(expected: str, log_print) -> None:
    cur = current_branch()
    if cur != expected:
        log_print(f"ERROR: Expected to run on branch '{expected}', but on '{cur}'. Aborting.")
        raise SystemExit(2)


def has_unpushed_commits() -> bool:
    branch = current_branch()
    if not branch:
        return False
    cp = _run(["git", "diff", "--quiet", f"origin/{branch}..HEAD"])  # exit code 1 if diff present
    return cp.returncode == 1

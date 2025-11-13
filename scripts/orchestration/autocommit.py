from __future__ import annotations

import os
from subprocess import run, PIPE
from pathlib import PurePosixPath
from typing import Callable, Iterable, Tuple, List, Set, Optional


def _run_list(cmd: Iterable[str]) -> List[str]:
    cp = run(list(cmd), stdout=PIPE, stderr=PIPE, text=True)
    if cp.returncode != 0:
        return []
    return [p for p in (cp.stdout or "").splitlines() if p.strip()]


def list_dirty_paths(include_ignored_untracked: bool = False) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Return (unstaged_mod, staged_mod, untracked, ignored_untracked) path lists.
    If include_ignored_untracked is True, also list ignored untracked files.
    """
    unstaged_mod = _run_list(["git", "diff", "--name-only", "--diff-filter=M"])
    staged_mod = _run_list(["git", "diff", "--cached", "--name-only", "--diff-filter=AM"])
    untracked = _run_list(["git", "ls-files", "--others", "--exclude-standard"])
    ignored_untracked: List[str] = []
    if include_ignored_untracked:
        # list ignored, untracked files according to .gitignore
        ignored_untracked = _run_list(["git", "ls-files", "--others", "-i", "--exclude-standard"])
    return unstaged_mod, staged_mod, untracked, ignored_untracked


def autocommit_reports(
    *,
    allowed_extensions: Set[str],
    max_file_bytes: int,
    max_total_bytes: int,
    force_add: bool,
    logger: Callable[[str], None],
    commit_message_prefix: str = "AUTO: reports evidence — tests: not run",
    skip_predicate: Optional[Callable[[str], bool]] = None,
    allowed_path_globs: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Stage and commit report-like artifacts filtered by extension and size caps.
    The optional skip_predicate can suppress specific paths from staging.
    Returns (committed, staged_paths, skipped_paths).
    """
    # Normalize extensions and path allowlist
    allowed_exts = {e.lower() for e in allowed_extensions}
    path_globs: Tuple[str, ...] = tuple(g for g in (allowed_path_globs or []) if g)
    unstaged_mod, staged_mod, untracked, ignored_untracked = list_dirty_paths(include_ignored_untracked=force_add)
    dirty_all: List[str] = []
    seen: Set[str] = set()
    for p in unstaged_mod + staged_mod + untracked + ignored_untracked:
        if p not in seen:
            dirty_all.append(p)
            seen.add(p)

    staged: List[str] = []
    skipped: List[str] = []
    total_bytes = 0

    for p in dirty_all:
        if path_globs:
            posix_path = PurePosixPath(p)
            if not any(posix_path.match(glob) for glob in path_globs):
                skipped.append(p)
                continue
        if skip_predicate and skip_predicate(p):
            skipped.append(p)
            continue
        # Determine extension
        ext = os.path.splitext(p)[1].lower()
        if ext not in allowed_exts:
            skipped.append(p)
            continue
        try:
            if not os.path.isfile(p):
                skipped.append(p)
                continue
            size = os.path.getsize(p)
            if size > max_file_bytes or (total_bytes + size) > max_total_bytes:
                skipped.append(p)
                continue
        except FileNotFoundError:
            skipped.append(p)
            continue

        # check-ignore
        ignored = False
        if force_add:
            chk = run(["git", "check-ignore", "-q", p])
            ignored = (chk.returncode == 0)

        if ignored and force_add:
            add = run(["git", "add", "-f", p], stdout=PIPE, stderr=PIPE, text=True)
        else:
            add = run(["git", "add", p], stdout=PIPE, stderr=PIPE, text=True)
        if add.returncode != 0:
            skipped.append(p)
            continue
        staged.append(p)
        total_bytes += size

    committed = False
    if staged:
        body = "\n\nFiles:\n" + "\n".join(f" - {x}" for x in staged)
        from .git_bus import commit  # local import to avoid cycle concerns at module import
        committed = commit(f"{commit_message_prefix}{body}")
        if committed:
            logger(f"[reports] Auto-committed {len(staged)} files ({total_bytes} bytes)")
        else:
            logger("[reports] WARNING: git commit failed; staged files remain staged")
    return committed, staged, skipped

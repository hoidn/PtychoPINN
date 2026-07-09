#!/usr/bin/env python
"""Deterministic main-overlay transform: produce a resnet-family-free tree.

Given a source tree-ish, materialize a transformed working tree and print its
git tree SHA. The transform is:

  (a) delete every path listed in ``main_overlay_exclude.txt``;
  (b) ``git apply`` every patch in ``main_overlay_patches/`` (family-entry
      strips on kept files), aborting loudly if any patch fails to apply;
  (c) graft every path listed in ``main_overlay_graft.txt`` from the tree-ish
      given via ``--graft-from`` (no-op when omitted) -- main-side
      infrastructure (e.g. the CI gate) that the fno-stable source tree does
      not carry, so a pure ``T(fno-stable)`` tree would silently lose it;
  (d) preserve every gitlink (submodule) in the source tree-ish except those
      listed in ``main_overlay_gitlink_exclude.txt`` -- ``git archive`` (the
      materialization mechanism) skips submodules entirely, so without this
      step every gitlink is silently dropped from the output tree;
  (e) run five built-in gates and refuse to emit a tree that fails them:
        - grep gate:     no family reference survives outside the ALLOW set;
        - dangling gate: no kept .py imports a repo-internal module the
                         transform removed (importers of deleted modules --
                         valid syntax, so grep and compile miss them);
        - import gate:   the materialized tree byte-compiles and the four
                         gate modules import cleanly;
        - patch-anchor:  enforced during (b) -- a non-applying patch names
                         itself and aborts (drift detection for future resyncs);
        - closure gate:  the EMITTED TREE OBJECT's path inventory equals
                         (source inventory - path excludes - gitlink
                         excludes) | graft paths, independently re-derived
                         from the transform's own stated intent and compared
                         against ``git ls-tree`` of the actual written tree --
                         not the scratch directory. This is the one gate that
                         catches paths present on disk that never made it
                         into the tree object (e.g. a materialized-tree
                         ``.gitignore`` silently excluding an intentionally
                         kept/grafted path from ``git add``).

Grafted paths are ordinary blobs written to the materialized tree before the
gates run, so the grep/dangling/import gates cover them naturally. Gitlinks
are inserted directly into the throwaway index used for the final
``write-tree`` (a gitlink has no working-tree content to place on disk or for
a text-scanning gate to examine).

Determinism: identical input tree-ish -> identical output tree SHA. The tree
SHA is computed against the repository object store via a throwaway index, so
it is content-addressed and independent of timestamps or scratch location.

This tool is intentionally checked in so ``main`` carries its own resync
machinery. See docs/plans/2026-07-07-rebase-fno-stable-onto-main.md (Task 2).

The last line of stdout is the emitted tree SHA. All progress and gate output
goes to stderr.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# Configuration (defaults for the real repository; tests inject their own).
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_EXCLUDE = HERE / "main_overlay_exclude.txt"
DEFAULT_PATCHES = HERE / "main_overlay_patches"
DEFAULT_GITLINK_EXCLUDE = HERE / "main_overlay_gitlink_exclude.txt"
DEFAULT_GRAFT_LIST = HERE / "main_overlay_graft.txt"

# The family fingerprint. Case-insensitive, matches the plan's audit sweep.
FAMILY_PATTERN = re.compile(
    r"hybrid_resnet|srunet|spectral_resnet|resnet_components|hybres",
    re.IGNORECASE,
)

# Paths where a family mention is an authorized record (prose) rather than a
# runnable surface. A file is ALLOWed if its repo-relative POSIX path starts
# with one of these prefixes or equals one of the explicit entries.
DEFAULT_ALLOW_PREFIXES: tuple[str, ...] = (
    "docs/",
    "plans/",
    "specs/",
    ".superpowers/",
    # The transform tooling necessarily names the family it strips.
    "scripts/main_overlay/main_overlay_patches/",
)
DEFAULT_ALLOW_FILES: frozenset[str] = frozenset(
    {
        "scripts/main_overlay/build_main_overlay.py",
        "scripts/main_overlay/main_overlay_exclude.txt",
        "tests/test_build_main_overlay.py",
        # Paper record/reporting surfaces: family named only in result tables,
        # figure paths, and provenance strings (no family import). Plan Task 2
        # ALLOW clause + Decision 3 ("prose/records stay").
        "scripts/studies/metrics_tables.py",
        "scripts/studies/paper_evidence_audit.py",
        "scripts/studies/paper_model_config_table.py",
        "scripts/studies/paper_results_refresh.py",
        "tests/studies/test_metrics_tables.py",
        "tests/studies/test_paper_evidence_audit.py",
        "tests/studies/test_paper_model_config_table.py",
        "tests/studies/test_paper_results_refresh.py",
        "tests/studies/test_paper_provenance.py",
        "tests/studies/test_paper_efficiency_table.py",
    }
)

# Modules that must import cleanly against the materialized tree.
DEFAULT_IMPORT_TARGETS: tuple[str, ...] = (
    "ptycho_torch.model",
    "ptycho_torch.generators.registry",
    "ptycho_torch.workflows.components",
    "ptycho_torch.config_bridge",
)

# Top-level package names whose imports are repo-internal (the dangling-import
# gate resolves these against the materialized tree; everything else is a
# third-party/stdlib import and is out of scope).
INTERNAL_TOP_PACKAGES: tuple[str, ...] = (
    "ptycho",
    "ptycho_torch",
    "scripts",
    "tests",
)

# Directories never worth scanning for family references (binary/vendored).
_SKIP_DIR_NAMES = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npz", ".npy", ".pt", ".pth",
    ".pyc", ".so", ".zip", ".gz", ".tar", ".ico", ".h5", ".hdf5", ".bin",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".webp",
}


class TransformError(RuntimeError):
    """Raised when a gate fails or a patch does not apply."""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _run(cmd: Sequence[str], *, cwd: Path | None = None, env: dict | None = None):
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, env=env,
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Stage 0: materialize.
# ---------------------------------------------------------------------------

def materialize(tree_ish: str, dest: Path, repo_root: Path) -> None:
    """Extract ``tree_ish`` from ``repo_root`` into ``dest`` (read-only w.r.t. repo).

    Uses ``git archive | tar -x`` so the repository working tree is never
    touched.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # git archive emits a binary tar stream; capture as bytes (no text decode).
    archive = subprocess.run(
        ["git", "archive", "--format=tar", tree_ish],
        cwd=str(repo_root), capture_output=True,
    )
    if archive.returncode != 0:
        raise TransformError(
            f"git archive {tree_ish!r} failed: "
            f"{archive.stderr.decode(errors='replace').strip()}"
        )
    tar = subprocess.run(
        ["tar", "-x", "-C", str(dest)], input=archive.stdout,
        capture_output=True,
    )
    if tar.returncode != 0:
        raise TransformError(f"tar extract failed: {tar.stderr.decode(errors='replace')}")


# ---------------------------------------------------------------------------
# Stage a: exclusions.
# ---------------------------------------------------------------------------

def read_path_list(path: Path) -> list[str]:
    """Read a newline path list, dropping blanks and ``#`` comments."""
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def apply_exclusions(tree: Path, entries: Iterable[str]) -> list[str]:
    """Delete each listed path from ``tree``.

    Entry grammar (POSIX, repo-relative):
      * ``dir/``      -> recursively delete a directory (must exist);
      * ``a/b*.py``   -> glob relative to ``tree`` (zero matches tolerated);
      * ``a/b.py``    -> exact file or directory (must exist).

    Exact and directory entries are required to exist: a missing target means
    the source tree drifted and the audit is stale -> fail loudly.
    """
    removed: list[str] = []
    for entry in entries:
        is_dir = entry.endswith("/")
        is_glob = any(ch in entry for ch in "*?[")
        if is_glob:
            matches = sorted(tree.glob(entry))
            if not matches:
                # Zero matches are tolerated by the grammar, but a glob that
                # stops matching usually means the source tree drifted -- surface
                # it so a resync notices instead of silently under-excluding.
                _log(f"[exclude] WARNING: glob matched zero paths (drift?): {entry!r}")
            for m in matches:
                _delete(m)
                removed.append(str(m.relative_to(tree)))
            continue
        target = tree / entry.rstrip("/")
        if not target.exists():
            raise TransformError(
                f"exclude entry does not exist in source tree (drift?): {entry!r}"
            )
        if is_dir and not target.is_dir():
            raise TransformError(f"exclude entry {entry!r} is not a directory")
        _delete(target)
        removed.append(entry.rstrip("/"))
    return removed


def _delete(target: Path) -> None:
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink()


# ---------------------------------------------------------------------------
# Stage b: patches (with the patch-anchor gate).
# ---------------------------------------------------------------------------

def apply_patches(tree: Path, patches_dir: Path) -> list[str]:
    """Apply every ``*.patch`` in ``patches_dir`` (sorted) inside ``tree``.

    ``git apply --check`` runs first; a non-applying patch aborts with its
    name -- the patch-anchor gate that flags anchor drift on future resyncs.
    """
    applied: list[str] = []
    if not patches_dir.exists():
        return applied
    for patch in sorted(patches_dir.glob("*.patch")):
        name = patch.name
        check = _run(["git", "apply", "--check", "-p1", str(patch)], cwd=tree)
        if check.returncode != 0:
            raise TransformError(
                f"PATCH-ANCHOR FAILURE: {name} does not apply to the source tree.\n"
                f"  git apply --check said: {check.stderr.strip()}\n"
                f"  Re-anchor this patch against the current tree before resyncing."
            )
        real = _run(["git", "apply", "-p1", str(patch)], cwd=tree)
        if real.returncode != 0:  # pragma: no cover - --check already passed
            raise TransformError(f"PATCH-ANCHOR FAILURE: {name} failed to apply: {real.stderr.strip()}")
        applied.append(name)
    return applied


# ---------------------------------------------------------------------------
# Stage c: graft (main-side infrastructure absent from the fno-stable source).
# ---------------------------------------------------------------------------

def apply_graft(
    tree: Path, graft_from: str, graft_list_path: Path, repo_root: Path,
) -> list[str]:
    """Write every path in ``graft_list_path`` into ``tree`` as it exists at
    ``graft_from`` (overwriting any same-named file already materialized from
    the source tree-ish). Fails loudly if a listed path is missing from
    ``graft_from``. Returns the list of grafted paths.
    """
    applied: list[str] = []
    if not graft_list_path.exists():
        return applied
    for rel in read_path_list(graft_list_path):
        ls = _run(["git", "ls-tree", graft_from, "--", rel], cwd=repo_root)
        if not ls.stdout.strip():
            raise TransformError(
                f"GRAFT FAILURE: {rel!r} is missing from graft source {graft_from!r}."
            )
        meta, _, _found_path = ls.stdout.strip().partition("\t")
        mode, _obj_type, blob_sha = meta.split()
        # Binary-safe: git cat-file -p emits the blob's raw bytes (npz/other
        # binary grafts included), so this must NOT go through _run()'s
        # text=True decoding -- that would corrupt or reject non-UTF8 bytes.
        cat = subprocess.run(
            ["git", "cat-file", "-p", blob_sha], cwd=str(repo_root),
            capture_output=True,
        )
        if cat.returncode != 0:
            raise TransformError(
                f"GRAFT FAILURE: cannot read blob for {rel!r} ({blob_sha}): "
                f"{cat.stderr.decode(errors='replace').strip()}"
            )
        dest = tree / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cat.stdout)
        if mode == "100755":
            dest.chmod(dest.stat().st_mode | 0o111)
        applied.append(rel)
    return applied


# ---------------------------------------------------------------------------
# Stage d: gitlink preservation (git archive drops all submodules).
# ---------------------------------------------------------------------------

def list_gitlinks(tree_ish: str, repo_root: Path) -> list[tuple[str, str]]:
    """Return ``(repo_relative_path, commit_sha)`` for every gitlink
    (mode 160000 entry) in ``tree_ish``."""
    out = _run(["git", "ls-tree", "-r", tree_ish], cwd=repo_root)
    if out.returncode != 0:
        raise TransformError(
            f"gitlink scan: git ls-tree {tree_ish!r} failed: {out.stderr.strip()}"
        )
    gitlinks: list[tuple[str, str]] = []
    for line in out.stdout.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[0] == "160000":
            gitlinks.append((path, parts[2]))
    return gitlinks


# ---------------------------------------------------------------------------
# Stage e: gates.
# ---------------------------------------------------------------------------

def _iter_text_files(tree: Path):
    for root, dirs, files in os.walk(tree):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES]
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in _BINARY_SUFFIXES:
                continue
            if fpath.is_symlink():
                continue
            yield fpath


def _is_allowed(rel: str, allow_prefixes: Sequence[str], allow_files) -> bool:
    if rel in allow_files:
        return True
    return any(rel.startswith(p) for p in allow_prefixes)


def grep_gate(
    tree: Path,
    *,
    pattern: re.Pattern = FAMILY_PATTERN,
    allow_prefixes: Sequence[str] = DEFAULT_ALLOW_PREFIXES,
    allow_files=DEFAULT_ALLOW_FILES,
) -> list[str]:
    """Return sorted repo-relative paths that carry a family reference and are
    NOT in the ALLOW set. Empty result == gate pass."""
    offenders: list[str] = []
    for fpath in _iter_text_files(tree):
        try:
            text = fpath.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        if not pattern.search(text):
            continue
        rel = fpath.relative_to(tree).as_posix()
        if not _is_allowed(rel, allow_prefixes, allow_files):
            offenders.append(rel)
    return sorted(offenders)


# ---------------------------------------------------------------------------
# Stage c: dangling-import gate (importers of transform-removed modules).
# ---------------------------------------------------------------------------

def _source_paths(tree_ish: str, repo_root: Path) -> frozenset[str]:
    """Repo-relative POSIX paths present in the SOURCE ``tree_ish``.

    Used to distinguish a module the transform REMOVED (present in source,
    absent from output) from a pre-existing dangling import (absent from both)
    -- only the former is this transform's responsibility to catch.
    """
    out = _run(["git", "ls-tree", "-r", "--name-only", tree_ish], cwd=repo_root)
    if out.returncode != 0:
        raise TransformError(
            f"dangling gate: git ls-tree {tree_ish!r} failed: {out.stderr.strip()}"
        )
    return frozenset(out.stdout.splitlines())


def _output_module_present(dotted: str, tree: Path) -> bool:
    """Does ``dotted`` resolve to a module file or package dir in ``tree``?"""
    parts = dotted.split(".")
    if (tree / Path(*parts)).with_suffix(".py").exists():
        return True
    return (tree / Path(*parts)).is_dir()


def _source_module_present(dotted: str, source_paths: frozenset[str]) -> bool:
    """Did ``dotted`` resolve to a module in the source path set?"""
    base = "/".join(dotted.split("."))
    if base + ".py" in source_paths:
        return True
    prefix = base + "/"
    return any(p.startswith(prefix) for p in source_paths)


def _iter_internal_import_targets(
    node: ast.AST, internal_top: Sequence[str],
) -> Iterator[tuple[str, int]]:
    """Yield ``(dotted_module, lineno)`` for every repo-internal import target.

    ``import a.b`` yields ``a.b``; ``from a.b import c`` yields both ``a.b`` and
    ``a.b.c`` (the latter catches ``from pkg import <removed submodule>``; a
    plain attribute name simply won't resolve as a source module and is ignored).
    """
    tops = set(internal_top)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.name.split(".")[0] in tops:
                    yield alias.name, sub.lineno
        elif isinstance(sub, ast.ImportFrom):
            if sub.level != 0 or not sub.module:
                continue
            if sub.module.split(".")[0] not in tops:
                continue
            yield sub.module, sub.lineno
            for alias in sub.names:
                if alias.name != "*":
                    yield f"{sub.module}.{alias.name}", sub.lineno


def dangling_import_gate(
    tree: Path,
    *,
    tree_ish: str,
    repo_root: Path,
    internal_top: Sequence[str] = INTERNAL_TOP_PACKAGES,
) -> list[str]:
    """Return sorted ``file:line -> module`` for every kept .py that imports a
    repo-internal module the transform removed. Empty result == gate pass.

    A dangling ``import`` is valid syntax and byte-compiles, so the grep and
    import gates miss it; this static AST pass is the one that catches importers
    of deleted modules. Pre-existing dangling imports (module absent from the
    source tree too) are ignored -- they predate and are out of scope for this
    transform.
    """
    source_paths = _source_paths(tree_ish, repo_root)
    offenders: list[str] = []
    for fpath in tree.rglob("*.py"):
        if any(part in _SKIP_DIR_NAMES for part in fpath.parts):
            continue
        rel = fpath.relative_to(tree).as_posix()
        try:
            node = ast.parse(fpath.read_text(errors="replace"), filename=rel)
        except SyntaxError:
            continue  # the import gate's compileall owns syntax errors
        for dotted, lineno in _iter_internal_import_targets(node, internal_top):
            if _output_module_present(dotted, tree):
                continue
            if not _source_module_present(dotted, source_paths):
                continue  # pre-existing dangler; not introduced by the transform
            offenders.append(f"{rel}:{lineno} -> {dotted}")
    return sorted(offenders)


def import_gate(
    tree: Path,
    *,
    import_targets: Sequence[str] = DEFAULT_IMPORT_TARGETS,
) -> None:
    """Byte-compile the tree, then import the four gate modules from it."""
    compiled = _run([sys.executable, "-m", "compileall", "-q", "."], cwd=tree)
    if compiled.returncode != 0:
        raise TransformError(
            "IMPORT GATE (compileall) failed:\n"
            + (compiled.stdout + compiled.stderr).strip()
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree) + os.pathsep + env.get("PYTHONPATH", "")
    stmt = "import " + ", ".join(import_targets)
    imported = _run([sys.executable, "-c", stmt], cwd=tree, env=env)
    if imported.returncode != 0:
        raise TransformError(
            f"IMPORT GATE failed for `{stmt}`:\n"
            + (imported.stdout + imported.stderr).strip()
        )


# ---------------------------------------------------------------------------
# Tree SHA.
# ---------------------------------------------------------------------------

def _purge_pycache(tree: Path) -> int:
    """Remove every ``__pycache__`` directory from ``tree``.

    ``import_gate``'s ``compileall`` step is a byte-compilation SIDE EFFECT,
    not intentional tree content -- with the forced ``git add -A -f .`` below
    (needed so a repo-wide ``.gitignore`` can't silently drop intentional
    paths), an un-purged ``__pycache__`` would otherwise be swept into the
    emitted tree object too. Called unconditionally so this holds regardless
    of which gates ran.
    """
    removed = 0
    for cache_dir in list(tree.rglob("__pycache__")):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            removed += 1
    return removed


def write_tree(
    tree: Path, repo_root: Path, *, gitlinks: Sequence[tuple[str, str]] = (),
) -> str:
    """Stage ``tree`` into a throwaway index against ``repo_root``'s object
    store and return the resulting tree SHA (content-addressed, deterministic).

    ``gitlinks`` are inserted via ``update-index --cacheinfo`` AFTER
    ``add -A -f .`` -- a gitlink has no working-tree file, and ``add -A``
    reconciles the index against the working tree, so an index entry with no
    matching on-disk path is treated as a deletion if inserted first.

    The add is forced (``-f``): the materialized ``tree`` directory is
    intentional content (archive output + patches + grafts) plus gate
    byproducts (``__pycache__``, purged below first). Without ``-f``,
    ``git add -A`` respects the materialized tree's OWN ``.gitignore`` (e.g. a
    repo-wide ``*.npz`` rule), silently dropping intentionally-kept or
    freshly-grafted paths from the index -- present on disk, absent from the
    emitted tree object. The closure gate (``tree_closure_gate``) is the
    independent check that would catch this class of defect even if ``-f``
    regressed.
    """
    purged = _purge_pycache(tree)
    if purged:
        _log(f"[tree] purged {purged} __pycache__ dir(s) before hashing")
    git_dir = str(repo_root / ".git")
    index_dir = tempfile.mkdtemp(prefix="overlay-index-")
    index_path = os.path.join(index_dir, "index")  # git creates it fresh
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = index_path
    try:
        add = _run(
            ["git", "--git-dir", git_dir, "--work-tree", ".", "add", "-A", "-f", "."],
            cwd=tree, env=env,
        )
        if add.returncode != 0:
            raise TransformError(f"git add (tree hashing) failed: {add.stderr.strip()}")
        for path, sha in gitlinks:
            ui = _run(
                ["git", "--git-dir", git_dir, "update-index", "--add",
                 "--cacheinfo", f"160000,{sha},{path}"],
                cwd=tree, env=env,
            )
            if ui.returncode != 0:
                raise TransformError(
                    f"gitlink insertion failed for {path!r} ({sha}): {ui.stderr.strip()}"
                )
        wt = _run(
            ["git", "--git-dir", git_dir, "--work-tree", ".", "write-tree"],
            cwd=tree, env=env,
        )
        if wt.returncode != 0:
            raise TransformError(f"git write-tree failed: {wt.stderr.strip()}")
        return wt.stdout.strip()
    finally:
        shutil.rmtree(index_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage f: tree-closure gate (does the emitted TREE OBJECT match intent?).
# ---------------------------------------------------------------------------

def _expand_removed_paths(
    source_paths: frozenset[str], removed_entries: Iterable[str],
) -> frozenset[str]:
    """Expand each top-level entry ``apply_exclusions`` reported as removed
    into the full set of SOURCE paths it accounts for.

    A directory entry (or a resolved glob match that is itself a directory)
    expands to every source path nested under it; a file entry matches only
    itself. Re-derives from ``source_paths`` rather than re-implementing
    ``apply_exclusions``'s glob/dir matching, so the two can never drift
    apart silently.
    """
    expanded: set[str] = set()
    for entry in removed_entries:
        prefix = entry + "/"
        nested = {p for p in source_paths if p.startswith(prefix)}
        if nested:
            expanded.update(nested)
        elif entry in source_paths:
            expanded.add(entry)
    return frozenset(expanded)


def tree_closure_gate(
    *,
    tree_sha: str,
    tree_ish: str,
    repo_root: Path,
    removed_entries: Sequence[str],
    gitlink_excluded: Iterable[str],
    graft_paths: Sequence[str],
) -> list[str]:
    """Verify the EMITTED TREE OBJECT's path inventory equals
    ``(source inventory - path excludes - gitlink excludes) | graft paths``.

    This is independently re-derived from the transform's own stated intent
    (the exclude list, the gitlink-exclude list, the graft list) and compared
    against ``git ls-tree`` of the tree ``write_tree`` actually wrote -- never
    against the scratch directory's filesystem state. A file can exist on
    disk in the materialized tree and still be silently absent from the
    emitted tree object (e.g. a repo-wide ``.gitignore`` rule quietly
    excluding it from ``git add``); this gate is the one that catches that
    class of defect. Patches change content, not paths, so plain set
    arithmetic is sufficient. Returns a list of delta lines (missing/extra
    paths); empty means the gate passes.
    """
    source_paths = _source_paths(tree_ish, repo_root)
    expected = set(source_paths)
    expected -= _expand_removed_paths(source_paths, removed_entries)
    expected -= set(gitlink_excluded)
    expected |= set(graft_paths)

    ls = _run(["git", "ls-tree", "-r", "--name-only", tree_sha], cwd=repo_root)
    if ls.returncode != 0:
        raise TransformError(
            f"closure gate: git ls-tree {tree_sha!r} failed: {ls.stderr.strip()}"
        )
    emitted = frozenset(ls.stdout.splitlines())

    lines: list[str] = []
    for p in sorted(expected - emitted):
        lines.append(f"MISSING from emitted tree: {p}")
    for p in sorted(emitted - expected):
        lines.append(f"UNEXPECTED in emitted tree: {p}")
    return lines


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def build_overlay(
    tree_ish: str,
    *,
    repo_root: Path = REPO_ROOT,
    exclude_path: Path = DEFAULT_EXCLUDE,
    patches_dir: Path = DEFAULT_PATCHES,
    gitlink_exclude_path: Path = DEFAULT_GITLINK_EXCLUDE,
    graft_from: str | None = None,
    graft_list_path: Path = DEFAULT_GRAFT_LIST,
    scratch: Path | None = None,
    run_import_gate: bool = True,
    run_grep_gate: bool = True,
    run_dangling_gate: bool = True,
    grep_pattern: re.Pattern = FAMILY_PATTERN,
    allow_prefixes: Sequence[str] = DEFAULT_ALLOW_PREFIXES,
    allow_files=DEFAULT_ALLOW_FILES,
    import_targets: Sequence[str] = DEFAULT_IMPORT_TARGETS,
    internal_top: Sequence[str] = INTERNAL_TOP_PACKAGES,
) -> str:
    """Run the full transform and return the emitted tree SHA."""
    owns_scratch = scratch is None
    if scratch is None:
        scratch = Path(tempfile.mkdtemp(prefix="main-overlay-"))
    tree = scratch / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    try:
        _log(f"[materialize] {tree_ish} -> {tree}")
        materialize(tree_ish, tree, repo_root)

        entries = read_path_list(exclude_path)
        removed = apply_exclusions(tree, entries)
        _log(f"[exclude] removed {len(removed)} path(s)")

        applied = apply_patches(tree, patches_dir)
        _log(f"[patch] applied {len(applied)} patch(es): {', '.join(applied) or '(none)'}")

        grafted: list[str] = []
        if graft_from is not None:
            grafted = apply_graft(tree, graft_from, graft_list_path, repo_root)
            _log(f"[graft] applied {len(grafted)} path(s) from {graft_from}: "
                 f"{', '.join(grafted) or '(none)'}")

        all_gitlinks = list_gitlinks(tree_ish, repo_root)
        gitlink_excluded = (
            set(read_path_list(gitlink_exclude_path))
            if gitlink_exclude_path.exists() else set()
        )
        gitlinks = [(p, s) for (p, s) in all_gitlinks if p not in gitlink_excluded]
        _log(
            f"[gitlink] found {len(all_gitlinks)} in source, "
            f"excluded {len(all_gitlinks) - len(gitlinks)}, preserving {len(gitlinks)}"
        )

        if run_grep_gate:
            offenders = grep_gate(
                tree, pattern=grep_pattern,
                allow_prefixes=allow_prefixes, allow_files=allow_files,
            )
            if offenders:
                raise TransformError(
                    "GREP GATE failed: family reference survives outside ALLOW in:\n  "
                    + "\n  ".join(offenders)
                )
            _log("[gate] grep: PASS (no family reference outside ALLOW)")

        if run_dangling_gate:
            danglers = dangling_import_gate(
                tree, tree_ish=tree_ish, repo_root=repo_root, internal_top=internal_top,
            )
            if danglers:
                raise TransformError(
                    "DANGLING-IMPORT GATE failed: kept file(s) import module(s) "
                    "the transform removed:\n  " + "\n  ".join(danglers)
                )
            _log("[gate] dangling: PASS (no kept file imports a removed module)")

        if run_import_gate:
            import_gate(tree, import_targets=import_targets)
            _log("[gate] import: PASS (compileall + gate-module imports)")

        sha = write_tree(tree, repo_root, gitlinks=gitlinks)

        closure_delta = tree_closure_gate(
            tree_sha=sha, tree_ish=tree_ish, repo_root=repo_root,
            removed_entries=removed, gitlink_excluded=gitlink_excluded,
            graft_paths=grafted,
        )
        if closure_delta:
            raise TransformError(
                "CLOSURE GATE failed: emitted tree object does not match "
                "(source - excludes) | grafts:\n  " + "\n  ".join(closure_delta)
            )
        _log("[gate] closure: PASS (emitted tree object matches source - excludes | grafts)")

        _log(f"[tree] {sha}")
        return sha
    finally:
        if owns_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("tree_ish", help="Source tree-ish (commit/tag/tree SHA).")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--gitlink-exclude", type=Path, default=DEFAULT_GITLINK_EXCLUDE)
    parser.add_argument("--graft-from", type=str, default=None,
                        help="Tree-ish to graft main-side infra paths from "
                             "(main_overlay_graft.txt); omit for a no-op.")
    parser.add_argument("--graft-list", type=Path, default=DEFAULT_GRAFT_LIST)
    parser.add_argument("--scratch", type=Path, default=None,
                        help="Reuse a scratch dir (default: a fresh temp dir).")
    parser.add_argument("--no-import-gate", action="store_true",
                        help="Skip compileall + import gate (grep/tree only).")
    parser.add_argument("--no-dangling-gate", action="store_true",
                        help="Skip the static dangling-import gate.")
    args = parser.parse_args(argv)

    try:
        sha = build_overlay(
            args.tree_ish,
            repo_root=args.repo_root.resolve(),
            exclude_path=args.exclude,
            patches_dir=args.patches,
            gitlink_exclude_path=args.gitlink_exclude,
            graft_from=args.graft_from,
            graft_list_path=args.graft_list,
            scratch=args.scratch,
            run_import_gate=not args.no_import_gate,
            run_dangling_gate=not args.no_dangling_gate,
        )
    except TransformError as exc:
        _log(f"ERROR: {exc}")
        return 1
    print(sha)  # LAST line of stdout, by contract.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

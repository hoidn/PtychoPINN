"""Unit tests for the deterministic main-overlay transform.

These exercise ``scripts/main_overlay/build_main_overlay.py`` against a tiny
synthetic git repo built inside ``tmp_path`` -- never the real tree. They cover
the four contract properties from the plan (Task 2.3): exclusion applied,
patch-anchor failure is loud, the grep gate catches a planted reference, and
determinism (two runs -> identical tree SHA).

The import gate is disabled here (synthetic files are not importable modules);
it is validated separately against the real tree.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "scripts" / "main_overlay" / "build_main_overlay.py"

_spec = importlib.util.spec_from_file_location("build_main_overlay", _MODULE_PATH)
bmo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bmo)

PATTERN = re.compile(r"FORBIDDEN", re.IGNORECASE)


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
    )
    assert out.returncode == 0, f"git {' '.join(args)} failed: {out.stderr}"
    return out.stdout


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "synthetic_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    for rel, content in files.items():
        fpath = repo / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)
    # -f: some fixtures deliberately commit a path a .gitignore in the SAME
    # commit would otherwise match (a genuinely tracked-but-ignoreable source
    # file); -f is a no-op for every other fixture (nothing else is ignored).
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _tree_files(repo: Path, tree_sha: str) -> set[str]:
    listing = _git(repo, "ls-tree", "-r", "--name-only", tree_sha)
    return set(listing.split())


def _write_exclude(tmp_path: Path, entries: list[str]) -> Path:
    p = tmp_path / "exclude.txt"
    p.write_text("\n".join(entries) + "\n")
    return p


def _write_list(tmp_path: Path, name: str, entries: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(entries) + "\n" if entries else "")
    return p


def _add_gitlink(repo: Path, path: str, sha: str = "a" * 40) -> None:
    """Insert a fake gitlink (submodule) entry into the fixture repo's HEAD,
    without any working-tree content -- mirrors how ``git archive`` sees a
    real submodule path (a 160000 index entry the archive mechanism skips)."""
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},{path}")
    _git(repo, "commit", "-qm", f"add gitlink {path}")


def _make_graft_source(repo: Path, files: dict) -> str:
    """Commit ``files`` (path -> str or bytes content) on a side branch
    (leaving HEAD where ``_make_repo`` left it) and return the new commit
    SHA -- a graft source whose paths are absent from HEAD's materialized
    tree. A ``bytes`` value proves the graft path is binary-safe."""
    original = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(repo, "checkout", "-qb", "graft_source")
    for rel, content in files.items():
        fpath = repo / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            fpath.write_bytes(content)
        else:
            fpath.write_text(content)
    _git(repo, "add", "-A", "-f")  # -f: graft sources may match a .gitignore rule
    _git(repo, "commit", "-qm", "graft source")
    sha = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", original)
    return sha


def _build(
    repo, tmp_path, *, exclude, patches_dir=None, allow_files=frozenset(),
    run_dangling_gate=True, gitlink_exclude=(), graft_from=None, graft_list=(),
):
    if patches_dir is None:
        patches_dir = tmp_path / "empty_patches"
        patches_dir.mkdir(exist_ok=True)
    return bmo.build_overlay(
        "HEAD",
        repo_root=repo,
        exclude_path=_write_exclude(tmp_path, exclude),
        patches_dir=patches_dir,
        gitlink_exclude_path=_write_list(tmp_path, "gitlink_exclude.txt", list(gitlink_exclude)),
        graft_from=graft_from,
        graft_list_path=_write_list(tmp_path, "graft_list.txt", list(graft_list)),
        run_import_gate=False,
        run_dangling_gate=run_dangling_gate,
        grep_pattern=PATTERN,
        allow_prefixes=(),
        allow_files=allow_files,
    )


def test_exclusion_removes_listed_paths(tmp_path):
    repo = _make_repo(tmp_path, {
        "keep.py": "print('clean')\n",
        "drop_me.py": "FORBIDDEN symbol\n",
        "pkg/also_drop.py": "FORBIDDEN too\n",
    })
    sha = _build(repo, tmp_path, exclude=["drop_me.py", "pkg/"])
    files = _tree_files(repo, sha)
    assert "keep.py" in files
    assert "drop_me.py" not in files
    assert "pkg/also_drop.py" not in files


def test_missing_exclude_entry_fails_loudly(tmp_path):
    repo = _make_repo(tmp_path, {"keep.py": "clean\n"})
    with pytest.raises(bmo.TransformError, match="does not exist"):
        _build(repo, tmp_path, exclude=["not_here.py"])


def test_patch_anchor_failure_is_loud(tmp_path):
    repo = _make_repo(tmp_path, {"target.txt": "line one\nline two\n"})
    patches = tmp_path / "patches"
    patches.mkdir()
    # A patch whose context does not match the fixture -> git apply --check fails.
    (patches / "drifted-anchor.patch").write_text(
        "--- a/target.txt\n"
        "+++ b/target.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-context that does not exist\n"
        "+replacement\n"
        " line two\n"
    )
    with pytest.raises(bmo.TransformError, match="drifted-anchor.patch"):
        _build(repo, tmp_path, exclude=[], patches_dir=patches)


def test_valid_patch_applies(tmp_path):
    repo = _make_repo(tmp_path, {"target.txt": "alpha\nbeta\n"})
    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / "rename-beta.patch").write_text(
        "--- a/target.txt\n"
        "+++ b/target.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " alpha\n"
        "-beta\n"
        "+gamma\n"
    )
    sha = _build(repo, tmp_path, exclude=[], patches_dir=patches)
    blob = _git(repo, "cat-file", "-p", f"{sha}:target.txt")
    assert "gamma" in blob and "beta" not in blob


def test_grep_gate_catches_planted_reference(tmp_path):
    repo = _make_repo(tmp_path, {
        "clean.py": "ok\n",
        "leaks.py": "FORBIDDEN reference left behind\n",
    })
    # leaks.py is neither excluded nor allowed -> the grep gate must fire.
    with pytest.raises(bmo.TransformError, match="GREP GATE"):
        _build(repo, tmp_path, exclude=[])


def test_grep_gate_respects_allow_list(tmp_path):
    repo = _make_repo(tmp_path, {
        "clean.py": "ok\n",
        "record.md": "FORBIDDEN mention permitted as prose\n",
    })
    sha = _build(repo, tmp_path, exclude=[], allow_files=frozenset({"record.md"}))
    assert "record.md" in _tree_files(repo, sha)


def test_dangling_import_gate_catches_importer_of_removed_module(tmp_path):
    """A kept file importing an EXCLUDED repo-internal module must be caught.

    This is the class of breakage a dangling ``import`` produces: valid syntax
    (compiles fine) and no family word (grep passes), yet the module is gone.
    ``scripts.*`` is one of the gate's repo-internal top packages.
    """
    repo = _make_repo(tmp_path, {
        "scripts/studies/dropme.py": "VALUE = 1\n",
        "scripts/studies/user.py": "from scripts.studies.dropme import VALUE\n",
        "keep.py": "ok\n",
    })
    # Exclude the module but keep its importer -> dangling import in the output.
    with pytest.raises(bmo.TransformError, match="DANGLING-IMPORT GATE") as exc:
        _build(repo, tmp_path, exclude=["scripts/studies/dropme.py"])
    message = str(exc.value)
    assert "scripts/studies/user.py:1" in message
    assert "scripts.studies.dropme" in message


def test_dangling_gate_off_lets_broken_importer_through(tmp_path):
    """RED-before/GREEN-after control: the SAME transform emits a tree when the
    dangling gate is disabled, proving the gate (not another stage) is what
    catches the broken importer."""
    repo = _make_repo(tmp_path, {
        "scripts/studies/dropme.py": "VALUE = 1\n",
        "scripts/studies/user.py": "from scripts.studies.dropme import VALUE\n",
    })
    sha = _build(
        repo, tmp_path, exclude=["scripts/studies/dropme.py"],
        run_dangling_gate=False,
    )
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert "scripts/studies/user.py" in _tree_files(repo, sha)


def test_dangling_gate_ignores_preexisting_broken_import(tmp_path):
    """An importer of a module that never existed in the SOURCE tree is a
    pre-existing dangler, out of the transform's scope -- it must NOT fail the
    gate (only modules the transform itself removed are the transform's fault)."""
    repo = _make_repo(tmp_path, {
        "scripts/studies/user.py": "from scripts.studies.neverexisted import X\n",
        "keep.py": "ok\n",
    })
    sha = _build(repo, tmp_path, exclude=[])  # dangling gate on by default
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_gitlink_preserved_in_output_tree(tmp_path):
    """``git archive`` (the materialization mechanism) skips submodules
    entirely; the gitlink-preservation stage must re-insert them."""
    repo = _make_repo(tmp_path, {"keep.py": "ok\n"})
    fake_sha = "b" * 40
    _add_gitlink(repo, "vendor/submod", fake_sha)
    sha = _build(repo, tmp_path, exclude=[])
    listing = _git(repo, "ls-tree", "-r", sha)
    assert f"160000 commit {fake_sha}\tvendor/submod" in listing


def test_gitlink_excluded_when_listed(tmp_path):
    repo = _make_repo(tmp_path, {"keep.py": "ok\n"})
    _add_gitlink(repo, "vendor/keepme", "b" * 40)
    _add_gitlink(repo, "vendor/dropme", "c" * 40)
    sha = _build(repo, tmp_path, exclude=[], gitlink_exclude=["vendor/dropme"])
    listing = _git(repo, "ls-tree", "-r", sha)
    assert "vendor/keepme" in listing
    assert "vendor/dropme" not in listing


def test_graft_writes_missing_file_with_source_content(tmp_path):
    """A path absent from the source tree-ish (main-side CI infra that
    fno-stable never carried) must appear in the output, with the graft
    source's content."""
    repo = _make_repo(tmp_path, {"keep.py": "ok\n"})
    graft_sha = _make_graft_source(repo, {"ci/infra.txt": "graft source content\n"})
    sha = _build(
        repo, tmp_path, exclude=[], graft_from=graft_sha, graft_list=["ci/infra.txt"],
    )
    blob = _git(repo, "cat-file", "-p", f"{sha}:ci/infra.txt")
    assert blob == "graft source content\n"


def test_graft_missing_path_fails_loudly(tmp_path):
    repo = _make_repo(tmp_path, {"keep.py": "ok\n"})
    graft_sha = _make_graft_source(repo, {"ci/infra.txt": "present\n"})
    with pytest.raises(bmo.TransformError, match="GRAFT FAILURE"):
        _build(
            repo, tmp_path, exclude=[], graft_from=graft_sha,
            graft_list=["ci/does_not_exist.txt"],
        )


def test_graft_is_binary_safe(tmp_path):
    """A grafted blob with non-UTF8 bytes (e.g. an .npz-like binary payload)
    must round-trip byte-for-byte -- text-mode decoding would corrupt or
    reject bytes like these."""
    payload = bytes(range(256)) * 200  # 51200 bytes, every byte value present
    repo = _make_repo(tmp_path, {"keep.py": "ok\n"})
    graft_sha = _make_graft_source(repo, {"data/blob.bin": payload})
    sha = _build(
        repo, tmp_path, exclude=[], graft_from=graft_sha, graft_list=["data/blob.bin"],
    )
    # Read back through git's own plumbing in binary mode for an exact check
    # (text mode would raise/corrupt on arbitrary byte values).
    out = subprocess.run(
        ["git", "cat-file", "-p", f"{sha}:data/blob.bin"], cwd=str(repo),
        capture_output=True,
    )
    assert out.returncode == 0
    assert out.stdout == payload


def test_graft_is_noop_when_graft_from_omitted(tmp_path):
    repo = _make_repo(tmp_path, {"keep.py": "ok\n"})
    # graft_list references a path that doesn't exist anywhere -- with
    # graft_from omitted (the default), the stage must not even look for it.
    sha = _build(repo, tmp_path, exclude=[], graft_list=["ci/does_not_exist.txt"])
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_gitignored_but_tracked_source_file_survives_into_tree(tmp_path):
    """A path that is TRACKED in the source commit but matched by the
    materialized tree's OWN .gitignore must still appear in the emitted tree
    object. Plain ``git add -A`` (no ``-f``) respects .gitignore and would
    silently drop it -- present on disk, absent from the tree -- even though
    it is exactly as intentional as any other kept file."""
    repo = _make_repo(tmp_path, {
        ".gitignore": "*.dat\n",
        "keep.py": "ok\n",
        "important.dat": "must survive\n",
    })
    sha = _build(repo, tmp_path, exclude=[])
    assert "important.dat" in _tree_files(repo, sha)


def test_graft_survives_destination_ignore_rule(tmp_path):
    """A grafted path that happens to match the DESTINATION tree's own
    .gitignore rule must still survive into the emitted tree object --
    grafted content is written straight to disk and is exactly as
    intentional as any other kept or excluded-from-exclusion file."""
    repo = _make_repo(tmp_path, {
        ".gitignore": "*.dat\n",
        "keep.py": "ok\n",
    })
    graft_sha = _make_graft_source(repo, {"data/graft.dat": "graft content\n"})
    sha = _build(
        repo, tmp_path, exclude=[], graft_from=graft_sha, graft_list=["data/graft.dat"],
    )
    assert "data/graft.dat" in _tree_files(repo, sha)


def test_closure_gate_catches_planted_inventory_loss(tmp_path, monkeypatch):
    """Directly prove the closure gate (not another stage) is what catches a
    path present in source intent (tracked, not excluded, not gitlink/graft
    related) but missing from the emitted TREE OBJECT -- monkeypatch
    write_tree to simulate the round-4 defect class (a path silently dropped
    at git-add time despite being present on disk moments before) and confirm
    build_overlay refuses to emit."""
    repo = _make_repo(tmp_path, {"keep.py": "ok\n", "other.py": "ok2\n"})
    real_write_tree = bmo.write_tree

    def _dropping_write_tree(tree, repo_root, *, gitlinks=()):
        (tree / "other.py").unlink()  # simulate a silent git-add drop
        return real_write_tree(tree, repo_root, gitlinks=gitlinks)

    monkeypatch.setattr(bmo, "write_tree", _dropping_write_tree)
    with pytest.raises(bmo.TransformError, match="CLOSURE GATE") as exc:
        _build(repo, tmp_path, exclude=[])
    assert "other.py" in str(exc.value)


def test_determinism_two_runs_same_tree_sha(tmp_path):
    repo = _make_repo(tmp_path, {
        "keep.py": "clean\n",
        "drop.py": "FORBIDDEN\n",
    })
    sha1 = _build(repo, tmp_path, exclude=["drop.py"])
    sha2 = _build(repo, tmp_path, exclude=["drop.py"])
    assert sha1 == sha2
    assert re.fullmatch(r"[0-9a-f]{40}", sha1)


def test_determinism_with_gitlinks_and_graft_active(tmp_path):
    repo = _make_repo(tmp_path, {"keep.py": "clean\n", "drop.py": "FORBIDDEN\n"})
    _add_gitlink(repo, "vendor/submod", "d" * 40)
    graft_sha = _make_graft_source(repo, {"ci/infra.txt": "infra\n"})
    kwargs = dict(exclude=["drop.py"], graft_from=graft_sha, graft_list=["ci/infra.txt"])
    sha1 = _build(repo, tmp_path, **kwargs)
    sha2 = _build(repo, tmp_path, **kwargs)
    assert sha1 == sha2
    assert re.fullmatch(r"[0-9a-f]{40}", sha1)

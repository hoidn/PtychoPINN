"""Boundary ratchets for the Phase 0 dead-code-and-guards consolidation.

Two self-retiring drift guards pin the architecture boundary between the
PyTorch backend (``ptycho_torch``) and the TensorFlow baseline (``ptycho``):

* Ratchet A freezes the set of ``ptycho_torch`` modules whose *cold import*
  eagerly loads TensorFlow.  Today that set is empty (``ptycho.raw_data`` was
  de-tainted in Phase 1).  Any new eager taint fails the test until the import
  is re-routed through lazy/deferred loading.
* Ratchet B freezes, per source file, the *count* of ``dill`` usage sites and
  ``torch.load(..., weights_only=False)`` sites in ``ptycho_torch``.  Counting
  per file (not per line) means benign line drift from later phases does not
  trip the ratchet, while a new site in an allowlisted file (count grows) or a
  removed site (count shrinks) still forces an explicit allowlist update.

Both allowlists may only shrink.  Per the roadmap completion clause, when an
allowlist becomes empty the corresponding assertion becomes a hard "nothing
tainted / no sites" check and the allowlist machinery is deleted.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PTYCHO_TORCH_DIR = REPO_ROOT / "ptycho_torch"


# ---------------------------------------------------------------------------
# Ratchet A: eager TensorFlow import taint
# ---------------------------------------------------------------------------

# Modules whose cold import currently loads TensorFlow.
TF_TAINTED_ALLOWLIST = frozenset()

# Script-style modules that execute on import and cannot be imported as plain
# modules (cold-import probe would fail for reasons unrelated to TF taint).
_PROBE_SKIP_MODULES = frozenset()

_PROBE_SCRIPT = """\
import json
import sys
import importlib

importlib.import_module(sys.argv[1])
print(json.dumps({"tainted": "tensorflow" in sys.modules}))
"""


def _probe_modules() -> list[str]:
    """Every importable ``ptycho_torch`` module, minus the script skip-set."""
    modules: list[str] = []
    for path in sorted(PTYCHO_TORCH_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        module = path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", ".")
        if module in _PROBE_SKIP_MODULES:
            continue
        modules.append(module)
    return modules


def _probe_taint(module: str) -> tuple[str, bool]:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, module],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"cold import of {module!r} failed (rc={completed.returncode}):\n"
            f"{completed.stderr}"
        )
    return module, json.loads(completed.stdout)["tainted"]


def test_tf_eager_import_taint_matches_allowlist() -> None:
    modules = _probe_modules()
    with ThreadPoolExecutor(max_workers=8) as pool:
        tainted = frozenset(
            module
            for module, is_tainted in pool.map(_probe_taint, modules)
            if is_tainted
        )

    new_taint = tainted - TF_TAINTED_ALLOWLIST
    assert not new_taint, (
        "New TF-tainted module(s) on cold import: "
        f"{sorted(new_taint)}. Re-route the import through lazy/deferred "
        "loading instead of growing the allowlist."
    )
    stale = TF_TAINTED_ALLOWLIST - tainted
    assert not stale, (
        "Allowlist entry/entries no longer TF-tainted: "
        f"{sorted(stale)}. Remove them from TF_TAINTED_ALLOWLIST "
        "(allowlist may only shrink)."
    )


_TF_FREE_ENTRYPOINTS = (
    "ptycho_torch.workflows.components",
    "ptycho_torch.inference",
    "ptycho_torch.train",
)


def test_torch_entrypoints_import_tf_free() -> None:
    """The torch orchestration/train/infer entrypoints must not load TF cold.

    ``torch`` itself loads eagerly — that is expected and asserted separately
    by ``tests/torch/test_ptycho_torch_lazy_import.py``; here we only pin that
    ``tensorflow`` stays out of ``sys.modules``.
    """
    for module in _TF_FREE_ENTRYPOINTS:
        name, is_tainted = _probe_taint(module)
        assert not is_tainted, f"{name} loads TensorFlow on cold import"


_BLOCK_TF_SCRIPT = """\
import sys


class _BlockTensorFlow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorflow" or fullname.startswith("tensorflow."):
            raise ImportError("tensorflow blocked for the TF-free import lock")
        return None


sys.meta_path.insert(0, _BlockTensorFlow())
import ptycho_torch  # noqa: E402
assert "tensorflow" not in sys.modules, "ptycho_torch pulled tensorflow in"
"""


def test_torch_package_imports_without_tf() -> None:
    """The ``ptycho_torch`` package cold-imports with TensorFlow blocked.

    Guards the package-level lazy surface (``_LAZY_EXPORTS`` + ``__getattr__``):
    importing ``ptycho_torch`` itself must never require TensorFlow, even when
    every ``tensorflow`` submodule import is made to fail.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCK_TF_SCRIPT],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        "ptycho_torch package failed to cold-import with TensorFlow blocked:\n"
        f"{completed.stderr}"
    )


# ---------------------------------------------------------------------------
# Ratchet B: serialization sites (per-file counts)
# ---------------------------------------------------------------------------

_DILL_KIND = "dill"
_WEIGHTS_ONLY_KIND = "weights_only_false"

# Per-file count of dill usage lines and weights_only=False lines.  Counts are
# positive-only; a file absent from the dict has zero sites of every kind.
_SERIALIZATION_ALLOWLIST: dict[str, dict[str, int]] = {}

_DILL_IMPORT_RE = re.compile(r"^\s*(import dill\b|from dill import\b)")
_DILL_CALL_RE = re.compile(r"\bdill\.(loads|dumps|dump|load)\b")
_WEIGHTS_ONLY_RE = re.compile(r"weights_only\s*=\s*False")


def _scan_serialization_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for path in PTYCHO_TORCH_DIR.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        dill_lines = 0
        weights_only_lines = 0
        for line in path.read_text().splitlines():
            if _DILL_IMPORT_RE.search(line) or _DILL_CALL_RE.search(line):
                dill_lines += 1
            if _WEIGHTS_ONLY_RE.search(line):
                weights_only_lines += 1
        entry: dict[str, int] = {}
        if dill_lines:
            entry[_DILL_KIND] = dill_lines
        if weights_only_lines:
            entry[_WEIGHTS_ONLY_KIND] = weights_only_lines
        if entry:
            counts[rel] = entry
    return counts


def test_serialization_sites_match_allowlist() -> None:
    scanned = _scan_serialization_counts()

    grown: list[str] = []
    for file, kinds in scanned.items():
        allowed = _SERIALIZATION_ALLOWLIST.get(file, {})
        for kind, count in kinds.items():
            limit = allowed.get(kind, 0)
            if count > limit:
                grown.append(
                    f"{file} {kind}: now {count}, allowlist {limit}"
                )
    assert not grown, (
        "New dill / weights_only=False site(s) in ptycho_torch: "
        f"{sorted(grown)}. Re-route serialization instead of growing the "
        "allowlist."
    )

    shrunk: list[str] = []
    for file, kinds in _SERIALIZATION_ALLOWLIST.items():
        present = scanned.get(file, {})
        for kind, count in kinds.items():
            if present.get(kind, 0) < count:
                shrunk.append(
                    f"{file} {kind}: now {present.get(kind, 0)}, "
                    f"allowlist {count}"
                )
    assert not shrunk, (
        "Allowlist site(s) removed or reduced: "
        f"{sorted(shrunk)}. Shrink the counts in _SERIALIZATION_ALLOWLIST "
        "(allowlist may only shrink)."
    )


# ---------------------------------------------------------------------------
# Ratchet C: from-disk module construction sites (load-path ratchet)
# ---------------------------------------------------------------------------

_PTYCHOPINN_KIND = "PtychoPINN_Lightning"
_LOAD_FROM_CHECKPOINT_KIND = "load_from_checkpoint"

# From-disk module-construction sites in ``ptycho_torch`` production files plus
# the one out-of-package migrator.  ``bundle_io.py`` routes its construction to
# ``application_factory.build_ptychopinn_application`` (hence
# ``application_factory.py`` here); the API loaders and ``lightning_utils``
# construct via ``load_from_checkpoint``; the migrator writes the current era by
# direct ``PtychoPINN_Lightning`` construction.  Exactly this set, no more, no
# less — a new or removed construction site fails the test.
_CONSTRUCTION_SITE_ALLOWLIST: dict[str, frozenset[str]] = {
    "ptycho_torch/application_factory.py": frozenset({_PTYCHOPINN_KIND}),
    "ptycho_torch/lightning_utils.py": frozenset({_LOAD_FROM_CHECKPOINT_KIND}),
    "ptycho_torch/api/base_api.py": frozenset({_LOAD_FROM_CHECKPOINT_KIND}),
    "ptycho_torch/api/mlflow_utils.py": frozenset({_LOAD_FROM_CHECKPOINT_KIND}),
    "scripts/migrate_legacy_bundle.py": frozenset({_PTYCHOPINN_KIND}),
}


def _scan_construction_sites() -> dict[str, frozenset[str]]:
    """Calls (not definitions) of the two module-construction entry points."""
    sites: dict[str, frozenset[str]] = {}
    paths = [*PTYCHO_TORCH_DIR.rglob("*.py")]
    paths.append(REPO_ROOT / "scripts" / "migrate_legacy_bundle.py")
    for path in paths:
        tree = ast.parse(path.read_text())
        kinds: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "PtychoPINN_Lightning":
                kinds.add(_PTYCHOPINN_KIND)
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "load_from_checkpoint"
            ):
                kinds.add(_LOAD_FROM_CHECKPOINT_KIND)
        if kinds:
            sites[path.relative_to(REPO_ROOT).as_posix()] = frozenset(kinds)
    return sites


def test_construction_sites_match_allowlist() -> None:
    scanned = _scan_construction_sites()

    grown: list[str] = []
    for file, kinds in scanned.items():
        extra = kinds - _CONSTRUCTION_SITE_ALLOWLIST.get(file, frozenset())
        if extra:
            grown.append(f"{file}: new {sorted(extra)}")
    assert not grown, (
        "New module-construction site(s) in ptycho_torch / migrator: "
        f"{sorted(grown)}. Route construction through "
        "ptycho_torch.application_factory.build_ptychopinn_application or "
        "checkpoint decode instead of growing the allowlist."
    )

    shrunk: list[str] = []
    for file, kinds in _CONSTRUCTION_SITE_ALLOWLIST.items():
        missing = kinds - scanned.get(file, frozenset())
        if missing:
            shrunk.append(f"{file}: removed {sorted(missing)}")
    assert not shrunk, (
        "Allowlisted construction site(s) removed: "
        f"{sorted(shrunk)}. Shrink _CONSTRUCTION_SITE_ALLOWLIST "
        "(allowlist may only shrink)."
    )

    gone = set(_CONSTRUCTION_SITE_ALLOWLIST) - set(scanned)
    assert not gone, (
        "Allowlisted file(s) no longer construct a module: "
        f"{sorted(gone)}. Remove them from _CONSTRUCTION_SITE_ALLOWLIST."
    )


# ---------------------------------------------------------------------------
# Ratchet C: params.cfg direct-write containment (W3.1)
# ---------------------------------------------------------------------------

# Direct dict writes bypass ``params.set()`` and therefore the seal machinery
# (``params._SEAL_WHITELIST`` warnings).  Every production write must either go
# through ``params.set()`` / ``update_legacy_dict`` or live in a recognized
# restore boundary below.  The allowlist may only shrink.
_DIRECT_WRITE_RE = re.compile(
    r"(?:params\.cfg|(?<![\w.])p\.cfg)\s*\[[^\]]+\]\s*=(?!=)"
    r"|(?:params\.cfg|(?<![\w.])p\.cfg)\s*\.update\("
    r"|(?<![\w.])cfg\s*\[[^\]]+\]\s*=(?!=)"
)

# file -> reason it may hold direct writes
PARAMS_WRITE_BOUNDARIES = {
    "ptycho/params.py": "owner module: set()/ensure_defaults internals",
    "ptycho/config/config.py": "update_legacy_dict projection internals",
    "ptycho/config/legacy_state.py": "scope save/restore internals",
    "ptycho/model.py": "frozen (CLAUDE.md directive 3): internal save/restore juggling",
    "ptycho/model_manager.py": "named archive-restore boundary (dill-era load)",
    "ptycho/workflows/bundle_loading.py": "named archive-restore boundary (adapter runtime params + gridsize sync)",
    "ptycho_torch/workflows/bundle_io.py": "spec-pinned CONFIG-001 bundle-load side effect",
    "ptycho/data_preprocessing.py": "deferred TF-only module (strangler Tranche-F); migrates under a future plan",
}


def _tracked_production_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "ptycho", "ptycho_torch", "scripts"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [f for f in out if f.endswith(".py")]


def test_params_cfg_direct_writes_are_allowlisted():
    offenders: dict[str, list[str]] = {}
    for rel in _tracked_production_files():
        text = (REPO_ROOT / rel).read_text()
        hits = []
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", ">>>", '"', "'")):
                continue
            if _DIRECT_WRITE_RE.search(line):
                hits.append(f"{rel}:{lineno}: {stripped[:80]}")
        if hits and rel not in PARAMS_WRITE_BOUNDARIES:
            offenders[rel] = hits
    assert not offenders, (
        "direct params.cfg write outside recognized boundaries — route through "
        "params.set() (seal-visible) or a named restore boundary:\n"
        + "\n".join(h for hs in offenders.values() for h in hs)
    )


def test_params_write_boundaries_are_live():
    stale = [f for f in PARAMS_WRITE_BOUNDARIES if not (REPO_ROOT / f).exists()]
    assert not stale, f"stale write-boundary entries: {stale}"


# ---------------------------------------------------------------------------
# Ratchet D: raw_data / loader params-free (W3.3 tranche 1)
# ---------------------------------------------------------------------------

# These data-path modules must not import ``ptycho.params``.  Their
# ``params.cfg`` fallback reads were removed in W3.3 tranche 1 and replaced with
# explicit arguments (loud ``ValueError`` when a caller omits them).  A new
# import here re-introduces the global bridge on the data path.
PARAMS_FREE_MODULES = (
    "ptycho/raw_data.py",
    "ptycho/loader.py",
)


def _params_import_offenders(module_ast: ast.AST) -> list[str]:
    """Import statements referencing ``ptycho.params`` (any spelling)."""
    offenders: list[str] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ptycho.params" or alias.name.endswith(".params"):
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            if module == "ptycho.params" or module.endswith(".params"):
                offenders.append(f"from {module} import ...")
            elif module == "params" and node.level >= 1:
                offenders.append("from .params import ...")
            elif "params" in names and (module == "ptycho" or (node.level >= 1 and not module)):
                offenders.append(f"from {module or '.'} import params")
    return offenders


def test_raw_data_and_loader_are_params_free():
    for rel in PARAMS_FREE_MODULES:
        text = (REPO_ROOT / rel).read_text()
        tree = ast.parse(text, filename=rel)
        offenders = _params_import_offenders(tree)
        assert not offenders, (
            f"{rel} imports ptycho.params ({offenders}); the params.cfg "
            "fallback was removed in W3.3 tranche 1 — pass explicit arguments."
        )
        # Strictly stronger pin: no spelling of "params" may remain (imports,
        # reads, or prose).  A legitimate future mention forces a conscious
        # ratchet revision here.
        assert "params" not in text, (
            f"{rel} still mentions 'params'; it must be params-free "
            "(imports, reads, and prose)."
        )


# ---------------------------------------------------------------------------
# Ratchet E: torch reconstruction path params.cfg-bridge-free (W3.4)
# ---------------------------------------------------------------------------

# The double-scoped adapter (``_reassemble_position_with_legacy_geometry``)
# that projected ``config`` into ``params.cfg`` so the frozen
# ``tf_helper.shift_and_sum`` could read its detector size was deleted in W3.4.
# ``reassemble_position`` now derives the size from its patch tensor and passes
# it explicitly; ``legacy.py`` must not re-introduce the legacy bridge.
TORCH_PARAMS_BRIDGE_FREE_MODULES = (
    "ptycho_torch/workflows/legacy.py",
)

_TORCH_PARAMS_BRIDGE_MARKERS = (
    "from ptycho import params",
    "import ptycho.params",
    "update_legacy_dict",
    "legacy_state",
    "legacy_params",
    "configured_params_scope",
)


def test_torch_reassembly_workflow_is_params_bridge_free():
    for rel in TORCH_PARAMS_BRIDGE_FREE_MODULES:
        text = (REPO_ROOT / rel).read_text()
        hits = [m for m in _TORCH_PARAMS_BRIDGE_MARKERS if m in text]
        assert not hits, (
            f"{rel} re-introduces the legacy params.cfg bridge ({hits}); "
            "W3.4 removed the double-scoped adapter — pass explicit geometry."
        )

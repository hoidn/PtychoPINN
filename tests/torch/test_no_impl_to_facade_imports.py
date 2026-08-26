"""Import-graph lint for the Phase 1 facade consolidation.

Two AST checks, no execution:

* Implementation modules must not import the ``components`` facades (directly or
  through the ``ptycho_torch.workflows`` / ``ptycho.workflows`` package proxies).
  Impl modules may only import their owned modules or siblings (relative imports)
  and the shared non-facade packages.
* The facades themselves must be pure re-export shims: a docstring, relative
  re-export blocks, and an ``__all__`` assignment — no functions, classes,
  try/except, calls, or eager (absolute) imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two explicit facade modules. Package initializers do not re-export them.
FACADE = frozenset(
    {
        "ptycho_torch.workflows.components",
        "ptycho_torch.workflows",
        "ptycho.workflows.components",
    }
)

# Every implementation module that lives behind a facade.  model_manager.py is
# included deliberately: its only facade reference is a docstring example, which
# the AST walk must ignore.
IMPL_MODULES = [
    "ptycho_torch.workflows.bundle_io",
    "ptycho_torch.workflows.containers",
    "ptycho_torch.workflows.dataloaders",
    "ptycho_torch.workflows.legacy",
    "ptycho_torch.workflows.lightning_service",
    "ptycho_torch.workflows.rect_s1s2",
    "ptycho_torch.batch_emission",
    "ptycho_torch.checkpoint_decode",
    "ptycho_torch.collate",
    "ptycho_torch.inference",
    "ptycho_torch.inference_validation",
    "ptycho_torch.train",
    "ptycho_torch.model_manager",
    "ptycho_torch.model_blocks",
    "ptycho_torch.reassembly_accumulators",
    "ptycho_torch.varpro",
    "ptycho.workflows.workflow_orchestration",
    "ptycho.workflows.bundle_loading",
    "ptycho.workflows.config_cli",
    "ptycho.workflows.backend_selector",
]

FACADE_FILES = [
    REPO_ROOT / "ptycho_torch" / "workflows" / "components.py",
    REPO_ROOT / "ptycho" / "workflows" / "components.py",
]


def _module_path(module: str) -> Path:
    return REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")


def _imported_roots(node: ast.AST) -> list[str]:
    """The dotted module path(s) imported by one Import/ImportFrom node.

    Relative imports (level > 0) are sibling imports and return no roots.
    """
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return []
        return [node.module] if node.module else []
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def test_impl_modules_do_not_import_facades() -> None:
    violations: list[str] = []
    for module in IMPL_MODULES:
        path = _module_path(module)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for root in _imported_roots(node):
                if root in FACADE:
                    violations.append(f"{module}:{node.lineno} imports {root!r}")
    assert not violations, (
        "Implementation module(s) import a facade or the workflows package "
        "proxy:\n" + "\n".join(sorted(violations))
    )


def test_facades_are_logic_free() -> None:
    violations: list[str] = []
    for path in FACADE_FILES:
        tree = ast.parse(path.read_text())
        for statement in tree.body:
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ) and isinstance(statement.value.value, str):
                continue  # module docstring
            if isinstance(statement, ast.ImportFrom) and statement.level > 0:
                continue  # relative re-export block
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in statement.targets
            ):
                continue  # __all__ = [...]
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{statement.lineno} "
                f"{type(statement).__name__}"
            )
    assert not violations, (
        "Facade(s) contain logic beyond docstring + re-export blocks + __all__:\n"
        + "\n".join(sorted(violations))
    )


def test_displaced_training_modules_are_deleted() -> None:
    for relative in (
        "ptycho/workflows/training.py",
        "ptycho_torch/workflows/orchestration.py",
        "ptycho_torch/data_adapter.py",
    ):
        assert not REPO_ROOT.joinpath(relative).exists(), relative

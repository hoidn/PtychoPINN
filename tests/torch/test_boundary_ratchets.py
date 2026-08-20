"""Self-retiring boundary ratchets: no dead API modules, no TF import in torch."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RETIRED_API_MODULES = ("example_predict_lightning.py", "trainer_api.py")


def test_retired_api_modules_are_absent():
    present = [
        name
        for name in RETIRED_API_MODULES
        if (REPO_ROOT / "ptycho_torch" / "api" / name).exists()
    ]
    assert not present, f"retired API modules still present: {present}"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_torch_tree_has_no_module_level_tensorflow_import():
    offenders = []
    for path in sorted((REPO_ROOT / "ptycho_torch").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if any(name.split(".")[0] == "tensorflow" for name in _module_imports(path)):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, f"module-level tensorflow import in torch tree: {offenders}"


def test_torch_workflow_import_does_not_load_tensorflow():
    code = (
        "import sys; import ptycho_torch.workflows.components as m; "
        "assert m is not None; "
        "sys.exit(1 if 'tensorflow' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "importing the torch workflow facade pulled in tensorflow\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_torch_tree_has_no_dill_and_no_unsafe_torch_load():
    offenders = []
    for path in sorted((REPO_ROOT / "ptycho_torch").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        if "import dill" in text or "weights_only=False" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, f"dill or unsafe torch.load in torch tree: {offenders}"

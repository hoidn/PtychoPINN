"""Utility for generating a Markdown index of the test suite."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

# Manual overrides for high-value files that need richer context than docstrings.
MANUAL_OVERRIDES: Dict[str, Dict[str, str]] = {
    "tests/test_integration_workflow.py": {
        "purpose": (
            "Validates the full train → save → load → infer workflow using"
            " subprocesses, ensuring model artifacts persist across CLI entrypoints."
        ),
        "notes": "Critical integration coverage for TensorFlow persistence.",
    },
    "tests/test_raw_data_grouping.py": {
        "purpose": (
            "Exhaustively checks the RawData sample-then-group implementation,"
            " covering shapes, bounds, locality, and oversized sampling requests."
        ),
    },
    "tests/image/test_registration.py": {
        "purpose": (
            "Covers registration alignment helpers, including translation detection,"
            " shift application, and complex-valued data handling."
        ),
    },
}


@dataclass
class TestEntry:
    """Represents a single test module for the index."""

    rel_path: Path
    purpose: str
    key_tests: str
    command: str
    notes: str

    @property
    def display_name(self) -> str:
        return f"`{self.rel_path.name}`"


def get_module_docstring(filepath: Path) -> str:
    """Extract the module-level docstring or provide a placeholder."""
    with open(filepath, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    doc = ast.get_docstring(tree)
    return doc.strip() if doc else "No module docstring found."


def get_test_functions(filepath: Path) -> str:
    """Extract names of tests (functions or methods) starting with ``test_``."""
    with open(filepath, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(f"`{node.name}`")

    names.sort()
    return ", ".join(names) if names else "N/A"


def _section_title(rel_parent: Path) -> str:
    """Create a human-friendly section title from a path relative to tests/."""
    if rel_parent == Path('.'):
        return "Core Library Tests (`tests/`)"
    parts = [p.capitalize().replace('_', ' ') for p in rel_parent.parts]
    return f"{'/'.join(parts)} Tests (`tests/{rel_parent.as_posix()}/`)"


def _module_command(rel_path: Path) -> str:
    module = rel_path.with_suffix('')
    dotted = ".".join((Path('tests') / module).parts)
    return f"python -m unittest {dotted}"


def _collect_entries() -> Dict[str, List[TestEntry]]:
    entries: Dict[str, List[TestEntry]] = {}

    for root, _, files in os.walk(TESTS_ROOT):
        root_path = Path(root)
        rel_parent = root_path.relative_to(TESTS_ROOT)
        section_key = _section_title(rel_parent)  # type: ignore[arg-type]

        for name in sorted(files):
            if not name.startswith("test_") or not name.endswith(".py"):
                continue
            rel_path = root_path.relative_to(REPO_ROOT) / name
            abs_path = root_path / name

            doc = get_module_docstring(abs_path)
            key_tests = get_test_functions(abs_path)
            command = _module_command((rel_parent / name))

            overrides = MANUAL_OVERRIDES.get(str(rel_path))
            purpose = overrides.get("purpose") if overrides else None
            notes = overrides.get("notes") if overrides else None

            entry = TestEntry(
                rel_path=rel_path,
                purpose=purpose or doc,
                key_tests=key_tests,
                command=command,
                notes=notes or "—",
            )
            entries.setdefault(section_key, []).append(entry)

    # Sort entries within each section by file name
    for item in entries.values():
        item.sort(key=lambda e: e.rel_path.name)
    return entries


def _format_table(entries: Iterable[TestEntry]) -> str:
    header = "| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |\n"
    header += "| :--- | :--- | :--- | :--- | :--- |\n"
    rows = []
    for entry in entries:
        rows.append(
            "| {name} | {purpose} | {key_tests} | `{command}` | {notes} |".format(
                name=entry.display_name,
                purpose=entry.purpose.replace("\n", " ").strip(),
                key_tests=entry.key_tests,
                command=entry.command,
                notes=entry.notes,
            )
        )
    return "\n".join([header] + rows)


def build_index() -> str:
    """Generate the full Markdown document."""
    sections = _collect_entries()

    lines: List[str] = [
        "# PtychoPINN Test Suite Index",
        "",
        "This document provides a comprehensive index of the automated tests in the `tests/` directory. "
        "Its purpose is to make the test suite discoverable, explain the scope of each test module, and provide "
        "direct commands for running specific tests.",
        "",
        "## How to Run Tests",
        "",
        "- **Run all tests:** `python -m unittest discover tests/`",
        "- **Run a specific file:** `python -m unittest tests.image.test_cropping`",
        "- **Run a specific test class:** `python -m unittest tests.test_integration_workflow.TestFullWorkflow`",
        "",
        "---",
        "",
        "## Test Modules",
        "",
    ]

    for section in sorted(sections.keys()):
        lines.append(f"### {section}")
        lines.append("")
        lines.append(_format_table(sections[section]))
        lines.append("")

    lines.extend([
        "---",
        "*This document can be automatically updated. Run ``python scripts/tools/generate_test_index.py`` to regenerate.*",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    print(build_index())


if __name__ == "__main__":
    main()

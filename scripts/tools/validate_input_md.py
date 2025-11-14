#!/usr/bin/env python
"""
Schema validator for ./input.md.

Checks:
- Required header fields at the top:
  - Mode: <TDD|Parity|Perf|Docs|none>
  - Focus: <initiative-id>
  - Selector: <pytest-node or none>
- For Mode != Docs:
  - Presence of a ### Workload Spec section.
  - Only the allowed top-level headings inside Workload Spec:
    ## Goal, ## Contracts, ## Interfaces, ## Pseudocode, ## Tasks, ## Selector, ## Artifacts
  - Every Task bullet under ## Tasks contains a path::symbol reference.
  - Exactly one fenced code block appears under ## Selector (the command block).
  - Every Contracts bullet under ## Contracts points at an existing file path (path:line).

Exit code:
- 0 on success (schema OK)
- 1 on any validation error (messages printed to stderr)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple


ALLOWED_MODES = {"TDD", "Parity", "Perf", "Docs", "none"}
ALLOWED_WS_HEADINGS = {
    "Goal",
    "Contracts",
    "Interfaces",
    "Pseudocode",
    "Tasks",
    "Selector",
    "Artifacts",
}


def load_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        sys.stderr.write(f"ERROR: input.md not found at {path}\n")
        sys.exit(1)


def parse_top_header(lines: List[str]) -> Tuple[str | None, str | None, str | None]:
    mode = focus = selector = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Mode:"):
            mode = stripped.split(":", 1)[1].strip() or None
        elif stripped.startswith("Focus:"):
            focus = stripped.split(":", 1)[1].strip() or None
        elif stripped.startswith("Selector:"):
            selector = stripped.split(":", 1)[1].strip() or None
        # Stop scanning once we hit an empty line after having seen all three
        if not stripped and mode and focus and selector:
            break
    return mode, focus, selector


def find_line_index(lines: List[str], prefix: str) -> int | None:
    for idx, line in enumerate(lines):
        if line.startswith(prefix):
            return idx
    return None


def collect_ws_headings(lines: List[str], ws_start: int) -> List[Tuple[str, int]]:
    headings: List[Tuple[str, int]] = []
    # Workload Spec section runs until the next "### " heading or EOF
    for idx in range(ws_start + 1, len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith("### "):
            break
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            headings.append((title, idx))
    return headings


def slice_section(lines: List[str], start_idx: int) -> List[str]:
    """Return lines belonging to a section starting at a '## ' heading."""
    out: List[str] = []
    for idx in range(start_idx + 1, len(lines)):
        stripped = lines[idx].lstrip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            break
        out.append(lines[idx])
    return out


def main(argv: List[str]) -> int:
    root = Path(os.getcwd())
    path = root / (argv[1] if len(argv) > 1 else "input.md")
    lines = load_lines(path)
    errors: List[str] = []

    mode, focus, selector = parse_top_header(lines)
    if mode is None:
        errors.append("Missing required header 'Mode: <TDD|Parity|Perf|Docs|none>' near top of input.md")
    elif mode not in ALLOWED_MODES:
        errors.append(f"Invalid Mode '{mode}'. Expected one of {sorted(ALLOWED_MODES)}")

    if focus is None:
        errors.append("Missing required header 'Focus: <initiative-id>' near top of input.md")

    if selector is None:
        errors.append("Missing required header 'Selector: <pytest-node or none>' near top of input.md")
    else:
        sel_norm = selector.strip().lower()
        if mode != "Docs" and sel_norm == "none":
            errors.append(
                "Mode is not 'Docs' but Selector is 'none'; implementation loops must use a concrete pytest node"
            )

    # For Mode == Docs we do not require a Workload Spec
    if mode != "Docs":
        ws_idx = find_line_index(lines, "### Workload Spec")
        if ws_idx is None:
            errors.append("Mode is not 'Docs' but no '### Workload Spec' section was found in input.md")
        else:
            headings = collect_ws_headings(lines, ws_idx)
            seen_titles = {title for title, _ in headings}

            # Check for unexpected headings
            for title, idx in headings:
                if title not in ALLOWED_WS_HEADINGS:
                    errors.append(
                        f"Unexpected heading '## {title}' in Workload Spec at line {idx+1}. "
                        f"Allowed headings: {sorted(ALLOWED_WS_HEADINGS)}"
                    )

            # Ensure core short-form headings appear at least once.
            # For trivial loops, a short-form spec with just Goal/Tasks/Selector is allowed.
            core_required = {"Goal", "Tasks", "Selector"}
            missing_core = [h for h in core_required if h not in seen_titles]
            if missing_core:
                errors.append(
                    "Workload Spec is missing required core headings: "
                    + ", ".join(f"'## {h}'" for h in missing_core)
                )

            # Validate Tasks section
            tasks_idx = next((idx for title, idx in headings if title == "Tasks"), None)
            if tasks_idx is not None:
                task_lines = slice_section(lines, tasks_idx)
                for i, raw in enumerate(task_lines):
                    stripped = raw.lstrip()
                    if stripped.startswith("-"):
                        if "::" not in stripped:
                            errors.append(
                                f"Task bullet under '## Tasks' at line {tasks_idx + 2 + i} "
                                f"must contain a 'path::symbol' reference"
                            )

            # Validate Contracts section
            contracts_idx = next((idx for title, idx in headings if title == "Contracts"), None)
            if contracts_idx is not None:
                contract_lines = slice_section(lines, contracts_idx)
                for i, raw in enumerate(contract_lines):
                    stripped = raw.lstrip()
                    if stripped.startswith("-"):
                        # Expect something like `- path/to/file:line`
                        text = stripped.lstrip("-").strip()
                        if not text:
                            errors.append(
                                f"Empty bullet under '## Contracts' at line {contracts_idx + 2 + i}"
                            )
                            continue
                        path_part = text.split()[0]
                        file_part = path_part.split(":", 1)[0]
                        file_path = root / file_part
                        if not file_path.exists():
                            errors.append(
                                f"'## Contracts' bullet at line {contracts_idx + 2 + i} "
                                f"references missing file '{file_part}'"
                            )

            # Validate Selector section: exactly one fenced code block
            selector_idx = next((idx for title, idx in headings if title == "Selector"), None)
            if selector_idx is not None:
                selector_lines = slice_section(lines, selector_idx)
                code_block_count = 0
                in_block = False
                for raw in selector_lines:
                    stripped = raw.strip()
                    if stripped.startswith("```"):
                        in_block = not in_block
                        if in_block:
                            code_block_count += 1
                if code_block_count == 0:
                    errors.append("Selector section must contain exactly one fenced code block with the command")
                elif code_block_count > 1:
                    errors.append(
                        f"Selector section contains {code_block_count} fenced code blocks; expected exactly one"
                    )

    if errors:
        for msg in errors:
            sys.stderr.write(f"ERROR: {msg}\n")
        return 1

    sys.stdout.write("input.md schema OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/usr/bin/env python
"""CLI wrapper for the PDEBench 128x128 image-suite study."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.pdebench_image128.darcy import main as darcy_main
from scripts.studies.pdebench_image128.preflight import main as preflight_main


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if "--task" in raw:
        task_index = raw.index("--task")
        task_value = raw[task_index + 1] if task_index + 1 < len(raw) else ""
        if task_value == "darcy":
            return darcy_main(raw)
    return preflight_main(raw)


if __name__ == "__main__":
    raise SystemExit(main())

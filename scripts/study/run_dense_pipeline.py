#!/usr/bin/env python3
"""
run_dense_pipeline.py — Neutral entrypoint for running the dense comparison pipeline.

This is a thin wrapper around the existing study runner currently located under
`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py`.

Goals
- Provide a stable, neutral script path under `scripts/` (not tied to plan names).
- Keep CLI flags intact. For convenience, `--output-root` is accepted as an
  alias of the upstream `--hub` flag.

Examples
  python scripts/study/run_dense_pipeline.py \
    --output-root outputs/dense_execution \
    --dose 100000 \
    --view dense \
    --splits train test \
    --clobber

Notes
- This wrapper simply forwards arguments to the upstream runner using the
  current Python interpreter. No behavior changes are introduced here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dense comparison pipeline (neutral entrypoint)",
        add_help=False,
    )
    # Accept both the upstream flag and a neutral alias
    parser.add_argument("--hub", dest="hub", type=str, default=None)
    parser.add_argument("--output-root", dest="output_root", type=str, default=None)
    # Collect the rest of the args verbatim to avoid drift
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    # Build the command to call the upstream runner
    repo_root = Path(__file__).resolve().parents[2]
    upstream = repo_root / (
        "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py"
    )

    if not upstream.exists():
        print(
            f"ERROR: Upstream runner not found at: {upstream}\n"
            f"This wrapper expects the study runner to be available.",
            file=sys.stderr,
        )
        return 2

    forwarded = []  # type: list[str]
    # Prefer explicit --hub; else allow --output-root as an alias
    if ns.hub:
        forwarded.extend(["--hub", ns.hub])
    elif ns.output_root:
        forwarded.extend(["--hub", ns.output_root])

    # Pass through the remainder unchanged
    remainder = ns.args
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    forwarded.extend(remainder)

    cmd = [sys.executable, str(upstream)] + forwarded
    result = subprocess.run(cmd)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

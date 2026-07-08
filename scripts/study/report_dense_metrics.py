#!/usr/bin/env python3
"""
report_dense_metrics.py — Neutral entrypoint for aggregate metrics reporting.

Thin wrapper that invokes the existing study helper located under:
  plans/active/.../bin/report_phase_g_dense_metrics.py

Usage mirrors the upstream script; this wrapper provides a stable path under scripts/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    upstream = repo_root / (
        "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py"
    )
    if not upstream.exists():
        print(f"ERROR: Upstream reporter not found: {upstream}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(upstream)] + sys.argv[1:]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())


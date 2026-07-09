#!/usr/bin/env python3
"""
ssim_grid.py — Neutral entrypoint for SSIM grid summary generation.

Thin wrapper that invokes the existing helper under:
  plans/active/.../bin/ssim_grid.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    upstream = repo_root / (
        "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py"
    )
    if not upstream.exists():
        print(f"ERROR: Upstream SSIM grid helper not found: {upstream}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(upstream)] + sys.argv[1:]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())


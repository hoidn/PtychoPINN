#!/usr/bin/env python3
"""
verify_dense_outputs.py — Neutral entrypoint for pipeline artifact verification.

Thin wrapper that invokes the existing verifier under:
  plans/active/.../bin/verify_dense_pipeline_artifacts.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    upstream = repo_root / (
        "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py"
    )
    if not upstream.exists():
        print(f"ERROR: Upstream verifier not found: {upstream}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(upstream)] + sys.argv[1:]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())


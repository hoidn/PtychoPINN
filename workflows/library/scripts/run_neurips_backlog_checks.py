#!/usr/bin/env python3
"""Run deterministic backlog check commands and emit a durable report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-path", required=True, help="JSON file containing a non-empty list of commands")
    parser.add_argument("--report-path", required=True, help="JSON report output path")
    parser.add_argument("--cwd", default=".", help="Working directory for check execution")
    args = parser.parse_args()

    checks_path = Path(args.checks_path)
    report_path = Path(args.report_path)
    cwd = Path(args.cwd)

    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if not isinstance(checks, list) or not checks or not all(isinstance(cmd, str) and cmd.strip() for cmd in checks):
        raise SystemExit("checks-path must contain a non-empty JSON list of commands")

    results: list[dict[str, object]] = []
    failed_count = 0
    for index, command in enumerate(checks, start=1):
        proc = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
        if proc.returncode != 0:
            failed_count += 1
        results.append(
            {
                "index": index,
                "command": command,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )

    payload = {
        "status": "PASS" if failed_count == 0 else "FAIL",
        "failed_count": failed_count,
        "command_count": len(checks),
        "checks_path": checks_path.as_posix(),
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

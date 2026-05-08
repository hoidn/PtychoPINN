#!/usr/bin/env python3
"""Run deterministic backlog check commands and emit a durable report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _existing_log_paths(report_path: Path, commands: list[str], cwd: Path) -> dict[int, str]:
    """Return preserved log_path fields when an existing report matches commands."""
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(commands):
        return {}
    if [item.get("command") if isinstance(item, dict) else None for item in results] != commands:
        return {}

    log_paths: dict[int, str] = {}
    for item in results:
        if not isinstance(item, dict):
            return {}
        index = item.get("index")
        log_path = item.get("log_path")
        if not isinstance(index, int) or not isinstance(log_path, str) or not log_path.strip():
            continue
        candidate = Path(log_path)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.is_file():
            log_paths[index] = log_path
    return log_paths


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

    preserved_log_paths = _existing_log_paths(report_path, checks, cwd)
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
        if index in preserved_log_paths:
            result = {
                "index": index,
                "command": command,
                "exit_code": proc.returncode,
                "log_path": preserved_log_paths[index],
            }
        else:
            result = {
                "index": index,
                "command": command,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        results.append(result)

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

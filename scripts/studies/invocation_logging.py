"""Helpers for persisting CLI invocation artifacts.

This module centralizes writing invocation metadata so study/orchestration
scripts can emit reproducible command traces in a consistent format.
"""

from __future__ import annotations

import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _to_jsonable(value: Any) -> Any:
    """Convert common CLI/argparse values into JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def build_command_line(script_path: str, argv: Iterable[str]) -> str:
    """Build a copy-paste command string for a script + argv payload."""
    tokens = ["python", script_path, *[str(arg) for arg in argv]]
    return shlex.join(tokens)


def write_invocation_artifacts(
    output_dir: Path,
    script_path: str,
    argv: Iterable[str],
    parsed_args: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Tuple[Path, Path]:
    """Write invocation JSON + shell script artifacts in output_dir.

    Returns:
        Tuple of (invocation_json_path, invocation_shell_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    argv_list = [str(arg) for arg in argv]
    command = build_command_line(script_path, argv_list)
    payload: Dict[str, Any] = {
        "script": str(script_path),
        "argv": argv_list,
        "command": command,
        "parsed_args": _to_jsonable(parsed_args),
        "cwd": str(Path.cwd()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    if extra:
        payload["extra"] = _to_jsonable(extra)

    json_path = output_dir / "invocation.json"
    sh_path = output_dir / "invocation.sh"
    json_path.write_text(json.dumps(payload, indent=2))
    sh_path.write_text(command + "\n")
    return json_path, sh_path


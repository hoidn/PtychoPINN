"""Helpers for persisting CLI invocation artifacts.

This module centralizes writing invocation metadata so study/orchestration
scripts can emit reproducible command traces in a consistent format.
"""

from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
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


def capture_runtime_provenance() -> Dict[str, Any]:
    """Capture the effective Python/runtime import provenance for study scripts."""
    payload: Dict[str, Any] = {
        "python_executable": str(Path(sys.executable).resolve()),
        "cwd": str(Path.cwd()),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    try:
        ptycho_torch = importlib.import_module("ptycho_torch")
    except Exception:
        payload["ptycho_torch_file"] = None
    else:
        payload["ptycho_torch_file"] = str(Path(ptycho_torch.__file__).resolve())
    return payload


def get_git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit for repo_root when available."""
    repo_root = Path(repo_root or Path.cwd())
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _capture_tmux_launcher_metadata() -> Dict[str, str] | None:
    session_name = os.environ.get("CODEX_TMUX_SESSION_NAME", "").strip()
    socket_path = os.environ.get("CODEX_TMUX_SOCKET_PATH", "").strip()
    attach_command = os.environ.get("CODEX_TMUX_ATTACH_COMMAND", "").strip()
    capture_command = os.environ.get("CODEX_TMUX_CAPTURE_COMMAND", "").strip()
    if not session_name:
        return None
    payload = {"session_name": session_name}
    if socket_path:
        payload["socket_path"] = socket_path
    if attach_command:
        payload["attach_command"] = attach_command
    if capture_command:
        payload["capture_command"] = capture_command
    return payload


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
    merged_extra: Dict[str, Any] = {}
    if extra:
        merged_extra.update(_to_jsonable(extra))
    tmux_payload = _capture_tmux_launcher_metadata()
    if tmux_payload:
        existing_tmux = merged_extra.get("tmux")
        if isinstance(existing_tmux, dict):
            merged = dict(existing_tmux)
            for key, value in tmux_payload.items():
                merged.setdefault(key, value)
            merged_extra["tmux"] = merged
        else:
            merged_extra["tmux"] = tmux_payload
    if merged_extra:
        payload["extra"] = merged_extra

    json_path = output_dir / "invocation.json"
    sh_path = output_dir / "invocation.sh"
    json_path.write_text(json.dumps(payload, indent=2))
    sh_path.write_text(command + "\n")
    return json_path, sh_path


def update_invocation_artifacts(
    json_path: Path,
    *,
    extra: Dict[str, Any] | None = None,
    **fields: Any,
) -> Path:
    """Update an existing invocation JSON payload with additional metadata."""
    json_path = Path(json_path)
    payload = json.loads(json_path.read_text())

    if extra:
        merged_extra = payload.get("extra", {})
        if not isinstance(merged_extra, dict):
            merged_extra = {}
        merged_extra.update(_to_jsonable(extra))
        payload["extra"] = merged_extra

    for key, value in fields.items():
        payload[key] = _to_jsonable(value)

    json_path.write_text(json.dumps(payload, indent=2))
    return json_path

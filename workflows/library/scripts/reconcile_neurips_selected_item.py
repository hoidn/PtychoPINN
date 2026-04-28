#!/usr/bin/env python3
"""Reconcile a selected NeurIPS backlog item into in_progress and update its plan."""

from __future__ import annotations

import argparse
from pathlib import Path


def _require_safe_repo_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Unsafe {label}: {path}")
    return path


def _require_backlog_state(path: Path, state: str) -> None:
    expected_prefix = ("docs", "backlog", state)
    if len(path.parts) < 4 or path.parts[:3] != expected_prefix:
        raise SystemExit(f"Expected {state} backlog path, got: {path}")


def _rewrite_plan_path(item_path: Path, plan_path: str) -> None:
    text = item_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SystemExit(f"Missing frontmatter in {item_path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"Missing frontmatter end fence in {item_path}")

    header = text[4:end].splitlines()
    body = text[end + len("\n---\n") :]
    updated = []
    replaced = False
    for line in header:
        if line.startswith("plan_path:"):
            updated.append(f"plan_path: {plan_path}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        raise SystemExit(f"plan_path missing in {item_path}")

    item_path.write_text("---\n" + "\n".join(updated) + "\n---\n" + body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-path", required=True, help="Repository-relative selected item path under active")
    parser.add_argument(
        "--in-progress-path",
        required=True,
        help="Repository-relative selected item path under in_progress",
    )
    parser.add_argument("--plan-path", required=True, help="Repository-relative approved plan path")
    parser.add_argument("--output-path", required=True, help="Path to write the reconciled item path")
    args = parser.parse_args()

    active_path = _require_safe_repo_path(args.active_path, label="active path")
    in_progress_path = _require_safe_repo_path(args.in_progress_path, label="in-progress path")
    plan_path = _require_safe_repo_path(args.plan_path, label="plan path")
    _require_backlog_state(active_path, "active")
    _require_backlog_state(in_progress_path, "in_progress")
    if active_path.name != in_progress_path.name:
        raise SystemExit(f"Selected item path names differ: {active_path} != {in_progress_path}")

    active_exists = active_path.is_file()
    in_progress_exists = in_progress_path.is_file()
    if active_exists and in_progress_exists:
        raise SystemExit(f"Selected item exists in both active and in_progress: {active_path.name}")
    if not active_exists and not in_progress_exists:
        raise SystemExit(f"Selected item exists in neither active nor in_progress: {active_path.name}")

    if active_exists:
        in_progress_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.rename(in_progress_path)

    _rewrite_plan_path(in_progress_path, plan_path.as_posix())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(in_progress_path.as_posix() + "\n", encoding="utf-8")
    print(in_progress_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

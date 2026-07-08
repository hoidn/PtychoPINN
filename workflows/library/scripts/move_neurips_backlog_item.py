#!/usr/bin/env python3
"""Move backlog items between active, in_progress, and done."""

from __future__ import annotations

import argparse
from pathlib import Path


ALLOWED_STATES = {"active", "in_progress", "done"}
ALLOWED_TRANSITIONS = {
    ("active", "in_progress"),
    ("in_progress", "done"),
    ("in_progress", "active"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-path", required=True, help="Current repository-relative backlog item path")
    parser.add_argument("--dest-state", required=True, choices=sorted(ALLOWED_STATES))
    args = parser.parse_args()

    item_path = Path(args.item_path)
    if item_path.is_absolute() or ".." in item_path.parts:
        raise SystemExit(f"Unsafe backlog item path: {item_path}")
    if len(item_path.parts) < 4 or item_path.parts[:2] != ("docs", "backlog"):
        raise SystemExit(f"Backlog item must live under docs/backlog/: {item_path}")
    source_state = item_path.parts[2]
    if source_state not in ALLOWED_STATES:
        raise SystemExit(f"Unsupported source queue state: {source_state}")

    transition = (source_state, args.dest_state)
    if transition not in ALLOWED_TRANSITIONS:
        raise SystemExit(f"Illegal queue transition: {source_state} -> {args.dest_state}")
    if not item_path.is_file():
        raise SystemExit(f"Backlog item does not exist: {item_path}")

    target_path = Path("docs/backlog") / args.dest_state / item_path.name
    if target_path.exists():
        raise SystemExit(f"Destination already exists: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    item_path.rename(target_path)
    print(target_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

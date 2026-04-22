#!/usr/bin/env python3
"""Maintain durable run state for the NeurIPS backlog drain workflow."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": 1,
            "run_id": None,
            "current_roadmap_path": None,
            "current_item": None,
            "completed_items": [],
            "blocked_items": {},
            "history": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", required=True, help="Repository-relative JSON state path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--roadmap-path")

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--run-id", required=True)
    select_parser.add_argument("--item-id", required=True)
    select_parser.add_argument("--item-path", required=True)
    select_parser.add_argument("--plan-path")
    select_parser.add_argument("--roadmap-path")

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("--item-id", required=True)
    complete_parser.add_argument("--item-path", required=True)
    complete_parser.add_argument("--plan-path", required=True)
    complete_parser.add_argument("--execution-report-path", required=True)
    complete_parser.add_argument("--roadmap-path")

    blocked_parser = subparsers.add_parser("block")
    blocked_parser.add_argument("--item-id", required=True)
    blocked_parser.add_argument("--item-path", required=True)
    blocked_parser.add_argument("--reason", required=True)
    blocked_parser.add_argument("--stage", required=True)
    blocked_parser.add_argument("--plan-path")
    blocked_parser.add_argument("--roadmap-path")

    args = parser.parse_args()
    state_path = Path(args.state_path)
    if state_path.is_absolute() or ".." in state_path.parts:
        raise SystemExit(f"Unsafe state path: {state_path}")

    state = _load_state(state_path)
    if args.command == "init":
        state["run_id"] = args.run_id
        state["initialized_at_utc"] = _timestamp()
        if args.roadmap_path:
            state["current_roadmap_path"] = args.roadmap_path
    elif args.command == "select":
        state["run_id"] = args.run_id
        state["current_item"] = {
            "item_id": args.item_id,
            "item_path": args.item_path,
            "selected_at_utc": _timestamp(),
            "plan_path": args.plan_path,
            "roadmap_path": args.roadmap_path or state.get("current_roadmap_path"),
        }
        if args.roadmap_path:
            state["current_roadmap_path"] = args.roadmap_path
        state["history"].append({"event": "select", **state["current_item"]})
    elif args.command == "complete":
        entry = {
            "item_id": args.item_id,
            "item_path": args.item_path,
            "completed_at_utc": _timestamp(),
            "plan_path": args.plan_path,
            "execution_report_path": args.execution_report_path,
            "roadmap_path": args.roadmap_path or state.get("current_roadmap_path"),
        }
        if args.item_id not in state["completed_items"]:
            state["completed_items"].append(args.item_id)
        state["blocked_items"].pop(args.item_id, None)
        state["current_item"] = None
        if args.roadmap_path:
            state["current_roadmap_path"] = args.roadmap_path
        state["history"].append({"event": "complete", **entry})
    elif args.command == "block":
        entry = {
            "item_id": args.item_id,
            "item_path": args.item_path,
            "blocked_at_utc": _timestamp(),
            "stage": args.stage,
            "reason": args.reason,
            "plan_path": args.plan_path,
            "roadmap_path": args.roadmap_path or state.get("current_roadmap_path"),
        }
        state["blocked_items"][args.item_id] = entry
        state["current_item"] = None
        if args.roadmap_path:
            state["current_roadmap_path"] = args.roadmap_path
        state["history"].append({"event": "block", **entry})
    else:
        raise SystemExit(f"Unsupported command: {args.command}")

    _write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

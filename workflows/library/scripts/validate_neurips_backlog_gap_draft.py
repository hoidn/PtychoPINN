#!/usr/bin/env python3
"""Validate a provider-drafted NeurIPS backlog gap item."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path.cwd()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_workspace_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path escapes repository root: {value}") from exc
    if ".." in Path(value).parts:
        raise SystemExit(f"Unsafe path contains '..': {value}")
    if must_exist and not resolved.exists():
        raise SystemExit(f"Path does not exist: {value}")
    return resolved


def _repo_relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _string_list(payload: dict[str, Any], key: str, *, required: bool = False) -> list[str]:
    value = payload.get(key)
    if value is None:
        if required:
            raise SystemExit(f"{key} is required")
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise SystemExit(f"{key} must be a non-empty string list" if required else f"{key} must be a string list")
    cleaned = [item.strip() for item in value]
    if required and not cleaned:
        raise SystemExit(f"{key} must be a non-empty string list")
    return cleaned


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SystemExit(f"Backlog item missing YAML frontmatter start fence: {path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"Backlog item missing YAML frontmatter end fence: {path}")

    payload: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text[4:end].splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  - ") or line.startswith("- "):
            if not current_list_key:
                raise SystemExit(f"List item without owning key in frontmatter: {path}: {line}")
            payload.setdefault(current_list_key, [])
            assert isinstance(payload[current_list_key], list)
            payload[current_list_key].append(line.split("- ", 1)[1].strip())
            continue
        if ":" not in line:
            raise SystemExit(f"Malformed frontmatter line in {path}: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            payload[key] = []
            current_list_key = key
        else:
            payload[key] = value
            current_list_key = None
    return payload


def _validate_policy(policy: dict[str, Any]) -> tuple[list[str], list[str]]:
    allowed = _string_list(policy, "allowed_roadmap_phase_prefixes", required=True)
    disallowed = _string_list(policy, "disallowed_roadmap_phase_prefixes")
    return allowed, disallowed


def _ensure_under(path: Path, root_rel: str, label: str) -> None:
    root = _resolve_workspace_path(root_rel)
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} must be under {root_rel}: {_repo_relpath(path)}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-request-path", required=True)
    parser.add_argument("--draft-bundle-path", required=True)
    parser.add_argument("--gate-policy-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gap_request_path = _resolve_workspace_path(args.gap_request_path, must_exist=True)
    draft_bundle_path = _resolve_workspace_path(args.draft_bundle_path, must_exist=True)
    gate_policy_path = _resolve_workspace_path(args.gate_policy_path, must_exist=True)
    output_path = _resolve_workspace_path(args.output)

    gap_request = _load_json(gap_request_path)
    draft = _load_json(draft_bundle_path)
    policy = _load_json(gate_policy_path)
    allowed, disallowed = _validate_policy({**policy, **gap_request})

    status = str(draft.get("draft_status") or "").strip()
    if status == "BLOCKED":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "draft_validation_status": "BLOCKED",
                    "reason": str(draft.get("reason") or "Backlog gap drafter blocked."),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    if status != "DRAFTED":
        raise SystemExit(f"draft_status must be DRAFTED or BLOCKED: {status}")

    item_rel = str(draft.get("backlog_item_path") or "").strip()
    plan_rel = str(draft.get("seed_plan_path") or "").strip()
    if not item_rel or not plan_rel:
        raise SystemExit("DRAFTED bundle requires backlog_item_path and seed_plan_path")

    item_path = _resolve_workspace_path(item_rel, must_exist=True)
    plan_path = _resolve_workspace_path(plan_rel, must_exist=True)
    _ensure_under(item_path, str(gap_request.get("gap_item_target_dir") or "docs/backlog/active"), "backlog_item_path")
    _ensure_under(plan_path, str(gap_request.get("gap_plan_target_root") or ""), "seed_plan_path")

    roadmap_path = str(gap_request.get("roadmap_path") or "").strip()
    draft_roadmap_path = str(draft.get("roadmap_path") or roadmap_path).strip()
    if roadmap_path and draft_roadmap_path != roadmap_path:
        raise SystemExit("Draft bundle attempted to change roadmap_path")

    frontmatter = _parse_frontmatter(item_path)
    check_commands = _string_list(frontmatter, "check_commands", required=True)
    phases = _string_list(frontmatter, "related_roadmap_phases", required=True)
    item_plan = str(frontmatter.get("plan_path") or "").strip()
    if item_plan != _repo_relpath(plan_path):
        raise SystemExit(f"plan_path must match seed_plan_path: {item_plan} != {_repo_relpath(plan_path)}")
    if not any(any(phase.startswith(prefix) for prefix in allowed) for phase in phases):
        raise SystemExit("Drafted backlog item has no allowed roadmap phase")
    if any(any(phase.startswith(prefix) for prefix in disallowed) for phase in phases):
        raise SystemExit("Drafted backlog item uses a disallowed roadmap phase")
    if not check_commands:
        raise SystemExit("Drafted backlog item must include check_commands")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "draft_validation_status": "VALID",
                "backlog_item_path": _repo_relpath(item_path),
                "seed_plan_path": _repo_relpath(plan_path),
                "reason": "Drafted backlog gap item passed deterministic validation.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

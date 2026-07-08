#!/usr/bin/env python3
"""Validate a provider-drafted NeurIPS backlog gap item."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


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

    try:
        parsed = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"Malformed YAML frontmatter in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"YAML frontmatter must be a mapping: {path}")
    return {str(key): value for key, value in parsed.items()}


def _write_validation_failure(output_path: Path, reason: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "draft_validation_status": "INVALID",
                "reason": reason,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _validate_backlog_item(
    item_path: Path,
    plan_path: Path,
    *,
    allowed: list[str],
    disallowed: list[str],
) -> None:
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


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(target)


def _run(args: argparse.Namespace) -> int:
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

    final_item_path = _resolve_workspace_path(item_rel)
    final_plan_path = _resolve_workspace_path(plan_rel)
    _ensure_under(final_item_path, str(gap_request.get("gap_item_target_dir") or "docs/backlog/active"), "backlog_item_path")
    _ensure_under(final_plan_path, str(gap_request.get("gap_plan_target_root") or ""), "seed_plan_path")

    roadmap_path = str(gap_request.get("roadmap_path") or "").strip()
    draft_roadmap_path = str(draft.get("roadmap_path") or roadmap_path).strip()
    if roadmap_path and draft_roadmap_path != roadmap_path:
        raise SystemExit("Draft bundle attempted to change roadmap_path")

    candidate_item_rel = str(draft.get("candidate_backlog_item_path") or "").strip()
    candidate_plan_rel = str(draft.get("candidate_plan_path") or "").strip()
    if candidate_item_rel or candidate_plan_rel:
        if not candidate_item_rel or not candidate_plan_rel:
            raise SystemExit("Candidate draft mode requires candidate_backlog_item_path and candidate_plan_path")
        candidate_item_path = _resolve_workspace_path(candidate_item_rel, must_exist=True)
        candidate_plan_path = _resolve_workspace_path(candidate_plan_rel, must_exist=True)
        _ensure_under(candidate_item_path, "state", "candidate_backlog_item_path")
        _ensure_under(candidate_plan_path, "state", "candidate_plan_path")
        _validate_backlog_item(candidate_item_path, final_plan_path, allowed=allowed, disallowed=disallowed)
        _atomic_copy(candidate_plan_path, final_plan_path)
        _atomic_copy(candidate_item_path, final_item_path)
    else:
        final_item_path = _resolve_workspace_path(item_rel, must_exist=True)
        final_plan_path = _resolve_workspace_path(plan_rel, must_exist=True)
        _ensure_under(final_item_path, str(gap_request.get("gap_item_target_dir") or "docs/backlog/active"), "backlog_item_path")
        _ensure_under(final_plan_path, str(gap_request.get("gap_plan_target_root") or ""), "seed_plan_path")
        _validate_backlog_item(final_item_path, final_plan_path, allowed=allowed, disallowed=disallowed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "draft_validation_status": "VALID",
                "backlog_item_path": _repo_relpath(final_item_path),
                "seed_plan_path": _repo_relpath(final_plan_path),
                "reason": "Drafted backlog gap item passed deterministic validation.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-request-path", required=True)
    parser.add_argument("--draft-bundle-path", required=True)
    parser.add_argument("--gate-policy-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        return _run(args)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            raise
        reason = str(exc.code)
        output_path = _resolve_workspace_path(args.output)
        _write_validation_failure(output_path, reason)
        print(reason, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

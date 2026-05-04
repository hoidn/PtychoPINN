#!/usr/bin/env python3
"""Filter NeurIPS backlog manifest items through a deterministic roadmap gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_neurips_backlog_manifest import _build_manifest_entries


REPO_ROOT = Path.cwd()
STATUSES = {"ELIGIBLE", "BACKLOG_GAP", "DONE", "BLOCKED"}


def _load_json(path: Path, *, missing_default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        if missing_default is not None:
            return missing_default
        raise SystemExit(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_workspace_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        if ".." in path.parts:
            raise SystemExit(f"Unsafe path contains '..': {value}")
        resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path escapes repository root: {value}") from exc
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


def _validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    allowed = _string_list(policy, "allowed_roadmap_phase_prefixes", required=True)
    disallowed = _string_list(policy, "disallowed_roadmap_phase_prefixes")
    gap_policy = str(policy.get("gap_policy") or "block").strip()
    if gap_policy not in {"draft_backlog_item", "block"}:
        raise SystemExit(f"Unsupported gap_policy: {gap_policy}")
    target_dir = str(policy.get("gap_item_target_dir") or "docs/backlog/active").strip()
    plan_root = str(policy.get("gap_plan_target_root") or "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps").strip()
    _resolve_workspace_path(target_dir)
    _resolve_workspace_path(plan_root)
    return {
        **policy,
        "allowed_roadmap_phase_prefixes": allowed,
        "disallowed_roadmap_phase_prefixes": disallowed,
        "gap_policy": gap_policy,
        "gap_item_target_dir": target_dir,
        "gap_plan_target_root": plan_root,
        "required_scope_summary": str(policy.get("required_scope_summary") or "").strip(),
        "current_gate_id": str(policy.get("current_gate_id") or "").strip(),
    }


def _completed_ids(progress: dict[str, Any], run_state: dict[str, Any]) -> set[str]:
    completed: set[str] = set()
    for key in ("completed_items", "completed_tranches"):
        value = run_state.get(key)
        if isinstance(value, list):
            completed.update(str(item) for item in value)
        value = progress.get(key)
        if isinstance(value, list):
            completed.update(str(item) for item in value)
    return completed


def _blocked_ids(run_state: dict[str, Any]) -> set[str]:
    blocked = run_state.get("blocked_items")
    if isinstance(blocked, dict):
        return {str(key) for key in blocked}
    return set()


def _refresh_manifest_from_backlog_root(manifest: dict[str, Any]) -> dict[str, Any]:
    backlog_root_value = str(manifest.get("backlog_root") or "").strip()
    if not backlog_root_value:
        return manifest

    backlog_root = _resolve_workspace_path(backlog_root_value, must_exist=True)
    if not backlog_root.is_dir():
        raise SystemExit(f"Manifest backlog_root is not a directory: {backlog_root_value}")

    entries, invalid_entries = _build_manifest_entries(sorted(backlog_root.glob("*.md")))
    return {
        **manifest,
        "manifest_version": 2,
        "active_count": len(entries),
        "total_active_count": len(entries) + len(invalid_entries),
        "invalid_count": len(invalid_entries),
        "items": entries,
        "invalid_items": invalid_entries,
        "refreshed_from_backlog_root": backlog_root_value,
    }


def _item_phase_status(item: dict[str, Any], allowed: list[str], disallowed: list[str]) -> tuple[bool, str]:
    phases = item.get("related_roadmap_phases")
    if not isinstance(phases, list) or not phases:
        return False, "no roadmap phase"
    clean_phases = [str(phase).strip() for phase in phases if str(phase).strip()]
    if any(any(phase.startswith(prefix) for prefix in disallowed) for phase in clean_phases):
        return False, "future roadmap phase"
    if not any(any(phase.startswith(prefix) for prefix in allowed) for phase in clean_phases):
        return False, "outside current roadmap gate"
    return True, "eligible roadmap phase"


def _classify_items(
    manifest: dict[str, Any],
    policy: dict[str, Any],
    progress: dict[str, Any],
    run_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed = _completed_ids(progress, run_state)
    blocked = _blocked_ids(run_state)
    items = manifest.get("items")
    if not isinstance(items, list):
        raise SystemExit("Manifest items must be a list")

    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise SystemExit("Manifest contains a non-object item")
        item_id = str(item.get("item_id") or "")
        reasons: list[str] = []
        if item_id in completed:
            reasons.append("already completed")
        if item_id in blocked:
            reasons.append("blocked in run state")
        prerequisites = item.get("prerequisites") or []
        if not isinstance(prerequisites, list):
            raise SystemExit(f"Invalid prerequisites for {item_id}")
        missing = [str(prereq) for prereq in prerequisites if str(prereq) not in completed]
        if missing:
            reasons.append("missing prerequisites: " + ", ".join(missing))
        phase_ok, phase_reason = _item_phase_status(
            item,
            policy["allowed_roadmap_phase_prefixes"],
            policy["disallowed_roadmap_phase_prefixes"],
        )
        if not phase_ok:
            reasons.append(phase_reason)

        if reasons:
            ineligible.append({**item, "ineligibility_reasons": reasons})
        else:
            eligible.append(item)
    return eligible, ineligible


def _has_current_phase_item(items: list[dict[str, Any]], allowed: list[str], disallowed: list[str]) -> bool:
    for item in items:
        phases = item.get("related_roadmap_phases") or []
        if not isinstance(phases, list):
            continue
        clean_phases = [str(phase).strip() for phase in phases if str(phase).strip()]
        if any(any(phase.startswith(prefix) for prefix in disallowed) for phase in clean_phases):
            continue
        if any(any(phase.startswith(prefix) for prefix in allowed) for phase in clean_phases):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--gate-policy-path", required=True)
    parser.add_argument("--progress-ledger-path", required=True)
    parser.add_argument("--run-state-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = _resolve_workspace_path(args.manifest_path, must_exist=True)
    policy_path = _resolve_workspace_path(args.gate_policy_path, must_exist=True)
    progress_path = _resolve_workspace_path(args.progress_ledger_path, must_exist=True)
    run_state_path = _resolve_workspace_path(args.run_state_path)
    output_path = _resolve_workspace_path(args.output)

    manifest = _refresh_manifest_from_backlog_root(_load_json(manifest_path))
    policy = _validate_policy(_load_json(policy_path))
    progress = _load_json(progress_path)
    run_state = _load_json(run_state_path, missing_default={"completed_items": [], "blocked_items": {}})

    eligible, ineligible = _classify_items(manifest, policy, progress, run_state)
    valid_items = manifest.get("items") if isinstance(manifest.get("items"), list) else []
    invalid_items = manifest.get("invalid_items") if isinstance(manifest.get("invalid_items"), list) else []
    total_active_count = int(manifest.get("total_active_count") or len(valid_items) + len(invalid_items))
    all_items = valid_items + invalid_items
    has_current_phase_item = _has_current_phase_item(
        all_items,
        policy["allowed_roadmap_phase_prefixes"],
        policy["disallowed_roadmap_phase_prefixes"],
    )
    if total_active_count == 0:
        gate_status = "DONE"
    elif eligible:
        gate_status = "ELIGIBLE"
    elif has_current_phase_item:
        gate_status = "BLOCKED"
    elif policy["gap_policy"] == "draft_backlog_item":
        gate_status = "BACKLOG_GAP"
    else:
        gate_status = "BLOCKED"
    assert gate_status in STATUSES

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eligible_manifest_path = output_path.parent / "eligible_manifest.json"
    gap_request_path = output_path.parent / "gap_request.json"

    eligible_manifest = {
        **manifest,
        "active_count": len(eligible),
        "items": eligible,
        "invalid_count": len(invalid_items),
        "invalid_items": invalid_items,
        "source_manifest_path": _repo_relpath(manifest_path),
        "roadmap_gate_status": gate_status,
    }
    eligible_manifest_path.write_text(json.dumps(eligible_manifest, indent=2) + "\n", encoding="utf-8")

    gap_request = {
        "current_gate_id": policy.get("current_gate_id"),
        "required_scope_summary": policy.get("required_scope_summary"),
        "allowed_roadmap_phase_prefixes": policy["allowed_roadmap_phase_prefixes"],
        "disallowed_roadmap_phase_prefixes": policy["disallowed_roadmap_phase_prefixes"],
        "gap_item_target_dir": policy["gap_item_target_dir"],
        "gap_plan_target_root": policy["gap_plan_target_root"],
        "source_manifest_path": _repo_relpath(manifest_path),
        "eligible_count": len(eligible),
        "ineligible_count": len(ineligible),
        "invalid_count": len(invalid_items),
        "ineligible_items": ineligible,
        "invalid_items": invalid_items,
        "roadmap_path": progress.get("roadmap_path"),
    }
    gap_request_path.write_text(json.dumps(gap_request, indent=2) + "\n", encoding="utf-8")

    payload = {
        "gate_status": gate_status,
        "eligible_manifest_path": _repo_relpath(eligible_manifest_path),
        "gap_request_path": _repo_relpath(gap_request_path),
        "eligible_count": len(eligible),
        "ineligible_count": len(ineligible),
        "invalid_count": len(invalid_items),
        "eligible_items": eligible,
        "ineligible_items": ineligible,
        "invalid_items": invalid_items,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

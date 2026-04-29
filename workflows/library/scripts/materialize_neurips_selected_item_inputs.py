#!/usr/bin/env python3
"""Materialize deterministic item inputs from backlog selection output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path.cwd()
ALLOWED_SELECTION_MODES = {"ACTIVE_SELECTION", "RECOVERED_IN_PROGRESS"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relpath(path: Path) -> str:
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"Path escapes repo root: {path}") from exc


def _parse_backlog_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SystemExit(f"Missing frontmatter start fence: {path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"Missing frontmatter end fence: {path}")
    return text[end + len("\n---\n") :].strip()


def _parse_frontmatter_and_body(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SystemExit(f"Missing frontmatter start fence: {path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"Missing frontmatter end fence: {path}")

    payload: dict[str, object] = {}
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
    return payload, text[end + len("\n---\n") :].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection_path = Path(args.selection_path)
    manifest_path = Path(args.manifest_path)
    state_root = Path(args.state_root)
    if not state_root.is_absolute():
        state_root = (REPO_ROOT / state_root).resolve()
    output_path = Path(args.output)

    selection = _load_json(selection_path)
    manifest = _load_json(manifest_path)

    if selection.get("selection_status") != "SELECTED":
        raise SystemExit("selection payload must have selection_status=SELECTED")

    item_id = selection.get("selected_item_id")
    item_path_rel = selection.get("selected_item_path")
    selection_mode = str(selection.get("selection_mode") or "ACTIVE_SELECTION").strip()
    if selection_mode not in ALLOWED_SELECTION_MODES:
        raise SystemExit(f"Unsupported selection_mode: {selection_mode}")
    if not isinstance(item_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", item_id):
        raise SystemExit(f"Unsafe or missing selected_item_id: {item_id}")
    manifest_entry: dict[str, object]
    selected_item_active_path: str
    selected_item_in_progress_path: str
    if selection_mode == "ACTIVE_SELECTION":
        if not isinstance(item_path_rel, str) or not item_path_rel.startswith("docs/backlog/active/"):
            raise SystemExit(f"Active selection must come from docs/backlog/active/: {item_path_rel}")
        manifest_items = manifest.get("items")
        if not isinstance(manifest_items, list):
            raise SystemExit("Manifest items payload is missing or invalid")
        found_entry = next(
            (
                entry
                for entry in manifest_items
                if entry.get("item_id") == item_id and entry.get("path") == item_path_rel
            ),
            None,
        )
        if found_entry is None:
            raise SystemExit(f"Selected item {item_id} was not found in the manifest")
        manifest_entry = found_entry
        selected_item_active_path = item_path_rel
        selected_item_in_progress_path = f"docs/backlog/in_progress/{Path(item_path_rel).name}"
    else:
        if not isinstance(item_path_rel, str) or not item_path_rel.startswith("docs/backlog/in_progress/"):
            raise SystemExit(f"Recovered selection must come from docs/backlog/in_progress/: {item_path_rel}")
        item_frontmatter, item_body = _parse_frontmatter_and_body(REPO_ROOT / item_path_rel)
        check_commands_value = item_frontmatter.get("check_commands")
        if not isinstance(check_commands_value, list) or not check_commands_value:
            raise SystemExit(f"Recovered item missing non-empty check_commands for {item_id}")
        manifest_entry = {
            "item_id": item_id,
            "title": item_id,
            "path": item_path_rel,
            "plan_path": str(item_frontmatter.get("plan_path") or "").strip(),
            "check_commands": check_commands_value,
            "summary": item_body.splitlines()[0].strip() if item_body.splitlines() else "",
        }
        selected_item_active_path = f"docs/backlog/active/{Path(item_path_rel).name}"
        selected_item_in_progress_path = item_path_rel

    item_path = REPO_ROOT / item_path_rel
    if not item_path.is_file():
        raise SystemExit(f"Selected backlog item does not exist: {item_path_rel}")

    check_commands = manifest_entry.get("check_commands")
    if not isinstance(check_commands, list) or not check_commands:
        raise SystemExit(f"Manifest entry missing non-empty check_commands for {item_id}")

    previous_plan_path = str(manifest_entry.get("plan_path") or "").strip()
    previous_plan_text = ""
    if previous_plan_path:
        previous_plan_target = REPO_ROOT / previous_plan_path
        if previous_plan_target.is_file():
            previous_plan_text = previous_plan_target.read_text(encoding="utf-8").strip()

    item_root = state_root / "items" / item_id
    item_root.mkdir(parents=True, exist_ok=True)

    check_commands_path = item_root / "check_commands.json"
    check_commands_path.write_text(json.dumps(check_commands, indent=2) + "\n", encoding="utf-8")

    check_commands_pointer = item_root / "check_commands_path.txt"
    check_commands_pointer.write_text(_repo_relpath(check_commands_path) + "\n", encoding="utf-8")

    context_path = item_root / "selected-item-context.md"
    context_lines = [
        f"# Selected Backlog Item Context: {item_id}",
        "",
        f"- item_id: `{item_id}`",
        f"- selection_mode: `{selection_mode}`",
        f"- authoritative_item_path: `docs/backlog/in_progress/{item_path.name}`",
        f"- selection_source_path: `{item_path_rel}`",
        f"- previous_plan_path: `{previous_plan_path or 'none'}`",
        "",
        "## Selection Summary",
        "",
        f"- title: {manifest_entry.get('title', item_id)}",
        f"- summary: {manifest_entry.get('summary', '').strip() or 'none'}",
        "",
        "## Required Check Commands",
        "",
    ]
    context_lines.extend(f"- `{command}`" for command in check_commands)
    context_lines.extend(
        [
            "",
            "## Backlog Item",
            "",
            _parse_backlog_body(item_path),
            "",
        ]
    )
    if previous_plan_text:
        context_lines.extend(
            [
                "## Previous Plan Background",
                "",
                previous_plan_text,
                "",
            ]
        )
    context_path.write_text("\n".join(context_lines).rstrip() + "\n", encoding="utf-8")

    payload = {
        "item_id": item_id,
        "selection_mode": selection_mode,
        "selected_item_active_path": selected_item_active_path,
        "selected_item_in_progress_path": selected_item_in_progress_path,
        "selected_item_context_path": _repo_relpath(context_path),
        "check_commands_path": _repo_relpath(check_commands_path),
        "check_commands_pointer_path": _repo_relpath(check_commands_pointer),
        "roadmap_sync_state_root": _repo_relpath(item_root / "roadmap-sync"),
        "plan_phase_state_root": _repo_relpath(item_root / "plan-phase"),
        "implementation_phase_state_root": _repo_relpath(item_root / "implementation-phase"),
        "plan_target_path": f"docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}/execution_plan.md",
        "candidate_plan_path": previous_plan_path,
        "plan_gate_recovery_bundle_path": _repo_relpath(item_root / "plan-gate-recovery.json"),
        "roadmap_sync_report_target_path": f"artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}-roadmap-sync.json",
        "plan_review_report_target_path": f"artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}-plan-review.json",
        "plan_gate_recovery_report_target_path": f"artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}-plan-recovery.md",
        "execution_report_target_path": f"artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}/execution_report.md",
        "checks_report_target_path": f"artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}-checks.json",
        "implementation_review_report_target_path": f"artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}-implementation-review.md",
        "item_summary_target_path": f"artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/{item_id}-summary.json",
        "outcome_bundle_path": _repo_relpath(item_root / "selected-item-outcome.json"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

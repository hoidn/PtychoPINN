#!/usr/bin/env python3
"""Build a typed manifest from NeurIPS backlog items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path.cwd()


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_frontmatter(text: str, source_path: Path) -> dict:
    if not text.startswith("---\n"):
        raise SystemExit(f"Backlog item missing YAML frontmatter start fence: {source_path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"Backlog item missing YAML frontmatter end fence: {source_path}")

    try:
        parsed = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"Malformed YAML frontmatter in {source_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"YAML frontmatter must be a mapping: {source_path}")
    payload: dict[str, object] = {str(key): value for key, value in parsed.items()}
    payload["_body"] = text[end + len("\n---\n") :]
    return payload


def _heading_title(body: str, source_path: Path) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise SystemExit(f"Backlog item missing top-level heading: {source_path}")


def _first_bullets(body: str, heading: str) -> list[str]:
    bullets: list[str] = []
    in_section = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_section = line == heading
            continue
        if in_section and line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets


def _summary_from_body(body: str) -> str:
    objective = _first_bullets(body, "## Objective")
    if objective:
        return objective[0]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _normalized_string_list(value: object, *, field: str, source_path: Path) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [part.strip() for part in str(value).split(",")]
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return []
    return cleaned


def _safe_relpath(path: Path, *, source_path: Path) -> str:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path escapes repo root for {source_path}: {path}") from exc
    return rel.as_posix()


def _entry_from_parsed(source_path: Path, parsed: dict[str, Any]) -> dict:
    parsed = dict(parsed)

    if "priority" not in parsed:
        raise SystemExit(f"Frontmatter missing required key priority: {source_path}")
    if "plan_path" not in parsed:
        raise SystemExit(f"Frontmatter missing required key plan_path: {source_path}")
    if "check_commands" not in parsed or not isinstance(parsed["check_commands"], list) or not parsed["check_commands"]:
        raise SystemExit(f"Frontmatter missing non-empty list check_commands: {source_path}")

    try:
        priority = int(str(parsed["priority"]).strip())
    except ValueError as exc:
        raise SystemExit(f"priority must be integer in {source_path}: {parsed['priority']}") from exc

    plan_path = Path(str(parsed["plan_path"]).strip())
    if plan_path.is_absolute() or ".." in plan_path.parts:
        raise SystemExit(f"Unsafe plan_path in {source_path}: {plan_path}")
    if not (REPO_ROOT / plan_path).is_file():
        raise SystemExit(f"plan_path target does not exist for {source_path}: {plan_path}")

    body = str(parsed.pop("_body"))
    title = _heading_title(body, source_path)
    item_id = source_path.stem
    check_commands = _normalized_string_list(parsed["check_commands"], field="check_commands", source_path=source_path)
    if not check_commands:
        raise SystemExit(f"check_commands must contain at least one non-empty command: {source_path}")

    return {
        "item_id": item_id,
        "title": title,
        "path": _safe_relpath(source_path, source_path=source_path),
        "status": "active",
        "priority": priority,
        "plan_path": plan_path.as_posix(),
        "check_commands": check_commands,
        "summary": _summary_from_body(body),
        "prerequisites": _normalized_string_list(parsed.get("prerequisites"), field="prerequisites", source_path=source_path),
        "related_roadmap_phases": _normalized_string_list(
            parsed.get("related_roadmap_phases"),
            field="related_roadmap_phases",
            source_path=source_path,
        ),
        "blocking_signals": _normalized_string_list(parsed.get("blocking_signals"), field="blocking_signals", source_path=source_path),
        "signals_for_selection": _normalized_string_list(
            parsed.get("signals_for_selection"),
            field="signals_for_selection",
            source_path=source_path,
        ),
    }


def _build_entry(source_path: Path) -> dict:
    parsed = _parse_frontmatter(source_path.read_text(encoding="utf-8"), source_path)
    return _entry_from_parsed(source_path, parsed)


def _invalid_entry(source_path: Path, reason: str, parsed: dict[str, Any] | None = None) -> dict:
    phases: list[str] = []
    priority: int | None = None
    plan_path = ""
    if parsed:
        phases = _normalized_string_list(parsed.get("related_roadmap_phases"), field="related_roadmap_phases", source_path=source_path)
        plan_path = str(parsed.get("plan_path") or "").strip()
        try:
            priority_value = parsed.get("priority")
            if priority_value is not None:
                priority = int(str(priority_value).strip())
        except ValueError:
            priority = None
    return {
        "item_id": source_path.stem,
        "path": _safe_relpath(source_path, source_path=source_path),
        "status": "invalid",
        "priority": priority,
        "plan_path": plan_path,
        "related_roadmap_phases": phases,
        "invalid_reasons": [reason],
    }


def _build_entry_or_invalid(source_path: Path) -> tuple[dict | None, dict | None]:
    parsed: dict[str, Any] | None = None
    try:
        parsed = _parse_frontmatter(source_path.read_text(encoding="utf-8"), source_path)
        return _entry_from_parsed(source_path, parsed), None
    except SystemExit as exc:
        reason = str(exc.code)
        if parsed is None:
            try:
                parsed = _parse_frontmatter(source_path.read_text(encoding="utf-8"), source_path)
            except SystemExit:
                parsed = None
        return None, _invalid_entry(source_path, reason, parsed)


def _build_manifest_entries(items: list[Path]) -> tuple[list[dict], list[dict]]:
    valid_entries: list[dict] = []
    invalid_entries: list[dict] = []
    for path in items:
        entry, invalid = _build_entry_or_invalid(path)
        if entry is not None:
            valid_entries.append(entry)
        elif invalid is not None:
            invalid_entries.append(invalid)
    valid_entries.sort(key=lambda row: (row["priority"], row["path"]))
    invalid_entries.sort(key=lambda row: ((row.get("priority") is None, row.get("priority") or 999999), row["path"]))
    return valid_entries, invalid_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backlog-root", required=True, help="Backlog directory root such as docs/backlog/active")
    parser.add_argument("--output", required=True, help="Manifest JSON path")
    args = parser.parse_args()

    backlog_root = (REPO_ROOT / args.backlog_root).resolve()
    if not backlog_root.is_dir():
        raise SystemExit(f"Backlog root does not exist: {backlog_root}")
    try:
        backlog_rel = backlog_root.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Backlog root must live under repo root: {backlog_root}") from exc

    items = sorted(backlog_root.glob("*.md"))
    entries, invalid_entries = _build_manifest_entries(items)

    payload = {
        "manifest_version": 2,
        "backlog_root": backlog_rel.as_posix(),
        "active_count": len(entries),
        "total_active_count": len(entries) + len(invalid_entries),
        "invalid_count": len(invalid_entries),
        "items": entries,
        "invalid_items": invalid_entries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a typed manifest from NeurIPS backlog items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    payload: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in text[4:end].splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  - ") or line.startswith("- "):
            key = current_list_key
            if not key:
                raise SystemExit(f"List item without owning key in frontmatter: {source_path}: {line}")
            payload.setdefault(key, [])
            assert isinstance(payload[key], list)
            payload[key].append(_strip_wrapping_quotes(line.split("- ", 1)[1].strip()))
            continue
        if ":" not in line:
            raise SystemExit(f"Malformed frontmatter line in {source_path}: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            payload[key] = []
            current_list_key = key
        else:
            payload[key] = _strip_wrapping_quotes(value)
            current_list_key = None
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


def _build_entry(source_path: Path) -> dict:
    parsed = _parse_frontmatter(source_path.read_text(encoding="utf-8"), source_path)

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
    entries = [_build_entry(path) for path in items]
    entries.sort(key=lambda row: (row["priority"], row["path"]))

    payload = {
        "manifest_version": 1,
        "backlog_root": backlog_rel.as_posix(),
        "active_count": len(entries),
        "items": entries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

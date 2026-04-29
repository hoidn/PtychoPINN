#!/usr/bin/env python3
"""Recover approved NeurIPS plan-gate outputs from durable item evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path.cwd()
ALLOWED_SELECTION_MODES = {"ACTIVE_SELECTION", "RECOVERED_IN_PROGRESS"}


def _write_json(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _missing(output_path: Path) -> int:
    _write_json(output_path, {"plan_gate_status": "MISSING"})
    return 0


def _safe_relpath(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    try:
        (REPO_ROOT / path).resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    return path


def _parse_frontmatter(path: Path) -> dict[str, object]:
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
            if current_list_key:
                payload.setdefault(current_list_key, [])
                assert isinstance(payload[current_list_key], list)
                payload[current_list_key].append(line.split("- ", 1)[1].strip())
            continue
        if ":" not in line:
            continue
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


def _write_recovery_report(path: Path, *, item_path: Path, plan_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# NeurIPS Plan Gate Recovery",
                "",
                f"- item_path: `{item_path.as_posix()}`",
                f"- recovered_plan_path: `{plan_path.as_posix()}`",
                "- validation: recovered item path is in `docs/backlog/in_progress/`",
                "- validation: plan path is repo-relative under `docs/plans/`",
                "- validation: plan file exists",
                "",
                "This report records deterministic recovery of a previously selected plan authority. It is not a fresh plan review.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-mode", required=True)
    parser.add_argument("--selected-item-path", required=True)
    parser.add_argument("--recovery-report-target-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    selection_mode = args.selection_mode.strip()
    if selection_mode not in ALLOWED_SELECTION_MODES:
        raise SystemExit(f"Unsupported selection mode: {selection_mode}")
    if selection_mode != "RECOVERED_IN_PROGRESS":
        return _missing(output_path)

    selected_item_path = _safe_relpath(args.selected_item_path.strip())
    if selected_item_path is None or selected_item_path.parts[:3] != ("docs", "backlog", "in_progress"):
        return _missing(output_path)
    item_target = REPO_ROOT / selected_item_path
    if not item_target.is_file():
        return _missing(output_path)

    frontmatter = _parse_frontmatter(item_target)
    plan_path = _safe_relpath(str(frontmatter.get("plan_path") or "").strip())
    if plan_path is None or plan_path.parts[:2] != ("docs", "plans"):
        return _missing(output_path)
    if not (REPO_ROOT / plan_path).is_file():
        return _missing(output_path)

    report_path = _safe_relpath(args.recovery_report_target_path.strip())
    if report_path is None or report_path.parts[:2] != ("artifacts", "review"):
        raise SystemExit(f"Unsafe recovery report path: {args.recovery_report_target_path}")
    _write_recovery_report(REPO_ROOT / report_path, item_path=selected_item_path, plan_path=plan_path)

    _write_json(
        output_path,
        {
            "plan_gate_status": "RECOVERED",
            "plan_path": plan_path.as_posix(),
            "plan_review_decision": "APPROVE",
            "plan_review_report_path": report_path.as_posix(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

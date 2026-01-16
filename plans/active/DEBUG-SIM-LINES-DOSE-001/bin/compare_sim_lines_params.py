#!/usr/bin/env python3
"""
Compare sim_lines_4x snapshot metadata against the legacy dose_experiments
parameter scan. Emits a Markdown table plus a structured JSON diff.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import re
from typing import Any, Dict, List, Tuple

CFG_ASSIGNMENT_RE = re.compile(r"""cfg\['(?P<key>[^']+)'\]\s*=\s*(?P<value>.+)""")

# Parameters we care about along with a human-friendly label
PARAMETERS: List[Tuple[str, str]] = [
    ("gridsize", "gridsize"),
    ("probe_big", "probe_big"),
    ("probe_mask", "probe_mask"),
    ("probe_scale", "probe_scale"),
    ("offset", "offset"),
    ("outer_offset_train", "outer_offset_train"),
    ("outer_offset_test", "outer_offset_test"),
    ("nimgs_train", "nimgs_train"),
    ("nimgs_test", "nimgs_test"),
    ("nphotons", "nphotons"),
    ("group_count", "group_count"),
    ("neighbor_count", "neighbor_count"),
    ("reassemble_M", "reassemble_M"),
    ("intensity_scale.trainable", "intensity_scale.trainable"),
    ("total_images", "total_images"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=pathlib.Path, help="sim_lines_4x snapshot JSON")
    parser.add_argument("--dose-config", required=True, type=pathlib.Path, help="dose_experiments_param_scan.md file")
    parser.add_argument("--output-markdown", required=True, type=pathlib.Path, help="Output Markdown file")
    parser.add_argument("--output-json", required=True, type=pathlib.Path, help="Output comparison JSON")
    return parser.parse_args()


def literal_eval(value: str) -> Any:
    """Best-effort literal parsing with graceful fallback for identifiers."""
    try:
        return ast.literal_eval(value)
    except Exception:
        stripped = value.strip()
        return stripped


def parse_dose_config(path: pathlib.Path) -> Dict[str, Any]:
    """
    Parse cfg[...] assignments from the init() function in the captured
    dose_experiments script.
    """
    text = path.read_text()
    assignments: Dict[str, Any] = {}
    in_init = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            in_init = stripped.startswith("def init")
        if not in_init:
            continue
        match = CFG_ASSIGNMENT_RE.search(stripped)
        if not match:
            continue
        key = match.group("key")
        value = match.group("value")
        if key not in assignments:
            assignments[key] = literal_eval(value)
    return assignments


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def load_sim_lines_snapshot(path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    data = json.loads(path.read_text())
    run_params = data.get("run_params", {})
    scenarios: Dict[str, Dict[str, Any]] = {}
    for scenario in data.get("scenarios", []):
        scenario_name = scenario.get("name") or scenario.get("inputs", {}).get("name")
        if not scenario_name:
            continue
        defaults = scenario.get("defaults", {})
        inputs = scenario.get("inputs", {})
        derived = scenario.get("derived_counts", {})
        scenario_params = {
            "gridsize": inputs.get("gridsize"),
            "probe_big": coalesce(inputs.get("probe_big"), defaults.get("probe_big")),
            "probe_mask": coalesce(inputs.get("probe_mask"), defaults.get("probe_mask")),
            "probe_scale": inputs.get("probe_scale"),
            "group_count": scenario.get("group_count"),
            "neighbor_count": scenario.get("neighbor_count"),
            "reassemble_M": scenario.get("reassemble_M"),
            "nphotons": run_params.get("nphotons"),
            "total_images": derived.get("total_images"),
            "nimgs_train": derived.get("train_count"),
            "nimgs_test": derived.get("test_count"),
            # Fields absent in snapshot are filled later with None.
        }
        scenarios[scenario_name] = scenario_params
    return run_params, scenarios


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isclose(value, int(value)):
            return int(value)
    return value


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return f"{value}"
    return str(value)


def compute_delta(base: Any, scenario: Any) -> str:
    if base is None and scenario is None:
        return "missing in both"
    if base == scenario:
        return "match"
    if isinstance(base, (int, float)) and isinstance(scenario, (int, float)):
        return f"{scenario - base:+}"
    if base is None:
        return "not set in dose_experiments"
    if scenario is None:
        return "not set in sim_lines"
    return "differs"


def build_markdown(
    dose_params: Dict[str, Any],
    scenarios: Dict[str, Dict[str, Any]],
    snapshot_path: pathlib.Path,
    dose_path: pathlib.Path,
) -> str:
    lines: List[str] = []
    lines.append("# SIM-LINES-4X vs dose_experiments Parameter Diff")
    lines.append("")
    lines.append(f"- Snapshot source: `{snapshot_path}`")
    lines.append(f"- Legacy defaults source: `{dose_path}` (init() assignments)")
    lines.append("- `—` indicates the parameter was not defined in that pipeline.")
    lines.append("")
    for scenario_name in sorted(scenarios.keys()):
        lines.append(f"## Scenario: {scenario_name}")
        lines.append("")
        lines.append("| Parameter | dose_experiments | sim_lines | Δ / note |")
        lines.append("|-----------|------------------|-----------|----------|")
        scenario_params = scenarios[scenario_name]
        for key, label in PARAMETERS:
            scenario_value = scenario_params.get(key)
            dose_value = dose_params.get(key)
            delta = compute_delta(dose_value, scenario_value)
            lines.append(
                f"| {label} | {format_value(dose_value)} | {format_value(scenario_value)} | {delta} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_diff_json(
    dose_params: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    diff = {
        "dose_experiments": dose_params,
        "scenarios": {},
    }
    for scenario_name, scenario_params in scenarios.items():
        entry = {"parameters": scenario_params, "diff": {}}
        for key, _ in PARAMETERS:
            entry["diff"][key] = {
                "dose_experiments": dose_params.get(key),
                "sim_lines": scenario_params.get(key),
                "delta": compute_delta(dose_params.get(key), scenario_params.get(key)),
            }
        diff["scenarios"][scenario_name] = entry
    return diff


def ensure_all_keys(
    dose_params: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]]
) -> None:
    for key, _ in PARAMETERS:
        dose_params.setdefault(key, None)
        for scenario in scenarios.values():
            scenario.setdefault(key, None)


def main() -> None:
    args = parse_args()
    dose_params = parse_dose_config(args.dose_config)
    run_params, scenarios = load_sim_lines_snapshot(args.snapshot)

    # Some parameters live in the global run params; surface them explicitly.
    if "intensity_scale.trainable" not in dose_params:
        dose_params["intensity_scale.trainable"] = dose_params.get("intensity_scale.trainable")
    for scenario in scenarios.values():
        scenario.setdefault("offset", run_params.get("offset"))
        scenario.setdefault("outer_offset_train", run_params.get("outer_offset_train"))
        scenario.setdefault("outer_offset_test", run_params.get("outer_offset_test"))
        scenario.setdefault("intensity_scale.trainable", run_params.get("intensity_scale.trainable"))

    ensure_all_keys(dose_params, scenarios)
    markdown = build_markdown(dose_params, scenarios, args.snapshot, args.dose_config)
    diff = build_diff_json(dose_params, scenarios)

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown)
    args.output_json.write_text(json.dumps(diff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

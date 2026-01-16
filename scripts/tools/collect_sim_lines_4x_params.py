#!/usr/bin/env python3
"""Capture the canonical SIM-LINES-4X scenario parameters."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Ensure repository root is on sys.path so script-relative imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptycho.config.config import ModelConfig  # noqa: E402
from scripts.studies.sim_lines_4x.pipeline import (  # noqa: E402
    CUSTOM_PROBE_PATH,
    RunParams,
    ScenarioSpec,
    derive_counts,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/sim_lines_4x")
RUNNER_SCRIPTS: Dict[str, Path] = {
    "gs1_ideal": Path("scripts/studies/sim_lines_4x/run_gs1_ideal.py"),
    "gs1_custom": Path("scripts/studies/sim_lines_4x/run_gs1_custom_probe.py"),
    "gs2_ideal": Path("scripts/studies/sim_lines_4x/run_gs2_ideal.py"),
    "gs2_custom": Path("scripts/studies/sim_lines_4x/run_gs2_custom_probe.py"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SIM-LINES-4X scenario parameters and write JSON."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file for the captured scenario metadata.",
    )
    return parser.parse_args()


def _resolve_probe_flags(spec: ScenarioSpec, model_defaults: ModelConfig) -> Dict[str, bool]:
    if spec.probe_mode == "idealized":
        default_big = False
        default_mask = True
    else:
        default_big = model_defaults.probe_big
        default_mask = model_defaults.probe_mask
    probe_big = spec.probe_big if spec.probe_big is not None else default_big
    probe_mask = spec.probe_mask if spec.probe_mask is not None else default_mask
    return {"probe_big": probe_big, "probe_mask": probe_mask}


def _build_paths(name: str) -> Dict[str, str]:
    scenario_dir = DEFAULT_OUTPUT_ROOT / name
    return {
        "runner_script": str(RUNNER_SCRIPTS[name]),
        "default_output_root": str(DEFAULT_OUTPUT_ROOT),
        "scenario_dir": str(scenario_dir),
        "train_dir": str(scenario_dir / "train_outputs"),
        "inference_dir": str(scenario_dir / "inference_outputs"),
        "run_log": str(scenario_dir / "run.log"),
        "run_metadata": str(scenario_dir / "run_metadata.json"),
    }


def _scenario_entries(params: RunParams) -> List[Dict[str, object]]:
    scenarios = [
        ScenarioSpec(name="gs1_ideal", gridsize=1, probe_mode="idealized", probe_scale=10.0),
        ScenarioSpec(name="gs1_custom", gridsize=1, probe_mode="custom", probe_scale=4.0),
        ScenarioSpec(name="gs2_ideal", gridsize=2, probe_mode="idealized", probe_scale=10.0),
        ScenarioSpec(name="gs2_custom", gridsize=2, probe_mode="custom", probe_scale=4.0),
    ]
    model_defaults = ModelConfig()
    entries: List[Dict[str, object]] = []
    for spec in scenarios:
        totals = derive_counts(params, spec.gridsize)
        entries.append(
            {
                "name": spec.name,
                "inputs": asdict(spec),
                "defaults": _resolve_probe_flags(spec, model_defaults),
                "derived_counts": {
                    "total_images": totals[0],
                    "train_count": totals[1],
                    "test_count": totals[2],
                },
                "group_count": params.group_count,
                "neighbor_count": params.neighbor_count,
                "reassemble_M": params.reassemble_M,
                "paths": _build_paths(spec.name),
            }
        )
    return entries


def main() -> None:
    args = _parse_args()
    params = RunParams()
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "custom_probe_path": str(CUSTOM_PROBE_PATH),
        "run_params": asdict(params),
        "scenarios": _scenario_entries(params),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

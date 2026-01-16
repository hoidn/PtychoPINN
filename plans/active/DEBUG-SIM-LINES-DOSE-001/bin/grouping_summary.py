#!/usr/bin/env python3
"""
Instrument grouping stats for SIM-LINES vs legacy dose_experiments scenarios.

The script rebuilds the nongrid simulation pipeline up to the grouping call,
then records JSON + Markdown summaries (train/test) including the requested
vs actual group counts and coordinate ranges. Grouping failures (e.g., not
enough neighbors for gridsize=2) are captured as error text so the Do Now can
document the block instead of crashing.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ptycho import params as legacy_params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from scripts.simulation.synthetic_helpers import (
    make_lines_object,
    make_probe,
    normalize_probe_guess,
    simulate_nongrid_raw_data,
    split_raw_data_by_axis,
)
from scripts.studies.sim_lines_4x.pipeline import (
    CUSTOM_PROBE_PATH,
    RunParams,
    ScenarioSpec,
    derive_counts,
)

DEFAULT_SNAPSHOT = Path(
    "plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json"
)


def str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="gs1_custom", help="Scenario name from the snapshot JSON")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to the sim_lines params snapshot JSON",
    )
    parser.add_argument("--label", help="Run label for metadata (defaults to scenario name)")
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path to write the JSON summary",
    )
    parser.add_argument(
        "--output-markdown",
        required=True,
        type=Path,
        help="Path to write the Markdown summary",
    )

    parser.add_argument("--gridsize", type=int, help="Override gridsize used for grouping")
    parser.add_argument("--group-count", type=int, help="Requested group count (nsamples)")
    parser.add_argument("--neighbor-count", type=int, help="K nearest neighbors for grouping")
    parser.add_argument("--total-images", type=int, help="Override total image count before splitting")
    parser.add_argument("--split-fraction", type=float, help="Override train/test split fraction")
    parser.add_argument("--image-multiplier", type=int, default=1, help="Multiplier applied before derive_counts")
    parser.add_argument("--group-multiplier", type=int, default=1, help="Multiplier applied to scenario group_count")
    parser.add_argument("--object-seed", type=int, help="Override object seed")
    parser.add_argument("--sim-seed", type=int, help="Override simulation/grouping seed")
    parser.add_argument("--probe-mode", choices={"custom", "idealized"}, help="Override probe mode")
    parser.add_argument("--probe-scale", type=float, help="Override probe scale factor")
    parser.add_argument("--probe-big", type=str2bool, help="Override probe_big flag (true/false)")
    parser.add_argument("--probe-mask", type=str2bool, help="Override probe_mask flag (true/false)")
    parser.add_argument("--neighbor-seed", type=int, help="Seed forwarded to generate_grouped_data (defaults to sim seed)")
    return parser.parse_args()


def load_snapshot(snapshot_path: Path) -> Tuple[RunParams, Dict[str, Dict[str, Any]]]:
    data = json.loads(snapshot_path.read_text())
    run_params = RunParams(**data["run_params"])
    scenarios: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("scenarios", []):
        name = entry.get("name") or entry.get("inputs", {}).get("name")
        if not name:
            continue
        scenarios[name] = entry
    if not scenarios:
        raise ValueError(f"No scenarios found in snapshot: {snapshot_path}")
    return run_params, scenarios


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def scenario_spec(entry: Mapping[str, Any]) -> ScenarioSpec:
    defaults = entry.get("defaults", {})
    inputs = entry.get("inputs", {})
    return ScenarioSpec(
        name=entry.get("name") or inputs.get("name") or "unknown",
        gridsize=inputs.get("gridsize"),
        probe_mode=inputs.get("probe_mode") or "custom",
        probe_scale=inputs.get("probe_scale"),
        probe_big=coalesce(inputs.get("probe_big"), defaults.get("probe_big")),
        probe_mask=coalesce(inputs.get("probe_mask"), defaults.get("probe_mask")),
    )


def dataclass_with_overrides(params: RunParams, args: argparse.Namespace) -> RunParams:
    updates: Dict[str, Any] = {}
    if args.object_seed is not None:
        updates["object_seed"] = args.object_seed
    if args.sim_seed is not None:
        updates["sim_seed"] = args.sim_seed
    if args.split_fraction is not None:
        updates["split_fraction"] = args.split_fraction
    return dataclasses.replace(params, **updates) if updates else params


def collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    override_keys = [
        "gridsize",
        "group_count",
        "neighbor_count",
        "total_images",
        "split_fraction",
        "object_seed",
        "sim_seed",
        "probe_mode",
        "probe_scale",
        "probe_big",
        "probe_mask",
        "neighbor_seed",
    ]
    overrides: Dict[str, Any] = {}
    for key in override_keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.image_multiplier not in (None, 1):
        overrides["image_multiplier"] = args.image_multiplier
    if args.group_multiplier not in (None, 1):
        overrides["group_multiplier"] = args.group_multiplier
    return overrides


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def describe_array(arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if arr is None:
        return None
    array = np.asarray(arr)
    flat = array.reshape(-1)
    summary: Dict[str, Any] = {"shape": list(array.shape)}
    if flat.size == 0:
        summary.update({"min": None, "max": None, "mean": None, "std": None})
        return summary

    summary.update(
        {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
        }
    )

    if array.ndim >= 2 and array.shape[-2] == 2:
        axis_moved = np.moveaxis(array, -2, 0).reshape(2, -1)
        axis_stats: List[Dict[str, Any]] = []
        for axis_index, axis_values in enumerate(axis_moved):
            if axis_values.size == 0:
                axis_stats.append(
                    {
                        "axis": axis_index,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                    }
                )
                continue
            axis_stats.append(
                {
                    "axis": axis_index,
                    "min": float(np.min(axis_values)),
                    "max": float(np.max(axis_values)),
                    "mean": float(np.mean(axis_values)),
                    "std": float(np.std(axis_values)),
                }
            )
        summary["axis_stats"] = axis_stats

    return summary


@contextlib.contextmanager
def suppress_stdout() -> Iterable[None]:
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def summarize_subset(
    name: str,
    raw_data,
    *,
    params: RunParams,
    gridsize: int,
    group_count: int,
    neighbor_count: int,
    neighbor_seed: Optional[int],
) -> Dict[str, Any]:
    point_count = int(raw_data.diff3d.shape[0]) if raw_data.diff3d is not None else 0
    summary: Dict[str, Any] = {
        "subset": name,
        "point_count": point_count,
        "requested_groups": group_count,
    }
    try:
        with suppress_stdout():
            grouped = raw_data.generate_grouped_data(
                params.N,
                K=neighbor_count,
                nsamples=group_count,
                gridsize=gridsize,
                seed=neighbor_seed,
            )
    except Exception as exc:  # pragma: no cover - instrumentation guard
        summary["status"] = "error"
        summary["error"] = str(exc)
        return summary

    diffraction = grouped.get("diffraction")
    nn_indices = grouped.get("nn_indices")
    summary.update(
        {
            "status": "ok",
            "actual_groups": int(diffraction.shape[0]) if diffraction is not None else 0,
            "diffraction_shape": list(diffraction.shape) if diffraction is not None else None,
            "nn_indices_shape": list(nn_indices.shape) if nn_indices is not None else None,
            "coords_offsets": describe_array(grouped.get("coords_offsets")),
            "coords_relative": describe_array(grouped.get("coords_relative")),
        }
    )
    if nn_indices is not None and nn_indices.size > 0:
        summary["nn_indices_min"] = int(np.min(nn_indices))
        summary["nn_indices_max"] = int(np.max(nn_indices))
    return summary


def build_markdown(metadata: Mapping[str, Any], subsets: Iterable[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# Grouping Summary — {metadata['label']}")
    lines.append("")
    lines.append(f"- Scenario: `{metadata['scenario']}`")
    lines.append(f"- Snapshot: `{metadata['snapshot']}`")
    lines.append(f"- Timestamp: {metadata['timestamp']}")
    lines.append(f"- Gridsize: {metadata['parameters']['gridsize']}")
    lines.append(f"- Group count: {metadata['parameters']['group_count']}")
    lines.append(f"- Neighbor count: {metadata['parameters']['neighbor_count']}")
    lines.append(f"- Split fraction: {metadata['parameters']['split_fraction']}")
    overrides = metadata.get("overrides") or {}
    if overrides:
        lines.append("")
        lines.append("## Overrides")
        lines.append("")
        for key, value in overrides.items():
            lines.append(f"- `{key}` = {value}")
    lines.append("")
    lines.append("## Subset Stats")
    lines.append("")
    for subset in subsets:
        lines.append(f"### {subset['subset'].title()} split")
        lines.append(f"- Raw points: {subset['point_count']}")
        lines.append(f"- Requested groups: {subset['requested_groups']}")
        if subset.get("status") == "ok":
            lines.append(f"- Actual groups: {subset.get('actual_groups')}")
            lines.append(f"- Diffraction shape: {subset.get('diffraction_shape')}")
            lines.append(f"- nn_indices shape: {subset.get('nn_indices_shape')}")
            if subset.get("nn_indices_min") is not None:
                lines.append(
                    f"- nn_indices min/max: {subset.get('nn_indices_min')} / {subset.get('nn_indices_max')}"
                )
            offsets = subset.get("coords_offsets") or {}
            rel = subset.get("coords_relative") or {}
            lines.append(
                f"- coords_offsets min/max: {offsets.get('min')} / {offsets.get('max')}"
            )
            lines.append(
                f"- coords_offsets mean/std: {offsets.get('mean')} / {offsets.get('std')}"
            )
            axis_offsets = offsets.get("axis_stats") or []
            if axis_offsets:
                axis_parts = ", ".join(
                    f"axis{axis['axis']} min/max={axis.get('min')}/{axis.get('max')} mean/std={axis.get('mean')}/{axis.get('std')}"
                    for axis in axis_offsets
                )
                lines.append(f"- coords_offsets axis stats: {axis_parts}")
            lines.append(
                f"- coords_relative min/max: {rel.get('min')} / {rel.get('max')}"
            )
            lines.append(
                f"- coords_relative mean/std: {rel.get('mean')} / {rel.get('std')}"
            )
            rel_axis = rel.get("axis_stats") or []
            if rel_axis:
                rel_parts = ", ".join(
                    f"axis{axis['axis']} min/max={axis.get('min')}/{axis.get('max')} mean/std={axis.get('mean')}/{axis.get('std')}"
                    for axis in rel_axis
                )
                lines.append(f"- coords_relative axis stats: {rel_parts}")
        else:
            lines.append(f"- Status: ERROR — {subset.get('error')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_params, scenarios = load_snapshot(args.snapshot)
    if args.scenario not in scenarios:
        available = ", ".join(sorted(scenarios.keys()))
        raise ValueError(f"Scenario '{args.scenario}' not found. Available: {available}")
    scenario_entry = scenarios[args.scenario]
    spec = scenario_spec(scenario_entry)
    params = dataclass_with_overrides(run_params, args)

    label = args.label or spec.name
    overrides = collect_overrides(args)

    gridsize = args.gridsize or spec.gridsize
    if gridsize is None:
        raise ValueError("Gridsize is required via scenario defaults or --gridsize")

    scenario_group_default = scenario_entry.get("group_count") or params.group_count
    if args.group_count is not None:
        group_count = args.group_count
    else:
        group_count = int(scenario_group_default * max(1, args.group_multiplier or 1))
    neighbor_count = args.neighbor_count or scenario_entry.get("neighbor_count") or params.neighbor_count

    if group_count <= 0:
        raise ValueError("group_count must be > 0")
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be > 0")

    probe_mode = args.probe_mode or spec.probe_mode
    probe_scale = args.probe_scale or spec.probe_scale

    defaults = ModelConfig()
    probe_big = (
        args.probe_big
        if args.probe_big is not None
        else (spec.probe_big if spec.probe_big is not None else defaults.probe_big)
    )
    probe_mask = (
        args.probe_mask
        if args.probe_mask is not None
        else (spec.probe_mask if spec.probe_mask is not None else defaults.probe_mask)
    )

    image_multiplier = args.image_multiplier or 1
    total_images, derived_train, derived_test = derive_counts(params, gridsize, image_multiplier=image_multiplier)
    if args.total_images is not None:
        total_images = args.total_images
        derived_test = int(round(total_images * params.split_fraction))
        derived_train = total_images - derived_test

    with suppress_stdout():
        object_guess = make_lines_object(params.object_size, seed=params.object_seed)
    if probe_mode == "custom":
        with suppress_stdout():
            probe_guess = make_probe(params.N, mode="custom", path=CUSTOM_PROBE_PATH)
    else:
        with suppress_stdout():
            probe_guess = make_probe(params.N, mode="idealized")
    probe_guess = normalize_probe_guess(probe_guess, probe_scale=probe_scale, N=params.N)

    with suppress_stdout():
        raw_data = simulate_nongrid_raw_data(
            object_guess,
            probe_guess,
            N=params.N,
            n_images=total_images,
            nphotons=params.nphotons,
            seed=params.sim_seed,
            buffer=params.buffer,
        )
    with suppress_stdout():
        train_raw, test_raw = split_raw_data_by_axis(
            raw_data,
            split_fraction=params.split_fraction,
            axis="y",
        )

    bridge_config = TrainingConfig(
        model=ModelConfig(
            N=params.N,
            gridsize=gridsize,
            probe_scale=probe_scale,
            probe_big=probe_big,
            probe_mask=probe_mask,
        ),
        n_groups=group_count,
        neighbor_count=neighbor_count,
        nphotons=params.nphotons,
    )
    update_legacy_dict(legacy_params.cfg, bridge_config)

    neighbor_seed = args.neighbor_seed if args.neighbor_seed is not None else params.sim_seed

    train_summary = summarize_subset(
        "train",
        train_raw,
        params=params,
        gridsize=gridsize,
        group_count=group_count,
        neighbor_count=neighbor_count,
        neighbor_seed=neighbor_seed,
    )
    test_summary = summarize_subset(
        "test",
        test_raw,
        params=params,
        gridsize=gridsize,
        group_count=group_count,
        neighbor_count=neighbor_count,
        neighbor_seed=neighbor_seed,
    )

    metadata = {
        "label": label,
        "scenario": spec.name,
        "snapshot": str(args.snapshot),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "N": params.N,
            "gridsize": gridsize,
            "group_count": group_count,
            "neighbor_count": neighbor_count,
            "total_images": total_images,
            "train_count_estimate": derived_train,
            "test_count_estimate": derived_test,
            "split_fraction": params.split_fraction,
            "object_seed": params.object_seed,
            "sim_seed": params.sim_seed,
            "probe_mode": probe_mode,
            "probe_scale": probe_scale,
            "probe_big": probe_big,
            "probe_mask": probe_mask,
        },
        "overrides": overrides,
    }

    summary = {
        "metadata": metadata,
        "subsets": {
            "train": train_summary,
            "test": test_summary,
        },
    }

    ensure_directory(args.output_json)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    ensure_directory(args.output_markdown)
    args.output_markdown.write_text(build_markdown(metadata, [train_summary, test_summary]))


if __name__ == "__main__":  # pragma: no cover
    main()

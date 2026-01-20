#!/usr/bin/env python3
"""Report padded-size vs observed offsets for SIM-LINES reassembly limits."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ptycho import params as legacy_params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.tf_helper import reassemble_whole_object
from ptycho.workflows.components import _update_max_position_jitter_from_offsets
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
    parser.add_argument("--scenario", required=True, help="Scenario name from the snapshot JSON")
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
    parser.add_argument(
        "--group-limit",
        type=int,
        default=64,
        help="Slice the first group-limit samples before reassembly (default: 64)",
    )
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
    if args.group_limit is not None and args.group_limit != 64:
        overrides["group_limit"] = args.group_limit
    return overrides


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def suppress_stdout() -> Iterable[None]:
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def describe_axis_abs(array: np.ndarray) -> Dict[str, Any]:
    axis_stats: list[Dict[str, Any]] = []
    axis_abs_max: list[float] = []
    if array is None:
        return {"shape": None, "axis_stats": [], "axis_abs_max": [], "max_abs": None}
    arr = np.asarray(array)
    summary: Dict[str, Any] = {"shape": list(arr.shape)}
    axis_dim = arr.shape[-2] if arr.ndim >= 2 else 1
    moved = np.moveaxis(arr, -2, 0) if arr.ndim >= 2 else arr[None, ...]
    moved = moved.reshape(axis_dim, -1)
    for axis_index in range(axis_dim):
        axis_values = moved[axis_index]
        if axis_values.size == 0:
            stats = {
                "axis": axis_index,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "abs_max": None,
            }
            axis_stats.append(stats)
            axis_abs_max.append(0.0)
            continue
        stats = {
            "axis": axis_index,
            "min": float(np.min(axis_values)),
            "max": float(np.max(axis_values)),
            "mean": float(np.mean(axis_values)),
            "std": float(np.std(axis_values)),
            "abs_max": float(np.max(np.abs(axis_values))),
        }
        axis_stats.append(stats)
        axis_abs_max.append(stats["abs_max"])
    summary["axis_stats"] = axis_stats
    summary["axis_abs_max"] = axis_abs_max
    summary["max_abs"] = float(max(axis_abs_max)) if axis_abs_max else None
    return summary


def bootstrap_config(
    params: RunParams,
    gridsize: int,
    group_count: int,
    neighbor_count: int,
    probe_scale: float,
    *,
    probe_big: bool,
    probe_mask: bool,
) -> TrainingConfig:
    config = TrainingConfig(
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
    update_legacy_dict(legacy_params.cfg, config)
    return config


def simulate_raw_dataset(
    params: RunParams,
    gridsize: int,
    probe_mode: str,
    probe_scale: float,
    total_images: int,
) -> Tuple[Any, Any]:
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
    return train_raw, test_raw


def subset_summary(
    name: str,
    raw_data,
    *,
    params: RunParams,
    config: TrainingConfig,
    gridsize: int,
    group_count: int,
    neighbor_count: int,
    neighbor_seed: Optional[int],
    group_limit: int,
) -> Dict[str, Any]:
    point_count = int(raw_data.diff3d.shape[0]) if raw_data.diff3d is not None else 0
    summary: Dict[str, Any] = {"subset": name, "point_count": point_count, "requested_groups": group_count}
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
    actual_groups = int(diffraction.shape[0]) if diffraction is not None else 0
    summary["status"] = "ok"
    summary["actual_groups"] = actual_groups
    summary["diffraction_shape"] = list(diffraction.shape) if diffraction is not None else None
    _update_max_position_jitter_from_offsets(grouped, config)
    padded_size = int(legacy_params.get_padded_size())

    coords_offsets = grouped.get("coords_offsets")
    coords_relative = grouped.get("coords_relative")
    combined_offsets = None
    if coords_offsets is not None and coords_relative is not None:
        combined_offsets = coords_offsets + coords_relative
    summary["offsets"] = {
        "coords_offsets": describe_axis_abs(coords_offsets) if coords_offsets is not None else None,
        "coords_relative": describe_axis_abs(coords_relative) if coords_relative is not None else None,
        "combined": describe_axis_abs(combined_offsets) if combined_offsets is not None else None,
    }

    required_canvas = None
    fits_canvas = None
    padded_delta = None
    padded_ratio = None
    max_abs_axis = None
    combined_max_abs = None
    if summary["offsets"]["combined"] and summary["offsets"]["combined"]["max_abs"] is not None:
        combined_max_abs = float(summary["offsets"]["combined"]["max_abs"])
    if summary["offsets"]["coords_offsets"] and summary["offsets"]["coords_offsets"]["max_abs"] is not None:
        max_abs_axis = float(summary["offsets"]["coords_offsets"]["max_abs"])
        required_canvas = int(math.ceil(params.N + (2.0 * max_abs_axis)))
        if (required_canvas - params.N) % 2 != 0:
            required_canvas += 1
        fits_canvas = padded_size >= required_canvas
        padded_delta = padded_size - required_canvas
        padded_ratio = float(padded_size) / float(required_canvas)
    summary["canvas"] = {
        "required_canvas": required_canvas,
        "fits_canvas": fits_canvas,
        "padded_size": padded_size,
        "padded_minus_required": padded_delta,
        "padded_to_required_ratio": padded_ratio,
        "max_combined_abs_offset": combined_max_abs,
    }

    limit = min(group_limit, actual_groups)
    if limit <= 0 or combined_offsets is None:
        summary["reassembly"] = {
            "group_limit_used": limit,
            "padded_sum": None,
            "required_sum": None,
            "loss_fraction": None,
        }
        return summary

    patches = tf.ones((limit, params.N, params.N, gridsize**2), dtype=tf.float32)
    offsets_tensor = tf.convert_to_tensor(combined_offsets[:limit].astype(np.float32))
    padded_canvas = env_reassemble(patches, offsets_tensor, size=padded_size)
    padded_sum = float(tf.reduce_sum(tf.math.abs(padded_canvas)).numpy())
    required_size = required_canvas if required_canvas is not None else padded_size
    if required_size % 2 != 0:
        required_size += 1
    required_canvas_tensor = env_reassemble(patches, offsets_tensor, size=required_size)
    required_sum = float(tf.reduce_sum(tf.math.abs(required_canvas_tensor)).numpy())
    loss_fraction = None
    if required_sum > 0:
        loss_fraction = float(1.0 - (padded_sum / required_sum))
    summary["reassembly"] = {
        "group_limit_used": limit,
        "padded_sum": padded_sum,
        "required_sum": required_sum,
        "loss_fraction": loss_fraction,
        "reassembly_canvas_size": required_size,
    }
    return summary


def env_reassemble(patches: tf.Tensor, offsets: tf.Tensor, *, size: int) -> tf.Tensor:
    return reassemble_whole_object(patches, offsets, size=size, norm=False)


def build_markdown(metadata: Mapping[str, Any], subsets: Iterable[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# Reassembly Limits — {metadata['label']}")
    lines.append("")
    lines.append(f"- Scenario: `{metadata['scenario']}`")
    lines.append(f"- Snapshot: `{metadata['snapshot']}`")
    lines.append(f"- Timestamp: {metadata['timestamp']}")
    lines.append(f"- Gridsize: {metadata['parameters']['gridsize']}")
    lines.append(f"- Group count: {metadata['parameters']['group_count']}")
    lines.append(f"- Neighbor count: {metadata['parameters']['neighbor_count']}")
    lines.append(f"- Padded size: {metadata['padded_size']}")
    lines.append(f"- Legacy offset: {metadata['legacy_offset']}")
    lines.append(f"- Legacy max_position_jitter: {metadata['legacy_max_position_jitter']}")
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
        status = subset.get("status")
        if status != "ok":
            lines.append(f"- Status: ERROR — {subset.get('error')}")
            lines.append("")
            continue
        lines.append(f"- Actual groups: {subset.get('actual_groups')}")
        lines.append(f"- Group limit used: {subset.get('reassembly', {}).get('group_limit_used')}")
        offsets = subset.get("offsets", {})
        combined = offsets.get("combined") or {}
        axis_stats = combined.get("axis_stats") or []
        if axis_stats:
            formatted = ", ".join(
                f"axis{axis['axis']} abs_max={axis.get('abs_max'):.3f}"
                for axis in axis_stats
                if axis.get("abs_max") is not None
            )
            lines.append(f"- Combined offset abs max per axis: {formatted}")
        canvas = subset.get("canvas") or {}
        lines.append(
            f"- Required canvas: {canvas.get('required_canvas')} (max |offset|={canvas.get('max_combined_abs_offset')})"
        )
        lines.append(
            f"- Fits padded size ({metadata['padded_size']}): {canvas.get('fits_canvas')} "
            f"(Δ={canvas.get('padded_minus_required')}, ratio={canvas.get('padded_to_required_ratio')})"
        )
        reassembly = subset.get("reassembly") or {}
        if reassembly.get("padded_sum") is not None:
            loss_pct = None
            if reassembly.get("loss_fraction") is not None:
                loss_pct = 100.0 * reassembly["loss_fraction"]
            lines.append(
                f"- Reassembly sums (size={reassembly.get('reassembly_canvas_size')}): "
                f"padded={reassembly.get('padded_sum'):.2f}, "
                f"required={reassembly.get('required_sum'):.2f}, "
                f"loss={loss_pct if loss_pct is not None else None}%"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.group_limit <= 0:
        raise ValueError("--group-limit must be positive")
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

    total_images, derived_train, derived_test = derive_counts(
        params,
        gridsize,
        image_multiplier=args.image_multiplier or 1,
    )
    if args.total_images is not None:
        total_images = args.total_images
        derived_test = int(round(total_images * params.split_fraction))
        derived_train = total_images - derived_test

    train_raw, test_raw = simulate_raw_dataset(
        params,
        gridsize,
        probe_mode=probe_mode,
        probe_scale=probe_scale,
        total_images=total_images,
    )

    config = bootstrap_config(
        params,
        gridsize,
        group_count,
        neighbor_count,
        probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )
    neighbor_seed = args.neighbor_seed if args.neighbor_seed is not None else params.sim_seed

    subsets = []
    for subset_name, subset_raw in (("train", train_raw), ("test", test_raw)):
        subset = subset_summary(
            subset_name,
            subset_raw,
            params=params,
            config=config,
            gridsize=gridsize,
            group_count=group_count,
            neighbor_count=neighbor_count,
            neighbor_seed=neighbor_seed,
            group_limit=args.group_limit,
        )
        subsets.append(subset)
        canvas = subset.get("canvas") or {}
        reassembly = subset.get("reassembly") or {}
        print(
            f"[{subset_name}] required_canvas={canvas.get('required_canvas')} "
            f"padded_size={canvas.get('padded_size')} fits={canvas.get('fits_canvas')} "
            f"loss_fraction={reassembly.get('loss_fraction')}"
        )

    padded_size = int(legacy_params.get_padded_size())
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
        "padded_size": padded_size,
        "legacy_offset": legacy_params.cfg.get("offset"),
        "legacy_max_position_jitter": legacy_params.cfg.get("max_position_jitter"),
        "overrides": overrides,
    }

    summary = {
        "metadata": metadata,
        "subsets": {item["subset"]: item for item in subsets},
    }

    ensure_directory(args.output_json)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    ensure_directory(args.output_markdown)
    args.output_markdown.write_text(build_markdown(metadata, subsets))


if __name__ == "__main__":  # pragma: no cover
    main()

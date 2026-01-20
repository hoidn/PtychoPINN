#!/usr/bin/env python3
"""Phase C2 runner for SIM-LINES scenarios (simulate → split → train → infer)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptycho import loader, nbutils, params as legacy_params, tf_helper
from ptycho.config.config import InferenceConfig, ModelConfig
from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
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
    build_training_config,
    derive_counts,
    run_training,
    save_training_bundle,
)

DEFAULT_SNAPSHOT = Path(
    "plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json"
)

STABLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "gs1_ideal": {
        "label": "stable_profile_gs1_ideal",
        "base_total_images": 512,
        "group_count": 256,
        "batch_size": 8,
    },
    "gs2_ideal": {
        "label": "stable_profile_gs2_ideal",
        "base_total_images": 256,
        "group_count": 128,
        "batch_size": 4,
        "neighbor_count": 4,
    },
}


def str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _namespace_to_dict(namespace: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse.Namespace to JSON-serializable dict."""

    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    return {key: _convert(val) for key, val in vars(namespace).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="Scenario name from the snapshot JSON")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to the sim_lines params snapshot JSON",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for run artifacts")
    parser.add_argument("--nepochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument(
        "--group-limit",
        type=int,
        default=64,
        help="Limit the number of grouped samples used during inference (default: 64)",
    )
    parser.add_argument("--image-multiplier", type=int, default=1, help="Scale total images before splitting")
    parser.add_argument("--group-multiplier", type=int, default=1, help="Scale requested group count")
    parser.add_argument("--group-count", type=int, help="Override scenario group count before multiplier")
    parser.add_argument("--gridsize", type=int, help="Override scenario gridsize")
    parser.add_argument("--probe-mode", choices={"custom", "idealized"}, help="Override probe mode")
    parser.add_argument("--probe-scale", type=float, help="Override probe scale factor")
    parser.add_argument("--probe-big", type=str2bool, help="Override probe_big flag (true/false)")
    parser.add_argument("--probe-mask", type=str2bool, help="Override probe_mask flag (true/false)")
    parser.add_argument("--object-seed", type=int, help="Override object seed")
    parser.add_argument("--sim-seed", type=int, help="Override simulation seed")
    parser.add_argument("--buffer", type=float, help="Override simulation buffer size")
    parser.add_argument("--neighbor-count", type=int, help="Override neighbor count for grouping")
    parser.add_argument("--nphotons", type=float, help="Override photon count")
    parser.add_argument("--split-fraction", type=float, help="Override train/test split fraction")
    parser.add_argument("--amp-vmax", type=float, help="Optional fixed vmax for amplitude PNGs")
    parser.add_argument(
        "--base-total-images",
        type=int,
        help="Override RunParams.base_total_images before gridsize scaling",
    )
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    return parser.parse_args()


def load_snapshot(
    snapshot_path: Path,
) -> Tuple[RunParams, Dict[str, Mapping[str, Any]], Path]:
    data = json.loads(snapshot_path.read_text())
    run_params = RunParams(**data["run_params"])
    scenarios: Dict[str, Mapping[str, Any]] = {}
    for entry in data.get("scenarios", []):
        name = entry.get("name") or entry.get("inputs", {}).get("name")
        if name:
            scenarios[name] = entry
    if not scenarios:
        raise ValueError(f"No scenarios found in snapshot: {snapshot_path}")
    custom_probe_path = Path(data.get("custom_probe_path", str(CUSTOM_PROBE_PATH)))
    return run_params, scenarios, custom_probe_path


def _warn_profile_override(field: str, preferred: Any, override: Any, label: str) -> None:
    flag = "--" + field.replace("_", "-")
    print(
        f"[runner][profile][warn] manual override {flag}={override} "
        f"disables {label} default {preferred}"
    )


def apply_stable_profile_if_needed(
    scenario_name: str,
    args: argparse.Namespace,
    params: RunParams,
    default_group_count: int,
) -> Tuple[RunParams, Dict[str, Any]]:
    profile = STABLE_PROFILES.get(scenario_name)
    if not profile:
        return params, {}
    label = profile.get("label", scenario_name)
    applied: Dict[str, Any] = {}
    skipped: Dict[str, Dict[str, Any]] = {}
    new_params = params
    profile_group_count: int | None = None
    profile_batch_size: int | None = None

    for field in ("base_total_images", "neighbor_count"):
        if field not in profile:
            continue
        preferred = profile[field]
        arg_value = getattr(args, field, None)
        if arg_value is None:
            new_params = dataclasses.replace(new_params, **{field: preferred})
            applied[field] = preferred
        else:
            skipped[field] = {"preferred": preferred, "override": arg_value, "reason": "cli_override"}
            _warn_profile_override(field, preferred, arg_value, label)

    if "group_count" in profile:
        preferred_group_count = profile["group_count"]
        if args.group_count is None:
            profile_group_count = preferred_group_count
            applied["group_count"] = preferred_group_count
        else:
            skipped["group_count"] = {
                "preferred": preferred_group_count,
                "override": args.group_count,
                "reason": "cli_override",
            }
            _warn_profile_override("group_count", preferred_group_count, args.group_count, label)

    if "batch_size" in profile:
        preferred_batch = profile["batch_size"]
        if args.batch_size is None:
            profile_batch_size = preferred_batch
            applied["batch_size"] = preferred_batch
        else:
            skipped["batch_size"] = {
                "preferred": preferred_batch,
                "override": args.batch_size,
                "reason": "cli_override",
            }
            _warn_profile_override("batch_size", preferred_batch, args.batch_size, label)

    metadata = {
        "name": scenario_name,
        "label": label,
        "defaults": {k: v for k, v in profile.items() if k != "label"},
        "applied": applied,
        "skipped": skipped,
        "active": bool(applied),
        "effective_group_count": profile_group_count or default_group_count,
        "effective_batch_size": profile_batch_size,
    }
    if applied:
        print(f"[runner][profile] Applied {label}: {applied}")
    return new_params, metadata


def scenario_spec_from_entry(entry: Mapping[str, Any], args: argparse.Namespace) -> ScenarioSpec:
    defaults = entry.get("defaults", {})
    inputs = entry.get("inputs", {})
    gridsize = args.gridsize if args.gridsize is not None else inputs.get("gridsize")
    probe_mode = args.probe_mode or inputs.get("probe_mode") or "custom"
    probe_scale = args.probe_scale if args.probe_scale is not None else inputs.get("probe_scale") or 1.0
    probe_big = (
        args.probe_big
        if args.probe_big is not None
        else inputs.get("probe_big", defaults.get("probe_big"))
    )
    probe_mask = (
        args.probe_mask
        if args.probe_mask is not None
        else inputs.get("probe_mask", defaults.get("probe_mask"))
    )
    return ScenarioSpec(
        name=entry.get("name") or inputs.get("name") or "unknown",
        gridsize=gridsize,
        probe_mode=probe_mode,
        probe_scale=probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )


def replace_run_params(params: RunParams, args: argparse.Namespace) -> RunParams:
    updates: Dict[str, Any] = {}
    if args.object_seed is not None:
        updates["object_seed"] = args.object_seed
    if args.sim_seed is not None:
        updates["sim_seed"] = args.sim_seed
    if args.buffer is not None:
        updates["buffer"] = args.buffer
    if args.nphotons is not None:
        updates["nphotons"] = args.nphotons
    if args.neighbor_count is not None:
        updates["neighbor_count"] = args.neighbor_count
    if args.split_fraction is not None:
        updates["split_fraction"] = args.split_fraction
    if args.base_total_images is not None:
        updates["base_total_images"] = args.base_total_images
    return dataclasses.replace(params, **updates) if updates else params


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_array_stats(array: np.ndarray) -> Dict[str, float | int]:
    stats = {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "nan_count": int(np.isnan(array).sum()),
    }
    return stats


def save_png(data: np.ndarray, path: Path, title: str, cmap: str, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def describe_offsets(global_offsets: np.ndarray) -> Dict[str, Any]:
    offsets = np.asarray(global_offsets)
    if offsets.ndim != 4:
        return {"axis_max_abs": [None, None], "max_abs": None}
    squeezed = np.squeeze(offsets, axis=1)  # (B, 2, C)
    reshaped = np.transpose(squeezed, (0, 2, 1)).reshape(-1, 2)
    abs_offsets = np.abs(reshaped)
    axis_max = abs_offsets.max(axis=0)
    max_abs = float(abs_offsets.max())
    return {"axis_max_abs": [float(axis_max[0]), float(axis_max[1])], "max_abs": max_abs}


def run_inference_and_reassemble(
    test_raw,
    scenario: ScenarioSpec,
    params: RunParams,
    model_dir: Path,
    group_count: int,
    group_limit: int,
    custom_probe_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    infer_config = InferenceConfig(
        model=ModelConfig(
            N=params.N,
            gridsize=scenario.gridsize,
            probe_scale=scenario.probe_scale,
            probe_big=scenario.probe_big,
            probe_mask=scenario.probe_mask,
        ),
        model_path=model_dir,
        test_data_file=custom_probe_path,
        n_groups=group_count,
        neighbor_count=params.neighbor_count,
        backend="tensorflow",
    )
    model, params_dict = load_inference_bundle_with_backend(model_dir, infer_config)
    nsamples = min(group_count, group_limit) if group_limit else group_count
    grouped = test_raw.generate_grouped_data(
        params_dict.get("N", params.N),
        K=params.neighbor_count,
        nsamples=nsamples,
        gridsize=params_dict.get("gridsize", scenario.gridsize),
    )
    container = loader.load(lambda: grouped, test_raw.probeGuess, which=None, create_split=False)
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(container, diffraction_to_obj=model)
    obj_image = tf_helper.reassemble_position(obj_tensor_full, global_offsets, M=params.reassemble_M)
    return np.abs(obj_image), np.angle(obj_image), global_offsets


def save_stats(
    output_dir: Path,
    amplitude: np.ndarray,
    phase: np.ndarray,
    offsets_summary: Dict[str, Any],
    padded_size: int,
    N_value: int,
) -> Dict[str, Any]:
    amplitude_stats = compute_array_stats(amplitude)
    phase_stats = compute_array_stats(phase)
    required_canvas = int(math.ceil(N_value + 2 * (offsets_summary.get("max_abs") or 0.0)))
    stats = {
        "amplitude": amplitude_stats,
        "phase": phase_stats,
        "offsets": offsets_summary,
        "padded_size": int(padded_size),
        "required_canvas": required_canvas,
        "fits_canvas": required_canvas <= padded_size,
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    ensure_dir(output_dir)
    snapshot_params, scenario_entries, custom_probe_path = load_snapshot(args.snapshot)
    if args.scenario not in scenario_entries:
        raise ValueError(f"Scenario '{args.scenario}' not found in snapshot {args.snapshot}")
    scenario_entry = scenario_entries[args.scenario]
    scenario = scenario_spec_from_entry(scenario_entry, args)
    params = replace_run_params(snapshot_params, args)
    if scenario.gridsize is None:
        raise ValueError("Scenario gridsize is undefined; provide --gridsize")

    scenario_dir = output_dir
    train_dir = scenario_dir / "train_outputs"
    inference_dir = scenario_dir / "inference_outputs"
    ensure_dir(train_dir)
    ensure_dir(inference_dir)

    group_count_snapshot = scenario_entry.get("group_count") or params.group_count
    params, profile_metadata = apply_stable_profile_if_needed(
        scenario.name,
        args,
        params,
        group_count_snapshot,
    )
    profile_group_count = profile_metadata.get("effective_group_count", group_count_snapshot)
    profile_batch_size = profile_metadata.get("effective_batch_size")

    group_count = args.group_count if args.group_count is not None else profile_group_count
    group_count = int(group_count * args.group_multiplier)

    total_images, train_count, test_count = derive_counts(
        params,
        scenario.gridsize,
        image_multiplier=args.image_multiplier,
    )
    if train_count <= 0 or test_count <= 0:
        raise ValueError("Train/test counts must be positive after derive_counts")
    if group_count > min(train_count, test_count):
        raise ValueError("group_count must be <= min(train_count, test_count)")

    print(
        f"[runner] Scenario={scenario.name} gridsize={scenario.gridsize} "
        f"probe_mode={scenario.probe_mode} probe_scale={scenario.probe_scale}"
    )
    print(
        f"[runner] total_images={total_images} train_count={train_count} test_count={test_count} "
        f"group_count={group_count} nepochs={args.nepochs}"
    )

    object_guess = make_lines_object(params.object_size, seed=params.object_seed)
    if scenario.probe_mode == "custom":
        probe_guess = make_probe(params.N, mode="custom", path=custom_probe_path)
    elif scenario.probe_mode == "idealized":
        probe_guess = make_probe(params.N, mode="idealized")
    else:
        raise ValueError(f"Unsupported probe mode: {scenario.probe_mode}")
    probe_guess = normalize_probe_guess(probe_guess, probe_scale=scenario.probe_scale, N=params.N)

    sim_buffer = params.buffer if args.buffer is None else args.buffer
    raw_data = simulate_nongrid_raw_data(
        object_guess,
        probe_guess,
        N=params.N,
        n_images=total_images,
        nphotons=params.nphotons,
        seed=params.sim_seed,
        buffer=sim_buffer,
    )
    train_raw, test_raw = split_raw_data_by_axis(raw_data, split_fraction=params.split_fraction, axis="y")

    train_config = build_training_config(
        params=params,
        gridsize=scenario.gridsize,
        group_count=group_count,
        output_dir=train_dir,
        nepochs=args.nepochs,
        probe_scale=scenario.probe_scale,
        probe_big=scenario.probe_big,
        probe_mask=scenario.probe_mask,
    )
    if args.batch_size is not None:
        train_config = dataclasses.replace(train_config, batch_size=args.batch_size)
    elif profile_batch_size is not None:
        train_config = dataclasses.replace(train_config, batch_size=profile_batch_size)

    start_time = time.time()
    run_training(train_raw, test_raw, train_config)
    save_training_bundle(train_dir)
    train_elapsed = time.time() - start_time
    print(f"[runner] Training complete in {train_elapsed:.2f}s; bundle saved to {train_dir}")

    amp, phase, global_offsets = run_inference_and_reassemble(
        test_raw=test_raw,
        scenario=scenario,
        params=params,
        model_dir=train_dir,
        group_count=group_count,
        group_limit=args.group_limit,
        custom_probe_path=custom_probe_path,
    )
    amplitude_path = inference_dir / "amplitude.npy"
    phase_path = inference_dir / "phase.npy"
    np.save(amplitude_path, amp.astype(np.float32))
    np.save(phase_path, phase.astype(np.float32))

    offsets_summary = describe_offsets(global_offsets)
    stats = save_stats(
        inference_dir,
        amp,
        phase,
        offsets_summary,
        padded_size=int(legacy_params.get_padded_size()),
        N_value=params.N,
    )

    amp_vmin = 0.0
    amp_vmax = args.amp_vmax if args.amp_vmax is not None else max(stats["amplitude"]["max"], 1e-9)
    phase_vmin = -math.pi
    phase_vmax = math.pi

    save_png(
        amp,
        inference_dir / "amplitude.png",
        f"{scenario.name} amplitude (vmax={amp_vmax:.3f})",
        cmap="magma",
        vmin=amp_vmin,
        vmax=amp_vmax,
    )
    save_png(
        phase,
        inference_dir / "phase.png",
        f"{scenario.name} phase",
        cmap="twilight",
        vmin=phase_vmin,
        vmax=phase_vmax,
    )

    metadata = {
        "scenario": scenario.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nepochs": args.nepochs,
        "group_limit": args.group_limit,
        "group_count": group_count,
        "total_images": total_images,
        "train_count": train_count,
        "test_count": test_count,
        "buffer": sim_buffer,
        "object_seed": params.object_seed,
        "sim_seed": params.sim_seed,
        "probe_mode": scenario.probe_mode,
        "probe_scale": scenario.probe_scale,
        "probe_big": scenario.probe_big,
        "probe_mask": scenario.probe_mask,
        "nphotons": params.nphotons,
        "neighbor_count": params.neighbor_count,
        "split_fraction": params.split_fraction,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "amp_png_range": {"vmin": amp_vmin, "vmax": amp_vmax},
        "cli_args": _namespace_to_dict(args),
        "artifacts": {
            "train_dir": str(train_dir),
            "inference_dir": str(inference_dir),
            "amplitude_npy": str(amplitude_path),
            "phase_npy": str(phase_path),
            "amplitude_png": str(inference_dir / "amplitude.png"),
            "phase_png": str(inference_dir / "phase.png"),
            "stats_json": str(inference_dir / "stats.json"),
        },
    }
    if profile_metadata:
        metadata["profile"] = profile_metadata
    (scenario_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[runner] Inference complete; stats saved to {inference_dir / 'stats.json'}")
    print(f"[runner] Outputs written to {scenario_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surfaced in CLI
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        raise

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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptycho import loader, nbutils, params as legacy_params, tf_helper
from ptycho.config.config import InferenceConfig, ModelConfig, update_legacy_dict
from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
from ptycho.model import _get_log_scale
from scripts.simulation.synthetic_helpers import (
    make_lines_object,
    make_probe,
    normalize_probe_guess,
    simulate_nongrid_raw_data,
    split_raw_data_by_axis,
)
from scripts.studies.sim_lines_4x.pipeline import (
    CUSTOM_PROBE_PATH,
    PREDICTION_SCALE_CHOICES,
    RunParams,
    ScenarioSpec,
    build_training_config,
    derive_counts,
    determine_prediction_scale,
    format_prediction_scale_note,
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
    parser.add_argument(
        "--prediction-scale-source",
        choices=PREDICTION_SCALE_CHOICES,
        default="none",
        help="Prediction scaling strategy (none, recorded, least_squares).",
    )
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


def _ensure_numpy(value: Any) -> np.ndarray:
    """Return a NumPy view of ``value`` even if it is a TensorFlow tensor."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:  # pragma: no cover - diagnostics only
            pass
    return np.asarray(value)


def compute_array_stats(array: np.ndarray) -> Dict[str, float | int | None]:
    arr = _ensure_numpy(array)
    nan_count = int(np.isnan(arr).sum())
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": nan_count,
        }
    finite_vals = arr[finite_mask]
    stats = {
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "mean": float(np.mean(finite_vals)),
        "std": float(np.std(finite_vals)),
        "nan_count": nan_count,
    }
    return stats


def format_array_stats(array: np.ndarray) -> Dict[str, Any]:
    arr = _ensure_numpy(array)
    payload = compute_array_stats(arr)
    payload.update(
        {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_count": int(np.isfinite(arr).sum()),
            "total_count": int(arr.size),
        }
    )
    return payload


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def extract_intensity_scaler_state() -> Dict[str, Any]:
    """Extract IntensityScaler state for diagnostics.

    Per `specs/spec-ptycho-core.md §Normalization Invariants`:
    - Captures the trained log_scale tf.Variable value
    - Computes exp(log_scale) to show the effective intensity scale multiplier
    - Records the trainable flag from params.cfg
    - Records the original params.cfg intensity_scale for comparison

    Returns dict with:
        log_scale_value: The current log_scale variable value
        exp_log_scale: exp(log_scale) - the effective scaling factor
        trainable: Whether the intensity_scale is trainable
        params_cfg_intensity_scale: The value from legacy_params.cfg['intensity_scale']
        delta: Difference between exp(log_scale) and params.cfg value
        ratio: Ratio of exp(log_scale) to params.cfg value
    """
    try:
        log_scale_var = _get_log_scale()
        log_scale_value = float(log_scale_var.numpy())
        exp_log_scale = float(np.exp(log_scale_value))
        trainable = bool(log_scale_var.trainable)
    except Exception as e:
        print(f"[runner][scaler_state][warn] Could not extract log_scale: {e}")
        log_scale_value = None
        exp_log_scale = None
        trainable = None

    params_cfg_scale = legacy_params.cfg.get("intensity_scale")
    params_cfg_trainable = legacy_params.cfg.get("intensity_scale.trainable")

    delta = None
    ratio = None
    if exp_log_scale is not None and params_cfg_scale is not None:
        try:
            delta = exp_log_scale - float(params_cfg_scale)
            ratio = exp_log_scale / float(params_cfg_scale) if float(params_cfg_scale) != 0 else None
        except (TypeError, ValueError):
            pass

    return {
        "log_scale_value": log_scale_value,
        "exp_log_scale": exp_log_scale,
        "trainable": trainable,
        "params_cfg_intensity_scale": _serialize_scalar(params_cfg_scale),
        "params_cfg_trainable": params_cfg_trainable,
        "delta": delta,
        "ratio": ratio,
        "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants",
    }


def record_intensity_stage(
    stages: list[Dict[str, Any]],
    name: str,
    array: np.ndarray,
    metadata: Dict[str, Any] | None = None,
) -> None:
    stats = format_array_stats(array)
    sanitized_stats = {key: _serialize_scalar(val) for key, val in stats.items()}
    entry: Dict[str, Any] = {"name": name, "stats": sanitized_stats}
    if metadata:
        entry["metadata"] = metadata
    stages.append(entry)


def _format_stage_stats_markdown(stage: Mapping[str, Any]) -> list[str]:
    stats = stage.get("stats", {})
    metadata = stage.get("metadata", {})
    fmt = lambda key: _format_optional(stats.get(key))
    lines = [f"### {stage.get('name', 'unknown')}\n"]
    if metadata:
        for key, value in metadata.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "| Metric | Value |",
            "| --- | --- |",
            f"| shape | {stats.get('shape')} |",
            f"| dtype | {stats.get('dtype')} |",
            f"| min | {fmt('min')} |",
            f"| max | {fmt('max')} |",
            f"| mean | {fmt('mean')} |",
            f"| std | {fmt('std')} |",
            f"| finite_count | {stats.get('finite_count')} |",
            f"| total_count | {stats.get('total_count')} |",
            f"| nan_count | {stats.get('nan_count')} |",
            "",
        ]
    )
    return lines


STAGE_ORDER = [
    "raw_diffraction",
    "grouped_diffraction",
    "grouped_X_full",
    "container_X",
]

STAGE_LABELS = {
    "raw_diffraction": "Raw diffraction",
    "grouped_diffraction": "Grouped diffraction",
    "grouped_X_full": "Grouped X (normalized)",
    "container_X": "Container X",
}

RATIO_TRANSITIONS = [
    ("raw_diffraction", "grouped_diffraction", "raw_to_grouped"),
    ("grouped_diffraction", "grouped_X_full", "grouped_to_normalized"),
    ("grouped_X_full", "container_X", "normalized_to_container"),
]


def _extract_stage_means(stages: list[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract mean values from recorded stages."""
    means: Dict[str, Optional[float]] = {}
    for stage in stages:
        name = stage.get("name")
        stats = stage.get("stats") or {}
        mean = stats.get("mean")
        if name and mean is not None:
            means[name] = float(mean)
    return means


def _compute_stage_ratios(
    stage_means: Dict[str, Optional[float]]
) -> Tuple[Dict[str, Optional[float]], Optional[Dict[str, Any]]]:
    """Compute stage-to-stage ratios and identify the largest drop."""
    ratios: Dict[str, Optional[float]] = {}
    transitions: list[Dict[str, Any]] = []

    for from_stage, to_stage, ratio_key in RATIO_TRANSITIONS:
        from_mean = stage_means.get(from_stage)
        to_mean = stage_means.get(to_stage)
        ratio: Optional[float] = None
        if from_mean is not None and to_mean is not None and abs(from_mean) > 1e-12:
            ratio = to_mean / from_mean
        ratios[ratio_key] = ratio
        transitions.append({
            "key": ratio_key,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "from_mean": from_mean,
            "to_mean": to_mean,
            "ratio": ratio,
        })

    # Identify largest drop (ratio < 1 with smallest value)
    valid_drops = [
        t for t in transitions
        if t["ratio"] is not None and t["ratio"] < 1.0
    ]
    if valid_drops:
        largest_drop = min(valid_drops, key=lambda t: t["ratio"])
    else:
        largest_drop = None

    return ratios, largest_drop


def _compute_normalize_gain(stages: list[Dict[str, Any]]) -> Optional[float]:
    """Compute the normalize_data gain as grouped_X_full.mean / grouped_diffraction.mean."""
    stage_means = _extract_stage_means(stages)
    grouped_mean = stage_means.get("grouped_diffraction")
    normalized_mean = stage_means.get("grouped_X_full")
    if grouped_mean is not None and normalized_mean is not None and abs(grouped_mean) > 1e-12:
        return normalized_mean / grouped_mean
    return None


def write_intensity_stats_outputs(
    stages: list[Dict[str, Any]],
    bundle_intensity_scale: Any,
    legacy_intensity_scale: Any,
    scenario_dir: Path,
    intensity_scaler_state: Optional[Dict[str, Any]] = None,
    training_container_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bundle_scale_serialized = _serialize_scalar(bundle_intensity_scale)
    legacy_scale_serialized = _serialize_scalar(legacy_intensity_scale)
    scale_delta = None
    if bundle_intensity_scale is not None and legacy_intensity_scale is not None:
        try:
            scale_delta = _serialize_scalar(bundle_intensity_scale - legacy_intensity_scale)
        except TypeError:
            scale_delta = None

    # Compute stage ratios and normalize_data gain
    stage_means = _extract_stage_means(stages)
    ratios, largest_drop = _compute_stage_ratios(stage_means)
    normalize_gain = _compute_normalize_gain(stages)

    payload: Dict[str, Any] = {
        "bundle_intensity_scale": bundle_scale_serialized,
        "legacy_params_intensity_scale": legacy_scale_serialized,
        "scale_delta": scale_delta,
        "stages": stages,
        "stage_means": stage_means,
        "ratios": ratios,
        "largest_drop": largest_drop,
        "normalize_gain": normalize_gain,
    }

    # Add IntensityScaler state if provided (D4 architecture diagnostics)
    if intensity_scaler_state is not None:
        payload["intensity_scaler_state"] = intensity_scaler_state

    # Add training container X stats if provided
    if training_container_stats is not None:
        payload["training_container_stats"] = training_container_stats

    json_path = scenario_dir / "intensity_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    md_lines = [
        "# Intensity Statistics",
        "",
        "**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`",
        "",
        f"- Bundle intensity_scale: {payload['bundle_intensity_scale']}",
        f"- Legacy params intensity_scale: {payload['legacy_params_intensity_scale']}",
        f"- bundle minus legacy delta: {payload['scale_delta']}",
        f"- normalize_data gain: {_format_optional(normalize_gain)}",
        f"- Stage count: {len(stages)}",
        "",
    ]

    # Add stage means table
    md_lines.extend([
        "## Stage Means",
        "",
        "| Stage | Mean |",
        "| --- | ---: |",
    ])
    for stage_key in STAGE_ORDER:
        label = STAGE_LABELS.get(stage_key, stage_key)
        mean_val = stage_means.get(stage_key)
        md_lines.append(f"| {label} | {_format_optional(mean_val)} |")
    md_lines.append("")

    # Add ratios table
    md_lines.extend([
        "## Stage Ratios",
        "",
        "| Transition | Ratio |",
        "| --- | ---: |",
    ])
    for from_stage, to_stage, ratio_key in RATIO_TRANSITIONS:
        from_label = STAGE_LABELS.get(from_stage, from_stage)
        to_label = STAGE_LABELS.get(to_stage, to_stage)
        ratio_val = ratios.get(ratio_key)
        md_lines.append(f"| {from_label} → {to_label} | {_format_optional(ratio_val)} |")
    md_lines.append("")

    # Add largest drop marker
    if largest_drop:
        from_label = STAGE_LABELS.get(largest_drop["from_stage"], largest_drop["from_stage"])
        to_label = STAGE_LABELS.get(largest_drop["to_stage"], largest_drop["to_stage"])
        md_lines.extend([
            "## Largest Drop",
            "",
            f"**{from_label} → {to_label}** (ratio={_format_optional(largest_drop['ratio'])})",
            "",
            "Per `specs/spec-ptycho-core.md §Normalization Invariants`, symmetry SHALL hold:",
            "- Training inputs: `X_scaled = s · X`",
            "- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)",
            "",
            "If the ratio deviates significantly from 1.0, investigate whether the normalization",
            "pipeline preserves the intensity_scale symmetry mandated by the spec.",
            "",
        ])

    # Add IntensityScaler state section (D4 architecture diagnostics)
    if intensity_scaler_state:
        md_lines.extend([
            "## IntensityScaler State",
            "",
            "**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`",
            "",
            "Per the architecture, `IntensityScaler` and `IntensityScaler_inv` layers use a shared `log_scale` tf.Variable.",
            "The effective scaling factor is `exp(log_scale)`. If this diverges from the recorded bundle/params.cfg value,",
            "it may indicate double-scaling or a training-time drift that contributes to amplitude bias.",
            "",
            "| Property | Value |",
            "| --- | ---: |",
            f"| log_scale (raw) | {_format_optional(intensity_scaler_state.get('log_scale_value'))} |",
            f"| exp(log_scale) | {_format_optional(intensity_scaler_state.get('exp_log_scale'))} |",
            f"| trainable | {intensity_scaler_state.get('trainable')} |",
            f"| params.cfg intensity_scale | {_format_optional(intensity_scaler_state.get('params_cfg_intensity_scale'))} |",
            f"| params.cfg trainable | {intensity_scaler_state.get('params_cfg_trainable')} |",
            f"| delta (exp - cfg) | {_format_optional(intensity_scaler_state.get('delta'))} |",
            f"| ratio (exp / cfg) | {_format_optional(intensity_scaler_state.get('ratio'))} |",
            "",
        ])

    # Add training container X stats section
    if training_container_stats:
        md_lines.extend([
            "## Training Container X Stats",
            "",
            "Statistics of the training container's X tensor after normalization.",
            "Per `specs/spec-ptycho-core.md §Normalization Invariants`, this should reflect `X_scaled = s · X`.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| shape | {training_container_stats.get('shape')} |",
            f"| dtype | {training_container_stats.get('dtype')} |",
            f"| min | {_format_optional(training_container_stats.get('min'))} |",
            f"| max | {_format_optional(training_container_stats.get('max'))} |",
            f"| mean | {_format_optional(training_container_stats.get('mean'))} |",
            f"| std | {_format_optional(training_container_stats.get('std'))} |",
            f"| nan_count | {training_container_stats.get('nan_count', 0)} |",
            "",
        ])

    # Add per-stage stats
    md_lines.append("## Per-Stage Statistics")
    md_lines.append("")
    for stage in stages:
        md_lines.extend(_format_stage_stats_markdown(stage))

    md_path = scenario_dir / "intensity_stats.md"
    md_path.write_text("\n".join(md_lines))

    result: Dict[str, Any] = {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "stages": stages,
        "stage_means": stage_means,
        "ratios": ratios,
        "largest_drop": largest_drop,
        "normalize_gain": normalize_gain,
        "bundle_intensity_scale": bundle_scale_serialized,
        "legacy_params_intensity_scale": legacy_scale_serialized,
        "scale_delta": scale_delta,
        "intensity_scale": bundle_scale_serialized or legacy_scale_serialized,
    }
    if intensity_scaler_state is not None:
        result["intensity_scaler_state"] = intensity_scaler_state
    if training_container_stats is not None:
        result["training_container_stats"] = training_container_stats
    return result


def summarize_bias(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float | None]:
    """Summarize prediction minus ground-truth bias."""
    pred_vals = np.asarray(pred, dtype=float)
    truth_vals = np.asarray(truth, dtype=float)
    diff = pred_vals - truth_vals
    mask = np.isfinite(diff)
    if not mask.any():
        return {"mean": None, "median": None, "p05": None, "p95": None}
    diff_use = diff[mask]
    return {
        "mean": float(np.mean(diff_use)),
        "median": float(np.median(diff_use)),
        "p05": float(np.percentile(diff_use, 5)),
        "p95": float(np.percentile(diff_use, 95)),
    }


def save_png(data: np.ndarray, path: Path, title: str, cmap: str, vmin: float, vmax: float) -> None:
    """Persist a 2D heatmap using matplotlib."""
    array2d = data
    if data.ndim > 2 and data.shape[-1] == 1:
        array2d = np.squeeze(data, axis=-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(array2d, cmap=cmap, vmin=vmin, vmax=vmax)
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


def center_crop(array: np.ndarray, size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Symmetrically crop an array to ``size`` keeping parity notes."""
    data = np.asarray(array)
    if size <= 0:
        raise ValueError(f"Center crop size must be positive (got {size})")
    if data.ndim < 2:
        raise ValueError("Center crop requires at least 2 dimensions")
    height, width = data.shape[0], data.shape[1]
    if size > height or size > width:
        raise ValueError(
            f"Center crop size {size} exceeds array dimensions {(height, width)}"
        )
    start_y = (height - size) // 2
    start_x = (width - size) // 2
    end_y = start_y + size
    end_x = start_x + size
    parity_warning = ((height - size) % 2 != 0) or ((width - size) % 2 != 0)
    if parity_warning:
        print(
            "[runner][crop][warn] Padded canvas minus object_size is odd; "
            f"cropping uses floor start (array_shape={(height, width)}, size={size})"
        )
    slices = (slice(start_y, end_y), slice(start_x, end_x))
    if data.ndim > 2:
        slices = slices + (slice(None),) * (data.ndim - 2)
    cropped = data[slices]
    metadata = {
        "input_shape": list(data.shape),
        "target_size": int(size),
        "start": [int(start_y), int(start_x)],
        "end": [int(end_y), int(end_x)],
        "parity_warning": parity_warning,
    }
    return cropped, metadata


def save_ground_truth_artifacts(
    object_guess: np.ndarray,
    scenario_dir: Path,
    amp_bounds: Tuple[float, float],
    target_size: int,
) -> Tuple[Dict[str, str], np.ndarray, np.ndarray]:
    """Persist ground-truth amplitude/phase arrays + PNGs."""
    amp_truth = np.abs(object_guess).astype(np.float32, copy=False)
    phase_truth = np.angle(object_guess).astype(np.float32, copy=False)
    amp_truth, _ = center_crop(amp_truth, target_size)
    phase_truth, _ = center_crop(phase_truth, target_size)

    amp_path = scenario_dir / "ground_truth_amp.npy"
    phase_path = scenario_dir / "ground_truth_phase.npy"
    np.save(amp_path, amp_truth)
    np.save(phase_path, phase_truth)

    amp_png = scenario_dir / "ground_truth_amp.png"
    phase_png = scenario_dir / "ground_truth_phase.png"
    amp_vmin, amp_vmax = amp_bounds
    save_png(
        amp_truth,
        amp_png,
        "Ground truth amplitude",
        cmap="magma",
        vmin=amp_vmin,
        vmax=amp_vmax,
    )
    save_png(
        phase_truth,
        phase_png,
        "Ground truth phase",
        cmap="twilight",
        vmin=-math.pi,
        vmax=math.pi,
    )
    artifact_map = {
        "amplitude_npy": str(amp_path),
        "phase_npy": str(phase_path),
        "amplitude_png": str(amp_png),
        "phase_png": str(phase_png),
    }
    return artifact_map, amp_truth, phase_truth


def _pearson_r(pred: np.ndarray, truth: np.ndarray) -> float | None:
    pred_flat = np.asarray(pred, dtype=float).ravel()
    truth_flat = np.asarray(truth, dtype=float).ravel()
    mask = np.isfinite(pred_flat) & np.isfinite(truth_flat)
    if not mask.any():
        return None
    pred_vals = pred_flat[mask]
    truth_vals = truth_flat[mask]
    if pred_vals.size < 2 or truth_vals.size < 2:
        return None
    pred_std = np.std(pred_vals)
    truth_std = np.std(truth_vals)
    if pred_std == 0.0 or truth_std == 0.0:
        return None
    cov = np.mean((pred_vals - pred_vals.mean()) * (truth_vals - truth_vals.mean()))
    return float(cov / (pred_std * truth_std))


def _compute_diff_metrics(
    diff: np.ndarray, pred: np.ndarray, truth: np.ndarray
) -> Dict[str, Any]:
    diff_vals = np.asarray(diff, dtype=float)
    pred_vals = np.asarray(pred, dtype=float)
    truth_vals = np.asarray(truth, dtype=float)
    mask = np.isfinite(diff_vals) & np.isfinite(pred_vals) & np.isfinite(truth_vals)
    if not mask.any():
        return {"count": 0, "mae": None, "rmse": None, "max_abs": None, "pearson_r": None}
    diff_use = diff_vals[mask]
    metrics = {
        "count": int(diff_use.size),
        "mae": float(np.mean(np.abs(diff_use))),
        "rmse": float(np.sqrt(np.mean(np.square(diff_use)))),
        "max_abs": float(np.max(np.abs(diff_use))),
        "pearson_r": _pearson_r(pred_vals[mask], truth_vals[mask]),
    }
    return metrics


def _extract_scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float | None]:
    scalars: Dict[str, float | None] = {}
    for key in ("mae", "rmse", "max_abs", "pearson_r"):
        value = metrics.get(key)
        scalars[key] = float(value) if value is not None else None
    return scalars


def write_diff_artifacts(
    amplitude_pred: np.ndarray,
    amplitude_truth: np.ndarray,
    phase_pred: np.ndarray,
    phase_truth: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate diff PNGs + metrics JSON for amplitude/phase."""
    comparison_path = output_dir / "comparison_metrics.json"
    amp_pred = np.squeeze(amplitude_pred)
    amp_truth = np.squeeze(amplitude_truth)
    phase_pred = np.squeeze(phase_pred)
    phase_truth = np.squeeze(phase_truth)

    amp_diff = amp_pred - amp_truth
    amp_diff_npy = output_dir / "amplitude_diff.npy"
    np.save(amp_diff_npy, amp_diff.astype(np.float32))
    amp_diff_png = output_dir / "amplitude_diff.png"
    amp_range = max(float(np.nanmax(np.abs(amp_truth))), 1e-9)
    save_png(
        amp_diff,
        amp_diff_png,
        "Amplitude difference",
        cmap="coolwarm",
        vmin=-amp_range,
        vmax=amp_range,
    )
    amp_metrics = _compute_diff_metrics(amp_diff, amp_pred, amp_truth)
    amp_metrics["color_range"] = [-amp_range, amp_range]
    amp_metrics["pred_stats"] = compute_array_stats(amp_pred)
    amp_metrics["truth_stats"] = compute_array_stats(amp_truth)
    amp_metrics["bias_summary"] = summarize_bias(amp_pred, amp_truth)

    phase_diff = np.angle(np.exp(1j * (phase_pred - phase_truth)))
    phase_diff_npy = output_dir / "phase_diff.npy"
    np.save(phase_diff_npy, phase_diff.astype(np.float32))
    phase_diff_png = output_dir / "phase_diff.png"
    save_png(
        phase_diff,
        phase_diff_png,
        "Phase difference",
        cmap="twilight",
        vmin=-math.pi,
        vmax=math.pi,
    )
    phase_metrics = _compute_diff_metrics(phase_diff, phase_pred, phase_truth)
    phase_metrics["color_range"] = [-math.pi, math.pi]
    phase_metrics["pred_stats"] = compute_array_stats(phase_pred)
    phase_metrics["truth_stats"] = compute_array_stats(phase_truth)
    phase_metrics["bias_summary"] = summarize_bias(phase_pred, phase_truth)

    metrics_payload = {
        "amplitude": amp_metrics,
        "phase": phase_metrics,
    }
    comparison_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    return {
        "metrics_path": str(comparison_path),
        "metrics": metrics_payload,
        "artifacts": {
            "amplitude_diff_npy": str(amp_diff_npy),
            "amplitude_diff_png": str(amp_diff_png),
            "phase_diff_npy": str(phase_diff_npy),
            "phase_diff_png": str(phase_diff_png),
        },
    }


def write_comparison_summary_markdown(
    scenario: str,
    amp_metrics: Mapping[str, Any],
    phase_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Emit a Markdown summary of prediction vs truth stats."""
    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return f"{float(value):.6g}"

    def _stat_table_lines(label: str, metrics: Mapping[str, Any]) -> list[str]:
        pred_stats = metrics.get("pred_stats", {})
        truth_stats = metrics.get("truth_stats", {})
        rows = [
            f"## {label}",
            "",
            "| Metric | Prediction | Ground Truth |",
            "| --- | --- | --- |",
        ]
        for key in ("min", "max", "mean", "std", "nan_count"):
            rows.append(
                f"| {key} | {fmt(pred_stats.get(key))} | {fmt(truth_stats.get(key))} |"
            )
        rows.append("")
        bias_stats = metrics.get("bias_summary", {})
        rows.extend(
            [
                "| Bias Metric (pred - truth) | Value |",
                "| --- | --- |",
                f"| mean | {fmt(bias_stats.get('mean'))} |",
                f"| median | {fmt(bias_stats.get('median'))} |",
                f"| p05 | {fmt(bias_stats.get('p05'))} |",
                f"| p95 | {fmt(bias_stats.get('p95'))} |",
                "",
            ]
        )
        return rows

    lines = [
        f"# {scenario} Ground-Truth Comparison Summary",
        "",
        "Bias values are reported as `(prediction - ground_truth)` to make shared intensity offsets obvious.",
        "",
    ]
    lines.extend(_stat_table_lines("Amplitude", amp_metrics))
    lines.extend(_stat_table_lines("Phase", phase_metrics))
    output_path.write_text("\n".join(lines))


def _coerce_value(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_value(v) for v in value]
        return float(coerced[-1]) if coerced else float("nan")
    return float(value)


def _coerce_sequence(seq: Any) -> list[float]:
    if seq is None:
        return []
    if isinstance(seq, np.ndarray):
        return [_coerce_value(v) for v in seq.tolist()]
    if isinstance(seq, (list, tuple)):
        return [_coerce_value(v) for v in seq]
    return [_coerce_value(seq)]


def coerce_history_for_json(history: Mapping[str, Any]) -> Dict[str, list[float]]:
    sanitized: Dict[str, list[float]] = {}
    for key, values in history.items():
        sanitized[key] = _coerce_sequence(values)
    return sanitized


def summarize_history(
    history: Mapping[str, Sequence[float]],
    epochs: Sequence[float] | None = None,
) -> Dict[str, Any]:
    metrics_summary: Dict[str, Any] = {}
    nan_metrics: Dict[str, Dict[str, Any]] = {}
    epoch_list = list(epochs) if epochs is not None else None

    for metric, values in history.items():
        series = _coerce_sequence(values)
        if not series:
            metrics_summary[metric] = {
                "count": 0,
                "last": None,
                "min": None,
                "max": None,
                "first_nan_step": None,
                "first_nan_epoch": None,
                "has_nan": False,
            }
            continue
        arr = np.asarray(series, dtype=float)
        first_bad_idx: int | None = None
        if arr.size:
            mask = ~np.isfinite(arr)
            bad_indices = np.where(mask)[0]
            if bad_indices.size:
                first_bad_idx = int(bad_indices[0])

        first_bad_epoch = None
        if first_bad_idx is not None and epoch_list and len(epoch_list) > first_bad_idx:
            first_bad_epoch = epoch_list[first_bad_idx]

        try:
            min_val = float(np.nanmin(arr))
        except (ValueError, FloatingPointError):
            min_val = None
        try:
            max_val = float(np.nanmax(arr))
        except (ValueError, FloatingPointError):
            max_val = None
        last_val = float(arr[-1]) if arr.size else None
        metric_summary = {
            "count": int(arr.size),
            "last": last_val,
            "min": min_val,
            "max": max_val,
            "first_nan_step": first_bad_idx,
            "first_nan_epoch": first_bad_epoch,
            "has_nan": first_bad_idx is not None,
        }
        metrics_summary[metric] = metric_summary
        if first_bad_idx is not None:
            nan_metrics[metric] = {
                "first_nan_step": first_bad_idx,
                "first_nan_epoch": first_bad_epoch,
            }
            print(
                f"[runner][history][warn] Metric '{metric}' reported NaN/inf at step "
                f"{first_bad_idx}{f' (epoch {first_bad_epoch})' if first_bad_epoch is not None else ''}"
            )

    if nan_metrics:
        metrics_list = ", ".join(sorted(nan_metrics))
        print(f"[runner][history][warn] NaN/inf detected in metrics: {metrics_list}")

    return {
        "metrics": metrics_summary,
        "nan_overview": {
            "has_nan": bool(nan_metrics),
            "metrics": nan_metrics,
        },
    }


def _format_optional(value: Any, precision: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{precision}g}"


def write_training_summary_markdown(
    scenario_name: str,
    summary: Mapping[str, Any],
    history_rel_path: str,
    summary_rel_path: str,
    output_path: Path,
) -> None:
    metrics = summary.get("metrics", {})
    nan_overview = summary.get("nan_overview", {})
    lines = [
        f"# {scenario_name} Training Summary",
        "",
        f"- History JSON: `{history_rel_path}`",
        f"- Summary JSON: `{summary_rel_path}`",
    ]
    if nan_overview.get("has_nan"):
        affected = ", ".join(sorted(nan_overview.get("metrics", {}).keys()))
        lines.append(f"- ⚠️ NaN/inf detected in metrics: {affected}")
    else:
        lines.append("- ✅ No NaN/inf detected in training history.")
    lines.append("")
    if metrics:
        lines.append("| Metric | Last | Min | Max | First NaN Step | First NaN Epoch |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for metric in sorted(metrics):
            entry = metrics[metric]
            lines.append(
                "| {metric} | {last} | {minv} | {maxv} | {step} | {epoch} |".format(
                    metric=metric,
                    last=_format_optional(entry.get("last")),
                    minv=_format_optional(entry.get("min")),
                    maxv=_format_optional(entry.get("max")),
                    step=entry.get("first_nan_step", "—")
                    if entry.get("first_nan_step") is not None
                    else "—",
                    epoch=entry.get("first_nan_epoch", "—")
                    if entry.get("first_nan_epoch") is not None
                    else "—",
                )
            )
    else:
        lines.append("No history metrics were reported for this run.")
    output_path.write_text("\n".join(lines) + "\n")


def run_inference_and_reassemble(
    test_raw,
    scenario: ScenarioSpec,
    params: RunParams,
    model_dir: Path,
    group_count: int,
    group_limit: int,
    custom_probe_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    intensity_stages: list[Dict[str, Any]] = []
    if getattr(test_raw, "diff3d", None) is not None:
        record_intensity_stage(
            intensity_stages,
            "raw_diffraction",
            test_raw.diff3d,
            metadata={
                "source": "RawData",
                "count": int(test_raw.diff3d.shape[0]),
            },
        )
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
    # CONFIG-001: Sync legacy params.cfg before loader/grouped-data (spec-inference-pipeline.md §1.1)
    update_legacy_dict(legacy_params.cfg, infer_config)
    model, params_dict = load_inference_bundle_with_backend(model_dir, infer_config)
    bundle_intensity_scale = params_dict.get("intensity_scale")
    legacy_intensity_scale = legacy_params.cfg.get("intensity_scale")
    recorded_scale = (
        bundle_intensity_scale if bundle_intensity_scale is not None else legacy_intensity_scale
    )
    nsamples = min(group_count, group_limit) if group_limit else group_count
    grouped = test_raw.generate_grouped_data(
        params_dict.get("N", params.N),
        K=params.neighbor_count,
        nsamples=nsamples,
        gridsize=params_dict.get("gridsize", scenario.gridsize),
    )
    grouped_diff = grouped.get("diffraction")
    if grouped_diff is not None:
        record_intensity_stage(
            intensity_stages,
            "grouped_diffraction",
            grouped_diff,
            metadata={
                "source": "RawData.generate_grouped_data",
                "count": int(grouped_diff.shape[0]),
                "gridsize": scenario.gridsize,
            },
        )
    if grouped.get("X_full") is not None:
        record_intensity_stage(
            intensity_stages,
            "grouped_X_full",
            grouped["X_full"],
            metadata={
                "source": "normalize_data",
                "count": int(grouped["X_full"].shape[0]),
            },
        )
    container = loader.load(lambda: grouped, test_raw.probeGuess, which=None, create_split=False)
    record_intensity_stage(
        intensity_stages,
        "container_X",
        container.X,
        metadata={
            "source": "PtychoDataContainer",
            "group_limit": group_limit,
        },
    )
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(container, diffraction_to_obj=model)
    obj_image = tf_helper.reassemble_position(obj_tensor_full, global_offsets, M=params.reassemble_M)
    intensity_info = {
        "stages": intensity_stages,
        "bundle_intensity_scale": bundle_intensity_scale,
        "legacy_params_intensity_scale": legacy_intensity_scale,
        "recorded_scale": recorded_scale,
    }
    return np.abs(obj_image), np.angle(obj_image), global_offsets, intensity_info


def save_stats(
    output_dir: Path,
    amplitude: np.ndarray,
    phase: np.ndarray,
    offsets_summary: Dict[str, Any],
    padded_size: int,
    N_value: int,
    extra_fields: Optional[Mapping[str, Any]] = None,
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
    if extra_fields:
        stats.update(extra_fields)
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

    # CONFIG-001: Sync legacy params.cfg before training/loader (spec-inference-pipeline.md §1.1)
    update_legacy_dict(legacy_params.cfg, train_config)

    start_time = time.time()
    training_results = run_training(train_raw, test_raw, train_config)
    save_training_bundle(train_dir)
    train_elapsed = time.time() - start_time
    print(f"[runner] Training complete in {train_elapsed:.2f}s; bundle saved to {train_dir}")

    # Capture IntensityScaler state after training (D4 architecture diagnostics)
    # Per specs/spec-ptycho-workflow.md §Loss and Optimization, the log_scale variable
    # is trained during training. We capture its state to trace any divergence.
    intensity_scaler_state = extract_intensity_scaler_state()
    print(
        f"[runner][scaler] exp(log_scale)={intensity_scaler_state.get('exp_log_scale')} "
        f"vs params.cfg={intensity_scaler_state.get('params_cfg_intensity_scale')} "
        f"(delta={intensity_scaler_state.get('delta')})"
    )

    history_payload_raw = training_results.get("history") or {}
    history_epochs_raw = training_results.get("history_epochs")
    coerced_history = coerce_history_for_json(history_payload_raw)
    coerced_epochs = _coerce_sequence(history_epochs_raw) if history_epochs_raw is not None else []
    history_json = {"metrics": coerced_history}
    if coerced_epochs:
        history_json["epochs"] = coerced_epochs
    history_path = train_dir / "history.json"
    history_path.write_text(json.dumps(history_json, indent=2, sort_keys=True))

    history_summary = summarize_history(coerced_history, coerced_epochs if coerced_epochs else None)
    history_summary_path = train_dir / "history_summary.json"
    history_summary_path.write_text(json.dumps(history_summary, indent=2, sort_keys=True))

    summary_md_path = scenario_dir / f"{scenario.name}_training_summary.md"
    history_rel = os.path.relpath(history_path, scenario_dir)
    summary_rel = os.path.relpath(history_summary_path, scenario_dir)
    summary_md_rel = os.path.relpath(summary_md_path, scenario_dir)
    write_training_summary_markdown(
        scenario.name,
        history_summary,
        history_rel,
        summary_rel,
        summary_md_path,
    )

    amp, phase, global_offsets, intensity_info = run_inference_and_reassemble(
        test_raw=test_raw,
        scenario=scenario,
        params=params,
        model_dir=train_dir,
        group_count=group_count,
        group_limit=args.group_limit,
        custom_probe_path=custom_probe_path,
    )
    target_size = params.object_size
    amp, crop_metadata = center_crop(amp, target_size)
    phase, _ = center_crop(phase, target_size)
    amplitude_unscaled = np.array(amp, copy=True)
    amplitude_unscaled_path = inference_dir / "amplitude_unscaled.npy"
    np.save(amplitude_unscaled_path, amplitude_unscaled.astype(np.float32))
    recorded_scale = intensity_info.get("recorded_scale")
    amp_truth_for_scale, _ = center_crop(np.abs(object_guess), target_size)
    scale_info = determine_prediction_scale(
        args.prediction_scale_source,
        recorded_scale,
        amplitude_unscaled,
        amp_truth_for_scale,
    )
    scale_note = format_prediction_scale_note(scale_info)
    if scale_info.get("applied") and isinstance(scale_info.get("value"), (int, float)):
        amp = amplitude_unscaled * float(scale_info["value"])
    else:
        amp = amplitude_unscaled
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
        extra_fields={"prediction_scale": scale_info},
    )
    stats["crop_metadata"] = crop_metadata

    amp_vmin = 0.0
    amp_vmax = args.amp_vmax if args.amp_vmax is not None else max(stats["amplitude"]["max"], 1e-9)
    phase_vmin = -math.pi
    phase_vmax = math.pi

    save_png(
        amp,
        inference_dir / "amplitude.png",
        f"{scenario.name} amplitude {f'[{scale_note}]' if scale_note else ''} (vmax={amp_vmax:.3f})",
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

    ground_truth_artifacts, amp_truth, phase_truth = save_ground_truth_artifacts(
        object_guess,
        scenario_dir,
        amp_bounds=(amp_vmin, amp_vmax),
        target_size=target_size,
    )
    comparison_payload = write_diff_artifacts(
        amplitude_pred=amp,
        amplitude_truth=amp_truth,
        phase_pred=phase,
        phase_truth=phase_truth,
        output_dir=scenario_dir,
    )
    comparison_summary = {
        "amplitude": _extract_scalar_metrics(comparison_payload["metrics"]["amplitude"]),
        "phase": _extract_scalar_metrics(comparison_payload["metrics"]["phase"]),
    }
    comparison_summary_path = scenario_dir / f"{scenario.name}_comparison_summary.md"
    write_comparison_summary_markdown(
        scenario.name,
        comparison_payload["metrics"]["amplitude"],
        comparison_payload["metrics"]["phase"],
        comparison_summary_path,
    )

    scale_note_path = inference_dir / "prediction_scale.txt"
    scale_note_path.write_text(
        json.dumps(
            {
                "mode": scale_info.get("mode"),
                "value": scale_info.get("value"),
                "applied": scale_info.get("applied"),
                "source": scale_info.get("source"),
                "recorded_scale": scale_info.get("recorded_scale"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
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
            "ground_truth_amplitude_npy": ground_truth_artifacts["amplitude_npy"],
            "ground_truth_phase_npy": ground_truth_artifacts["phase_npy"],
            "ground_truth_amplitude_png": ground_truth_artifacts["amplitude_png"],
            "ground_truth_phase_png": ground_truth_artifacts["phase_png"],
            "comparison_metrics": comparison_payload["metrics_path"],
            "amplitude_diff_npy": comparison_payload["artifacts"]["amplitude_diff_npy"],
            "amplitude_diff_png": comparison_payload["artifacts"]["amplitude_diff_png"],
            "phase_diff_npy": comparison_payload["artifacts"]["phase_diff_npy"],
            "phase_diff_png": comparison_payload["artifacts"]["phase_diff_png"],
            "comparison_summary": str(comparison_summary_path),
            "prediction_scale_json": str(scale_note_path),
        },
    }
    metadata["prediction_scale"] = scale_info
    metadata["artifacts"]["amplitude_unscaled_npy"] = str(amplitude_unscaled_path)
    if scale_note:
        metadata["prediction_scale_note"] = scale_note
    metadata["training_history"] = {
        "history_json": str(history_path),
        "summary_json": str(history_summary_path),
        "summary_markdown": str(summary_md_path),
        "nan_overview": history_summary.get("nan_overview", {}),
    }
    metadata["training_nan_overview"] = history_summary.get("nan_overview", {})
    metadata["training_history_path"] = history_rel
    metadata["training_summary_path"] = summary_rel
    metadata["training_summary_markdown_path"] = summary_md_rel
    metadata["ground_truth_amp_path"] = ground_truth_artifacts["amplitude_npy"]
    metadata["ground_truth_phase_path"] = ground_truth_artifacts["phase_npy"]
    metadata["comparison_metrics_path"] = comparison_payload["metrics_path"]
    metadata["comparison_metrics_summary"] = comparison_summary
    metadata["ground_truth"] = ground_truth_artifacts
    metadata["comparison"] = {
        "metrics_json": comparison_payload["metrics_path"],
        "metrics": comparison_payload["metrics"],
        "metrics_summary": comparison_summary,
        "diff_pngs": comparison_payload["artifacts"],
        "summary_markdown": str(comparison_summary_path),
    }
    # Extract training container X stats from the stages if available
    training_container_stats = None
    for stage in intensity_info.get("stages", []):
        if stage.get("name") == "container_X":
            training_container_stats = stage.get("stats")
            break

    intensity_record = write_intensity_stats_outputs(
        stages=intensity_info.get("stages", []),
        bundle_intensity_scale=intensity_info.get("bundle_intensity_scale"),
        legacy_intensity_scale=intensity_info.get("legacy_params_intensity_scale"),
        scenario_dir=scenario_dir,
        intensity_scaler_state=intensity_scaler_state,
        training_container_stats=training_container_stats,
    )
    intensity_record["prediction_scale"] = scale_info
    metadata["intensity_stats"] = intensity_record
    metadata["intensity_stats_path"] = intensity_record.get("json_path")
    metadata["intensity_stats_markdown"] = intensity_record.get("markdown_path")
    artifacts = metadata.setdefault("artifacts", {})
    if intensity_record.get("json_path"):
        artifacts["intensity_stats_json"] = intensity_record["json_path"]
    if intensity_record.get("markdown_path"):
        artifacts["intensity_stats_markdown"] = intensity_record["markdown_path"]
    metadata["crop_metadata"] = crop_metadata
    # Add IntensityScaler state to run_metadata for D4 architecture diagnostics
    metadata["intensity_scaler_state"] = intensity_scaler_state
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

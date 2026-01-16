#!/usr/bin/env python3
"""Generate probe normalization stats for SIM-LINES legacy vs modern pipelines."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ptycho import params as legacy_params
from ptycho import probe as probe_module
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from scripts.simulation.synthetic_helpers import make_probe, normalize_probe_guess
from scripts.studies.sim_lines_4x.pipeline import CUSTOM_PROBE_PATH, RunParams, ScenarioSpec

DEFAULT_SNAPSHOT = Path(
    "plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json"
)


@dataclass(frozen=True)
class ScenarioMetadata:
    """Flattened scenario description pulled from the snapshot."""

    name: str
    gridsize: int
    probe_mode: str
    probe_scale: float
    probe_big: bool
    probe_mask: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="Scenario name from the snapshot JSON")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to the sim_lines params snapshot JSON",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path to write the JSON report",
    )
    parser.add_argument(
        "--output-markdown",
        required=True,
        type=Path,
        help="Path to write the Markdown report",
    )
    return parser.parse_args()


def load_snapshot(snapshot_path: Path) -> Tuple[RunParams, Dict[str, Mapping[str, Any]], Path]:
    data = json.loads(snapshot_path.read_text())
    run_params = RunParams(**data["run_params"])
    scenarios: Dict[str, Mapping[str, Any]] = {}
    for entry in data.get("scenarios", []):
        name = entry.get("name") or entry.get("inputs", {}).get("name")
        if not name:
            continue
        scenarios[name] = entry
    if not scenarios:
        raise ValueError(f"No scenarios found in snapshot: {snapshot_path}")
    custom_probe_entry = data.get("custom_probe_path")
    custom_probe_path = Path(custom_probe_entry) if custom_probe_entry else CUSTOM_PROBE_PATH
    return run_params, scenarios, custom_probe_path


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def scenario_metadata(entry: Mapping[str, Any]) -> ScenarioMetadata:
    defaults = entry.get("defaults", {})
    inputs = entry.get("inputs", {})
    spec = ScenarioSpec(
        name=entry.get("name") or inputs.get("name") or "unknown",
        gridsize=inputs.get("gridsize"),
        probe_mode=inputs.get("probe_mode") or "custom",
        probe_scale=inputs.get("probe_scale"),
        probe_big=coalesce(inputs.get("probe_big"), defaults.get("probe_big")),
        probe_mask=coalesce(inputs.get("probe_mask"), defaults.get("probe_mask")),
    )

    defaults_model = ModelConfig()
    return ScenarioMetadata(
        name=spec.name,
        gridsize=int(spec.gridsize),
        probe_mode=spec.probe_mode,
        probe_scale=float(spec.probe_scale),
        probe_big=defaults_model.probe_big if spec.probe_big is None else bool(spec.probe_big),
        probe_mask=defaults_model.probe_mask if spec.probe_mask is None else bool(spec.probe_mask),
    )


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def bootstrap_legacy_config(run_params: RunParams, scenario: ScenarioMetadata) -> None:
    model_config = ModelConfig(
        N=run_params.N,
        gridsize=scenario.gridsize,
        probe_scale=scenario.probe_scale,
        probe_big=scenario.probe_big,
        probe_mask=scenario.probe_mask,
    )
    config = TrainingConfig(
        model=model_config,
        n_groups=run_params.group_count,
        nphotons=run_params.nphotons,
        neighbor_count=run_params.neighbor_count,
    )
    update_legacy_dict(legacy_params.cfg, config)
    legacy_params.cfg["default_probe_scale"] = scenario.probe_scale


def compute_mask(N: int) -> np.ndarray:
    mask = probe_module.get_probe_mask_real(N)
    return np.asarray(mask[..., 0], dtype=np.float32)


def compute_norm_factor(probe_guess: np.ndarray, mask: np.ndarray, scale: float) -> float:
    tamped = mask * np.abs(probe_guess)
    mean_amp = float(np.mean(tamped))
    if mean_amp == 0.0:
        raise ValueError("Probe normalization failed: zero mean amplitude inside mask")
    return float(scale * mean_amp)


def tensor_to_numpy(probe_tensor: tf.Tensor) -> np.ndarray:
    array = probe_tensor.numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array.astype(np.complex64)


def build_legacy_probe(
    run_params: RunParams,
    scenario: ScenarioMetadata,
    *,
    custom_probe_path: Path,
    mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    bootstrap_legacy_config(run_params, scenario)
    if scenario.probe_mode == "idealized":
        probe_module.set_default_probe()
        probe_tensor = legacy_params.get("probe")
        normalized = tensor_to_numpy(probe_tensor)
        base_probe = probe_module.get_default_probe(run_params.N, fmt="np")
    else:
        source_path = custom_probe_path if custom_probe_path else CUSTOM_PROBE_PATH
        probe_guess = make_probe(run_params.N, mode="custom", path=source_path)
        tf_probe = tf.convert_to_tensor(probe_guess[..., None], dtype=tf.complex64)
        probe_module.set_probe(tf_probe)
        probe_tensor = legacy_params.get("probe")
        normalized = tensor_to_numpy(probe_tensor)
        base_probe = probe_guess
    norm = compute_norm_factor(np.asarray(base_probe), mask, scenario.probe_scale)
    return normalized, norm


def build_sim_lines_probe(
    run_params: RunParams,
    scenario: ScenarioMetadata,
    *,
    custom_probe_path: Path,
    mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    if scenario.probe_mode == "idealized":
        legacy_params.cfg["default_probe_scale"] = scenario.probe_scale
        probe_guess = make_probe(run_params.N, mode="idealized")
    else:
        source_path = custom_probe_path if custom_probe_path else CUSTOM_PROBE_PATH
        probe_guess = make_probe(run_params.N, mode="custom", path=source_path)
    normalized = normalize_probe_guess(
        np.asarray(probe_guess),
        probe_scale=scenario.probe_scale,
        N=run_params.N,
    )
    norm = compute_norm_factor(np.asarray(probe_guess), mask, scenario.probe_scale)
    return normalized, norm


def amplitude_stats(probe: np.ndarray) -> Dict[str, float]:
    amp = np.abs(probe)
    return {
        "min": float(np.min(amp)),
        "max": float(np.max(amp)),
        "mean": float(np.mean(amp)),
        "std": float(np.std(amp)),
        "l2": float(np.linalg.norm(amp)),
    }


def ratio_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def comparison_metrics(legacy_probe: np.ndarray, modern_probe: np.ndarray) -> Dict[str, Any]:
    legacy_amp = np.abs(legacy_probe)
    modern_amp = np.abs(modern_probe)
    ratio = np.divide(
        legacy_amp,
        modern_amp,
        out=np.ones_like(legacy_amp),
        where=modern_amp > 0,
    )
    amp_delta = legacy_amp - modern_amp
    complex_delta = legacy_probe - modern_probe
    return {
        "amp_ratio": ratio_stats(ratio),
        "amp_delta": {
            "mean": float(np.mean(amp_delta)),
            "std": float(np.std(amp_delta)),
            "min": float(np.min(amp_delta)),
            "max": float(np.max(amp_delta)),
        },
        "l2_delta": float(np.linalg.norm(complex_delta)),
        "max_abs_delta": float(np.max(np.abs(complex_delta))),
    }


def build_metadata(
    scenario: ScenarioMetadata,
    run_params: RunParams,
    snapshot: Path,
    custom_probe_path: Path,
) -> Dict[str, Any]:
    return {
        "scenario": scenario.name,
        "gridsize": scenario.gridsize,
        "probe_mode": scenario.probe_mode,
        "probe_scale": scenario.probe_scale,
        "probe_big": scenario.probe_big,
        "probe_mask": scenario.probe_mask,
        "nphotons": run_params.nphotons,
        "snapshot": str(snapshot),
        "custom_probe_path": str(custom_probe_path),
    }


def build_markdown(
    metadata: Mapping[str, Any],
    *,
    legacy_stats: Mapping[str, Any],
    modern_stats: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# Probe Normalization Report — {metadata['scenario']}")
    lines.append("")
    lines.append(f"- Snapshot: `{metadata['snapshot']}`")
    lines.append(f"- Probe mode: `{metadata['probe_mode']}`")
    lines.append(f"- Probe scale: {metadata['probe_scale']}")
    lines.append(f"- Gridsize: {metadata['gridsize']}")
    lines.append(f"- Custom probe path: `{metadata['custom_probe_path']}`")
    lines.append("")
    lines.append("## Branch Stats")
    lines.append("")
    lines.append("| Branch | Norm factor | Amp min | Amp max | Amp mean | Amp std | L2 norm |")
    lines.append("|--------|-------------|---------|---------|----------|---------|---------|")
    lines.append(
        "| legacy | {norm:.6f} | {min:.6f} | {max:.6f} | {mean:.6f} | {std:.6f} | {l2:.6f} |".format(
            norm=legacy_stats["normalization_factor"],
            **legacy_stats["amplitude"],
        )
    )
    lines.append(
        "| sim_lines | {norm:.6f} | {min:.6f} | {max:.6f} | {mean:.6f} | {std:.6f} | {l2:.6f} |".format(
            norm=modern_stats["normalization_factor"],
            **modern_stats["amplitude"],
        )
    )
    lines.append("")
    lines.append("## Comparison (legacy ÷ sim_lines)")
    lines.append("")
    ratio = comparison["amp_ratio"]
    lines.append(
        "- Amp ratio min/max/mean/std: {min:.6f} / {max:.6f} / {mean:.6f} / {std:.6f}".format(**ratio)
    )
    amp_delta = comparison["amp_delta"]
    lines.append(
        "- Amp delta min/max/mean/std: {min:.6f} / {max:.6f} / {mean:.6f} / {std:.6f}".format(**amp_delta)
    )
    lines.append(f"- L2 delta: {comparison['l2_delta']:.6f}")
    lines.append(f"- Max |legacy - sim_lines|: {comparison['max_abs_delta']:.6f}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_params, scenario_entries, custom_probe_path = load_snapshot(args.snapshot)
    if args.scenario not in scenario_entries:
        available = ", ".join(sorted(scenario_entries.keys()))
        raise ValueError(f"Scenario '{args.scenario}' not found. Available: {available}")

    scenario_entry = scenario_entries[args.scenario]
    scenario = scenario_metadata(scenario_entry)
    mask = compute_mask(run_params.N)

    legacy_probe, legacy_norm = build_legacy_probe(
        run_params,
        scenario,
        custom_probe_path=custom_probe_path,
        mask=mask,
    )
    modern_probe, modern_norm = build_sim_lines_probe(
        run_params,
        scenario,
        custom_probe_path=custom_probe_path,
        mask=mask,
    )

    legacy_stats = {
        "normalization_factor": legacy_norm,
        "amplitude": amplitude_stats(legacy_probe),
    }
    modern_stats = {
        "normalization_factor": modern_norm,
        "amplitude": amplitude_stats(modern_probe),
    }
    comparison = comparison_metrics(legacy_probe, modern_probe)
    metadata = build_metadata(scenario, run_params, args.snapshot, custom_probe_path)

    report = {
        "metadata": metadata,
        "legacy": legacy_stats,
        "sim_lines": modern_stats,
        "comparison": comparison,
    }

    ensure_directory(args.output_json)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    ensure_directory(args.output_markdown)
    args.output_markdown.write_text(
        build_markdown(metadata, legacy_stats=legacy_stats, modern_stats=modern_stats, comparison=comparison)
    )

    print(
        f"Scenario {scenario.name}: legacy norm={legacy_norm:.6f}, sim_lines norm={modern_norm:.6f}, "
        f"L2 delta={comparison['l2_delta']:.6f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python3
"""
Normalization parity capture CLI for dose_experiments-style runs.

Simulates the nongrid dataset using dose_experiments_param_scan.md defaults
(gridsize=2, probe_scale=4, neighbor_count=5, etc.), splits by y-axis,
records stage telemetry via record_intensity_stage, and emits both JSON
and Markdown outputs for normalization parity analysis.

Spec reference: specs/spec-ptycho-core.md §Normalization Invariants
Finding reference: CONFIG-001, SIM-LINES-CONFIG-001, NORMALIZATION-001
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ptycho import params as legacy_params
from ptycho.config.config import ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.raw_data import RawData
from scripts.simulation.synthetic_helpers import (
    make_lines_object,
    make_probe,
    normalize_probe_guess,
    simulate_nongrid_raw_data,
    split_raw_data_by_axis,
)

# Import intensity stage helpers from the existing runner
# The bin directory is not a package, so we need to import via sys.path manipulation
import importlib.util as _import_util

def _import_runner_helpers():  # type: ignore[misc]
    """Import helpers from run_phase_c2_scenario.py without requiring package structure."""
    _runner_path = Path(__file__).parent / "run_phase_c2_scenario.py"
    _spec = _import_util.spec_from_file_location("_runner", _runner_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load module spec from {_runner_path}")
    _module = _import_util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)  # type: ignore[union-attr]
    return _module

_runner = _import_runner_helpers()  # type: ignore[misc]
RATIO_TRANSITIONS = _runner.RATIO_TRANSITIONS
STAGE_LABELS = _runner.STAGE_LABELS
STAGE_ORDER = _runner.STAGE_ORDER
record_intensity_stage = _runner.record_intensity_stage
write_intensity_stats_outputs = _runner.write_intensity_stats_outputs

# Dose experiments defaults from dose_experiments_param_scan.md
DOSE_EXPERIMENTS_DEFAULTS = {
    "gridsize": 2,
    "offset": 4,
    "max_position_jitter": 10,
    "n_filters_scale": 2,
    "object_big": True,
    "intensity_scale_trainable": True,
    "probe_trainable": False,
    "outer_offset_train": 8,
    "outer_offset_test": 20,
    "nimgs_train": 2,
    "nimgs_test": 2,
}


@dataclasses.dataclass
class CaptureConfig:
    """Configuration for dose normalization capture."""

    scenario: str
    output_dir: Path
    total_images: int
    group_count: int
    neighbor_count: int
    nphotons: float
    buffer: float
    object_size: int
    sim_seed: int
    object_seed: int
    probe_mode: str
    probe_scale: float
    gridsize: int
    split_fraction: float
    custom_probe_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name (e.g., dose_legacy_gs2)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--total-images",
        type=int,
        default=1024,
        help="Total raw images to simulate (default: 1024)",
    )
    parser.add_argument(
        "--group-count",
        type=int,
        default=64,
        help="Number of groups for grouping (default: 64)",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=5,
        help="Neighbor count for KDTree grouping (default: 5)",
    )
    parser.add_argument(
        "--nphotons",
        type=float,
        default=1e9,
        help="Photon count (default: 1e9)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Simulation buffer size (default: 10.0)",
    )
    parser.add_argument(
        "--object-size",
        type=int,
        default=392,
        help="Object size in pixels (default: 392)",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=42,
        help="Simulation random seed (default: 42)",
    )
    parser.add_argument(
        "--object-seed",
        type=int,
        default=42,
        help="Object generation random seed (default: 42)",
    )
    parser.add_argument(
        "--custom-probe-path",
        type=Path,
        default=Path("ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz"),
        help="Path to custom probe NPZ file",
    )
    parser.add_argument(
        "--probe-mode",
        choices=["custom", "idealized"],
        default="custom",
        help="Probe mode (default: custom)",
    )
    parser.add_argument(
        "--probe-scale",
        type=float,
        default=4.0,
        help="Probe scale factor (default: 4.0)",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=2,
        help="Grouping gridsize (default: 2, per dose_experiments)",
    )
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=0.5,
        help="Train/test split fraction (default: 0.5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing outputs before writing",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CaptureConfig:
    """Build CaptureConfig from CLI args."""
    return CaptureConfig(
        scenario=args.scenario,
        output_dir=args.output_dir,
        total_images=args.total_images,
        group_count=args.group_count,
        neighbor_count=args.neighbor_count,
        nphotons=args.nphotons,
        buffer=args.buffer,
        object_size=args.object_size,
        sim_seed=args.sim_seed,
        object_seed=args.object_seed,
        probe_mode=args.probe_mode,
        probe_scale=args.probe_scale,
        gridsize=args.gridsize,
        split_fraction=args.split_fraction,
        custom_probe_path=args.custom_probe_path,
    )


def config_to_dict(cfg: CaptureConfig) -> Dict[str, Any]:
    """Convert CaptureConfig to JSON-serializable dict."""
    return {
        "scenario": cfg.scenario,
        "output_dir": str(cfg.output_dir),
        "total_images": cfg.total_images,
        "group_count": cfg.group_count,
        "neighbor_count": cfg.neighbor_count,
        "nphotons": cfg.nphotons,
        "buffer": cfg.buffer,
        "object_size": cfg.object_size,
        "sim_seed": cfg.sim_seed,
        "object_seed": cfg.object_seed,
        "probe_mode": cfg.probe_mode,
        "probe_scale": cfg.probe_scale,
        "gridsize": cfg.gridsize,
        "split_fraction": cfg.split_fraction,
        "custom_probe_path": str(cfg.custom_probe_path) if cfg.custom_probe_path else None,
        "dose_experiments_defaults": DOSE_EXPERIMENTS_DEFAULTS,
    }


def compute_dataset_intensity_scale(raw_data: RawData, nphotons: float) -> float:
    """
    Compute dataset-derived intensity scale per spec-ptycho-core.md §Normalization Invariants.

    Dataset-derived mode (preferred):
        s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])

    Since we have diff3d (amplitude), |Ψ|² = diff3d², so:
        s = sqrt(nphotons / mean(sum(diff3d², axis=(1,2))))
    """
    if raw_data.diff3d is None:
        raise ValueError("raw_data.diff3d is required for intensity scale computation")
    diff_squared = raw_data.diff3d.astype(np.float64) ** 2
    mean_sum_intensity = np.mean(np.sum(diff_squared, axis=(1, 2)))
    return float(np.sqrt(nphotons / mean_sum_intensity))


def compute_closedform_fallback_scale(nphotons: float, N: int) -> float:
    """
    Compute closed-form fallback intensity scale per spec-ptycho-core.md §Normalization Invariants.

    Closed-form fallback:
        s ≈ sqrt(nphotons) / (N/2)
    """
    return float(np.sqrt(nphotons) / (N / 2))


def run_capture(cfg: CaptureConfig) -> Dict[str, Any]:
    """
    Execute normalization capture pipeline.

    Steps:
    1. Generate object via make_lines_object
    2. Build probe (custom or idealized)
    3. Simulate nongrid raw data
    4. Split by y-axis
    5. Group train split
    6. Record stage telemetry
    7. Compute intensity scales
    8. Write outputs
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[capture][{cfg.scenario}] Starting at {timestamp}")

    # Determine N based on probe or use default
    # The custom probe (Run1084_recon3_postPC_shrunk_3.npz) is 64x64
    # Idealized probes can be generated at any size
    # Check probe shape first if custom mode
    if cfg.probe_mode == "custom" and cfg.custom_probe_path and cfg.custom_probe_path.exists():
        with np.load(cfg.custom_probe_path) as data:
            if "probeGuess" in data:
                N = data["probeGuess"].shape[0]
                print(f"[capture] N derived from custom probe: {N}")
            else:
                N = 64  # Fallback to 64 which is common
    else:
        N = 64  # Default for idealized probes
    # Note: C = cfg.gridsize ** 2 channels per group, used implicitly in grouping

    # Step 1: Generate object
    print(f"[capture] Generating lines object (size={cfg.object_size}, seed={cfg.object_seed})")
    object_guess = make_lines_object(cfg.object_size, seed=cfg.object_seed)
    print(f"[capture] Object shape: {object_guess.shape}, dtype: {object_guess.dtype}")

    # Step 2: Build and normalize probe
    print(f"[capture] Building probe (mode={cfg.probe_mode}, scale={cfg.probe_scale})")
    if cfg.probe_mode == "custom" and cfg.custom_probe_path:
        if not cfg.custom_probe_path.exists():
            raise FileNotFoundError(f"Custom probe not found: {cfg.custom_probe_path}")
        probe_raw = make_probe(N, mode="custom", path=cfg.custom_probe_path)
    else:
        probe_raw = make_probe(N, mode="idealized")
    probe_guess = normalize_probe_guess(probe_raw, probe_scale=cfg.probe_scale, N=N)
    print(f"[capture] Probe shape: {probe_guess.shape}, mean|amp|: {np.mean(np.abs(probe_guess)):.6f}")

    # Step 3: Simulate nongrid raw data (CONFIG-001 compliant via synthetic_helpers)
    print(f"[capture] Simulating {cfg.total_images} raw images (nphotons={cfg.nphotons:.2e})")
    raw_data = simulate_nongrid_raw_data(
        object_guess=object_guess,
        probe_guess=probe_guess,
        N=N,
        n_images=cfg.total_images,
        nphotons=cfg.nphotons,
        seed=cfg.sim_seed,
        buffer=cfg.buffer,
        sim_gridsize=1,
        use_cache=True,
    )
    print(f"[capture] RawData: diff3d shape {raw_data.diff3d.shape}")

    # Initialize intensity stage telemetry
    stages: list[Dict[str, Any]] = []

    # Record raw diffraction stats
    record_intensity_stage(
        stages,
        "raw_diffraction",
        raw_data.diff3d,
        metadata={"n_images": cfg.total_images, "source": "simulate_nongrid_raw_data"},
    )

    # Step 4: Split by y-axis
    print(f"[capture] Splitting by y-axis (fraction={cfg.split_fraction})")
    train_raw, test_raw = split_raw_data_by_axis(
        raw_data, split_fraction=cfg.split_fraction, axis="y"
    )
    print(f"[capture] Train split: {train_raw.diff3d.shape[0]} images, Test split: {test_raw.diff3d.shape[0]} images")

    # Step 5: Build training config and sync legacy params (CONFIG-001)
    train_config = TrainingConfig(
        model=ModelConfig(N=N, gridsize=cfg.gridsize),
        n_groups=cfg.group_count,
        nphotons=cfg.nphotons,
    )
    # CONFIG-001: Bridge before any legacy module usage
    update_legacy_dict(legacy_params.cfg, train_config)
    print(f"[capture] CONFIG-001 bridge complete (gridsize={cfg.gridsize}, n_groups={cfg.group_count})")

    # Step 6: Generate grouped data from train split
    print(f"[capture] Grouping train split (groups={cfg.group_count}, neighbors={cfg.neighbor_count})")
    grouped = train_raw.generate_grouped_data(
        N=N,
        K=cfg.neighbor_count,
        nsamples=cfg.group_count,
        gridsize=cfg.gridsize,
        seed=cfg.sim_seed,
    )

    # Record grouped diffraction stats
    grouped_diffraction = grouped.get("diffraction")
    if grouped_diffraction is not None:
        record_intensity_stage(
            stages,
            "grouped_diffraction",
            grouped_diffraction,
            metadata={
                "n_groups": grouped_diffraction.shape[0] if grouped_diffraction is not None else 0,
                "gridsize": cfg.gridsize,
                "neighbor_count": cfg.neighbor_count,
            },
        )
        print(f"[capture] Grouped diffraction shape: {grouped_diffraction.shape}")

    # Step 7: Normalize grouped data (this is where normalize_data is applied)
    from ptycho.raw_data import normalize_data

    grouped_X_full = normalize_data(grouped, N)
    grouped["X_full"] = grouped_X_full
    record_intensity_stage(
        stages,
        "grouped_X_full",
        grouped_X_full,
        metadata={"normalization": "normalize_data", "N": N},
    )
    print(f"[capture] Normalized X_full shape: {grouped_X_full.shape}")

    # Step 8: Build container (simulates what loader does)
    from ptycho.workflows.components import create_ptycho_data_container

    container = create_ptycho_data_container(
        train_raw,
        train_config,
    )
    container_X = container._X_np if hasattr(container, "_X_np") else None
    if container_X is None and hasattr(container, "X"):
        import tensorflow as tf
        container_X_tf = getattr(container, "X", None)
        if container_X_tf is not None:
            container_X = container_X_tf.numpy() if isinstance(container_X_tf, tf.Tensor) else np.asarray(container_X_tf)
    if container_X is not None:
        record_intensity_stage(
            stages,
            "container_X",
            container_X,
            metadata={"source": "PtychoDataContainer"},
        )
        print(f"[capture] Container X shape: {container_X.shape}")

    # Step 9: Compute intensity scales
    dataset_intensity_scale = compute_dataset_intensity_scale(raw_data, cfg.nphotons)
    closedform_intensity_scale = compute_closedform_fallback_scale(cfg.nphotons, N)
    scale_ratio = dataset_intensity_scale / closedform_intensity_scale if closedform_intensity_scale != 0 else None

    # Helper for safe float formatting
    def _fmt(v: Optional[float]) -> str:
        return f"{v:.6f}" if v is not None else "N/A"

    print(f"[capture] Dataset-derived intensity_scale: {dataset_intensity_scale:.6f}")
    print(f"[capture] Closed-form fallback: {closedform_intensity_scale:.6f}")
    print(f"[capture] Ratio (dataset/closedform): {_fmt(scale_ratio)}")

    # Step 10: Write outputs
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Write intensity stats using the shared helper
    stats_result = write_intensity_stats_outputs(
        stages=stages,
        bundle_intensity_scale=dataset_intensity_scale,
        legacy_intensity_scale=closedform_intensity_scale,
        scenario_dir=cfg.output_dir,
    )

    # Also write as dose_normalization_stats.{json,md}
    dose_stats_payload = {
        "scenario": cfg.scenario,
        "timestamp": timestamp,
        "dataset_intensity_scale": dataset_intensity_scale,
        "closedform_intensity_scale": closedform_intensity_scale,
        "scale_ratio": scale_ratio,
        "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants",
        "stages": stages,
        "stage_means": stats_result.get("stage_means", {}),
        "ratios": stats_result.get("ratios", {}),
        "largest_drop": stats_result.get("largest_drop"),
        "normalize_gain": stats_result.get("normalize_gain"),
    }

    dose_json_path = cfg.output_dir / "dose_normalization_stats.json"
    dose_json_path.write_text(json.dumps(dose_stats_payload, indent=2, sort_keys=True))

    # Build Markdown summary
    dose_md_lines = [
        "# Dose Normalization Statistics",
        "",
        f"**Scenario:** {cfg.scenario}",
        f"**Timestamp:** {timestamp}",
        "",
        "## Spec Reference",
        "",
        "Per `specs/spec-ptycho-core.md §Normalization Invariants`:",
        "",
        "- Dataset-derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`",
        "- Closed-form fallback: `s ≈ sqrt(nphotons) / (N/2)`",
        "",
        "In both modes symmetry SHALL hold:",
        "- Training inputs: `X_scaled = s · X`",
        "- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)",
        "",
        "## Intensity Scales",
        "",
        f"| Source | Value |",
        f"| --- | ---: |",
        f"| Dataset-derived | {dataset_intensity_scale:.6f} |",
        f"| Closed-form fallback | {closedform_intensity_scale:.6f} |",
        f"| Ratio (dataset/closedform) | {scale_ratio:.6f} |" if scale_ratio is not None else "| Ratio (dataset/closedform) | N/A |",
        "",
        "## Stage Flow",
        "",
        "| Stage | Mean |",
        "| --- | ---: |",
    ]

    for stage_key in STAGE_ORDER:
        label = STAGE_LABELS.get(stage_key, stage_key)
        mean_val = stats_result.get("stage_means", {}).get(stage_key)
        dose_md_lines.append(f"| {label} | {_fmt(mean_val)} |")

    dose_md_lines.extend([
        "",
        "## Stage Ratios",
        "",
        "| Transition | Ratio |",
        "| --- | ---: |",
    ])
    for from_stage, to_stage, ratio_key in RATIO_TRANSITIONS:
        from_label = STAGE_LABELS.get(from_stage, from_stage)
        to_label = STAGE_LABELS.get(to_stage, to_stage)
        ratio_val = stats_result.get("ratios", {}).get(ratio_key)
        dose_md_lines.append(f"| {from_label} → {to_label} | {_fmt(ratio_val)} |")

    if stats_result.get("largest_drop"):
        drop = stats_result["largest_drop"]
        dose_md_lines.extend([
            "",
            "## Largest Drop",
            "",
            f"**{STAGE_LABELS.get(drop['from_stage'], drop['from_stage'])} → "
            f"{STAGE_LABELS.get(drop['to_stage'], drop['to_stage'])}** "
            f"(ratio={_fmt(drop.get('ratio'))})",
        ])

    dose_md_path = cfg.output_dir / "dose_normalization_stats.md"
    dose_md_path.write_text("\n".join(dose_md_lines))

    # Write capture config
    config_path = cfg.output_dir / "capture_config.json"
    config_path.write_text(json.dumps(config_to_dict(cfg), indent=2, sort_keys=True))

    # Write capture summary
    summary_lines = [
        "# Capture Summary",
        "",
        f"**Scenario:** {cfg.scenario}",
        f"**Timestamp:** {timestamp}",
        "",
        "## Configuration",
        "",
        f"- Total images: {cfg.total_images}",
        f"- Group count: {cfg.group_count}",
        f"- Neighbor count: {cfg.neighbor_count}",
        f"- Gridsize: {cfg.gridsize}",
        f"- nphotons: {cfg.nphotons:.2e}",
        f"- Probe mode: {cfg.probe_mode}",
        f"- Probe scale: {cfg.probe_scale}",
        "",
        "## Intensity Scale Summary",
        "",
        f"- Dataset-derived: {dataset_intensity_scale:.6f}",
        f"- Closed-form fallback: {closedform_intensity_scale:.6f}",
        f"- Ratio: {_fmt(scale_ratio)}",
        "",
        "## CONFIG-001 Bridge",
        "",
        "Per SIM-LINES-CONFIG-001 and CONFIG-001, `update_legacy_dict(params.cfg, config)` was called",
        "before grouping and container creation to ensure legacy modules see correct parameters.",
        "",
        "## Spec Citation",
        "",
        "Per `specs/spec-ptycho-core.md §Normalization Invariants`:",
        "",
        "> Dataset-level `intensity_scale` `s` is a learned or fixed parameter used symmetrically.",
        "> Two compliant calculation modes are allowed:",
        "> 1) Dataset-derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`",
        "> 2) Closed-form fallback: `s ≈ sqrt(nphotons) / (N/2)`",
        "",
        "## Artifacts",
        "",
        f"- `capture_config.json`: Full capture configuration",
        f"- `dose_normalization_stats.json`: Intensity scale and stage statistics (JSON)",
        f"- `dose_normalization_stats.md`: Intensity scale and stage statistics (Markdown)",
        f"- `intensity_stats.json`: Detailed stage telemetry (JSON)",
        f"- `intensity_stats.md`: Detailed stage telemetry (Markdown)",
    ]
    summary_path = cfg.output_dir / "capture_summary.md"
    summary_path.write_text("\n".join(summary_lines))

    print(f"[capture] Outputs written to {cfg.output_dir}")
    return {
        "scenario": cfg.scenario,
        "timestamp": timestamp,
        "output_dir": str(cfg.output_dir),
        "dataset_intensity_scale": dataset_intensity_scale,
        "closedform_intensity_scale": closedform_intensity_scale,
        "scale_ratio": scale_ratio,
        "stats_result": stats_result,
    }


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle overwrite
    if args.output_dir.exists():
        if args.overwrite:
            print(f"[capture] --overwrite: Clearing existing outputs at {args.output_dir}")
            for item in args.output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        else:
            existing_files = list(args.output_dir.glob("*.json")) + list(args.output_dir.glob("*.md"))
            if existing_files:
                print(f"[capture][ERROR] Output directory contains existing files: {args.output_dir}")
                print("[capture][ERROR] Use --overwrite to clear existing outputs")
                return 1

    try:
        cfg = config_from_args(args)
        result = run_capture(cfg)
        print(f"[capture] SUCCESS: {result['scenario']} complete")
        return 0
    except Exception as e:
        print(f"[capture][ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Write blocker log if output dir exists
        if args.output_dir.exists():
            blocker_path = args.output_dir / "blocker.log"
            blocker_path.write_text(f"ERROR: {type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            print(f"[capture] Blocker log written to {blocker_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Rebuild the canonical lines_256 synthetic dataset pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ptycho.workflows.grid_lines_workflow import GridLinesConfig
from scripts.studies.grid_study_dataset_builder import build_datasets


DEFAULT_OUTPUT_ROOT = Path("outputs/lines_256_arch_improvement")
DEFAULT_PROBE_NPZ = Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")


def _canonical_config(
    output_root: Path,
    probe_npz: Path,
    probe_transform_pipeline: str | None = None,
) -> GridLinesConfig:
    if probe_transform_pipeline is None:
        probe_scale_mode = "pad_preserve"
        probe_smoothing_sigma = 0.5
    else:
        probe_scale_mode = "pipeline"
        probe_smoothing_sigma = 0.0
    return GridLinesConfig(
        N=256,
        gridsize=1,
        output_dir=output_root,
        probe_npz=probe_npz,
        nimgs_train=2,
        nimgs_test=1,
        nphotons=1e9,
        size=392,
        offset=4,
        outer_offset_train=8,
        outer_offset_test=20,
        probe_source="custom",
        probe_scale_mode=probe_scale_mode,
        probe_smoothing_sigma=probe_smoothing_sigma,
        probe_transform_pipeline=probe_transform_pipeline,
        set_phi=True,
    )


def build_lines_256_dataset(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    probe_npz: Path = DEFAULT_PROBE_NPZ,
    probe_transform_pipeline: str | None = None,
) -> Dict[str, Any]:
    """Build the authoritative lines_256 dataset pair and return a summary."""
    cfg = _canonical_config(
        output_root=Path(output_root),
        probe_npz=Path(probe_npz),
        probe_transform_pipeline=probe_transform_pipeline,
    )
    bundles = build_datasets(
        dataset_source="synthetic_lines",
        cfg=cfg,
        required_ns=[256],
    )
    bundle = bundles[256]
    return {
        "train_npz": Path(bundle["train_npz"]),
        "test_npz": Path(bundle["test_npz"]),
        "gt_recon": Path(bundle["gt_recon"]),
        "set_phi": cfg.set_phi,
        "config": {
            "N": cfg.N,
            "gridsize": cfg.gridsize,
            "output_dir": str(cfg.output_dir),
            "probe_npz": str(cfg.probe_npz),
            "nimgs_train": cfg.nimgs_train,
            "nimgs_test": cfg.nimgs_test,
            "nphotons": cfg.nphotons,
            "size": cfg.size,
            "offset": cfg.offset,
            "outer_offset_train": cfg.outer_offset_train,
            "outer_offset_test": cfg.outer_offset_test,
            "probe_source": cfg.probe_source,
            "probe_scale_mode": cfg.probe_scale_mode,
            "probe_smoothing_sigma": cfg.probe_smoothing_sigma,
            "probe_transform_pipeline": cfg.probe_transform_pipeline,
            "set_phi": cfg.set_phi,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Synthetic dataset output root.",
    )
    parser.add_argument(
        "--probe-npz",
        type=Path,
        default=DEFAULT_PROBE_NPZ,
        help="Probe NPZ used to seed the synthetic lines dataset.",
    )
    parser.add_argument(
        "--probe-transform-pipeline",
        type=str,
        default=None,
        help="Optional explicit probe transform pipeline, e.g. 'smooth:0.5|pad:128|interp:256'.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_lines_256_dataset(
        output_root=args.output_root,
        probe_npz=args.probe_npz,
        probe_transform_pipeline=args.probe_transform_pipeline,
    )
    serializable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in summary.items()
    }
    print(json.dumps(serializable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

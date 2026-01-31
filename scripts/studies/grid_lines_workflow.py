#!/usr/bin/env python3
"""CLI wrapper for grid_lines_workflow."""

import argparse
import os
from pathlib import Path

from ptycho.workflows.grid_lines_workflow import GridLinesConfig, run_grid_lines_workflow


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, choices=[64, 128])
    parser.add_argument("--gridsize", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-npz",
        type=Path,
        default=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    )
    parser.add_argument("--nimgs-train", type=int, default=2)
    parser.add_argument("--nimgs-test", type=int, default=2)
    parser.add_argument("--nphotons", type=float, default=1e9)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nll-weight", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--realspace-weight", type=float, default=0.0)
    parser.add_argument("--probe-smoothing-sigma", type=float, default=0.5)
    parser.add_argument("--probe-mask-diameter", type=int, default=None)
    parser.add_argument(
        "--probe-source",
        choices=["custom", "ideal_disk"],
        default="custom",
        help="Probe source for grid-lines datasets.",
    )
    parser.add_argument(
        "--probe-scale-mode",
        choices=["pad_extrapolate", "interpolate"],
        default="pad_extrapolate",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> GridLinesConfig:
    return GridLinesConfig(
        N=args.N,
        gridsize=args.gridsize,
        output_dir=args.output_dir,
        probe_npz=args.probe_npz,
        nimgs_train=args.nimgs_train,
        nimgs_test=args.nimgs_test,
        nphotons=args.nphotons,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        nll_weight=args.nll_weight,
        mae_weight=args.mae_weight,
        realspace_weight=args.realspace_weight,
        probe_smoothing_sigma=args.probe_smoothing_sigma,
        probe_mask_diameter=args.probe_mask_diameter,
        probe_source=args.probe_source,
        probe_scale_mode=args.probe_scale_mode,
    )


def main() -> None:
    os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    args = parse_args()
    cfg = build_config(args)
    run_grid_lines_workflow(cfg)


if __name__ == "__main__":
    main()

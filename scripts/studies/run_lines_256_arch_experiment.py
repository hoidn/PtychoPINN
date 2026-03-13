#!/usr/bin/env python3
"""Thin wrapper for repeatable lines_256 hybrid-resnet architecture experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from scripts.studies.invocation_logging import write_invocation_artifacts

LINES_256_TRAIN_NPZ = Path(
    "outputs/hybrid_resnet_structural_rerun_20260226T110719Z/"
    "datasets/custom_npz_builder_n256/datasets/N256/gs1/train.npz"
)
LINES_256_TEST_NPZ = Path(
    "outputs/hybrid_resnet_structural_rerun_20260226T110719Z/"
    "datasets/custom_npz_builder_n256/datasets/N256/gs1/test.npz"
)

FIXED_SEED = 3
FIXED_EPOCHS = 20
FIXED_N = 256
FIXED_GRIDSIZE = 1
FIXED_ARCHITECTURE = "hybrid_resnet"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lines_256 hybrid-resnet experiment with fixed dataset, seed, "
            "resolution, and epoch budget."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--infer-batch-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--gradient-clip-algorithm",
        choices=["norm", "value", "agc"],
        default="norm",
    )
    parser.add_argument("--fno-modes", type=int, default=12)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-blocks", type=int, default=4)
    parser.add_argument(
        "--hybrid-skip-connections",
        dest="hybrid_skip_connections",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-hybrid-skip-connections",
        dest="hybrid_skip_connections",
        action="store_false",
    )
    parser.add_argument("--hybrid-downsample-steps", type=int, default=2)
    parser.add_argument(
        "--hybrid-downsample-op",
        choices=["stride_conv", "avgpool_conv", "blurpool_conv"],
        default="stride_conv",
    )
    parser.add_argument("--hybrid-resnet-blocks", type=int, default=6)
    parser.add_argument(
        "--hybrid-skip-style",
        choices=["add", "concat", "gated_add"],
        default="add",
    )
    return parser.parse_args(argv)


def build_runner_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "scripts/studies/grid_lines_torch_runner.py",
        "--train-npz",
        str(LINES_256_TRAIN_NPZ),
        "--test-npz",
        str(LINES_256_TEST_NPZ),
        "--output-dir",
        str(args.output_dir),
        "--architecture",
        FIXED_ARCHITECTURE,
        "--seed",
        str(FIXED_SEED),
        "--epochs",
        str(FIXED_EPOCHS),
        "--N",
        str(FIXED_N),
        "--gridsize",
        str(FIXED_GRIDSIZE),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--infer-batch-size",
        str(args.infer_batch_size),
        "--grad-clip",
        str(args.grad_clip),
        "--gradient-clip-algorithm",
        args.gradient_clip_algorithm,
        "--fno-modes",
        str(args.fno_modes),
        "--fno-width",
        str(args.fno_width),
        "--fno-blocks",
        str(args.fno_blocks),
        "--hybrid-downsample-steps",
        str(args.hybrid_downsample_steps),
        "--hybrid-downsample-op",
        args.hybrid_downsample_op,
        "--hybrid-resnet-blocks",
        str(args.hybrid_resnet_blocks),
        "--hybrid-skip-style",
        args.hybrid_skip_style,
        "--no-probe-mask",
        "--no-torch-mae-pred-l2-match-target",
    ]
    if args.hybrid_skip_connections:
        cmd.append("--hybrid-skip-connections")
    else:
        cmd.append("--no-hybrid-skip-connections")
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/run_lines_256_arch_experiment.py",
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "fixed_train_npz": str(LINES_256_TRAIN_NPZ),
            "fixed_test_npz": str(LINES_256_TEST_NPZ),
            "fixed_seed": FIXED_SEED,
            "fixed_epochs": FIXED_EPOCHS,
            "fixed_N": FIXED_N,
            "fixed_gridsize": FIXED_GRIDSIZE,
            "fixed_architecture": FIXED_ARCHITECTURE,
        },
    )

    cmd = build_runner_cmd(args)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)

    (args.output_dir / "driver_stdout.log").write_text(completed.stdout)
    (args.output_dir / "driver_stderr.log").write_text(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            "lines_256 arch experiment failed "
            f"(exit={completed.returncode}); see "
            f"{args.output_dir / 'driver_stdout.log'} and "
            f"{args.output_dir / 'driver_stderr.log'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

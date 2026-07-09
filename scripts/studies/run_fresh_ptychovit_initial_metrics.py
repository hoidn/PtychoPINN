#!/usr/bin/env python3
"""Run a fresh checkpoint-restored PtychoViT initial baseline on lines synthetic data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_PTYCHOVIT_REPO = Path("/home/ollie/Documents/ptycho-vit")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed output-dir with a checkpoint and run grid_lines_compare_wrapper in "
            "fresh selected-model mode for pinn_ptychovit on lines synthetic data."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ptychovit-repo", type=Path, default=DEFAULT_PTYCHOVIT_REPO)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--gridsize", type=int, default=1)
    parser.add_argument("--nimgs-train", type=int, default=1)
    parser.add_argument("--nimgs-test", type=int, default=1)
    parser.add_argument("--torch-epochs", type=int, default=None)
    parser.add_argument("--set-phi", action="store_true")
    return parser.parse_args(argv)


def _ensure_output_dir(output_dir: Path, *, force_clean: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force_clean:
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {output_dir}. "
                "Use --force-clean to delete and rerun."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _seed_checkpoint(checkpoint: Path, output_dir: Path) -> Path:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    run_dir = output_dir / "runs" / "pinn_ptychovit"
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / "best_model.pth"

    if checkpoint.resolve() != dst.resolve():
        shutil.copy2(checkpoint, dst)
    return dst


def _build_wrapper_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "scripts/studies/grid_lines_compare_wrapper.py",
        "--N",
        str(args.N),
        "--gridsize",
        str(args.gridsize),
        "--output-dir",
        str(args.output_dir),
        "--architectures",
        "hybrid",
        "--models",
        "pinn_ptychovit",
        "--model-n",
        "pinn_ptychovit=256",
        "--ptychovit-repo",
        str(args.ptychovit_repo),
        "--nimgs-train",
        str(args.nimgs_train),
        "--nimgs-test",
        str(args.nimgs_test),
    ]
    if args.set_phi:
        cmd.append("--set-phi")
    if args.torch_epochs is not None:
        cmd.extend(["--torch-epochs", str(args.torch_epochs)])
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    _ensure_output_dir(args.output_dir, force_clean=args.force_clean)
    seeded_checkpoint = _seed_checkpoint(args.checkpoint, args.output_dir)

    cmd = _build_wrapper_cmd(args)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)

    run_dir = args.output_dir / "runs" / "pinn_ptychovit"
    (run_dir / "driver_stdout.log").write_text(completed.stdout)
    (run_dir / "driver_stderr.log").write_text(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            "fresh initial metrics run failed "
            f"(exit={completed.returncode}); see {run_dir / 'driver_stdout.log'} "
            f"and {run_dir / 'driver_stderr.log'}"
        )

    print(f"Seeded checkpoint: {seeded_checkpoint}")
    print(f"Run complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Orchestrate TF grid-lines workflow + Torch runners and merge metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple, Optional

from ptycho.workflows.grid_lines_workflow import GridLinesConfig
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig


def _parse_architectures(value: str) -> Tuple[str, ...]:
    return tuple(a.strip() for a in value.split(",") if a.strip())


def run_grid_lines_compare(
    *,
    N: int,
    gridsize: int,
    output_dir: Path,
    probe_npz: Path,
    architectures: Iterable[str],
    seed: int = 0,
    nimgs_train: int = 2,
    nimgs_test: int = 2,
    nphotons: float = 1e9,
    nepochs: int = 60,
    batch_size: int = 16,
    nll_weight: float = 0.0,
    mae_weight: float = 1.0,
    realspace_weight: float = 0.0,
    probe_smoothing_sigma: float = 0.5,
    set_phi: bool = False,
    torch_epochs: Optional[int] = None,
    torch_batch_size: Optional[int] = None,
    torch_learning_rate: float = 1e-3,
    torch_infer_batch_size: int = 16,
    torch_gradient_clip_val: float = 1.0,
    torch_loss_mode: str = "mae",
    fno_modes: int = 12,
    fno_width: int = 32,
    fno_blocks: int = 4,
    fno_cnn_blocks: int = 2,
) -> dict:
    os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    output_dir = Path(output_dir)
    architectures = tuple(architectures)

    dataset_dir = output_dir / "datasets" / f"N{N}" / f"gs{gridsize}"
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"

    tf_metrics = {}
    if ("cnn" in architectures or "baseline" in architectures) or not train_npz.exists() or not test_npz.exists():
        tf_cfg = GridLinesConfig(
            N=N,
            gridsize=gridsize,
            output_dir=output_dir,
            probe_npz=probe_npz,
            nimgs_train=nimgs_train,
            nimgs_test=nimgs_test,
            nphotons=nphotons,
            nepochs=nepochs,
            batch_size=batch_size,
            nll_weight=nll_weight,
            mae_weight=mae_weight,
            realspace_weight=realspace_weight,
            probe_smoothing_sigma=probe_smoothing_sigma,
            set_phi=set_phi,
        )
        from ptycho.workflows import grid_lines_workflow as tf_workflow
        tf_result = tf_workflow.run_grid_lines_workflow(tf_cfg)
        train_npz = Path(tf_result["train_npz"])
        test_npz = Path(tf_result["test_npz"])
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        tf_metrics = json.loads(metrics_path.read_text())

    merged = {}
    if "cnn" in architectures and "pinn" in tf_metrics:
        merged["pinn"] = tf_metrics["pinn"]
    if "baseline" in architectures and "baseline" in tf_metrics:
        merged["baseline"] = tf_metrics["baseline"]

    for arch in architectures:
        if arch in ("fno", "hybrid"):
            torch_cfg = TorchRunnerConfig(
                train_npz=train_npz,
                test_npz=test_npz,
                output_dir=output_dir,
                architecture=arch,
                seed=seed,
                epochs=torch_epochs or nepochs,
                batch_size=torch_batch_size or batch_size,
                learning_rate=torch_learning_rate,
                infer_batch_size=torch_infer_batch_size,
                gradient_clip_val=torch_gradient_clip_val,
                N=N,
                gridsize=gridsize,
                torch_loss_mode=torch_loss_mode,
                fno_modes=fno_modes,
                fno_width=fno_width,
                fno_blocks=fno_blocks,
                fno_cnn_blocks=fno_cnn_blocks,
            )
            from scripts.studies import grid_lines_torch_runner as torch_runner
            torch_result = torch_runner.run_grid_lines_torch(torch_cfg)
            if "metrics" in torch_result:
                merged[f"pinn_{arch}"] = torch_result["metrics"]

    order = ["gt"]
    if "cnn" in architectures:
        order.append("pinn")
    if "baseline" in architectures:
        order.append("baseline")
    if "fno" in architectures:
        order.append("pinn_fno")
    if "hybrid" in architectures:
        order.append("pinn_hybrid")

    from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals
    render_grid_lines_visuals(output_dir, order=tuple(order))

    metrics_path.write_text(json.dumps(merged, indent=2, default=str))
    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "metrics": merged,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run grid-lines comparison across backends")
    parser.add_argument("--N", type=int, required=True, choices=[64, 128])
    parser.add_argument("--gridsize", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-npz",
        type=Path,
        default=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    )
    parser.add_argument("--architectures", type=str, default="cnn,baseline,fno,hybrid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nimgs-train", type=int, default=2)
    parser.add_argument("--nimgs-test", type=int, default=2)
    parser.add_argument("--nphotons", type=float, default=1e9)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nll-weight", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--realspace-weight", type=float, default=0.0)
    parser.add_argument("--probe-smoothing-sigma", type=float, default=0.5)
    parser.add_argument("--set-phi", action="store_true", help="Enable non-zero phase in synthetic grid data.")
    parser.add_argument("--torch-epochs", type=int, default=None)
    parser.add_argument("--torch-batch-size", type=int, default=None)
    parser.add_argument("--torch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-infer-batch-size", type=int, default=16)
    parser.add_argument(
        "--torch-grad-clip",
        type=float,
        default=1.0,
        help="Torch gradient clipping max norm (<=0 disables clipping).",
    )
    parser.add_argument("--torch-loss-mode", type=str, default="mae", choices=["poisson", "mae"])
    parser.add_argument("--fno-modes", type=int, default=12)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-blocks", type=int, default=4)
    parser.add_argument("--fno-cnn-blocks", type=int, default=2)
    args = parser.parse_args(argv)
    args.architectures = _parse_architectures(args.architectures)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_grid_lines_compare(
        N=args.N,
        gridsize=args.gridsize,
        output_dir=args.output_dir,
        probe_npz=args.probe_npz,
        architectures=args.architectures,
        seed=args.seed,
        nimgs_train=args.nimgs_train,
        nimgs_test=args.nimgs_test,
        nphotons=args.nphotons,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        nll_weight=args.nll_weight,
        mae_weight=args.mae_weight,
        realspace_weight=args.realspace_weight,
        probe_smoothing_sigma=args.probe_smoothing_sigma,
        set_phi=args.set_phi,
        torch_epochs=args.torch_epochs,
        torch_batch_size=args.torch_batch_size,
        torch_learning_rate=args.torch_learning_rate,
        torch_infer_batch_size=args.torch_infer_batch_size,
        torch_gradient_clip_val=args.torch_grad_clip,
        torch_loss_mode=args.torch_loss_mode,
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_blocks=args.fno_blocks,
        fno_cnn_blocks=args.fno_cnn_blocks,
    )


if __name__ == "__main__":
    main()

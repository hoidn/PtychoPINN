#!/usr/bin/env python3
"""Checkpoint-replay parity runner for shift_sum vs batched position reassembly."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, _harmonize_prediction_shape
from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference
from scripts.studies.invocation_logging import write_invocation_artifacts


def _load_recon(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "YY_pred" not in data.files:
            raise KeyError(f"Missing YY_pred in recon artifact: {path}")
        return np.asarray(data["YY_pred"], dtype=np.complex64)


def _amp_stats(arr: np.ndarray) -> dict[str, float]:
    amp = np.abs(np.asarray(arr))
    return {
        "amp_mean": float(np.mean(amp)),
        "amp_q99": float(np.quantile(amp, 0.99)),
        "amp_max": float(np.max(amp)),
    }


def summarize_backend_parity(
    *,
    shift_sum_recon: Path,
    batched_recon: Path,
    out_json: Path,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Summarize parity metrics between shift_sum and batched recon artifacts."""
    shift = _load_recon(Path(shift_sum_recon))
    batched_raw = _load_recon(Path(batched_recon))
    batched = _harmonize_prediction_shape(batched_raw, shift)
    shift = np.squeeze(np.asarray(shift, dtype=np.complex64))
    batched = np.squeeze(np.asarray(batched, dtype=np.complex64))

    shift_stats = _amp_stats(shift)
    batched_stats = _amp_stats(batched)

    amp_shift = np.abs(shift)
    amp_batched = np.abs(batched)
    phase_shift = np.angle(shift)
    phase_batched = np.angle(batched)

    amp_mae = float(np.mean(np.abs(amp_shift - amp_batched)))
    amp_rmse = float(np.sqrt(np.mean((amp_shift - amp_batched) ** 2)))
    phase_mae = float(np.mean(np.abs(np.angle(np.exp(1j * (phase_shift - phase_batched))))))

    amp_max_rel_error = float(
        abs(batched_stats["amp_max"] - shift_stats["amp_max"]) / max(shift_stats["amp_max"], 1e-12)
    )
    amp_q99_rel_error = float(
        abs(batched_stats["amp_q99"] - shift_stats["amp_q99"]) / max(shift_stats["amp_q99"], 1e-12)
    )
    complex_allclose = bool(np.allclose(batched, shift, atol=atol, rtol=rtol))
    parity_pass = bool(complex_allclose and amp_max_rel_error <= rtol and amp_q99_rel_error <= rtol)

    summary = {
        "shift_sum_recon": str(shift_sum_recon),
        "batched_recon": str(batched_recon),
        "thresholds": {"atol": float(atol), "rtol": float(rtol)},
        "shape_shift_sum": list(shift.shape),
        "shape_batched_aligned": list(batched.shape),
        "shift_sum_stats": shift_stats,
        "batched_stats": batched_stats,
        "diff_metrics": {
            "amp_mae": amp_mae,
            "amp_rmse": amp_rmse,
            "phase_mae": phase_mae,
            "amp_max_rel_error": amp_max_rel_error,
            "amp_q99_rel_error": amp_q99_rel_error,
            "complex_allclose": complex_allclose,
        },
        "parity_pass": parity_pass,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def run_checkpoint_replay_parity(
    *,
    model_pt: Path,
    dataset_name: str,
    train_npz: Path,
    test_npz: Path,
    output_dir: Path,
    base_cfg: TorchRunnerConfig,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Run checkpoint replay twice (shift_sum, batched) and write parity summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_recons: dict[str, str] = {}
    for backend in ("shift_sum", "batched"):
        backend_root = output_dir / backend
        cfg = replace(
            base_cfg,
            train_npz=Path(train_npz),
            test_npz=Path(test_npz),
            output_dir=backend_root,
            reassembly_mode="position",
            position_reassembly_backend=backend,
        )
        results = run_cross_dataset_hybrid_inference(
            model_pt=Path(model_pt),
            dataset_npzs={dataset_name: Path(test_npz)},
            output_dir=backend_root,
            base_cfg=cfg,
        )
        backend_recons[backend] = str(results[dataset_name]["recon_npz"])

    summary = summarize_backend_parity(
        shift_sum_recon=Path(backend_recons["shift_sum"]),
        batched_recon=Path(backend_recons["batched"]),
        out_json=output_dir / "summary.json",
        atol=atol,
        rtol=rtol,
    )
    manifest = {
        "model_pt": str(model_pt),
        "dataset_name": dataset_name,
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "backend_recons": backend_recons,
        "summary_json": str(output_dir / "summary.json"),
        "parity_pass": bool(summary["parity_pass"]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run checkpoint replay parity between shift_sum and batched position reassembly."
    )
    parser.add_argument("--model-pt", type=Path, required=True)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="replay")
    parser.add_argument("--architecture", type=str, default="hybrid_resnet")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--gridsize", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--infer-batch-size", type=int, default=128)
    parser.add_argument("--position-reassembly-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/position_reassembly_checkpoint_replay.py",
        argv=(argv if argv is not None else sys.argv[1:]),
        parsed_args=vars(args),
    )

    base_cfg = TorchRunnerConfig(
        train_npz=args.train_npz,
        test_npz=args.test_npz,
        output_dir=args.output_dir,
        architecture=args.architecture,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        infer_batch_size=args.infer_batch_size,
        N=args.n,
        gridsize=args.gridsize,
        reassembly_mode="position",
        position_reassembly_backend="shift_sum",
        position_reassembly_batch_size=args.position_reassembly_batch_size,
    )
    manifest = run_checkpoint_replay_parity(
        model_pt=args.model_pt,
        dataset_name=args.dataset_name,
        train_npz=args.train_npz,
        test_npz=args.test_npz,
        output_dir=args.output_dir,
        base_cfg=base_cfg,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

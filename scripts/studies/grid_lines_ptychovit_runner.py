#!/usr/bin/env python3
"""Subprocess runner for PtychoViT study integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass
class PtychoViTRunnerConfig:
    ptychovit_repo: Path
    output_dir: Path
    train_dp: Path
    test_dp: Path
    train_para: Path | None = None
    test_para: Path | None = None
    model_n: int = 256
    mode: str = "inference"  # inference|finetune


def run_grid_lines_ptychovit(cfg: PtychoViTRunnerConfig) -> dict:
    if cfg.model_n != 256:
        raise ValueError("pinn_ptychovit currently supports only N=256")
    if cfg.mode not in {"inference", "finetune"}:
        raise ValueError(f"Unsupported mode '{cfg.mode}'")

    logs_dir = cfg.output_dir / "runs" / "pinn_ptychovit"
    logs_dir.mkdir(parents=True, exist_ok=True)
    recon_npz = cfg.output_dir / "recons" / "pinn_ptychovit" / "recon.npz"

    cmd = [
        "python",
        "scripts/studies/ptychovit_bridge_entrypoint.py",
        "--ptychovit-repo",
        str(cfg.ptychovit_repo),
        "--train-dp",
        str(cfg.train_dp),
        "--test-dp",
        str(cfg.test_dp),
        "--mode",
        cfg.mode,
        "--output-dir",
        str(logs_dir),
        "--recon-npz",
        str(recon_npz),
    ]
    if cfg.train_para is not None:
        cmd.extend(["--train-para", str(cfg.train_para)])
    if cfg.test_para is not None:
        cmd.extend(["--test-para", str(cfg.test_para)])

    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    (logs_dir / "stdout.log").write_text(completed.stdout)
    (logs_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"ptychovit subprocess failed (exit={completed.returncode})")
    if not recon_npz.exists():
        raise RuntimeError(
            f"ptychovit subprocess succeeded but recon artifact missing: {recon_npz}"
        )

    return {
        "status": "ok",
        "run_dir": str(logs_dir),
        "model_id": "pinn_ptychovit",
        "recon_npz": str(recon_npz),
    }


#!/usr/bin/env python3
"""Bridge entrypoint invoked by grid-lines PtychoViT subprocess runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = value.strip().lower()
    if raw in {"1", "true", "t", "yes", "y"}:
        return True
    if raw in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{value}'")


def _infer_para_path(dp_path: Path) -> Path:
    if dp_path.stem.endswith("_dp"):
        return dp_path.with_name(f"{dp_path.stem[:-3]}_para{dp_path.suffix}")
    return dp_path.with_name(f"{dp_path.stem}_para{dp_path.suffix}")


def _copy_pair(dp_path: Path, para_path: Path, out_dir: Path, prefix: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dp_dst = out_dir / f"{prefix}_dp.hdf5"
    para_dst = out_dir / f"{prefix}_para.hdf5"
    shutil.copy2(dp_path, dp_dst)
    shutil.copy2(para_path, para_dst)
    return dp_dst, para_dst


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _prepare_runtime_training_config(args) -> tuple[Path, Dict[str, Any]]:
    repo_cfg_path = args.ptychovit_repo / "config.yaml"
    if not repo_cfg_path.exists():
        raise FileNotFoundError(f"PtychoViT config.yaml not found at {repo_cfg_path}")
    config = _load_yaml(repo_cfg_path)

    train_para = args.train_para or _infer_para_path(args.train_dp)
    test_para = args.test_para or _infer_para_path(args.test_dp)
    if not train_para.exists():
        raise FileNotFoundError(f"train para file not found: {train_para}")
    if not test_para.exists():
        raise FileNotFoundError(f"test para file not found: {test_para}")

    work_dir = args.output_dir / "bridge_work"
    data_dir = work_dir / "data"
    train_dp, train_para = _copy_pair(args.train_dp, train_para, data_dir, prefix="train")
    test_dp, test_para = _copy_pair(args.test_dp, test_para, data_dir, prefix="test")

    config.setdefault("data", {})
    config.setdefault("training", {})
    config.setdefault("paths", {})
    config.setdefault("trainer", {})
    config.setdefault("wandb", {})

    config["data"]["data_path"] = str(data_dir)
    config["data"]["test_path"] = str(test_dp)
    config["data"]["normalization_dict_path"] = None
    config["data"]["test_normalization"] = None
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False
    config["data"]["persistent_workers"] = False
    config["data"]["use_cuda_prefetcher"] = False
    config["data"]["train_split"] = 0.9
    config["data"]["sharding_strategy"] = "static"
    config["data"]["cache_object"] = bool(config["data"].get("cache_object", False))
    config["data"]["max_probe_modes"] = int(config["data"].get("max_probe_modes", 8))

    config["training"]["epochs"] = int(args.epochs)
    config["training"]["batch_size"] = int(args.batch_size)
    config["training"]["learning_rate"] = float(args.learning_rate)
    config["training"]["resume_from_checkpoint"] = bool(args.resume_from_checkpoint)
    config["training"]["validation_plot_freq"] = max(1, int(args.epochs) + 1)
    config["training"]["test_plot_freq"] = max(1, int(args.epochs) + 1)

    config["wandb"]["enabled"] = False

    config["paths"]["model_save_path"] = str(args.output_dir)
    config["trainer"]["run_num"] = int(args.run_num)

    runtime_cfg_path = work_dir / "config.yaml"
    _save_yaml(runtime_cfg_path, config)
    return runtime_cfg_path, config


def _run_training_subprocess(args, runtime_cfg_path: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    repo_str = str(args.ptychovit_repo)
    env["PYTHONPATH"] = repo_str if "PYTHONPATH" not in env else f"{repo_str}:{env['PYTHONPATH']}"
    cmd = ["python", str(args.ptychovit_repo / "main.py")]
    completed = subprocess.run(
        cmd,
        cwd=str(runtime_cfg_path.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _resolve_run_dir(args) -> Path:
    return args.output_dir / f"run{int(args.run_num)}"


def _collect_checkpoint_artifacts(args, run_dir: Path) -> Dict[str, str]:
    required = {
        "best_model": run_dir / "best_model.pth",
        "checkpoint_model": run_dir / "checkpoint_model.pth",
        "checkpoint_state": run_dir / "checkpoint.state",
    }
    for label, path in required.items():
        if not path.exists():
            raise RuntimeError(f"Missing required training artifact {label}: {path}")

    copied: Dict[str, str] = {}
    for label, src in required.items():
        dst_name = src.name
        dst = args.output_dir / dst_name
        shutil.copy2(src, dst)
        copied[label] = str(dst)

    run_cfg = run_dir / "config.yaml"
    if run_cfg.exists():
        cfg_dst = args.output_dir / "config.yaml"
        shutil.copy2(run_cfg, cfg_dst)
        copied["config"] = str(cfg_dst)
    return copied


def _load_checkpoint_path(args) -> Path:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        return checkpoint

    default_ckpt = args.output_dir / "best_model.pth"
    if default_ckpt.exists():
        return default_ckpt
    raise FileNotFoundError(
        "No checkpoint provided and no default best_model.pth found in output-dir"
    )


def _run_model_inference(args, checkpoint_path: Path) -> Path:
    repo_str = str(args.ptychovit_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from data import PtychographyDataset  # pylint: disable=import-error
    from model.model import PtychoViT  # pylint: disable=import-error

    config_path = args.output_dir / "config.yaml"
    if config_path.exists():
        config = _load_yaml(config_path)
    else:
        config = _load_yaml(args.ptychovit_repo / "config.yaml")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    dataset = PtychographyDataset(
        file_path=str(args.test_dp),
        scale=float(data_cfg.get("scale", 10000.0)),
        normalization_dict_path=data_cfg.get("test_normalization"),
        apply_noise=False,
        cache_object=bool(data_cfg.get("cache_object", False)),
        max_probe_modes=int(data_cfg.get("max_probe_modes", 8)),
    )
    if len(dataset) == 0:
        raise RuntimeError("PtychoViT inference dataset is empty")

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.infer_batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PtychoViT(config=model_cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for diff_amp, _amp_patch, _phase_patch, probe, _probe_position, norm, scale in loader:
            input_diff = diff_amp.to(device=device, dtype=torch.float32)
            input_probe = torch.view_as_real(probe.clone().detach()).to(device=device)
            input_norm = torch.as_tensor(norm, dtype=torch.float32, device=device)
            input_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
            _pred_diff, pred_amp, pred_ph = model(input_diff, input_probe, input_norm, input_scale)
            pred_amp = pred_amp.squeeze(1)
            pred_ph = pred_ph.squeeze(1)
            pred_complex = torch.complex(pred_amp * torch.cos(pred_ph), pred_amp * torch.sin(pred_ph))
            preds.append(pred_complex.detach().cpu().numpy())

    pred_stack = np.concatenate(preds, axis=0).astype(np.complex64)
    recon = np.mean(pred_stack, axis=0).astype(np.complex64)

    args.recon_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.recon_npz,
        YY_pred=recon,
        amp=np.abs(recon).astype(np.float32),
        phase=np.angle(recon).astype(np.float32),
    )
    return args.recon_npz


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PtychoViT bridge subprocess entrypoint")
    parser.add_argument("--ptychovit-repo", type=Path, required=True)
    parser.add_argument("--train-dp", type=Path, required=True)
    parser.add_argument("--test-dp", type=Path, required=True)
    parser.add_argument("--train-para", type=Path, default=None)
    parser.add_argument("--test-para", type=Path, default=None)
    parser.add_argument("--mode", type=str, choices=["inference", "finetune"], default="inference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recon-npz", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=_parse_bool, default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--infer-batch-size", type=int, default=8)
    parser.add_argument("--run-num", type=int, default=1)
    args = parser.parse_args(argv)
    if args.recon_npz is None:
        args.recon_npz = args.output_dir / "recons" / "pinn_ptychovit" / "recon.npz"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.ptychovit_repo.exists():
        raise FileNotFoundError(f"ptychovit repo not found: {args.ptychovit_repo}")

    training_returncode = None
    training_stdout = ""
    training_stderr = ""
    checkpoint_artifacts: Dict[str, str] = {}

    if args.mode == "finetune":
        runtime_cfg_path, _config = _prepare_runtime_training_config(args)
        training_returncode, training_stdout, training_stderr = _run_training_subprocess(args, runtime_cfg_path)
        if training_returncode != 0:
            raise RuntimeError(
                f"PtychoViT training subprocess failed (exit={training_returncode})\n"
                f"stdout:\n{training_stdout}\n\nstderr:\n{training_stderr}"
            )
        run_dir = _resolve_run_dir(args)
        checkpoint_artifacts = _collect_checkpoint_artifacts(args, run_dir)

    checkpoint_path = None
    if args.mode == "inference":
        try:
            checkpoint_path = _load_checkpoint_path(args)
        except FileNotFoundError:
            runtime_cfg_path, _config = _prepare_runtime_training_config(args)
            training_returncode, training_stdout, training_stderr = _run_training_subprocess(args, runtime_cfg_path)
            if training_returncode != 0:
                raise RuntimeError(
                    f"PtychoViT bootstrap training subprocess failed (exit={training_returncode})\n"
                    f"stdout:\n{training_stdout}\n\nstderr:\n{training_stderr}"
                )
            run_dir = _resolve_run_dir(args)
            checkpoint_artifacts = _collect_checkpoint_artifacts(args, run_dir)
            checkpoint_path = Path(checkpoint_artifacts["best_model"])
    elif args.mode == "finetune":
        checkpoint_path = Path(checkpoint_artifacts["best_model"])

    recon_path = _run_model_inference(args, checkpoint_path=checkpoint_path)

    manifest = {
        "mode": args.mode,
        "ptychovit_repo": str(args.ptychovit_repo),
        "train_dp": str(args.train_dp),
        "test_dp": str(args.test_dp),
        "train_para": str(args.train_para) if args.train_para else None,
        "test_para": str(args.test_para) if args.test_para else None,
        "recon_npz": str(recon_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "resume_from_checkpoint": bool(args.resume_from_checkpoint),
        "training_returncode": training_returncode,
        "checkpoint_artifacts": checkpoint_artifacts,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

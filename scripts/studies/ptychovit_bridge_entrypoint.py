#!/usr/bin/env python3
"""Bridge entrypoint invoked by grid-lines PtychoViT subprocess runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any, Dict
import h5py
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


def _object_name_from_dp_path(dp_path: Path) -> str:
    stem = dp_path.stem
    if stem.endswith("_dp"):
        return stem[:-3]
    return stem


def _read_dp_max(dp_path: Path) -> float:
    with h5py.File(dp_path, "r") as handle:
        if "dp" not in handle:
            raise KeyError(f"Missing 'dp' dataset in {dp_path}")
        dp = np.asarray(handle["dp"])
    if dp.size == 0:
        raise ValueError(f"Empty 'dp' dataset in {dp_path}")
    max_value = float(np.max(dp))
    if not np.isfinite(max_value):
        raise ValueError(f"Non-finite max(dp) in {dp_path}: {max_value}")
    return max_value


def _write_runtime_normalization_dict(train_dp: Path, test_dp: Path, out_path: Path) -> Path:
    payload = {
        _object_name_from_dp_path(train_dp): _read_dp_max(train_dp),
        _object_name_from_dp_path(test_dp): _read_dp_max(test_dp),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return out_path


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

    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "bridge_work"
    data_dir = work_dir / "data"
    train_dp, train_para = _copy_pair(args.train_dp, train_para, data_dir, prefix="train")
    test_dp, test_para = _copy_pair(args.test_dp, test_para, data_dir, prefix="test")
    normalization_dict_path = _write_runtime_normalization_dict(
        train_dp=train_dp,
        test_dp=test_dp,
        out_path=data_dir / "normalization.pkl",
    )

    config.setdefault("data", {})
    config.setdefault("training", {})
    config.setdefault("paths", {})
    config.setdefault("trainer", {})
    config.setdefault("wandb", {})

    config["data"]["data_path"] = str(data_dir.resolve())
    config["data"]["test_path"] = str(test_dp.resolve())
    config["data"]["normalization_dict_path"] = str(normalization_dict_path.resolve())
    config["data"]["test_normalization"] = str(normalization_dict_path.resolve())
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

    config["paths"]["model_save_path"] = str(output_dir)
    config["trainer"]["run_num"] = int(args.run_num)

    runtime_cfg_path = work_dir / "config.yaml"
    _save_yaml(runtime_cfg_path, config)
    return runtime_cfg_path, config


def _run_training_subprocess(args, runtime_cfg_path: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    repo_str = str(args.ptychovit_repo)
    env["PYTHONPATH"] = repo_str if "PYTHONPATH" not in env else f"{repo_str}:{env['PYTHONPATH']}"
    # Upstream training code has known mixed-device loss paths on GPU; force CPU for stable bridge execution.
    env["CUDA_VISIBLE_DEVICES"] = ""
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


def _stitch_complex_predictions_fallback(
    *,
    patches: np.ndarray,
    positions_px: np.ndarray,
    object_shape: tuple[int, int],
) -> np.ndarray:
    """Integer placement fallback when upstream Fourier-shift helper is unavailable."""
    h_obj, w_obj = int(object_shape[0]), int(object_shape[1])
    canvas = np.zeros((h_obj, w_obj), dtype=np.complex64)
    occupancy = np.zeros((h_obj, w_obj), dtype=np.float32)

    n_scan, h_patch, w_patch = patches.shape
    half_h = (h_patch - 1.0) / 2.0
    half_w = (w_patch - 1.0) / 2.0
    for i in range(n_scan):
        cy = float(positions_px[i, 0])
        cx = float(positions_px[i, 1])
        top = int(np.floor(cy - half_h))
        left = int(np.floor(cx - half_w))
        bottom = top + h_patch
        right = left + w_patch

        dst_y0 = max(0, top)
        dst_x0 = max(0, left)
        dst_y1 = min(h_obj, bottom)
        dst_x1 = min(w_obj, right)
        if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
            continue

        src_y0 = dst_y0 - top
        src_x0 = dst_x0 - left
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        canvas[dst_y0:dst_y1, dst_x0:dst_x1] += patches[i, src_y0:src_y1, src_x0:src_x1]
        occupancy[dst_y0:dst_y1, dst_x0:dst_x1] += 1.0

    return (canvas / np.clip(occupancy, a_min=1.0, a_max=None)).astype(np.complex64)


def _stitch_complex_predictions(
    *,
    patches: np.ndarray,
    positions_px: np.ndarray,
    object_shape: tuple[int, int],
) -> np.ndarray:
    """Assemble predicted scan patches into object space with occupancy normalization."""
    patches = np.asarray(patches, dtype=np.complex64)
    positions_px = np.asarray(positions_px, dtype=np.float32)
    if patches.ndim != 3:
        raise ValueError(f"Expected predicted patches with shape [N,H,W], got {patches.shape}")
    if positions_px.ndim != 2 or positions_px.shape[1] != 2:
        raise ValueError(f"Expected probe positions with shape [N,2], got {positions_px.shape}")
    if patches.shape[0] != positions_px.shape[0]:
        raise ValueError(
            f"Scan-count mismatch between patches ({patches.shape[0]}) and positions ({positions_px.shape[0]})"
        )
    if patches.shape[0] == 0:
        raise ValueError("No predicted patches available for stitching")

    try:
        from utils.ptychi_utils import place_patches_fourier_shift  # pylint: disable=import-error

        patch_tensor = torch.from_numpy(patches)
        position_tensor = torch.from_numpy(positions_px)
        canvas = torch.zeros(tuple(int(v) for v in object_shape), dtype=torch.complex64, device="cpu")
        occupancy = torch.zeros(tuple(int(v) for v in object_shape), dtype=torch.float32, device="cpu")
        canvas = place_patches_fourier_shift(
            canvas,
            position_tensor,
            patch_tensor,
            op="add",
            adjoint_mode=False,
            pad=0,
        )
        occupancy = place_patches_fourier_shift(
            occupancy,
            position_tensor,
            torch.ones_like(patch_tensor, dtype=torch.float32),
            op="add",
            adjoint_mode=False,
            pad=0,
        )
        stitched = canvas / torch.clip(occupancy, min=1.0)
        return stitched.detach().cpu().numpy().astype(np.complex64)
    except Exception:
        return _stitch_complex_predictions_fallback(
            patches=patches,
            positions_px=positions_px,
            object_shape=object_shape,
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
    probe_positions = []
    with torch.no_grad():
        for diff_amp, _amp_patch, _phase_patch, probe, probe_position, norm, scale in loader:
            input_diff = diff_amp.to(device=device, dtype=torch.float32)
            input_probe = torch.view_as_real(probe.clone().detach()).to(device=device)
            input_norm = torch.as_tensor(norm, dtype=torch.float32, device=device)
            input_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
            _pred_diff, pred_amp, pred_ph = model(input_diff, input_probe, input_norm, input_scale)
            pred_amp = pred_amp.squeeze(1)
            pred_ph = pred_ph.squeeze(1)
            pred_complex = torch.complex(pred_amp * torch.cos(pred_ph), pred_amp * torch.sin(pred_ph))
            preds.append(pred_complex.detach().cpu().numpy())
            probe_positions.append(probe_position.detach().cpu().numpy())

    pred_stack = np.concatenate(preds, axis=0).astype(np.complex64)
    position_stack = np.concatenate(probe_positions, axis=0).astype(np.float32)
    if len(dataset.object_shape) != 2:
        raise ValueError(f"Expected object_shape rank-2, got {dataset.object_shape}")
    recon = _stitch_complex_predictions(
        patches=pred_stack,
        positions_px=position_stack,
        object_shape=(int(dataset.object_shape[0]), int(dataset.object_shape[1])),
    )

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
    if args.mode == "finetune" and not bool(args.resume_from_checkpoint):
        raise ValueError(
            "finetune mode requires --resume-from-checkpoint=true to avoid accidental scratch retraining"
        )

    training_returncode = None
    training_stdout = ""
    training_stderr = ""
    checkpoint_artifacts: Dict[str, str] = {}
    runtime_cfg_path: Path | None = None
    runtime_cfg: Dict[str, Any] | None = None

    if args.mode == "inference":
        # Always materialize runtime config + normalization dictionary for inference,
        # even when loading an existing checkpoint.
        runtime_cfg_path, runtime_cfg = _prepare_runtime_training_config(args)
        shutil.copy2(runtime_cfg_path, args.output_dir / "config.yaml")
        args.train_dp = Path(runtime_cfg["data"]["data_path"]) / "train_dp.hdf5"
        args.test_dp = Path(runtime_cfg["data"]["test_path"])

    if args.mode == "finetune":
        runtime_cfg_path, runtime_cfg = _prepare_runtime_training_config(args)
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
            if runtime_cfg_path is None:
                runtime_cfg_path, runtime_cfg = _prepare_runtime_training_config(args)
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

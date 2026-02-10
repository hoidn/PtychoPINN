#!/usr/bin/env python3
"""Orchestrate TF grid-lines workflow + Torch runners and merge metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from ptycho.image.harmonize import resize_complex_to_shape
from ptycho.workflows.grid_lines_workflow import GridLinesConfig
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig


LEGACY_ARCH_TO_MODEL = {
    "cnn": "pinn",
    "baseline": "baseline",
    "fno": "pinn_fno",
    "hybrid": "pinn_hybrid",
    "stable_hybrid": "pinn_stable_hybrid",
    "fno_vanilla": "pinn_fno_vanilla",
    "hybrid_resnet": "pinn_hybrid_resnet",
}

MODEL_TO_LEGACY_ARCH = {model_id: arch for arch, model_id in LEGACY_ARCH_TO_MODEL.items()}
SUPPORTED_MODEL_IDS = set(LEGACY_ARCH_TO_MODEL.values()) | {"pinn_ptychovit"}
MODEL_DEFAULT_N = {"pinn_ptychovit": 256}


def _parse_architectures(value: str) -> Tuple[str, ...]:
    return tuple(a.strip() for a in value.split(",") if a.strip())


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_models(value: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _parse_model_n(value: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not value:
        return out
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --model-n entry '{item}'; expected model_id=N")
        name, raw_n = item.split("=", 1)
        out[name.strip()] = int(raw_n)
    return out


def validate_model_specs(models: Tuple[str, ...], model_n: Dict[str, int]) -> None:
    for model_id in models:
        if model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"Unsupported model '{model_id}'")
    for model_id, n_value in model_n.items():
        if model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"Unsupported model '{model_id}' in --model-n")
        if n_value <= 0:
            raise ValueError(f"Invalid N for model '{model_id}': {n_value}")
    if "pinn_ptychovit" in models and model_n.get("pinn_ptychovit", 256) != 256:
        raise ValueError("pinn_ptychovit currently supports only N=256")


def compute_required_ns(models: Tuple[str, ...], model_n: Dict[str, int], default_n: int) -> list[int]:
    return sorted({model_n.get(model_id, default_n) for model_id in models})


def resolve_model_ns(
    models: Tuple[str, ...],
    model_n_overrides: Dict[str, int],
    default_n: int,
) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    for model_id in models:
        resolved[model_id] = int(model_n_overrides.get(model_id, MODEL_DEFAULT_N.get(model_id, default_n)))
    return resolved


def _load_recon_complex(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "YY_pred" not in data:
            raise KeyError(f"Recon artifact missing YY_pred: {path}")
        return np.squeeze(np.asarray(data["YY_pred"])).astype(np.complex64)


def evaluate_selected_models(recon_paths: Dict[str, Path], gt_path: Path) -> Dict[str, Dict[str, object]]:
    """Evaluate selected model reconstructions on canonical GT object grid."""
    from ptycho.evaluation import eval_reconstruction

    gt_ref = _load_recon_complex(Path(gt_path))
    target_hw = (int(gt_ref.shape[0]), int(gt_ref.shape[1]))
    out: Dict[str, Dict[str, object]] = {}
    for model_id, recon_path in recon_paths.items():
        pred = _load_recon_complex(Path(recon_path))
        pred_ref = resize_complex_to_shape(pred, target_hw)
        metrics = eval_reconstruction(pred_ref[..., None], gt_ref[..., None], label=model_id)
        out[model_id] = {
            "reference_shape": [target_hw[0], target_hw[1]],
            "metrics": metrics,
        }
    return out


def run_grid_lines_compare(
    *,
    N: int,
    gridsize: int,
    output_dir: Path,
    probe_npz: Path,
    architectures: Iterable[str],
    models: Optional[Tuple[str, ...]] = None,
    model_n: Optional[Dict[str, int]] = None,
    reuse_existing_recons: bool = False,
    ptychovit_repo: Optional[Path] = None,
    seed: Optional[int] = None,
    nimgs_train: int = 2,
    nimgs_test: int = 2,
    nphotons: float = 1e9,
    nepochs: int = 60,
    batch_size: int = 16,
    nll_weight: float = 0.0,
    mae_weight: float = 1.0,
    realspace_weight: float = 0.0,
    probe_smoothing_sigma: float = 0.5,
    probe_mask_diameter: Optional[int] = None,
    probe_source: str = "custom",
    probe_scale_mode: str = "pad_extrapolate",
    set_phi: bool = False,
    torch_epochs: Optional[int] = None,
    torch_batch_size: Optional[int] = None,
    torch_learning_rate: float = 1e-3,
    torch_infer_batch_size: int = 16,
    torch_gradient_clip_val: float = 0.0,
    torch_gradient_clip_algorithm: str = "norm",
    torch_output_mode: str = "real_imag",
    torch_loss_mode: str = "mae",
    torch_log_grad_norm: bool = False,
    torch_grad_norm_log_freq: int = 1,
    fno_modes: int = 12,
    fno_width: int = 32,
    fno_blocks: int = 4,
    fno_cnn_blocks: int = 2,
    fno_input_transform: str = "none",
    torch_max_hidden_channels: Optional[int] = None,
    torch_resnet_width: Optional[int] = None,
    torch_optimizer: str = "adam",
    torch_weight_decay: float = 0.0,
    torch_momentum: float = 0.9,
    torch_beta1: float = 0.9,
    torch_beta2: float = 0.999,
    torch_scheduler: str = "Default",
    torch_lr_warmup_epochs: int = 0,
    torch_lr_min_ratio: float = 0.1,
    torch_plateau_factor: float = 0.5,
    torch_plateau_patience: int = 2,
    torch_plateau_min_lr: float = 1e-4,
    torch_plateau_threshold: float = 0.0,
) -> dict:
    os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    output_dir = Path(output_dir)
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**32)
        print(f"[grid_lines_compare_wrapper] Using random seed {seed}")
    architectures = tuple(architectures)
    model_n = dict(model_n or {})
    selected_models: Tuple[str, ...]
    if models:
        selected_models = tuple(models)
    else:
        selected_models = tuple(
            LEGACY_ARCH_TO_MODEL[arch]
            for arch in architectures
            if arch in LEGACY_ARCH_TO_MODEL
        )
    resolved_model_n = resolve_model_ns(selected_models, model_n, default_n=N)
    validate_model_specs(selected_models, resolved_model_n)
    required_ns = compute_required_ns(selected_models, resolved_model_n, default_n=N)

    if models:
        from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair
        from ptycho.interop.ptychovit.validate import validate_hdf5_pair
        from ptycho.workflows import grid_lines_workflow as tf_workflow
        from scripts.studies import grid_lines_torch_runner as torch_runner
        from scripts.studies.grid_lines_ptychovit_runner import (
            PtychoViTRunnerConfig,
            run_grid_lines_ptychovit,
        )

        precomputed_gt = output_dir / "recons" / "gt" / "recon.npz"
        precomputed_recons = {
            model_id: output_dir / "recons" / model_id / "recon.npz"
            for model_id in selected_models
            if (output_dir / "recons" / model_id / "recon.npz").exists()
        }
        if reuse_existing_recons and precomputed_gt.exists() and len(precomputed_recons) == len(selected_models):
            metrics_by_model = evaluate_selected_models(precomputed_recons, precomputed_gt)
            metrics_by_model_path = output_dir / "metrics_by_model.json"
            metrics_by_model_path.write_text(json.dumps(metrics_by_model, indent=2, default=_json_default))
            legacy_metrics = {
                model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()
            }
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(legacy_metrics, indent=2, default=_json_default))
            for model_id in selected_models:
                run_dir = output_dir / "runs" / model_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "stdout.log").write_text(
                    "Skipped backend execution; reused existing reconstruction artifact.\n"
                )
                (run_dir / "stderr.log").write_text("")

            from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals

            render_grid_lines_visuals(output_dir, order=tuple(["gt", *precomputed_recons.keys()]))
            return {
                "train_npz": "",
                "test_npz": "",
                "metrics": legacy_metrics,
                "metrics_by_model": metrics_by_model,
                "gt_recon": str(precomputed_gt),
                "recon_paths": {k: str(v) for k, v in precomputed_recons.items()},
            }

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
            probe_mask_diameter=probe_mask_diameter,
            probe_source=probe_source,
            probe_scale_mode=probe_scale_mode,
            set_phi=set_phi,
        )
        bundles_by_n = tf_workflow.build_grid_lines_datasets_by_n(tf_cfg, required_ns=required_ns)
        gt_candidates = {bundle["gt_recon"] for bundle in bundles_by_n.values() if "gt_recon" in bundle}
        if not gt_candidates:
            raise RuntimeError("Dataset builders did not provide canonical gt_recon path")
        if len(gt_candidates) != 1:
            raise RuntimeError("Multiple canonical GT paths detected across N bundles")
        gt_path = Path(next(iter(gt_candidates)))

        recon_paths: Dict[str, Path] = {}
        for model_id in selected_models:
            n_for_model = resolved_model_n[model_id]
            bundle = bundles_by_n[n_for_model]
            train_npz = Path(bundle["train_npz"])
            test_npz = Path(bundle["test_npz"])
            existing_recon = output_dir / "recons" / model_id / "recon.npz"
            if reuse_existing_recons and existing_recon.exists():
                recon_paths[model_id] = existing_recon
                continue

            if model_id in {
                "pinn_fno",
                "pinn_hybrid",
                "pinn_stable_hybrid",
                "pinn_fno_vanilla",
                "pinn_hybrid_resnet",
            }:
                arch = MODEL_TO_LEGACY_ARCH[model_id]
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
                    gradient_clip_algorithm=torch_gradient_clip_algorithm,
                    generator_output_mode=torch_output_mode,
                    N=n_for_model,
                    gridsize=gridsize,
                    torch_loss_mode=torch_loss_mode,
                    fno_modes=fno_modes,
                    fno_width=fno_width,
                    fno_blocks=fno_blocks,
                    fno_cnn_blocks=fno_cnn_blocks,
                    fno_input_transform=fno_input_transform,
                    max_hidden_channels=torch_max_hidden_channels,
                    resnet_width=torch_resnet_width,
                    optimizer=torch_optimizer,
                    weight_decay=torch_weight_decay,
                    momentum=torch_momentum,
                    adam_beta1=torch_beta1,
                    adam_beta2=torch_beta2,
                    log_grad_norm=torch_log_grad_norm,
                    grad_norm_log_freq=torch_grad_norm_log_freq,
                    scheduler=torch_scheduler,
                    lr_warmup_epochs=torch_lr_warmup_epochs,
                    lr_min_ratio=torch_lr_min_ratio,
                    plateau_factor=torch_plateau_factor,
                    plateau_patience=torch_plateau_patience,
                    plateau_min_lr=torch_plateau_min_lr,
                    plateau_threshold=torch_plateau_threshold,
                )
                torch_result = torch_runner.run_grid_lines_torch(torch_cfg)
                recon_path = torch_result.get("recon_npz")
                if recon_path is None:
                    recon_path = output_dir / "recons" / model_id / "recon.npz"
                recon_paths[model_id] = Path(recon_path)
                continue

            if model_id == "pinn_ptychovit":
                interop_dir = output_dir / "interop" / "pinn_ptychovit" / f"N{n_for_model}"
                train_pair = convert_npz_split_to_hdf5_pair(
                    npz_path=train_npz,
                    out_dir=interop_dir / "train",
                    object_name=f"grid_lines_train_N{n_for_model}",
                )
                test_pair = convert_npz_split_to_hdf5_pair(
                    npz_path=test_npz,
                    out_dir=interop_dir / "test",
                    object_name=f"grid_lines_test_N{n_for_model}",
                )
                validate_hdf5_pair(train_pair.dp_hdf5, train_pair.para_hdf5)
                validate_hdf5_pair(test_pair.dp_hdf5, test_pair.para_hdf5)

                canonical_interop_dir = output_dir / "interop"
                canonical_interop_dir.mkdir(parents=True, exist_ok=True)
                canonical_train_dp = canonical_interop_dir / "train_dp.hdf5"
                canonical_train_para = canonical_interop_dir / "train_para.hdf5"
                canonical_test_dp = canonical_interop_dir / "test_dp.hdf5"
                canonical_test_para = canonical_interop_dir / "test_para.hdf5"
                shutil.copy2(train_pair.dp_hdf5, canonical_train_dp)
                shutil.copy2(train_pair.para_hdf5, canonical_train_para)
                shutil.copy2(test_pair.dp_hdf5, canonical_test_dp)
                shutil.copy2(test_pair.para_hdf5, canonical_test_para)

                pvit_cfg = PtychoViTRunnerConfig(
                    ptychovit_repo=Path(ptychovit_repo or "/home/ollie/Documents/ptycho-vit"),
                    output_dir=output_dir,
                    train_dp=train_pair.dp_hdf5,
                    test_dp=test_pair.dp_hdf5,
                    train_para=train_pair.para_hdf5,
                    test_para=test_pair.para_hdf5,
                    model_n=n_for_model,
                    mode="inference",
                )
                pvit_result = run_grid_lines_ptychovit(pvit_cfg)
                recon_paths[model_id] = Path(pvit_result["recon_npz"])
                continue

            if model_id in {"pinn", "baseline"}:
                tf_model_cfg = GridLinesConfig(
                    N=n_for_model,
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
                    probe_mask_diameter=probe_mask_diameter,
                    probe_source=probe_source,
                    probe_scale_mode=probe_scale_mode,
                    set_phi=set_phi,
                )
                tf_result = tf_workflow.run_grid_lines_workflow(tf_model_cfg)
                recon_paths[model_id] = Path(output_dir / "recons" / model_id / "recon.npz")
                if not recon_paths[model_id].exists():
                    raise RuntimeError(
                        f"Expected recon artifact missing for {model_id}: {recon_paths[model_id]}"
                    )
                _ = tf_result
                continue

            raise ValueError(f"Unsupported model '{model_id}'")

        metrics_by_model = evaluate_selected_models(recon_paths, gt_path)
        metrics_by_model_path = output_dir / "metrics_by_model.json"
        metrics_by_model_path.write_text(json.dumps(metrics_by_model, indent=2, default=_json_default))

        legacy_metrics = {
            model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(legacy_metrics, indent=2, default=_json_default))

        from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals

        render_grid_lines_visuals(output_dir, order=tuple(["gt", *recon_paths.keys()]))
        first_bundle = bundles_by_n[required_ns[0]]
        return {
            "train_npz": str(first_bundle["train_npz"]),
            "test_npz": str(first_bundle["test_npz"]),
            "metrics": legacy_metrics,
            "metrics_by_model": metrics_by_model,
            "gt_recon": str(gt_path),
            "recon_paths": {k: str(v) for k, v in recon_paths.items()},
        }

    dataset_dir = output_dir / "datasets" / f"N{N}" / f"gs{gridsize}"
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"

    tf_metrics = {}
    selected_architectures = architectures if not models else tuple(
        MODEL_TO_LEGACY_ARCH[m]
        for m in selected_models
        if m in MODEL_TO_LEGACY_ARCH
    )

    if ("cnn" in selected_architectures or "baseline" in selected_architectures) or not train_npz.exists() or not test_npz.exists():
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
            probe_mask_diameter=probe_mask_diameter,
            probe_source=probe_source,
            probe_scale_mode=probe_scale_mode,
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
    if "cnn" in selected_architectures and "pinn" in tf_metrics:
        merged["pinn"] = tf_metrics["pinn"]
    if "baseline" in selected_architectures and "baseline" in tf_metrics:
        merged["baseline"] = tf_metrics["baseline"]

    for arch in selected_architectures:
        if arch in ("fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"):
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
                gradient_clip_algorithm=torch_gradient_clip_algorithm,
                generator_output_mode=torch_output_mode,
                N=N,
                gridsize=gridsize,
                torch_loss_mode=torch_loss_mode,
                fno_modes=fno_modes,
                fno_width=fno_width,
                fno_blocks=fno_blocks,
                fno_cnn_blocks=fno_cnn_blocks,
                fno_input_transform=fno_input_transform,
                max_hidden_channels=torch_max_hidden_channels,
                resnet_width=torch_resnet_width,
                optimizer=torch_optimizer,
                weight_decay=torch_weight_decay,
                momentum=torch_momentum,
                adam_beta1=torch_beta1,
                adam_beta2=torch_beta2,
                log_grad_norm=torch_log_grad_norm,
                grad_norm_log_freq=torch_grad_norm_log_freq,
                scheduler=torch_scheduler,
                lr_warmup_epochs=torch_lr_warmup_epochs,
                lr_min_ratio=torch_lr_min_ratio,
                plateau_factor=torch_plateau_factor,
                plateau_patience=torch_plateau_patience,
                plateau_min_lr=torch_plateau_min_lr,
                plateau_threshold=torch_plateau_threshold,
            )
            from scripts.studies import grid_lines_torch_runner as torch_runner
            torch_result = torch_runner.run_grid_lines_torch(torch_cfg)
            if "metrics" in torch_result:
                merged[f"pinn_{arch}"] = torch_result["metrics"]

    order = ["gt"]
    if "cnn" in selected_architectures:
        order.append("pinn")
    if "baseline" in selected_architectures:
        order.append("baseline")
    if "fno" in selected_architectures:
        order.append("pinn_fno")
    if "hybrid" in selected_architectures:
        order.append("pinn_hybrid")
    if "stable_hybrid" in selected_architectures:
        order.append("pinn_stable_hybrid")
    if "fno_vanilla" in selected_architectures:
        order.append("pinn_fno_vanilla")
    if "hybrid_resnet" in selected_architectures:
        order.append("pinn_hybrid_resnet")

    from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals
    render_grid_lines_visuals(output_dir, order=tuple(order))

    metrics_path.write_text(json.dumps(merged, indent=2, default=_json_default))
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
    parser.add_argument(
        "--architectures",
        type=str,
        default="cnn,baseline,fno,hybrid,stable_hybrid,fno_vanilla,hybrid_resnet",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional explicit model IDs (overrides --architectures).",
    )
    parser.add_argument(
        "--model-n",
        type=str,
        default="",
        help="Optional per-model N overrides as comma-separated model_id=N entries.",
    )
    parser.add_argument(
        "--reuse-existing-recons",
        action="store_true",
        help="Reuse existing recon artifacts in output-dir instead of re-running selected backends.",
    )
    parser.add_argument(
        "--ptychovit-repo",
        type=Path,
        default=Path("/home/ollie/Documents/ptycho-vit"),
        help="Path to local ptycho-vit checkout for subprocess execution.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if omitted)")
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
    parser.add_argument("--set-phi", action="store_true", help="Enable non-zero phase in synthetic grid data.")
    parser.add_argument("--torch-epochs", type=int, default=None)
    parser.add_argument("--torch-batch-size", type=int, default=None)
    parser.add_argument("--torch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-infer-batch-size", type=int, default=16)
    parser.add_argument(
        "--torch-grad-clip",
        type=float,
        default=0.0,
        help="Torch gradient clipping max norm (<=0 disables clipping).",
    )
    parser.add_argument(
        "--torch-grad-clip-algorithm",
        type=str,
        default="norm",
        choices=["norm", "value", "agc"],
        help="Torch gradient clipping algorithm.",
    )
    parser.add_argument(
        "--torch-output-mode",
        type=str,
        default="real_imag",
        choices=["real_imag", "amp_phase_logits", "amp_phase"],
        help="Torch generator output mode.",
    )
    parser.add_argument("--torch-loss-mode", type=str, default="mae", choices=["poisson", "mae"])
    parser.add_argument("--torch-log-grad-norm", action="store_true")
    parser.add_argument("--torch-grad-norm-log-freq", type=int, default=1)
    parser.add_argument("--fno-modes", type=int, default=12)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-blocks", type=int, default=4)
    parser.add_argument("--fno-cnn-blocks", type=int, default=2)
    parser.add_argument(
        "--fno-input-transform",
        type=str,
        default="none",
        choices=["none", "sqrt", "log1p", "instancenorm"],
    )
    parser.add_argument("--torch-max-hidden-channels", type=int, default=None,
                        help="Cap on hidden channels in Hybrid encoder (default: no cap)")
    parser.add_argument("--torch-resnet-width", type=int, default=None,
                        help="Fixed bottleneck width for hybrid_resnet (must be divisible by 4)")
    parser.add_argument("--torch-optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"], help="Optimizer algorithm")
    parser.add_argument("--torch-weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--torch-momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--torch-beta1", type=float, default=0.9, help="Adam/AdamW beta1")
    parser.add_argument("--torch-beta2", type=float, default=0.999, help="Adam/AdamW beta2")
    parser.add_argument("--torch-scheduler", type=str, default="Default",
                        choices=["Default", "Exponential", "WarmupCosine", "ReduceLROnPlateau"])
    parser.add_argument("--torch-lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--torch-lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--torch-plateau-factor", type=float, default=0.5)
    parser.add_argument("--torch-plateau-patience", type=int, default=2)
    parser.add_argument("--torch-plateau-min-lr", type=float, default=1e-4)
    parser.add_argument("--torch-plateau-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.architectures = _parse_architectures(args.architectures)
    args.models = _parse_models(args.models) if args.models else None
    args.model_n = _parse_model_n(args.model_n)
    if args.models:
        validate_model_specs(args.models, resolve_model_ns(args.models, args.model_n, default_n=args.N))
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_grid_lines_compare(
        N=args.N,
        gridsize=args.gridsize,
        output_dir=args.output_dir,
        probe_npz=args.probe_npz,
        architectures=args.architectures,
        models=args.models,
        model_n=args.model_n,
        reuse_existing_recons=args.reuse_existing_recons,
        ptychovit_repo=args.ptychovit_repo,
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
        probe_mask_diameter=args.probe_mask_diameter,
        probe_source=args.probe_source,
        probe_scale_mode=args.probe_scale_mode,
        set_phi=args.set_phi,
        torch_epochs=args.torch_epochs,
        torch_batch_size=args.torch_batch_size,
        torch_learning_rate=args.torch_learning_rate,
        torch_infer_batch_size=args.torch_infer_batch_size,
        torch_gradient_clip_val=args.torch_grad_clip,
        torch_gradient_clip_algorithm=args.torch_grad_clip_algorithm,
        torch_output_mode=args.torch_output_mode,
        torch_loss_mode=args.torch_loss_mode,
        torch_log_grad_norm=args.torch_log_grad_norm,
        torch_grad_norm_log_freq=args.torch_grad_norm_log_freq,
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_blocks=args.fno_blocks,
        fno_cnn_blocks=args.fno_cnn_blocks,
        fno_input_transform=args.fno_input_transform,
        torch_max_hidden_channels=args.torch_max_hidden_channels,
        torch_resnet_width=args.torch_resnet_width,
        torch_optimizer=args.torch_optimizer,
        torch_weight_decay=args.torch_weight_decay,
        torch_momentum=args.torch_momentum,
        torch_beta1=args.torch_beta1,
        torch_beta2=args.torch_beta2,
        torch_scheduler=args.torch_scheduler,
        torch_lr_warmup_epochs=args.torch_lr_warmup_epochs,
        torch_lr_min_ratio=args.torch_lr_min_ratio,
        torch_plateau_factor=args.torch_plateau_factor,
        torch_plateau_patience=args.torch_plateau_patience,
        torch_plateau_min_lr=args.torch_plateau_min_lr,
        torch_plateau_threshold=args.torch_plateau_threshold,
    )


if __name__ == "__main__":
    main()

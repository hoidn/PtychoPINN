#!/usr/bin/env python3
"""Checkpoint-reuse inference helpers for cross-dataset hybrid_resnet evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ptycho.config.config import PyTorchExecutionConfig
from ptycho_torch.config_factory import create_training_payload
from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
from ptycho_torch.generators.registry import resolve_generator
from scripts.studies.grid_lines_torch_runner import (
    TorchRunnerConfig,
    _harmonize_prediction_shape,
    _reassemble_with_coords_offsets,
    _stitch_for_metrics,
    load_cached_dataset_with_metadata,
    run_torch_inference,
    to_complex_patches,
)


def _build_model_for_config(cfg: TorchRunnerConfig):
    overrides = {
        "N": int(cfg.N),
        "gridsize": int(cfg.gridsize),
        "n_groups": 1,
        "batch_size": int(cfg.batch_size),
        "epochs": 1,
        "learning_rate": float(cfg.learning_rate),
        "architecture": str(cfg.architecture),
        "fno_modes": int(cfg.fno_modes),
        "fno_width": int(cfg.fno_width),
        "fno_blocks": int(cfg.fno_blocks),
        "fno_cnn_blocks": int(cfg.fno_cnn_blocks),
        "fno_input_transform": str(cfg.fno_input_transform),
        "max_hidden_channels": cfg.max_hidden_channels,
        "resnet_width": cfg.resnet_width,
        "generator_output_mode": str(cfg.generator_output_mode),
        "object_big": False,
        "probe_big": False,
    }
    payload = create_training_payload(
        train_data_file=Path(cfg.train_npz),
        output_dir=Path(cfg.output_dir),
        overrides=overrides,
        execution_config=PyTorchExecutionConfig(
            learning_rate=float(cfg.learning_rate),
            deterministic=True,
            logger_backend="none",
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
    )
    generator = resolve_generator(payload.tf_training_config)
    model = generator.build_model(
        {
            "model_config": payload.pt_model_config,
            "data_config": payload.pt_data_config,
            "training_config": payload.pt_training_config,
            "inference_config": PTInferenceConfig(),
        }
    )
    return model


def _load_model_from_checkpoint(config: TorchRunnerConfig, checkpoint_path: Path):
    model = _build_model_for_config(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _select_ground_truth(data: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ("YY_ground_truth", "YY_full", "objectGuess"):
        if key in data:
            return np.asarray(data[key])
    raise ValueError("Test data must provide one of: YY_ground_truth, YY_full, objectGuess.")


def _run_single_dataset_inference(
    model,
    config: TorchRunnerConfig,
    dataset_name: str,
    test_npz: Path,
    recon_npz: Path,
    allow_oom_fallback: bool = True,
) -> Path:
    _ = dataset_name
    test_data, test_metadata = load_cached_dataset_with_metadata(Path(test_npz))
    predictions = run_torch_inference(model, test_data, config, metadata=test_metadata)
    pred_for_metrics = to_complex_patches(predictions) if predictions.shape[-1] == 2 else predictions

    ground_truth = _select_ground_truth(test_data)
    if config.reassembly_mode == "position":
        pred_for_metrics = _reassemble_with_coords_offsets(
            pred_for_metrics,
            test_data,
            M=config.N,
            backend=config.position_reassembly_backend,
            batch_size=config.position_reassembly_batch_size,
            allow_oom_fallback=allow_oom_fallback,
        )
    elif pred_for_metrics.ndim >= 3:
        pred_h, pred_w = pred_for_metrics.shape[-3], pred_for_metrics.shape[-2]
        gt_h, gt_w = ground_truth.shape[-2], ground_truth.shape[-1]
        if (pred_h, pred_w) != (gt_h, gt_w):
            norm_y_i = test_data.get("norm_Y_I", 1.0)
            pred_for_metrics = _stitch_for_metrics(pred_for_metrics, config, test_metadata, norm_y_i)

    pred_for_metrics = _harmonize_prediction_shape(pred_for_metrics, ground_truth)
    recon_complex = np.asarray(pred_for_metrics, dtype=np.complex64)
    recon_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        recon_npz,
        YY_pred=recon_complex,
        amp=np.abs(recon_complex).astype(np.float32),
        phase=np.angle(recon_complex).astype(np.float32),
    )
    return recon_npz


def run_cross_dataset_hybrid_inference(
    *,
    model_pt: Path,
    dataset_npzs: Dict[str, Path],
    output_dir: Path,
    base_cfg: TorchRunnerConfig,
    allow_oom_fallback: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Reuse one trained checkpoint to run hybrid_resnet inference on multiple datasets."""
    model_pt = Path(model_pt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_cfg.reassembly_mode != "position":
        raise ValueError(
            "Cross-dataset hybrid inference requires position reassembly mode for external offsets."
        )
    allowed_backends = {"auto", "shift_sum", "batched"}
    if base_cfg.position_reassembly_backend not in allowed_backends:
        raise ValueError(
            f"Unsupported position reassembly backend {base_cfg.position_reassembly_backend!r}; "
            f"expected one of {sorted(allowed_backends)}."
        )

    model = _load_model_from_checkpoint(base_cfg, model_pt)
    results: Dict[str, Dict[str, str]] = {}
    for dataset_name, test_npz in dataset_npzs.items():
        dataset_out = output_dir / dataset_name
        cfg = replace(base_cfg, output_dir=dataset_out, test_npz=Path(test_npz))
        recon_npz = dataset_out / "recons" / "pinn_hybrid_resnet" / "recon.npz"
        recon_path = _run_single_dataset_inference(
            model=model,
            config=cfg,
            dataset_name=dataset_name,
            test_npz=Path(test_npz),
            recon_npz=recon_npz,
            allow_oom_fallback=allow_oom_fallback,
        )
        results[dataset_name] = {
            "recon_npz": str(recon_path),
            "test_npz": str(test_npz),
        }

    manifest_path = output_dir / "hybrid_cross_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_pt": str(model_pt),
                "architecture": base_cfg.architecture,
                "position_reassembly_backend": base_cfg.position_reassembly_backend,
                "allow_oom_fallback": bool(allow_oom_fallback),
                "datasets": {name: payload["test_npz"] for name, payload in results.items()},
            },
            indent=2,
        )
    )
    return results

#!/usr/bin/env python3
"""Torch runner for grid-lines workflow with FNO/hybrid architectures.

This runner executes PyTorch-based training and inference for FNO and hybrid
architectures, consuming cached NPZ datasets from the TensorFlow grid-lines
workflow and producing compatible metrics JSON for comparison.

Contract:
    Inputs:
        - train_npz: Path to cached training dataset (from grid_lines_workflow)
        - test_npz: Path to cached test dataset (from grid_lines_workflow)
        - output_dir: Base output directory for artifacts
        - architecture: 'fno' or 'hybrid'
        - seed: Random seed for reproducibility
        - Training hyperparams (epochs, batch_size, learning_rate)

    Outputs:
        - Artifacts under output_dir/runs/pinn_<arch>/
        - Metrics JSON compatible with TF workflow (same keys)

Usage:
    python grid_lines_torch_runner.py \\
        --train-npz datasets/train.npz \\
        --test-npz datasets/test.npz \\
        --output-dir outputs/grid_lines \\
        --architecture fno \\
        --seed 42

See also:
    - ptycho/workflows/grid_lines_workflow.py (TF harness)
    - ptycho_torch/generators/ (architecture implementations)
    - docs/plans/2026-01-27-grid-lines-workflow.md
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


PAPER_MODEL_LABELS = {
    "ffno": "FFNO + PINN",
    "fno": "FNO + PINN",
    "hybrid": "Hybrid + PINN",
    "stable_hybrid": "Stable Hybrid + PINN",
    "fno_vanilla": "FNO Vanilla + PINN",
    "neuralop_uno": "U-NO + PINN",
    "hybrid_resnet": "Hybrid ResNet + PINN",
    "spectral_resnet_bottleneck_net": "Spectral ResNet Bottleneck + PINN",
    "spectral_resnet_bottleneck_linear_decoder": "Spectral ResNet Linear Decoder + PINN",
    "hybrid_resnet_ffno_bottleneck": "Hybrid ResNet FFNO Bottleneck + PINN",
}


def _runner_model_id(
    architecture: str,
    training_procedure: str,
    model_id_override: Optional[str] = None,
) -> str:
    if model_id_override:
        return str(model_id_override)
    if training_procedure == "supervised":
        return f"supervised_{architecture}"
    return f"pinn_{architecture}"


def _paper_model_label(
    architecture: str,
    training_procedure: str,
    model_label_override: Optional[str] = None,
) -> str:
    if model_label_override:
        return str(model_label_override)
    label = PAPER_MODEL_LABELS.get(architecture, architecture)
    if training_procedure == "supervised":
        if label.endswith(" + PINN"):
            return label[:-7] + " + supervised"
        return f"{label} + supervised"
    return label


def _json_default(value: object) -> object:
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


def derive_channel_count(gridsize: int) -> int:
    """Derive channel count (C) from gridsize.

    The grid-lines datasets store diffraction patches with channel dimension
    equal to gridsize**2. Keeping this explicit avoids mismatches when
    gridsize=1 (C=1) vs gridsize=2 (C=4).
    """
    return int(gridsize) * int(gridsize)


def to_complex_patches(real_imag: np.ndarray) -> np.ndarray:
    """Convert real/imag output tensor to complex patches.

    FNO/Hybrid models output predictions in real/imag format with shape
    (..., 2) where the last dimension contains [real, imag]. This function
    converts that to complex64 format.

    Args:
        real_imag: Array with shape (..., 2) containing real and imaginary parts

    Returns:
        Complex array with shape (...) (last dimension collapsed)

    See also:
        docs/plans/2026-01-27-fno-hybrid-testing-gaps-addendum.md Task 3
    """
    real = real_imag[..., 0]
    imag = real_imag[..., 1]
    return (real + 1j * imag).astype(np.complex64)


@dataclass
class TorchRunnerConfig:
    """Configuration for Torch grid-lines runner."""
    train_npz: Path
    test_npz: Path
    output_dir: Path
    architecture: str  # 'ffno', 'fno', or 'hybrid'
    training_procedure: str = "pinn"
    model_id_override: Optional[str] = None
    model_label_override: Optional[str] = None
    seed: int = 42
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    infer_batch_size: int = 128
    gradient_clip_val: Optional[float] = 0.0
    gradient_clip_algorithm: str = 'norm'  # 'norm', 'value', or 'agc'
    generator_output_mode: str = "real_imag"
    N: int = 64
    gridsize: int = 1
    probe_source: Optional[str] = None
    torch_loss_mode: str = "mae"
    torch_mae_pred_l2_match_target: bool = False
    probe_mask: bool = False
    probe_mask_sigma: float = 1.0
    probe_mask_diameter: Optional[float] = None
    fno_modes: int = 12
    fno_width: int = 32
    fno_blocks: int = 4
    fno_cnn_blocks: int = 2
    fno_input_transform: str = "none"
    max_hidden_channels: Optional[int] = None
    resnet_width: Optional[int] = None
    hybrid_skip_connections: bool = False
    hybrid_downsample_steps: int = 2
    hybrid_downsample_op: str = "stride_conv"
    hybrid_encoder_conv_hidden_scale: float = 2.0
    hybrid_encoder_spectral_hidden_scale: float = 1.0
    # Legacy absolute-width aliases retained for compatibility with older runbooks.
    hybrid_encoder_conv_hidden_channels: Optional[int] = None
    hybrid_encoder_spectral_hidden_channels: Optional[int] = None
    hybrid_resnet_blocks: int = 6
    hybrid_skip_style: str = "add"
    spectral_bottleneck_blocks: int = 6
    spectral_bottleneck_modes: int = 12
    spectral_bottleneck_share_weights: bool = True
    spectral_bottleneck_gate_init: float = 0.1
    spectral_bottleneck_gate_mode: str = "shared"
    optimizer: str = 'adam'  # 'adam', 'adamw', or 'sgd'
    weight_decay: float = 0.0
    momentum: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    log_grad_norm: bool = False
    grad_norm_log_freq: int = 1
    enable_checkpointing: bool = True
    scheduler: str = 'Default'
    lr_warmup_epochs: int = 0
    lr_min_ratio: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 5e-5
    plateau_threshold: float = 0.0
    # Recon logging
    logger_backend: Optional[str] = "csv"  # 'csv', 'mlflow', etc.; set None to disable
    recon_log_every_n_epochs: Optional[int] = None
    recon_log_num_patches: int = 4
    recon_log_fixed_indices: Optional[List[int]] = None
    recon_log_stitch: bool = False
    recon_log_max_stitch_samples: Optional[int] = None
    reassembly_mode: str = "grid_lines"  # "grid_lines" | "position"
    position_reassembly_backend: str = "auto"  # "auto" | "shift_sum" | "batched"
    position_reassembly_batch_size: int = 64
    # None means auto default (min(patch_h, patch_w) // 4).
    position_crop_border: Optional[int] = None


def _build_runner_cli_argv(cfg: TorchRunnerConfig) -> List[str]:
    argv = [
        "--train-npz", str(cfg.train_npz),
        "--test-npz", str(cfg.test_npz),
        "--output-dir", str(cfg.output_dir),
        "--architecture", str(cfg.architecture),
        "--training-procedure", str(cfg.training_procedure),
        "--seed", str(cfg.seed),
        "--epochs", str(cfg.epochs),
        "--batch-size", str(cfg.batch_size),
        "--learning-rate", str(cfg.learning_rate),
        "--infer-batch-size", str(cfg.infer_batch_size),
        "--grad-clip", str(cfg.gradient_clip_val or 0.0),
        "--gradient-clip-algorithm", str(cfg.gradient_clip_algorithm),
        "--output-mode", str(cfg.generator_output_mode),
        "--torch-loss-mode", str(cfg.torch_loss_mode),
        "--fno-modes", str(cfg.fno_modes),
        "--fno-width", str(cfg.fno_width),
        "--fno-blocks", str(cfg.fno_blocks),
        "--fno-cnn-blocks", str(cfg.fno_cnn_blocks),
        "--optimizer", str(cfg.optimizer),
        "--weight-decay", str(cfg.weight_decay),
        "--momentum", str(cfg.momentum),
        "--beta1", str(cfg.adam_beta1),
        "--beta2", str(cfg.adam_beta2),
        "--scheduler", str(cfg.scheduler),
        "--lr-warmup-epochs", str(cfg.lr_warmup_epochs),
        "--lr-min-ratio", str(cfg.lr_min_ratio),
        "--plateau-factor", str(cfg.plateau_factor),
        "--plateau-patience", str(cfg.plateau_patience),
        "--plateau-min-lr", str(cfg.plateau_min_lr),
        "--plateau-threshold", str(cfg.plateau_threshold),
        "--N", str(cfg.N),
        "--gridsize", str(cfg.gridsize),
    ]
    if cfg.probe_source:
        argv.extend(["--probe-source", str(cfg.probe_source)])
    if cfg.torch_mae_pred_l2_match_target:
        argv.append("--torch-mae-pred-l2-match-target")
    else:
        argv.append("--no-torch-mae-pred-l2-match-target")
    if cfg.probe_mask:
        argv.append("--probe-mask")
    else:
        argv.append("--no-probe-mask")
    argv.extend(["--probe-mask-sigma", str(cfg.probe_mask_sigma)])
    if cfg.probe_mask_diameter is not None:
        argv.extend(["--probe-mask-diameter", str(cfg.probe_mask_diameter)])
    if cfg.hybrid_skip_connections:
        argv.append("--hybrid-skip-connections")
    else:
        argv.append("--no-hybrid-skip-connections")
    argv.extend(
        [
            "--hybrid-downsample-steps", str(cfg.hybrid_downsample_steps),
            "--hybrid-downsample-op", str(cfg.hybrid_downsample_op),
            "--hybrid-encoder-conv-hidden-scale", str(cfg.hybrid_encoder_conv_hidden_scale),
            "--hybrid-encoder-spectral-hidden-scale", str(cfg.hybrid_encoder_spectral_hidden_scale),
            "--hybrid-resnet-blocks", str(cfg.hybrid_resnet_blocks),
            "--hybrid-skip-style", str(cfg.hybrid_skip_style),
            "--spectral-bottleneck-blocks", str(cfg.spectral_bottleneck_blocks),
            "--spectral-bottleneck-modes", str(cfg.spectral_bottleneck_modes),
            "--spectral-bottleneck-gate-init", str(cfg.spectral_bottleneck_gate_init),
            "--spectral-bottleneck-gate-mode", str(cfg.spectral_bottleneck_gate_mode),
        ]
    )
    if cfg.spectral_bottleneck_share_weights:
        argv.append("--spectral-bottleneck-share-weights")
    else:
        argv.append("--no-spectral-bottleneck-share-weights")
    if cfg.hybrid_encoder_conv_hidden_channels is not None:
        argv.extend(["--hybrid-encoder-conv-hidden", str(cfg.hybrid_encoder_conv_hidden_channels)])
    if cfg.hybrid_encoder_spectral_hidden_channels is not None:
        argv.extend(["--hybrid-encoder-spectral-hidden", str(cfg.hybrid_encoder_spectral_hidden_channels)])
    if cfg.resnet_width is not None:
        argv.extend(["--torch-resnet-width", str(cfg.resnet_width)])
    if cfg.log_grad_norm:
        argv.append("--log-grad-norm")
    argv.extend(["--grad-norm-log-freq", str(cfg.grad_norm_log_freq)])
    if cfg.position_crop_border is not None:
        argv.extend(["--position-crop-border", str(cfg.position_crop_border)])
    if cfg.logger_backend is None:
        argv.extend(["--torch-logger", "none"])
    else:
        argv.extend(["--torch-logger", str(cfg.logger_backend)])
    if cfg.recon_log_every_n_epochs is not None:
        argv.extend(["--recon-log-every-n-epochs", str(cfg.recon_log_every_n_epochs)])
    argv.extend(["--recon-log-num-patches", str(cfg.recon_log_num_patches)])
    if cfg.recon_log_fixed_indices:
        argv.extend(["--recon-log-fixed-indices", *[str(v) for v in cfg.recon_log_fixed_indices]])
    if cfg.recon_log_stitch:
        argv.append("--recon-log-stitch")
    if cfg.recon_log_max_stitch_samples is not None:
        argv.extend(["--recon-log-max-stitch-samples", str(cfg.recon_log_max_stitch_samples)])
    return argv


def _write_runner_invocation_artifacts(
    cfg: TorchRunnerConfig,
    *,
    argv: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    from scripts.studies.invocation_logging import (
        capture_runtime_provenance,
        get_git_commit,
        write_invocation_artifacts,
    )

    run_dir = cfg.output_dir / "runs" / _runner_model_id(
        cfg.architecture,
        cfg.training_procedure,
        cfg.model_id_override,
    )
    invocation_extra: Dict[str, Any] = {
        "runtime_provenance": capture_runtime_provenance(),
        "git_commit": get_git_commit(REPO_ROOT),
        "invocation_mode": "library",
    }
    if extra:
        invocation_extra.update(extra)
    json_path, _ = write_invocation_artifacts(
        output_dir=run_dir,
        script_path="scripts/studies/grid_lines_torch_runner.py",
        argv=list(argv) if argv is not None else _build_runner_cli_argv(cfg),
        parsed_args=asdict(cfg),
        extra=invocation_extra,
    )
    return json_path


def _validate_position_reassembly_config(cfg: TorchRunnerConfig) -> None:
    allowed = {"auto", "shift_sum", "batched"}
    if cfg.position_reassembly_backend not in allowed:
        raise ValueError(
            f"Unsupported position_reassembly_backend={cfg.position_reassembly_backend!r}; "
            f"expected one of {sorted(allowed)}."
        )
    if int(cfg.position_reassembly_batch_size) <= 0:
        raise ValueError(
            f"position_reassembly_batch_size must be > 0 (got {cfg.position_reassembly_batch_size})."
        )
    if cfg.position_crop_border is not None and int(cfg.position_crop_border) < 0:
        raise ValueError(
            f"position_crop_border must be >= 0 when set (got {cfg.position_crop_border})."
        )


def _resolve_position_crop_border(
    patch_h: int,
    patch_w: int,
    configured: Optional[int],
) -> int:
    """Resolve and clamp border used to derive the effective reassembly window (M)."""
    h = int(patch_h)
    w = int(patch_w)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid patch shape for crop resolution: ({h}, {w})")

    if configured is None:
        crop = min(h, w) // 4
    else:
        crop = int(configured)

    if crop < 0:
        raise ValueError(f"position_crop_border must be >= 0 (got {crop}).")

    crop_max = max(0, (min(h, w) - 1) // 2)
    return int(max(0, min(crop, crop_max)))


def _resolve_position_reassembly_m(
    requested_m: int,
    patch_h: int,
    patch_w: int,
    crop_border: int,
) -> int:
    """Resolve effective M from requested window and border-derived usable patch shape."""
    req = int(requested_m)
    h = int(patch_h)
    w = int(patch_w)
    crop = int(crop_border)
    if req <= 0:
        raise ValueError(f"Requested reassembly M must be positive, got {req}.")
    if h <= 0 or w <= 0:
        raise ValueError(f"Patch spatial shape must be positive, got ({h}, {w}).")
    if crop < 0:
        raise ValueError(f"Crop border must be >= 0, got {crop}.")

    usable_h = h - 2 * crop
    usable_w = w - 2 * crop
    if usable_h <= 0 or usable_w <= 0:
        raise ValueError(
            f"Resolved crop border {crop} leaves non-positive usable patch shape "
            f"({usable_h}, {usable_w}) from ({h}, {w})."
        )
    return int(min(req, usable_h, usable_w))


def load_cached_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load cached NPZ dataset from grid-lines workflow.

    Expected keys (from grid_lines_workflow.save_split_npz):
        - diffraction: Input diffraction patterns
        - Y_I: Amplitude ground truth
        - Y_phi: Phase ground truth
        - coords_nominal: Nominal scan positions
        - coords_true: True scan positions
        - YY_full: Full object ground truth
        - probeGuess: Probe function (optional)
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    required_keys = ['diffraction', 'Y_I', 'Y_phi', 'coords_nominal']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in {npz_path}")
    return data


def load_cached_dataset_with_metadata(
    npz_path: Path,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
    """Load cached NPZ dataset and optional metadata."""
    from ptycho.metadata import MetadataManager

    data, metadata = MetadataManager.load_with_metadata(str(npz_path))
    required_keys = ['diffraction', 'Y_I', 'Y_phi', 'coords_nominal']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in {npz_path}")
    return data, metadata


def _reshape_coords(coords: Optional[np.ndarray], n_samples: int, channels: int) -> np.ndarray:
    if coords is None:
        return np.zeros((n_samples, 1, 2, channels), dtype=np.float32)
    coords_np = np.asarray(coords)
    if coords_np.ndim == 2 and coords_np.shape[1] == 2:
        if coords_np.shape[0] == n_samples * channels:
            coords_np = coords_np.reshape(n_samples, channels, 2)
        elif coords_np.shape[0] == n_samples:
            coords_np = np.repeat(coords_np[:, None, :], channels, axis=1)
        else:
            coords_np = np.zeros((n_samples, channels, 2), dtype=np.float32)
        coords_np = coords_np.transpose(0, 2, 1)
        coords_np = coords_np[:, None, :, :]
    elif coords_np.ndim == 3 and coords_np.shape[2] == 2:
        coords_np = coords_np.transpose(0, 2, 1)
        coords_np = coords_np[:, None, :, :]
    elif coords_np.ndim == 4 and coords_np.shape[1] == 1 and coords_np.shape[2] == 2:
        coords_np = coords_np
    else:
        coords_np = np.zeros((n_samples, 1, 2, channels), dtype=np.float32)
    return coords_np.astype(np.float32)


def _select_coords_relative(
    data: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]],
    n_samples: int,
    channels: int,
) -> np.ndarray:
    from ptycho_torch.coords import coords_relative_from_nominal

    coords_rel = data.get("coords_relative")
    if coords_rel is not None:
        return _reshape_coords(coords_rel, n_samples, channels)
    coords_nom = _reshape_coords(data.get("coords_nominal"), n_samples, channels)
    coords_type = (metadata or {}).get("additional_parameters", {}).get("coords_type")
    if coords_type == "relative" or coords_type is None:
        return coords_nom
    if coords_type == "nominal":
        return coords_relative_from_nominal(coords_nom)
    raise ValueError(f"Unknown coords_type='{coords_type}'.")


def _coords_relative_for_inference(
    data: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]],
    n_samples: int,
    channels: int,
) -> np.ndarray:
    coords_rel = _select_coords_relative(data, metadata, n_samples, channels)
    return np.transpose(coords_rel, (0, 3, 1, 2)).astype(np.float32)


def _configure_stitching_params(cfg: TorchRunnerConfig, metadata: Optional[Dict[str, Any]]) -> None:
    if not metadata:
        raise ValueError("Missing metadata; cannot stitch predictions for metrics.")

    additional = metadata.get("additional_parameters", {})
    nimgs_test = additional.get("nimgs_test")
    outer_offset_test = additional.get("outer_offset_test")
    if nimgs_test is None or outer_offset_test is None:
        raise ValueError("Metadata missing nimgs_test/outer_offset_test for stitching.")

    from ptycho import params as p

    p.cfg["N"] = cfg.N
    p.cfg["gridsize"] = cfg.gridsize
    p.set("nimgs_test", nimgs_test)
    p.set("outer_offset_test", outer_offset_test)


def _stitch_for_metrics(
    pred_complex: np.ndarray,
    cfg: TorchRunnerConfig,
    metadata: Optional[Dict[str, Any]],
    norm_Y_I: float,
) -> np.ndarray:
    from ptycho.workflows.grid_lines_workflow import stitch_predictions

    _configure_stitching_params(cfg, metadata)
    return stitch_predictions(pred_complex, float(norm_Y_I), part="complex")


def _normalize_position_inputs(
    pred_complex: np.ndarray,
    test_data: Dict[str, np.ndarray],
    *,
    position_crop_border: Optional[int] = None,
    runtime_contract_out: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Normalize prediction and offset shapes for position reassembly backends."""
    coords_offsets = test_data.get("coords_offsets")
    if coords_offsets is None:
        raise ValueError("Position reassembly requires 'coords_offsets' in test data.")

    coords_np = np.asarray(coords_offsets)
    if coords_np.ndim != 4 or coords_np.shape[1] != 1 or coords_np.shape[2] != 2:
        raise ValueError(
            f"coords_offsets has unsupported shape {coords_np.shape}; expected (B, 1, 2, C1)."
        )

    pred_np = np.asarray(pred_complex)
    # Normalize channel-first prediction layout (B, C, H, W) to channel-last.
    if pred_np.ndim == 4 and pred_np.shape[1] <= 8 and pred_np.shape[-1] > 8:
        pred_np = np.transpose(pred_np, (0, 2, 3, 1))
    if pred_np.ndim == 2:
        patches = pred_np[None, :, :, None]
    elif pred_np.ndim == 3:
        patches = pred_np[:, :, :, None]
    elif pred_np.ndim == 4:
        batch, h, w, channels = pred_np.shape
        patches = np.transpose(pred_np, (0, 3, 1, 2)).reshape(batch * channels, h, w, 1)
    else:
        raise ValueError(f"Unsupported prediction shape for position reassembly: {pred_np.shape}")

    forwarded_shape = tuple(int(v) for v in patches.shape)
    if patches.shape[-1] != 1:
        raise ValueError(
            f"Position reassembly expects singleton channel patches after normalization, got {patches.shape}."
        )
    resolved_crop_border = _resolve_position_crop_border(
        int(patches.shape[1]),
        int(patches.shape[2]),
        position_crop_border,
    )

    offsets_b12c = np.asarray(coords_np).astype(np.float64)  # (B,1,2,C)
    offsets_b112 = np.transpose(offsets_b12c, (0, 1, 3, 2))  # (B,1,C,2) => (B,1,1,2) when C=1
    if offsets_b112.shape[0] != patches.shape[0]:
        if patches.shape[0] % offsets_b112.shape[0] != 0:
            raise ValueError(
                f"Cannot align coords_offsets batch {offsets_b112.shape[0]} with patches {patches.shape[0]}."
            )
        repeats = patches.shape[0] // offsets_b112.shape[0]
        offsets_b112 = np.repeat(offsets_b112, repeats, axis=0)
        offsets_b12c = np.repeat(offsets_b12c, repeats, axis=0)

    if runtime_contract_out is not None:
        runtime_contract_out["position_crop_border_configured"] = (
            None if position_crop_border is None else int(position_crop_border)
        )
        runtime_contract_out["position_crop_border_resolved"] = int(resolved_crop_border)
        # Deprecated pre/post crop keys retained for compatibility; runner no longer pre-crops tensors.
        runtime_contract_out["position_patch_shape_pre_crop"] = list(forwarded_shape)
        runtime_contract_out["position_patch_shape_post_crop"] = list(forwarded_shape)
        runtime_contract_out["position_patch_shape_forwarded"] = list(forwarded_shape)

    return patches.astype(np.complex64), offsets_b12c, offsets_b112, int(resolved_crop_border)


def _reassemble_position_shift_sum(
    patches: np.ndarray,
    offsets_b112: np.ndarray,
    M: int,
) -> np.ndarray:
    from ptycho import tf_helper as hh

    stitched = hh.reassemble_position(patches, offsets_b112, M=M)
    return np.squeeze(np.asarray(stitched)).astype(np.complex64)


def _reassemble_position_batched(
    patches: np.ndarray,
    offsets_b12c: np.ndarray,
    M: int,
    batch_size: int,
) -> np.ndarray:
    from ptycho import tf_helper as hh

    # Keep batched backend numerically aligned with shift_sum by using the same
    # reassemble_position normalization path, but with smaller streaming chunks.
    offsets_b112 = np.transpose(np.asarray(offsets_b12c), (0, 1, 3, 2)).astype(np.float64)
    stitched = hh.reassemble_position(
        patches,
        offsets_b112,
        M=M,
        chunk_size=max(1, int(batch_size)),
    )
    return np.squeeze(np.asarray(stitched)).astype(np.complex64)


def _choose_position_backend(
    pred_complex: np.ndarray,
    test_data: Dict[str, np.ndarray],
    configured: str,
) -> str:
    _ = (pred_complex, test_data)
    if configured != "auto":
        return configured
    # Keep auto on shift_sum for reconstruction parity; only fall back to batched
    # in _reassemble_with_coords_offsets when shift_sum raises OOM.
    return "shift_sum"


def _reassemble_with_coords_offsets(
    pred_complex: np.ndarray,
    test_data: Dict[str, np.ndarray],
    M: int = 20,
    backend: str = "shift_sum",
    batch_size: int = 64,
    position_crop_border: Optional[int] = None,
    allow_oom_fallback: bool = True,
    runtime_contract_out: Dict[str, object] | None = None,
) -> np.ndarray:
    """Reassemble predicted patches using coords_offsets (external dataset mode)."""
    patches, offsets_b12c, offsets_b112, resolved_crop_border = _normalize_position_inputs(
        pred_complex,
        test_data,
        position_crop_border=position_crop_border,
        runtime_contract_out=runtime_contract_out,
    )
    effective_m = _resolve_position_reassembly_m(
        requested_m=int(M),
        patch_h=int(patches.shape[1]),
        patch_w=int(patches.shape[2]),
        crop_border=int(resolved_crop_border),
    )
    if effective_m <= 0:
        raise ValueError(
            f"Effective reassembly M must be positive after crop resolution, got {effective_m}"
        )
    selected_backend = _choose_position_backend(pred_complex, test_data, configured=backend)
    if runtime_contract_out is not None:
        runtime_contract_out.update(
            {
                "requested_reassembly_backend": str(backend),
                "resolved_reassembly_backend": str(selected_backend),
                "allow_oom_fallback": bool(allow_oom_fallback),
                "fallback_used": False,
                "position_reassembly_M_requested": int(M),
                "position_reassembly_M_effective": int(effective_m),
            }
        )
    if selected_backend == "shift_sum":
        try:
            return _reassemble_position_shift_sum(patches, offsets_b112, M=effective_m)
        except Exception as exc:
            import tensorflow as tf

            if (
                allow_oom_fallback
                and backend in {"auto", "shift_sum"}
                and isinstance(exc, tf.errors.ResourceExhaustedError)
            ):
                if runtime_contract_out is not None:
                    runtime_contract_out["resolved_reassembly_backend"] = "batched"
                    runtime_contract_out["fallback_used"] = True
                logger.warning(
                    "Shift-sum OOM under backend=%s; retrying with batched position reassembly",
                    backend,
                )
                return _reassemble_position_batched(
                    patches,
                    offsets_b12c,
                    M=effective_m,
                    batch_size=batch_size,
                )
            raise
    if selected_backend == "batched":
        return _reassemble_position_batched(
            patches,
            offsets_b12c,
            M=effective_m,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported position reassembly backend: {selected_backend!r}")


def _harmonize_prediction_shape(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Align prediction spatial shape to ground truth shape for metric evaluation."""
    pred = np.squeeze(np.asarray(prediction))
    gt = np.squeeze(np.asarray(ground_truth))

    if pred.ndim == 2 and gt.ndim == 2 and pred.shape != gt.shape:
        from ptycho.image.harmonize import resize_complex_to_shape

        pred = resize_complex_to_shape(pred, (int(gt.shape[0]), int(gt.shape[1])))
    return pred


def _normalize_eval_image_array(arr: np.ndarray) -> np.ndarray:
    """Canonicalize eval arrays to complex 2D when they encode channels."""
    image = np.squeeze(np.asarray(arr))
    if image.ndim != 3:
        return image

    # Handle channel-first encodings shaped (C, H, W).
    if image.shape[0] in {1, 2} and image.shape[1] > 8 and image.shape[2] > 8:
        if image.shape[0] == 1:
            return image[0]
        if np.isrealobj(image):
            return image[0] + 1j * image[1]
        return image[0]

    # Handle channel-last encodings shaped (H, W, C).
    if image.shape[-1] in {1, 2} and image.shape[0] > 8 and image.shape[1] > 8:
        if image.shape[-1] == 1:
            return image[..., 0]
        if np.isrealobj(image):
            return image[..., 0] + 1j * image[..., 1]
        return image[..., 0]

    # Fallback for singleton channel dimensions.
    if image.shape[0] == 1:
        return image[0]
    if image.shape[-1] == 1:
        return image[..., 0]
    return image


def setup_torch_configs(cfg: TorchRunnerConfig):
    """Set up PyTorch configuration objects from runner config.

    Returns:
        Tuple of (TrainingConfig, PyTorchExecutionConfig)
    """
    from typing import cast, Literal
    from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig

    # Cast N and architecture to their Literal types
    N_literal = cast(Literal[64, 128, 256], cfg.N)
    arch_literal = cast(
        Literal['cnn', 'ffno', 'fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'neuralop_uno', 'hybrid_resnet', 'spectral_resnet_bottleneck_net', 'spectral_resnet_bottleneck_linear_decoder', 'hybrid_resnet_ffno_bottleneck'],
        cfg.architecture,
    )
    if cfg.architecture in {
        "hybrid_resnet",
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
    }:
        if cfg.fno_blocks < 3:
            raise ValueError(
                f"{cfg.architecture} requires --fno-blocks >= 3 to downsample to N/4 "
                f"(got {cfg.fno_blocks})."
            )
        if cfg.hybrid_downsample_steps not in {1, 2}:
            raise ValueError(
                "hybrid_downsample_steps must be in [1, 2] "
                f"(got {cfg.hybrid_downsample_steps})."
            )
        valid_downsample_ops = {"stride_conv", "avgpool_conv", "blurpool_conv"}
        if cfg.hybrid_downsample_op not in valid_downsample_ops:
            raise ValueError(
                f"hybrid_downsample_op must be one of {sorted(valid_downsample_ops)} "
                f"(got {cfg.hybrid_downsample_op!r})."
            )
        if (
            not math.isfinite(float(cfg.hybrid_encoder_conv_hidden_scale))
            or float(cfg.hybrid_encoder_conv_hidden_scale) <= 0.0
        ):
            raise ValueError(
                "hybrid_encoder_conv_hidden_scale must be finite and > 0 "
                f"(got {cfg.hybrid_encoder_conv_hidden_scale})."
            )
        if (
            not math.isfinite(float(cfg.hybrid_encoder_spectral_hidden_scale))
            or float(cfg.hybrid_encoder_spectral_hidden_scale) <= 0.0
        ):
            raise ValueError(
                "hybrid_encoder_spectral_hidden_scale must be finite and > 0 "
                f"(got {cfg.hybrid_encoder_spectral_hidden_scale})."
            )
        if (
            cfg.hybrid_encoder_conv_hidden_channels is not None
            and cfg.hybrid_encoder_conv_hidden_channels <= 0
        ):
            raise ValueError(
                "hybrid_encoder_conv_hidden_channels must be positive when set "
                f"(got {cfg.hybrid_encoder_conv_hidden_channels})."
            )
        if (
            cfg.hybrid_encoder_spectral_hidden_channels is not None
            and cfg.hybrid_encoder_spectral_hidden_channels <= 0
        ):
            raise ValueError(
                "hybrid_encoder_spectral_hidden_channels must be positive when set "
                f"(got {cfg.hybrid_encoder_spectral_hidden_channels})."
            )
        if cfg.hybrid_resnet_blocks <= 0:
            raise ValueError(
                f"hybrid_resnet_blocks must be positive (got {cfg.hybrid_resnet_blocks})."
            )
        valid_skip_styles = {"add", "concat", "gated_add"}
        if cfg.hybrid_skip_style not in valid_skip_styles:
            raise ValueError(
                f"hybrid_skip_style must be one of {sorted(valid_skip_styles)} "
                f"(got {cfg.hybrid_skip_style!r})."
            )
        if cfg.resnet_width is not None:
            if cfg.resnet_width <= 0:
                raise ValueError(
                    f"--torch-resnet-width must be positive when set (got {cfg.resnet_width})."
                )
            if cfg.resnet_width % 4 != 0:
                raise ValueError(
                    "--torch-resnet-width must be divisible by 4 so the CycleGAN "
                    f"upsamplers produce integer channel sizes (got {cfg.resnet_width})."
                )
    if cfg.architecture in {
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
    }:
        if cfg.spectral_bottleneck_blocks <= 0:
            raise ValueError(
                f"spectral_bottleneck_blocks must be positive (got {cfg.spectral_bottleneck_blocks})."
            )
        if cfg.spectral_bottleneck_modes <= 0:
            raise ValueError(
                f"spectral_bottleneck_modes must be positive (got {cfg.spectral_bottleneck_modes})."
            )
        if not math.isfinite(float(cfg.spectral_bottleneck_gate_init)):
            raise ValueError(
                "spectral_bottleneck_gate_init must be finite "
                f"(got {cfg.spectral_bottleneck_gate_init})."
            )
        if cfg.spectral_bottleneck_gate_mode not in {"shared", "per_block"}:
            raise ValueError(
                "spectral_bottleneck_gate_mode must be one of ['per_block', 'shared'] "
                f"(got {cfg.spectral_bottleneck_gate_mode!r})."
            )
    if cfg.architecture == "neuralop_uno":
        if cfg.N != 128:
            raise ValueError(
                "neuralop_uno only supports the locked Lines128 CDI contract "
                f"(N=128); got N={cfg.N}."
            )
        if cfg.gridsize != 1:
            raise ValueError(
                "neuralop_uno only supports the locked gridsize=1 CDI contract; "
                f"got gridsize={cfg.gridsize}."
            )
        if cfg.generator_output_mode != "real_imag":
            raise ValueError(
                "neuralop_uno only supports generator_output_mode='real_imag'."
            )

    training_config = TrainingConfig(
        model=ModelConfig(
            N=N_literal,
            gridsize=cfg.gridsize,
            model_type=cast(Literal['pinn', 'supervised'], cfg.training_procedure),
            architecture=arch_literal,
            fno_modes=cfg.fno_modes,
            fno_width=cfg.fno_width,
            fno_blocks=cfg.fno_blocks,
            fno_cnn_blocks=cfg.fno_cnn_blocks,
            fno_input_transform=cfg.fno_input_transform,
            max_hidden_channels=cfg.max_hidden_channels,
            resnet_width=cfg.resnet_width,
            generator_output_mode=cfg.generator_output_mode,
            object_big=False,
            probe_big=False,
            probe_mask=cfg.probe_mask,
            probe_mask_sigma=cfg.probe_mask_sigma,
            probe_mask_diameter=cfg.probe_mask_diameter,
        ),
        train_data_file=cfg.train_npz,
        test_data_file=cfg.test_npz,
        output_dir=cfg.output_dir,
        nepochs=cfg.epochs,
        batch_size=cfg.batch_size,
        backend='pytorch',
        torch_loss_mode=cfg.torch_loss_mode,
        torch_mae_pred_l2_match_target=cfg.torch_mae_pred_l2_match_target,
    )
    training_config.log_grad_norm = cfg.log_grad_norm
    training_config.grad_norm_log_freq = cfg.grad_norm_log_freq
    training_config.subsample_seed = cfg.seed
    training_config.gradient_clip_val = cfg.gradient_clip_val
    training_config.gradient_clip_algorithm = cfg.gradient_clip_algorithm
    training_config.optimizer = cfg.optimizer
    training_config.weight_decay = cfg.weight_decay
    training_config.momentum = cfg.momentum
    training_config.adam_beta1 = cfg.adam_beta1
    training_config.adam_beta2 = cfg.adam_beta2
    training_config.learning_rate = cfg.learning_rate
    training_config.scheduler = cfg.scheduler
    training_config.lr_warmup_epochs = cfg.lr_warmup_epochs
    training_config.lr_min_ratio = cfg.lr_min_ratio
    training_config.plateau_factor = cfg.plateau_factor
    training_config.plateau_patience = cfg.plateau_patience
    training_config.plateau_min_lr = cfg.plateau_min_lr
    training_config.plateau_threshold = cfg.plateau_threshold

    execution_config = PyTorchExecutionConfig(
        learning_rate=cfg.learning_rate,
        deterministic=True,
        gradient_clip_val=cfg.gradient_clip_val,
        enable_progress_bar=True,
        enable_checkpointing=cfg.enable_checkpointing,
        logger_backend=cfg.logger_backend,
        hybrid_skip_connections=cfg.hybrid_skip_connections,
        hybrid_downsample_steps=cfg.hybrid_downsample_steps,
        hybrid_downsample_op=cfg.hybrid_downsample_op,
        hybrid_encoder_conv_hidden_scale=cfg.hybrid_encoder_conv_hidden_scale,
        hybrid_encoder_spectral_hidden_scale=cfg.hybrid_encoder_spectral_hidden_scale,
        hybrid_encoder_conv_hidden_channels=cfg.hybrid_encoder_conv_hidden_channels,
        hybrid_encoder_spectral_hidden_channels=cfg.hybrid_encoder_spectral_hidden_channels,
        hybrid_resnet_blocks=cfg.hybrid_resnet_blocks,
        hybrid_skip_style=cfg.hybrid_skip_style,
        spectral_bottleneck_blocks=cfg.spectral_bottleneck_blocks,
        spectral_bottleneck_modes=cfg.spectral_bottleneck_modes,
        spectral_bottleneck_share_weights=cfg.spectral_bottleneck_share_weights,
        spectral_bottleneck_gate_init=cfg.spectral_bottleneck_gate_init,
        spectral_bottleneck_gate_mode=cfg.spectral_bottleneck_gate_mode,
        recon_log_every_n_epochs=cfg.recon_log_every_n_epochs,
        recon_log_num_patches=cfg.recon_log_num_patches,
        recon_log_fixed_indices=cfg.recon_log_fixed_indices,
        recon_log_stitch=cfg.recon_log_stitch,
        recon_log_max_stitch_samples=cfg.recon_log_max_stitch_samples,
    )

    return training_config, execution_config


def run_torch_training(
    cfg: TorchRunnerConfig,
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    train_metadata: Optional[Dict[str, Any]] = None,
    test_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run PyTorch training using the Lightning workflow.

    Args:
        cfg: Runner configuration
        train_data: Loaded training dataset
        test_data: Loaded test dataset (unused in scaffold, for future validation)

    Returns:
        Training results dict with model and history

    Note:
        Uses torchapi-devel Lightning workflow via _train_with_lightning.
    """
    import torch
    from ptycho_torch.workflows.components import _train_with_lightning

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set up configs
    training_config, execution_config = setup_torch_configs(cfg)

    X = np.asarray(train_data["diffraction"])
    if X.ndim == 3:
        X = X[..., np.newaxis]
    n_samples = X.shape[0]
    channels = X.shape[-1]
    coords = _select_coords_relative(train_data, train_metadata, n_samples, channels)
    probe = train_data.get("probeGuess")
    if probe is None:
        probe = np.ones((cfg.N, cfg.N), dtype=np.complex64)

    train_container = {
        "X": X,
        "coords_nominal": _reshape_coords(train_data.get("coords_nominal"), n_samples, channels),
        "coords_relative": coords,
        "probe": probe,
    }
    if cfg.training_procedure == "supervised":
        label_amp = train_data.get("Y_I")
        label_phase = train_data.get("Y_phi")
        if label_amp is None or label_phase is None:
            raise RuntimeError(
                "Supervised grid-lines Torch runs require Y_I and Y_phi so they can be bridged "
                "to label_amp/label_phase."
            )
        train_container["label_amp"] = np.asarray(label_amp, dtype=np.float32)
        train_container["label_phase"] = np.asarray(label_phase, dtype=np.float32)

    test_container = None
    if test_data:
        X_te = np.asarray(test_data["diffraction"])
        if X_te.ndim == 3:
            X_te = X_te[..., np.newaxis]
        n_te = X_te.shape[0]
        channels_te = X_te.shape[-1]
        coords_te = _select_coords_relative(test_data, test_metadata, n_te, channels_te)
        test_probe = test_data.get("probeGuess", probe)
        test_container = {
            "X": X_te,
            "coords_nominal": _reshape_coords(test_data.get("coords_nominal"), n_te, channels_te),
            "coords_relative": coords_te,
            "probe": test_probe,
        }
        if cfg.training_procedure == "supervised":
            label_amp = test_data.get("Y_I")
            label_phase = test_data.get("Y_phi")
            if label_amp is None or label_phase is None:
                raise RuntimeError(
                    "Supervised grid-lines Torch runs require Y_I and Y_phi so they can be bridged "
                    "to label_amp/label_phase."
                )
            test_container["label_amp"] = np.asarray(label_amp, dtype=np.float32)
            test_container["label_phase"] = np.asarray(label_phase, dtype=np.float32)

    results = _train_with_lightning(
        train_container,
        test_container,
        training_config,
        execution_config=execution_config,
    )
    results["generator"] = cfg.architecture
    return results


def run_torch_inference(
    model: Any,
    test_data: Dict[str, np.ndarray],
    cfg: TorchRunnerConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Run inference using trained PyTorch model.

    Args:
        model: Trained PyTorch model
        test_data: Test dataset
        cfg: Runner configuration

    Returns:
        Reconstructed complex object predictions

    Note:
        Use Lightning forward_predict signature: (x, positions, probe, input_scale_factor).
        Inference is batched to avoid GPU OOM on dense datasets.
    """
    import torch

    if model is None:
        raise ValueError("Model is required for inference")

    def _resolve_inference_device() -> torch.device:
        # Prefer current non-CPU model placement when present. If the model ended
        # up on CPU after trainer.fit(), explicitly re-select the best available
        # accelerator for inference.
        if hasattr(model, "parameters"):
            try:
                first_param = next(model.parameters())
            except StopIteration:
                first_param = None
            if first_param is not None and first_param.device.type != "cpu":
                return first_param.device

        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    X_np = np.asarray(test_data['diffraction'])
    if X_np.ndim == 3:
        X_np = X_np[..., np.newaxis]
    if X_np.ndim == 4 and X_np.shape[1] <= 8 and X_np.shape[-1] > 8:
        X_np = np.transpose(X_np, (0, 2, 3, 1))

    n_samples = X_np.shape[0]
    channels = X_np.shape[-1]
    coords_np = _coords_relative_for_inference(test_data, metadata, n_samples, channels)
    probe_np = test_data.get('probeGuess')
    if probe_np is None:
        probe_np = np.ones((cfg.N, cfg.N), dtype=np.complex64)

    X_test = torch.from_numpy(X_np).float().permute(0, 3, 1, 2)
    coords_test = torch.from_numpy(coords_np).float()
    probe_test = torch.from_numpy(probe_np).to(torch.complex64)

    device = _resolve_inference_device()
    if hasattr(model, "to") and callable(model.to):
        model = model.to(device)
    model.eval()
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    preds = []
    batch_size = max(1, cfg.infer_batch_size)
    probe_test = probe_test.to(device)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = X_test[start:end].to(device)
            coords_batch = coords_test[start:end].to(device)
            scale_batch = torch.ones((end - start, 1, 1, 1), device=device, dtype=torch.float32)
            probe_batch = probe_test
            batch_pred = target_model.forward_predict(x_batch, coords_batch, probe_batch, scale_batch)
            preds.append(batch_pred.detach().cpu())

    predictions = torch.cat(preds, dim=0)
    return predictions.numpy()


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    label: str,
) -> Dict[str, float]:
    """Compute reconstruction metrics compatible with TF workflow.

    Args:
        predictions: Model predictions (complex)
        ground_truth: Ground truth (complex)
        label: Label for metrics (e.g., 'pinn_fno')

    Returns:
        Metrics dict with MSE, SSIM, etc.
    """
    from ptycho.evaluation import eval_reconstruction

    pred = _normalize_eval_image_array(np.asarray(predictions))
    gt = _normalize_eval_image_array(np.asarray(ground_truth))

    if pred.ndim == 2 and gt.ndim == 2 and pred.shape != gt.shape:
        from ptycho.image.harmonize import resize_complex_to_shape

        pred = resize_complex_to_shape(pred, (int(gt.shape[0]), int(gt.shape[1])))

    if pred.ndim == 2:
        pred = pred[..., None]
    if gt.ndim == 2:
        gt = gt[..., None]
    return eval_reconstruction(
        pred,
        gt,
        label=label,
    )


def _collect_visual_order(output_dir: Path, architecture: str) -> Tuple[str, ...]:
    recon_dir = output_dir / "recons"
    if not recon_dir.exists():
        return ()
    existing = {path.name for path in recon_dir.iterdir() if path.is_dir()}
    arch_label = _runner_model_id(architecture, "pinn")
    preferred = [
        "gt",
        "baseline",
        "pinn",
        "supervised_ffno",
        "pinn_ffno",
        "pinn_fno",
        "pinn_hybrid",
        "pinn_stable_hybrid",
        "pinn_fno_vanilla",
        "pinn_hybrid_resnet",
    ]
    if arch_label not in preferred:
        preferred.append(arch_label)
    order = [label for label in preferred if label in existing]
    extras = sorted(existing - set(order))
    return tuple(order + extras)


def _save_gt_recon_if_missing(output_dir: Path, ground_truth: np.ndarray) -> None:
    if ground_truth is None:
        return
    gt_path = output_dir / "recons" / "gt" / "recon.npz"
    if gt_path.exists():
        return
    from ptycho.workflows.grid_lines_workflow import save_recon_artifact
    if not np.iscomplexobj(ground_truth):
        ground_truth = np.asarray(ground_truth).astype(np.complex64)
    save_recon_artifact(output_dir, "gt", ground_truth)


def save_run_artifacts(
    cfg: TorchRunnerConfig,
    results: Dict[str, Any],
    metrics: Dict[str, float],
    randomness_contract: Dict[str, int | None],
    *,
    recon_path: Optional[Path] = None,
) -> Path:
    """Save run artifacts to output directory.

    Creates:
        - output_dir/runs/pinn_<arch>/model.pt
        - output_dir/runs/pinn_<arch>/metrics.json
        - output_dir/runs/pinn_<arch>/history.json
    """
    model_id = _runner_model_id(cfg.architecture, cfg.training_procedure)
    model_id = _runner_model_id(cfg.architecture, cfg.training_procedure, cfg.model_id_override)
    run_dir = cfg.output_dir / "runs" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # Save training history
    history_path = run_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(results.get('history', {}), f, indent=2, default=_json_default)

    randomness_path = run_dir / "randomness_contract.json"
    with open(randomness_path, 'w') as f:
        json.dump(randomness_contract, f, indent=2)

    config_payload = {
        "torch_runner_config": asdict(cfg),
        "model_label": _paper_model_label(
            cfg.architecture,
            cfg.training_procedure,
            cfg.model_label_override,
        ),
        "training_procedure": cfg.training_procedure,
        "train_npz": str(cfg.train_npz),
        "test_npz": str(cfg.test_npz),
        "recon_npz": str(recon_path) if recon_path is not None else None,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2, default=_json_default)

    # Save model checkpoint
    model_to_save = results.get('model')
    if model_to_save is None and isinstance(results.get('models'), dict):
        model_to_save = results['models'].get('diffraction_to_obj')
    if model_to_save is not None:
        import torch
        model_path = run_dir / "model.pt"
        torch.save(model_to_save.state_dict(), model_path)

    logger.info(f"Saved artifacts to {run_dir}")
    return run_dir


def _build_randomness_contract(cfg: TorchRunnerConfig) -> Dict[str, int | None]:
    """Describe the effective seed policy used by the Torch runner."""
    return {
        "requested_seed": int(cfg.seed),
        "effective_subsample_seed": int(cfg.seed),
        "effective_lightning_seed": int(cfg.seed),
    }


def _history_series(history: object, *keys: str) -> List[float]:
    if not isinstance(history, dict):
        return []
    for key in keys:
        values = history.get(key)
        if isinstance(values, list):
            try:
                return [float(value) for value in values]
            except Exception:
                return []
    return []


def _history_validation_loss(history: object) -> Dict[str, object]:
    val_loss = _history_series(history, "val_loss")
    if val_loss:
        return {"status": "emitted", "value": float(val_loss[-1])}
    return {"status": "no_validation_series", "value": None}


def _torch_hardware_summary() -> Dict[str, object]:
    try:
        import torch
    except Exception:
        return {"backend": "pytorch", "accelerator": "unknown"}

    accelerator = "cpu"
    if torch.cuda.is_available():
        try:
            accelerator = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            accelerator = "cuda"
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            accelerator = "mps"
    return {
        "backend": "pytorch",
        "accelerator": accelerator,
    }


def _build_paper_row_payload(
    cfg: TorchRunnerConfig,
    *,
    metrics: Dict[str, Any],
    history: object,
    model_params: int,
    train_wall_time_sec: float,
    inference_time_s: float,
    run_dir: Optional[Path] = None,
    recon_path: Optional[Path] = None,
    invocation_json: Optional[Path] = None,
) -> Dict[str, object]:
    train_loss_series = _history_series(history, "train_loss", "loss")
    final_completed_epoch = int(len(train_loss_series) or cfg.epochs)
    final_train_loss = float(train_loss_series[-1]) if train_loss_series else None
    payload = {
        "model_label": _paper_model_label(
            cfg.architecture,
            cfg.training_procedure,
            cfg.model_label_override,
        ),
        "architecture_id": str(cfg.architecture),
        "training_procedure": cfg.training_procedure,
        "N": int(cfg.N),
        "parameter_count": int(model_params),
        "epoch_budget": int(cfg.epochs),
        "final_completed_epoch": final_completed_epoch,
        "final_train_loss": final_train_loss,
        "validation_loss": _history_validation_loss(history),
        "runtime_summary": {
            "train_wall_time_sec": float(train_wall_time_sec),
            "inference_time_sec": float(inference_time_s),
        },
        "hardware_summary": _torch_hardware_summary(),
        "row_status": "paper_grade",
        "caveats": [],
        "metrics": dict(metrics),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        payload["config"] = {"json": str((run_dir / "config.json").relative_to(cfg.output_dir))}
        payload["outputs"] = {
            "metrics_json": str((run_dir / "metrics.json").relative_to(cfg.output_dir)),
            "history_json": str((run_dir / "history.json").relative_to(cfg.output_dir)),
            "recon_npz": str(Path(recon_path).relative_to(cfg.output_dir)) if recon_path is not None else "",
            "model_artifact": str((run_dir / "model.pt").relative_to(cfg.output_dir)),
        }
    if invocation_json is not None:
        invocation_json = Path(invocation_json)
        payload["invocation"] = {
            "json": str(invocation_json.relative_to(cfg.output_dir)),
            "shell": str(invocation_json.with_suffix(".sh").relative_to(cfg.output_dir)),
        }
        invocation_payload = json.loads(invocation_json.read_text(encoding="utf-8"))
        extra = invocation_payload.get("extra", {})
        runtime = extra.get("runtime_provenance", {})
        git_commit = extra.get("git_commit")
        if runtime:
            payload["environment"] = dict(runtime)
        if isinstance(git_commit, str) and git_commit:
            payload["git"] = {"commit": git_commit}
    payload.setdefault(
        "dataset",
        {
            "train_npz": str(cfg.train_npz),
            "test_npz": str(cfg.test_npz),
            "probe_source": str(cfg.probe_source),
        },
    )
    payload.setdefault(
        "splits",
        {
            "gridsize": int(cfg.gridsize),
        },
    )
    payload.setdefault("randomness", _build_randomness_contract(cfg))
    payload.setdefault(
        "visuals",
        {
            "amp_phase_png": (
                f"visuals/amp_phase_"
                f"{_runner_model_id(cfg.architecture, cfg.training_procedure, cfg.model_id_override)}.png"
            ),
            "amp_phase_error_png": (
                f"visuals/amp_phase_error_"
                f"{_runner_model_id(cfg.architecture, cfg.training_procedure, cfg.model_id_override)}.png"
            ),
        },
    )
    return payload


def run_grid_lines_torch(
    cfg: TorchRunnerConfig,
    *,
    invocation_argv: Optional[List[str]] = None,
    invocation_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main entry point for Torch grid-lines runner.

    Orchestrates: load data → train → infer → compute metrics → save artifacts

    Args:
        cfg: Runner configuration

    Returns:
        Dict with metrics, artifact paths, and run metadata
    """
    model_id = _runner_model_id(
        cfg.architecture,
        cfg.training_procedure,
        cfg.model_id_override,
    )
    logger.info(
        "Starting Torch grid-lines runner: arch=%s training_procedure=%s",
        cfg.architecture,
        cfg.training_procedure,
    )
    _validate_position_reassembly_config(cfg)
    from scripts.studies.invocation_logging import update_invocation_artifacts

    invocation_json = _write_runner_invocation_artifacts(
        cfg,
        argv=invocation_argv,
        extra=invocation_extra,
    )

    try:
        # Step 1: Load cached datasets
        logger.info(f"Loading train data from {cfg.train_npz}")
        train_data, train_metadata = load_cached_dataset_with_metadata(cfg.train_npz)

        logger.info(f"Loading test data from {cfg.test_npz}")
        test_data, test_metadata = load_cached_dataset_with_metadata(cfg.test_npz)
        if cfg.probe_source:
            meta_probe_source = None
            if test_metadata:
                meta_probe_source = test_metadata.get("additional_parameters", {}).get("probe_source")
            if meta_probe_source and meta_probe_source != cfg.probe_source:
                logger.warning(
                    "Probe source mismatch: CLI=%s metadata=%s",
                    cfg.probe_source,
                    meta_probe_source,
                )

        # Step 2: Train model
        logger.info(f"Training {cfg.architecture} model...")
        train_start = time.perf_counter()
        results = run_torch_training(cfg, train_data, test_data, train_metadata=train_metadata, test_metadata=test_metadata)
        train_wall_time_sec = time.perf_counter() - train_start

        # Step 3: Run inference
        logger.info("Running inference...")
        model = results.get('model')
        if model is None and isinstance(results.get('models'), dict):
            model = results['models'].get('diffraction_to_obj')

        model_params = 0
        if model is not None and hasattr(model, "parameters"):
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        if cuda_available:
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictions = run_torch_inference(model, test_data, cfg, metadata=test_metadata)
        if cuda_available:
            torch.cuda.synchronize()
        inference_time_s = time.perf_counter() - start

        # Step 3b: Convert real/imag predictions to complex if needed
        # FNO/Hybrid models output (B, H, W, C, 2) format; convert to complex
        predictions_complex = None
        if predictions.ndim >= 2 and predictions.shape[-1] == 2:
            predictions_complex = to_complex_patches(predictions)
            logger.info(f"Converted predictions to complex: {predictions_complex.shape}")

        # Step 4: Compute metrics
        logger.info("Computing metrics...")
        ground_truth = test_data.get("YY_ground_truth")
        if ground_truth is None:
            ground_truth = test_data.get("YY_full")
        if ground_truth is None:
            ground_truth = test_data.get("objectGuess")
        if ground_truth is None:
            raise ValueError("Test data must provide one of: YY_ground_truth, YY_full, objectGuess.")
        # Use complex predictions for metrics if available
        pred_for_metrics = predictions_complex if predictions_complex is not None else predictions
        position_reassembly_runtime_contract: Dict[str, object] = {}
        if cfg.reassembly_mode == "position":
            pred_for_metrics = _reassemble_with_coords_offsets(
                pred_for_metrics,
                test_data,
                M=cfg.N,
                backend=cfg.position_reassembly_backend,
                batch_size=cfg.position_reassembly_batch_size,
                position_crop_border=cfg.position_crop_border,
                runtime_contract_out=position_reassembly_runtime_contract,
            )
        elif pred_for_metrics.ndim >= 3:
            pred_h, pred_w = pred_for_metrics.shape[-3], pred_for_metrics.shape[-2]
            gt_h, gt_w = ground_truth.shape[-2], ground_truth.shape[-1]
            if (pred_h, pred_w) != (gt_h, gt_w):
                norm_Y_I = test_data.get("norm_Y_I", 1.0)
                pred_for_metrics = _stitch_for_metrics(
                    pred_for_metrics,
                    cfg,
                    test_metadata,
                    norm_Y_I,
                )
        pred_for_metrics = _harmonize_prediction_shape(pred_for_metrics, ground_truth)
        from ptycho.workflows.grid_lines_workflow import save_recon_artifact
        recon_target = pred_for_metrics
        if not np.iscomplexobj(recon_target):
            recon_target = recon_target.astype(np.complex64)
        recon_path = save_recon_artifact(cfg.output_dir, model_id, recon_target)
        metrics = compute_metrics(
            pred_for_metrics,
            ground_truth,
            model_id,
        )

        # Step 5: Save artifacts
        randomness_contract = _build_randomness_contract(cfg)
        run_dir = save_run_artifacts(
            cfg,
            results,
            metrics,
            randomness_contract,
            recon_path=Path(recon_path),
        )

        logger.info(f"Torch runner complete. Artifacts in {run_dir}")

        result_dict = {
            'architecture': cfg.architecture,
            'model_id': model_id,
            'run_dir': str(run_dir),
            'metrics': metrics,
            'history': results.get('history', {}),
            'recon_path': str(recon_path),
            'recon_npz': str(recon_path),
            'model_params': int(model_params),
            'inference_time_s': float(inference_time_s),
            'position_reassembly_runtime_contract': position_reassembly_runtime_contract,
            'randomness_contract': randomness_contract,
        }
        result_dict['paper_row_payload'] = _build_paper_row_payload(
            cfg,
            metrics=metrics,
            history=results.get('history', {}),
            model_params=int(model_params),
            train_wall_time_sec=float(train_wall_time_sec),
            inference_time_s=float(inference_time_s),
            run_dir=Path(run_dir),
            recon_path=Path(recon_path),
            invocation_json=invocation_json,
        )

        # Step 6: Render post-run visuals (best-effort)
        try:
            from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals
            _save_gt_recon_if_missing(cfg.output_dir, ground_truth)
            order = _collect_visual_order(cfg.output_dir, cfg.architecture)
            if order:
                visuals = render_grid_lines_visuals(cfg.output_dir, order=order)
                result_dict["visuals"] = visuals
        except Exception as e:
            logger.warning("Failed to render visuals: %s", e)

        # Include complex predictions if conversion was done
        if predictions_complex is not None:
            result_dict['predictions_complex'] = predictions_complex

        update_invocation_artifacts(
            invocation_json,
            status="completed",
            exit_code=0,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            run_dir=str(run_dir),
        )
        return result_dict
    except Exception as exc:
        update_invocation_artifacts(
            invocation_json,
            status="failed",
            exit_code=1,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Torch runner for grid-lines FNO/hybrid architectures"
    )
    parser.add_argument("--train-npz", type=Path, required=True,
                        help="Path to cached training NPZ")
    parser.add_argument("--test-npz", type=Path, required=True,
                        help="Path to cached test NPZ")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for artifacts")
    parser.add_argument("--architecture", type=str, required=True,
                        choices=['ffno', 'fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'neuralop_uno', 'hybrid_resnet', 'spectral_resnet_bottleneck_net', 'spectral_resnet_bottleneck_linear_decoder', 'hybrid_resnet_ffno_bottleneck'],
                        help="Generator architecture to use")
    parser.add_argument(
        "--training-procedure",
        type=str,
        default="pinn",
        choices=["pinn", "supervised"],
        help="Training procedure contract for this Torch row.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (random if omitted)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--infer-batch-size", type=int, default=128,
                        help="Inference batch size (OOM guard)")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient clipping max norm (<=0 disables clipping)")
    parser.add_argument("--gradient-clip-algorithm", choices=['norm', 'value', 'agc'],
                        default='norm', help='Gradient clipping algorithm')
    parser.add_argument("--output-mode", type=str, default="real_imag",
                        choices=["real_imag", "amp_phase_logits", "amp_phase"],
                        help="Generator output mode for Torch models")
    parser.add_argument("--log-grad-norm", action="store_true",
                        help="Log gradient norms during training")
    parser.add_argument("--grad-norm-log-freq", type=int, default=1,
                        help="Log grad norms every N steps")
    parser.add_argument("--torch-loss-mode", type=str, default="mae",
                        choices=["poisson", "mae"],
                        help="Training loss mode ('poisson' or 'mae')")
    parser.add_argument(
        "--torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_true",
        default=False,
        help="Enable MAE mode where each prediction sample is scaled to match target L2 energy.",
    )
    parser.add_argument(
        "--no-torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_false",
        help="Disable prediction-to-target L2 matching in MAE mode.",
    )
    parser.add_argument(
        "--probe-mask",
        dest="probe_mask",
        action="store_true",
        default=False,
        help="Enable probe mask during forward-model illumination.",
    )
    parser.add_argument(
        "--no-probe-mask",
        dest="probe_mask",
        action="store_false",
        help="Disable probe mask during forward-model illumination.",
    )
    parser.add_argument(
        "--probe-mask-sigma",
        type=float,
        default=1.0,
        help="Gaussian edge sigma for probe mask smoothing (0.0 gives hard edge).",
    )
    parser.add_argument(
        "--probe-mask-diameter",
        type=float,
        default=None,
        help="Optional probe-mask diameter in normalized units.",
    )
    parser.add_argument(
        "--hybrid-skip-connections",
        dest="hybrid_skip_connections",
        action="store_true",
        default=False,
        help="Enable hybrid_resnet encoder-decoder skip fusion.",
    )
    parser.add_argument(
        "--no-hybrid-skip-connections",
        dest="hybrid_skip_connections",
        action="store_false",
        help="Disable hybrid_resnet encoder-decoder skip fusion.",
    )
    parser.add_argument(
        "--hybrid-downsample-steps",
        type=int,
        default=2,
        help="Hybrid ResNet downsample schedule depth (1 or 2).",
    )
    parser.add_argument(
        "--hybrid-downsample-op",
        type=str,
        default="stride_conv",
        choices=["stride_conv", "avgpool_conv", "blurpool_conv"],
        help="Hybrid ResNet downsample operator family.",
    )
    parser.add_argument(
        "--hybrid-encoder-conv-hidden-scale",
        type=float,
        default=2.0,
        help=(
            "Scale factor for hybrid_resnet encoder local-conv branch width; "
            "resolved per block as round(stage_channels * scale)."
        ),
    )
    parser.add_argument(
        "--hybrid-encoder-spectral-hidden-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for hybrid_resnet encoder spectral branch width; "
            "resolved per block as round(stage_channels * scale)."
        ),
    )
    parser.add_argument(
        "--hybrid-encoder-conv-hidden",
        type=int,
        default=None,
        help="Legacy absolute width for hybrid_resnet encoder local-conv branch.",
    )
    parser.add_argument(
        "--hybrid-encoder-spectral-hidden",
        type=int,
        default=None,
        help="Legacy absolute width for hybrid_resnet encoder spectral branch.",
    )
    parser.add_argument(
        "--hybrid-encoder-conv-hidden-channels",
        dest="hybrid_encoder_conv_hidden",
        type=int,
        default=None,
        help="Alias for --hybrid-encoder-conv-hidden.",
    )
    parser.add_argument(
        "--hybrid-encoder-spectral-hidden-channels",
        dest="hybrid_encoder_spectral_hidden",
        type=int,
        default=None,
        help="Alias for --hybrid-encoder-spectral-hidden.",
    )
    parser.add_argument(
        "--hybrid-resnet-blocks",
        type=int,
        default=6,
        help="Hybrid ResNet bottleneck block count.",
    )
    parser.add_argument(
        "--hybrid-skip-style",
        type=str,
        default="add",
        choices=["add", "concat", "gated_add"],
        help="Hybrid skip-fusion style for hybrid_resnet.",
    )
    parser.add_argument(
        "--spectral-bottleneck-blocks",
        type=int,
        default=6,
        help="Spectral ResNet bottleneck depth.",
    )
    parser.add_argument(
        "--spectral-bottleneck-modes",
        type=int,
        default=12,
        help="Low-mode count per axis for the shared spectral bottleneck branch.",
    )
    parser.add_argument(
        "--spectral-bottleneck-share-weights",
        dest="spectral_bottleneck_share_weights",
        action="store_true",
        default=True,
        help="Share one factorized spectral operator across bottleneck depth.",
    )
    parser.add_argument(
        "--no-spectral-bottleneck-share-weights",
        dest="spectral_bottleneck_share_weights",
        action="store_false",
        help="Use separate spectral operators per bottleneck block.",
    )
    parser.add_argument(
        "--spectral-bottleneck-gate-init",
        type=float,
        default=0.1,
        help="Initial scalar gate for the spectral residual branch.",
    )
    parser.add_argument(
        "--spectral-bottleneck-gate-mode",
        type=str,
        default="shared",
        choices=["shared", "per_block"],
        help="Gate sharing policy for the spectral residual branch.",
    )
    parser.add_argument("--torch-resnet-width", type=int, default=None,
                        help="Hybrid ResNet bottleneck width (must be divisible by 4)")
    parser.add_argument("--fno-modes", type=int, default=12,
                        help="FNO spectral modes")
    parser.add_argument("--fno-width", type=int, default=32,
                        help="FNO hidden width")
    parser.add_argument("--fno-blocks", type=int, default=4,
                        help="FNO spectral block count")
    parser.add_argument("--fno-cnn-blocks", type=int, default=2,
                        help="FNO CNN refiner block count")
    parser.add_argument("--probe-source", type=str, default=None,
                        choices=["custom", "ideal_disk"],
                        help="Expected probe source in dataset metadata")
    parser.add_argument("--sim-backend", type=str, default=None,
                        help="Optional simulation backend selector (legacy compatibility)")
    parser.add_argument("--N", type=int, default=64,
                        help="Patch size N")
    parser.add_argument("--gridsize", type=int, default=1,
                        help="Grid size for stitching")
    parser.add_argument(
        "--position-crop-border",
        type=int,
        default=None,
        help=(
            "Optional border (pixels/side) used to derive effective position-reassembly M. "
            "Default auto-resolves to min(patch_h, patch_w)//4; set 0 to preserve full-window M."
        ),
    )
    parser.add_argument("--optimizer", choices=['adam', 'adamw', 'sgd'], default='adam',
                        help="Optimizer algorithm")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (L2 penalty)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam/AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam/AdamW beta2")
    parser.add_argument("--scheduler", choices=['Default', 'Exponential', 'WarmupCosine', 'ReduceLROnPlateau'], default='Default',
                        help="LR scheduler type")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0,
                        help="Number of warmup epochs for WarmupCosine scheduler")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1,
                        help="Minimum LR ratio for WarmupCosine scheduler (eta_min = base_lr * ratio)")
    parser.add_argument("--plateau-factor", type=float, default=0.5,
                        help="ReduceLROnPlateau factor")
    parser.add_argument("--plateau-patience", type=int, default=2,
                        help="ReduceLROnPlateau patience")
    parser.add_argument("--plateau-min-lr", type=float, default=5e-5,
                        help="ReduceLROnPlateau min lr")
    parser.add_argument("--plateau-threshold", type=float, default=0.0,
                        help="ReduceLROnPlateau threshold")
    # Recon logging CLI flags
    parser.add_argument("--torch-logger", type=str, default="csv",
                        choices=["csv", "tensorboard", "mlflow", "none"],
                        help="Logger backend (default: csv). Use 'none' to disable.")
    parser.add_argument("--recon-log-every-n-epochs", type=int, default=None,
                        help="Log reconstructions every N epochs (default: disabled)")
    parser.add_argument("--recon-log-num-patches", type=int, default=4,
                        help="Number of fixed patch indices to log (default: 4)")
    parser.add_argument("--recon-log-fixed-indices", type=int, nargs='+', default=None,
                        help="Explicit patch indices to log (default: auto-select)")
    parser.add_argument("--recon-log-stitch", action="store_true", default=False,
                        help="Log stitched full-resolution reconstructions")
    parser.add_argument("--recon-log-max-stitch-samples", type=int, default=None,
                        help="Cap on stitched samples (default: no limit)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    from scripts.studies.invocation_logging import capture_runtime_provenance

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    if args.seed is None:
        logging.info("Using random seed %s", seed)

    logger_backend = None if args.torch_logger == "none" else args.torch_logger

    cfg = TorchRunnerConfig(
        train_npz=args.train_npz,
        test_npz=args.test_npz,
        output_dir=args.output_dir,
        architecture=args.architecture,
        training_procedure=args.training_procedure,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        infer_batch_size=args.infer_batch_size,
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        generator_output_mode=args.output_mode,
        N=args.N,
        gridsize=args.gridsize,
        probe_source=args.probe_source,
        torch_loss_mode=args.torch_loss_mode,
        torch_mae_pred_l2_match_target=args.torch_mae_pred_l2_match_target,
        probe_mask=args.probe_mask,
        probe_mask_sigma=args.probe_mask_sigma,
        probe_mask_diameter=args.probe_mask_diameter,
        hybrid_skip_connections=args.hybrid_skip_connections,
        hybrid_downsample_steps=args.hybrid_downsample_steps,
        hybrid_downsample_op=args.hybrid_downsample_op,
        hybrid_encoder_conv_hidden_scale=args.hybrid_encoder_conv_hidden_scale,
        hybrid_encoder_spectral_hidden_scale=args.hybrid_encoder_spectral_hidden_scale,
        hybrid_encoder_conv_hidden_channels=args.hybrid_encoder_conv_hidden,
        hybrid_encoder_spectral_hidden_channels=args.hybrid_encoder_spectral_hidden,
        hybrid_resnet_blocks=args.hybrid_resnet_blocks,
        hybrid_skip_style=args.hybrid_skip_style,
        spectral_bottleneck_blocks=args.spectral_bottleneck_blocks,
        spectral_bottleneck_modes=args.spectral_bottleneck_modes,
        spectral_bottleneck_share_weights=args.spectral_bottleneck_share_weights,
        spectral_bottleneck_gate_init=args.spectral_bottleneck_gate_init,
        spectral_bottleneck_gate_mode=args.spectral_bottleneck_gate_mode,
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_blocks=args.fno_blocks,
        fno_cnn_blocks=args.fno_cnn_blocks,
        resnet_width=args.torch_resnet_width,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        log_grad_norm=args.log_grad_norm,
        grad_norm_log_freq=args.grad_norm_log_freq,
        scheduler=args.scheduler,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min_ratio=args.lr_min_ratio,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_min_lr=args.plateau_min_lr,
        plateau_threshold=args.plateau_threshold,
        logger_backend=logger_backend,
        recon_log_every_n_epochs=args.recon_log_every_n_epochs,
        recon_log_num_patches=args.recon_log_num_patches,
        recon_log_fixed_indices=args.recon_log_fixed_indices,
        recon_log_stitch=args.recon_log_stitch,
        recon_log_max_stitch_samples=args.recon_log_max_stitch_samples,
        position_crop_border=args.position_crop_border,
    )

    result = run_grid_lines_torch(
        cfg,
        invocation_argv=raw_argv,
        invocation_extra={
            "runtime_provenance": capture_runtime_provenance(),
            "invocation_mode": "cli",
        },
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

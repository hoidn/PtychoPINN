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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
    architecture: str  # 'fno' or 'hybrid'
    seed: int = 42
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    infer_batch_size: int = 16
    gradient_clip_val: Optional[float] = 0.0
    gradient_clip_algorithm: str = 'norm'  # 'norm', 'value', or 'agc'
    generator_output_mode: str = "real_imag"
    N: int = 64
    gridsize: int = 1
    torch_loss_mode: str = "mae"
    fno_modes: int = 12
    fno_width: int = 32
    fno_blocks: int = 4
    fno_cnn_blocks: int = 2
    fno_input_transform: str = "none"
    max_hidden_channels: Optional[int] = None
    resnet_width: Optional[int] = None
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
    plateau_min_lr: float = 1e-4
    plateau_threshold: float = 0.0
    # Recon logging
    logger_backend: Optional[str] = None  # 'csv', 'mlflow', etc.
    recon_log_every_n_epochs: Optional[int] = None
    recon_log_num_patches: int = 4
    recon_log_fixed_indices: Optional[List[int]] = None
    recon_log_stitch: bool = False
    recon_log_max_stitch_samples: Optional[int] = None


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
        Literal['cnn', 'fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'hybrid_resnet'],
        cfg.architecture,
    )
    if cfg.architecture == "hybrid_resnet":
        if cfg.fno_blocks < 3:
            raise ValueError(
                "hybrid_resnet requires --fno-blocks >= 3 to downsample to N/4 "
                f"(got {cfg.fno_blocks})."
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

    model_config = ModelConfig(
        N=N_literal,
        gridsize=cfg.gridsize,
        architecture=arch_literal,
        fno_modes=cfg.fno_modes,
        fno_width=cfg.fno_width,
        fno_blocks=cfg.fno_blocks,
        fno_cnn_blocks=cfg.fno_cnn_blocks,
        fno_input_transform=cfg.fno_input_transform,
        max_hidden_channels=cfg.max_hidden_channels,
        resnet_width=cfg.resnet_width,
        generator_output_mode=cfg.generator_output_mode,
    )

    training_config = TrainingConfig(
        model=model_config,
        train_data_file=cfg.train_npz,
        test_data_file=cfg.test_npz,
        nepochs=cfg.epochs,
        batch_size=cfg.batch_size,
        backend='pytorch',
        torch_loss_mode=cfg.torch_loss_mode,
    )
    training_config.log_grad_norm = cfg.log_grad_norm
    training_config.grad_norm_log_freq = cfg.grad_norm_log_freq
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
        enable_checkpointing=cfg.enable_checkpointing,
        logger_backend=cfg.logger_backend,
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

    X = np.asarray(train_data["diffraction"])
    if X.ndim == 3:
        X = X[..., np.newaxis]
    n_samples = X.shape[0]
    channels = X.shape[-1]
    coords = _reshape_coords(train_data.get("coords_nominal"), n_samples, channels)
    probe = train_data.get("probeGuess")
    if probe is None:
        probe = np.ones((cfg.N, cfg.N), dtype=np.complex64)

    train_container = {
        "X": X,
        "coords_nominal": coords,
        "probe": probe,
    }

    test_container = None
    if test_data:
        X_te = np.asarray(test_data["diffraction"])
        if X_te.ndim == 3:
            X_te = X_te[..., np.newaxis]
        n_te = X_te.shape[0]
        channels_te = X_te.shape[-1]
        coords_te = _reshape_coords(test_data.get("coords_nominal"), n_te, channels_te)
        test_probe = test_data.get("probeGuess", probe)
        test_container = {
            "X": X_te,
            "coords_nominal": coords_te,
            "probe": test_probe,
        }

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

    def _normalize_coords(coords: Optional[np.ndarray], n_samples: int, channels: int) -> np.ndarray:
        if coords is None:
            return np.zeros((n_samples, channels, 1, 2), dtype=np.float32)
        coords_np = np.asarray(coords)
        if coords_np.ndim == 2 and coords_np.shape[1] == 2:
            if coords_np.shape[0] == n_samples * channels:
                coords_np = coords_np.reshape(n_samples, channels, 2)
            elif coords_np.shape[0] == n_samples:
                coords_np = np.repeat(coords_np[:, None, :], channels, axis=1)
            else:
                coords_np = np.zeros((n_samples, channels, 2), dtype=np.float32)
            coords_np = coords_np[:, :, None, :]
        elif coords_np.ndim == 3 and coords_np.shape[2] == 2:
            if coords_np.shape[1] != channels:
                coords_np = coords_np[:, :channels, :]
            coords_np = coords_np[:, :, None, :]
        elif coords_np.ndim == 4:
            if coords_np.shape[1] == 1 and coords_np.shape[2] == 2:
                coords_np = np.transpose(coords_np, (0, 3, 1, 2))
            elif coords_np.shape[2] == 1 and coords_np.shape[3] == 2:
                coords_np = coords_np
            else:
                coords_np = np.zeros((n_samples, channels, 1, 2), dtype=np.float32)
        else:
            coords_np = np.zeros((n_samples, channels, 1, 2), dtype=np.float32)
        return coords_np.astype(np.float32)

    if model is None:
        raise ValueError("Model is required for inference")

    X_np = np.asarray(test_data['diffraction'])
    if X_np.ndim == 3:
        X_np = X_np[..., np.newaxis]
    if X_np.ndim == 4 and X_np.shape[1] <= 8 and X_np.shape[-1] > 8:
        X_np = np.transpose(X_np, (0, 2, 3, 1))

    n_samples = X_np.shape[0]
    channels = X_np.shape[-1]
    coords_np = _normalize_coords(test_data.get('coords_nominal'), n_samples, channels)
    probe_np = test_data.get('probeGuess')
    if probe_np is None:
        probe_np = np.ones((cfg.N, cfg.N), dtype=np.complex64)

    X_test = torch.from_numpy(X_np).float().permute(0, 3, 1, 2)
    coords_test = torch.from_numpy(coords_np).float()
    probe_test = torch.from_numpy(probe_np).to(torch.complex64)

    model.eval()
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    preds = []
    batch_size = max(1, cfg.infer_batch_size)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = X_test[start:end].to(device)
            coords_batch = coords_test[start:end].to(device)
            scale_batch = torch.ones((end - start, 1, 1, 1), device=device, dtype=torch.float32)
            probe_batch = probe_test.to(device)
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
    return eval_reconstruction(predictions, ground_truth, label=label)


def save_run_artifacts(
    cfg: TorchRunnerConfig,
    results: Dict[str, Any],
    metrics: Dict[str, float],
) -> Path:
    """Save run artifacts to output directory.

    Creates:
        - output_dir/runs/pinn_<arch>/model.pt
        - output_dir/runs/pinn_<arch>/metrics.json
        - output_dir/runs/pinn_<arch>/history.json
    """
    run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save training history
    history_path = run_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(results.get('history', {}), f, indent=2)

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


def run_grid_lines_torch(cfg: TorchRunnerConfig) -> Dict[str, Any]:
    """Main entry point for Torch grid-lines runner.

    Orchestrates: load data → train → infer → compute metrics → save artifacts

    Args:
        cfg: Runner configuration

    Returns:
        Dict with metrics, artifact paths, and run metadata
    """
    logger.info(f"Starting Torch grid-lines runner: arch={cfg.architecture}")

    # Step 1: Load cached datasets
    logger.info(f"Loading train data from {cfg.train_npz}")
    train_data = load_cached_dataset(cfg.train_npz)

    logger.info(f"Loading test data from {cfg.test_npz}")
    test_data, test_metadata = load_cached_dataset_with_metadata(cfg.test_npz)

    # Step 2: Train model
    logger.info(f"Training {cfg.architecture} model...")
    results = run_torch_training(cfg, train_data, test_data)

    # Step 3: Run inference
    logger.info("Running inference...")
    model = results.get('model')
    if model is None and isinstance(results.get('models'), dict):
        model = results['models'].get('diffraction_to_obj')

    model_params = 0
    if model is not None and hasattr(model, "parameters"):
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    import time
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available:
        torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = run_torch_inference(model, test_data, cfg)
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
    ground_truth = test_data.get('YY_ground_truth', test_data['YY_full'])
    # Use complex predictions for metrics if available
    pred_for_metrics = predictions_complex if predictions_complex is not None else predictions
    if pred_for_metrics.ndim >= 3:
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
    from ptycho.workflows.grid_lines_workflow import save_recon_artifact
    recon_target = pred_for_metrics
    if not np.iscomplexobj(recon_target):
        recon_target = recon_target.astype(np.complex64)
    recon_path = save_recon_artifact(cfg.output_dir, f"pinn_{cfg.architecture}", recon_target)
    metrics = compute_metrics(pred_for_metrics, ground_truth, f"pinn_{cfg.architecture}")

    # Step 5: Save artifacts
    run_dir = save_run_artifacts(cfg, results, metrics)

    logger.info(f"Torch runner complete. Artifacts in {run_dir}")

    result_dict = {
        'architecture': cfg.architecture,
        'run_dir': str(run_dir),
        'metrics': metrics,
        'history': results.get('history', {}),
        'recon_path': str(recon_path),
        'model_params': int(model_params),
        'inference_time_s': float(inference_time_s),
    }

    # Include complex predictions if conversion was done
    if predictions_complex is not None:
        result_dict['predictions_complex'] = predictions_complex

    return result_dict


def main() -> None:
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
                        choices=['fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'hybrid_resnet'],
                        help="Generator architecture to use")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (random if omitted)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--infer-batch-size", type=int, default=16,
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
    parser.add_argument("--N", type=int, default=64,
                        help="Patch size N")
    parser.add_argument("--gridsize", type=int, default=1,
                        help="Grid size for stitching")
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
    parser.add_argument("--plateau-min-lr", type=float, default=1e-4,
                        help="ReduceLROnPlateau min lr")
    parser.add_argument("--plateau-threshold", type=float, default=0.0,
                        help="ReduceLROnPlateau threshold")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    if args.seed is None:
        logging.info("Using random seed %s", seed)

    cfg = TorchRunnerConfig(
        train_npz=args.train_npz,
        test_npz=args.test_npz,
        output_dir=args.output_dir,
        architecture=args.architecture,
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
        torch_loss_mode=args.torch_loss_mode,
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
    )

    result = run_grid_lines_torch(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

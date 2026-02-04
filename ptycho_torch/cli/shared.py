"""
Shared CLI helper functions for PyTorch backend (ADR-003 Phase D.B).

This module centralizes common CLI functionality extracted from training and
inference CLI scripts during the thin wrapper refactor. Functions are designed
to be stateless with clear contracts, raising exceptions for validation errors
rather than exiting directly (allowing callers to format user-friendly messages).

Component Responsibilities:
- resolve_accelerator(): Handle --device → --accelerator backward compatibility
- build_execution_config_from_args(): Construct PyTorchExecutionConfig with validation
- validate_paths(): Check file existence and create output directory

References:
- Blueprint: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md
- Design Decisions: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md
- Spec: specs/ptychodus_api_spec.md §7 (CLI execution config flags)
"""

import warnings
import argparse
from pathlib import Path
from typing import Optional


def resolve_accelerator(accelerator: str = 'auto', device: Optional[str] = None) -> str:
    """
    Resolve accelerator from CLI args, handling --device deprecation and auto-detection.

    Args:
        accelerator: Value from --accelerator flag (default: 'auto')
        device: Value from --device flag (deprecated, optional)

    Returns:
        Resolved accelerator string ('cpu', 'gpu', 'cuda', 'tpu', 'mps')

    Emits:
        DeprecationWarning if device is specified
        UserWarning if 'auto' resolves to CPU due to unavailable CUDA (POLICY-001)

    Examples:
        >>> resolve_accelerator('cpu', None)
        'cpu'
        >>> resolve_accelerator('auto', 'cuda')  # Legacy --device usage
        'gpu'
        >>> resolve_accelerator('cpu', 'cuda')  # Conflict: accelerator wins
        'cpu'
        >>> resolve_accelerator('auto', None)  # Auto-detection: CUDA if available, else CPU with warning
        'cuda'

    Notes:
        - 'auto' now auto-detects: prefers CUDA if available, falls back to CPU with POLICY-001 warning
        - Legacy --device='cuda' maps to accelerator='gpu' (Lightning convention)
        - If both flags specified, --accelerator takes precedence
        - Emits DeprecationWarning for --device usage
        - POLICY-001 enforcement: GPU baseline is required; CPU fallback emits actionable warning
    """
    resolved = accelerator

    if device and accelerator == 'auto':
        # Map legacy --device to --accelerator
        warnings.warn(
            "--device is deprecated. Use --accelerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        resolved = 'cpu' if device == 'cpu' else 'gpu'

    elif device and accelerator != 'auto':
        # Conflict: accelerator takes precedence
        warnings.warn(
            "--device is deprecated. Use --accelerator instead. Ignoring --device value.",
            DeprecationWarning,
            stacklevel=2
        )
        # resolved = accelerator (no change)

    # Auto-detection: prefer CUDA, fallback to CPU with POLICY-001 warning
    if resolved == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                resolved = 'cuda'
            else:
                resolved = 'cpu'
                warnings.warn(
                    "POLICY-001: PyTorch GPU baseline is recommended. "
                    "CUDA is not available; falling back to CPU. "
                    "Install CUDA-enabled PyTorch for optimal performance: "
                    "see https://pytorch.org/get-started/locally/",
                    UserWarning,
                    stacklevel=2
                )
        except ImportError:
            # torch not available (should not happen with POLICY-001, but handle gracefully)
            resolved = 'cpu'
            warnings.warn(
                "POLICY-001: PyTorch is not available. Falling back to CPU accelerator. "
                "Install PyTorch with CUDA support: see https://pytorch.org/get-started/locally/",
                UserWarning,
                stacklevel=2
            )

    return resolved


def build_execution_config_from_args(
    args: argparse.Namespace,
    mode: str = 'training'
):
    """
    Build PyTorchExecutionConfig from CLI args with validation and warnings.

    Args:
        args: Parsed argparse.Namespace containing execution config flags
        mode: 'training' or 'inference' (controls field availability)

    Returns:
        PyTorchExecutionConfig instance

    Raises:
        ValueError: If mode is invalid or validation fails (caught in dataclass __post_init__)

    Emits:
        UserWarning if deterministic=True and num_workers > 0 (training mode only)

    Examples:
        >>> args = argparse.Namespace(accelerator='cpu', deterministic=True, num_workers=0, learning_rate=1e-3, disable_mlflow=False, quiet=False)
        >>> config = build_execution_config_from_args(args, mode='training')
        >>> config.accelerator
        'cpu'

    Notes:
        - Calls resolve_accelerator() to handle --device deprecation
        - Maps --quiet OR --disable_mlflow to enable_progress_bar field
        - Emits UserWarning for deterministic+num_workers performance caveat
        - Validation (accelerator whitelist, non-negative workers, etc.) handled in PyTorchExecutionConfig.__post_init__()
    """
    from ptycho.config.config import PyTorchExecutionConfig

    # Resolve accelerator (handles --device deprecation)
    resolved_accelerator = resolve_accelerator(
        args.accelerator,
        getattr(args, 'device', None)
    )

    # Map disable_mlflow/quiet to enable_progress_bar
    quiet_mode = getattr(args, 'quiet', False) or getattr(args, 'disable_mlflow', False)
    enable_progress_bar = not quiet_mode

    # Handle logger backend with deprecation (Phase EB3.B - ADR-003)
    logger_backend_raw = getattr(args, 'logger_backend', 'mlflow')  # default to MLflow
    if getattr(args, 'disable_mlflow', False):
        # Emit deprecation warning for --disable_mlflow
        warnings.warn(
            "The --disable_mlflow flag is deprecated. "
            "Use --logger none instead to disable all experiment loggers, "
            "or --quiet to suppress progress bars without disabling metrics logging.",
            DeprecationWarning,
            stacklevel=2
        )
        logger_backend_raw = 'none'  # Override with none

    # Map 'none' string to None type for PyTorchExecutionConfig
    logger_backend = None if logger_backend_raw == 'none' else logger_backend_raw

    # Emit deterministic+num_workers warning (training only)
    if mode == 'training' and args.deterministic and args.num_workers > 0:
        warnings.warn(
            f"Deterministic mode with num_workers={args.num_workers} may cause performance degradation. "
            f"Consider setting --num-workers 0 for reproducibility.",
            UserWarning,
            stacklevel=2
        )

    # Construct config (validation happens in __post_init__)
    if mode == 'training':
        return PyTorchExecutionConfig(
            accelerator=resolved_accelerator,
            deterministic=args.deterministic,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            enable_progress_bar=enable_progress_bar,
            # Checkpoint/early-stop knobs (Phase EB1.C - ADR-003)
            enable_checkpointing=getattr(args, 'enable_checkpointing', True),
            checkpoint_save_top_k=getattr(args, 'checkpoint_save_top_k', 1),
            checkpoint_monitor_metric=getattr(args, 'checkpoint_monitor_metric', 'val_loss'),
            checkpoint_mode=getattr(args, 'checkpoint_mode', 'min'),
            early_stop_patience=getattr(args, 'early_stop_patience', 100),
            # Optimization knobs (Phase EB2.A2 - ADR-003)
            scheduler=getattr(args, 'scheduler', 'Default'),
            accum_steps=getattr(args, 'accumulate_grad_batches', 1),
            # Logger backend (Phase EB3.B - ADR-003)
            logger_backend=logger_backend,
            # Recon logging knobs
            recon_log_every_n_epochs=getattr(args, 'recon_log_every_n_epochs', None),
            recon_log_num_patches=getattr(args, 'recon_log_num_patches', 4),
            recon_log_fixed_indices=getattr(args, 'recon_log_fixed_indices', None),
            recon_log_stitch=getattr(args, 'recon_log_stitch', False),
            recon_log_max_stitch_samples=getattr(args, 'recon_log_max_stitch_samples', None),
        )
    elif mode == 'inference':
        return PyTorchExecutionConfig(
            accelerator=resolved_accelerator,
            num_workers=args.num_workers,
            inference_batch_size=getattr(args, 'inference_batch_size', None),
            enable_progress_bar=enable_progress_bar,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'training' or 'inference'.")


def validate_paths(
    train_file: Optional[Path],
    test_file: Optional[Path],
    output_dir: Path,
) -> None:
    """
    Validate input NPZ files exist and create output directory.

    Args:
        train_file: Path to training NPZ file (required for training CLI, optional for inference)
        test_file: Path to test NPZ file (optional)
        output_dir: Directory for outputs (will be created if missing)

    Raises:
        FileNotFoundError: If train_file or test_file does not exist

    Side Effects:
        Creates output_dir and any parent directories (mkdir -p behavior)

    Examples:
        >>> validate_paths(Path('data/train.npz'), None, Path('outputs/'))
        # Creates outputs/ if missing, raises if data/train.npz missing

    Notes:
        - Accepts None for train_file (inference mode) or test_file (optional validation data)
        - Uses Path.mkdir(parents=True, exist_ok=True) for directory creation
        - Raises descriptive FileNotFoundError with path included in message
    """
    if train_file and not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")

    if test_file and not test_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_file}")

    # Create output directory (mkdir -p)
    output_dir.mkdir(parents=True, exist_ok=True)

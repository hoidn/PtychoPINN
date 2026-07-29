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

import argparse
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ptycho_torch.execution_request import ExecutionRequest, ResolutionNotice


_NATIVE_TRAINING = "native-training"
_NATIVE_INFERENCE = "native-inference"
_UNIFIED_TRAINING = "unified-training"
_UNIFIED_INFERENCE = "unified-inference"
_TRAINING_LANES = frozenset({_NATIVE_TRAINING, _UNIFIED_TRAINING})
_INFERENCE_LANES = frozenset({_NATIVE_INFERENCE, _UNIFIED_INFERENCE})
_ALL_LANES = _TRAINING_LANES | _INFERENCE_LANES


@dataclass(frozen=True)
class _ExecutionOptionBinding:
    """A raw option's canonical fields, argparse destination, and CLI lanes."""

    option: str
    fields: tuple[str, ...]
    destination: str
    lanes: frozenset[str]


def _binding(
    option: str,
    fields: str | tuple[str, ...],
    destination: str,
    lanes: frozenset[str],
) -> _ExecutionOptionBinding:
    if isinstance(fields, str):
        fields = (fields,)
    return _ExecutionOptionBinding(option, fields, destination, lanes)


# This is the single suppliedness registry for all supported execution CLIs.
# Destination names remain parser details; only ``fields`` enter a request.
_EXECUTION_OPTION_BINDINGS = (
    _binding(
        "--accelerator",
        "accelerator",
        "accelerator",
        frozenset({_NATIVE_TRAINING, _NATIVE_INFERENCE}),
    ),
    _binding(
        "--torch-accelerator",
        "accelerator",
        "torch_accelerator",
        frozenset({_UNIFIED_TRAINING, _UNIFIED_INFERENCE}),
    ),
    _binding(
        "--device",
        "accelerator",
        "device",
        frozenset({_NATIVE_TRAINING, _NATIVE_INFERENCE}),
    ),
    _binding(
        "--deterministic",
        "deterministic",
        "deterministic",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--no-deterministic",
        "deterministic",
        "deterministic",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-deterministic",
        "deterministic",
        "torch_deterministic",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--num-workers",
        "num_workers",
        "num_workers",
        frozenset({_NATIVE_TRAINING, _NATIVE_INFERENCE}),
    ),
    _binding(
        "--torch-num-workers",
        "num_workers",
        "torch_num_workers",
        frozenset({_UNIFIED_TRAINING, _UNIFIED_INFERENCE}),
    ),
    _binding(
        "--learning-rate",
        "learning_rate",
        "learning_rate",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-learning-rate",
        "learning_rate",
        "torch_learning_rate",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--scheduler",
        "scheduler",
        "scheduler",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-scheduler",
        "scheduler",
        "torch_scheduler",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--accumulate-grad-batches",
        "accum_steps",
        "accumulate_grad_batches",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-accumulate-grad-batches",
        "accum_steps",
        "torch_accumulate_grad_batches",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--logger",
        "logger_backend",
        "logger_backend",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-logger",
        "logger_backend",
        "torch_logger",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--quiet",
        "enable_progress_bar",
        "quiet",
        frozenset(
            {
                _NATIVE_TRAINING,
                _NATIVE_INFERENCE,
                _UNIFIED_TRAINING,
            }
        ),
    ),
    _binding(
        "--disable_mlflow",
        ("logger_backend", "enable_progress_bar"),
        "disable_mlflow",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--enable-checkpointing",
        "enable_checkpointing",
        "enable_checkpointing",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--disable-checkpointing",
        "enable_checkpointing",
        "enable_checkpointing",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-enable-checkpointing",
        "enable_checkpointing",
        "torch_enable_checkpointing",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--checkpoint-save-top-k",
        "checkpoint_save_top_k",
        "checkpoint_save_top_k",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-checkpoint-save-top-k",
        "checkpoint_save_top_k",
        "torch_checkpoint_save_top_k",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--checkpoint-monitor",
        "checkpoint_monitor_metric",
        "checkpoint_monitor_metric",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--checkpoint-mode",
        "checkpoint_mode",
        "checkpoint_mode",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--early-stop-patience",
        "early_stop_patience",
        "early_stop_patience",
        frozenset({_NATIVE_TRAINING}),
    ),
    _binding(
        "--torch-recon-log-every-n-epochs",
        "recon_log_every_n_epochs",
        "torch_recon_log_every_n_epochs",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--torch-recon-log-num-patches",
        "recon_log_num_patches",
        "torch_recon_log_num_patches",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--torch-recon-log-fixed-indices",
        "recon_log_fixed_indices",
        "torch_recon_log_fixed_indices",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--torch-recon-log-stitch",
        "recon_log_stitch",
        "torch_recon_log_stitch",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--torch-recon-log-max-stitch-samples",
        "recon_log_max_stitch_samples",
        "torch_recon_log_max_stitch_samples",
        frozenset({_UNIFIED_TRAINING}),
    ),
    _binding(
        "--inference-batch-size",
        "inference_batch_size",
        "inference_batch_size",
        frozenset({_NATIVE_INFERENCE}),
    ),
    _binding(
        "--torch-inference-batch-size",
        "inference_batch_size",
        "torch_inference_batch_size",
        frozenset({_UNIFIED_INFERENCE}),
    ),
)
_EXECUTION_OPTION_BY_NAME = {
    binding.option: binding for binding in _EXECUTION_OPTION_BINDINGS
}

_TRAINING_EXECUTION_DEFAULTS: Mapping[str, Any] = {
    "accelerator": "auto",
    "deterministic": True,
    "num_workers": 0,
    "learning_rate": 1e-3,
    "enable_progress_bar": True,
    "enable_checkpointing": True,
    "checkpoint_save_top_k": 1,
    "checkpoint_monitor_metric": "val_loss",
    "checkpoint_mode": "min",
    "early_stop_patience": 100,
    "scheduler": "Default",
    "accum_steps": 1,
    "logger_backend": "csv",
    "recon_log_every_n_epochs": None,
    "recon_log_num_patches": 4,
    "recon_log_fixed_indices": None,
    "recon_log_stitch": False,
    "recon_log_max_stitch_samples": None,
}
_INFERENCE_EXECUTION_DEFAULTS: Mapping[str, Any] = {
    "accelerator": "auto",
    "num_workers": 0,
    "inference_batch_size": None,
    "enable_progress_bar": True,
}


def _normalize_option_spelling(option: str) -> str:
    return option.split("=", 1)[0]


def _normalize_lane(lane: str | None) -> str | None:
    if lane is None:
        return None
    normalized = lane.replace("_", "-")
    if normalized not in _ALL_LANES:
        raise ValueError(
            f"Invalid execution CLI lane: {lane}. "
            f"Expected one of {sorted(_ALL_LANES)}."
        )
    return normalized


def canonicalize_execution_options(
    explicit_options: Iterable[str],
    *,
    lane: str | None = None,
) -> tuple[set[str], set[str]]:
    """Map raw supplied option spellings to canonical execution fields."""

    normalized_lane = _normalize_lane(lane)
    explicit_fields: set[str] = set()
    explicit_sources: set[str] = set()
    for raw_option in explicit_options:
        if not isinstance(raw_option, str) or not raw_option.startswith("--"):
            continue
        option = _normalize_option_spelling(raw_option)
        binding = _EXECUTION_OPTION_BY_NAME.get(option)
        if binding is None:
            continue
        if normalized_lane is not None and normalized_lane not in binding.lanes:
            continue
        explicit_fields.update(binding.fields)
        explicit_sources.add(option)
    return explicit_fields, explicit_sources


def _execution_defaults(mode: str) -> Mapping[str, Any]:
    if mode == "training":
        return _TRAINING_EXECUTION_DEFAULTS
    if mode == "inference":
        return _INFERENCE_EXECUTION_DEFAULTS
    raise ValueError(
        f"Invalid mode: {mode}. Expected 'training' or 'inference'."
    )


def _mode_lanes(mode: str) -> frozenset[str]:
    _execution_defaults(mode)
    return _TRAINING_LANES if mode == "training" else _INFERENCE_LANES


def _sources_for_mode(
    explicit_sources: Iterable[str],
    mode: str,
) -> set[str]:
    lanes = _mode_lanes(mode)
    return {
        source
        for source in explicit_sources
        if _EXECUTION_OPTION_BY_NAME[source].lanes & lanes
    }


def _binding_value(
    args: argparse.Namespace,
    field: str,
    explicit_sources: set[str],
    default: Any,
    *,
    none_means_default: bool,
) -> Any:
    """Read the parser destination for an explicitly supplied canonical field."""

    for binding in _EXECUTION_OPTION_BINDINGS:
        if binding.option not in explicit_sources or field not in binding.fields:
            continue
        destination_candidates = [binding.destination]
        if field not in destination_candidates:
            destination_candidates.append(field)
        for destination in destination_candidates:
            if hasattr(args, destination):
                value = getattr(args, destination)
                if value is None and none_means_default:
                    return default
                return value
    return default


def _device_notice(ignored: bool) -> ResolutionNotice:
    suffix = " Ignoring --device value." if ignored else ""
    return ResolutionNotice(
        DeprecationWarning,
        f"--device is deprecated. Use --accelerator instead.{suffix}",
    )


def _disable_mlflow_notice() -> ResolutionNotice:
    return ResolutionNotice(
        DeprecationWarning,
        "The --disable_mlflow flag is deprecated. "
        "Use --logger none instead to disable all experiment loggers, "
        "or --quiet to suppress progress bars without disabling metrics logging.",
    )


def _deterministic_worker_notice(num_workers: int) -> ResolutionNotice:
    return ResolutionNotice(
        UserWarning,
        f"Deterministic mode with num_workers={num_workers} may cause "
        "performance degradation. Consider setting --num-workers 0 for "
        "reproducibility.",
    )


def _normalize_execution_namespace(
    args: argparse.Namespace,
    mode: str = "training",
    *,
    explicit_sources: Iterable[str],
    none_means_default: bool = True,
) -> tuple[dict[str, Any], tuple[ResolutionNotice, ...]]:
    """Normalize parser destinations into primitive execution fields."""

    defaults = _execution_defaults(mode)
    sources = _sources_for_mode(explicit_sources, mode)
    values = dict(defaults)
    notices: list[ResolutionNotice] = []

    canonical_accelerator_sources = {
        "--accelerator",
        "--torch-accelerator",
    } & sources
    deprecated_device_supplied = "--device" in sources
    accelerator = _binding_value(
        args,
        "accelerator",
        canonical_accelerator_sources,
        defaults["accelerator"],
        none_means_default=none_means_default,
    )
    device = None
    if deprecated_device_supplied:
        device = _binding_value(
            args,
            "accelerator",
            {"--device"},
            None,
            none_means_default=none_means_default,
        )
    if device:
        if canonical_accelerator_sources:
            notices.append(_device_notice(ignored=True))
        else:
            accelerator = "cpu" if device == "cpu" else "gpu"
            notices.append(_device_notice(ignored=False))
    values["accelerator"] = accelerator

    derived_sources = {"--quiet", "--disable_mlflow"}
    for field, default in defaults.items():
        if field in {"accelerator", "enable_progress_bar"}:
            continue
        field_sources = {
            source
            for source in sources
            if field in _EXECUTION_OPTION_BY_NAME[source].fields
            and source not in derived_sources
        }
        if not field_sources:
            continue
        values[field] = _binding_value(
            args,
            field,
            field_sources,
            default,
            none_means_default=none_means_default,
        )

    if values.get("logger_backend") == "none":
        values["logger_backend"] = None

    quiet = False
    if "--quiet" in sources:
        quiet = bool(
            _binding_value(
                args,
                "enable_progress_bar",
                {"--quiet"},
                False,
                none_means_default=none_means_default,
            )
        )
    disable_mlflow = False
    if "--disable_mlflow" in sources:
        disable_mlflow = bool(
            _binding_value(
                args,
                "logger_backend",
                {"--disable_mlflow"},
                False,
                none_means_default=none_means_default,
            )
        )
    if {"--quiet", "--disable_mlflow"} & sources:
        values["enable_progress_bar"] = not (quiet or disable_mlflow)
    if disable_mlflow:
        values["logger_backend"] = None
        notices.append(_disable_mlflow_notice())

    if (
        mode == "training"
        and values["deterministic"]
        and values["num_workers"] > 0
    ):
        notices.append(_deterministic_worker_notice(values["num_workers"]))

    return values, tuple(notices)


def build_execution_request_from_args(
    args: argparse.Namespace,
    mode: str = "training",
    *,
    explicit_options: Iterable[str] = (),
    lane: str | None = None,
) -> ExecutionRequest:
    """Build a pure execution request while retaining raw-option provenance."""

    explicit_fields, explicit_sources = canonicalize_execution_options(
        explicit_options,
        lane=lane,
    )
    allowed_fields = set(_execution_defaults(mode))
    explicit_fields &= allowed_fields
    explicit_sources = _sources_for_mode(explicit_sources, mode)
    values, notices = _normalize_execution_namespace(
        args,
        mode=mode,
        explicit_sources=explicit_sources,
    )
    return ExecutionRequest(
        values=values,
        explicit_fields=frozenset(explicit_fields),
        notices=notices,
    )


def resolve_accelerator(accelerator: str = 'auto', device: Optional[str] = None) -> str:
    """
    Resolve accelerator from CLI args, handling --device deprecation and auto-detection.

    Args:
        accelerator: Value from --accelerator flag (default: 'auto')
        device: Value from --device flag (deprecated, optional)

    Returns:
        Resolved accelerator string ('cpu', 'gpu', 'cuda', 'mps')

    Raises:
        ValueError: If TPU is requested. This runtime has no Torch-XLA contract.

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
        - TPU execution is rejected because this runtime does not support Torch-XLA
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

    if resolved == 'tpu':
        raise ValueError(
            "Torch-XLA TPU execution is unsupported by this PyTorch runtime. "
            "Use --accelerator cpu, gpu/cuda, or mps."
        )

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
    logger_backend_raw = getattr(args, 'logger_backend', 'csv')  # default to CSV
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

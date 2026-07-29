"""
Shared CLI helper functions for PyTorch backend (ADR-003 Phase D.B).

This module centralizes common CLI functionality extracted from training and
inference CLI scripts during the thin wrapper refactor. Functions are designed
to be stateless with clear contracts, raising exceptions for validation errors
rather than exiting directly (allowing callers to format user-friendly messages).

Component Responsibilities:
- build_execution_request_from_args(): Preserve explicit runtime intent
- build_training_config_patch_from_args(): Preserve explicit optimizer intent
- validate_paths(): Check file existence and create output directory

References:
- Blueprint: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md
- Design Decisions: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md
- Spec: specs/ptychodus_api_spec.md §7 (CLI execution config flags)
"""

import argparse
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


@dataclass(frozen=True)
class _TrainingOptionBinding:
    """An explicitly supplied optimizer option and its canonical owner."""

    option: str
    field: str
    destination: str
    lanes: frozenset[str]


def _training_binding(
    option: str,
    field: str,
    destination: str | None = None,
    *,
    lanes: frozenset[str],
) -> _TrainingOptionBinding:
    return _TrainingOptionBinding(
        option=option,
        field=field,
        destination=destination or field,
        lanes=lanes,
    )


# Canonical public spellings precede Torch aliases so one owner has an
# unambiguous value when both spellings are supplied.
_TRAINING_OPTION_BINDINGS = (
    _training_binding(
        "--learning-rate",
        "learning_rate",
        lanes=frozenset({_NATIVE_TRAINING}),
    ),
    _training_binding(
        "--scheduler",
        "scheduler",
        lanes=_TRAINING_LANES,
    ),
    _training_binding(
        "--accumulate-grad-batches",
        "accum_steps",
        "accumulate_grad_batches",
        lanes=frozenset({_NATIVE_TRAINING}),
    ),
    *(
        _training_binding(
            f"--{field}",
            field,
            lanes=frozenset({_UNIFIED_TRAINING}),
        )
        for field in (
            "gradient_clip_val",
            "gradient_clip_algorithm",
            "optimizer",
            "momentum",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "lr_warmup_epochs",
            "lr_min_ratio",
            "plateau_factor",
            "plateau_patience",
            "plateau_min_lr",
            "plateau_threshold",
        )
    ),
    _training_binding(
        "--torch-learning-rate",
        "learning_rate",
        "torch_learning_rate",
        lanes=frozenset({_UNIFIED_TRAINING}),
    ),
    _training_binding(
        "--torch-accumulate-grad-batches",
        "accum_steps",
        "torch_accumulate_grad_batches",
        lanes=frozenset({_UNIFIED_TRAINING}),
    ),
    _training_binding(
        "--torch-scheduler",
        "scheduler",
        "torch_scheduler",
        lanes=frozenset({_UNIFIED_TRAINING}),
    ),
    *(
        _training_binding(
            option,
            field,
            destination,
            lanes=frozenset({_UNIFIED_TRAINING}),
        )
        for option, field, destination in (
            (
                "--torch-plateau-factor",
                "plateau_factor",
                "torch_plateau_factor",
            ),
            (
                "--torch-plateau-patience",
                "plateau_patience",
                "torch_plateau_patience",
            ),
            (
                "--torch-plateau-min-lr",
                "plateau_min_lr",
                "torch_plateau_min_lr",
            ),
            (
                "--torch-plateau-threshold",
                "plateau_threshold",
                "torch_plateau_threshold",
            ),
        )
    ),
)

_TRAINING_EXECUTION_DEFAULTS: Mapping[str, Any] = {
    "accelerator": "auto",
    "deterministic": True,
    "num_workers": 0,
    "enable_progress_bar": True,
    "enable_checkpointing": True,
    "checkpoint_save_top_k": 1,
    "checkpoint_monitor_metric": "val_loss",
    "checkpoint_mode": "min",
    "early_stop_patience": 100,
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


def build_training_config_patch_from_args(
    args: argparse.Namespace,
    *,
    explicit_options: Iterable[str] = (),
    lane: str,
) -> dict[str, Any]:
    """Return explicitly supplied optimizer values under canonical ownership."""

    normalized_lane = _normalize_lane(lane)
    if normalized_lane not in _TRAINING_LANES:
        raise ValueError(
            "Training config patches require a training CLI lane."
        )
    supplied_options = {
        _normalize_option_spelling(option)
        for option in explicit_options
        if isinstance(option, str) and option.startswith("--")
    }
    patch: dict[str, Any] = {}
    for binding in _TRAINING_OPTION_BINDINGS:
        if (
            normalized_lane not in binding.lanes
            or binding.option not in supplied_options
            or binding.field in patch
            or not hasattr(args, binding.destination)
        ):
            continue
        patch[binding.field] = getattr(args, binding.destination)
    return patch


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

"""Effective Lightning runtime and child-process provenance records."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import torch
import torch.distributed as dist


def _runtime_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_runtime_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _runtime_json_value(item) for key, item in value.items()}
    value_type = type(value)
    return {"class": f"{value_type.__module__}.{value_type.__qualname__}"}


def _runtime_class(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _runtime_attributes(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    settings = {"class": _runtime_class(value)}
    for name in names:
        try:
            attribute = getattr(value, name)
        except (AttributeError, RuntimeError):
            continue
        if attribute is not None:
            settings[name] = _runtime_json_value(attribute)
    return settings


def _callback_runtime(callback: Any) -> dict[str, Any]:
    return _runtime_attributes(
        callback,
        (
            "monitor",
            "mode",
            "save_top_k",
            "save_last",
            "patience",
            "dirpath",
        ),
    )


def _logger_runtime(logger: Any) -> dict[str, Any]:
    return _runtime_attributes(
        logger,
        ("name", "version", "save_dir", "log_dir", "root_dir", "experiment_name"),
    )


def _logger_list(value: Any) -> list[Any]:
    if value in (None, False):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _trainer_kwargs_runtime(trainer_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in trainer_kwargs.items():
        if key == "callbacks":
            serialized[key] = [_callback_runtime(callback) for callback in value]
        elif key == "logger":
            serialized[key] = [
                _logger_runtime(logger) for logger in _logger_list(value)
            ]
        else:
            serialized[key] = _runtime_json_value(value)
    return serialized


def strategy_runtime(strategy: Any) -> dict[str, Any]:
    """Serialize the effective Lightning strategy without initializing DDP."""

    backend = getattr(strategy, "process_group_backend", None)
    if backend is None:
        backend = getattr(strategy, "_process_group_backend", None)
    if backend is None:
        resolve_backend = getattr(strategy, "_get_process_group_backend", None)
        if callable(resolve_backend):
            backend = resolve_backend()
    if backend is None and dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
    settings = _runtime_attributes(strategy, ("root_device", "parallel_devices"))
    settings["process_group_backend"] = backend
    start_method = getattr(strategy, "_start_method", None)
    try:
        launcher = getattr(strategy, "launcher", None)
    except (AttributeError, RuntimeError):
        launcher = None
    if start_method is None and launcher is not None:
        start_method = getattr(launcher, "_start_method", None)
    if start_method is not None:
        settings["start_method"] = start_method
    if launcher is not None:
        settings["launcher"] = {"class": _runtime_class(launcher)}
    return settings


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(
            "cannot resolve one unambiguous NVIDIA driver version"
        ) from error
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    unique_values = set(values)
    if completed.returncode != 0 or len(unique_values) != 1:
        raise RuntimeError("cannot resolve one unambiguous NVIDIA driver version")
    return unique_values.pop()


def _cuda_fingerprint(*, device: torch.device, precision: Any) -> dict[str, Any] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    import lightning
    import numpy as np

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    cudnn_version = torch.backends.cudnn.version()
    return {
        "gpu_name": torch.cuda.get_device_name(device_index),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "driver_version": _nvidia_driver_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(cudnn_version) if cudnn_version is not None else 0,
        "lightning_version": str(lightning.__version__),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "precision": str(precision),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def build_effective_runtime(
    resolved_seed: int,
    trainer_kwargs: Mapping[str, Any],
    execution_config: Any,
    dataloader_settings: Mapping[str, Any] | None = None,
    trainer: Any = None,
) -> dict[str, Any]:
    """Capture requested and effective runtime from one constructed Trainer."""

    if trainer is None:
        raise ValueError("trainer is required to record effective runtime")
    if dataloader_settings is None:
        num_workers = execution_config.num_workers
        dataloader_settings = {
            "num_workers": num_workers,
            "pin_memory": execution_config.pin_memory,
            "persistent_workers": (
                execution_config.persistent_workers if num_workers > 0 else False
            ),
            "prefetch_factor": (
                (execution_config.prefetch_factor or 2) if num_workers > 0 else None
            ),
        }
    precision_value = getattr(trainer.precision_plugin, "precision", None)
    root_device = trainer.strategy.root_device
    mps_backend = getattr(torch.backends, "mps", None)
    environment = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": bool(mps_backend is not None and mps_backend.is_available()),
        "fingerprint": _cuda_fingerprint(
            device=root_device,
            precision=precision_value,
        ),
    }
    effective = {
        "precision": {
            "value": precision_value,
            "plugin": _runtime_class(trainer.precision_plugin),
        },
        "deterministic": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
        "environment": environment,
        "accelerator": {
            "class": _runtime_class(trainer.accelerator),
            "device_type": root_device.type,
            "trainer_value": _runtime_json_value(trainer_kwargs.get("accelerator")),
        },
        "devices": {
            "count": trainer.num_devices,
            "ids": _runtime_json_value(trainer.device_ids),
            "trainer_value": _runtime_json_value(trainer_kwargs.get("devices")),
        },
        "strategy": {
            **strategy_runtime(trainer.strategy),
            "trainer_value": _runtime_json_value(trainer_kwargs.get("strategy")),
        },
        "callbacks": [_callback_runtime(callback) for callback in trainer.callbacks],
        "loggers": [_logger_runtime(logger) for logger in trainer.loggers],
        "dataloader": _runtime_json_value(dataloader_settings),
    }
    requested = {
        "accelerator": execution_config.accelerator,
        "devices": execution_config.devices,
        "strategy": execution_config.strategy,
        "deterministic": execution_config.deterministic,
        "precision": execution_config.precision,
        "enable_progress_bar": execution_config.enable_progress_bar,
        "enable_checkpointing": execution_config.enable_checkpointing,
        "checkpoint_save_top_k": execution_config.checkpoint_save_top_k,
        "checkpoint_monitor_metric": execution_config.checkpoint_monitor_metric,
        "checkpoint_mode": execution_config.checkpoint_mode,
        "early_stop_patience": execution_config.early_stop_patience,
        "dataloader": {
            "num_workers": execution_config.num_workers,
            "pin_memory": execution_config.pin_memory,
            "persistent_workers": execution_config.persistent_workers,
            "prefetch_factor": execution_config.prefetch_factor,
        },
    }
    return {
        "seed": resolved_seed,
        "requested": requested,
        "effective": effective,
        "trainer_kwargs": _trainer_kwargs_runtime(trainer_kwargs),
        "dataloader": _runtime_json_value(dataloader_settings),
        "precision": precision_value,
    }


def write_effective_runtime_json(
    path: Path,
    runtime: Mapping[str, Any],
) -> None:
    """Atomically publish a completed effective-runtime record."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(runtime, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

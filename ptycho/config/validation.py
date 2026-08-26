"""Structural and resource validation for public configuration records."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from .config import (
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    resolve_model_object_policy,
)


_MODEL_CONFIG_ADAPTER = TypeAdapter(ModelConfig)
_TRAINING_CONFIG_ADAPTER = TypeAdapter(TrainingConfig)
_INFERENCE_CONFIG_ADAPTER = TypeAdapter(InferenceConfig)


def _raise_public_validation_error(
    error: ValidationError,
    *,
    root: str,
) -> None:
    messages = []
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in detail["loc"])
        path = f"{root}.{location}" if location else root
        messages.append(f"{path}: {detail['msg']}")
    raise ValueError("; ".join(messages)) from error


def _validate_public_structure(
    adapter: TypeAdapter,
    value: Any,
    *,
    root: str,
    strict: bool,
) -> Any:
    try:
        return adapter.validate_python(
            value,
            strict=strict,
            context={"strict_instance": strict},
        )
    except ValidationError as error:
        _raise_public_validation_error(error, root=root)


def _object_policy_backend(backend: Any) -> Literal["torch", "tensorflow"]:
    return "torch" if backend == "pytorch" else "tensorflow"


def validate_model_config_structure(config: ModelConfig) -> None:
    """Validate public model types, domains, ranges, and policy joins."""

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    _validate_public_structure(
        _MODEL_CONFIG_ADAPTER,
        config,
        root="model",
        strict=True,
    )

    resolve_model_object_policy(config, warn_deprecated=False)


def validate_training_config_structure(config: TrainingConfig) -> None:
    """Validate a complete training record without filesystem checks."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    _validate_public_structure(
        _TRAINING_CONFIG_ADAPTER,
        config,
        root="training",
        strict=True,
    )
    validate_model_config_structure(config.model)

    if config.realspace_mae_weight > 0 and config.realspace_weight <= 0:
        raise ValueError("realspace_mae_weight requires positive realspace_weight")

    resolve_model_object_policy(
        config.model,
        backend=_object_policy_backend(config.backend),
        warn_deprecated=False,
    )


def validate_inference_config_structure(config: InferenceConfig) -> None:
    """Validate a complete inference record without filesystem checks."""

    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")
    _validate_public_structure(
        _INFERENCE_CONFIG_ADAPTER,
        config,
        root="inference",
        strict=True,
    )
    validate_model_config_structure(config.model)

    resolve_model_object_policy(
        config.model,
        backend=_object_policy_backend(config.backend),
        warn_deprecated=False,
    )


def validate_runnable_training_config(config: TrainingConfig) -> None:
    """Validate requirements needed to begin a training run."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if config.train_data_file is None:
        raise ValueError("train_data_file is required for runnable training")
    if not config.train_data_file.exists():
        raise ValueError(f"train_data_file must exist: {config.train_data_file}")
    if not config.train_data_file.is_file():
        raise ValueError(
            f"train_data_file must be a regular file: {config.train_data_file}"
        )
    if not os.access(config.train_data_file, os.R_OK):
        raise ValueError(f"train_data_file must be readable: {config.train_data_file}")
    if config.nepochs <= 0:
        raise ValueError(f"nepochs must be positive, got {config.nepochs!r}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size!r}")
    if config.nphotons <= 0:
        raise ValueError(f"nphotons must be positive, got {config.nphotons!r}")
    if config.training_groups is None or config.training_groups <= 0:
        raise ValueError(
            f"training_groups must be positive, got {config.training_groups!r}"
        )
    if config.train_raw_selection is not None and config.train_raw_selection <= 0:
        raise ValueError(
            "train_raw_selection must be positive when set, "
            f"got {config.train_raw_selection!r}"
        )


def validate_inference_resources(config: InferenceConfig) -> None:
    """Validate resources needed by a resolved inference request."""

    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")

    model_path = config.model_path
    if not model_path.exists():
        raise ValueError(f"model_path must exist: {model_path}")
    if not model_path.is_dir():
        raise ValueError(
            f"model_path must be a directory containing wts.h5.zip, got {model_path}"
        )

    model_archive = model_path / "wts.h5.zip"
    if not model_archive.exists():
        raise ValueError(
            f"{config.backend} model_path must contain wts.h5.zip: {model_archive}"
        )
    if not model_archive.is_file():
        raise ValueError(f"model archive must be a regular file: {model_archive}")
    if not os.access(model_archive, os.R_OK):
        raise ValueError(f"model archive must be readable: {model_archive}")

    test_data_path = config.test_data_file
    if not test_data_path.exists():
        raise ValueError(f"test_data_file must exist: {test_data_path}")
    if not test_data_path.is_file():
        raise ValueError(f"test_data_file must be a regular file: {test_data_path}")
    if not os.access(test_data_path, os.R_OK):
        raise ValueError(f"test_data_file must be readable: {test_data_path}")


__all__ = [
    "validate_inference_config_structure",
    "validate_inference_resources",
    "validate_model_config_structure",
    "validate_runnable_training_config",
    "validate_training_config_structure",
]

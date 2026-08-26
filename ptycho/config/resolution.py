"""Pure source resolution for the public configuration dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
import os
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin
import warnings

from pydantic import TypeAdapter, ValidationError

from .config import (
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    resolve_model_object_policy,
)


_MODEL_INPUT_NAMES = frozenset(f.name for f in fields(ModelConfig))
_TRAINING_INPUT_NAMES = frozenset(f.name for f in fields(TrainingConfig)) - {"model"}
_INFERENCE_INPUT_NAMES = frozenset(f.name for f in fields(InferenceConfig)) - {"model"}

assert _MODEL_INPUT_NAMES.isdisjoint(_TRAINING_INPUT_NAMES)
assert _MODEL_INPUT_NAMES.isdisjoint(_INFERENCE_INPUT_NAMES)


def _is_path_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _is_path_annotation(get_args(annotation)[0])
    return annotation is Path


def _required_path_field_names(config_type) -> frozenset[str]:
    """Required (no-default) Path-typed field names from type hints."""
    names = set()
    for item in fields(config_type):
        if item.name == "model":
            continue
        if item.default is not MISSING or item.default_factory is not MISSING:
            continue
        if _is_path_annotation(item.type):
            names.add(item.name)
    return frozenset(names)


_INFERENCE_REQUIRED_PATH_FIELDS = _required_path_field_names(InferenceConfig)


_N_IMAGES_DEPRECATION_MESSAGE = (
    "Parameter 'n_images' is deprecated and will be removed in a future "
    "version. Use the canonical group-count field ('training_groups' for "
    "training, 'inference_groups' for inference) instead, which always means "
    "the number of groups regardless of gridsize."
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
        return adapter.validate_python(value, strict=strict, context={"strict_instance": strict})
    except ValidationError as error:
        _raise_public_validation_error(error, root=root)


def _sorted_names(names: set[Any]) -> list[Any]:
    return sorted(names, key=lambda name: str(name))


def _values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


def _dataclass_source_mapping(
    value: TrainingConfig | InferenceConfig,
) -> dict[str, Any]:
    if not isinstance(value.model, ModelConfig):
        raise ValueError(
            f"{type(value).__name__}.model must be a ModelConfig, "
            f"got {type(value.model).__name__}"
        )
    return {
        **{
            item.name: getattr(value, item.name)
            for item in fields(value)
            if item.name != "model"
        },
        "model": {
            item.name: getattr(value.model, item.name) for item in fields(value.model)
        },
    }


def _materialize_source(
    value: Mapping[str, Any] | TrainingConfig | InferenceConfig | None,
    *,
    source: str,
    config_type: type[TrainingConfig] | type[InferenceConfig],
    allow_dataclass: bool,
) -> Mapping[str, Any]:
    if value is None:
        return {}
    if allow_dataclass and isinstance(value, config_type):
        return _dataclass_source_mapping(value)
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        raise ValueError(
            f"{source} configuration must be a mapping or "
            f"{config_type.__name__}, got {type(value).__name__}"
        )
    raise ValueError(
        f"{source} configuration source must be a mapping, got {type(value).__name__}"
    )


def _normalize_public_source(
    values: Mapping[str, Any] | TrainingConfig | InferenceConfig | None,
    *,
    source: str,
    config_type: type[TrainingConfig] | type[InferenceConfig],
    workflow_names: frozenset[str],
    allow_dataclass: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = _materialize_source(
        values,
        source=source,
        config_type=config_type,
        allow_dataclass=allow_dataclass,
    )

    nested_model: Mapping[str, Any]
    if "model" in values:
        nested_model = values["model"]
        if not isinstance(nested_model, Mapping):
            raise ValueError(
                f"{source} configuration field 'model' must be a mapping, "
                f"got {type(nested_model).__name__}"
            )
    else:
        nested_model = {}

    known_root_names = _MODEL_INPUT_NAMES | workflow_names | {"model"}
    unknown_root_names = set(values) - known_root_names
    if unknown_root_names:
        raise ValueError(
            f"{source} configuration has unknown root fields "
            f"{_sorted_names(unknown_root_names)}"
        )

    unknown_model_names = set(nested_model) - _MODEL_INPUT_NAMES
    if unknown_model_names:
        raise ValueError(
            f"{source} configuration has unknown model fields "
            f"{_sorted_names(unknown_model_names)}"
        )

    model_values = dict(nested_model)
    for name in sorted(_MODEL_INPUT_NAMES):
        if name not in values:
            continue
        if name in nested_model and not _values_equal(
            values[name],
            nested_model[name],
        ):
            raise ValueError(
                f"{source} field {name!r} has conflicting flat "
                f"{name!r} and model.{name} values"
            )
        model_values[name] = values[name]

    workflow_values = {name: values[name] for name in workflow_names if name in values}
    return model_values, workflow_values


def _resolve_group_alias(
    values: Mapping[str, Any],
    *,
    source: str,
    canonical_name: str,
) -> tuple[dict[str, Any], bool]:
    resolved = dict(values)
    if "n_images" not in resolved:
        return resolved, False

    legacy = resolved["n_images"]
    canonical = resolved.get(canonical_name)
    if (
        canonical is not None
        and legacy is not None
        and not _values_equal(canonical, legacy)
    ):
        raise ValueError(
            f"{source} field 'n_images' conflicts with canonical '{canonical_name}'"
        )
    if canonical is None:
        resolved[canonical_name] = legacy
    resolved["n_images"] = None
    return resolved, legacy is not None


def _warn_deprecated_group_alias() -> None:
    warnings.warn(
        _N_IMAGES_DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=3,
    )


def _object_policy_backend(backend: Any) -> str:
    return "torch" if backend == "pytorch" else "tensorflow"


def resolve_training_config(
    file_mapping: Mapping[str, Any] | TrainingConfig | None,
    explicit_cli_patch: Mapping[str, Any] | None,
) -> TrainingConfig:
    """Resolve file/programmatic and explicit CLI values into a fresh config."""

    file_model, file_training = _normalize_public_source(
        file_mapping,
        source="file",
        config_type=TrainingConfig,
        workflow_names=_TRAINING_INPUT_NAMES,
        allow_dataclass=True,
    )
    cli_model, cli_training = _normalize_public_source(
        explicit_cli_patch,
        source="explicit CLI",
        config_type=TrainingConfig,
        workflow_names=_TRAINING_INPUT_NAMES,
        allow_dataclass=False,
    )
    file_training, file_used_alias = _resolve_group_alias(
        file_training,
        source="file",
        canonical_name="training_groups",
    )
    cli_training, cli_used_alias = _resolve_group_alias(
        cli_training,
        source="explicit CLI",
        canonical_name="training_groups",
    )

    training_values = dict(file_training)
    training_values.update(cli_training)
    model_values = dict(file_model)
    model_values.update(cli_model)
    candidate = _validate_public_structure(
        _TRAINING_CONFIG_ADAPTER,
        {"model": model_values, **training_values},
        root="training",
        strict=False,
    )
    raw_model = candidate.model
    candidate.model = resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(candidate.backend),
        warn_deprecated=False,
    )
    validate_training_config_structure(candidate)
    resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(candidate.backend),
        warn_deprecated=True,
    )
    if file_used_alias or cli_used_alias:
        _warn_deprecated_group_alias()
    return candidate


def resolve_inference_config(
    file_mapping: Mapping[str, Any] | InferenceConfig | None,
    explicit_cli_patch: Mapping[str, Any] | None,
) -> InferenceConfig:
    """Resolve file/programmatic and explicit CLI values into a fresh config."""

    file_model, file_inference = _normalize_public_source(
        file_mapping,
        source="file",
        config_type=InferenceConfig,
        workflow_names=_INFERENCE_INPUT_NAMES,
        allow_dataclass=True,
    )
    cli_model, cli_inference = _normalize_public_source(
        explicit_cli_patch,
        source="explicit CLI",
        config_type=InferenceConfig,
        workflow_names=_INFERENCE_INPUT_NAMES,
        allow_dataclass=False,
    )
    file_inference, file_used_alias = _resolve_group_alias(
        file_inference,
        source="file",
        canonical_name="inference_groups",
    )
    cli_inference, cli_used_alias = _resolve_group_alias(
        cli_inference,
        source="explicit CLI",
        canonical_name="inference_groups",
    )

    inference_values = dict(file_inference)
    inference_values.update(cli_inference)
    missing = sorted(
        name
        for name in _INFERENCE_REQUIRED_PATH_FIELDS
        if name not in inference_values
    )
    if missing:
        raise ValueError(
            f"inference configuration is missing required fields {missing}"
        )
    model_values = dict(file_model)
    model_values.update(cli_model)
    candidate = _validate_public_structure(
        _INFERENCE_CONFIG_ADAPTER,
        {"model": model_values, **inference_values},
        root="inference",
        strict=False,
    )
    raw_model = candidate.model
    candidate.model = resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(candidate.backend),
        warn_deprecated=False,
    )
    validate_inference_config_structure(candidate)
    resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(candidate.backend),
        warn_deprecated=True,
    )
    if file_used_alias or cli_used_alias:
        _warn_deprecated_group_alias()
    return candidate


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


def _validate_sampling_semantics(
    config: TrainingConfig | InferenceConfig,
) -> None:
    if config.enable_oversampling and config.model.gridsize > 1:
        pool_size = (
            config.neighbor_count
            if config.neighbor_pool_size is None
            else config.neighbor_pool_size
        )
        group_size = config.model.gridsize**2
        if pool_size < group_size:
            raise ValueError(
                "oversampling requires neighbor_pool_size or neighbor_count "
                f">= gridsize² ({group_size}), got {pool_size}"
            )


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
    _validate_sampling_semantics(config)

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
    _validate_sampling_semantics(config)

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
        raise ValueError(f"training_groups must be positive, got {config.training_groups!r}")
    if config.train_raw_selection is not None and config.train_raw_selection <= 0:
        raise ValueError(
            f"train_raw_selection must be positive when set, got {config.train_raw_selection!r}"
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
    "resolve_inference_config",
    "resolve_training_config",
    "validate_inference_config_structure",
    "validate_inference_resources",
    "validate_model_config_structure",
    "validate_runnable_training_config",
    "validate_training_config_structure",
]

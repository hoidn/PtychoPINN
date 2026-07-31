"""Pure source resolution for the public configuration dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import os
from typing import Any
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

# Backward-compat mapping: old flat training field names → (sub_config_name, new_field_name).
# Allows callers to still pass flat field names that are now nested inside TrainingConfig.
_TRAINING_FLAT_FIELD_MAP: dict[str, tuple[str, str]] = {
    "train_data_file": ("data", "train_data_file"),
    "test_data_file": ("data", "test_data_file"),
    "nphotons": ("data", "nphotons"),
    "n_groups": ("sampling", "n_groups"),
    "n_images": ("sampling", "n_images"),
    "n_subsample": ("sampling", "n_subsample"),
    "subsample_seed": ("sampling", "subsample_seed"),
    "neighbor_count": ("sampling", "neighbor_count"),
    "enable_oversampling": ("sampling", "enable_oversampling"),
    "neighbor_pool_size": ("sampling", "neighbor_pool_size"),
    "sequential_sampling": ("sampling", "sequential_sampling"),
    "torch_loss_mode": ("loss", "torch_loss_mode"),
    "torch_mae_pred_l2_match_target": ("loss", "torch_mae_pred_l2_match_target"),
    "mae_weight": ("tf_loss", "mae_weight"),
    "nll_weight": ("tf_loss", "nll_weight"),
    "realspace_mae_weight": ("tf_loss", "realspace_mae_weight"),
    "realspace_weight": ("tf_loss", "realspace_weight"),
    "gradient_clip_val": ("gradient_clip", "val"),
    "gradient_clip_algorithm": ("gradient_clip", "algorithm"),
}


def _lift_flat_training_fields(values: Any) -> Any:
    """Translate any legacy flat field names to the nested sub-config format.

    Non-mapping values are returned unchanged so that _normalize_public_source
    can raise the appropriate type error.
    """
    if not isinstance(values, Mapping):
        return values
    result: dict[str, Any] = {}
    for key, value in values.items():
        if key in _TRAINING_FLAT_FIELD_MAP:
            sub, field_name = _TRAINING_FLAT_FIELD_MAP[key]
            existing = result.get(sub, {})
            if not isinstance(existing, dict):
                existing = {}
            existing[field_name] = value
            result[sub] = existing
        else:
            result[key] = value
    return result


_N_IMAGES_DEPRECATION_MESSAGE = (
    "Parameter 'n_images' is deprecated and will be removed in a future "
    "version. Use 'n_groups' instead, which always means the number of "
    "groups regardless of gridsize."
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


def _normalize_public_source(
    values: Mapping[str, Any],
    *,
    source: str,
    workflow_names: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise ValueError(
            f"{source} configuration source must be a mapping, "
            f"got {type(values).__name__}"
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
            values[name], nested_model[name]
        ):
            raise ValueError(
                f"{source} field {name!r} has conflicting flat "
                f"{name!r} and model.{name} values"
            )
        model_values[name] = values[name]

    workflow_values = {
        name: values[name] for name in workflow_names if name in values
    }
    return model_values, workflow_values


def _resolve_group_alias(
    values: Mapping[str, Any],
    *,
    source: str,
) -> tuple[dict[str, Any], bool]:
    resolved = dict(values)

    # For TrainingConfig the group fields are nested inside "sampling".
    # For InferenceConfig they remain at the top level.
    if "sampling" in resolved:
        container = dict(resolved.get("sampling", {}))
        in_sampling = True
    else:
        container = resolved
        in_sampling = False

    if "n_images" not in container:
        return resolved, False

    legacy = container["n_images"]
    canonical = container.get("n_groups")
    if (
        canonical is not None
        and legacy is not None
        and canonical != legacy
    ):
        location = "sampling" if in_sampling else "root"
        raise ValueError(
            f"{source} {location} field 'n_images' conflicts with canonical 'n_groups'"
        )
    if canonical is None:
        container["n_groups"] = legacy
    container["n_images"] = None

    if in_sampling:
        resolved["sampling"] = container
    return resolved, legacy is not None


def _warn_deprecated_group_alias() -> None:
    warnings.warn(
        _N_IMAGES_DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=3,
    )


def _object_policy_backend(backend: Any) -> str:
    return "torch" if backend == "pytorch" else "tensorflow"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base, deep-merging nested dicts."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_training_config(
    file_mapping: Mapping[str, Any] | None,
    explicit_cli_patch: Mapping[str, Any] | None,
) -> TrainingConfig:
    """Resolve file and explicitly supplied CLI values into a fresh config."""

    file_model, file_training = _normalize_public_source(
        _lift_flat_training_fields({} if file_mapping is None else file_mapping),
        source="file",
        workflow_names=_TRAINING_INPUT_NAMES,
    )
    cli_model, cli_training = _normalize_public_source(
        _lift_flat_training_fields({} if explicit_cli_patch is None else explicit_cli_patch),
        source="explicit CLI",
        workflow_names=_TRAINING_INPUT_NAMES,
    )
    file_training, file_used_alias = _resolve_group_alias(
        file_training,
        source="file",
    )
    cli_training, cli_used_alias = _resolve_group_alias(
        cli_training,
        source="explicit CLI",
    )

    training_values = _deep_merge(dict(file_training), dict(cli_training))
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
    file_mapping: Mapping[str, Any] | None,
    explicit_cli_patch: Mapping[str, Any] | None,
) -> InferenceConfig:
    """Resolve file and explicitly supplied CLI values into a fresh config."""

    file_model, file_inference = _normalize_public_source(
        {} if file_mapping is None else file_mapping,
        source="file",
        workflow_names=_INFERENCE_INPUT_NAMES,
    )
    cli_model, cli_inference = _normalize_public_source(
        {} if explicit_cli_patch is None else explicit_cli_patch,
        source="explicit CLI",
        workflow_names=_INFERENCE_INPUT_NAMES,
    )
    file_inference, file_used_alias = _resolve_group_alias(
        file_inference,
        source="file",
    )
    cli_inference, cli_used_alias = _resolve_group_alias(
        cli_inference,
        source="explicit CLI",
    )

    inference_values = dict(file_inference)
    inference_values.update(cli_inference)
    missing = sorted(
        name
        for name in ("model_path", "test_data_file")
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
    """Validate public model types, domains, ranges, and object-policy joins."""

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
    if isinstance(config, TrainingConfig):
        enable_oversampling = config.sampling.enable_oversampling
        neighbor_count = config.sampling.neighbor_count
        neighbor_pool_size = config.sampling.neighbor_pool_size
    else:
        enable_oversampling = config.enable_oversampling
        neighbor_count = config.neighbor_count
        neighbor_pool_size = config.neighbor_pool_size
    if enable_oversampling and config.model.gridsize > 1:
        pool_size = (
            neighbor_count
            if neighbor_pool_size is None
            else neighbor_pool_size
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

    if config.tf_loss.realspace_mae_weight > 0 and config.tf_loss.realspace_weight <= 0:
        raise ValueError(
            "realspace_mae_weight requires positive realspace_weight"
        )

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
    """Validate requirements that are needed to begin a training run."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if config.data.train_data_file is None:
        raise ValueError("train_data_file is required for runnable training")
    if not config.data.train_data_file.exists():
        raise ValueError(
            f"train_data_file must exist: {config.data.train_data_file}"
        )
    if not config.data.train_data_file.is_file():
        raise ValueError(
            f"train_data_file must be a regular file: {config.data.train_data_file}"
        )
    if not os.access(config.data.train_data_file, os.R_OK):
        raise ValueError(
            f"train_data_file must be readable: {config.data.train_data_file}"
        )
    if config.nepochs <= 0:
        raise ValueError(f"nepochs must be positive, got {config.nepochs!r}")
    if config.batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {config.batch_size!r}"
        )
    if config.data.nphotons <= 0:
        raise ValueError(
            f"nphotons must be positive, got {config.data.nphotons!r}"
        )
    if config.sampling.n_groups is None or config.sampling.n_groups <= 0:
        raise ValueError(
            f"n_groups must be positive, got {config.sampling.n_groups!r}"
        )
    if config.sampling.n_subsample is not None and config.sampling.n_subsample <= 0:
        raise ValueError(
            f"n_subsample must be positive when set, "
            f"got {config.sampling.n_subsample!r}"
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
            "model_path must be a directory containing wts.h5.zip, "
            f"got {model_path}"
        )

    # Both supported backend loaders consume the established bundle layout:
    # a model directory containing the wts.h5.zip archive.
    model_archive = model_path / "wts.h5.zip"
    if not model_archive.exists():
        raise ValueError(
            f"{config.backend} model_path must contain wts.h5.zip: "
            f"{model_archive}"
        )
    if not model_archive.is_file():
        raise ValueError(
            f"model archive must be a regular file: {model_archive}"
        )
    if not os.access(model_archive, os.R_OK):
        raise ValueError(f"model archive must be readable: {model_archive}")

    test_data_path = config.test_data_file
    if not test_data_path.exists():
        raise ValueError(
            f"test_data_file must exist: {test_data_path}"
        )
    if not test_data_path.is_file():
        raise ValueError(
            f"test_data_file must be a regular file: {test_data_path}"
        )
    if not os.access(test_data_path, os.R_OK):
        raise ValueError(
            f"test_data_file must be readable: {test_data_path}"
        )


__all__ = [
    "resolve_inference_config",
    "resolve_training_config",
    "validate_inference_config_structure",
    "validate_inference_resources",
    "validate_model_config_structure",
    "validate_runnable_training_config",
    "validate_training_config_structure",
]

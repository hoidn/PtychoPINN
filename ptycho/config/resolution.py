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

# Flat names accepted from YAML / CLI for training config.  TrainingConfig now
# groups several sub-configs (DataConfig, SamplingConfig, …) under nested fields,
# but callers still supply flat keys, so we enumerate every accepted flat name
# explicitly and map them to the nested structure in resolve_training_config.
_TRAINING_INPUT_NAMES = frozenset({
    # Direct TrainingConfig fields
    "batch_size", "nepochs", "positions_provided", "probe_trainable",
    "intensity_scale_trainable", "output_dir", "backend",
    # DataConfig (flat equivalents)
    "train_data_file", "test_data_file", "nphotons",
    # SamplingConfig
    "n_groups", "n_images", "n_subsample", "subsample_seed",
    "neighbor_count", "enable_oversampling", "neighbor_pool_size", "sequential_sampling",
    # LossConfig
    "torch_loss_mode", "torch_mae_pred_l2_match_target",
    # TFLossConfig
    "mae_weight", "nll_weight", "realspace_mae_weight", "realspace_weight",
    # GradientClipConfig (remapped names)
    "gradient_clip_val", "gradient_clip_algorithm",
    # OptimizerConfig (remapped names)
    "optimizer", "weight_decay", "momentum", "adam_beta1", "adam_beta2",
    # SchedulerConfig (remapped names)
    "scheduler", "lr_warmup_epochs", "lr_min_ratio",
    "plateau_factor", "plateau_patience", "plateau_min_lr", "plateau_threshold",
})

_INFERENCE_INPUT_NAMES = frozenset(f.name for f in fields(InferenceConfig)) - {"model"}

assert _MODEL_INPUT_NAMES.isdisjoint(_TRAINING_INPUT_NAMES)
assert _MODEL_INPUT_NAMES.isdisjoint(_INFERENCE_INPUT_NAMES)


_N_IMAGES_DEPRECATION_MESSAGE = (
    "Parameter 'n_images' is deprecated and will be removed in a future "
    "version. Use 'n_groups' instead, which always means the number of "
    "groups regardless of gridsize."
)


_MODEL_CONFIG_ADAPTER = TypeAdapter(ModelConfig)
_TRAINING_CONFIG_ADAPTER = TypeAdapter(TrainingConfig)
_INFERENCE_CONFIG_ADAPTER = TypeAdapter(InferenceConfig)


def _flatten_to_nested_training_values(flat: dict[str, Any]) -> dict[str, Any]:
    """Map flat CLI/YAML training keys into TrainingConfig's nested structure."""
    nested: dict[str, Any] = {}
    data: dict[str, Any] = {}
    sampling: dict[str, Any] = {}
    loss: dict[str, Any] = {}
    tf_loss: dict[str, Any] = {}
    gradient_clip: dict[str, Any] = {}
    optimizer: dict[str, Any] = {}
    optimizer_sgd: dict[str, Any] = {}
    optimizer_adam: dict[str, Any] = {}
    scheduler: dict[str, Any] = {}

    for key, value in flat.items():
        if key in ("train_data_file", "test_data_file", "nphotons"):
            data[key] = value
        elif key in ("n_groups", "n_images", "n_subsample", "subsample_seed",
                     "neighbor_count", "enable_oversampling", "neighbor_pool_size",
                     "sequential_sampling"):
            sampling[key] = value
        elif key in ("torch_loss_mode", "torch_mae_pred_l2_match_target"):
            loss[key] = value
        elif key in ("mae_weight", "nll_weight", "realspace_mae_weight", "realspace_weight"):
            tf_loss[key] = value
        elif key == "gradient_clip_val":
            gradient_clip["val"] = value
        elif key == "gradient_clip_algorithm":
            gradient_clip["algorithm"] = value
        elif key == "optimizer":
            optimizer["algorithm"] = value
        elif key == "weight_decay":
            optimizer["weight_decay"] = value
        elif key == "momentum":
            optimizer_sgd["momentum"] = value
        elif key == "adam_beta1":
            optimizer_adam["beta1"] = value
        elif key == "adam_beta2":
            optimizer_adam["beta2"] = value
        elif key == "scheduler":
            scheduler["kind"] = value
        elif key in ("lr_warmup_epochs", "lr_min_ratio", "plateau_factor",
                     "plateau_patience", "plateau_min_lr", "plateau_threshold"):
            scheduler[key] = value
        else:
            nested[key] = value

    if data:
        nested["data"] = data
    if sampling:
        nested["sampling"] = sampling
    if loss:
        nested["loss"] = loss
    if tf_loss:
        nested["tf_loss"] = tf_loss
    if gradient_clip:
        nested["gradient_clip"] = gradient_clip
    if optimizer_sgd:
        optimizer["sgd"] = optimizer_sgd
    if optimizer_adam:
        optimizer["adam"] = optimizer_adam
    if optimizer:
        nested["optimizer"] = optimizer
    if scheduler:
        nested["scheduler"] = scheduler

    return nested


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
    if "n_images" not in resolved:
        return resolved, False

    legacy = resolved["n_images"]
    canonical = resolved.get("n_groups")
    if (
        canonical is not None
        and legacy is not None
        and canonical != legacy
    ):
        raise ValueError(
            f"{source} field 'n_images' conflicts with canonical 'n_groups'"
        )
    if canonical is None:
        resolved["n_groups"] = legacy
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
    file_mapping: Mapping[str, Any] | None,
    explicit_cli_patch: Mapping[str, Any] | None,
) -> TrainingConfig:
    """Resolve file and explicitly supplied CLI values into a fresh config."""

    file_model, file_training = _normalize_public_source(
        {} if file_mapping is None else file_mapping,
        source="file",
        workflow_names=_TRAINING_INPUT_NAMES,
    )
    cli_model, cli_training = _normalize_public_source(
        {} if explicit_cli_patch is None else explicit_cli_patch,
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

    training_values = dict(file_training)
    training_values.update(cli_training)
    model_values = dict(file_model)
    model_values.update(cli_model)
    nested_training = _flatten_to_nested_training_values(training_values)
    candidate = _validate_public_structure(
        _TRAINING_CONFIG_ADAPTER,
        {"model": model_values, **nested_training},
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
    # TrainingConfig nests sampling fields; InferenceConfig keeps them flat.
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

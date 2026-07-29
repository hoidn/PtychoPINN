"""Pure source resolution for the public configuration dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math
import os
from pathlib import Path
from typing import Any
import warnings

from .config import (
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    resolve_model_object_policy,
)


# Family ownership is explicit. The assertions make adding a dataclass field
# without assigning it to a source family fail at import time.
_MODEL_INPUT_NAMES = frozenset(
    {
        "N",
        "gridsize",
        "n_filters_scale",
        "model_type",
        "architecture",
        "fno_modes",
        "fno_width",
        "fno_blocks",
        "fno_cnn_blocks",
        "learned_input_channels",
        "max_hidden_channels",
        "resnet_width",
        "fno_input_transform",
        "generator_output_mode",
        "amp_activation",
        "object_big",
        "object_layout",
        "training_canvas",
        "training_patch_weighting",
        "probe_big",
        "probe_mask",
        "probe_mask_sigma",
        "probe_mask_diameter",
        "pad_object",
        "probe_scale",
        "gaussian_smoothing_sigma",
    }
)

_TRAINING_INPUT_NAMES = frozenset(
    {
        "train_data_file",
        "test_data_file",
        "batch_size",
        "nepochs",
        "mae_weight",
        "nll_weight",
        "realspace_mae_weight",
        "realspace_weight",
        "nphotons",
        "n_groups",
        "n_images",
        "n_subsample",
        "subsample_seed",
        "neighbor_count",
        "enable_oversampling",
        "neighbor_pool_size",
        "positions_provided",
        "probe_trainable",
        "intensity_scale_trainable",
        "output_dir",
        "sequential_sampling",
        "backend",
        "torch_loss_mode",
        "torch_mae_pred_l2_match_target",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "optimizer",
        "momentum",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "scheduler",
        "lr_warmup_epochs",
        "lr_min_ratio",
        "plateau_factor",
        "plateau_patience",
        "plateau_min_lr",
        "plateau_threshold",
    }
)

_INFERENCE_INPUT_NAMES = frozenset(
    {
        "model_path",
        "test_data_file",
        "n_groups",
        "n_images",
        "n_subsample",
        "subsample_seed",
        "neighbor_count",
        "enable_oversampling",
        "neighbor_pool_size",
        "debug",
        "output_dir",
        "backend",
    }
)

assert _MODEL_INPUT_NAMES == frozenset(item.name for item in fields(ModelConfig))
assert _TRAINING_INPUT_NAMES == frozenset(
    item.name for item in fields(TrainingConfig) if item.name != "model"
)
assert _INFERENCE_INPUT_NAMES == frozenset(
    item.name for item in fields(InferenceConfig) if item.name != "model"
)
assert _MODEL_INPUT_NAMES.isdisjoint(_TRAINING_INPUT_NAMES)
assert _MODEL_INPUT_NAMES.isdisjoint(_INFERENCE_INPUT_NAMES)


_MODEL_ARCHITECTURES = frozenset(
    {
        "cnn",
        "ffno",
        "fno",
        "hybrid",
        "stable_hybrid",
        "fno_vanilla",
        "neuralop_uno",
        "hybrid_resnet",
        "hybrid_resnet_ffno_ptychoblock_encoder",
        "hybrid_resnet_ptychoblock_ffno_encoder",
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
        "hybrid_resnet_convnext_bottleneck",
    }
)
_RESNET_SHELL_ARCHITECTURES = frozenset(
    {
        "hybrid_resnet",
        "hybrid_resnet_ffno_ptychoblock_encoder",
        "hybrid_resnet_ptychoblock_ffno_encoder",
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
        "hybrid_resnet_convnext_bottleneck",
    }
)
_MODEL_TYPES = frozenset({"pinn", "supervised"})
_FNO_INPUT_TRANSFORMS = frozenset({"none", "sqrt", "log1p", "instancenorm"})
_GENERATOR_OUTPUT_MODES = frozenset({"real_imag", "amp_phase_logits", "amp_phase"})
_AMP_ACTIVATIONS = frozenset({"sigmoid", "swish", "softplus", "relu"})
_OBJECT_LAYOUTS = frozenset({"single_patch", "grouped_patches"})
_TRAINING_CANVASES = frozenset({"independent", "relative_overlap"})
_TRAINING_PATCH_WEIGHTINGS = frozenset({"central_mask", "uniform", "probe"})
_PUBLIC_BACKENDS = frozenset({"tensorflow", "pytorch"})
_TORCH_LOSS_MODES = frozenset({"poisson", "mae"})
_GRADIENT_CLIP_ALGORITHMS = frozenset({"norm", "value", "agc"})
_OPTIMIZERS = frozenset({"adam", "adamw", "sgd"})
_SCHEDULERS = frozenset({"Default", "Exponential", "WarmupCosine", "ReduceLROnPlateau"})

_TRAINING_PATH_NAMES = frozenset({"train_data_file", "test_data_file", "output_dir"})
_INFERENCE_PATH_NAMES = frozenset({"model_path", "test_data_file", "output_dir"})

_N_IMAGES_DEPRECATION_MESSAGE = (
    "Parameter 'n_images' is deprecated and will be removed in a future "
    "version. Use 'n_groups' instead, which always means the number of "
    "groups regardless of gridsize."
)


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
) -> tuple[dict[str, Any], bool]:
    resolved = dict(values)
    if "n_images" not in resolved:
        return resolved, False

    legacy = resolved["n_images"]
    canonical = resolved.get("n_groups")
    if (
        canonical is not None
        and legacy is not None
        and not _values_equal(canonical, legacy)
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


def _materialize_paths(
    values: Mapping[str, Any],
    *,
    path_names: frozenset[str],
) -> dict[str, Any]:
    materialized = dict(values)
    for name in path_names:
        if name not in materialized or materialized[name] is None:
            continue
        try:
            materialized[name] = Path(materialized[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must be path-like, got {materialized[name]!r}"
            ) from error
    return materialized


def _object_policy_backend(backend: Any) -> str:
    _require_literal("backend", backend, _PUBLIC_BACKENDS)
    return "torch" if backend == "pytorch" else "tensorflow"


def _construct_model(
    file_values: Mapping[str, Any],
    cli_values: Mapping[str, Any],
    *,
    backend: Any,
) -> tuple[ModelConfig, ModelConfig]:
    merged_values = dict(file_values)
    merged_values.update(cli_values)
    raw_model = ModelConfig(**merged_values)
    validate_model_config_structure(raw_model)
    resolved_model = resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(backend),
        warn_deprecated=False,
    )
    return raw_model, resolved_model


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
    )
    cli_training, cli_used_alias = _resolve_group_alias(
        cli_training,
        source="explicit CLI",
    )

    training_values = dict(file_training)
    training_values.update(cli_training)
    training_values = _materialize_paths(
        training_values,
        path_names=_TRAINING_PATH_NAMES,
    )
    backend = training_values.get("backend", "tensorflow")
    raw_model, model = _construct_model(file_model, cli_model, backend=backend)
    candidate = TrainingConfig(model=model, **training_values)
    validate_training_config_structure(candidate)
    resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(backend),
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
    inference_values = _materialize_paths(
        inference_values,
        path_names=_INFERENCE_PATH_NAMES,
    )
    backend = inference_values.get("backend", "tensorflow")
    raw_model, model = _construct_model(file_model, cli_model, backend=backend)
    candidate = InferenceConfig(model=model, **inference_values)
    validate_inference_config_structure(candidate)
    resolve_model_object_policy(
        raw_model,
        backend=_object_policy_backend(backend),
        warn_deprecated=True,
    )
    if file_used_alias or cli_used_alias:
        _warn_deprecated_group_alias()
    return candidate


def _require_exact_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    allowed: frozenset[int] | None = None,
) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact built-in integer, got {value!r}")
    if allowed is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")


def _require_optional_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
) -> None:
    if value is not None:
        _require_exact_int(name, value, minimum=minimum)


def _require_exact_bool(name: str, value: Any) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact built-in boolean, got {value!r}")


def _require_optional_bool(name: str, value: Any) -> None:
    if value is not None:
        _require_exact_bool(name, value)


def _require_literal(
    name: str,
    value: Any,
    choices: frozenset[str],
    *,
    optional: bool = False,
) -> None:
    if optional and value is None:
        return
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    optional: bool = False,
) -> None:
    if optional and value is None:
        return
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in number, got {value!r}")
    if minimum is not None:
        below = value < minimum if minimum_inclusive else value <= minimum
        if below:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}, got {value!r}")
    if maximum is not None:
        above = value > maximum if maximum_inclusive else value >= maximum
        if above:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}, got {value!r}")


def _require_path(name: str, value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, Path):
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be a pathlib.Path{suffix}, got {value!r}")


def validate_model_config_structure(config: ModelConfig) -> None:
    """Validate public model types, domains, ranges, and policy joins."""

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")

    _require_exact_int(
        "N",
        config.N,
        allowed=frozenset({64, 128, 256}),
    )
    for name in (
        "gridsize",
        "n_filters_scale",
        "fno_modes",
        "fno_width",
        "fno_blocks",
        "learned_input_channels",
    ):
        _require_exact_int(name, getattr(config, name), minimum=1)
    _require_exact_int(
        "fno_cnn_blocks",
        config.fno_cnn_blocks,
        minimum=0,
    )
    for name in ("max_hidden_channels", "resnet_width"):
        _require_optional_int(name, getattr(config, name), minimum=1)

    _require_literal("model_type", config.model_type, _MODEL_TYPES)
    _require_literal(
        "architecture",
        config.architecture,
        _MODEL_ARCHITECTURES,
    )
    _require_literal(
        "fno_input_transform",
        config.fno_input_transform,
        _FNO_INPUT_TRANSFORMS,
    )
    _require_literal(
        "generator_output_mode",
        config.generator_output_mode,
        _GENERATOR_OUTPUT_MODES,
    )
    _require_literal(
        "amp_activation",
        config.amp_activation,
        _AMP_ACTIVATIONS,
    )
    _require_optional_bool("object_big", config.object_big)
    _require_literal(
        "object_layout",
        config.object_layout,
        _OBJECT_LAYOUTS,
        optional=True,
    )
    _require_literal(
        "training_canvas",
        config.training_canvas,
        _TRAINING_CANVASES,
        optional=True,
    )
    _require_literal(
        "training_patch_weighting",
        config.training_patch_weighting,
        _TRAINING_PATCH_WEIGHTINGS,
        optional=True,
    )
    for name in ("probe_big", "probe_mask", "pad_object"):
        _require_exact_bool(name, getattr(config, name))

    _require_number("probe_mask_sigma", config.probe_mask_sigma, minimum=0)
    _require_number(
        "probe_mask_diameter",
        config.probe_mask_diameter,
        minimum=0,
        minimum_inclusive=False,
        optional=True,
    )
    _require_number(
        "probe_scale",
        config.probe_scale,
        minimum=0,
        minimum_inclusive=False,
    )
    _require_number(
        "gaussian_smoothing_sigma",
        config.gaussian_smoothing_sigma,
        minimum=0,
    )

    if config.architecture in _RESNET_SHELL_ARCHITECTURES:
        if config.fno_blocks < 3:
            raise ValueError(
                f"{config.architecture} requires fno_blocks >= 3 "
                f"(got fno_blocks={config.fno_blocks})"
            )
        if config.resnet_width is not None and config.resnet_width % 4 != 0:
            raise ValueError(
                "resnet_width must be divisible by 4 so the CycleGAN "
                f"upsamplers produce integer channels (got {config.resnet_width})"
            )

    resolve_model_object_policy(config, warn_deprecated=False)


def _validate_sampling_structure(
    config: TrainingConfig | InferenceConfig,
) -> None:
    for name in ("n_groups", "n_images", "n_subsample", "subsample_seed"):
        _require_optional_int(name, getattr(config, name), minimum=0)
    _require_exact_int("neighbor_count", config.neighbor_count, minimum=1)
    _require_exact_bool("enable_oversampling", config.enable_oversampling)
    _require_optional_int(
        "neighbor_pool_size",
        config.neighbor_pool_size,
        minimum=1,
    )

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
    validate_model_config_structure(config.model)

    _require_path("train_data_file", config.train_data_file, optional=True)
    _require_path("test_data_file", config.test_data_file, optional=True)
    _require_path("output_dir", config.output_dir)
    _require_exact_int("batch_size", config.batch_size, minimum=1)
    if config.batch_size & (config.batch_size - 1):
        raise ValueError(
            f"batch_size must be a positive power of 2, got {config.batch_size!r}"
        )
    _require_exact_int("nepochs", config.nepochs, minimum=0)
    for name in (
        "mae_weight",
        "nll_weight",
        "realspace_mae_weight",
        "realspace_weight",
    ):
        _require_number(
            name,
            getattr(config, name),
            minimum=0,
            maximum=1,
        )
    _require_number("nphotons", config.nphotons, minimum=0)
    _validate_sampling_structure(config)

    for name in (
        "positions_provided",
        "probe_trainable",
        "intensity_scale_trainable",
        "sequential_sampling",
        "torch_mae_pred_l2_match_target",
    ):
        _require_exact_bool(name, getattr(config, name))
    _require_literal("backend", config.backend, _PUBLIC_BACKENDS)
    _require_literal(
        "torch_loss_mode",
        config.torch_loss_mode,
        _TORCH_LOSS_MODES,
    )
    _require_number(
        "gradient_clip_val",
        config.gradient_clip_val,
        minimum=0,
        optional=True,
    )
    _require_literal(
        "gradient_clip_algorithm",
        config.gradient_clip_algorithm,
        _GRADIENT_CLIP_ALGORITHMS,
    )
    _require_literal("optimizer", config.optimizer, _OPTIMIZERS)
    _require_number("momentum", config.momentum, minimum=0, maximum=1)
    _require_number("weight_decay", config.weight_decay, minimum=0)
    for name in ("adam_beta1", "adam_beta2"):
        _require_number(
            name,
            getattr(config, name),
            minimum=0,
            maximum=1,
            maximum_inclusive=False,
        )
    _require_literal("scheduler", config.scheduler, _SCHEDULERS)
    _require_exact_int(
        "lr_warmup_epochs",
        config.lr_warmup_epochs,
        minimum=0,
    )
    _require_number(
        "lr_min_ratio",
        config.lr_min_ratio,
        minimum=0,
        maximum=1,
    )
    _require_number(
        "plateau_factor",
        config.plateau_factor,
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    _require_exact_int(
        "plateau_patience",
        config.plateau_patience,
        minimum=0,
    )
    _require_number("plateau_min_lr", config.plateau_min_lr, minimum=0)
    _require_number(
        "plateau_threshold",
        config.plateau_threshold,
        minimum=0,
    )

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
    validate_model_config_structure(config.model)

    _require_path("model_path", config.model_path)
    _require_path("test_data_file", config.test_data_file)
    _require_path("output_dir", config.output_dir)
    _validate_sampling_structure(config)
    _require_exact_bool("debug", config.debug)
    _require_literal("backend", config.backend, _PUBLIC_BACKENDS)

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
    if config.n_groups is None or config.n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {config.n_groups!r}")
    if config.n_subsample is not None and config.n_subsample <= 0:
        raise ValueError(
            f"n_subsample must be positive when set, got {config.n_subsample!r}"
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

"""Versioned configuration resolution for the public synthetic workflow.

The workflow exposes five user-authored namespaces.  Torch ``DataConfig`` is
materialized from those namespaces and persisted on the resolved record, but
is deliberately not accepted as another input namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import (
    MISSING,
    dataclass,
    field,
    fields,
    is_dataclass,
    make_dataclass,
    replace,
)
import hashlib
import json
import math
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import (
    BeforeValidator,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    with_config,
)

from ptycho.config.config import (
    SimulationConfig,
    simulation_config_from_mapping,
    simulation_config_to_dict,
    validate_simulation_config,
)
from ptycho.config.strict_types import (
    _require_exact_int,
    _require_exact_str,
    _StrictBool,
    _StrictFiniteNonNegativeNumber,
    _StrictFiniteNumber,
    _StrictFinitePositiveNumber,
    _StrictHalfOpenUnitNumber,
    _StrictNonNegativeInt,
    _StrictOpenUnitNumber,
    _StrictPositiveInt,
)
from ptycho_torch.config_params import DataConfig, ModelConfig as TorchModelConfig


__all__ = [
    "SyntheticSimulationConfig",
    "SyntheticModelConfig",
    "SyntheticTrainingConfig",
    "SyntheticInferenceConfig",
    "SyntheticWorkflowConfig",
    "ResolvedDataConfig",
    "ResolvedSyntheticWorkflow",
    "UNSET",
    "resolve_synthetic_workflow",
    "materialize_data_config",
    "synthetic_workflow_to_dict",
    "synthetic_workflow_sha256",
]


_PROFILE_NAME = "synthetic-lines"
_RECIPE_VERSION = "synthetic-lines-v1"
_SCHEMA_VERSION = "synthetic-workflow-v1"


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _UnsetType()

_CONFIG = ConfigDict(
    extra="forbid",
    revalidate_instances="always",
    validate_default=True,
)

_StrictText = Annotated[str, BeforeValidator(_require_exact_str)]
_LIVE_MODEL_TYPE_HINTS = get_type_hints(TorchModelConfig)
_LiveArchitecture = _LIVE_MODEL_TYPE_HINTS["architecture"]
_Architecture = Annotated[
    _LiveArchitecture,
    BeforeValidator(_require_exact_str),
]
_ScaleContractVersion = Annotated[
    Literal["legacy_v1"],
    BeforeValidator(_require_exact_str),
]
_MeasurementDomain = Annotated[
    Literal["normalized_amplitude"],
    BeforeValidator(_require_exact_str),
]
_PhysicsForwardMode = Annotated[
    Literal["amplitude"],
    BeforeValidator(_require_exact_str),
]
_SupportedN = Annotated[
    Literal[64, 128, 256],
    BeforeValidator(_require_exact_int),
]


@with_config(_CONFIG)
@dataclass(frozen=True)
class SyntheticSimulationConfig:
    """The two complete split recipes and their shared semantic identity."""

    train: SimulationConfig
    test: SimulationConfig
    scale_contract_version: _ScaleContractVersion = "legacy_v1"
    measurement_domain: _MeasurementDomain = "normalized_amplitude"
    object_recipe: _StrictText = "lines-object-v1"
    shared_object: _StrictBool = True


_GainProvenance = Annotated[
    Literal[
        "pending_training_split_derivation",
        "explicit",
    ],
    BeforeValidator(_require_exact_str),
]


def _frozen_snapshot_type(
    record_name: str,
    live_type: type[Any],
    *,
    annotation_overrides: Mapping[str, Any] | None = None,
    extra_definitions: tuple[tuple[str, Any, Any], ...] = (),
    doc: str,
) -> type[Any]:
    """Project a live dataclass field surface into a frozen identity record."""

    annotation_overrides = annotation_overrides or {}
    definitions: list[tuple[str, Any, Any]] = []
    for item in fields(live_type):
        annotation = annotation_overrides.get(item.name, item.type)
        if item.default is not MISSING:
            definition = field(default=item.default)
        elif item.default_factory is not MISSING:
            definition = field(default_factory=item.default_factory)
        else:
            definition = field()
        definitions.append((item.name, annotation, definition))
    definitions.extend(extra_definitions)
    record_type = make_dataclass(
        record_name,
        definitions,
        frozen=True,
        namespace={"__doc__": doc},
    )
    record_type.__module__ = __name__
    return with_config(_CONFIG)(record_type)


SyntheticModelConfig = _frozen_snapshot_type(
    "SyntheticModelConfig",
    TorchModelConfig,
    annotation_overrides={
        "amplitude_physics_gain": _StrictFinitePositiveNumber | None,
    },
    extra_definitions=(
        (
            "amplitude_physics_gain_provenance",
            _GainProvenance,
            field(default="pending_training_split_derivation"),
        ),
    ),
    doc="Immutable semantic snapshot of every live Torch ModelConfig field.",
)
ResolvedDataConfig = _frozen_snapshot_type(
    "ResolvedDataConfig",
    DataConfig,
    doc="Immutable semantic snapshot of every live Torch DataConfig field.",
)


@with_config(_CONFIG)
@dataclass(frozen=True)
class SyntheticTrainingConfig:
    """Optimization plus raw-selection/grouping policy."""

    train_raw_selection: _StrictPositiveInt = 4096
    training_groups: _StrictPositiveInt = 1024
    validation_groups: _StrictPositiveInt = 1024
    neighbor_count: _StrictPositiveInt = 4
    neighbor_pool_size: _StrictPositiveInt = 4
    enable_oversampling: _StrictBool = False
    sequential_sampling: _StrictBool = False
    subsample_seed: _StrictNonNegativeInt | None = None
    epochs: _StrictPositiveInt = 50
    batch_size: _StrictPositiveInt = 16
    framework: Annotated[
        Literal["Default", "Lightning"],
        BeforeValidator(_require_exact_str),
    ] = "Lightning"
    orchestrator: Annotated[
        Literal["Mlflow", "Lightning"],
        BeforeValidator(_require_exact_str),
    ] = "Lightning"
    learning_rate: _StrictFinitePositiveNumber = 2e-4
    optimizer: Annotated[
        Literal["adam", "adamw", "sgd"],
        BeforeValidator(_require_exact_str),
    ] = "adam"
    momentum: _StrictFiniteNonNegativeNumber = 0.9
    weight_decay: _StrictFiniteNonNegativeNumber = 0.0
    adam_beta1: _StrictHalfOpenUnitNumber = 0.9
    adam_beta2: _StrictHalfOpenUnitNumber = 0.999
    scheduler: Annotated[
        Literal[
            "Default",
            "Exponential",
            "MultiStage",
            "Adaptive",
            "WarmupCosine",
            "ReduceLROnPlateau",
        ],
        BeforeValidator(_require_exact_str),
    ] = "ReduceLROnPlateau"
    lr_warmup_epochs: _StrictNonNegativeInt = 0
    lr_min_ratio: _StrictFiniteNonNegativeNumber = 0.1
    plateau_factor: _StrictOpenUnitNumber = 0.5
    plateau_patience: _StrictNonNegativeInt = 2
    plateau_min_lr: _StrictFiniteNonNegativeNumber = 1e-4
    plateau_threshold: _StrictFiniteNonNegativeNumber = 0.0
    accum_steps: _StrictPositiveInt = 1
    gradient_clip_val: _StrictFiniteNonNegativeNumber | None = None
    gradient_clip_algorithm: Annotated[
        Literal["norm", "value", "agc"],
        BeforeValidator(_require_exact_str),
    ] = "norm"
    epochs_fine_tune: _StrictNonNegativeInt = 0
    fine_tune_gamma: _StrictFiniteNonNegativeNumber = 0.1
    stage_1_epochs: _StrictNonNegativeInt = 0
    stage_2_epochs: _StrictNonNegativeInt = 0
    stage_3_epochs: _StrictNonNegativeInt = 0
    physics_weight_schedule: _StrictText = "cosine"
    stage_3_lr_factor: _StrictFiniteNonNegativeNumber = 0.1
    torch_loss_mode: Annotated[
        Literal["poisson", "mae"],
        BeforeValidator(_require_exact_str),
    ] = "mae"
    torch_mae_pred_l2_match_target: _StrictBool = True
    nll: _StrictBool = False


@with_config(_CONFIG)
@dataclass(frozen=True)
class SyntheticInferenceConfig:
    """Reconstruction-only policy; never overwrites persisted DataConfig."""

    middle_trim: _StrictNonNegativeInt = 32
    batch_size: _StrictPositiveInt = 16
    experiment_number: _StrictNonNegativeInt = 0
    pad_eval: _StrictBool = True
    window: _StrictNonNegativeInt = 20
    reconstruction_method: Annotated[
        Literal["barycentric"],
        BeforeValidator(_require_exact_str),
    ] = "barycentric"
    patch_weighting: Annotated[
        Literal["uniform", "probe"],
        BeforeValidator(_require_exact_str),
    ] = "probe"
    groups_per_center: _StrictPositiveInt = 1
    varpro_scaling: _StrictBool = True
    log_patch_stats: _StrictBool = False
    patch_stats_limit: _StrictPositiveInt | None = None


@with_config(_CONFIG)
@dataclass(frozen=True)
class SyntheticWorkflowConfig:
    """Stage selection, output identity, and unresolved runtime request."""

    stages: tuple[
        Literal["simulate", "train", "reconstruct", "evaluate"], ...
    ] = ("simulate", "train", "reconstruct", "evaluate")
    output_root: Path = Path("synthetic_outputs")
    reuse_complete_artifacts: _StrictBool = True
    accelerator: Annotated[
        Literal["auto", "cpu", "gpu", "cuda", "mps"],
        BeforeValidator(_require_exact_str),
    ] = "auto"
    devices: _StrictPositiveInt = 1
    strategy: _StrictText = "auto"
    deterministic: _StrictBool = True
    precision: Annotated[
        Literal["32-true", "16-mixed", "bf16-mixed"],
        BeforeValidator(_require_exact_str),
    ] = "32-true"
    num_workers: _StrictNonNegativeInt = 0
    pin_memory: _StrictBool = False
    persistent_workers: _StrictBool = False
    prefetch_factor: _StrictPositiveInt | None = None
    enable_progress_bar: _StrictBool = False
    enable_checkpointing: _StrictBool = True
    checkpoint_save_top_k: _StrictNonNegativeInt = 1
    checkpoint_monitor_metric: _StrictText = "val_loss"
    checkpoint_mode: Annotated[
        Literal["min", "max"],
        BeforeValidator(_require_exact_str),
    ] = "min"
    early_stop_patience: _StrictNonNegativeInt = 100
    logger_backend: Annotated[
        Literal["csv", "tensorboard", "mlflow"],
        BeforeValidator(_require_exact_str),
    ] | None = "csv"
    recon_log_every_n_epochs: _StrictPositiveInt | None = None
    recon_log_num_patches: _StrictPositiveInt = 4
    recon_log_fixed_indices: tuple[_StrictNonNegativeInt, ...] | None = None
    recon_log_stitch: _StrictBool = False
    recon_log_max_stitch_samples: _StrictPositiveInt | None = None


@dataclass(frozen=True)
class ResolvedSyntheticWorkflow:
    """Complete semantic snapshot including the derived persisted DataConfig."""

    schema_version: str
    profile: str
    recipe_version: str
    simulation: SyntheticSimulationConfig
    model: SyntheticModelConfig
    training: SyntheticTrainingConfig
    inference: SyntheticInferenceConfig
    workflow: SyntheticWorkflowConfig
    data: ResolvedDataConfig


_SIMULATION_NAMESPACE_ADAPTER = TypeAdapter(SyntheticSimulationConfig)
_TRAINING_ADAPTER = TypeAdapter(SyntheticTrainingConfig)
_INFERENCE_ADAPTER = TypeAdapter(SyntheticInferenceConfig)
_WORKFLOW_ADAPTER = TypeAdapter(SyntheticWorkflowConfig)
_N_ADAPTER = TypeAdapter(_SupportedN)
_GRID_SIZE_ADAPTER = TypeAdapter(_StrictPositiveInt)
_SEED_ADAPTER = TypeAdapter(_StrictNonNegativeInt | None)
_COUNT_ADAPTER = TypeAdapter(_StrictPositiveInt)


def _finite_numeric_annotation(annotation: Any) -> Any:
    """Replace live float annotations with the repository finite-number type."""

    if annotation is float:
        return _StrictFiniteNumber
    if get_origin(annotation) not in {Union, UnionType}:
        return annotation
    arguments = get_args(annotation)
    if float not in arguments:
        return annotation
    rewritten = [
        _StrictFiniteNumber if argument is float else argument
        for argument in arguments
    ]
    result = rewritten[0]
    for argument in rewritten[1:]:
        result = result | argument
    return result


_MODEL_VALIDATION_TYPES = {
    item.name: _finite_numeric_annotation(_LIVE_MODEL_TYPE_HINTS[item.name])
    for item in fields(TorchModelConfig)
}
_MODEL_VALIDATION_TYPES.update(
    {
        "architecture": _Architecture,
        "fno_modes": _StrictPositiveInt,
        "fno_width": _StrictPositiveInt,
        "fno_blocks": _StrictPositiveInt,
        "fno_cnn_blocks": _StrictNonNegativeInt,
        "learned_input_channels": _StrictPositiveInt,
        "max_hidden_channels": _StrictPositiveInt | None,
        "resnet_width": _StrictPositiveInt | None,
        "C_model": _StrictPositiveInt,
        "C_forward": _StrictPositiveInt,
        "object_big": _StrictBool | None,
        "probe_big": _StrictBool,
        "probe_mask": _StrictBool,
        "probe_mask_sigma": _StrictFiniteNonNegativeNumber,
        "probe_mask_diameter": _StrictFinitePositiveNumber | None,
        "physics_forward_mode": _PhysicsForwardMode,
        "rect_s1s2_trainable": _StrictBool,
        "amplitude_physics_gain": _StrictFinitePositiveNumber | None,
        "pad_object": _StrictBool,
        "gaussian_smoothing_sigma": _StrictFiniteNonNegativeNumber,
    }
)
_STRICT_FIELD_CONFIG = ConfigDict(strict=True, arbitrary_types_allowed=True)
_MODEL_FIELD_ADAPTERS = {
    name: TypeAdapter(annotation, config=_STRICT_FIELD_CONFIG)
    for name, annotation in _MODEL_VALIDATION_TYPES.items()
}
_GAIN_PROVENANCE_ADAPTER = TypeAdapter(_GainProvenance)

_LIVE_DATA_TYPE_HINTS = get_type_hints(DataConfig)
_DATA_VALIDATION_TYPES = {
    item.name: _finite_numeric_annotation(_LIVE_DATA_TYPE_HINTS[item.name])
    for item in fields(DataConfig)
}
_DATA_VALIDATION_TYPES.update(
    {
        "nphotons": _StrictFinitePositiveNumber,
        "N": _StrictPositiveInt,
        "C": _StrictPositiveInt,
        "K": _StrictPositiveInt,
        "K_quadrant": _StrictPositiveInt,
        "n_subsample": _StrictPositiveInt,
        "subsample_seed": _StrictNonNegativeInt | None,
        "grid_size": tuple[_StrictPositiveInt, _StrictPositiveInt],
        "min_neighbor_distance": _StrictFiniteNonNegativeNumber,
        "max_neighbor_distance": _StrictFinitePositiveNumber,
        "probe_scale": _StrictFinitePositiveNumber,
        "probe_normalize": _StrictBool,
        "phase_subtraction": _StrictBool,
        "x_bounds": tuple[
            _StrictFiniteNumber,
            _StrictFiniteNumber,
        ],
        "y_bounds": tuple[
            _StrictFiniteNumber,
            _StrictFiniteNumber,
        ],
    }
)
_DATA_FIELD_ADAPTERS = {
    name: TypeAdapter(annotation, config=_STRICT_FIELD_CONFIG)
    for name, annotation in _DATA_VALIDATION_TYPES.items()
}


def _torch_model_defaults() -> dict[str, Any]:
    defaults = TorchModelConfig()
    return {item.name: getattr(defaults, item.name) for item in fields(defaults)}


_PROFILE_VALUES: dict[str, dict[str, Any]] = {
    "simulation": {
        "N": 128,
        "gridsize": 1,
        "seed": 3,
        "train_patterns": 4096,
        "test_patterns": 1024,
        "scale_contract_version": "legacy_v1",
        "measurement_domain": "normalized_amplitude",
        "object_recipe": "lines-object-v1",
        "shared_object": True,
        "probe": {
            "source": "ideal",
            "source_path": None,
            "transform_pipeline": None,
            "mask_diameter": None,
            "ideal_scale": 0.7,
        },
        "object": {
            "kind": "lines",
            "image_size": (392, 392),
            "objects_per_probe": 1,
            "set_phi": True,
        },
        "scan": {
            "kind": "nongrid",
            "grid_size": None,
            "offset": 4,
            "outer_offset_train": 8,
            "outer_offset_test": 20,
            "buffer": 64,
        },
        "detector": {
            "photons_per_pattern": 1e9,
            "beamstop_diameter": None,
        },
    },
    "model": {
        **_torch_model_defaults(),
        "mode": "Unsupervised",
        "architecture": "cnn",
        "fno_modes": 12,
        "fno_width": 32,
        "fno_blocks": 4,
        "fno_cnn_blocks": 2,
        "learned_input_channels": 1,
        "fno_input_transform": "none",
        "max_hidden_channels": None,
        "resnet_width": None,
        "generator_output_mode": "real_imag",
        "C_model": None,
        "C_forward": None,
        "object_big": None,
        "object_layout": None,
        "training_canvas": None,
        "training_patch_weighting": None,
        "probe_big": None,
        "probe_mask": False,
        "probe_mask_sigma": 1.0,
        "probe_mask_diameter": None,
        "physics_forward_mode": "amplitude",
        "rect_s1s2_trainable": True,
        "rect_s1s2_init": "ones",
        "amplitude_physics_gain": None,
        "amplitude_physics_gain_provenance": (
            "pending_training_split_derivation"
        ),
        "pad_object": True,
        "gaussian_smoothing_sigma": 0.0,
        "loss_function": "MAE",
    },
    "training": {
        item.name: getattr(SyntheticTrainingConfig(), item.name)
        for item in fields(SyntheticTrainingConfig)
    },
    "inference": {
        item.name: getattr(SyntheticInferenceConfig(), item.name)
        for item in fields(SyntheticInferenceConfig)
    },
    "workflow": {
        item.name: getattr(SyntheticWorkflowConfig(), item.name)
        for item in fields(SyntheticWorkflowConfig)
    },
}


_FLAT_ALIASES: dict[str, tuple[str, ...]] = {
    "N": ("simulation", "N"),
    "gridsize": ("simulation", "gridsize"),
    "seed": ("simulation", "seed"),
    "train_patterns": ("simulation", "train_patterns"),
    "test_patterns": ("simulation", "test_patterns"),
    "ideal_scale": ("simulation", "probe", "ideal_scale"),
    "architecture": ("model", "architecture"),
    "epochs": ("training", "epochs"),
    "batch_size": ("training", "batch_size"),
    "optimizer": ("training", "optimizer"),
    "learning_rate": ("training", "learning_rate"),
    "scheduler": ("training", "scheduler"),
    "train_raw_selection": ("training", "train_raw_selection"),
    "training_groups": ("training", "training_groups"),
    "validation_groups": ("training", "validation_groups"),
    "neighbor_count": ("training", "neighbor_count"),
    "neighbor_pool_size": ("training", "neighbor_pool_size"),
    "groups_per_center": ("inference", "groups_per_center"),
    "inference_batch_size": ("inference", "batch_size"),
    "accelerator": ("workflow", "accelerator"),
    "devices": ("workflow", "devices"),
    "strategy": ("workflow", "strategy"),
    "precision": ("workflow", "precision"),
    "deterministic": ("workflow", "deterministic"),
    "num_workers": ("workflow", "num_workers"),
    "logger_backend": ("workflow", "logger_backend"),
    "checkpoint_save_top_k": ("workflow", "checkpoint_save_top_k"),
    "output_root": ("workflow", "output_root"),
}


def _values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


def _put_patch_value(
    target: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
    *,
    source: str,
) -> None:
    current = target
    for part in path[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"{source} field {'.'.join(path)} has conflicting declarations")
        current = existing
    leaf = path[-1]
    if leaf in current and not _values_equal(current[leaf], value):
        raise ValueError(
            f"{source} field {'.'.join(path)} has conflicting duplicate declarations"
        )
    current[leaf] = value


def _normalize_nested_patch(
    value: Any,
    template: Any,
    *,
    path: tuple[str, ...],
    source: str,
    omit_unset: bool,
    output: dict[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} field {'.'.join(path)} must be a mapping")
    unknown = set(value) - set(template)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(
            f"{source} configuration has unknown field(s) under "
            f"{'.'.join(path)}: {names}"
        )
    for name, item in value.items():
        child_path = (*path, name)
        if omit_unset and item is UNSET:
            continue
        if child_path == (
            "model",
            "amplitude_physics_gain_provenance",
        ):
            raise ValueError(
                f"{source} field model.amplitude_physics_gain_provenance "
                "is derived and cannot be supplied"
            )
        child_template = template[name]
        if isinstance(child_template, Mapping):
            _normalize_nested_patch(
                item,
                child_template,
                path=child_path,
                source=source,
                omit_unset=omit_unset,
                output=output,
            )
        else:
            _put_patch_value(output, child_path, item, source=source)


def _normalize_source(
    values: Mapping[str, Any] | None,
    *,
    source: str,
    omit_unset: bool,
) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{source} configuration must be a mapping")
    output: dict[str, Any] = {}
    for name, value in values.items():
        if omit_unset and value is UNSET:
            continue
        if name in _PROFILE_VALUES:
            _normalize_nested_patch(
                value,
                _PROFILE_VALUES[name],
                path=(name,),
                source=source,
                omit_unset=omit_unset,
                output=output,
            )
            continue
        alias = _FLAT_ALIASES.get(name)
        if alias is None:
            raise ValueError(f"{source} configuration has unknown field {name!r}")
        _put_patch_value(output, alias, value, source=source)
    return output


def _merged_mapping(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in base.items():
        if isinstance(value, Mapping):
            result[name] = _merged_mapping(value, {})
        elif isinstance(value, list):
            result[name] = list(value)
        elif isinstance(value, tuple):
            result[name] = tuple(value)
        else:
            result[name] = value
    for name, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(name), Mapping):
            result[name] = _merged_mapping(result[name], value)
        else:
            result[name] = value
    return result


def _raise_adapter_error(error: ValidationError, *, root: str) -> None:
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


def _adapt(adapter: TypeAdapter, value: Any, *, root: str) -> Any:
    try:
        return adapter.validate_python(value)
    except ValidationError as error:
        _raise_adapter_error(error, root=root)


def _derived_grouping_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    import numpy as np

    grouping_sequence = np.random.SeedSequence(seed).spawn(7)[5]
    return int(grouping_sequence.generate_state(1, dtype=np.uint32)[0])


def _custom_probe_shape(source_path: str | Path) -> tuple[int, int]:
    """Load the same custom probe view used by generation and validate its shape."""

    import numpy as np

    from ptycho.metadata import MetadataManager

    source_data, _ = MetadataManager.load_with_metadata(str(source_path))
    if "probeGuess" not in source_data:
        raise KeyError(f"probeGuess missing from {source_path}")
    probe = np.asarray(source_data["probeGuess"], dtype=np.complex64).squeeze()
    if probe.ndim != 2 or probe.shape[0] != probe.shape[1]:
        raise ValueError(
            "simulation.probe.source_path probeGuess must be square after squeeze, "
            f"got shape {probe.shape}"
        )
    return probe.shape


def _resolve_simulation(values: Mapping[str, Any]) -> SyntheticSimulationConfig:
    N = _adapt(_N_ADAPTER, values["N"], root="simulation.N")
    gridsize = _adapt(
        _GRID_SIZE_ADAPTER,
        values["gridsize"],
        root="simulation.gridsize",
    )
    seed = _adapt(_SEED_ADAPTER, values["seed"], root="simulation.seed")
    train_patterns = _adapt(
        _COUNT_ADAPTER,
        values["train_patterns"],
        root="simulation.train_patterns",
    )
    test_patterns = _adapt(
        _COUNT_ADAPTER,
        values["test_patterns"],
        root="simulation.test_patterns",
    )

    probe = dict(values["probe"])
    if probe["source"] == "custom" and probe["source_path"] is None:
        raise ValueError(
            "simulation.probe.source_path is required when "
            "simulation.probe.source='custom'"
        )
    source_shape = (N, N)
    if probe["source"] == "custom":
        source_shape = _custom_probe_shape(probe["source_path"])
    if probe["transform_pipeline"] is None:
        if source_shape[0] == N:
            probe["transform_pipeline"] = f"smooth:0.5|pad_preserve:{N}"
        elif source_shape[0] < N:
            probe["transform_pipeline"] = f"pad_extrapolate:{N}|smooth:0.5"
        else:
            raise ValueError(
                "custom probe is larger than simulation.N; supply an explicit "
                "valid downsampling transform such as interp"
            )
    from ptycho.simulation.probe_transform import normalize_probe_transform_pipeline

    probe["transform_pipeline"], _ = normalize_probe_transform_pipeline(
        target_N=N,
        probe_shape=source_shape,
        probe_scale_mode="pipeline",
        probe_smoothing_sigma=0.0,
        probe_transform_pipeline=probe["transform_pipeline"],
    )

    expected_grid = (gridsize, gridsize)
    scan = dict(values["scan"])
    if scan["grid_size"] is None:
        scan["grid_size"] = expected_grid

    def split_config(pattern_count: int) -> SimulationConfig:
        return simulation_config_from_mapping(
            {
                "N": N,
                "seed": seed,
                "probe": probe,
                "object": {
                    **values["object"],
                    "diffractions_per_object": pattern_count,
                },
                "scan": {
                    **scan,
                    "train_groups": 1,
                    "test_groups": 1,
                },
                "detector": values["detector"],
            }
        )

    train = split_config(train_patterns)
    test = split_config(test_patterns)
    if train.scan.grid_size != expected_grid:
        raise ValueError(
            "simulation.scan.grid_size conflicts with simulation.gridsize: "
            f"expected {expected_grid}, got {train.scan.grid_size}"
        )
    return _adapt(
        _SIMULATION_NAMESPACE_ADAPTER,
        {
            "train": train,
            "test": test,
            "scale_contract_version": values["scale_contract_version"],
            "measurement_domain": values["measurement_domain"],
            "object_recipe": values["object_recipe"],
            "shared_object": values["shared_object"],
        },
        root="simulation",
    )


def _adapt_data(values: Mapping[str, Any]) -> ResolvedDataConfig:
    validated = {
        name: _adapt(adapter, values[name], root=f"data.{name}")
        for name, adapter in _DATA_FIELD_ADAPTERS.items()
    }
    expected_channels = validated["grid_size"][0] * validated["grid_size"][1]
    if validated["C"] != expected_channels:
        raise ValueError(
            "data.C conflicts with data.grid_size: "
            f"expected {expected_channels}, got {validated['C']}"
        )
    if validated["min_neighbor_distance"] > validated["max_neighbor_distance"]:
        raise ValueError(
            "data.min_neighbor_distance must be <= data.max_neighbor_distance"
        )
    for name in ("x_bounds", "y_bounds"):
        lower, upper = validated[name]
        if lower >= upper:
            raise ValueError(f"data.{name} must be strictly increasing")
    return ResolvedDataConfig(**validated)


def _snapshot_data_config(data: DataConfig) -> ResolvedDataConfig:
    if not isinstance(data, DataConfig):
        raise TypeError("data must be a DataConfig")
    return _adapt_data(
        {item.name: getattr(data, item.name) for item in fields(data)}
    )


def _adapt_model(values: Mapping[str, Any]) -> SyntheticModelConfig:
    validated = {
        name: _adapt(adapter, values[name], root=f"model.{name}")
        for name, adapter in _MODEL_FIELD_ADAPTERS.items()
    }
    provenance = _adapt(
        _GAIN_PROVENANCE_ADAPTER,
        values["amplitude_physics_gain_provenance"],
        root="model.amplitude_physics_gain_provenance",
    )
    if validated["probe_mask_tensor"] is not None:
        raise ValueError(
            "model.probe_mask_tensor is a legacy runtime tensor and cannot be "
            "part of the JSON workflow identity; use model.probe_mask controls"
        )
    try:
        TorchModelConfig(**validated)
    except (TypeError, ValueError) as error:
        message = str(error)
        field_name = next(
            (
                name
                for name in sorted(validated, key=len, reverse=True)
                if name in message
            ),
            None,
        )
        root = f"model.{field_name}" if field_name is not None else "model"
        raise ValueError(f"{root}: {message}") from error
    return SyntheticModelConfig(
        **validated,
        amplitude_physics_gain_provenance=provenance,
    )


def _resolve_model(
    values: Mapping[str, Any],
    *,
    C: int,
    N: int,
    gridsize: int,
) -> SyntheticModelConfig:
    candidate = dict(values)
    if gridsize == 1:
        expected = {
            "object_big": False,
            "object_layout": "single_patch",
            "training_canvas": "independent",
            "training_patch_weighting": "central_mask",
            "probe_big": False,
        }
    else:
        expected = {
            "object_big": True,
            "object_layout": "grouped_patches",
            "training_canvas": "relative_overlap",
            "training_patch_weighting": "probe",
            "probe_big": True,
        }
    expected.update({"C_model": C, "C_forward": C})
    for name, expected_value in expected.items():
        supplied = candidate[name]
        if supplied is not None and not _values_equal(supplied, expected_value):
            raise ValueError(
                f"model.{name} conflicts with derived gridsize geometry: "
                f"expected {expected_value!r}, got {supplied!r}"
            )
        candidate[name] = expected_value

    gain = candidate["amplitude_physics_gain"]
    if gain is None:
        candidate["amplitude_physics_gain_provenance"] = (
            "pending_training_split_derivation"
        )
    else:
        candidate["amplitude_physics_gain_provenance"] = "explicit"

    model = _adapt_model(candidate)
    if model.architecture == "neuralop_uno":
        if N != 128:
            raise ValueError(
                "simulation.N must be 128 when "
                "model.architecture='neuralop_uno'"
            )
        if gridsize != 1:
            raise ValueError(
                "simulation.gridsize must be 1 when "
                "model.architecture='neuralop_uno'"
            )
        if model.generator_output_mode != "real_imag":
            raise ValueError(
                "model.generator_output_mode must be 'real_imag' when "
                "model.architecture='neuralop_uno'"
            )
    if gridsize > 1 and model.architecture != "cnn":
        raise ValueError(
            f"model.architecture={model.architecture!r} does not support "
            f"gridsize={gridsize}; expected 'cnn'"
        )
    return model


def _validate_sampling(
    *,
    C: int,
    train_patterns: int,
    test_patterns: int,
    training: SyntheticTrainingConfig,
) -> None:
    if training.neighbor_count < C:
        raise ValueError(
            f"training.neighbor_count must be >= C={C}, got "
            f"{training.neighbor_count}"
        )
    if training.neighbor_count > training.train_raw_selection:
        raise ValueError(
            "training.train_raw_selection must be >= training.neighbor_count, "
            f"got {training.train_raw_selection} < {training.neighbor_count}"
        )
    if training.train_raw_selection > train_patterns:
        raise ValueError(
            "training.train_raw_selection must be <= simulation.train_patterns, "
            f"got {training.train_raw_selection} > {train_patterns}"
        )
    if test_patterns < C:
        raise ValueError(
            f"simulation.test_patterns must be >= C={C}, got {test_patterns}"
        )
    if training.training_groups > training.train_raw_selection:
        detail = " with oversampling disabled" if not training.enable_oversampling else ""
        raise ValueError(
            "training.training_groups must be <= training.train_raw_selection"
            f"{detail}, got {training.training_groups} > "
            f"{training.train_raw_selection}"
        )
    if training.validation_groups > test_patterns:
        raise ValueError(
            "training.validation_groups must be <= simulation.test_patterns, "
            f"got {training.validation_groups} > {test_patterns}"
        )
    if training.enable_oversampling and training.neighbor_pool_size < C:
        raise ValueError(
            f"training.neighbor_pool_size must be >= C={C} when "
            "training.enable_oversampling is true"
        )


def _validate_loss_identity(
    model: SyntheticModelConfig,
    training: SyntheticTrainingConfig,
) -> None:
    expected_loss, expected_nll = {
        "mae": ("MAE", False),
        "poisson": ("Poisson", True),
    }[training.torch_loss_mode]
    if model.loss_function != expected_loss:
        raise ValueError(
            f"model.loss_function must be {expected_loss!r} when "
            f"training.torch_loss_mode={training.torch_loss_mode!r}"
        )
    if training.nll is not expected_nll:
        raise ValueError(
            f"training.nll must be {expected_nll!r} when "
            f"training.torch_loss_mode={training.torch_loss_mode!r}"
        )


def _validate_stages(workflow: SyntheticWorkflowConfig) -> None:
    order = {name: index for index, name in enumerate((
        "simulate",
        "train",
        "reconstruct",
        "evaluate",
    ))}
    indices = [order[name] for name in workflow.stages]
    if not indices:
        raise ValueError("workflow.stages must contain at least one stage")
    if len(set(indices)) != len(indices):
        raise ValueError("workflow.stages must not contain duplicates")
    if indices != sorted(indices):
        raise ValueError("workflow.stages must follow workflow order")


def _record_values(record: Any) -> dict[str, Any]:
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("record must be a dataclass instance")
    return {item.name: getattr(record, item.name) for item in fields(record)}


def _derive_data_snapshot(
    simulation: SyntheticSimulationConfig,
    training: SyntheticTrainingConfig,
    *,
    C: int,
    gridsize: int,
) -> ResolvedDataConfig:
    return _snapshot_data_config(
        DataConfig(
            nphotons=float(simulation.train.detector.photons_per_pattern),
            scale_contract_version=simulation.scale_contract_version,
            measurement_domain=simulation.measurement_domain,
            N=simulation.train.N,
            C=C,
            K=training.neighbor_count,
            n_subsample=training.train_raw_selection,
            subsample_seed=training.subsample_seed,
            grid_size=(gridsize, gridsize),
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
            probe_scale=4.0,
            probe_normalize=True,
        )
    )


def _raise_first_record_difference(
    root: str,
    observed: Any,
    expected: Any,
) -> None:
    for item in fields(expected):
        observed_value = getattr(observed, item.name)
        expected_value = getattr(expected, item.name)
        field_path = f"{root}.{item.name}"
        if (
            is_dataclass(observed_value)
            and not isinstance(observed_value, type)
            and is_dataclass(expected_value)
            and not isinstance(expected_value, type)
            and type(observed_value) is type(expected_value)
        ):
            _raise_first_record_difference(
                field_path,
                observed_value,
                expected_value,
            )
            continue
        if not _values_equal(observed_value, expected_value):
            raise ValueError(
                f"{field_path} conflicts with resolved identity: "
                f"expected {expected_value!r}, got {observed_value!r}"
            )


def _validate_resolved_workflow(resolved: ResolvedSyntheticWorkflow) -> None:
    if not isinstance(resolved, ResolvedSyntheticWorkflow):
        raise TypeError("resolved must be a ResolvedSyntheticWorkflow")
    for name, expected_type in (
        ("simulation", SyntheticSimulationConfig),
        ("model", SyntheticModelConfig),
        ("training", SyntheticTrainingConfig),
        ("inference", SyntheticInferenceConfig),
        ("workflow", SyntheticWorkflowConfig),
        ("data", ResolvedDataConfig),
    ):
        observed = getattr(resolved, name)
        if type(observed) is not expected_type:
            raise ValueError(
                f"{name} must be a {expected_type.__name__}, "
                f"got {type(observed).__name__}"
            )
    for name, expected in (
        ("schema_version", _SCHEMA_VERSION),
        ("profile", _PROFILE_NAME),
        ("recipe_version", _RECIPE_VERSION),
    ):
        observed = getattr(resolved, name)
        if not _values_equal(observed, expected):
            raise ValueError(
                f"{name} must be {expected!r}, got {observed!r}"
            )

    simulation = _adapt(
        _SIMULATION_NAMESPACE_ADAPTER,
        _record_values(resolved.simulation),
        root="simulation",
    )
    _raise_first_record_difference(
        "simulation",
        resolved.simulation,
        simulation,
    )
    validate_simulation_config(simulation.train)
    validate_simulation_config(simulation.test)
    for split_name, split in (
        ("train", simulation.train),
        ("test", simulation.test),
    ):
        for group_name in ("train_groups", "test_groups"):
            observed = getattr(split.scan, group_name)
            if observed != 1:
                raise ValueError(
                    f"simulation.{split_name}.scan.{group_name} must be 1, "
                    f"got {observed}"
                )
    expected_test = replace(
        simulation.train,
        object=replace(
            simulation.train.object,
            diffractions_per_object=(
                simulation.test.object.diffractions_per_object
            ),
        ),
    )
    _raise_first_record_difference(
        "simulation.test",
        simulation.test,
        expected_test,
    )
    train_grid = simulation.train.scan.grid_size
    if train_grid[0] != train_grid[1]:
        raise ValueError("simulation.train.scan.grid_size must be square")
    gridsize = train_grid[0]
    C = gridsize**2

    model = _adapt_model(_record_values(resolved.model))
    _raise_first_record_difference("model", resolved.model, model)
    expected_model = _resolve_model(
        _record_values(model),
        C=C,
        N=simulation.train.N,
        gridsize=gridsize,
    )
    _raise_first_record_difference("model", model, expected_model)
    training = _adapt(
        _TRAINING_ADAPTER,
        _record_values(resolved.training),
        root="training",
    )
    _raise_first_record_difference("training", resolved.training, training)
    inference = _adapt(
        _INFERENCE_ADAPTER,
        _record_values(resolved.inference),
        root="inference",
    )
    _raise_first_record_difference("inference", resolved.inference, inference)
    workflow = _adapt(
        _WORKFLOW_ADAPTER,
        _record_values(resolved.workflow),
        root="workflow",
    )
    _raise_first_record_difference("workflow", resolved.workflow, workflow)
    _validate_stages(workflow)
    _validate_sampling(
        C=C,
        train_patterns=simulation.train.object.diffractions_per_object,
        test_patterns=simulation.test.object.diffractions_per_object,
        training=training,
    )
    _validate_loss_identity(model, training)

    data = _adapt_data(_record_values(resolved.data))
    _raise_first_record_difference("data", resolved.data, data)
    expected_data = _derive_data_snapshot(
        simulation,
        training,
        C=C,
        gridsize=gridsize,
    )
    _raise_first_record_difference("data", data, expected_data)


def resolve_synthetic_workflow(
    *,
    profile: str = _PROFILE_NAME,
    file_values: Mapping[str, Any] | None = None,
    cli_values: Mapping[str, Any] | None = None,
) -> ResolvedSyntheticWorkflow:
    """Resolve explicit CLI values over file values over a named profile."""

    if profile != _PROFILE_NAME:
        raise ValueError(
            f"unknown synthetic workflow profile {profile!r}; expected "
            f"{_PROFILE_NAME!r}"
        )
    file_patch = _normalize_source(
        file_values,
        source="file",
        omit_unset=False,
    )
    cli_patch = _normalize_source(
        cli_values,
        source="explicit CLI",
        omit_unset=True,
    )
    merged = _merged_mapping(_PROFILE_VALUES, file_patch)
    merged = _merged_mapping(merged, cli_patch)

    simulation = _resolve_simulation(merged["simulation"])
    gridsize = simulation.train.scan.grid_size[0]
    C = gridsize**2
    model = _resolve_model(
        merged["model"],
        C=C,
        N=simulation.train.N,
        gridsize=gridsize,
    )

    training_values = dict(merged["training"])
    if training_values["subsample_seed"] is None:
        training_values["subsample_seed"] = _derived_grouping_seed(
            simulation.train.seed
        )
    training = _adapt(_TRAINING_ADAPTER, training_values, root="training")
    inference = _adapt(
        _INFERENCE_ADAPTER,
        merged["inference"],
        root="inference",
    )
    workflow = _adapt(_WORKFLOW_ADAPTER, merged["workflow"], root="workflow")
    _validate_stages(workflow)

    _validate_sampling(
        C=C,
        train_patterns=simulation.train.object.diffractions_per_object,
        test_patterns=simulation.test.object.diffractions_per_object,
        training=training,
    )
    _validate_loss_identity(model, training)

    data = _derive_data_snapshot(
        simulation,
        training,
        C=C,
        gridsize=gridsize,
    )
    resolved = ResolvedSyntheticWorkflow(
        schema_version=_SCHEMA_VERSION,
        profile=_PROFILE_NAME,
        recipe_version=_RECIPE_VERSION,
        simulation=simulation,
        model=model,
        training=training,
        inference=inference,
        workflow=workflow,
        data=data,
    )
    synthetic_workflow_to_dict(resolved)
    return resolved


def materialize_data_config(resolved: ResolvedSyntheticWorkflow) -> DataConfig:
    """Return a fresh mutable Torch DataConfig for one runtime consumer."""

    _validate_resolved_workflow(resolved)
    return DataConfig(
        **{
            item.name: getattr(resolved.data, item.name)
            for item in fields(resolved.data)
        }
    )


def _semantic_value(value: Any) -> Any:
    if isinstance(value, SimulationConfig):
        return simulation_config_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            item.name: _semantic_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(name): _semantic_value(item)
            for name, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("synthetic workflow semantic values must be finite")
    return value


def synthetic_workflow_to_dict(
    resolved: ResolvedSyntheticWorkflow,
) -> dict[str, Any]:
    """Return the one complete JSON-compatible semantic workflow snapshot."""

    _validate_resolved_workflow(resolved)
    return _semantic_value(resolved)


def synthetic_workflow_sha256(
    resolved: ResolvedSyntheticWorkflow | Mapping[str, Any],
) -> str:
    """Return location-independent SHA-256 identity for a resolved workflow."""

    if isinstance(resolved, ResolvedSyntheticWorkflow):
        payload = synthetic_workflow_to_dict(resolved)
    elif isinstance(resolved, Mapping):
        payload = _semantic_value(resolved)
    else:
        raise TypeError("resolved must be a workflow record or semantic mapping")
    expected_roots = {
        "schema_version",
        "profile",
        "recipe_version",
        "simulation",
        "model",
        "training",
        "inference",
        "workflow",
        "data",
    }
    if set(payload) != expected_roots:
        raise ValueError("resolved workflow digest input fields are not exact")
    workflow = payload["workflow"]
    simulation = payload["simulation"]
    if not isinstance(workflow, dict) or not isinstance(simulation, dict):
        raise ValueError("resolved workflow digest namespaces must be mappings")
    workflow.pop("output_root", None)
    for split in ("train", "test"):
        split_config = simulation.get(split)
        if not isinstance(split_config, dict) or not isinstance(
            split_config.get("probe"), dict
        ):
            raise ValueError(f"resolved workflow simulation.{split}.probe is invalid")
        probe = split_config["probe"]
        if probe.get("source") == "custom":
            probe["source_path"] = "<sealed-custom-probe>"

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

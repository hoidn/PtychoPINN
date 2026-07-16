"""Authoritative Torch configuration resolution for ablation studies.

Manifest expansion owns matrix semantics. This module accepts one already-expanded
dotted override mapping and constructs the exact objects consumed by the Torch
training path without using the legacy flat configuration factory.
"""

from __future__ import annotations

import difflib
import json
import math
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from ptycho.config.config import (
    PyTorchExecutionConfig,
    SimulationConfig,
    simulation_config_from_mapping,
)
from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TensorType,
    TrainingConfig,
)
from ptycho_torch.scaling_contract import (
    ResolvedScaleContract,
    resolve_scale_contract,
    validate_amplitude_physics_gain,
)

from .datasets import (
    DatasetCompatibilityError,
    DatasetCompatibilityRequirements,
    ValidatedDataset,
    ValidatedDatasetBundle,
    validate_dataset_compatibility,
)


class ConfigResolutionError(ValueError):
    """Raised when an expanded study override cannot be resolved safely."""


SIMULATION_PATHS = frozenset(
    {
        "simulation.N",
        "simulation.seed",
        "simulation.probe.source",
        "simulation.probe.source_path",
        "simulation.probe.transform_pipeline",
        "simulation.probe.mask_diameter",
        "simulation.object.kind",
        "simulation.object.image_size",
        "simulation.object.objects_per_probe",
        "simulation.object.diffractions_per_object",
        "simulation.object.set_phi",
        "simulation.scan.kind",
        "simulation.scan.grid_size",
        "simulation.scan.offset",
        "simulation.scan.outer_offset_train",
        "simulation.scan.outer_offset_test",
        "simulation.scan.train_groups",
        "simulation.scan.test_groups",
        "simulation.scan.buffer",
        "simulation.detector.photons_per_pattern",
        "simulation.detector.beamstop_diameter",
    }
)


def resolve_simulation_namespace(values: Mapping[str, Any]) -> SimulationConfig:
    """Resolve immutable recursive ``simulation.*`` paths outside arm config."""
    if not isinstance(values, Mapping):
        raise ConfigResolutionError("simulation values must be a mapping")
    unknown = sorted(set(values) - SIMULATION_PATHS)
    if unknown:
        raise ConfigResolutionError(
            f"simulation path {unknown[0]!r} is not allowlisted"
        )
    nested: dict[str, Any] = {}
    for path, value in values.items():
        cursor = nested
        parts = path.split(".")[1:]
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    try:
        return simulation_config_from_mapping(nested)
    except (TypeError, ValueError) as exc:
        raise ConfigResolutionError(f"simulation config is invalid: {exc}") from exc


NAMESPACE_OWNERS: dict[str, str | type[Any]] = {
    "dataset": "immutable dataset descriptor",
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "inference": InferenceConfig,
    "execution": PyTorchExecutionConfig,
}


ALLOWLISTS: dict[str, frozenset[str]] = {
    "dataset": frozenset({"id"}),
    "data": frozenset(
        {
            "scale_contract_version",
            "measurement_domain",
            "N",
            "C",
            "n_subsample",
            "grid_size",
            "normalize",
            "probe_scale",
            "probe_normalize",
            "data_scaling",
            "phase_subtraction",
            "x_bounds",
            "y_bounds",
        }
    ),
    "model": frozenset(
        {
            "mode",
            "architecture",
            "fno_modes",
            "fno_width",
            "fno_blocks",
            "fno_cnn_blocks",
            "learned_input_channels",
            "fno_input_transform",
            "max_hidden_channels",
            "resnet_width",
            "hybrid_skip_connections",
            "hybrid_downsample_steps",
            "hybrid_downsample_op",
            "hybrid_encoder_conv_hidden_scale",
            "hybrid_encoder_spectral_hidden_scale",
            "hybrid_encoder_conv_hidden_channels",
            "hybrid_encoder_spectral_hidden_channels",
            "hybrid_resnet_blocks",
            "hybrid_skip_style",
            "hybrid_resnet_bottleneck_layerscale_mode",
            "hybrid_resnet_bottleneck_layerscale_value",
            "hybrid_encoder_fusion_mode",
            "hybrid_encoder_layerscale_init",
            "hybrid_encoder_branch_gate_init",
            "hybrid_encoder_branch_select",
            "generator_output_mode",
            "cnn_output_mode",
            "use_shared_decoder",
            "C_model",
            "n_filters_scale",
            "amp_activation",
            "batch_norm",
            "probe_mask",
            "probe_mask_tensor",
            "probe_mask_sigma",
            "probe_mask_diameter",
            "decoder_last_c_outer_fraction",
            "decoder_last_amp_channels",
            "cbam_encoder",
            "cbam_bottleneck",
            "object_big",
            "probe_big",
            "offset",
            "C_forward",
            "training_patch_weighting",
            "physics_forward_mode",
            "rect_s1s2_trainable",
            "loss_function",
            "amplitude_physics_gain",
            "amp_loss",
            "phase_loss",
            "amp_loss_coeff",
            "phase_loss_coeff",
        }
    ),
    "training": frozenset(
        {
            "learning_rate",
            "epochs",
            "batch_size",
            "epochs_fine_tune",
            "fine_tune_gamma",
            "scheduler",
            "lr_warmup_epochs",
            "lr_min_ratio",
            "plateau_factor",
            "plateau_patience",
            "plateau_min_lr",
            "plateau_threshold",
            "accum_steps",
            "gradient_clip_val",
            "gradient_clip_algorithm",
            "optimizer",
            "momentum",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "log_grad_norm",
            "grad_norm_log_freq",
            "torch_loss_mode",
            "torch_mae_pred_l2_match_target",
            "experiment_name",
            "notes",
            "model_name",
        }
    ),
    "inference": frozenset(
        {
            "middle_trim",
            "batch_size",
            "patch_weighting",
            "varpro_scaling",
        }
    ),
    "execution": frozenset(
        {
            "accelerator",
            "devices",
            "strategy",
            "deterministic",
            "precision",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "enable_progress_bar",
            "enable_checkpointing",
            "checkpoint_save_top_k",
            "checkpoint_monitor_metric",
            "checkpoint_mode",
            "early_stop_patience",
        }
    ),
}


@dataclass(frozen=True)
class ArchitecturePolicy:
    """Closed construction and applicability policy for one proven architecture."""

    applicable_paths: frozenset[str]
    output_path: str
    learned_input_channels: int | None = None
    minimum_fno_blocks: int | None = None
    resnet_width_divisor: int | None = None
    minimum_hidden_per_downsample: bool = False
    object_big_required_for_multichannel: bool = False
    fine_tuning_supported: bool = False


ARCHITECTURE_POLICIES: Mapping[str, ArchitecturePolicy] = MappingProxyType(
    {
        "cnn": ArchitecturePolicy(
            applicable_paths=frozenset(
                {
                    "model.use_shared_decoder",
                    "model.n_filters_scale",
                    "model.batch_norm",
                    "model.cbam_encoder",
                    "model.cbam_bottleneck",
                    "model.decoder_last_c_outer_fraction",
                    "model.decoder_last_amp_channels",
                }
            ),
            output_path="model.cnn_output_mode",
            object_big_required_for_multichannel=True,
            fine_tuning_supported=True,
        ),
        "fno": ArchitecturePolicy(
            applicable_paths=frozenset(
                {
                    "model.generator_output_mode",
                    "model.fno_modes",
                    "model.fno_width",
                    "model.fno_blocks",
                    "model.fno_cnn_blocks",
                    "model.fno_input_transform",
                    "model.learned_input_channels",
                }
            ),
            output_path="model.generator_output_mode",
            learned_input_channels=1,
            minimum_fno_blocks=1,
        ),
        "hybrid_resnet": ArchitecturePolicy(
            applicable_paths=frozenset(
                {
                    "model.generator_output_mode",
                    "model.fno_modes",
                    "model.fno_width",
                    "model.fno_blocks",
                    "model.fno_input_transform",
                    "model.learned_input_channels",
                    "model.max_hidden_channels",
                    "model.resnet_width",
                    "model.hybrid_skip_connections",
                    "model.hybrid_downsample_steps",
                    "model.hybrid_downsample_op",
                    "model.hybrid_encoder_conv_hidden_scale",
                    "model.hybrid_encoder_spectral_hidden_scale",
                    "model.hybrid_encoder_conv_hidden_channels",
                    "model.hybrid_encoder_spectral_hidden_channels",
                    "model.hybrid_resnet_blocks",
                    "model.hybrid_resnet_bottleneck_layerscale_mode",
                    "model.hybrid_resnet_bottleneck_layerscale_value",
                    "model.hybrid_encoder_fusion_mode",
                    "model.hybrid_encoder_layerscale_init",
                    "model.hybrid_encoder_branch_gate_init",
                    "model.hybrid_encoder_branch_select",
                }
            ),
            output_path="model.generator_output_mode",
            learned_input_channels=1,
            minimum_fno_blocks=3,
            resnet_width_divisor=4,
            minimum_hidden_per_downsample=True,
        ),
    }
)


@dataclass(frozen=True)
class ExecutionPlatformPolicy:
    """Closed study policy for an effective Lightning accelerator."""

    strategies: frozenset[str]
    precisions: frozenset[str]
    requires_single_device: bool = False


_CUDA_EXECUTION_POLICY = ExecutionPlatformPolicy(
    strategies=frozenset({"auto", "ddp"}),
    precisions=frozenset({"32-true", "16-mixed", "bf16-mixed"}),
    requires_single_device=True,
)
EXECUTION_PLATFORM_POLICIES: Mapping[str, ExecutionPlatformPolicy] = MappingProxyType(
    {
        "cpu": ExecutionPlatformPolicy(
            strategies=frozenset({"auto", "ddp"}),
            precisions=frozenset({"32-true", "bf16-mixed"}),
            requires_single_device=True,
        ),
        "cuda": _CUDA_EXECUTION_POLICY,
        "gpu": _CUDA_EXECUTION_POLICY,
    }
)


REJECTED_PATH_REASONS: dict[str, str] = {
    "data.nphotons": "inert for immutable measured datasets",
    "data.subsample_seed": "unsupported because seed is runtime run identity",
    "data.K": "unsupported; Task4 v1 fixes grouping to Nearest with K=6",
    "data.K_quadrant": "unsupported; alternate grouping policies are not proven",
    "data.neighbor_function": "unsupported; Task4 v1 fixes grouping to Nearest",
    "data.min_neighbor_distance": "unsupported; distance grouping is not proven",
    "data.max_neighbor_distance": "unsupported; distance grouping is not proven",
    "data.scan_pattern": "unsupported; quadrant grouping is not proven",
    "model.edge_pad": "inert in canonical model construction",
    "model.pad_object": "inert in canonical model construction",
    "model.gaussian_smoothing_sigma": "inert in canonical model construction",
    "model.eca_encoder": "inert because the encoder never applies the ECA module",
    "model.eca_decoder": "inert because canonical decoders bypass attention blocks",
    "model.cbam_decoder": "inert because canonical decoders bypass attention blocks",
    "model.spatial_decoder": "inert because canonical decoders bypass attention blocks",
    "model.decoder_spatial_kernel": "inert because canonical decoders bypass attention blocks",
    "training.nll": "unsupported input; derived from training.torch_loss_mode",
    "inference.pad_eval": "inert in reconstruct_image_barycentric",
    "inference.window": "inert in reconstruct_image_barycentric",
    "inference.log_patch_stats": "inert in reconstruct_image_barycentric",
    "inference.patch_stats_limit": "inert in reconstruct_image_barycentric",
    "execution.logger_backend": "inert in train_lightning_only",
    "execution.recon_log_every_n_epochs": "inert in train_lightning_only",
    "execution.recon_log_num_patches": "inert in train_lightning_only",
    "execution.recon_log_fixed_indices": "inert in train_lightning_only",
    "execution.recon_log_stitch": "inert in train_lightning_only",
    "execution.recon_log_max_stitch_samples": "inert in train_lightning_only",
}


TRAINING_TO_EXECUTION_ALIASES: dict[str, str] = {
    "learning_rate": "learning_rate",
    "scheduler": "scheduler",
    "gradient_clip_val": "gradient_clip_val",
    "gradient_clip_algorithm": "gradient_clip_algorithm",
    "accum_steps": "accum_steps",
}

EXECUTION_TO_TRAINING_ALIASES: dict[str, str] = {
    "strategy": "strategy",
    "devices": "n_devices",
    "num_workers": "num_workers",
}

OPTIONAL_TOML_SENTINELS: dict[str, tuple[str, Any]] = {
    "model.max_hidden_channels": ("auto", None),
    "model.resnet_width": ("auto", None),
    "model.probe_mask_tensor": ("auto", None),
    "model.probe_mask_diameter": ("auto", None),
    "model.amp_loss": ("disabled", None),
    "model.phase_loss": ("disabled", None),
    "training.gradient_clip_val": ("disabled", None),
    "execution.prefetch_factor": ("auto", None),
}

_CLAIM_GRADE_DISCRIMINATORS = frozenset(
    {
        "model.mode",
        "model.architecture",
        "model.object_big",
        "model.physics_forward_mode",
        "model.probe_mask",
        "model.amp_loss",
        "model.phase_loss",
        "training.scheduler",
        "training.gradient_clip_val",
        "training.optimizer",
        "training.log_grad_norm",
        "training.torch_loss_mode",
        "execution.num_workers",
        "execution.enable_checkpointing",
    }
)

# Introduced after the original strict checked manifest was pinned. It is accepted
# and provenance-bearing when supplied, but old strict manifests retain the
# validated ModelConfig default.
_BACKWARD_COMPATIBLE_OPTIONAL_EXPLICIT_PATHS = frozenset(
    {
        "model.amplitude_physics_gain",
        "model.hybrid_encoder_conv_hidden_channels",
        "model.hybrid_encoder_spectral_hidden_channels",
        "model.hybrid_resnet_bottleneck_layerscale_mode",
        "model.hybrid_resnet_bottleneck_layerscale_value",
        "model.hybrid_encoder_fusion_mode",
        "model.hybrid_encoder_layerscale_init",
        "model.hybrid_encoder_branch_gate_init",
        "model.hybrid_encoder_branch_select",
    }
)


def _validate_registry() -> None:
    accepted: set[str] = set()
    for namespace, allowed in ALLOWLISTS.items():
        owner = NAMESPACE_OWNERS[namespace]
        if isinstance(owner, str):
            continue
        declared = {item.name for item in fields(owner)}
        missing = allowed - declared
        if missing:
            raise RuntimeError(
                f"allowlist fields missing from {owner.__name__}: {sorted(missing)}"
            )
        paths = {f"{namespace}.{field_name}" for field_name in allowed}
        if accepted & paths:
            raise RuntimeError("configuration path has multiple owners")
        accepted.update(paths)
    for architecture, policy in ARCHITECTURE_POLICIES.items():
        invalid = policy.applicable_paths - accepted
        if invalid:
            raise RuntimeError(
                f"architecture {architecture!r} references non-allowlisted paths: "
                f"{sorted(invalid)}"
            )


_validate_registry()
_VALID_PATHS = frozenset(
    f"{namespace}.{field_name}"
    for namespace, allowed in ALLOWLISTS.items()
    for field_name in allowed
)
_TYPE_HINTS: dict[str, dict[str, Any]] = {
    namespace: get_type_hints(owner)
    for namespace, owner in NAMESPACE_OWNERS.items()
    if isinstance(owner, type)
}


def _architecture_policy(architecture: str) -> ArchitecturePolicy:
    try:
        return ARCHITECTURE_POLICIES[architecture]
    except KeyError as exc:
        supported = ", ".join(sorted(ARCHITECTURE_POLICIES))
        raise ConfigResolutionError(
            f"model.architecture={architecture!r} is unsupported by this study; "
            f"supported architectures: {supported}"
        ) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_native(
    value: Any,
    path: str,
    active_containers: set[int] | None = None,
) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigResolutionError(f"{path} must be finite")
        return value
    if isinstance(value, (tuple, list)):
        active = active_containers if active_containers is not None else set()
        identity = id(value)
        if identity in active:
            raise ConfigResolutionError(
                f"{path} contains a recursive non-JSON container mutation"
            )
        active.add(identity)
        try:
            return [
                _json_native(item, f"{path}[{index}]", active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise ConfigResolutionError(f"{path} is not JSON-native: {type(value).__name__}")


def _snapshot_config(config: Any, namespace: str) -> dict[str, Any]:
    return {
        item.name: _json_native(getattr(config, item.name), f"{namespace}.{item.name}")
        for item in fields(config)
    }


def _build_snapshot(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    datagen_config: DatagenConfig,
    execution_config: PyTorchExecutionConfig,
    profile: ResolvedScaleContract,
    active: bool,
    dataset_id: str | None,
) -> dict[str, Any]:
    return {
        "ci_scaling_active": active,
        "data": _snapshot_config(data_config, "data"),
        "datagen": _snapshot_config(datagen_config, "datagen"),
        "dataset_id": dataset_id,
        "execution": _snapshot_config(execution_config, "execution"),
        "inference": _snapshot_config(inference_config, "inference"),
        "model": _snapshot_config(model_config, "model"),
        "profile": {
            "measurement_domain": profile.measurement_domain,
            "scale_contract_version": profile.version,
        },
        "training": _snapshot_config(training_config, "training"),
    }


@dataclass(frozen=True)
class ResolvedTorchConfigs:
    """Resolved configs plus a mutation-detecting canonical provenance record."""

    data_config: DataConfig
    model_config: ModelConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    datagen_config: DatagenConfig
    execution_config: PyTorchExecutionConfig
    ci_scaling_active: bool
    profile: ResolvedScaleContract
    dataset_id: str | None
    _snapshot_json: str
    _snapshot_frozen: Any
    _identities: tuple[int, int, int, int, int, int]

    @property
    def existing_config(
        self,
    ) -> tuple[DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig]:
        """Return the tuple order required by ``train_lightning_only.main``."""
        self.validate_integrity()
        return (
            self.data_config,
            self.model_config,
            self.training_config,
            self.inference_config,
            self.datagen_config,
        )

    @property
    def snapshot(self) -> dict[str, Any]:
        """Return an isolated JSON-native copy of the complete resolved snapshot."""
        self.validate_integrity()
        return json.loads(self._snapshot_json)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a copy-safe JSON-native representation."""
        self.validate_integrity()
        return json.loads(self._snapshot_json)

    @property
    def canonical_json(self) -> str:
        self.validate_integrity()
        return self._snapshot_json

    def validate_integrity(self) -> None:
        configs = (
            self.data_config,
            self.model_config,
            self.training_config,
            self.inference_config,
            self.datagen_config,
            self.execution_config,
        )
        if tuple(map(id, configs)) != self._identities:
            raise ConfigResolutionError(
                "resolved config object identity changed after resolution"
            )
        current = _build_snapshot(
            *configs,
            self.profile,
            self.ci_scaling_active,
            self.dataset_id,
        )
        if (
            _freeze_json(current) != self._snapshot_frozen
            or _canonical_json(current) != self._snapshot_json
        ):
            raise ConfigResolutionError("resolved config mutated after resolution")

    def assert_unmodified(self) -> None:
        self.validate_integrity()

    def to_json(self) -> str:
        self.validate_integrity()
        return self._snapshot_json


def _suggestion(path: str) -> str:
    matches = difflib.get_close_matches(path, sorted(_VALID_PATHS), n=3, cutoff=0.55)
    return (
        f" Did you mean {', '.join(repr(item) for item in matches)}?" if matches else ""
    )


def _split_and_validate_path(path: object) -> tuple[str, str]:
    if not isinstance(path, str):
        raise ConfigResolutionError(
            f"override path must be a string, got {type(path).__name__}"
        )
    if path in REJECTED_PATH_REASONS:
        raise ConfigResolutionError(
            f"override path {path!r} is {REJECTED_PATH_REASONS[path]}"
        )
    if path not in _VALID_PATHS:
        raise ConfigResolutionError(
            f"override path {path!r} is not allowlisted.{_suggestion(path)}"
        )
    namespace, field_name = path.split(".", 1)
    return namespace, field_name


def _describe(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return "one of " + ", ".join(repr(item) for item in args)
    if origin in (Union, types.UnionType):
        non_none = [item for item in args if item is not type(None)]
        suffix = " or null" if len(non_none) != len(args) else ""
        if len(non_none) == 1:
            return _describe(non_none[0]) + suffix
        return " or ".join(_describe(item) for item in non_none) + suffix
    if origin is tuple:
        return "array"
    if origin is list:
        return "array"
    return {bool: "boolean", int: "exact integer", float: "number", str: "string"}.get(
        annotation, getattr(annotation, "__name__", str(annotation))
    )


def _coerce_literal(annotation: Any, value: Any, path: str) -> Any:
    choices = get_args(annotation)
    for choice in choices:
        if choice is None and value is None:
            return None
        if type(value) is type(choice) and value == choice:
            return value
    raise ConfigResolutionError(
        f"{path} expected one of {list(choices)!r}; got {value!r}"
    )


def _coerce_union(annotation: Any, value: Any, path: str) -> Any:
    args = get_args(annotation)
    if value is None and type(None) in args:
        return None
    errors: list[ConfigResolutionError] = []
    for candidate in args:
        if candidate is type(None):
            continue
        try:
            return _coerce_value(candidate, value, path)
        except ConfigResolutionError as exc:
            errors.append(exc)
    expected = _describe(annotation)
    raise ConfigResolutionError(
        f"{path} expected {expected}; got {value!r}"
    ) from errors[-1]


def _coerce_tuple(annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigResolutionError(
            f"{path} expected an array; got {type(value).__name__}"
        )
    args = get_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            _coerce_value(args[0], item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if len(value) != len(args):
        raise ConfigResolutionError(
            f"{path} expected {len(args)} items; got {len(value)}"
        )
    return tuple(
        _coerce_value(item_type, item, f"{path}[{index}]")
        for index, (item_type, item) in enumerate(zip(args, value))
    )


def _coerce_list(annotation: Any, value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigResolutionError(
            f"{path} expected an array; got {type(value).__name__}"
        )
    (item_type,) = get_args(annotation)
    return [
        _coerce_value(item_type, item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]


def _coerce_primitive(annotation: Any, value: Any, path: str) -> Any:
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigResolutionError(f"{path} expected a boolean; got {value!r}")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigResolutionError(
                f"{path} expected an exact integer; got {value!r}"
            )
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigResolutionError(f"{path} expected a number; got {value!r}")
        try:
            result = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ConfigResolutionError(f"{path} must be finite") from exc
        if not math.isfinite(result):
            raise ConfigResolutionError(f"{path} must be finite")
        return result
    raise ConfigResolutionError(
        f"{path} has unsupported primitive annotation {annotation!r}"
    )


def _coerce_scalar(annotation: Any, value: Any, path: str) -> Any:
    if annotation in {bool, int, float}:
        return _coerce_primitive(annotation, value, path)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigResolutionError(f"{path} expected a string; got {value!r}")
        return value
    if isinstance(value, annotation):
        return value
    raise ConfigResolutionError(
        f"{path} expected {_describe(annotation)}; got {value!r}"
    )


def _coerce_value(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    if origin is Literal:
        return _coerce_literal(annotation, value, path)
    if origin in (Union, types.UnionType):
        return _coerce_union(annotation, value, path)
    if origin is tuple:
        return _coerce_tuple(annotation, value, path)
    if origin is list:
        return _coerce_list(annotation, value, path)
    return _coerce_scalar(annotation, value, path)


def _validate_override_json_native(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigResolutionError(f"{path} must be finite")
        return
    if isinstance(value, list) or (
        path in {"data.grid_size", "data.x_bounds", "data.y_bounds"}
        and isinstance(value, tuple)
    ):
        for index, item in enumerate(value):
            _validate_override_json_native(item, f"{path}[{index}]")
        return
    raise ConfigResolutionError(
        f"{path} must be JSON-native; got {type(value).__name__}"
    )


def _numeric_mask(value: Any, path: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
        raise ConfigResolutionError(f"{path} must be a JSON numeric array")
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ConfigResolutionError(f"{path} must contain finite numbers") from exc
        if not math.isfinite(number):
            raise ConfigResolutionError(f"{path} must contain finite numbers")
        if number < 0:
            raise ConfigResolutionError(f"{path} must contain nonnegative numbers")
        return value
    if not value:
        raise ConfigResolutionError(f"{path} must not be empty")
    converted = [
        _numeric_mask(item, f"{path}[{index}]") for index, item in enumerate(value)
    ]
    child_shapes = {_mask_shape(item) for item in converted}
    if len(child_shapes) != 1:
        raise ConfigResolutionError(f"{path} must be a rectangular numeric array")
    return converted


def _mask_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return (len(value),) + _mask_shape(value[0])


def _coerce_mask(field_name: str, value: Any, path: str) -> Any:
    if isinstance(value, TensorType):
        raise ConfigResolutionError(
            f"{path} rejects tensor values; use TOML/JSON controls"
        )
    if value is None:
        return None
    if field_name == "probe_mask" and type(value) is bool:
        return value
    if not isinstance(value, list):
        supported = (
            "a boolean or numeric array"
            if field_name == "probe_mask"
            else "a numeric array"
        )
        raise ConfigResolutionError(f"{path} must be {supported}")
    return _numeric_mask(value, path)


def _validated_dataset_id(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ConfigResolutionError(f"{path} must be a nonempty valid string")
    return value


def _normalize_required_capabilities(value: object) -> tuple[str, ...]:
    error = "required_capabilities must be an iterable of nonempty strings"
    if isinstance(value, (str, bytes)):
        raise ConfigResolutionError(error)
    try:
        capabilities = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise ConfigResolutionError(error) from exc
    if any(
        not isinstance(capability, str) or not capability.strip()
        for capability in capabilities
    ):
        raise ConfigResolutionError(error)
    return cast(tuple[str, ...], capabilities)


def _coerce_overrides(
    overrides: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not isinstance(overrides, Mapping):
        raise ConfigResolutionError("overrides must be a mapping of dotted paths")
    grouped: dict[str, dict[str, Any]] = {
        namespace: {} for namespace in NAMESPACE_OWNERS if namespace != "dataset"
    }
    dataset_id: str | None = None
    for raw_path, value in overrides.items():
        namespace, field_name = _split_and_validate_path(raw_path)
        if namespace == "dataset":
            dataset_id = _validated_dataset_id(value, "dataset.id")
            continue
        if field_name not in {"probe_mask", "probe_mask_tensor"}:
            _validate_override_json_native(value, raw_path)
        sentinel = OPTIONAL_TOML_SENTINELS.get(raw_path)
        if sentinel is not None and isinstance(value, str) and value == sentinel[0]:
            grouped[namespace][field_name] = sentinel[1]
            continue
        if field_name in {"probe_mask", "probe_mask_tensor"}:
            grouped[namespace][field_name] = _coerce_mask(field_name, value, raw_path)
        else:
            grouped[namespace][field_name] = _coerce_value(
                _TYPE_HINTS[namespace][field_name], value, raw_path
            )
    return grouped, dataset_id


def _require_positive(path: str, value: int | float) -> None:
    try:
        valid = (
            not isinstance(value, bool) and math.isfinite(float(value)) and value > 0
        )
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ConfigResolutionError(f"{path} must be positive")


def _require_nonnegative(path: str, value: int | float) -> None:
    try:
        valid = (
            not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0
        )
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ConfigResolutionError(f"{path} must be nonnegative")


def _validate_neighbor_policy(config: DataConfig) -> None:
    if config.neighbor_function != "Nearest" or config.K != 6:
        raise ConfigResolutionError(
            "Task4 v1 runtime compatibility requires fixed Nearest grouping with K=6"
        )


def _validate_data(config: DataConfig) -> None:
    for name in ("N", "C", "K", "K_quadrant", "n_subsample", "probe_scale"):
        _require_positive(f"data.{name}", getattr(config, name))
    if config.N not in {64, 128, 256}:
        raise ConfigResolutionError("data.N must be 64, 128, or 256")
    for index, size in enumerate(config.grid_size):
        _require_positive(f"data.grid_size[{index}]", size)
    if config.C != config.grid_size[0] * config.grid_size[1]:
        raise ConfigResolutionError(
            "data.C must equal data.grid_size[0] * data.grid_size[1]"
        )
    _validate_neighbor_policy(config)
    _require_nonnegative("data.min_neighbor_distance", config.min_neighbor_distance)
    if config.max_neighbor_distance <= config.min_neighbor_distance:
        raise ConfigResolutionError(
            "data.max_neighbor_distance must exceed data.min_neighbor_distance"
        )
    for name in ("x_bounds", "y_bounds"):
        lower, upper = getattr(config, name)
        if not 0 <= lower < upper <= 1:
            raise ConfigResolutionError(
                f"data.{name} must satisfy 0 <= lower < upper <= 1"
            )


def _validate_probe_mask_shapes(config: ModelConfig, data_config: DataConfig) -> None:
    expected_shapes = {
        (data_config.N, data_config.N),
        (data_config.N, data_config.N, 1),
        (1, 1, data_config.N, data_config.N),
    }
    for name in ("probe_mask", "probe_mask_tensor"):
        value = getattr(config, name)
        if value is None or isinstance(value, bool):
            continue
        shape = _mask_shape(value)
        if shape not in expected_shapes:
            raise ConfigResolutionError(
                f"model.{name} must have runtime mask shape ({data_config.N}, {data_config.N}), "
                f"({data_config.N}, {data_config.N}, 1), or "
                f"(1, 1, {data_config.N}, {data_config.N}); got {shape}"
            )


def _validate_model_semantics(config: ModelConfig) -> None:
    if config.hybrid_downsample_steps not in {1, 2}:
        raise ConfigResolutionError("model.hybrid_downsample_steps must be 1 or 2")
    if config.amp_activation not in {"silu", "sigmoid"}:
        raise ConfigResolutionError("model.amp_activation must be 'silu' or 'sigmoid'")


def _validate_architecture_constraints(config: ModelConfig) -> None:
    policy = _architecture_policy(config.architecture)
    if (
        policy.learned_input_channels is not None
        and config.learned_input_channels != policy.learned_input_channels
    ):
        raise ConfigResolutionError(
            "model.learned_input_channels must be "
            f"{policy.learned_input_channels} for proven canonical forwards"
        )
    if (
        policy.minimum_fno_blocks is not None
        and config.fno_blocks < policy.minimum_fno_blocks
    ):
        raise ConfigResolutionError(
            "model.fno_blocks must be at least "
            f"{policy.minimum_fno_blocks} for {config.architecture}"
        )
    if (
        policy.resnet_width_divisor is not None
        and config.resnet_width is not None
        and config.resnet_width % policy.resnet_width_divisor != 0
    ):
        raise ConfigResolutionError(
            "model.resnet_width must be divisible by "
            f"{policy.resnet_width_divisor} for CycleGAN upsamplers"
        )
    if policy.minimum_hidden_per_downsample:
        minimum_hidden = 2**config.hybrid_downsample_steps
        if (
            config.max_hidden_channels is not None
            and config.max_hidden_channels < minimum_hidden
        ):
            raise ConfigResolutionError(
                "model.max_hidden_channels must be at least "
                f"{minimum_hidden} for {config.hybrid_downsample_steps} downsample steps"
            )


def _validate_cnn_decoder(config: ModelConfig) -> None:
    if config.architecture != "cnn":
        return
    if config.decoder_last_amp_channels not in {1, config.C_model}:
        raise ConfigResolutionError(
            "model.decoder_last_amp_channels must be 1 or model.C_model"
        )
    minimum_fraction = 1.0 / (32 * config.n_filters_scale)
    if not minimum_fraction <= config.decoder_last_c_outer_fraction <= 0.5:
        raise ConfigResolutionError(
            "model.decoder_last_c_outer_fraction must be in the effective range "
            f"[{minimum_fraction:g}, 0.5]"
        )


def _validate_model(config: ModelConfig, data_config: DataConfig) -> None:
    if config.C_model != data_config.C:
        raise ConfigResolutionError("model.C_model must equal data.C")
    if config.C_forward != data_config.C:
        raise ConfigResolutionError("model.C_forward must equal data.C")
    policy = _architecture_policy(config.architecture)
    if (
        policy.object_big_required_for_multichannel
        and not config.object_big
        and config.C_model > 1
    ):
        raise ConfigResolutionError(
            "model.architecture='cnn' with model.object_big=false requires "
            "C=1 (data.C=model.C_model=1) because the canonical encoder expects "
            "one channel"
        )
    positive = (
        "fno_modes",
        "fno_width",
        "fno_blocks",
        "fno_cnn_blocks",
        "learned_input_channels",
        "hybrid_encoder_conv_hidden_scale",
        "hybrid_encoder_spectral_hidden_scale",
        "hybrid_resnet_blocks",
        "spectral_bottleneck_blocks",
        "spectral_bottleneck_modes",
        "n_filters_scale",
        "probe_mask_sigma",
    )
    for name in positive:
        _require_positive(f"model.{name}", getattr(config, name))
    for name in ("max_hidden_channels", "resnet_width", "probe_mask_diameter"):
        value = getattr(config, name)
        if value is not None:
            _require_positive(f"model.{name}", value)
    for name in ("offset", "amp_loss_coeff", "phase_loss_coeff"):
        _require_nonnegative(f"model.{name}", getattr(config, name))
    _validate_model_semantics(config)
    _validate_architecture_constraints(config)
    _validate_cnn_decoder(config)
    _validate_probe_mask_shapes(config, data_config)
    try:
        validate_amplitude_physics_gain(config)
    except (TypeError, ValueError) as error:
        raise ConfigResolutionError(str(error)) from error


def _validate_scheduler(config: TrainingConfig) -> None:
    if config.lr_warmup_epochs > config.epochs:
        raise ConfigResolutionError(
            "training.lr_warmup_epochs must not exceed training.epochs"
        )
    if not 0 < config.lr_min_ratio <= 1:
        raise ConfigResolutionError("training.lr_min_ratio must be in (0, 1]")
    if not 0 < config.plateau_factor < 1:
        raise ConfigResolutionError("training.plateau_factor must be in (0, 1)")
    _require_positive("training.plateau_min_lr", config.plateau_min_lr)
    if (
        config.scheduler == "ReduceLROnPlateau"
        and config.plateau_min_lr > config.learning_rate
    ):
        raise ConfigResolutionError(
            "training.plateau_min_lr must not exceed training.learning_rate"
        )
    if config.scheduler in {"MultiStage", "Adaptive"}:
        raise ConfigResolutionError(
            f"training.scheduler={config.scheduler!r} is inert in single-loss training"
        )


def _validate_optimizer(config: TrainingConfig) -> None:
    if config.gradient_clip_val is not None:
        _require_positive("training.gradient_clip_val", config.gradient_clip_val)
    if config.optimizer not in {"adam", "adamw", "sgd"}:
        raise ConfigResolutionError(
            "training.optimizer must be 'adam', 'adamw', or 'sgd'"
        )
    if config.gradient_clip_algorithm not in {"norm", "value", "agc"}:
        raise ConfigResolutionError(
            "training.gradient_clip_algorithm must be 'norm', 'value', or 'agc'"
        )
    if not 0 <= config.momentum < 1:
        raise ConfigResolutionError("training.momentum must be in [0, 1)")
    for name in ("adam_beta1", "adam_beta2"):
        if not 0 <= getattr(config, name) < 1:
            raise ConfigResolutionError(f"training.{name} must be in [0, 1)")


def _validate_training(config: TrainingConfig) -> None:
    positive = (
        "learning_rate",
        "epochs",
        "batch_size",
        "accum_steps",
        "fine_tune_gamma",
        "grad_norm_log_freq",
    )
    nonnegative = (
        "epochs_fine_tune",
        "lr_warmup_epochs",
        "plateau_patience",
        "plateau_threshold",
        "weight_decay",
    )
    for name in positive:
        _require_positive(f"training.{name}", getattr(config, name))
    for name in nonnegative:
        _require_nonnegative(f"training.{name}", getattr(config, name))
    if config.fine_tune_gamma > 1:
        raise ConfigResolutionError("training.fine_tune_gamma must be in (0, 1]")
    _validate_scheduler(config)
    _validate_optimizer(config)


def _validate_inference(config: InferenceConfig, data_config: DataConfig) -> None:
    _require_positive("inference.middle_trim", config.middle_trim)
    if config.middle_trim % 2 != 0:
        raise ConfigResolutionError("inference.middle_trim must be even")
    if config.middle_trim > data_config.N:
        raise ConfigResolutionError("inference.middle_trim must not exceed data.N")
    _require_positive("inference.batch_size", config.batch_size)


def _validate_execution_platform(values: Mapping[str, Any]) -> None:
    if "devices" in values:
        devices = values["devices"]
        if devices != "auto" and (type(devices) is not int or devices <= 0):
            raise ConfigResolutionError(
                "execution.devices must be a positive integer or 'auto'"
            )
        if devices != 1 and values.get("accelerator") != "mps":
            raise ConfigResolutionError(
                "canonical ablation requires execution.devices=1 because canonical "
                "held-out mmap/reassembly and framework peak-memory evidence are "
                "single-device-only"
            )
    strategy = values.get("strategy", "auto")
    if strategy not in {"auto", "ddp"}:
        raise ConfigResolutionError(
            f"execution.strategy={strategy!r} is unsupported by the study resolver; "
            "supported strategies are 'auto' and 'ddp'"
        )


def _validate_resolved_execution_platform(
    config: PyTorchExecutionConfig,
) -> None:
    if config.accelerator == "mps":
        raise ConfigResolutionError(
            "execution.accelerator='mps' is unsupported for canonical ablation: "
            "MPS lacks required float64 device tensors for reassembly and "
            "fitted-count accumulation"
        )
    try:
        policy = EXECUTION_PLATFORM_POLICIES[config.accelerator]
    except KeyError as exc:
        raise ConfigResolutionError(
            f"execution.accelerator={config.accelerator!r} did not resolve to a "
            "supported effective accelerator"
        ) from exc
    if config.strategy not in policy.strategies:
        raise ConfigResolutionError(
            f"execution.strategy={config.strategy!r} is unsupported for "
            f"execution.accelerator={config.accelerator!r}"
        )
    if config.precision not in policy.precisions:
        detail = (
            "; Lightning rewrites CPU 16-mixed to bf16-mixed"
            if config.accelerator == "cpu" and config.precision == "16-mixed"
            else ""
        )
        raise ConfigResolutionError(
            f"execution.precision={config.precision!r} is unsupported for "
            f"execution.accelerator={config.accelerator!r}{detail}"
        )
    if policy.requires_single_device and config.devices != 1:
        raise ConfigResolutionError(
            "canonical ablation requires execution.devices=1 because canonical "
            "held-out mmap/reassembly and framework peak-memory evidence are "
            "single-device-only"
        )


def _validate_execution_loader(values: Mapping[str, Any]) -> None:
    num_workers = values.get("num_workers", 0)
    _require_nonnegative("execution.num_workers", num_workers)
    if values.get("persistent_workers", False) and num_workers == 0:
        raise ConfigResolutionError(
            "execution.persistent_workers requires execution.num_workers > 0"
        )
    if values.get("prefetch_factor") is not None and num_workers == 0:
        raise ConfigResolutionError(
            "execution.prefetch_factor requires execution.num_workers > 0"
        )
    for name in ("prefetch_factor", "early_stop_patience"):
        value = values.get(name)
        if value is not None:
            _require_positive(f"execution.{name}", value)


def _validate_execution_logging(values: Mapping[str, Any]) -> None:
    monitor = values.get("checkpoint_monitor_metric", "val_loss")
    if monitor != "val_loss":
        raise ConfigResolutionError(
            "execution.checkpoint_monitor_metric must be 'val_loss'; the runtime "
            "resolves this sentinel to the model's emitted validation loss metric"
        )
    if "checkpoint_save_top_k" in values:
        _require_nonnegative(
            "execution.checkpoint_save_top_k", values["checkpoint_save_top_k"]
        )


def _validate_execution_values(values: Mapping[str, Any]) -> None:
    _validate_execution_platform(values)
    _validate_execution_loader(values)
    _validate_execution_logging(values)


def _applicable_data_paths(
    data_config: DataConfig,
    model_config: ModelConfig,
) -> set[str]:
    paths = {f"data.{name}" for name in ALLOWLISTS["data"]}
    legacy_amplitude = (
        data_config.scale_contract_version == "legacy_v1"
        and data_config.measurement_domain == "normalized_amplitude"
        and model_config.physics_forward_mode == "amplitude"
    )
    if not legacy_amplitude:
        paths -= {"data.normalize", "data.data_scaling"}
    if model_config.mode != "Supervised":
        paths.discard("data.phase_subtraction")
    return paths


def _applicable_probe_mask_paths(model_config: ModelConfig) -> set[str]:
    paths = {"model.probe_mask"}
    if model_config.probe_mask is not True:
        return paths
    paths.add("model.probe_mask_tensor")
    if model_config.probe_mask_tensor is None:
        paths.update({"model.probe_mask_sigma", "model.probe_mask_diameter"})
    return paths


def _applicable_architecture_paths(model_config: ModelConfig) -> set[str]:
    architecture = model_config.architecture
    policy = _architecture_policy(architecture)
    paths = set(policy.applicable_paths)
    if model_config.mode == "Unsupervised":
        paths.add(policy.output_path)
    if architecture == "cnn" and (
        model_config.mode == "Supervised" or model_config.cnn_output_mode != "real_imag"
    ):
        paths.add("model.amp_activation")
    if architecture == "hybrid_resnet" and model_config.hybrid_skip_connections:
        paths.add("model.hybrid_skip_style")
    return paths


def _applicable_model_paths(model_config: ModelConfig) -> set[str]:
    paths = {
        "model.mode",
        "model.architecture",
        "model.C_model",
        "model.C_forward",
        "model.object_big",
        "model.probe_big",
        "model.physics_forward_mode",
        "model.loss_function",
        "model.amp_loss",
        "model.phase_loss",
        "model.rect_s1s2_trainable",
        "model.amplitude_physics_gain",
    }
    paths.update(_applicable_architecture_paths(model_config))
    paths.update(_applicable_probe_mask_paths(model_config))
    if model_config.object_big:
        paths.update({"model.offset", "model.training_patch_weighting"})
    if model_config.amp_loss is not None:
        paths.add("model.amp_loss_coeff")
    if model_config.phase_loss is not None:
        paths.add("model.phase_loss_coeff")
    return paths


def _applicable_training_paths(
    training_config: TrainingConfig,
    model_config: ModelConfig,
) -> set[str]:
    paths = {
        "training.learning_rate",
        "training.epochs",
        "training.batch_size",
        "training.scheduler",
        "training.accum_steps",
        "training.gradient_clip_val",
        "training.optimizer",
        "training.weight_decay",
        "training.log_grad_norm",
        "training.torch_loss_mode",
        "training.experiment_name",
        "training.notes",
        "training.model_name",
    }
    if _architecture_policy(model_config.architecture).fine_tuning_supported:
        paths.add("training.epochs_fine_tune")
        if training_config.epochs_fine_tune > 0:
            paths.add("training.fine_tune_gamma")
    if training_config.scheduler == "WarmupCosine":
        paths.update({"training.lr_warmup_epochs", "training.lr_min_ratio"})
    if training_config.scheduler == "ReduceLROnPlateau":
        paths.update(
            {
                "training.plateau_factor",
                "training.plateau_patience",
                "training.plateau_min_lr",
                "training.plateau_threshold",
            }
        )
    if training_config.gradient_clip_val is not None:
        paths.add("training.gradient_clip_algorithm")
    if training_config.optimizer == "sgd":
        paths.add("training.momentum")
    if training_config.optimizer in {"adam", "adamw"}:
        paths.update({"training.adam_beta1", "training.adam_beta2"})
    if training_config.log_grad_norm:
        paths.add("training.grad_norm_log_freq")
    if training_config.torch_loss_mode == "mae":
        paths.add("training.torch_mae_pred_l2_match_target")
    return paths


def _applicable_execution_paths(execution_config: PyTorchExecutionConfig) -> set[str]:
    paths = {
        "execution.accelerator",
        "execution.devices",
        "execution.strategy",
        "execution.deterministic",
        "execution.precision",
        "execution.num_workers",
        "execution.pin_memory",
        "execution.enable_progress_bar",
        "execution.enable_checkpointing",
    }
    if execution_config.num_workers > 0:
        paths.update({"execution.persistent_workers", "execution.prefetch_factor"})
    if execution_config.enable_checkpointing:
        paths.update(
            {
                "execution.checkpoint_save_top_k",
                "execution.checkpoint_monitor_metric",
                "execution.checkpoint_mode",
                "execution.early_stop_patience",
            }
        )
    return paths


def _applicable_config_paths(configs: _ConfigObjects) -> frozenset[str]:
    data, model, training, _inference, _datagen, execution = configs
    paths = _applicable_data_paths(data, model)
    paths.update(_applicable_model_paths(model))
    paths.update(_applicable_training_paths(training, model))
    paths.update(f"inference.{name}" for name in ALLOWLISTS["inference"])
    paths.update(_applicable_execution_paths(execution))
    return frozenset(paths)


def _training_device(accelerator: str) -> str:
    return "cuda" if accelerator in {"cuda", "gpu"} else accelerator


def _output_mode(config: ModelConfig) -> tuple[str, str]:
    policy = _architecture_policy(config.architecture)
    field_name = policy.output_path.removeprefix("model.")
    return policy.output_path, cast(str, getattr(config, field_name))


def _validate_loss_consistency(
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> None:
    expected_loss = "Poisson" if training_config.torch_loss_mode == "poisson" else "MAE"
    expected_nll = training_config.torch_loss_mode == "poisson"
    if (
        model_config.loss_function != expected_loss
        or training_config.nll is not expected_nll
    ):
        raise ConfigResolutionError(
            "training.torch_loss_mode, training.nll, and model.loss_function must agree"
        )
    if model_config.mode == "Supervised" and training_config.torch_loss_mode != "mae":
        raise ConfigResolutionError(
            "Supervised mode requires training.torch_loss_mode='mae'"
        )


def _validate_legacy_profile(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
) -> None:
    if model_config.physics_forward_mode != "amplitude":
        raise ConfigResolutionError(
            "legacy_v1 requires model.physics_forward_mode='amplitude'"
        )
    if inference_config.varpro_scaling:
        raise ConfigResolutionError("legacy_v1 requires inference.varpro_scaling=false")
    if model_config.rect_s1s2_trainable:
        raise ConfigResolutionError(
            "legacy_v1 requires model.rect_s1s2_trainable=false"
        )


def _validate_ci_controls(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
) -> None:
    mode_path, output_mode = _output_mode(model_config)
    requirements = {
        "inference.varpro_scaling": inference_config.varpro_scaling is True,
        "inference.patch_weighting": inference_config.patch_weighting == "probe",
        mode_path: output_mode == "real_imag",
        "model.rect_s1s2_trainable": model_config.rect_s1s2_trainable is True,
    }
    failed = [path for path, valid in requirements.items() if not valid]
    if failed:
        raise ConfigResolutionError(
            "CI scaling requires compatible values for " + ", ".join(failed)
        )


def _validate_loss_and_profile(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> tuple[ResolvedScaleContract, bool]:
    try:
        profile = resolve_scale_contract(
            data_config.scale_contract_version,
            data_config.measurement_domain,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigResolutionError(f"data profile contradiction: {exc}") from exc
    _validate_loss_consistency(model_config, training_config)
    active = (
        profile.version == "ci_intensity_v2"
        and profile.measurement_domain == "count_intensity"
        and model_config.mode == "Unsupervised"
        and model_config.physics_forward_mode == "rectangular_scaled"
        and training_config.torch_loss_mode == "poisson"
    )
    if model_config.physics_forward_mode == "rectangular_scaled" and not active:
        raise ConfigResolutionError(
            "model.physics_forward_mode='rectangular_scaled' requires the CI/count, "
            "Unsupervised, Poisson profile"
        )
    if profile.version == "legacy_v1":
        _validate_legacy_profile(model_config, inference_config)
    if active:
        _validate_ci_controls(model_config, inference_config)
    return profile, active


def _derive_execution_values(
    execution_values: dict[str, Any],
    training_config: TrainingConfig,
    model_config: ModelConfig,
) -> dict[str, Any]:
    result = dict(execution_values)
    for training_name, execution_name in TRAINING_TO_EXECUTION_ALIASES.items():
        result[execution_name] = getattr(training_config, training_name)
    return result


def _assert_aliases(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    execution_config: PyTorchExecutionConfig,
) -> None:
    for training_name, execution_name in TRAINING_TO_EXECUTION_ALIASES.items():
        if getattr(training_config, training_name) != getattr(
            execution_config, execution_name
        ):
            raise ConfigResolutionError(
                f"training.{training_name} alias does not match execution"
            )
    if training_config.device != _training_device(execution_config.accelerator):
        raise ConfigResolutionError(
            "training.device alias does not match execution.accelerator"
        )
    for execution_name, training_name in EXECUTION_TO_TRAINING_ALIASES.items():
        if getattr(execution_config, execution_name) != getattr(
            training_config, training_name
        ):
            raise ConfigResolutionError(
                f"training.{training_name} alias does not match execution"
            )
def _select_validated_dataset(
    dataset: ValidatedDataset | ValidatedDatasetBundle,
    dataset_id: str | None,
) -> ValidatedDataset:
    if isinstance(dataset, ValidatedDataset):
        if dataset_id is not None and dataset.descriptor.id != dataset_id:
            raise ConfigResolutionError(
                f"dataset.id={dataset_id!r} does not match validated dataset {dataset.descriptor.id!r}"
            )
        return dataset
    if not isinstance(dataset, ValidatedDatasetBundle):
        raise ConfigResolutionError(
            "dataset must be a sealed ValidatedDataset or ValidatedDatasetBundle"
        )
    if dataset_id is None:
        if len(dataset) != 1:
            raise ConfigResolutionError(
                "dataset.id is required when a validated bundle has multiple datasets"
            )
        return next(iter(dataset.values()))
    try:
        return dataset[dataset_id]
    except KeyError as exc:
        raise ConfigResolutionError(
            f"dataset.id={dataset_id!r} is not in the validated bundle"
        ) from exc


def _validate_dataset_for_configs(
    dataset: ValidatedDataset | ValidatedDatasetBundle,
    dataset_id: str | None,
    profile: ResolvedScaleContract,
    requested_n: int,
    requested_c: int,
    required_capabilities: Iterable[str],
) -> str:
    selected = _select_validated_dataset(dataset, dataset_id)
    detector_shape = selected.descriptor.detector_shape
    expected_shape = (requested_n, requested_n)
    if detector_shape != expected_shape:
        raise ConfigResolutionError(
            f"dataset {selected.descriptor.id!r} detector_shape={detector_shape} "
            f"does not match data.N={requested_n} expected shape {expected_shape}"
        )
    ci_profile = profile.version == "ci_intensity_v2"
    requirements = DatasetCompatibilityRequirements(
        scale_contract_version=cast(
            Literal["ci_intensity_v2", "legacy_v1"], profile.version
        ),
        measurement_domain=cast(
            Literal["count_intensity", "normalized_amplitude"],
            profile.measurement_domain,
        ),
        probe_calibration="count_amplitude" if ci_profile else "legacy_normalized",
        probe_gauge="physical_count_amplitude" if ci_profile else "legacy_normalized",
        requested_C=requested_c,
        required_capabilities=frozenset(required_capabilities),
    )
    try:
        validate_dataset_compatibility(selected, requirements)
    except DatasetCompatibilityError as exc:
        raise ConfigResolutionError(f"dataset compatibility failed: {exc}") from exc
    return selected.descriptor.id


def _supplied_config_paths(
    overrides: Mapping[str, Any],
    epochs: int | None,
) -> set[str]:
    supplied = set(overrides)
    if epochs is not None:
        supplied.add("training.epochs")
    supplied.discard("dataset.id")
    return supplied


def _require_claim_grade_discriminators(overrides: Mapping[str, Any]) -> None:
    required = set(_CLAIM_GRADE_DISCRIMINATORS)
    if (
        overrides.get("model.architecture") == "cnn"
        and overrides.get("model.mode") == "Unsupervised"
    ):
        required.add("model.cnn_output_mode")
    architecture = overrides.get("model.architecture")
    policy = (
        ARCHITECTURE_POLICIES.get(architecture)
        if isinstance(architecture, str)
        else None
    )
    if policy is not None and policy.fine_tuning_supported:
        required.add("training.epochs_fine_tune")
    missing = sorted(required - set(overrides))
    if missing:
        raise ConfigResolutionError(
            "claim-grade resolution requires explicit applicability discriminators; missing: "
            + ", ".join(missing)
        )


def _validate_supplied_applicability(
    supplied: set[str],
    applicable_paths: frozenset[str],
) -> None:
    non_applicable = sorted(supplied - applicable_paths)
    if non_applicable:
        path = non_applicable[0]
        raise ConfigResolutionError(
            f"override path {path!r} is non-applicable for the resolved configuration"
        )


def _require_explicit_paths(
    supplied: set[str],
    applicable_paths: frozenset[str],
) -> None:
    missing = sorted(
        applicable_paths - supplied - _BACKWARD_COMPATIBLE_OPTIONAL_EXPLICIT_PATHS
    )
    if missing:
        raise ConfigResolutionError(
            "claim-grade resolution requires explicit allowlisted paths; missing: "
            + ", ".join(missing)
        )


def _validate_claim_checkpoint_contract(
    execution_config: PyTorchExecutionConfig,
) -> None:
    if not execution_config.enable_checkpointing:
        raise ConfigResolutionError(
            "claim-grade execution.enable_checkpointing must be true"
        )
    if execution_config.checkpoint_save_top_k < 1:
        raise ConfigResolutionError(
            "claim-grade execution.checkpoint_save_top_k must be at least 1"
        )
    if execution_config.checkpoint_mode != "min":
        raise ConfigResolutionError(
            "claim-grade execution.checkpoint_mode must be 'min' for val_loss"
        )
    if execution_config.checkpoint_monitor_metric != "val_loss":
        raise ConfigResolutionError(
            "claim-grade execution.checkpoint_monitor_metric must be 'val_loss'"
        )


def _apply_runtime_owned_values(
    grouped: dict[str, dict[str, Any]],
    epochs: int | None,
    output_root: str | Path | None,
) -> None:
    if epochs is not None:
        grouped["training"]["epochs"] = _coerce_value(int, epochs, "epochs")
    if output_root is None:
        return
    if not isinstance(output_root, (str, Path)) or not str(output_root):
        raise ConfigResolutionError("output_root must be a nonempty path")
    grouped["training"]["output_dir"] = str(output_root)


_ConfigObjects = tuple[
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    DatagenConfig,
    PyTorchExecutionConfig,
]


def _construct_configs(grouped: dict[str, dict[str, Any]]) -> _ConfigObjects:
    _validate_execution_values(grouped["execution"])
    try:
        data_values = dict(grouped["data"])
        data_values.update(
            K=6,
            K_quadrant=30,
            neighbor_function="Nearest",
            min_neighbor_distance=0.0,
            max_neighbor_distance=3.0,
            scan_pattern="Isotropic",
        )
        data_config = DataConfig(**data_values)
        model_values = dict(grouped["model"])
        if model_values.get("physics_forward_mode", "amplitude") == "amplitude":
            model_values["rect_s1s2_trainable"] = False
        model_config = ModelConfig(**model_values)
        training_values = dict(grouped["training"])
        training_values["nll"] = (
            training_values.get("torch_loss_mode", "poisson") == "poisson"
        )
        training_values["framework"] = "Lightning"
        training_values["orchestrator"] = "Lightning"
        if not _architecture_policy(model_config.architecture).fine_tuning_supported:
            training_values["epochs_fine_tune"] = 0
            training_values["fine_tune_gamma"] = 0.1
        preliminary_training = TrainingConfig(**training_values)
        _validate_training(preliminary_training)
        inference_config = InferenceConfig(**grouped["inference"])
        execution_values = _derive_execution_values(
            grouped["execution"], preliminary_training, model_config
        )
        execution_config = PyTorchExecutionConfig(**execution_values)
        _validate_resolved_execution_platform(execution_config)
        training_values.update(
            device=_training_device(execution_config.accelerator),
            strategy=execution_config.strategy,
            n_devices=execution_config.devices,
            num_workers=execution_config.num_workers,
        )
        training_config = TrainingConfig(**training_values)
        datagen_config = DatagenConfig()
    except ConfigResolutionError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise ConfigResolutionError(f"config construction failed: {exc}") from exc
    return (
        data_config,
        model_config,
        training_config,
        inference_config,
        datagen_config,
        execution_config,
    )


def resolve_torch_configs(
    overrides: Mapping[str, Any],
    *,
    epochs: int | None = None,
    output_root: str | Path | None = None,
    dataset: ValidatedDataset | ValidatedDatasetBundle | None = None,
    dataset_id: str | None = None,
    required_capabilities: Iterable[str] = (),
    require_all_explicit: bool = False,
) -> ResolvedTorchConfigs:
    """Resolve one expanded override mapping into authoritative Torch configs."""
    requested_capabilities = _normalize_required_capabilities(required_capabilities)
    validated_keyword_dataset_id = (
        None if dataset_id is None else _validated_dataset_id(dataset_id, "dataset_id")
    )
    if dataset is None and requested_capabilities:
        raise ConfigResolutionError(
            "required_capabilities require a sealed validated dataset"
        )
    grouped, override_dataset_id = _coerce_overrides(overrides)
    if (
        grouped["model"].get("physics_forward_mode") == "amplitude"
        and grouped["model"].get("rect_s1s2_trainable") is True
    ):
        raise ConfigResolutionError(
            "amplitude forward mode requires model.rect_s1s2_trainable=false"
        )
    if validated_keyword_dataset_id is not None and override_dataset_id not in {
        None,
        validated_keyword_dataset_id,
    }:
        raise ConfigResolutionError(
            "dataset.id was assigned more than once with different values"
        )
    selected_dataset_id = validated_keyword_dataset_id or override_dataset_id
    if require_all_explicit:
        _require_claim_grade_discriminators(overrides)
    _apply_runtime_owned_values(grouped, epochs, output_root)
    configs = _construct_configs(grouped)
    (
        data_config,
        model_config,
        training_config,
        inference_config,
        datagen_config,
        execution_config,
    ) = configs
    if require_all_explicit:
        _validate_claim_checkpoint_contract(execution_config)
    applicable_paths = _applicable_config_paths(configs)
    supplied_paths = _supplied_config_paths(overrides, epochs)
    _validate_supplied_applicability(supplied_paths, applicable_paths)
    if require_all_explicit:
        _require_explicit_paths(supplied_paths, applicable_paths)

    _validate_data(data_config)
    _validate_model(model_config, data_config)
    _validate_training(training_config)
    _validate_inference(inference_config, data_config)
    _assert_aliases(model_config, training_config, execution_config)
    profile, active = _validate_loss_and_profile(
        data_config,
        model_config,
        training_config,
        inference_config,
    )
    if dataset is not None:
        selected_dataset_id = _validate_dataset_for_configs(
            dataset,
            selected_dataset_id,
            profile,
            data_config.N,
            data_config.C,
            requested_capabilities,
        )

    snapshot = _build_snapshot(
        *configs,
        profile,
        active,
        selected_dataset_id,
    )
    return ResolvedTorchConfigs(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        inference_config=inference_config,
        datagen_config=datagen_config,
        execution_config=execution_config,
        ci_scaling_active=active,
        profile=profile,
        dataset_id=selected_dataset_id,
        _snapshot_json=_canonical_json(snapshot),
        _snapshot_frozen=_freeze_json(snapshot),
        _identities=(
            id(data_config),
            id(model_config),
            id(training_config),
            id(inference_config),
            id(datagen_config),
            id(execution_config),
        ),
    )

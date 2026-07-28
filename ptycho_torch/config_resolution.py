"""Pure, phase-aware normalization for Torch configuration patches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from ptycho.config.config import TrainingConfig as PublicTrainingConfig
from ptycho_torch.execution_request import (
    NormalizedExecutionInput,
    OPTIMIZER_EXECUTION_COMPAT_FIELDS,
    ResolutionNotice,
)
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.object_compatibility import (
    resolve_torch_model_object_policy,
)
from ptycho_torch.scaling_contract import (
    resolve_scale_contract,
    validate_amplitude_physics_gain,
    validate_contract_coherence,
)


InputOwner = Literal[
    "data",
    "model",
    "training",
    "inference",
    "bridge",
    "derived_constraint",
    "execution_compatibility",
]

TRAINING_OWNER_FIELDS = frozenset(
    {
        "learning_rate",
        "scheduler",
        "lr_warmup_epochs",
        "lr_min_ratio",
        "plateau_factor",
        "plateau_patience",
        "plateau_min_lr",
        "plateau_threshold",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accum_steps",
        "optimizer",
        "momentum",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "epochs_fine_tune",
        "fine_tune_gamma",
        "stage_1_epochs",
        "stage_2_epochs",
        "stage_3_epochs",
        "physics_weight_schedule",
        "stage_3_lr_factor",
    }
)


@dataclass(frozen=True)
class InputRule:
    """One explicitly accepted phase input and its single owner."""

    canonical: str
    owner: InputOwner
    aliases: tuple[str, ...] = ()


def _snapshot_mutable_value(value: object) -> object:
    """Shallow-copy ordinary mutable containers without cloning tensors."""

    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, set):
        return set(value)
    return value


@dataclass(frozen=True)
class NormalizedPatch:
    """Immutable snapshots of normalized values and deferred resolution data."""

    phase: Literal["training", "inference"]
    values: Mapping[str, object]
    audit: Mapping[str, object]
    aliases: Mapping[str, tuple[str, ...]]
    notices: tuple[ResolutionNotice, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"training", "inference"}:
            raise ValueError(
                "phase must be 'training' or 'inference', "
                f"got {self.phase!r}"
            )
        object.__setattr__(
            self,
            "values",
            MappingProxyType(
                {
                    name: _snapshot_mutable_value(value)
                    for name, value in self.values.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "audit",
            MappingProxyType(
                {
                    name: _snapshot_mutable_value(value)
                    for name, value in self.audit.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "aliases",
            MappingProxyType(
                {
                    canonical: tuple(source_names)
                    for canonical, source_names in self.aliases.items()
                }
            ),
        )
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class TorchConfigBaseline:
    """Caller-owned phase baseline records used to construct fresh candidates."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig | None
    inference: InferenceConfig
    training_provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_provenance",
            MappingProxyType(dict(sorted(self.training_provenance.items()))),
        )


@dataclass(frozen=True)
class ProbeSizeObservation:
    """Read-only probe-size observation with deferred diagnostics."""

    value: int
    notices: tuple[ResolutionNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class TrainingObservations:
    """Read-only inputs observed by the training factory before resolution."""

    train_data_file: Path
    output_dir: Path
    inferred_probe_size: int
    photon_metadata: float | None = None
    notices: tuple[ResolutionNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_data_file", Path(self.train_data_file))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class InferenceObservations:
    """Read-only inputs observed by the inference factory before resolution."""

    model_path: Path
    test_data_file: Path
    output_dir: Path
    inferred_probe_size: int
    notices: tuple[ResolutionNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        object.__setattr__(self, "test_data_file", Path(self.test_data_file))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class ResolvedTrainingBundle:
    """Fresh, validated Torch training records and deterministic audit."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    bridge: Mapping[str, object]
    audit: Mapping[str, object]
    aliases: Mapping[str, tuple[str, ...]]
    notices: tuple[ResolutionNotice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bridge",
            MappingProxyType(
                {
                    name: _snapshot_mutable_value(value)
                    for name, value in self.bridge.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "audit",
            MappingProxyType(
                {
                    name: _snapshot_mutable_value(value)
                    for name, value in sorted(self.audit.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "aliases",
            MappingProxyType(
                {
                    name: tuple(sources)
                    for name, sources in sorted(self.aliases.items())
                }
            ),
        )
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class ResolvedInferenceBundle:
    """Fresh inference-time records without checkpoint model identity."""

    data: DataConfig
    model: ModelConfig
    inference: InferenceConfig
    bridge: Mapping[str, object]
    audit: Mapping[str, object]
    aliases: Mapping[str, tuple[str, ...]]
    notices: tuple[ResolutionNotice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bridge",
            MappingProxyType(
                {
                    name: _snapshot_mutable_value(value)
                    for name, value in self.bridge.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "audit",
            MappingProxyType(
                {
                    name: _snapshot_mutable_value(value)
                    for name, value in sorted(self.audit.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "aliases",
            MappingProxyType(
                {
                    name: tuple(sources)
                    for name, sources in sorted(self.aliases.items())
                }
            ),
        )
        object.__setattr__(self, "notices", tuple(self.notices))


_TRAINING_INPUTS_BY_OWNER: tuple[
    tuple[InputOwner, tuple[str, ...]], ...
] = (
    (
        "data",
        (
            "nphotons",
            "scale_contract_version",
            "measurement_domain",
            "N",
            "K",
            "K_quadrant",
            "n_subsample",
            "subsample_seed",
            "grid_size",
            "neighbor_function",
            "min_neighbor_distance",
            "max_neighbor_distance",
            "scan_pattern",
            "normalize",
            "probe_scale",
            "probe_normalize",
            "data_scaling",
            "phase_subtraction",
            "x_bounds",
            "y_bounds",
        ),
    ),
    (
        "model",
        (
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
            "spectral_bottleneck_blocks",
            "spectral_bottleneck_modes",
            "spectral_bottleneck_share_weights",
            "spectral_bottleneck_gate_init",
            "spectral_bottleneck_gate_mode",
            "generator_output_mode",
            "cnn_output_mode",
            "use_shared_decoder",
            "intensity_scale_trainable",
            "intensity_scale",
            "max_position_jitter",
            "num_datasets",
            "n_filters_scale",
            "amp_activation",
            "batch_norm",
            "probe_mask",
            "probe_mask_tensor",
            "probe_mask_sigma",
            "probe_mask_diameter",
            "edge_pad",
            "decoder_last_c_outer_fraction",
            "decoder_last_amp_channels",
            "use_legacy_decoder_channel_override",
            "eca_encoder",
            "cbam_encoder",
            "cbam_bottleneck",
            "cbam_decoder",
            "eca_decoder",
            "spatial_decoder",
            "decoder_spatial_kernel",
            "object_big",
            "object_layout",
            "training_canvas",
            "probe_big",
            "offset",
            "training_patch_weighting",
            "physics_forward_mode",
            "rect_s1s2_trainable",
            "rect_s1s2_init",
            "amplitude_physics_gain",
            "pad_object",
            "gaussian_smoothing_sigma",
            "amp_loss",
            "phase_loss",
            "amp_loss_coeff",
            "phase_loss_coeff",
        ),
    ),
    (
        "training",
        (
            "training_directories",
            "framework",
            "orchestrator",
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
            "stage_1_epochs",
            "stage_2_epochs",
            "stage_3_epochs",
            "physics_weight_schedule",
            "stage_3_lr_factor",
            "torch_loss_mode",
            "torch_mae_pred_l2_match_target",
            "experiment_name",
            "notes",
            "model_name",
            "test_data_file",
            "n_groups",
        ),
    ),
    (
        "inference",
        (
            "patch_weighting",
            "varpro_scaling",
            "log_patch_stats",
            "patch_stats_limit",
        ),
    ),
    (
        "derived_constraint",
        (
            "C",
            "C_model",
            "C_forward",
            "loss_function",
            "nll",
            "train_data_file",
            "output_dir",
        ),
    ),
)

_INFERENCE_INPUTS_BY_OWNER: tuple[
    tuple[InputOwner, tuple[str, ...]], ...
] = (
    (
        "data",
        (
            "N",
            "K",
            "grid_size",
            "probe_scale",
            "subsample_seed",
            "scale_contract_version",
            "measurement_domain",
        ),
    ),
    (
        "model",
        (
            "mode",
            "amp_activation",
            "n_filters_scale",
            "object_big",
            "object_layout",
            "training_canvas",
            "training_patch_weighting",
            "probe_big",
            "probe_mask",
            "probe_mask_tensor",
            "probe_mask_sigma",
            "probe_mask_diameter",
            "pad_object",
            "gaussian_smoothing_sigma",
        ),
    ),
    (
        "inference",
        (
            "batch_size",
            "patch_weighting",
            "varpro_scaling",
            "log_patch_stats",
            "patch_stats_limit",
        ),
    ),
    (
        "bridge",
        (
            "n_groups",
            "n_subsample",
        ),
    ),
    (
        "derived_constraint",
        (
            "C",
            "C_model",
            "C_forward",
            "model_path",
            "test_data_file",
            "output_dir",
        ),
    ),
)

_TRAINING_ALIASES = MappingProxyType(
    {
        "gridsize": "grid_size",
        "neighbor_count": "K",
        "model_type": "mode",
        "max_epochs": "epochs",
    }
)
_INFERENCE_ALIASES = MappingProxyType(
    {
        "gridsize": "grid_size",
        "neighbor_count": "K",
        "model_type": "mode",
    }
)


def _declare_rules(
    inputs_by_owner: tuple[tuple[InputOwner, tuple[str, ...]], ...],
    aliases: Mapping[str, str],
) -> tuple[InputRule, ...]:
    aliases_by_canonical: dict[str, list[str]] = {}
    for alias, canonical in aliases.items():
        aliases_by_canonical.setdefault(canonical, []).append(alias)
    return tuple(
        InputRule(
            canonical=name,
            owner=owner,
            aliases=tuple(aliases_by_canonical.get(name, ())),
        )
        for owner, names in inputs_by_owner
        for name in names
    )


TRAINING_INPUT_RULES = _declare_rules(
    _TRAINING_INPUTS_BY_OWNER,
    _TRAINING_ALIASES,
)
INFERENCE_INPUT_RULES = _declare_rules(
    _INFERENCE_INPUTS_BY_OWNER,
    _INFERENCE_ALIASES,
)

# These names intentionally match the current factory compatibility maps. The
# factory continues to own its existing definitions until its planned
# delegation; Task 1 only establishes the pure resolver-side declarations.
DEPRECATED_EXECUTION_MODEL_ALIASES = MappingProxyType(
    {
        "ffno_encoder_blocks": "fno_blocks",
        "ffno_encoder_modes": "fno_modes",
        "spectral_bottleneck_blocks": "spectral_bottleneck_blocks",
        "spectral_bottleneck_modes": "spectral_bottleneck_modes",
        "spectral_bottleneck_share_weights": (
            "spectral_bottleneck_share_weights"
        ),
        "spectral_bottleneck_gate_init": "spectral_bottleneck_gate_init",
        "spectral_bottleneck_gate_mode": "spectral_bottleneck_gate_mode",
    }
)
UNOWNED_EXECUTION_MODEL_ALIASES = frozenset(
    {
        "ffno_encoder_share_weights",
        "ffno_encoder_gate_init",
        "ffno_encoder_norm",
        "ffno_encoder_mlp_ratio",
    }
)
EXECUTION_OWNED_TRAINING_FIELDS = frozenset(
    {"device", "strategy", "n_devices", "num_workers"}
)


def _canonicalize_value(
    *,
    canonical: str,
    source_name: str,
    value: object,
) -> object:
    if source_name == "gridsize":
        return (value, value)
    if canonical == "grid_size" and isinstance(value, list):
        return tuple(value)
    return value


def _normalize_raw_patch(
    patch: Mapping[str, object],
    *,
    phase: Literal["training", "inference"],
    rules: tuple[InputRule, ...],
) -> tuple[
    dict[str, object],
    dict[str, tuple[str, ...]],
]:
    copied_patch = dict(patch)
    if phase == "training":
        execution_owned = sorted(
            set(copied_patch) & EXECUTION_OWNED_TRAINING_FIELDS
        )
        if execution_owned:
            raise ValueError(
                "training input field(s) "
                + ", ".join(execution_owned)
                + " are execution-owned; supply them through "
                "ExecutionRequest instead of the scientific configuration "
                "patch"
            )
    canonical_names = {rule.canonical for rule in rules}
    alias_names = {alias for rule in rules for alias in rule.aliases}
    unknown = sorted(set(copied_patch) - canonical_names - alias_names)
    if unknown:
        raise ValueError(
            f"unknown {phase} input field(s): " + ", ".join(unknown)
        )

    values: dict[str, object] = {}
    alias_provenance: dict[str, tuple[str, ...]] = {}
    for rule in rules:
        sources = (
            (rule.canonical,) + rule.aliases
            if rule.canonical in copied_patch
            else rule.aliases
        )
        supplied = [
            (
                source_name,
                _canonicalize_value(
                    canonical=rule.canonical,
                    source_name=source_name,
                    value=copied_patch[source_name],
                ),
            )
            for source_name in sources
            if source_name in copied_patch
        ]
        if not supplied:
            continue

        selected_source, selected_value = supplied[0]
        for source_name, value in supplied[1:]:
            if value != selected_value:
                alias_name = (
                    source_name
                    if source_name in rule.aliases
                    else selected_source
                )
                raise ValueError(
                    f"compatibility alias {alias_name!r} for canonical "
                    f"{rule.canonical!r} conflicts with the supplied "
                    f"canonical value"
                )
        values[rule.canonical] = selected_value
        used_aliases = tuple(
            source_name
            for source_name, _ in supplied
            if source_name in rule.aliases
        )
        if used_aliases:
            alias_provenance[rule.canonical] = used_aliases

    return values, alias_provenance


def normalize_training_patch(
    patch: Mapping[str, object],
    *,
    normalized_execution: NormalizedExecutionInput | None = None,
) -> NormalizedPatch:
    """Normalize one training patch without mutation, warnings, or defaults."""

    values, aliases = _normalize_raw_patch(
        patch,
        phase="training",
        rules=TRAINING_INPUT_RULES,
    )
    notices: tuple[ResolutionNotice, ...] = ()

    if normalized_execution is not None:
        if not isinstance(normalized_execution, NormalizedExecutionInput):
            raise TypeError(
                "normalized_execution must be a NormalizedExecutionInput "
                "or None"
            )
        explicit = normalized_execution.explicit_fields
        unowned = sorted(explicit & UNOWNED_EXECUTION_MODEL_ALIASES)
        if unowned:
            raise ValueError(
                "deprecated execution topology field(s) "
                + ", ".join(unowned)
                + " have no ModelConfig owner"
            )

        consumed: list[str] = []
        for source_name, target_name in (
            DEPRECATED_EXECUTION_MODEL_ALIASES.items()
        ):
            if source_name not in explicit:
                continue
            alias_value = normalized_execution.values[source_name]
            if target_name in values and values[target_name] != alias_value:
                raise ValueError(
                    f"deprecated execution alias {source_name!r} for "
                    f"canonical ModelConfig field {target_name!r} conflicts "
                    f"with the supplied canonical value"
                )
            values[target_name] = alias_value
            aliases[target_name] = tuple(
                (*aliases.get(target_name, ()), source_name)
            )
            consumed.append(source_name)

        if consumed:
            notices = (
                ResolutionNotice(
                    DeprecationWarning,
                    "PyTorchExecutionConfig topology fields are deprecated "
                    "aliases; pass their canonical values through "
                    "ModelConfig instead: "
                    + ", ".join(sorted(consumed)),
                ),
            )

    ordered_values = dict(sorted(values.items()))
    return NormalizedPatch(
        phase="training",
        values=ordered_values,
        audit=ordered_values,
        aliases={
            canonical: aliases[canonical]
            for canonical in sorted(aliases)
        },
        notices=notices,
    )


def normalize_inference_patch(
    patch: Mapping[str, object],
) -> NormalizedPatch:
    """Normalize one inference patch without mutation, warnings, or defaults."""

    values, aliases = _normalize_raw_patch(
        patch,
        phase="inference",
        rules=INFERENCE_INPUT_RULES,
    )
    ordered_values = dict(sorted(values.items()))
    return NormalizedPatch(
        phase="inference",
        values=ordered_values,
        audit=ordered_values,
        aliases={
            canonical: aliases[canonical]
            for canonical in sorted(aliases)
        },
        notices=(),
    )


def _fresh_config(config, changes: Mapping[str, object] | None = None):
    values = {
        field_info.name: _snapshot_mutable_value(
            getattr(config, field_info.name)
        )
        for field_info in fields(config)
    }
    values.update(
        {
            name: _snapshot_mutable_value(value)
            for name, value in (changes or {}).items()
        }
    )
    return type(config)(**values)


def training_factory_baseline(
    *,
    training_baseline: PublicTrainingConfig | TrainingConfig | None = None,
) -> TorchConfigBaseline:
    """Return the explicit historical training-factory baseline."""

    data = DataConfig(
        nphotons=1e9,
        N=64,
        C=1,
        grid_size=(1, 1),
    )
    model = resolve_torch_model_object_policy(
        ModelConfig(
            C_model=1,
            C_forward=1,
            loss_function="Poisson",
        )
    )
    training = TrainingConfig()
    training_provenance = {
        name: "torch_default"
        for name in TRAINING_OWNER_FIELDS
        if hasattr(training, name)
    }
    if training_baseline is not None:
        if not isinstance(
            training_baseline,
            (PublicTrainingConfig, TrainingConfig),
        ):
            raise TypeError(
                "training_baseline must be a public TrainingConfig, "
                "Torch TrainingConfig, or None"
            )
        available = {
            field_info.name for field_info in fields(training_baseline)
        }
        baseline_changes = {
            name: _snapshot_mutable_value(
                getattr(training_baseline, name)
            )
            for name in TRAINING_OWNER_FIELDS & available
        }
        training = _fresh_config(training, baseline_changes)
        training_provenance.update(
            {
                name: "training_baseline"
                for name in baseline_changes
            }
        )
    return TorchConfigBaseline(
        data=data,
        model=model,
        training=training,
        inference=InferenceConfig(),
        training_provenance=training_provenance,
    )


def inference_factory_baseline() -> TorchConfigBaseline:
    """Return the explicit historical inference-factory baseline."""

    data = DataConfig(
        N=64,
        C=1,
        K=4,
        grid_size=(1, 1),
        scale_contract_version="ci_intensity_v2",
        measurement_domain="count_intensity",
    )
    model = resolve_torch_model_object_policy(
        ModelConfig(
            C_model=1,
            C_forward=1,
        )
    )
    return TorchConfigBaseline(
        data=data,
        model=model,
        training=None,
        inference=InferenceConfig(batch_size=16),
    )


def observe_probe_size(data_file: Path) -> ProbeSizeObservation:
    """Inspect one NPZ without emitting the legacy fallback warning."""

    import numpy as np

    path = Path(data_file)
    fallback = 64

    def fallback_observation(message: str) -> ProbeSizeObservation:
        return ProbeSizeObservation(
            fallback,
            (
                ResolutionNotice(
                    UserWarning,
                    f"{message} Using fallback N={fallback}.",
                ),
            ),
        )

    try:
        with np.load(path, allow_pickle=False) as npz_data:
            if "probeGuess" not in npz_data:
                return fallback_observation(
                    f"probeGuess key missing from {path}."
                )
            probe = npz_data["probeGuess"]
            if probe.ndim < 2:
                return fallback_observation(
                    "probeGuess has invalid shape "
                    f"{probe.shape} (expected 2D square array)."
                )
            inferred = int(probe.shape[0])
            if probe.shape[0] != probe.shape[1]:
                return ProbeSizeObservation(
                    inferred,
                    (
                        ResolutionNotice(
                            UserWarning,
                            "probeGuess is non-square "
                            f"{probe.shape}. Using first dimension "
                            f"N={inferred}.",
                        ),
                    ),
                )
            return ProbeSizeObservation(inferred)
    except FileNotFoundError:
        return fallback_observation(f"Data file {path} not found.")
    except Exception as exc:
        return fallback_observation(
            f"Error reading probeGuess from {path}: {exc}."
        )


def _owned_values(
    normalized: NormalizedPatch,
    rules: tuple[InputRule, ...],
    owner: InputOwner,
) -> dict[str, object]:
    names = {rule.canonical for rule in rules if rule.owner == owner}
    return {
        name: value
        for name, value in normalized.values.items()
        if name in names
    }


def _derive_channel_count(grid_size: object) -> tuple[tuple[int, int], int]:
    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)
    if (
        not isinstance(grid_size, tuple)
        or len(grid_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in grid_size
        )
    ):
        raise ValueError(
            "grid_size must contain exactly two positive integers, "
            f"got {grid_size!r}"
        )
    normalized_grid = (grid_size[0], grid_size[1])
    return normalized_grid, normalized_grid[0] * normalized_grid[1]


def _check_derived_channel_constraints(
    normalized: NormalizedPatch,
    derived_channels: int,
) -> None:
    for field_name in ("C", "C_model", "C_forward"):
        if (
            field_name in normalized.values
            and normalized.values[field_name] != derived_channels
        ):
            raise ValueError(
                f"{field_name}={normalized.values[field_name]!r} conflicts "
                "with the channel count derived from grid_size "
                f"({derived_channels})"
            )


def _check_path_constraint(
    normalized: NormalizedPatch,
    field_name: str,
    observed: Path,
) -> None:
    if field_name not in normalized.values:
        return
    supplied = Path(normalized.values[field_name])
    if supplied != observed:
        raise ValueError(
            f"{field_name}={supplied} conflicts with authoritative factory "
            f"argument {observed}"
        )


def _require_positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{field_name} must be a positive integer, got {value!r}"
        )
    return value


def _require_positive_number(value: object, field_name: str) -> object:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(
            f"{field_name} must be positive and finite, got {value!r}"
        )
    return value


def _loss_identity(torch_loss_mode: object) -> tuple[str, bool]:
    if torch_loss_mode == "poisson":
        return "Poisson", True
    if torch_loss_mode == "mae":
        return "MAE", False
    raise ValueError(
        "torch_loss_mode must be 'poisson' or 'mae', "
        f"got {torch_loss_mode!r}"
    )


def _validate_training_owner_domains(config: TrainingConfig) -> None:
    schedulers = {
        "Default",
        "Exponential",
        "MultiStage",
        "Adaptive",
        "WarmupCosine",
        "ReduceLROnPlateau",
    }
    if config.scheduler not in schedulers:
        raise ValueError(
            f"scheduler must be one of {sorted(schedulers)}, "
            f"got {config.scheduler!r}"
        )
    clip_algorithms = {"norm", "value", "agc"}
    if config.gradient_clip_algorithm not in clip_algorithms:
        raise ValueError(
            "gradient_clip_algorithm must be one of "
            f"{sorted(clip_algorithms)}, "
            f"got {config.gradient_clip_algorithm!r}"
        )
    if (
        isinstance(config.learning_rate, bool)
        or not isinstance(config.learning_rate, (int, float))
        or not math.isfinite(float(config.learning_rate))
        or config.learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be finite and positive, "
            f"got {config.learning_rate!r}"
        )
    if (
        isinstance(config.accum_steps, bool)
        or not isinstance(config.accum_steps, int)
        or config.accum_steps <= 0
    ):
        raise ValueError(
            "accum_steps must be a positive integer, "
            f"got {config.accum_steps!r}"
        )
    if config.gradient_clip_val is not None and (
        isinstance(config.gradient_clip_val, bool)
        or not isinstance(config.gradient_clip_val, (int, float))
        or not math.isfinite(float(config.gradient_clip_val))
        or config.gradient_clip_val < 0
    ):
        raise ValueError(
            "gradient_clip_val must be finite and non-negative or None, "
            f"got {config.gradient_clip_val!r}"
        )


def _reject_derived_loss_conflicts(
    normalized: NormalizedPatch,
    loss_function: str,
    nll: bool,
) -> None:
    if (
        "loss_function" in normalized.values
        and normalized.values["loss_function"] != loss_function
    ):
        raise ValueError(
            "loss_function="
            f"{normalized.values['loss_function']!r} conflicts with "
            f"torch_loss_mode-derived loss {loss_function!r}"
        )
    if "nll" in normalized.values and normalized.values["nll"] is not nll:
        raise ValueError(
            f"nll={normalized.values['nll']!r} conflicts with "
            f"torch_loss_mode-derived nll={nll!r}"
        )


def _reject_half_configured_ci(
    normalized: NormalizedPatch,
    physics_forward_mode: str,
) -> None:
    if physics_forward_mode == "rectangular_scaled":
        return
    if normalized.values.get("rect_s1s2_init") == "data":
        raise ValueError(
            "rect_s1s2_init='data' is a CI-contract knob (one-batch s1/s2 "
            "calibration for the rectangular_scaled forward) but "
            f"physics_forward_mode resolved to {physics_forward_mode!r}, "
            "where it is silently ignored. Half-configured CI is fail-closed: "
            "use create_training_payload(..., profile='ci') (or --profile ci "
            "on the training CLI) for the coherent PtychoPINN-CI bundle, or "
            "drop rect_s1s2_init."
        )


def _prepare_object_policy_changes(
    normalized: NormalizedPatch,
    model_changes: dict[str, object],
) -> None:
    """Keep the unset-aware object policy atomic across a resolved baseline."""

    object_big_supplied = "object_big" in normalized.values
    layout_supplied = "object_layout" in normalized.values
    canvas_supplied = "training_canvas" in normalized.values
    if object_big_supplied and not (layout_supplied or canvas_supplied):
        model_changes["object_layout"] = None
        model_changes["training_canvas"] = None
    elif layout_supplied or canvas_supplied:
        model_changes.setdefault("object_big", None)
        model_changes.setdefault("object_layout", None)
        model_changes.setdefault("training_canvas", None)


def _validate_data_and_model(
    data: DataConfig,
    model: ModelConfig,
) -> None:
    _require_positive_integer(data.N, "N")
    _require_positive_number(data.nphotons, "nphotons")
    resolve_scale_contract(
        data.scale_contract_version,
        data.measurement_domain,
    )
    validate_amplitude_physics_gain(model)


def _required_group_count(
    normalized: NormalizedPatch,
    baseline_value: int | None,
) -> int:
    value = normalized.values.get("n_groups", baseline_value)
    if value is None:
        raise ValueError(
            "n_groups is required in the phase patch (no default)"
        )
    return _require_positive_integer(value, "n_groups")


def _resolve_training_owner_precedence(
    *,
    baseline: TrainingConfig,
    baseline_provenance: Mapping[str, str],
    normalized: NormalizedPatch,
    normalized_execution: NormalizedExecutionInput | None,
) -> tuple[TrainingConfig, dict[str, str]]:
    """Resolve the one TrainingConfig owner before derived/path joins."""

    changes: dict[str, object] = {}
    provenance = {
        name: baseline_provenance.get(name, "training_baseline")
        for name in TRAINING_OWNER_FIELDS
        if hasattr(baseline, name)
    }
    if normalized_execution is not None:
        for field_name in OPTIMIZER_EXECUTION_COMPAT_FIELDS:
            if field_name in normalized_execution.explicit_fields:
                changes[field_name] = normalized_execution.values[field_name]
                provenance[field_name] = "execution_compatibility"

    canonical_changes = _owned_values(
        normalized,
        TRAINING_INPUT_RULES,
        "training",
    )
    changes.update(canonical_changes)
    for field_name in canonical_changes:
        if field_name in TRAINING_OWNER_FIELDS:
            provenance[field_name] = "canonical_override"

    return _fresh_config(baseline, changes), provenance


def resolve_optimizer_ownership(
    *,
    training_baseline: PublicTrainingConfig | TrainingConfig | None,
    normalized_execution: NormalizedExecutionInput | None,
    canonical_training_patch: Mapping[str, object],
) -> tuple[TrainingConfig, dict[str, str]]:
    """Compatibility facade over the canonical training-owner resolver."""

    baseline = training_factory_baseline(
        training_baseline=training_baseline,
    )
    assert baseline.training is not None
    normalized = normalize_training_patch(canonical_training_patch)
    resolved, provenance = _resolve_training_owner_precedence(
        baseline=baseline.training,
        baseline_provenance=baseline.training_provenance,
        normalized=normalized,
        normalized_execution=normalized_execution,
    )
    _validate_training_owner_domains(resolved)
    return resolved, dict(sorted(provenance.items()))


def resolve_training_bundle(
    *,
    baseline: TorchConfigBaseline,
    normalized: NormalizedPatch,
    observations: TrainingObservations,
    normalized_execution: NormalizedExecutionInput | None = None,
) -> ResolvedTrainingBundle:
    """Construct and validate a fresh training candidate without side effects."""

    if not isinstance(baseline, TorchConfigBaseline):
        raise TypeError("baseline must be a TorchConfigBaseline")
    if baseline.training is None:
        raise ValueError("training resolution requires a training baseline")
    if not isinstance(normalized, NormalizedPatch):
        raise TypeError("normalized must be a NormalizedPatch")
    if normalized.phase != "training":
        raise ValueError(
            f"{normalized.phase} NormalizedPatch cannot be used for "
            "training resolution"
        )
    if not isinstance(observations, TrainingObservations):
        raise TypeError("observations must be TrainingObservations")
    if normalized_execution is not None and not isinstance(
        normalized_execution,
        NormalizedExecutionInput,
    ):
        raise TypeError(
            "normalized_execution must be a NormalizedExecutionInput or None"
        )

    _check_path_constraint(
        normalized,
        "train_data_file",
        observations.train_data_file,
    )
    _check_path_constraint(
        normalized,
        "output_dir",
        observations.output_dir,
    )

    data_changes = _owned_values(normalized, TRAINING_INPUT_RULES, "data")
    model_changes = _owned_values(normalized, TRAINING_INPUT_RULES, "model")
    inference_changes = _owned_values(
        normalized,
        TRAINING_INPUT_RULES,
        "inference",
    )
    _prepare_object_policy_changes(normalized, model_changes)

    grid_size, channels = _derive_channel_count(
        data_changes.get("grid_size", baseline.data.grid_size)
    )
    _check_derived_channel_constraints(normalized, channels)

    if "N" in normalized.values:
        resolved_N = normalized.values["N"]
        N_source = "explicit"
    else:
        resolved_N = observations.inferred_probe_size
        N_source = "observation"
    resolved_N = _require_positive_integer(resolved_N, "N")

    if "nphotons" in normalized.values:
        resolved_nphotons = normalized.values["nphotons"]
        nphotons_source = "explicit"
    elif observations.photon_metadata is not None:
        resolved_nphotons = observations.photon_metadata
        nphotons_source = "metadata"
    else:
        resolved_nphotons = baseline.data.nphotons
        nphotons_source = "declared_default"
    resolved_nphotons = _require_positive_number(
        resolved_nphotons,
        "nphotons",
    )

    data_changes.update(
        {
            "grid_size": grid_size,
            "C": channels,
            "N": resolved_N,
            "nphotons": resolved_nphotons,
        }
    )
    data = _fresh_config(baseline.data, data_changes)

    candidate_training, training_provenance = (
        _resolve_training_owner_precedence(
            baseline=baseline.training,
            baseline_provenance=baseline.training_provenance,
            normalized=normalized,
            normalized_execution=normalized_execution,
        )
    )

    n_groups = _required_group_count(
        normalized,
        baseline.training.n_groups,
    )
    candidate_training = _fresh_config(
        candidate_training,
        {
            "n_groups": n_groups,
            "train_data_file": str(observations.train_data_file),
            "output_dir": str(observations.output_dir),
            "test_data_file": (
                str(candidate_training.test_data_file)
                if candidate_training.test_data_file is not None
                else None
            ),
        },
    )
    loss_function, nll = _loss_identity(
        candidate_training.torch_loss_mode
    )
    _reject_derived_loss_conflicts(normalized, loss_function, nll)
    candidate_training = _fresh_config(
        candidate_training,
        {"nll": nll},
    )
    _validate_training_owner_domains(candidate_training)

    model_changes.update(
        {
            "C_model": channels,
            "C_forward": channels,
            "loss_function": loss_function,
        }
    )
    model = resolve_torch_model_object_policy(
        _fresh_config(baseline.model, model_changes)
    )
    inference = _fresh_config(baseline.inference, inference_changes)

    _reject_half_configured_ci(
        normalized,
        model.physics_forward_mode,
    )
    _validate_data_and_model(data, model)
    validate_contract_coherence(data, model, candidate_training)

    bridge: dict[str, object] = {
        "train_data_file": observations.train_data_file,
        "output_dir": observations.output_dir,
        "n_groups": n_groups,
        "nphotons": data.nphotons,
    }
    if candidate_training.test_data_file is not None:
        bridge["test_data_file"] = candidate_training.test_data_file
    if "n_subsample" in normalized.values:
        bridge["n_subsample"] = data.n_subsample
    if "subsample_seed" in normalized.values:
        bridge["subsample_seed"] = data.subsample_seed

    audit: dict[str, object] = dict(normalized.audit)
    audit.update(
        {
            name: getattr(candidate_training, name)
            for name in sorted(TRAINING_OWNER_FIELDS)
            if hasattr(candidate_training, name)
        }
    )
    audit.update(
        {
            "N": data.N,
            "N_source": N_source,
            "nphotons": data.nphotons,
            "nphotons_source": nphotons_source,
            "grid_size": data.grid_size,
            "C": data.C,
            "C_source": "derived:grid_size",
            "C_model": model.C_model,
            "C_model_source": "derived:grid_size",
            "C_forward": model.C_forward,
            "C_forward_source": "derived:grid_size",
            "loss_function": model.loss_function,
            "loss_function_source": "derived:torch_loss_mode",
            "nll": candidate_training.nll,
            "nll_source": "derived:torch_loss_mode",
            "amplitude_physics_gain": model.amplitude_physics_gain,
            "train_data_file": candidate_training.train_data_file,
            "output_dir": candidate_training.output_dir,
            "training_config_provenance": MappingProxyType(
                dict(sorted(training_provenance.items()))
            ),
        }
    )
    return ResolvedTrainingBundle(
        data=data,
        model=model,
        training=candidate_training,
        inference=inference,
        bridge=bridge,
        audit=audit,
        aliases=normalized.aliases,
        notices=(*normalized.notices, *observations.notices),
    )


def resolve_inference_bundle(
    *,
    baseline: TorchConfigBaseline,
    normalized: NormalizedPatch,
    observations: InferenceObservations,
) -> ResolvedInferenceBundle:
    """Construct inference runtime records without deriving ModelSpec identity."""

    if not isinstance(baseline, TorchConfigBaseline):
        raise TypeError("baseline must be a TorchConfigBaseline")
    if not isinstance(normalized, NormalizedPatch):
        raise TypeError("normalized must be a NormalizedPatch")
    if normalized.phase != "inference":
        raise ValueError(
            f"{normalized.phase} NormalizedPatch cannot be used for "
            "inference resolution"
        )
    if not isinstance(observations, InferenceObservations):
        raise TypeError("observations must be InferenceObservations")

    for field_name, observed in (
        ("model_path", observations.model_path),
        ("test_data_file", observations.test_data_file),
        ("output_dir", observations.output_dir),
    ):
        _check_path_constraint(normalized, field_name, observed)

    data_changes = _owned_values(normalized, INFERENCE_INPUT_RULES, "data")
    model_changes = _owned_values(
        normalized,
        INFERENCE_INPUT_RULES,
        "model",
    )
    _prepare_object_policy_changes(normalized, model_changes)
    inference_changes = _owned_values(
        normalized,
        INFERENCE_INPUT_RULES,
        "inference",
    )

    grid_size, channels = _derive_channel_count(
        data_changes.get("grid_size", baseline.data.grid_size)
    )
    _check_derived_channel_constraints(normalized, channels)

    if "N" in normalized.values:
        resolved_N = normalized.values["N"]
        N_source = "explicit"
    else:
        resolved_N = observations.inferred_probe_size
        N_source = "observation"
    resolved_N = _require_positive_integer(resolved_N, "N")
    data_changes.update(
        {
            "grid_size": grid_size,
            "C": channels,
            "N": resolved_N,
        }
    )
    data = _fresh_config(baseline.data, data_changes)

    model_changes.update(
        {
            "C_model": channels,
            "C_forward": channels,
        }
    )
    model = resolve_torch_model_object_policy(
        _fresh_config(baseline.model, model_changes)
    )
    inference = _fresh_config(baseline.inference, inference_changes)

    _validate_data_and_model(data, model)

    baseline_group_count = None
    n_groups = _required_group_count(normalized, baseline_group_count)
    bridge: dict[str, object] = {
        "model_path": observations.model_path,
        "test_data_file": observations.test_data_file,
        "output_dir": observations.output_dir,
        "n_groups": n_groups,
    }
    if "n_subsample" in normalized.values:
        bridge["n_subsample"] = normalized.values["n_subsample"]
    if "subsample_seed" in normalized.values:
        bridge["subsample_seed"] = data.subsample_seed

    audit: dict[str, object] = dict(normalized.audit)
    audit.update(
        {
            "N": data.N,
            "N_source": N_source,
            "grid_size": data.grid_size,
            "C": data.C,
            "C_source": "derived:grid_size",
            "C_model": model.C_model,
            "C_model_source": "derived:grid_size",
            "C_forward": model.C_forward,
            "C_forward_source": "derived:grid_size",
            "model_path": str(observations.model_path),
            "test_data_file": str(observations.test_data_file),
            "output_dir": str(observations.output_dir),
        }
    )
    return ResolvedInferenceBundle(
        data=data,
        model=model,
        inference=inference,
        bridge=bridge,
        audit=audit,
        aliases=normalized.aliases,
        notices=(*normalized.notices, *observations.notices),
    )

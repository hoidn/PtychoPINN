"""Pure, phase-aware normalization for Torch configuration patches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args, get_type_hints

from ptycho.config.config import TrainingConfig as PublicTrainingConfig
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.execution_request import (
    ResolutionNotice,
)
from ptycho_torch.object_compatibility import (
    resolve_torch_model_object_policy,
)
from ptycho_torch.rect_s1s2_initialization import (
    validate_rect_s1s2_initialization_mode,
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

# Closed value domain derived from the authoritative Literal annotation (a
# resolver-layer twin, not an era schema). Divergence is impossible by
# construction; the Literal is the single source.
SUPPORTED_TORCH_ARCHITECTURES = frozenset(
    get_args(get_type_hints(ModelConfig)["architecture"])
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


def _path_field_names(config_type) -> frozenset[str]:
    """Path-typed field names derived from the dataclass type hints."""
    hints = get_type_hints(config_type)
    return frozenset(
        f.name for f in fields(config_type) if hints.get(f.name) is Path
    )


TRAINING_OBSERVATION_PATH_FIELDS = _path_field_names(TrainingObservations)
INFERENCE_OBSERVATION_PATH_FIELDS = _path_field_names(InferenceObservations)


def _freeze_mapping_values(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            name: _snapshot_mutable_value(value)
            for name, value in sorted(values.items())
        }
    )


def _freeze_aliases(
    aliases: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(
        {
            name: tuple(sources)
            for name, sources in sorted(aliases.items())
        }
    )


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
        object.__setattr__(self, "bridge", _freeze_mapping_values(self.bridge))
        object.__setattr__(self, "audit", _freeze_mapping_values(self.audit))
        object.__setattr__(self, "aliases", _freeze_aliases(self.aliases))
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
        object.__setattr__(self, "bridge", _freeze_mapping_values(self.bridge))
        object.__setattr__(self, "audit", _freeze_mapping_values(self.audit))
        object.__setattr__(self, "aliases", _freeze_aliases(self.aliases))
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
            "neighbor_count",
            "K_quadrant",
            "n_raw_frames_selected",
            "subsample_seed",
            "gridsize",
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
            "ffno_encoder_blocks",
            "ffno_encoder_modes",
            "ffno_encoder_share_weights",
            "ffno_encoder_gate_init",
            "ffno_encoder_norm",
            "ffno_encoder_mlp_ratio",
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
            "training_groups",
        ),
    ),
    (
        "inference",
        (
            "middle_trim",
            "inference_batch_size",
            "experiment_number",
            "pad_eval",
            "window",
            "patch_weighting",
            "varpro_scaling",
            "log_patch_stats",
            "patch_stats_limit",
        ),
    ),
    (
        "bridge",
        (
            "enable_oversampling",
            "neighbor_pool_size",
            "sequential_sampling",
        ),
    ),
    (
        "derived_constraint",
        (
            "loss_function",
            "nll",
            "train_data_file",
            "output_dir",
        ),
    ),
)

_TRAINING_ALIASES = MappingProxyType(
    {
        "model_type": "mode",
        "max_epochs": "epochs",
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

_INFERENCE_INPUTS_BY_OWNER: tuple[
    tuple[InputOwner, tuple[str, ...]], ...
] = (
    (
        "data",
        (
            "N",
            "neighbor_count",
            "gridsize",
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
            "inference_groups",
            "n_raw_frames_selected",
        ),
    ),
    (
        "derived_constraint",
        (
            "model_path",
            "test_data_file",
            "output_dir",
        ),
    ),
)

_INFERENCE_ALIASES = MappingProxyType(
    {
        "model_type": "mode",
        # Documented external-contract fence (not a fallback): the legacy
        # inference-group-count spelling "training_groups" is permanently
        # accepted; specs/ptychodus_api_spec.md §4.6 and config_factory.py
        # docstrings historically named this key for the inference patch.
        # Normalization maps it to the canonical "inference_groups".
        "training_groups": "inference_groups",
    }
)
INFERENCE_INPUT_RULES = _declare_rules(
    _INFERENCE_INPUTS_BY_OWNER,
    _INFERENCE_ALIASES,
)

EXECUTION_OWNED_TRAINING_FIELDS = frozenset(
    {"device", "strategy", "n_devices", "num_workers"}
)

# --- Import-time ownership tripwires (W1) ----------------------------------
# The owner tables are the resolver's "retain manual" surface. A torch
# dataclass field must be owned by exactly one InputOwner: a name claimed by
# two owners would resolve through two precedence paths and silently corrupt
# the patch; a non-execution torch field missing from the training table would
# silently never resolve. Name-set tripwire idiom, precedented at
# ptycho_torch/model_spec.py:239-268.
_TORCH_RESOLVED_FIELDS = frozenset(
    field.name
    for config_type in (DataConfig, ModelConfig, TrainingConfig, InferenceConfig)
    for field in fields(config_type)
)


def _assert_single_ownership(
    table: tuple[tuple[InputOwner, tuple[str, ...]], ...],
    label: str,
) -> None:
    """Fail import if any name is owned by more than one InputOwner."""
    owner_by_name: dict[str, InputOwner] = {}
    for owner, names in table:
        for name in names:
            previous = owner_by_name.setdefault(name, owner)
            assert previous == owner, (
                f"{label}: {name!r} is owned by both "
                f"{previous!r} and {owner!r}"
            )


_assert_single_ownership(_TRAINING_INPUTS_BY_OWNER, "_TRAINING_INPUTS_BY_OWNER")
_assert_single_ownership(_INFERENCE_INPUTS_BY_OWNER, "_INFERENCE_INPUTS_BY_OWNER")

_TRAINING_RESOLVER_NAMES = frozenset(
    name for _, names in _TRAINING_INPUTS_BY_OWNER for name in names
)
assert _TORCH_RESOLVED_FIELDS - EXECUTION_OWNED_TRAINING_FIELDS <= _TRAINING_RESOLVER_NAMES, (
    "torch dataclass fields the training resolver can set are undeclared in "
    "_TRAINING_INPUTS_BY_OWNER: "
    f"{sorted(_TORCH_RESOLVED_FIELDS - EXECUTION_OWNED_TRAINING_FIELDS - _TRAINING_RESOLVER_NAMES)}"
)


def _canonicalize_value(
    *,
    canonical: str,
    source_name: str,
    value: object,
) -> object:
    return value



def _normalize_raw_patch(
    patch: Mapping[str, object],
    *,
    phase: Literal["training", "inference"],
    rules: tuple[InputRule, ...],
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
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
                    "canonical value"
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
) -> NormalizedPatch:
    """Normalize one canonical training patch without side effects."""

    values, aliases = _normalize_raw_patch(
        patch,
        phase="training",
        rules=TRAINING_INPUT_RULES,
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
        notices=(),
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
        gridsize=1,
    )
    model = resolve_torch_model_object_policy(
        ModelConfig(
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
        neighbor_count=4,
        gridsize=1,
        scale_contract_version="ci_intensity_v2",
        measurement_domain="count_intensity",
    )
    model = resolve_torch_model_object_policy(
        ModelConfig()
    )
    return TorchConfigBaseline(
        data=data,
        model=model,
        training=None,
        inference=InferenceConfig(batch_size=16),
    )


def observe_probe_size(data_file: Path) -> ProbeSizeObservation:
    """Inspect one NPZ without emitting the legacy fallback warning."""

    from ptycho.acquisition import inspect_probe_size

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
        return ProbeSizeObservation(inspect_probe_size(path))
    except FileNotFoundError:
        return fallback_observation(f"Data file {path} not found.")
    except Exception as exc:
        if "missing required key probeGuess" in str(exc):
            return fallback_observation(f"probeGuess key missing from {path}.")
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


def _derive_channel_count(gridsize: object) -> tuple[int, int]:
    """Derive ``(gridsize, channels)`` with ``channels = gridsize**2``.

    Audit note (Task 3.3): this is the SINGLE derivation site for the
    C-channel count in the Torch config resolver — both the training and
    inference resolution paths call it. ``C`` is never stored on a resolved
    owner; it is re-derived from ``gridsize`` at consumption. The stored
    ``C_model``/``C_forward`` family survives only in the artifact-era
    decode (``ModelSpec.from_payload`` and the ``artifact_schema`` upgrade),
    never on a current-era dataclass.
    """
    if (
        isinstance(gridsize, bool)
        or not isinstance(gridsize, int)
        or gridsize <= 0
    ):
        raise ValueError(
            f"gridsize must be a positive integer, got {gridsize!r}"
        )
    return gridsize, gridsize * gridsize


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


def _require_exact_nonnegative_integer(
    value: object,
    field_name: str,
) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field_name} must be an exact nonnegative integer, got {value!r}"
        )
    return value


def _require_exact_positive_integer(
    value: object,
    field_name: str,
) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{field_name} must be an exact positive integer, got {value!r}"
        )
    return value


def _require_exact_boolean(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field_name} must be an exact boolean, got {value!r}"
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


def _validate_model_domains(config: ModelConfig) -> None:
    if config.architecture not in SUPPORTED_TORCH_ARCHITECTURES:
        raise ValueError(
            "architecture must be one of "
            f"{sorted(SUPPORTED_TORCH_ARCHITECTURES)}, "
            f"got {config.architecture!r}"
        )
    validate_rect_s1s2_initialization_mode(config.rect_s1s2_init)


def _validate_inference_domains(config: InferenceConfig) -> None:
    for field_name in ("middle_trim", "window", "experiment_number"):
        _require_exact_nonnegative_integer(
            getattr(config, field_name),
            field_name,
        )
    _require_exact_positive_integer(config.batch_size, "batch_size")
    for field_name in ("pad_eval", "varpro_scaling", "log_patch_stats"):
        _require_exact_boolean(getattr(config, field_name), field_name)
    if config.patch_weighting not in {"uniform", "probe"}:
        raise ValueError(
            "patch_weighting must be 'uniform' or 'probe', "
            f"got {config.patch_weighting!r}"
        )
    if config.patch_stats_limit is not None:
        _require_exact_positive_integer(
            config.patch_stats_limit,
            "patch_stats_limit",
        )


def _validate_training_bridge_domains(
    bridge_values: Mapping[str, object],
    data: DataConfig,
) -> None:
    for field_name in ("enable_oversampling", "sequential_sampling"):
        if field_name in bridge_values:
            _require_exact_boolean(bridge_values[field_name], field_name)

    neighbor_pool_size = bridge_values.get("neighbor_pool_size")
    if neighbor_pool_size is not None:
        neighbor_pool_size = _require_exact_positive_integer(
            neighbor_pool_size,
            "neighbor_pool_size",
        )

    enable_oversampling = bridge_values.get("enable_oversampling", False)
    if enable_oversampling and data.gridsize != 1:
        effective_pool_size = (
            data.neighbor_count if neighbor_pool_size is None else neighbor_pool_size
        )
        derived_channels = data.gridsize * data.gridsize
        if effective_pool_size < derived_channels:
            raise ValueError(
                "enable_oversampling requires neighbor_pool_size or neighbor_count >= "
                f"derived C={derived_channels} for gridsize={data.gridsize}, "
                f"got {effective_pool_size}"
            )


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
    _require_positive_integer(config.epochs, "epochs")
    _require_positive_integer(config.batch_size, "batch_size")
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
    if normalized.values.get("rect_s1s2_init") == "dose_closure":
        raise ValueError(
            "rect_s1s2_init='dose_closure' is a CI-contract knob (closed-form "
            "gauge initialization for the rectangular_scaled forward) but "
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
    _validate_model_domains(model)
    resolve_scale_contract(
        data.scale_contract_version,
        data.measurement_domain,
    )
    validate_amplitude_physics_gain(model)


def _required_group_count(
    normalized: NormalizedPatch,
    baseline_value: int | None,
    *,
    key: str,
) -> int:
    value = normalized.values.get(key, baseline_value)
    if value is None:
        raise ValueError(
            f"{key} is required in the phase patch (no default)"
        )
    return _require_positive_integer(value, key)


def _resolve_training_owner_precedence(
    *,
    baseline: TrainingConfig,
    baseline_provenance: Mapping[str, str],
    normalized: NormalizedPatch,
) -> tuple[TrainingConfig, dict[str, str]]:
    """Resolve the one TrainingConfig owner before derived/path joins."""

    changes: dict[str, object] = {}
    provenance = {
        name: baseline_provenance.get(name, "training_baseline")
        for name in TRAINING_OWNER_FIELDS
        if hasattr(baseline, name)
    }
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


def resolve_training_bundle(
    *,
    baseline: TorchConfigBaseline,
    normalized: NormalizedPatch,
    observations: TrainingObservations,
) -> ResolvedTrainingBundle:
    """Construct and validate a fresh training candidate without side effects."""

    if not isinstance(baseline, TorchConfigBaseline):
        raise TypeError("baseline must be TorchConfigBaseline")
    if normalized.phase != "training":
        raise ValueError(
            f"{normalized.phase} NormalizedPatch cannot be used for "
            "training resolution"
        )
    if not isinstance(observations, TrainingObservations):
        raise TypeError("observations must be TrainingObservations")
    for field_name in sorted(TRAINING_OBSERVATION_PATH_FIELDS):
        _check_path_constraint(
            normalized,
            field_name,
            getattr(observations, field_name),
        )

    data_changes = _owned_values(normalized, TRAINING_INPUT_RULES, "data")
    model_changes = _owned_values(normalized, TRAINING_INPUT_RULES, "model")
    inference_changes = _owned_values(
        normalized,
        TRAINING_INPUT_RULES,
        "inference",
    )
    if "inference_batch_size" in inference_changes:
        inference_changes["batch_size"] = inference_changes.pop(
            "inference_batch_size"
        )
    _prepare_object_policy_changes(normalized, model_changes)

    gridsize, channels = _derive_channel_count(
        data_changes.get("gridsize", baseline.data.gridsize)
    )


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
            "gridsize": gridsize,
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
        )
    )

    training_groups = _required_group_count(
        normalized,
        baseline.training.training_groups,
        key="training_groups",
    )
    candidate_training = _fresh_config(
        candidate_training,
        {
            "training_groups": training_groups,
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
    _validate_inference_domains(inference)
    validate_contract_coherence(data, model, candidate_training)

    bridge: dict[str, object] = {
        "train_data_file": observations.train_data_file,
        "output_dir": observations.output_dir,
        "training_groups": training_groups,
        "nphotons": data.nphotons,
    }
    if candidate_training.test_data_file is not None:
        bridge["test_data_file"] = candidate_training.test_data_file
    if "n_raw_frames_selected" in normalized.values:
        bridge["train_raw_selection"] = data.n_raw_frames_selected

    if "subsample_seed" in normalized.values:
        bridge["subsample_seed"] = data.subsample_seed
    bridge_values = {
        name: normalized.values[name]
        for name in (
            "enable_oversampling",
            "neighbor_pool_size",
            "sequential_sampling",
        )
        if name in normalized.values
    }
    _validate_training_bridge_domains(bridge_values, data)
    for name in (
        "enable_oversampling",
        "neighbor_pool_size",
        "sequential_sampling",
    ):
        if name in bridge_values:
            bridge[name] = bridge_values[name]

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
            "gridsize": data.gridsize,
            "C": data.gridsize * data.gridsize,
            "C_source": "derived:gridsize",
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

    for field_name in sorted(INFERENCE_OBSERVATION_PATH_FIELDS):
        _check_path_constraint(
            normalized,
            field_name,
            getattr(observations, field_name),
        )

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

    gridsize, channels = _derive_channel_count(
        data_changes.get("gridsize", baseline.data.gridsize)
    )


    if "N" in normalized.values:
        resolved_N = normalized.values["N"]
        N_source = "explicit"
    else:
        resolved_N = observations.inferred_probe_size
        N_source = "observation"
    resolved_N = _require_positive_integer(resolved_N, "N")
    data_changes.update(
        {
            "gridsize": gridsize,
            "N": resolved_N,
        }
    )

    data = _fresh_config(baseline.data, data_changes)


    model = resolve_torch_model_object_policy(
        _fresh_config(baseline.model, model_changes)
    )
    inference = _fresh_config(baseline.inference, inference_changes)

    _validate_data_and_model(data, model)
    _validate_inference_domains(inference)

    inference_groups = _required_group_count(
        normalized, None, key="inference_groups"
    )
    bridge: dict[str, object] = {
        "model_path": observations.model_path,
        "test_data_file": observations.test_data_file,
        "output_dir": observations.output_dir,
        "inference_groups": inference_groups,
    }
    if "n_raw_frames_selected" in normalized.values:
        bridge["inference_raw_selection"] = normalized.values["n_raw_frames_selected"]

    if "subsample_seed" in normalized.values:
        bridge["subsample_seed"] = data.subsample_seed

    audit: dict[str, object] = dict(normalized.audit)
    audit.update(
        {
            "N": data.N,
            "N_source": N_source,
            "gridsize": data.gridsize,
            "C": data.gridsize * data.gridsize,
            "C_source": "derived:gridsize",
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

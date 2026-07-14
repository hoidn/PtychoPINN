"""Closed configuration namespace and gate contract for the bridge ladder.

Declares the flat ladder-config field set (rung-changeable fields plus the
invariants that hold the reference condition fixed), the SSIM gate value
object, the classifiable per-rung difference vocabulary, and the
per-group dataset-recipe delta scopes. Shared by the spec parser, the
execution flow, and the walk; imports nothing heavy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

from .runtime_errors import StudyRequestError
from .runtime_reference_spec import ExpectedDifference

RETAINED_GATE_POLICY = "retained_ssim_v1"
ABSOLUTE_GATE_POLICY = "absolute_ssim_delta_v1"
LOCKED_MAX_ABS_AMP_SSIM_DELTA = 0.02
LOCKED_MAX_ABS_PHASE_SSIM_DELTA = 0.01
GATE_POLICY = RETAINED_GATE_POLICY
#: Threshold provenance required before any rung may execute. Historical
#: retained policies may parse ``proposed*`` for review; the absolute policy
#: requires locked provenance at parse time.
THRESHOLDS_LOCKED = "locked"


def absolute_ssim_delta(current: float, control: float) -> float:
    """Absolute decimalized delta with exact configured-boundary behavior."""
    return float(abs(Decimal(str(float(current))) - Decimal(str(float(control)))))

#: Classifiable per-rung internal differences. They explain gaps but never
#: independently fail a rung once predeclared; unclassified ones fail closed.
LADDER_DIFFERENCE_IDS = frozenset(
    {"canvas_equivalence", "mask_equivalence", "effective_probe_identity"}
)
DIFFERENCE_CLASSIFICATIONS = frozenset(
    {"harmless", "performance_relevant", "comparison_invalidating"}
)

ENUM_CONFIG_FIELDS: dict[str, tuple[Any, ...]] = {
    "loader": ("dictionary", "mmap"),
    # Generic-loader scan-position bounds filter (rung-1 split): "off" is the
    # dictionary path's effective behavior (no filtering); "endpoint" is the
    # endpoint arm's (0.1, 0.9) range filter (see BOUNDS_FILTER_MODES).
    "mmap_bounds_filter": ("off", "endpoint"),
    # Which scale convention the generic loader applies: "loader" is the
    # explicitly selected Batch RMS + physics regime; "dictionary_parity"
    # engages normalize='None' (unit scalars), the canonical bridge convention
    # for simulator-owned normalized amplitudes. Inert under dictionary loading.
    "mmap_scale_convention": ("loader", "dictionary_parity"),
    "mmap_probe_batch_shape": ("modes", "dictionary_flat"),
    # Train-sampler regime of the generic loader (V3b sampler isolation):
    # "sequential" is the CURRENT real mmap behavior (SequentialSampler,
    # fixed raster order); "shuffled" injects a deterministically seeded
    # per-epoch RandomSampler through the same TensorDictDataLoader.
    "mmap_train_sampler": ("sequential", "shuffled"),
    "gated_evaluator": ("historical", "generic"),
    "training_patch_weighting": ("central_mask", "probe"),
    "measurement_domain": ("normalized_amplitude", "count_intensity"),
    "torch_loss_mode": ("mae", "poisson"),
    "count_scale_mode": ("off", "auto"),
    "scale_contract_version": ("legacy_v1", "ci_intensity_v2"),
    "physics_forward_mode": ("amplitude", "rectangular_scaled"),
}
_BOOL_FIELDS = frozenset(
    {"probe_normalize", "rect_s1s2_trainable", "varpro_scaling", "enable_checkpointing"}
)
_INT_FIELDS = frozenset(
    {
        "N",
        "gridsize",
        "position_crop_border",
        "seed",
        "epochs",
        "batch_size",
        "infer_batch_size",
        "plateau_patience",
        "fno_modes",
        "fno_width",
        "fno_blocks",
        "fno_cnn_blocks",
    }
)
_NUMBER_FIELDS = frozenset(
    {
        "learning_rate",
        "hybrid_encoder_conv_hidden_scale",
        "amplitude_physics_gain",
        "plateau_factor",
        "plateau_min_lr",
        "plateau_threshold",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
    }
)
_TEXT_FIELDS = frozenset(
    {
        "dataset",
        "architecture",
        "scheduler",
        "optimizer",
        "generator_output_mode",
        "probe_source",
        "logger_backend",
    }
)

#: Sealed-evidence migration whitelist (task-21c review I2, controller
#: decision): evidence sealed BEFORE a config field existed may be treated
#: as carrying that field's assumed default, ONLY for fields listed here.
#: Justification (reviewer-verified): the default branch of each listed
#: field is behaviorally equivalent to the pre-field path (for
#: mmap_scale_convention, "loader" explicitly emits DataConfig's default
#: normalize="Batch") and the staged dataset hashes recompute bit-identical,
#: so pre-field sealed runs are semantically identical to default-valued
#: configs. Every other field remains strict-equality fail-closed.
#: NOTE: the controller's decision message named the assumed default
#: "batch_derived"; the field's actual enum literal for the Batch-derived
#: regime is "loader" (see ENUM_CONFIG_FIELDS) — the semantic intent is
#: implemented with the real literal and flagged in the fix report.
MIGRATED_CONFIG_FIELDS: dict[str, Any] = {
    "mmap_scale_convention": "loader",
    "mmap_probe_batch_shape": "modes",
    # Same precedent: the "sequential" default is bit-preserving (no
    # injection engages), so pre-field sealed runs equal default configs.
    "mmap_train_sampler": "sequential",
}

#: Fields a rung group may change. Everything else is a ladder invariant that
#: holds the reference condition fixed across every rung (epoch budget, seeds,
#: architecture width, and optimizer are gate-4/Task-22 subjects, not rungs).
MUTABLE_CONFIG_FIELDS = frozenset(
    {
        "dataset",
        "loader",
        "mmap_bounds_filter",
        "mmap_scale_convention",
        "mmap_train_sampler",
        "mmap_probe_batch_shape",
        "gated_evaluator",
        "probe_normalize",
        "N",
        "gridsize",
        "position_crop_border",
        "training_patch_weighting",
        "measurement_domain",
        "torch_loss_mode",
        "count_scale_mode",
        "scale_contract_version",
        "physics_forward_mode",
        "amplitude_physics_gain",
        "rect_s1s2_trainable",
        "varpro_scaling",
    }
)
CONFIG_FIELDS = frozenset(
    set(ENUM_CONFIG_FIELDS)
    | _BOOL_FIELDS
    | _INT_FIELDS
    | _NUMBER_FIELDS
    | _TEXT_FIELDS
)
INVARIANT_CONFIG_FIELDS = CONFIG_FIELDS - MUTABLE_CONFIG_FIELDS

#: Grid-lines recipe fields a group may step between consecutive rung
#: datasets ("id" is always dataset-specific and ignored). Groups absent here
#: must keep the rung dataset identical to its predecessor's.
DATASET_RECIPE_DELTA_BY_GROUP: dict[str, frozenset[str]] = {
    "loader_schema": frozenset(),
    "probe_source_transform": frozenset(
        {
            "probe_archive",
            "probe_archive_sha256",
            "raw_probe_array_sha256",
            "transformed_probe_sha256",
            "probe_smoothing_sigma",
        }
    ),
    "detector_size": frozenset({"N", "transformed_probe_sha256"}),
    "measurement_domain_loss": frozenset(),
}
#: Dataset expression steps a group must perform (or preserve, when absent).
DATASET_EXPRESSION_STEP_BY_GROUP: dict[str, tuple[str, str]] = {
    "loader_schema": ("dictionary", "generic_amplitude"),
    "measurement_domain_loss": ("generic_amplitude", "generic_count_intensity"),
}
RECIPE_COMPARE_FIELDS = (
    "generator",
    "probe_archive_declared",
    "probe_archive_sha256",
    "raw_probe_array_sha256",
    "transformed_probe_sha256",
    "probe_scale_mode",
    "probe_smoothing_sigma",
    "set_phi",
    "N",
    "gridsize",
    "size",
    "offset",
    "outer_offset_train",
    "outer_offset_test",
    "nimgs_train",
    "nimgs_test",
    "nphotons",
)
RECIPE_FIELD_ALIASES = {"probe_archive_declared": "probe_archive"}

#: Endpoint-arm fields that are neither ladder config fields nor declared
#: residuals because their value is PROVABLY IDENTICAL between the ladder's
#: execution defaults and the endpoint declaration (no silent omissions).
#: Machine-checked in tests/studies/test_grid_lines_bridge_ladder.py
#: (test_endpoint_gap_fields_are_proven_inert) against both the endpoint spec
#: (hybrid_resnet_ci_compatibility.toml) and the live config defaults.
#: Proof notes:
#: - data.probe_scale: DataConfig default 4.0 == endpoint 4.0. It feeds the
#:   mmap loader's normalize_probe_like_tf, whose product is sealed per rung
#:   as effective_probe_sha256 — any deviation surfaces as evidence.
#: - data.x_bounds/y_bounds were REMOVED from this mapping (rung-1 split):
#:   value-equality proved constancy across mmap rungs, not inertness of the
#:   rung-0->1 step — the filter is behaviorally active only under the mmap
#:   loader. Bounds behavior is a ladder field now (mmap_bounds_filter,
#:   "endpoint" == the endpoint declaration, see
#:   runtime_ladder_mmap.BOUNDS_FILTER_MODES).
#: - model.offset: ModelConfig default 6 == endpoint 6 (active only for the
#:   C>1 grouping path, i.e., from the grouping rung on).
#: - model.amp_loss/phase_loss: endpoint "disabled" maps to None in the study
#:   configuration layer (configuration.py override sentinels); ModelConfig
#:   defaults are already None, so the declaration is a no-op.
#: - model.hybrid_encoder_spectral_hidden_scale: TorchRunnerConfig and
#:   ModelConfig defaults 1.0 == endpoint 1.0 (unlike the conv scale, which
#:   differs and is the declared architecture_width residual).
ENDPOINT_PROVEN_INERT_FIELDS: dict[str, Any] = {
    "data.probe_scale": 4.0,
    "model.offset": 6,
    "model.amp_loss": "disabled",
    "model.phase_loss": "disabled",
    "model.hybrid_encoder_spectral_hidden_scale": 1.0,
}

#: Ladder-config fields forwarded verbatim onto TorchRunnerConfig operands.
RUNNER_PASSTHROUGH_FIELDS = (
    "architecture",
    "N",
    "gridsize",
    "seed",
    "epochs",
    "batch_size",
    "infer_batch_size",
    "learning_rate",
    "scheduler",
    "plateau_factor",
    "plateau_patience",
    "plateau_min_lr",
    "plateau_threshold",
    "optimizer",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "generator_output_mode",
    "probe_source",
    "fno_modes",
    "fno_width",
    "fno_blocks",
    "fno_cnn_blocks",
    "enable_checkpointing",
    "logger_backend",
    "training_patch_weighting",
    "physics_forward_mode",
    "amplitude_physics_gain",
    "torch_loss_mode",
    "hybrid_encoder_conv_hidden_scale",
    "position_crop_border",
    "measurement_domain",
    "scale_contract_version",
    "count_scale_mode",
    "rect_s1s2_trainable",
)


@dataclass(frozen=True)
class LadderGate:
    """Predeclared, policy-specific SSIM rung gate."""

    policy: str
    threshold_provenance: str
    retained_amp_ssim_min_fraction: float | None = None
    retained_phase_ssim_min_fraction: float | None = None
    absolute_amp_ssim_floor: float | None = None
    max_abs_amp_ssim_delta: float | None = None
    max_abs_phase_ssim_delta: float | None = None

    def __post_init__(self) -> None:
        if self.policy not in {RETAINED_GATE_POLICY, ABSOLUTE_GATE_POLICY}:
            raise StudyRequestError(
                "gate.policy must be 'retained_ssim_v1' or "
                "'absolute_ssim_delta_v1'"
            )
        if self.policy == ABSOLUTE_GATE_POLICY:
            if self.threshold_provenance != THRESHOLDS_LOCKED:
                raise StudyRequestError(
                    "gate.threshold_provenance must be 'locked' for "
                    "absolute_ssim_delta_v1"
                )
            for name in (
                "max_abs_amp_ssim_delta",
                "max_abs_phase_ssim_delta",
            ):
                value = getattr(self, name)
                if (
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise StudyRequestError(
                        f"gate.{name} must be a finite nonnegative number"
                    )
            locked_values = {
                "max_abs_amp_ssim_delta": LOCKED_MAX_ABS_AMP_SSIM_DELTA,
                "max_abs_phase_ssim_delta": LOCKED_MAX_ABS_PHASE_SSIM_DELTA,
            }
            for name, locked_value in locked_values.items():
                if float(getattr(self, name)) != locked_value:
                    raise StudyRequestError(
                        f"gate.{name} must equal the locked value {locked_value}"
                    )
            if any(
                value is not None
                for value in (
                    self.retained_amp_ssim_min_fraction,
                    self.retained_phase_ssim_min_fraction,
                    self.absolute_amp_ssim_floor,
                )
            ):
                raise StudyRequestError(
                    "absolute_ssim_delta_v1 may not declare retained-SSIM fields"
                )
            return
        if self.threshold_provenance != THRESHOLDS_LOCKED and not str(
            self.threshold_provenance
        ).startswith("proposed"):
            raise StudyRequestError(
                "gate.threshold_provenance must be 'locked' or start with "
                "'proposed'"
            )
        for name in (
            "retained_amp_ssim_min_fraction",
            "retained_phase_ssim_min_fraction",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise StudyRequestError(f"gate.{name} must be in (0, 1]")
        if (
            not isinstance(self.absolute_amp_ssim_floor, Real)
            or not math.isfinite(float(self.absolute_amp_ssim_floor))
            or not -1.0 <= float(self.absolute_amp_ssim_floor) <= 1.0
        ):
            raise StudyRequestError("gate.absolute_amp_ssim_floor must be in [-1, 1]")
        if (
            self.max_abs_amp_ssim_delta is not None
            or self.max_abs_phase_ssim_delta is not None
        ):
            raise StudyRequestError(
                "retained_ssim_v1 may not declare absolute-delta fields"
            )

    @property
    def locked(self) -> bool:
        return self.threshold_provenance == THRESHOLDS_LOCKED


def config_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Fields whose values differ between two resolved ladder configs."""
    if set(previous) != set(current):
        raise StudyRequestError("ladder configs must share the closed field set")
    return {
        field: (previous[field], current[field])
        for field in previous
        if previous[field] != current[field]
    }


def closed_table(
    value: Any, *, path: str, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyRequestError(f"{path} must be a table")
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise StudyRequestError(f"{path} has unexpected keys {unexpected}")
    missing = sorted(required - set(value))
    if missing:
        raise StudyRequestError(f"{path} is missing required keys {missing}")
    return dict(value)


def required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise StudyRequestError(f"{path} must be nonempty text")
    return value


def validate_config_value(field: str, value: Any, path: str) -> None:
    if field in ENUM_CONFIG_FIELDS:
        if value not in ENUM_CONFIG_FIELDS[field]:
            raise StudyRequestError(
                f"{path}.{field} must be one of {ENUM_CONFIG_FIELDS[field]}"
            )
    elif field in _BOOL_FIELDS:
        if type(value) is not bool:
            raise StudyRequestError(f"{path}.{field} must be boolean")
    elif field in _INT_FIELDS:
        if type(value) is not int or value < 0:
            raise StudyRequestError(f"{path}.{field} must be a nonnegative integer")
    elif field in _NUMBER_FIELDS:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise StudyRequestError(f"{path}.{field} must be numeric")
        if field == "amplitude_physics_gain" and float(value) <= 0.0:
            raise StudyRequestError(
                f"{path}.amplitude_physics_gain must be positive"
            )
    elif field in _TEXT_FIELDS:
        required_text(value, f"{path}.{field}")
    else:  # pragma: no cover - guarded by the closed-table check
        raise StudyRequestError(f"{path}.{field} is not a ladder config field")


def parse_config(value: Any, path: str) -> dict[str, Any]:
    table = closed_table(
        value, path=path, allowed=set(CONFIG_FIELDS), required=set(CONFIG_FIELDS)
    )
    for field, item in table.items():
        validate_config_value(field, item, path)
    return table


def validate_dataset_step(
    rung_path: str, group: str, previous: Any, current: Any
) -> None:
    """Constrain consecutive rung dataset recipes to the group's scope."""
    allowed = DATASET_RECIPE_DELTA_BY_GROUP.get(group)
    if allowed is None:
        if current.id != previous.id:
            raise StudyRequestError(
                f"{rung_path}: group {group!r} may not switch the rung dataset"
            )
        return
    changed = {
        RECIPE_FIELD_ALIASES.get(field, field)
        for field in RECIPE_COMPARE_FIELDS
        if getattr(previous.recipe, field) != getattr(current.recipe, field)
    }
    if not changed <= set(allowed):
        raise StudyRequestError(
            f"{rung_path}: dataset recipe changes {sorted(changed - set(allowed))} "
            f"exceed the {group!r} group's dataset scope"
        )
    expected = DATASET_EXPRESSION_STEP_BY_GROUP.get(group)
    if expected is not None:
        if (previous.expression, current.expression) != expected:
            raise StudyRequestError(
                f"{rung_path}: group {group!r} must step the dataset expression "
                f"{expected[0]!r} -> {expected[1]!r}"
            )
    elif current.expression != previous.expression:
        raise StudyRequestError(
            f"{rung_path}: group {group!r} may not change the dataset expression"
        )


def parse_ladder_expected_differences(
    value: Any, path: str
) -> dict[str, ExpectedDifference]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise StudyRequestError(f"{path} must be a table")
    differences: dict[str, ExpectedDifference] = {}
    for field, entry in value.items():
        if field not in LADDER_DIFFERENCE_IDS:
            raise StudyRequestError(
                f"{path}.{field} is not a classifiable ladder difference"
            )
        table = closed_table(
            entry,
            path=f"{path}.{field}",
            allowed={"classification", "justification"},
            required={"classification", "justification"},
        )
        classification = table["classification"]
        if classification not in DIFFERENCE_CLASSIFICATIONS:
            raise StudyRequestError(
                f"{path}.{field}.classification must be one of "
                + ", ".join(sorted(DIFFERENCE_CLASSIFICATIONS))
            )
        justification = required_text(
            table["justification"], f"{path}.{field}.justification"
        )
        differences[field] = ExpectedDifference(field, classification, justification)
    return differences

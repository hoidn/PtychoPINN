"""Torch artifact eras and section upgrade rules."""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any, Mapping

from ptycho_torch.config_params import DataConfig, InferenceConfig, TrainingConfig


TORCH_ARTIFACT_BACKEND = "pytorch"
ARTIFACT_SCHEMA_V1_VERSION = "torch-artifact-v1"
ARTIFACT_SCHEMA_V2_VERSION = "torch-artifact-v2"
ARTIFACT_SCHEMA_V3_VERSION = "torch-artifact-v3"
ARTIFACT_SCHEMA_V4_VERSION = "torch-artifact-v4"
ARTIFACT_SCHEMA_V5_VERSION = "torch-artifact-v5"
CURRENT_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_V5_VERSION
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = (
    ARTIFACT_SCHEMA_V1_VERSION,
    ARTIFACT_SCHEMA_V2_VERSION,
    ARTIFACT_SCHEMA_V3_VERSION,
    ARTIFACT_SCHEMA_V4_VERSION,
    ARTIFACT_SCHEMA_V5_VERSION,
)
RUNTIME_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = (
    ARTIFACT_SCHEMA_V3_VERSION,
    ARTIFACT_SCHEMA_V4_VERSION,
    ARTIFACT_SCHEMA_V5_VERSION,
)


ARTIFACT_V1_DATA_FIELDS = (
    "nphotons",
    "scale_contract_version",
    "measurement_domain",
    "N",
    "C",
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
)
ARTIFACT_V1_TRAINING_FIELDS = (
    "training_directories",
    "nll",
    "device",
    "strategy",
    "n_devices",
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
    "num_workers",
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
    "output_dir",
    "train_data_file",
    "test_data_file",
    "n_groups",
)
ARTIFACT_V1_INFERENCE_FIELDS = (
    "middle_trim",
    "batch_size",
    "experiment_number",
    "pad_eval",
    "window",
    "patch_weighting",
    "varpro_scaling",
    "log_patch_stats",
    "patch_stats_limit",
)

# Frozen v2-era section schemas: identical to v1 in all three sections, so
# they alias the v1 literals (still independent of current dataclass
# reflection; the v2 fixtures pin these tuples). The upgrade shims carry the
# inter-era deltas.
ARTIFACT_V2_DATA_FIELDS = ARTIFACT_V1_DATA_FIELDS
ARTIFACT_V2_TRAINING_FIELDS = ARTIFACT_V1_TRAINING_FIELDS
ARTIFACT_V2_INFERENCE_FIELDS = ARTIFACT_V1_INFERENCE_FIELDS

# Declared v3-era section schemas. Data drops stored C/grid_size and splits
# n_subsample into n_raw_frames_selected + groups_per_center; gridsize is
# derived from grid_size in the upgrade functions. Data stays a literal (it
# encodes the v2->v3 rename history); training and inference are unchanged
# from v2 and alias the frozen literals.
ARTIFACT_V3_DATA_FIELDS = (
    "nphotons",
    "scale_contract_version",
    "measurement_domain",
    "N",
    "gridsize",
    "K",
    "K_quadrant",
    "n_raw_frames_selected",
    "groups_per_center",
    "subsample_seed",
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
)
ARTIFACT_V3_TRAINING_FIELDS = ARTIFACT_V2_TRAINING_FIELDS
ARTIFACT_V3_INFERENCE_FIELDS = ARTIFACT_V1_INFERENCE_FIELDS

# Declared v4-era section schemas. Data renames K -> neighbor_count and drops
# groups_per_center (inference meaning now a runtime argument, never persisted);
# training renames n_groups -> training_groups. Inference identical across eras.
ARTIFACT_V4_DATA_FIELDS = (
    "nphotons",
    "scale_contract_version",
    "measurement_domain",
    "N",
    "gridsize",
    "neighbor_count",
    "K_quadrant",
    "n_raw_frames_selected",
    "subsample_seed",
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
)
ARTIFACT_V4_TRAINING_FIELDS = (
    "training_directories",
    "nll",
    "device",
    "strategy",
    "n_devices",
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
    "num_workers",
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
    "output_dir",
    "train_data_file",
    "test_data_file",
    "training_groups",
)
ARTIFACT_V4_INFERENCE_FIELDS = ARTIFACT_V1_INFERENCE_FIELDS

# Declared v5-era section schemas (centered-nearest grouping). Data drops the
# K-choose-C oversampling family (K_quadrant, neighbor_function,
# min/max_neighbor_distance, scan_pattern) and carries group_padding_step
# instead. Training and inference are unchanged from v4 and alias the frozen
# literals.
ARTIFACT_V5_DATA_FIELDS: tuple[str, ...] = (
    "nphotons",
    "scale_contract_version",
    "measurement_domain",
    "N",
    "gridsize",
    "neighbor_count",
    "group_padding_step",
    "n_raw_frames_selected",
    "subsample_seed",
    "normalize",
    "probe_scale",
    "probe_normalize",
    "data_scaling",
    "phase_subtraction",
    "x_bounds",
    "y_bounds",
)
ARTIFACT_V5_TRAINING_FIELDS: tuple[str, ...] = ARTIFACT_V4_TRAINING_FIELDS
ARTIFACT_V5_INFERENCE_FIELDS: tuple[str, ...] = ARTIFACT_V1_INFERENCE_FIELDS


def _derive_fields(config_type) -> tuple[str, ...]:
    """Persisted-dataclass field names in declaration order (the v5 wire shape)."""
    return tuple(field.name for field in fields(config_type))


def _field_set(values: tuple[str, ...]) -> set[str]:
    return set(values)


# Derived current-era (v5) section schemas (W2.2): the only writable era's wire
# shape is exactly the persisted torch dataclass fields. Runtime encode/decode
# consumes these derived tuples; the v4 DATA/TRAINING literals above are frozen
# (they encode the pre-centered wire) and stay independent of dataclass
# reflection.
_DERIVED_V5_DATA_FIELDS = _derive_fields(DataConfig)
_DERIVED_V5_TRAINING_FIELDS = _derive_fields(TrainingConfig)
_DERIVED_V5_INFERENCE_FIELDS = _derive_fields(InferenceConfig)

# W1.1 import-time totality tripwire, re-expressed as the W2.2 transition
# assert (derived == literal): the v5 tuples guard the only writable era, so
# their content must exactly equal the persisted torch dataclass fields. A
# dataclass field addition/rename fails here at import instead of silently
# drifting the decode schema.
assert _field_set(_DERIVED_V5_DATA_FIELDS) == _field_set(ARTIFACT_V5_DATA_FIELDS), (
    "ARTIFACT_V5_DATA_FIELDS drifted from DataConfig fields: "
    f"extra={sorted(_field_set(ARTIFACT_V5_DATA_FIELDS) - _field_set(_DERIVED_V5_DATA_FIELDS))} "
    f"missing={sorted(_field_set(_DERIVED_V5_DATA_FIELDS) - _field_set(ARTIFACT_V5_DATA_FIELDS))}"
)
assert _field_set(_DERIVED_V5_TRAINING_FIELDS) == _field_set(
    ARTIFACT_V5_TRAINING_FIELDS
), (
    "ARTIFACT_V5_TRAINING_FIELDS drifted from TrainingConfig fields: "
    f"extra={sorted(_field_set(ARTIFACT_V5_TRAINING_FIELDS) - _field_set(_DERIVED_V5_TRAINING_FIELDS))} "
    f"missing={sorted(_field_set(_DERIVED_V5_TRAINING_FIELDS) - _field_set(ARTIFACT_V5_TRAINING_FIELDS))}"
)
assert _field_set(_DERIVED_V5_INFERENCE_FIELDS) == _field_set(
    ARTIFACT_V5_INFERENCE_FIELDS
), (
    "ARTIFACT_V5_INFERENCE_FIELDS drifted from InferenceConfig fields: "
    f"extra={sorted(_field_set(ARTIFACT_V5_INFERENCE_FIELDS) - _field_set(_DERIVED_V5_INFERENCE_FIELDS))} "
    f"missing={sorted(_field_set(_DERIVED_V5_INFERENCE_FIELDS) - _field_set(ARTIFACT_V5_INFERENCE_FIELDS))}"
)


def _config_field_names(config_type) -> set[str]:
    return set(_derive_fields(config_type))


def _require_exact_config_payload(
    payload: Mapping[str, Any],
    config_type,
    *,
    era: str,
    section: str,
    expected_fields=None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{era} {section} must be a mapping")
    expected = (
        _config_field_names(config_type)
        if expected_fields is None
        else set(expected_fields)
    )
    received = set(payload)
    if received != expected:
        raise ValueError(
            f"{era} {section} field set is not exact; "
            f"missing={sorted(expected - received)}, unknown={sorted(received - expected)}"
        )
    values = copy.deepcopy(dict(payload))
    if config_type is DataConfig:
        for name in ("grid_size", "x_bounds", "y_bounds"):
            if name in values:
                values[name] = tuple(values[name])
    return values


_DATA_FIELDS_BY_ERA = {
    ARTIFACT_SCHEMA_V1_VERSION: ARTIFACT_V1_DATA_FIELDS,
    ARTIFACT_SCHEMA_V2_VERSION: ARTIFACT_V2_DATA_FIELDS,
    ARTIFACT_SCHEMA_V3_VERSION: ARTIFACT_V3_DATA_FIELDS,
    ARTIFACT_SCHEMA_V4_VERSION: ARTIFACT_V4_DATA_FIELDS,
    ARTIFACT_SCHEMA_V5_VERSION: _DERIVED_V5_DATA_FIELDS,
}
_TRAINING_FIELDS_BY_ERA = {
    ARTIFACT_SCHEMA_V1_VERSION: ARTIFACT_V1_TRAINING_FIELDS,
    ARTIFACT_SCHEMA_V2_VERSION: ARTIFACT_V2_TRAINING_FIELDS,
    ARTIFACT_SCHEMA_V3_VERSION: ARTIFACT_V3_TRAINING_FIELDS,
    ARTIFACT_SCHEMA_V4_VERSION: ARTIFACT_V4_TRAINING_FIELDS,
    ARTIFACT_SCHEMA_V5_VERSION: _DERIVED_V5_TRAINING_FIELDS,
}
_INFERENCE_FIELDS_BY_ERA = {
    ARTIFACT_SCHEMA_V1_VERSION: ARTIFACT_V1_INFERENCE_FIELDS,
    ARTIFACT_SCHEMA_V2_VERSION: ARTIFACT_V2_INFERENCE_FIELDS,
    ARTIFACT_SCHEMA_V3_VERSION: ARTIFACT_V3_INFERENCE_FIELDS,
    ARTIFACT_SCHEMA_V4_VERSION: ARTIFACT_V4_INFERENCE_FIELDS,
    ARTIFACT_SCHEMA_V5_VERSION: _DERIVED_V5_INFERENCE_FIELDS,
}


def validate_legacy_channel_faithfulness(
    data_section: Mapping[str, Any],
    model_fields: Mapping[str, Any],
    *,
    era: str,
) -> None:
    """Require legacy/v1/v2 stored channel identity to agree with gridsize.

    The v1/v2 wire stores the channel count redundantly (data ``C`` and model
    ``C_model``/``C_forward``) beside ``grid_size``. Faithful payloads agree on
    ``gridsize**2``; disagreement means the payload is internally inconsistent
    and must be rejected, never silently reinterpreted by dropping the stored
    values.
    """
    grid_h, grid_w = data_section["grid_size"]
    if grid_h != grid_w:
        raise ValueError(
            f"non-square grid_size=({grid_h}, {grid_w}) cannot upgrade to a v3 gridsize"
        )
    derived = grid_h * grid_h
    stored_c = data_section["C"]
    if stored_c != derived:
        raise ValueError(
            f"{era} data section channel identity is unfaithful: stored "
            f"C={stored_c} conflicts with derived gridsize**2={derived}"
        )
    missing = [k for k in ("C_model", "C_forward") if k not in model_fields]
    if missing:
        raise ValueError(
            f"{era} model section lacks channel identity fields {missing}; "
            "cannot validate faithfulness against the data section"
        )
    c_model = model_fields["C_model"]
    c_forward = model_fields["C_forward"]
    if c_model != derived:
        raise ValueError(
            f"{era} model section channel identity is unfaithful: stored "
            f"C_model={c_model} conflicts with derived gridsize**2={derived}"
        )
    if c_forward != derived:
        raise ValueError(
            f"{era} model section channel identity is unfaithful: stored "
            f"C_forward={c_forward} conflicts with derived gridsize**2={derived}"
        )


def _project_legacy_data_to_runtime(values: dict[str, Any]) -> dict[str, Any]:
    """Map a v1/v2 data section onto the current (v5) DataConfig."""
    grid_h, grid_w = values.pop("grid_size")
    if grid_h != grid_w:
        raise ValueError(
            f"non-square grid_size=({grid_h}, {grid_w}) cannot upgrade to a v5 gridsize"
        )
    values["gridsize"] = grid_h
    values.pop("C")
    values["n_raw_frames_selected"] = values.pop("n_subsample")
    values["neighbor_count"] = values.pop("K")
    return values


def _upgrade_pre_centered_data(era: str, values: dict[str, Any]) -> None:
    """Convert a pre-v5 data section to the centered-nearest era in place.

    The centered-nearest grouping contract supports exactly one derived
    channel (gridsize**2 == 1). Multi-channel pre-v5 payloads cannot be
    projected faithfully and must be retrained under torch-artifact-v5.
    Single-channel payloads drop the retired K-choose-C policy fields and map
    ``max_neighbor_distance`` onto ``group_padding_step``.
    """
    gridsize = values["gridsize"]
    derived = gridsize * gridsize
    if derived != 1:
        raise ValueError(
            f"{era} data section derives C={derived} from gridsize={gridsize}; "
            "the centered-nearest grouping contract supports exactly one "
            "derived channel; retrain under torch-artifact-v5"
        )
    for name in (
        "K_quadrant",
        "neighbor_function",
        "min_neighbor_distance",
        "scan_pattern",
    ):
        values.pop(name, None)
    if "max_neighbor_distance" in values:
        values["group_padding_step"] = values.pop("max_neighbor_distance")


def _upgrade_data_section(
    era: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    values = _require_exact_config_payload(
        payload,
        DataConfig,
        era=era,
        section="data_config",
        expected_fields=_DATA_FIELDS_BY_ERA[era],
    )
    if era in (ARTIFACT_SCHEMA_V1_VERSION, ARTIFACT_SCHEMA_V2_VERSION):
        values = _project_legacy_data_to_runtime(values)
    elif era == ARTIFACT_SCHEMA_V3_VERSION:
        values["neighbor_count"] = values.pop("K")
        values.pop("groups_per_center")
    if era != ARTIFACT_SCHEMA_V5_VERSION:
        _upgrade_pre_centered_data(era, values)
    return values


def _upgrade_training_section(
    era: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    values = _require_exact_config_payload(
        payload,
        TrainingConfig,
        era=era,
        section="training_config",
        expected_fields=_TRAINING_FIELDS_BY_ERA[era],
    )
    if era in (
        ARTIFACT_SCHEMA_V1_VERSION,
        ARTIFACT_SCHEMA_V2_VERSION,
        ARTIFACT_SCHEMA_V3_VERSION,
    ):
        values["training_groups"] = values.pop("n_groups")
    return values


def _upgrade_inference_section(
    era: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _require_exact_config_payload(
        payload,
        InferenceConfig,
        era=era,
        section="inference_config",
        expected_fields=_INFERENCE_FIELDS_BY_ERA[era],
    )

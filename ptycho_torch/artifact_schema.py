"""Pure classification and upgrade helpers for supported Torch artifacts."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

import torch

from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model_spec import (
    PORTABLE_V1_MODEL_FIELDS,
    PORTABLE_V2_MODEL_FIELDS,
    ModelSpec,
    derive_model_spec,
)
from ptycho_torch.object_compatibility import resolve_torch_model_object_policy
from ptycho_torch.scaling_contract import (
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
    resolve_scale_contract,
)


TORCH_ARTIFACT_BACKEND = "pytorch"
ARTIFACT_SCHEMA_V1_VERSION = "torch-artifact-portable-v1"
CURRENT_ARTIFACT_SCHEMA_VERSION = "torch-artifact-portable-v2"
TORCH_BUNDLE_VERSION = "2.0-pytorch"
REQUIRED_BUNDLE_ROLES = frozenset({"autoencoder", "diffraction_to_obj"})

PORTABLE_V1_DATA_FIELDS = (
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
PORTABLE_V1_TRAINING_FIELDS = (
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
PORTABLE_V1_INFERENCE_FIELDS = (
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

_CONFIG_SCHEMA_FIELDS = {
    DataConfig: PORTABLE_V1_DATA_FIELDS,
    TrainingConfig: PORTABLE_V1_TRAINING_FIELDS,
    InferenceConfig: PORTABLE_V1_INFERENCE_FIELDS,
}
for _config_type, _schema_fields in _CONFIG_SCHEMA_FIELDS.items():
    _runtime_fields = tuple(item.name for item in fields(_config_type))
    if len(_schema_fields) != len(set(_schema_fields)):
        raise RuntimeError(
            f"{_config_type.__name__} artifact schema contains duplicate fields"
        )
    if set(_runtime_fields) != set(_schema_fields):
        raise RuntimeError(
            f"{_config_type.__name__} fields changed without an artifact schema "
            f"revision: runtime={_runtime_fields!r}"
        )


@dataclass(frozen=True)
class DecodedArtifactIdentity:
    model_spec: ModelSpec
    data_config: DataConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    ci_statistics: Optional[dict[str, list[float]]]


def _config_field_names(config_type) -> set[str]:
    if config_type is ModelConfig:
        return set(PORTABLE_V2_MODEL_FIELDS) | {"object_big"}
    return set(_CONFIG_SCHEMA_FIELDS[config_type])


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
            values[name] = tuple(values[name])
    return values


def _normalize_statistics(statistics) -> Optional[dict[str, list[float]]]:
    if statistics is None:
        return None
    if not isinstance(statistics, Mapping):
        raise ValueError("ci_statistics must be a mapping or null")
    return {
        str(name): torch.as_tensor(value).detach().cpu().reshape(-1).tolist()
        for name, value in statistics.items()
    }


def encode_artifact_identity(
    model_spec: ModelSpec,
    data_config: DataConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    *,
    ci_statistics=None,
) -> dict[str, Any]:
    return {
        "backend": TORCH_ARTIFACT_BACKEND,
        "schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
        "model_spec": model_spec.to_payload(),
        "data_config": asdict(data_config),
        "training_config": asdict(training_config),
        "inference_config": asdict(inference_config),
        "ci_statistics": _normalize_statistics(ci_statistics),
    }


def decode_artifact_identity(payload: Mapping[str, Any]) -> DecodedArtifactIdentity:
    if not isinstance(payload, Mapping):
        raise ValueError("Torch artifact identity must be a mapping")
    backend = payload.get("backend")
    if backend != TORCH_ARTIFACT_BACKEND:
        raise ValueError(
            f"unsupported artifact backend {backend!r}; expected 'pytorch'"
        )
    schema = payload.get("schema_version")
    if schema not in {
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"unsupported Torch artifact schema {schema!r}; "
            f"expected {ARTIFACT_SCHEMA_V1_VERSION!r} or "
            f"{CURRENT_ARTIFACT_SCHEMA_VERSION!r}"
        )
    expected_keys = {
        "backend",
        "schema_version",
        "model_spec",
        "data_config",
        "training_config",
        "inference_config",
        "ci_statistics",
    }
    received = set(payload)
    if received != expected_keys:
        raise ValueError(
            "current Torch artifact identity keys are not exact; "
            f"missing={sorted(expected_keys - received)}, "
            f"unknown={sorted(received - expected_keys)}"
        )

    spec = ModelSpec.from_payload(payload["model_spec"])
    is_v1 = schema == ARTIFACT_SCHEMA_V1_VERSION
    data = DataConfig(
        **_require_exact_config_payload(
            payload["data_config"],
            DataConfig,
            era=CURRENT_ARTIFACT_SCHEMA_VERSION,
            section="data_config",
            expected_fields=PORTABLE_V1_DATA_FIELDS if is_v1 else None,
        )
    )
    training = TrainingConfig(
        **_require_exact_config_payload(
            payload["training_config"],
            TrainingConfig,
            era=CURRENT_ARTIFACT_SCHEMA_VERSION,
            section="training_config",
            expected_fields=PORTABLE_V1_TRAINING_FIELDS if is_v1 else None,
        )
    )
    inference = InferenceConfig(
        **_require_exact_config_payload(
            payload["inference_config"],
            InferenceConfig,
            era=CURRENT_ARTIFACT_SCHEMA_VERSION,
            section="inference_config",
            expected_fields=PORTABLE_V1_INFERENCE_FIELDS if is_v1 else None,
        )
    )
    model = spec.to_model_config()
    if model.C_model != data.C or model.C_forward != data.C:
        raise ValueError(
            "Torch artifact ModelSpec/data channel join is inconsistent: "
            f"C_model={model.C_model}, C_forward={model.C_forward}, data C={data.C}"
        )
    return DecodedArtifactIdentity(
        model_spec=spec,
        data_config=data,
        training_config=training,
        inference_config=inference,
        ci_statistics=_normalize_statistics(payload["ci_statistics"]),
    )


def upgrade_unversioned_sections(
    *,
    data_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    explicit_profile: Optional[tuple[str, str]] = None,
    metadata_free_legacy: bool = False,
    parity_scale_mode: str = "off",
    parity_fixed_delta: float = 0.0,
    parity_init_scheme: str = "default",
    ci_statistics=None,
) -> DecodedArtifactIdentity:
    """Upgrade only the two declared unversioned eras without current defaults."""
    raw_data = copy.deepcopy(dict(data_config))
    profile_fields = {"scale_contract_version", "measurement_domain"}
    if metadata_free_legacy:
        if explicit_profile != (LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE):
            raise ValueError(
                "metadata-free legacy artifact requires explicit "
                "legacy_v1/normalized_amplitude profile"
            )
        if profile_fields & set(raw_data):
            raise ValueError(
                "metadata-free legacy artifact unexpectedly contains partial or "
                "complete scale profile metadata"
            )
        raw_data.update(
            scale_contract_version=LEGACY_SCALE_CONTRACT,
            measurement_domain=NORMALIZED_AMPLITUDE,
        )
    data_values = _require_exact_config_payload(
        raw_data,
        DataConfig,
        era="unversioned",
        section="data_config",
        expected_fields=PORTABLE_V1_DATA_FIELDS,
    )
    raw_model = copy.deepcopy(dict(model_config))
    received_model_fields = set(raw_model)
    v1_model_fields = set(PORTABLE_V1_MODEL_FIELDS)
    v2_model_fields = set(PORTABLE_V2_MODEL_FIELDS)
    current_model_fields = _config_field_names(ModelConfig)
    if received_model_fields == v1_model_fields:
        model = resolve_torch_model_object_policy(ModelConfig(**raw_model))
    elif received_model_fields == v2_model_fields:
        model = resolve_torch_model_object_policy(
            ModelConfig(object_big=None, **raw_model)
        )
    elif received_model_fields == current_model_fields:
        model = resolve_torch_model_object_policy(ModelConfig(**raw_model))
    else:
        variants = (v1_model_fields, v2_model_fields, current_model_fields)
        closest = min(
            variants,
            key=lambda expected: len(expected ^ received_model_fields),
        )
        raise ValueError(
            "unversioned model_config field set is not a declared portable-v1, "
            "portable-v2, or dual-written current shape; "
            f"missing={sorted(closest - received_model_fields)}, "
            f"unknown={sorted(received_model_fields - closest)}"
        )
    training_values = _require_exact_config_payload(
        training_config,
        TrainingConfig,
        era="unversioned",
        section="training_config",
        expected_fields=PORTABLE_V1_TRAINING_FIELDS,
    )
    inference_values = _require_exact_config_payload(
        inference_config,
        InferenceConfig,
        era="unversioned",
        section="inference_config",
        expected_fields=PORTABLE_V1_INFERENCE_FIELDS,
    )

    data = DataConfig(**data_values)
    training = TrainingConfig(**training_values)
    inference = InferenceConfig(**inference_values)
    resolve_scale_contract(data.scale_contract_version, data.measurement_domain)
    canonical = to_model_config(data, model)
    spec = derive_model_spec(
        canonical,
        model,
        data,
        parity_scale_mode=parity_scale_mode,
        parity_fixed_delta=parity_fixed_delta,
        parity_init_scheme=parity_init_scheme,
    )
    return DecodedArtifactIdentity(
        model_spec=spec,
        data_config=data,
        training_config=training,
        inference_config=inference,
        ci_statistics=_normalize_statistics(ci_statistics),
    )


def validate_torch_bundle_manifest(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("wts.h5.zip manifest must be a mapping")
    version = manifest.get("version")
    if version != TORCH_BUNDLE_VERSION:
        raise ValueError(
            f"unsupported wts.h5.zip manifest version {version!r}; "
            f"expected {TORCH_BUNDLE_VERSION!r}"
        )
    roles = set(manifest.get("models", ()))
    if roles != REQUIRED_BUNDLE_ROLES:
        raise ValueError(
            "wts.h5.zip roles must contain autoencoder and diffraction_to_obj "
            f"exactly; found {sorted(roles)}"
        )
    backend = manifest.get("backend")
    if backend is None:
        if "artifact_schema_version" in manifest:
            raise ValueError(
                "wts.h5.zip current schema marker requires backend='pytorch'"
            )
        return "metadata-free-legacy"
    if backend != TORCH_ARTIFACT_BACKEND:
        raise ValueError(f"unsupported wts.h5.zip backend {backend!r}")
    schema = manifest.get("artifact_schema_version")
    if schema is None:
        return "metadata-free-legacy"
    if schema not in {
        ARTIFACT_SCHEMA_V1_VERSION,
        CURRENT_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported wts.h5.zip artifact schema {schema!r}")
    return schema


_TENSOR_TAG = "__ptychopinn_torch_tensor__"


def _to_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_complex():
            data = {
                "real": tensor.real.reshape(-1).tolist(),
                "imag": tensor.imag.reshape(-1).tolist(),
            }
        else:
            data = tensor.reshape(-1).tolist()
        return {
            _TENSOR_TAG: True,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": data,
        }
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    return value


def _from_json_value(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get(_TENSOR_TAG) is True:
        dtype_name = value["dtype"]
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"unsupported serialized torch dtype {dtype_name!r}")
        data = value["data"]
        if isinstance(data, Mapping):
            tensor = torch.complex(
                torch.tensor(data["real"]), torch.tensor(data["imag"])
            ).to(dtype=dtype)
        else:
            tensor = torch.tensor(data, dtype=dtype)
        return tensor.reshape(tuple(value["shape"]))
    if isinstance(value, Mapping):
        return {str(key): _from_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_json_value(item) for item in value]
    return value


def to_json_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _to_json_value(payload)


def from_json_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _from_json_value(payload)

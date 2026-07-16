"""Versioned internal structural identity for Torch model construction.

``ModelSpec`` is not a public configuration surface.  It seals the result of
the canonical config bridge plus the declared Torch-only structural extensions
so construction and rebuild use one closed, versioned identity.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import torch

from ptycho.config.config import (
    ModelConfig as CanonicalModelConfig,
    resolve_model_object_policy,
)
from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch.object_compatibility import (
    resolve_model_object_compatibility,
    resolve_torch_model_object_policy,
)


MODEL_SPEC_V1_VERSION = "torch-model-spec-portable-v1"
CURRENT_MODEL_SPEC_VERSION = "torch-model-spec-portable-v2"

# Frozen field order for the family-free Torch model schema carried by main.
# This is deliberately not derived from ``fields(ModelConfig)``: adding or
# removing a structural field requires an explicit schema decision and version
# change rather than silently changing the meaning of an existing identifier.
PORTABLE_V1_MODEL_FIELDS = (
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
    "C_model",
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
    "probe_big",
    "offset",
    "C_forward",
    "training_patch_weighting",
    "physics_forward_mode",
    "rect_s1s2_trainable",
    "rect_s1s2_init",
    "amplitude_physics_gain",
    "pad_object",
    "gaussian_smoothing_sigma",
    "loss_function",
    "amp_loss",
    "phase_loss",
    "amp_loss_coeff",
    "phase_loss_coeff",
)

PORTABLE_V2_MODEL_FIELDS = (
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
    "C_model",
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
    "object_layout",
    "training_canvas",
    "probe_big",
    "offset",
    "C_forward",
    "training_patch_weighting",
    "physics_forward_mode",
    "rect_s1s2_trainable",
    "rect_s1s2_init",
    "amplitude_physics_gain",
    "pad_object",
    "gaussian_smoothing_sigma",
    "loss_function",
    "amp_loss",
    "phase_loss",
    "amp_loss_coeff",
    "phase_loss_coeff",
)

MODEL_SPEC_V1_MODEL_FIELDS = PORTABLE_V1_MODEL_FIELDS
MODEL_SPEC_V2_MODEL_FIELDS = PORTABLE_V2_MODEL_FIELDS

_RUNTIME_MODEL_FIELDS = tuple(item.name for item in fields(ModelConfig))
for _schema_fields in (PORTABLE_V1_MODEL_FIELDS, PORTABLE_V2_MODEL_FIELDS):
    if len(_schema_fields) != len(set(_schema_fields)):
        raise RuntimeError("portable ModelSpec field declaration contains duplicates")
if set(_RUNTIME_MODEL_FIELDS) != set(PORTABLE_V2_MODEL_FIELDS) | {"object_big"}:
    raise RuntimeError(
        "Torch ModelConfig fields changed without a ModelSpec schema revision: "
        f"runtime={_RUNTIME_MODEL_FIELDS!r}"
    )

# Values owned by the public canonical model handshake and represented in the
# Torch structural config. Keys are Torch ModelConfig field names; values are
# their canonical source names.
_CANONICAL_TO_TORCH = {
    "mode": "model_type",
    "architecture": "architecture",
    "fno_modes": "fno_modes",
    "fno_width": "fno_width",
    "fno_blocks": "fno_blocks",
    "fno_cnn_blocks": "fno_cnn_blocks",
    "learned_input_channels": "learned_input_channels",
    "fno_input_transform": "fno_input_transform",
    "max_hidden_channels": "max_hidden_channels",
    "resnet_width": "resnet_width",
    "generator_output_mode": "generator_output_mode",
    "n_filters_scale": "n_filters_scale",
    "amp_activation": "amp_activation",
    "probe_mask": "probe_mask",
    "probe_mask_sigma": "probe_mask_sigma",
    "probe_mask_diameter": "probe_mask_diameter",
    "object_layout": "object_layout",
    "training_canvas": "training_canvas",
    "training_patch_weighting": "training_patch_weighting",
    "probe_big": "probe_big",
    "pad_object": "pad_object",
    "gaussian_smoothing_sigma": "gaussian_smoothing_sigma",
}

CANONICAL_MODEL_FIELDS = frozenset(_CANONICAL_TO_TORCH)
TORCH_COMPATIBILITY_ALIAS_FIELDS = frozenset({"object_big"})
TORCH_EXTENSION_FIELDS = frozenset(
    PORTABLE_V2_MODEL_FIELDS
) - CANONICAL_MODEL_FIELDS

_PARITY_SCALE_MODES = frozenset({"off", "tied", "input", "output", "fixed"})
_PARITY_INIT_SCHEMES = frozenset({"default", "tf_glorot"})


def _copy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _canonical_expected_value(
    torch_name: str,
    canonical: CanonicalModelConfig,
    torch_model: ModelConfig,
) -> Any:
    value = getattr(canonical, _CANONICAL_TO_TORCH[torch_name])
    if torch_name == "mode":
        return {"pinn": "Unsupervised", "supervised": "Supervised"}[value]
    if torch_name == "probe_mask":
        # The canonical handshake owns boolean enablement. Torch may carry the
        # enabled mask as either a bool or an explicit tensor.
        return bool(value)
    return value


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        if left is None or right is None:
            return left is right
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
    return left == right


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """Sealed current Torch structural identity.

    Stored values are private defensive copies. Every construction receives a
    fresh ``ModelConfig`` so mutable tensor payloads cannot alias one another.
    """

    schema_version: str
    _model_fields: Mapping[str, Any] = field(repr=False)
    parity_scale_mode: str = "off"
    parity_fixed_delta: float = 0.0
    parity_init_scheme: str = "default"

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_MODEL_SPEC_VERSION:
            raise ValueError(
                f"unsupported current ModelSpec schema {self.schema_version!r}; "
                f"expected {CURRENT_MODEL_SPEC_VERSION!r}"
            )
        expected = set(PORTABLE_V2_MODEL_FIELDS)
        received = set(self._model_fields)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(
                f"ModelSpec fields must exactly match ModelConfig; missing={missing}, "
                f"unknown={extra}"
            )
        resolve_torch_model_object_policy(
            ModelConfig(
                object_big=None,
                **{
                    name: _copy_value(value)
                    for name, value in self._model_fields.items()
                },
            )
        )
        if self.parity_scale_mode not in _PARITY_SCALE_MODES:
            raise ValueError(f"invalid parity_scale_mode={self.parity_scale_mode!r}")
        if self.parity_init_scheme not in _PARITY_INIT_SCHEMES:
            raise ValueError(f"invalid parity_init_scheme={self.parity_init_scheme!r}")
        if not math.isfinite(float(self.parity_fixed_delta)):
            raise ValueError("parity_fixed_delta must be finite")
        object.__setattr__(
            self,
            "_model_fields",
            {name: _copy_value(value) for name, value in self._model_fields.items()},
        )

    @property
    def architecture(self) -> str:
        return str(self._model_fields["architecture"])

    @property
    def object_compatibility(self):
        """Return the versioned interpretation of the authoritative object axes."""
        return resolve_model_object_compatibility(self.to_model_config())

    def to_model_config(self) -> ModelConfig:
        raw = ModelConfig(
            object_big=None,
            **{
                name: _copy_value(value)
                for name, value in self._model_fields.items()
            },
        )
        return resolve_torch_model_object_policy(raw)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_config": {
                name: _copy_value(value) for name, value in self._model_fields.items()
            },
            "parity_scale_mode": self.parity_scale_mode,
            "parity_fixed_delta": float(self.parity_fixed_delta),
            "parity_init_scheme": self.parity_init_scheme,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("ModelSpec payload must be a mapping")
        expected = {
            "schema_version",
            "model_config",
            "parity_scale_mode",
            "parity_fixed_delta",
            "parity_init_scheme",
        }
        received = set(payload)
        if received != expected:
            raise ValueError(
                "ModelSpec payload keys are not current-schema exact; "
                f"missing={sorted(expected - received)}, unknown={sorted(received - expected)}"
            )
        schema_version = payload["schema_version"]
        model_fields = payload["model_config"]
        if not isinstance(model_fields, Mapping):
            raise ValueError("ModelSpec model_config must be a mapping")
        if schema_version == MODEL_SPEC_V1_VERSION:
            expected_v1 = set(PORTABLE_V1_MODEL_FIELDS)
            received_v1 = set(model_fields)
            if received_v1 != expected_v1:
                raise ValueError(
                    "torch-model-spec-portable-v1 model fields are not exact; "
                    f"missing={sorted(expected_v1 - received_v1)}, "
                    f"unknown={sorted(received_v1 - expected_v1)}"
                )
            legacy_big = model_fields["object_big"]
            if type(legacy_big) is not bool:
                raise ValueError(
                    "torch-model-spec-portable-v1 object_big must be bool"
                )
            values = {
                name: _copy_value(value)
                for name, value in model_fields.items()
                if name != "object_big"
            }
            if legacy_big:
                values["object_layout"] = "grouped_patches"
                values["training_canvas"] = "relative_overlap"
            else:
                values["object_layout"] = "single_patch"
                values["training_canvas"] = "independent"
        elif schema_version == CURRENT_MODEL_SPEC_VERSION:
            values = dict(model_fields)
        else:
            raise ValueError(
                f"unsupported ModelSpec schema {schema_version!r}; expected "
                f"{MODEL_SPEC_V1_VERSION!r} or {CURRENT_MODEL_SPEC_VERSION!r}"
            )
        return cls(
            schema_version=CURRENT_MODEL_SPEC_VERSION,
            _model_fields=values,
            parity_scale_mode=payload["parity_scale_mode"],
            parity_fixed_delta=float(payload["parity_fixed_delta"]),
            parity_init_scheme=payload["parity_init_scheme"],
        )


def derive_model_spec(
    canonical_model: CanonicalModelConfig,
    torch_model: ModelConfig,
    data_config: DataConfig,
    *,
    parity_scale_mode: str = "off",
    parity_fixed_delta: float = 0.0,
    parity_init_scheme: str = "default",
) -> ModelSpec:
    """Close canonical/shared fields and Torch extensions into current identity."""
    if not isinstance(canonical_model, CanonicalModelConfig):
        raise TypeError("canonical_model must be ptycho.config.config.ModelConfig")
    if not isinstance(torch_model, ModelConfig):
        raise TypeError("torch_model must be ptycho_torch.config_params.ModelConfig")
    if not isinstance(data_config, DataConfig):
        raise TypeError("data_config must be ptycho_torch.config_params.DataConfig")

    canonical_model = resolve_model_object_policy(
        canonical_model,
        backend="torch",
        warn_deprecated=False,
    )
    torch_model = resolve_torch_model_object_policy(torch_model)

    for torch_name, canonical_name in _CANONICAL_TO_TORCH.items():
        expected = _canonical_expected_value(torch_name, canonical_model, torch_model)
        actual = getattr(torch_model, torch_name)
        if torch_name == "probe_mask":
            actual = (
                torch_model.probe_mask_tensor is not None
                or isinstance(actual, torch.Tensor)
                or bool(actual)
            )
        elif torch_name == "amp_activation" and actual in {"silu", "SiLU"}:
            # Canonical TensorFlow spelling is ``swish``; the Torch structural
            # owner accepts both native SiLU spellings and the canonical alias.
            actual = "swish"
        if not _values_match(actual, expected):
            raise ValueError(
                f"structural field {torch_name}={actual!r} conflicts with canonical "
                f"ModelConfig.{canonical_name}={getattr(canonical_model, canonical_name)!r}"
            )

    if canonical_model.N != data_config.N:
        raise ValueError(
            f"canonical ModelConfig.N={canonical_model.N} conflicts with "
            f"data_config.N={data_config.N}"
        )
    expected_grid = (canonical_model.gridsize, canonical_model.gridsize)
    if tuple(data_config.grid_size) != expected_grid:
        raise ValueError(
            f"canonical gridsize={canonical_model.gridsize} conflicts with "
            f"data_config.grid_size={data_config.grid_size}"
        )
    if not _values_match(canonical_model.probe_scale, data_config.probe_scale):
        raise ValueError(
            f"canonical probe_scale={canonical_model.probe_scale} conflicts with "
            f"data_config.probe_scale={data_config.probe_scale}"
        )
    for name in ("C_model", "C_forward"):
        if getattr(torch_model, name) != data_config.C:
            raise ValueError(
                f"{name}={getattr(torch_model, name)} conflicts with data_config.C={data_config.C}"
            )

    values = {
        name: _copy_value(getattr(torch_model, name))
        for name in PORTABLE_V2_MODEL_FIELDS
    }
    return ModelSpec(
        schema_version=CURRENT_MODEL_SPEC_VERSION,
        _model_fields=values,
        parity_scale_mode=parity_scale_mode,
        parity_fixed_delta=float(parity_fixed_delta),
        parity_init_scheme=parity_init_scheme,
    )

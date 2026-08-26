"""One checkpoint-decode boundary for identity agreement checks.

Every Lightning-native loader routes the raw saved ``hyper_parameters`` dict
through :func:`decode_checkpoint_hparams` *before* strict state loading.

Two eras are distinguished:

- **New era** (current ``torch-artifact`` schema): the checkpoint single-writes
  identity payload under ``artifact_identity``. There is nothing to
  cross-check inside the checkpoint; decoding it is the validation.

- **Old eras** (``torch-model-spec-v1``/``-v2``): the checkpoint dual-writes
  the four ``asdict`` config dicts beside ``model_spec``. The agreement checks
  that previously lived in ``PtychoPINN_Lightning.__init__`` remain here
  (field-set-exact, model-config agreement, parity identity).

Pre-spec checkpoints (no ``model_spec`` key) are the legacy protocol: they
pass through unchanged and are reported loudly so their retirement horizon
stays visible.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

import torch

from ptycho_torch.artifact_schema import (
    ARTIFACT_V1_DATA_FIELDS,
    ARTIFACT_V1_INFERENCE_FIELDS,
    ARTIFACT_V1_TRAINING_FIELDS,
    ARTIFACT_V2_DATA_FIELDS,
    ARTIFACT_V2_INFERENCE_FIELDS,
    ARTIFACT_V2_TRAINING_FIELDS,
    decode_artifact_identity,
    validate_legacy_channel_faithfulness,
)
from ptycho_torch.config_params import ModelConfig
from ptycho_torch.model_spec import (
    MODEL_SPEC_V1_MODEL_FIELDS,
    MODEL_SPEC_V1_VERSION,
    MODEL_SPEC_V2_MODEL_FIELDS,
    ModelSpec,
)
from ptycho_torch.object_compatibility import resolve_torch_model_object_policy

logger = logging.getLogger(__name__)

_DERIVED_MODEL_FIELDS = frozenset({"C_model", "C_forward"})


def decode_checkpoint_hparams(hparams: dict) -> dict:
    """Validate raw Lightning hyperparameters against the sealed identity.

    New-era checkpoints carry only the sealed ``artifact_identity`` payload;
    decoding it validates it and there is no dual-write to cross-check. Old-era
    checkpoints dual-write the four config dicts beside ``model_spec`` and are
    agreement-checked exactly as before. A missing ``model_spec`` marks a
    pre-spec checkpoint (the legacy protocol): it is reported loudly and
    returned unchanged.
    """
    if not isinstance(hparams, dict):
        raise TypeError("checkpoint hyperparameters must be a dict")

    artifact_identity = hparams.get("artifact_identity")
    if artifact_identity is not None:
        decode_artifact_identity(artifact_identity)
        return hparams

    model_spec_payload = hparams.get("model_spec")
    if model_spec_payload is None:
        logger.warning(
            "checkpoint carries no model_spec; treating it as the pre-spec "
            "legacy protocol and skipping the dual-write agreement checks"
        )
        return hparams

    schema = (
        model_spec_payload.get("schema_version")
        if isinstance(model_spec_payload, dict)
        else None
    )
    is_v1 = schema == MODEL_SPEC_V1_VERSION
    section_fields = (
        ("model_config", hparams.get("model_config"),
         MODEL_SPEC_V1_MODEL_FIELDS if is_v1 else MODEL_SPEC_V2_MODEL_FIELDS),
        ("data_config", hparams.get("data_config"),
         ARTIFACT_V1_DATA_FIELDS if is_v1 else ARTIFACT_V2_DATA_FIELDS),
        ("training_config", hparams.get("training_config"),
         ARTIFACT_V1_TRAINING_FIELDS if is_v1 else ARTIFACT_V2_TRAINING_FIELDS),
        ("inference_config", hparams.get("inference_config"),
         ARTIFACT_V1_INFERENCE_FIELDS if is_v1 else ARTIFACT_V2_INFERENCE_FIELDS),
    )
    for section_name, value, expected in section_fields:
        if not isinstance(value, dict):
            continue
        received = set(value)
        expected = set(expected)
        if received != expected:
            raise ValueError(
                f"current checkpoint {section_name} field set is not exact; "
                f"missing={sorted(expected - received)}, "
                f"unknown={sorted(received - expected)}"
            )
    raw_data = hparams.get("data_config")
    raw_model = hparams.get("model_config")
    if isinstance(raw_data, dict) and "grid_size" in raw_data:
        validate_legacy_channel_faithfulness(
            raw_data,
            raw_model if isinstance(raw_model, dict) else {},
            era="checkpoint",
        )
    decoded_model_spec = (
        model_spec_payload
        if isinstance(model_spec_payload, ModelSpec)
        else ModelSpec.from_payload(model_spec_payload)
    )
    sealed_model_config = decoded_model_spec.to_model_config()

    raw_model_config = hparams.get("model_config")
    if isinstance(raw_model_config, dict):
        dropped = {
            name: value
            for name, value in raw_model_config.items()
            if name not in _DERIVED_MODEL_FIELDS
        }
        supplied = resolve_torch_model_object_policy(ModelConfig(**dropped))
        mismatches = []
        for item in fields(ModelConfig):
            supplied_value = getattr(supplied, item.name)
            sealed_value = getattr(sealed_model_config, item.name)
            if isinstance(supplied_value, torch.Tensor) or isinstance(
                sealed_value, torch.Tensor
            ):
                equal = (
                    isinstance(supplied_value, torch.Tensor)
                    and isinstance(sealed_value, torch.Tensor)
                    and torch.equal(supplied_value, sealed_value)
                )
            else:
                equal = supplied_value == sealed_value
            if not equal:
                mismatches.append(item.name)
        if mismatches:
            raise ValueError(
                "checkpoint ModelSpec conflicts with dual-written model_config "
                f"field(s): {sorted(mismatches)}"
            )

    parity_scale_mode = hparams.get("parity_scale_mode", "off")
    parity_fixed_delta = hparams.get("parity_fixed_delta", 0.0)
    parity_init_scheme = hparams.get("parity_init_scheme", "default")
    if (
        parity_scale_mode != decoded_model_spec.parity_scale_mode
        or float(parity_fixed_delta) != decoded_model_spec.parity_fixed_delta
        or parity_init_scheme != decoded_model_spec.parity_init_scheme
    ):
        raise ValueError(
            "checkpoint ModelSpec parity identity conflicts with dual-written "
            "Lightning parity hyperparameters"
        )

    return hparams

"""One checkpoint-decode boundary for the bundle-vs-runtime agreement checks.

Every Lightning-native loader routes the raw saved ``hyper_parameters`` dict
through :func:`decode_checkpoint_hparams` *before* strict state loading.

Two eras are distinguished:

- **Spec era** (current portable model-spec schema): the checkpoint dual-writes
  the four ``asdict`` config dicts beside ``model_spec``. The agreement checks
  that previously lived inline in ``PtychoPINN_Lightning.__init__`` live here
  (field-set-exact, model-config agreement, parity identity).

- **Pre-spec era** (no ``model_spec`` key): the legacy protocol. It passes
  through unchanged and is reported loudly so its retirement horizon stays
  visible.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

import torch

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model_spec import (
    MODEL_SPEC_V1_MODEL_FIELDS,
    MODEL_SPEC_V1_VERSION,
    ModelSpec,
)
from ptycho_torch.object_compatibility import resolve_torch_model_object_policy

logger = logging.getLogger(__name__)


def decode_checkpoint_hparams(hparams: dict) -> dict:
    """Validate raw Lightning hyperparameters against the sealed ModelSpec identity.

    Old-era checkpoints dual-write the four config dicts beside ``model_spec``
    and are agreement-checked exactly as ``PtychoPINN_Lightning.__init__`` did
    before the split. A missing ``model_spec`` marks a pre-spec checkpoint (the
    legacy protocol): it is reported loudly and returned unchanged.
    """
    if not isinstance(hparams, dict):
        raise TypeError("checkpoint hyperparameters must be a dict")

    model_spec = hparams.get("model_spec")
    if model_spec is None:
        logger.warning(
            "checkpoint carries no model_spec; treating it as the pre-spec "
            "legacy protocol and skipping the dual-write agreement checks"
        )
        return hparams

    model_config = hparams.get("model_config")
    data_config = hparams.get("data_config")
    training_config = hparams.get("training_config")
    inference_config = hparams.get("inference_config")
    parity_scale_mode = hparams.get("parity_scale_mode", "off")
    parity_fixed_delta = hparams.get("parity_fixed_delta", 0.0)
    parity_init_scheme = hparams.get("parity_init_scheme", "default")

    # Field-set-exact: the dual-written dicts must carry exactly the frozen
    # field surface for their era.
    for section_name, value, config_type in (
        ("model_config", model_config, ModelConfig),
        ("data_config", data_config, DataConfig),
        ("training_config", training_config, TrainingConfig),
        ("inference_config", inference_config, InferenceConfig),
    ):
        if not isinstance(value, dict):
            continue
        if (
            section_name == "model_config"
            and isinstance(model_spec, dict)
            and model_spec.get("schema_version") == MODEL_SPEC_V1_VERSION
        ):
            expected = set(MODEL_SPEC_V1_MODEL_FIELDS)
        else:
            expected = {item.name for item in fields(config_type)}
        received = set(value)
        if received != expected:
            raise ValueError(
                f"current checkpoint {section_name} field set is not exact; "
                f"missing={sorted(expected - received)}, "
                f"unknown={sorted(received - expected)}"
            )

    # Dict -> dataclass coercion (mirrors the constructor's load path).
    if isinstance(model_config, dict):
        model_config = ModelConfig(**model_config)
    if isinstance(data_config, dict):
        data_config = DataConfig(**data_config)
    if isinstance(training_config, dict):
        training_config = TrainingConfig(**training_config)
    if isinstance(inference_config, dict):
        inference_config = InferenceConfig(**inference_config)
    model_config = resolve_torch_model_object_policy(model_config)

    decoded_model_spec = (
        model_spec
        if isinstance(model_spec, ModelSpec)
        else ModelSpec.from_payload(model_spec)
    )
    sealed_model_config = decoded_model_spec.to_model_config()
    model_config = resolve_torch_model_object_policy(model_config)
    mismatches = []
    for item in fields(ModelConfig):
        supplied = getattr(model_config, item.name)
        sealed = getattr(sealed_model_config, item.name)
        if isinstance(supplied, torch.Tensor) or isinstance(sealed, torch.Tensor):
            equal = (
                isinstance(supplied, torch.Tensor)
                and isinstance(sealed, torch.Tensor)
                and torch.equal(supplied, sealed)
            )
        else:
            equal = supplied == sealed
        if not equal:
            mismatches.append(item.name)
    if mismatches:
        raise ValueError(
            "checkpoint ModelSpec conflicts with dual-written model_config "
            f"field(s): {sorted(mismatches)}"
        )
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

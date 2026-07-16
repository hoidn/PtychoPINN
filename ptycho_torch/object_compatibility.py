"""Torch adapter for the framework-neutral legacy object compatibility map."""

from __future__ import annotations

from ptycho.object_compatibility import (
    LegacyObjectFields,
    ObjectCompatibilitySpec,
    resolve_object_compatibility_spec,
)
from ptycho.config.config import ModelConfig as CanonicalModelConfig
from ptycho_torch.config_params import ModelConfig


def resolve_model_object_compatibility(
    model_config: ModelConfig | CanonicalModelConfig,
) -> ObjectCompatibilitySpec:
    if isinstance(model_config, ModelConfig):
        training_patch_weighting = model_config.training_patch_weighting
    elif isinstance(model_config, CanonicalModelConfig):
        # The public canonical handshake does not own the Torch-only weighting
        # axis; its historical projection uses the Torch default.
        training_patch_weighting = "central_mask"
    else:
        raise TypeError(
            "model_config must be a canonical or Torch ModelConfig"
        )
    return resolve_object_compatibility_spec(
        LegacyObjectFields(
            object_big=model_config.object_big,
            training_patch_weighting=training_patch_weighting,
            pad_object=model_config.pad_object,
            probe_big=model_config.probe_big,
        )
    )

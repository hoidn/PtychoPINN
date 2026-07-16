"""Torch adapter for the framework-neutral legacy object compatibility map."""

from __future__ import annotations

from dataclasses import replace

from ptycho.config.config import (
    ModelConfig as CanonicalModelConfig,
    resolve_model_object_policy,
)
from ptycho.object_compatibility import (
    ObjectCompatibilitySpec,
    resolve_public_object_policy,
)
from ptycho_torch.config_params import ModelConfig


def resolve_torch_model_object_policy(
    model_config: ModelConfig,
    *,
    warn_deprecated: bool = False,
) -> ModelConfig:
    if not isinstance(model_config, ModelConfig):
        raise TypeError("model_config must be a Torch ModelConfig")
    policy = resolve_public_object_policy(
        object_big=model_config.object_big,
        object_layout=model_config.object_layout,
        training_canvas=model_config.training_canvas,
        training_patch_weighting=model_config.training_patch_weighting,
        pad_object=model_config.pad_object,
        probe_big=model_config.probe_big,
        backend="torch",
        warn_deprecated=warn_deprecated,
    )
    return replace(
        model_config,
        object_big=policy.object_big,
        object_layout=policy.object_layout,
        training_canvas=policy.training_canvas,
        training_patch_weighting=policy.training_patch_weighting,
    )


def resolve_model_object_compatibility(
    model_config: ModelConfig | CanonicalModelConfig,
) -> ObjectCompatibilitySpec:
    if isinstance(model_config, ModelConfig):
        resolved = resolve_torch_model_object_policy(model_config)
        policy = resolve_public_object_policy(
            object_big=resolved.object_big,
            object_layout=resolved.object_layout,
            training_canvas=resolved.training_canvas,
            training_patch_weighting=resolved.training_patch_weighting,
            pad_object=resolved.pad_object,
            probe_big=resolved.probe_big,
            backend="torch",
            warn_deprecated=False,
        )
    elif isinstance(model_config, CanonicalModelConfig):
        resolved = resolve_model_object_policy(
            model_config,
            backend="torch",
            warn_deprecated=False,
        )
        policy = resolve_public_object_policy(
            object_big=resolved.object_big,
            object_layout=resolved.object_layout,
            training_canvas=resolved.training_canvas,
            training_patch_weighting=resolved.training_patch_weighting,
            pad_object=resolved.pad_object,
            probe_big=resolved.probe_big,
            backend="torch",
            warn_deprecated=False,
        )
    else:
        raise TypeError(
            "model_config must be a canonical or Torch ModelConfig"
        )
    return policy.compatibility

"""Torch-side training orchestration bodies relocated from ``ptycho.workflows.training``.

The shared TF-side training module keeps its spec-pinned public entry point
(``run_training_workflow``) but delegates every torch-specific branch and
factory/config adapter to the functions in this module.  Nothing here imports
TensorFlow; all heavy torch imports stay lazy inside the functions that need
them.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ptycho.config import TrainingConfig


def _torch_model_from_snapshot(resolved: Any):
    from ptycho_torch.config_params import ModelConfig as TorchModelConfig

    values = {
        item.name: getattr(resolved.model, item.name)
        for item in fields(TorchModelConfig)
    }
    if values["amplitude_physics_gain"] is None:
        values["amplitude_physics_gain"] = 1.0
    return TorchModelConfig(**values)


def _public_config_from_synthetic(request: Any) -> TrainingConfig:
    resolved = request.resolved_synthetic_workflow
    from ptycho.workflows.synthetic_config import materialize_data_config
    from ptycho_torch.config_bridge import to_model_config

    data_config = materialize_data_config(resolved)
    public_model = to_model_config(
        data_config,
        _torch_model_from_snapshot(resolved),
    )
    training = resolved.training
    return TrainingConfig(
        model=public_model,
        train_data_file=Path(request.train_data_file),
        test_data_file=Path(request.test_data_file),
        batch_size=training.batch_size,
        nepochs=training.epochs,
        nphotons=data_config.nphotons,
        training_groups=training.training_groups,
        train_raw_selection=training.train_raw_selection,
        subsample_seed=training.subsample_seed,
        neighbor_count=training.neighbor_count,
        enable_oversampling=training.enable_oversampling,
        neighbor_pool_size=training.neighbor_pool_size,
        output_dir=Path(request.output_dir),
        sequential_sampling=training.sequential_sampling,
        backend="pytorch",
        torch_loss_mode=training.torch_loss_mode,
        torch_mae_pred_l2_match_target=(
            training.torch_mae_pred_l2_match_target
        ),
        gradient_clip_val=training.gradient_clip_val,
        gradient_clip_algorithm=training.gradient_clip_algorithm,
        optimizer=training.optimizer,
        momentum=training.momentum,
        weight_decay=training.weight_decay,
        adam_beta1=training.adam_beta1,
        adam_beta2=training.adam_beta2,
        scheduler=training.scheduler,
        lr_warmup_epochs=training.lr_warmup_epochs,
        lr_min_ratio=training.lr_min_ratio,
        plateau_factor=training.plateau_factor,
        plateau_patience=training.plateau_patience,
        plateau_min_lr=training.plateau_min_lr,
        plateau_threshold=training.plateau_threshold,
    )


def _materialize_torch_container(
    grouped: dict,
    raw_data: Any,
    *,
    data_adapter: str | None = None,
):
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

    container = PtychoDataContainerTorch(grouped, raw_data.probeGuess)
    metadata = getattr(raw_data, "metadata", None)
    if metadata is not None:
        container.metadata = metadata
    if data_adapter is not None:
        _apply_backend_data_adapter(container, grouped, data_adapter)
    return container


def _apply_backend_data_adapter(
    container: Any,
    grouped: Mapping[str, Any],
    adapter_name: str,
) -> Any:
    """Apply one resolved synthetic ingestion policy to an in-memory container."""

    from ptycho_torch.data_adapter import resolve_data_adapter

    policy = resolve_data_adapter(adapter_name)
    if not policy.explicit_unit_scales:
        return container

    import torch

    raw_grouped = np.ascontiguousarray(
        np.asarray(grouped["diffraction"], dtype=np.float32)
    )
    raw_tensor = torch.from_numpy(raw_grouped).to(torch.float32)
    unit = torch.ones((1, 1, 1), dtype=torch.float32)
    values = {
        "X": raw_tensor,
        "rms_scaling_constant": unit.clone(),
        "physics_scaling_constant": unit.clone(),
    }
    if isinstance(container, dict):
        container.update(values)
    else:
        for name, value in values.items():
            setattr(container, name, value)
    return container


def _legacy_execution_and_patch(request: Any, config: TrainingConfig):
    if config.backend != "pytorch":
        return None, None
    from ptycho_torch.cli.shared import (
        build_execution_request_from_args,
        build_training_config_patch_from_args,
    )

    execution = build_execution_request_from_args(
        request.legacy_args,
        mode="training",
        explicit_options=request.raw_argv,
        lane="unified-training",
    )
    patch = build_training_config_patch_from_args(
        request.legacy_args,
        explicit_options=request.raw_argv,
        lane="unified-training",
    )
    return execution, patch


def _synthetic_execution_request(resolved: Any):
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    execution_names = {item.name for item in fields(PyTorchExecutionConfig)}
    values = {
        item.name: getattr(resolved.workflow, item.name)
        for item in fields(resolved.workflow)
        if item.name in execution_names
    }
    return ExecutionRequest(values=values, explicit_fields=frozenset(values))


def _base_factory_overrides(config: TrainingConfig) -> dict[str, Any]:
    from ptycho_torch.config_factory import build_training_factory_overrides

    return build_training_factory_overrides(config)


def _synthetic_factory_overrides(
    resolved: Any,
    config: TrainingConfig,
) -> dict[str, Any]:
    from ptycho_torch.config_params import DataConfig, ModelConfig as TorchModelConfig

    overrides = _base_factory_overrides(config)
    for item in fields(DataConfig):
        overrides[item.name] = getattr(resolved.data, item.name)
    for item in fields(TorchModelConfig):
        value = getattr(resolved.model, item.name)
        if item.name == "amplitude_physics_gain" and value is None:
            continue
        overrides[item.name] = value
    for name in (
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
        "stage_1_epochs",
        "stage_2_epochs",
        "stage_3_epochs",
        "physics_weight_schedule",
        "stage_3_lr_factor",
        "torch_loss_mode",
        "torch_mae_pred_l2_match_target",
        "nll",
    ):
        overrides[name] = getattr(resolved.training, name)
    from ptycho_torch.config_params import InferenceConfig as TorchInferenceConfig

    for item in fields(TorchInferenceConfig):
        target = (
            "inference_batch_size"
            if item.name == "batch_size"
            else item.name
        )
        overrides[target] = getattr(resolved.inference, item.name)
    overrides["training_groups"] = resolved.training.training_groups
    overrides["n_raw_frames_selected"] = resolved.training.train_raw_selection
    return overrides

def _validate_payload_selection_identity(
    selected_config: TrainingConfig,
    payload_config: TrainingConfig,
) -> None:
    """Fail if factory resolution changes fields already used for selection."""

    paths = (
        "train_data_file",
        "test_data_file",
        "output_dir",
        "training_groups",
        "train_raw_selection",
        "subsample_seed",
        "neighbor_count",
        "enable_oversampling",
        "neighbor_pool_size",
        "sequential_sampling",
        "nphotons",
    )
    for name in paths:
        if getattr(selected_config, name) != getattr(payload_config, name):
            raise ValueError(
                "resolved Torch payload changed post-selection identity field "
                f"{name!r}: {getattr(selected_config, name)!r} != "
                f"{getattr(payload_config, name)!r}"
            )
    for name in ("N", "gridsize"):
        before = getattr(selected_config.model, name)
        after = getattr(payload_config.model, name)
        if before != after:
            raise ValueError(
                "resolved Torch payload changed post-selection model field "
                f"{name!r}: {before!r} != {after!r}"
            )


def _validate_synthetic_payload_identity(
    resolved: Any,
    payload: Any,
    gain_record: Any,
) -> None:
    """Check the synthetic data/loss/gain owners agree after factory resolution."""

    if payload.pt_data_config.n_raw_frames_selected != resolved.training.train_raw_selection:
        raise ValueError("resolved payload changed synthetic train raw selection")
    if payload.pt_data_config.neighbor_count != resolved.training.neighbor_count:
        raise ValueError("resolved payload changed synthetic neighbor count")
    if payload.pt_training_config.nll is not resolved.training.nll:
        raise ValueError("resolved payload changed synthetic nll identity")
    if payload.tf_training_config.torch_loss_mode != resolved.training.torch_loss_mode:
        raise ValueError("resolved payload changed synthetic Torch loss mode")
    if (
        payload.tf_training_config.torch_mae_pred_l2_match_target
        is not resolved.training.torch_mae_pred_l2_match_target
    ):
        raise ValueError("resolved payload changed synthetic MAE L2-match identity")
    if payload.pt_model_config.amplitude_physics_gain != gain_record.value:
        raise ValueError("resolved payload did not consume the selected-data gain")
    for item in fields(payload.pt_inference_config):
        if getattr(payload.pt_inference_config, item.name) != getattr(
            resolved.inference,
            item.name,
        ):
            raise ValueError(
                "resolved payload changed synthetic inference field "
                f"{item.name!r}"
            )


def _resolve_workflow_batch_order_recipe(request: Any) -> str:
    """Resolve the versioned Torch example-order contract exactly once."""

    from ptycho_torch.batch_order import (
        DEFAULT_BATCH_ORDER_RECIPE,
        validate_batch_order_recipe,
    )

    explicit = request.batch_order_recipe
    if request.resolved_synthetic_workflow is None:
        return validate_batch_order_recipe(
            DEFAULT_BATCH_ORDER_RECIPE if explicit is None else explicit
        )
    configured = request.resolved_synthetic_workflow.training.batch_order_recipe
    if explicit is not None and explicit != configured:
        raise ValueError(
            "request batch_order_recipe conflicts with resolved synthetic "
            f"identity: expected {configured!r}, got {explicit!r}"
        )
    return validate_batch_order_recipe(configured)

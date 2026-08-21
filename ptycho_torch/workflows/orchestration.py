"""Torch-side training orchestration bodies relocated from ``ptycho.workflows.training``.

Every function here is a thin lazy adapter to a pure ``ptycho_torch`` factory or
a torch-only branch previously inlined in the TF-side training workflow.  The
TF-side ``ptycho/workflows/training.py`` delegates to this module, so the
shared entry point keeps its public surface while torch knowledge lives under
``ptycho_torch/``.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from ptycho.config import TrainingConfig


def resolve_training_payload(**kwargs):
    """Lazy adapter to the pure Torch training factory."""

    from ptycho_torch.config_factory import resolve_training_payload as resolve

    return resolve(**kwargs)


def resolve_amplitude_physics_gain(*args, **kwargs):
    """Lazy adapter to the selected-data scaling contract."""

    from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain as resolve

    return resolve(*args, **kwargs)


def load_inference_bundle_torch(*args, **kwargs):
    """Lazy adapter to strict Torch bundle reload."""

    from .bundle_io import load_inference_bundle_torch as load

    return load(*args, **kwargs)


def _torch_model_from_snapshot(resolved: Any):
    from ptycho_torch.config_params import ModelConfig as TorchModelConfig

    values = {
        item.name: getattr(resolved.model, item.name)
        for item in fields(TorchModelConfig)
    }
    if values["amplitude_physics_gain"] is None:
        values["amplitude_physics_gain"] = 1.0
    return TorchModelConfig(**values)


def _public_config_from_synthetic(
    request: Any,
) -> TrainingConfig:
    resolved = request.resolved_synthetic_workflow
    from ptycho.config.config import (
        AdamConfig,
        DataConfig as PublicDataConfig,
        GradientClipConfig,
        LossConfig,
        OptimizerConfig,
        SamplingConfig,
        SchedulerConfig,
        SgdConfig,
    )
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
        batch_size=training.batch_size,
        nepochs=training.epochs,
        output_dir=Path(request.output_dir),
        backend="pytorch",
        data=PublicDataConfig(
            train_data_file=Path(request.train_data_file),
            test_data_file=Path(request.test_data_file),
            nphotons=data_config.nphotons,
        ),
        sampling=SamplingConfig(
            n_groups=training.training_groups,
            n_subsample=training.train_raw_selection,
            subsample_seed=training.subsample_seed,
            neighbor_count=training.neighbor_count,
            enable_oversampling=training.enable_oversampling,
            neighbor_pool_size=training.neighbor_pool_size,
            sequential_sampling=training.sequential_sampling,
        ),
        loss=LossConfig(
            torch_loss_mode=training.torch_loss_mode,
            torch_mae_pred_l2_match_target=(
                training.torch_mae_pred_l2_match_target
            ),
        ),
        gradient_clip=GradientClipConfig(
            val=training.gradient_clip_val,
            algorithm=training.gradient_clip_algorithm,
        ),
        optimizer=OptimizerConfig(
            algorithm=training.optimizer,
            weight_decay=training.weight_decay,
            sgd=SgdConfig(momentum=training.momentum),
            adam=AdamConfig(
                beta1=training.adam_beta1,
                beta2=training.adam_beta2,
            ),
        ),
        scheduler=SchedulerConfig(
            kind=training.scheduler,
            lr_warmup_epochs=training.lr_warmup_epochs,
            lr_min_ratio=training.lr_min_ratio,
            plateau_factor=training.plateau_factor,
            plateau_patience=training.plateau_patience,
            plateau_min_lr=training.plateau_min_lr,
            plateau_threshold=training.plateau_threshold,
        ),
    )


def _materialize_torch_container(
    grouped: dict,
    raw_data: Any,
    config: TrainingConfig,
):
    from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

    container = PtychoDataContainerTorch(grouped, raw_data.probeGuess)
    metadata = getattr(raw_data, "metadata", None)
    if metadata is not None:
        container.metadata = metadata
    return container


def _legacy_execution_and_patch(request: Any, config):
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
    overrides["n_groups"] = resolved.training.training_groups
    overrides["n_raw_frames_selected"] = resolved.training.train_raw_selection
    return overrides


def _selected_truth_patches(train_raw: Any, *, N: int) -> np.ndarray:
    """Return one raw truth patch for every finalized selected frame.

    ``RawData.Y`` is already frame-aligned after ``load_data`` subsampling.  A
    flat acquisition may instead carry only the full object; in that case use
    the same singleton-coordinate convention as ``RawData.from_simulation`` so
    gain identity is independent of later neighbor grouping.
    """

    sample_count = int(np.asarray(train_raw.diff3d).shape[0])
    if getattr(train_raw, "Y", None) is not None:
        patches = np.asarray(train_raw.Y)
        if patches.shape[0] != sample_count:
            raise ValueError(
                "selected RawData.Y must contain one truth patch per selected "
                f"frame; got {patches.shape[0]} patches for {sample_count} frames"
            )
        return np.ascontiguousarray(patches)

    object_guess = getattr(train_raw, "objectGuess", None)
    if object_guess is None:
        raise ValueError(
            "selected training data requires RawData.Y or objectGuess to derive "
            "amplitude_physics_gain"
        )
    from ptycho.raw_data import get_image_patches, get_relative_coords

    coords = np.zeros((sample_count, 1, 2, 1), dtype=np.float64)
    coords[:, 0, 0, 0] = np.asarray(train_raw.xcoords)
    coords[:, 0, 1, 0] = np.asarray(train_raw.ycoords)
    global_offsets, local_offsets = get_relative_coords(coords)
    patches = get_image_patches(
        object_guess,
        global_offsets,
        local_offsets,
        N=N,
        gridsize=1,
    )
    return np.ascontiguousarray(np.asarray(patches))


def _resolve_gain(resolved: Any, train_raw: Any):
    """Resolve gain from the finalized raw training selection, never groups."""

    model = resolved.model
    override = (
        model.amplitude_physics_gain
        if model.amplitude_physics_gain_provenance == "explicit"
        else None
    )
    needs_truth = (
        model.physics_forward_mode == "amplitude" and override is None
    )
    return resolve_amplitude_physics_gain(
        np.ascontiguousarray(np.asarray(train_raw.diff3d)),
        (
            _selected_truth_patches(train_raw, N=resolved.data.N)
            if needs_truth
            else None
        ),
        np.ascontiguousarray(np.asarray(train_raw.probeGuess)),
        probe_scale=resolved.data.probe_scale,
        probe_mask=model.probe_mask,
        probe_mask_tensor=model.probe_mask_tensor,
        probe_mask_sigma=model.probe_mask_sigma,
        probe_mask_diameter=model.probe_mask_diameter,
        override=override,
        physics_forward_mode=model.physics_forward_mode,
    )


def _validate_payload_selection_identity(
    selected_config: TrainingConfig,
    payload_config: TrainingConfig,
) -> None:
    """Fail if factory resolution changes fields already used for selection."""

    values = (
        (
            "data.train_data_file",
            selected_config.data.train_data_file,
            payload_config.data.train_data_file,
        ),
        (
            "data.test_data_file",
            selected_config.data.test_data_file,
            payload_config.data.test_data_file,
        ),
        (
            "output_dir",
            selected_config.output_dir,
            payload_config.output_dir,
        ),
        (
            "sampling.n_groups",
            selected_config.sampling.n_groups,
            payload_config.sampling.n_groups,
        ),
        (
            "sampling.n_subsample",
            selected_config.sampling.n_subsample,
            payload_config.sampling.n_subsample,
        ),
        (
            "sampling.subsample_seed",
            selected_config.sampling.subsample_seed,
            payload_config.sampling.subsample_seed,
        ),
        (
            "sampling.neighbor_count",
            selected_config.sampling.neighbor_count,
            payload_config.sampling.neighbor_count,
        ),
        (
            "sampling.enable_oversampling",
            selected_config.sampling.enable_oversampling,
            payload_config.sampling.enable_oversampling,
        ),
        (
            "sampling.neighbor_pool_size",
            selected_config.sampling.neighbor_pool_size,
            payload_config.sampling.neighbor_pool_size,
        ),
        (
            "sampling.sequential_sampling",
            selected_config.sampling.sequential_sampling,
            payload_config.sampling.sequential_sampling,
        ),
        (
            "data.nphotons",
            selected_config.data.nphotons,
            payload_config.data.nphotons,
        ),
    )
    for name, before, after in values:
        if before != after:
            raise ValueError(
                "resolved Torch payload changed post-selection identity field "
                f"{name!r}: {before!r} != {after!r}"
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
    if payload.pt_data_config.K != resolved.training.neighbor_count:
        raise ValueError("resolved payload changed synthetic neighbor count")
    if payload.pt_training_config.nll is not resolved.training.nll:
        raise ValueError("resolved payload changed synthetic nll identity")
    if (
        payload.tf_training_config.loss.torch_loss_mode
        != resolved.training.torch_loss_mode
    ):
        raise ValueError("resolved payload changed synthetic Torch loss mode")
    if (
        payload.tf_training_config.loss.torch_mae_pred_l2_match_target
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

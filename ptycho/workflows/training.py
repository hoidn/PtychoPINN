"""Shared generic training workflow used by public training entry points.

The boundary resolves public configuration, selects and groups flat raw
acquisitions, resolves the Torch factory exactly once, and delegates model and
Trainer construction to the existing backend implementations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, fields
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ptycho.config import (
    TrainingConfig,
    validate_runnable_training_config,
    validate_training_config_structure,
)
from ptycho.metadata import MetadataManager
from ptycho.workflows.backend_selector import run_cdi_example_with_backend
from ptycho.workflows.components import load_data, save_outputs, setup_configuration


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingWorkflowRequest:
    """One legacy-CLI or resolved-synthetic training request."""

    legacy_args: argparse.Namespace | None = None
    raw_argv: tuple[str, ...] = ()
    resolved_synthetic_workflow: Any | None = None
    train_data_file: Path | None = None
    test_data_file: Path | None = None
    output_dir: Path | None = None
    do_stitching: bool = False
    torch_training_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_argv", tuple(self.raw_argv))
        if self.torch_training_seed is not None:
            if (
                isinstance(self.torch_training_seed, bool)
                or not isinstance(self.torch_training_seed, int)
            ):
                raise TypeError(
                    "torch_training_seed must be a nonnegative integer"
                )
            if self.torch_training_seed < 0:
                raise ValueError(
                    "torch_training_seed must be a nonnegative integer"
                )
        legacy = self.legacy_args is not None
        synthetic = self.resolved_synthetic_workflow is not None
        if legacy == synthetic:
            raise ValueError(
                "exactly one of legacy_args or resolved_synthetic_workflow "
                "must be supplied"
            )
        if synthetic:
            missing = [
                name
                for name in ("train_data_file", "test_data_file", "output_dir")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "resolved synthetic training requires " + ", ".join(missing)
                )
            if self.do_stitching:
                raise ValueError(
                    "resolved synthetic training requires do_stitching=False; "
                    "barycentric reconstruction is a separate stage"
                )


@dataclass(frozen=True)
class TrainingWorkflowResult:
    """Structured training output and persisted scientific identity."""

    public_config: TrainingConfig
    backend_results: Mapping[str, Any]
    reconstruction_amplitude: Any | None
    reconstruction_phase: Any | None
    bundle_path: Path | None
    pt_data_config: Any | None
    pt_model_config: Any | None
    model_spec: Any | None
    train_group_count: int
    validation_group_count: int
    amplitude_physics_gain_record: Any | None
    amplitude_physics_gain_metadata: Mapping[str, Any] | None
    torch_training_seed: int | None


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

    from ptycho_torch.workflows.components import load_inference_bundle_torch as load

    return load(*args, **kwargs)


def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:
    """Retain the legacy CLI explanation for ``--n-images``."""

    if gridsize == 1:
        return (
            n_images,
            f"Parameter interpretation: --n-images={n_images} refers to "
            "individual images (gridsize=1)",
        )
    total_patterns = n_images * gridsize * gridsize
    return (
        n_images,
        f"Parameter interpretation: --n-images={n_images} refers to neighbor "
        f"groups (gridsize={gridsize}, total patterns={total_patterns})",
    )


def interpret_sampling_parameters(config: TrainingConfig):
    """Resolve raw selection and exact group counts without side effects."""

    gridsize = config.model.gridsize
    sampling = config.sampling
    enable_oversampling = sampling.enable_oversampling
    neighbor_pool_size = sampling.neighbor_pool_size
    if sampling.n_subsample is not None:
        n_subsample = sampling.n_subsample
        n_groups = sampling.n_groups
        if gridsize == 1:
            message = (
                f"Independent sampling control: subsampling {n_subsample} images, "
                f"using {n_groups} groups for training"
            )
        else:
            message = (
                f"Independent sampling control: subsampling {n_subsample} images, "
                f"creating {n_groups} groups (approx "
                f"{n_groups * gridsize * gridsize} patterns from groups)"
            )
    else:
        n_subsample = sampling.n_groups
        n_groups = sampling.n_groups
        if gridsize == 1:
            message = f"Legacy mode: using {n_groups} groups (gridsize=1)"
        else:
            message = (
                f"Legacy mode: --n-groups={n_groups} refers to neighbor groups "
                f"(gridsize={gridsize}, approx "
                f"{n_groups * gridsize * gridsize} patterns)"
            )
    if enable_oversampling:
        pool = neighbor_pool_size or sampling.neighbor_count
        message += f" [Oversampling enabled: K={pool}]"
    return (
        n_subsample,
        n_groups,
        enable_oversampling,
        neighbor_pool_size,
        message,
    )


def _metadata_photon_count(path: Path) -> float | None:
    try:
        _, metadata = MetadataManager.load_with_metadata(str(path))
    except Exception as error:
        logger.debug("No metadata found or error reading metadata: %s", error)
        return None
    if not metadata:
        return None
    if "nphotons" in metadata:
        return float(metadata["nphotons"])
    physics = metadata.get("physics_parameters")
    if isinstance(physics, Mapping) and "nphotons" in physics:
        return float(physics["nphotons"])
    return None


def _resolve_metadata_photons(config: TrainingConfig) -> TrainingConfig:
    resolved = _metadata_photon_count(Path(config.data.train_data_file))
    if resolved is None:
        return config
    logger.info(
        "Overriding nphotons from config (%.1e) with dataset metadata: %.1e",
        config.data.nphotons,
        resolved,
    )
    return config.model_copy(update={
        "data": config.data.model_copy(update={"nphotons": resolved})
    })


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
    request: TrainingWorkflowRequest,
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


def _resolve_public_config(request: TrainingWorkflowRequest) -> TrainingConfig:
    if request.legacy_args is not None:
        args = argparse.Namespace(**vars(request.legacy_args))
        if hasattr(args, "train_data_file_path"):
            args.train_data_file = args.train_data_file_path
            delattr(args, "train_data_file_path")
        config = setup_configuration(args, getattr(args, "config", None))
    else:
        config = _public_config_from_synthetic(request)
    config = _resolve_metadata_photons(config)
    validate_training_config_structure(config)
    validate_runnable_training_config(config)
    return config


def _group_raw_data(
    raw_data: Any,
    config: TrainingConfig,
    path: Path,
    *,
    require_exact: bool = True,
    group_count: int | None = None,
) -> dict:
    sampling = config.sampling
    expected_groups = sampling.n_groups if group_count is None else group_count
    grouped = raw_data.generate_grouped_data(
        N=config.model.N,
        K=sampling.neighbor_count,
        nsamples=expected_groups,
        dataset_path=str(path),
        seed=sampling.subsample_seed,
        sequential_sampling=sampling.sequential_sampling,
        gridsize=config.model.gridsize,
        enable_oversampling=sampling.enable_oversampling,
        neighbor_pool_size=sampling.neighbor_pool_size,
    )
    actual = int(np.asarray(grouped["nn_indices"]).shape[0])
    if require_exact and actual != expected_groups:
        raise ValueError(
            f"grouping produced {actual} groups; expected exactly {expected_groups}"
        )
    return grouped


def _materialize_backend_container(
    grouped: dict,
    raw_data: Any,
    config: TrainingConfig,
):
    if config.backend == "pytorch":
        from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

        container = PtychoDataContainerTorch(grouped, raw_data.probeGuess)
        metadata = getattr(raw_data, "metadata", None)
        if metadata is not None:
            container.metadata = metadata
        return container
    from ptycho import loader, params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import legacy_params_scope

    with legacy_params_scope():
        update_legacy_dict(params.cfg, config)
        return loader.load(
            lambda: grouped,
            raw_data.probeGuess,
            which=None,
            create_split=False,
        )


@contextmanager
def _legacy_data_preparation_scope(config: TrainingConfig):
    """Project one request while legacy data consumers run, then restore it."""

    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import legacy_params_scope

    with legacy_params_scope():
        update_legacy_dict(params.cfg, config)
        yield


def _legacy_execution_and_patch(request: TrainingWorkflowRequest, config):
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
    overrides["n_subsample"] = resolved.training.train_raw_selection
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


def _validate_selected_raw_count(train_raw: Any, *, expected: int) -> None:
    """Reject a clamped synthetic selection before deriving its identity."""

    actual = int(np.asarray(train_raw.diff3d).shape[0])
    if actual != expected:
        raise ValueError(
            "synthetic training requested exactly "
            f"{expected} raw frames but selected {actual}"
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

    if payload.pt_data_config.n_subsample != resolved.training.train_raw_selection:
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


def _persist_tensorflow_outputs(
    config: TrainingConfig,
    amplitude: Any,
    phase: Any,
    results: Mapping[str, Any],
) -> None:
    from ptycho import model_manager, params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import configured_params_scope, legacy_params_scope

    with legacy_params_scope():
        with configured_params_scope():
            update_legacy_dict(params.cfg, config)
            model_manager.save(str(config.output_dir))
    save_outputs(amplitude, phase, dict(results), str(config.output_dir))


def run_training_workflow(
    request: TrainingWorkflowRequest,
) -> TrainingWorkflowResult:
    """Execute one generic training workflow through the existing backend."""

    if not isinstance(request, TrainingWorkflowRequest):
        raise TypeError("request must be a TrainingWorkflowRequest")
    config = _resolve_public_config(request)
    torch_training_seed = request.torch_training_seed
    if request.resolved_synthetic_workflow is not None:
        from ptycho.simulation.flat_acquisition import derive_seed_lineage

        seed_lineage = derive_seed_lineage(
            request.resolved_synthetic_workflow.simulation.train.seed
        )
        derived_torch_seed = seed_lineage["torch"]
        if (
            torch_training_seed is not None
            and torch_training_seed != derived_torch_seed
        ):
            raise ValueError(
                "torch_training_seed disagrees with the synthetic workflow "
                "Torch child seed"
            )
        torch_training_seed = derived_torch_seed
        if torch_training_seed == config.sampling.subsample_seed:
            raise ValueError(
                "the synthetic Torch and grouping seed streams must be distinct"
            )
    n_subsample, n_groups, _, _, message = interpret_sampling_parameters(config)
    logger.info(message)
    config = config.model_copy(update={
        "sampling": config.sampling.model_copy(update={"n_groups": n_groups})
    })

    with _legacy_data_preparation_scope(config):
        train_raw = load_data(
            str(config.data.train_data_file),
            n_images=n_groups,
            n_subsample=n_subsample,
            subsample_seed=config.sampling.subsample_seed,
        )
    if request.resolved_synthetic_workflow is not None:
        _validate_selected_raw_count(train_raw, expected=n_subsample)
    gain_record = None
    payload = None
    execution_request = None
    factory_overrides = None
    if config.backend == "pytorch":
        if request.resolved_synthetic_workflow is not None:
            resolved = request.resolved_synthetic_workflow
            gain_record = _resolve_gain(resolved, train_raw)
            execution_request = _synthetic_execution_request(resolved)
            factory_overrides = _synthetic_factory_overrides(resolved, config)
            factory_overrides.update(gain_record.factory_overrides())
        else:
            execution_request, cli_patch = _legacy_execution_and_patch(
                request,
                config,
            )
            explicit_fields = getattr(
                execution_request,
                "explicit_fields",
                None,
            )
            if explicit_fields is not None and not explicit_fields:
                logger.info(
                    "POLICY-001: No --torch-* execution flags provided. "
                    "Backend will use GPU-first defaults (auto-detects CUDA "
                    "if available, else CPU). CPU-only users should pass "
                    "--torch-accelerator cpu."
                )
            # Preserve the established precedence: reconstruct the complete
            # public-config baseline, then overlay only explicit CLI aliases.
            factory_overrides = _base_factory_overrides(config)
            factory_overrides.update(cli_patch)
        payload = resolve_training_payload(
            train_data_file=Path(config.data.train_data_file),
            output_dir=Path(config.output_dir),
            overrides=factory_overrides,
            execution_config=execution_request,
            training_baseline=config,
        )
        payload_config = payload.tf_training_config
        _validate_payload_selection_identity(config, payload_config)
        if request.resolved_synthetic_workflow is not None:
            _validate_synthetic_payload_identity(resolved, payload, gain_record)
        config = payload_config
        validate_training_config_structure(config)
        validate_runnable_training_config(config)

    with _legacy_data_preparation_scope(config):
        test_raw = None
        if config.data.test_data_file is not None:
            test_raw = load_data(
                str(config.data.test_data_file),
                n_images=None,
                n_subsample=None,
            )

        train_grouped = _group_raw_data(
            train_raw,
            config,
            Path(config.data.train_data_file),
            require_exact=request.resolved_synthetic_workflow is not None,
        )
        validation_grouped = None
        if test_raw is not None:
            validation_grouped = _group_raw_data(
                test_raw,
                config,
                Path(config.data.test_data_file),
                require_exact=request.resolved_synthetic_workflow is not None,
                group_count=(
                    request.resolved_synthetic_workflow.training.validation_groups
                    if request.resolved_synthetic_workflow is not None
                    else None
                ),
            )
        train_container = _materialize_backend_container(
            train_grouped,
            train_raw,
            config,
        )
        validation_container = (
            _materialize_backend_container(validation_grouped, test_raw, config)
            if validation_grouped is not None
            else None
        )

    amplitude, phase, backend_results = run_cdi_example_with_backend(
        train_container,
        validation_container,
        config,
        do_stitching=request.do_stitching,
        torch_execution_config=None,
        torch_factory_overrides=None,
        torch_resolved_payload=payload,
        torch_amplitude_physics_gain_record=gain_record,
        torch_training_seed=torch_training_seed,
    )
    if config.backend == "tensorflow":
        _persist_tensorflow_outputs(config, amplitude, phase, backend_results)

    bundle_path = backend_results.get("bundle_path")
    if bundle_path is not None:
        bundle_path = Path(bundle_path)
    elif config.backend == "pytorch":
        candidate = Path(config.output_dir) / "wts.h5.zip"
        if candidate.is_file():
            bundle_path = candidate

    if gain_record is not None:
        if bundle_path is None or not bundle_path.is_file():
            raise FileNotFoundError(
                f"synthetic training bundle was not written: {bundle_path}"
            )
        from ptycho.config.legacy_state import isolated_archived_params_scope

        with isolated_archived_params_scope():
            _, loaded = load_inference_bundle_torch(
                Path(config.output_dir),
            )
            if loaded.get("amplitude_physics_gain_record") is None:
                raise ValueError(
                    "strict reload did not return the persisted "
                    "amplitude_physics_gain_record"
                )

    return TrainingWorkflowResult(
        public_config=config,
        backend_results=backend_results,
        reconstruction_amplitude=amplitude,
        reconstruction_phase=phase,
        bundle_path=bundle_path,
        pt_data_config=(payload.pt_data_config if payload is not None else None),
        pt_model_config=(payload.pt_model_config if payload is not None else None),
        model_spec=(payload.model_spec if payload is not None else None),
        train_group_count=int(np.asarray(train_grouped["nn_indices"]).shape[0]),
        validation_group_count=(
            int(np.asarray(validation_grouped["nn_indices"]).shape[0])
            if validation_grouped is not None
            else 0
        ),
        amplitude_physics_gain_record=gain_record,
        amplitude_physics_gain_metadata=(
            gain_record.to_metadata() if gain_record is not None else None
        ),
        torch_training_seed=torch_training_seed,
    )


__all__ = [
    "TrainingWorkflowRequest",
    "TrainingWorkflowResult",
    "interpret_n_images_parameter",
    "interpret_sampling_parameters",
    "run_training_workflow",
]

"""Architecture-neutral canonical Torch execution for one ablation run.

This module drives the repository's canonical training, checkpoint-selection,
reload, mmap, and reassembly entry points in the exact order fixed by the
2026-07-09 CI model-compatibility ablation design ("Training And Inference
Flow"). It sees only resolved configuration objects and immutable dataset
paths; it contains no architecture selection logic of any kind.

Call order (verbatim contract, asserted by tests/studies/
test_torch_ablation_runtime.py):

1. ``train_lightning_only.main(existing_config=..., execution_config=...,
   seed=..., return_training_result=True)`` on a staged train-only NPZ dir.
2. ``lightning_utils.find_best_checkpoint``.
3. ``torch.load(best_checkpoint)["state_dict"]`` loaded into the in-memory
   trained model with ``load_state_dict(strict=True)`` — the pre-reload
   reference always carries the persisted best state, never final-epoch
   in-memory weights.
4. ``dataloader.PtychoDataset`` over the staged held-out NPZ.
5. ``reassembly.reconstruct_image_barycentric(best-state reference model,
   structured_diagnostics=True)``.
6. ``lightning_utils.load_checkpoint_with_configs`` (production loader).
7. ``reassembly.reconstruct_image_barycentric(reloaded model,
   structured_diagnostics=True)``.
8. ``reassembly.evaluate_fitted_count_metrics`` — CI count arms only; legacy
   arms record the typed not-applicable marker instead.
"""

from __future__ import annotations

import copy
import dataclasses
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import ResolvedTorchConfigs
from .runtime_errors import RuntimeExecutionError, sha256_file
from .runtime_errors import stage as _stage

#: Checkpoint-reload parity tolerances fixed by the design's stability gate.
RELOAD_RTOL = 1e-5
RELOAD_ATOL = 1e-6
#: Metric path under which the reload parity error is published.
RELOAD_METRIC_PATH = "stability.reload_max_abs_error"
RELOAD_ALLCLOSE_METRIC_PATH = "stability.reload_allclose"

_CI_SCALE_CONTRACT = "ci_intensity_v2"
_ACCELERATOR_CLASS_SUFFIX = {
    "cpu": ".CPUAccelerator",
    "cuda": ".CUDAAccelerator",
    "mps": ".MPSAccelerator",
}


@dataclass(frozen=True)
class MilestoneRunResult:
    """Compact canonical evaluation for one requested post-epoch checkpoint."""

    milestone_epoch: int
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_epoch: int
    training_history: Any
    reference_texture: Any
    reference_canvas: Any
    reference_diagnostics: Any
    reloaded_texture: Any
    reloaded_canvas: Any
    reloaded_diagnostics: Any
    count_metrics: Any
    reload_max_abs_error: float
    reload_allclose: bool


@dataclass(frozen=True)
class _CheckpointEvaluation:
    reference_texture: Any
    reference_canvas: Any
    reference_diagnostics: Any
    reloaded_texture: Any
    reloaded_canvas: Any
    reloaded_diagnostics: Any
    count_metrics: Any
    reload_max_abs_error: float
    count_sample_identity: dict[str, Any] | None


@dataclass(frozen=True)
class CanonicalRunResult:
    """Complete typed evidence produced by one canonical run."""

    run_dir: Path
    best_checkpoint: Path
    best_checkpoint_sha256: str
    best_checkpoint_epoch: int | None
    fixed_batch_identity: dict[str, Any]
    effective_runtime: dict[str, Any]
    reference_texture: Any
    reference_canvas: Any
    reference_diagnostics: Any
    reloaded_texture: Any
    reloaded_canvas: Any
    reloaded_diagnostics: Any
    count_metrics: Any
    reload_max_abs_error: float
    reload_allclose: bool
    train_seconds: float
    peak_memory_bytes: int | None
    #: JSON-able per-run training history from the trainer (losses, gradient
    #: norms, output statistics); None when the trainer produced none.
    training_history: Any = None
    milestones: tuple[MilestoneRunResult, ...] = ()


#: Backwards-compatible alias for the extracted torch-free helper.
_sha256_file = sha256_file


def _stage_single_npz(source: Path, directory: Path) -> Path:
    """Copy exactly one immutable NPZ into a fresh isolated directory."""
    source_path = Path(source)
    if not source_path.is_file():
        raise RuntimeExecutionError("staging", f"missing NPZ file: {source_path}")
    directory.mkdir(parents=True, exist_ok=False)
    staged = directory / source_path.name
    shutil.copy2(source_path, staged)
    entries = sorted(entry.name for entry in directory.iterdir())
    if entries != [staged.name]:
        raise RuntimeExecutionError(
            "staging", f"staged directory is not isolated: {entries}"
        )
    return staged


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _reload_device(accelerator: str) -> str:
    if accelerator == "gpu":
        return "cuda"
    if accelerator == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return accelerator


def _device_type(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split(":", 1)[0]


def _runtime_resolution_problems(
    *,
    requested_accelerator: Any,
    requested_devices: Any,
    requested_strategy: Any,
    effective_accelerator: Mapping[str, Any],
    effective_devices: Mapping[str, Any],
    effective_strategy: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> list[str]:
    """Validate resolved Lightning hardware evidence against supplied semantics."""
    problems: list[str] = []
    cuda_available = environment.get("cuda_available") is True
    cuda_count = environment.get("cuda_device_count")
    if (
        not isinstance(cuda_count, int)
        or isinstance(cuda_count, bool)
        or cuda_count < 0
    ):
        problems.append("effective.environment.cuda_device_count is invalid")
        cuda_count = 0
    mps_available = environment.get("mps_available") is True

    if requested_accelerator == "auto":
        expected_device_type = "cuda" if cuda_available else "cpu"
    elif requested_accelerator in {"gpu", "cuda"}:
        expected_device_type = "cuda"
    elif requested_accelerator in {"cpu", "mps"}:
        expected_device_type = requested_accelerator
    else:
        problems.append(
            f"requested.accelerator={requested_accelerator!r} is unsupported"
        )
        return problems

    actual_device_type = effective_accelerator.get("device_type")
    if actual_device_type != expected_device_type:
        problems.append(
            "effective.accelerator.device_type="
            f"{actual_device_type!r} expected {expected_device_type!r}"
        )
    accelerator_class = str(effective_accelerator.get("class", ""))
    expected_accelerator_suffix = _ACCELERATOR_CLASS_SUFFIX[expected_device_type]
    if not accelerator_class.endswith(expected_accelerator_suffix):
        problems.append(
            "effective.accelerator.class="
            f"{accelerator_class!r} expected *{expected_accelerator_suffix}"
        )
    if expected_device_type == "cuda" and not cuda_available:
        problems.append("effective accelerator resolved cuda without CUDA availability")
    if expected_device_type == "mps" and not mps_available:
        problems.append("effective accelerator resolved mps without MPS availability")

    expected_count = (
        cuda_count
        if requested_devices == "auto" and expected_device_type == "cuda"
        else 1
        if requested_devices == "auto"
        else requested_devices
    )
    actual_count = effective_devices.get("count")
    if actual_count != expected_count:
        problems.append(
            f"effective.devices.count={actual_count!r} expected {expected_count!r}"
        )
    actual_ids = effective_devices.get("ids")
    expected_ids = (
        list(range(expected_count)) if isinstance(expected_count, int) else []
    )
    if actual_ids != expected_ids:
        problems.append(
            f"effective.devices.ids={actual_ids!r} expected {expected_ids!r}"
        )

    root_device_type = _device_type(effective_strategy.get("root_device"))
    if root_device_type != expected_device_type:
        problems.append(
            "effective.strategy.root_device="
            f"{effective_strategy.get('root_device')!r} "
            f"expected {expected_device_type!r}"
        )
    strategy_class = str(effective_strategy.get("class", ""))
    if requested_strategy == "auto":
        expected_strategy_suffix = (
            ".SingleDeviceStrategy" if expected_count == 1 else ".DDPStrategy"
        )
    else:
        expected_strategy_suffix = {
            "ddp": ".DDPStrategy",
            "ddp_spawn": ".DDPStrategy",
            "fsdp": ".FSDPStrategy",
            "deepspeed": ".DeepSpeedStrategy",
        }.get(requested_strategy)
    if expected_strategy_suffix and not strategy_class.endswith(
        expected_strategy_suffix
    ):
        problems.append(
            "effective.strategy.class="
            f"{strategy_class!r} expected *{expected_strategy_suffix}"
        )
    parallel_devices = effective_strategy.get("parallel_devices", [])
    single_device_auto = requested_strategy == "auto" and expected_count == 1
    if single_device_auto:
        if parallel_devices not in ([], None):
            problems.append(
                "effective.strategy.parallel_devices must be empty for one-device auto"
            )
    elif (
        not isinstance(parallel_devices, list)
        or len(parallel_devices) != expected_count
        or any(
            _device_type(device) != expected_device_type for device in parallel_devices
        )
    ):
        problems.append(
            "effective.strategy.parallel_devices do not match resolved devices"
        )
    return problems


def _assert_reconstruction_precision(
    diagnostics: Any,
    expected_precision: str,
    stage: str,
) -> None:
    actual = getattr(diagnostics, "effective_precision", None)
    if actual != expected_precision:
        raise RuntimeExecutionError(
            stage,
            f"effective reconstruction precision {actual!r} "
            f"expected {expected_precision!r}",
        )


def _assert_effective_runtime(
    payload: Any, execution: Any, seed: int, model: Any
) -> None:
    """Hard-compare recorded effective runtime values with the resolved config."""
    if not isinstance(payload, Mapping):
        raise RuntimeExecutionError(
            "effective_runtime", "training result carries no effective runtime record"
        )
    problems: list[str] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{name}={actual!r} expected {expected!r}")

    requested = payload.get("requested")
    requested = requested if isinstance(requested, Mapping) else {}
    requested_loader = requested.get("dataloader")
    requested_loader = requested_loader if isinstance(requested_loader, Mapping) else {}
    effective = payload.get("effective")
    effective = effective if isinstance(effective, Mapping) else {}
    trainer_kwargs = payload.get("trainer_kwargs")
    trainer_kwargs = trainer_kwargs if isinstance(trainer_kwargs, Mapping) else {}
    effective_loader = effective.get("dataloader")
    effective_loader = effective_loader if isinstance(effective_loader, Mapping) else {}
    recorded_loader = payload.get("dataloader")
    recorded_loader = recorded_loader if isinstance(recorded_loader, Mapping) else {}
    effective_accelerator = effective.get("accelerator")
    effective_accelerator = (
        effective_accelerator if isinstance(effective_accelerator, Mapping) else {}
    )
    effective_devices = effective.get("devices")
    effective_devices = (
        effective_devices if isinstance(effective_devices, Mapping) else {}
    )
    effective_precision = effective.get("precision")
    effective_precision = (
        effective_precision if isinstance(effective_precision, Mapping) else {}
    )
    effective_strategy = effective.get("strategy")
    effective_strategy = (
        effective_strategy if isinstance(effective_strategy, Mapping) else {}
    )
    callbacks = effective.get("callbacks")
    callbacks = callbacks if isinstance(callbacks, list) else []
    effective_deterministic = effective.get("deterministic")
    effective_deterministic = (
        effective_deterministic if isinstance(effective_deterministic, Mapping) else {}
    )
    environment = effective.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}

    def expected_worker_settings() -> dict[str, Any]:
        worker_count = (
            0 if "spawn" in str(execution.strategy) else execution.num_workers
        )
        return {
            "num_workers": worker_count,
            "pin_memory": execution.pin_memory,
            "persistent_workers": (
                execution.persistent_workers if worker_count > 0 else False
            ),
            "prefetch_factor": (
                (execution.prefetch_factor or 2) if worker_count > 0 else None
            ),
        }

    def callback_with_suffix(suffix: str) -> Mapping[str, Any] | None:
        matches = [
            callback
            for callback in callbacks
            if isinstance(callback, Mapping)
            and str(callback.get("class", "")).endswith(suffix)
        ]
        if len(matches) != 1:
            problems.append(
                f"effective.callbacks.{suffix}=count {len(matches)!r} expected 1"
            )
            return None
        return matches[0]

    def callback_count(suffix: str) -> int:
        return sum(
            1
            for callback in callbacks
            if isinstance(callback, Mapping)
            and str(callback.get("class", "")).endswith(suffix)
        )

    check("seed", payload.get("seed"), seed)
    from ptycho_torch.reassembly import resolve_inference_precision_for_device

    effective_device_type = effective_accelerator.get("device_type")
    expected_precision = resolve_inference_precision_for_device(
        execution.precision,
        effective_device_type,
    )
    check("precision", payload.get("precision"), expected_precision)
    check("requested.accelerator", requested.get("accelerator"), execution.accelerator)
    check("requested.devices", requested.get("devices"), execution.devices)
    check("requested.strategy", requested.get("strategy"), execution.strategy)
    check(
        "requested.deterministic",
        requested.get("deterministic"),
        execution.deterministic,
    )
    check("requested.precision", requested.get("precision"), execution.precision)
    check(
        "requested.enable_progress_bar",
        requested.get("enable_progress_bar"),
        execution.enable_progress_bar,
    )
    check(
        "requested.enable_checkpointing",
        requested.get("enable_checkpointing"),
        execution.enable_checkpointing,
    )
    check(
        "requested.dataloader.num_workers",
        requested_loader.get("num_workers"),
        execution.num_workers,
    )
    check(
        "requested.dataloader.pin_memory",
        requested_loader.get("pin_memory"),
        execution.pin_memory,
    )
    check(
        "trainer_kwargs.accelerator",
        trainer_kwargs.get("accelerator"),
        "gpu" if execution.accelerator == "cuda" else execution.accelerator,
    )
    check("trainer_kwargs.devices", trainer_kwargs.get("devices"), execution.devices)
    if "strategy" not in trainer_kwargs:
        problems.append("trainer_kwargs.strategy is missing")
    check(
        "trainer_kwargs.deterministic",
        trainer_kwargs.get("deterministic"),
        execution.deterministic,
    )
    check(
        "trainer_kwargs.precision",
        trainer_kwargs.get("precision"),
        execution.precision,
    )
    check(
        "trainer_kwargs.enable_progress_bar",
        trainer_kwargs.get("enable_progress_bar"),
        execution.enable_progress_bar,
    )
    check(
        "trainer_kwargs.enable_checkpointing",
        trainer_kwargs.get("enable_checkpointing"),
        execution.enable_checkpointing,
    )
    check(
        "effective.accelerator.trainer_value",
        effective_accelerator.get("trainer_value"),
        trainer_kwargs.get("accelerator"),
    )
    expected_device_type = {"cuda": "cuda", "gpu": "cuda"}.get(
        execution.accelerator, execution.accelerator
    )
    if expected_device_type != "auto":
        check(
            "effective.accelerator.device_type",
            effective_accelerator.get("device_type"),
            expected_device_type,
        )
    check(
        "effective.devices.trainer_value",
        effective_devices.get("trainer_value"),
        trainer_kwargs.get("devices"),
    )
    check(
        "effective.strategy.trainer_value",
        effective_strategy.get("trainer_value"),
        trainer_kwargs.get("strategy"),
    )
    if not effective_strategy.get("class"):
        problems.append("effective.strategy.class is missing")
    check(
        "effective.precision.value",
        effective_precision.get("value"),
        expected_precision,
    )
    expected_deterministic = execution.deterministic in {True, "warn"}
    check(
        "effective.deterministic.algorithms_enabled",
        effective_deterministic.get("algorithms_enabled"),
        expected_deterministic,
    )
    check(
        "effective.deterministic.warn_only",
        effective_deterministic.get("warn_only"),
        execution.deterministic == "warn",
    )
    problems.extend(
        _runtime_resolution_problems(
            requested_accelerator=execution.accelerator,
            requested_devices=execution.devices,
            requested_strategy=execution.strategy,
            effective_accelerator=effective_accelerator,
            effective_devices=effective_devices,
            effective_strategy=effective_strategy,
            environment=environment,
        )
    )
    for name, expected in expected_worker_settings().items():
        check(f"effective.dataloader.{name}", effective_loader.get(name), expected)
        check(f"dataloader.{name}", recorded_loader.get(name), expected)
    if execution.num_workers > 0:
        check(
            "requested.dataloader.persistent_workers",
            requested_loader.get("persistent_workers"),
            execution.persistent_workers,
        )
        check(
            "requested.dataloader.prefetch_factor",
            requested_loader.get("prefetch_factor"),
            execution.prefetch_factor,
        )
    if execution.enable_checkpointing:
        expected_monitor = (
            model.val_loss_name
            if execution.checkpoint_monitor_metric == "val_loss"
            else execution.checkpoint_monitor_metric
        )
        checkpoint = callback_with_suffix(".ModelCheckpoint")
        early_stopping = callback_with_suffix(".EarlyStopping")
        if checkpoint is not None:
            check(
                "effective.callbacks.ModelCheckpoint.save_top_k",
                checkpoint.get("save_top_k"),
                execution.checkpoint_save_top_k,
            )
            check(
                "effective.callbacks.ModelCheckpoint.monitor",
                checkpoint.get("monitor"),
                expected_monitor,
            )
            check(
                "effective.callbacks.ModelCheckpoint.mode",
                checkpoint.get("mode"),
                execution.checkpoint_mode,
            )
        if early_stopping is not None:
            check(
                "effective.callbacks.EarlyStopping.patience",
                early_stopping.get("patience"),
                execution.early_stop_patience,
            )
            check(
                "effective.callbacks.EarlyStopping.monitor",
                early_stopping.get("monitor"),
                expected_monitor,
            )
            check(
                "effective.callbacks.EarlyStopping.mode",
                early_stopping.get("mode"),
                execution.checkpoint_mode,
            )
    else:
        for suffix in (".ModelCheckpoint", ".EarlyStopping"):
            if callback_count(suffix):
                problems.append(
                    f"effective.callbacks.{suffix} must be absent when disabled"
                )
    progress_count = callback_count(".TQDMProgressBar")
    expected_progress_count = int(execution.enable_progress_bar)
    if progress_count != expected_progress_count:
        problems.append(
            "effective.callbacks.TQDMProgressBar="
            f"count {progress_count!r} expected {expected_progress_count!r}"
        )
    if problems:
        raise RuntimeExecutionError(
            "effective_runtime",
            "effective runtime diverges from resolved execution config: "
            + "; ".join(problems),
        )


def _normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, Enum):
        return _normalized(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


_CONFIG_MEMBER_NAMES = ("data", "model", "training", "inference", "datagen")


def _assert_config_round_trip(configs: ResolvedTorchConfigs, persisted: Any) -> None:
    """Assert every persisted checkpoint config field matches resolution."""
    expected = configs.existing_config
    if not isinstance(persisted, (tuple, list)) or len(persisted) != len(expected):
        raise RuntimeExecutionError(
            "config_round_trip",
            "production loader must return the five persisted config objects",
        )
    mismatches: list[str] = []
    for name, expected_config, persisted_config in zip(
        _CONFIG_MEMBER_NAMES, expected, persisted
    ):
        try:
            expected_fields = dataclasses.asdict(expected_config)
            persisted_fields = dataclasses.asdict(persisted_config)
        except TypeError as error:
            raise RuntimeExecutionError(
                "config_round_trip",
                f"persisted {name} config is not a config dataclass: {error}",
            ) from error
        for key in sorted(set(expected_fields) | set(persisted_fields)):
            if _normalized(expected_fields.get(key)) != _normalized(
                persisted_fields.get(key)
            ):
                mismatches.append(f"{name}.{key}")
    if mismatches:
        raise RuntimeExecutionError(
            "config_round_trip",
            "persisted checkpoint configs diverge from the resolved configuration: "
            + ", ".join(mismatches),
        )


def _assert_reload_parity(
    reference_texture: Any,
    reference_canvas: Any,
    reloaded_texture: Any,
    reloaded_canvas: Any,
) -> float:
    """Enforce rtol/atol parity and return the published max abs error."""
    max_error = 0.0
    failures: list[str] = []
    pairs = (
        ("texture", reference_texture, reloaded_texture),
        ("stitched_canvas", reference_canvas, reloaded_canvas),
    )
    for name, reference, reloaded in pairs:
        reference_array = _as_array(reference)
        reloaded_array = _as_array(reloaded)
        if reference_array.shape != reloaded_array.shape:
            raise RuntimeExecutionError(
                "reload_parity",
                f"{name} shape changed after reload: "
                f"{reference_array.shape} -> {reloaded_array.shape}",
            )
        difference = np.abs(reference_array - reloaded_array)
        error = float(difference.max()) if difference.size else 0.0
        max_error = max(max_error, error)
        if not np.allclose(
            reloaded_array, reference_array, rtol=RELOAD_RTOL, atol=RELOAD_ATOL
        ):
            failures.append(f"{name} max_abs_error={error!r}")
    if failures:
        raise RuntimeExecutionError(
            "reload_parity",
            "production checkpoint reload broke parity with the persisted best "
            f"state (rtol={RELOAD_RTOL!r}, atol={RELOAD_ATOL!r}): "
            + "; ".join(failures),
        )
    return max_error


def _build_count_metric_loader(subset: Any, configs: ResolvedTorchConfigs) -> Any:
    """Build the deterministic held-out loader for the count second pass."""
    import torch

    from ptycho_torch.dataloader import Collate, TensorDictDataLoader

    device = torch.device(_reload_device(configs.execution_config.accelerator))
    return TensorDictDataLoader(
        subset,
        batch_size=configs.inference_config.batch_size,
        num_workers=configs.training_config.num_workers,
        collate_fn=Collate(device=device),
    )


def _strict_load_checkpoint(
    checkpoint: Path,
    model: Any,
    *,
    expected_epoch: int | None = None,
) -> int | None:
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    checkpoint_epoch = payload.get("epoch")
    if type(checkpoint_epoch) is not int or checkpoint_epoch < 0:
        checkpoint_epoch = None
    if expected_epoch is not None and checkpoint_epoch != expected_epoch:
        raise RuntimeExecutionError(
            "milestone_checkpoints",
            f"checkpoint {checkpoint} has payload epoch {checkpoint_epoch!r}; "
            f"expected {expected_epoch}",
        )
    return checkpoint_epoch


def _checkpoint_stage(name: str, milestone_epoch: int | None) -> str:
    if milestone_epoch is None:
        return name
    return f"milestone_epoch_{milestone_epoch:04d}_{name}"


def _evaluate_checkpoint(
    configs: ResolvedTorchConfigs,
    *,
    checkpoint: Path,
    trained_model: Any,
    heldout_dataset: Any,
    inference_precision: str,
    milestone_epoch: int | None = None,
) -> _CheckpointEvaluation:
    from ptycho_torch import lightning_utils, reassembly
    from ptycho_torch.reassembly_diagnostics import not_applicable

    execution = configs.execution_config
    with _stage(_checkpoint_stage("reference_reconstruction", milestone_epoch)):
        reference_canvas, _, reference_diagnostics, reference_texture = (
            reassembly.reconstruct_image_barycentric(
                trained_model,
                heldout_dataset,
                configs.training_config,
                configs.data_config,
                configs.model_config,
                configs.inference_config,
                verbose=False,
                structured_diagnostics=True,
                precision=inference_precision,
                compute_count_metrics=False,
            )
        )
    _assert_reconstruction_precision(
        reference_diagnostics,
        inference_precision,
        _checkpoint_stage("reference_precision", milestone_epoch),
    )

    with _stage(_checkpoint_stage("production_reload", milestone_epoch)):
        reloaded_model, persisted_configs = (
            lightning_utils.load_checkpoint_with_configs(
                str(checkpoint),
                type(trained_model),
                device=_reload_device(execution.accelerator),
            )
        )
    _assert_config_round_trip(configs, persisted_configs)

    with _stage(_checkpoint_stage("reloaded_reconstruction", milestone_epoch)):
        reloaded_canvas, reloaded_subset, reloaded_diagnostics, reloaded_texture = (
            reassembly.reconstruct_image_barycentric(
                reloaded_model,
                heldout_dataset,
                configs.training_config,
                configs.data_config,
                configs.model_config,
                configs.inference_config,
                verbose=False,
                structured_diagnostics=True,
                precision=inference_precision,
                compute_count_metrics=False,
            )
        )
    _assert_reconstruction_precision(
        reloaded_diagnostics,
        inference_precision,
        _checkpoint_stage("reloaded_precision", milestone_epoch),
    )
    reload_max_abs_error = _assert_reload_parity(
        reference_texture, reference_canvas, reloaded_texture, reloaded_canvas
    )

    count_sample_identity = None
    if configs.ci_scaling_active:
        with _stage(_checkpoint_stage("count_metrics", milestone_epoch)):
            if configs.profile.version != _CI_SCALE_CONTRACT:
                raise RuntimeExecutionError(
                    "count_metrics",
                    f"unexpected active scale profile {configs.profile.version!r}",
                )
            infer_loader = _build_count_metric_loader(reloaded_subset, configs)
            grouping_method = getattr(reloaded_subset, "group_coords_enabled", None)
            grouped = (
                bool(grouping_method())
                if callable(grouping_method)
                else bool(configs.model_config.object_big)
            )
            valid_per_file = getattr(reloaded_subset, "valid_indices_per_file", None)
            local_to_source_ids = (
                None if grouped or not valid_per_file else valid_per_file[0]
            )
            count_metrics = reassembly.evaluate_fitted_count_metrics(
                reloaded_model,
                infer_loader,
                configs.data_config,
                configs.model_config,
                s1=reloaded_diagnostics.s1,
                s2=reloaded_diagnostics.s2,
                device=_reload_device(execution.accelerator),
                scale_profile=configs.profile.version,
                precision=inference_precision,
                local_to_source_ids=local_to_source_ids,
            )
            sample_digest = getattr(count_metrics, "sample_identity_digest", "")
            if sample_digest:
                count_sample_identity = {
                    "policy": "exact_loader_source_ids_v1",
                    "sha256": sample_digest,
                    "n_samples": count_metrics.n_samples,
                }
    else:
        count_metrics = not_applicable()

    return _CheckpointEvaluation(
        reference_texture=reference_texture,
        reference_canvas=reference_canvas,
        reference_diagnostics=reference_diagnostics,
        reloaded_texture=reloaded_texture,
        reloaded_canvas=reloaded_canvas,
        reloaded_diagnostics=reloaded_diagnostics,
        count_metrics=count_metrics,
        reload_max_abs_error=reload_max_abs_error,
        count_sample_identity=count_sample_identity,
    )


def _training_history_prefix(history: Any, checkpoint_epoch: int) -> Any:
    if not isinstance(history, Mapping):
        return history
    prefix = copy.deepcopy(history)
    series = prefix.get("series")
    if not isinstance(series, Mapping):
        return prefix
    for entry in series.values():
        if not isinstance(entry, dict):
            continue
        epochs = entry.get("epoch")
        if not isinstance(epochs, list):
            continue
        indices = [
            index
            for index, epoch in enumerate(epochs)
            if type(epoch) is int and epoch <= checkpoint_epoch
        ]
        for key, values in tuple(entry.items()):
            if isinstance(values, list) and len(values) == len(epochs):
                entry[key] = [values[index] for index in indices]
    return prefix


def execute_canonical_run(
    configs: ResolvedTorchConfigs,
    *,
    seed: int,
    train_npz: Path,
    test_npz: Path,
    work_dir: Path,
    milestone_epochs: tuple[int, ...] = (),
) -> CanonicalRunResult:
    """Execute the canonical train/reload/mmap/reconstruct flow for one run."""
    if not isinstance(configs, ResolvedTorchConfigs):
        raise RuntimeExecutionError(
            "preflight", "configs must be a ResolvedTorchConfigs instance"
        )
    import torch

    from ptycho_torch import dataloader, lightning_utils, reassembly
    from ptycho_torch import train_lightning_only

    if any(type(epoch) is not int or epoch <= 0 for epoch in milestone_epochs):
        raise RuntimeExecutionError(
            "preflight", "milestone epochs must be positive integers"
        )
    if tuple(sorted(set(milestone_epochs))) != milestone_epochs:
        raise RuntimeExecutionError(
            "preflight", "milestone epochs must be strictly increasing"
        )

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    execution = configs.execution_config
    existing_config = configs.existing_config
    staged_train = _stage_single_npz(train_npz, work / "staged_train")
    peak_memory_device = torch.device(_reload_device(execution.accelerator))
    if peak_memory_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(peak_memory_device)

    train_start = time.perf_counter()
    with _stage("training"):
        training_kwargs = dict(
            existing_config=existing_config,
            output_dir=str(work / "training"),
            execution_config=execution,
            seed=seed,
            return_training_result=True,
        )
        if milestone_epochs:
            training_kwargs["milestone_epochs"] = milestone_epochs
        training = train_lightning_only.main(
            str(staged_train.parent), **training_kwargs
        )
    train_seconds = time.perf_counter() - train_start
    _assert_effective_runtime(
        training.effective_runtime,
        execution,
        seed,
        training.model,
    )
    effective_device_type = training.effective_runtime["effective"]["accelerator"][
        "device_type"
    ]
    inference_precision = reassembly.resolve_inference_precision_for_device(
        execution.precision,
        effective_device_type,
    )

    milestone_checkpoints: dict[int, Path] = {}
    if milestone_epochs:
        captured = getattr(training, "milestone_checkpoints", None)
        if not isinstance(captured, Mapping):
            raise RuntimeExecutionError(
                "milestone_checkpoints",
                "training did not return requested milestone checkpoints",
            )
        missing = [epoch for epoch in milestone_epochs if epoch not in captured]
        if missing:
            raise RuntimeExecutionError(
                "milestone_checkpoints",
                "missing requested milestone checkpoints: "
                + ", ".join(str(epoch) for epoch in missing),
            )
        milestone_checkpoints = {
            epoch: Path(captured[epoch]) for epoch in milestone_epochs
        }
        missing_files = [
            epoch
            for epoch, checkpoint in milestone_checkpoints.items()
            if not checkpoint.is_file()
        ]
        if missing_files:
            raise RuntimeExecutionError(
                "milestone_checkpoints",
                "requested milestone checkpoint files are missing: "
                + ", ".join(str(epoch) for epoch in missing_files),
            )

    with _stage("checkpoint_selection"):
        best_checkpoint = lightning_utils.find_best_checkpoint(Path(training.run_dir))
        if best_checkpoint is None:
            raise RuntimeExecutionError(
                "checkpoint_selection",
                f"no checkpoint found under {training.run_dir}",
            )
        best_checkpoint = Path(best_checkpoint)
        best_checkpoint_sha256 = _sha256_file(best_checkpoint)

    with _stage("best_state_reload"):
        checkpoint_epoch = _strict_load_checkpoint(best_checkpoint, training.model)

    staged_test = _stage_single_npz(test_npz, work / "staged_test")
    fixed_batch_identity = {
        "policy": "full_heldout_set_v1",
        "test_npz_sha256": _sha256_file(staged_test),
        "source_test_npz": str(Path(test_npz)),
    }

    with _stage("heldout_dataset"):
        heldout_dataset = dataloader.PtychoDataset(
            str(staged_test.parent),
            configs.model_config,
            configs.data_config,
            training_config=configs.training_config,
            data_dir=str(work / "heldout_mmap" / "memmap"),
            remake_map=True,
        )

    best_evaluation = _evaluate_checkpoint(
        configs,
        checkpoint=best_checkpoint,
        trained_model=training.model,
        heldout_dataset=heldout_dataset,
        inference_precision=inference_precision,
    )
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(peak_memory_device))
        if peak_memory_device.type == "cuda"
        else None
    )

    if best_evaluation.count_sample_identity is not None:
        fixed_batch_identity["count_sample_identity"] = (
            best_evaluation.count_sample_identity
        )

    training_history = getattr(training, "training_history", None)
    milestones: list[MilestoneRunResult] = []
    for milestone_epoch in milestone_epochs:
        milestone_checkpoint = milestone_checkpoints[milestone_epoch]
        milestone_sha256 = _sha256_file(milestone_checkpoint)
        with _stage(f"milestone_epoch_{milestone_epoch:04d}_state_reload"):
            payload_epoch = _strict_load_checkpoint(
                milestone_checkpoint,
                training.model,
                expected_epoch=milestone_epoch - 1,
            )
        evaluation = _evaluate_checkpoint(
            configs,
            checkpoint=milestone_checkpoint,
            trained_model=training.model,
            heldout_dataset=heldout_dataset,
            inference_precision=inference_precision,
            milestone_epoch=milestone_epoch,
        )
        milestones.append(
            MilestoneRunResult(
                milestone_epoch=milestone_epoch,
                checkpoint=milestone_checkpoint,
                checkpoint_sha256=milestone_sha256,
                checkpoint_epoch=payload_epoch,
                training_history=_training_history_prefix(
                    training_history, payload_epoch
                ),
                reference_texture=_as_array(evaluation.reference_texture),
                reference_canvas=_as_array(evaluation.reference_canvas),
                reference_diagnostics=evaluation.reference_diagnostics,
                reloaded_texture=_as_array(evaluation.reloaded_texture),
                reloaded_canvas=_as_array(evaluation.reloaded_canvas),
                reloaded_diagnostics=evaluation.reloaded_diagnostics,
                count_metrics=evaluation.count_metrics,
                reload_max_abs_error=evaluation.reload_max_abs_error,
                reload_allclose=True,
            )
        )
        del evaluation


    return CanonicalRunResult(
        run_dir=Path(training.run_dir),
        best_checkpoint=best_checkpoint,
        best_checkpoint_sha256=best_checkpoint_sha256,
        best_checkpoint_epoch=checkpoint_epoch,
        fixed_batch_identity=fixed_batch_identity,
        effective_runtime=dict(training.effective_runtime),
        reference_texture=best_evaluation.reference_texture,
        reference_canvas=best_evaluation.reference_canvas,
        reference_diagnostics=best_evaluation.reference_diagnostics,
        reloaded_texture=best_evaluation.reloaded_texture,
        reloaded_canvas=best_evaluation.reloaded_canvas,
        reloaded_diagnostics=best_evaluation.reloaded_diagnostics,
        count_metrics=best_evaluation.count_metrics,
        reload_max_abs_error=best_evaluation.reload_max_abs_error,
        reload_allclose=True,
        train_seconds=train_seconds,
        peak_memory_bytes=peak_memory_bytes,
        # Absent history (e.g. stubbed trainers) stays an explicit None; the
        # record layer publishes a typed absence instead of fabricating one.
        training_history=training_history,
        milestones=tuple(milestones),
    )

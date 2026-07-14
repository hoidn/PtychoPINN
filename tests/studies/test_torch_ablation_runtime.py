"""Canonical-call contract tests for the architecture-neutral ablation runtime.

These tests monkeypatch the exact Torch entry points the runtime must drive and
assert the verbatim call order from the Task 9 brief:

    train_lightning_only.main(return_training_result=True)
    find_best_checkpoint
    torch.load(best_checkpoint)["state_dict"] -> load_state_dict(strict=True)
    PtychoDataset
    reconstruct_image_barycentric(best_state_reference_model, structured_diagnostics=True)
    load_checkpoint_with_configs
    reconstruct_image_barycentric(reloaded_model, structured_diagnostics=True)
    evaluate_fitted_count_metrics  # CI only
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.studies.ablation.configuration import (
    ConfigResolutionError,
    resolve_torch_configs,
)
from scripts.studies.ablation import manifest
from scripts.studies.ablation import runtime
from scripts.studies.ablation import runtime_attempts
from scripts.studies.ablation import runtime_execution
from scripts.studies.ablation import runtime_records
from scripts.studies.ablation import runtime_study
from scripts.studies.ablation.artifacts import GitIdentity, TrainingFingerprintInput


# --------------------------------------------------------------------------
# Resolved-config helpers (mirrors the proven Task 5 minimal override sets).
# --------------------------------------------------------------------------


def _ci_overrides() -> dict[str, Any]:
    return {
        "data.N": 64,
        "data.C": 4,
        "data.scale_contract_version": "ci_intensity_v2",
        "data.measurement_domain": "count_intensity",
        "model.mode": "Unsupervised",
        "model.architecture": "hybrid_resnet",
        "model.generator_output_mode": "real_imag",
        "model.C_model": 4,
        "model.C_forward": 4,
        "model.physics_forward_mode": "rectangular_scaled",
        "model.rect_s1s2_trainable": True,
        "model.loss_function": "Poisson",
        "training.torch_loss_mode": "poisson",
        "training.learning_rate": 2e-4,
        "training.epochs": 1,
        "training.batch_size": 2,
        "inference.patch_weighting": "probe",
        "inference.varpro_scaling": True,
        "execution.accelerator": "cpu",
        "execution.devices": 1,
        "execution.strategy": "auto",
        "execution.precision": "32-true",
        "execution.num_workers": 0,
        "execution.pin_memory": False,
        "execution.enable_checkpointing": True,
    }


def _legacy_overrides() -> dict[str, Any]:
    overrides = _ci_overrides()
    overrides.update(
        {
            "data.scale_contract_version": "legacy_v1",
            "data.measurement_domain": "normalized_amplitude",
            "model.physics_forward_mode": "amplitude",
            "model.loss_function": "MAE",
            "training.torch_loss_mode": "mae",
            "inference.varpro_scaling": False,
        }
    )
    del overrides["model.rect_s1s2_trainable"]
    return overrides


def _effective_runtime(execution: Any, seed: int) -> dict[str, Any]:
    """Build an effective-runtime payload consistent with the execution config."""
    device_type = (
        "cuda" if execution.accelerator in {"cuda", "gpu"} else execution.accelerator
    )
    trainer_accelerator = (
        "gpu" if execution.accelerator == "cuda" else execution.accelerator
    )
    device_count = execution.devices if isinstance(execution.devices, int) else 1
    effective_precision = (
        "bf16-mixed"
        if device_type == "cpu" and execution.precision == "16-mixed"
        else execution.precision
    )
    deterministic_enabled = execution.deterministic in {True, "warn"}
    return {
        "seed": seed,
        "precision": effective_precision,
        "requested": {
            "accelerator": execution.accelerator,
            "devices": execution.devices,
            "strategy": execution.strategy,
            "deterministic": execution.deterministic,
            "precision": execution.precision,
            "enable_progress_bar": execution.enable_progress_bar,
            "enable_checkpointing": execution.enable_checkpointing,
            "checkpoint_save_top_k": execution.checkpoint_save_top_k,
            "checkpoint_monitor_metric": execution.checkpoint_monitor_metric,
            "checkpoint_mode": execution.checkpoint_mode,
            "early_stop_patience": execution.early_stop_patience,
            "dataloader": {
                "num_workers": execution.num_workers,
                "pin_memory": execution.pin_memory,
                "persistent_workers": execution.persistent_workers,
                "prefetch_factor": execution.prefetch_factor,
            },
        },
        "effective": {
            "precision": {"value": effective_precision, "plugin": "StubPrecision"},
            "deterministic": {
                "algorithms_enabled": deterministic_enabled,
                "warn_only": execution.deterministic == "warn",
            },
            "environment": {
                "cuda_available": device_type == "cuda",
                "cuda_device_count": device_count if device_type == "cuda" else 0,
                "mps_available": device_type == "mps",
            },
            "accelerator": {
                "class": f"Stub.{device_type.upper()}Accelerator",
                "device_type": device_type,
                "trainer_value": trainer_accelerator,
            },
            "devices": {
                "count": device_count,
                "ids": list(range(device_count)),
                "trainer_value": execution.devices,
            },
            "strategy": {
                "class": (
                    "Stub.SingleDeviceStrategy"
                    if device_count == 1
                    else "Stub.DDPStrategy"
                ),
                "trainer_value": execution.strategy,
                "root_device": device_type,
                "parallel_devices": (
                    [] if device_count == 1 else [device_type] * device_count
                ),
            },
            "callbacks": (
                [
                    {
                        "class": "Stub.ModelCheckpoint",
                        "monitor": execution.checkpoint_monitor_metric,
                        "mode": execution.checkpoint_mode,
                        "save_top_k": execution.checkpoint_save_top_k,
                    },
                    {
                        "class": "Stub.EarlyStopping",
                        "monitor": execution.checkpoint_monitor_metric,
                        "mode": execution.checkpoint_mode,
                        "patience": execution.early_stop_patience,
                    },
                ]
                if execution.enable_checkpointing
                else []
            )
            + (
                [{"class": "Stub.TQDMProgressBar"}]
                if execution.enable_progress_bar
                else []
            ),
            "loggers": [],
            "dataloader": {
                "num_workers": execution.num_workers,
                "pin_memory": execution.pin_memory,
                "persistent_workers": (
                    execution.persistent_workers if execution.num_workers > 0 else False
                ),
                "prefetch_factor": (
                    execution.prefetch_factor if execution.num_workers > 0 else None
                ),
            },
        },
        "trainer_kwargs": {
            "accelerator": trainer_accelerator,
            "devices": execution.devices,
            "strategy": execution.strategy,
            "deterministic": execution.deterministic,
            "precision": execution.precision,
            "enable_progress_bar": execution.enable_progress_bar,
            "enable_checkpointing": execution.enable_checkpointing,
        },
        "dataloader": {
            "num_workers": execution.num_workers,
            "pin_memory": execution.pin_memory,
            "persistent_workers": (
                execution.persistent_workers if execution.num_workers > 0 else False
            ),
            "prefetch_factor": (
                execution.prefetch_factor if execution.num_workers > 0 else None
            ),
        },
    }


class _StubModel:
    """Duck-typed in-memory trained model recording best-state loading."""

    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self._calls = calls
        self.loaded_state: Any = None
        self.val_loss_name = "val_loss"

    def load_state_dict(self, state_dict: Any, strict: bool = False) -> None:
        self._calls.append(("load_state_dict", strict, state_dict))
        self.loaded_state = state_dict


class _StubDiagnostics:
    schema_version = 1
    s1 = 1.25
    s2 = 0.5
    condition = 3.0
    unit_objective = 2.0
    fitted_objective = 1.5
    inference_time = 0.5
    assembly_time = 0.25
    solve_time = 0.01
    accepted_patches = 24
    total_patches = 25
    patches_accepted = 24
    patches_total = 25
    count_metrics = None

    def __init__(self, canvas_shape: tuple[int, int], effective_precision: str) -> None:
        self.effective_precision = effective_precision
        height, width = canvas_shape
        scan = (width // 2 - 0.5, height // 2 - 0.5)
        self.canvas_anchor = {
            "scan_com": np.asarray(scan, dtype=np.float64),
            "canvas_shape": (height, width),
            "canvas_origin_offset": (
                width // 2 - scan[0],
                height // 2 - scan[1],
            ),
        }
        self.canvas_weights = np.ones(canvas_shape, dtype=np.float32)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "s1": self.s1,
            "s2": self.s2,
        }


class _Harness:
    """Installs entry-point stubs and records the exact call sequence."""

    CHECKPOINT_BYTES = b"stub-checkpoint-payload"

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        resolved: Any,
        *,
        reload_delta: float = 0.0,
        effective_runtime: dict[str, Any] | None = None,
        persisted_configs: tuple[Any, ...] | None = None,
        training_history: dict[str, Any] | None = None,
        reference_effective_precision: str | None = None,
        reloaded_effective_precision: str | None = None,
        milestone_epochs: tuple[int, ...] = (),
        missing_milestone_epochs: tuple[int, ...] = (),
        milestone_payload_epochs: dict[int, int] | None = None,
    ) -> None:
        import torch
        from ptycho_torch import dataloader, lightning_utils, reassembly
        from ptycho_torch import train_lightning_only
        from ptycho_torch.reassembly_diagnostics import FittedCountMetrics

        self.calls: list[tuple[Any, ...]] = []
        self.resolved = resolved
        self.trained_model = _StubModel(self.calls)
        self.reloaded_model = _StubModel(self.calls)
        self.state_dict = {"weight": 1.0}
        self.canvas = np.linspace(0.0, 1.0, 144, dtype=np.float64).reshape(
            12, 12
        ) + 1j * np.linspace(1.0, 0.0, 144, dtype=np.float64).reshape(12, 12)
        self.reload_delta = reload_delta
        self.subset = SimpleNamespace(name="stub-subset")
        self.count_result = FittedCountMetrics(
            relative_l2_intensity_error=0.05,
            mean_raw_poisson_nll=1.5,
            n_samples=25,
            n_pixels=1600,
            effective_mask_digest="stub-mask-digest",
        )
        self.run_dir = tmp_path / "training-run"
        self.best_checkpoint = self.run_dir / "checkpoints" / "best-checkpoint.ckpt"
        self.effective_runtime_payload = effective_runtime
        self.persisted_configs = persisted_configs
        self.training_history = training_history
        expected_precision = _effective_runtime(resolved.execution_config, 11)[
            "precision"
        ]
        self.reference_effective_precision = (
            reference_effective_precision or expected_precision
        )
        self.reloaded_effective_precision = (
            reloaded_effective_precision or expected_precision
        )
        self.dataset_instances: list[Any] = []
        self.reconstruction_datasets: list[Any] = []
        self.train_kwargs: dict[str, Any] = {}
        self.milestone_checkpoints = {
            epoch: self.run_dir / "checkpoints" / "milestones" / f"epoch-{epoch:04d}.ckpt"
            for epoch in milestone_epochs
            if epoch not in missing_milestone_epochs
        }
        self.milestone_payload_epochs = milestone_payload_epochs or {}

        train_seed = {}
        harness = self

        def train_main(ptycho_dir: str, *args: Any, **kwargs: Any) -> Any:
            harness.train_kwargs = dict(kwargs)
            staged = tuple(sorted(p.name for p in Path(ptycho_dir).iterdir()))
            harness.calls.append(
                (
                    "train_main",
                    staged,
                    kwargs.get("existing_config"),
                    kwargs.get("execution_config"),
                    kwargs.get("seed"),
                    kwargs.get("return_training_result"),
                )
            )
            train_seed["seed"] = kwargs.get("seed")
            harness.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            harness.best_checkpoint.write_bytes(harness.CHECKPOINT_BYTES)
            for checkpoint in harness.milestone_checkpoints.values():
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(
                    harness.CHECKPOINT_BYTES + checkpoint.name.encode("ascii")
                )
            payload = (
                harness.effective_runtime_payload
                if harness.effective_runtime_payload is not None
                else _effective_runtime(kwargs["execution_config"], kwargs.get("seed"))
            )
            training_result = SimpleNamespace(
                run_dir=harness.run_dir,
                model=harness.trained_model,
                effective_runtime=payload,
                milestone_checkpoints=dict(harness.milestone_checkpoints),
            )
            if harness.training_history is not None:
                training_result.training_history = harness.training_history
            return training_result

        def find_best_checkpoint(run_dir: Path) -> Path:
            harness.calls.append(("find_best_checkpoint", Path(run_dir)))
            return harness.best_checkpoint

        def torch_load(path: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            harness.calls.append(("torch_load", str(path)))
            checkpoint = Path(path)
            external_epoch = next(
                (
                    epoch
                    for epoch, milestone_path in harness.milestone_checkpoints.items()
                    if milestone_path == checkpoint
                ),
                None,
            )
            payload_epoch = (
                3
                if external_epoch is None
                else harness.milestone_payload_epochs.get(
                    external_epoch, external_epoch - 1
                )
            )
            return {"state_dict": harness.state_dict, "epoch": payload_epoch}

        class StubPtychoDataset:
            def __init__(
                self,
                ptycho_dir: str,
                model_config: Any,
                data_config: Any,
                training_config: Any = None,
                data_dir: str = "data/memmap",
                **kwargs: Any,
            ) -> None:
                staged = tuple(sorted(p.name for p in Path(ptycho_dir).iterdir()))
                harness.calls.append(
                    ("ptycho_dataset", staged, model_config, data_config)
                )
                self.remake_map = kwargs.get("remake_map")
                harness.dataset_instances.append(self)

        def reconstruct(
            model: Any,
            ptycho_dset: Any,
            training_config: Any,
            data_config: Any,
            model_config: Any,
            inference_config: Any,
            **kwargs: Any,
        ) -> tuple[Any, Any, Any, Any]:
            harness.reconstruction_datasets.append(ptycho_dset)
            harness.calls.append(
                (
                    "reconstruct_image_barycentric",
                    model,
                    inference_config,
                    kwargs.get("structured_diagnostics"),
                    kwargs.get("precision"),
                    kwargs.get("compute_count_metrics"),
                )
            )
            if model is harness.trained_model:
                canvas = harness.canvas.copy()
            else:
                canvas = harness.canvas + harness.reload_delta
            diagnostics_precision = (
                harness.reference_effective_precision
                if model is harness.trained_model
                else harness.reloaded_effective_precision
            )
            diagnostics = _StubDiagnostics(canvas.shape, diagnostics_precision)
            return canvas, harness.subset, diagnostics, canvas.copy()

        def load_checkpoint_with_configs(
            checkpoint_path: str, model_class: Any, device: str = "cuda", **kwargs: Any
        ) -> tuple[Any, tuple[Any, ...]]:
            harness.calls.append(
                ("load_checkpoint_with_configs", checkpoint_path, model_class, device)
            )
            persisted = (
                harness.persisted_configs
                if harness.persisted_configs is not None
                else harness.resolved.existing_config
            )
            return harness.reloaded_model, persisted

        def evaluate_fitted_count_metrics(
            model: Any,
            infer_loader: Any,
            data_config: Any,
            model_config: Any,
            **kwargs: Any,
        ) -> Any:
            harness.calls.append(
                (
                    "evaluate_fitted_count_metrics",
                    model,
                    kwargs.get("s1"),
                    kwargs.get("s2"),
                    kwargs.get("scale_profile"),
                    kwargs.get("precision"),
                )
            )
            return harness.count_result

        monkeypatch.setattr(train_lightning_only, "main", train_main)
        monkeypatch.setattr(
            lightning_utils, "find_best_checkpoint", find_best_checkpoint
        )
        monkeypatch.setattr(torch, "load", torch_load)
        monkeypatch.setattr(dataloader, "PtychoDataset", StubPtychoDataset)
        monkeypatch.setattr(reassembly, "reconstruct_image_barycentric", reconstruct)
        monkeypatch.setattr(
            lightning_utils,
            "load_checkpoint_with_configs",
            load_checkpoint_with_configs,
        )
        monkeypatch.setattr(
            reassembly, "evaluate_fitted_count_metrics", evaluate_fitted_count_metrics
        )

    def names(self) -> list[str]:
        return [entry[0] for entry in self.calls]


def _write_npz_pair(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "immutable-data"
    data_dir.mkdir()
    train = data_dir / "train.npz"
    test = data_dir / "test.npz"
    train.write_bytes(b"immutable-train-npz")
    test.write_bytes(b"immutable-test-npz")
    return train, test


def _execute(
    harness: _Harness,
    tmp_path: Path,
    *,
    seed: int = 11,
    milestone_epochs: tuple[int, ...] = (),
) -> Any:
    train, test = _write_npz_pair(tmp_path)
    return runtime.execute_canonical_run(
        harness.resolved,
        seed=seed,
        train_npz=train,
        test_npz=test,
        work_dir=tmp_path / "work",
        milestone_epochs=milestone_epochs,
    )


CI_EXPECTED_ORDER = [
    "train_main",
    "find_best_checkpoint",
    "torch_load",
    "load_state_dict",
    "ptycho_dataset",
    "reconstruct_image_barycentric",
    "load_checkpoint_with_configs",
    "reconstruct_image_barycentric",
    "evaluate_fitted_count_metrics",
]


def test_ci_run_follows_exact_canonical_call_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    _execute(harness, tmp_path)

    assert harness.names() == CI_EXPECTED_ORDER


def test_milestone_checkpoint_evaluation_reuses_canonical_call_order_and_mmap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested = (5, 20, 40, 80)
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved, milestone_epochs=requested)

    result = _execute(harness, tmp_path, milestone_epochs=requested)

    assert harness.train_kwargs["milestone_epochs"] == requested
    assert harness.names() == CI_EXPECTED_ORDER + [
        name
        for _ in requested
        for name in (
            "torch_load",
            "load_state_dict",
            "reconstruct_image_barycentric",
            "load_checkpoint_with_configs",
            "reconstruct_image_barycentric",
            "evaluate_fitted_count_metrics",
        )
    ]
    assert len(harness.dataset_instances) == 1
    assert set(map(id, harness.reconstruction_datasets)) == {
        id(harness.dataset_instances[0])
    }
    assert result.best_checkpoint == harness.best_checkpoint
    assert result.best_checkpoint_epoch == 3
    assert tuple(milestone.milestone_epoch for milestone in result.milestones) == requested
    assert tuple(milestone.checkpoint_epoch for milestone in result.milestones) == (
        4,
        19,
        39,
        79,
    )
    assert all(milestone.count_metrics is harness.count_result for milestone in result.milestones)


def test_milestone_capture_does_not_drift_main_records_or_verdict_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ticks = iter((0.0, 1.0, 10.0, 11.0))
    monkeypatch.setattr(runtime_execution.time, "perf_counter", lambda: next(ticks))
    resolved = resolve_torch_configs(_ci_overrides())

    absent_root = tmp_path / "absent"
    absent_root.mkdir()
    absent_harness = _Harness(monkeypatch, absent_root, resolved)
    absent = _execute(absent_harness, absent_root)

    enabled_root = tmp_path / "enabled"
    enabled_root.mkdir()
    enabled_harness = _Harness(
        monkeypatch, enabled_root, resolved, milestone_epochs=(5,)
    )
    enabled = _execute(enabled_harness, enabled_root, milestone_epochs=(5,))

    descriptor = SimpleNamespace(truth="none", truth_location="none", test=None)
    absent_records, absent_arrays = runtime.build_run_metric_records(
        resolved, absent, descriptor
    )
    enabled_records, enabled_arrays = runtime.build_run_metric_records(
        resolved, enabled, descriptor
    )

    assert enabled.best_checkpoint_sha256 == absent.best_checkpoint_sha256
    assert enabled.best_checkpoint_epoch == absent.best_checkpoint_epoch
    assert runtime_records.records_to_payload(
        enabled_records
    ) == runtime_records.records_to_payload(absent_records)
    assert enabled_arrays.keys() == absent_arrays.keys()
    for name in absent_arrays:
        assert np.array_equal(enabled_arrays[name], absent_arrays[name])

    run = SimpleNamespace(id="run-1", arm_id="arm-1", dataset_id="dataset", seed=3)
    absent_attempt, _ = runtime_attempts.success_rows(
        run, "none", absent_records, absent_arrays
    )
    enabled_attempt, _ = runtime_attempts.success_rows(
        run, "none", enabled_records, enabled_arrays
    )
    assert enabled_attempt == absent_attempt


def test_milestones_preserve_main_cuda_peak_and_retain_cpu_arrays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    overrides = _ci_overrides()
    overrides["execution.accelerator"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    resolved = resolve_torch_configs(overrides)
    main_peak = 123_456_789

    def execute_case(
        root: Path, milestone_epochs: tuple[int, ...]
    ) -> tuple[Any, _Harness]:
        from ptycho_torch import reassembly

        root.mkdir()
        harness = _Harness(
            monkeypatch, root, resolved, milestone_epochs=milestone_epochs
        )
        reconstruct = reassembly.reconstruct_image_barycentric

        def reconstruct_with_tensors(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            canvas, subset, diagnostics, texture = reconstruct(*args, **kwargs)
            return (
                torch.as_tensor(canvas),
                subset,
                diagnostics,
                torch.as_tensor(texture),
            )

        def reset_peak(device: Any) -> None:
            harness.calls.append(("reset_peak_memory_stats", str(device)))

        def max_allocated(device: Any) -> int:
            reconstruction_count = harness.names().count(
                "reconstruct_image_barycentric"
            )
            harness.calls.append(
                ("max_memory_allocated", str(device), reconstruction_count)
            )
            return main_peak if reconstruction_count == 2 else main_peak + 1

        monkeypatch.setattr(
            reassembly, "reconstruct_image_barycentric", reconstruct_with_tensors
        )
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_allocated)
        return _execute(
            harness, root, milestone_epochs=milestone_epochs
        ), harness

    absent, absent_harness = execute_case(tmp_path / "absent", ())
    enabled, enabled_harness = execute_case(tmp_path / "enabled", (5,))

    assert absent.peak_memory_bytes == main_peak
    assert enabled.peak_memory_bytes == main_peak
    assert absent_harness.names().count("max_memory_allocated") == 1
    assert enabled_harness.names().count("max_memory_allocated") == 1
    milestone = enabled.milestones[0]
    assert all(
        isinstance(value, np.ndarray)
        for value in (
            milestone.reference_canvas,
            milestone.reference_texture,
            milestone.reloaded_canvas,
            milestone.reloaded_texture,
        )
    )
    assert all(
        isinstance(value, torch.Tensor)
        for value in (
            enabled.reference_canvas,
            enabled.reference_texture,
            enabled.reloaded_canvas,
            enabled.reloaded_texture,
        )
    )


def test_prior_milestone_reconstruction_tensors_are_released_before_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import gc
    import weakref

    import torch
    from ptycho_torch import reassembly

    requested = (5, 20)
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved, milestone_epochs=requested)
    reconstruct = reassembly.reconstruct_image_barycentric
    first_milestone_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    alive_at_second_milestone: list[int] = []

    def reconstruct_with_lifetime_probe(
        *args: Any, **kwargs: Any
    ) -> tuple[Any, ...]:
        if len(harness.reconstruction_datasets) == 4:
            gc.collect()
            alive_at_second_milestone.append(
                sum(reference() is not None for reference in first_milestone_refs)
            )
        canvas, subset, diagnostics, texture = reconstruct(*args, **kwargs)
        tensor_canvas = torch.as_tensor(canvas)
        tensor_texture = torch.as_tensor(texture)
        if len(harness.reconstruction_datasets) in (3, 4):
            first_milestone_refs.extend(
                (weakref.ref(tensor_canvas), weakref.ref(tensor_texture))
            )
        return tensor_canvas, subset, diagnostics, tensor_texture

    monkeypatch.setattr(
        reassembly, "reconstruct_image_barycentric", reconstruct_with_lifetime_probe
    )

    result = _execute(harness, tmp_path, milestone_epochs=requested)

    assert len(result.milestones) == 2
    assert alive_at_second_milestone == [0]


def test_missing_requested_milestone_fails_without_best_checkpoint_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested = (5, 20)
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(
        monkeypatch,
        tmp_path,
        resolved,
        milestone_epochs=requested,
        missing_milestone_epochs=(20,),
    )

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path, milestone_epochs=requested)

    assert excinfo.value.stage == "milestone_checkpoints"
    assert "20" in str(excinfo.value)
    assert "find_best_checkpoint" not in harness.names()


def test_milestone_history_is_prefix_through_payload_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    history = {
        "schema_version": "training_history_v1",
        "series": {
            "val_loss": {
                "step": [1, 2, 3, 4, 5, 6],
                "epoch": [0, 1, 2, 3, 4, 5],
                "value": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        },
    }
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(
        monkeypatch,
        tmp_path,
        resolved,
        milestone_epochs=(5,),
        training_history=history,
    )

    result = _execute(harness, tmp_path, milestone_epochs=(5,))

    assert result.milestones[0].training_history["series"]["val_loss"] == {
        "step": [1, 2, 3, 4, 5],
        "epoch": [0, 1, 2, 3, 4],
        "value": [6.0, 5.0, 4.0, 3.0, 2.0],
    }


def test_training_call_stages_train_only_npz_and_full_config_tuple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    _execute(harness, tmp_path, seed=17)

    _, staged, existing_config, execution_config, seed, return_result = harness.calls[0]
    assert staged == ("train.npz",)
    assert len(existing_config) == 5
    assert existing_config == resolved.existing_config
    assert existing_config[4] is resolved.datagen_config
    assert execution_config is resolved.execution_config
    assert seed == 17
    assert return_result is True


def test_heldout_dataset_initializes_its_isolated_memory_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    _execute(harness, tmp_path)

    assert harness.dataset_instances[0].remake_map is True


def test_best_state_is_loaded_strict_into_in_memory_model_before_reference_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    _execute(harness, tmp_path)

    torch_load_call = harness.calls[2]
    assert torch_load_call == ("torch_load", str(harness.best_checkpoint))
    load_call = harness.calls[3]
    assert load_call == ("load_state_dict", True, harness.state_dict)
    # The reference pass runs on the in-memory model carrying the best state,
    # never on a separately loaded model and never on final-epoch weights.
    reference_call = harness.calls[5]
    assert reference_call[0] == "reconstruct_image_barycentric"
    assert reference_call[1] is harness.trained_model
    reloaded_call = harness.calls[7]
    assert reloaded_call[1] is harness.reloaded_model


def test_both_reconstructions_use_resolved_inference_config_and_precision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    _execute(harness, tmp_path)

    reconstructions = [
        call for call in harness.calls if call[0] == "reconstruct_image_barycentric"
    ]
    assert len(reconstructions) == 2
    for call in reconstructions:
        assert call[2] is resolved.inference_config
        assert call[3] is True
        assert call[4] == resolved.execution_config.precision
        assert call[5] is False


@pytest.mark.parametrize(
    ("precision_kwarg", "expected"),
    [("bf16-mixed", "bf16-mixed")],
)
def test_cpu_mixed_precision_uses_semantic_effective_precision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    precision_kwarg: str,
    expected: str,
) -> None:
    overrides = _ci_overrides()
    overrides["execution.precision"] = precision_kwarg
    resolved = resolve_torch_configs(overrides)
    harness = _Harness(monkeypatch, tmp_path, resolved)

    _execute(harness, tmp_path)

    reconstruction_precisions = [
        call[4] for call in harness.calls if call[0] == "reconstruct_image_barycentric"
    ]
    assert reconstruction_precisions == [expected, expected]
    assert harness.calls[-1][5] == expected


@pytest.mark.parametrize(
    ("reference_precision", "reloaded_precision", "expected_stage"),
    [
        ("16-mixed", "32-true", "reference_precision"),
        ("32-true", "16-mixed", "reloaded_precision"),
    ],
)
def test_reconstruction_effective_precision_mismatch_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reference_precision: str,
    reloaded_precision: str,
    expected_stage: str,
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(
        monkeypatch,
        tmp_path,
        resolved,
        reference_effective_precision=reference_precision,
        reloaded_effective_precision=reloaded_precision,
    )

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == expected_stage


def test_count_pass_uses_reloaded_model_and_reloaded_scales(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    result = _execute(harness, tmp_path)

    evaluate_call = harness.calls[-1]
    assert evaluate_call[0] == "evaluate_fitted_count_metrics"
    assert evaluate_call[1] is harness.reloaded_model
    diagnostics = _StubDiagnostics((12, 12), "32-true")
    assert evaluate_call[2] == pytest.approx(diagnostics.s1)
    assert evaluate_call[3] == pytest.approx(diagnostics.s2)
    assert evaluate_call[4] == "ci_intensity_v2"
    assert evaluate_call[5] == "32-true"
    assert result.count_metrics is harness.count_result


def test_cuda_peak_memory_covers_training_reload_reassembly_and_count_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    overrides = _ci_overrides()
    overrides["execution.accelerator"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    resolved = resolve_torch_configs(overrides)
    harness = _Harness(monkeypatch, tmp_path, resolved)

    def reset_peak(device: Any) -> None:
        harness.calls.append(("reset_peak_memory_stats", str(device)))

    def max_allocated(device: Any) -> int:
        harness.calls.append(("max_memory_allocated", str(device)))
        return 987_654_321

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_allocated)

    result = _execute(harness, tmp_path)

    assert result.peak_memory_bytes == 987_654_321
    assert harness.names().count("reset_peak_memory_stats") == 1
    assert harness.names().count("max_memory_allocated") == 1
    assert harness.names().index("reset_peak_memory_stats") < harness.names().index(
        "train_main"
    )
    assert harness.names().index("max_memory_allocated") > harness.names().index(
        "evaluate_fitted_count_metrics"
    )


def test_cpu_run_does_not_report_or_query_cuda_peak_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    def unexpected_cuda_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU execution must not query CUDA peak memory")

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", unexpected_cuda_call)

    result = _execute(harness, tmp_path)

    assert result.peak_memory_bytes is None


def test_persists_checkpoint_sha_and_fixed_batch_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    result = _execute(harness, tmp_path)

    assert result.best_checkpoint == harness.best_checkpoint
    assert (
        result.best_checkpoint_sha256
        == hashlib.sha256(_Harness.CHECKPOINT_BYTES).hexdigest()
    )
    assert result.best_checkpoint_epoch == 3
    assert (
        result.fixed_batch_identity["test_npz_sha256"]
        == hashlib.sha256(b"immutable-test-npz").hexdigest()
    )


def test_legacy_run_skips_count_pass_with_typed_not_applicable_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_legacy_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    result = _execute(harness, tmp_path)

    assert harness.names() == CI_EXPECTED_ORDER[:-1]
    assert result.count_metrics.to_jsonable() == {
        "status": "not_applicable",
        "reason": "legacy_normalized_amplitude",
    }


def test_reload_parity_tolerances_and_stability_namespace_are_pinned() -> None:
    assert runtime.RELOAD_RTOL == 1e-5
    assert runtime.RELOAD_ATOL == 1e-6
    assert runtime.RELOAD_METRIC_PATH == "stability.reload_max_abs_error"
    assert runtime_execution.RELOAD_ALLCLOSE_METRIC_PATH == "stability.reload_allclose"


def test_reload_parity_violation_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved, reload_delta=1e-3)

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "reload_parity"
    assert "1e-3" in str(excinfo.value) or "0.001" in str(excinfo.value)


def test_reload_max_error_within_tolerance_is_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    delta = 5e-7
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved, reload_delta=delta)

    result = _execute(harness, tmp_path)

    assert result.reload_max_abs_error == pytest.approx(delta, rel=1e-6)
    assert result.reload_allclose is True


def test_effective_runtime_mismatch_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    payload = _effective_runtime(resolved.execution_config, 11)
    payload["precision"] = "16-mixed"
    harness = _Harness(monkeypatch, tmp_path, resolved, effective_runtime=payload)

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "effective_runtime"


def test_canonical_auto_devices_cannot_reach_runtime_or_heldout_mmap() -> None:
    overrides = _ci_overrides()
    overrides["execution.devices"] = "auto"

    with pytest.raises(
        ConfigResolutionError,
        match=r"held-out mmap/reassembly.*single-device-only",
    ):
        resolve_torch_configs(overrides)


def _cpu_auto_resolution_evidence() -> dict[str, Any]:
    return {
        "effective_accelerator": {
            "class": "lightning.pytorch.accelerators.CPUAccelerator",
            "device_type": "cpu",
        },
        "effective_devices": {"count": 1, "ids": [0]},
        "effective_strategy": {
            "class": "lightning.pytorch.strategies.SingleDeviceStrategy",
            "root_device": "cpu",
            "parallel_devices": [],
        },
        "environment": {
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_available": False,
        },
    }


def test_cpu_auto_runtime_resolution_accepts_real_semantics() -> None:
    evidence = _cpu_auto_resolution_evidence()

    problems = runtime_execution._runtime_resolution_problems(
        requested_accelerator="auto",
        requested_devices="auto",
        requested_strategy="auto",
        **evidence,
    )

    assert problems == []


@pytest.mark.parametrize(
    ("path", "replacement", "problem_fragment"),
    [
        (("effective_devices", "count"), 2, "devices.count"),
        (("effective_devices", "ids"), [], "devices.ids"),
        (("effective_devices", "ids"), [1], "devices.ids"),
        (
            ("effective_strategy", "class"),
            "lightning.pytorch.strategies.DDPStrategy",
            "strategy.class",
        ),
        (("effective_strategy", "root_device"), "cuda:0", "root_device"),
        (
            ("effective_accelerator", "class"),
            "lightning.pytorch.accelerators.CUDAAccelerator",
            "accelerator.class",
        ),
    ],
)
def test_cpu_auto_runtime_resolution_rejects_fabricated_evidence(
    path: tuple[str, str], replacement: Any, problem_fragment: str
) -> None:
    evidence = _cpu_auto_resolution_evidence()
    evidence[path[0]][path[1]] = replacement

    problems = runtime_execution._runtime_resolution_problems(
        requested_accelerator="auto",
        requested_devices="auto",
        requested_strategy="auto",
        **evidence,
    )

    assert any(problem_fragment in problem for problem in problems)


def test_cpu_two_device_auto_strategy_requires_distributed_evidence() -> None:
    problems = runtime_execution._runtime_resolution_problems(
        requested_accelerator="cpu",
        requested_devices=2,
        requested_strategy="auto",
        effective_accelerator={
            "class": "lightning.pytorch.accelerators.CPUAccelerator",
            "device_type": "cpu",
        },
        effective_devices={"count": 2, "ids": [0, 1]},
        effective_strategy={
            "class": "lightning.pytorch.strategies.DDPStrategy",
            "root_device": "cpu",
            "parallel_devices": ["cpu", "cpu"],
        },
        environment={
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_available": False,
        },
    )

    assert problems == []


def _cpu_one_device_ddp_resolution_evidence() -> dict[str, Any]:
    return {
        "effective_accelerator": {
            "class": "lightning.pytorch.accelerators.CPUAccelerator",
            "device_type": "cpu",
        },
        "effective_devices": {"count": 1, "ids": [0]},
        "effective_strategy": {
            "class": "lightning.pytorch.strategies.DDPStrategy",
            "root_device": "cpu",
            "parallel_devices": ["cpu"],
        },
        "environment": {
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_available": False,
        },
    }


def test_cpu_one_device_explicit_ddp_accepts_parallel_device_evidence() -> None:
    problems = runtime_execution._runtime_resolution_problems(
        requested_accelerator="cpu",
        requested_devices=1,
        requested_strategy="ddp",
        **_cpu_one_device_ddp_resolution_evidence(),
    )

    assert problems == []


@pytest.mark.parametrize(
    ("parallel_devices", "problem_fragment"),
    [
        ([], "parallel_devices"),
        (["cuda:0"], "parallel_devices"),
        (["cpu", "cpu"], "parallel_devices"),
    ],
)
def test_cpu_one_device_explicit_ddp_rejects_bad_parallel_device_evidence(
    parallel_devices: list[str], problem_fragment: str
) -> None:
    evidence = _cpu_one_device_ddp_resolution_evidence()
    evidence["effective_strategy"]["parallel_devices"] = parallel_devices

    problems = runtime_execution._runtime_resolution_problems(
        requested_accelerator="cpu",
        requested_devices=1,
        requested_strategy="ddp",
        **evidence,
    )

    assert any(problem_fragment in problem for problem in problems)


@pytest.mark.parametrize(
    ("enable_progress_bar", "mutated_callbacks"),
    [
        (True, []),
        (False, [{"class": "Stub.TQDMProgressBar"}]),
    ],
)
def test_effective_runtime_rejects_progress_callback_presence_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enable_progress_bar: bool,
    mutated_callbacks: list[dict[str, Any]],
) -> None:
    overrides = _ci_overrides()
    overrides["execution.enable_progress_bar"] = enable_progress_bar
    overrides["execution.enable_checkpointing"] = False
    resolved = resolve_torch_configs(overrides)
    payload = _effective_runtime(resolved.execution_config, 11)
    payload["effective"]["callbacks"] = mutated_callbacks
    harness = _Harness(monkeypatch, tmp_path, resolved, effective_runtime=payload)

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "effective_runtime"


@pytest.mark.parametrize("removed_suffix", ["ModelCheckpoint", "EarlyStopping"])
def test_effective_runtime_rejects_missing_checkpoint_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    removed_suffix: str,
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    payload = _effective_runtime(resolved.execution_config, 11)
    payload["effective"]["callbacks"] = [
        callback
        for callback in payload["effective"]["callbacks"]
        if not callback["class"].endswith(removed_suffix)
    ]
    harness = _Harness(monkeypatch, tmp_path, resolved, effective_runtime=payload)

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "effective_runtime"


def test_effective_runtime_rejects_checkpoint_callbacks_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    overrides = _ci_overrides()
    overrides["execution.enable_checkpointing"] = False
    resolved = resolve_torch_configs(overrides)
    payload = _effective_runtime(resolved.execution_config, 11)
    payload["effective"]["callbacks"] = [
        {"class": "Stub.ModelCheckpoint"},
        {"class": "Stub.EarlyStopping"},
    ]
    harness = _Harness(monkeypatch, tmp_path, resolved, effective_runtime=payload)

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "effective_runtime"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("effective", "accelerator", "trainer_value"), "gpu"),
        (("effective", "devices", "trainer_value"), 2),
        (("effective", "strategy", "trainer_value"), "ddp"),
        (("trainer_kwargs", "deterministic"), False),
        (("effective", "deterministic", "algorithms_enabled"), False),
        (("effective", "precision", "value"), "16-mixed"),
        (("effective", "dataloader", "num_workers"), 1),
        (("effective", "dataloader", "pin_memory"), True),
        (("trainer_kwargs", "enable_progress_bar"), True),
        (("trainer_kwargs", "enable_checkpointing"), False),
        (("effective", "dataloader", "persistent_workers"), False),
        (("effective", "dataloader", "prefetch_factor"), 4),
        (("effective", "callbacks", 0, "save_top_k"), 2),
        (("effective", "callbacks", 0, "monitor"), "wrong_metric"),
        (("effective", "callbacks", 0, "mode"), "max"),
        (("effective", "callbacks", 1, "patience"), 1),
    ],
)
def test_effective_runtime_rejects_each_execution_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    overrides = _ci_overrides()
    overrides.update(
        {
            "execution.num_workers": 2,
            "execution.persistent_workers": True,
            "execution.prefetch_factor": 3,
        }
    )
    resolved = resolve_torch_configs(overrides)
    payload = _effective_runtime(resolved.execution_config, 11)
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    harness = _Harness(monkeypatch, tmp_path, resolved, effective_runtime=payload)

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "effective_runtime"


@pytest.mark.parametrize("member_index", [2, 4])
def test_persisted_config_round_trip_mismatch_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, member_index: int
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    persisted = list(resolved.existing_config)
    if member_index == 2:
        persisted[2] = dataclasses.replace(persisted[2], epochs=99)
    else:
        persisted[4] = dataclasses.replace(persisted[4], objects_per_probe=99)
    harness = _Harness(
        monkeypatch, tmp_path, resolved, persisted_configs=tuple(persisted)
    )

    with pytest.raises(runtime.RuntimeExecutionError) as excinfo:
        _execute(harness, tmp_path)

    assert excinfo.value.stage == "config_round_trip"


def test_matching_config_round_trip_returns_effective_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    result = _execute(harness, tmp_path, seed=11)

    assert result.effective_runtime["seed"] == 11
    assert result.effective_runtime["precision"] == "32-true"


def _history() -> dict[str, Any]:
    return {
        "schema_version": "training_history_v1",
        "source": "lightning_csv_logger",
        "metrics_csv": "unused/metrics.csv",
        "train_loss_name": "poisson_train_loss",
        "val_loss_name": "poisson_val_loss",
        "gradient_clip_val": 3.0,
        "gradient_clip_algorithm": "norm",
        "series": {
            "poisson_train_loss_epoch": {
                "step": [4, 9],
                "epoch": [0, 1],
                "value": [2.0, 1.0],
            },
            "grad_norm_preclip_step": {
                "step": [0, 1, 2, 3],
                "epoch": [0, 0, 1, 1],
                "value": [4.0, 2.0, 1.0, 0.5],
            },
        },
    }


def test_training_history_is_carried_through_canonical_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    history = _history()
    harness = _Harness(monkeypatch, tmp_path, resolved, training_history=history)

    result = _execute(harness, tmp_path)

    assert result.training_history == history


def test_absent_training_history_degrades_to_typed_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    harness = _Harness(monkeypatch, tmp_path, resolved)

    result = _execute(harness, tmp_path)

    assert result.training_history is None
    payload = runtime.training_history_payload(None)
    assert payload["schema_version"] == "ablation_training_history_v1"
    assert payload["available"] is False
    assert payload["reason"]
    assert runtime.training_history_records(None) == ()
    assert runtime.history_report_curves(None) == ((), ())


def test_training_history_publishes_flattened_stability_operands() -> None:
    records = runtime.training_history_records(_history())

    values = {record.path: record.value for record in records}
    assert values == {
        "stability.loss_final": 1.0,
        "stability.loss_all_finite": 1.0,
        "stability.gradient_norm_max": 4.0,
        "stability.gradient_norm_final": 0.5,
        "stability.gradient_norm_all_finite": 1.0,
        "stability.clip_fraction": 0.25,
    }
    assert {record.basis for record in records} == {"training_history"}


def test_finite_history_yields_report_curves() -> None:
    assert runtime.history_report_curves(_history()) == (
        (2.0, 1.0),
        (4.0, 2.0, 1.0, 0.5),
    )


def test_non_finite_history_is_flagged_never_fabricated() -> None:
    history = _history()
    history["series"]["poisson_train_loss_epoch"]["value"] = [2.0, float("nan")]
    history["series"]["grad_norm_preclip_step"]["value"] = [
        4.0,
        float("inf"),
        1.0,
        0.5,
    ]

    values = {
        record.path: record.value
        for record in runtime.training_history_records(history)
    }
    assert values["stability.loss_all_finite"] == 0.0
    assert "stability.loss_final" not in values
    assert values["stability.gradient_norm_all_finite"] == 0.0
    assert values["stability.gradient_norm_max"] == 4.0
    assert values["stability.gradient_norm_final"] == 0.5
    assert values["stability.clip_fraction"] == pytest.approx(1.0 / 3.0)
    assert runtime.history_report_curves(history) == ((), ())
    payload = runtime.training_history_payload(history)
    encoded = json.dumps(payload, allow_nan=False)  # strict-JSON safe
    decoded = json.loads(encoded)
    assert decoded["history"]["series"]["poisson_train_loss_epoch"]["value"] == [
        2.0,
        "NaN",
    ]


def test_stored_history_curves_rebuild_from_artifact(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-1"
    attempt.mkdir()
    payload = runtime.training_history_payload(_history())
    (attempt / "training_history.json").write_text(
        json.dumps(payload, allow_nan=False), encoding="utf-8"
    )

    assert runtime.stored_history_curves(attempt) == (
        (2.0, 1.0),
        (4.0, 2.0, 1.0, 0.5),
    )
    assert runtime.stored_history_curves(tmp_path / "missing") == ((), ())


def test_runtime_and_driver_modules_are_architecture_neutral() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sources = sorted(
        (repo_root / "scripts" / "studies" / "ablation").glob("runtime*.py")
    )
    sources.append(repo_root / "scripts" / "studies" / "torch_ablation_driver.py")
    assert sources, "runtime modules must exist"
    pattern = re.compile(
        r"\b(hybrid_resnet|stable_hybrid|fno|ffno|cnn)\b", re.IGNORECASE
    )
    for source in sources:
        assert not pattern.search(source.read_text(encoding="utf-8")), (
            f"architecture-specific token found in {source.name}"
        )


def test_real_execution_forwards_manifest_strictness_with_validated_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    request = runtime.StudyRequest(
        spec=spec,
        only="architecture=hybrid_resnet,physics_profile=ci_nll",
    )
    loaded = runtime.load_study(request)
    run = loaded.selected[0]
    validated = object()
    captured: dict[str, Any] = {}

    class StopAfterResolution(RuntimeError):
        pass

    def capture_resolution(overrides: dict[str, Any], **kwargs: Any) -> None:
        captured.update(kwargs)
        raise StopAfterResolution

    monkeypatch.setattr(runtime_study, "resolve_torch_configs", capture_resolution)

    with pytest.raises(StopAfterResolution):
        runtime_study._execute_or_reuse_run(
            loaded,
            run,
            validated,  # type: ignore[arg-type]
            tmp_path,
            request,
            object(),  # type: ignore[arg-type]
            "environment",
        )

    assert captured["dataset"] is validated
    assert captured["dataset_id"] == "deadleaves_ci_3p5m"
    assert captured["require_all_explicit"] is True


def test_checked_dry_run_has_twelve_arms_without_dataset_or_gpu_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry-run crossed into dataset or GPU execution")

    monkeypatch.setattr(runtime_study, "_load_validated_datasets", forbidden)
    monkeypatch.setattr(runtime_execution, "execute_canonical_run", forbidden)
    request = runtime.StudyRequest(spec=spec, dry_run=True)
    loaded = runtime.load_study(request)

    rendered = runtime.render_dry_run(loaded, request)
    run_lines = [line for line in rendered.splitlines() if line.startswith("run ")]
    arm_lines = [line for line in rendered.splitlines() if line.startswith("arm ")]

    assert len(run_lines) == len(set(run_lines)) == 36
    assert len(arm_lines) == 12
    assert "dataset_validation=deferred" in rendered


def _canonical_claim_grade_runs(loaded: Any) -> tuple[Any, ...]:
    template = loaded.selected[0]
    valid_arms = (
        ("hybrid_resnet", "ci_nll"),
        ("cnn", "ci_nll"),
        ("hybrid_resnet", "legacy_nll"),
        ("hybrid_resnet", "legacy_mae"),
        ("cnn", "legacy_nll"),
        ("cnn", "legacy_mae"),
    )
    return tuple(
        dataclasses.replace(
            template,
            id=f"{family}-{architecture}-{profile}-seed-{seed}",
            arm_id=f"{family}-{architecture}-{profile}",
            dataset_id=f"{family}-dataset",
            seed=seed,
            dimensions=manifest.FrozenDict(
                {
                    "object_family": family,
                    "architecture": architecture,
                    "physics_profile": profile,
                }
            ),
        )
        for family in ("deadleaves", "lines")
        for architecture, profile in valid_arms
        for seed in (3, 17, 29)
    )


def _canonical_protocol_configs(runs: tuple[Any, ...]) -> dict[str, Any]:
    return {
        run.id: {
            "data": {"N": 64},
            "training": {"epochs": run.overrides["training.epochs"]},
            "dataset_id": run.dataset_id,
        }
        for run in runs
    }


def _canonical_claim_grade_manifest(
    loaded: Any,
    runs: tuple[Any, ...],
    configs: dict[str, Any],
    dataset_profiles: dict[str, str] | None = None,
) -> Any:
    manifest_with_budget = dataclasses.replace(
        loaded.manifest, budget_threshold_contract_locked=True
    )
    return dataclasses.replace(
        manifest_with_budget,
        expected_protocol_sha256=manifest.protocol_fingerprint(
            manifest_with_budget,
            runs,
            configs,
            dataset_profiles=dataset_profiles,
        ),
    )


def _dataset_profiles(runs: tuple[Any, ...], profile: str) -> dict[str, str]:
    return {run.dataset_id: profile for run in runs}


def _exact_bridge_evidence(study_manifest: Any) -> Any:
    from scripts.studies.ablation.verdicts import IntegrationBridgeEvidence

    requirement = study_manifest.integration_bridge_requirement
    assert requirement is not None
    payload = {
            "schema_version": "hybrid_resnet_integration_bridge_evidence_v3",
            "contract": requirement.to_mapping(),
            "checkpoint_sha256": "1" * 64,
            "selected_checkpoint": "artifacts/checkpoints/best.ckpt",
            "train_npz_sha256": "5" * 64,
            "test_npz_sha256": "6" * 64,
            "pre_stitch_patch_sha256": "2" * 64,
            "historical_canvas_sha256": "3" * 64,
            "ground_truth_sha256": "7" * 64,
            "generic_canvas_sha256": "3" * 64,
            "historical_mask_sha256": "4" * 64,
            "generic_mask_sha256": "4" * 64,
            "canvases_equivalent": True,
            "masks_equivalent": True,
            "no_resize_asserted": True,
            "gauge_handling": "declared_none",
            "recorded_differences": [],
            "fixture_amp_mae": 0.08,
            "fixture_phase_mae": 0.12,
            "fixture_amp_ssim": 0.88,
            "fixture_phase_ssim": 0.96,
            "architecture": "hybrid_resnet",
            "generator_output_mode": "real_imag",
            "hybrid_encoder_conv_hidden_scale": 2.0,
            "training_patch_weighting": "central_mask",
            "physics_forward_mode": "amplitude",
            "amplitude_physics_gain": 16.0,
            "torch_loss_mode": "mae",
            "seed": 3,
            "epochs": 5,
        }
    return IntegrationBridgeEvidence.from_sealed_artifact_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def test_claim_grade_eligibility_has_closed_canonical_reason_order() -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))

    eligible, reasons = manifest.claim_grade_eligibility(
        loaded.manifest,
        loaded.study.runs,
        loaded.selected,
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert eligible is False
    assert reasons == (
        "manifest_budget_mismatch",
        "integration_bridge_prerequisite",
    )

    canonical = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(canonical)
    declared = _canonical_claim_grade_manifest(loaded, canonical, configs)
    eligible, reasons = manifest.claim_grade_eligibility(
        declared,
        canonical,
        canonical,
        resolved_run_configs=configs,
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert eligible is True
    assert reasons == ()

    eligible, reasons = manifest.claim_grade_eligibility(
        declared,
        canonical[:-3],
        canonical[:-3],
        resolved_run_configs={run.id: configs[run.id] for run in canonical[:-3]},
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert eligible is False
    assert reasons == ("manifest_budget_mismatch",)

    eligible, reasons = manifest.claim_grade_eligibility(
        declared,
        canonical,
        canonical[:-1],
        resolved_run_configs=configs,
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=True,
        seeds_override=True,
        matrix_filter=True,
        dataset_override=True,
        external_dataset_spec=True,
        dirty_checkout=True,
        extra_reasons=("dirty_checkout", "epochs_override", "fixture_dataset"),
    )
    assert eligible is False
    assert reasons == (
        "epochs_override",
        "seeds_override",
        "matrix_filter",
        "dataset_override",
        "external_dataset_spec",
        "dirty_checkout",
        "manifest_budget_mismatch",
        "fixture_dataset",
    )


def test_fixture_dataset_reason_is_closed_ordered_and_protocol_bound() -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    canonical = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(canonical)
    claim_profiles = _dataset_profiles(canonical, "claim_grade")
    fixture_profiles = dict(claim_profiles)
    fixture_profiles[next(iter(sorted(fixture_profiles)))] = "fixture"

    declared_claim = _canonical_claim_grade_manifest(
        loaded, canonical, configs, claim_profiles
    )
    eligible, reasons = manifest.claim_grade_eligibility(
        declared_claim,
        canonical,
        canonical,
        resolved_run_configs=configs,
        dataset_profiles=claim_profiles,
        integration_bridge_evidence=_exact_bridge_evidence(declared_claim),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert (eligible, reasons) == (True, ())

    declared_fixture = _canonical_claim_grade_manifest(
        loaded, canonical, configs, fixture_profiles
    )
    eligible, reasons = manifest.claim_grade_eligibility(
        declared_fixture,
        canonical,
        canonical,
        resolved_run_configs=configs,
        dataset_profiles=fixture_profiles,
        integration_bridge_evidence=_exact_bridge_evidence(declared_fixture),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert (eligible, reasons) == (False, ("fixture_dataset",))

    eligible, reasons = manifest.claim_grade_eligibility(
        declared_claim,
        canonical,
        canonical,
        resolved_run_configs=configs,
        dataset_profiles=fixture_profiles,
        integration_bridge_evidence=_exact_bridge_evidence(declared_claim),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert eligible is False
    assert reasons == ("manifest_budget_mismatch", "fixture_dataset")
    assert manifest.protocol_fingerprint(
        declared_claim, canonical, configs, dataset_profiles=claim_profiles
    ) != manifest.protocol_fingerprint(
        declared_claim, canonical, configs, dataset_profiles=fixture_profiles
    )


def test_runtime_records_selected_validated_dataset_profiles(tmp_path: Path) -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    request = runtime.StudyRequest(spec=spec)
    loaded = runtime.load_study(request)
    selected_ids = {run.dataset_id for run in loaded.selected}
    validated = {
        dataset_id: SimpleNamespace(
            bundle=SimpleNamespace(materialization_profile="fixture")
        )
        for dataset_id in selected_ids
    }

    profiles = runtime_study._selected_dataset_materialization_profiles(
        loaded.selected, validated
    )
    expansion, invocation = runtime_study._study_records(
        loaded, request, tmp_path, profiles
    )

    assert profiles == {dataset_id: "fixture" for dataset_id in sorted(selected_ids)}
    assert expansion["dataset_materialization_profiles"] == profiles
    assert invocation["dataset_materialization_profiles"] == profiles


def test_honest_fixture_bundle_disqualifies_canonical_runtime(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation.datasets import load_checked_dataset_bundle
    from tests.studies.test_ci_compatibility_materializer import _materialize

    bundle_root = tmp_path / "fixture_bundle"
    bundle = load_checked_dataset_bundle(
        _materialize(bundle_root), repo_root=bundle_root
    )
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    canonical = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(canonical)
    selected_ids = {run.dataset_id for run in canonical}
    validated = {
        dataset_id: bundle[
            "deadleaves_ci_3p5m"
            if dataset_id.startswith("deadleaves")
            else "lines_ci_3p5m"
        ]
        for dataset_id in selected_ids
    }
    profiles = runtime_study._selected_dataset_materialization_profiles(
        canonical, validated
    )
    declared = _canonical_claim_grade_manifest(loaded, canonical, configs, profiles)

    eligible, reasons = manifest.claim_grade_eligibility(
        declared,
        canonical,
        canonical,
        resolved_run_configs=configs,
        dataset_profiles=profiles,
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )

    assert set(profiles.values()) == {"fixture"}
    assert (eligible, reasons) == (False, ("fixture_dataset",))

    from scripts.studies.ablation.reporting import ReportInput, write_report
    from scripts.studies.ablation.verdicts import GateResult, Verdict

    report_root = tmp_path / "report"
    write_report(
        ReportInput(
            "fixture-runtime",
            (),
            (),
            (GateResult.active("numeric", Verdict.PASS),),
            claim_grade_eligible=eligible,
            claim_grade_disqualifying_reasons=reasons,
            actual_protocol_sha256=declared.expected_protocol_sha256,
            expected_protocol_sha256=declared.expected_protocol_sha256,
            invocation={"dataset_materialization_profiles": profiles},
        ),
        report_root,
    )
    invocation = json.loads((report_root / "invocation.json").read_text())
    assert invocation["dataset_materialization_profiles"] == profiles
    assert invocation["claim_grade_disqualifying_reasons"] == ["fixture_dataset"]
    assert "NON_CLAIM_GRADE" in (report_root / "report.md").read_text()


@pytest.mark.parametrize(
    "drift", ("dataset", "config", "study_id", "gate", "comparison")
)
def test_declared_protocol_fingerprint_rejects_resolved_protocol_drift(
    drift: str,
) -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    runs = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(runs)
    declared = _canonical_claim_grade_manifest(loaded, runs, configs)
    actual = declared.expected_protocol_sha256
    assert actual is not None

    eligible, reasons = manifest.claim_grade_eligibility(
        declared,
        runs,
        runs,
        resolved_run_configs=configs,
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert eligible is True
    assert reasons == ()
    assert manifest.claim_grade_protocol_fingerprints(declared, runs, configs) == (
        actual,
        actual,
    )

    drifted_manifest = declared
    drifted_runs = runs
    drifted_configs = configs
    if drift == "config":
        drifted_configs = {key: dict(value) for key, value in configs.items()}
        drifted_configs[runs[0].id] = {
            **drifted_configs[runs[0].id],
            "training": {"epochs": 21},
        }
    elif drift == "dataset":
        changed_dataset = dataclasses.replace(
            runs[0].dataset,
            metadata=manifest.FrozenDict(
                {**dict(runs[0].dataset.metadata), "train_sha256": "f" * 64}
            ),
        )
        drifted_runs = (
            dataclasses.replace(runs[0], dataset=changed_dataset),
            *runs[1:],
        )
    elif drift == "study_id":
        drifted_manifest = dataclasses.replace(declared, study_id="changed-study")
    elif drift == "gate":
        drifted_manifest = dataclasses.replace(
            declared,
            gates=(
                dataclasses.replace(
                    declared.gates[0], threshold=declared.gates[0].threshold + 1
                ),
                *declared.gates[1:],
            ),
        )
    else:
        drifted_manifest = dataclasses.replace(
            declared,
            comparisons=(
                dataclasses.replace(
                    declared.comparisons[0],
                    threshold=declared.comparisons[0].threshold + 0.1,
                ),
                *declared.comparisons[1:],
            ),
        )

    eligible, reasons = manifest.claim_grade_eligibility(
        drifted_manifest,
        drifted_runs,
        drifted_runs,
        resolved_run_configs=drifted_configs,
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    assert eligible is False
    assert reasons == ("manifest_budget_mismatch",)
    actual_drifted, expected = manifest.claim_grade_protocol_fingerprints(
        drifted_manifest, drifted_runs, drifted_configs
    )
    assert actual_drifted != expected == actual


@pytest.mark.parametrize(
    ("rule_collection", "field"),
    tuple(("gates", field.name) for field in dataclasses.fields(manifest.Gate))
    + tuple(
        ("comparisons", field.name) for field in dataclasses.fields(manifest.Comparison)
    ),
)
def test_protocol_fingerprint_binds_every_verdict_contract_field(
    rule_collection: str,
    field: str,
) -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    runs = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(runs)
    declared = _canonical_claim_grade_manifest(loaded, runs, configs)
    rules = getattr(declared, rule_collection)
    rule = rules[0]
    current = getattr(rule, field)
    if isinstance(current, manifest.FrozenDict):
        changed: object = manifest.FrozenDict({"protocol_drift": "changed"})
    elif isinstance(current, tuple):
        changed = (*current, "protocol_drift")
    elif isinstance(current, str):
        changed = f"{current}_drift"
    elif isinstance(current, (int, float)):
        changed = current + 1
    elif field in {"threshold", "min_successful", "requested", "min_pairs"}:
        changed = 1
    else:
        changed = "protocol_drift"
    drifted_rules = (dataclasses.replace(rule, **{field: changed}), *rules[1:])
    drifted = dataclasses.replace(declared, **{rule_collection: drifted_rules})

    assert manifest.protocol_fingerprint(drifted, runs, configs) != (
        declared.expected_protocol_sha256
    )


def test_protocol_fingerprint_binds_integration_bridge_requirement() -> None:
    from scripts.studies.ablation.verdicts import IntegrationBridgeRequirement

    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    runs = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(runs)
    declared = _canonical_claim_grade_manifest(loaded, runs, configs)
    requirement = declared.integration_bridge_requirement
    assert requirement is not None
    changed = requirement.to_mapping()
    changed["epochs"] = 6
    drifted = dataclasses.replace(
        declared,
        integration_bridge_requirement=(
            IntegrationBridgeRequirement.from_mapping(changed)
        ),
    )

    assert manifest.protocol_fingerprint(drifted, runs, configs) != (
        declared.expected_protocol_sha256
    )


def test_manifest_schema_parses_declared_protocol_fingerprint() -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    raw = loaded.manifest.to_dict()
    raw["claim_grade"] = {"expected_protocol_sha256": "a" * 64}

    parsed = manifest._parse_manifest(raw)

    assert parsed.expected_protocol_sha256 == "a" * 64


def test_manifest_clone_preserves_frozen_source_identity() -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))

    cloned = manifest._parse_manifest(loaded.manifest.to_dict())

    assert cloned.source_bytes == loaded.manifest.source_bytes
    assert cloned.source_sha256 == loaded.manifest.source_sha256
    assert cloned.source_stat == loaded.manifest.source_stat


def test_frozen_source_manifest_rejects_mutation_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.toml"
    source.write_text("[schema]\nversion = 1\n", encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    frozen = runtime_study._freeze_source_manifest(source, expected)
    assert frozen.data == b"[schema]\nversion = 1\n"
    assert frozen.sha256 == expected

    source.write_text("[schema]\nversion = 2\n", encoding="utf-8")
    with pytest.raises(runtime.StudyRequestError, match="changed during study"):
        frozen.verify_unchanged()
    with pytest.raises(runtime.StudyRequestError, match="hash"):
        runtime_study._freeze_source_manifest(source, expected)


def test_non_claim_grade_report_cannot_publish_pass(tmp_path: Path) -> None:
    reporting = __import__(
        "scripts.studies.ablation.reporting", fromlist=["ReportInput"]
    )
    from scripts.studies.ablation.verdicts import GateResult, Verdict

    study = reporting.ReportInput(
        "smoke",
        (),
        (),
        (GateResult.active("numeric", Verdict.PASS),),
        claim_grade_eligible=False,
        claim_grade_disqualifying_reasons=("epochs_override",),
    )

    artifacts = reporting.write_report(study, tmp_path)
    report = (tmp_path / "report.md").read_text()
    invocation = json.loads((tmp_path / "invocation.json").read_text())
    completion = json.loads((tmp_path / "report_completion.json").read_text())
    verdicts = json.loads((tmp_path / "verdicts.json").read_text())
    assert artifacts.aggregate_verdict is Verdict.PASS
    assert "NON_CLAIM_GRADE" in report
    assert "Bounded compatibility conclusion: **PASS**" not in report
    assert invocation["claim_grade_eligible"] is False
    assert completion["claim_grade_eligible"] is False
    assert verdicts["aggregate_verdict"] == "pass"
    assert verdicts["published_conclusion"] == "NON_CLAIM_GRADE"


def test_terminal_failure_changes_verdict_not_protocol_eligibility() -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    loaded = runtime.load_study(runtime.StudyRequest(spec=spec))
    canonical = _canonical_claim_grade_runs(loaded)
    configs = _canonical_protocol_configs(canonical)
    declared = _canonical_claim_grade_manifest(loaded, canonical, configs)
    eligible_before = manifest.claim_grade_eligibility(
        declared,
        canonical,
        canonical,
        resolved_run_configs=configs,
        integration_bridge_evidence=_exact_bridge_evidence(declared),
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )
    failed_rows = (
        runtime_study.ReportRow.failed(
            loaded.selected[0].id,
            loaded.selected[0].arm_id,
            loaded.selected[0].dataset_id,
            loaded.selected[0].seed,
            stage="training",
            error="failed",
        ).attempt,
    )
    verdicts = runtime_study._evaluate_rules(loaded, failed_rows, None)

    assert eligible_before == (True, ())
    assert any(
        result.verdict is not None and result.verdict.value != "pass"
        for result in verdicts
    )


def test_successful_attempt_seals_one_row_csv_and_source_copies(tmp_path: Path) -> None:
    from scripts.studies.ablation.runtime_attempts import write_run_artifacts

    class Payload:
        def to_jsonable(self) -> dict[str, Any]:
            return {"value": 1}

    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    run = SimpleNamespace(id="run-1", arm_id="arm-1", dataset_id="dataset", seed=3)
    resolved = SimpleNamespace(canonical_json='{"training":{"epochs":10}}')
    result = SimpleNamespace(
        effective_runtime={},
        best_checkpoint_sha256="a" * 64,
        best_checkpoint=tmp_path / "best.ckpt",
        fixed_batch_identity={},
        run_dir=tmp_path,
        reference_diagnostics=Payload(),
        reloaded_diagnostics=Payload(),
        count_metrics=Payload(),
        reload_max_abs_error=0.0,
        reload_allclose=True,
        reference_texture=np.ones((2, 2)),
        reference_canvas=np.ones((2, 2)),
        reloaded_texture=np.ones((2, 2)),
        reloaded_canvas=np.ones((2, 2)),
        training_history=None,
    )
    from scripts.studies.ablation.metrics import build_metric_record

    metric_records = (
        build_metric_record(
            "runtime.train_seconds", 1.0, basis="wall_clock", alignment="none"
        ),
    )
    training_input = TrainingFingerprintInput(
        schema_version=1,
        manifest_sha256="b" * 64,
        logical_run_id="run-1",
        resolved_configs={"training": {"epochs": 10}},
        seed=3,
        git=GitIdentity(commit="c" * 40, clean=True),
        environment_digest="d" * 64,
        content_sha256s={"dataset.train": "e" * 64},
    )
    required = write_run_artifacts(
        attempt,
        run,
        resolved,
        result,
        metric_records,
        {"reconstruction": np.ones((2, 2))},
        training_input,
        "f" * 64,
        "0" * 64,
        source_manifest=b'[study]\nid = "frozen"\n',
    )

    with (attempt / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert (attempt / "source_manifest.toml").read_bytes().startswith(b"[study]")
    assert json.loads((attempt / "source_config.json").read_text()) == {
        "training": {"epochs": 10}
    }
    assert {"metrics.csv", "source_manifest.toml", "source_config.json"} <= set(
        required
    )

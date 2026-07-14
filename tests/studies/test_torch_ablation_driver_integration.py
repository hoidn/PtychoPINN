"""CLI and tiny two-arm integration tests for the Torch ablation driver.

Execution tests are CPU-only: the canonical Torch entry points are replaced by
deterministic stubs (a real tiny ``nn.Linear`` checkpoint keeps ``torch.load``
plus ``load_state_dict(strict=True)`` real), while dataset preflight,
configuration resolution, fingerprints, artifacts, metrics, verdicts, and
reporting all run for real against tiny 64x64 fixtures.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.studies.torch_ablation_driver as driver
from scripts.studies.ablation import reporting, runtime_study
from scripts.studies.ablation.artifacts import GitIdentity
from tests.studies.ablation_dataset_fixtures import (
    _bundle,
    _dose as fixture_dose,
    _file_sha256 as fixture_file_sha256,
    _refresh_provenance,
)

DETECTOR = 64
TRUTH_SHAPE = (12, 12)
STUDY_ID = "tinystudy"
ARM_A = f"{STUDY_ID}--ci_ds--a"
RUN_A = f"{ARM_A}--seed-7"
RUN_B = f"{STUDY_ID}--ci_ds--b--seed-7"
CONVERGENCE_SPEC = Path("scripts/studies/specs/grid_lines_ci_convergence.toml")


def test_trajectory_runtime_reuses_canonical_milestone_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    milestones = tuple(
        SimpleNamespace(
            milestone_epoch=epoch,
            checkpoint_sha256=f"{epoch:064x}",
        )
        for epoch in (5, 20, 40, 80)
    )
    arrays_by_epoch = {
        epoch: {"reconstruction": np.full((2, 3), epoch)}
        for epoch in (5, 20, 40, 80)
    }
    calls: list[int] = []

    def build_records(resolved: object, result: object, descriptor: object):
        assert resolved is resolved_config
        assert descriptor is dataset_descriptor
        calls.append(result.milestone_epoch)
        return (
            (SimpleNamespace(path="stability.amp_variance", value=1.0),),
            arrays_by_epoch[result.milestone_epoch],
        )

    monkeypatch.setattr(
        runtime_study, "build_milestone_metric_records", build_records
    )
    resolved_config = object()
    dataset_descriptor = object()

    evidence = runtime_study._build_milestone_evidence(
        resolved_config,
        SimpleNamespace(milestones=milestones),
        dataset_descriptor,
    )

    assert calls == [5, 20, 40, 80]
    assert [item.epoch for item in evidence] == [5, 20, 40, 80]
    assert [
        item.arrays is arrays_by_epoch[item.epoch]
        for item in evidence
    ] == [True, True, True, True]


def test_grid_lines_ci_convergence_preflight_uses_derived_legacy_gain() -> None:
    loaded = runtime_study.load_study(
        runtime_study.StudyRequest(
            spec=Path("scripts/studies/specs/grid_lines_ci_convergence.toml"),
            dry_run=True,
        )
    )

    ci_scaling, resolved = runtime_study._preflight_selected_configs(loaded)

    assert len(resolved) == 6
    assert {
        config["model"]["amplitude_physics_gain"] for config in resolved.values()
    } == {1.0, 12.452229360013307}
    for run_id, config in resolved.items():
        expected = 1.0 if "--ci_nll--" in run_id else 12.452229360013307
        assert config["model"]["amplitude_physics_gain"] == expected
    assert sum(ci_scaling.values()) == 2


def test_grid_lines_ci_convergence_preflight_supplies_complete_provenance_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = runtime_study.load_study(
        runtime_study.StudyRequest(spec=CONVERGENCE_SPEC, dry_run=True)
    )
    expected_ids = {
        "deadleaves_ci_3p5m",
        "deadleaves_legacy_amp",
        "lines_ci_3p5m",
        "lines_legacy_amp",
    }

    def load_complete_bundle(
        descriptors: dict[str, dict[str, Any]], *, repo_root: Path
    ):
        del repo_root
        assert set(descriptors) == expected_ids
        return {dataset_id: object() for dataset_id in descriptors}

    monkeypatch.setattr(
        runtime_study, "load_checked_dataset_bundle", load_complete_bundle
    )

    validated = runtime_study._load_validated_datasets(loaded)

    assert set(validated) == {"lines_ci_3p5m", "lines_legacy_amp"}


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [
        (None, (5, 20, 40, 80)),
        (1, ()),
        (20, (5, 20)),
        (80, (5, 20, 40, 80)),
    ],
)
def test_convergence_request_resolves_effective_milestones(
    epochs: int | None, expected: tuple[int, ...]
) -> None:
    loaded = runtime_study.load_study(
        runtime_study.StudyRequest(
            spec=CONVERGENCE_SPEC,
            dry_run=True,
            epochs=epochs,
        )
    )

    assert loaded.effective_milestones == expected


def test_diagnostics_with_omitted_epochs_rejects_default_budget_overrun(
    workspace: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = workspace["spec"].read_text(encoding="utf-8")
    text = text.replace(
        "[base.overrides]",
        "[diagnostics]\nmilestones = [51]\n\n[base.overrides]",
        1,
    ).replace('"training.epochs" = 1\n', "", 1)
    spec = tmp_path / "default-epoch-budget.toml"
    spec.write_text(text, encoding="utf-8")

    code = driver.main(["--spec", str(spec), "--dry-run"])

    error = capsys.readouterr().err
    assert code == 2
    assert "requires explicit training.epochs" in error
    assert "diagnostics milestone 51" in error


# --------------------------------------------------------------------------
# Workspace fixtures: datasets, checked manifest, standalone descriptor.
# --------------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def _truth_array() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return (rng.normal(size=TRUTH_SHAPE) + 1j * rng.normal(size=TRUTH_SHAPE)).astype(
        np.complex64
    )


def _study_dataset(
    base: Path,
    dataset_id: str,
    *,
    kind: str = "synthetic",
    truth: str = "object_truth",
    rebase_paths: bool = True,
) -> dict[str, Any]:
    """Build one preflight-valid 64x64 count-intensity dataset fixture."""
    sub = base / dataset_id
    probe = np.ones((DETECTOR, DETECTOR), dtype=np.complex64)
    descriptor = _bundle(
        sub, dataset_id=dataset_id, kind=kind, truth=truth, probe_array=probe
    )
    descriptor["detector_shape"] = [DETECTOR, DETECTOR]
    measurement = np.full((25, DETECTOR, DETECTOR), 250_000, dtype=np.uint32)
    truth_array = _truth_array()
    for split in ("train", "test"):
        path = sub / descriptor[split]
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload[descriptor["measurement_key"]] = measurement.copy()
        payload["objectGuess"] = truth_array.copy()
        np.savez(path, **payload)
        descriptor[f"{split}_sha256"] = fixture_file_sha256(path)
    if truth == "reference_reconstruction":
        reference = sub / descriptor["reference"]
        np.savez(reference, object=truth_array.copy())
        descriptor["reference_sha256"] = fixture_file_sha256(reference)
    descriptor["dose"] = {
        "train": fixture_dose(measurement),
        "test": fixture_dose(measurement),
    }
    _refresh_provenance(sub, descriptor)
    descriptor.pop("_id")
    if rebase_paths:
        for key in ("train", "test", "provenance", "reference"):
            if key in descriptor:
                descriptor[key] = f"{dataset_id}/{descriptor[key]}"
    return descriptor


_DESCRIPTOR_SCALAR_KEYS = (
    "kind",
    "format",
    "scale_contract_version",
    "measurement_domain",
    "truth",
    "truth_location",
    "truth_key",
    "measurement_key",
    "probe_key",
    "x_key",
    "y_key",
    "coords_convention",
    "detector_shape",
    "grouping_max_C",
    "probe_modes",
    "train",
    "test",
    "reference",
    "provenance",
    "train_sha256",
    "test_sha256",
    "reference_sha256",
    "provenance_sha256",
)


def _descriptor_lines(prefix: str, descriptor: dict[str, Any]) -> list[str]:
    lines = [f"[{prefix}]"]
    for key in _DESCRIPTOR_SCALAR_KEYS:
        if key in descriptor:
            lines.append(f"{key} = {_toml_value(descriptor[key])}")
    lines.append("")
    lines.append(f"[{prefix}.probe]")
    for key, value in descriptor["probe"].items():
        lines.append(f"{key} = {_toml_value(value)}")
    for split in ("train", "test"):
        lines.append("")
        lines.append(f"[{prefix}.dose.{split}]")
        for key, value in descriptor["dose"][split].items():
            lines.append(f"{key} = {_toml_value(value)}")
    return lines


_BASE_OVERRIDES: dict[str, Any] = {
    "dataset.id": "ci_ds",
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


def _manifest_text(datasets: dict[str, dict[str, Any]]) -> str:
    lines = [
        "[schema]",
        "version = 1",
        "",
        "[study]",
        f'id = "{STUDY_ID}"',
        "seeds = [7]",
        "",
        "[base.overrides]",
    ]
    lines += [
        f"{json.dumps(key)} = {_toml_value(value)}"
        for key, value in _BASE_OVERRIDES.items()
    ]
    lines += [
        "",
        "[[matrix.dimensions]]",
        'name = "variant"',
        "",
        "[[matrix.dimensions.values]]",
        'id = "a"',
        "[matrix.dimensions.values.overrides]",
        '"training.batch_size" = 2',
        "",
        "[[matrix.dimensions.values]]",
        'id = "b"',
        "[matrix.dimensions.values.overrides]",
        '"training.batch_size" = 4',
    ]
    for dataset_id, descriptor in datasets.items():
        lines.append("")
        lines += _descriptor_lines(f"datasets.{dataset_id}", descriptor)
    lines += [
        "",
        "[[gates]]",
        'id = "seed_success"',
        'target = { variant = "a" }',
        'operator = "status_count_ge"',
        'status = "success"',
        "requested = 1",
        "threshold = 1",
        "",
        "[[gates]]",
        'id = "reload_stability"',
        'target = { variant = "a" }',
        'operator = "le"',
        'metric = "stability.reload_max_abs_error"',
        'aggregation = "median"',
        "threshold = 1e-05",
        "min_successful = 1",
        "",
        "[[gates]]",
        'id = "truth_recognizability"',
        'target = { variant = "a" }',
        'operator = "ge"',
        'metric = "truth_quality.amp_pearson"',
        'aggregation = "median"',
        "threshold = -1.0",
        "min_successful = 1",
        'requires = ["has_object_truth"]',
        'when_dataset_kind = "synthetic"',
        'on_missing_capability = "not_applicable"',
        "",
        "[[gates]]",
        'id = "truth_amp_ssim"',
        'target = { variant = "a" }',
        'operator = "ge"',
        'metric = "truth_quality.amp_ssim"',
        'aggregation = "median"',
        "threshold = -1.0",
        "min_successful = 1",
        'requires = ["has_object_truth"]',
        'when_dataset_kind = "synthetic"',
        'on_missing_capability = "not_applicable"',
        "",
        "[[gates]]",
        'id = "reference_agreement_check"',
        'target = { variant = "a" }',
        'operator = "ge"',
        'metric = "reference_agreement.amp_pearson"',
        'aggregation = "median"',
        "threshold = -1.0",
        "min_successful = 1",
        'requires = ["has_reference"]',
        'when_dataset_kind = "experimental"',
        'on_missing_capability = "not_applicable"',
        "",
        "[[gates]]",
        'id = "visual"',
        'target = { variant = "a" }',
        'operator = "manual_review"',
    ]
    return "\n".join(lines) + "\n"


def _standalone_spec_text(descriptor: dict[str, Any]) -> str:
    lines = ["[schema]", "version = 1", ""]
    body = _descriptor_lines("dataset", descriptor)
    body.insert(1, 'id = "spec_ds"')
    return "\n".join(lines + body) + "\n"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("ablation-driver-ws")
    datasets = {
        "ci_ds": _study_dataset(root, "ci_ds"),
        "exp_ref": _study_dataset(
            root, "exp_ref", kind="experimental", truth="reference_reconstruction"
        ),
        "exp_blind": _study_dataset(
            root, "exp_blind", kind="experimental", truth="none"
        ),
    }
    spec = root / "study.toml"
    spec.write_text(_manifest_text(datasets), encoding="utf-8")
    standalone_descriptor = _study_dataset(
        root / "standalone",
        "spec_ds",
        kind="experimental",
        truth="reference_reconstruction",
        rebase_paths=False,
    )
    dataset_spec = root / "standalone" / "spec_ds" / "dataset.toml"
    dataset_spec.write_text(
        _standalone_spec_text(standalone_descriptor), encoding="utf-8"
    )
    return {"root": root, "spec": spec, "dataset_spec": dataset_spec}


# --------------------------------------------------------------------------
# Execution stubs.
# --------------------------------------------------------------------------


def _effective_runtime(execution: Any, seed: int) -> dict[str, Any]:
    device_type = (
        "cuda" if execution.accelerator in {"cuda", "gpu"} else execution.accelerator
    )
    device_count = execution.devices if isinstance(execution.devices, int) else 1
    effective_precision = (
        "bf16-mixed"
        if device_type == "cpu" and execution.precision == "16-mixed"
        else execution.precision
    )
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
            "precision": {"value": effective_precision, "plugin": "Stub"},
            "deterministic": {
                "algorithms_enabled": execution.deterministic in {True, "warn"},
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
                "trainer_value": execution.accelerator,
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
            "accelerator": execution.accelerator,
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


class _IntegrationDiagnostics:
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
            "canvas_origin_offset": (width // 2 - scan[0], height // 2 - scan[1]),
        }
        self.canvas_weights = np.ones(canvas_shape, dtype=np.float32)

    def to_jsonable(self) -> dict[str, Any]:
        return {"schema_version": 1, "s1": self.s1, "s2": self.s2}


class _ExecutionStubs:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail_training: bool = False):
        import torch
        from torch import nn
        from ptycho_torch import dataloader, lightning_utils, reassembly
        from ptycho_torch import train_lightning_only
        from ptycho_torch.reassembly_diagnostics import FittedCountMetrics

        self.train_calls: list[tuple[str, int]] = []
        self.milestone_calls: list[tuple[int, ...]] = []
        self.configs_by_checkpoint: dict[str, tuple[Any, ...]] = {}
        rng = np.random.default_rng(42)
        self.canvas = (
            rng.normal(size=TRUTH_SHAPE) + 1j * rng.normal(size=TRUTH_SHAPE)
        ).astype(np.complex128)
        counter = itertools.count(1)
        stubs = self

        def train_main(ptycho_dir: str, *args: Any, **kwargs: Any) -> Any:
            stubs.train_calls.append((ptycho_dir, kwargs.get("seed")))
            stubs.milestone_calls.append(kwargs.get("milestone_epochs", ()))
            if fail_training:
                raise RuntimeError("stub training failure")
            run_dir = Path(kwargs["output_dir"]) / f"stub_run_{next(counter)}"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            torch.manual_seed(0)
            model = nn.Linear(2, 2)
            model.val_loss_name = "val_loss"
            best = checkpoint_dir / "best-checkpoint.ckpt"
            torch.save({"state_dict": model.state_dict()}, best)
            stubs.configs_by_checkpoint[str(best)] = kwargs["existing_config"]
            milestone_checkpoints: dict[int, Path] = {}
            for epoch in kwargs.get("milestone_epochs", ()):
                checkpoint = checkpoint_dir / f"milestone-epoch-{epoch:04d}.ckpt"
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch - 1},
                    checkpoint,
                )
                stubs.configs_by_checkpoint[str(checkpoint)] = kwargs[
                    "existing_config"
                ]
                milestone_checkpoints[epoch] = checkpoint
            return SimpleNamespace(
                run_dir=run_dir,
                model=model,
                effective_runtime=_effective_runtime(
                    kwargs["execution_config"], kwargs["seed"]
                ),
                training_history={
                    "schema_version": "training_history_v1",
                    "source": "lightning_csv_logger",
                    "metrics_csv": str(run_dir / "metrics.csv"),
                    "train_loss_name": "poisson_train_loss",
                    "val_loss_name": "poisson_val_loss",
                    "gradient_clip_val": None,
                    "gradient_clip_algorithm": "norm",
                    "series": {
                        "poisson_train_loss_epoch": {
                            "step": [4],
                            "epoch": [0],
                            "value": [0.5],
                        },
                        "grad_norm_preclip_step": {
                            "step": [0, 1],
                            "epoch": [0, 0],
                            "value": [2.0, 1.0],
                        },
                    },
                },
                milestone_checkpoints=milestone_checkpoints,
            )

        class StubPtychoDataset:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        def reconstruct(
            model: Any,
            ptycho_dset: Any,
            training_config: Any,
            data_config: Any,
            model_config: Any,
            inference_config: Any,
            **kwargs: Any,
        ) -> tuple[Any, Any, Any, Any]:
            canvas = stubs.canvas.copy()
            return (
                canvas,
                SimpleNamespace(name="stub-subset"),
                _IntegrationDiagnostics(canvas.shape, kwargs["precision"]),
                canvas.copy(),
            )

        def load_checkpoint_with_configs(
            checkpoint_path: str, model_class: Any, device: str = "cuda", **kwargs: Any
        ) -> tuple[Any, tuple[Any, ...]]:
            torch.manual_seed(0)
            model = nn.Linear(2, 2)
            model.val_loss_name = "val_loss"
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(payload["state_dict"], strict=True)
            return model, stubs.configs_by_checkpoint[checkpoint_path]

        def evaluate_fitted_count_metrics(*args: Any, **kwargs: Any) -> Any:
            return FittedCountMetrics(
                relative_l2_intensity_error=0.05,
                mean_raw_poisson_nll=1.5,
                n_samples=25,
                n_pixels=25 * DETECTOR * DETECTOR,
                effective_mask_digest="stub-mask-digest",
                sample_ids=tuple(range(25)),
            )

        monkeypatch.setattr(train_lightning_only, "main", train_main)
        monkeypatch.setattr(dataloader, "PtychoDataset", StubPtychoDataset)
        monkeypatch.setattr(reassembly, "reconstruct_image_barycentric", reconstruct)
        monkeypatch.setattr(
            lightning_utils,
            "load_checkpoint_with_configs",
            load_checkpoint_with_configs,
        )
        monkeypatch.setattr(
            reassembly,
            "evaluate_fitted_count_metrics",
            evaluate_fitted_count_metrics,
        )
        monkeypatch.setattr(
            runtime_study,
            "_git_identity",
            lambda: GitIdentity(commit="c" * 40, clean=True),
        )
        monkeypatch.setattr(runtime_study, "_environment_digest", lambda: "e" * 64)


def _forbid(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("forbidden call during --dry-run")


@pytest.fixture()
def no_data_or_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from ptycho_torch import train_lightning_only

    monkeypatch.setattr(np, "load", _forbid)
    monkeypatch.setattr(torch.cuda, "init", _forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbid)
    monkeypatch.setattr(train_lightning_only, "main", _forbid)


def _metric_paths(output_root: Path) -> dict[str, set[str]]:
    payload = json.loads(
        (output_root / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    paths: dict[str, set[str]] = {}
    for row in payload["rows"]:
        paths.setdefault(row["run_id"], set()).add(row["metric_path"])
    return paths


def _verdict_rows(output_root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((output_root / "verdicts.json").read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["rows"]}


# --------------------------------------------------------------------------
# CLI dry-run and argument tests.
# --------------------------------------------------------------------------


def test_dry_run_expands_matrix_without_loading_npz_or_cuda(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(["--spec", str(workspace["spec"]), "--dry-run"])

    output = capsys.readouterr().out
    assert code == 0
    assert "runs 2" in output
    assert RUN_A in output
    assert RUN_B in output
    assert f"gate seed_success target={ARM_A}" in output
    assert "applicability=active" in output
    assert "config_validation resolved_ok arms=2 strict=false" in output
    assert "milestones declared=" not in output


def test_convergence_dry_run_discloses_effective_milestones(
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        ["--spec", str(CONVERGENCE_SPEC), "--dry-run", "--epochs", "20"]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "milestones declared=[5, 20, 40, 80] effective=[5, 20]" in output


def test_checked_strict_dry_run_validates_explicit_paths_without_npz_or_cuda(
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")

    code = driver.main(["--spec", str(spec), "--dry-run"])

    output = capsys.readouterr().out
    assert code == 0
    expected_arm_ids = {
        f"hybrid-resnet-ci-compatibility--{family}_{dataset_suffix}--{family}--{architecture}--{physics_profile}"
        for family in ("deadleaves", "lines")
        for dataset_suffix, architecture, physics_profile in (
            ("ci_3p5m", "hybrid_resnet", "ci_nll"),
            ("ci_3p5m", "cnn", "ci_nll"),
            ("legacy_amp", "hybrid_resnet", "legacy_nll"),
            ("legacy_amp", "hybrid_resnet", "legacy_mae"),
            ("legacy_amp", "cnn", "legacy_nll"),
            ("legacy_amp", "cnn", "legacy_mae"),
        )
    }
    lines = output.splitlines()
    arm_ids = {
        line.removeprefix("arm ").split(" overrides ", maxsplit=1)[0]
        for line in lines
        if line.startswith("arm ")
    }
    run_ids = {
        line.removeprefix("run ").split(" dataset=", maxsplit=1)[0]
        for line in lines
        if line.startswith("run ")
    }

    assert lines.count("runs 36") == 1
    assert arm_ids == expected_arm_ids
    assert run_ids == {
        f"{arm_id}--seed-{seed}" for arm_id in expected_arm_ids for seed in (3, 17, 29)
    }
    assert (
        "config_validation strict_explicit_ok arms=12 dataset_validation=deferred"
        in lines
    )


def test_dry_run_accepts_clean_checkout_external_dataset_bundle_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import torch
    from ptycho_torch import train_lightning_only

    clean = tmp_path / "clean-checkout"
    external_root = tmp_path / "source-artifacts" / "datasets_v2"
    descriptor = _study_dataset(external_root, "ci_ds", rebase_paths=False)
    external_bundle = external_root / "ci_ds"
    lexical_root = Path(".artifacts/ci_compatibility/datasets_v2")
    for field in ("train", "test", "provenance"):
        descriptor[field] = (lexical_root / descriptor[field]).as_posix()
    spec = clean / "study.toml"
    spec.parent.mkdir(parents=True)
    spec.write_text(_manifest_text({"ci_ds": descriptor}), encoding="utf-8")
    bundle_link = clean / lexical_root
    bundle_link.parent.mkdir(parents=True)
    bundle_link.symlink_to(external_bundle, target_is_directory=True)
    monkeypatch.chdir(clean)
    monkeypatch.setattr(np, "load", _forbid)
    monkeypatch.setattr(torch.cuda, "init", _forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbid)
    monkeypatch.setattr(train_lightning_only, "main", _forbid)

    code = driver.main(["--spec", str(spec), "--dry-run"])

    captured = capsys.readouterr()
    output = captured.out
    assert code == 0, captured.err
    assert output.splitlines().count("runs 2") == 1
    assert output.count("\nrun ") == 2
    assert "config_validation resolved_ok arms=2 strict=false" in output


def test_dry_run_rejects_mps_before_data_cuda_or_training(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_text = (
        workspace["spec"]
        .read_text(encoding="utf-8")
        .replace(
            '"execution.accelerator" = "cpu"',
            '"execution.accelerator" = "mps"',
        )
    )
    spec = tmp_path / "mps-study.toml"
    spec.write_text(spec_text, encoding="utf-8")

    code = driver.main(["--spec", str(spec), "--dry-run"])

    assert code == 2
    error = capsys.readouterr().err
    assert "execution.accelerator='mps'" in error
    assert "float64" in error
    assert "reassembly" in error
    assert "count" in error


def test_execution_rejects_selected_arm_mps_before_data_or_output_mutation(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_text = (
        workspace["spec"]
        .read_text(encoding="utf-8")
        .replace(
            '"training.batch_size" = 4',
            '"training.batch_size" = 4\n"execution.accelerator" = "mps"',
        )
    )
    spec = tmp_path / "selected-arm-mps-study.toml"
    spec.write_text(spec_text, encoding="utf-8")
    output_root = tmp_path / "must-not-exist"
    monkeypatch.setattr(runtime_study, "_load_validated_datasets", _forbid)

    code = driver.main(
        [
            "--spec",
            str(spec),
            "--only",
            "variant=b",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "execution.accelerator='mps'" in error
    assert "float64" in error
    assert not output_root.exists()


@pytest.mark.parametrize("devices", [2, "auto"])
def test_dry_run_rejects_base_multi_device_before_data_cuda_or_training(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    devices: int | str,
) -> None:
    spec_text = (
        workspace["spec"]
        .read_text(encoding="utf-8")
        .replace(
            '"execution.devices" = 1',
            f'"execution.devices" = {_toml_value(devices)}',
        )
    )
    spec = tmp_path / f"multi-device-{devices}-study.toml"
    spec.write_text(spec_text, encoding="utf-8")

    code = driver.main(["--spec", str(spec), "--dry-run", "--epochs", "2"])

    assert code == 2
    error = capsys.readouterr().err
    assert "canonical ablation" in error
    assert "held-out mmap/reassembly" in error
    assert "framework peak-memory evidence" in error


def test_execution_rejects_selected_arm_multi_device_before_mmap_or_output(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_text = (
        workspace["spec"]
        .read_text(encoding="utf-8")
        .replace(
            '"training.batch_size" = 4',
            '"training.batch_size" = 4\n"execution.devices" = 2',
        )
    )
    spec = tmp_path / "selected-arm-multi-device-study.toml"
    spec.write_text(spec_text, encoding="utf-8")
    output_root = tmp_path / "must-not-exist-multi-device"
    monkeypatch.setattr(runtime_study, "_load_validated_datasets", _forbid)

    code = driver.main(
        [
            "--spec",
            str(spec),
            "--only",
            "variant=b",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "canonical ablation" in error
    assert "held-out mmap/reassembly" in error
    assert not output_root.exists()


def test_dry_run_applies_seed_epoch_and_output_overrides(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--dry-run",
            "--seeds",
            "5,9",
            "--epochs",
            "3",
            "--output-root",
            str(tmp_path / "plan-out"),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "runs 4" in output
    assert f"{ARM_A}--seed-5" in output
    assert f"{ARM_A}--seed-9" in output
    assert '"training.epochs": 3' in output
    assert str(tmp_path / "plan-out") in output


def test_dry_run_only_selector_filters_runs(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        ["--spec", str(workspace["spec"]), "--dry-run", "--only", "variant=a"]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "runs 1" in output
    assert RUN_A in output
    assert RUN_B not in output


def test_dry_run_unknown_selector_fails(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        ["--spec", str(workspace["spec"]), "--dry-run", "--only", "variant=zzz"]
    )

    assert code == 2
    assert "variant" in capsys.readouterr().err


def test_dry_run_dataset_selection_retargets_gates(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        ["--spec", str(workspace["spec"]), "--dry-run", "--dataset", "exp_ref"]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert f"{STUDY_ID}--exp_ref--a--seed-7" in output
    assert f"gate truth_recognizability target={STUDY_ID}--exp_ref--a" in output
    assert "applicability=not_applicable" in output


def test_dry_run_unknown_dataset_fails(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        ["--spec", str(workspace["spec"]), "--dry-run", "--dataset", "nope"]
    )

    assert code == 2
    assert "nope" in capsys.readouterr().err


def test_dry_run_dataset_spec_is_schema_only(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--dry-run",
            "--dataset-spec",
            str(workspace["dataset_spec"]),
            "--dataset",
            "spec_ds",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert f"{STUDY_ID}--spec_ds--a--seed-7" in output


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unknown_dataset_field", "unknown field"),
        ("missing_train_sha256", "train_sha256"),
        ("missing_provenance", "provenance"),
        ("malformed_dose", "dose.test"),
        ("malformed_profile_pair", "supported pair"),
    ],
)
def test_checked_dry_run_rejects_complete_dataset_schema_errors_without_side_effects(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected: str,
) -> None:
    text = workspace["spec"].read_text(encoding="utf-8")
    if case == "unknown_dataset_field":
        text = text.replace(
            "[datasets.ci_ds]\n", "[datasets.ci_ds]\nunknown_field = true\n", 1
        )
    elif case == "missing_train_sha256":
        text = re.sub(
            r'^train_sha256 = "[0-9a-f]{64}"\n', "", text, count=1, flags=re.M
        )
    elif case == "missing_provenance":
        text = re.sub(r'^provenance = ".*"\n', "", text, count=1, flags=re.M)
    elif case == "malformed_dose":
        text = re.sub(
            r"(\[datasets\.ci_ds\.dose\.test\]\n)counts_mean = [^\n]+\n",
            r"\1",
            text,
            count=1,
        )
    else:
        text = text.replace(
            'scale_contract_version = "ci_intensity_v2"',
            'scale_contract_version = "legacy_v1"',
            1,
        )
    spec = tmp_path / f"invalid-checked-{case}.toml"
    spec.write_text(text, encoding="utf-8")
    output_root = tmp_path / f"output-{case}"

    code = driver.main(
        ["--spec", str(spec), "--dry-run", "--output-root", str(output_root)]
    )

    error = capsys.readouterr().err
    assert code == 2
    assert "datasets.ci_ds" in error
    assert expected in error
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unknown_top_level", "unknown table"),
        ("unknown_dataset_field", "unknown field"),
        ("missing_train_sha256", "train_sha256"),
        ("missing_provenance", "provenance"),
        ("malformed_dose", "dose.test"),
        ("malformed_probe_pair", "calibration/gauge pair"),
    ],
)
def test_standalone_dry_run_rejects_complete_schema_errors_without_side_effects(
    workspace: dict[str, Path],
    no_data_or_cuda: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected: str,
) -> None:
    text = workspace["dataset_spec"].read_text(encoding="utf-8")
    if case == "unknown_top_level":
        text += "\n[unexpected]\nvalue = true\n"
    elif case == "unknown_dataset_field":
        text = text.replace(
            '[dataset]\nid = "spec_ds"\n',
            '[dataset]\nid = "spec_ds"\nunknown_field = true\n',
            1,
        )
    elif case == "missing_train_sha256":
        text = re.sub(
            r'^train_sha256 = "[0-9a-f]{64}"\n', "", text, count=1, flags=re.M
        )
    elif case == "missing_provenance":
        text = re.sub(r'^provenance = ".*"\n', "", text, count=1, flags=re.M)
    elif case == "malformed_dose":
        text = re.sub(
            r"(\[dataset\.dose\.test\]\n)counts_mean = [^\n]+\n",
            r"\1",
            text,
            count=1,
        )
    else:
        text = text.replace(
            'calibration = "count_amplitude"',
            'calibration = "legacy_normalized"',
            1,
        )
    dataset_spec = tmp_path / f"invalid-standalone-{case}.toml"
    dataset_spec.write_text(text, encoding="utf-8")
    output_root = tmp_path / f"output-{case}"

    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--dry-run",
            "--dataset-spec",
            str(dataset_spec),
            "--dataset",
            "spec_ds",
            "--output-root",
            str(output_root),
        ]
    )

    error = capsys.readouterr().err
    assert code == 2
    assert "--dataset-spec" in error
    assert expected in error
    assert not output_root.exists()


def test_resume_and_rerun_are_mutually_exclusive(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = driver.main(["--spec", str(workspace["spec"]), "--resume", "--rerun"])

    assert code == 2


def test_execution_requires_an_output_root(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = driver.main(["--spec", str(workspace["spec"])])

    assert code == 2
    assert "output" in capsys.readouterr().err.lower()


def test_execution_passes_effective_milestones_to_training(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    text = workspace["spec"].read_text(encoding="utf-8")
    text = text.replace(
        "[base.overrides]",
        "[diagnostics]\nmilestones = [5, 20, 40, 80]\n\n[base.overrides]",
        1,
    ).replace('"training.epochs" = 1', '"training.epochs" = 80', 1)
    spec = tmp_path / "milestone-study.toml"
    spec.write_text(text, encoding="utf-8")
    stubs = _ExecutionStubs(monkeypatch)

    code = driver.main(
        [
            "--spec",
            str(spec),
            "--epochs",
            "1",
            "--output-root",
            str(tmp_path / "milestone-output"),
        ]
    )

    assert code == 0
    assert stubs.milestone_calls == [(), ()]


def test_nonempty_milestone_schedule_persists_execution_and_artifacts(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    text = workspace["spec"].read_text(encoding="utf-8").replace(
        "[base.overrides]",
        "[diagnostics]\nmilestones = [1]\n\n[base.overrides]",
        1,
    )
    spec = tmp_path / "one-milestone-study.toml"
    spec.write_text(text, encoding="utf-8")
    output_root = tmp_path / "one-milestone-output"
    stubs = _ExecutionStubs(monkeypatch)

    code = driver.main(
        ["--spec", str(spec), "--output-root", str(output_root)]
    )

    assert code == 0
    assert stubs.milestone_calls == [(1,), (1,)]
    invocation = json.loads(
        (output_root / "invocation.json").read_text(encoding="utf-8")
    )
    expansion = json.loads(
        (output_root / "expansion.json").read_text(encoding="utf-8")
    )
    assert invocation["epochs"] is None
    assert {
        run["overrides"]["training.epochs"]
        for run in expansion["selected_runs"]
    } == {1}
    for run_id in (RUN_A, RUN_B):
        attempt = output_root / "runs" / run_id / "attempt-1"
        source_config = json.loads(
            (attempt / "source_config.json").read_text(encoding="utf-8")
        )
        trajectory = json.loads(
            (attempt / "milestone_trajectory.json").read_text(encoding="utf-8")
        )
        assert source_config["training"]["epochs"] == 1
        assert [row["epoch"] for row in trajectory] == [1]
        for name in (
            "milestone_trajectory.csv",
            "milestone_reconstruction_grid.png",
            "milestone_review.json",
        ):
            assert (attempt / name).is_file()


def test_dimension_epoch_override_below_milestones_fails_before_dataset_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = CONVERGENCE_SPEC.read_text(encoding="utf-8").replace(
        '"model.architecture" = "cnn"',
        '"model.architecture" = "cnn"\n"training.epochs" = 20',
        1,
    )
    spec = tmp_path / "invalid-milestone-budget.toml"
    spec.write_text(text, encoding="utf-8")
    output_root = tmp_path / "must-not-exist"
    monkeypatch.setattr(runtime_study, "_load_validated_datasets", _forbid)

    code = driver.main(
        ["--spec", str(spec), "--output-root", str(output_root)]
    )

    error = capsys.readouterr().err
    assert code == 2
    assert "training.epochs=20" in error
    assert "diagnostics milestone 80" in error
    assert not output_root.exists()


# --------------------------------------------------------------------------
# Tiny integration study: execute, complete, resume, rerun.
# --------------------------------------------------------------------------


def test_tiny_study_executes_completes_and_resumes(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    stubs = _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "study-out"

    code = driver.main(
        ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
    )

    assert code == 0
    assert len(stubs.train_calls) == 2
    # Deterministic run ids own the run directories.
    for run_id in (RUN_A, RUN_B):
        completion = output_root / "runs" / run_id / "attempt-1" / "completion.json"
        assert completion.is_file()
        checkpoint = json.loads(
            (completion.parent / "checkpoint.json").read_text(encoding="utf-8")
        )
        assert checkpoint["best_checkpoint_sha256"]
        assert checkpoint["fixed_batch_identity"]["test_npz_sha256"]
        # Pre-reload best-state reference texture/canvas are persisted evidence.
        assert (completion.parent / "arrays" / "prereload_texture.npy").is_file()
        assert (completion.parent / "arrays" / "prereload_canvas.npy").is_file()
    # Study-level artifacts and machine-readable expansion records exist.
    for name in (
        "report.md",
        "aggregate_metrics.json",
        "aggregate_metrics.csv",
        "arm_seed_status.json",
        "verdicts.json",
        "expansion.json",
    ):
        assert (output_root / name).is_file()
    metric_paths = _metric_paths(output_root)
    assert "truth_quality.amp_pearson" in metric_paths[RUN_A]
    assert "truth_quality.amp_ssim" in metric_paths[RUN_A]
    assert "truth_quality.absolute_amp_nrmse" in metric_paths[RUN_A]
    assert "stability.reload_max_abs_error" in metric_paths[RUN_A]
    assert "measurement_consistency.relative_l2_intensity_error" in metric_paths[RUN_A]
    assert "measurement_consistency.varpro.s1" in metric_paths[RUN_A]
    assert "runtime.train_seconds" in metric_paths[RUN_A]
    assert not any(
        path.startswith("reference_agreement.") for path in metric_paths[RUN_A]
    )
    verdicts = _verdict_rows(output_root)
    assert verdicts["seed_success"]["verdict"] == "pass"
    assert verdicts["reload_stability"]["verdict"] == "pass"
    assert verdicts["truth_recognizability"]["verdict"] == "pass"
    assert verdicts["truth_amp_ssim"]["verdict"] == "pass"
    assert verdicts["truth_amp_ssim"]["reason"] != "missing_or_invalid_operand"
    assert verdicts["reference_agreement_check"]["applicability"] == "not_applicable"
    assert verdicts["visual"]["verdict"] == "inconclusive"
    report_text = (output_root / "report.md").read_text(encoding="utf-8")
    truth_section = report_text.split("### truth_quality", 1)[1].split(
        "### reference_agreement", 1
    )[0]
    assert f"- {ARM_A}: AVAILABLE (" in truth_section
    assert "truth_quality.amp_ssim" in truth_section

    # A plain second invocation must refuse to clobber completed evidence.
    code = driver.main(
        ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
    )
    assert code == 2
    assert len(stubs.train_calls) == 2

    # Resume validates fingerprints and reuses both completed runs.
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--resume",
        ]
    )
    assert code == 0
    assert len(stubs.train_calls) == 2
    assert _verdict_rows(output_root)["seed_success"]["verdict"] == "pass"
    resumed_paths = _metric_paths(output_root)
    assert "truth_quality.amp_pearson" in resumed_paths[RUN_A]

    # Rerun archives the completed attempts and re-executes both runs.
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--rerun",
        ]
    )
    assert code == 0
    assert len(stubs.train_calls) == 4
    run_entries = sorted(
        entry.name for entry in (output_root / "runs" / RUN_A).iterdir()
    )
    assert "attempt-2" in run_entries
    assert "archive" in run_entries
    archived = list((output_root / "runs" / RUN_A / "archive").iterdir())
    assert archived, "rerun must archive the prior completed attempt"


def test_provenance_and_training_history_artifacts_flow_through(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "provenance-out"

    code = driver.main(
        ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
    )

    assert code == 0
    attempt = output_root / "runs" / RUN_A / "attempt-1"
    provenance = json.loads((attempt / "provenance.json").read_text(encoding="utf-8"))
    expansion = json.loads((output_root / "expansion.json").read_text(encoding="utf-8"))
    fingerprints = json.loads(
        (attempt / "fingerprints.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads((attempt / "checkpoint.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "ablation_provenance_v1"
    assert provenance["manifest_sha256"] == expansion["manifest_sha256"]
    assert provenance["logical_run_id"] == RUN_A
    assert provenance["seed"] == 7
    assert provenance["git"] == {
        "commit": "c" * 40,
        "clean": True,
        "tracked_patch_sha256": None,
        "untracked_sources": [],
    }
    assert provenance["environment_digest"] == "e" * 64
    assert {"dataset.train", "dataset.test", "dataset.provenance"} <= set(
        provenance["content_sha256s"]
    )
    assert provenance["resolved_configs"]
    assert provenance["claim_grade"] is True
    assert (
        provenance["selected_checkpoint_sha256"] == checkpoint["best_checkpoint_sha256"]
    )
    assert provenance["fingerprints"] == {
        "training": fingerprints["training"],
        "inference": fingerprints["inference"],
    }

    history = json.loads(
        (attempt / "training_history.json").read_text(encoding="utf-8")
    )
    assert history["available"] is True
    assert history["history"]["series"]["poisson_train_loss_epoch"]["value"] == [0.5]

    completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
    hashed_paths = {artifact["path"] for artifact in completion["artifacts"]}
    assert {"provenance.json", "training_history.json"} <= hashed_paths

    metric_paths = _metric_paths(output_root)
    assert {
        "stability.loss_final",
        "stability.loss_all_finite",
        "stability.gradient_norm_max",
        "stability.gradient_norm_final",
        "stability.gradient_norm_all_finite",
        "stability.clip_fraction",
    } <= metric_paths[RUN_A]

    mapping = json.loads(
        (output_root / "figure_row_mapping.json").read_text(encoding="utf-8")
    )
    assert RUN_A in mapping["training_gradient_curves.png"]
    assert RUN_B in mapping["training_gradient_curves.png"]

    # Resume rebuilds the report curves from the stored per-run history artifact.
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--resume",
        ]
    )
    assert code == 0
    resumed = json.loads(
        (output_root / "figure_row_mapping.json").read_text(encoding="utf-8")
    )
    assert RUN_A in resumed["training_gradient_curves.png"]


def test_visual_review_import_controls_manual_gate(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "review-out"
    assert (
        driver.main(
            ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
        )
        == 0
    )
    assert _verdict_rows(output_root)["visual"]["verdict"] == "inconclusive"
    grid_path = output_root / "reconstruction_truth_error_grid.png"
    grid_bytes = grid_path.read_bytes()
    grid_sha256 = hashlib.sha256(grid_bytes).hexdigest()

    review = tmp_path / "review.json"
    payload = {
        "schema_version": "visual_review_v1",
        "reviewer": "tester",
        "timestamp": "2026-07-10T00:00:00Z",
        "figure_sha256": "b" * 64,
        "decision": "approve",
        "recognizable": True,
        "flat": False,
        "checkerboard": False,
        "mirrored": False,
        "saturation": False,
        "collapse": False,
        "notes": "looks fine",
    }
    review.write_text(json.dumps(payload), encoding="utf-8")
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--resume",
            "--visual-review",
            str(review),
        ]
    )
    assert code == 2
    assert _verdict_rows(output_root)["visual"]["verdict"] == "inconclusive"

    review.write_text(
        json.dumps(
            {
                **payload,
                "figure_sha256": grid_sha256,
            }
        ),
        encoding="utf-8",
    )
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--resume",
            "--visual-review",
            str(review),
        ]
    )
    assert code == 0
    assert _verdict_rows(output_root)["visual"]["verdict"] == "pass"
    assert grid_path.read_bytes() == grid_bytes
    assert hashlib.sha256(grid_path.read_bytes()).hexdigest() == grid_sha256
    stored_review = json.loads((output_root / "visual_review.json").read_text())
    assert stored_review["schema_version"] == "visual_review_v1"
    assert stored_review["figure_sha256"] == grid_sha256

    bad_review = tmp_path / "bad_review.json"
    bad_review.write_text(json.dumps({"schema_version": "bogus"}), encoding="utf-8")
    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--resume",
            "--visual-review",
            str(bad_review),
        ]
    )
    assert code == 2


def _completed_same_path_review(output_root: Path) -> dict[str, object]:
    grid_sha256 = hashlib.sha256(
        (output_root / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    family = {
        "decision": "approve",
        "recognizable": True,
        "flat": False,
        "checkerboard": False,
        "mirrored": False,
        "saturation": False,
        "collapse": False,
        "notes": "same-path reviewed",
    }
    return {
        "schema_version": "visual_review_v1",
        "reviewer": "same-path-reviewer",
        "timestamp": "2026-07-11T18:00:00Z",
        "figure_sha256": grid_sha256,
        "families": {"deadleaves": family, "lines": family},
    }


def test_same_path_visual_review_import_reseals_completed_report(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "same-path-review"
    assert (
        driver.main(
            ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
        )
        == 0
    )
    old_completion = json.loads((output_root / "report_completion.json").read_text())
    old_review_hash = next(
        item["sha256"]
        for item in old_completion["artifacts"]
        if item["path"] == "visual_review.json"
    )
    review_path = output_root / "visual_review.json"
    review_path.write_text(
        json.dumps(_completed_same_path_review(output_root)), encoding="utf-8"
    )

    assert (
        driver.main(
            [
                "--spec",
                str(workspace["spec"]),
                "--output-root",
                str(output_root),
                "--resume",
                "--visual-review",
                str(review_path),
            ]
        )
        == 0
    )

    completion = reporting.verify_completed_report(output_root)
    new_review_hash = next(
        item["sha256"]
        for item in completion["artifacts"]
        if item["path"] == "visual_review.json"
    )
    assert new_review_hash == hashlib.sha256(review_path.read_bytes()).hexdigest()
    assert new_review_hash != old_review_hash
    assert not list((output_root / "runs").glob("*/attempt-2"))


def test_external_visual_review_symlink_is_unprivileged_and_load_bound(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "external-symlink-review"
    assert (
        driver.main(
            ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
        )
        == 0
    )
    family_aware = _completed_same_path_review(output_root)
    family = family_aware["families"]["deadleaves"]
    original = {
        "schema_version": "visual_review_v1",
        "reviewer": "external-symlink-original",
        "timestamp": family_aware["timestamp"],
        "figure_sha256": family_aware["figure_sha256"],
        **family,
    }
    changed = {**original, "reviewer": "external-symlink-changed"}
    target = tmp_path / "external-symlink-target.json"
    target.write_text(json.dumps(original), encoding="utf-8")
    review_link = tmp_path / "external-symlink-input.json"
    review_link.symlink_to(target)
    original_dataset_load = runtime_study._load_validated_datasets

    def change_target_after_review_load(loaded):
        target.write_text(json.dumps(changed), encoding="utf-8")
        return original_dataset_load(loaded)

    monkeypatch.setattr(
        runtime_study, "_load_validated_datasets", change_target_after_review_load
    )

    assert (
        driver.main(
            [
                "--spec",
                str(workspace["spec"]),
                "--output-root",
                str(output_root),
                "--resume",
                "--visual-review",
                str(review_link),
            ]
        )
        == 0
    )

    reporting.verify_completed_report(output_root)
    stored = json.loads((output_root / "visual_review.json").read_text())
    assert stored["reviewer"] == "external-symlink-original"
    assert json.loads(target.read_text())["reviewer"] == "external-symlink-changed"
    assert not list((output_root / "runs").glob("*/attempt-2"))


@pytest.mark.parametrize(
    "failure",
    (
        "corrupt_review",
        "pending_review",
        "legacy_review",
        "legacy_family_unaware_v1",
        "wrong_grid",
        "normalized_alias",
        "hardlink_alias",
        "extra_mismatch",
        "review_symlink",
        "artifact_symlink",
        "missing_artifact",
        "missing_completion",
        "completion_schema",
        "completion_inclusion",
    ),
)
def test_same_path_visual_review_recovery_fails_closed(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / f"same-path-{failure}"
    assert (
        driver.main(
            ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
        )
        == 0
    )
    completion_path = output_root / "report_completion.json"
    completion_before = completion_path.read_bytes()
    review_path = output_root / "visual_review.json"
    valid = _completed_same_path_review(output_root)
    review_path.write_text(json.dumps(valid), encoding="utf-8")
    review_argument = review_path
    expected_usage_error = failure in {
        "corrupt_review",
        "pending_review",
        "legacy_review",
        "wrong_grid",
    }
    if failure == "corrupt_review":
        review_path.write_text("{", encoding="utf-8")
    elif failure == "pending_review":
        review_path.write_text(
            json.dumps(
                {
                    "schema_version": "visual_review_pending_v1",
                    "state": "pending",
                    "figure_path": "reconstruction_truth_error_grid.png",
                    "figure_sha256": valid["figure_sha256"],
                    "families": ["deadleaves", "lines"],
                    "instructions": "pending",
                }
            ),
            encoding="utf-8",
        )
    elif failure == "legacy_review":
        review_path.write_text(
            json.dumps({**valid, "schema_version": "visual_review_v0"}),
            encoding="utf-8",
        )
    elif failure == "legacy_family_unaware_v1":
        family = valid["families"]["deadleaves"]
        review_path.write_text(
            json.dumps(
                {
                    "schema_version": "visual_review_v1",
                    "reviewer": valid["reviewer"],
                    "timestamp": valid["timestamp"],
                    "figure_sha256": valid["figure_sha256"],
                    **family,
                }
            ),
            encoding="utf-8",
        )
    elif failure == "wrong_grid":
        review_path.write_text(
            json.dumps({**valid, "figure_sha256": "a" * 64}), encoding="utf-8"
        )
    elif failure == "normalized_alias":
        alias_parent = output_root / "alias"
        alias_parent.mkdir()
        review_argument = alias_parent / ".." / "visual_review.json"
        assert ".." in str(review_argument)
    elif failure == "hardlink_alias":
        review_argument = tmp_path / "hardlink-review.json"
        review_argument.hardlink_to(review_path)
    elif failure == "extra_mismatch":
        (output_root / "report.md").write_text("second mismatch", encoding="utf-8")
    elif failure == "review_symlink":
        external = tmp_path / f"{failure}.json"
        external.write_text(json.dumps(valid), encoding="utf-8")
        review_path.unlink()
        review_path.symlink_to(external)
    elif failure == "artifact_symlink":
        report = output_root / "report.md"
        external = tmp_path / f"{failure}.md"
        report.replace(external)
        report.symlink_to(external)
    elif failure == "missing_artifact":
        (output_root / "report.md").unlink()
    elif failure == "missing_completion":
        completion_path.unlink()
    elif failure == "completion_schema":
        completion = json.loads(completion_before)
        completion["schema_version"] = "ablation_report_completion_v0"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        completion_before = completion_path.read_bytes()
    elif failure == "completion_inclusion":
        completion = json.loads(completion_before)
        completion["artifacts"] = completion["artifacts"][1:]
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        completion_before = completion_path.read_bytes()

    argv = [
        "--spec",
        str(workspace["spec"]),
        "--output-root",
        str(output_root),
        "--resume",
        "--visual-review",
        str(review_argument),
    ]
    if expected_usage_error:
        assert driver.main(argv) == 2
    else:
        with pytest.raises(reporting.ReportingError):
            driver.main(argv)
    if failure == "missing_completion":
        assert not completion_path.exists()
    else:
        assert completion_path.read_bytes() == completion_before


def test_experimental_reference_dataset_uses_reference_namespace(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "exp-ref-out"
    run_id = f"{STUDY_ID}--exp_ref--a--seed-7"

    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--dataset",
            "exp_ref",
            "--only",
            "variant=a",
        ]
    )

    assert code == 0
    metric_paths = _metric_paths(output_root)
    assert set(metric_paths) == {run_id}
    assert "reference_agreement.amp_pearson" in metric_paths[run_id]
    assert "reference_agreement.amp_ssim" in metric_paths[run_id]
    assert not any(path.startswith("truth_quality.") for path in metric_paths[run_id])
    verdicts = _verdict_rows(output_root)
    assert verdicts["truth_recognizability"]["applicability"] == "not_applicable"
    assert verdicts["reference_agreement_check"]["verdict"] == "pass"
    assert verdicts["reference_agreement_check"]["contributing_run_ids"] == [run_id]


def test_experimental_no_reference_dataset_uses_measurement_only(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "exp-blind-out"
    run_id = f"{STUDY_ID}--exp_blind--a--seed-7"

    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--dataset",
            "exp_blind",
            "--only",
            "variant=a",
        ]
    )

    assert code == 0
    metric_paths = _metric_paths(output_root)
    image_paths = {
        path
        for path in metric_paths[run_id]
        if path.startswith(("truth_quality.", "reference_agreement."))
    }
    assert image_paths == set()
    assert "measurement_consistency.mean_raw_poisson_nll" in metric_paths[run_id]
    verdicts = _verdict_rows(output_root)
    assert verdicts["truth_recognizability"]["applicability"] == "not_applicable"
    assert verdicts["reference_agreement_check"]["applicability"] == "not_applicable"
    assert "missing_capability" in verdicts["reference_agreement_check"]["reason"]
    report_text = (output_root / "report.md").read_text(encoding="utf-8")
    blind_arm = f"{STUDY_ID}--exp_blind--a"
    truth_section = report_text.split("### truth_quality", 1)[1].split(
        "### reference_agreement", 1
    )[0]
    reference_section = report_text.split("### reference_agreement", 1)[1].split(
        "### measurement_consistency", 1
    )[0]
    assert (
        f"- {blind_arm}: NOT_APPLICABLE (declared truth role is none)" in truth_section
    )
    assert (
        f"- {blind_arm}: NOT_APPLICABLE (declared truth role is none)"
        in reference_section
    )


def test_dataset_spec_execution_runs_standalone_descriptor(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    _ExecutionStubs(monkeypatch)
    output_root = tmp_path / "spec-out"
    run_id = f"{STUDY_ID}--spec_ds--a--seed-7"

    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--dataset-spec",
            str(workspace["dataset_spec"]),
            "--dataset",
            "spec_ds",
            "--only",
            "variant=a",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    metric_paths = _metric_paths(output_root)
    assert set(metric_paths) == {run_id}
    assert "reference_agreement.amp_pearson" in metric_paths[run_id]
    assert (output_root / "runs" / run_id / "attempt-1" / "completion.json").is_file()


def test_fail_fast_stops_after_first_failure(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    stubs = _ExecutionStubs(monkeypatch, fail_training=True)
    output_root = tmp_path / "fail-fast-out"

    code = driver.main(
        [
            "--spec",
            str(workspace["spec"]),
            "--output-root",
            str(output_root),
            "--fail-fast",
        ]
    )

    assert code == 1
    assert len(stubs.train_calls) == 1
    statuses = json.loads(
        (output_root / "arm_seed_status.json").read_text(encoding="utf-8")
    )
    by_run = {row["run_id"]: row for row in statuses["rows"]}
    assert by_run[RUN_A]["status"] == "failed"
    assert by_run[RUN_A]["failure_stage"] == "training"
    assert by_run[RUN_B]["status"] == "missing"


def test_without_fail_fast_a_failed_arm_does_not_stop_other_arms(
    workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(workspace["root"])
    stubs = _ExecutionStubs(monkeypatch, fail_training=True)
    output_root = tmp_path / "continue-out"

    code = driver.main(
        ["--spec", str(workspace["spec"]), "--output-root", str(output_root)]
    )

    assert code == 0
    assert len(stubs.train_calls) == 2
    statuses = json.loads(
        (output_root / "arm_seed_status.json").read_text(encoding="utf-8")
    )
    by_run = {row["run_id"]: row for row in statuses["rows"]}
    assert by_run[RUN_A]["status"] == "failed"
    assert by_run[RUN_B]["status"] == "failed"
    verdicts = _verdict_rows(output_root)
    assert verdicts["seed_success"]["verdict"] == "fail"

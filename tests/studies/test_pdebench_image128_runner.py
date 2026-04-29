import csv
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch


class _WorkerRuntime:
    def __init__(self, *, broadcast_value):
        self.requested_device = "cpu"
        self.device = torch.device("cpu")
        self.rank = 1
        self.local_rank = 1
        self.world_size = 2
        self.backend = "gloo"
        self.distributed_enabled = False
        self.launched_via_torchrun = True
        self._broadcast_value = broadcast_value

    @property
    def is_rank_zero(self):
        return False

    def build_training_loader(self, dataset, *, batch_size, num_workers, collate_fn, shuffle=False):
        from torch.utils.data import DataLoader

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            ),
            None,
        )

    def maybe_reset_peak_memory_stats(self):
        return None

    def reduce_sum(self, value):
        return float(value)

    def reduce_sum_dict(self, values):
        return {str(key): float(value) for key, value in values.items()}

    def barrier(self):
        return None

    def max_cuda_memory_bytes(self):
        return None

    def training_runtime_payload(self):
        return {
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "distributed_enabled": False,
            "distributed_world_size": 2,
            "distributed_rank": 1,
            "distributed_local_rank": 1,
            "distributed_backend": "gloo",
            "launched_via_torchrun": True,
        }

    def broadcast_object(self, value, *, src=0):
        return value if value is not None else dict(self._broadcast_value)


class _DistributedRankZeroRuntime:
    def __init__(self, *, world_size=2):
        self.requested_device = "cpu"
        self.device = torch.device("cpu")
        self.rank = 0
        self.local_rank = 0
        self.world_size = int(world_size)
        self.backend = "gloo"
        self.distributed_enabled = True
        self.launched_via_torchrun = True

    @property
    def is_rank_zero(self):
        return True

    def build_training_loader(self, dataset, *, batch_size, num_workers, collate_fn, shuffle=False):
        from torch.utils.data import DataLoader

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            ),
            None,
        )

    def maybe_reset_peak_memory_stats(self):
        return None

    def reduce_sum(self, value):
        return float(value) * float(self.world_size)

    def reduce_sum_dict(self, values):
        return {str(key): float(value) * float(self.world_size) for key, value in values.items()}

    def barrier(self):
        return None

    def max_cuda_memory_bytes(self):
        return None

    def training_runtime_payload(self):
        return {
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "distributed_enabled": True,
            "distributed_world_size": int(self.world_size),
            "distributed_rank": 0,
            "distributed_local_rank": 0,
            "distributed_backend": "gloo",
            "launched_via_torchrun": True,
        }

    def broadcast_object(self, value, *, src=0):
        return value


def _write_tiny_darcy(path: Path, *, n: int = 12) -> Path:
    rng = np.random.default_rng(7)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["beta"] = 1.0
        handle.create_dataset("nu", data=rng.normal(size=(n, 8, 8)).astype(np.float32))
        handle.create_dataset("tensor", data=rng.normal(size=(n, 1, 8, 8)).astype(np.float32))
    return path


def _stub_skimage(monkeypatch):
    if "skimage" in sys.modules:
        return
    draw_module = types.ModuleType("skimage.draw")
    morphology_module = types.ModuleType("skimage.morphology")
    skimage_module = types.ModuleType("skimage")
    skimage_module.draw = draw_module
    skimage_module.morphology = morphology_module
    monkeypatch.setitem(sys.modules, "skimage", skimage_module)
    monkeypatch.setitem(sys.modules, "skimage.draw", draw_module)
    monkeypatch.setitem(sys.modules, "skimage.morphology", morphology_module)


def _stub_pdebench_models_module(monkeypatch):
    module_name = "scripts.studies.pdebench_image128.models"
    if module_name in sys.modules:
        return

    fake_models = types.ModuleType(module_name)

    class ModelBuildBlocker(Exception):
        reason = "blocked"

        def to_payload(self, *, run_id):
            return {"run_id": run_id, "blocker_reason": self.reason}

    def build_model_from_profile(*args, **kwargs):
        raise AssertionError("test must monkeypatch build_model_from_profile after import")

    def describe_model(model, *, profile):
        return {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "profile_config": profile.to_model_config(),
        }

    fake_models.ModelBuildBlocker = ModelBuildBlocker
    fake_models.build_model_from_profile = build_model_from_profile
    fake_models.describe_model = describe_model
    monkeypatch.setitem(sys.modules, module_name, fake_models)


def _write_tiny_cfd_cns(path: Path, *, n: int = 5, t: int = 4) -> Path:
    rng = np.random.default_rng(11)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["eta"] = 0.01
        handle.attrs["zeta"] = 0.01
        for field in ("density", "Vx", "Vy", "pressure"):
            handle.create_dataset(field, data=rng.normal(size=(n, t, 8, 8)).astype(np.float32))
        handle.create_dataset("x-coordinate", data=np.linspace(0.0, 0.875, 8, dtype=np.float32))
        handle.create_dataset("y-coordinate", data=np.linspace(0.0, 0.875, 8, dtype=np.float32))
        handle.create_dataset("t-coordinate", data=np.linspace(0.0, 0.2, t + 1, dtype=np.float32))
    return path


def _write_fake_cfd_cns_compare_run(
    run_root: Path,
    *,
    profile_id: str,
    epochs: int,
    err_nrmse: float,
    runtime_sec: float = 100.0,
    batch_size: int = 4,
    history_len: int = 2,
    max_windows_per_trajectory: int = 8,
    split_counts: dict[str, int] | None = None,
    dataset_file: str = "/tmp/fake-cfd-cns.hdf5",
    training_loss: str = "mse",
    field_order: list[str] | None = None,
    target_offset: float = 0.0,
    with_npz: bool = True,
    mode: str = "pilot",
    evidence_scope: str = "capped_decision_support_only",
    metric_interpretation: str = "decision_support_not_benchmark_performance",
) -> Path:
    field_order = field_order or ["density", "Vx", "Vy", "pressure"]
    split_counts = split_counts or {"train": 512, "val": 64, "test": 64}
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "invocation.json").write_text(
        json.dumps(
            {
                "parsed_args": {
                    "task_id": "2d_cfd_cns",
                    "mode": mode,
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "history_len": int(history_len),
                    "max_windows_per_trajectory": int(max_windows_per_trajectory),
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "2d_cfd_cns",
                "data_file": dataset_file,
                "history_len": int(history_len),
                "field_order": field_order,
                "field_axis_order": "NTHW",
                "sample_contract": f"concat u[t-{history_len}:t] -> u[t]",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "history_len": int(history_len),
                "max_windows_per_trajectory": int(max_windows_per_trajectory),
                "split_counts": dict(split_counts),
                "source_file": {"path": dataset_file},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metric_payload = {
        "profile_id": profile_id,
        "status": "completed",
        "runtime_sec": float(runtime_sec),
        "training_loss": training_loss,
        "err_RMSE": float(err_nrmse * 10.0),
        "err_nRMSE": float(err_nrmse),
        "relative_l2": float(err_nrmse),
        "fRMSE_low": float(err_nrmse * 2.0),
        "fRMSE_mid": float(err_nrmse * 1.5),
        "fRMSE_high": float(err_nrmse * 0.5),
        "parameter_count": 123456,
        "metric_units": "denormalized_state_units",
        "fourier_metric_units": "denormalized_state_units_fft_ortho",
    }
    (run_root / f"metrics_{profile_id}.json").write_text(
        json.dumps(metric_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / f"model_profile_{profile_id}.json").write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "parameter_count": 123456,
                "profile_config": {"profile_id": profile_id},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "comparison_summary.json").write_text(
        json.dumps(
            {
                "task_id": "2d_cfd_cns",
                "mode": mode,
                "history_len": int(history_len),
                "evidence_scope": evidence_scope,
                "metric_interpretation": metric_interpretation,
                "profile_results": [metric_payload],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_root / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_id",
                "status",
                "err_RMSE",
                "err_nRMSE",
                "relative_l2",
                "fRMSE_low",
                "fRMSE_mid",
                "fRMSE_high",
                "parameter_count",
                "blocker_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "profile_id": profile_id,
                "status": "completed",
                "err_RMSE": metric_payload["err_RMSE"],
                "err_nRMSE": metric_payload["err_nRMSE"],
                "relative_l2": metric_payload["relative_l2"],
                "fRMSE_low": metric_payload["fRMSE_low"],
                "fRMSE_mid": metric_payload["fRMSE_mid"],
                "fRMSE_high": metric_payload["fRMSE_high"],
                "parameter_count": metric_payload["parameter_count"],
                "blocker_reason": "",
            }
        )

    if with_npz:
        channels = len(field_order)
        target = np.arange(channels * 4, dtype=np.float32).reshape(channels, 2, 2) + np.float32(target_offset)
        prediction = target + np.float32(err_nrmse)
        abs_error = np.abs(prediction - target)
        np.savez_compressed(
            run_root / f"comparison_{profile_id}_sample0.npz",
            prediction=prediction,
            target=target,
            abs_error=abs_error,
            field_order=np.asarray(field_order),
        )

    return run_root


def test_literature_context_payload_contains_values_and_caveats(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_literature_context

    path = write_literature_context(tmp_path, task_id="darcy")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["task_id"] == "darcy"
    assert payload["access_date"] == "2026-04-20"
    assert payload["calibration_targets"]["pdebench_unet"]["nRMSE"] == 3.3e-2
    assert payload["calibration_targets"]["pdebench_fno"]["RMSE"] == 1.2e-2
    assert payload["calibration_targets"]["hamlet"]["nRMSE"] == 1.40e-2
    assert all("protocol_caveat" in item for item in payload["calibration_targets"].values())


def test_comparison_summary_rejects_tiny_unet_as_strong_baseline(tmp_path):
    from scripts.studies.pdebench_image128.reporting import build_comparison_summary

    try:
        build_comparison_summary(
            task_id="darcy",
            mode="benchmark",
            output_root=tmp_path,
            profile_results=[
                {"profile_id": "hybrid_resnet_base", "status": "completed"},
                {"profile_id": "fno_base", "status": "completed"},
                {"profile_id": "unet_tiny_smoke", "status": "completed"},
            ],
        )
    except ValueError as exc:
        assert "unet_tiny_smoke" in str(exc)
    else:
        raise AssertionError("benchmark summary must reject tiny U-Net as a strong baseline")


def test_cfd_cns_comparison_summary_uses_task_specific_required_primary_profiles(tmp_path):
    from scripts.studies.pdebench_image128.reporting import build_comparison_summary

    payload = build_comparison_summary(
        task_id="2d_cfd_cns",
        mode="benchmark",
        output_root=tmp_path,
        profile_results=[
            {"profile_id": "hybrid_resnet_cns", "status": "completed"},
            {"profile_id": "fno_base", "status": "completed"},
            {"profile_id": "unet_strong", "status": "completed"},
        ],
    )

    assert payload["required_primary_profiles"] == ["fno_base", "hybrid_resnet_cns", "unet_strong"]
    assert payload["performance_assessment_complete"] is True


def test_cfd_cns_pilot_comparison_summary_marks_capped_decision_support_only(tmp_path):
    from scripts.studies.pdebench_image128.reporting import build_comparison_summary

    payload = build_comparison_summary(
        task_id="2d_cfd_cns",
        mode="pilot",
        output_root=tmp_path,
        profile_results=[
            {"profile_id": "spectral_resnet_bottleneck_base", "status": "completed"},
            {"profile_id": "spectral_resnet_bottleneck_noshare", "status": "completed"},
        ],
    )

    assert payload["mode"] == "pilot"
    assert payload["evidence_scope"] == "capped_decision_support_only"
    assert payload["metric_interpretation"] == "decision_support_not_benchmark_performance"
    assert payload["performance_assessment_complete"] is False


def test_reference_run_manifest_records_row_local_contract_fields(tmp_path):
    from scripts.studies.pdebench_image128.reporting import build_reference_run_manifest, write_reference_run_manifest

    payload = build_reference_run_manifest(
        task_id="2d_cfd_cns",
        dataset_file="/tmp/fake-cfd-cns.hdf5",
        split_counts={"train": 512, "val": 64, "test": 64},
        max_windows_per_trajectory=8,
        history_len=2,
        training_loss="mse",
        batch_size=4,
        metric_family=["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"],
        required_rows={
            "10ep": [
                {
                    "run_root": "runs/spectral-10ep",
                    "profile_id": "spectral_resnet_bottleneck_base",
                    "epochs": 10,
                    "source_document": "docs/example.md",
                }
            ]
        },
        optional_rows={
            "10ep": [
                {
                    "run_root": "runs/hybrid-10ep",
                    "profile_id": "hybrid_resnet_cns",
                    "epochs": 10,
                    "source_document": "docs/example.md",
                }
            ]
        },
    )
    path = write_reference_run_manifest(payload, tmp_path / "reference_runs.json")

    written = json.loads(path.read_text(encoding="utf-8"))
    row = written["required_rows"]["10ep"][0]

    assert row["run_root"] == "runs/spectral-10ep"
    assert row["profile_id"] == "spectral_resnet_bottleneck_base"
    assert row["epochs"] == 10
    assert row["dataset_file"] == "/tmp/fake-cfd-cns.hdf5"
    assert row["split_counts"] == {"train": 512, "val": 64, "test": 64}
    assert row["max_windows_per_trajectory"] == 8
    assert row["history_len"] == 2
    assert row["training_loss"] == "mse"
    assert row["batch_size"] == 4
    assert row["metric_family"] == ["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"]


def test_cross_run_compare_writes_merged_outputs_and_gallery(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_cross_run_compare

    author_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "author",
        profile_id="author_ffno_cns_base",
        epochs=10,
        err_nrmse=0.42,
    )
    spectral_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "spectral",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.11,
    )
    fno_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "fno",
        profile_id="fno_base",
        epochs=10,
        err_nrmse=0.23,
    )
    unet_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "unet",
        profile_id="unet_strong",
        epochs=10,
        err_nrmse=0.31,
    )
    hybrid_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "hybrid",
        profile_id="hybrid_resnet_cns",
        epochs=10,
        err_nrmse=0.19,
    )

    json_path, csv_path, payload = write_cross_run_compare(
        output_root=tmp_path / "out",
        epoch_label="10ep",
        expected_epochs=10,
        author_run_root=author_root,
        author_profile_id="author_ffno_cns_base",
        required_reference_rows=[
            {
                "run_root": str(spectral_root),
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/spectral.md",
            },
            {
                "run_root": str(fno_root),
                "profile_id": "fno_base",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/fno.md",
            },
            {
                "run_root": str(unet_root),
                "profile_id": "unet_strong",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/unet.md",
            },
        ],
        optional_reference_rows=[
            {
                "run_root": str(hybrid_root),
                "profile_id": "hybrid_resnet_cns",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/hybrid.md",
            }
        ],
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert (tmp_path / "out" / "compare_10ep_sample0.png").exists()
    assert (tmp_path / "out" / "compare_10ep_sample0_error.png").exists()
    assert payload["cross_run_gallery_blocked"] is None
    assert [row["profile_id"] for row in payload["profile_results"]] == [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
        "hybrid_resnet_cns",
    ]

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["profile_id"] for row in rows] == [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
        "hybrid_resnet_cns",
    ]


def test_cross_run_compare_accepts_generic_fresh_run_fields_for_modes32_lane(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_cross_run_compare

    fresh_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "fresh",
        profile_id="spectral_resnet_bottleneck_modes32",
        epochs=10,
        err_nrmse=0.09,
    )
    spectral_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "spectral",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.11,
    )
    fno_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "fno",
        profile_id="fno_base",
        epochs=10,
        err_nrmse=0.23,
    )
    unet_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "unet",
        profile_id="unet_strong",
        epochs=10,
        err_nrmse=0.31,
    )

    json_path, csv_path, payload = write_cross_run_compare(
        output_root=tmp_path / "out",
        epoch_label="10ep",
        expected_epochs=10,
        fresh_run_root=fresh_root,
        fresh_profile_id="spectral_resnet_bottleneck_modes32",
        required_reference_rows=[
            {
                "run_root": str(spectral_root),
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/spectral.md",
            },
            {
                "run_root": str(fno_root),
                "profile_id": "fno_base",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/fno.md",
            },
            {
                "run_root": str(unet_root),
                "profile_id": "unet_strong",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/unet.md",
            },
        ],
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["fresh_profile_id"] == "spectral_resnet_bottleneck_modes32"
    assert payload["fresh_run_root"] == str(fresh_root)
    assert "author_profile_id" not in payload
    assert "author_run_root" not in payload
    assert [row["profile_id"] for row in payload["profile_results"]] == [
        "spectral_resnet_bottleneck_modes32",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]


def test_cross_run_compare_accepts_multiple_fresh_rows_plus_frozen_context(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_cross_run_compare

    fresh_root = tmp_path / "fresh-sharing"
    _write_fake_cfd_cns_compare_run(
        fresh_root,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.11,
    )
    _write_fake_cfd_cns_compare_run(
        fresh_root,
        profile_id="spectral_resnet_bottleneck_noshare",
        epochs=40,
        err_nrmse=0.09,
    )
    fno_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "fno",
        profile_id="fno_base",
        epochs=40,
        err_nrmse=0.23,
    )
    unet_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "unet",
        profile_id="unet_strong",
        epochs=40,
        err_nrmse=0.31,
    )
    hybrid_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "hybrid",
        profile_id="hybrid_resnet_cns",
        epochs=40,
        err_nrmse=0.19,
    )

    json_path, csv_path, payload = write_cross_run_compare(
        output_root=tmp_path / "out",
        epoch_label="sharing_40ep",
        expected_epochs=40,
        fresh_run_root=fresh_root,
        fresh_profile_ids=[
            "spectral_resnet_bottleneck_base",
            "spectral_resnet_bottleneck_noshare",
        ],
        required_reference_rows=[
            {
                "run_root": str(fno_root),
                "profile_id": "fno_base",
                "epochs": 40,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/fno.md",
            },
            {
                "run_root": str(unet_root),
                "profile_id": "unet_strong",
                "epochs": 40,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/unet.md",
            },
        ],
        optional_reference_rows=[
            {
                "run_root": str(hybrid_root),
                "profile_id": "hybrid_resnet_cns",
                "epochs": 40,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/hybrid.md",
            }
        ],
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["fresh_profile_ids"] == [
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_noshare",
    ]
    assert "fresh_profile_id" not in payload
    assert [row["profile_id"] for row in payload["profile_results"]] == [
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_noshare",
        "fno_base",
        "unet_strong",
        "hybrid_resnet_cns",
    ]
    assert [row["profile_id"] for row in payload["included_required_reference_rows"]] == [
        "fno_base",
        "unet_strong",
    ]
    assert [row["profile_id"] for row in payload["included_optional_reference_rows"]] == [
        "hybrid_resnet_cns",
    ]


def test_cross_run_compare_rejects_fixed_contract_mismatch_for_modes32_lane(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_cross_run_compare

    fresh_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "fresh",
        profile_id="spectral_resnet_bottleneck_modes32",
        epochs=10,
        err_nrmse=0.09,
        batch_size=2,
    )
    spectral_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "spectral",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.11,
        batch_size=4,
    )

    try:
        write_cross_run_compare(
            output_root=tmp_path / "out",
            epoch_label="10ep",
            expected_epochs=10,
            fresh_run_root=fresh_root,
            fresh_profile_id="spectral_resnet_bottleneck_modes32",
            required_reference_rows=[
                {
                    "run_root": str(spectral_root),
                    "profile_id": "spectral_resnet_bottleneck_base",
                    "epochs": 10,
                    "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                    "split_counts": {"train": 512, "val": 64, "test": 64},
                    "max_windows_per_trajectory": 8,
                    "history_len": 2,
                    "training_loss": "mse",
                    "batch_size": 4,
                    "metric_family": [
                        "err_RMSE",
                        "err_nRMSE",
                        "relative_l2",
                        "fRMSE_low",
                        "fRMSE_mid",
                        "fRMSE_high",
                    ],
                    "source_document": "docs/spectral.md",
                }
            ],
        )
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("cross-run compare must reject fixed-contract mismatches for reused anchors")


def test_cross_run_compare_records_gallery_blocker_when_targets_do_not_align(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_cross_run_compare

    author_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "author",
        profile_id="author_ffno_cns_base",
        epochs=40,
        err_nrmse=0.42,
    )
    spectral_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "spectral",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.11,
        target_offset=5.0,
    )
    fno_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "fno",
        profile_id="fno_base",
        epochs=40,
        err_nrmse=0.23,
        with_npz=False,
    )
    unet_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "unet",
        profile_id="unet_strong",
        epochs=40,
        err_nrmse=0.31,
    )

    json_path, csv_path, payload = write_cross_run_compare(
        output_root=tmp_path / "out",
        epoch_label="40ep",
        expected_epochs=40,
        author_run_root=author_root,
        author_profile_id="author_ffno_cns_base",
        required_reference_rows=[
            {
                "run_root": str(spectral_root),
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 40,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/spectral.md",
            },
            {
                "run_root": str(fno_root),
                "profile_id": "fno_base",
                "epochs": 40,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/fno.md",
            },
            {
                "run_root": str(unet_root),
                "profile_id": "unet_strong",
                "epochs": 40,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/unet.md",
            },
        ],
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert not (tmp_path / "out" / "compare_40ep_sample0.png").exists()
    assert not (tmp_path / "out" / "compare_40ep_sample0_error.png").exists()
    assert payload["cross_run_gallery_blocked"] is not None
    assert payload["cross_run_gallery_blocked"]["reason"] in {
        "missing_sample_artifact",
        "target_mismatch",
    }


def test_scaling_trend_split_cap_delta_writes_outputs_and_deltas(tmp_path):
    from scripts.studies.pdebench_image128.reporting import (
        build_reference_run_manifest,
        write_reference_run_manifest,
        write_split_cap_scaling_trend,
    )

    profile_ids = [
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_shared_blocks10",
    ]
    metric_family = [
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
    ]
    dataset_file = "/tmp/fake-cfd-cns.hdf5"

    base_512 = _write_fake_cfd_cns_compare_run(
        tmp_path / "cap512-base",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.060,
        runtime_sec=1100.0,
        split_counts={"train": 512, "val": 64, "test": 64},
    )
    shared10_512 = _write_fake_cfd_cns_compare_run(
        tmp_path / "cap512-shared10",
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.055,
        runtime_sec=1350.0,
        split_counts={"train": 512, "val": 64, "test": 64},
    )
    cap1024 = tmp_path / "cap1024"
    _write_fake_cfd_cns_compare_run(
        cap1024,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.044,
        runtime_sec=2200.0,
        split_counts={"train": 1024, "val": 128, "test": 128},
    )
    _write_fake_cfd_cns_compare_run(
        cap1024,
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.045,
        runtime_sec=2750.0,
        split_counts={"train": 1024, "val": 128, "test": 128},
    )
    cap2048 = tmp_path / "cap2048"
    _write_fake_cfd_cns_compare_run(
        cap2048,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.033,
        runtime_sec=3600.0,
        split_counts={"train": 2048, "val": 256, "test": 256},
    )
    _write_fake_cfd_cns_compare_run(
        cap2048,
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.034,
        runtime_sec=4300.0,
        split_counts={"train": 2048, "val": 256, "test": 256},
    )

    manifest_512 = write_reference_run_manifest(
        build_reference_run_manifest(
            task_id="2d_cfd_cns",
            dataset_file=dataset_file,
            split_counts={"train": 512, "val": 64, "test": 64},
            max_windows_per_trajectory=8,
            history_len=2,
            training_loss="mse",
            batch_size=4,
            metric_family=metric_family,
            required_rows={
                "40ep": [
                    {
                        "run_root": str(base_512),
                        "profile_id": "spectral_resnet_bottleneck_base",
                        "epochs": 40,
                        "source_document": "docs/base512.md",
                    },
                    {
                        "run_root": str(shared10_512),
                        "profile_id": "spectral_resnet_bottleneck_shared_blocks10",
                        "epochs": 40,
                        "source_document": "docs/shared10-512.md",
                    },
                ]
            },
        ),
        tmp_path / "reference_runs_512cap_40ep.json",
    )
    manifest_1024 = write_reference_run_manifest(
        build_reference_run_manifest(
            task_id="2d_cfd_cns",
            dataset_file=dataset_file,
            split_counts={"train": 1024, "val": 128, "test": 128},
            max_windows_per_trajectory=8,
            history_len=2,
            training_loss="mse",
            batch_size=4,
            metric_family=metric_family,
            required_rows={
                "40ep": [
                    {
                        "run_root": str(cap1024),
                        "profile_id": "spectral_resnet_bottleneck_base",
                        "epochs": 40,
                        "source_document": "docs/base1024.md",
                    },
                    {
                        "run_root": str(cap1024),
                        "profile_id": "spectral_resnet_bottleneck_shared_blocks10",
                        "epochs": 40,
                        "source_document": "docs/shared10-1024.md",
                    },
                ]
            },
        ),
        tmp_path / "reference_runs_1024cap_40ep.json",
    )

    json_path, csv_path, payload = write_split_cap_scaling_trend(
        output_root=tmp_path / "out",
        profile_ids=profile_ids,
        reference_manifest_paths=[manifest_512, manifest_1024],
        fresh_run_root=cap2048,
        fresh_profile_ids=profile_ids,
        fresh_source_document="fresh_run",
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["evidence_scope"] == "capped_decision_support_only"
    assert payload["metric_interpretation"] == "decision_support_not_benchmark_performance"
    assert payload["allowed_contract_delta"]["delta_kind"] == "split_counts_only"
    assert payload["cap_sequence"] == ["512cap", "1024cap", "2048cap"]
    assert payload["cross_run_gallery_blocked"] is None
    assert (tmp_path / "out" / "compare_scaling_512_1024_2048_sample0.png").exists()
    assert (tmp_path / "out" / "compare_scaling_512_1024_2048_sample0_error.png").exists()

    profiles = {item["profile_id"]: item for item in payload["profiles"]}
    base = profiles["spectral_resnet_bottleneck_base"]
    assert base["metrics_by_cap"]["512cap"]["err_nRMSE"] == 0.060
    assert base["metrics_by_cap"]["1024cap"]["err_nRMSE"] == 0.044
    assert base["metrics_by_cap"]["2048cap"]["err_nRMSE"] == 0.033
    assert base["delta_1024_minus_512"]["err_nRMSE"] == -0.016
    assert base["delta_2048_minus_1024"]["err_nRMSE"] == -0.011
    assert base["runtime_delta_1024_minus_512"] == 1100.0
    assert base["runtime_delta_2048_minus_1024"] == 1400.0
    assert base["improvement_per_added_training_trajectory"]["1024_minus_512"]["err_nRMSE"] == 0.016 / 512.0
    assert base["improvement_per_added_training_trajectory"]["2048_minus_1024"]["relative_l2"] == 0.011 / 1024.0
    assert base["improvement_per_added_training_trajectory"]["2048_minus_1024"]["fRMSE_high"] == 0.0055 / 1024.0


def test_split_cap_delta_rejects_contract_drift_outside_split_counts(tmp_path):
    from scripts.studies.pdebench_image128.reporting import (
        build_reference_run_manifest,
        write_reference_run_manifest,
        write_split_cap_scaling_trend,
    )

    base_512 = _write_fake_cfd_cns_compare_run(
        tmp_path / "cap512-base",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.060,
        split_counts={"train": 512, "val": 64, "test": 64},
    )
    shared10_512 = _write_fake_cfd_cns_compare_run(
        tmp_path / "cap512-shared10",
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.055,
        split_counts={"train": 512, "val": 64, "test": 64},
    )
    bad_2048 = tmp_path / "cap2048-bad"
    _write_fake_cfd_cns_compare_run(
        bad_2048,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.033,
        batch_size=2,
        split_counts={"train": 2048, "val": 256, "test": 256},
    )
    _write_fake_cfd_cns_compare_run(
        bad_2048,
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.034,
        batch_size=2,
        split_counts={"train": 2048, "val": 256, "test": 256},
    )

    manifest_512 = write_reference_run_manifest(
        build_reference_run_manifest(
            task_id="2d_cfd_cns",
            dataset_file="/tmp/fake-cfd-cns.hdf5",
            split_counts={"train": 512, "val": 64, "test": 64},
            max_windows_per_trajectory=8,
            history_len=2,
            training_loss="mse",
            batch_size=4,
            metric_family=["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"],
            required_rows={
                "40ep": [
                    {"run_root": str(base_512), "profile_id": "spectral_resnet_bottleneck_base", "epochs": 40, "source_document": "docs/base512.md"},
                    {"run_root": str(shared10_512), "profile_id": "spectral_resnet_bottleneck_shared_blocks10", "epochs": 40, "source_document": "docs/shared10-512.md"},
                ]
            },
        ),
        tmp_path / "reference_runs_512cap_40ep.json",
    )

    try:
        write_split_cap_scaling_trend(
            output_root=tmp_path / "out",
            profile_ids=["spectral_resnet_bottleneck_base", "spectral_resnet_bottleneck_shared_blocks10"],
            reference_manifest_paths=[manifest_512],
            fresh_run_root=bad_2048,
            fresh_profile_ids=["spectral_resnet_bottleneck_base", "spectral_resnet_bottleneck_shared_blocks10"],
            fresh_source_document="fresh_run",
        )
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("split-cap scaling trend must reject contract drift outside split_counts")


def test_scaling_trend_records_nonfatal_gallery_blocker(tmp_path):
    from scripts.studies.pdebench_image128.reporting import (
        build_reference_run_manifest,
        write_reference_run_manifest,
        write_split_cap_scaling_trend,
    )

    base_512 = _write_fake_cfd_cns_compare_run(
        tmp_path / "cap512-base",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.060,
        split_counts={"train": 512, "val": 64, "test": 64},
    )
    shared10_512 = _write_fake_cfd_cns_compare_run(
        tmp_path / "cap512-shared10",
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.055,
        split_counts={"train": 512, "val": 64, "test": 64},
        target_offset=5.0,
    )
    cap2048 = tmp_path / "cap2048"
    _write_fake_cfd_cns_compare_run(
        cap2048,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=40,
        err_nrmse=0.033,
        split_counts={"train": 2048, "val": 256, "test": 256},
        with_npz=False,
    )
    _write_fake_cfd_cns_compare_run(
        cap2048,
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        epochs=40,
        err_nrmse=0.034,
        split_counts={"train": 2048, "val": 256, "test": 256},
    )

    manifest_512 = write_reference_run_manifest(
        build_reference_run_manifest(
            task_id="2d_cfd_cns",
            dataset_file="/tmp/fake-cfd-cns.hdf5",
            split_counts={"train": 512, "val": 64, "test": 64},
            max_windows_per_trajectory=8,
            history_len=2,
            training_loss="mse",
            batch_size=4,
            metric_family=["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"],
            required_rows={
                "40ep": [
                    {"run_root": str(base_512), "profile_id": "spectral_resnet_bottleneck_base", "epochs": 40, "source_document": "docs/base512.md"},
                    {"run_root": str(shared10_512), "profile_id": "spectral_resnet_bottleneck_shared_blocks10", "epochs": 40, "source_document": "docs/shared10-512.md"},
                ]
            },
        ),
        tmp_path / "reference_runs_512cap_40ep.json",
    )

    json_path, csv_path, payload = write_split_cap_scaling_trend(
        output_root=tmp_path / "out",
        profile_ids=["spectral_resnet_bottleneck_base", "spectral_resnet_bottleneck_shared_blocks10"],
        reference_manifest_paths=[manifest_512],
        fresh_run_root=cap2048,
        fresh_profile_ids=["spectral_resnet_bottleneck_base", "spectral_resnet_bottleneck_shared_blocks10"],
        fresh_source_document="fresh_run",
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["gallery_artifacts"] is None
    assert payload["cross_run_gallery_blocked"]["reason"] in {"missing_sample_artifact", "target_mismatch"}


def test_history1_cross_run_compare_records_allowed_delta_and_rows(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_history_delta_compare

    history1_root = tmp_path / "history1"
    _write_fake_cfd_cns_compare_run(
        history1_root,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.08,
        history_len=1,
    )
    _write_fake_cfd_cns_compare_run(
        history1_root,
        profile_id="hybrid_resnet_cns",
        epochs=10,
        err_nrmse=0.10,
        history_len=1,
    )
    _write_fake_cfd_cns_compare_run(
        history1_root,
        profile_id="fno_base",
        epochs=10,
        err_nrmse=0.12,
        history_len=1,
    )
    _write_fake_cfd_cns_compare_run(
        history1_root,
        profile_id="unet_strong",
        epochs=10,
        err_nrmse=0.21,
        history_len=1,
    )

    spectral_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "history2-spectral",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.11,
        history_len=2,
    )
    hybrid_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "history2-hybrid",
        profile_id="hybrid_resnet_cns",
        epochs=10,
        err_nrmse=0.19,
        history_len=2,
    )
    fno_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "history2-fno",
        profile_id="fno_base",
        epochs=10,
        err_nrmse=0.23,
        history_len=2,
    )
    unet_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "history2-unet",
        profile_id="unet_strong",
        epochs=10,
        err_nrmse=0.31,
        history_len=2,
    )

    json_path, csv_path, payload = write_history_delta_compare(
        output_root=tmp_path / "out",
        epoch_label="10ep",
        fresh_run_root=history1_root,
        fresh_profile_ids=[
            "spectral_resnet_bottleneck_base",
            "hybrid_resnet_cns",
            "fno_base",
            "unet_strong",
        ],
        reference_rows=[
            {
                "run_root": str(spectral_root),
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/spectral.md",
            },
            {
                "run_root": str(hybrid_root),
                "profile_id": "hybrid_resnet_cns",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/hybrid.md",
            },
            {
                "run_root": str(fno_root),
                "profile_id": "fno_base",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/fno.md",
            },
            {
                "run_root": str(unet_root),
                "profile_id": "unet_strong",
                "epochs": 10,
                "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                "split_counts": {"train": 512, "val": 64, "test": 64},
                "max_windows_per_trajectory": 8,
                "history_len": 2,
                "training_loss": "mse",
                "batch_size": 4,
                "metric_family": [
                    "err_RMSE",
                    "err_nRMSE",
                    "relative_l2",
                    "fRMSE_low",
                    "fRMSE_mid",
                    "fRMSE_high",
                ],
                "source_document": "docs/unet.md",
            },
        ],
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["fixed_contract"]["dataset_file"] == "/tmp/fake-cfd-cns.hdf5"
    assert payload["allowed_contract_delta"] == {
        "delta_kind": "history_len_only",
        "reference_history_len": 2,
        "fresh_history_len": 1,
        "reference_sample_contract": "concat u[t-2:t] -> u[t]",
        "fresh_sample_contract": "concat u[t-1:t] -> u[t]",
        "reference_input_channels": 8,
        "fresh_input_channels": 4,
        "target_channels": 4,
    }
    assert [item["profile_id"] for item in payload["fresh_profile_results"]] == [
        "spectral_resnet_bottleneck_base",
        "hybrid_resnet_cns",
        "fno_base",
        "unet_strong",
    ]
    assert [item["profile_id"] for item in payload["reference_profile_results"]] == [
        "spectral_resnet_bottleneck_base",
        "hybrid_resnet_cns",
        "fno_base",
        "unet_strong",
    ]
    assert payload["cross_run_gallery_blocked"] is None
    assert (tmp_path / "out" / "compare_10ep_sample0.png").exists()
    assert (tmp_path / "out" / "compare_10ep_sample0_error.png").exists()


def test_history1_cross_run_compare_rejects_fixed_contract_mismatch(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_history_delta_compare

    history1_root = tmp_path / "history1"
    _write_fake_cfd_cns_compare_run(
        history1_root,
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.08,
        history_len=1,
        batch_size=2,
    )

    spectral_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "history2-spectral",
        profile_id="spectral_resnet_bottleneck_base",
        epochs=10,
        err_nrmse=0.11,
        history_len=2,
        batch_size=4,
    )

    try:
        write_history_delta_compare(
            output_root=tmp_path / "out",
            epoch_label="10ep",
            fresh_run_root=history1_root,
            fresh_profile_ids=["spectral_resnet_bottleneck_base"],
            reference_rows=[
                {
                    "run_root": str(spectral_root),
                    "profile_id": "spectral_resnet_bottleneck_base",
                    "epochs": 10,
                    "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                    "split_counts": {"train": 512, "val": 64, "test": 64},
                    "max_windows_per_trajectory": 8,
                    "history_len": 2,
                    "training_loss": "mse",
                    "batch_size": 4,
                    "metric_family": [
                        "err_RMSE",
                        "err_nRMSE",
                        "relative_l2",
                        "fRMSE_low",
                        "fRMSE_mid",
                        "fRMSE_high",
                    ],
                    "source_document": "docs/spectral.md",
                }
            ],
        )
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("history compare must reject fixed-contract mismatches outside the history delta")


def test_history1_cross_run_compare_rejects_hybrid_base_proxy_anchor(tmp_path):
    from scripts.studies.pdebench_image128.reporting import write_history_delta_compare

    history1_root = tmp_path / "history1"
    _write_fake_cfd_cns_compare_run(
        history1_root,
        profile_id="hybrid_resnet_cns",
        epochs=40,
        err_nrmse=0.10,
        history_len=1,
    )

    hybrid_base_root = _write_fake_cfd_cns_compare_run(
        tmp_path / "history2-hybrid-base",
        profile_id="hybrid_resnet_base",
        epochs=40,
        err_nrmse=0.15,
        history_len=2,
    )

    try:
        write_history_delta_compare(
            output_root=tmp_path / "out",
            epoch_label="40ep",
            fresh_run_root=history1_root,
            fresh_profile_ids=["hybrid_resnet_cns"],
            reference_rows=[
                {
                    "run_root": str(hybrid_base_root),
                    "profile_id": "hybrid_resnet_base",
                    "epochs": 40,
                    "dataset_file": "/tmp/fake-cfd-cns.hdf5",
                    "split_counts": {"train": 512, "val": 64, "test": 64},
                    "max_windows_per_trajectory": 8,
                    "history_len": 2,
                    "training_loss": "mse",
                    "batch_size": 4,
                    "metric_family": [
                        "err_RMSE",
                        "err_nRMSE",
                        "relative_l2",
                        "fRMSE_low",
                        "fRMSE_mid",
                        "fRMSE_high",
                    ],
                    "source_document": "docs/hybrid-base.md",
                }
            ],
        )
    except ValueError as exc:
        assert "profile_id" in str(exc)
        assert "hybrid_resnet_base" in str(exc)
    else:
        raise AssertionError("history compare must reject hybrid_resnet_base as a proxy anchor")


def test_validate_darcy_benchmark_budget_requires_full_split_and_primary_profiles():
    from scripts.studies.pdebench_image128.run_config import validate_darcy_run_budget

    valid = validate_darcy_run_budget(
        {
            "task_id": "darcy",
            "mode": "benchmark",
            "train_count": 8000,
            "val_count": 1000,
            "test_count": 1000,
            "primary_profiles": ["hybrid_resnet_base", "fno_base", "unet_strong"],
            "training_seed": 20260420,
            "loss": "relative_l2",
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_lr": 1e-5,
            "plateau_threshold": 0.0,
            "batch_size": 8,
            "epochs": 1,
            "precision": "float32",
            "device": "cpu",
            "num_workers": 0,
        }
    )
    assert valid["primary_profiles"] == ["hybrid_resnet_base", "fno_base", "unet_strong"]
    assert valid["loss"] == "relative_l2"
    assert valid["plateau_min_lr"] == 1e-5

    invalid_floor = dict(valid)
    invalid_floor["plateau_min_lr"] = 1e-4
    try:
        validate_darcy_run_budget(invalid_floor)
    except ValueError as exc:
        assert "plateau_min_lr" in str(exc)
    else:
        raise AssertionError("Darcy benchmark budget must reject scheduler floors above 1e-5")

    invalid = dict(valid)
    invalid["train_count"] = 512
    try:
        validate_darcy_run_budget(invalid)
    except ValueError as exc:
        assert "full train split" in str(exc)
    else:
        raise AssertionError("benchmark budget must reject capped training counts")


def test_darcy_readiness_runner_writes_required_artifacts(tmp_path):
    from scripts.studies.pdebench_image128.darcy import run_darcy

    data_root = tmp_path / "data"
    data_file = _write_tiny_darcy(data_root / "darcy" / "2D_DarcyFlow_beta1.0_Train.hdf5")
    output_root = tmp_path / "out"

    exit_code = run_darcy(
        task_id="darcy",
        mode="readiness",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["unet_tiny_smoke"],
        epochs=1,
        batch_size=2,
        max_train_samples=4,
        max_val_samples=2,
        max_test_samples=2,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=["--task", "darcy", "--mode", "readiness"],
    )

    assert exit_code == 0
    required = [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest.json",
        "normalization_stats_input.json",
        "normalization_stats_target.json",
        "model_profile_unet_tiny_smoke.json",
        "metrics_unet_tiny_smoke.json",
        "comparison_unet_tiny_smoke_sample0.png",
        "comparison_unet_tiny_smoke_sample0.npz",
        "comparison_summary.json",
        "comparison_summary.csv",
        "literature_context.json",
        "invocation.json",
        "invocation.sh",
    ]
    for name in required:
        assert (output_root / name).exists(), name

    summary = json.loads((output_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_scope"] == "smoke_feasibility_only"
    assert summary["performance_assessment_complete"] is False
    assert summary["distributed_enabled"] is False
    assert summary["distributed_world_size"] == 1
    metrics = json.loads((output_root / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert metrics["distributed_enabled"] is False
    assert metrics["distributed_world_size"] == 1
    with (output_root / "comparison_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["profile_id"] == "unet_tiny_smoke"


def test_cfd_cns_readiness_runner_writes_history_window_artifacts(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import run_cfd_cns

    data_root = tmp_path / "data"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    output_root = tmp_path / "out"

    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="readiness",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["unet_tiny_smoke"],
        history_len=2,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=["--task", "2d_cfd_cns", "--mode", "readiness"],
    )

    assert exit_code == 0
    required = [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest.json",
        "normalization_stats_state.json",
        "model_profile_unet_tiny_smoke.json",
        "metrics_unet_tiny_smoke.json",
        "comparison_unet_tiny_smoke_sample0.png",
        "comparison_unet_tiny_smoke_sample0.npz",
        "comparison_summary.json",
        "comparison_summary.csv",
        "invocation.json",
        "invocation.sh",
    ]
    for name in required:
        assert (output_root / name).exists(), name

    manifest = json.loads((output_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["history_len"] == 2
    assert manifest["field_order"] == ["density", "Vx", "Vy", "pressure"]
    metrics = json.loads((output_root / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert metrics["training_loss"] == "mse"
    assert metrics["horizon"] == "one_step"
    assert metrics["physics_regularization_enabled"] is False
    assert metrics["physics_loss_terms"] == []
    assert metrics["train_split_eval"]["split_name"] == "train"
    assert metrics["train_split_eval"]["horizon"] == "one_step"
    assert metrics["train_split_eval"]["num_eval_samples"] == 2
    assert "relative_l2" in metrics["train_split_eval"]
    assert "fRMSE_high" in metrics["train_split_eval"]
    summary = json.loads((output_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["task_id"] == "2d_cfd_cns"
    assert summary["history_len"] == 2
    assert summary["evidence_scope"] == "smoke_feasibility_only"
    assert summary["distributed_enabled"] is False
    assert summary["distributed_world_size"] == 1
    assert "fRMSE_high" in summary["profile_results"][0]
    assert metrics["distributed_enabled"] is False
    assert metrics["distributed_world_size"] == 1


def test_cfd_cns_inspect_runner_writes_split_manifest(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import run_cfd_cns

    data_root = tmp_path / "data"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    output_root = tmp_path / "inspect"

    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="inspect",
        data_root=data_root,
        output_root=output_root,
        profile_ids=None,
        history_len=2,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=["--task", "2d_cfd_cns", "--mode", "inspect", "--history-len", "2"],
    )

    assert exit_code == 0
    required = [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest.json",
        "invocation.json",
        "invocation.sh",
    ]
    for name in required:
        assert (output_root / name).exists(), name

    split_manifest = json.loads((output_root / "split_manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["history_len"] == 2
    assert split_manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert split_manifest["max_windows_per_trajectory"] == 1
    assert split_manifest["run_mode"] == "inspect"
    assert split_manifest["full_split_counts"] == {"train": 3, "val": 1, "test": 1}


def test_cfd_cns_readiness_runner_artifacts_history1_contract(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import run_cfd_cns
    from scripts.studies.pdebench_image128.data import MultiFieldHistoryWindowDataset

    data_root = tmp_path / "data"
    data_file = _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    output_root = tmp_path / "out"

    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="readiness",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["unet_tiny_smoke"],
        history_len=1,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=["--task", "2d_cfd_cns", "--mode", "readiness", "--history-len", "1"],
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["history_len"] == 1
    assert manifest["sample_contract"] == "concat u[t-1:t] -> u[t]"

    summary = json.loads((output_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["task_id"] == "2d_cfd_cns"
    assert summary["history_len"] == 1
    assert summary["evidence_scope"] == "smoke_feasibility_only"

    dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=("density", "Vx", "Vy", "pressure"),
        trajectory_ids=[0],
        axis_order="NTHW",
        history_len=1,
        max_windows_per_trajectory=1,
    )
    sample = dataset[0]
    assert tuple(sample["input"].shape) == (4, 8, 8)
    assert tuple(sample["target"].shape) == (4, 8, 8)
    assert sample["input_time_indices"] == [0]
    assert sample["target_time_index"] == 1


def test_cfd_cns_readiness_runner_records_physics_regularization_metadata(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import PhysicsRegularizationConfig, run_cfd_cns

    data_root = tmp_path / "data"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    output_root = tmp_path / "out"

    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="readiness",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["unet_tiny_smoke"],
        history_len=2,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        physics_config=PhysicsRegularizationConfig(
            enabled=True,
            positivity_weight=1.0,
            continuity_weight=0.5,
            global_mass_weight=0.25,
        ),
        raw_argv=["--task", "2d_cfd_cns", "--mode", "readiness", "--physics-regularization", "on"],
    )

    assert exit_code == 0
    metrics = json.loads((output_root / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert metrics["physics_regularization_enabled"] is True
    assert metrics["physics_loss_terms"] == ["positivity", "continuity", "global_mass"]
    assert metrics["physics_loss_weights"] == {
        "positivity": 1.0,
        "continuity": 0.5,
        "global_mass": 0.25,
    }
    assert "physics_last_epoch" in metrics


def test_cfd_cns_passes_task_metadata_into_model_builder(tmp_path, monkeypatch):
    from scripts.studies.pdebench_image128 import cfd_cns
    from scripts.studies.pdebench_image128.data import (
        MultiFieldHistoryWindowDataset,
        inspect_cfd_cns_hdf5,
    )
    from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats

    data_file = _write_tiny_cfd_cns(
        tmp_path / "data" / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    metadata = inspect_cfd_cns_hdf5(data_file, history_len=2)
    state_stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=metadata["field_order"],
        axis_order=metadata["field_axis_order"],
        train_trajectory_ids=[0, 1],
    )
    train_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[0, 1],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )
    eval_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[2],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )

    class TinyRecorderModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(8, 4, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    seen: dict[str, object] = {}

    def fake_build_model_from_profile(profile, *, in_channels, out_channels, spatial_shape, task_metadata=None):
        seen["task_metadata"] = task_metadata
        return TinyRecorderModel()

    monkeypatch.setattr(cfd_cns, "build_model_from_profile", fake_build_model_from_profile)

    result = cfd_cns._run_profile(
        profile_id="unet_tiny_smoke",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        state_stats=state_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="runner-metadata-test",
        metadata=metadata,
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        device_name="cpu",
        num_workers=0,
        physics_config=cfd_cns.PhysicsRegularizationConfig(),
    )

    assert result["status"] == "completed"
    assert seen["task_metadata"] is not None
    assert seen["task_metadata"]["task_id"] == "2d_cfd_cns"
    assert seen["task_metadata"]["history_len"] == 2
    assert seen["task_metadata"]["field_order"] == ["density", "Vx", "Vy", "pressure"]
    assert seen["task_metadata"]["dx"] > 0.0
    assert seen["task_metadata"]["dy"] > 0.0
    assert seen["task_metadata"]["dt"] > 0.0


def test_cfd_cns_gnot_profile_uses_paper_default_training_recipe(tmp_path, monkeypatch):
    from scripts.studies.pdebench_image128 import cfd_cns
    from scripts.studies.pdebench_image128.data import (
        MultiFieldHistoryWindowDataset,
        inspect_cfd_cns_hdf5,
    )
    from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats

    data_file = _write_tiny_cfd_cns(
        tmp_path / "data" / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    metadata = inspect_cfd_cns_hdf5(data_file, history_len=2)
    state_stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=metadata["field_order"],
        axis_order=metadata["field_axis_order"],
        train_trajectory_ids=[0, 1],
    )
    train_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[0, 1],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )
    eval_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[2],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )

    class TinyRecorderModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(8, 4, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    monkeypatch.setattr(
        cfd_cns,
        "build_model_from_profile",
        lambda *args, **kwargs: TinyRecorderModel(),
    )

    result = cfd_cns._run_profile(
        profile_id="gnot_cns_base",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        state_stats=state_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="runner-gnot-training-recipe-test",
        metadata=metadata,
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        device_name="cpu",
        num_workers=0,
        physics_config=cfd_cns.PhysicsRegularizationConfig(),
    )

    assert result["status"] == "completed"
    metrics = json.loads((tmp_path / "out" / "metrics_gnot_cns_base.json").read_text(encoding="utf-8"))
    assert metrics["training_loss"] == "relative_l2"
    assert metrics["optimizer"] == "AdamW"
    assert metrics["scheduler"] == "OneCycleLR"
    assert metrics["learning_rate"] == 1e-3
    assert metrics["weight_decay"] == 5e-5
    assert metrics["model_profile"]["profile_config"]["gnot_hidden"] == 128


def test_cfd_cns_author_ffno_profile_uses_equal_footing_training_recipe(tmp_path, monkeypatch):
    from scripts.studies.pdebench_image128 import cfd_cns
    from scripts.studies.pdebench_image128.data import (
        MultiFieldHistoryWindowDataset,
        inspect_cfd_cns_hdf5,
    )
    from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats

    data_file = _write_tiny_cfd_cns(
        tmp_path / "data" / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    metadata = inspect_cfd_cns_hdf5(data_file, history_len=2)
    state_stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=metadata["field_order"],
        axis_order=metadata["field_axis_order"],
        train_trajectory_ids=[0, 1],
    )
    train_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[0, 1],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )
    eval_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[2],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )

    class TinyRecorderModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(8, 4, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    monkeypatch.setattr(
        cfd_cns,
        "build_model_from_profile",
        lambda *args, **kwargs: TinyRecorderModel(),
    )

    result = cfd_cns._run_profile(
        profile_id="author_ffno_cns_base",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        state_stats=state_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="runner-author-ffno-training-recipe-test",
        metadata=metadata,
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        device_name="cpu",
        num_workers=0,
        physics_config=cfd_cns.PhysicsRegularizationConfig(),
    )

    assert result["status"] == "completed"
    metrics = json.loads((tmp_path / "out" / "metrics_author_ffno_cns_base.json").read_text(encoding="utf-8"))
    assert metrics["training_loss"] == "mse"
    assert metrics["optimizer"] == "Adam"
    assert metrics["scheduler"] == "ReduceLROnPlateau"
    assert metrics["learning_rate"] == 2e-4
    assert metrics["weight_decay"] == 0.0


def test_darcy_worker_runtime_skips_rank_zero_eval_and_artifacts(tmp_path, monkeypatch):
    _stub_skimage(monkeypatch)
    _stub_pdebench_models_module(monkeypatch)
    from scripts.studies.pdebench_image128 import darcy
    from scripts.studies.pdebench_image128.data import DarcyStaticOperatorDataset
    from scripts.studies.pdebench_image128.normalization import compute_static_operator_stats

    data_file = _write_tiny_darcy(tmp_path / "data" / "darcy" / "2D_DarcyFlow_beta1.0_Train.hdf5")
    input_stats, target_stats = compute_static_operator_stats(
        data_file=data_file,
        input_dataset="nu",
        target_dataset="tensor",
        train_indices=[0, 1, 2, 3],
    )
    train_dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        sample_indices=[0, 1, 2, 3],
        input_stats=input_stats,
        target_stats=target_stats,
    )
    eval_dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        sample_indices=[4, 5],
        input_stats=input_stats,
        target_stats=target_stats,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(1, 1, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    monkeypatch.setattr(darcy, "build_model_from_profile", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(darcy, "_write_json", lambda path, payload: (_ for _ in ()).throw(AssertionError("worker should not write json")))
    monkeypatch.setattr(
        darcy,
        "_evaluate_loader",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker should not evaluate")),
        raising=False,
    )

    result = darcy._run_profile(
        profile_id="unet_tiny_smoke",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        target_stats=target_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="darcy-worker-runtime-test",
        epochs=1,
        batch_size=2,
        learning_rate=2e-4,
        loss_name="relative_l2",
        scheduler_name="ReduceLROnPlateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_min_lr=1e-5,
        plateau_threshold=0.0,
        device_name="cpu",
        num_workers=0,
        runtime=_WorkerRuntime(
            broadcast_value={
                "profile_id": "unet_tiny_smoke",
                "status": "completed",
                "err_RMSE": 0.1,
                "err_nRMSE": 0.1,
                "relative_l2": 0.1,
                "parameter_count": 2,
            }
        ),
    )

    assert result["status"] == "completed"
    assert not (tmp_path / "out" / "metrics_unet_tiny_smoke.json").exists()


def test_cfd_cns_worker_runtime_skips_rank_zero_eval_and_artifacts(tmp_path, monkeypatch):
    _stub_skimage(monkeypatch)
    _stub_pdebench_models_module(monkeypatch)
    from scripts.studies.pdebench_image128 import cfd_cns
    from scripts.studies.pdebench_image128.data import MultiFieldHistoryWindowDataset, inspect_cfd_cns_hdf5
    from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats

    data_file = _write_tiny_cfd_cns(
        tmp_path / "data" / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    metadata = inspect_cfd_cns_hdf5(data_file, history_len=2)
    state_stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=metadata["field_order"],
        axis_order=metadata["field_axis_order"],
        train_trajectory_ids=[0, 1],
    )
    train_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[0, 1],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )
    eval_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[2],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(8, 4, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    monkeypatch.setattr(cfd_cns, "build_model_from_profile", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(cfd_cns, "_write_json", lambda path, payload: (_ for _ in ()).throw(AssertionError("worker should not write json")))
    monkeypatch.setattr(
        cfd_cns,
        "_evaluate_loader",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker should not evaluate")),
    )

    result = cfd_cns._run_profile(
        profile_id="unet_tiny_smoke",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        state_stats=state_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="cfd-worker-runtime-test",
        metadata=metadata,
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        device_name="cpu",
        num_workers=0,
        physics_config=cfd_cns.PhysicsRegularizationConfig(),
        runtime=_WorkerRuntime(
            broadcast_value={
                "profile_id": "unet_tiny_smoke",
                "status": "completed",
                "err_RMSE": 0.1,
                "err_nRMSE": 0.1,
                "relative_l2": 0.1,
                "fRMSE_low": 0.1,
                "fRMSE_mid": 0.1,
                "fRMSE_high": 0.1,
                "parameter_count": 2,
            }
        ),
    )

    assert result["status"] == "completed"
    assert not (tmp_path / "out" / "metrics_unet_tiny_smoke.json").exists()


def test_darcy_distributed_train_batches_counts_global_batches_once_per_epoch(tmp_path, monkeypatch):
    _stub_skimage(monkeypatch)
    _stub_pdebench_models_module(monkeypatch)
    from scripts.studies.pdebench_image128 import darcy
    from scripts.studies.pdebench_image128.data import DarcyStaticOperatorDataset
    from scripts.studies.pdebench_image128.normalization import compute_static_operator_stats

    data_file = _write_tiny_darcy(tmp_path / "data" / "darcy" / "2D_DarcyFlow_beta1.0_Train.hdf5")
    input_stats, target_stats = compute_static_operator_stats(
        data_file=data_file,
        input_dataset="nu",
        target_dataset="tensor",
        train_indices=[0, 1, 2, 3],
    )
    train_dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        sample_indices=[0, 1, 2, 3],
        input_stats=input_stats,
        target_stats=target_stats,
    )
    eval_dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        sample_indices=[4, 5],
        input_stats=input_stats,
        target_stats=target_stats,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(1, 1, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    monkeypatch.setattr(darcy, "build_model_from_profile", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(darcy, "DistributedDataParallel", lambda model, **kwargs: model)

    result = darcy._run_profile(
        profile_id="unet_tiny_smoke",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        target_stats=target_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="darcy-ddp-train-batches-test",
        epochs=2,
        batch_size=2,
        learning_rate=2e-4,
        loss_name="relative_l2",
        scheduler_name="ReduceLROnPlateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_min_lr=1e-5,
        plateau_threshold=0.0,
        device_name="cpu",
        num_workers=0,
        runtime=_DistributedRankZeroRuntime(world_size=2),
    )

    assert result["status"] == "completed"
    metrics = json.loads((tmp_path / "out" / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert metrics["distributed_enabled"] is True
    assert metrics["distributed_world_size"] == 2
    assert metrics["train_batches"] == 8


def test_cfd_cns_distributed_train_batches_counts_global_batches_once_per_epoch(tmp_path, monkeypatch):
    _stub_skimage(monkeypatch)
    _stub_pdebench_models_module(monkeypatch)
    from scripts.studies.pdebench_image128 import cfd_cns
    from scripts.studies.pdebench_image128.data import MultiFieldHistoryWindowDataset, inspect_cfd_cns_hdf5
    from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats

    data_file = _write_tiny_cfd_cns(
        tmp_path / "data" / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    metadata = inspect_cfd_cns_hdf5(data_file, history_len=2)
    state_stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=metadata["field_order"],
        axis_order=metadata["field_axis_order"],
        train_trajectory_ids=[0, 1],
    )
    train_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[0, 1],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )
    eval_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=[2],
        axis_order=metadata["field_axis_order"],
        history_len=2,
        normalization=state_stats,
        max_windows_per_trajectory=1,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(8, 4, kernel_size=1)

        def forward(self, x):
            return self.proj(x)

    monkeypatch.setattr(cfd_cns, "build_model_from_profile", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(cfd_cns, "DistributedDataParallel", lambda model, **kwargs: model)

    result = cfd_cns._run_profile(
        profile_id="unet_tiny_smoke",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        state_stats=state_stats,
        spatial_shape=(8, 8),
        output_root=tmp_path / "out",
        run_id="cfd-ddp-train-batches-test",
        metadata=metadata,
        epochs=2,
        batch_size=1,
        learning_rate=2e-4,
        device_name="cpu",
        num_workers=0,
        physics_config=cfd_cns.PhysicsRegularizationConfig(),
        runtime=_DistributedRankZeroRuntime(world_size=2),
    )

    assert result["status"] == "completed"
    metrics = json.loads((tmp_path / "out" / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert metrics["distributed_enabled"] is True
    assert metrics["distributed_world_size"] == 2
    assert metrics["train_batches"] == 8


def test_darcy_relative_l2_training_loss_is_sample_mean():
    import torch

    from scripts.studies.pdebench_image128.darcy import _relative_l2_sample_mean_loss

    prediction = torch.tensor(
        [
            [[[2.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 3.0], [0.0, 0.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 0.0]]],
        ]
    )

    expected = torch.tensor([(1.0 / 1.0), (2.0 / 1.0)]).mean()
    assert torch.isclose(_relative_l2_sample_mean_loss(prediction, target), expected)


def test_image128_cli_keeps_preflight_default_and_supports_darcy_readiness(tmp_path):
    import subprocess
    import sys

    data_root = tmp_path / "data"
    _write_tiny_darcy(data_root / "darcy" / "2D_DarcyFlow_beta1.0_Train.hdf5")

    preflight_root = tmp_path / "preflight"
    markdown_path = tmp_path / "preflight.md"
    preflight = subprocess.run(
        [
            sys.executable,
            "scripts/studies/run_pdebench_image128_suite.py",
            "--data-root",
            str(data_root),
            "--output-root",
            str(preflight_root),
            "--markdown-path",
            str(markdown_path),
            "--no-sha256",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert (preflight_root / "pdebench_image128_suite_preflight.json").exists()

    readiness_root = tmp_path / "readiness"
    readiness = subprocess.run(
        [
            sys.executable,
            "scripts/studies/run_pdebench_image128_suite.py",
            "--task",
            "darcy",
            "--mode",
            "readiness",
            "--data-root",
            str(data_root),
            "--output-root",
            str(readiness_root),
            "--profiles",
            "unet_tiny_smoke",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "2",
            "--max-test-samples",
            "2",
            "--device",
            "cpu",
            "--allow-existing-output-root",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert readiness.returncode == 0, readiness.stderr
    assert (readiness_root / "comparison_summary.json").exists()
    metrics = json.loads((readiness_root / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert metrics["training_loss"] == "relative_l2"

    cfd_root = tmp_path / "cfd_cns_readiness"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )
    cfd = subprocess.run(
        [
            sys.executable,
            "scripts/studies/run_pdebench_image128_suite.py",
            "--task",
            "2d_cfd_cns",
            "--mode",
            "readiness",
            "--data-root",
            str(data_root),
            "--output-root",
            str(cfd_root),
            "--profiles",
            "unet_tiny_smoke",
            "--history-len",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-train-trajectories",
            "2",
            "--max-val-trajectories",
            "1",
            "--max-test-trajectories",
            "1",
            "--max-windows-per-trajectory",
            "1",
            "--device",
            "cpu",
            "--physics-regularization",
            "on",
            "--physics-loss-weights",
            "pos=1.0,cont=0.5,mass=0.25",
            "--allow-existing-output-root",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert cfd.returncode == 0, cfd.stderr
    assert (cfd_root / "comparison_summary.json").exists()
    cfd_metrics = json.loads((cfd_root / "metrics_unet_tiny_smoke.json").read_text(encoding="utf-8"))
    assert cfd_metrics["training_loss"] == "mse"
    assert cfd_metrics["physics_regularization_enabled"] is True
    assert cfd_metrics["physics_loss_terms"] == ["positivity", "continuity", "global_mass"]


def test_cfd_cns_pilot_runner_requires_explicit_profiles_and_writes_required_artifacts(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import run_cfd_cns

    data_root = tmp_path / "data"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )

    missing_profiles_root = tmp_path / "pilot_missing_profiles"
    try:
        run_cfd_cns(
            task_id="2d_cfd_cns",
            mode="pilot",
            data_root=data_root,
            output_root=missing_profiles_root,
            history_len=2,
            epochs=1,
            batch_size=2,
            max_train_trajectories=2,
            max_val_trajectories=1,
            max_test_trajectories=1,
            max_windows_per_trajectory=1,
            device="cpu",
            num_workers=0,
            allow_existing_output_root=True,
            raw_argv=["--task", "2d_cfd_cns", "--mode", "pilot"],
        )
    except ValueError as exc:
        assert "profile_ids" in str(exc)
    else:
        raise AssertionError("pilot mode must require explicit profile_ids")

    output_root = tmp_path / "pilot_run"
    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="pilot",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["spectral_resnet_bottleneck_base"],
        history_len=2,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=[
            "--task",
            "2d_cfd_cns",
            "--mode",
            "pilot",
            "--profiles",
            "spectral_resnet_bottleneck_base",
        ],
    )

    assert exit_code == 0
    required = [
        "invocation.json",
        "invocation.sh",
        "dataset_manifest.json",
        "split_manifest.json",
        "comparison_summary.json",
        "comparison_summary.csv",
        "model_profile_spectral_resnet_bottleneck_base.json",
        "metrics_spectral_resnet_bottleneck_base.json",
        "comparison_spectral_resnet_bottleneck_base_sample0.png",
        "comparison_spectral_resnet_bottleneck_base_sample0.npz",
    ]
    for name in required:
        assert (output_root / name).exists(), name

    model_profile = json.loads(
        (output_root / "model_profile_spectral_resnet_bottleneck_base.json").read_text(encoding="utf-8")
    )
    profile_config = model_profile["profile_config"]
    assert profile_config["hybrid_skip_connections"] is True
    assert profile_config["hybrid_skip_style"] == "add"
    assert profile_config["hybrid_upsampler"] == "pixelshuffle"
    assert profile_config["spectral_bottleneck_share_weights"] is True
    assert profile_config["spectral_bottleneck_blocks"] == 6

    summary = json.loads((output_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "pilot"
    assert summary["evidence_scope"] == "capped_decision_support_only"
    assert summary["metric_interpretation"] == "decision_support_not_benchmark_performance"
    assert summary["performance_assessment_complete"] is False


def test_cfd_cns_pilot_runner_writes_modes32_profile_config_with_explicit_mode_values(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import run_cfd_cns

    data_root = tmp_path / "data"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )

    output_root = tmp_path / "pilot_modes32"
    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="pilot",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["spectral_resnet_bottleneck_modes32"],
        history_len=2,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=[
            "--task",
            "2d_cfd_cns",
            "--mode",
            "pilot",
            "--profiles",
            "spectral_resnet_bottleneck_modes32",
        ],
    )

    assert exit_code == 0
    model_profile = json.loads(
        (output_root / "model_profile_spectral_resnet_bottleneck_modes32.json").read_text(encoding="utf-8")
    )
    profile_config = model_profile["profile_config"]
    assert profile_config["fno_modes"] == 32
    assert profile_config["spectral_bottleneck_modes"] == 32


def test_cfd_cns_pilot_runner_writes_modes24_profile_config_with_explicit_mode_values(tmp_path):
    from scripts.studies.pdebench_image128.cfd_cns import run_cfd_cns

    data_root = tmp_path / "data"
    _write_tiny_cfd_cns(
        data_root / "2d_cfd_cns" / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    )

    output_root = tmp_path / "pilot_modes24"
    exit_code = run_cfd_cns(
        task_id="2d_cfd_cns",
        mode="pilot",
        data_root=data_root,
        output_root=output_root,
        profile_ids=["spectral_resnet_bottleneck_modes24"],
        history_len=2,
        epochs=1,
        batch_size=2,
        max_train_trajectories=2,
        max_val_trajectories=1,
        max_test_trajectories=1,
        max_windows_per_trajectory=1,
        device="cpu",
        num_workers=0,
        allow_existing_output_root=True,
        raw_argv=[
            "--task",
            "2d_cfd_cns",
            "--mode",
            "pilot",
            "--profiles",
            "spectral_resnet_bottleneck_modes24",
        ],
    )

    assert exit_code == 0
    model_profile = json.loads(
        (output_root / "model_profile_spectral_resnet_bottleneck_modes24.json").read_text(encoding="utf-8")
    )
    profile_config = model_profile["profile_config"]
    assert profile_config["fno_modes"] == 24
    assert profile_config["spectral_bottleneck_modes"] == 24

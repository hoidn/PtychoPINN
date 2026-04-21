import csv
import json
from pathlib import Path

import h5py
import numpy as np


def _write_tiny_darcy(path: Path, *, n: int = 12) -> Path:
    rng = np.random.default_rng(7)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["beta"] = 1.0
        handle.create_dataset("nu", data=rng.normal(size=(n, 8, 8)).astype(np.float32))
        handle.create_dataset("tensor", data=rng.normal(size=(n, 1, 8, 8)).astype(np.float32))
    return path


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
    summary = json.loads((output_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["task_id"] == "2d_cfd_cns"
    assert summary["history_len"] == 2
    assert summary["evidence_scope"] == "smoke_feasibility_only"
    assert "fRMSE_high" in summary["profile_results"][0]


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

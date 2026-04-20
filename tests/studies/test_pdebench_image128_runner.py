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
            "loss": "mae",
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_lr": 1e-4,
            "plateau_threshold": 0.0,
            "batch_size": 8,
            "epochs": 1,
            "precision": "float32",
            "device": "cpu",
            "num_workers": 0,
        }
    )
    assert valid["primary_profiles"] == ["hybrid_resnet_base", "fno_base", "unet_strong"]

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

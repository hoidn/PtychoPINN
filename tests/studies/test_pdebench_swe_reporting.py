import csv
import json
from pathlib import Path

import pytest


def _write_metrics(root: Path, profile_id: str, *, run_id="run-a", test_nrmse=1.0) -> None:
    profile_root = root / "runs" / profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "pid": 123,
        "profile_id": profile_id,
        "data_file": "/data/2D_rdb_NA_NA.h5",
        "split_manifest_run": "split-a",
        "normalization_stats": "norm-a",
        "horizon": "one_step",
        "metric_units": "denormalized_physical_units",
        "eval": {
            "val": {"err_RMSE": test_nrmse + 0.1, "err_nRMSE": test_nrmse + 0.1},
            "test": {"err_RMSE": test_nrmse, "err_nRMSE": test_nrmse},
        },
        "runtime_sec": 1.5,
        "peak_cuda_memory_bytes": None,
        "model_description": {"parameter_count": 10, "profile_config": {"profile_id": profile_id}},
    }
    (profile_root / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_blocker(root: Path, profile_id: str, *, run_id="run-a") -> None:
    profile_root = root / "runs" / profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "pid": 123,
        "profile_id": profile_id,
        "reason": "model_dependency_unavailable",
        "message": "synthetic blocker",
    }
    (profile_root / "blocker.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collate_comparison_writes_deterministic_csv_and_gate_inputs(tmp_path):
    from scripts.studies.pdebench_swe.reporting import collate_comparison

    _write_metrics(tmp_path, "hybrid_resnet_base", test_nrmse=1.08)
    _write_metrics(tmp_path, "fno_base", test_nrmse=1.0)
    _write_metrics(tmp_path, "unet_base", test_nrmse=1.2)

    summary = collate_comparison(
        tmp_path,
        primary_profiles=["hybrid_resnet_base", "fno_base", "unet_base"],
        ablation_profiles=[],
        run_id="run-a",
    )

    assert summary["primary_profiles_complete"] is True
    assert summary["baseline_profiles_complete"] is True
    assert summary["hybrid_test_err_nRMSE"] == 1.08
    assert summary["best_baseline_test_err_nRMSE"] == 1.0
    assert summary["relative_gap_vs_best_baseline"] == pytest.approx(0.08)
    assert summary["recommended_decision_input"] == "primary_viable"

    rows = list(csv.DictReader((tmp_path / "comparison_summary.csv").open(newline="", encoding="utf-8")))
    assert [row["profile_id"] for row in rows] == ["hybrid_resnet_base", "fno_base", "unet_base"]
    assert list(rows[0].keys()) == [
        "profile_id",
        "status",
        "test_err_nRMSE",
        "test_err_RMSE",
        "val_err_nRMSE",
        "val_err_RMSE",
        "runtime_sec",
        "parameter_count",
        "blocker_reason",
    ]


def test_collate_comparison_marks_missing_local_baseline_as_incomplete(tmp_path):
    from scripts.studies.pdebench_swe.reporting import collate_comparison

    _write_metrics(tmp_path, "hybrid_resnet_base", test_nrmse=1.0)
    _write_metrics(tmp_path, "unet_base", test_nrmse=1.2)
    _write_blocker(tmp_path, "fno_base")

    summary = collate_comparison(
        tmp_path,
        primary_profiles=["hybrid_resnet_base", "fno_base", "unet_base"],
        ablation_profiles=[],
        run_id="run-a",
    )

    assert summary["primary_profiles_complete"] is False
    assert summary["baseline_profiles_complete"] is False
    assert summary["profiles"]["fno_base"]["status"] == "blocker"
    assert summary["recommended_decision_input"] == "block_local_baseline_incomplete"


def test_collate_comparison_requires_both_local_baselines_even_when_profile_override_omits_one(tmp_path):
    from scripts.studies.pdebench_swe.reporting import collate_comparison

    _write_metrics(tmp_path, "hybrid_resnet_base", test_nrmse=1.0)
    _write_metrics(tmp_path, "unet_base", test_nrmse=1.2)

    summary = collate_comparison(
        tmp_path,
        primary_profiles=["hybrid_resnet_base", "unet_base"],
        ablation_profiles=[],
        run_id="run-a",
    )

    assert summary["primary_profiles_complete"] is True
    assert summary["baseline_profiles_complete"] is False
    assert summary["required_baseline_profiles"] == ["fno_base", "unet_base"]
    assert summary["recommended_decision_input"] == "block_local_baseline_incomplete"


def test_collate_comparison_rejects_mismatched_run_ids(tmp_path):
    from scripts.studies.pdebench_swe.reporting import collate_comparison

    _write_metrics(tmp_path, "hybrid_resnet_base", run_id="run-a", test_nrmse=1.0)
    _write_metrics(tmp_path, "fno_base", run_id="run-b", test_nrmse=1.0)
    _write_metrics(tmp_path, "unet_base", run_id="run-a", test_nrmse=1.0)

    with pytest.raises(ValueError, match="run_id"):
        collate_comparison(
            tmp_path,
            primary_profiles=["hybrid_resnet_base", "fno_base", "unet_base"],
            ablation_profiles=[],
            run_id="run-a",
        )

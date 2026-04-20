import json

import pytest


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_collator_handles_metrics_and_blocker_files(tmp_path):
    from scripts.studies.openfwi_flatvel_a.reporting import collate_comparison

    run_root = tmp_path / "run"
    common = {
        "run_id": "r1",
        "split_manifest": "split_manifest.json",
        "normalization_stats": "normalization_stats.json",
        "metric_units": "denormalized_velocity",
    }
    _write_json(run_root / "runs/hybrid_resnet_smoke/metrics.json", {**common, "MAE": 3.0, "RMSE": 4.0, "SSIM": 0.5, "runtime_sec": 1.0, "model_description": {"parameter_count": 10}})
    _write_json(run_root / "runs/unet_smoke/metrics.json", {**common, "MAE": 2.0, "RMSE": 3.0, "SSIM": 0.6, "runtime_sec": 1.2, "model_description": {"parameter_count": 8}})
    _write_json(run_root / "official_inversionnet_blocker.json", {"run_id": "r1", "status": "blocked", "reason": "official_repo_missing"})

    summary = collate_comparison(run_root, profiles=["hybrid_resnet_smoke", "unet_smoke"], run_id="r1")

    assert summary["hybrid_MAE"] == pytest.approx(3.0)
    assert summary["best_baseline_MAE"] == pytest.approx(2.0)
    assert summary["relative_gap_vs_best_baseline"] == pytest.approx(0.5)
    assert summary["recommended_decision_input"] == "proceed_candidate"
    assert (run_root / "comparison_summary.json").exists()
    assert (run_root / "comparison_summary.csv").exists()


def test_collator_rejects_mismatched_run_ids(tmp_path):
    from scripts.studies.openfwi_flatvel_a.reporting import collate_comparison

    run_root = tmp_path / "run"
    _write_json(run_root / "runs/hybrid_resnet_smoke/metrics.json", {"run_id": "r1", "MAE": 1.0})
    _write_json(run_root / "runs/unet_smoke/metrics.json", {"run_id": "other", "MAE": 1.0})

    with pytest.raises(ValueError, match="run_id"):
        collate_comparison(run_root, profiles=["hybrid_resnet_smoke", "unet_smoke"], run_id="r1")


def test_collator_emits_data_access_block_decision(tmp_path):
    from scripts.studies.openfwi_flatvel_a.reporting import collate_comparison

    run_root = tmp_path / "run"
    _write_json(run_root / "data_access_blocker.json", {"run_id": "r1", "reason": "missing_required_shards"})

    summary = collate_comparison(run_root, profiles=["hybrid_resnet_smoke", "unet_smoke"], run_id="r1")

    assert summary["data_access_complete"] is False
    assert summary["recommended_decision_input"] == "block_data_access"


def test_collator_ignores_stale_data_access_blocker_when_current_metrics_exist(tmp_path):
    from scripts.studies.openfwi_flatvel_a.reporting import collate_comparison

    run_root = tmp_path / "run"
    common = {
        "run_id": "current",
        "split_manifest": "split_manifest.json",
        "normalization_stats": "normalization_stats.json",
        "metric_units": "denormalized_velocity",
    }
    _write_json(run_root / "data_access_blocker.json", {"run_id": "old", "reason": "missing_required_shards"})
    _write_json(run_root / "runs/hybrid_resnet_smoke/metrics.json", {**common, "MAE": 3.0, "RMSE": 4.0, "SSIM": 0.5})
    _write_json(run_root / "runs/unet_smoke/metrics.json", {**common, "MAE": 2.0, "RMSE": 3.0, "SSIM": 0.6})

    summary = collate_comparison(run_root, profiles=["hybrid_resnet_smoke", "unet_smoke"], run_id="current")

    assert summary["data_access_complete"] is True
    assert summary["recommended_decision_input"] == "proceed_candidate"
    assert summary["ignored_stale_data_access_blocker"]["run_id"] == "old"

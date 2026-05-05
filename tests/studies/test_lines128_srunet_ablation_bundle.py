"""Tests for the lines128 SRU-Net branch / objective ablation bundle helper."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_row_artifacts(
    row_dir: Path,
    metrics: dict,
    *,
    with_completion: bool = False,
    with_invocation: bool = True,
) -> None:
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if with_invocation:
        (row_dir / "invocation.json").write_text(json.dumps({"argv": ["stub"]}), encoding="utf-8")
    if with_completion:
        (row_dir / "launcher_completion.json").write_text(
            json.dumps({"evidence_source": "stub_completion"}), encoding="utf-8"
        )


def test_build_ablation_bundle_promotes_baseline_and_collates_fresh_rows(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import (
        ABLATION_ROW_IDS,
        CLAIM_BOUNDARY,
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.10})

    run_root = tmp_path / "fresh_run"
    fresh_metrics = {
        "pinn_hybrid_resnet_encoder_conv_only": {"mae": 0.12},
        "pinn_hybrid_resnet_encoder_spectral_only": {"mae": 0.13},
        "supervised_hybrid_resnet": {"mae": 0.11},
    }
    for model_id, m in fresh_metrics.items():
        _write_row_artifacts(run_root / "runs" / model_id, m, with_completion=True)

    bundle_dir = tmp_path / "bundle"
    manifest = build_ablation_bundle(
        run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
    )

    assert manifest["claim_boundary"] == CLAIM_BOUNDARY
    assert manifest["fixed_contract_id"] == "cdi_lines128_seed3"
    row_ids = [row["model_id"] for row in manifest["rows"]]
    assert row_ids[0] == "pinn_hybrid_resnet"
    assert tuple(row_ids[1:]) == ABLATION_ROW_IDS

    assert (manifest["rows"][0]["evidence_source"]) == "promoted_by_lineage"
    for row in manifest["rows"][1:]:
        assert row["evidence_source"] == "fresh_run"

    metrics_payload = json.loads((bundle_dir / "ablation_metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["claim_boundary"] == CLAIM_BOUNDARY
    assert metrics_payload["metrics_by_model"]["pinn_hybrid_resnet"] == {"mae": 0.10}
    assert metrics_payload["metrics_by_model"]["pinn_hybrid_resnet_encoder_conv_only"] == {"mae": 0.12}
    assert metrics_payload["metrics_by_model"]["supervised_hybrid_resnet"] == {"mae": 0.11}


def test_build_ablation_bundle_records_branch_select_overrides(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import build_ablation_bundle

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.10})

    run_root = tmp_path / "fresh_run"
    for model_id in (
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ):
        _write_row_artifacts(run_root / "runs" / model_id, {"mae": 0.1})

    bundle_dir = tmp_path / "bundle"
    manifest = build_ablation_bundle(
        run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
    )

    rows_by_id = {row["model_id"]: row for row in manifest["rows"]}
    assert rows_by_id["pinn_hybrid_resnet_encoder_conv_only"]["overrides"] == {
        "hybrid_encoder_branch_select": "conv_only"
    }
    assert rows_by_id["pinn_hybrid_resnet_encoder_spectral_only"]["overrides"] == {
        "hybrid_encoder_branch_select": "spectral_only"
    }
    assert rows_by_id["supervised_hybrid_resnet"]["overrides"] == {}
    assert rows_by_id["supervised_hybrid_resnet"]["training_procedure"] == "supervised"


def test_build_ablation_bundle_fails_when_baseline_row_missing(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import build_ablation_bundle

    run_root = tmp_path / "fresh_run"
    for model_id in (
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ):
        _write_row_artifacts(run_root / "runs" / model_id, {"mae": 0.1})

    with pytest.raises(FileNotFoundError, match="pinn_hybrid_resnet"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=tmp_path / "missing_baseline",
            bundle_dir=tmp_path / "bundle",
        )


def test_build_ablation_bundle_fails_when_fresh_row_missing(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import build_ablation_bundle

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.10})

    run_root = tmp_path / "fresh_run"
    # Only two of three present
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_encoder_conv_only", {"mae": 0.1}
    )
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_encoder_spectral_only", {"mae": 0.1}
    )

    with pytest.raises(FileNotFoundError, match="supervised_hybrid_resnet"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=baseline_root,
            bundle_dir=tmp_path / "bundle",
        )

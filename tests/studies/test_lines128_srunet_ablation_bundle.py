"""Tests for the lines128 SRU-Net branch / objective ablation bundle helper."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_row_artifacts(
    row_dir: Path,
    metrics: dict,
    *,
    with_completion: bool = True,
    completion_filename: str = "exit_code_proof.json",
    completion_payload: dict | None = None,
    with_invocation: bool = True,
) -> None:
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if with_invocation:
        (row_dir / "invocation.json").write_text(json.dumps({"argv": ["stub"]}), encoding="utf-8")
    if with_completion:
        payload = completion_payload or {
            "exit_code": 0,
            "invocation_status": "completed",
            "proof_source": "stub_proof",
        }
        (row_dir / completion_filename).write_text(json.dumps(payload), encoding="utf-8")


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

    for row in manifest["rows"]:
        assert row["completion_proof_present"] is True
        assert row["completion_proof_filename"] == "exit_code_proof.json"

    metrics_payload = json.loads((bundle_dir / "ablation_metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["claim_boundary"] == CLAIM_BOUNDARY
    assert metrics_payload["metrics_by_model"]["pinn_hybrid_resnet"] == {"mae": 0.10}
    assert metrics_payload["metrics_by_model"]["pinn_hybrid_resnet_encoder_conv_only"] == {"mae": 0.12}
    assert metrics_payload["metrics_by_model"]["supervised_hybrid_resnet"] == {"mae": 0.11}

    rows_by_id = {row["model_id"]: row for row in metrics_payload["rows"]}
    for model_id in ABLATION_ROW_IDS:
        provenance = rows_by_id[model_id]["row_provenance"]
        assert provenance["completion_proof_present"] is True
        assert provenance["completion_proof_filename"] == "exit_code_proof.json"
        assert rows_by_id[model_id]["completion_proof"]["exit_code"] == 0


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


def test_build_ablation_bundle_fails_when_completion_proof_missing(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import build_ablation_bundle

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.10})

    run_root = tmp_path / "fresh_run"
    # Two rows have completion proof; one is intentionally missing it.
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_encoder_conv_only", {"mae": 0.12}
    )
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_encoder_spectral_only", {"mae": 0.13}
    )
    _write_row_artifacts(
        run_root / "runs" / "supervised_hybrid_resnet",
        {"mae": 0.11},
        with_completion=False,
    )

    with pytest.raises(FileNotFoundError, match="completion-proof"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=baseline_root,
            bundle_dir=tmp_path / "bundle",
        )


def test_build_ablation_bundle_refuses_to_overwrite_existing_outputs(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import (
        BUNDLE_OUTPUT_FILENAMES,
        build_ablation_bundle,
    )

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
    bundle_dir.mkdir(parents=True)
    sentinel = {"sentinel": "prior_bundle_artifact"}
    for name in BUNDLE_OUTPUT_FILENAMES:
        (bundle_dir / name).write_text(json.dumps(sentinel), encoding="utf-8")

    with pytest.raises(FileExistsError, match="append-only"):
        build_ablation_bundle(
            run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
        )

    for name in BUNDLE_OUTPUT_FILENAMES:
        assert json.loads((bundle_dir / name).read_text(encoding="utf-8")) == sentinel


def test_build_ablation_bundle_accepts_legacy_launcher_completion(tmp_path: Path):
    from scripts.studies.lines128_srunet_ablation_bundle import build_ablation_bundle

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.10})

    run_root = tmp_path / "fresh_run"
    for model_id in (
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ):
        _write_row_artifacts(
            run_root / "runs" / model_id,
            {"mae": 0.1},
            completion_filename="launcher_completion.json",
            completion_payload={"evidence_source": "stub_legacy_completion"},
        )

    bundle_dir = tmp_path / "bundle"
    build_ablation_bundle(
        run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
    )

    metrics_payload = json.loads((bundle_dir / "ablation_metrics.json").read_text(encoding="utf-8"))
    rows_by_id = {row["model_id"]: row for row in metrics_payload["rows"]}
    for model_id in (
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ):
        provenance = rows_by_id[model_id]["row_provenance"]
        assert provenance["completion_proof_present"] is True
        assert provenance["completion_proof_filename"] == "launcher_completion.json"

"""Tests for the lines128 SRU-Net ConvNeXt-bottleneck ablation bundle helper."""
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
        (row_dir / "invocation.json").write_text(
            json.dumps({"argv": ["stub"]}), encoding="utf-8"
        )
    if with_completion:
        payload = completion_payload or {
            "exit_code": 0,
            "invocation_status": "completed",
            "proof_source": "stub_proof",
        }
        (row_dir / completion_filename).write_text(json.dumps(payload), encoding="utf-8")


def test_build_bundle_promotes_baseline_and_collates_only_convnext_row(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        ABLATION_ROW_IDS,
        CLAIM_BOUNDARY,
        EVIDENCE_SCOPE,
        FIXED_CONTRACT_ID,
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.10})

    run_root = tmp_path / "fresh_run"
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_convnext_bottleneck",
        {"mae": 0.09},
    )

    bundle_dir = tmp_path / "bundle"
    manifest = build_ablation_bundle(
        run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
    )

    assert manifest["claim_boundary"] == CLAIM_BOUNDARY
    assert manifest["evidence_scope"] == EVIDENCE_SCOPE
    assert manifest["fixed_contract_id"] == FIXED_CONTRACT_ID
    row_ids = [row["model_id"] for row in manifest["rows"]]
    assert row_ids == ["pinn_hybrid_resnet", *ABLATION_ROW_IDS]

    assert manifest["rows"][0]["evidence_source"] == "promoted_by_lineage"
    assert manifest["rows"][0]["architecture_id"] == "hybrid_resnet"
    fresh = manifest["rows"][1]
    assert fresh["evidence_source"] == "fresh_run"
    assert fresh["architecture_id"] == "hybrid_resnet_convnext_bottleneck"
    assert fresh["model_label"] == "Hybrid ResNet (ConvNeXt bottleneck) + PINN"
    assert fresh["overrides"]["changed_factor"] == "bottleneck_block_family_only"

    for row in manifest["rows"]:
        assert row["completion_proof_present"] is True
        assert row["completion_proof_filename"] == "exit_code_proof.json"

    metrics_payload = json.loads(
        (bundle_dir / "ablation_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics_payload["claim_boundary"] == CLAIM_BOUNDARY
    assert metrics_payload["metrics_by_model"]["pinn_hybrid_resnet"] == {"mae": 0.10}
    assert metrics_payload["metrics_by_model"][
        "pinn_hybrid_resnet_convnext_bottleneck"
    ] == {"mae": 0.09}


def test_build_bundle_records_completion_proof_per_row(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.1})

    run_root = tmp_path / "fresh_run"
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_convnext_bottleneck",
        {"mae": 0.09},
    )

    bundle_dir = tmp_path / "bundle"
    build_ablation_bundle(
        run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
    )

    metrics_payload = json.loads(
        (bundle_dir / "ablation_metrics.json").read_text(encoding="utf-8")
    )
    rows_by_id = {row["model_id"]: row for row in metrics_payload["rows"]}
    fresh = rows_by_id["pinn_hybrid_resnet_convnext_bottleneck"]
    assert fresh["row_provenance"]["completion_proof_present"] is True
    assert fresh["row_provenance"]["completion_proof_filename"] == "exit_code_proof.json"
    assert fresh["completion_proof"]["exit_code"] == 0


def test_build_bundle_accepts_legacy_launcher_completion_filename(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(
        baseline_root / "runs" / "pinn_hybrid_resnet",
        {"mae": 0.1},
        completion_filename="launcher_completion.json",
    )

    run_root = tmp_path / "fresh_run"
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_convnext_bottleneck",
        {"mae": 0.09},
        completion_filename="launcher_completion.json",
    )

    bundle_dir = tmp_path / "bundle"
    manifest = build_ablation_bundle(
        run_root=run_root, baseline_root=baseline_root, bundle_dir=bundle_dir
    )
    fresh = manifest["rows"][1]
    assert fresh["completion_proof_filename"] == "launcher_completion.json"


def test_build_bundle_fails_when_baseline_row_missing(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    baseline_root.mkdir(parents=True)

    run_root = tmp_path / "fresh_run"
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_convnext_bottleneck",
        {"mae": 0.09},
    )

    with pytest.raises(FileNotFoundError, match="baseline pinn_hybrid_resnet"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=baseline_root,
            bundle_dir=tmp_path / "bundle",
        )


def test_build_bundle_fails_when_fresh_row_missing(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.1})

    run_root = tmp_path / "fresh_run"
    run_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="fresh row directory"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=baseline_root,
            bundle_dir=tmp_path / "bundle",
        )


def test_build_bundle_fails_when_fresh_row_has_no_completion_proof(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.1})

    run_root = tmp_path / "fresh_run"
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_convnext_bottleneck",
        {"mae": 0.09},
        with_completion=False,
    )

    with pytest.raises(FileNotFoundError, match="completion-proof"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=baseline_root,
            bundle_dir=tmp_path / "bundle",
        )


def test_build_bundle_refuses_to_overwrite_existing_outputs(tmp_path: Path):
    from scripts.studies.lines128_srunet_convnext_bottleneck_ablation import (
        BUNDLE_OUTPUT_FILENAMES,
        build_ablation_bundle,
    )

    baseline_root = tmp_path / "baseline_root"
    _write_row_artifacts(baseline_root / "runs" / "pinn_hybrid_resnet", {"mae": 0.1})

    run_root = tmp_path / "fresh_run"
    _write_row_artifacts(
        run_root / "runs" / "pinn_hybrid_resnet_convnext_bottleneck",
        {"mae": 0.09},
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / BUNDLE_OUTPUT_FILENAMES[0]).write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="append-only"):
        build_ablation_bundle(
            run_root=run_root,
            baseline_root=baseline_root,
            bundle_dir=bundle_dir,
        )

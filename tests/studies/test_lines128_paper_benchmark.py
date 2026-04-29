"""Tests for the lines128 paper benchmark harness preflight layer."""

import json
from pathlib import Path

import pytest


def _write_decision_artifact(path: Path, *, contract_note_status: str = "resolved") -> Path:
    payload = {
        "selected_fno_comparator": "fno_vanilla",
        "seed_policy": {"type": "fixed", "seed": 3},
        "go_no_go": {
            "state": "go_for_harness_preflight_only",
            "full_benchmark_launch_authorized": False,
        },
        "contract_note": {
            "path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md",
            "status": contract_note_status,
        },
        "rows": [
            {
                "model_id": "pinn_hybrid_resnet",
                "model_label": "Hybrid ResNet",
                "architecture": "hybrid_resnet",
                "status": "supported_for_harness",
                "required_for_minimum_subset": True,
            },
            {
                "model_id": "pinn",
                "model_label": "PtychoPINN (CNN)",
                "architecture": "cnn",
                "status": "supported_for_harness",
                "required_for_minimum_subset": True,
            },
            {
                "model_id": "pinn_fno_vanilla",
                "model_label": "FNO Vanilla",
                "architecture": "fno_vanilla",
                "status": "supported_for_harness",
                "required_for_minimum_subset": True,
            },
            {
                "model_id": "pinn_spectral_resnet_bottleneck_net",
                "model_label": "Spectral ResNet Bottleneck",
                "architecture": "spectral_resnet_bottleneck_net",
                "status": "supported_for_harness",
                "required_for_minimum_subset": False,
            },
            {
                "model_id": "pinn_ffno",
                "model_label": "FFNO",
                "architecture": "ffno",
                "status": "row_blocker",
                "required_for_minimum_subset": False,
                "blocker_reason": "example blocker",
            },
        ],
        "fixed_sample_ids": [0, 1],
        "shared_visual_scales": {"amp": {"vmin": 0.0, "vmax": 1.0}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_preflight_requires_decision_artifact(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    with pytest.raises(FileNotFoundError, match="decision artifact"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
        )


def test_preflight_rejects_unresolved_contract_note(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    decision_artifact = _write_decision_artifact(
        tmp_path / "decision.json",
        contract_note_status="unresolved",
    )

    with pytest.raises(ValueError, match="contract note"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )


def test_preflight_emits_validation_bundle_with_blocker_rows(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    result = run_lines128_paper_benchmark_preflight(
        decision_artifact=decision_artifact,
        output_dir=tmp_path / "out",
    )

    metrics_payload = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert result["selected_models"] == [
        "pinn_hybrid_resnet",
        "pinn",
        "pinn_fno_vanilla",
        "pinn_spectral_resnet_bottleneck_net",
    ]
    assert metrics_payload["selected_fno_comparator"] == "fno_vanilla"
    assert metrics_payload["row_statuses"]["pinn_ffno"]["status"] == "row_blocker"
    assert metrics_payload["row_statuses"]["pinn_ffno"]["reason"] == "example blocker"
    assert metrics_payload["visual_collation"]["fixed_sample_ids"] == [0, 1]

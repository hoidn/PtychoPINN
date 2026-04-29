"""Tests for the lines128 paper benchmark harness preflight layer."""

import json
from pathlib import Path

import pytest


def _fixed_contract_payload() -> dict:
    return {
        "N": 128,
        "gridsize": 1,
        "dataset_source": "synthetic_lines",
        "set_phi": True,
        "probe_source": "custom",
        "probe_npz": "datasets/Run1084_recon3_postPC_shrunk_3.npz",
        "probe_scale_mode": "pad_extrapolate",
        "probe_smoothing_sigma": 0.5,
        "probe_mask_diameter": None,
        "nimgs_train": 2,
        "nimgs_test": 2,
        "nphotons": 1e9,
        "seed": 3,
        "torch_epochs": 40,
        "torch_learning_rate": 2e-4,
        "torch_scheduler": "ReduceLROnPlateau",
        "torch_plateau_factor": 0.5,
        "torch_plateau_patience": 2,
        "torch_plateau_min_lr": 1e-4,
        "torch_plateau_threshold": 0.0,
        "torch_loss_mode": "mae",
        "torch_mae_pred_l2_match_target": False,
        "torch_output_mode": "real_imag",
        "fno_modes": 12,
        "fno_width": 32,
        "fno_blocks": 4,
        "fno_cnn_blocks": 2,
    }


def _fixed_contract_provenance_payload() -> dict:
    return {
        key: {
            "confidence": "high",
            "sources": [
                "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md",
                "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md",
            ],
        }
        for key in _fixed_contract_payload()
    }


def _write_decision_artifact(
    path: Path,
    *,
    contract_note_status: str = "resolved",
    contract_note_path: str = "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md",
    fixed_contract: dict | None = None,
    fixed_contract_provenance: dict | None = None,
    seed_policy: dict | None = None,
    go_no_go: dict | None = None,
) -> Path:
    payload = {
        "selected_fno_comparator": "fno_vanilla",
        "seed_policy": seed_policy or {"type": "fixed", "seed": 3},
        "go_no_go": go_no_go or {
            "state": "go_for_harness_preflight_only",
            "full_benchmark_launch_authorized": False,
        },
        "contract_note": {
            "path": contract_note_path,
            "status": contract_note_status,
        },
        "fixed_contract": fixed_contract or _fixed_contract_payload(),
        "fixed_contract_provenance": fixed_contract_provenance or _fixed_contract_provenance_payload(),
        "approved_deviations": [],
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


def test_preflight_rejects_missing_contract_note_file(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    decision_artifact = _write_decision_artifact(
        tmp_path / "decision.json",
        contract_note_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/missing_contract_note.md",
    )

    with pytest.raises(FileNotFoundError, match="contract note"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )


def test_preflight_rejects_missing_fixed_contract(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    payload = json.loads(decision_artifact.read_text(encoding="utf-8"))
    payload.pop("fixed_contract")
    decision_artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed contract"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )


def test_preflight_rejects_seed_policy_that_does_not_match_fixed_contract(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    decision_artifact = _write_decision_artifact(
        tmp_path / "decision.json",
        seed_policy={"type": "fixed", "seed": 7},
    )

    with pytest.raises(ValueError, match="seed policy"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )


def test_preflight_rejects_full_benchmark_go_no_go_state(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    decision_artifact = _write_decision_artifact(
        tmp_path / "decision.json",
        go_no_go={
            "state": "go_for_full_benchmark",
            "full_benchmark_launch_authorized": True,
        },
    )

    with pytest.raises(ValueError, match="go/no-go"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )


def test_preflight_delegates_to_compare_wrapper_and_emits_validation_bundle_with_blocker_rows(
    tmp_path,
    monkeypatch,
):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    captured = {}

    def fake_run_grid_lines_compare(**kwargs):
        captured.update(kwargs)
        return {
            "mode": "preflight_only",
            "selected_models": list(kwargs["models"]),
            "resolved_model_n": dict(kwargs["model_n"]),
            "seed": kwargs["seed"],
            "contract": {
                "N": kwargs["N"],
                "gridsize": kwargs["gridsize"],
                "dataset_source": kwargs["dataset_source"],
                "set_phi": kwargs["set_phi"],
                "probe_source": kwargs["probe_source"],
                "probe_npz": str(kwargs["probe_npz"]),
                "probe_scale_mode": kwargs["probe_scale_mode"],
                "probe_smoothing_sigma": kwargs["probe_smoothing_sigma"],
                "probe_mask_diameter": kwargs.get("probe_mask_diameter"),
                "nimgs_train": kwargs["nimgs_train"],
                "nimgs_test": kwargs["nimgs_test"],
                "nphotons": kwargs["nphotons"],
                "seed": kwargs["seed"],
                "torch_epochs": kwargs["torch_epochs"],
                "torch_learning_rate": kwargs["torch_learning_rate"],
                "torch_scheduler": kwargs["torch_scheduler"],
                "torch_plateau_factor": kwargs["torch_plateau_factor"],
                "torch_plateau_patience": kwargs["torch_plateau_patience"],
                "torch_plateau_min_lr": kwargs["torch_plateau_min_lr"],
                "torch_plateau_threshold": kwargs["torch_plateau_threshold"],
                "torch_loss_mode": kwargs["torch_loss_mode"],
                "torch_mae_pred_l2_match_target": kwargs["torch_mae_pred_l2_match_target"],
                "torch_output_mode": kwargs["torch_output_mode"],
                "fno_modes": kwargs["fno_modes"],
                "fno_width": kwargs["fno_width"],
                "fno_blocks": kwargs["fno_blocks"],
                "fno_cnn_blocks": kwargs["fno_cnn_blocks"],
            },
            "row_plan": [
                {
                    "model_id": model_id,
                    "status": "supported_for_harness",
                    "backend": "tf" if model_id == "pinn" else "torch",
                    "N": kwargs["model_n"].get(model_id, kwargs["N"]),
                }
                for model_id in kwargs["models"]
            ],
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    result = run_lines128_paper_benchmark_preflight(
        decision_artifact=decision_artifact,
        output_dir=tmp_path / "out",
    )

    metrics_payload = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert captured["preflight_only"] is True
    assert captured["seed"] == 3
    assert captured["models"] == (
        "pinn_hybrid_resnet",
        "pinn",
        "pinn_fno_vanilla",
        "pinn_spectral_resnet_bottleneck_net",
    )
    assert result["selected_models"] == [
        "pinn_hybrid_resnet",
        "pinn",
        "pinn_fno_vanilla",
        "pinn_spectral_resnet_bottleneck_net",
    ]
    assert result["compare_preflight"]["mode"] == "preflight_only"
    assert metrics_payload["selected_fno_comparator"] == "fno_vanilla"
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    assert metrics_payload["row_statuses"]["pinn_ffno"]["status"] == "row_blocker"
    assert metrics_payload["row_statuses"]["pinn_ffno"]["reason"] == "example blocker"
    assert metrics_payload["visual_collation"]["fixed_sample_ids"] == [0, 1]


def test_preflight_rejects_compare_preflight_contract_drift(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    def fake_run_grid_lines_compare(**kwargs):
        return {
            "mode": "preflight_only",
            "selected_models": list(kwargs["models"]),
            "resolved_model_n": dict(kwargs["model_n"]),
            "seed": kwargs["seed"],
            "contract": {
                **_fixed_contract_payload(),
                "torch_epochs": 41,
            },
            "row_plan": [
                {
                    "model_id": model_id,
                    "status": "supported_for_harness",
                    "backend": "tf" if model_id == "pinn" else "torch",
                    "N": kwargs["model_n"].get(model_id, kwargs["N"]),
                }
                for model_id in kwargs["models"]
            ],
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")

    with pytest.raises(ValueError, match="contract drift"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )

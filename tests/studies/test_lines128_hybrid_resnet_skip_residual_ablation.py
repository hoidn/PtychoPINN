"""Tests for the Lines128 Hybrid ResNet skip/residual ablation helper."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.studies.lines128_hybrid_resnet_skip_residual_ablation import (
    build_row_runner_config,
    prepare_execution_scaffold,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _baseline_manifest_payload() -> dict:
    return {
        "selected_fno_comparator": "fno_vanilla",
        "seed_policy": {"type": "fixed", "seed": 3},
        "fixed_contract": {
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
        },
    }


def _baseline_metrics_payload() -> dict:
    return {
        "visual_collation": {
            "fixed_sample_ids": [0, 1],
            "shared_visual_scales": {
                "amp": {"vmin": 0.0, "vmax": 1.0},
                "phase": {"vmin": -3.14159, "vmax": 3.14159},
            },
        }
    }


def _baseline_invocation_payload(output_dir: Path) -> dict:
    return {
        "parsed_args": {
            "train_npz": "datasets/N128/gs1/train.npz",
            "test_npz": "datasets/N128/gs1/test.npz",
            "output_dir": str(output_dir),
            "architecture": "hybrid_resnet",
            "training_procedure": "pinn",
            "seed": 3,
            "epochs": 40,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "infer_batch_size": 128,
            "gradient_clip_val": 0.0,
            "gradient_clip_algorithm": "norm",
            "generator_output_mode": "real_imag",
            "N": 128,
            "gridsize": 1,
            "torch_loss_mode": "mae",
            "torch_mae_pred_l2_match_target": False,
            "probe_mask": False,
            "probe_mask_sigma": 1.0,
            "probe_mask_diameter": None,
            "fno_modes": 12,
            "fno_width": 32,
            "fno_blocks": 4,
            "fno_cnn_blocks": 2,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "momentum": 0.9,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "scheduler": "ReduceLROnPlateau",
            "lr_warmup_epochs": 0,
            "lr_min_ratio": 0.1,
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_lr": 1e-4,
            "plateau_threshold": 0.0,
            "hybrid_skip_connections": False,
            "hybrid_downsample_steps": 2,
            "hybrid_downsample_op": "stride_conv",
            "hybrid_encoder_conv_hidden_scale": 2.0,
            "hybrid_encoder_spectral_hidden_scale": 1.0,
            "hybrid_resnet_blocks": 6,
            "hybrid_skip_style": "add",
        }
    }


def test_prepare_execution_scaffold_writes_manifest_surfaces(tmp_path):
    baseline_root = tmp_path / "baseline"
    artifact_root = tmp_path / "ablation"
    summary_path = tmp_path / "summary.md"
    plan_path = Path("docs/plans/plan.md")
    _write_json(baseline_root / "paper_benchmark_manifest.json", _baseline_manifest_payload())
    _write_json(baseline_root / "metrics.json", _baseline_metrics_payload())
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "invocation.json",
        _baseline_invocation_payload(baseline_root),
    )

    result = prepare_execution_scaffold(
        baseline_root=baseline_root,
        artifact_root=artifact_root,
        summary_path=summary_path,
        plan_path=plan_path,
    )

    assert summary_path.exists()
    assert Path(result["execution_manifest_path"]).exists()
    assert Path(result["row_contract_audit_path"]).exists()
    assert Path(result["cross_reference_manifest_path"]).exists()

    execution_manifest = json.loads(Path(result["execution_manifest_path"]).read_text(encoding="utf-8"))
    assert execution_manifest["baseline_source_root"] == str(baseline_root)
    assert execution_manifest["baseline_row_id"] == "pinn_hybrid_resnet"
    assert execution_manifest["mandatory_fresh_rows"] == [
        "pinn_hybrid_resnet_skip_add",
        "pinn_hybrid_resnet_residual_fixed",
        "pinn_hybrid_resnet_skip_add_residual_fixed",
    ]


def test_build_row_runner_config_applies_skip_and_fixed_residual_changes(tmp_path):
    baseline_root = tmp_path / "baseline"
    artifact_root = tmp_path / "ablation"
    summary_path = tmp_path / "summary.md"
    _write_json(baseline_root / "paper_benchmark_manifest.json", _baseline_manifest_payload())
    _write_json(baseline_root / "metrics.json", _baseline_metrics_payload())
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "invocation.json",
        _baseline_invocation_payload(baseline_root),
    )
    prepare_execution_scaffold(
        baseline_root=baseline_root,
        artifact_root=artifact_root,
        summary_path=summary_path,
        plan_path=Path("docs/plans/plan.md"),
    )

    residual_cfg = build_row_runner_config(
        artifact_root=artifact_root,
        row_id="pinn_hybrid_resnet_residual_fixed",
    )
    interaction_cfg = build_row_runner_config(
        artifact_root=artifact_root,
        row_id="pinn_hybrid_resnet_skip_add_residual_fixed",
    )

    assert residual_cfg.hybrid_resnet_bottleneck_layerscale_mode == "fixed"
    assert residual_cfg.hybrid_resnet_bottleneck_layerscale_value == 1.0
    assert residual_cfg.hybrid_skip_connections is False
    assert interaction_cfg.hybrid_skip_connections is True
    assert interaction_cfg.hybrid_skip_style == "add"
    assert interaction_cfg.hybrid_resnet_bottleneck_layerscale_mode == "fixed"

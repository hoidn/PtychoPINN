"""Tests for the Lines128 Hybrid ResNet skip/residual ablation helper."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

import scripts.studies.lines128_hybrid_resnet_skip_residual_ablation as ablation
from scripts.studies.lines128_hybrid_resnet_skip_residual_ablation import (
    build_row_runner_config,
    collate_ablation_bundle,
    prepare_execution_scaffold,
    run_fresh_row,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_numpy_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=np.zeros((1,), dtype=np.float32))


def _write_torch_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.zeros(1)}, path)


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
        "script": "scripts/studies/grid_lines_torch_runner.py",
        "argv": ["--output-dir", str(output_dir / "training_runs" / "pinn_hybrid_resnet")],
        "command": "python scripts/studies/grid_lines_torch_runner.py --output-dir ...",
        "parsed_args": {
            "train_npz": str(output_dir / "datasets" / "N128" / "gs1" / "train.npz"),
            "test_npz": str(output_dir / "datasets" / "N128" / "gs1" / "test.npz"),
            "output_dir": str(output_dir / "training_runs" / "pinn_hybrid_resnet"),
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
        },
        "cwd": str(output_dir),
        "timestamp_utc": "2026-05-01T00:00:00+00:00",
        "pid": 101,
        "extra": {
            "runtime_provenance": {
                "python_executable": "python",
                "python_version": "3.11.9",
                "host": "test-host",
                "torch_version": "2.5.1",
                "cuda_version": "12.1",
                "gpu": "test-gpu",
            },
            "git_commit": "deadbeef",
            "invocation_mode": "cli",
        },
        "status": "completed",
        "exit_code": 0,
        "finished_at_utc": "2026-05-01T00:10:00+00:00",
        "run_dir": str(output_dir / "runs" / "pinn_hybrid_resnet"),
    }


def _row_metrics_payload() -> dict:
    return {
        "mae": [0.1, 0.2],
        "mse": [0.01, 0.02],
        "psnr": [30.0, 29.0],
        "ssim": [0.95, 0.94],
        "ms_ssim": [0.96, 0.95],
        "frc50": [42.0, 41.0],
        "frc1over7": [21.0, 20.0],
        "frc": [[0.1, 0.2], [0.3, 0.4]],
    }


def _row_history_payload() -> dict:
    return {
        "train_loss": [0.3, 0.2, 0.1],
        "val_loss": [0.4, 0.3, 0.2],
    }


def _fresh_row_invocation_payload(*, artifact_root: Path, row_id: str) -> dict:
    training_root = artifact_root / "training_runs" / row_id
    return {
        "script": "scripts/studies/grid_lines_torch_runner.py",
        "argv": ["--output-dir", str(training_root)],
        "command": "python scripts/studies/grid_lines_torch_runner.py --output-dir ...",
        "parsed_args": {
            "train_npz": str(artifact_root / "datasets" / "N128" / "gs1" / "train.npz"),
            "test_npz": str(artifact_root / "datasets" / "N128" / "gs1" / "test.npz"),
            "output_dir": str(training_root),
            "seed": 3,
            "epochs": 40,
            "hybrid_skip_connections": row_id != "pinn_hybrid_resnet_residual_fixed",
            "hybrid_skip_style": "add",
            "hybrid_resnet_bottleneck_layerscale_mode": (
                "fixed" if "residual_fixed" in row_id else "learned"
            ),
            "hybrid_resnet_bottleneck_layerscale_value": 1.0 if "residual_fixed" in row_id else None,
        },
        "cwd": str(artifact_root),
        "timestamp_utc": "2026-05-01T01:00:00+00:00",
        "pid": 202,
        "extra": {
            "runtime_provenance": {
                "python_executable": "python",
                "python_version": "3.11.9",
                "host": "test-host",
                "torch_version": "2.5.1",
                "cuda_version": "12.1",
                "gpu": "test-gpu",
            },
            "git_commit": "feedface",
            "invocation_mode": "cli",
            "ablation_row_id": row_id,
            "ablation_bundle_root": str(artifact_root),
        },
        "status": "completed",
        "exit_code": 0,
        "finished_at_utc": "2026-05-01T01:18:00+00:00",
        "run_dir": str(artifact_root / "runs" / row_id),
    }


def _create_visuals(root: Path, row_ids: list[str]) -> None:
    for row_id in row_ids:
        _write_text(root / "visuals" / f"amp_phase_{row_id}.png", "png")
        _write_text(root / "visuals" / f"amp_phase_error_{row_id}.png", "png")


def _seed_baseline_artifacts(baseline_root: Path) -> None:
    _write_json(baseline_root / "paper_benchmark_manifest.json", _baseline_manifest_payload())
    _write_json(baseline_root / "metrics.json", _baseline_metrics_payload())
    _write_numpy_archive(baseline_root / "datasets" / "N128" / "gs1" / "train.npz")
    _write_numpy_archive(baseline_root / "datasets" / "N128" / "gs1" / "test.npz")
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "invocation.json",
        _baseline_invocation_payload(baseline_root),
    )
    _write_text(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "invocation.sh",
        "#!/usr/bin/env bash\n",
    )
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "config.json",
        {"torch_runner_config": {"seed": 3}},
    )
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "history.json",
        _row_history_payload(),
    )
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "metrics.json",
        _row_metrics_payload(),
    )
    _write_torch_model(baseline_root / "runs" / "pinn_hybrid_resnet" / "model.pt")
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "randomness_contract.json",
        {"seed": 3, "requested_seed": 3},
    )
    _write_text(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "stdout.log",
        "baseline row completed\n",
    )
    _write_text(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "stderr.log",
        "baseline row stderr\n",
    )
    _write_json(
        baseline_root / "runs" / "pinn_hybrid_resnet" / "exit_code_proof.json",
        {
            "model_id": "pinn_hybrid_resnet",
            "validated_at_utc": "2026-05-01T00:11:00+00:00",
            "proof_source": "test",
            "exit_code": 0,
            "invocation_json": "runs/pinn_hybrid_resnet/invocation.json",
            "invocation_status": "completed",
            "stdout_log": "runs/pinn_hybrid_resnet/stdout.log",
            "stderr_log": "runs/pinn_hybrid_resnet/stderr.log",
        },
    )
    _write_numpy_archive(baseline_root / "recons" / "pinn_hybrid_resnet" / "recon.npz")
    _write_numpy_archive(baseline_root / "recons" / "gt" / "recon.npz")


def _seed_fresh_row_artifacts(artifact_root: Path, row_id: str) -> None:
    _write_json(
        artifact_root / "runs" / row_id / "invocation.json",
        _fresh_row_invocation_payload(artifact_root=artifact_root, row_id=row_id),
    )
    _write_text(artifact_root / "runs" / row_id / "invocation.sh", "#!/usr/bin/env bash\n")
    _write_json(
        artifact_root / "runs" / row_id / "config.json",
        {"torch_runner_config": {"seed": 3}},
    )
    _write_json(
        artifact_root / "runs" / row_id / "history.json",
        _row_history_payload(),
    )
    _write_json(
        artifact_root / "runs" / row_id / "metrics.json",
        _row_metrics_payload(),
    )
    _write_torch_model(artifact_root / "runs" / row_id / "model.pt")
    _write_json(
        artifact_root / "runs" / row_id / "randomness_contract.json",
        {"seed": 3, "requested_seed": 3},
    )
    _write_numpy_archive(artifact_root / "recons" / row_id / "recon.npz")
    _write_text(
        artifact_root / "training_runs" / row_id / "lightning_logs" / "version_0" / "metrics.csv",
        "epoch,train_loss,val_loss\n",
    )


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
    cross_reference_manifest = json.loads(
        Path(result["cross_reference_manifest_path"]).read_text(encoding="utf-8")
    )
    assert cross_reference_manifest["encoder_fusion_backlog"]["path"] == (
        "docs/backlog/active/2026-04-21-hybrid-resnet-encoder-fusion-variants.md"
    )


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
    assert residual_cfg.output_dir == artifact_root / "training_runs" / "pinn_hybrid_resnet_residual_fixed"
    assert interaction_cfg.output_dir == (
        artifact_root / "training_runs" / "pinn_hybrid_resnet_skip_add_residual_fixed"
    )
    assert residual_cfg.artifact_root == artifact_root
    assert interaction_cfg.artifact_root == artifact_root


def test_collate_ablation_bundle_marks_fresh_rows_as_direct_outputs(tmp_path, monkeypatch):
    baseline_root = tmp_path / "baseline"
    artifact_root = tmp_path / "ablation"
    summary_path = tmp_path / "summary.md"
    _seed_baseline_artifacts(baseline_root)
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    _write_numpy_archive(tmp_path / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz")
    prepare_execution_scaffold(
        baseline_root=baseline_root,
        artifact_root=artifact_root,
        summary_path=summary_path,
        plan_path=Path("docs/plans/plan.md"),
    )
    _write_numpy_archive(artifact_root / "datasets" / "N128" / "gs1" / "train.npz")
    _write_numpy_archive(artifact_root / "datasets" / "N128" / "gs1" / "test.npz")
    _seed_fresh_row_artifacts(artifact_root, "pinn_hybrid_resnet_skip_add")
    _create_visuals(
        artifact_root,
        ["pinn_hybrid_resnet", "pinn_hybrid_resnet_skip_add"],
    )

    bundle = collate_ablation_bundle(
        artifact_root=artifact_root,
        baseline_root=baseline_root,
    )

    metrics_payload = json.loads(Path(bundle["metrics_json"]).read_text(encoding="utf-8"))
    fresh_row = metrics_payload["rows"]["pinn_hybrid_resnet_skip_add"]
    assert fresh_row["runtime_summary"]["recovered_from_existing_artifacts"] is False
    assert "recovered_from_existing_artifacts" not in fresh_row["caveats"]
    assert "fresh_ablation_row" in fresh_row["caveats"]
    assert metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet_skip_add"] == []
    for key in ("stdout_log", "stderr_log", "exit_code_proof_json"):
        path = artifact_root / fresh_row["outputs"][key]
        assert path.exists(), key

    model_manifest = json.loads(Path(bundle["model_manifest_json"]).read_text(encoding="utf-8"))
    fresh_manifest_row = next(
        row for row in model_manifest["rows"] if row["model_id"] == "pinn_hybrid_resnet_skip_add"
    )
    assert fresh_manifest_row["runtime_summary"]["recovered_from_existing_artifacts"] is False
    assert fresh_manifest_row["missing_fields"] == []


def test_run_fresh_row_relaunches_when_row_local_training_root_is_missing(tmp_path, monkeypatch):
    baseline_root = tmp_path / "baseline"
    artifact_root = tmp_path / "ablation"
    summary_path = tmp_path / "summary.md"
    _seed_baseline_artifacts(baseline_root)
    prepare_execution_scaffold(
        baseline_root=baseline_root,
        artifact_root=artifact_root,
        summary_path=summary_path,
        plan_path=Path("docs/plans/plan.md"),
    )
    row_id = "pinn_hybrid_resnet_skip_add"
    _seed_fresh_row_artifacts(artifact_root, row_id)
    # Leave the expected training root absent so reuse must be rejected.
    training_root = artifact_root / "training_runs" / row_id
    shutil.rmtree(training_root)

    calls: list[Path] = []

    def _fake_run_grid_lines_torch(cfg, invocation_extra=None):
        calls.append(Path(cfg.output_dir))
        _write_text(training_root / "lightning_logs" / "version_0" / "metrics.csv", "epoch\n")
        _write_json(
            artifact_root / "runs" / row_id / "invocation.json",
            _fresh_row_invocation_payload(artifact_root=artifact_root, row_id=row_id),
        )
        _write_text(artifact_root / "runs" / row_id / "stdout.log", "direct fresh row stdout\n")
        _write_text(artifact_root / "runs" / row_id / "stderr.log", "direct fresh row stderr\n")
        _write_numpy_archive(artifact_root / "recons" / row_id / "recon.npz")
        return {"recon_npz": artifact_root / "recons" / row_id / "recon.npz"}

    monkeypatch.setattr(ablation, "run_grid_lines_torch", _fake_run_grid_lines_torch)

    result = run_fresh_row(artifact_root=artifact_root, row_id=row_id)

    assert calls == [training_root]
    assert result["status"] == "completed_row"

"""Tests for the Lines128 Hybrid ResNet encoder-fusion variants helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import scripts.studies.lines128_hybrid_resnet_encoder_fusion_variants as helper
from scripts.studies.lines128_hybrid_resnet_encoder_fusion_variants import (
    BASELINE_ROW_ID,
    MANDATORY_FRESH_ROWS,
    RUN_ID,
    build_row_runner_config,
    collate_ablation_bundle,
    prepare_execution_scaffold,
    repair_existing_run_provenance,
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


def _baseline_invocation_payload(training_root: Path) -> dict:
    return {
        "script": "scripts/studies/grid_lines_torch_runner.py",
        "argv": ["--output-dir", str(training_root)],
        "command": "python scripts/studies/grid_lines_torch_runner.py --output-dir ...",
        "parsed_args": {
            "train_npz": str(training_root.parents[2] / "datasets" / "N128" / "gs1" / "train.npz"),
            "test_npz": str(training_root.parents[2] / "datasets" / "N128" / "gs1" / "test.npz"),
            "output_dir": str(training_root),
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
            "hybrid_encoder_fusion_mode": "baseline",
            "hybrid_encoder_layerscale_init": 0.1,
            "hybrid_encoder_branch_gate_init": 0.1,
            "torch_logger": "csv",
            "recon_log_num_patches": 4,
        },
        "cwd": str(training_root.parents[2]),
        "timestamp_utc": "2026-05-02T10:00:00+00:00",
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
        "finished_at_utc": "2026-05-02T10:10:00+00:00",
        "run_dir": str(training_root.parents[1] / "runs" / BASELINE_ROW_ID),
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


def _row_history_payload(min_val: float) -> list[float]:
    return [0.145, 0.09, min_val]


def _seed_baseline_artifacts(baseline_root: Path) -> None:
    _write_json(baseline_root / "paper_benchmark_manifest.json", _baseline_manifest_payload())
    _write_json(baseline_root / "metrics.json", _baseline_metrics_payload())
    _write_numpy_archive(baseline_root / "datasets" / "N128" / "gs1" / "train.npz")
    _write_numpy_archive(baseline_root / "datasets" / "N128" / "gs1" / "test.npz")
    _write_numpy_archive(baseline_root / "recons" / "gt" / "recon.npz")
    _write_numpy_archive(baseline_root / "recons" / BASELINE_ROW_ID / "recon.npz")
    baseline_training_root = baseline_root / "training_runs" / BASELINE_ROW_ID
    _write_json(
        baseline_root / "runs" / BASELINE_ROW_ID / "invocation.json",
        _baseline_invocation_payload(baseline_training_root),
    )
    _write_text(baseline_root / "runs" / BASELINE_ROW_ID / "invocation.sh", "#!/usr/bin/env bash\n")
    _write_json(
        baseline_root / "runs" / BASELINE_ROW_ID / "config.json",
        {"torch_runner_config": {"seed": 3}},
    )
    _write_json(baseline_root / "runs" / BASELINE_ROW_ID / "history.json", _row_history_payload(0.03))
    _write_json(baseline_root / "runs" / BASELINE_ROW_ID / "metrics.json", _row_metrics_payload())
    _write_torch_model(baseline_root / "runs" / BASELINE_ROW_ID / "model.pt")
    _write_json(
        baseline_root / "runs" / BASELINE_ROW_ID / "randomness_contract.json",
        {"seed": 3, "requested_seed": 3},
    )
    _write_text(baseline_root / "runs" / BASELINE_ROW_ID / "stdout.log", "baseline stdout\n")
    _write_text(baseline_root / "runs" / BASELINE_ROW_ID / "stderr.log", "baseline stderr\n")
    _write_json(
        baseline_root / "runs" / BASELINE_ROW_ID / "launcher_completion.json",
        {"model_id": BASELINE_ROW_ID},
    )


def _fresh_invocation_payload(*, run_root: Path, row_id: str, fusion_mode: str) -> dict:
    return {
        "script": "scripts/studies/grid_lines_torch_runner.py",
        "argv": ["--output-dir", str(run_root)],
        "command": "python scripts/studies/grid_lines_torch_runner.py --output-dir ...",
        "parsed_args": {
            "train_npz": str(run_root.parents[2] / "complete_table" / "datasets" / "N128" / "gs1" / "train.npz"),
            "test_npz": str(run_root.parents[2] / "complete_table" / "datasets" / "N128" / "gs1" / "test.npz"),
            "output_dir": str(run_root),
            "architecture": "hybrid_resnet",
            "training_procedure": "pinn",
            "seed": 3,
            "epochs": 40,
            "hybrid_encoder_fusion_mode": fusion_mode,
            "hybrid_encoder_layerscale_init": 0.1,
            "hybrid_encoder_branch_gate_init": 0.1,
        },
        "cwd": str(run_root.parents[3]),
        "timestamp_utc": "2026-05-02T10:43:19+00:00",
        "pid": 202,
        "status": "completed",
        "exit_code": 0,
        "finished_at_utc": "2026-05-02T10:59:00+00:00",
        "run_dir": str(run_root / "runs" / row_id),
    }


def _write_fake_checkpoint(path: Path, fusion_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 39,
            "global_step": 22480,
            "hyper_parameters": {
                "generator_overrides": {
                    "hybrid_encoder_fusion_mode": fusion_mode,
                }
            },
        },
        path,
    )


def _seed_existing_shared_run(bundle_root: Path) -> Path:
    run_root = bundle_root / "runs" / RUN_ID
    _write_text(
        run_root / "launch_rows.log",
        "\n".join(
            [
                "==========================================",
                "Launching row: pinn_hybrid_resnet_encoder_layerscale",
                "Fusion mode:   layerscale",
                "Saved artifacts to /tmp/runs/pinn_hybrid_resnet_encoder_layerscale",
                "Torch runner complete. Artifacts in /tmp/runs/pinn_hybrid_resnet_encoder_layerscale",
                "==========================================",
                "Launching row: pinn_hybrid_resnet_encoder_branch_gated",
                "Fusion mode:   branch_gated",
                "Saved artifacts to /tmp/runs/pinn_hybrid_resnet_encoder_branch_gated",
                "Torch runner complete. Artifacts in /tmp/runs/pinn_hybrid_resnet_encoder_branch_gated",
                "==========================================",
                "Launching row: pinn_hybrid_resnet_encoder_branch_gated_layerscale",
                "Fusion mode:   branch_gated_layerscale",
                "Saved artifacts to /tmp/runs/pinn_hybrid_resnet_encoder_branch_gated_layerscale",
                "Torch runner complete. Artifacts in /tmp/runs/pinn_hybrid_resnet_encoder_branch_gated_layerscale",
            ])
        + "\n",
    )

    versions = {
        "layerscale": "version_0",
        "branch_gated": "version_1",
        "branch_gated_layerscale": "version_2",
    }
    for fusion_mode, version in versions.items():
        _write_text(
            run_root / "lightning_logs" / version / "hparams.yaml",
            f"generator_overrides:\n  hybrid_encoder_fusion_mode: {fusion_mode}\n",
        )
        _write_text(
            run_root / "lightning_logs" / version / "metrics.csv",
            "step,metric\n0,1.0\n",
        )

    _write_fake_checkpoint(
        run_root / "checkpoints" / "epoch=epoch=39-mae_val=mae_val_loss=0.0315.ckpt",
        "layerscale",
    )
    _write_fake_checkpoint(run_root / "checkpoints" / "last.ckpt", "layerscale")
    _write_fake_checkpoint(
        run_root / "checkpoints" / "epoch=epoch=38-mae_val=mae_val_loss=0.0288.ckpt",
        "branch_gated",
    )
    _write_fake_checkpoint(run_root / "checkpoints" / "last-v1.ckpt", "branch_gated")
    _write_fake_checkpoint(
        run_root / "checkpoints" / "epoch=epoch=38-mae_val=mae_val_loss=0.0320.ckpt",
        "branch_gated_layerscale",
    )
    _write_fake_checkpoint(run_root / "checkpoints" / "last-v2.ckpt", "branch_gated_layerscale")

    row_modes = {
        "pinn_hybrid_resnet_encoder_layerscale": ("layerscale", 0.0315),
        "pinn_hybrid_resnet_encoder_branch_gated": ("branch_gated", 0.0288),
        "pinn_hybrid_resnet_encoder_branch_gated_layerscale": ("branch_gated_layerscale", 0.0320),
    }
    for row_id, (fusion_mode, min_val) in row_modes.items():
        _write_json(run_root / "runs" / row_id / "invocation.json", _fresh_invocation_payload(run_root=run_root, row_id=row_id, fusion_mode=fusion_mode))
        _write_text(run_root / "runs" / row_id / "invocation.sh", "#!/usr/bin/env bash\n")
        _write_json(
            run_root / "runs" / row_id / "config.json",
            {
                "torch_runner_config": {
                    "output_dir": str(run_root),
                    "hybrid_encoder_fusion_mode": fusion_mode,
                }
            },
        )
        _write_json(run_root / "runs" / row_id / "history.json", _row_history_payload(min_val))
        _write_json(run_root / "runs" / row_id / "metrics.json", _row_metrics_payload())
        _write_torch_model(run_root / "runs" / row_id / "model.pt")
        _write_json(
            run_root / "runs" / row_id / "randomness_contract.json",
            {"seed": 3, "requested_seed": 3},
        )
        _write_numpy_archive(run_root / "recons" / row_id / "recon.npz")

    _write_numpy_archive(run_root / "recons" / "gt" / "recon.npz")
    return run_root


def test_build_row_runner_config_uses_row_local_training_root(tmp_path):
    baseline_root = tmp_path / "complete_table"
    bundle_root = tmp_path / "bundle"
    _seed_baseline_artifacts(baseline_root)

    prepare_execution_scaffold(
        baseline_root=baseline_root,
        artifact_root=bundle_root,
        summary_path=tmp_path / "summary.md",
        plan_path=tmp_path / "plan.md",
    )

    cfg = build_row_runner_config(
        artifact_root=bundle_root,
        row_id="pinn_hybrid_resnet_encoder_branch_gated",
    )

    expected_run_root = bundle_root / "runs" / RUN_ID
    assert cfg.output_dir == expected_run_root / "training_runs" / "pinn_hybrid_resnet_encoder_branch_gated"
    assert cfg.artifact_root == expected_run_root
    assert cfg.model_id_override == "pinn_hybrid_resnet_encoder_branch_gated"
    assert cfg.hybrid_encoder_fusion_mode == "branch_gated"


def test_repair_existing_run_provenance_rebuilds_row_local_artifacts_and_manifest(tmp_path):
    baseline_root = tmp_path / "complete_table"
    bundle_root = tmp_path / "bundle"
    _seed_baseline_artifacts(baseline_root)
    prepare_execution_scaffold(
        baseline_root=baseline_root,
        artifact_root=bundle_root,
        summary_path=tmp_path / "summary.md",
        plan_path=tmp_path / "plan.md",
    )
    run_root = _seed_existing_shared_run(bundle_root)

    repair_existing_run_provenance(artifact_root=bundle_root)
    collate_ablation_bundle(artifact_root=bundle_root, baseline_root=baseline_root)

    for row_id in MANDATORY_FRESH_ROWS:
        training_root = run_root / "training_runs" / row_id
        assert training_root.exists()
        assert (training_root / "lightning_logs" / "version_0" / "hparams.yaml").exists()
        assert (training_root / "checkpoints" / "last.ckpt").exists()
        row_root = run_root / "runs" / row_id
        assert (row_root / "stdout.log").exists()
        assert (row_root / "stderr.log").exists()
        assert (row_root / "launcher_completion.json").exists()

    manifest_payload = json.loads((bundle_root / "model_manifest.json").read_text(encoding="utf-8"))
    rows = {
        row_id: row
        for row_id, row in manifest_payload["rows"].items()
        if row_id != BASELINE_ROW_ID
    }
    for row_id in MANDATORY_FRESH_ROWS:
        artifacts = rows[row_id]["artifacts"]
        assert artifacts["stdout_log"].endswith(f"runs/{RUN_ID}/runs/{row_id}/stdout.log")
        assert artifacts["stderr_log"].endswith(f"runs/{RUN_ID}/runs/{row_id}/stderr.log")
        assert artifacts["launcher_completion_json"].endswith(
            f"runs/{RUN_ID}/runs/{row_id}/launcher_completion.json"
        )
        assert artifacts["training_output_dir"].endswith(f"runs/{RUN_ID}/training_runs/{row_id}")
        assert artifacts["checkpoint_dir"].endswith(f"runs/{RUN_ID}/training_runs/{row_id}/checkpoints")

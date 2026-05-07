# tests/test_grid_lines_compare_wrapper.py
"""Tests for grid_lines_compare_wrapper orchestration."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _full_pair_metrics(amp: float, phase: float):
    return {
        "mae": [amp, phase],
        "mse": [amp * 0.1, phase * 0.1],
        "psnr": [70.0 + amp, 65.0 + phase],
        "ssim": [0.9, 0.8],
        "ms_ssim": [0.85, 0.75],
        "frc50": [64, 48],
    }


def _mock_dataset_builder(monkeypatch):
    def fake_build_grid_lines_datasets(cfg, *args, **kwargs):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        train_npz = datasets_dir / "train.npz"
        test_npz = datasets_dir / "test.npz"
        train_npz.write_bytes(b"stub")
        test_npz.write_bytes(b"stub")
        gt_dir = cfg.output_dir / "recons" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
        np.savez(gt_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"train_npz": str(train_npz), "test_npz": str(test_npz), "gt_recon": str(gt_dir / "recon.npz")}

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets",
        fake_build_grid_lines_datasets,
    )


def test_wrapper_merges_metrics(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({
            "pinn": {"mse": 0.1},
            "baseline": {"mse": 0.2},
        }))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    def fake_torch_run(cfg):
        run_dir = cfg.output_dir / "runs" / f"pinn_{cfg.architecture}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps({"mse": 0.3}))
        return {"metrics": {"mse": 0.3}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    result = run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn", "baseline", "fno", "hybrid"),
        probe_npz=Path("dummy_probe.npz"),
    )
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn" in merged
    assert "baseline" in merged
    assert "pinn_fno" in merged
    assert "pinn_hybrid" in merged
    assert "metrics" in result


def test_wrapper_routes_spectral_resnet_bottleneck_from_explicit_models(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    _mock_dataset_builder(monkeypatch)
    captured = {}

    def fake_torch_run(cfg):
        captured["architecture"] = cfg.architecture
        model_id = f"pinn_{cfg.architecture}"
        recon_dir = cfg.output_dir / "recons" / model_id
        recon_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
        np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"recon_npz": str(recon_dir / "recon.npz")}

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.02, 0.03),
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=Path("dummy_probe.npz"),
        architectures=(),
        models=("pinn_spectral_resnet_bottleneck_net",),
        model_n={"pinn_spectral_resnet_bottleneck_net": 128},
    )

    assert captured["architecture"] == "spectral_resnet_bottleneck_net"
    assert "pinn_spectral_resnet_bottleneck_net" in result["metrics"]


def test_wrapper_preflight_only_validates_supported_rows_without_running_backends(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    probe_path = tmp_path / "probe.npz"
    probe_path.write_bytes(b"stub")
    captured = {"architectures": []}

    def fake_setup_torch_configs(cfg):
        captured["architectures"].append(cfg.architecture)
        return object(), object()

    def fail_torch_run(cfg):
        raise AssertionError("preflight_only should not launch the torch runner")

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.setup_torch_configs", fake_setup_torch_configs)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fail_torch_run)

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=probe_path,
        architectures=(),
        models=("pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"),
        model_n={
            "pinn": 128,
            "pinn_hybrid_resnet": 128,
            "pinn_fno_vanilla": 128,
        },
        preflight_only=True,
        seed=3,
        set_phi=True,
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert result["mode"] == "preflight_only"
    assert result["selected_models"] == [
        "pinn",
        "pinn_hybrid_resnet",
        "pinn_fno_vanilla",
    ]
    assert captured["architectures"] == ["hybrid_resnet", "fno_vanilla"]
    assert result["row_plan"][0]["model_id"] == "pinn"
    assert result["row_plan"][0]["backend"] == "tf"


def test_wrapper_preflight_only_accepts_row_specs_with_same_base_architecture_override(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    probe_path = tmp_path / "probe.npz"
    probe_path.write_bytes(b"stub")
    captured = {}

    def fake_setup_torch_configs(cfg):
        captured["architecture"] = cfg.architecture
        captured["model_id_override"] = cfg.model_id_override
        captured["hybrid_downsample_steps"] = cfg.hybrid_downsample_steps
        return object(), object()

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.setup_torch_configs", fake_setup_torch_configs)

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=probe_path,
        architectures=(),
        models=("pinn_spectral_resnet_bottleneck_ds1",),
        row_specs=(
            {
                "model_id": "pinn_spectral_resnet_bottleneck_ds1",
                "architecture": "spectral_resnet_bottleneck_net",
                "training_procedure": "pinn",
                "overrides": {"hybrid_downsample_steps": 1},
            },
        ),
        model_n={"pinn_spectral_resnet_bottleneck_ds1": 128},
        preflight_only=True,
        seed=3,
        set_phi=True,
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert result["mode"] == "preflight_only"
    assert result["selected_models"] == ["pinn_spectral_resnet_bottleneck_ds1"]
    assert captured["architecture"] == "spectral_resnet_bottleneck_net"
    assert captured["model_id_override"] == "pinn_spectral_resnet_bottleneck_ds1"
    assert captured["hybrid_downsample_steps"] == 1
    assert result["row_plan"][0]["model_id"] == "pinn_spectral_resnet_bottleneck_ds1"


def test_wrapper_preflight_only_routes_ffno_depth24_row_with_override(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    probe_path = tmp_path / "probe.npz"
    probe_path.write_bytes(b"stub")
    captured = {}

    def fake_setup_torch_configs(cfg):
        captured["architecture"] = cfg.architecture
        captured["model_id_override"] = cfg.model_id_override
        captured["model_label_override"] = cfg.model_label_override
        captured["fno_blocks"] = cfg.fno_blocks
        captured["fno_cnn_blocks"] = cfg.fno_cnn_blocks
        return object(), object()

    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner.setup_torch_configs",
        fake_setup_torch_configs,
    )

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=probe_path,
        architectures=(),
        models=("pinn_ffno_depth24",),
        model_n={"pinn_ffno_depth24": 128},
        preflight_only=True,
        seed=3,
        set_phi=True,
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        fno_blocks=4,
        fno_cnn_blocks=0,
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert result["mode"] == "preflight_only"
    assert result["selected_models"] == ["pinn_ffno_depth24"]
    assert captured["architecture"] == "ffno"
    assert captured["model_id_override"] == "pinn_ffno_depth24"
    assert captured["model_label_override"] == "FFNO-24 + PINN"
    assert captured["fno_blocks"] == 24
    assert captured["fno_cnn_blocks"] == 0
    assert result["row_plan"][0]["model_id"] == "pinn_ffno_depth24"
    assert result["row_plan"][0]["overrides"]["fno_blocks"] == 24


def test_wrapper_emits_row_payloads_for_minimum_subset_execution(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    _mock_dataset_builder(monkeypatch)

    def fake_tf_run(cfg, tf_models=None):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        for model_id in ("baseline", "pinn"):
            run_dir = cfg.output_dir / "runs" / model_id
            run_dir.mkdir(parents=True, exist_ok=True)
            recon_dir = cfg.output_dir / "recons" / model_id
            recon_dir.mkdir(parents=True, exist_ok=True)
            gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
            np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
            (run_dir / "invocation.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "finished_at_utc": "2026-04-29T00:00:00+00:00",
                        "exit_code": 0,
                        "pid": 12345,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "invocation.sh").write_text("python fake_tf_run.py\n", encoding="utf-8")
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "history.json").write_text("{}", encoding="utf-8")
            (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            (run_dir / "stdout.log").write_text(f"[row:{model_id}] tf stdout\n", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
            "row_payloads": {
                "baseline": {
                    "model_label": "CDI CNN + supervised",
                    "architecture_id": "cnn",
                    "training_procedure": "supervised",
                    "N": 128,
                    "parameter_count": 10,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.4,
                    "validation_loss": {"status": "not_emitted", "value": None},
                    "runtime_summary": {"train_wall_time_sec": 4.0, "inference_time_sec": 0.2},
                    "hardware_summary": {"backend": "tensorflow", "accelerator": "rtx3090"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "metrics": _full_pair_metrics(0.2, 0.3),
                },
                "pinn": {
                    "model_label": "CDI CNN + PINN",
                    "architecture_id": "cnn",
                    "training_procedure": "pinn",
                    "N": 128,
                    "parameter_count": 11,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.3,
                    "validation_loss": {"status": "not_emitted", "value": None},
                    "runtime_summary": {"train_wall_time_sec": 5.0, "inference_time_sec": 0.3},
                    "hardware_summary": {"backend": "tensorflow", "accelerator": "rtx3090"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "metrics": _full_pair_metrics(0.19, 0.29),
                },
            },
        }

    def fake_torch_run(cfg):
        model_id = f"pinn_{cfg.architecture}"
        run_dir = cfg.output_dir / "runs" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        recon_dir = cfg.output_dir / "recons" / model_id
        recon_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
        np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        (run_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "finished_at_utc": "2026-04-29T00:00:00+00:00",
                    "exit_code": 0,
                    "pid": 12345,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "invocation.sh").write_text("python fake_torch_run.py\n", encoding="utf-8")
        (run_dir / "config.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "stdout.log").write_text(f"[row:{model_id}] torch stdout\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        return {
            "recon_npz": str(recon_dir / "recon.npz"),
            "paper_row_payload": {
                "model_label": "Torch row",
                "architecture_id": cfg.architecture,
                "training_procedure": "pinn",
                "N": cfg.N,
                "parameter_count": 12,
                "epoch_budget": cfg.epochs,
                "final_completed_epoch": cfg.epochs,
                "final_train_loss": 0.2,
                "validation_loss": {"status": "not_emitted", "value": None},
                "runtime_summary": {"train_wall_time_sec": 6.0, "inference_time_sec": 0.4},
                "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
                "row_status": "paper_grade",
                "caveats": [],
            },
        }

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.01, 0.03),
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(),
        models=("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"),
        model_n={
            "baseline": 128,
            "pinn": 128,
            "pinn_hybrid_resnet": 128,
            "pinn_fno_vanilla": 128,
        },
        probe_npz=tmp_path / "probe.npz",
        seed=3,
        set_phi=True,
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert set(result["row_payloads"]) == {
        "baseline",
        "pinn",
        "pinn_hybrid_resnet",
        "pinn_fno_vanilla",
    }
    assert result["row_payloads"]["baseline"]["training_procedure"] == "supervised"
    assert result["row_payloads"]["pinn_hybrid_resnet"]["architecture_id"] == "hybrid_resnet"
    assert "metrics" in result["row_payloads"]["pinn_fno_vanilla"]
    baseline_payload = result["row_payloads"]["baseline"]
    assert baseline_payload["git"]["dirty_state_note"]["source"]
    assert baseline_payload["environment"]["host"]
    assert "torch_version" in baseline_payload["environment"]
    assert baseline_payload["dataset"]["manifest_json"] == "dataset_identity_manifest.json"
    assert (tmp_path / baseline_payload["dataset"]["manifest_json"]).exists()
    assert baseline_payload["splits"]["manifest_json"] == "split_manifest.json"
    assert (tmp_path / baseline_payload["splits"]["manifest_json"]).exists()
    assert baseline_payload["outputs"]["stdout_log"] == "runs/baseline/stdout.log"
    assert baseline_payload["outputs"]["exit_code_proof_json"] == "runs/baseline/exit_code_proof.json"
    assert (tmp_path / baseline_payload["outputs"]["exit_code_proof_json"]).exists()
    model_manifest = json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    assert model_manifest["rows"][0]["model_id"] == "baseline"
    assert model_manifest["rows"][0]["training_procedure"] == "supervised"
    assert [row["model_id"] for row in model_manifest["rows"]] == [
        "baseline",
        "pinn",
        "pinn_hybrid_resnet",
        "pinn_fno_vanilla",
    ]


def test_wrapper_preserves_distinct_tf_row_logs(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    _mock_dataset_builder(monkeypatch)

    def fake_tf_run(cfg, tf_models=None):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        row_payloads = {}
        for model_id in tf_models or ():
            run_dir = cfg.output_dir / "runs" / model_id
            run_dir.mkdir(parents=True, exist_ok=True)
            recon_dir = cfg.output_dir / "recons" / model_id
            recon_dir.mkdir(parents=True, exist_ok=True)
            gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
            np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
            (run_dir / "invocation.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "finished_at_utc": "2026-04-29T00:00:00+00:00",
                        "exit_code": 0,
                        "pid": 12345,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "invocation.sh").write_text("python fake_tf_run.py\n", encoding="utf-8")
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "history.json").write_text("{}", encoding="utf-8")
            (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            (run_dir / "stdout.log").write_text(
                f"[row:{model_id}] distinct stdout\n",
                encoding="utf-8",
            )
            (run_dir / "stderr.log").write_text(
                f"[row:{model_id}] distinct stderr\n",
                encoding="utf-8",
            )
            row_payloads[model_id] = {
                "model_label": PAPER_MODEL_LABELS[model_id] if "PAPER_MODEL_LABELS" in globals() else model_id,
                "architecture_id": "cnn",
                "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                "N": cfg.N,
                "parameter_count": 10,
                "epoch_budget": 40,
                "final_completed_epoch": 40,
                "final_train_loss": 0.4,
                "validation_loss": {"status": "not_emitted", "value": None},
                "runtime_summary": {"train_wall_time_sec": 4.0, "inference_time_sec": 0.2},
                "hardware_summary": {"backend": "tensorflow", "accelerator": "rtx3090"},
                "row_status": "paper_grade",
                "caveats": [],
                "metrics": _full_pair_metrics(0.2, 0.3),
            }
        print("shared wrapper output that should stay out of row-local logs")
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
            "row_payloads": row_payloads,
        }

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.02, 0.03),
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=Path("dummy_probe.npz"),
        architectures=(),
        models=("baseline", "pinn"),
        model_n={"baseline": 128, "pinn": 128},
        seed=3,
        set_phi=True,
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    baseline_stdout = (tmp_path / "runs" / "baseline" / "stdout.log").read_text(encoding="utf-8")
    pinn_stdout = (tmp_path / "runs" / "pinn" / "stdout.log").read_text(encoding="utf-8")
    assert baseline_stdout == "[row:baseline] distinct stdout\n"
    assert pinn_stdout == "[row:pinn] distinct stdout\n"
    assert baseline_stdout != pinn_stdout


def test_wrapper_uses_locked_epoch_budget_for_tf_rows_in_explicit_model_mode(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    _mock_dataset_builder(monkeypatch)
    captured = {}

    def fake_tf_run(cfg, tf_models=None):
        captured["nepochs"] = cfg.nepochs
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        for model_id in ("baseline", "pinn"):
            recon_dir = cfg.output_dir / "recons" / model_id
            recon_dir.mkdir(parents=True, exist_ok=True)
            gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
            np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
            "row_payloads": {},
        }

    def fake_torch_run(cfg):
        model_id = f"pinn_{cfg.architecture}"
        recon_dir = cfg.output_dir / "recons" / model_id
        recon_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
        np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"recon_npz": str(recon_dir / "recon.npz"), "paper_row_payload": {}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.01, 0.03),
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(),
        models=("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"),
        model_n={
            "baseline": 128,
            "pinn": 128,
            "pinn_hybrid_resnet": 128,
            "pinn_fno_vanilla": 128,
        },
        probe_npz=tmp_path / "probe.npz",
        seed=3,
        set_phi=True,
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert captured["nepochs"] == 40


def test_wrapper_backfills_tf_row_n_when_payload_emits_none(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    _mock_dataset_builder(monkeypatch)

    def fake_tf_run(cfg, tf_models=None):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        for model_id in ("baseline", "pinn"):
            recon_dir = cfg.output_dir / "recons" / model_id
            recon_dir.mkdir(parents=True, exist_ok=True)
            gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
            np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
            "row_payloads": {
                "baseline": {"N": None},
                "pinn": {"N": None},
            },
        }

    def fake_torch_run(cfg):
        model_id = f"pinn_{cfg.architecture}"
        recon_dir = cfg.output_dir / "recons" / model_id
        recon_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((cfg.N, cfg.N)) + 1j * np.ones((cfg.N, cfg.N))).astype(np.complex64)
        np.savez(recon_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"recon_npz": str(recon_dir / "recon.npz"), "paper_row_payload": {"N": cfg.N}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.01, 0.03),
    )
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(),
        models=("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"),
        model_n={
            "baseline": 128,
            "pinn": 128,
            "pinn_hybrid_resnet": 128,
            "pinn_fno_vanilla": 128,
        },
        probe_npz=tmp_path / "probe.npz",
        seed=3,
        set_phi=True,
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert result["row_payloads"]["baseline"]["N"] == 128
    assert result["row_payloads"]["pinn"]["N"] == 128


def test_wrapper_writes_metrics_table_tex_architecture_mode(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "pinn": _full_pair_metrics(0.02, 0.05),
                    "baseline": _full_pair_metrics(0.03, 0.07),
                }
            )
        )
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
        }

    def fake_torch_run(cfg):
        _ = cfg
        return {"metrics": _full_pair_metrics(0.01, 0.04)}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn", "baseline", "ffno"),
        probe_npz=Path("dummy_probe.npz"),
    )
    table_path = tmp_path / "metrics_table.tex"
    csv_path = tmp_path / "metrics_table.csv"
    assert table_path.exists()
    assert csv_path.exists()
    table_text = table_path.read_text()
    csv_text = csv_path.read_text()
    assert "N & Model" in table_text
    assert "MAE" in table_text
    assert "64" in table_text
    assert "PtychoPINN (CNN)" in table_text
    assert "Baseline" in table_text
    assert "FFNO" in table_text
    assert "model_id,model_label,N" in csv_text
    assert "pinn_ffno,FFNO + PINN,64" in csv_text


def test_wrapper_writes_metrics_table_tex_models_reuse_path(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    recon_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.01, 0.03),
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid",),
        model_n={"pinn_hybrid": 128},
        probe_npz=Path("dummy_probe.npz"),
        reuse_existing_recons=True,
    )
    table_path = tmp_path / "metrics_table.tex"
    assert table_path.exists()
    table_text = table_path.read_text()
    assert "N & Model" in table_text
    assert "128" in table_text
    assert "Hybrid" in table_text


def test_wrapper_reuse_path_recovers_row_payloads_from_existing_artifacts(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    for model_id in ("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"):
        recon_path = tmp_path / "recons" / model_id / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    (tmp_path / "baseline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "baseline" / "baseline.keras").write_text("stub", encoding="utf-8")
    (tmp_path / "pinn").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pinn" / "wts.h5.zip").write_text("stub", encoding="utf-8")
    (tmp_path / "live_stdout.log").write_text(
        "\n".join(
            [
                "Epoch 40/40",
                "loss: 13.2960 - pred_intensity_loss: -1833342.8750",
                "conv2d_12_loss: 0.0148 - conv2d_19_loss: 0.1235 - loss: 0.1383 - val_loss: 0.1379",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "invocation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "finished_at_utc": "2026-04-29T22:00:00+00:00",
                "parsed_args": {"reuse_existing_recons": True},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "launcher_stderr.log").write_text(
        "\n".join(
            [
                f"Saved artifacts to {tmp_path / 'runs' / 'pinn_hybrid_resnet'}",
                f"Torch runner complete. Artifacts in {tmp_path / 'runs' / 'pinn_hybrid_resnet'}",
                f"Saved artifacts to {tmp_path / 'runs' / 'pinn_fno_vanilla'}",
                f"Torch runner complete. Artifacts in {tmp_path / 'runs' / 'pinn_fno_vanilla'}",
            ]
        ),
        encoding="utf-8",
    )

    for model_id in ("pinn_hybrid_resnet", "pinn_fno_vanilla"):
        run_dir = tmp_path / "runs" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "history.json").write_text(
            json.dumps({"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "timestamp_utc": "2026-04-29T21:00:00+00:00",
                    "finished_at_utc": "2026-04-29T21:00:05+00:00",
                    "parsed_args": {"epochs": 40},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "model.pt").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.01, 0.03),
    )

    class _FakeModel:
        def __init__(self, count):
            self._count = count

        def count_params(self):
            return self._count

    monkeypatch.setattr(
        "tensorflow.keras.models.load_model",
        lambda path, compile=False: _FakeModel(10),
    )
    monkeypatch.setattr(
        "ptycho.model_manager.ModelManager.load_multiple_models",
        lambda path, model_names=None: {"autoencoder": _FakeModel(11)},
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_compare_wrapper._count_torch_state_dict_parameters",
        lambda path: 12 if "hybrid" in str(path) else 13,
    )

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(),
        models=("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"),
        model_n={
            "baseline": 128,
            "pinn": 128,
            "pinn_hybrid_resnet": 128,
            "pinn_fno_vanilla": 128,
        },
        probe_npz=Path("dummy_probe.npz"),
        reuse_existing_recons=True,
    )

    assert set(result["row_payloads"]) == {
        "baseline",
        "pinn",
        "pinn_hybrid_resnet",
        "pinn_fno_vanilla",
    }
    assert result["row_payloads"]["baseline"]["parameter_count"] == 10
    assert result["row_payloads"]["pinn"]["parameter_count"] == 11
    assert result["row_payloads"]["pinn_hybrid_resnet"]["parameter_count"] == 12
    assert result["row_payloads"]["pinn_fno_vanilla"]["parameter_count"] == 13
    assert result["row_payloads"]["baseline"]["final_train_loss"] == pytest.approx(0.1383)
    assert result["row_payloads"]["pinn"]["final_train_loss"] == pytest.approx(13.2960)
    assert result["row_payloads"]["pinn_hybrid_resnet"]["final_completed_epoch"] == 2
    assert result["row_payloads"]["baseline"]["row_status"] == "decision_support"
    assert result["row_payloads"]["pinn"]["row_status"] == "decision_support"


def test_wrapper_preflight_supports_supervised_ffno_row(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    probe_path = tmp_path / "probe.npz"
    probe_path.write_bytes(b"stub")
    captured = {}

    def fake_setup_torch_configs(cfg):
        captured["architecture"] = cfg.architecture
        captured["training_procedure"] = cfg.training_procedure
        return object(), object()

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.setup_torch_configs", fake_setup_torch_configs)

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=probe_path,
        architectures=(),
        models=("supervised_ffno",),
        model_n={"supervised_ffno": 128},
        preflight_only=True,
        seed=3,
        set_phi=True,
        probe_scale_mode="pad_extrapolate",
        torch_epochs=40,
        torch_learning_rate=2e-4,
        torch_scheduler="ReduceLROnPlateau",
        torch_plateau_factor=0.5,
        torch_plateau_patience=2,
        torch_plateau_min_lr=1e-4,
        torch_plateau_threshold=0.0,
        torch_loss_mode="mae",
        torch_output_mode="real_imag",
        nimgs_train=2,
        nimgs_test=2,
        nphotons=1e9,
    )

    assert result["mode"] == "preflight_only"
    assert result["selected_models"] == ["supervised_ffno"]
    assert captured["architecture"] == "ffno"
    assert captured["training_procedure"] == "supervised"
    assert result["row_plan"][0]["model_id"] == "supervised_ffno"
    assert result["row_plan"][0]["status"] == "supported_for_harness"


def test_recover_torch_row_payload_marks_current_root_rows_as_fresh(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import _recover_torch_row_payload

    run_dir = tmp_path / "runs" / "supervised_ffno"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(
        json.dumps({"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}),
        encoding="utf-8",
    )
    (run_dir / "invocation.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-30T17:09:15.697374+00:00",
                "finished_at_utc": "2026-04-30T17:22:54.631787+00:00",
                "parsed_args": {
                    "epochs": 40,
                    "output_dir": str(tmp_path),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model.pt").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_compare_wrapper._count_torch_state_dict_parameters",
        lambda path: 123,
    )

    payload = _recover_torch_row_payload(
        output_dir=tmp_path,
        model_id="supervised_ffno",
        n_value=128,
        metrics={},
    )

    assert payload["runtime_summary"]["recovered_from_existing_artifacts"] is False
    assert payload["runtime_summary"]["row_payload_rebuilt_from_row_artifacts"] is True
    assert payload["runtime_summary"]["command_wall_time_sec"] == pytest.approx(818.934413)
    assert "recovered_from_existing_artifacts" not in payload["caveats"]
    assert "row_payload_rebuilt_from_row_artifacts" in payload["caveats"]


def test_enrich_paper_row_payload_recovers_missing_direct_runner_logs(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import (
        _enrich_paper_row_payload,
        _recover_torch_row_payload,
    )
    from scripts.studies.metrics_tables import _missing_paper_fields

    run_dir = tmp_path / "runs" / "supervised_ffno"
    run_dir.mkdir(parents=True, exist_ok=True)
    recons_dir = tmp_path / "recons" / "supervised_ffno"
    recons_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = tmp_path / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    probe_npz = tmp_path / "probe.npz"
    for path in (train_npz, test_npz, probe_npz):
        path.write_bytes(b"stub")

    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(recons_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
    (visuals_dir / "amp_phase_supervised_ffno.png").write_bytes(b"png")
    (visuals_dir / "amp_phase_error_supervised_ffno.png").write_bytes(b"png")

    (run_dir / "history.json").write_text(
        json.dumps({"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}),
        encoding="utf-8",
    )
    (run_dir / "invocation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pid": 12345,
                "exit_code": 0,
                "timestamp_utc": "2026-04-30T17:09:15.697374+00:00",
                "finished_at_utc": "2026-04-30T17:22:54.631787+00:00",
                "parsed_args": {
                    "epochs": 40,
                    "seed": 3,
                    "output_dir": str(tmp_path),
                    "train_npz": str(train_npz),
                    "test_npz": str(test_npz),
                },
                "extra": {
                    "git_commit": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "invocation.sh").write_text("python grid_lines_torch_runner.py\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(_full_pair_metrics(0.01, 0.02)), encoding="utf-8")
    (run_dir / "model.pt").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_compare_wrapper._count_torch_state_dict_parameters",
        lambda path: 123,
    )

    recovered = _recover_torch_row_payload(
        output_dir=tmp_path,
        model_id="supervised_ffno",
        n_value=128,
        metrics=_full_pair_metrics(0.01, 0.02),
    )
    enriched = _enrich_paper_row_payload(
        model_id="supervised_ffno",
        payload=recovered,
        output_dir=tmp_path,
        train_npz=train_npz,
        test_npz=test_npz,
        seed=3,
        nimgs_train=2,
        nimgs_test=2,
        gridsize=1,
        set_phi=True,
        probe_npz=probe_npz,
        dataset_source="synthetic_lines",
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
    )

    assert enriched["row_status"] == "paper_grade"
    assert "row_output_logs_recovered_from_invocation" in enriched["caveats"]
    assert enriched["outputs"]["exit_code_proof_json"] == "runs/supervised_ffno/exit_code_proof.json"
    assert (run_dir / "stdout.log").exists()
    assert (run_dir / "stderr.log").exists()
    assert "Recovered row log placeholder" in (run_dir / "stdout.log").read_text(encoding="utf-8")
    assert _missing_paper_fields(
        enriched,
        model_id="supervised_ffno",
        require_row_provenance=True,
        output_dir=tmp_path,
    ) == []


def test_enrich_paper_row_payload_respects_locked_decision_support_status(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import (
        _enrich_paper_row_payload,
        _recover_torch_row_payload,
    )

    run_dir = tmp_path / "runs" / "supervised_ffno"
    recons_dir = tmp_path / "recons" / "supervised_ffno"
    visuals_dir = tmp_path / "visuals"
    run_dir.mkdir(parents=True)
    recons_dir.mkdir(parents=True)
    visuals_dir.mkdir(parents=True)

    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    probe_npz = tmp_path / "probe.npz"
    for path in (train_npz, test_npz, probe_npz):
        path.write_bytes(b"stub")

    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(recons_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
    (visuals_dir / "amp_phase_supervised_ffno.png").write_bytes(b"png")
    (visuals_dir / "amp_phase_error_supervised_ffno.png").write_bytes(b"png")

    (run_dir / "history.json").write_text(
        json.dumps({"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}),
        encoding="utf-8",
    )
    (run_dir / "invocation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pid": 12345,
                "exit_code": 0,
                "timestamp_utc": "2026-04-30T17:09:15.697374+00:00",
                "finished_at_utc": "2026-04-30T17:22:54.631787+00:00",
                "parsed_args": {
                    "epochs": 40,
                    "seed": 3,
                    "output_dir": str(tmp_path),
                    "train_npz": str(train_npz),
                    "test_npz": str(test_npz),
                },
                "extra": {
                    "git_commit": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "invocation.sh").write_text("python grid_lines_torch_runner.py\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(_full_pair_metrics(0.01, 0.02)), encoding="utf-8")
    (run_dir / "model.pt").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_compare_wrapper._count_torch_state_dict_parameters",
        lambda path: 123,
    )

    recovered = _recover_torch_row_payload(
        output_dir=tmp_path,
        model_id="supervised_ffno",
        n_value=128,
        metrics=_full_pair_metrics(0.01, 0.02),
    )
    enriched = _enrich_paper_row_payload(
        model_id="supervised_ffno",
        payload=recovered,
        output_dir=tmp_path,
        train_npz=train_npz,
        test_npz=test_npz,
        seed=3,
        nimgs_train=2,
        nimgs_test=2,
        gridsize=1,
        set_phi=True,
        probe_npz=probe_npz,
        dataset_source="synthetic_lines",
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
        row_spec={"row_status": "decision_support", "lock_row_status": True},
    )

    assert enriched["row_status"] == "decision_support"


def test_compare_wrapper_script_path_bootstraps_repo_imports():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/studies/grid_lines_compare_wrapper.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ModuleNotFoundError" not in result.stderr


def test_wrapper_reuse_path_does_not_attribute_tf_runtime_to_wrapper_repair_pass(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    for model_id in ("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"):
        recon_path = tmp_path / "recons" / model_id / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    (tmp_path / "baseline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "baseline" / "baseline.keras").write_text("stub", encoding="utf-8")
    (tmp_path / "pinn").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pinn" / "wts.h5.zip").write_text("stub", encoding="utf-8")
    (tmp_path / "live_stdout.log").write_text(
        "\n".join(
            [
                "Epoch 40/40",
                "loss: 13.2960 - pred_intensity_loss: -1833342.8750",
                "conv2d_12_loss: 0.0148 - conv2d_19_loss: 0.1235 - loss: 0.1383 - val_loss: 0.1379",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "invocation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "timestamp_utc": "2026-04-30T11:24:58.708304+00:00",
                "finished_at_utc": "2026-04-30T11:25:11.467518+00:00",
                "parsed_args": {"reuse_existing_recons": True},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "launcher_stderr.log").write_text(
        "\n".join(
            [
                f"Saved artifacts to {tmp_path / 'runs' / 'pinn_hybrid_resnet'}",
                f"Torch runner complete. Artifacts in {tmp_path / 'runs' / 'pinn_hybrid_resnet'}",
                f"Saved artifacts to {tmp_path / 'runs' / 'pinn_fno_vanilla'}",
                f"Torch runner complete. Artifacts in {tmp_path / 'runs' / 'pinn_fno_vanilla'}",
            ]
        ),
        encoding="utf-8",
    )

    tf_row_invocations = {
        "baseline": {
            "timestamp_utc": "2026-04-30T09:05:32.963983+00:00",
            "finished_at_utc": "2026-04-30T09:05:32.964175+00:00",
        },
        "pinn": {
            "timestamp_utc": "2026-04-30T08:55:15.794780+00:00",
            "finished_at_utc": "2026-04-30T08:55:15.795060+00:00",
        },
    }
    for model_id, timing in tf_row_invocations.items():
        run_dir = tmp_path / "runs" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "exit_code": 0,
                    "timestamp_utc": timing["timestamp_utc"],
                    "finished_at_utc": timing["finished_at_utc"],
                    "extra": {
                        "invocation_mode": "library",
                        "row_model_id": model_id,
                        "shared_root_output_dir": str(tmp_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "history.json").write_text("{}", encoding="utf-8")

    for model_id in ("pinn_hybrid_resnet", "pinn_fno_vanilla"):
        run_dir = tmp_path / "runs" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "history.json").write_text(
            json.dumps({"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "timestamp_utc": "2026-04-29T21:00:00+00:00",
                    "finished_at_utc": "2026-04-29T21:00:05+00:00",
                    "parsed_args": {"epochs": 40},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "model.pt").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: _full_pair_metrics(0.01, 0.03),
    )

    class _FakeModel:
        def __init__(self, count):
            self._count = count

        def count_params(self):
            return self._count

    monkeypatch.setattr(
        "tensorflow.keras.models.load_model",
        lambda path, compile=False: _FakeModel(10),
    )
    monkeypatch.setattr(
        "ptycho.model_manager.ModelManager.load_multiple_models",
        lambda path, model_names=None: {"autoencoder": _FakeModel(11)},
    )
    monkeypatch.setattr(
        "scripts.studies.grid_lines_compare_wrapper._count_torch_state_dict_parameters",
        lambda path: 12 if "hybrid" in str(path) else 13,
    )

    result = run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(),
        models=("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"),
        model_n={
            "baseline": 128,
            "pinn": 128,
            "pinn_hybrid_resnet": 128,
            "pinn_fno_vanilla": 128,
        },
        probe_npz=Path("dummy_probe.npz"),
        reuse_existing_recons=True,
    )

    for model_id in ("baseline", "pinn"):
        runtime_summary = result["row_payloads"][model_id]["runtime_summary"]
        assert runtime_summary["recovered_from_existing_artifacts"] is True
        assert runtime_summary["runtime_source"] == "unavailable_under_recovery"
        assert "command_wall_time_sec" not in runtime_summary
        assert "wrapper bundle-repair pass" in runtime_summary["runtime_unavailable_reason"]


def test_wrapper_accepts_architecture_list(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "cnn,fno",
    ])
    assert args.architectures == ("cnn", "fno")


def test_wrapper_parse_args_rejects_single_image_frc_controls(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    with pytest.raises(SystemExit):
        parse_args([
            "--N", "64",
            "--gridsize", "1",
            "--output-dir", str(tmp_path),
            "--torch-no-single-image-frc",
        ])


def test_parse_args_default_architectures_excludes_baseline(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert "baseline" not in args.architectures


def test_wrapper_keeps_architectures_backward_compat(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "cnn,baseline,fno",
    ])
    assert args.architectures == ("cnn", "baseline", "fno")


def test_wrapper_accepts_models_and_model_n(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--models", "pinn_hybrid,pinn_ptychovit",
        "--model-n", "pinn_hybrid=128,pinn_ptychovit=256",
    ])
    assert args.models == ("pinn_hybrid", "pinn_ptychovit")
    assert args.model_n["pinn_hybrid"] == 128
    assert args.model_n["pinn_ptychovit"] == 256


def test_parse_args_accepts_external_raw_npz_mode(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--dataset-source",
            "external_raw_npz",
            "--train-data",
            "datasets/fly64/fly001_64_train_converted.npz",
            "--test-data",
            "datasets/fly64/fly001_64_train_converted.npz",
            "--models",
            "pinn_hybrid_resnet",
        ]
    )
    assert args.dataset_source == "external_raw_npz"
    assert args.train_data.name.endswith(".npz")
    assert args.test_data.name.endswith(".npz")


def test_parse_args_external_raw_requires_train_and_test_data(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--N",
                "64",
                "--gridsize",
                "1",
                "--output-dir",
                str(tmp_path),
                "--dataset-source",
                "external_raw_npz",
                "--models",
                "pinn_hybrid_resnet",
            ]
        )


def test_parse_args_accepts_position_reassembly_strategy_flags(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args(
        [
            "--N",
            "128",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--dataset-source",
            "external_raw_npz",
            "--train-data",
            "train.npz",
            "--test-data",
            "test.npz",
            "--torch-position-reassembly-backend",
            "batched",
            "--torch-position-reassembly-batch-size",
            "32",
        ]
    )
    assert args.torch_position_reassembly_backend == "batched"
    assert args.torch_position_reassembly_batch_size == 32


def test_external_raw_rejects_tf_and_ptychovit_models(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    with pytest.raises(ValueError, match="external_raw_npz.*Torch model IDs"):
        run_grid_lines_compare(
            N=64,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=Path("dummy_probe.npz"),
            architectures=("cnn",),
            models=("pinn",),
            dataset_source="external_raw_npz",
            train_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
            test_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
        )


def test_parse_args_reuse_existing_recons_defaults_false(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
    ])
    assert args.reuse_existing_recons is False


def test_parse_args_reuse_existing_recons_flag_sets_true(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--reuse-existing-recons",
    ])
    assert args.reuse_existing_recons is True


def test_wrapper_rejects_ptychovit_non_256():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    with pytest.raises(ValueError, match="pinn_ptychovit.*N=256"):
        validate_model_specs(
            models=("pinn_ptychovit",),
            model_n={"pinn_ptychovit": 128},
        )


def test_compute_required_ns_from_models_and_model_n():
    from scripts.studies.grid_lines_compare_wrapper import compute_required_ns

    required = compute_required_ns(
        models=("pinn_hybrid", "pinn_ptychovit"),
        model_n={"pinn_hybrid": 128, "pinn_ptychovit": 256},
        default_n=128,
    )
    assert required == [128, 256]


def test_resolve_model_ns_defaults_ptychovit_to_256():
    from scripts.studies.grid_lines_compare_wrapper import resolve_model_ns

    resolved = resolve_model_ns(
        models=("pinn_ptychovit",),
        model_n_overrides={},
        default_n=128,
    )
    assert resolved["pinn_ptychovit"] == 256


def test_wrapper_models_mode_honors_tf_model_n(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    train_64 = tmp_path / "datasets" / "N64" / "gs1" / "train.npz"
    test_64 = tmp_path / "datasets" / "N64" / "gs1" / "test.npz"
    for path in (train_64, test_64):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ok=np.array([1], dtype=np.int8))

    def fake_build_by_n(base_cfg, required_ns):
        _ = base_cfg
        assert sorted(required_ns) == [64]
        return {
            64: {"train_npz": str(train_64), "test_npz": str(test_64), "gt_recon": str(gt_recon), "tag": "N64"},
        }

    captured = {}

    def fake_tf_run(cfg):
        captured["N"] = cfg.N
        recon_path = cfg.output_dir / "recons" / "pinn" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"train_npz": str(train_64), "test_npz": str(test_64), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label, **kwargs: {"mse": float(np.mean(np.abs(pred - gt)))})

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        models=("pinn",),
        model_n={"pinn": 64},
        probe_npz=Path("dummy_probe.npz"),
    )
    assert captured["N"] == 64


def test_wrapper_writes_metrics_by_model_for_selected_models(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    train_128 = tmp_path / "datasets" / "N128" / "gs1" / "train.npz"
    test_128 = tmp_path / "datasets" / "N128" / "gs1" / "test.npz"
    train_256 = tmp_path / "datasets" / "N256" / "gs1" / "train.npz"
    test_256 = tmp_path / "datasets" / "N256" / "gs1" / "test.npz"
    for path in (train_128, test_128, train_256, test_256):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ok=np.array([1], dtype=np.int8))

    def fake_build_by_n(base_cfg, required_ns):
        _ = base_cfg
        assert sorted(required_ns) == [128, 256]
        return {
            128: {"train_npz": str(train_128), "test_npz": str(test_128), "gt_recon": str(gt_recon), "tag": "N128"},
            256: {"train_npz": str(train_256), "test_npz": str(test_256), "gt_recon": str(gt_recon), "tag": "N256"},
        }

    def fake_torch_run(cfg):
        recon_path = cfg.output_dir / "recons" / "pinn_hybrid" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    def fake_ptychovit_run(cfg):
        _ = cfg
        recon_path = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((256, 256)) + 1j * np.ones((256, 256))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "status": "ok"}

    def fake_convert(npz_path, out_dir, object_name, pixel_size_m=1.0):
        from ptycho.interop.ptychovit.contracts import PtychoViTHdf5Pair

        _ = (npz_path, object_name, pixel_size_m)
        out_dir.mkdir(parents=True, exist_ok=True)
        dp = out_dir / "x_dp.hdf5"
        para = out_dir / "x_para.hdf5"
        with h5py.File(dp, "w") as dp_file:
            dp_file.create_dataset("dp", data=np.ones((2, 8, 8), dtype=np.float32))
        with h5py.File(para, "w") as para_file:
            obj = para_file.create_dataset("object", data=np.ones((1, 16, 16), dtype=np.complex64))
            obj.attrs["pixel_height_m"] = 1.0
            obj.attrs["pixel_width_m"] = 1.0
            probe = para_file.create_dataset("probe", data=np.ones((1, 1, 8, 8), dtype=np.complex64))
            probe.attrs["pixel_height_m"] = 1.0
            probe.attrs["pixel_width_m"] = 1.0
            para_file.create_dataset("probe_position_x_m", data=np.zeros(2, dtype=np.float64))
            para_file.create_dataset("probe_position_y_m", data=np.zeros(2, dtype=np.float64))
        return PtychoViTHdf5Pair(dp_hdf5=dp, para_hdf5=para, object_name="x")

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("scripts.studies.grid_lines_ptychovit_runner.run_grid_lines_ptychovit", fake_ptychovit_run)
    monkeypatch.setattr("ptycho.interop.ptychovit.convert.convert_npz_split_to_hdf5_pair", fake_convert)
    monkeypatch.setattr("ptycho.interop.ptychovit.validate.validate_hdf5_pair", lambda dp_path, para_path: None)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label, **kwargs: {"mse": float(np.mean(np.abs(pred - gt)))})

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid", "pinn_ptychovit"),
        model_n={"pinn_hybrid": 128, "pinn_ptychovit": 256},
        probe_npz=Path("dummy_probe.npz"),
    )
    metrics = json.loads((tmp_path / "metrics_by_model.json").read_text())
    assert "pinn_hybrid" in metrics
    assert "pinn_ptychovit" in metrics
    mse_val = metrics["pinn_hybrid"]["metrics"]["mse"]
    if isinstance(mse_val, list):
        mse_val = mse_val[0]
    assert isinstance(mse_val, float)
    table_text = (tmp_path / "metrics_table.tex").read_text()
    assert "128" in table_text
    assert "256" in table_text


def test_wrapper_does_not_reuse_precomputed_recons_by_default(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    stale_recon = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    stale_recon.parent.mkdir(parents=True, exist_ok=True)
    stale = (np.zeros((392, 392)) + 1j * np.zeros((392, 392))).astype(np.complex64)
    np.savez(stale_recon, YY_pred=stale, amp=np.abs(stale), phase=np.angle(stale))

    train_128 = tmp_path / "datasets" / "N128" / "gs1" / "train.npz"
    test_128 = tmp_path / "datasets" / "N128" / "gs1" / "test.npz"
    for path in (train_128, test_128):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ok=np.array([1], dtype=np.int8))

    called = {"build": 0, "torch": 0}

    def fake_build_by_n(base_cfg, required_ns):
        _ = base_cfg
        assert sorted(required_ns) == [128]
        called["build"] += 1
        return {
            128: {"train_npz": str(train_128), "test_npz": str(test_128), "gt_recon": str(gt_recon), "tag": "N128"},
        }

    def fake_torch_run(cfg):
        called["torch"] += 1
        recon_path = cfg.output_dir / "recons" / "pinn_hybrid" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: {"mse": float(np.mean(np.abs(pred - gt)))},
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid",),
        model_n={"pinn_hybrid": 128},
        probe_npz=Path("dummy_probe.npz"),
    )
    assert called["build"] == 1
    assert called["torch"] == 1


def test_wrapper_reuses_precomputed_recons_when_opted_in(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_recon = tmp_path / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_recon, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    recon_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    called = {"build": 0, "torch": 0}

    def fake_build_by_n(base_cfg, required_ns):
        _ = (base_cfg, required_ns)
        called["build"] += 1
        raise AssertionError("build_grid_lines_datasets_by_n should not run when reusing precomputed artifacts")

    def fake_torch_run(cfg):
        _ = cfg
        called["torch"] += 1
        raise AssertionError("run_grid_lines_torch should not run when reusing precomputed artifacts")

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fake_build_by_n)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr(
        "ptycho.evaluation.eval_reconstruction",
        lambda pred, gt, label, **kwargs: {"mse": float(np.mean(np.abs(pred - gt)))},
    )

    run_grid_lines_compare(
        N=128,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid",),
        models=("pinn_hybrid",),
        model_n={"pinn_hybrid": 128},
        probe_npz=Path("dummy_probe.npz"),
        reuse_existing_recons=True,
    )
    assert called["build"] == 0
    assert called["torch"] == 0
    assert not (tmp_path / "runs" / "pinn_hybrid" / "stdout.log").exists()
    assert not (tmp_path / "runs" / "pinn_hybrid" / "exit_code_proof.json").exists()


def test_harmonized_metrics_run_on_canonical_gt_grid(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import evaluate_selected_models

    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((392, 392)) + 1j * np.ones((392, 392))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    pred_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(pred_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    out = evaluate_selected_models({"pinn_hybrid": pred_path}, gt_path)
    assert out["pinn_hybrid"]["reference_shape"] == [392, 392]


def test_evaluate_selected_models_does_not_pass_single_image_frc(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import evaluate_selected_models

    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    pred_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(pred_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    captured = {}

    def fake_eval(pred_obj, gt_obj, label="", **kwargs):
        _ = (pred_obj, gt_obj, label)
        captured["kwargs"] = dict(kwargs)
        return {"mse": 0.0}

    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

    _ = evaluate_selected_models({"pinn_hybrid": pred_path}, gt_path)
    assert not any(key.startswith("single_image_frc") for key in captured["kwargs"])


def test_evaluate_selected_models_omits_binomial_single_image_frc_metrics(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import evaluate_selected_models

    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    pred_path = tmp_path / "recons" / "pinn_hybrid" / "recon.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(pred_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))

    def fake_eval(pred_obj, gt_obj, label="", **kwargs):
        _ = (pred_obj, gt_obj, label, kwargs)
        return {"mse": 0.0}

    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", fake_eval)

    out = evaluate_selected_models({"pinn_hybrid": pred_path}, gt_path)
    metrics = out["pinn_hybrid"]["metrics"]
    assert "single_frc50_binomial" not in metrics
    assert "single_frc1over7_binomial" not in metrics


def test_wrapper_defaults_torch_loss_mode_mae(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
    ])
    assert getattr(args, "torch_loss_mode", None) == "mae"
    assert getattr(args, "torch_mae_pred_l2_match_target", None) is False


def test_wrapper_passes_torch_loss_mode_to_runner(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    captured = {}

    def fake_torch_run(cfg):
        captured["torch_loss_mode"] = getattr(cfg, "torch_loss_mode", None)
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["torch_loss_mode"] == "mae"


def test_wrapper_passes_torch_mae_pred_l2_match_target_to_runner(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    captured = {}

    def fake_torch_run(cfg):
        captured["torch_mae_pred_l2_match_target"] = getattr(cfg, "torch_mae_pred_l2_match_target", None)
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
        torch_mae_pred_l2_match_target=True,
    )

    assert captured["torch_mae_pred_l2_match_target"] is True


def test_wrapper_does_not_pass_single_image_frc_controls_to_torch_runner(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    captured = {}

    def fake_torch_run(cfg):
        captured["has_single_image_frc"] = hasattr(cfg, "single_image_frc")
        captured["has_split_mode"] = hasattr(cfg, "single_image_frc_split_mode")
        captured["has_rng_seed"] = hasattr(cfg, "single_image_frc_rng_seed")
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["has_single_image_frc"] is False
    assert captured["has_split_mode"] is False
    assert captured["has_rng_seed"] is False


def test_wrapper_cnn_only_excludes_baseline_metrics(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(
            json.dumps({"pinn": {"mse": 0.1}, "baseline": {"mse": 0.2}})
        )
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
        }

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        probe_npz=Path("dummy_probe.npz"),
    )
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn" in merged
    assert "baseline" not in merged


def test_wrapper_torch_only_uses_dataset_builder_not_tf_workflow(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    called = {"build_datasets": False, "tf_workflow": False}

    def fake_build_grid_lines_datasets(cfg):
        called["build_datasets"] = True
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        train_npz = datasets_dir / "train.npz"
        test_npz = datasets_dir / "test.npz"
        train_npz.write_bytes(b"stub")
        test_npz.write_bytes(b"stub")
        gt_dir = cfg.output_dir / "recons" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(gt_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
        return {"train_npz": str(train_npz), "test_npz": str(test_npz), "gt_recon": str(gt_dir / "recon.npz")}

    def fake_tf_workflow(cfg):
        called["tf_workflow"] = True
        raise AssertionError("Torch-only run should not invoke TF training workflow")

    def fake_torch_run(cfg):
        recon_dir = cfg.output_dir / "recons" / f"pinn_{cfg.architecture}"
        recon_dir.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_dir / "recon.npz", YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"metrics": {"mse": 0.3}, "recon_npz": str(recon_dir / "recon.npz")}

    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets",
        fake_build_grid_lines_datasets,
    )
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_workflow)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr(
        "ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals",
        lambda output_dir, order: {},
    )

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert called["build_datasets"] is True
    assert called["tf_workflow"] is False


def test_wrapper_passes_grad_clip_algorithm(monkeypatch, tmp_path):
    """Test that --torch-grad-clip-algorithm threads through to TorchRunnerConfig."""
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args default
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
    ])
    assert args.torch_grad_clip_algorithm == "norm"

    # Test parse_args with explicit value
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--torch-grad-clip-algorithm", "agc",
    ])
    assert args.torch_grad_clip_algorithm == "agc"

    # Test end-to-end passthrough to TorchRunnerConfig
    captured = {}

    def fake_torch_run(cfg):
        captured["gradient_clip_algorithm"] = cfg.gradient_clip_algorithm
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("fno",),
        probe_npz=Path("dummy_probe.npz"),
        torch_gradient_clip_algorithm="agc",
    )

    assert captured["gradient_clip_algorithm"] == "agc"


def test_wrapper_renders_visuals(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({"pinn": {}, "baseline": {}}))
        return {
            "train_npz": str(datasets_dir / "train.npz"),
            "test_npz": str(datasets_dir / "test.npz"),
        }

    def fake_torch_run(cfg):
        recon_dir = cfg.output_dir / "recons" / f"pinn_{cfg.architecture}"
        recon_dir.mkdir(parents=True, exist_ok=True)
        (recon_dir / "recon.npz").write_bytes(b"stub")
        return {"metrics": {"mse": 0.3}}

    called = {}

    def fake_render(output_dir, order):
        called["order"] = order
        visuals = output_dir / "visuals"
        visuals.mkdir(parents=True, exist_ok=True)
        out = visuals / "compare_amp_phase.png"
        out.write_bytes(b"stub")
        return {"compare": str(out)}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", fake_render)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn", "baseline", "fno"),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert called["order"] == ("gt", "pinn", "baseline", "pinn_fno")


def test_wrapper_handles_stable_hybrid(monkeypatch, tmp_path):
    """Test that stable_hybrid is wired through to Torch runner and merged metrics.

    Task ID: FNO-STABILITY-OVERHAUL-001 Task 2.3
    """
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args accepts 'stable_hybrid'
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", "stable_hybrid",
    ])
    assert "stable_hybrid" in args.architectures

    # Test end-to-end: stable_hybrid invokes torch runner and merges metrics
    captured = {}

    def fake_torch_run(cfg):
        captured["architecture"] = cfg.architecture
        return {"metrics": {"mse": 0.25}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("stable_hybrid",),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["architecture"] == "stable_hybrid"
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert "pinn_stable_hybrid" in merged


@pytest.mark.parametrize(
    "arch,metric_key",
    [
        ("ffno", "pinn_ffno"),
        ("fno_vanilla", "pinn_fno_vanilla"),
        ("hybrid_resnet", "pinn_hybrid_resnet"),
    ],
)
def test_wrapper_handles_new_architectures(monkeypatch, tmp_path, arch, metric_key):
    """Test that new torch architectures are wired through and merged."""
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", arch,
    ])
    assert arch in args.architectures

    captured = {}

    def fake_torch_run(cfg):
        captured["architecture"] = cfg.architecture
        return {"metrics": {"mse": 0.25}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=(arch,),
        probe_npz=Path("dummy_probe.npz"),
    )

    assert captured["architecture"] == arch
    merged = json.loads((tmp_path / "metrics.json").read_text())
    assert metric_key in merged

def test_wrapper_passes_max_hidden_channels():
    """--torch-max-hidden-channels 512 propagates to the mocked runner call."""
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1",
        "--output-dir", "/tmp/test_out",
        "--architectures", "hybrid",
        "--torch-max-hidden-channels", "512",
    ])
    assert args.torch_max_hidden_channels == 512


def test_wrapper_accepts_resnet_width():
    """--torch-resnet-width propagates to parsed args."""
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1",
        "--output-dir", "/tmp/test_out",
        "--architectures", "hybrid_resnet",
        "--torch-resnet-width", "256",
    ])
    assert args.torch_resnet_width == 256


def test_wrapper_accepts_plateau_params(tmp_path):
    """Test ReduceLROnPlateau args parse in grid_lines_compare_wrapper."""
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "hybrid",
        "--torch-scheduler", "ReduceLROnPlateau",
        "--torch-plateau-factor", "0.25",
        "--torch-plateau-patience", "5",
        "--torch-plateau-min-lr", "1e-5",
        "--torch-plateau-threshold", "1e-3",
    ])
    assert args.torch_scheduler == "ReduceLROnPlateau"
    assert args.torch_plateau_factor == 0.25
    assert args.torch_plateau_patience == 5
    assert args.torch_plateau_min_lr == 1e-5
    assert args.torch_plateau_threshold == 1e-3


def test_wrapper_passes_optimizer(monkeypatch, tmp_path):
    """Test --torch-optimizer and related flags thread through to runner.

    Task ID: FNO-STABILITY-OVERHAUL-001 Phase 8 Task 1
    """
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", "stable_hybrid",
        "--torch-optimizer", "sgd",
        "--torch-momentum", "0.9",
        "--torch-weight-decay", "0.01",
        "--torch-beta1", "0.9",
        "--torch-beta2", "0.999",
    ])
    assert args.torch_optimizer == "sgd"
    assert args.torch_momentum == 0.9
    assert args.torch_weight_decay == 0.01

    # Test end-to-end passthrough
    captured = {}

    def fake_torch_run(cfg):
        captured["optimizer"] = cfg.optimizer
        captured["momentum"] = cfg.momentum
        captured["weight_decay"] = cfg.weight_decay
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("stable_hybrid",),
        probe_npz=Path("dummy_probe.npz"),
        torch_optimizer="sgd",
        torch_momentum=0.9,
        torch_weight_decay=0.01,
    )

    assert captured["optimizer"] == "sgd"
    assert captured["momentum"] == 0.9
    assert captured["weight_decay"] == 0.01


def test_wrapper_passes_scheduler_knobs(monkeypatch, tmp_path):
    """Test --torch-scheduler and related flags thread through to runner."""
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    # Test parse_args
    args = parse_args([
        "--N", "64", "--gridsize", "1", "--output-dir", str(tmp_path),
        "--architectures", "stable_hybrid",
        "--torch-scheduler", "WarmupCosine",
        "--torch-lr-warmup-epochs", "5",
        "--torch-lr-min-ratio", "0.05",
    ])
    assert args.torch_scheduler == "WarmupCosine"
    assert args.torch_lr_warmup_epochs == 5
    assert args.torch_lr_min_ratio == 0.05

    # Test end-to-end passthrough
    captured = {}

    def fake_torch_run(cfg):
        captured["scheduler"] = cfg.scheduler
        captured["lr_warmup_epochs"] = cfg.lr_warmup_epochs
        captured["lr_min_ratio"] = cfg.lr_min_ratio
        return {"metrics": {"mse": 0.3}}

    _mock_dataset_builder(monkeypatch)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("stable_hybrid",),
        probe_npz=Path("dummy_probe.npz"),
        torch_scheduler="WarmupCosine",
        torch_lr_warmup_epochs=5,
        torch_lr_min_ratio=0.05,
    )

    assert captured["scheduler"] == "WarmupCosine"
    assert captured["lr_warmup_epochs"] == 5
    assert captured["lr_min_ratio"] == 0.05


def test_external_raw_mode_uses_shared_builder_not_synthetic_builder(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    called = {"shared_builder": False, "synthetic_builder": False}
    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))

    train_npz = tmp_path / "datasets" / "N64" / "gs1" / "train.npz"
    test_npz = tmp_path / "datasets" / "N64" / "gs1" / "test.npz"
    for path in (train_npz, test_npz):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, diffraction=np.ones((2, 64, 64, 1), dtype=np.float32))

    def fake_shared_builder(*, dataset_source, cfg, required_ns, train_data=None, test_data=None, **kwargs):
        called["shared_builder"] = True
        assert dataset_source == "external_raw_npz"
        assert train_data is not None
        assert test_data is not None
        assert sorted(required_ns) == [64]
        return {
            64: {
                "train_npz": str(train_npz),
                "test_npz": str(test_npz),
                "gt_recon": str(gt_path),
                "tag": "N64",
            }
        }

    def fail_synthetic_builder(*args, **kwargs):
        called["synthetic_builder"] = True
        raise AssertionError("synthetic builder should not run in external_raw_npz mode")

    def fake_torch_run(cfg):
        recon_path = tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("scripts.studies.grid_study_dataset_builder.build_datasets", fake_shared_builder)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n", fail_synthetic_builder)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label, **kwargs: {"mse": 0.0})

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid_resnet",),
        models=("pinn_hybrid_resnet",),
        probe_npz=Path("dummy_probe.npz"),
        dataset_source="external_raw_npz",
        train_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
        test_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
    )
    assert called["shared_builder"] is True
    assert called["synthetic_builder"] is False


def test_external_raw_mode_sets_torch_reassembly_mode_position(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
    train_npz = tmp_path / "datasets" / "N64" / "gs1" / "train.npz"
    test_npz = tmp_path / "datasets" / "N64" / "gs1" / "test.npz"
    for path in (train_npz, test_npz):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, diffraction=np.ones((2, 64, 64, 1), dtype=np.float32))

    monkeypatch.setattr(
        "scripts.studies.grid_study_dataset_builder.build_datasets",
        lambda **kwargs: {
            64: {
                "train_npz": str(train_npz),
                "test_npz": str(test_npz),
                "gt_recon": str(gt_path),
                "tag": "N64",
            }
        },
    )
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label, **kwargs: {"mse": 0.0})

    captured = {}

    def fake_torch_run(cfg):
        captured["reassembly_mode"] = cfg.reassembly_mode
        recon_path = tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid_resnet",),
        models=("pinn_hybrid_resnet",),
        probe_npz=Path("dummy_probe.npz"),
        dataset_source="external_raw_npz",
        train_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
        test_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
    )
    assert captured["reassembly_mode"] == "position"


def test_external_mode_passes_position_strategy_to_torch_runner(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    gt_path = tmp_path / "recons" / "gt" / "recon.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
    np.savez(gt_path, YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
    train_npz = tmp_path / "datasets" / "N64" / "gs1" / "train.npz"
    test_npz = tmp_path / "datasets" / "N64" / "gs1" / "test.npz"
    for path in (train_npz, test_npz):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, diffraction=np.ones((2, 64, 64, 1), dtype=np.float32))

    monkeypatch.setattr(
        "scripts.studies.grid_study_dataset_builder.build_datasets",
        lambda **kwargs: {
            64: {
                "train_npz": str(train_npz),
                "test_npz": str(test_npz),
                "gt_recon": str(gt_path),
                "tag": "N64",
            }
        },
    )
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", lambda output_dir, order: {})
    monkeypatch.setattr("ptycho.evaluation.eval_reconstruction", lambda pred, gt, label, **kwargs: {"mse": 0.0})

    captured = {}

    def fake_torch_run(cfg):
        captured["backend"] = cfg.position_reassembly_backend
        captured["batch_size"] = cfg.position_reassembly_batch_size
        recon_path = tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz"
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        pred = (np.ones((64, 64)) + 1j * np.ones((64, 64))).astype(np.complex64)
        np.savez(recon_path, YY_pred=pred, amp=np.abs(pred), phase=np.angle(pred))
        return {"recon_npz": str(recon_path), "metrics": {"mse": 0.1}}

    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("hybrid_resnet",),
        models=("pinn_hybrid_resnet",),
        probe_npz=Path("dummy_probe.npz"),
        dataset_source="external_raw_npz",
        train_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
        test_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
        torch_position_reassembly_backend="batched",
        torch_position_reassembly_batch_size=32,
    )
    assert captured["backend"] == "batched"
    assert captured["batch_size"] == 32


def test_wrapper_accepts_plateau_scheduler(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "hybrid",
        "--torch-scheduler", "ReduceLROnPlateau",
    ])
    assert args.torch_scheduler == "ReduceLROnPlateau"


def test_compare_wrapper_probe_mask_diameter_passthrough(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare, parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--probe-mask-diameter", "64",
    ])
    assert args.probe_mask_diameter == 64

    captured = {}

    def fake_tf_run(cfg):
        captured["probe_mask_diameter"] = cfg.probe_mask_diameter
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text("{}")
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn",),
        probe_npz=Path("dummy_probe.npz"),
        probe_mask_diameter=64,
    )

    assert captured["probe_mask_diameter"] == 64


def test_main_writes_cli_invocation_artifacts(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_compare_wrapper as wrapper

    called = {"run": False}

    def fake_run_grid_lines_compare(**kwargs):
        called["run"] = True
        return {"metrics": {}}

    monkeypatch.setattr(wrapper, "run_grid_lines_compare", fake_run_grid_lines_compare)
    monkeypatch.setattr(
        "scripts.studies.invocation_logging.capture_runtime_provenance",
        lambda: {"python_executable": "/usr/bin/python3", "pythonpath": "/tmp/session_repo"},
    )
    monkeypatch.setattr("scripts.studies.invocation_logging.get_git_commit", lambda repo_root=None: "abc123")

    wrapper.main(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--architectures",
            "cnn",
        ]
    )

    assert called["run"] is True
    inv_json = tmp_path / "invocation.json"
    inv_sh = tmp_path / "invocation.sh"
    assert inv_json.exists()
    assert inv_sh.exists()
    payload = json.loads(inv_json.read_text())
    assert "grid_lines_compare_wrapper.py" in payload["command"]
    assert "--architectures" in payload["argv"]
    assert payload["extra"]["runtime_provenance"]["python_executable"] == "/usr/bin/python3"
    assert payload["extra"]["git_commit"] == "abc123"
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["finished_at_utc"]


def test_main_writes_root_launcher_logs(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_compare_wrapper as wrapper

    def fake_run_grid_lines_compare(**kwargs):
        print("wrapper stdout sentinel")
        print("wrapper stderr sentinel", file=sys.stderr)
        return {"metrics": {}}

    monkeypatch.setattr(wrapper, "run_grid_lines_compare", fake_run_grid_lines_compare)
    monkeypatch.setattr(
        "scripts.studies.invocation_logging.capture_runtime_provenance",
        lambda: {"python_executable": "/usr/bin/python3", "pythonpath": "/tmp/session_repo"},
    )
    monkeypatch.setattr("scripts.studies.invocation_logging.get_git_commit", lambda repo_root=None: "abc123")

    wrapper.main(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--architectures",
            "cnn",
        ]
    )

    assert (tmp_path / "launcher_stdout.log").read_text(encoding="utf-8") == "wrapper stdout sentinel\n"
    assert (tmp_path / "launcher_stderr.log").read_text(encoding="utf-8") == "wrapper stderr sentinel\n"


def test_main_finalizes_launcher_completion_for_fresh_single_row_training(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_compare_wrapper as wrapper

    def fake_run_grid_lines_compare(**kwargs):
        run_dir = tmp_path / "runs" / "supervised_ffno"
        recon_dir = tmp_path / "recons" / "supervised_ffno"
        run_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")
        (recon_dir / "recon.npz").write_text("stub", encoding="utf-8")
        print("Saved artifacts to runs/supervised_ffno", file=sys.stderr)
        print("Torch runner complete. Artifacts in runs/supervised_ffno", file=sys.stderr)
        return {"metrics": {}}

    monkeypatch.setattr(wrapper, "run_grid_lines_compare", fake_run_grid_lines_compare)
    monkeypatch.setattr(
        "scripts.studies.invocation_logging.capture_runtime_provenance",
        lambda: {"python_executable": "/usr/bin/python3", "pythonpath": "/tmp/session_repo"},
    )
    monkeypatch.setattr("scripts.studies.invocation_logging.get_git_commit", lambda repo_root=None: "abc123")

    wrapper.main(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--models",
            "supervised_ffno",
        ]
    )

    completion_path = tmp_path / "runs" / "supervised_ffno" / "launcher_completion.json"
    assert completion_path.exists()
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    assert payload["evidence_source"] == "wrapper_launcher_stderr_row_completion_markers"
    assert payload["launcher_stderr_log"] == "launcher_stderr.log"


def test_main_finalizes_launcher_completion_after_invocation_completion(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_compare_wrapper as wrapper

    def fake_run_grid_lines_compare(**kwargs):
        run_dir = tmp_path / "runs" / "pinn_ffno"
        recon_dir = tmp_path / "recons" / "pinn_ffno"
        run_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")
        (recon_dir / "recon.npz").write_text("stub", encoding="utf-8")
        print("DEBUG eval_reconstruction [pinn_ffno]: amp_target stats: mean=1.0")
        print("DEBUG eval_reconstruction [pinn_ffno]: amp_pred stats: mean=2.0")
        return {"metrics": {}}

    monkeypatch.setattr(wrapper, "run_grid_lines_compare", fake_run_grid_lines_compare)
    monkeypatch.setattr(
        "scripts.studies.invocation_logging.capture_runtime_provenance",
        lambda: {"python_executable": "/usr/bin/python3", "pythonpath": "/tmp/session_repo"},
    )
    monkeypatch.setattr("scripts.studies.invocation_logging.get_git_commit", lambda repo_root=None: "abc123")

    wrapper.main(
        [
            "--N",
            "64",
            "--gridsize",
            "1",
            "--output-dir",
            str(tmp_path),
            "--models",
            "pinn_ffno",
            "--reuse-existing-recons",
        ]
    )

    completion_path = tmp_path / "runs" / "pinn_ffno" / "launcher_completion.json"
    assert completion_path.exists()
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    assert payload["evidence_source"] == "wrapper_launcher_stdout_eval_markers"
    assert payload["launcher_stdout_log"] == "launcher_stdout.log"


def test_default_torch_row_specs_register_neuralop_uno_rows():
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        PAPER_MODEL_LABELS,
        PAPER_TRAINING_PROCEDURE_OVERRIDES,
        TORCH_MODEL_IDS,
        _torch_model_route,
    )

    pinn_spec = DEFAULT_TORCH_ROW_SPECS["pinn_neuralop_uno"]
    sup_spec = DEFAULT_TORCH_ROW_SPECS["supervised_neuralop_uno"]
    assert pinn_spec["architecture"] == "neuralop_uno"
    assert pinn_spec["training_procedure"] == "pinn"
    assert sup_spec["architecture"] == "neuralop_uno"
    assert sup_spec["training_procedure"] == "supervised"
    assert "pinn_neuralop_uno" in TORCH_MODEL_IDS
    assert "supervised_neuralop_uno" in TORCH_MODEL_IDS
    assert PAPER_MODEL_LABELS["pinn_neuralop_uno"] == "U-NO + PINN"
    assert PAPER_MODEL_LABELS["supervised_neuralop_uno"] == "U-NO + supervised"
    assert PAPER_TRAINING_PROCEDURE_OVERRIDES["supervised_neuralop_uno"] == "supervised"

    arch_pinn, proc_pinn = _torch_model_route("pinn_neuralop_uno")
    arch_sup, proc_sup = _torch_model_route("supervised_neuralop_uno")
    assert (arch_pinn, proc_pinn) == ("neuralop_uno", "pinn")
    assert (arch_sup, proc_sup) == ("neuralop_uno", "supervised")


def test_validate_model_specs_accepts_neuralop_uno_rows():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    validate_model_specs(
        ("pinn_neuralop_uno", "supervised_neuralop_uno"),
        {"pinn_neuralop_uno": 128, "supervised_neuralop_uno": 128},
    )


def test_default_torch_row_specs_register_srunet_branch_objective_ablation_rows():
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        PAPER_MODEL_LABELS,
        PAPER_TRAINING_PROCEDURE_OVERRIDES,
        TORCH_MODEL_IDS,
        _torch_model_route,
    )

    conv_spec = DEFAULT_TORCH_ROW_SPECS["pinn_hybrid_resnet_encoder_conv_only"]
    spec_spec = DEFAULT_TORCH_ROW_SPECS["pinn_hybrid_resnet_encoder_spectral_only"]
    sup_spec = DEFAULT_TORCH_ROW_SPECS["supervised_hybrid_resnet"]

    assert conv_spec["architecture"] == "hybrid_resnet"
    assert conv_spec["training_procedure"] == "pinn"
    assert conv_spec["overrides"]["hybrid_encoder_branch_select"] == "conv_only"
    assert spec_spec["architecture"] == "hybrid_resnet"
    assert spec_spec["training_procedure"] == "pinn"
    assert spec_spec["overrides"]["hybrid_encoder_branch_select"] == "spectral_only"
    assert sup_spec["architecture"] == "hybrid_resnet"
    assert sup_spec["training_procedure"] == "supervised"

    assert "pinn_hybrid_resnet_encoder_conv_only" in TORCH_MODEL_IDS
    assert "pinn_hybrid_resnet_encoder_spectral_only" in TORCH_MODEL_IDS
    assert "supervised_hybrid_resnet" in TORCH_MODEL_IDS

    assert PAPER_MODEL_LABELS["pinn_hybrid_resnet_encoder_conv_only"] == \
        "Hybrid ResNet (conv-only encoder) + PINN"
    assert PAPER_MODEL_LABELS["pinn_hybrid_resnet_encoder_spectral_only"] == \
        "Hybrid ResNet (spectral-only encoder) + PINN"
    assert PAPER_MODEL_LABELS["supervised_hybrid_resnet"] == "Hybrid ResNet + supervised"

    assert PAPER_TRAINING_PROCEDURE_OVERRIDES["supervised_hybrid_resnet"] == "supervised"

    for model_id in (
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ):
        arch, _ = _torch_model_route(model_id)
        assert arch == "hybrid_resnet"


def test_validate_model_specs_accepts_srunet_branch_objective_ablation_rows():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    validate_model_specs(
        (
            "pinn_hybrid_resnet_encoder_conv_only",
            "pinn_hybrid_resnet_encoder_spectral_only",
            "supervised_hybrid_resnet",
        ),
        {
            "pinn_hybrid_resnet_encoder_conv_only": 128,
            "pinn_hybrid_resnet_encoder_spectral_only": 128,
            "supervised_hybrid_resnet": 128,
        },
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ],
)
def test_srunet_branch_objective_ablation_rows_lock_decision_support_status(model_id):
    from scripts.studies.grid_lines_compare_wrapper import DEFAULT_TORCH_ROW_SPECS

    spec = DEFAULT_TORCH_ROW_SPECS[model_id]
    assert spec["row_status"] == "decision_support_append_only"
    assert spec["lock_row_status"] is True


def test_default_torch_row_specs_register_hybrid_resnet_convnext_bottleneck_row():
    """The ConvNeXt-bottleneck SRU-Net row spec, label, and arch route are stable."""
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        PAPER_MODEL_LABELS,
        TORCH_MODEL_IDS,
        _torch_model_route,
    )

    spec = DEFAULT_TORCH_ROW_SPECS["pinn_hybrid_resnet_convnext_bottleneck"]
    assert spec["model_id"] == "pinn_hybrid_resnet_convnext_bottleneck"
    assert spec["architecture"] == "hybrid_resnet_convnext_bottleneck"
    assert spec["training_procedure"] == "pinn"
    assert spec["row_status"] == "decision_support_append_only"
    assert spec["lock_row_status"] is True

    assert "pinn_hybrid_resnet_convnext_bottleneck" in TORCH_MODEL_IDS
    assert (
        PAPER_MODEL_LABELS["pinn_hybrid_resnet_convnext_bottleneck"]
        == "Hybrid ResNet (ConvNeXt bottleneck) + PINN"
    )

    arch, training_procedure = _torch_model_route(
        "pinn_hybrid_resnet_convnext_bottleneck"
    )
    assert arch == "hybrid_resnet_convnext_bottleneck"
    assert training_procedure == "pinn"


def test_validate_model_specs_accepts_hybrid_resnet_convnext_bottleneck_row():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    validate_model_specs(
        ("pinn_hybrid_resnet_convnext_bottleneck",),
        {"pinn_hybrid_resnet_convnext_bottleneck": 128},
    )


def test_default_torch_row_specs_register_ffno_ptychoblock_encoder_row():
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        PAPER_MODEL_LABELS,
        TORCH_MODEL_IDS,
        _torch_model_route,
    )

    spec = DEFAULT_TORCH_ROW_SPECS["pinn_hybrid_resnet_ffno_ptychoblock_encoder"]
    assert spec["model_id"] == "pinn_hybrid_resnet_ffno_ptychoblock_encoder"
    assert spec["architecture"] == "hybrid_resnet_ffno_ptychoblock_encoder"
    assert spec["training_procedure"] == "pinn"
    assert spec["row_status"] == "decision_support_append_only"
    assert spec["lock_row_status"] is True

    assert "pinn_hybrid_resnet_ffno_ptychoblock_encoder" in TORCH_MODEL_IDS
    assert (
        PAPER_MODEL_LABELS["pinn_hybrid_resnet_ffno_ptychoblock_encoder"]
        == "Hybrid ResNet (FFNO->PtychoBlock encoder) + PINN"
    )

    arch, training_procedure = _torch_model_route(
        "pinn_hybrid_resnet_ffno_ptychoblock_encoder"
    )
    assert arch == "hybrid_resnet_ffno_ptychoblock_encoder"
    assert training_procedure == "pinn"


def test_default_torch_row_specs_register_ptychoblock_ffno_encoder_row():
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        PAPER_MODEL_LABELS,
        TORCH_MODEL_IDS,
        _torch_model_route,
    )

    spec = DEFAULT_TORCH_ROW_SPECS["pinn_hybrid_resnet_ptychoblock_ffno_encoder"]
    assert spec["model_id"] == "pinn_hybrid_resnet_ptychoblock_ffno_encoder"
    assert spec["architecture"] == "hybrid_resnet_ptychoblock_ffno_encoder"
    assert spec["training_procedure"] == "pinn"
    assert spec["row_status"] == "decision_support_append_only"
    assert spec["lock_row_status"] is True

    assert "pinn_hybrid_resnet_ptychoblock_ffno_encoder" in TORCH_MODEL_IDS
    assert (
        PAPER_MODEL_LABELS["pinn_hybrid_resnet_ptychoblock_ffno_encoder"]
        == "Hybrid ResNet (PtychoBlock->FFNO encoder) + PINN"
    )

    arch, training_procedure = _torch_model_route(
        "pinn_hybrid_resnet_ptychoblock_ffno_encoder"
    )
    assert arch == "hybrid_resnet_ptychoblock_ffno_encoder"
    assert training_procedure == "pinn"


def test_default_torch_row_specs_register_ffno_depth24_row():
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        PAPER_MODEL_LABELS,
        TORCH_MODEL_IDS,
        _torch_model_route,
    )

    default_ffno = DEFAULT_TORCH_ROW_SPECS["pinn_ffno"]
    depth24_spec = DEFAULT_TORCH_ROW_SPECS["pinn_ffno_depth24"]

    assert default_ffno["overrides"] == {}
    assert depth24_spec["model_id"] == "pinn_ffno_depth24"
    assert depth24_spec["architecture"] == "ffno"
    assert depth24_spec["training_procedure"] == "pinn"
    assert depth24_spec["model_label"] == "FFNO-24 + PINN"
    assert depth24_spec["overrides"]["fno_blocks"] == 24
    assert depth24_spec["overrides"]["fno_cnn_blocks"] == 0
    assert depth24_spec["row_status"] == "decision_support_append_only"
    assert depth24_spec["lock_row_status"] is True

    assert "pinn_ffno_depth24" in TORCH_MODEL_IDS
    assert PAPER_MODEL_LABELS["pinn_ffno_depth24"] == "FFNO-24 + PINN"

    arch, training_procedure = _torch_model_route("pinn_ffno_depth24")
    assert arch == "ffno"
    assert training_procedure == "pinn"


def test_validate_model_specs_accepts_ffno_depth24_row():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    validate_model_specs(
        ("pinn_ffno", "pinn_ffno_depth24"),
        {"pinn_ffno": 128, "pinn_ffno_depth24": 128},
    )


def test_validate_model_specs_accepts_ffno_ptychoblock_encoder_row():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    validate_model_specs(
        ("pinn_hybrid_resnet_ffno_ptychoblock_encoder",),
        {"pinn_hybrid_resnet_ffno_ptychoblock_encoder": 128},
    )


def test_validate_model_specs_accepts_ptychoblock_ffno_encoder_row():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs

    validate_model_specs(
        ("pinn_hybrid_resnet_ptychoblock_ffno_encoder",),
        {"pinn_hybrid_resnet_ptychoblock_ffno_encoder": 128},
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "pinn_hybrid_resnet_encoder_conv_only",
        "pinn_hybrid_resnet_encoder_spectral_only",
        "supervised_hybrid_resnet",
    ],
)
def test_enrich_paper_row_payload_keeps_branch_objective_rows_off_paper_grade(
    monkeypatch, tmp_path, model_id
):
    from scripts.studies.grid_lines_compare_wrapper import (
        DEFAULT_TORCH_ROW_SPECS,
        _enrich_paper_row_payload,
        _recover_torch_row_payload,
    )

    run_dir = tmp_path / "runs" / model_id
    recons_dir = tmp_path / "recons" / model_id
    visuals_dir = tmp_path / "visuals"
    run_dir.mkdir(parents=True)
    recons_dir.mkdir(parents=True)
    visuals_dir.mkdir(parents=True)

    train_npz = tmp_path / "train.npz"
    test_npz = tmp_path / "test.npz"
    probe_npz = tmp_path / "probe.npz"
    for path in (train_npz, test_npz, probe_npz):
        path.write_bytes(b"stub")

    gt = (np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64)
    np.savez(recons_dir / "recon.npz", YY_pred=gt, amp=np.abs(gt), phase=np.angle(gt))
    (visuals_dir / f"amp_phase_{model_id}.png").write_bytes(b"png")
    (visuals_dir / f"amp_phase_error_{model_id}.png").write_bytes(b"png")

    (run_dir / "history.json").write_text(
        json.dumps({"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}),
        encoding="utf-8",
    )
    (run_dir / "invocation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pid": 12345,
                "exit_code": 0,
                "timestamp_utc": "2026-04-30T17:09:15.697374+00:00",
                "finished_at_utc": "2026-04-30T17:22:54.631787+00:00",
                "parsed_args": {
                    "epochs": 40,
                    "seed": 3,
                    "output_dir": str(tmp_path),
                    "train_npz": str(train_npz),
                    "test_npz": str(test_npz),
                },
                "extra": {"git_commit": "abc123"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "invocation.sh").write_text("python grid_lines_torch_runner.py\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(_full_pair_metrics(0.01, 0.02)), encoding="utf-8")
    (run_dir / "model.pt").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.studies.grid_lines_compare_wrapper._count_torch_state_dict_parameters",
        lambda path: 123,
    )

    recovered = _recover_torch_row_payload(
        output_dir=tmp_path,
        model_id=model_id,
        n_value=128,
        metrics=_full_pair_metrics(0.01, 0.02),
    )
    enriched = _enrich_paper_row_payload(
        model_id=model_id,
        payload=recovered,
        output_dir=tmp_path,
        train_npz=train_npz,
        test_npz=test_npz,
        seed=3,
        nimgs_train=2,
        nimgs_test=2,
        gridsize=1,
        set_phi=True,
        probe_npz=probe_npz,
        dataset_source="synthetic_lines",
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
        row_spec=DEFAULT_TORCH_ROW_SPECS[model_id],
    )

    assert enriched["row_status"] == "decision_support_append_only"


def test_compare_wrapper_cli_accepts_manifest_claim_boundary():
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args(
        [
            "--N",
            "128",
            "--gridsize",
            "1",
            "--output-dir",
            "/tmp/example",
            "--manifest-claim-boundary",
            "decision_support_append_only",
        ]
    )

    assert args.manifest_claim_boundary == "decision_support_append_only"


def test_compare_wrapper_cli_manifest_claim_boundary_default():
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args(
        [
            "--N",
            "128",
            "--gridsize",
            "1",
            "--output-dir",
            "/tmp/example",
        ]
    )

    assert args.manifest_claim_boundary == "grid_lines_compare_bundle"

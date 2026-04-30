"""Tests for the lines128 paper benchmark harness preflight layer."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def _minimum_subset_rows_payload() -> list[dict]:
    return [
        {
            "model_id": "baseline",
            "model_label": "CDI CNN + supervised",
            "architecture_id": "cnn",
            "training_procedure": "supervised",
            "required_for_minimum_subset": True,
        },
        {
            "model_id": "pinn",
            "model_label": "CDI CNN + PINN",
            "architecture_id": "cnn",
            "training_procedure": "pinn",
            "required_for_minimum_subset": True,
        },
        {
            "model_id": "pinn_hybrid_resnet",
            "model_label": "Hybrid ResNet + PINN",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "required_for_minimum_subset": True,
        },
        {
            "model_id": "pinn_fno_vanilla",
            "model_label": "FNO Vanilla + PINN",
            "architecture_id": "fno_vanilla",
            "training_procedure": "pinn",
            "required_for_minimum_subset": True,
        },
    ]


def _write_execution_authority_note(path: Path, *, rows: list[dict] | None = None) -> Path:
    payload = {
        "state": "go_for_minimum_subset_execution",
        "prerequisite_preflight_decision_artifact": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json",
        "prerequisite_preflight_note": "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md",
        "selected_fno_comparator": "fno_vanilla",
        "seed_policy": {"type": "fixed", "seed": 3},
        "fixed_contract": _fixed_contract_payload(),
        "fixed_sample_ids": [0, 1],
        "shared_visual_scales": {
            "amp": {"vmin": 0.0, "vmax": 1.0},
            "phase": {"vmin": -3.14, "vmax": 3.14},
        },
        "rows": rows or _minimum_subset_rows_payload(),
        "later_complete_table_rows": [
            "pinn_spectral_resnet_bottleneck_net",
            "pinn_ffno",
        ],
    }
    note = "\n".join(
        [
            "# Lines128 Minimum Paper Table Execution Authority",
            "",
            "This note supersedes the preflight-only state for this backlog item.",
            "",
            "<!-- lines128_execution_authority_json:start -->",
            json.dumps(payload, indent=2),
            "<!-- lines128_execution_authority_json:end -->",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note, encoding="utf-8")
    return path


def _write_execution_manifest(path: Path, *, rows: list[dict] | None = None) -> Path:
    payload = {
        "state": "go_for_minimum_subset_execution",
        "selected_fno_comparator": "fno_vanilla",
        "seed_policy": {"type": "fixed", "seed": 3},
        "fixed_contract": _fixed_contract_payload(),
        "fixed_sample_ids": [0, 1],
        "shared_visual_scales": {
            "amp": {"vmin": 0.0, "vmax": 1.0},
            "phase": {"vmin": -3.14, "vmax": 3.14},
        },
        "rows": rows or _minimum_subset_rows_payload(),
        "later_complete_table_rows": [
            "pinn_spectral_resnet_bottleneck_net",
            "pinn_ffno",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, contents: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _materialize_minimum_subset_bundle_artifacts(
    output_dir: Path,
    *,
    omit_relative_paths: set[str] | None = None,
    include_wrapper_artifacts: bool = True,
) -> None:
    omit_relative_paths = set(omit_relative_paths or set())

    def write_relative(relative_path: str, contents: str = "x") -> None:
        if relative_path in omit_relative_paths:
            return
        _write_text(output_dir / relative_path, contents)

    if include_wrapper_artifacts:
        for relative_path in ("invocation.json", "invocation.sh"):
            if relative_path == "invocation.json":
                write_relative(
                    relative_path,
                    json.dumps(
                        {
                            "status": "completed",
                            "exit_code": 0,
                            "finished_at_utc": "2026-04-29T00:00:00+00:00",
                            "pid": 999,
                            "extra": {
                                "tmux": {
                                    "session_name": "lines128-minimum-subset",
                                    "socket_path": "/tmp/lines128-minimum-subset.sock",
                                    "attach_command": "tmux -S /tmp/lines128-minimum-subset.sock attach -t lines128-minimum-subset",
                                    "capture_command": "tmux -S /tmp/lines128-minimum-subset.sock capture-pane -p -t lines128-minimum-subset:0.0",
                                }
                            }
                        }
                    ),
                )
            else:
                write_relative(relative_path, "python fake_row.py\n")
    for relative_path in ("train.npz", "test.npz"):
        write_relative(relative_path)
    _write_text(
        output_dir / "dataset_identity_manifest.json",
        json.dumps(
            {
                "train_npz": {"size_bytes": 1, "sha256": "train-sha", "source": "synthetic_lines"},
                "test_npz": {"size_bytes": 1, "sha256": "test-sha", "source": "synthetic_lines"},
            }
        ),
    )
    _write_text(
        output_dir / "split_manifest.json",
        json.dumps({"seed": 3, "nimgs_train": 2, "nimgs_test": 2, "gridsize": 1, "set_phi": True}),
    )
    gt_recon = output_dir / "recons" / "gt" / "recon.npz"
    gt_recon.parent.mkdir(parents=True, exist_ok=True)
    if "recons/gt/recon.npz" not in omit_relative_paths:
        np.savez(gt_recon, YY_pred=np.ones((2, 2), dtype=np.complex64))

    for model_id in ("baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"):
        for relative_path in (
            f"runs/{model_id}/invocation.json",
            f"runs/{model_id}/invocation.sh",
            f"runs/{model_id}/config.json",
            f"runs/{model_id}/history.json",
            f"runs/{model_id}/metrics.json",
            f"runs/{model_id}/stdout.log",
            f"runs/{model_id}/stderr.log",
            f"recons/{model_id}/recon.npz",
            f"visuals/amp_phase_{model_id}.png",
            f"visuals/amp_phase_error_{model_id}.png",
        ):
            if relative_path.endswith("invocation.json"):
                write_relative(
                    relative_path,
                    json.dumps(
                        {
                            "argv": ["--seed", "3"],
                            "parsed_args": {"seed": 3, "grid_lines_config": {"seed": 3}},
                            "status": "completed",
                            "exit_code": 0,
                            "finished_at_utc": "2026-04-29T00:00:00+00:00",
                            "pid": 12345,
                        }
                    ),
                )
            elif relative_path.endswith("invocation.sh"):
                write_relative(relative_path, "python fake_row.py\n")
            elif relative_path.endswith("config.json"):
                write_relative(relative_path, json.dumps({"seed": 3, "grid_lines_config": {"seed": 3}}))
            elif relative_path.endswith("stdout.log"):
                write_relative(relative_path, f"[row:{model_id}] fixture stdout\n")
            elif relative_path.endswith("stderr.log"):
                write_relative(relative_path, "")
            else:
                write_relative(relative_path, "{}")
        _write_text(
            output_dir / "runs" / model_id / "exit_code_proof.json",
            json.dumps(
                {
                    "model_id": model_id,
                    "exit_code": 0,
                    "proof_source": "test",
                    "invocation_json": f"runs/{model_id}/invocation.json",
                    "invocation_status": "completed",
                    "stdout_log": f"runs/{model_id}/stdout.log",
                    "stderr_log": f"runs/{model_id}/stderr.log",
                }
            ),
        )

    for relative_path in (
        "visuals/amp_phase_gt.png",
        "visuals/compare_amp_phase.png",
        "visuals/frc_curves.png",
    ):
        write_relative(relative_path, "png")


def test_preflight_requires_decision_artifact(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    with pytest.raises(FileNotFoundError, match="decision artifact"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
        )


def test_cli_help_runs_from_script_path():
    completed = subprocess.run(
        [sys.executable, "scripts/studies/lines128_paper_benchmark.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "lines128" in completed.stdout.lower()


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


def test_minimum_subset_rejects_execution_manifest_drift(tmp_path):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    drift_rows = _minimum_subset_rows_payload()
    drift_rows[-1] = {
        **drift_rows[-1],
        "model_label": "Wrong FNO label",
    }
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
        rows=drift_rows,
    )

    with pytest.raises(ValueError, match="execution manifest"):
        run_lines128_paper_benchmark(
            decision_artifact=decision_artifact,
            execution_authority_note=authority_note,
            execution_manifest=execution_manifest,
            output_dir=tmp_path / "out",
        )


def test_minimum_subset_executes_four_locked_rows_and_emits_bundle(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark

    captured = {}
    output_dir = tmp_path / "out"

    def fake_run_grid_lines_compare(**kwargs):
        captured.update(kwargs)
        _materialize_minimum_subset_bundle_artifacts(output_dir)

        def _row(model_id, model_label, architecture_id, training_procedure, parameter_count, final_train_loss, runtime, metrics):
            return {
                "model_label": model_label,
                "architecture_id": architecture_id,
                "training_procedure": training_procedure,
                "N": 128,
                "parameter_count": parameter_count,
                "epoch_budget": 40,
                "final_completed_epoch": 40,
                "final_train_loss": final_train_loss,
                "validation_loss": {"status": "no_validation_series", "value": None},
                "runtime_summary": {"train_wall_time_sec": runtime, "inference_time_sec": runtime / 20.0},
                "hardware_summary": {"backend": "tensorflow" if architecture_id == "cnn" else "pytorch", "accelerator": "rtx3090"},
                "row_status": "paper_grade",
                "caveats": [],
                "invocation": {"json": f"runs/{model_id}/invocation.json", "shell": f"runs/{model_id}/invocation.sh"},
                "config": {"json": f"runs/{model_id}/config.json"},
                "git": {"commit": "abc123", "dirty_state_note": {"source": "test", "dirty": False}},
                "environment": {
                    "python_executable": "/usr/bin/python",
                    "python_version": "3.11.0",
                    "torch_version": "2.4.1",
                    "cuda_version": "12.1",
                    "gpu": "rtx3090",
                    "host": "test-host",
                },
                "dataset": {
                    "train_npz": "train.npz",
                    "test_npz": "test.npz",
                    "dataset_source": "synthetic_lines",
                    "manifest_json": "dataset_identity_manifest.json",
                },
                "splits": {
                    "nimgs_train": 2,
                    "nimgs_test": 2,
                    "gridsize": 1,
                    "set_phi": True,
                    "seed": 3,
                    "manifest_json": "split_manifest.json",
                },
                "randomness": {"requested_seed": 3},
                "outputs": {
                    "metrics_json": f"runs/{model_id}/metrics.json",
                    "history_json": f"runs/{model_id}/history.json",
                    "recon_npz": f"recons/{model_id}/recon.npz",
                    "stdout_log": f"runs/{model_id}/stdout.log",
                    "stderr_log": f"runs/{model_id}/stderr.log",
                    "exit_code_proof_json": f"runs/{model_id}/exit_code_proof.json",
                },
                "visuals": {
                    "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
                    "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
                },
                "metrics": metrics,
            }
        return {
            "selected_models": list(kwargs["models"]),
            "train_npz": str(tmp_path / "train.npz"),
            "test_npz": str(tmp_path / "test.npz"),
            "gt_recon": str(output_dir / "recons" / "gt" / "recon.npz"),
            "recon_paths": {
                model_id: str(output_dir / "recons" / model_id / "recon.npz")
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
            "row_payloads": {
                "baseline": _row("baseline", "CDI CNN + supervised", "cnn", "supervised", 100, 0.4, 9.0, {
                    "mae": (0.2, 0.3),
                    "mse": (0.02, 0.03),
                    "psnr": (60.0, 55.0),
                    "ssim": (0.7, 0.6),
                    "ms_ssim": (0.65, 0.55),
                    "frc50": (32, 24),
                }),
                "pinn": _row("pinn", "CDI CNN + PINN", "cnn", "pinn", 101, 0.3, 10.0, {
                    "mae": (0.19, 0.29),
                    "mse": (0.019, 0.029),
                    "psnr": (61.0, 56.0),
                    "ssim": (0.71, 0.61),
                    "ms_ssim": (0.66, 0.56),
                    "frc50": (33, 25),
                }),
                "pinn_hybrid_resnet": _row("pinn_hybrid_resnet", "Hybrid ResNet + PINN", "hybrid_resnet", "pinn", 102, 0.2, 11.0, {
                    "mae": (0.1, 0.2),
                    "mse": (0.01, 0.02),
                    "psnr": (70.0, 65.0),
                    "ssim": (0.9, 0.8),
                    "ms_ssim": (0.85, 0.75),
                    "frc50": (64, 48),
                }),
                "pinn_fno_vanilla": _row("pinn_fno_vanilla", "FNO Vanilla + PINN", "fno_vanilla", "pinn", 103, 0.25, 12.0, {
                    "mae": (0.12, 0.22),
                    "mse": (0.012, 0.022),
                    "psnr": (68.0, 63.0),
                    "ssim": (0.88, 0.78),
                    "ms_ssim": (0.83, 0.73),
                    "frc50": (60, 44),
                }),
            },
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )
    result = run_lines128_paper_benchmark(
        decision_artifact=decision_artifact,
        execution_authority_note=authority_note,
        execution_manifest=execution_manifest,
        output_dir=output_dir,
    )

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((output_dir / "model_manifest.json").read_text(encoding="utf-8"))
    assert captured["models"] == (
        "baseline",
        "pinn",
        "pinn_hybrid_resnet",
        "pinn_fno_vanilla",
    )
    assert metrics_payload["benchmark_status"] == "paper_complete"
    assert metrics_payload["selected_fno_comparator"] == "fno_vanilla"
    assert model_manifest["rows"][0]["training_procedure"] == "supervised"
    assert result["required_rows"] == [
        "baseline",
        "pinn",
        "pinn_hybrid_resnet",
        "pinn_fno_vanilla",
    ]


def test_minimum_subset_can_request_existing_recon_reuse(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark

    captured = {}

    def fake_run_grid_lines_compare(**kwargs):
        captured.update(kwargs)
        return {
            "row_payloads": {
                model_id: {
                    "model_label": model_id,
                    "architecture_id": "cnn" if model_id in {"baseline", "pinn"} else model_id.replace("pinn_", ""),
                    "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                    "N": 128,
                    "parameter_count": 1,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "not_emitted", "value": None},
                    "runtime_summary": {"recovered_from_existing_artifacts": True},
                    "hardware_summary": {"backend": "test"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "metrics": {
                        "mae": (0.1, 0.1),
                        "mse": (0.01, 0.01),
                        "psnr": (1.0, 1.0),
                        "ssim": (0.9, 0.9),
                        "ms_ssim": (0.8, 0.8),
                        "frc50": (2, 2),
                    },
                }
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            }
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )

    run_lines128_paper_benchmark(
        decision_artifact=decision_artifact,
        execution_authority_note=authority_note,
        execution_manifest=execution_manifest,
        output_dir=tmp_path / "out",
        reuse_existing_recons=True,
    )

    assert captured["reuse_existing_recons"] is True


def test_minimum_subset_emits_wrapper_manifest_for_shared_provenance(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark
    output_dir = tmp_path / "out"

    def fake_run_grid_lines_compare(**kwargs):
        _materialize_minimum_subset_bundle_artifacts(output_dir)
        return {
            "train_npz": str(tmp_path / "train.npz"),
            "test_npz": str(tmp_path / "test.npz"),
            "gt_recon": str(output_dir / "recons" / "gt" / "recon.npz"),
            "recon_paths": {
                model_id: str(output_dir / "recons" / model_id / "recon.npz")
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
            "row_payloads": {
                model_id: {
                    "model_label": {
                        "baseline": "CDI CNN + supervised",
                        "pinn": "CDI CNN + PINN",
                        "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
                        "pinn_fno_vanilla": "FNO Vanilla + PINN",
                    }[model_id],
                    "architecture_id": "cnn" if model_id in {"baseline", "pinn"} else model_id.replace("pinn_", ""),
                    "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                    "N": 128,
                    "parameter_count": 1,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.05},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.1},
                    "hardware_summary": {"backend": "test"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "invocation": {"json": f"runs/{model_id}/invocation.json", "shell": f"runs/{model_id}/invocation.sh"},
                    "config": {"json": f"runs/{model_id}/config.json"},
                    "git": {"commit": "abc123", "dirty_state_note": {"source": "test", "dirty": False}},
                    "environment": {
                        "python_executable": "/usr/bin/python",
                        "python_version": "3.11.0",
                        "torch_version": "2.4.1",
                        "cuda_version": "12.1",
                        "gpu": "rtx3090",
                        "host": "test-host",
                    },
                    "dataset": {
                        "train_npz": "train.npz",
                        "test_npz": "test.npz",
                        "dataset_source": "synthetic_lines",
                        "manifest_json": "dataset_identity_manifest.json",
                    },
                    "splits": {
                        "nimgs_train": 2,
                        "nimgs_test": 2,
                        "gridsize": 1,
                        "set_phi": True,
                        "seed": 3,
                        "manifest_json": "split_manifest.json",
                    },
                    "randomness": {"requested_seed": 3},
                    "outputs": {
                        "metrics_json": f"runs/{model_id}/metrics.json",
                        "history_json": f"runs/{model_id}/history.json",
                        "recon_npz": f"recons/{model_id}/recon.npz",
                        "stdout_log": f"runs/{model_id}/stdout.log",
                        "stderr_log": f"runs/{model_id}/stderr.log",
                        "exit_code_proof_json": f"runs/{model_id}/exit_code_proof.json",
                    },
                    "visuals": {
                        "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
                        "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
                    },
                    "metrics": {
                        "mae": (0.1, 0.1),
                        "mse": (0.01, 0.01),
                        "psnr": (1.0, 1.0),
                        "ssim": (0.9, 0.9),
                        "ms_ssim": (0.8, 0.8),
                        "frc50": (2, 2),
                    },
                }
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )

    run_lines128_paper_benchmark(
        decision_artifact=decision_artifact,
        execution_authority_note=authority_note,
        execution_manifest=execution_manifest,
        output_dir=output_dir,
    )

    manifest_path = output_dir / "paper_benchmark_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark_status"] == "paper_complete"
    assert manifest["selected_fno_comparator"] == "fno_vanilla"
    assert manifest["dataset"]["train_npz"].endswith("train.npz")
    assert manifest["dataset"]["manifest_json"] == "dataset_identity_manifest.json"
    assert manifest["git"]["dirty_state_note"]["source"]
    assert "python_version" in manifest["environment"]
    assert "torch_version" in manifest["environment"]
    assert "cuda_version" in manifest["environment"]
    assert "gpu" in manifest["environment"]
    assert "host" in manifest["environment"]
    assert manifest["rows"][0]["row_root"] == "runs/baseline"


def test_minimum_subset_downgrades_without_wrapper_launcher_contract(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark

    output_dir = tmp_path / "out"

    def fake_run_grid_lines_compare(**_kwargs):
        _materialize_minimum_subset_bundle_artifacts(output_dir)
        _write_text(
            output_dir / "invocation.json",
            json.dumps(
                {
                    "status": "completed",
                    "finished_at_utc": "2026-04-29T00:00:00+00:00",
                    "pid": 999,
                }
            ),
        )
        return {
            "train_npz": str(tmp_path / "train.npz"),
            "test_npz": str(tmp_path / "test.npz"),
            "gt_recon": str(output_dir / "recons" / "gt" / "recon.npz"),
            "recon_paths": {
                model_id: str(output_dir / "recons" / model_id / "recon.npz")
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
            "row_payloads": {
                model_id: {
                    "model_label": {
                        "baseline": "CDI CNN + supervised",
                        "pinn": "CDI CNN + PINN",
                        "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
                        "pinn_fno_vanilla": "FNO Vanilla + PINN",
                    }[model_id],
                    "architecture_id": "cnn" if model_id in {"baseline", "pinn"} else model_id.replace("pinn_", ""),
                    "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                    "N": 128,
                    "parameter_count": 1,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.05},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.1},
                    "hardware_summary": {"backend": "test"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "invocation": {"json": f"runs/{model_id}/invocation.json", "shell": f"runs/{model_id}/invocation.sh"},
                    "config": {"json": f"runs/{model_id}/config.json"},
                    "git": {"commit": "abc123", "dirty_state_note": {"source": "test", "dirty": False}},
                    "environment": {
                        "python_executable": "/usr/bin/python",
                        "python_version": "3.11.0",
                        "torch_version": "2.4.1",
                        "cuda_version": "12.1",
                        "gpu": "rtx3090",
                        "host": "test-host",
                    },
                    "dataset": {
                        "train_npz": "train.npz",
                        "test_npz": "test.npz",
                        "dataset_source": "synthetic_lines",
                        "manifest_json": "dataset_identity_manifest.json",
                    },
                    "splits": {
                        "nimgs_train": 2,
                        "nimgs_test": 2,
                        "gridsize": 1,
                        "set_phi": True,
                        "seed": 3,
                        "manifest_json": "split_manifest.json",
                    },
                    "randomness": {"requested_seed": 3},
                    "outputs": {
                        "metrics_json": f"runs/{model_id}/metrics.json",
                        "history_json": f"runs/{model_id}/history.json",
                        "recon_npz": f"recons/{model_id}/recon.npz",
                        "stdout_log": f"runs/{model_id}/stdout.log",
                        "stderr_log": f"runs/{model_id}/stderr.log",
                        "exit_code_proof_json": f"runs/{model_id}/exit_code_proof.json",
                    },
                    "visuals": {
                        "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
                        "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
                    },
                    "metrics": {
                        "mae": (0.1, 0.1),
                        "mse": (0.01, 0.01),
                        "psnr": (1.0, 1.0),
                        "ssim": (0.9, 0.9),
                        "ms_ssim": (0.8, 0.8),
                        "frc50": (2, 2),
                    },
                }
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )

    run_lines128_paper_benchmark(
        decision_artifact=decision_artifact,
        execution_authority_note=authority_note,
        execution_manifest=execution_manifest,
        output_dir=output_dir,
    )

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    assert "launcher_invocation_contract" in metrics_payload["missing_bundle_artifacts"]


def test_main_finalizes_bundle_after_wrapper_invocation_completes(tmp_path, monkeypatch):
    from scripts.studies import lines128_paper_benchmark as benchmark

    output_dir = tmp_path / "out"

    def fake_run_grid_lines_compare(**_kwargs):
        _materialize_minimum_subset_bundle_artifacts(
            output_dir,
            include_wrapper_artifacts=False,
        )
        return {
            "train_npz": str(output_dir / "train.npz"),
            "test_npz": str(output_dir / "test.npz"),
            "gt_recon": str(output_dir / "recons" / "gt" / "recon.npz"),
            "recon_paths": {
                model_id: str(output_dir / "recons" / model_id / "recon.npz")
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
            "row_payloads": {
                model_id: {
                    "model_label": {
                        "baseline": "CDI CNN + supervised",
                        "pinn": "CDI CNN + PINN",
                        "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
                        "pinn_fno_vanilla": "FNO Vanilla + PINN",
                    }[model_id],
                    "architecture_id": "cnn" if model_id in {"baseline", "pinn"} else model_id.replace("pinn_", ""),
                    "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                    "N": 128,
                    "parameter_count": 1,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.05},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.1},
                    "hardware_summary": {"backend": "test"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "invocation": {"json": f"runs/{model_id}/invocation.json", "shell": f"runs/{model_id}/invocation.sh"},
                    "config": {"json": f"runs/{model_id}/config.json"},
                    "git": {"commit": "abc123", "dirty_state_note": {"source": "test", "dirty": False}},
                    "environment": {
                        "python_executable": "/usr/bin/python",
                        "python_version": "3.11.0",
                        "torch_version": "2.4.1",
                        "cuda_version": "12.1",
                        "gpu": "rtx3090",
                        "host": "test-host",
                    },
                    "dataset": {
                        "train_npz": "train.npz",
                        "test_npz": "test.npz",
                        "dataset_source": "synthetic_lines",
                        "manifest_json": "dataset_identity_manifest.json",
                    },
                    "splits": {
                        "nimgs_train": 2,
                        "nimgs_test": 2,
                        "gridsize": 1,
                        "set_phi": True,
                        "seed": 3,
                        "manifest_json": "split_manifest.json",
                    },
                    "randomness": {"requested_seed": 3},
                    "outputs": {
                        "metrics_json": f"runs/{model_id}/metrics.json",
                        "history_json": f"runs/{model_id}/history.json",
                        "recon_npz": f"recons/{model_id}/recon.npz",
                        "stdout_log": f"runs/{model_id}/stdout.log",
                        "stderr_log": f"runs/{model_id}/stderr.log",
                        "exit_code_proof_json": f"runs/{model_id}/exit_code_proof.json",
                    },
                    "visuals": {
                        "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
                        "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
                    },
                    "metrics": {
                        "mae": (0.1, 0.1),
                        "mse": (0.01, 0.01),
                        "psnr": (1.0, 1.0),
                        "ssim": (0.9, 0.9),
                        "ms_ssim": (0.8, 0.8),
                        "frc50": (2, 2),
                    },
                }
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )
    monkeypatch.setenv("CODEX_TMUX_SESSION_NAME", "lines128-test")
    monkeypatch.setenv("CODEX_TMUX_SOCKET_PATH", "/tmp/lines128-test.sock")
    monkeypatch.setenv(
        "CODEX_TMUX_ATTACH_COMMAND",
        "tmux -S /tmp/lines128-test.sock attach -t lines128-test",
    )
    monkeypatch.setenv(
        "CODEX_TMUX_CAPTURE_COMMAND",
        "tmux -S /tmp/lines128-test.sock capture-pane -p -t lines128-test:0.0",
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )

    benchmark.main(
        [
            "--mode",
            "minimum_subset",
            "--decision-artifact",
            str(decision_artifact),
            "--execution-authority-note",
            str(authority_note),
            "--execution-manifest",
            str(execution_manifest),
            "--output-dir",
            str(output_dir),
        ]
    )

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "paper_benchmark_manifest.json").read_text(encoding="utf-8"))
    wrapper_invocation_payload = json.loads((output_dir / "invocation.json").read_text(encoding="utf-8"))
    assert wrapper_invocation_payload["status"] == "completed"
    assert wrapper_invocation_payload["exit_code"] == 0
    assert metrics_payload["benchmark_status"] == "paper_complete"
    assert metrics_payload["missing_bundle_artifacts"] == []
    assert manifest_payload["benchmark_status"] == "paper_complete"


def test_main_finalizes_recovered_torch_launcher_completion_after_wrapper_completion(tmp_path, monkeypatch):
    from scripts.studies import lines128_paper_benchmark as benchmark

    output_dir = tmp_path / "out"

    def fake_run_grid_lines_compare(**_kwargs):
        _materialize_minimum_subset_bundle_artifacts(
            output_dir,
            include_wrapper_artifacts=False,
        )
        _write_text(
            output_dir / "launcher_stderr.log",
            "\n".join(
                [
                    f"Saved artifacts to {output_dir / 'runs' / 'pinn_hybrid_resnet'}",
                    f"Torch runner complete. Artifacts in {output_dir / 'runs' / 'pinn_hybrid_resnet'}",
                    f"Saved artifacts to {output_dir / 'runs' / 'pinn_fno_vanilla'}",
                    f"Torch runner complete. Artifacts in {output_dir / 'runs' / 'pinn_fno_vanilla'}",
                ]
            )
            + "\n",
        )
        row_payloads = {}
        for model_id, backend in (
            ("baseline", "tensorflow"),
            ("pinn", "tensorflow"),
            ("pinn_hybrid_resnet", "pytorch"),
            ("pinn_fno_vanilla", "pytorch"),
        ):
            row_payloads[model_id] = {
                "model_label": {
                    "baseline": "CDI CNN + supervised",
                    "pinn": "CDI CNN + PINN",
                    "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
                    "pinn_fno_vanilla": "FNO Vanilla + PINN",
                }[model_id],
                "architecture_id": "cnn" if model_id in {"baseline", "pinn"} else model_id.replace("pinn_", ""),
                "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                "N": 128,
                "parameter_count": 1,
                "epoch_budget": 40,
                "final_completed_epoch": 40,
                "final_train_loss": 0.1,
                "validation_loss": {"status": "emitted", "value": 0.05},
                "runtime_summary": {
                    "recovered_from_existing_artifacts": True,
                    "runtime_source": "unavailable_under_recovery" if backend == "tensorflow" else "row_invocation",
                },
                "hardware_summary": {"backend": backend},
                "row_status": "paper_grade",
                "caveats": ["recovered_from_existing_artifacts"],
                "invocation": {"json": f"runs/{model_id}/invocation.json", "shell": f"runs/{model_id}/invocation.sh"},
                "config": {"json": f"runs/{model_id}/config.json"},
                "git": {"commit": "abc123", "dirty_state_note": {"source": "test", "dirty": False}},
                "environment": {
                    "python_executable": "/usr/bin/python",
                    "python_version": "3.11.0",
                    "torch_version": "2.4.1",
                    "cuda_version": "12.1",
                    "gpu": "rtx3090",
                    "host": "test-host",
                },
                "dataset": {
                    "train_npz": "train.npz",
                    "test_npz": "test.npz",
                    "dataset_source": "synthetic_lines",
                    "manifest_json": "dataset_identity_manifest.json",
                },
                "splits": {
                    "nimgs_train": 2,
                    "nimgs_test": 2,
                    "gridsize": 1,
                    "set_phi": True,
                    "seed": 3,
                    "manifest_json": "split_manifest.json",
                },
                "randomness": {"requested_seed": 3},
                "outputs": {
                    "metrics_json": f"runs/{model_id}/metrics.json",
                    "history_json": f"runs/{model_id}/history.json",
                    "recon_npz": f"recons/{model_id}/recon.npz",
                    "stdout_log": f"runs/{model_id}/stdout.log",
                    "stderr_log": f"runs/{model_id}/stderr.log",
                    "exit_code_proof_json": f"runs/{model_id}/exit_code_proof.json",
                },
                "visuals": {
                    "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
                    "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
                },
                "metrics": {
                    "mae": (0.1, 0.1),
                    "mse": (0.2, 0.2),
                    "psnr": (1.0, 1.0),
                    "ssim": (0.9, 0.9),
                    "ms_ssim": (0.8, 0.8),
                    "frc50": (0.7, 0.7),
                },
            }
        return {
            "train_npz": str(output_dir / "train.npz"),
            "test_npz": str(output_dir / "test.npz"),
            "gt_recon": str(output_dir / "recons" / "gt" / "recon.npz"),
            "recon_paths": {
                model_id: str(output_dir / "recons" / model_id / "recon.npz")
                for model_id in row_payloads
            },
            "row_payloads": row_payloads,
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )
    monkeypatch.setenv("CODEX_TMUX_SESSION_NAME", "lines128-test")
    monkeypatch.setenv("CODEX_TMUX_SOCKET_PATH", "/tmp/lines128-test.sock")
    monkeypatch.setenv(
        "CODEX_TMUX_ATTACH_COMMAND",
        "tmux -S /tmp/lines128-test.sock attach -t lines128-test",
    )
    monkeypatch.setenv(
        "CODEX_TMUX_CAPTURE_COMMAND",
        "tmux -S /tmp/lines128-test.sock capture-pane -p -t lines128-test:0.0",
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )

    benchmark.main(
        [
            "--mode",
            "minimum_subset",
            "--decision-artifact",
            str(decision_artifact),
            "--execution-authority-note",
            str(authority_note),
            "--execution-manifest",
            str(execution_manifest),
            "--output-dir",
            str(output_dir),
            "--reuse-existing-recons",
        ]
    )

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "paper_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "paper_complete"
    assert metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"] == []
    assert metrics_payload["missing_fields_by_row"]["pinn_fno_vanilla"] == []
    assert (
        metrics_payload["rows"]["pinn_hybrid_resnet"]["outputs"]["launcher_completion_json"]
        == "runs/pinn_hybrid_resnet/launcher_completion.json"
    )
    assert (
        metrics_payload["rows"]["pinn_fno_vanilla"]["outputs"]["launcher_completion_json"]
        == "runs/pinn_fno_vanilla/launcher_completion.json"
    )
    assert manifest_payload["benchmark_status"] == "paper_complete"
    assert (
        manifest_payload["rows"][2]["artifacts"]["launcher_completion_json"]
        == "runs/pinn_hybrid_resnet/launcher_completion.json"
    )
    assert (
        manifest_payload["rows"][3]["artifacts"]["launcher_completion_json"]
        == "runs/pinn_fno_vanilla/launcher_completion.json"
    )


def test_minimum_subset_downgrades_when_required_bundle_visual_is_missing(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark

    output_dir = tmp_path / "out"

    def fake_run_grid_lines_compare(**kwargs):
        _materialize_minimum_subset_bundle_artifacts(
            output_dir,
            omit_relative_paths={"visuals/frc_curves.png"},
        )
        return {
            "train_npz": str(tmp_path / "train.npz"),
            "test_npz": str(tmp_path / "test.npz"),
            "gt_recon": str(output_dir / "recons" / "gt" / "recon.npz"),
            "recon_paths": {
                model_id: str(output_dir / "recons" / model_id / "recon.npz")
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
            "row_payloads": {
                model_id: {
                    "model_label": model_id,
                    "architecture_id": "cnn" if model_id in {"baseline", "pinn"} else model_id.replace("pinn_", ""),
                    "training_procedure": "supervised" if model_id == "baseline" else "pinn",
                    "N": 128,
                    "parameter_count": 1,
                    "epoch_budget": 40,
                    "final_completed_epoch": 40,
                    "final_train_loss": 0.1,
                    "validation_loss": {"status": "emitted", "value": 0.05},
                    "runtime_summary": {"train_wall_time_sec": 1.0, "inference_time_sec": 0.1},
                    "hardware_summary": {"backend": "test"},
                    "row_status": "paper_grade",
                    "caveats": [],
                    "invocation": {"json": f"runs/{model_id}/invocation.json", "shell": f"runs/{model_id}/invocation.sh"},
                    "config": {"json": f"runs/{model_id}/config.json"},
                    "git": {"commit": "abc123", "dirty_state_note": {"source": "test", "dirty": False}},
                    "environment": {
                        "python_executable": "/usr/bin/python",
                        "python_version": "3.11.0",
                        "torch_version": "2.4.1",
                        "cuda_version": "12.1",
                        "gpu": "rtx3090",
                        "host": "test-host",
                    },
                    "dataset": {
                        "train_npz": "train.npz",
                        "test_npz": "test.npz",
                        "dataset_source": "synthetic_lines",
                        "manifest_json": "dataset_identity_manifest.json",
                    },
                    "splits": {
                        "nimgs_train": 2,
                        "nimgs_test": 2,
                        "gridsize": 1,
                        "set_phi": True,
                        "seed": 3,
                        "manifest_json": "split_manifest.json",
                    },
                    "randomness": {"requested_seed": 3},
                    "outputs": {
                        "metrics_json": f"runs/{model_id}/metrics.json",
                        "history_json": f"runs/{model_id}/history.json",
                        "recon_npz": f"recons/{model_id}/recon.npz",
                        "stdout_log": f"runs/{model_id}/stdout.log",
                        "stderr_log": f"runs/{model_id}/stderr.log",
                        "exit_code_proof_json": f"runs/{model_id}/exit_code_proof.json",
                    },
                    "visuals": {
                        "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
                        "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
                    },
                    "metrics": {
                        "mae": (0.1, 0.1),
                        "mse": (0.01, 0.01),
                        "psnr": (1.0, 1.0),
                        "ssim": (0.9, 0.9),
                        "ms_ssim": (0.8, 0.8),
                        "frc50": (2, 2),
                    },
                }
                for model_id in (
                    "baseline",
                    "pinn",
                    "pinn_hybrid_resnet",
                    "pinn_fno_vanilla",
                )
            },
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")
    authority_note = _write_execution_authority_note(tmp_path / "authority.md")
    execution_manifest = _write_execution_manifest(
        tmp_path / "execution" / "benchmark_execution_decisions.json",
    )

    run_lines128_paper_benchmark(
        decision_artifact=decision_artifact,
        execution_authority_note=authority_note,
        execution_manifest=execution_manifest,
        output_dir=output_dir,
    )

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((output_dir / "paper_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    assert "visuals/frc_curves.png" in metrics_payload["missing_bundle_artifacts"]
    assert manifest_payload["benchmark_status"] == "benchmark_incomplete"


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
    assert metrics_payload["visual_collation"]["shared_visual_scales"] == {
        "amp": {"vmin": 0.0, "vmax": 1.0},
    }


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


def test_preflight_rejects_compare_preflight_when_selected_models_omit_supported_row(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    def fake_run_grid_lines_compare(**kwargs):
        selected_models = [model_id for model_id in kwargs["models"] if model_id != "pinn_fno_vanilla"]
        return {
            "mode": "preflight_only",
            "selected_models": selected_models,
            "resolved_model_n": dict(kwargs["model_n"]),
            "seed": kwargs["seed"],
            "contract": {
                **_fixed_contract_payload(),
                "probe_npz": str(kwargs["probe_npz"]),
            },
            "row_plan": [
                {
                    "model_id": model_id,
                    "status": "supported_for_harness",
                    "backend": "tf" if model_id == "pinn" else "torch",
                    "N": kwargs["model_n"].get(model_id, kwargs["N"]),
                }
                for model_id in selected_models
            ],
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")

    with pytest.raises(ValueError, match="selected_models"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )


def test_preflight_rejects_compare_preflight_when_row_plan_omits_supported_row(tmp_path, monkeypatch):
    from scripts.studies.lines128_paper_benchmark import run_lines128_paper_benchmark_preflight

    def fake_run_grid_lines_compare(**kwargs):
        return {
            "mode": "preflight_only",
            "selected_models": list(kwargs["models"]),
            "resolved_model_n": dict(kwargs["model_n"]),
            "seed": kwargs["seed"],
            "contract": {
                **_fixed_contract_payload(),
                "probe_npz": str(kwargs["probe_npz"]),
            },
            "row_plan": [
                {
                    "model_id": model_id,
                    "status": "supported_for_harness",
                    "backend": "tf" if model_id == "pinn" else "torch",
                    "N": kwargs["model_n"].get(model_id, kwargs["N"]),
                }
                for model_id in kwargs["models"]
                if model_id != "pinn_fno_vanilla"
            ],
        }

    monkeypatch.setattr(
        "scripts.studies.lines128_paper_benchmark.run_grid_lines_compare",
        fake_run_grid_lines_compare,
    )

    decision_artifact = _write_decision_artifact(tmp_path / "decision.json")

    with pytest.raises(ValueError, match="row_plan"):
        run_lines128_paper_benchmark_preflight(
            decision_artifact=decision_artifact,
            output_dir=tmp_path / "out",
        )

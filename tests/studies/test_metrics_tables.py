"""Tests for study metrics table rendering."""

import json
from pathlib import Path

import numpy as np

from scripts.studies.metrics_tables import (
    METRICS,
    _build_main_table,
    write_paper_benchmark_bundle,
)


def _write_text(path: Path, contents: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _paper_grade_row_payload() -> dict:
    return {
        "model_label": "Hybrid ResNet",
        "architecture_id": "hybrid_resnet",
        "training_procedure": "pinn",
        "N": 128,
        "parameter_count": 123456,
        "epoch_budget": 40,
        "final_completed_epoch": 40,
        "final_train_loss": 0.123,
        "validation_loss": {"status": "no_validation_series", "value": None},
        "runtime_summary": {"train_wall_time_sec": 12.5, "inference_time_sec": 1.5},
        "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
        "row_status": "paper_grade",
        "caveats": [],
        "invocation": {
            "json": "runs/pinn_hybrid_resnet/invocation.json",
            "shell": "runs/pinn_hybrid_resnet/invocation.sh",
        },
        "config": {"json": "runs/pinn_hybrid_resnet/config.json"},
        "git": {
            "commit": "abc123",
            "dirty_state_note": {
                "source": "test_fixture",
                "dirty": False,
            },
        },
        "environment": {
            "python_executable": "/usr/bin/python",
            "python_version": "3.11.0",
            "torch_version": "2.4.1",
            "cuda_version": "12.1",
            "gpu": "rtx3090",
            "host": "test-host",
        },
        "dataset": {
            "train_npz": "datasets/train.npz",
            "test_npz": "datasets/test.npz",
            "dataset_source": "synthetic_lines",
            "manifest_json": "dataset_identity_manifest.json",
        },
        "splits": {
            "nimgs_train": 2,
            "nimgs_test": 2,
            "seed": 3,
            "manifest_json": "split_manifest.json",
        },
        "randomness": {"requested_seed": 3},
        "outputs": {
            "metrics_json": "runs/pinn_hybrid_resnet/metrics.json",
            "history_json": "runs/pinn_hybrid_resnet/history.json",
            "recon_npz": "recons/pinn_hybrid_resnet/recon.npz",
            "stdout_log": "runs/pinn_hybrid_resnet/stdout.log",
            "stderr_log": "runs/pinn_hybrid_resnet/stderr.log",
            "exit_code_proof_json": "runs/pinn_hybrid_resnet/exit_code_proof.json",
        },
        "visuals": {
            "amp_phase_png": "visuals/amp_phase_pinn_hybrid_resnet.png",
            "amp_phase_error_png": "visuals/amp_phase_error_pinn_hybrid_resnet.png",
        },
        "metrics": {
            "mae": (0.1, 0.2),
            "mse": (0.01, 0.02),
            "psnr": (70.0, 65.0),
            "ssim": (0.9, 0.8),
            "ms_ssim": (0.85, 0.75),
            "frc50": (64, 48),
        },
    }


def test_main_table_keeps_one_cell_per_metric_when_values_missing():
    metrics = {
        "pinn": {
            "mae": (0.1, 0.2),
            # intentionally omit the rest
        }
    }
    table = _build_main_table(metrics, model_ns={"pinn": 64})
    model_line = next(line for line in table.splitlines() if "PtychoPINN (CNN)" in line)
    cells = [cell.strip() for cell in model_line.rstrip("\\").split("&")]
    assert len(cells) == 2 + len(METRICS)
    assert cells[0] == "64"
    assert cells[1] == "PtychoPINN (CNN)"
    # all non-MAE metric cells should be a single '-' placeholder
    assert all(cell == "-" for cell in cells[3:])


def test_main_table_header_omits_binomial_single_image_frc_columns():
    metrics = {"pinn": {"mae": (0.1, 0.2)}}
    table = _build_main_table(metrics, model_ns={"pinn": 64})
    header = next(line for line in table.splitlines() if line.startswith("N & Model"))
    assert "1FRC50 Bin (A/P)" not in header
    assert "1FRC1/7 Bin (A/P)" not in header
    assert "single_frc50_binomial" not in {metric for metric, _ in METRICS}
    assert "single_frc1over7_binomial" not in {metric for metric, _ in METRICS}


def test_main_table_escapes_underscore_in_fallback_model_labels():
    metrics = {"pinn_custom_model": {"mae": (0.1, 0.2)}}
    table = _build_main_table(metrics, model_ns={"pinn_custom_model": 64})
    assert "pinn\\_custom\\_model" in table


def test_write_paper_bundle_marks_benchmark_incomplete_when_required_fields_missing(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "N": 128,
            "metrics": {
                "mae": (0.1, 0.2),
                "mse": (0.01, 0.02),
                "psnr": (70.0, 65.0),
                "ssim": (0.9, 0.8),
                "ms_ssim": (0.85, 0.75),
                "frc50": (64, 48),
            },
        }
    }

    paths = write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    schema_payload = json.loads((tmp_path / "metric_schema.json").read_text(encoding="utf-8"))
    assert paths["metric_schema_json"].endswith("metric_schema.json")
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    assert "parameter_count" in metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"]
    assert schema_payload["status_values"] == ["paper_complete", "benchmark_incomplete"]
    assert schema_payload["row_status_values"] == [
        "paper_grade",
        "decision_support",
        "blocked",
        "not_protocol_compatible",
    ]
    field_definitions = {field["name"]: field for field in schema_payload["field_definitions"]}
    assert field_definitions["parameter_count"]["units"] == "parameters"
    assert field_definitions["validation_loss"]["nullable"] is True
    metric_fields = {field["name"]: field for field in schema_payload["metric_fields"]}
    assert metric_fields["psnr"]["units"] == {"amplitude": "dB", "phase": "dB"}
    assert metric_fields["frc50"]["nullable"] is False


def test_write_paper_bundle_marks_paper_complete_when_required_fields_present(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "no_validation_series", "value": None},
            "runtime_summary": {"train_wall_time_sec": 12.5, "inference_time_sec": 1.5},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": (0.1, 0.2),
                "mse": (0.01, 0.02),
                "psnr": (70.0, 65.0),
                "ssim": (0.9, 0.8),
                "ms_ssim": (0.85, 0.75),
                "frc50": (64, 48),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "paper_complete"
    assert metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"] == []


def test_write_paper_bundle_accepts_numpy_metric_pairs(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "no_validation_series", "value": None},
            "runtime_summary": {"train_wall_time_sec": 12.5},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": np.array([0.1, 0.2], dtype=np.float32),
                "mse": np.array([0.01, 0.02], dtype=np.float32),
                "psnr": np.array([70.0, 65.0], dtype=np.float32),
                "ssim": np.array([0.9, 0.8], dtype=np.float32),
                "ms_ssim": np.array([0.85, 0.75], dtype=np.float32),
                "frc50": np.array([64, 48], dtype=np.float32),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "paper_complete"
    assert metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"] == []


def test_write_paper_bundle_accepts_numpy_scalar_pairs_in_tuples(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "no_validation_series", "value": None},
            "runtime_summary": {"train_wall_time_sec": 12.5},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": (np.float32(0.1), np.float32(0.2)),
                "mse": (np.float32(0.01), np.float32(0.02)),
                "psnr": (np.float32(70.0), np.float32(65.0)),
                "ssim": (np.float32(0.9), np.float32(0.8)),
                "ms_ssim": (np.float32(0.85), np.float32(0.75)),
                "frc50": (np.float32(64), np.float32(48)),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "paper_complete"
    assert metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"] == []


def test_write_paper_bundle_emits_model_manifest_with_row_identity_fields(tmp_path):
    row_payloads = {
        "baseline": {
            "model_label": "CDI CNN + supervised",
            "architecture_id": "cnn",
            "training_procedure": "supervised",
            "N": 128,
            "parameter_count": 111,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.5,
            "validation_loss": {"status": "not_emitted", "value": None},
            "runtime_summary": {"train_wall_time_sec": 9.0, "inference_time_sec": 0.4},
            "hardware_summary": {"backend": "tensorflow", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": (0.2, 0.3),
                "mse": (0.02, 0.03),
                "psnr": (60.0, 55.0),
                "ssim": (0.7, 0.6),
                "ms_ssim": (0.65, 0.55),
                "frc50": (32, 24),
            },
        }
    }

    paths = write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("baseline",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
    )

    manifest = json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    assert paths["model_manifest_json"].endswith("model_manifest.json")
    assert manifest["benchmark_status"] == "paper_complete"
    assert manifest["claim_boundary"] == "minimum_draftable_cdi_subset"
    assert manifest["rows"][0]["model_id"] == "baseline"
    assert manifest["rows"][0]["model_label"] == "CDI CNN + supervised"
    assert manifest["rows"][0]["architecture_id"] == "cnn"
    assert manifest["rows"][0]["training_procedure"] == "supervised"
    assert manifest["rows"][0]["row_status"] == "paper_grade"


def test_write_paper_bundle_marks_benchmark_incomplete_when_required_row_status_is_blocked(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "no_validation_series", "value": None},
            "runtime_summary": {"train_wall_time_sec": 12.5, "inference_time_sec": 1.5},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": (0.1, 0.2),
                "mse": (0.01, 0.02),
                "psnr": (70.0, 65.0),
                "ssim": (0.9, 0.8),
                "ms_ssim": (0.85, 0.75),
                "frc50": (64, 48),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
        row_statuses={"pinn_hybrid_resnet": {"status": "row_blocker", "reason": "route failed"}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"


def test_write_paper_bundle_serializes_numpy_scalars_in_nested_payloads(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": np.float32(0.123),
            "validation_loss": {"status": "emitted", "value": np.float32(0.045)},
            "runtime_summary": {"train_wall_time_sec": np.float32(12.5), "inference_time_sec": np.float32(1.5)},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": (0.1, 0.2),
                "mse": (0.01, 0.02),
                "psnr": (70.0, 65.0),
                "ssim": (0.9, 0.8),
                "ms_ssim": (0.85, 0.75),
                "frc50": (64, 48),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": np.float32(0.0), "vmax": np.float32(1.0)}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["rows"]["pinn_hybrid_resnet"]["final_train_loss"] == 0.12300000339746475
    assert metrics_payload["rows"]["pinn_hybrid_resnet"]["validation_loss"]["value"] == 0.04500000178813934
    assert metrics_payload["visual_collation"]["shared_visual_scales"]["amp"]["vmax"] == 1.0


def test_write_paper_bundle_requires_paper_grade_row_status_for_paper_complete(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "no_validation_series", "value": None},
            "runtime_summary": {"train_wall_time_sec": 12.5, "inference_time_sec": 1.5},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "completed",
            "caveats": [],
            "metrics": {
                "mae": (0.1, 0.2),
                "mse": (0.01, 0.02),
                "psnr": (70.0, 65.0),
                "ssim": (0.9, 0.8),
                "ms_ssim": (0.85, 0.75),
                "frc50": (64, 48),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
        require_row_provenance=True,
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    assert "row_status" in metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"]


def test_write_paper_bundle_requires_row_provenance_for_paper_complete(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "emitted", "value": 0.045},
            "runtime_summary": {"train_wall_time_sec": 12.5, "inference_time_sec": 1.5},
            "hardware_summary": {"backend": "pytorch", "accelerator": "rtx3090"},
            "row_status": "paper_grade",
            "caveats": [],
            "metrics": {
                "mae": (0.1, 0.2),
                "mse": (0.01, 0.02),
                "psnr": (70.0, 65.0),
                "ssim": (0.9, 0.8),
                "ms_ssim": (0.85, 0.75),
                "frc50": (64, 48),
            },
        }
    }

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
        require_row_provenance=True,
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    missing = set(metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"])
    assert {
        "invocation",
        "config",
        "git",
        "environment",
        "dataset",
        "splits",
        "randomness",
        "outputs",
        "visuals",
    } <= missing


def test_write_paper_bundle_rejects_shallow_paper_grade_provenance(tmp_path):
    row_payload = _paper_grade_row_payload()
    row_payload["git"] = {"commit": "abc123"}
    row_payload["environment"] = {"python_executable": "/usr/bin/python"}
    row_payload["dataset"] = {
        "train_npz": "datasets/train.npz",
        "test_npz": "datasets/test.npz",
    }
    row_payload["splits"] = {"nimgs_train": 2, "nimgs_test": 2}
    row_payload["outputs"] = {
        "metrics_json": "runs/pinn_hybrid_resnet/metrics.json",
        "history_json": "runs/pinn_hybrid_resnet/history.json",
        "recon_npz": "recons/pinn_hybrid_resnet/recon.npz",
    }
    row_payloads = {"pinn_hybrid_resnet": row_payload}

    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.sh", "#!/usr/bin/env bash\n")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "config.json", "{}")
    _write_text(tmp_path / "datasets" / "train.npz")
    _write_text(tmp_path / "datasets" / "test.npz")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "metrics.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "history.json", "{}")
    _write_text(tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz")
    _write_text(tmp_path / "visuals" / "amp_phase_pinn_hybrid_resnet.png")
    _write_text(tmp_path / "visuals" / "amp_phase_error_pinn_hybrid_resnet.png")

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
        require_row_provenance=True,
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    missing = set(metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"])
    assert {"git", "environment", "dataset", "splits", "outputs"} <= missing


def test_write_paper_bundle_requires_existing_provenance_paths_for_paper_complete(tmp_path):
    row_payloads = {"pinn_hybrid_resnet": _paper_grade_row_payload()}

    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.sh", "#!/usr/bin/env bash\n")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "config.json", "{}")
    _write_text(tmp_path / "datasets" / "train.npz")
    _write_text(tmp_path / "datasets" / "test.npz")
    _write_text(tmp_path / "dataset_identity_manifest.json", "{}")
    _write_text(tmp_path / "split_manifest.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "metrics.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "history.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "stdout.log")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "stderr.log")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "exit_code_proof.json", "{}")
    _write_text(tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz")
    _write_text(tmp_path / "visuals" / "amp_phase_pinn_hybrid_resnet.png")
    # Intentionally omit amp_phase_error_png to prove the validator checks the referenced file.

    write_paper_benchmark_bundle(
        output_dir=tmp_path,
        row_payloads=row_payloads,
        required_rows=("pinn_hybrid_resnet",),
        fixed_sample_ids=[0, 1],
        shared_visual_scales={"amp": {"vmin": 0.0, "vmax": 1.0}},
        require_row_provenance=True,
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["benchmark_status"] == "benchmark_incomplete"
    assert "visuals" in metrics_payload["missing_fields_by_row"]["pinn_hybrid_resnet"]

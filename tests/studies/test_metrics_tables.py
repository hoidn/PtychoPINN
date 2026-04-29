"""Tests for study metrics table rendering."""

import json

import numpy as np

from scripts.studies.metrics_tables import (
    METRICS,
    _build_main_table,
    write_paper_benchmark_bundle,
)


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
            "row_status": "completed",
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
    assert manifest["rows"][0]["row_status"] == "completed"


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
        shared_visual_scales={"amp": {"vmin": np.float32(0.0), "vmax": np.float32(1.0)}},
    )

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["rows"]["pinn_hybrid_resnet"]["final_train_loss"] == 0.12300000339746475
    assert metrics_payload["rows"]["pinn_hybrid_resnet"]["validation_loss"]["value"] == 0.04500000178813934
    assert metrics_payload["visual_collation"]["shared_visual_scales"]["amp"]["vmax"] == 1.0

"""Tests for study metrics table rendering."""

import json

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


def test_write_paper_bundle_marks_paper_complete_when_required_fields_present(tmp_path):
    row_payloads = {
        "pinn_hybrid_resnet": {
            "model_label": "Hybrid ResNet",
            "N": 128,
            "parameter_count": 123456,
            "epoch_budget": 40,
            "final_completed_epoch": 40,
            "final_train_loss": 0.123,
            "validation_loss": {"status": "no_validation_series", "value": None},
            "runtime_hardware_summary": {"train_wall_time_sec": 12.5, "accelerator": "rtx3090"},
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

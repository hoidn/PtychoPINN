import pytest

from scripts.studies.paper_results_refresh import (
    cdi_display_metrics,
    detect_cns_history5_gaps,
    render_brdt_metrics_table,
)


def test_detect_cns_history5_gaps_reports_missing_main_comparators():
    available = {
        "author_ffno_cns_base": {"history_len": 5, "epochs": 40},
        "spectral_resnet_bottleneck_base": {"history_len": 5, "epochs": 40},
    }
    required = [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]

    gaps = detect_cns_history5_gaps(available, required_rows=required)

    assert gaps == ["fno_base", "unet_strong"]


def test_render_brdt_metrics_table_keeps_blocked_classical_row():
    metrics = {
        "rows": [
            {"row_id": "classical_born_backprop", "row_status": "blocked"},
            {
                "row_id": "hybrid_resnet",
                "row_status": "completed",
                "image": {"image_relative_l2_phys": 0.319},
                "measurement": {"meas_relative_l2": 0.1992},
                "supporting": {"psnr_phys": 29.741, "ssim_phys": 0.9471},
            },
        ]
    }

    tex = render_brdt_metrics_table(metrics)

    assert "Classical Born backprop" in tex
    assert "blocked" in tex
    assert "Hybrid ResNet" in tex
    assert "0.319" in tex
    assert "0.199" in tex


def test_cdi_display_metrics_derives_rmse_and_keeps_uno_rows():
    metrics = {
        "pinn_hybrid_resnet": {
            "metrics": {
                "mae": [0.0269, 0.0721],
                "mse": [0.0016, 0.0081],
                "ssim": [0.9881, 0.9947],
            }
        },
        "pinn_neuralop_uno": {
            "metrics": {
                "mae": [0.0932, 0.0683],
                "mse": [0.0121, 0.0049],
                "ssim": [0.8280, 0.9569],
            }
        },
        "supervised_ffno": {
            "metrics": {
                "mae": [0.3864, 0.0466],
                "mse": [0.2664, 0.0033],
                "ssim": [0.2484, 0.9372],
            }
        },
    }

    rows = cdi_display_metrics(metrics)

    row_by_id = {row["row_id"]: row for row in rows}
    assert "supervised_ffno" in row_by_id
    assert row_by_id["pinn_neuralop_uno"]["amp_rmse"] == pytest.approx(0.11)
    assert row_by_id["pinn_hybrid_resnet"]["phase_rmse"] == pytest.approx(0.09)

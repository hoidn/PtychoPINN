import json

import numpy as np
import pytest

from scripts.studies.paper_results_refresh import (
    CNS_H2_FIXED_CONTRACT,
    CNS_H5_FIXED_CONTRACT,
    align_phase_to_reference,
    cdi_display_metrics,
    center_crop_bounds,
    detect_cns_history5_gaps,
    evaluate_cns_h5_lane,
    load_cns_h5_candidate_rows,
    render_brdt_metrics_table,
    render_cdi_objective_comparison_table,
    render_cdi_pinn_metrics_table,
    render_cns_matched_condition_tex,
    robust_display_bounds,
    select_cns_matched_condition,
    shared_display_bounds,
    gt_anchored_phase_bounds,
    srunet_matched_phase_bounds,
    write_cdi_phase_zoom_figure,
    write_cdi_phase_zoom_per_panel_figure,
    write_cns_matched_condition_assets,
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


def test_center_crop_bounds_uses_half_extent():
    assert center_crop_bounds((270, 270), fraction=0.5) == (67, 202, 67, 202)


def test_align_phase_to_reference_removes_global_offset():
    reference = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    shifted = reference + 1.25

    aligned = align_phase_to_reference(shifted, reference)

    np.testing.assert_allclose(aligned, reference, atol=1e-6)


def test_write_cdi_phase_zoom_figure_uses_phase_only_panel_order(tmp_path):
    recons = tmp_path / "recons"
    for row_id, phase in {
        "gt": np.linspace(-0.4, 0.8, 64, dtype=np.float32).reshape(8, 8),
        "pinn": np.linspace(-0.5, 0.9, 64, dtype=np.float32).reshape(8, 8),
        "pinn_fno_vanilla": np.linspace(-0.3, 0.6, 64, dtype=np.float32).reshape(8, 8),
        "pinn_ffno": np.linspace(-0.2, 0.7, 64, dtype=np.float32).reshape(8, 8),
        "pinn_neuralop_uno": np.linspace(-0.25, 0.65, 64, dtype=np.float32).reshape(8, 8),
        "pinn_hybrid_resnet": np.linspace(-0.35, 0.75, 64, dtype=np.float32).reshape(8, 8),
    }.items():
        row_dir = recons / row_id
        row_dir.mkdir(parents=True)
        np.savez(row_dir / "recon.npz", phase=phase)
    output = tmp_path / "phase_zoom.png"

    meta = write_cdi_phase_zoom_figure(recons_root=recons, output_path=output)

    assert output.exists()
    assert meta["visible_rows"] == [
        "gt",
        "pinn",
        "pinn_fno_vanilla",
        "pinn_ffno",
        "pinn_neuralop_uno",
        "pinn_hybrid_resnet",
    ]
    assert meta["display_channel"] == "phase"
    assert meta["crop_fraction"] == 0.5
    assert meta["phase_colormap"] == "twilight"
    assert meta["phase_display_scale"] == "gt_crop_min_to_gt_crop_p99_after_alignment"
    assert meta["phase_display_bounds"] != [-np.pi, np.pi]


def test_shared_display_bounds_uses_data_range_with_degenerate_padding():
    vmin, vmax = shared_display_bounds(
        [
            np.array([[0.1, 0.4]], dtype=np.float32),
            np.array([[-0.2, 0.3]], dtype=np.float32),
        ]
    )

    assert vmin == pytest.approx(-0.2)
    assert vmax == pytest.approx(0.4)

    vmin, vmax = shared_display_bounds([np.zeros((2, 2), dtype=np.float32)])

    assert (vmin, vmax) == (-1.0, 1.0)


def test_robust_display_bounds_uses_percentiles():
    values = np.array([[-10.0, 0.0, 1.0, 2.0, 100.0]], dtype=np.float32)

    vmin, vmax = robust_display_bounds(values, lower_quantile=0.2, upper_quantile=0.8)

    assert vmin == pytest.approx(float(np.nanquantile(values, 0.2)))
    assert vmax == pytest.approx(float(np.nanquantile(values, 0.8)))


def test_gt_anchored_phase_bounds_uses_gt_crop_min_and_p99():
    display_phases = {
        "gt": np.array(
            [
                [-0.5, 0.1, 0.7, 1.1],
                [-0.4, 0.2, 0.8, 1.2],
                [-0.3, 0.3, 0.9, 1.3],
                [-0.2, 0.4, 1.0, 1.4],
            ],
            dtype=np.float32,
        ),
        "pinn_hybrid_resnet": np.array(
            [
                [-0.4, 0.0, 0.2, 0.3],
                [-0.3, 0.1, 0.4, 0.5],
                [-0.2, 0.2, 0.6, 0.7],
                [-0.1, 0.3, 0.8, 0.9],
            ],
            dtype=np.float32,
        ),
    }

    vmin, vmax = gt_anchored_phase_bounds(display_phases, [1, 3, 1, 3])

    assert vmin == pytest.approx(0.2)
    assert vmax == pytest.approx(np.quantile([0.2, 0.8, 0.3, 0.9], 0.99))


def test_gt_anchored_phase_bounds_ignores_extra_row_outliers():
    display_phases = {
        "gt": np.array([[-0.5, 0.1], [0.2, 1.4]], dtype=np.float32),
        "pinn": np.array([[-1.2, 0.0], [0.3, 1.8]], dtype=np.float32),
        "pinn_hybrid_resnet": np.array([[-0.4, 0.1], [0.4, 0.7]], dtype=np.float32),
    }

    vmin, vmax = gt_anchored_phase_bounds(display_phases, [0, 2, 0, 2])

    assert vmin == pytest.approx(-0.5)
    assert vmax == pytest.approx(np.quantile([-0.5, 0.1, 0.2, 1.4], 0.99))


def test_srunet_matched_phase_bounds_aliases_gt_anchored_rule():
    display_phases = {
        "gt": np.array([[-0.5, 0.1], [0.2, 1.4]], dtype=np.float32),
        "pinn_hybrid_resnet": np.array([[-0.4, 0.1], [0.4, 0.7]], dtype=np.float32),
    }

    assert srunet_matched_phase_bounds(display_phases, [0, 2, 0, 2]) == gt_anchored_phase_bounds(
        display_phases,
        [0, 2, 0, 2],
    )


def test_write_cdi_phase_zoom_per_panel_figure_records_independent_bounds(tmp_path):
    recons = tmp_path / "recons"
    for row_id, phase in {
        "gt": np.linspace(-0.4, 0.8, 64, dtype=np.float32).reshape(8, 8),
        "pinn": np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(8, 8),
        "pinn_fno_vanilla": np.linspace(-0.3, 0.6, 64, dtype=np.float32).reshape(8, 8),
        "pinn_ffno": np.linspace(-0.2, 0.7, 64, dtype=np.float32).reshape(8, 8),
        "pinn_neuralop_uno": np.linspace(-0.25, 0.65, 64, dtype=np.float32).reshape(8, 8),
        "pinn_hybrid_resnet": np.linspace(-0.35, 0.75, 64, dtype=np.float32).reshape(8, 8),
    }.items():
        row_dir = recons / row_id
        row_dir.mkdir(parents=True)
        np.savez(row_dir / "recon.npz", phase=phase)
    output = tmp_path / "phase_zoom_per_panel.png"

    meta = write_cdi_phase_zoom_per_panel_figure(recons_root=recons, output_path=output)

    assert output.exists()
    assert meta["phase_display_scale"] == "per_panel_p01_to_p99_after_alignment"
    assert set(meta["phase_display_bounds_by_row"]) == {
        "gt",
        "pinn",
        "pinn_fno_vanilla",
        "pinn_ffno",
        "pinn_neuralop_uno",
        "pinn_hybrid_resnet",
    }
    assert meta["phase_display_bounds_by_row"]["pinn"] != meta["phase_display_bounds_by_row"]["gt"]
    assert "not comparable across panels" in meta["caption_note"]


def test_render_brdt_metrics_table_keeps_model_based_classical_row():
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

    assert "Model-based Born inverse" in tex
    assert "Status" not in tex
    assert "blocked" not in tex
    assert "SRU-Net" in tex
    assert "Hybrid ResNet" not in tex
    assert r"\textbf{0.319}" in tex
    assert r"\textbf{0.199}" in tex


def test_render_brdt_metrics_table_bolds_best_values_by_direction():
    metrics = {
        "rows": [
            {
                "row_id": "unet",
                "row_status": "completed",
                "image": {"image_relative_l2_phys": 0.2},
                "measurement": {"meas_relative_l2": 0.8},
                "supporting": {"psnr_phys": 24.0, "ssim_phys": 0.7},
            },
            {
                "row_id": "hybrid_resnet",
                "paper_label": "Hybrid ResNet",
                "row_status": "completed",
                "image": {"image_relative_l2_phys": 0.3},
                "measurement": {"meas_relative_l2": 0.5},
                "supporting": {"psnr_phys": 29.0, "ssim_phys": 0.9},
            },
        ]
    }

    tex = render_brdt_metrics_table(metrics)

    assert r"\textbf{0.200}" in tex
    assert r"\textbf{0.500}" in tex
    assert r"\textbf{29.00}" in tex
    assert r"\textbf{0.900}" in tex
    assert "Hybrid ResNet" not in tex


def test_cdi_display_metrics_keeps_uno_rows_without_rmse():
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
    assert "amp_rmse" not in row_by_id["pinn_neuralop_uno"]
    assert "phase_rmse" not in row_by_id["pinn_hybrid_resnet"]
    assert row_by_id["pinn_neuralop_uno"]["phase_mse"] == pytest.approx(0.0049)


def test_render_cdi_pinn_metrics_table_keeps_only_pinn_rows():
    rows = [
        {
            "row_id": "pinn",
            "model": "CNN",
            "training": "PINN",
            "amp_mae": 0.1,
            "phase_mae": 0.2,
            "amp_mse": 0.01,
            "phase_mse": 0.04,
            "amp_ssim": 0.9,
            "phase_ssim": 0.8,
        },
        {
            "row_id": "baseline",
            "model": "CNN",
            "training": "supervised",
            "amp_mae": 0.3,
            "phase_mae": 0.4,
            "amp_mse": 0.09,
            "phase_mse": 0.16,
            "amp_ssim": 0.7,
            "phase_ssim": 0.6,
        },
    ]

    tex = render_cdi_pinn_metrics_table(rows)

    assert "CNN" in tex
    assert r"Amp MAE $\downarrow$" in tex
    assert r"Phase SSIM $\uparrow$" in tex
    assert "0.1000" in tex
    assert "0.3000" not in tex
    assert "Training" not in tex


def test_render_cdi_pinn_metrics_table_bolds_best_values_across_pinn_models():
    rows = [
        {
            "row_id": "pinn_a",
            "model": "A",
            "training": "PINN",
            "amp_mae": 0.1,
            "phase_mae": 0.4,
            "amp_mse": 0.01,
            "phase_mse": 0.16,
            "amp_ssim": 0.7,
            "phase_ssim": 0.6,
        },
        {
            "row_id": "pinn_b",
            "model": "B",
            "training": "PINN",
            "amp_mae": 0.2,
            "phase_mae": 0.2,
            "amp_mse": 0.04,
            "phase_mse": 0.04,
            "amp_ssim": 0.9,
            "phase_ssim": 0.8,
        },
    ]

    tex = render_cdi_pinn_metrics_table(rows)

    assert r"\textbf{0.1000}" in tex
    assert r"\textbf{0.2000}" in tex
    assert r"\textbf{0.9000}" in tex
    assert r"\textbf{0.8000}" in tex


def test_render_cdi_objective_comparison_table_keeps_only_paired_models():
    rows = [
        {
            "row_id": "pinn",
            "model": "CNN",
            "training": "PINN",
            "amp_mae": 0.1,
            "phase_mae": 0.2,
            "amp_mse": 0.01,
            "phase_mse": 0.04,
            "amp_ssim": 0.9,
            "phase_ssim": 0.8,
        },
        {
            "row_id": "baseline",
            "model": "CNN",
            "training": "supervised",
            "amp_mae": 0.3,
            "phase_mae": 0.4,
            "amp_mse": 0.09,
            "phase_mse": 0.16,
            "amp_ssim": 0.7,
            "phase_ssim": 0.6,
        },
        {
            "row_id": "pinn_hybrid_resnet",
            "model": "SRU-Net",
            "training": "PINN",
            "amp_mae": 0.05,
            "phase_mae": 0.1,
            "amp_mse": 0.0025,
            "phase_mse": 0.01,
            "amp_ssim": 0.95,
            "phase_ssim": 0.9,
        },
    ]

    tex = render_cdi_objective_comparison_table(rows)

    assert r"\multicolumn{5}{l}{\textit{CNN}}" in tex
    assert "Amp MSE" not in tex
    assert "Phase MSE" not in tex
    assert "PINN" in tex
    assert "Supervised" in tex
    assert "SRU-Net" not in tex


def _h2_lane_fixture():
    rows_by_id = {
        "author_ffno_cns_base": {
            "row_id": "author_ffno_cns_base",
            "split_label": "2048 / 256 / 256",
            "err_nRMSE": 0.0263,
            "err_RMSE": 0.6364,
            "relative_l2": 0.0263,
            "fRMSE_low": 1.5135,
            "fRMSE_mid": 0.0697,
            "fRMSE_high": 0.0672,
            "parameter_count": 1073672,
            "runtime_sec": 18909.0,
            "source_run_root": "h2/author_ffno",
        },
        "spectral_resnet_bottleneck_base": {
            "row_id": "spectral_resnet_bottleneck_base",
            "split_label": "2048 / 256 / 256",
            "err_nRMSE": 0.0422,
            "err_RMSE": 1.0198,
            "relative_l2": 0.0422,
            "fRMSE_low": 2.3713,
            "fRMSE_mid": 0.2231,
            "fRMSE_high": 0.3118,
            "parameter_count": 8391814,
            "runtime_sec": 4311.2,
            "source_run_root": "h2/spectral",
        },
        "fno_base": {
            "row_id": "fno_base",
            "split_label": "2048 / 256 / 256",
            "err_nRMSE": 0.0507,
            "err_RMSE": 1.2268,
            "relative_l2": 0.0507,
            "fRMSE_low": 2.8394,
            "fRMSE_mid": 0.1917,
            "fRMSE_high": 0.4954,
            "parameter_count": 357860,
            "runtime_sec": 3135.4,
            "source_run_root": "h2/fno",
        },
        "unet_strong": {
            "row_id": "unet_strong",
            "split_label": "2048 / 256 / 256",
            "err_nRMSE": 0.6432,
            "err_RMSE": 15.555,
            "relative_l2": 0.6432,
            "fRMSE_low": 37.154,
            "fRMSE_mid": 0.4026,
            "fRMSE_high": 0.7355,
            "parameter_count": 7764580,
            "runtime_sec": 2898.5,
            "source_run_root": "h2/unet",
        },
    }
    return {
        "lane_id": "h2_2048_256_256_40ep",
        "contract": dict(CNS_H2_FIXED_CONTRACT),
        "summary_authority": "docs/plans/.../h2_summary.md",
        "contract_authority": "docs/plans/.../h2_contract.md",
        "source_bundle_json": "fixture/h2.json",
        "rows_by_id": rows_by_id,
    }


def _h5_row(profile_id: str, **overrides):
    base = {
        "profile_id": profile_id,
        "history_len": 5,
        "epochs": 40,
        "batch_size": 4,
        "training_loss": "mse",
        "split_counts": {"train": 512, "val": 64, "test": 64},
        "max_windows_per_trajectory": 8,
        "metric_family": [
            "err_RMSE",
            "err_nRMSE",
            "relative_l2",
            "fRMSE_low",
            "fRMSE_mid",
            "fRMSE_high",
        ],
        "status": "completed",
        "err_nRMSE": 0.04,
        "err_RMSE": 1.0,
        "relative_l2": 0.04,
        "fRMSE_low": 2.0,
        "fRMSE_mid": 0.1,
        "fRMSE_high": 0.3,
        "parameter_count": 100000,
        "runtime_sec": 1000.0,
        "run_root": f"h5/{profile_id}",
    }
    base.update(overrides)
    return base


def _h5_lane_fixture(rows=None):
    if rows is None:
        rows = {
            "author_ffno_cns_base": _h5_row(
                "author_ffno_cns_base", err_nRMSE=0.0198, fRMSE_high=0.1018
            ),
            "spectral_resnet_bottleneck_base": _h5_row(
                "spectral_resnet_bottleneck_base", err_nRMSE=0.0331, fRMSE_high=0.2622
            ),
            "fno_base": _h5_row("fno_base", err_nRMSE=0.0384, fRMSE_high=0.4329),
            "unet_strong": _h5_row("unet_strong", err_nRMSE=0.5386, fRMSE_high=1.7428),
        }
    return {
        "lane_id": "h5_512_64_64_40ep",
        "contract": dict(CNS_H5_FIXED_CONTRACT),
        "summary_authority": "docs/plans/.../h5_summary.md",
        "source_compare_json": "fixture/h5.json",
        "rows_by_id": rows,
    }


def test_evaluate_cns_h5_lane_complete_when_all_rows_consistent():
    eval_payload = evaluate_cns_h5_lane(_h5_lane_fixture())
    assert eval_payload["is_complete_and_consistent"] is True
    assert eval_payload["missing_rows"] == []
    assert eval_payload["inconsistent_rows"] == []


def test_evaluate_cns_h5_lane_flags_missing_rows():
    fixture = _h5_lane_fixture()
    del fixture["rows_by_id"]["fno_base"]
    eval_payload = evaluate_cns_h5_lane(fixture)
    assert eval_payload["is_complete_and_consistent"] is False
    assert "fno_base" in eval_payload["missing_rows"]


def test_evaluate_cns_h5_lane_flags_inconsistent_rows():
    fixture = _h5_lane_fixture()
    fixture["rows_by_id"]["unet_strong"] = _h5_row("unet_strong", history_len=2)
    eval_payload = evaluate_cns_h5_lane(fixture)
    assert eval_payload["is_complete_and_consistent"] is False
    inconsistent_ids = [item["row_id"] for item in eval_payload["inconsistent_rows"]]
    assert "unet_strong" in inconsistent_ids


def test_select_cns_matched_condition_picks_h5_when_complete():
    decision = select_cns_matched_condition(
        h2_lane=_h2_lane_fixture(), h5_lane=_h5_lane_fixture()
    )
    assert decision["selected_lane_id"] == "h5_512_64_64_40ep"
    assert decision["selected_contract"]["history_len"] == 5
    assert [row["row_id"] for row in decision["selected_rows"]] == [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]
    assert decision["rejected_candidate"] is None
    assert decision["claim_boundary"] == "bounded_capped_decision_support_only"


def test_select_cns_matched_condition_falls_back_when_h5_incomplete():
    h5 = _h5_lane_fixture()
    del h5["rows_by_id"]["unet_strong"]
    decision = select_cns_matched_condition(h2_lane=_h2_lane_fixture(), h5_lane=h5)
    assert decision["selected_lane_id"] == "h2_2048_256_256_40ep"
    assert decision["selected_contract"]["history_len"] == 2
    assert decision["rejected_candidate"]["lane_id"] == "h5_512_64_64_40ep"
    assert "unet_strong" in decision["rejected_candidate"]["missing_rows"]


def test_select_cns_matched_condition_falls_back_when_h5_inconsistent():
    h5 = _h5_lane_fixture()
    h5["rows_by_id"]["fno_base"] = _h5_row("fno_base", training_loss="mae")
    decision = select_cns_matched_condition(h2_lane=_h2_lane_fixture(), h5_lane=h5)
    assert decision["selected_lane_id"] == "h2_2048_256_256_40ep"
    rejected_inconsistent_ids = [
        item["row_id"] for item in decision["rejected_candidate"]["inconsistent_rows"]
    ]
    assert "fno_base" in rejected_inconsistent_ids


def test_render_cns_matched_condition_tex_uses_manuscript_labels():
    decision = select_cns_matched_condition(
        h2_lane=_h2_lane_fixture(), h5_lane=_h5_lane_fixture()
    )
    tex = render_cns_matched_condition_tex(decision)
    assert "FFNO" in tex
    assert "SRU-Net" in tex
    assert "FNO" in tex
    assert "U-Net" in tex
    # nRMSE format
    assert "0.0198" in tex


def test_write_cns_matched_condition_assets_emits_required_payload(tmp_path):
    decision = select_cns_matched_condition(
        h2_lane=_h2_lane_fixture(), h5_lane=_h5_lane_fixture()
    )
    paths = write_cns_matched_condition_assets(decision, output_root=tmp_path)
    for key in [
        "matched_condition_decision_json",
        "cns_paper_table_rows_json",
        "cns_paper_table_rows_csv",
        "cns_paper_table_rows_tex",
        "source_lineage_json",
        "figure_selection_json",
    ]:
        assert paths[key]
    decision_payload = json.loads(
        (tmp_path / "matched_condition_decision.json").read_text()
    )
    assert decision_payload["selected_lane_id"] == "h5_512_64_64_40ep"
    table_payload = json.loads((tmp_path / "cns_paper_table_rows.json").read_text())
    assert [row["row_id"] for row in table_payload["rows"]] == [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]
    assert table_payload["fixed_contract"]["history_len"] == 5
    figure_payload = json.loads((tmp_path / "figure_selection.json").read_text())
    assert figure_payload["same_condition_visuals_available"] is False


def test_evaluate_cns_h5_lane_flags_max_windows_per_trajectory_mismatch():
    fixture = _h5_lane_fixture()
    fixture["rows_by_id"]["author_ffno_cns_base"] = _h5_row(
        "author_ffno_cns_base", max_windows_per_trajectory=4
    )
    eval_payload = evaluate_cns_h5_lane(fixture)
    assert eval_payload["is_complete_and_consistent"] is False
    issues_by_row = {
        item["row_id"]: item["issues"] for item in eval_payload["inconsistent_rows"]
    }
    assert "max_windows_per_trajectory_mismatch" in issues_by_row["author_ffno_cns_base"]


def test_evaluate_cns_h5_lane_flags_missing_max_windows_per_trajectory():
    fixture = _h5_lane_fixture()
    row = _h5_row("fno_base")
    del row["max_windows_per_trajectory"]
    fixture["rows_by_id"]["fno_base"] = row
    eval_payload = evaluate_cns_h5_lane(fixture)
    assert eval_payload["is_complete_and_consistent"] is False
    issues_by_row = {
        item["row_id"]: item["issues"] for item in eval_payload["inconsistent_rows"]
    }
    assert "max_windows_per_trajectory_missing" in issues_by_row["fno_base"]


def test_select_cns_matched_condition_falls_back_when_max_windows_per_trajectory_disagrees():
    h5 = _h5_lane_fixture()
    h5["rows_by_id"]["unet_strong"] = _h5_row("unet_strong", max_windows_per_trajectory=16)
    decision = select_cns_matched_condition(h2_lane=_h2_lane_fixture(), h5_lane=h5)
    assert decision["selected_lane_id"] == "h2_2048_256_256_40ep"
    rejected_inconsistent_ids = [
        item["row_id"] for item in decision["rejected_candidate"]["inconsistent_rows"]
    ]
    assert "unet_strong" in rejected_inconsistent_ids


def test_load_cns_h5_candidate_rows_rejects_missing_top_level_contract(tmp_path):
    compare_path = tmp_path / "compare_no_contract.json"
    compare_path.write_text(
        json.dumps({"profile_results": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing a top-level 'contract' block"):
        load_cns_h5_candidate_rows(compare_json=compare_path)


def test_load_cns_h5_candidate_rows_rejects_top_level_contract_disagreement(tmp_path):
    contract = dict(CNS_H5_FIXED_CONTRACT)
    contract["max_windows_per_trajectory"] = 4
    compare_path = tmp_path / "compare_bad_contract.json"
    compare_path.write_text(
        json.dumps({"contract": contract, "profile_results": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="max_windows_per_trajectory_mismatch"):
        load_cns_h5_candidate_rows(compare_json=compare_path)


def test_load_cns_h5_candidate_rows_rejects_top_level_contract_history_disagreement(tmp_path):
    contract = dict(CNS_H5_FIXED_CONTRACT)
    contract["history_len"] = 4
    compare_path = tmp_path / "compare_history_mismatch.json"
    compare_path.write_text(
        json.dumps({"contract": contract, "profile_results": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="history_len_mismatch"):
        load_cns_h5_candidate_rows(compare_json=compare_path)


def test_load_cns_h5_candidate_rows_returns_source_contract_when_consistent(tmp_path):
    contract = dict(CNS_H5_FIXED_CONTRACT)
    compare_path = tmp_path / "compare_ok.json"
    compare_path.write_text(
        json.dumps({"contract": contract, "profile_results": []}),
        encoding="utf-8",
    )
    h5 = load_cns_h5_candidate_rows(compare_json=compare_path)
    assert h5["lane_id"] == "h5_512_64_64_40ep"
    assert h5["source_contract"]["history_len"] == 5
    assert h5["source_contract"]["max_windows_per_trajectory"] == 8

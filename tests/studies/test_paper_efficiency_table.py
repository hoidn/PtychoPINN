import json
from pathlib import Path

from scripts.studies.paper_efficiency_table import (
    _collect_cdi_rows,
    classify_inference_throughput,
    collect_efficiency_rows,
    group_rows_by_benchmark,
    normalize_efficiency_row,
    render_efficiency_table_tex,
)


def test_normalize_efficiency_row_preserves_runtime_source_field():
    row = normalize_efficiency_row(
        benchmark="PDEBench CNS",
        row_id="spectral_resnet_bottleneck_base",
        model_label="SRU-Net*",
        source_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json",
        payload={
            "parameter_count": 8186726,
            "runtime_sec": 1861.63,
            "hardware_runtime_note": "RTX 3090 provenance field",
        },
        claim_boundary="bounded_capped_decision_support_only",
    )

    assert row.training_runtime_seconds == 1861.63
    assert row.training_runtime_source_field == "runtime_sec"
    assert row.training_runtime_status == "provenance_context"
    assert row.source_path.endswith("pdebench_cns_matched_condition_metrics.json")


def test_normalize_efficiency_row_accepts_brdt_train_runtime_and_eval_throughput():
    row = normalize_efficiency_row(
        benchmark="BRDT",
        row_id="ffno",
        model_label="FFNO",
        source_path=".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/combined_metrics.json",
        payload={
            "runtime": {
                "parameter_count": 36674,
                "device_name": "NVIDIA GeForce RTX 3090",
                "wall_time_train_s": 171.4,
            },
            "samples_per_second": 375.5,
        },
        claim_boundary="paper_approved_secondary_brdt",
    )

    assert row.parameter_count == 36674
    assert row.training_runtime_seconds == 171.4
    assert row.training_runtime_source_field == "wall_time_train_s"
    assert row.inference_throughput_status == "measured"
    assert row.inference_samples_per_second == 375.5
    assert row.claim_boundary == "paper_approved_secondary_brdt"


def test_classify_inference_throughput_keeps_measured_value():
    result = classify_inference_throughput(
        {"samples_per_second": 385.3},
        source_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.json",
    )

    assert result.status == "measured"
    assert result.samples_per_second == 385.3
    assert result.source_field == "samples_per_second"


def test_classify_inference_throughput_does_not_promote_training_runtime():
    result = classify_inference_throughput(
        {"runtime_sec": 1861.63},
        source_path="docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json",
    )

    assert result.status == "missing"
    assert result.samples_per_second is None
    assert result.source_field is None


def test_group_rows_by_benchmark_preserves_claim_boundaries():
    rows = [
        {
            "benchmark": "CDI Lines128",
            "row_id": "pinn_hybrid_resnet",
            "claim_boundary": "paper_table",
        },
        {
            "benchmark": "PDEBench CNS",
            "row_id": "spectral_resnet_bottleneck_base",
            "claim_boundary": "bounded_capped_decision_support_only",
        },
        {
            "benchmark": "BRDT",
            "row_id": "ffno",
            "claim_boundary": "decision_support_convergence_followup",
        },
    ]

    grouped = group_rows_by_benchmark(rows)

    assert list(grouped) == ["CDI Lines128", "PDEBench CNS", "BRDT"]
    assert grouped["PDEBench CNS"][0]["claim_boundary"] == (
        "bounded_capped_decision_support_only"
    )


def test_write_paper_efficiency_table_records_per_row_active_ffno_provenance(tmp_path):
    from scripts.studies.paper_efficiency_table import write_paper_efficiency_table

    write_paper_efficiency_table(
        repo_root=Path.cwd(),
        output_dir=tmp_path,
        cdi_final_ffno_pair_key="depth24_no_refiner",
    )
    payload = json.loads(
        (tmp_path / "paper_efficiency_table.json").read_text(encoding="utf-8")
    )
    rows_by_id = {
        (row["benchmark"], row["row_id"]): row for row in payload["rows"]
    }

    pinn_ffno = rows_by_id[("CDI", "pinn_ffno")]
    assert pinn_ffno["final_ffno_pair_key"] == "depth24_no_refiner"
    assert pinn_ffno["final_ffno_depth_label"] == "depth24_no_refiner"
    assert pinn_ffno["claim_boundary"] == (
        "complete_lines128_cdi_benchmark_plus_uno_extension_"
        "with_final_depth24_no_refiner_ffno_pair"
    )
    assert pinn_ffno["source_root"].endswith(
        "2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z"
    )
    assert pinn_ffno["source_metrics_json"].endswith(
        "ffno_depth24_20260507T052301Z/runs/pinn_ffno_depth24/metrics.json"
    )
    assert "historical_proxy_lineage" not in pinn_ffno

    supervised_ffno = rows_by_id[("CDI", "supervised_ffno")]
    assert supervised_ffno["final_ffno_pair_key"] == "depth24_no_refiner"
    assert supervised_ffno["claim_boundary"] == pinn_ffno["claim_boundary"]
    assert supervised_ffno["historical_proxy_lineage"]["metrics_json"].endswith(
        "supervised_ffno_extension_20260430T180217Z/runs/supervised_ffno/metrics.json"
    )

    pinn_resnet = rows_by_id[("CDI", "pinn_hybrid_resnet")]
    assert "final_ffno_pair_key" not in pinn_resnet
    assert "source_root" not in pinn_resnet


def test_collect_efficiency_rows_includes_paper_approved_brdt_rows():
    rows = collect_efficiency_rows()
    brdt_rows = [row for row in rows if row.benchmark == "BRDT"]

    assert {row.row_id for row in brdt_rows} == {"sru_net", "ffno"}
    assert all(row.inference_throughput_status == "measured" for row in brdt_rows)
    assert all(row.claim_boundary == "paper_evidence_brdt_additive" for row in brdt_rows)
    assert {row.row_id: row.model_label for row in brdt_rows}["sru_net"] == "SRU-Net"


def test_collect_efficiency_rows_uses_unique_model_config_counts_for_cdi():
    rows = collect_efficiency_rows()
    cdi_rows = {
        row.row_id: row
        for row in rows
        if row.benchmark == "CDI"
    }

    assert cdi_rows["pinn_hybrid_resnet"].parameter_count == 9_003_299
    assert cdi_rows["pinn_hybrid_resnet"].parameter_count_source_field == "unique_trainable_params"
    assert cdi_rows["pinn_ffno"].parameter_count == 124_966
    assert cdi_rows["pinn_ffno"].model_label == "FFNO + PINN"
    assert cdi_rows["supervised_ffno"].parameter_count == 124_966
    assert cdi_rows["supervised_ffno"].model_label == "FFNO + supervised"


def test_collect_efficiency_rows_can_switch_to_depth24_pair():
    rows = _collect_cdi_rows(Path.cwd(), cdi_final_ffno_pair_key="depth24_no_refiner")
    cdi_rows = {row.row_id: row for row in rows}

    assert cdi_rows["pinn_ffno"].parameter_count == 701_628
    assert cdi_rows["pinn_ffno"].source_path.endswith(
        "2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/model_manifest.json"
    )
    assert cdi_rows["supervised_ffno"].parameter_count == 136_355
    assert cdi_rows["supervised_ffno"].source_path.endswith(
        "2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z/model_manifest.json"
    )


def test_collect_cdi_rows_uses_final_pair_claim_boundary_for_active_ffno_rows():
    rows = _collect_cdi_rows(Path.cwd(), cdi_final_ffno_pair_key="depth24_no_refiner")
    cdi_rows = {row.row_id: row for row in rows}

    depth24_boundary = (
        "complete_lines128_cdi_benchmark_plus_uno_extension_"
        "with_final_depth24_no_refiner_ffno_pair"
    )
    assert cdi_rows["pinn_ffno"].claim_boundary == depth24_boundary
    assert cdi_rows["supervised_ffno"].claim_boundary == depth24_boundary
    # Non-FFNO CDI rows continue to inherit their lineage claim boundary.
    assert cdi_rows["pinn_hybrid_resnet"].claim_boundary != depth24_boundary


def test_render_efficiency_table_tex_groups_rows_and_escapes_fields():
    tex = render_efficiency_table_tex(
        [
            {
                "benchmark": "PDEBench CNS",
                "row_id": "spectral_resnet_bottleneck_base",
                "model_label": "SRU-Net*",
                "parameter_count": 8186726,
                "training_runtime_seconds": 1861.63,
                "training_runtime_source_field": "runtime_sec",
                "inference_throughput_status": "missing",
                "inference_samples_per_second": None,
                "claim_boundary": "bounded_capped_decision_support_only",
            }
        ]
    )

    assert "PDEBench CNS" in tex
    assert "spectral\\_resnet\\_bottleneck\\_base" in tex
    assert "runtime\\_sec" in tex
    assert "normalized throughput" not in tex.lower()

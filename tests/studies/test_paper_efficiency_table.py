from scripts.studies.paper_efficiency_table import (
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


def test_collect_efficiency_rows_includes_paper_approved_brdt_rows():
    rows = collect_efficiency_rows()
    brdt_rows = [row for row in rows if row.benchmark == "BRDT"]

    assert {row.row_id for row in brdt_rows} == {"hybrid_resnet", "ffno"}
    assert all(row.inference_throughput_status == "measured" for row in brdt_rows)
    assert all(row.claim_boundary == "paper_approved_secondary_brdt" for row in brdt_rows)


def test_collect_efficiency_rows_uses_unique_model_config_counts_for_cdi():
    rows = collect_efficiency_rows()
    cdi_rows = {
        row.row_id: row
        for row in rows
        if row.benchmark == "Synthetic CDI"
    }

    assert cdi_rows["pinn_hybrid_resnet"].parameter_count == 9_003_299
    assert cdi_rows["pinn_hybrid_resnet"].parameter_count_source_field == "unique_trainable_params"
    assert cdi_rows["pinn_ffno"].parameter_count == 161_958


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

"""Tests for the CDI hybrid-spectral to FFNO parameter-space study harness."""

from __future__ import annotations

import json
from pathlib import Path


AUTHORITATIVE_ROOT = (
    Path(".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog")
    / "2026-04-29-cdi-lines128-paper-benchmark-execution/runs"
    / "complete_table_20260430T150757Z_repair_tmux"
)


def test_preflight_artifacts_freeze_exact_six_row_matrix(tmp_path):
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
        build_preflight_artifacts,
    )

    note_path = tmp_path / "cdi_preflight.md"
    matrix_path = tmp_path / "preflight" / "study_matrix.json"
    reference_path = tmp_path / "preflight" / "reference_runs.json"

    result = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path,
        note_path=note_path,
        matrix_path=matrix_path,
        reference_runs_path=reference_path,
    )

    assert result["study_matrix_path"] == matrix_path
    assert result["reference_runs_path"] == reference_path
    assert result["note_path"] == note_path
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert [row["model_id"] for row in matrix["rows"]] == [
        "pinn_hybrid_resnet",
        "pinn_spectral_resnet_bottleneck_net",
        "pinn_ffno",
        "pinn_spectral_resnet_bottleneck_ds1",
        "pinn_spectral_resnet_bottleneck_linear_decoder",
        "pinn_hybrid_resnet_ffno_bottleneck",
    ]


def test_reference_runs_manifest_maps_reused_rows_to_authoritative_root(tmp_path):
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import (
        build_preflight_artifacts,
    )

    result = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path,
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )

    payload = json.loads(Path(result["reference_runs_path"]).read_text(encoding="utf-8"))

    assert payload["authoritative_root"] == str(AUTHORITATIVE_ROOT)
    assert [row["model_id"] for row in payload["reused_rows"]] == [
        "pinn_hybrid_resnet",
        "pinn_spectral_resnet_bottleneck_net",
        "pinn_ffno",
    ]
    assert payload["reused_rows"][0]["run_dir"].endswith("runs/pinn_hybrid_resnet")


def test_runbook_preflight_only_validates_fresh_rows_without_training(monkeypatch, tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        run_cdi_parameter_space_study,
    )

    captured = {}

    def fake_compare(**kwargs):
        captured["models"] = kwargs["models"]
        captured["preflight_only"] = kwargs["preflight_only"]
        captured["row_specs"] = kwargs["row_specs"]
        return {
            "mode": "preflight_only",
            "selected_models": list(kwargs["models"]),
        }

    monkeypatch.setattr(
        "scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space.run_grid_lines_compare",
        fake_compare,
    )

    result = run_cdi_parameter_space_study(
        authoritative_root=AUTHORITATIVE_ROOT,
        output_root=tmp_path / "study",
        preflight_root=tmp_path / "preflight",
        note_path=tmp_path / "cdi_preflight.md",
        preflight_only=True,
    )

    assert captured["preflight_only"] is True
    assert captured["models"] == (
        "pinn_spectral_resnet_bottleneck_ds1",
        "pinn_spectral_resnet_bottleneck_linear_decoder",
        "pinn_hybrid_resnet_ffno_bottleneck",
    )
    assert [spec["model_id"] for spec in captured["row_specs"]] == list(captured["models"])
    assert result["preflight_validation"]["selected_models"] == list(captured["models"])


def test_validate_bundle_reports_missing_frozen_rows(tmp_path):
    from scripts.studies.runbooks.run_cdi_hybrid_spectral_ffno_parameter_space import (
        validate_cdi_parameter_space_bundle,
    )
    from scripts.studies.cdi_hybrid_spectral_ffno_parameter_space import build_preflight_artifacts

    paths = build_preflight_artifacts(
        authoritative_root=AUTHORITATIVE_ROOT,
        artifact_root=tmp_path / "study",
        note_path=tmp_path / "cdi_preflight.md",
        matrix_path=tmp_path / "preflight" / "study_matrix.json",
        reference_runs_path=tmp_path / "preflight" / "reference_runs.json",
    )

    report = validate_cdi_parameter_space_bundle(
        output_root=tmp_path / "study",
        study_matrix_path=paths["study_matrix_path"],
    )

    assert report["ok"] is False
    assert "pinn_hybrid_resnet" in report["missing_rows"]

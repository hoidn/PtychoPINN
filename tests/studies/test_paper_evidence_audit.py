"""Tests for the NeurIPS paper evidence audit."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_MODULE = "scripts.studies.paper_evidence_audit"
NEURIPS_ROOT = Path("/home/ollie/Documents/neurips")


def _load_audit_module():
    try:
        return importlib.import_module(AUDIT_MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing audit module {AUDIT_MODULE}: {exc}")


def _default_inputs():
    module = _load_audit_module()
    return module.build_default_input_manifest(REPO_ROOT)


def test_default_inputs_promote_2048cap_cns_authority_and_preserve_512_history():
    inputs = _default_inputs()
    cns_inputs = inputs["cns"]

    assert cns_inputs["bundle_summary_path"] == (
        "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md"
    )
    assert cns_inputs["bundle_root"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap"
    )
    assert cns_inputs["locked_rows_path"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json"
    )
    assert cns_inputs["historical_bundle_summary_path"] == (
        "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md"
    )
    assert cns_inputs["historical_bundle_root"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle"
    )


def test_build_manifest_from_real_authorities_preserves_distinct_pillar_statuses():
    module = _load_audit_module()
    manifest = module.build_manifest(_default_inputs(), repo_root=REPO_ROOT)

    cdi = manifest["pillar_summaries"]["cdi"]
    cns = manifest["pillar_summaries"]["cns"]

    assert cdi["headline_status"] == "paper_grade"
    assert cdi["claim_boundary"] == "complete_lines128_cdi_benchmark"
    assert cdi["selected_fno_comparator"] == "fno_vanilla"

    assert cns["headline_status"] == "capped_decision_support"
    assert cns["claim_boundary"] == "bounded_capped_decision_support_only"
    assert cns["bundle_status"] == "paper_complete"

    validation = module.validate_manifest(manifest)
    assert validation["cdi_headline_is_paper_grade"] is True
    assert validation["cns_headline_is_capped_decision_support"] is True
    assert validation["no_current_full_training_pillar"] is True
    assert validation["adjacent_cdi_context_not_promoted"] is True
    assert validation["no_output_targets_under_neurips"] is True


def test_load_cdi_authority_detects_same_pillar_source_disagreement(tmp_path):
    module = _load_audit_module()
    inputs = _default_inputs()
    cdi_inputs = dict(inputs["cdi"])

    model_manifest = json.loads(Path(cdi_inputs["model_manifest_path"]).read_text(encoding="utf-8"))
    model_manifest["benchmark_status"] = "benchmark_incomplete"
    authoritative_root = REPO_ROOT / cdi_inputs["authoritative_root"]
    tampered_model_manifest = authoritative_root / "model_manifest_tampered.json"
    tampered_model_manifest.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    cdi_inputs["model_manifest_path"] = str(tampered_model_manifest.relative_to(REPO_ROOT))

    try:
        with pytest.raises(ValueError, match="CDI.*benchmark status"):
            module.load_cdi_authority(cdi_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_model_manifest.unlink(missing_ok=True)


def test_load_cdi_authority_rejects_non_authoritative_bundle_file_paths(tmp_path):
    module = _load_audit_module()
    inputs = _default_inputs()
    cdi_inputs = dict(inputs["cdi"])

    copied_paths = {}
    for key in ("paper_manifest_path", "metrics_path", "model_manifest_path"):
        source_path = REPO_ROOT / cdi_inputs[key]
        copied_path = tmp_path / source_path.name
        copied_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied_paths[key] = str(copied_path)
    cdi_inputs.update(copied_paths)

    with pytest.raises(ValueError, match="CDI.*authoritative root identity"):
        module.load_cdi_authority(cdi_inputs, repo_root=REPO_ROOT)


def test_load_cns_authority_preserves_capped_claim_boundary():
    module = _load_audit_module()
    inputs = _default_inputs()

    cns = module.load_cns_authority(inputs["cns"], repo_root=REPO_ROOT)

    assert cns["headline_status"] == "capped_decision_support"
    assert cns["claim_boundary"] == "bounded_capped_decision_support_only"
    assert cns["bundle_status"] == "paper_complete"
    assert cns["headline_row_ids"] == [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]
    assert all(row["row_status"] == "capped_decision_support" for row in cns["headline_rows"])
    assert cns["source_summary"] == (
        "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md"
    )
    assert cns["source_root"] == (
        ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh"
    )
    assert cns["manuscript_headline_role"] == "matched_condition_history_len_5_512_64_64_40ep"
    assert sorted(cns["larger_cap_context"]["headline_row_ids"]) == [
        "author_ffno_cns_base",
        "fno_base",
        "spectral_resnet_bottleneck_base",
        "unet_strong",
    ]
    assert cns["larger_cap_context"]["bundle_root"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap"
    )


def test_load_cns_authority_rejects_matched_condition_lane_change():
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    decision = json.loads(
        Path(cns_inputs["matched_condition_decision_path"]).read_text(encoding="utf-8")
    )
    decision["selected_lane_id"] = "h2_2048_256_256_40ep"
    refresh_root = REPO_ROOT / cns_inputs["matched_condition_refresh_root"]
    tampered_decision = refresh_root / "matched_condition_decision_tampered.json"
    tampered_decision.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    cns_inputs["matched_condition_decision_path"] = str(
        tampered_decision.relative_to(REPO_ROOT)
    )

    try:
        with pytest.raises(ValueError, match="selected_lane_id"):
            module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_decision.unlink(missing_ok=True)


def test_load_cns_authority_rejects_matched_condition_claim_boundary_drift():
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    decision = json.loads(
        Path(cns_inputs["matched_condition_decision_path"]).read_text(encoding="utf-8")
    )
    decision["claim_boundary"] = "paper_grade"
    refresh_root = REPO_ROOT / cns_inputs["matched_condition_refresh_root"]
    tampered_decision = refresh_root / "matched_condition_decision_boundary_tampered.json"
    tampered_decision.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    cns_inputs["matched_condition_decision_path"] = str(
        tampered_decision.relative_to(REPO_ROOT)
    )

    try:
        with pytest.raises(ValueError, match="capped claim boundary"):
            module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_decision.unlink(missing_ok=True)


def test_load_cns_authority_rejects_matched_condition_row_roster_drift():
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    decision = json.loads(
        Path(cns_inputs["matched_condition_decision_path"]).read_text(encoding="utf-8")
    )
    decision["selected_row_ids"] = decision["selected_row_ids"][:2]
    decision["selected_rows"] = decision["selected_rows"][:2]
    refresh_root = REPO_ROOT / cns_inputs["matched_condition_refresh_root"]
    tampered_decision = refresh_root / "matched_condition_decision_roster_tampered.json"
    tampered_decision.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    cns_inputs["matched_condition_decision_path"] = str(
        tampered_decision.relative_to(REPO_ROOT)
    )

    try:
        with pytest.raises(ValueError, match="row roster"):
            module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_decision.unlink(missing_ok=True)


def test_load_cns_authority_detects_same_pillar_source_disagreement():
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    bundle_validation = json.loads(Path(cns_inputs["bundle_validation_path"]).read_text(encoding="utf-8"))
    bundle_validation["table_headline_row_ids"] = ["spectral_resnet_bottleneck_base", "fno_base"]
    bundle_root = REPO_ROOT / cns_inputs["bundle_root"]
    tampered_bundle_validation = bundle_root / "bundle_validation_tampered.json"
    tampered_bundle_validation.write_text(json.dumps(bundle_validation, indent=2), encoding="utf-8")
    cns_inputs["bundle_validation_path"] = str(tampered_bundle_validation.relative_to(REPO_ROOT))

    try:
        with pytest.raises(ValueError, match="CNS source disagreement on headline roster"):
            module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_bundle_validation.unlink(missing_ok=True)


def test_load_cns_authority_detects_conflicting_table_bundle_status_within_authoritative_root():
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    table_rows = json.loads(Path(cns_inputs["table_rows_path"]).read_text(encoding="utf-8"))
    table_rows["benchmark_status"] = "benchmark_incomplete"
    bundle_root = REPO_ROOT / cns_inputs["bundle_root"]
    tampered_table_rows = bundle_root / "cns_paper_table_rows_tampered.json"
    tampered_table_rows.write_text(json.dumps(table_rows, indent=2), encoding="utf-8")
    cns_inputs["table_rows_path"] = str(tampered_table_rows.relative_to(REPO_ROOT))

    try:
        with pytest.raises(ValueError, match="CNS source disagreement on benchmark status"):
            module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_table_rows.unlink(missing_ok=True)


def test_load_cns_authority_detects_same_root_figure_manifest_roster_disagreement():
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    figure_manifest = json.loads(Path(cns_inputs["figure_manifest_path"]).read_text(encoding="utf-8"))
    figure_manifest["rows_in_visual_bundle"] = ["spectral_resnet_bottleneck_base", "fno_base"]
    bundle_root = REPO_ROOT / cns_inputs["bundle_root"]
    tampered_figure_manifest = bundle_root / "figure_manifest_tampered.json"
    tampered_figure_manifest.write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    cns_inputs["figure_manifest_path"] = str(tampered_figure_manifest.relative_to(REPO_ROOT))

    try:
        with pytest.raises(ValueError, match="CNS source disagreement on visual roster"):
            module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)
    finally:
        tampered_figure_manifest.unlink(missing_ok=True)


def test_load_cns_authority_rejects_non_authoritative_bundle_file_paths(tmp_path):
    module = _load_audit_module()
    inputs = _default_inputs()
    cns_inputs = dict(inputs["cns"])

    copied_paths = {}
    for key in (
        "table_rows_path",
        "bundle_validation_path",
        "figure_manifest_path",
        "fixed_sample_manifest_path",
    ):
        source_path = REPO_ROOT / cns_inputs[key]
        copied_path = tmp_path / source_path.name
        copied_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied_paths[key] = str(copied_path)
    cns_inputs.update(copied_paths)

    with pytest.raises(ValueError, match="CNS.*authoritative root identity"):
        module.load_cns_authority(cns_inputs, repo_root=REPO_ROOT)


def test_repo_local_output_guard_rejects_neurips_output_root():
    module = _load_audit_module()

    with pytest.raises(ValueError, match=str(NEURIPS_ROOT)):
        module.ensure_repo_local_output_path(NEURIPS_ROOT / "index.md")


def test_direct_script_entrypoint_runs_audit_successfully():
    script_path = REPO_ROOT / "scripts/studies/paper_evidence_audit.py"

    result = subprocess.run(
        ["python", str(script_path), "--repo-root", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["manifest_path"] == "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json"
    assert payload["summary_path"] == (
        "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md"
    )
    assert payload["validation_path"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/"
        "audit_validation.json"
    )


def test_summary_mentions_same_authorities_and_claim_limits():
    module = _load_audit_module()
    inputs = _default_inputs()
    manifest = module.build_manifest(inputs, repo_root=REPO_ROOT)
    summary = module.render_audit_summary(manifest)

    assert inputs["cdi"]["authoritative_root"] in summary
    assert inputs["cns"]["bundle_root"] in summary
    assert inputs["cns"]["historical_bundle_root"] in summary
    assert "paper_grade" in summary
    assert "capped_decision_support" in summary
    assert "full-training CNS competitiveness claims remain blocked" in summary

    sync_validation = module.validate_summary_sync(manifest, summary)
    assert sync_validation["cdi_root_present"] is True
    assert sync_validation["cns_root_present"] is True
    assert sync_validation["cdi_status_present"] is True
    assert sync_validation["cns_status_present"] is True
    assert sync_validation["claim_boundaries_present"] is True


def test_manifest_and_summary_use_only_frozen_status_vocabulary():
    module = _load_audit_module()
    manifest = module.build_manifest(_default_inputs(), repo_root=REPO_ROOT)
    summary = module.render_audit_summary(manifest)
    allowed_statuses = set(manifest["status_vocabulary"])

    manifest_statuses = {
        manifest["pillar_summaries"]["cdi"]["headline_status"],
        manifest["pillar_summaries"]["cns"]["headline_status"],
        *(context["row_status"] for context in manifest["pillar_summaries"]["cdi"]["adjacent_context"]),
        *(context["status"] for context in manifest["pillar_summaries"]["cns"]["adjacent_context"]),
        *(entry["status"] for entry in manifest["blocked_claims"]),
    }

    assert manifest_statuses <= allowed_statuses
    assert "excluded_adjacent_context" not in summary


def test_brdt_authority_is_loaded_and_promoted_as_additive_secondary():
    module = _load_audit_module()
    inputs = _default_inputs()

    brdt_inputs = inputs["brdt"]
    assert brdt_inputs["authoritative_root"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-05-07-brdt-sinogram-input-40ep-paper-evidence"
    )
    assert brdt_inputs["expected_claim_boundary"] == "paper_evidence_brdt_additive"

    brdt = module.load_brdt_authority(brdt_inputs, repo_root=REPO_ROOT)
    assert brdt["headline_status"] == "paper_approved_secondary"
    assert brdt["claim_boundary"] == "paper_evidence_brdt_additive"
    assert brdt["bundle_status"] == "passed"
    assert brdt["headline_row_ids"] == ["ffno", "sru_net"]
    assert any(row["row_id"] == "classical_born_backprop" for row in brdt["reference_rows"])


def test_build_manifest_includes_brdt_additive_secondary_authority():
    module = _load_audit_module()
    manifest = module.build_manifest(_default_inputs(), repo_root=REPO_ROOT)

    additive = manifest["additive_secondary_authorities"]["brdt"]
    assert additive["claim_boundary"] == "paper_evidence_brdt_additive"
    assert additive["source_root"] == (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-05-07-brdt-sinogram-input-40ep-paper-evidence"
    )

    boundaries = {entry["claim_boundary"] for entry in manifest["claim_boundary_registry"]}
    assert "paper_evidence_brdt_additive" in boundaries

    brdt_row_ids = {row["row_id"] for row in manifest["row_registry"] if row["pillar_id"] == "brdt"}
    assert {"ffno", "sru_net", "classical_born_backprop"} <= brdt_row_ids

    blocked_ids = {entry["claim_id"] for entry in manifest["blocked_claims"]}
    assert "brdt_replaces_cdi_or_cns_pillar" in blocked_ids
    assert "same_protocol_full_training_brdt_competitiveness" in blocked_ids

    validation = module.validate_manifest(manifest)
    assert validation["brdt_additive_authority_present"] is True
    assert validation["brdt_headline_is_paper_approved_secondary"] is True
    assert validation["brdt_does_not_replace_pillars"] is True
    assert validation["all_statuses_within_frozen_vocabulary"] is True


def test_audit_summary_records_brdt_additive_authority_and_lineage():
    module = _load_audit_module()
    manifest = module.build_manifest(_default_inputs(), repo_root=REPO_ROOT)
    summary = module.render_audit_summary(manifest)

    brdt = manifest["additive_secondary_authorities"]["brdt"]
    assert "Additive Secondary BRDT Context" in summary
    assert brdt["source_root"] in summary
    assert brdt["claim_boundary"] in summary
    assert brdt["headline_status"] in summary
    for root in brdt["historical_lineage_roots"]:
        assert root in summary

    sync = module.validate_summary_sync(manifest, summary)
    assert sync["brdt_root_present"] is True
    assert sync["brdt_status_present"] is True
    assert sync["brdt_claim_boundary_present"] is True


def test_summary_records_emitted_paths_and_verification_logs():
    module = _load_audit_module()
    manifest = module.build_manifest(_default_inputs(), repo_root=REPO_ROOT)
    summary = module.render_audit_summary(manifest)

    assert manifest["output_targets"]["manifest_path"] in summary
    assert manifest["output_targets"]["summary_path"] in summary
    assert manifest["output_targets"]["audit_validation_path"] in summary
    for log_path in manifest["output_targets"]["verification_logs"].values():
        assert log_path in summary

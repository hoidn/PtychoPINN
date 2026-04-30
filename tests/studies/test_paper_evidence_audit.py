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
    tampered_model_manifest = tmp_path / "model_manifest.json"
    tampered_model_manifest.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    cdi_inputs["model_manifest_path"] = str(tampered_model_manifest)

    with pytest.raises(ValueError, match="CDI.*benchmark status"):
        module.load_cdi_authority(cdi_inputs, repo_root=REPO_ROOT)


def test_load_cns_authority_preserves_capped_claim_boundary():
    module = _load_audit_module()
    inputs = _default_inputs()

    cns = module.load_cns_authority(inputs["cns"], repo_root=REPO_ROOT)

    assert cns["headline_status"] == "capped_decision_support"
    assert cns["claim_boundary"] == "bounded_capped_decision_support_only"
    assert cns["bundle_status"] == "paper_complete"
    assert cns["headline_row_ids"] == [
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
        "author_ffno_cns_base",
    ]
    assert all(row["row_status"] == "capped_decision_support" for row in cns["headline_rows"])


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
    assert "paper_grade" in summary
    assert "capped_decision_support" in summary
    assert "full-training CNS competitiveness claims remain blocked" in summary

    sync_validation = module.validate_summary_sync(manifest, summary)
    assert sync_validation["cdi_root_present"] is True
    assert sync_validation["cns_root_present"] is True
    assert sync_validation["cdi_status_present"] is True
    assert sync_validation["cns_status_present"] is True
    assert sync_validation["claim_boundaries_present"] is True

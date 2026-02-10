from pathlib import Path
import re


def test_ptychovit_doc_records_interop_contract_source():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "Interop Contract Source" in text
    assert "Checkpoint Contract Source" in text
    assert "source_repo" in text
    assert "source_commit" in text
    assert "TBD" not in text
    assert re.search(r"source_commit:\s*`?[0-9a-f]{7,40}`?", text)


def test_docs_index_links_ptychovit_workflow():
    text = Path("docs/index.md").read_text()
    assert "workflows/ptychovit.md" in text


def test_studies_readme_points_to_workflow_doc():
    text = Path("scripts/studies/README.md").read_text()
    assert "docs/workflows/ptychovit.md" in text


def test_ptychovit_workflow_documents_fresh_initial_metrics_runbook():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "Fresh Initial Baseline (Checkpoint-Restored, Lines Synthetic)" in text
    assert "--models pinn_ptychovit" in text
    assert "--model-n pinn_ptychovit=256" in text
    assert "--reuse-existing-recons" in text


def test_ptychovit_workflow_calls_out_nonzero_scan_positions_and_norm_dict_requirement():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "probe_position_x_m/probe_position_y_m must be non-constant" in text
    assert "normalization dictionary" in text
    assert "Normalization file not found" in text


def test_ptychovit_workflow_requires_position_aware_stitching_contract():
    workflow_text = Path("docs/workflows/ptychovit.md").read_text()
    spec_text = Path("specs/ptychovit_interop_contract.md").read_text()
    assert "position-aware stitching" in workflow_text
    assert "scan-wise mean aggregation" in workflow_text
    assert "position-aware stitching" in spec_text
    assert "scan-wise mean aggregation" in spec_text

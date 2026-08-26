from pathlib import Path
import re


WORKFLOW = Path("docs/workflows/ptychovit.md")
SPEC = Path("specs/ptychovit_interop_contract.md")
CURRENT_DOCS = (
    WORKFLOW,
    Path("docs/DATA_MANAGEMENT_GUIDE.md"),
    Path("docs/studies/index.md"),
    Path("scripts/studies/README.md"),
)
RETIRED_ENTRYPOINTS = (
    "grid_lines_compare_wrapper.py",
    "grid_lines_torch_runner.py",
    "grid_lines_ptychovit_runner.py",
    "run_fresh_ptychovit_initial_metrics.py",
    "verify_fresh_ptychovit_initial_metrics.py",
    "ptychovit_input_optimization_diagnostic.py",
    "nersc_orchestration.py",
    "hybrid_checkpoint_inference.py",
    "run_nersc_scan807_cameraman_study.py",
    "run_nersc_scan807_cameraman_study_n256.py",
    "run_nersc_scan807_cameraman_study_n128_factorial.py",
    "collate_nersc_n128_factorial_results.py",
)


def test_ptychovit_doc_records_versioned_interop_sources():
    text = WORKFLOW.read_text()
    assert "Interop Contract Source" in text
    assert "Checkpoint Contract Source" in text
    assert "source_repo" in text
    assert "source_commit" in text
    assert "TBD" not in text
    assert re.search(r"source_commit:\s*`?[0-9a-f]{7,40}`?", text)


def test_docs_route_to_the_retained_ptychovit_workflow():
    assert "workflows/ptychovit.md" in Path("docs/index.md").read_text()
    assert "docs/workflows/ptychovit.md" in Path(
        "scripts/studies/README.md"
    ).read_text()


def test_workflow_names_only_the_retained_runtime_interfaces():
    text = WORKFLOW.read_text()
    assert "ptycho.interop.ptychovit" in text
    assert "scripts/studies/ptychovit_bridge_entrypoint.py" in text
    assert "scripts/studies/nersc_pair_adapter.py" in text


def test_workflow_documents_runtime_normalization_requirements():
    text = WORKFLOW.read_text()
    assert "normalization dictionary" in text
    assert "data.normalization_dict_path" in text
    assert "data.test_normalization" in text
    assert "Normalization file not found" in text


def test_workflow_and_spec_record_current_position_aware_assembly():
    workflow_text = WORKFLOW.read_text()
    spec_text = SPEC.read_text()
    for text in (workflow_text, spec_text):
        assert "position-aware" in text
        assert "scan-wise mean aggregation" in text.lower()
        assert "_stitch_complex_predictions" in text
    assert "Known gap" not in spec_text


def test_workflow_documents_reverse_pair_probe_policy_and_caveat():
    text = WORKFLOW.read_text()
    assert "pair_to_external_npz" in text
    assert "incoherent_aggregate" in text
    assert "first_mode" in text
    assert "approximation" in text


def test_studies_index_keeps_retired_nersc_design_provenance():
    text = Path("docs/studies/index.md").read_text()
    assert "Retired NERSC/PtychoViT comparison studies" in text
    assert "2026-02-17-nersc-ptychovit-hybrid-orchestration-plan.md" in text
    assert "2026-02-18-nersc-downsample-policy-flip-implementation.md" in text


def test_current_docs_do_not_advertise_retired_study_entrypoints():
    documented = "\n".join(path.read_text() for path in CURRENT_DOCS)
    for entrypoint in RETIRED_ENTRYPOINTS:
        assert entrypoint not in documented

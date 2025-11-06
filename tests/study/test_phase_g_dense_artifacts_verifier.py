#!/usr/bin/env python3
"""
Tests for Phase G dense pipeline artifact inventory validation.

Covers the TDD cycle for validate_artifact_inventory() in verify_dense_pipeline_artifacts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_verify_dense_pipeline_artifact_inventory_blocks_missing_entries(tmp_path: Path) -> None:
    """
    Test that verify_dense_pipeline_artifacts.py fails when artifact_inventory.txt is missing
    or incomplete.

    Acceptance:
    - Create a hub with analysis/ directory but NO artifact_inventory.txt
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for artifact inventory
    - Assert error message mentions "artifact_inventory.txt not found" or similar

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Setup: Create incomplete hub (missing artifact_inventory.txt)
    hub = tmp_path / "incomplete_hub"
    analysis = hub / "analysis"
    analysis.mkdir(parents=True)

    # Create some artifacts but NOT artifact_inventory.txt
    (analysis / "metrics_summary.json").write_text(json.dumps({
        "n_jobs": 6,
        "n_success": 6,
        "n_failed": 0,
        "aggregate_metrics": {},
        "phase_c_metadata_compliance": {}
    }))
    (analysis / "comparison_manifest.json").write_text(json.dumps({
        "n_jobs": 6,
        "n_success": 6,
        "n_failed": 0,
        "jobs": []
    }))

    # Create CLI directory so other checks don't all fail
    cli_dir = hub / "cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "phase_g_train.log").write_text("fake log")

    report_path = tmp_path / "report_missing.json"

    # Execute: Import and run the verifier
    import sys
    import importlib.util

    verifier_script = Path(__file__).parent.parent.parent / "plans" / "active" / \
                      "STUDY-SYNTH-FLY64-DOSE-OVERLAP-001" / "bin" / "verify_dense_pipeline_artifacts.py"

    spec = importlib.util.spec_from_file_location("verifier", verifier_script)
    verifier = importlib.util.module_from_spec(spec)

    # Monkeypatch sys.argv
    original_argv = sys.argv
    sys.argv = [
        str(verifier_script),
        "--hub", str(hub),
        "--report", str(report_path),
    ]

    try:
        spec.loader.exec_module(verifier)
        exit_code = verifier.main()
    except SystemExit as e:
        exit_code = e.code
    finally:
        sys.argv = original_argv

    # Assert: Verification failed
    assert exit_code != 0, "Expected non-zero exit code when artifact_inventory.txt is missing"

    # Load report and check for inventory validation failure
    assert report_path.exists(), "Report JSON should be created even on failure"

    with report_path.open('r') as f:
        report_data = json.load(f)

    assert not report_data['all_valid'], "Expected all_valid=False when inventory missing"
    assert report_data['n_invalid'] > 0, "Expected at least one invalid check"

    # Find the artifact_inventory validation result
    inventory_check = None
    for check in report_data['validations']:
        if 'artifact inventory' in check.get('description', '').lower():
            inventory_check = check
            break

    assert inventory_check is not None, \
        "Expected validation check for artifact inventory, available checks: " + \
        ", ".join([v.get('description', '') for v in report_data['validations']])
    assert not inventory_check['valid'], "Expected artifact inventory check to be invalid"
    assert 'error' in inventory_check, "Expected error field in failed inventory check"


def test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle(tmp_path: Path) -> None:
    """
    Test that verify_dense_pipeline_artifacts.py succeeds when artifact_inventory.txt
    is present and properly formatted.

    Acceptance:
    - Create a complete hub with all required artifacts including artifact_inventory.txt
    - artifact_inventory.txt must contain POSIX-relative paths (no absolute paths, no backslashes)
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with status 0
    - Assert the verification report JSON shows all_valid=True
    - Assert the artifact_inventory validation check passes

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Setup: Create complete hub
    hub = tmp_path / "complete_hub"
    analysis = hub / "analysis"
    analysis.mkdir(parents=True)
    cli_dir = hub / "cli"
    cli_dir.mkdir(parents=True)

    # Create all required artifacts
    metrics_summary = {
        "n_jobs": 6,
        "n_success": 6,
        "n_failed": 0,
        "aggregate_metrics": {
            "PtychoPINN": {"ms_ssim": {"amplitude": 0.95, "phase": 0.92}, "mae": {"amplitude": 0.05, "phase": 0.08}},
            "Baseline": {"ms_ssim": {"amplitude": 0.90, "phase": 0.87}, "mae": {"amplitude": 0.10, "phase": 0.13}},
            "PtyChi": {"ms_ssim": {"amplitude": 0.88, "phase": 0.85}, "mae": {"amplitude": 0.12, "phase": 0.15}},
        },
        "phase_c_metadata_compliance": {
            "dose_1000": {"train": {"compliant": True}, "test": {"compliant": True}},
        }
    }
    (analysis / "metrics_summary.json").write_text(json.dumps(metrics_summary))

    (analysis / "comparison_manifest.json").write_text(json.dumps({
        "n_jobs": 6,
        "n_success": 6,
        "n_failed": 0,
        "jobs": []
    }))

    (analysis / "metrics_summary.md").write_text("# Metrics Summary\n\nFake summary\n")
    (analysis / "aggregate_highlights.txt").write_text("Fake highlights\n")

    digest_content = """# Phase G Dense Metrics Digest

## Pipeline Summary

Completed successfully.

## Highlights

MS-SSIM improvements observed.
"""
    (analysis / "metrics_digest.md").write_text(digest_content)

    # Create metrics delta artifacts
    delta_summary = {
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {
            "vs_Baseline": {
                "ms_ssim": {"amplitude": 0.05, "phase": 0.05},
                "mae": {"amplitude": -0.05, "phase": -0.05}
            },
            "vs_PtyChi": {
                "ms_ssim": {"amplitude": 0.07, "phase": 0.07},
                "mae": {"amplitude": -0.07, "phase": -0.07}
            }
        }
    }
    (analysis / "metrics_delta_summary.json").write_text(json.dumps(delta_summary))

    highlights_txt = """MS-SSIM Δ (PtychoPINN - Baseline) Amplitude: +0.05, Phase: +0.05
MS-SSIM Δ (PtychoPINN - PtyChi) Amplitude: +0.07, Phase: +0.07
MAE Δ (PtychoPINN - Baseline) Amplitude: -0.05, Phase: -0.05
MAE Δ (PtychoPINN - PtyChi) Amplitude: -0.07, Phase: -0.07
"""
    (analysis / "metrics_delta_highlights.txt").write_text(highlights_txt)

    # Create artifact_inventory.txt with POSIX-relative paths
    inventory_content = """analysis/aggregate_highlights.txt
analysis/comparison_manifest.json
analysis/metrics_delta_highlights.txt
analysis/metrics_delta_summary.json
analysis/metrics_digest.md
analysis/metrics_summary.json
analysis/metrics_summary.md
cli/phase_g_train.log
"""
    (analysis / "artifact_inventory.txt").write_text(inventory_content)

    # Create CLI log
    (cli_dir / "phase_g_train.log").write_text("fake log")

    report_path = tmp_path / "report_complete.json"

    # Execute: Import and run the verifier
    import sys
    import importlib.util

    verifier_script = Path(__file__).parent.parent.parent / "plans" / "active" / \
                      "STUDY-SYNTH-FLY64-DOSE-OVERLAP-001" / "bin" / "verify_dense_pipeline_artifacts.py"

    spec = importlib.util.spec_from_file_location("verifier", verifier_script)
    verifier = importlib.util.module_from_spec(spec)

    # Monkeypatch sys.argv
    original_argv = sys.argv
    sys.argv = [
        str(verifier_script),
        "--hub", str(hub),
        "--report", str(report_path),
    ]

    try:
        spec.loader.exec_module(verifier)
        exit_code = verifier.main()
    except SystemExit as e:
        exit_code = e.code
    finally:
        sys.argv = original_argv

    # Assert: Verification succeeded
    assert exit_code == 0, f"Expected exit code 0 with complete bundle, got {exit_code}"

    # Load report and check all validations passed
    assert report_path.exists(), "Report JSON should be created"

    with report_path.open('r') as f:
        report_data = json.load(f)

    assert report_data['all_valid'], \
        f"Expected all_valid=True with complete bundle. Failures: {[v for v in report_data['validations'] if not v['valid']]}"
    assert report_data['n_invalid'] == 0, "Expected no invalid checks"

    # Find the artifact_inventory validation result
    inventory_check = None
    for check in report_data['validations']:
        if 'artifact inventory' in check.get('description', '').lower():
            inventory_check = check
            break

    assert inventory_check is not None, \
        "Expected validation check for artifact inventory"
    assert inventory_check['valid'], \
        f"Expected artifact inventory check to be valid, got error: {inventory_check.get('error', 'none')}"

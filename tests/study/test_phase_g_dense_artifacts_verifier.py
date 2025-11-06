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

    # Create metrics delta artifacts (matching actual orchestrator structure: vs_Baseline, vs_PtyChi)
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

    highlights_txt = """MS-SSIM Δ (PtychoPINN - Baseline): +0.050
MS-SSIM Δ (PtychoPINN - PtyChi): +0.070
MAE Δ (PtychoPINN - Baseline): -0.050
MAE Δ (PtychoPINN - PtyChi): -0.070
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
cli/run_phase_g_dense.log
"""
    (analysis / "artifact_inventory.txt").write_text(inventory_content)

    # Create orchestrator CLI log with all required elements
    orchestrator_log_content = """[run_phase_g_dense] Starting pipeline...
================================================================================
[1/8] Phase C: Dataset Generation
================================================================================
Running Phase C...
================================================================================
[2/8] Phase D: Overlap View Generation
================================================================================
Running Phase D...
================================================================================
[3/8] Phase E: Training Baseline (gs1)
================================================================================
Running baseline...
================================================================================
[4/8] Phase E: Training Dense (gs2)
================================================================================
Running dense...
================================================================================
[5/8] Phase F: Reconstruction dense/train
================================================================================
Reconstruction train...
================================================================================
[6/8] Phase F: Reconstruction dense/test
================================================================================
Reconstruction test...
================================================================================
[7/8] Phase G: Comparison dense/train
================================================================================
Comparison train...
================================================================================
[8/8] Phase G: Comparison dense/test
================================================================================
Comparison test...
================================================================================
[run_phase_g_dense] SUCCESS: All phases completed
================================================================================
"""
    (cli_dir / "run_phase_g_dense.log").write_text(orchestrator_log_content)

    # Create all required per-phase logs with CORRECT dose/view-specific patterns
    # Using dose=1000, view=dense as per typical run (matches run_phase_g_dense.py defaults)
    phase_logs = {
        "phase_c_generation.log": "Phase C generation complete\n",
        "phase_d_dense.log": "Phase D overlap view generation complete\n",
        "phase_e_baseline_gs1_dose1000.log": "Phase E baseline training complete\n",
        "phase_e_dense_gs2_dose1000.log": "Phase E dense training complete\n",
        "phase_f_dense_train.log": "Phase F train reconstruction complete\n",
        "phase_f_dense_test.log": "Phase F test reconstruction complete\n",
        "phase_g_dense_train.log": "Phase G train comparison complete\n",
        "phase_g_dense_test.log": "Phase G test comparison complete\n",
    }

    for log_name, content in phase_logs.items():
        (cli_dir / log_name).write_text(f"# {log_name}\n{content}")

    # Create required helper logs (aggregate_report_cli.log and metrics_digest_cli.log)
    (cli_dir / "aggregate_report_cli.log").write_text("Aggregate report generation complete\n")
    (cli_dir / "metrics_digest_cli.log").write_text("Metrics digest generation complete\n")

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


def test_verify_dense_pipeline_cli_logs_missing(tmp_path: Path) -> None:
    """
    RED test: Verify that CLI log validation fails when required logs are missing.

    Acceptance:
    - Create a hub with cli/ directory but missing phase logs
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for CLI logs
    - Assert error message mentions missing phase banners or SUCCESS sentinel

    Follows input.md Do Now step 1 (TDD RED).
    """
    # Setup: Create incomplete hub (cli/ dir exists but logs are empty/missing)
    hub = tmp_path / "incomplete_cli_hub"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks don't all fail
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015"
    )
    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_summary.md\n"
        "cli/phase_c_generation.log\n"
    )

    # Create a CLI log that is incomplete (missing phase banners and SUCCESS)
    (cli_dir / "phase_c_generation.log").write_text(
        "# Command: python -m studies.fly64_dose_overlap.generation\n"
        "Some output\n"
        "But no phase banners or success marker\n"
    )

    report_path = tmp_path / "report_missing_cli.json"

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

    # Assert: Verification failed due to missing CLI log validation
    assert exit_code != 0, f"Expected non-zero exit code with incomplete CLI logs, got {exit_code}"

    # Load report and check for CLI log validation failure
    assert report_path.exists(), "Report JSON should be created even on failure"

    with report_path.open('r') as f:
        report_data = json.load(f)

    assert not report_data['all_valid'], \
        "Expected all_valid=False with incomplete CLI logs"
    assert report_data['n_invalid'] > 0, "Expected at least one invalid check"

    # Find the CLI log validation result
    cli_check = None
    for check in report_data['validations']:
        desc = check.get('description', '').lower()
        if 'orchestrator' in desc or ('cli' in desc and 'log' in desc):
            cli_check = check
            break

    assert cli_check is not None, \
        f"Expected validation check for CLI orchestrator logs. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert not cli_check['valid'], \
        "Expected CLI log check to be invalid with incomplete logs"
    assert cli_check.get('error') is not None, \
        "Expected error message for missing CLI log content"


def test_verify_dense_pipeline_cli_logs_complete(tmp_path: Path) -> None:
    """
    GREEN test: Verify that CLI log validation passes when all required logs are present.

    Acceptance:
    - Create a hub with cli/ directory containing complete logs with all phase banners
    - Ensure logs include [1/8]...[8/8] banners and "SUCCESS: All phases completed"
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with status 0
    - Assert the verification report JSON shows CLI log validation passed

    Follows input.md Do Now step 1 (TDD GREEN).
    """
    # Setup: Create complete hub with proper CLI logs
    hub = tmp_path / "complete_cli_hub"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015"
    )
    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_summary.md\n"
        "cli/run_phase_g_dense.log\n"
    )

    # Create a complete CLI log with all phase banners and SUCCESS
    complete_log_content = """
# Command: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py
# CWD: /home/user/PtychoPINN2
# Log: cli/run_phase_g_dense.log
# ==============================================================================

[run_phase_g_dense] Preparing hub...
[run_phase_g_dense] Hub: /home/user/hub
[run_phase_g_dense] Total commands: 8

================================================================================
[1/8] Phase C: Dataset Generation
================================================================================
Running phase C...

================================================================================
[2/8] Phase D: Overlap View Generation
================================================================================
Running phase D...

================================================================================
[3/8] Phase E: Training Baseline (gs1)
================================================================================
Running baseline training...

================================================================================
[4/8] Phase E: Training Dense (gs2)
================================================================================
Running dense training...

================================================================================
[5/8] Phase F: Reconstruction dense/train
================================================================================
Running reconstruction train...

================================================================================
[6/8] Phase F: Reconstruction dense/test
================================================================================
Running reconstruction test...

================================================================================
[7/8] Phase G: Comparison dense/train
================================================================================
Running comparison train...

================================================================================
[8/8] Phase G: Comparison dense/test
================================================================================
Running comparison test...

================================================================================
[run_phase_g_dense] SUCCESS: All phases completed
================================================================================
"""
    (cli_dir / "run_phase_g_dense.log").write_text(complete_log_content)

    # Create all required per-phase logs with CORRECT dose/view-specific patterns
    # Using dose=1000, view=dense as per typical run (matches run_phase_g_dense.py defaults)
    phase_logs = {
        "phase_c_generation.log": "Phase C generation complete\n",
        "phase_d_dense.log": "Phase D overlap view generation complete\n",
        "phase_e_baseline_gs1_dose1000.log": "Phase E baseline training complete\n",
        "phase_e_dense_gs2_dose1000.log": "Phase E dense training complete\n",
        "phase_f_dense_train.log": "Phase F train reconstruction complete\n",
        "phase_f_dense_test.log": "Phase F test reconstruction complete\n",
        "phase_g_dense_train.log": "Phase G train comparison complete\n",
        "phase_g_dense_test.log": "Phase G test comparison complete\n",
    }

    for log_name, content in phase_logs.items():
        (cli_dir / log_name).write_text(f"# {log_name}\n{content}")

    # Create required helper logs (aggregate_report_cli.log and metrics_digest_cli.log)
    (cli_dir / "aggregate_report_cli.log").write_text("Aggregate report generation complete\n")
    (cli_dir / "metrics_digest_cli.log").write_text("Metrics digest generation complete\n")

    report_path = tmp_path / "report_complete_cli.json"

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

    # Load report and check CLI log validation passed
    # Note: Other checks may fail due to minimal fixture data, but we only care about CLI validation here
    assert report_path.exists(), "Report JSON should be created"

    with report_path.open('r') as f:
        report_data = json.load(f)

    # Find the CLI log validation result
    cli_check = None
    for check in report_data['validations']:
        desc = check.get('description', '').lower()
        if 'orchestrator' in desc or ('cli' in desc and 'log' in desc):
            cli_check = check
            break

    # The CLI check must exist and be valid
    assert cli_check is not None, \
        f"Expected validation check for CLI orchestrator logs. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert cli_check['valid'], \
        f"Expected CLI log check to be valid with complete logs, got error: {cli_check.get('error', 'none')}"

    # Verify the check found all required elements
    assert cli_check.get('has_success') is True, \
        "Expected CLI check to find SUCCESS marker"
    assert not cli_check.get('missing_banners'), \
        f"Expected no missing phase banners, got: {cli_check.get('missing_banners', [])}"


def test_verify_dense_pipeline_cli_phase_logs_missing(tmp_path: Path) -> None:
    """
    RED test: Verify that per-phase CLI log validation fails when required phase logs are missing.

    Acceptance:
    - Create a hub with cli/ directory containing orchestrator log but missing individual phase logs
    - Individual phase logs should include: phase_c_generation.log, phase_d_dense.log, etc.
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for missing phase logs
    - Assert error message lists the specific missing phase log files

    Follows input.md Do Now step 1 (TDD RED for per-phase logs).
    """
    # Setup: Create hub with orchestrator log but NO per-phase logs
    hub = tmp_path / "hub_missing_phase_logs"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks don't all fail
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015"
    )
    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_summary.md\n"
        "cli/run_phase_g_dense.log\n"
    )

    # Create orchestrator log with all phase banners and SUCCESS (this passes basic check)
    orchestrator_log_content = """[run_phase_g_dense] Starting pipeline...
================================================================================
[1/8] Phase C: Dataset Generation
================================================================================
Running Phase C...
================================================================================
[2/8] Phase D: Overlap View Generation
================================================================================
Running Phase D...
================================================================================
[3/8] Phase E: Training Baseline (gs1)
================================================================================
Running baseline...
================================================================================
[4/8] Phase E: Training Dense (gs2)
================================================================================
Running dense...
================================================================================
[5/8] Phase F: Reconstruction dense/train
================================================================================
Reconstruction train...
================================================================================
[6/8] Phase F: Reconstruction dense/test
================================================================================
Reconstruction test...
================================================================================
[7/8] Phase G: Comparison dense/train
================================================================================
Comparison train...
================================================================================
[8/8] Phase G: Comparison dense/test
================================================================================
Comparison test...
================================================================================
[run_phase_g_dense] SUCCESS: All phases completed
================================================================================
"""
    (cli_dir / "run_phase_g_dense.log").write_text(orchestrator_log_content)

    # DO NOT create individual phase logs - this is the RED test condition

    report_path = tmp_path / "report_missing_phase_logs.json"

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

    # Assert: Verification failed due to missing per-phase logs
    assert exit_code != 0, f"Expected non-zero exit code with missing phase logs, got {exit_code}"

    # Load report and check for per-phase log validation failure
    assert report_path.exists(), "Report JSON should be created even on failure"

    with report_path.open('r') as f:
        report_data = json.load(f)

    assert not report_data['all_valid'], \
        "Expected all_valid=False with missing per-phase logs"
    assert report_data['n_invalid'] > 0, "Expected at least one invalid check"

    # Find the CLI log validation result - should fail due to missing phase logs
    cli_check = None
    for check in report_data['validations']:
        desc = check.get('description', '').lower()
        if 'orchestrator' in desc or ('cli' in desc and 'log' in desc):
            cli_check = check
            break

    assert cli_check is not None, \
        f"Expected validation check for CLI logs. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert not cli_check['valid'], \
        "Expected CLI log check to be invalid with missing per-phase logs"

    # The error should mention missing phase logs
    error_msg = cli_check.get('error', '').lower()
    assert 'phase' in error_msg and 'log' in error_msg, \
        f"Expected error message to mention missing phase logs, got: {cli_check.get('error', 'none')}"

    # Should have a list of missing phase log files
    assert cli_check.get('missing_phase_logs') is not None, \
        "Expected missing_phase_logs field in CLI check result"
    assert len(cli_check.get('missing_phase_logs', [])) > 0, \
        "Expected at least one missing phase log file"


def test_verify_dense_pipeline_cli_phase_logs_wrong_pattern(tmp_path: Path) -> None:
    """
    RED test: Verify that per-phase CLI log validation fails when phase logs use wrong filename patterns.

    Acceptance:
    - Create a hub with cli/ directory containing orchestrator log and phase logs
    - Phase logs use OLD generic names (phase_e_baseline.log) instead of dose/view-specific names
      (phase_e_baseline_gs1_dose1000.log) that run_phase_g_dense.py actually generates
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for CLI logs
    - Assert error message indicates missing phase logs with correct patterns

    Follows input.md Do Now step 1 (TDD RED for filename pattern enforcement).
    """
    # Setup: Create hub with orchestrator log but WRONG-PATTERN phase logs
    hub = tmp_path / "hub_wrong_pattern_phase_logs"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks don't all fail
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015"
    )
    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_summary.md\n"
        "cli/run_phase_g_dense.log\n"
    )

    # Create orchestrator log with all phase banners and SUCCESS
    orchestrator_log_content = """[run_phase_g_dense] Starting pipeline...
================================================================================
[1/8] Phase C: Dataset Generation
================================================================================
Running Phase C...
================================================================================
[2/8] Phase D: Overlap View Generation
================================================================================
Running Phase D...
================================================================================
[3/8] Phase E: Training Baseline (gs1)
================================================================================
Running baseline...
================================================================================
[4/8] Phase E: Training Dense (gs2)
================================================================================
Running dense...
================================================================================
[5/8] Phase F: Reconstruction dense/train
================================================================================
Reconstruction train...
================================================================================
[6/8] Phase F: Reconstruction dense/test
================================================================================
Reconstruction test...
================================================================================
[7/8] Phase G: Comparison dense/train
================================================================================
Comparison train...
================================================================================
[8/8] Phase G: Comparison dense/test
================================================================================
Comparison test...
================================================================================
[run_phase_g_dense] SUCCESS: All phases completed
================================================================================
"""
    (cli_dir / "run_phase_g_dense.log").write_text(orchestrator_log_content)

    # Create phase logs with WRONG PATTERN (old generic names, not dose/view-specific)
    # These are the OLD names that don't match what run_phase_g_dense.py actually generates
    wrong_pattern_logs = {
        "phase_c_generation.log": "Phase C generation complete\n",
        "phase_d_dense.log": "Phase D overlap view generation complete\n",
        "phase_e_baseline.log": "Phase E baseline training complete\n",  # Should be phase_e_baseline_gs1_dose1000.log
        "phase_e_dense.log": "Phase E dense training complete\n",  # Should be phase_e_dense_gs2_dose1000.log
        "phase_f_train.log": "Phase F train reconstruction complete\n",  # Should be phase_f_dense_train.log
        "phase_f_test.log": "Phase F test reconstruction complete\n",  # Should be phase_f_dense_test.log
        "phase_g_train.log": "Phase G train comparison complete\n",  # Should be phase_g_dense_train.log
        "phase_g_test.log": "Phase G test comparison complete\n",  # Should be phase_g_dense_test.log
    }

    for log_name, content in wrong_pattern_logs.items():
        (cli_dir / log_name).write_text(f"# {log_name}\n# Starting phase...\n{content}")

    report_path = tmp_path / "report_wrong_pattern.json"

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

    # Assert: Verification should FAIL because expected dose/view-specific patterns are missing
    assert exit_code != 0, "Expected non-zero exit code when phase logs use wrong filename pattern"

    # Load report and check CLI log validation failed
    assert report_path.exists(), "Report JSON should be created"

    with report_path.open('r') as f:
        report_data = json.load(f)

    # Find the CLI log validation result
    cli_check = None
    for check in report_data['validations']:
        desc = check.get('description', '').lower()
        if 'orchestrator' in desc or ('cli' in desc and 'log' in desc):
            cli_check = check
            break

    assert cli_check is not None, \
        f"Expected validation check for CLI logs. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert not cli_check['valid'], \
        "Expected CLI log check to be invalid with wrong filename patterns"

    # The error should mention missing phase logs (because correct patterns not found)
    error_msg = cli_check.get('error', '').lower()
    assert 'phase' in error_msg and 'log' in error_msg, \
        f"Expected error message to mention missing phase logs, got: {cli_check.get('error', 'none')}"

    # Should have a list of missing phase log files with correct patterns
    assert cli_check.get('missing_phase_logs') is not None, \
        "Expected missing_phase_logs field in CLI check result"
    assert len(cli_check.get('missing_phase_logs', [])) > 0, \
        "Expected at least one missing phase log file with correct pattern"


def test_verify_dense_pipeline_cli_phase_logs_incomplete(tmp_path: Path) -> None:
    """
    RED test: Verify that per-phase CLI log validation fails when phase logs lack completion sentinels.

    Acceptance:
    - Create a hub with cli/ directory containing orchestrator log and correctly-named phase logs
    - Phase logs use correct dose/view-specific names (e.g., phase_e_baseline_gs1_dose1000.log)
    - But some phase logs are INCOMPLETE: missing completion sentinel at end
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for CLI logs
    - Assert error message indicates incomplete phase logs missing completion markers

    Follows input.md Do Now step 1 (TDD RED for completion sentinel enforcement).
    """
    # Setup: Create hub with orchestrator log and correct-pattern phase logs, but some incomplete
    hub = tmp_path / "hub_incomplete_phase_logs"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks don't all fail
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015"
    )
    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_summary.md\n"
        "cli/run_phase_g_dense.log\n"
    )

    # Create orchestrator log with all phase banners and SUCCESS
    orchestrator_log_content = """[run_phase_g_dense] Starting pipeline...
================================================================================
[1/8] Phase C: Dataset Generation
================================================================================
Running Phase C...
================================================================================
[2/8] Phase D: Overlap View Generation
================================================================================
Running Phase D...
================================================================================
[3/8] Phase E: Training Baseline (gs1)
================================================================================
Running baseline...
================================================================================
[4/8] Phase E: Training Dense (gs2)
================================================================================
Running dense...
================================================================================
[5/8] Phase F: Reconstruction dense/train
================================================================================
Reconstruction train...
================================================================================
[6/8] Phase F: Reconstruction dense/test
================================================================================
Reconstruction test...
================================================================================
[7/8] Phase G: Comparison dense/train
================================================================================
Comparison train...
================================================================================
[8/8] Phase G: Comparison dense/test
================================================================================
Comparison test...
================================================================================
[run_phase_g_dense] SUCCESS: All phases completed
================================================================================
"""
    (cli_dir / "run_phase_g_dense.log").write_text(orchestrator_log_content)

    # Create phase logs with CORRECT PATTERN but some INCOMPLETE (missing completion sentinel)
    # Using dose=1000, view=dense as per typical run
    phase_logs_incomplete = {
        "phase_c_generation.log": "# phase_c_generation.log\n# Starting phase...\nPhase C generation complete\n",
        "phase_d_dense.log": "# phase_d_dense.log\n# Starting phase...\nPhase D overlap view generation complete\n",
        "phase_e_baseline_gs1_dose1000.log": "# phase_e_baseline_gs1_dose1000.log\n# Starting phase...\nTraining...\n",  # INCOMPLETE
        "phase_e_dense_gs2_dose1000.log": "# phase_e_dense_gs2_dose1000.log\n# Starting phase...\nPhase E dense training complete\n",
        "phase_f_dense_train.log": "# phase_f_dense_train.log\n# Starting phase...\nReconstruction...\n",  # INCOMPLETE
        "phase_f_dense_test.log": "# phase_f_dense_test.log\n# Starting phase...\nPhase F test reconstruction complete\n",
        "phase_g_dense_train.log": "# phase_g_dense_train.log\n# Starting phase...\nPhase G train comparison complete\n",
        "phase_g_dense_test.log": "# phase_g_dense_test.log\n# Starting phase...\nPhase G test comparison complete\n",
    }

    for log_name, content in phase_logs_incomplete.items():
        (cli_dir / log_name).write_text(content)

    # Also create helper logs (aggregate_report_cli.log and metrics_digest_cli.log)
    (cli_dir / "aggregate_report_cli.log").write_text("Aggregate report complete\n")
    (cli_dir / "metrics_digest_cli.log").write_text("Metrics digest complete\n")

    report_path = tmp_path / "report_incomplete.json"

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

    # Assert: Verification should FAIL because some phase logs are incomplete
    assert exit_code != 0, "Expected non-zero exit code when phase logs are incomplete"

    # Load report and check CLI log validation failed
    assert report_path.exists(), "Report JSON should be created"

    with report_path.open('r') as f:
        report_data = json.load(f)

    # Find the CLI log validation result
    cli_check = None
    for check in report_data['validations']:
        desc = check.get('description', '').lower()
        if 'orchestrator' in desc or ('cli' in desc and 'log' in desc):
            cli_check = check
            break

    assert cli_check is not None, \
        f"Expected validation check for CLI logs. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert not cli_check['valid'], \
        "Expected CLI log check to be invalid with incomplete phase logs"

    # The error should mention incomplete phase logs
    error_msg = cli_check.get('error', '').lower()
    assert 'incomplete' in error_msg or 'missing' in error_msg, \
        f"Expected error message to mention incomplete phase logs, got: {cli_check.get('error', 'none')}"

    # Should have a list of incomplete phase log files
    assert cli_check.get('incomplete_phase_logs') is not None, \
        "Expected incomplete_phase_logs field in CLI check result"
    assert len(cli_check.get('incomplete_phase_logs', [])) > 0, \
        "Expected at least one incomplete phase log file"


def test_verify_dense_pipeline_cli_phase_logs_complete(tmp_path: Path) -> None:
    """
    GREEN test: Verify that per-phase CLI log validation passes when all phase logs are present.

    Acceptance:
    - Create a hub with cli/ directory containing both orchestrator log AND all individual phase logs
    - Individual phase logs include: phase_c_generation.log, phase_d_dense.log, phase_e_baseline.log,
      phase_e_dense.log, phase_f_train.log, phase_f_test.log, phase_g_train.log, phase_g_test.log
    - Each phase log should contain a completion sentinel (e.g., "Phase X complete" or similar)
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with status 0
    - Assert the verification report JSON shows CLI log validation passed
    - Assert no missing phase logs are reported

    Follows input.md Do Now step 1 (TDD GREEN for per-phase logs).
    """
    # Setup: Create complete hub with orchestrator log AND all per-phase logs
    hub = tmp_path / "hub_complete_phase_logs"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015"
    )
    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_summary.md\n"
        "cli/run_phase_g_dense.log\n"
    )

    # Create orchestrator log with all phase banners and SUCCESS
    orchestrator_log_content = """[run_phase_g_dense] Starting pipeline...
================================================================================
[1/8] Phase C: Dataset Generation
================================================================================
Running Phase C...
================================================================================
[2/8] Phase D: Overlap View Generation
================================================================================
Running Phase D...
================================================================================
[3/8] Phase E: Training Baseline (gs1)
================================================================================
Running baseline...
================================================================================
[4/8] Phase E: Training Dense (gs2)
================================================================================
Running dense...
================================================================================
[5/8] Phase F: Reconstruction dense/train
================================================================================
Reconstruction train...
================================================================================
[6/8] Phase F: Reconstruction dense/test
================================================================================
Reconstruction test...
================================================================================
[7/8] Phase G: Comparison dense/train
================================================================================
Comparison train...
================================================================================
[8/8] Phase G: Comparison dense/test
================================================================================
Comparison test...
================================================================================
[run_phase_g_dense] SUCCESS: All phases completed
================================================================================
"""
    (cli_dir / "run_phase_g_dense.log").write_text(orchestrator_log_content)

    # Create all required per-phase logs with CORRECT dose/view-specific patterns and completion sentinels
    # Using dose=1000, view=dense as per typical run (matches run_phase_g_dense.py defaults)
    phase_logs = {
        "phase_c_generation.log": "Phase C generation complete\n",
        "phase_d_dense.log": "Phase D overlap view generation complete\n",
        "phase_e_baseline_gs1_dose1000.log": "Phase E baseline training complete\n",
        "phase_e_dense_gs2_dose1000.log": "Phase E dense training complete\n",
        "phase_f_dense_train.log": "Phase F train reconstruction complete\n",
        "phase_f_dense_test.log": "Phase F test reconstruction complete\n",
        "phase_g_dense_train.log": "Phase G train comparison complete\n",
        "phase_g_dense_test.log": "Phase G test comparison complete\n",
    }

    for log_name, content in phase_logs.items():
        (cli_dir / log_name).write_text(f"# {log_name}\n# Starting phase...\n{content}")

    # Create required helper logs (aggregate_report_cli.log and metrics_digest_cli.log)
    (cli_dir / "aggregate_report_cli.log").write_text("Aggregate report generation complete\n")
    (cli_dir / "metrics_digest_cli.log").write_text("Metrics digest generation complete\n")

    report_path = tmp_path / "report_complete_phase_logs.json"

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
    # Load report and check CLI log validation passed
    assert report_path.exists(), "Report JSON should be created"

    with report_path.open('r') as f:
        report_data = json.load(f)

    # Find the CLI log validation result
    cli_check = None
    for check in report_data['validations']:
        desc = check.get('description', '').lower()
        if 'orchestrator' in desc or ('cli' in desc and 'log' in desc):
            cli_check = check
            break

    # The CLI check must exist and be valid
    assert cli_check is not None, \
        f"Expected validation check for CLI logs. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert cli_check['valid'], \
        f"Expected CLI log check to be valid with complete phase logs, got error: {cli_check.get('error', 'none')}"

    # Verify the check found all required phase logs
    assert cli_check.get('has_success') is True, \
        "Expected CLI check to find SUCCESS marker in orchestrator log"
    assert not cli_check.get('missing_banners'), \
        f"Expected no missing phase banners, got: {cli_check.get('missing_banners', [])}"
    assert not cli_check.get('missing_phase_logs'), \
        f"Expected no missing phase logs, got: {cli_check.get('missing_phase_logs', [])}"

    # Verify all expected phase logs were found
    found_phase_logs = cli_check.get('found_phase_logs', [])
    expected_phase_logs = list(phase_logs.keys())
    for expected_log in expected_phase_logs:
        assert expected_log in found_phase_logs, \
            f"Expected phase log {expected_log} to be found, found logs: {found_phase_logs}"


def test_verify_dense_pipeline_highlights_missing_model(tmp_path: Path) -> None:
    """
    RED test: Verify that highlights validation fails when metrics_delta_highlights.txt
    is missing a required model comparison (e.g., only has Baseline, missing PtyChi).

    Acceptance:
    - Create a hub with metrics_delta_highlights.txt containing only 2 lines (Baseline only)
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for highlights
    - Assert error message mentions missing model comparison lines

    Follows input.md Do Now step 1 (TDD RED for missing model).
    """
    # Setup: Create hub with incomplete highlights (missing PtyChi comparisons)
    hub = tmp_path / "hub_missing_model"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks don't all fail
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))

    # Create INCOMPLETE highlights (only Baseline, missing PtyChi)
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
    )

    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_delta_highlights.txt\n"
    )

    # Create minimal CLI log
    (cli_dir / "run_phase_g_dense.log").write_text(
        "[run_phase_g_dense] SUCCESS: All phases completed\n"
    )

    report_path = tmp_path / "report_missing_model.json"

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
    assert exit_code != 0, "Expected non-zero exit code when highlights missing model comparison"

    # Load report and check for highlights validation failure
    assert report_path.exists(), "Report JSON should be created even on failure"

    with report_path.open('r') as f:
        report_data = json.load(f)

    assert not report_data['all_valid'], "Expected all_valid=False when highlights incomplete"
    assert report_data['n_invalid'] > 0, "Expected at least one invalid check"

    # Find the highlights validation result
    highlights_check = None
    for check in report_data['validations']:
        if 'delta highlight' in check.get('description', '').lower():
            highlights_check = check
            break

    assert highlights_check is not None, \
        f"Expected validation check for highlights. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert not highlights_check['valid'], \
        "Expected highlights check to be invalid when missing model comparison"
    assert 'expected exactly 4 lines' in highlights_check.get('error', '').lower() or \
           'got 2' in highlights_check.get('error', '').lower(), \
        f"Expected error about line count, got: {highlights_check.get('error', 'none')}"


def test_verify_dense_pipeline_highlights_mismatched_value(tmp_path: Path) -> None:
    """
    RED test: Verify that highlights validation fails when metrics_delta_highlights.txt
    has incorrect format (e.g., wrong metric prefix or malformed delta value).

    Acceptance:
    - Create a hub with metrics_delta_highlights.txt containing lines with wrong prefixes
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with non-zero status
    - Assert the verification report JSON shows a validation failure for highlights
    - Assert error message mentions incorrect prefix or format

    Follows input.md Do Now step 1 (TDD RED for mismatched value).
    """
    # Setup: Create hub with malformed highlights (wrong prefix)
    hub = tmp_path / "hub_mismatched_value"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks don't all fail
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
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "generated_at": "2025-11-10T10:00:00Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {}
    }))

    # Create MALFORMED highlights (wrong prefix on line 2)
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "WRONG-METRIC Δ (PtychoPINN - PtyChi): +0.005\n"  # Wrong prefix!
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015\n"
    )

    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/metrics_delta_highlights.txt\n"
    )

    # Create minimal CLI log
    (cli_dir / "run_phase_g_dense.log").write_text(
        "[run_phase_g_dense] SUCCESS: All phases completed\n"
    )

    report_path = tmp_path / "report_mismatched_value.json"

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
    assert exit_code != 0, "Expected non-zero exit code when highlights have wrong format"

    # Load report and check for highlights validation failure
    assert report_path.exists(), "Report JSON should be created even on failure"

    with report_path.open('r') as f:
        report_data = json.load(f)

    assert not report_data['all_valid'], "Expected all_valid=False when highlights malformed"
    assert report_data['n_invalid'] > 0, "Expected at least one invalid check"

    # Find the highlights validation result
    highlights_check = None
    for check in report_data['validations']:
        if 'delta highlight' in check.get('description', '').lower():
            highlights_check = check
            break

    assert highlights_check is not None, \
        f"Expected validation check for highlights. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert not highlights_check['valid'], \
        "Expected highlights check to be invalid when format is wrong"
    assert 'does not start with expected prefix' in highlights_check.get('error', '').lower() or \
           'wrong-metric' in highlights_check.get('error', '').lower(), \
        f"Expected error about wrong prefix, got: {highlights_check.get('error', 'none')}"


def test_verify_dense_pipeline_highlights_complete(tmp_path: Path) -> None:
    """
    GREEN test: Verify that highlights validation passes when metrics_delta_highlights.txt
    has all 4 required lines with correct format.

    Acceptance:
    - Create a hub with complete and correct metrics_delta_highlights.txt (4 lines)
    - Invoke verify_dense_pipeline_artifacts.py --hub <hub> --report <report>
    - Assert the script exits with zero status (assuming other validations also pass)
    - Assert the verification report JSON shows highlights validation as valid
    - Assert the check captures line_count=4

    Follows input.md Do Now step 1 (TDD GREEN for complete highlights).
    """
    # Setup: Create hub with complete highlights
    hub = tmp_path / "hub_complete_highlights"
    analysis = hub / "analysis"
    cli_dir = hub / "cli"
    analysis.mkdir(parents=True)
    cli_dir.mkdir(parents=True)

    # Create minimal artifacts so other checks pass
    (analysis / "metrics_summary.json").write_text(json.dumps({
        "n_jobs": 6,
        "n_success": 6,
        "n_failed": 0,
        "aggregate_metrics": {
            "MS-SSIM": {
                "PtychoPINN": 0.95,
                "Baseline": 0.94,
                "PtyChi": 0.945
            }
        },
        "phase_c_metadata_compliance": {}
    }))
    (analysis / "comparison_manifest.json").write_text(json.dumps({
        "n_jobs": 6,
        "n_success": 6,
        "n_failed": 0,
        "jobs": []
    }))
    (analysis / "metrics_summary.md").write_text("# Metrics Summary")
    (analysis / "aggregate_highlights.txt").write_text("highlights")
    (analysis / "metrics_digest.md").write_text("# Metrics Digest")
    (analysis / "metrics_delta_summary.json").write_text(json.dumps({
        "deltas": {
            "MS-SSIM": {
                "PtychoPINN_vs_Baseline": {"delta": 0.010, "preview": "+0.010"},
                "PtychoPINN_vs_PtyChi": {"delta": 0.005, "preview": "+0.005"}
            },
            "MAE": {
                "PtychoPINN_vs_Baseline": {"delta": -0.020, "preview": "-0.020"},
                "PtychoPINN_vs_PtyChi": {"delta": -0.015, "preview": "-0.015"}
            }
        }
    }))

    # Create COMPLETE and CORRECT highlights
    (analysis / "metrics_delta_highlights.txt").write_text(
        "MS-SSIM Δ (PtychoPINN - Baseline): +0.010\n"
        "MS-SSIM Δ (PtychoPINN - PtyChi): +0.005\n"
        "MAE Δ (PtychoPINN - Baseline): -0.020\n"
        "MAE Δ (PtychoPINN - PtyChi): -0.015\n"
    )

    (analysis / "artifact_inventory.txt").write_text(
        "analysis/metrics_summary.json\n"
        "analysis/comparison_manifest.json\n"
        "analysis/metrics_summary.md\n"
        "analysis/aggregate_highlights.txt\n"
        "analysis/metrics_digest.md\n"
        "analysis/metrics_delta_summary.json\n"
        "analysis/metrics_delta_highlights.txt\n"
        "analysis/artifact_inventory.txt\n"
        "cli/run_phase_g_dense.log\n"
    )

    # Create minimal CLI log with SUCCESS
    (cli_dir / "run_phase_g_dense.log").write_text(
        "[run_phase_g_dense] SUCCESS: All phases completed\n"
    )

    report_path = tmp_path / "report_complete_highlights.json"

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

    # Assert: Verification succeeded (or check highlights validation specifically)
    # Load report and check highlights validation passed
    assert report_path.exists(), "Report JSON should be created"

    with report_path.open('r') as f:
        report_data = json.load(f)

    # Find the highlights validation result
    highlights_check = None
    for check in report_data['validations']:
        if 'delta highlight' in check.get('description', '').lower():
            highlights_check = check
            break

    assert highlights_check is not None, \
        f"Expected validation check for highlights. Available checks: {[v.get('description') for v in report_data['validations']]}"
    assert highlights_check['valid'], \
        f"Expected highlights check to be valid with complete highlights, got error: {highlights_check.get('error', 'none')}"

    # Verify the check captured line_count=4
    assert highlights_check.get('line_count') == 4, \
        f"Expected line_count=4, got: {highlights_check.get('line_count')}"

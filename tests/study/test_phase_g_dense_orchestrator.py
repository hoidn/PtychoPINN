"""Tests for Phase G dense orchestrator summary helper."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


def _import_orchestrator_module():
    """Import the orchestrator module using spec loader."""
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py"
    spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_generate_overlap_views(monkeypatch: pytest.MonkeyPatch) -> None:
    import studies.fly64_dose_overlap.overlap as overlap_module
    monkeypatch.setattr(overlap_module, "generate_overlap_views", lambda **kwargs: None)


def _import_summarize_phase_g_outputs():
    """Import summarize_phase_g_outputs() from the orchestrator script using spec loader."""
    return _import_orchestrator_module().summarize_phase_g_outputs


def test_run_phase_g_dense_collect_only_generates_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """
    Test that main() with --collect-only prints planned commands without executing them.

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Runs with --collect-only into a tmp hub directory
    - Asserts stdout contains expected command substrings (Phase C/D/E/F/G markers)
    - Verifies no Phase C outputs are created (dry-run mode, no filesystem side effects)
    - Ensures AUTHORITATIVE_CMDS_DOC environment variable is respected
    - Returns 0 exit code on success

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import main() from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory
    hub = tmp_path / "collect_only_hub"
    hub.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--collect-only",
        ],
    )

    # Execute: Call main() (should print commands and return 0)
    exit_code = main()

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from --collect-only mode, got {exit_code}"

    # Assert: Capture stdout and verify expected command substrings
    captured = capsys.readouterr()
    stdout = captured.out

    # Check for phase markers in stdout
    assert "Phase C: Dataset Generation" in stdout, "Missing Phase C command in --collect-only output"
    assert "Phase D: Overlap View Generation" in stdout, "Missing Phase D command in --collect-only output"
    assert "Phase E: Training Baseline (gs1)" in stdout, "Missing Phase E baseline command in --collect-only output"
    assert "Phase E: Training Dense (gs2)" in stdout, "Missing Phase E dense command in --collect-only output"
    assert "Phase F: Reconstruction" in stdout, "Missing Phase F command in --collect-only output"
    assert "Phase G: Comparison" in stdout, "Missing Phase G command in --collect-only output"

    # Check for specific command keywords
    assert "studies.fly64_dose_overlap.generation" in stdout, "Missing generation module in command output"
    assert "__PHASE_D_PROGRAMMATIC__" in stdout, "Missing programmatic Phase D marker in command output"
    assert "studies.fly64_dose_overlap.training" in stdout, "Missing training module in command output"
    assert "studies.fly64_dose_overlap.reconstruction" in stdout, "Missing reconstruction module in command output"
    assert "studies.fly64_dose_overlap.comparison" in stdout, "Missing comparison module in command output"

    # Check for reporting helper command
    assert "report_phase_g_dense_metrics.py" in stdout, "Missing reporting helper command in --collect-only output"
    assert "aggregate_report.md" in stdout, "Missing aggregate_report.md output path in reporting helper command"
    assert "aggregate_highlights.txt" in stdout, "Missing aggregate_highlights.txt output path in reporting helper command"

    # Check for analyze digest command
    assert "analyze_dense_metrics.py" in stdout, "Missing analyze_dense_metrics.py command in --collect-only output"
    assert "metrics_digest.md" in stdout, "Missing metrics_digest.md output path in analyze command"

    # Check for ssim_grid command
    assert "ssim_grid.py" in stdout, "Missing ssim_grid.py command in --collect-only output"
    assert "ssim_grid_cli.log" in stdout, "Missing ssim_grid_cli.log output path in ssim_grid command"

    # Check for post-verify commands (default: enabled)
    assert "verify_dense_pipeline_artifacts.py" in stdout, "Missing verify_dense_pipeline_artifacts.py command in --collect-only output"
    assert "verification_report.json" in stdout, "Missing verification_report.json output path in verify command"
    assert "verify_dense_stdout.log" in stdout, "Missing verify_dense_stdout.log output path in verify command"
    assert "check_dense_highlights_match.py" in stdout, "Missing check_dense_highlights_match.py command in --collect-only output"
    assert "check_dense_highlights.log" in stdout, "Missing check_dense_highlights.log output path in check command"

    # Assert: No Phase C outputs created (dry-run mode)
    phase_c_root = hub / "data" / "phase_c"
    if phase_c_root.exists():
        phase_c_files = list(phase_c_root.rglob("*.npz"))
        assert len(phase_c_files) == 0, f"--collect-only mode should not create Phase C outputs, found: {phase_c_files}"


def test_run_phase_g_dense_post_verify_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that main() invokes post-verify commands (verify_dense_pipeline_artifacts.py + check_dense_highlights_match.py) with correct paths.

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Stubs prepare_hub, validate_phase_c_metadata, summarize_phase_g_outputs (no-op for speed)
    - Monkeypatches run_command to record invocations
    - Runs main() without --skip-post-verify (default: post-verify enabled)
    - Asserts post-verify commands are invoked AFTER ssim_grid
    - Validates verify command includes --hub, --report, --dose, --view flags
    - Validates check command includes --hub flag
    - Validates log paths point to analysis/verify_dense_stdout.log and analysis/check_dense_highlights.log
    - Ensures AUTHORITATIVE_CMDS_DOC environment variable is respected
    - Returns 0 exit code on success

    Follows TYPE-PATH-001 (Path normalization), DATA-001, TEST-CLI-001.
    """
    # Import main() and helper functions from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory
    hub = tmp_path / "post_verify_hub"
    hub.mkdir(parents=True)

    # Create expected directory structure for Phase C→G
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)
    cli_log_dir = hub / "cli"
    cli_log_dir.mkdir(parents=True)
    phase_g_root = hub / "analysis"
    phase_g_root.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse (NO --skip-post-verify, so post-verify enabled)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--clobber",  # Required to pass prepare_hub check
        ],
    )

    # Stub heavy helpers to no-op (we only care about run_command invocations)
    def stub_prepare_hub(hub_path, clobber):
        """No-op stub for prepare_hub."""
        pass

    def stub_validate_phase_c_metadata(hub_path):
        """No-op stub for validate_phase_c_metadata."""
        pass

    def stub_summarize_phase_g_outputs(hub_path):
        """Create metrics_summary.json with test data for delta computation."""
        analysis = Path(hub_path) / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)

        # Create metrics_summary.json with aggregate_metrics for delta computation
        summary_data = {
            "n_jobs": 2,
            "n_success": 2,
            "n_failed": 0,
            "jobs": [],
            "aggregate_metrics": {
                "PtychoPINN": {
                    "ms_ssim": {"mean_amplitude": 0.950, "mean_phase": 0.920},
                    "mae": {"mean_amplitude": 0.025, "mean_phase": 0.035}
                },
                "Baseline": {
                    "ms_ssim": {"mean_amplitude": 0.930, "mean_phase": 0.900},
                    "mae": {"mean_amplitude": 0.030, "mean_phase": 0.040}
                },
                "PtyChi": {
                    "ms_ssim": {"mean_amplitude": 0.940, "mean_phase": 0.910},
                    "mae": {"mean_amplitude": 0.027, "mean_phase": 0.037}
                }
            }
        }

        import json
        metrics_summary_path = analysis / "metrics_summary.json"
        with metrics_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    def stub_generate_artifact_inventory(hub_path):
        """Create artifact_inventory.txt to satisfy orchestrator validation."""
        analysis = Path(hub_path) / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        inventory_path = analysis / "artifact_inventory.txt"
        inventory_path.write_text("# Stub artifact inventory\n", encoding="utf-8")

    monkeypatch.setattr(module, "prepare_hub", stub_prepare_hub)
    monkeypatch.setattr(module, "validate_phase_c_metadata", stub_validate_phase_c_metadata)
    monkeypatch.setattr(module, "summarize_phase_g_outputs", stub_summarize_phase_g_outputs)
    monkeypatch.setattr(module, "generate_artifact_inventory", stub_generate_artifact_inventory)
    _stub_generate_overlap_views(monkeypatch)

    # Record run_command invocations
    run_command_calls = []

    def stub_run_command(cmd, log_path):
        """Record cmd and log_path, create required files for orchestrator progression."""
        run_command_calls.append((cmd, log_path))
        cmd_str = " ".join(str(c) for c in cmd)

        # When reporting helper is invoked, create highlights file
        if "report_phase_g_dense_metrics.py" in cmd_str and "--highlights" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--highlights" and i + 1 < len(cmd):
                    highlights_path = Path(cmd[i + 1])
                    highlights_path.parent.mkdir(parents=True, exist_ok=True)
                    highlights_path.write_text("MS-SSIM Deltas\n", encoding="utf-8")
                    break

        # When analyze digest is invoked, create digest file
        if "analyze_dense_metrics.py" in cmd_str and "--output" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--output" and i + 1 < len(cmd):
                    digest_path = Path(cmd[i + 1])
                    digest_path.parent.mkdir(parents=True, exist_ok=True)
                    digest_path.write_text("# Digest\n", encoding="utf-8")
                    break

        # When ssim_grid is invoked, create summary file
        if "ssim_grid.py" in cmd_str and "--hub" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--hub" and i + 1 < len(cmd):
                    hub_path = Path(cmd[i + 1])
                    ssim_grid_summary_path = hub_path / "analysis" / "ssim_grid_summary.md"
                    ssim_grid_summary_path.parent.mkdir(parents=True, exist_ok=True)
                    ssim_grid_summary_path.write_text("# SSIM Grid\n", encoding="utf-8")
                    break

        # When verify_dense_pipeline_artifacts is invoked, create report file
        if "verify_dense_pipeline_artifacts.py" in cmd_str and "--report" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--report" and i + 1 < len(cmd):
                    report_path = Path(cmd[i + 1])
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text('{"valid": true}\n', encoding="utf-8")
                    break

        # When check_dense_highlights_match is invoked, no file creation needed (just stdout)
        # (check_dense_highlights_match.py outputs to stdout, which is captured by run_command log_path)

    monkeypatch.setattr(module, "run_command", stub_run_command)

    # Execute: Call main() (should execute Phase C→G pipeline + reporting helper + analyze digest + ssim_grid + post-verify)
    exit_code = main()

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from real execution mode, got {exit_code}"

    # Assert: run_command should have been called for all phases + reporting helper + analyze digest + ssim_grid + post-verify (2 commands)
    # Expected: 1 Phase C + 1 Phase D + 2 Phase E + 2 Phase F + 2 Phase G + 1 reporting + 1 analyze + 1 ssim_grid + 2 post-verify = 13 total
    assert len(run_command_calls) >= 12, f"Expected at least 12 run_command calls (C/E/F/G phases + reporting + analyze + ssim_grid + post-verify), got {len(run_command_calls)}"

    # Find ssim_grid, verify, and check command indices
    ssim_grid_idx = None
    verify_idx = None
    check_idx = None

    for idx, (cmd, log_path) in enumerate(run_command_calls):
        cmd_str = " ".join(str(c) for c in cmd)
        if "ssim_grid.py" in cmd_str:
            ssim_grid_idx = idx
        if "verify_dense_pipeline_artifacts.py" in cmd_str:
            verify_idx = idx
        if "check_dense_highlights_match.py" in cmd_str:
            check_idx = idx

    # Assert: All three commands should be invoked
    assert ssim_grid_idx is not None, "ssim_grid.py was not invoked"
    assert verify_idx is not None, "verify_dense_pipeline_artifacts.py was not invoked"
    assert check_idx is not None, "check_dense_highlights_match.py was not invoked"

    # Assert: post-verify commands should be invoked AFTER ssim_grid
    assert verify_idx > ssim_grid_idx, \
        f"verify_dense_pipeline_artifacts.py should be invoked after ssim_grid.py, but got order: ssim_grid={ssim_grid_idx}, verify={verify_idx}"
    assert check_idx > ssim_grid_idx, \
        f"check_dense_highlights_match.py should be invoked after ssim_grid.py, but got order: ssim_grid={ssim_grid_idx}, check={check_idx}"
    assert check_idx > verify_idx, \
        f"check_dense_highlights_match.py should be invoked after verify_dense_pipeline_artifacts.py, but got order: verify={verify_idx}, check={check_idx}"

    # Validate verify command
    verify_cmd, verify_log_path = run_command_calls[verify_idx]
    verify_cmd_str = " ".join(str(c) for c in verify_cmd)

    assert "verify_dense_pipeline_artifacts.py" in verify_cmd_str, \
        f"Expected verify_dense_pipeline_artifacts.py in command, got: {verify_cmd_str}"
    assert "--hub" in verify_cmd_str, f"Missing --hub flag in verify command: {verify_cmd_str}"
    assert "--report" in verify_cmd_str, f"Missing --report flag in verify command: {verify_cmd_str}"
    assert "verification_report.json" in verify_cmd_str, f"Missing verification_report.json in verify command: {verify_cmd_str}"
    assert "--dose" in verify_cmd_str, f"Missing --dose flag in verify command: {verify_cmd_str}"
    assert "1000" in verify_cmd_str, f"Missing dose value 1000 in verify command: {verify_cmd_str}"
    assert "--view" in verify_cmd_str, f"Missing --view flag in verify command: {verify_cmd_str}"
    assert "dense" in verify_cmd_str, f"Missing view value 'dense' in verify command: {verify_cmd_str}"

    # Validate verify log_path points to analysis/verify_dense_stdout.log
    assert "verify_dense_stdout.log" in str(verify_log_path), \
        f"Expected verify log path to be analysis/verify_dense_stdout.log, got: {verify_log_path}"

    # Validate check command
    check_cmd, check_log_path = run_command_calls[check_idx]
    check_cmd_str = " ".join(str(c) for c in check_cmd)

    assert "check_dense_highlights_match.py" in check_cmd_str, \
        f"Expected check_dense_highlights_match.py in command, got: {check_cmd_str}"
    assert "--hub" in check_cmd_str, f"Missing --hub flag in check command: {check_cmd_str}"

    # Validate check log_path points to analysis/check_dense_highlights.log
    assert "check_dense_highlights.log" in str(check_log_path), \
        f"Expected check log path to be analysis/check_dense_highlights.log, got: {check_log_path}"


def test_summarize_phase_g_outputs(tmp_path: Path) -> None:
    """
    Test that summarize_phase_g_outputs() validates manifest, extracts metrics, and emits summaries.

    Acceptance:
    - Parses comparison_manifest.json from hub/analysis/
    - Fails fast if n_failed > 0 or metrics CSV missing
    - Extracts MS-SSIM + MAE (amplitude/phase/value) from per-job comparison_metrics.csv
    - Writes deterministic JSON + Markdown summary to hub/analysis/

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import the function under test
    summarize_phase_g_outputs = _import_summarize_phase_g_outputs()

    # Setup: Create temp hub with manifest + per-job metrics
    hub = tmp_path / "phase_g_hub"
    analysis = hub / "analysis"
    analysis.mkdir(parents=True)

    # Write comparison_manifest.json (2 successful jobs)
    manifest_data = {
        'n_jobs': 2,
        'n_executed': 2,
        'n_success': 2,
        'n_failed': 0,
        'execution_results': [
            {
                'dose': 1000,
                'view': 'dense',
                'split': 'train',
                'returncode': 0,
                'log_path': 'cli/phase_g_dense_train.log',
            },
            {
                'dose': 1000,
                'view': 'dense',
                'split': 'test',
                'returncode': 0,
                'log_path': 'cli/phase_g_dense_test.log',
            },
        ],
    }
    manifest_path = analysis / 'comparison_manifest.json'
    with manifest_path.open('w') as f:
        json.dump(manifest_data, f, indent=2)

    # Write per-job comparison_metrics.csv files (tidy format with model/metric/amplitude/phase/value columns)
    # Job 1: dense/train
    job1_dir = analysis / 'dose_1000' / 'dense' / 'train'
    job1_dir.mkdir(parents=True)
    job1_csv = job1_dir / 'comparison_metrics.csv'
    with job1_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'metric', 'amplitude', 'phase', 'value'])
        writer.writeheader()
        # MS-SSIM (amplitude, phase)
        writer.writerow({'model': 'PtychoPINN', 'metric': 'ms_ssim', 'amplitude': '0.9500', 'phase': '0.9200', 'value': ''})
        writer.writerow({'model': 'Baseline', 'metric': 'ms_ssim', 'amplitude': '0.9300', 'phase': '0.9000', 'value': ''})
        # MAE (amplitude, phase)
        writer.writerow({'model': 'PtychoPINN', 'metric': 'mae', 'amplitude': '0.0250', 'phase': '0.0350', 'value': ''})
        writer.writerow({'model': 'Baseline', 'metric': 'mae', 'amplitude': '0.0300', 'phase': '0.0400', 'value': ''})
        # Computation time (scalar value)
        writer.writerow({'model': 'PtychoPINN', 'metric': 'computation_time_s', 'amplitude': '', 'phase': '', 'value': '12.5'})

    # Job 2: dense/test
    job2_dir = analysis / 'dose_1000' / 'dense' / 'test'
    job2_dir.mkdir(parents=True)
    job2_csv = job2_dir / 'comparison_metrics.csv'
    with job2_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'metric', 'amplitude', 'phase', 'value'])
        writer.writeheader()
        writer.writerow({'model': 'PtychoPINN', 'metric': 'ms_ssim', 'amplitude': '0.9450', 'phase': '0.9150', 'value': ''})
        writer.writerow({'model': 'Baseline', 'metric': 'ms_ssim', 'amplitude': '0.9250', 'phase': '0.8950', 'value': ''})
        writer.writerow({'model': 'PtychoPINN', 'metric': 'mae', 'amplitude': '0.0270', 'phase': '0.0370', 'value': ''})
        writer.writerow({'model': 'Baseline', 'metric': 'mae', 'amplitude': '0.0320', 'phase': '0.0420', 'value': ''})

    # Setup Phase C data with metadata (PHASEC-METADATA-001)
    phase_c_root = hub / 'data' / 'phase_c'
    dose_dir = phase_c_root / 'dose_1000'
    dose_dir.mkdir(parents=True)

    # Import MetadataManager to create compliant NPZ files
    from ptycho.metadata import MetadataManager

    for split in ['train', 'test']:
        npz_path = dose_dir / f'patched_{split}.npz'
        # Create minimal NPZ with metadata containing canonical transformation
        metadata = {
            'data_transformations': [
                {'tool': 'transpose_rename_convert', 'timestamp': '2025-11-06T084736Z'}
            ]
        }
        # Save with metadata (need minimal data arrays)
        import numpy as np
        data_dict = {
            'diffraction': np.zeros((2, 64, 64), dtype=np.float32),
            'objectGuess': np.zeros((128, 128), dtype=np.complex64),
            'probeGuess': np.zeros((64, 64), dtype=np.complex64),
            'x': np.zeros(2, dtype=np.float32),
            'y': np.zeros(2, dtype=np.float32),
        }
        MetadataManager.save_with_metadata(str(npz_path), data_dict, metadata)

    # Execute: Call the helper
    summarize_phase_g_outputs(hub)

    # Assert: JSON summary exists and contains expected structure
    json_summary_path = analysis / 'metrics_summary.json'
    assert json_summary_path.exists(), f"Missing JSON summary: {json_summary_path}"

    with json_summary_path.open() as f:
        summary_data = json.load(f)

    assert summary_data['n_jobs'] == 2
    assert summary_data['n_success'] == 2
    assert summary_data['n_failed'] == 0
    assert len(summary_data['jobs']) == 2

    # Check first job metrics extraction
    job1_summary = summary_data['jobs'][0]
    assert job1_summary['dose'] == 1000
    assert job1_summary['view'] == 'dense'
    assert job1_summary['split'] == 'train'
    assert 'metrics' in job1_summary

    # Validate MS-SSIM extraction (amplitude, phase)
    pinn_ms_ssim = next((m for m in job1_summary['metrics'] if m['model'] == 'PtychoPINN' and m['metric'] == 'ms_ssim'), None)
    assert pinn_ms_ssim is not None
    assert pinn_ms_ssim['amplitude'] == 0.95
    assert pinn_ms_ssim['phase'] == 0.92

    baseline_mae = next((m for m in job1_summary['metrics'] if m['model'] == 'Baseline' and m['metric'] == 'mae'), None)
    assert baseline_mae is not None
    assert baseline_mae['amplitude'] == 0.03
    assert baseline_mae['phase'] == 0.04

    # Validate aggregate metrics in JSON
    assert 'aggregate_metrics' in summary_data, "Missing aggregate_metrics field in JSON"
    agg = summary_data['aggregate_metrics']

    # Check PtychoPINN aggregates
    assert 'PtychoPINN' in agg
    pinn_agg = agg['PtychoPINN']

    # PtychoPINN MS-SSIM: mean_amp=(0.95+0.945)/2=0.9475, best_amp=max(0.95,0.945)=0.95
    #                      mean_phase=(0.92+0.915)/2=0.9175, best_phase=max(0.92,0.915)=0.92
    assert 'ms_ssim' in pinn_agg
    pinn_ms_ssim_agg = pinn_agg['ms_ssim']
    assert abs(pinn_ms_ssim_agg['mean_amplitude'] - 0.9475) < 1e-6
    assert abs(pinn_ms_ssim_agg['best_amplitude'] - 0.95) < 1e-6
    assert abs(pinn_ms_ssim_agg['mean_phase'] - 0.9175) < 1e-6
    assert abs(pinn_ms_ssim_agg['best_phase'] - 0.92) < 1e-6

    # PtychoPINN MAE: mean_amp=(0.025+0.027)/2=0.026, mean_phase=(0.035+0.037)/2=0.036
    assert 'mae' in pinn_agg
    pinn_mae_agg = pinn_agg['mae']
    assert abs(pinn_mae_agg['mean_amplitude'] - 0.026) < 1e-6
    assert abs(pinn_mae_agg['mean_phase'] - 0.036) < 1e-6
    # MAE should not have 'best' fields (mean only)
    assert 'best_amplitude' not in pinn_mae_agg
    assert 'best_phase' not in pinn_mae_agg

    # Check Baseline aggregates
    assert 'Baseline' in agg
    baseline_agg = agg['Baseline']

    # Baseline MS-SSIM: mean_amp=(0.93+0.925)/2=0.9275, best_amp=max(0.93,0.925)=0.93
    #                    mean_phase=(0.90+0.895)/2=0.8975, best_phase=max(0.90,0.895)=0.90
    assert 'ms_ssim' in baseline_agg
    baseline_ms_ssim_agg = baseline_agg['ms_ssim']
    assert abs(baseline_ms_ssim_agg['mean_amplitude'] - 0.9275) < 1e-6
    assert abs(baseline_ms_ssim_agg['best_amplitude'] - 0.93) < 1e-6
    assert abs(baseline_ms_ssim_agg['mean_phase'] - 0.8975) < 1e-6
    assert abs(baseline_ms_ssim_agg['best_phase'] - 0.90) < 1e-6

    # Baseline MAE: mean_amp=(0.03+0.032)/2=0.031, mean_phase=(0.04+0.042)/2=0.041
    assert 'mae' in baseline_agg
    baseline_mae_agg = baseline_agg['mae']
    assert abs(baseline_mae_agg['mean_amplitude'] - 0.031) < 1e-6
    assert abs(baseline_mae_agg['mean_phase'] - 0.041) < 1e-6

    # Assert: Phase C metadata compliance in JSON (PHASEC-METADATA-001)
    assert 'phase_c_metadata_compliance' in summary_data, "Missing phase_c_metadata_compliance field in JSON"
    phase_c_compliance = summary_data['phase_c_metadata_compliance']

    # Should have dose_1000 key
    assert 'dose_1000' in phase_c_compliance, "Missing dose_1000 in phase_c_metadata_compliance"
    dose_1000_compliance = phase_c_compliance['dose_1000']

    # Should have train and test splits
    assert 'train' in dose_1000_compliance, "Missing train split in dose_1000 compliance"
    assert 'test' in dose_1000_compliance, "Missing test split in dose_1000 compliance"

    # Both splits should be compliant (has_metadata=True, has_canonical_transform=True, compliant=True)
    for split in ['train', 'test']:
        split_data = dose_1000_compliance[split]
        assert split_data['has_metadata'] is True, f"Split {split} should have metadata"
        assert split_data['has_canonical_transform'] is True, f"Split {split} should have canonical transform"
        assert split_data['compliant'] is True, f"Split {split} should be compliant"
        assert 'npz_path' in split_data, f"Split {split} should have npz_path"
        assert f'patched_{split}.npz' in split_data['npz_path'], f"Split {split} npz_path should contain patched_{split}.npz"

    # Assert: Markdown summary exists and contains key metrics
    md_summary_path = analysis / 'metrics_summary.md'
    assert md_summary_path.exists(), f"Missing Markdown summary: {md_summary_path}"

    md_content = md_summary_path.read_text()
    assert '# Phase G Metrics Summary' in md_content
    assert 'dense/train' in md_content
    assert 'dense/test' in md_content
    assert 'MS-SSIM' in md_content or 'ms_ssim' in md_content
    assert 'MAE' in md_content or 'mae' in md_content
    assert 'PtychoPINN' in md_content
    assert 'Baseline' in md_content

    # Validate Aggregate Metrics section in Markdown
    assert '## Aggregate Metrics' in md_content, "Missing '## Aggregate Metrics' section in Markdown"
    assert 'Summary statistics across all jobs per model' in md_content

    # Check that aggregate tables are present with expected structure
    # Should have model headings followed by MS-SSIM and MAE tables
    assert '### Baseline' in md_content or '### PtychoPINN' in md_content
    assert '**MS-SSIM:**' in md_content
    assert '**MAE:**' in md_content
    assert '| Mean |' in md_content
    assert '| Best |' in md_content  # Only for MS-SSIM

    # Validate Phase C Metadata Compliance section in Markdown (PHASEC-METADATA-001)
    assert '## Phase C Metadata Compliance' in md_content, "Missing '## Phase C Metadata Compliance' section in Markdown"
    assert 'Validation of Phase C NPZ files' in md_content, "Missing validation description in Markdown"
    assert 'dose_1000' in md_content, "Missing dose_1000 in Markdown compliance table"
    assert 'train' in md_content and 'test' in md_content, "Missing train/test splits in Markdown"
    # Check for checkmarks indicating compliance
    assert '✓' in md_content, "Missing compliance checkmarks (✓) in Markdown"
    assert 'patched_train.npz' in md_content or 'patched_test.npz' in md_content, "Missing NPZ file paths in Markdown"

    # Validate specific aggregate values appear in Markdown (formatted to 3 decimals)
    # PtychoPINN mean MS-SSIM amplitude should be 0.948 (0.9475 rounded to 3 decimals)
    assert '0.948' in md_content or '0.947' in md_content  # Allow rounding variation


def test_summarize_phase_g_outputs_fails_on_missing_manifest(tmp_path: Path) -> None:
    """Test that summarize_phase_g_outputs() raises RuntimeError if comparison_manifest.json is missing."""
    summarize_phase_g_outputs = _import_summarize_phase_g_outputs()

    hub = tmp_path / "empty_hub"
    hub.mkdir()

    with pytest.raises(RuntimeError, match="comparison_manifest.json.*not found"):
        summarize_phase_g_outputs(hub)


def test_summarize_phase_g_outputs_fails_on_execution_failures(tmp_path: Path) -> None:
    """Test that summarize_phase_g_outputs() raises RuntimeError if n_failed > 0."""
    summarize_phase_g_outputs = _import_summarize_phase_g_outputs()

    hub = tmp_path / "failed_hub"
    analysis = hub / "analysis"
    analysis.mkdir(parents=True)

    # Write manifest with 1 failure
    manifest_data = {
        'n_jobs': 2,
        'n_executed': 2,
        'n_success': 1,
        'n_failed': 1,
        'execution_results': [
            {'dose': 1000, 'view': 'dense', 'split': 'train', 'returncode': 0, 'log_path': 'cli/train.log'},
            {'dose': 1000, 'view': 'dense', 'split': 'test', 'returncode': 1, 'log_path': 'cli/test.log'},
        ],
    }
    manifest_path = analysis / 'comparison_manifest.json'
    with manifest_path.open('w') as f:
        json.dump(manifest_data, f, indent=2)

    with pytest.raises(RuntimeError, match="n_failed.*> 0"):
        summarize_phase_g_outputs(hub)


def test_summarize_phase_g_outputs_fails_on_missing_csv(tmp_path: Path) -> None:
    """Test that summarize_phase_g_outputs() raises RuntimeError if comparison_metrics.csv is missing for a successful job."""
    summarize_phase_g_outputs = _import_summarize_phase_g_outputs()

    hub = tmp_path / "missing_csv_hub"
    analysis = hub / "analysis"
    analysis.mkdir(parents=True)

    manifest_data = {
        'n_jobs': 1,
        'n_executed': 1,
        'n_success': 1,
        'n_failed': 0,
        'execution_results': [
            {'dose': 1000, 'view': 'dense', 'split': 'train', 'returncode': 0, 'log_path': 'cli/train.log'},
        ],
    }
    manifest_path = analysis / 'comparison_manifest.json'
    with manifest_path.open('w') as f:
        json.dump(manifest_data, f, indent=2)

    # No CSV file created

    with pytest.raises(RuntimeError, match="comparison_metrics.csv.*not found"):
        summarize_phase_g_outputs(hub)


def _import_validate_phase_c_metadata():
    """Import validate_phase_c_metadata() from the orchestrator script using spec loader."""
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py"
    spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_phase_c_metadata


def test_validate_phase_c_metadata_requires_metadata(tmp_path: Path) -> None:
    """
    Test that validate_phase_c_metadata() raises RuntimeError when Phase C NPZ outputs lack _metadata.

    Acceptance:
    - Normalizes hub path via TYPE-PATH-001
    - Checks both train/test splits exist under phase_c_root (modern layout: dose_*/patched_{train,test}.npz)
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Raises RuntimeError mentioning '_metadata' if metadata is None
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract), PHASEC-METADATA-001.
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs without metadata (modern patched layout)
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create minimal NPZ files WITHOUT _metadata using modern dose_*/patched_{train,test}.npz layout
    import numpy as np

    dose_dir = phase_c_root / "dose_1000"
    dose_dir.mkdir(parents=True)

    for split in ["train", "test"]:
        # Modern layout: dose_*/patched_{train,test}.npz
        npz_path = dose_dir / f"patched_{split}.npz"
        np.savez_compressed(
            npz_path,
            diffraction=np.random.rand(10, 64, 64).astype(np.float32),
            objectGuess=np.random.rand(128, 128).astype(np.complex64),
            probeGuess=np.random.rand(64, 64).astype(np.complex64),
        )

    # Execute: Call the guard (should raise RuntimeError mentioning '_metadata')
    with pytest.raises(RuntimeError, match="_metadata"):
        validate_phase_c_metadata(hub)


def test_validate_phase_c_metadata_requires_canonical_transform(tmp_path: Path) -> None:
    """
    Test that validate_phase_c_metadata() raises RuntimeError when Phase C NPZ outputs have _metadata but missing transpose_rename_convert transformation.

    Acceptance:
    - Normalizes hub path via TYPE-PATH-001
    - Checks both train/test splits exist under phase_c_root (modern layout: dose_*/patched_{train,test}.npz)
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Checks metadata["data_transformations"] for "transpose_rename_convert" (case-sensitive list membership)
    - Raises RuntimeError mentioning both '_metadata' and 'transpose_rename_convert' if transformation is missing
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract), PHASEC-METADATA-001.
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs with _metadata but missing canonical transformation (modern patched layout)
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create NPZ files WITH _metadata but WITHOUT transpose_rename_convert using modern layout
    import numpy as np
    from ptycho.metadata import MetadataManager

    dose_dir = phase_c_root / "dose_1000"
    dose_dir.mkdir(parents=True)

    for split in ["train", "test"]:
        # Modern layout: dose_*/patched_{train,test}.npz
        npz_path = dose_dir / f"patched_{split}.npz"

        # Create minimal NPZ data
        data_dict = {
            'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
            'objectGuess': np.random.rand(128, 128).astype(np.complex64),
            'probeGuess': np.random.rand(64, 64).astype(np.complex64),
        }

        # Create metadata with a DIFFERENT transformation (not transpose_rename_convert)
        metadata = {
            "schema_version": "1.0",
            "data_transformations": [
                {
                    "tool": "some_other_tool",
                    "timestamp": "2025-11-07T19:00:00Z",
                    "operation": "other_operation",
                    "parameters": {}
                }
            ]
        }

        # Save with metadata but missing the required transformation
        MetadataManager.save_with_metadata(str(npz_path), data_dict, metadata)

    # Execute: Call the guard (should raise RuntimeError mentioning both '_metadata' and 'transpose_rename_convert')
    with pytest.raises(RuntimeError, match=r"transpose_rename_convert"):
        validate_phase_c_metadata(hub)


def test_validate_phase_c_metadata_accepts_valid_metadata(tmp_path: Path) -> None:
    """
    Test that validate_phase_c_metadata() succeeds when Phase C NPZ outputs have proper metadata with transpose_rename_convert.

    Acceptance:
    - Normalizes hub path via TYPE-PATH-001
    - Checks both train/test splits exist under phase_c_root (modern layout: dose_*/patched_{train,test}.npz)
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Verifies metadata["data_transformations"] contains "transpose_rename_convert" transformation
    - Succeeds without raising when transformation is present
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract), PHASEC-METADATA-001.
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs with complete valid metadata (modern patched layout)
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create NPZ files WITH proper metadata including transpose_rename_convert using modern layout
    import numpy as np
    from ptycho.metadata import MetadataManager

    dose_dir = phase_c_root / "dose_1000"
    dose_dir.mkdir(parents=True)

    for split in ["train", "test"]:
        # Modern layout: dose_*/patched_{train,test}.npz
        npz_path = dose_dir / f"patched_{split}.npz"

        # Create minimal NPZ data
        data_dict = {
            'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
            'objectGuess': np.random.rand(128, 128).astype(np.complex64),
            'probeGuess': np.random.rand(64, 64).astype(np.complex64),
        }

        # Create metadata with the required transformation
        metadata = {
            "schema_version": "1.0",
            "data_transformations": []
        }

        # Use add_transformation_record to properly embed the canonical transformation
        metadata = MetadataManager.add_transformation_record(
            metadata,
            tool_name="transpose_rename_convert",
            operation="canonicalize_npz_format",
            parameters={"target_format": "NHW"}
        )

        # Save with proper metadata
        MetadataManager.save_with_metadata(str(npz_path), data_dict, metadata)

    # Execute: Call the guard (should succeed without raising)
    validate_phase_c_metadata(hub)


def test_validate_phase_c_metadata_handles_patched_layout(tmp_path: Path) -> None:
    """
    Test that validate_phase_c_metadata() succeeds with modern Phase C layout (dose_*/patched_{train,test}.npz).

    Acceptance (PHASEC-METADATA-001):
    - Normalizes hub path via TYPE-PATH-001
    - Walks dose_* directories (not dose_*_{split} subdirs)
    - Checks dose_*/patched_train.npz and dose_*/patched_test.npz exist
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Verifies metadata["data_transformations"] contains "transpose_rename_convert" transformation
    - Succeeds without raising when transformation is present
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract), PHASEC-METADATA-001.
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs with modern patched layout
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create dose directories with patched_{train,test}.npz files
    import numpy as np
    from ptycho.metadata import MetadataManager

    for dose in [1000, 10000, 100000]:
        dose_dir = phase_c_root / f"dose_{dose}"
        dose_dir.mkdir(parents=True)

        for split in ["train", "test"]:
            npz_path = dose_dir / f"patched_{split}.npz"

            # Create minimal NPZ data
            data_dict = {
                'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
                'objectGuess': np.random.rand(128, 128).astype(np.complex64),
                'probeGuess': np.random.rand(64, 64).astype(np.complex64),
            }

            # Create metadata with the required transformation
            metadata = {
                "schema_version": "1.0",
                "data_transformations": []
            }

            # Use add_transformation_record to properly embed the canonical transformation
            metadata = MetadataManager.add_transformation_record(
                metadata,
                tool_name="transpose_rename_convert",
                operation="canonicalize_npz_format",
                parameters={"target_format": "NHW"}
            )

            # Save with proper metadata
            MetadataManager.save_with_metadata(str(npz_path), data_dict, metadata)

    # Execute: Call the guard (should succeed without raising)
    validate_phase_c_metadata(hub)


def _import_prepare_hub():
    """Import prepare_hub() from the orchestrator script using spec loader."""
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py"
    spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_hub


def test_prepare_hub_detects_stale_outputs(tmp_path: Path) -> None:
    """
    Test that prepare_hub() raises RuntimeError when hub contains stale Phase C outputs and clobber=False.

    Acceptance:
    - Normalizes hub path via TYPE-PATH-001
    - Detects existing Phase C outputs under hub/data/phase_c/
    - Raises RuntimeError with actionable guidance mentioning both the stale path and --clobber remedy
    - Does not delete/modify hub contents when clobber=False (default read-only behavior)

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import the function under test
    prepare_hub = _import_prepare_hub()

    # Setup: Create fake hub with existing Phase C outputs
    hub = tmp_path / "stale_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create a fake Phase C output file to simulate stale state
    stale_file = phase_c_root / "dose_1000_train" / "fly64_train_simulated.npz"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("fake stale data")

    # Execute: Call prepare_hub with clobber=False (should raise RuntimeError)
    with pytest.raises(RuntimeError, match=r"(?=.*stale)(?=.*--clobber)"):
        prepare_hub(hub, clobber=False)

    # Assert: Stale file should still exist (read-only, no deletion)
    assert stale_file.exists(), "prepare_hub should not delete files when clobber=False"


def test_prepare_hub_clobbers_previous_outputs(tmp_path: Path) -> None:
    """
    Test that prepare_hub() removes stale outputs and produces a clean hub when clobber=True.

    Acceptance:
    - Normalizes hub path via TYPE-PATH-001
    - Detects existing Phase C outputs under hub/data/phase_c/
    - When clobber=True, archives or deletes prior data (implementation choice)
    - Produces clean hub directory structure ready for new pipeline run
    - Does not raise errors when clobber=True

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import the function under test
    prepare_hub = _import_prepare_hub()

    # Setup: Create fake hub with existing Phase C outputs
    hub = tmp_path / "clobber_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create multiple fake Phase C output files
    for split in ["train", "test"]:
        stale_dir = phase_c_root / f"dose_1000_{split}"
        stale_dir.mkdir(parents=True)
        stale_file = stale_dir / f"fly64_{split}_simulated.npz"
        stale_file.write_text("fake stale data")

    # Execute: Call prepare_hub with clobber=True (should succeed and clean up)
    prepare_hub(hub, clobber=True)

    # Assert: Hub should be clean (Phase C outputs either moved or deleted)
    # Check that phase_c_root either doesn't exist or is empty
    if phase_c_root.exists():
        # If it exists, it should be empty or contain only archive metadata
        remaining_files = list(phase_c_root.rglob("*.npz"))
        assert len(remaining_files) == 0, f"prepare_hub should remove/archive all .npz files when clobber=True, found: {remaining_files}"


def test_run_phase_g_dense_exec_invokes_reporting_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that main() in real execution mode invokes the reporting helper after Phase C→G pipeline.

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Stubs prepare_hub, validate_phase_c_metadata, summarize_phase_g_outputs (no-op for speed)
    - Monkeypatches run_command to record invocations
    - Runs main() without --collect-only to trigger real execution path
    - Asserts final run_command call targets report_phase_g_dense_metrics.py script
    - Validates command includes --metrics metrics_summary.json and --output aggregate_report.md
    - Validates log_path points to cli/aggregate_report_cli.log
    - Ensures AUTHORITATIVE_CMDS_DOC environment variable is respected
    - Returns 0 exit code on success

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import main() and helper functions from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory
    hub = tmp_path / "exec_hub"
    hub.mkdir(parents=True)

    # Create expected directory structure for Phase C→G
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)
    cli_log_dir = hub / "cli"
    cli_log_dir.mkdir(parents=True)
    phase_g_root = hub / "analysis"
    phase_g_root.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse (NO --collect-only, so real execution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--clobber",  # Required to pass prepare_hub check
        ],
    )

    # Stub heavy helpers to no-op (we only care about run_command invocations)
    def stub_prepare_hub(hub_path, clobber):
        """No-op stub for prepare_hub."""
        pass

    def stub_validate_phase_c_metadata(hub_path):
        """No-op stub for validate_phase_c_metadata."""
        pass

    def stub_summarize_phase_g_outputs(hub_path):
        """No-op stub for summarize_phase_g_outputs."""
        pass

    monkeypatch.setattr(module, "prepare_hub", stub_prepare_hub)
    monkeypatch.setattr(module, "validate_phase_c_metadata", stub_validate_phase_c_metadata)
    monkeypatch.setattr(module, "summarize_phase_g_outputs", stub_summarize_phase_g_outputs)
    _stub_generate_overlap_views(monkeypatch)

    # Record run_command invocations
    run_command_calls = []

    def stub_run_command(cmd, log_path):
        """Record cmd and log_path for assertions, and create highlights file when reporting helper is invoked."""
        run_command_calls.append((cmd, log_path))
        # When reporting helper is invoked, create the highlights file to satisfy orchestrator expectations
        cmd_str = " ".join(str(c) for c in cmd)
        if "report_phase_g_dense_metrics.py" in cmd_str and "--highlights" in cmd_str:
            # Extract highlights path from command
            for i, part in enumerate(cmd):
                if str(part) == "--highlights" and i + 1 < len(cmd):
                    highlights_path = Path(cmd[i + 1])
                    highlights_path.parent.mkdir(parents=True, exist_ok=True)
                    # Write minimal highlights content
                    highlights_path.write_text("Minimal highlights for test\n", encoding="utf-8")
                    break

    monkeypatch.setattr(module, "run_command", stub_run_command)

    # Execute: Call main() (should execute Phase C→G pipeline + reporting helper)
    exit_code = main()

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from real execution mode, got {exit_code}"

    # Assert: run_command should have been called multiple times (Phase C/D/E/F/G + reporting helper + analyze digest + ssim_grid)
    # Expected: 1 Phase C + 1 Phase D + 2 Phase E (baseline gs1 + dense gs2) + 2 Phase F (train/test) + 2 Phase G (train/test) + 1 reporting helper + 1 analyze digest + 1 ssim_grid = 11 total
    assert len(run_command_calls) >= 11, f"Expected at least 11 run_command calls (C/D/E/F/G phases + reporting + analyze + ssim_grid), got {len(run_command_calls)}"

    # Find the reporting helper command (no longer predictable by position due to ssim_grid)
    reporting_helper_idx = None
    for idx, (cmd, log_path) in enumerate(run_command_calls):
        cmd_str = " ".join(str(c) for c in cmd)
        if "report_phase_g_dense_metrics.py" in cmd_str:
            reporting_helper_idx = idx
            break

    # Validate reporting helper was invoked
    assert reporting_helper_idx is not None, "reporting helper (report_phase_g_dense_metrics.py) was not invoked"

    reporting_helper_cmd, reporting_helper_log_path = run_command_calls[reporting_helper_idx]

    # Validate command targets report_phase_g_dense_metrics.py
    assert any("report_phase_g_dense_metrics.py" in str(part) for part in reporting_helper_cmd), \
        f"Expected reporting helper command to target report_phase_g_dense_metrics.py, got: {reporting_helper_cmd}"

    # Validate command includes required flags
    cmd_str = " ".join(str(c) for c in reporting_helper_cmd)
    assert "--metrics" in cmd_str, f"Missing --metrics flag in reporting helper command: {cmd_str}"
    assert "metrics_summary.json" in cmd_str, f"Missing metrics_summary.json in reporting helper command: {cmd_str}"
    assert "--output" in cmd_str, f"Missing --output flag in reporting helper command: {cmd_str}"
    assert "aggregate_report.md" in cmd_str, f"Missing aggregate_report.md in reporting helper command: {cmd_str}"
    assert "--highlights" in cmd_str, f"Missing --highlights flag in reporting helper command: {cmd_str}"
    assert "aggregate_highlights.txt" in cmd_str, f"Missing aggregate_highlights.txt in reporting helper command: {cmd_str}"

    # Validate log_path points to cli/aggregate_report_cli.log
    assert "aggregate_report_cli.log" in str(reporting_helper_log_path), \
        f"Expected reporting helper log path to be cli/aggregate_report_cli.log, got: {reporting_helper_log_path}"


def test_run_phase_g_dense_exec_prints_highlights_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """
    Test that main() in real execution mode prints an "Aggregate highlights" preview after reporting helper.

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Stubs prepare_hub, validate_phase_c_metadata, summarize_phase_g_outputs (no-op for speed)
    - Stubs run_command to write deterministic highlights text when reporting helper is invoked
    - Runs main() without --collect-only to trigger real execution path
    - Captures stdout via capsys
    - Asserts stdout contains "Aggregate highlights preview" banner
    - Asserts stdout contains sample highlights content (MS-SSIM/MAE deltas)
    - Returns 0 exit code on success

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import main() and helper functions from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory
    hub = tmp_path / "exec_hub"
    hub.mkdir(parents=True)

    # Create expected directory structure for Phase C→G
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)
    cli_log_dir = hub / "cli"
    cli_log_dir.mkdir(parents=True)
    phase_g_root = hub / "analysis"
    phase_g_root.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse (NO --collect-only, so real execution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--clobber",  # Required to pass prepare_hub check
        ],
    )

    # Stub heavy helpers to no-op (we only care about run_command invocations)
    def stub_prepare_hub(hub_path, clobber):
        """No-op stub for prepare_hub."""
        pass

    def stub_validate_phase_c_metadata(hub_path):
        """No-op stub for validate_phase_c_metadata."""
        pass

    def stub_summarize_phase_g_outputs(hub_path):
        """No-op stub for summarize_phase_g_outputs."""
        pass

    monkeypatch.setattr(module, "prepare_hub", stub_prepare_hub)
    monkeypatch.setattr(module, "validate_phase_c_metadata", stub_validate_phase_c_metadata)
    monkeypatch.setattr(module, "summarize_phase_g_outputs", stub_summarize_phase_g_outputs)
    _stub_generate_overlap_views(monkeypatch)

    # Create deterministic highlights file when reporting helper is invoked
    def stub_run_command(cmd, log_path):
        """Stub that writes highlights file when reporting helper is invoked."""
        cmd_str = " ".join(str(c) for c in cmd)
        if "report_phase_g_dense_metrics.py" in cmd_str and "--highlights" in cmd_str:
            # Extract highlights path from command
            # Command format: [..., '--highlights', 'path/to/aggregate_highlights.txt', ...]
            for i, part in enumerate(cmd):
                if str(part) == "--highlights" and i + 1 < len(cmd):
                    highlights_path = Path(cmd[i + 1])
                    highlights_path.parent.mkdir(parents=True, exist_ok=True)
                    # Write deterministic highlights content
                    highlights_path.write_text(
                        "Phase G Dense Metrics — Highlights\n"
                        "==================================================\n"
                        "\n"
                        "MS-SSIM Deltas (PtychoPINN - Baseline):\n"
                        "  Amplitude (mean): +0.123\n"
                        "  Phase (mean):     +0.045\n"
                        "\n"
                        "MS-SSIM Deltas (PtychoPINN - PtyChi):\n"
                        "  Amplitude (mean): +0.067\n"
                        "  Phase (mean):     +0.012\n"
                        "\n"
                        "MAE Deltas (PtychoPINN - Baseline):\n"
                        "  [Note: Negative = PtychoPINN better (lower error)]\n"
                        "  Amplitude (mean): -0.008\n"
                        "  Phase (mean):     -0.003\n"
                        "\n"
                        "MAE Deltas (PtychoPINN - PtyChi):\n"
                        "  [Note: Negative = PtychoPINN better (lower error)]\n"
                        "  Amplitude (mean): -0.005\n"
                        "  Phase (mean):     -0.001\n",
                        encoding="utf-8"
                    )
                    break

    monkeypatch.setattr(module, "run_command", stub_run_command)

    # Execute: Call main() (should execute Phase C→G pipeline + reporting helper + highlights preview)
    exit_code = main()

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from real execution mode, got {exit_code}"

    # Capture stdout
    captured = capsys.readouterr()
    stdout = captured.out

    # Assert: stdout should contain "Aggregate highlights preview" banner
    assert "Aggregate highlights preview" in stdout, \
        f"Expected stdout to contain 'Aggregate highlights preview' banner, got:\n{stdout}"

    # Assert: stdout should contain sample highlights content
    assert "MS-SSIM Deltas (PtychoPINN - Baseline):" in stdout, \
        f"Expected highlights preview to contain MS-SSIM delta header, got:\n{stdout}"
    assert "Amplitude (mean): +0.123" in stdout, \
        f"Expected highlights preview to contain amplitude delta value, got:\n{stdout}"
    assert "MAE Deltas (PtychoPINN - Baseline):" in stdout, \
        f"Expected highlights preview to contain MAE delta header, got:\n{stdout}"

    # Assert: stdout should contain hub-relative path references (TYPE-PATH-001, TEST-CLI-001)
    assert "CLI logs: cli" in stdout, \
        f"Expected stdout to contain 'CLI logs: cli' (hub-relative path), got:\n{stdout}"
    assert "Analysis outputs: analysis" in stdout, \
        f"Expected stdout to contain 'Analysis outputs: analysis' (hub-relative path), got:\n{stdout}"
    assert "analysis/artifact_inventory.txt" in stdout, \
        f"Expected stdout to contain 'analysis/artifact_inventory.txt' reference, got:\n{stdout}"


def test_run_phase_g_dense_exec_runs_analyze_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that main() in real execution mode invokes analyze_dense_metrics.py after reporting helper
    and emits MS-SSIM/MAE delta summary to stdout.

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Stubs prepare_hub, validate_phase_c_metadata, summarize_phase_g_outputs (seeds metrics_summary.json)
    - Monkeypatches run_command to record invocations and create required files
    - Runs main() without --collect-only to trigger real execution path
    - Asserts analyze_dense_metrics.py is invoked after report_phase_g_dense_metrics.py
    - Validates analyze command includes --metrics, --highlights, --output flags
    - Validates log_path points to cli/metrics_digest_cli.log
    - Validates stdout contains delta block with four lines (MS-SSIM vs Baseline/PtyChi, MAE vs Baseline/PtyChi)
    - Ensures AUTHORITATIVE_CMDS_DOC environment variable is respected
    - Returns 0 exit code on success

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Import main() and helper functions from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory
    hub = tmp_path / "exec_hub"
    hub.mkdir(parents=True)

    # Create expected directory structure for Phase C→G
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)
    cli_log_dir = hub / "cli"
    cli_log_dir.mkdir(parents=True)
    phase_g_root = hub / "analysis"
    phase_g_root.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse (NO --collect-only, so real execution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--clobber",  # Required to pass prepare_hub check
        ],
    )

    # Stub heavy helpers to no-op (we only care about run_command invocations)
    def stub_prepare_hub(hub_path, clobber):
        """No-op stub for prepare_hub."""
        pass

    def stub_validate_phase_c_metadata(hub_path):
        """No-op stub for validate_phase_c_metadata."""
        pass

    def stub_summarize_phase_g_outputs(hub_path):
        """Create metrics_summary.json with test data for delta computation."""
        analysis = Path(hub_path) / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)

        # Create metrics_summary.json with aggregate_metrics for delta computation
        summary_data = {
            "n_jobs": 2,
            "n_success": 2,
            "n_failed": 0,
            "jobs": [],
            "aggregate_metrics": {
                "PtychoPINN": {
                    "ms_ssim": {
                        "mean_amplitude": 0.950,
                        "best_amplitude": 0.955,
                        "mean_phase": 0.920,
                        "best_phase": 0.925
                    },
                    "mae": {
                        "mean_amplitude": 0.025,
                        "mean_phase": 0.035
                    }
                },
                "Baseline": {
                    "ms_ssim": {
                        "mean_amplitude": 0.930,
                        "best_amplitude": 0.935,
                        "mean_phase": 0.900,
                        "best_phase": 0.905
                    },
                    "mae": {
                        "mean_amplitude": 0.030,
                        "mean_phase": 0.040
                    }
                },
                "PtyChi": {
                    "ms_ssim": {
                        "mean_amplitude": 0.940,
                        "best_amplitude": 0.945,
                        "mean_phase": 0.910,
                        "best_phase": 0.915
                    },
                    "mae": {
                        "mean_amplitude": 0.027,
                        "mean_phase": 0.037
                    }
                }
            }
        }

        import json
        metrics_summary_path = analysis / "metrics_summary.json"
        with metrics_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    def stub_generate_artifact_inventory(hub_path):
        """Create artifact_inventory.txt with test data listing key artifacts."""
        analysis = Path(hub_path) / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)

        inventory_path = analysis / "artifact_inventory.txt"

        # List key artifacts that should be present after pipeline execution
        artifacts = [
            "analysis/aggregate_highlights.txt",
            "analysis/aggregate_report.md",
            "analysis/comparison_manifest.json",
            "analysis/metrics_delta_highlights.txt",
            "analysis/metrics_delta_summary.json",
            "analysis/metrics_digest.md",
            "analysis/metrics_summary.json",
            "analysis/metrics_summary.md",
            "cli/metrics_digest_cli.log",
            "cli/aggregate_report_cli.log",
        ]

        # Write sorted artifact list (deterministic)
        with inventory_path.open("w", encoding="utf-8") as f:
            for artifact in sorted(artifacts):
                f.write(f"{artifact}\n")

    monkeypatch.setattr(module, "prepare_hub", stub_prepare_hub)
    monkeypatch.setattr(module, "validate_phase_c_metadata", stub_validate_phase_c_metadata)
    monkeypatch.setattr(module, "summarize_phase_g_outputs", stub_summarize_phase_g_outputs)
    monkeypatch.setattr(module, "generate_artifact_inventory", stub_generate_artifact_inventory)
    _stub_generate_overlap_views(monkeypatch)

    # Record run_command invocations
    run_command_calls = []

    def stub_run_command(cmd, log_path):
        """Record cmd and log_path, create required files for orchestrator progression."""
        run_command_calls.append((cmd, log_path))
        cmd_str = " ".join(str(c) for c in cmd)

        # Create log file for every invocation (TEST-CLI-001)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"Stub log for: {cmd_str}\n", encoding="utf-8")

        # When reporting helper is invoked, create highlights file
        if "report_phase_g_dense_metrics.py" in cmd_str and "--highlights" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--highlights" and i + 1 < len(cmd):
                    highlights_path = Path(cmd[i + 1])
                    highlights_path.parent.mkdir(parents=True, exist_ok=True)
                    highlights_path.write_text("MS-SSIM Deltas (PtychoPINN - Baseline):\n  Amplitude (mean): +0.050\n", encoding="utf-8")
                    break

        # When analyze digest is invoked, create digest file
        if "analyze_dense_metrics.py" in cmd_str and "--output" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--output" and i + 1 < len(cmd):
                    digest_path = Path(cmd[i + 1])
                    digest_path.parent.mkdir(parents=True, exist_ok=True)
                    digest_path.write_text("# Phase G Dense Metrics Digest\n", encoding="utf-8")
                    break

        # When ssim_grid is invoked, create summary file and log
        if "ssim_grid.py" in cmd_str and "--hub" in cmd_str:
            # ssim_grid.py creates analysis/ssim_grid_summary.md
            for i, part in enumerate(cmd):
                if str(part) == "--hub" and i + 1 < len(cmd):
                    hub_path = Path(cmd[i + 1])
                    ssim_grid_summary_path = hub_path / "analysis" / "ssim_grid_summary.md"
                    ssim_grid_summary_path.parent.mkdir(parents=True, exist_ok=True)
                    ssim_grid_summary_path.write_text("# SSIM Grid Summary (Phase-Only)\n", encoding="utf-8")
                    # Create ssim_grid CLI log
                    ssim_grid_log_path = hub_path / "cli" / "ssim_grid_cli.log"
                    ssim_grid_log_path.parent.mkdir(parents=True, exist_ok=True)
                    ssim_grid_log_path.write_text("SSIM grid generation complete\n", encoding="utf-8")
                    break

        # When verify_dense_pipeline_artifacts is invoked, create report and log files
        if "verify_dense_pipeline_artifacts.py" in cmd_str and "--report" in cmd_str:
            for i, part in enumerate(cmd):
                if str(part) == "--report" and i + 1 < len(cmd):
                    report_path = Path(cmd[i + 1])
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text('{"valid": true}\n', encoding="utf-8")
                    # Create verification log in analysis/
                    hub_path = report_path.parent.parent  # analysis -> hub
                    verify_log_path = hub_path / "analysis" / "verify_dense_stdout.log"
                    verify_log_path.parent.mkdir(parents=True, exist_ok=True)
                    verify_log_path.write_text("Verification complete\n", encoding="utf-8")
                    break

        # When check_dense_highlights_match is invoked, create log file
        if "check_dense_highlights_match.py" in cmd_str:
            # Extract hub path from command to create log in analysis/
            for i, part in enumerate(cmd):
                if str(part) == "--hub" and i + 1 < len(cmd):
                    hub_path = Path(cmd[i + 1])
                    check_log_path = hub_path / "analysis" / "check_dense_highlights.log"
                    check_log_path.parent.mkdir(parents=True, exist_ok=True)
                    check_log_path.write_text("Highlights check complete\n", encoding="utf-8")
                    break

    monkeypatch.setattr(module, "run_command", stub_run_command)

    # Execute: Call main() with stdout capture (should execute Phase C→G pipeline + reporting helper + analyze digest)
    import io
    import contextlib

    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        exit_code = main()

    stdout = stdout_buffer.getvalue()

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from real execution mode, got {exit_code}"

    # Assert: run_command should have been called for all phases + reporting helper + analyze digest + ssim_grid
    # Expected: 1 Phase C + 1 Phase D + 2 Phase E (baseline gs1 + dense gs2) + 2 Phase F (train/test) + 2 Phase G (train/test) + 1 reporting helper + 1 analyze digest + 1 ssim_grid = 11 total
    assert len(run_command_calls) >= 11, f"Expected at least 11 run_command calls (C/D/E/F/G phases + reporting + analyze + ssim_grid), got {len(run_command_calls)}"

    # Find reporting helper, analyze digest, and ssim_grid calls
    reporting_helper_idx = None
    analyze_digest_idx = None
    ssim_grid_idx = None

    for idx, (cmd, log_path) in enumerate(run_command_calls):
        cmd_str = " ".join(str(c) for c in cmd)
        if "report_phase_g_dense_metrics.py" in cmd_str:
            reporting_helper_idx = idx
        if "analyze_dense_metrics.py" in cmd_str:
            analyze_digest_idx = idx
        if "ssim_grid.py" in cmd_str:
            ssim_grid_idx = idx

    # Assert: All three helpers should be invoked
    assert reporting_helper_idx is not None, "reporting helper (report_phase_g_dense_metrics.py) was not invoked"
    assert analyze_digest_idx is not None, "analyze digest (analyze_dense_metrics.py) was not invoked"
    assert ssim_grid_idx is not None, "ssim_grid (ssim_grid.py) was not invoked"

    # Assert: analyze_dense_metrics.py should be invoked AFTER report_phase_g_dense_metrics.py
    assert analyze_digest_idx > reporting_helper_idx, \
        f"analyze_dense_metrics.py should be invoked after report_phase_g_dense_metrics.py, but got order: reporting={reporting_helper_idx}, analyze={analyze_digest_idx}"

    # Assert: ssim_grid.py should be invoked AFTER analyze_dense_metrics.py
    assert ssim_grid_idx > analyze_digest_idx, \
        f"ssim_grid.py should be invoked after analyze_dense_metrics.py, but got order: analyze={analyze_digest_idx}, ssim_grid={ssim_grid_idx}"

    # Validate analyze digest command
    analyze_cmd, analyze_log_path = run_command_calls[analyze_digest_idx]
    analyze_cmd_str = " ".join(str(c) for c in analyze_cmd)

    assert "analyze_dense_metrics.py" in analyze_cmd_str, \
        f"Expected analyze_dense_metrics.py in command, got: {analyze_cmd_str}"
    assert "--metrics" in analyze_cmd_str, f"Missing --metrics flag in analyze command: {analyze_cmd_str}"
    assert "metrics_summary.json" in analyze_cmd_str, f"Missing metrics_summary.json in analyze command: {analyze_cmd_str}"
    assert "--highlights" in analyze_cmd_str, f"Missing --highlights flag in analyze command: {analyze_cmd_str}"
    assert "aggregate_highlights.txt" in analyze_cmd_str, f"Missing aggregate_highlights.txt in analyze command: {analyze_cmd_str}"
    assert "--output" in analyze_cmd_str, f"Missing --output flag in analyze command: {analyze_cmd_str}"
    assert "metrics_digest.md" in analyze_cmd_str, f"Missing metrics_digest.md in analyze command: {analyze_cmd_str}"

    # Validate log_path points to cli/metrics_digest_cli.log
    assert "metrics_digest_cli.log" in str(analyze_log_path), \
        f"Expected analyze digest log path to be cli/metrics_digest_cli.log, got: {analyze_log_path}"

    # Assert: Success banner should include metrics digest paths (TYPE-PATH-001)
    assert "Metrics digest:" in stdout, \
        f"Expected success banner to include 'Metrics digest:' line, got:\n{stdout}"
    assert "Metrics digest log:" in stdout, \
        f"Expected success banner to include 'Metrics digest log:' line, got:\n{stdout}"

    # Assert: "Metrics digest:" banner line appears exactly once (TYPE-PATH-001 regression guard)
    digest_banner_count = stdout.count("Metrics digest: ")
    assert digest_banner_count == 1, \
        f"Expected 'Metrics digest: ' to appear exactly once in stdout (TYPE-PATH-001), but found {digest_banner_count} occurrences. " \
        f"This guards against duplicate banner regressions. Got:\n{stdout}"

    # Assert: "Metrics digest log:" banner line appears exactly once (TYPE-PATH-001 regression guard)
    digest_log_banner_count = stdout.count("Metrics digest log: ")
    assert digest_log_banner_count == 1, \
        f"Expected 'Metrics digest log: ' to appear exactly once in stdout (TYPE-PATH-001), but found {digest_log_banner_count} occurrences. " \
        f"This guards against duplicate banner regressions. Got:\n{stdout}"

    # Validate digest paths appear in stdout (should show relative paths per TYPE-PATH-001)
    assert "metrics_digest.md" in stdout, \
        f"Expected stdout to mention metrics_digest.md path, got:\n{stdout}"
    assert "metrics_digest_cli.log" in stdout, \
        f"Expected stdout to mention metrics_digest_cli.log path, got:\n{stdout}"

    # Assert: SSIM Grid banner lines must be present (TEST-CLI-001)
    assert "SSIM Grid Summary (phase-only):" in stdout, \
        f"Expected success banner to include 'SSIM Grid Summary (phase-only):' line, got:\n{stdout}"
    assert "ssim_grid_summary.md" in stdout, \
        f"Expected stdout to mention ssim_grid_summary.md path, got:\n{stdout}"
    assert "SSIM Grid log:" in stdout, \
        f"Expected success banner to include 'SSIM Grid log:' line, got:\n{stdout}"
    assert "ssim_grid_cli.log" in stdout, \
        f"Expected stdout to mention ssim_grid_cli.log path, got:\n{stdout}"

    # Assert: Verification banner lines must be present (TEST-CLI-001)
    assert "Verification report:" in stdout, \
        f"Expected success banner to include 'Verification report:' line, got:\n{stdout}"
    assert "verification_report.json" in stdout, \
        f"Expected stdout to mention verification_report.json path, got:\n{stdout}"
    assert "Verification log:" in stdout, \
        f"Expected success banner to include 'Verification log:' line, got:\n{stdout}"
    assert "verify_dense_stdout.log" in stdout, \
        f"Expected stdout to mention verify_dense_stdout.log path, got:\n{stdout}"
    assert "Highlights check log:" in stdout, \
        f"Expected success banner to include 'Highlights check log:' line, got:\n{stdout}"
    assert "check_dense_highlights.log" in stdout, \
        f"Expected stdout to mention check_dense_highlights.log path, got:\n{stdout}"

    # Assert: stdout should contain delta summary block with four delta lines
    # Expected deltas (computed from stub data above):
    # MS-SSIM vs Baseline: mean_amp = 0.950 - 0.930 = +0.020, mean_phase = 0.920 - 0.900 = +0.020
    # MS-SSIM vs PtyChi: mean_amp = 0.950 - 0.940 = +0.010, mean_phase = 0.920 - 0.910 = +0.010
    # MAE vs Baseline: mean_amp = 0.025 - 0.030 = -0.005, mean_phase = 0.035 - 0.040 = -0.005
    # MAE vs PtyChi: mean_amp = 0.025 - 0.027 = -0.002, mean_phase = 0.035 - 0.037 = -0.002

    assert "MS-SSIM Δ (PtychoPINN - Baseline)" in stdout, \
        f"Expected stdout to contain MS-SSIM delta line vs Baseline, got:\n{stdout}"
    assert "+0.020" in stdout, \
        f"Expected stdout to contain +0.020 delta value (MS-SSIM amplitude vs Baseline), got:\n{stdout}"

    assert "MS-SSIM Δ (PtychoPINN - PtyChi)" in stdout, \
        f"Expected stdout to contain MS-SSIM delta line vs PtyChi, got:\n{stdout}"
    assert "+0.010" in stdout, \
        f"Expected stdout to contain +0.010 delta value (MS-SSIM amplitude vs PtyChi), got:\n{stdout}"

    assert "MAE Δ (PtychoPINN - Baseline)" in stdout, \
        f"Expected stdout to contain MAE delta line vs Baseline, got:\n{stdout}"
    assert "-0.005" in stdout, \
        f"Expected stdout to contain -0.005 delta value (MAE amplitude vs Baseline), got:\n{stdout}"

    assert "MAE Δ (PtychoPINN - PtyChi)" in stdout, \
        f"Expected stdout to contain MAE delta line vs PtyChi, got:\n{stdout}"
    assert "-0.002" in stdout, \
        f"Expected stdout to contain -0.002 delta value (MAE amplitude vs PtyChi), got:\n{stdout}"

    # Assert: metrics_delta_summary.json should be created in analysis/
    delta_json_path = phase_g_root / "metrics_delta_summary.json"
    assert delta_json_path.exists(), \
        f"Expected metrics_delta_summary.json to exist at {delta_json_path}, but it was not found"

    # Assert: JSON should contain valid delta structure with numeric values
    import json
    with delta_json_path.open("r", encoding="utf-8") as f:
        delta_data = json.load(f)

    # Validate top-level structure
    assert "deltas" in delta_data, "Expected 'deltas' key in metrics_delta_summary.json"
    deltas = delta_data["deltas"]

    # Validate Baseline deltas (PtychoPINN - Baseline)
    assert "vs_Baseline" in deltas, "Expected 'vs_Baseline' key in deltas"
    baseline_deltas = deltas["vs_Baseline"]
    assert "ms_ssim" in baseline_deltas, "Expected 'ms_ssim' in vs_Baseline deltas"
    assert "mae" in baseline_deltas, "Expected 'mae' in vs_Baseline deltas"

    # Validate MS-SSIM vs Baseline values (numeric floats, not formatted strings)
    ms_ssim_baseline = baseline_deltas["ms_ssim"]
    assert "amplitude" in ms_ssim_baseline, "Expected 'amplitude' in ms_ssim vs_Baseline"
    assert "phase" in ms_ssim_baseline, "Expected 'phase' in ms_ssim vs_Baseline"
    assert abs(ms_ssim_baseline["amplitude"] - 0.020) < 1e-6, \
        f"Expected ms_ssim amplitude delta vs Baseline to be ~0.020, got {ms_ssim_baseline['amplitude']}"
    assert abs(ms_ssim_baseline["phase"] - 0.020) < 1e-6, \
        f"Expected ms_ssim phase delta vs Baseline to be ~0.020, got {ms_ssim_baseline['phase']}"

    # Validate MAE vs Baseline values (negative deltas expected)
    mae_baseline = baseline_deltas["mae"]
    assert "amplitude" in mae_baseline, "Expected 'amplitude' in mae vs_Baseline"
    assert "phase" in mae_baseline, "Expected 'phase' in mae vs_Baseline"
    assert abs(mae_baseline["amplitude"] - (-0.005)) < 1e-6, \
        f"Expected mae amplitude delta vs Baseline to be ~-0.005, got {mae_baseline['amplitude']}"
    assert abs(mae_baseline["phase"] - (-0.005)) < 1e-6, \
        f"Expected mae phase delta vs Baseline to be ~-0.005, got {mae_baseline['phase']}"

    # Validate PtyChi deltas (PtychoPINN - PtyChi)
    assert "vs_PtyChi" in deltas, "Expected 'vs_PtyChi' key in deltas"
    ptychi_deltas = deltas["vs_PtyChi"]
    assert "ms_ssim" in ptychi_deltas, "Expected 'ms_ssim' in vs_PtyChi deltas"
    assert "mae" in ptychi_deltas, "Expected 'mae' in vs_PtyChi deltas"

    # Validate MS-SSIM vs PtyChi values
    ms_ssim_ptychi = ptychi_deltas["ms_ssim"]
    assert abs(ms_ssim_ptychi["amplitude"] - 0.010) < 1e-6, \
        f"Expected ms_ssim amplitude delta vs PtyChi to be ~0.010, got {ms_ssim_ptychi['amplitude']}"
    assert abs(ms_ssim_ptychi["phase"] - 0.010) < 1e-6, \
        f"Expected ms_ssim phase delta vs PtyChi to be ~0.010, got {ms_ssim_ptychi['phase']}"

    # Validate MAE vs PtyChi values (negative deltas expected)
    mae_ptychi = ptychi_deltas["mae"]
    assert abs(mae_ptychi["amplitude"] - (-0.002)) < 1e-6, \
        f"Expected mae amplitude delta vs PtyChi to be ~-0.002, got {mae_ptychi['amplitude']}"
    assert abs(mae_ptychi["phase"] - (-0.002)) < 1e-6, \
        f"Expected mae phase delta vs PtyChi to be ~-0.002, got {mae_ptychi['phase']}"

    # Assert: Success banner should mention metrics_delta_summary.json path
    assert "metrics_delta_summary.json" in stdout, \
        f"Expected success banner to mention metrics_delta_summary.json, got:\n{stdout}"

    # Assert: artifact_inventory.txt should be created in analysis/ directory
    inventory_path = phase_g_root / "artifact_inventory.txt"
    assert inventory_path.exists(), \
        f"Expected artifact_inventory.txt to exist at {inventory_path}, but it was not found"

    # Assert: Inventory file should be non-empty and contain POSIX-style relative paths
    inventory_content = inventory_path.read_text(encoding="utf-8")
    assert inventory_content.strip(), \
        f"Expected artifact_inventory.txt to be non-empty, but it was empty"

    # Assert: Inventory should list key artifacts (at minimum: metrics_summary.json, aggregate_report.md)
    assert "metrics_summary.json" in inventory_content, \
        f"Expected artifact_inventory.txt to list metrics_summary.json, got:\n{inventory_content}"
    assert "aggregate_report.md" in inventory_content, \
        f"Expected artifact_inventory.txt to list aggregate_report.md, got:\n{inventory_content}"

    # Assert: JSON should contain provenance metadata fields (generated_at, source_metrics)
    assert "generated_at" in delta_data, "Expected 'generated_at' key in metrics_delta_summary.json"
    assert "source_metrics" in delta_data, "Expected 'source_metrics' key in metrics_delta_summary.json"

    # Validate generated_at is ISO8601 UTC timestamp string
    generated_at = delta_data["generated_at"]
    assert isinstance(generated_at, str), f"Expected generated_at to be string, got {type(generated_at)}"
    assert generated_at.endswith("Z"), f"Expected generated_at to be UTC (end with 'Z'), got {generated_at}"
    # Basic ISO8601 format check: YYYY-MM-DDTHH:MM:SSZ pattern
    import re
    iso8601_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    assert re.match(iso8601_pattern, generated_at), \
        f"Expected generated_at to match ISO8601 UTC format (YYYY-MM-DDTHH:MM:SSZ), got {generated_at}"

    # Validate source_metrics is relative POSIX path (TYPE-PATH-001)
    source_metrics = delta_data["source_metrics"]
    assert isinstance(source_metrics, str), f"Expected source_metrics to be string, got {type(source_metrics)}"
    assert not source_metrics.startswith("/"), \
        f"Expected source_metrics to be relative path (TYPE-PATH-001), got absolute: {source_metrics}"
    assert "metrics_summary.json" in source_metrics, \
        f"Expected source_metrics to reference metrics_summary.json, got {source_metrics}"

    # Assert: metrics_delta_highlights.txt should be created in analysis/
    highlights_txt_path = phase_g_root / "metrics_delta_highlights.txt"
    assert highlights_txt_path.exists(), \
        f"Expected metrics_delta_highlights.txt to exist at {highlights_txt_path}, but it was not found"

    # Read highlights file content
    highlights_content = highlights_txt_path.read_text(encoding="utf-8")

    # Assert: highlights file should contain exactly 4 delta lines (MS-SSIM vs Baseline/PtyChi, MAE vs Baseline/PtyChi)
    # Expected format: "MS-SSIM Δ (PtychoPINN - Baseline)  : amplitude +0.020  phase +0.020"
    assert "MS-SSIM Δ (PtychoPINN - Baseline)" in highlights_content, \
        f"Expected highlights to contain MS-SSIM delta line vs Baseline, got:\n{highlights_content}"
    assert "amplitude +0.020" in highlights_content, \
        f"Expected highlights to contain amplitude +0.020 (MS-SSIM vs Baseline), got:\n{highlights_content}"
    assert "phase +0.020" in highlights_content, \
        f"Expected highlights to contain phase +0.020 (MS-SSIM vs Baseline), got:\n{highlights_content}"

    assert "MS-SSIM Δ (PtychoPINN - PtyChi)" in highlights_content, \
        f"Expected highlights to contain MS-SSIM delta line vs PtyChi, got:\n{highlights_content}"
    assert "amplitude +0.010" in highlights_content, \
        f"Expected highlights to contain amplitude +0.010 (MS-SSIM vs PtyChi), got:\n{highlights_content}"
    assert "phase +0.010" in highlights_content, \
        f"Expected highlights to contain phase +0.010 (MS-SSIM vs PtyChi), got:\n{highlights_content}"

    assert "MAE Δ (PtychoPINN - Baseline)" in highlights_content, \
        f"Expected highlights to contain MAE delta line vs Baseline, got:\n{highlights_content}"
    assert "amplitude -0.005" in highlights_content, \
        f"Expected highlights to contain amplitude -0.005 (MAE vs Baseline), got:\n{highlights_content}"
    assert "phase -0.005" in highlights_content, \
        f"Expected highlights to contain phase -0.005 (MAE vs Baseline), got:\n{highlights_content}"

    assert "MAE Δ (PtychoPINN - PtyChi)" in highlights_content, \
        f"Expected highlights to contain MAE delta line vs PtyChi, got:\n{highlights_content}"
    assert "amplitude -0.002" in highlights_content, \
        f"Expected highlights to contain amplitude -0.002 (MAE vs PtyChi), got:\n{highlights_content}"
    assert "phase -0.002" in highlights_content, \
        f"Expected highlights to contain phase -0.002 (MAE vs PtyChi), got:\n{highlights_content}"

    # Assert: Success banner should mention metrics_delta_highlights.txt path (TYPE-PATH-001)
    assert "metrics_delta_highlights.txt" in stdout, \
        f"Expected success banner to mention metrics_delta_highlights.txt, got:\n{stdout}"

    # Validate ssim_grid command
    ssim_grid_cmd, ssim_grid_log_path = run_command_calls[ssim_grid_idx]
    ssim_grid_cmd_str = " ".join(str(c) for c in ssim_grid_cmd)

    assert "ssim_grid.py" in ssim_grid_cmd_str, \
        f"Expected ssim_grid.py in command, got: {ssim_grid_cmd_str}"
    assert "--hub" in ssim_grid_cmd_str, f"Missing --hub flag in ssim_grid command: {ssim_grid_cmd_str}"

    # Validate log_path points to cli/ssim_grid_cli.log
    assert "ssim_grid_cli.log" in str(ssim_grid_log_path), \
        f"Expected ssim_grid log path to be cli/ssim_grid_cli.log, got: {ssim_grid_log_path}"

    # Assert: ssim_grid_summary.md should be created in analysis/
    ssim_grid_summary_path = phase_g_root / "ssim_grid_summary.md"
    assert ssim_grid_summary_path.exists(), \
        f"Expected ssim_grid_summary.md to exist at {ssim_grid_summary_path}, but it was not found"

    # Assert: Success banner should mention ssim_grid_summary.md path (TYPE-PATH-001)
    assert "ssim_grid_summary.md" in stdout, \
        f"Expected success banner to mention ssim_grid_summary.md, got:\n{stdout}"


def test_persist_delta_highlights_creates_preview(tmp_path: Path) -> None:
    """
    Test that persist_delta_highlights() creates both highlights.txt and preview.txt with correct precision.

    Acceptance:
    - Feeds synthetic aggregate metrics (PtychoPINN/Baseline/PtyChi) through the helper
    - Asserts highlight lines contain signed values with correct precision:
      * MS-SSIM: ±0.000 (3 decimals)
      * MAE: ±0.000000 (6 decimals)
    - Verifies preview file exists with four phase-only lines
    - Checks returned numeric deltas (e.g., Baseline.mae.phase == -0.000025)
    - Does NOT perform file I/O outside tmp_path (helper receives output_dir argument)

    Follows input.md Do Now step 2 (TDD RED→GREEN for preview helper).
    """
    # Import the helper (it doesn't exist yet, so this will fail)
    module = _import_orchestrator_module()
    persist_delta_highlights = module.persist_delta_highlights

    # Setup: Create synthetic aggregate metrics matching metrics_summary.json structure
    aggregate_metrics = {
        "PtychoPINN": {
            "ms_ssim": {
                "mean_amplitude": 0.950,
                "mean_phase": 0.920
            },
            "mae": {
                "mean_amplitude": 0.025000,
                "mean_phase": 0.035000
            }
        },
        "Baseline": {
            "ms_ssim": {
                "mean_amplitude": 0.940,
                "mean_phase": 0.905
            },
            "mae": {
                "mean_amplitude": 0.030000,
                "mean_phase": 0.035025
            }
        },
        "PtyChi": {
            "ms_ssim": {
                "mean_amplitude": 0.945,
                "mean_phase": 0.912
            },
            "mae": {
                "mean_amplitude": 0.027500,
                "mean_phase": 0.035018
            }
        }
    }

    output_dir = tmp_path / "analysis"
    output_dir.mkdir(parents=True)

    # Execute: Call the helper (should write both txt files and return delta_summary dict)
    delta_summary = persist_delta_highlights(
        aggregate_metrics=aggregate_metrics,
        output_dir=output_dir,
        hub=tmp_path
    )

    # Assert: Verify highlights.txt exists with 4 lines (both amplitude and phase for each baseline)
    highlights_txt_path = output_dir / "metrics_delta_highlights.txt"
    assert highlights_txt_path.exists(), f"Expected highlights.txt at {highlights_txt_path}"
    highlights_content = highlights_txt_path.read_text(encoding="utf-8")
    highlights_lines = [line for line in highlights_content.strip().split("\n") if line]
    assert len(highlights_lines) == 4, f"Expected 4 highlight lines, got {len(highlights_lines)}: {highlights_lines}"

    # Assert: Check MS-SSIM precision (±0.000, 3 decimals)
    # PtychoPINN - Baseline: ms_ssim.phase = 0.920 - 0.905 = 0.015 → +0.015
    assert "MS-SSIM Δ (PtychoPINN - Baseline)  : amplitude +0.010  phase +0.015" in highlights_content, \
        f"Expected MS-SSIM vs Baseline with ±0.000 precision, got:\n{highlights_content}"

    # Assert: Check MAE precision (±0.000000, 6 decimals)
    # PtychoPINN - Baseline: mae.phase = 0.035000 - 0.035025 = -0.000025 → -0.000025
    assert "MAE Δ (PtychoPINN - Baseline)      : amplitude -0.005000  phase -0.000025" in highlights_content, \
        f"Expected MAE vs Baseline with ±0.000000 precision, got:\n{highlights_content}"

    # Assert: Check PtyChi deltas
    # MS-SSIM: 0.920 - 0.912 = 0.008 → +0.008
    assert "MS-SSIM Δ (PtychoPINN - PtyChi)    : amplitude +0.005  phase +0.008" in highlights_content, \
        f"Expected MS-SSIM vs PtyChi with ±0.000 precision, got:\n{highlights_content}"

    # MAE: 0.035000 - 0.035018 = -0.000018 → -0.000018
    assert "MAE Δ (PtychoPINN - PtyChi)        : amplitude -0.002500  phase -0.000018" in highlights_content, \
        f"Expected MAE vs PtyChi with ±0.000000 precision, got:\n{highlights_content}"

    # Assert: Verify preview.txt exists with 4 phase-only lines
    preview_txt_path = output_dir / "metrics_delta_highlights_preview.txt"
    assert preview_txt_path.exists(), f"Expected preview.txt at {preview_txt_path}"
    preview_content = preview_txt_path.read_text(encoding="utf-8")
    preview_lines = [line for line in preview_content.strip().split("\n") if line]
    assert len(preview_lines) == 4, f"Expected 4 preview lines (phase-only), got {len(preview_lines)}: {preview_lines}"

    # Assert: Preview should contain phase deltas only (not amplitude)
    assert "MS-SSIM Δ (PtychoPINN - Baseline): +0.015" in preview_content, \
        f"Expected preview MS-SSIM vs Baseline phase: +0.015, got:\n{preview_content}"
    assert "MS-SSIM Δ (PtychoPINN - PtyChi): +0.008" in preview_content, \
        f"Expected preview MS-SSIM vs PtyChi phase: +0.008, got:\n{preview_content}"
    assert "MAE Δ (PtychoPINN - Baseline): -0.000025" in preview_content, \
        f"Expected preview MAE vs Baseline phase: -0.000025, got:\n{preview_content}"
    assert "MAE Δ (PtychoPINN - PtyChi): -0.000018" in preview_content, \
        f"Expected preview MAE vs PtyChi phase: -0.000018, got:\n{preview_content}"

    # Assert: Preview should NOT contain "amplitude" keyword
    assert "amplitude" not in preview_content.lower(), \
        f"Expected preview to be phase-only (no 'amplitude'), got:\n{preview_content}"

    # Assert: Verify JSON structure returned
    assert delta_summary is not None, "Expected delta_summary dict to be returned"
    assert "deltas" in delta_summary, f"Expected 'deltas' key in delta_summary, got: {delta_summary.keys()}"
    assert "vs_Baseline" in delta_summary["deltas"], f"Expected 'vs_Baseline' in deltas, got: {delta_summary['deltas'].keys()}"
    assert "vs_PtyChi" in delta_summary["deltas"], f"Expected 'vs_PtyChi' in deltas, got: {delta_summary['deltas'].keys()}"

    # Assert: Check returned numeric deltas (raw floats, not formatted strings)
    baseline_deltas = delta_summary["deltas"]["vs_Baseline"]
    assert baseline_deltas["mae"]["phase"] == pytest.approx(-0.000025, abs=1e-9), \
        f"Expected Baseline MAE phase delta == -0.000025, got: {baseline_deltas['mae']['phase']}"

    ptychi_deltas = delta_summary["deltas"]["vs_PtyChi"]
    assert ptychi_deltas["mae"]["phase"] == pytest.approx(-0.000018, abs=1e-9), \
        f"Expected PtyChi MAE phase delta == -0.000018, got: {ptychi_deltas['mae']['phase']}"


def test_run_phase_g_dense_collect_only_post_verify_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """
    Test that main() with --collect-only --post-verify-only prints only verification commands (SSIM grid + verify + check).

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Runs with --collect-only --post-verify-only into a tmp hub directory
    - Asserts stdout contains only SSIM grid, verify, and check commands with hub-relative log paths
    - Verifies Phase C→F commands are NOT printed
    - Returns 0 exit code on success

    Follows TYPE-PATH-001 (hub-relative paths), TEST-CLI-001 (collect-only mode validation).
    """
    # Import main() from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory
    hub = tmp_path / "post_verify_only_collect_hub"
    hub.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--collect-only",
            "--post-verify-only",
        ],
    )

    # Execute: Call main() (should print commands and return 0)
    exit_code = main()

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from --collect-only --post-verify-only mode, got {exit_code}"

    # Assert: Capture stdout and verify expected command substrings
    captured = capsys.readouterr()
    stdout = captured.out

    # Check for post-verify-only mode banner
    assert "Post-verify-only mode: skipping Phase C→F" in stdout, "Missing post-verify-only mode banner"

    # Check for verification commands
    assert "ssim_grid.py" in stdout, "Missing ssim_grid.py command in --post-verify-only output"
    assert "verify_dense_pipeline_artifacts.py" in stdout, "Missing verify_dense_pipeline_artifacts.py command in --post-verify-only output"
    assert "check_dense_highlights_match.py" in stdout, "Missing check_dense_highlights_match.py command in --post-verify-only output"

    # Check for hub-relative log paths (TYPE-PATH-001)
    assert "cli/ssim_grid_cli.log" in stdout or "ssim_grid_cli.log" in stdout, "Missing hub-relative ssim_grid_cli.log path"
    assert "analysis/verify_dense_stdout.log" in stdout or "verify_dense_stdout.log" in stdout, "Missing hub-relative verify_dense_stdout.log path"
    assert "analysis/check_dense_highlights.log" in stdout or "check_dense_highlights.log" in stdout, "Missing hub-relative check_dense_highlights.log path"

    # Assert: Phase C→F commands should NOT be present
    assert "Phase C: Dataset Generation" not in stdout, "Phase C command should not be printed in --post-verify-only mode"
    assert "studies.fly64_dose_overlap.generation" not in stdout, "Generation module should not be present in --post-verify-only output"
    assert "Phase D: Overlap View Generation" not in stdout, "Phase D command should not be printed in --post-verify-only mode"
    assert "studies.fly64_dose_overlap.training" not in stdout, "Training module should not be present in --post-verify-only output"
    assert "studies.fly64_dose_overlap.reconstruction" not in stdout, "Reconstruction module should not be present in --post-verify-only output"
    assert "studies.fly64_dose_overlap.comparison" not in stdout, "Comparison module should not be present in --post-verify-only output"


def test_run_phase_g_dense_post_verify_only_executes_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """
    Test that main() with --post-verify-only executes verification commands in correct order (SSIM grid → verify → check).

    Acceptance:
    - Loads main() from orchestrator script via importlib
    - Monkeypatches run_command to capture invocations and generate_artifact_inventory to record call
    - Runs with --post-verify-only into a tmp hub directory
    - Asserts run_command was called exactly 3 times with correct command order: [ssim_grid, verify, check]
    - Asserts generate_artifact_inventory was called exactly once
    - Asserts stdout contains 'analysis/artifact_inventory.txt' path
    - Returns 0 exit code on success

    Follows DATA-001 (artifact inventory regeneration), TEST-CLI-001 (command capture).
    """
    # Import main() from orchestrator
    module = _import_orchestrator_module()
    main = module.main

    # Setup: Create tmp hub directory with required structure
    hub = tmp_path / "post_verify_only_exec_hub"
    hub.mkdir(parents=True)
    cli_log_dir = hub / "cli"
    cli_log_dir.mkdir(parents=True)
    phase_g_root = hub / "analysis"
    phase_g_root.mkdir(parents=True)

    # Set AUTHORITATIVE_CMDS_DOC to satisfy orchestrator env check
    monkeypatch.setenv("AUTHORITATIVE_CMDS_DOC", "./docs/TESTING_GUIDE.md")

    # Prepare sys.argv for argparse
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase_g_dense.py",
            "--hub", str(hub),
            "--dose", "1000",
            "--view", "dense",
            "--splits", "train", "test",
            "--post-verify-only",
        ],
    )

    # Record run_command invocations
    run_command_calls = []

    def stub_run_command(cmd, log_path, env=None, cwd=None):
        """Record cmd and log_path without executing."""
        run_command_calls.append((cmd, log_path))
        # Create log file to satisfy orchestrator progression
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"Stub log for: {' '.join(str(c) for c in cmd)}\n", encoding="utf-8")

        # Create expected output files based on command type
        cmd_str = " ".join(str(c) for c in cmd)
        if "ssim_grid.py" in cmd_str:
            # Create SSIM grid summary
            ssim_summary = phase_g_root / "ssim_grid_summary.md"
            ssim_summary.write_text("# SSIM Grid Summary\nStub SSIM grid content\n", encoding="utf-8")
        elif "verify_dense_pipeline_artifacts.py" in cmd_str:
            # Create verification report and log
            verify_report = phase_g_root / "verification_report.json"
            verify_report.write_text('{"status": "stub"}', encoding="utf-8")
            verify_log = phase_g_root / "verify_dense_stdout.log"
            verify_log.write_text("Stub verification log\n", encoding="utf-8")
        elif "check_dense_highlights_match.py" in cmd_str:
            # Create highlights check log
            check_log = phase_g_root / "check_dense_highlights.log"
            check_log.write_text("Stub highlights check log\n", encoding="utf-8")

    # Record generate_artifact_inventory invocations
    inventory_calls = []

    def stub_generate_artifact_inventory(hub_path):
        """Record hub_path and create artifact_inventory.txt file."""
        inventory_calls.append(hub_path)
        # Create artifact_inventory.txt to satisfy orchestrator validation
        inventory_path = Path(hub_path) / "analysis" / "artifact_inventory.txt"
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text("# Stub artifact inventory\n", encoding="utf-8")

    monkeypatch.setattr(module, "run_command", stub_run_command)
    monkeypatch.setattr(module, "generate_artifact_inventory", stub_generate_artifact_inventory)

    # Execute: Call main() (should execute verification commands and return 0)
    exit_code = main()

    # Capture stdout
    captured = capsys.readouterr()
    stdout = captured.out

    # Assert: Exit code should be 0
    assert exit_code == 0, f"Expected exit code 0 from --post-verify-only mode, got {exit_code}"

    # Assert: run_command was called exactly 3 times
    assert len(run_command_calls) == 3, f"Expected 3 run_command calls, got {len(run_command_calls)}"

    # Assert: Command order is correct: SSIM grid → verify → check
    cmd_0, log_0 = run_command_calls[0]
    cmd_1, log_1 = run_command_calls[1]
    cmd_2, log_2 = run_command_calls[2]

    cmd_0_str = " ".join(str(c) for c in cmd_0)
    cmd_1_str = " ".join(str(c) for c in cmd_1)
    cmd_2_str = " ".join(str(c) for c in cmd_2)

    assert "ssim_grid.py" in cmd_0_str, f"Expected ssim_grid.py in first command, got: {cmd_0_str}"
    assert "verify_dense_pipeline_artifacts.py" in cmd_1_str, f"Expected verify_dense_pipeline_artifacts.py in second command, got: {cmd_1_str}"
    assert "check_dense_highlights_match.py" in cmd_2_str, f"Expected check_dense_highlights_match.py in third command, got: {cmd_2_str}"

    # Assert: generate_artifact_inventory was called exactly once
    assert len(inventory_calls) == 1, f"Expected 1 generate_artifact_inventory call, got {len(inventory_calls)}"
    assert inventory_calls[0] == hub, f"Expected hub={hub}, got: {inventory_calls[0]}"

    # Assert: stdout contains artifact_inventory.txt path (TYPE-PATH-001, DATA-001)
    assert "analysis/artifact_inventory.txt" in stdout, \
        f"Expected success banner to contain 'analysis/artifact_inventory.txt', but stdout was:\n{stdout}"

    # Assert: stdout contains hub-relative CLI logs and Analysis outputs paths (TYPE-PATH-001)
    assert "CLI logs: cli" in stdout, \
        f"Expected success banner to contain 'CLI logs: cli' (hub-relative), but stdout was:\n{stdout}"
    assert "Analysis outputs: analysis" in stdout, \
        f"Expected success banner to contain 'Analysis outputs: analysis' (hub-relative), but stdout was:\n{stdout}"

    # Assert: stdout contains SSIM Grid summary and log paths (TYPE-PATH-001, TEST-CLI-001)
    assert "SSIM Grid Summary (phase-only): analysis/ssim_grid_summary.md" in stdout, \
        f"Expected success banner to contain SSIM Grid Summary path, but stdout was:\n{stdout}"
    assert "SSIM Grid log: cli/ssim_grid_cli.log" in stdout, \
        f"Expected success banner to contain SSIM Grid log path, but stdout was:\n{stdout}"

    # Assert: stdout contains verification report and log paths (TYPE-PATH-001, TEST-CLI-001)
    assert "Verification report: analysis/verification_report.json" in stdout, \
        f"Expected success banner to contain Verification report path, but stdout was:\n{stdout}"
    assert "Verification log: analysis/verify_dense_stdout.log" in stdout, \
        f"Expected success banner to contain Verification log path, but stdout was:\n{stdout}"

    # Assert: stdout contains highlights check log path (TYPE-PATH-001, TEST-CLI-001)
    assert "Highlights check log: analysis/check_dense_highlights.log" in stdout, \
        f"Expected success banner to contain Highlights check log path, but stdout was:\n{stdout}"

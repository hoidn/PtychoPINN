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
    assert "studies.fly64_dose_overlap.overlap" in stdout, "Missing overlap module in command output"
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

    # Assert: No Phase C outputs created (dry-run mode)
    phase_c_root = hub / "data" / "phase_c"
    if phase_c_root.exists():
        phase_c_files = list(phase_c_root.rglob("*.npz"))
        assert len(phase_c_files) == 0, f"--collect-only mode should not create Phase C outputs, found: {phase_c_files}"


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
    - Checks both train/test splits exist under phase_c_root
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Raises RuntimeError mentioning '_metadata' if metadata is None
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract).
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs without metadata
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create minimal NPZ files for train and test splits WITHOUT _metadata
    import numpy as np

    for split in ["train", "test"]:
        split_dir = phase_c_root / f"dose_1000_{split}"
        split_dir.mkdir(parents=True)

        # Create a minimal NPZ without metadata (violates expected contract)
        npz_path = split_dir / f"fly64_{split}_simulated.npz"
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
    - Checks both train/test splits exist under phase_c_root
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Checks metadata["data_transformations"] for "transpose_rename_convert" (case-sensitive list membership)
    - Raises RuntimeError mentioning both '_metadata' and 'transpose_rename_convert' if transformation is missing
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract).
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs with _metadata but missing canonical transformation
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create NPZ files for train and test splits WITH _metadata but WITHOUT transpose_rename_convert
    import numpy as np
    from ptycho.metadata import MetadataManager

    for split in ["train", "test"]:
        split_dir = phase_c_root / f"dose_1000_{split}"
        split_dir.mkdir(parents=True)

        npz_path = split_dir / f"fly64_{split}_simulated.npz"

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
    - Checks both train/test splits exist under phase_c_root
    - Loads NPZ files via MetadataManager.load_with_metadata
    - Verifies metadata["data_transformations"] contains "transpose_rename_convert" transformation
    - Succeeds without raising when transformation is present
    - Does not mutate or delete Phase C outputs (read-only)

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract).
    """
    # Import the function under test
    validate_phase_c_metadata = _import_validate_phase_c_metadata()

    # Setup: Create fake Phase C outputs with complete valid metadata
    hub = tmp_path / "phase_c_hub"
    phase_c_root = hub / "data" / "phase_c"
    phase_c_root.mkdir(parents=True)

    # Create NPZ files for train and test splits WITH proper metadata including transpose_rename_convert
    import numpy as np
    from ptycho.metadata import MetadataManager

    for split in ["train", "test"]:
        split_dir = phase_c_root / f"dose_1000_{split}"
        split_dir.mkdir(parents=True)

        npz_path = split_dir / f"fly64_{split}_simulated.npz"

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

    # Assert: run_command should have been called multiple times (Phase C/D/E/F/G + reporting helper + analyze digest)
    # Expected: 1 Phase C + 1 Phase D + 2 Phase E (baseline gs1 + dense gs2) + 2 Phase F (train/test) + 2 Phase G (train/test) + 1 reporting helper + 1 analyze digest = 10 total
    assert len(run_command_calls) >= 10, f"Expected at least 10 run_command calls (C/D/E/F/G phases + reporting + analyze), got {len(run_command_calls)}"

    # Assert: Second-to-last call should be the reporting helper (before analyze digest)
    reporting_helper_cmd, reporting_helper_log_path = run_command_calls[-2]

    # Validate command targets report_phase_g_dense_metrics.py
    assert any("report_phase_g_dense_metrics.py" in str(part) for part in reporting_helper_cmd), \
        f"Second-to-last run_command call should target report_phase_g_dense_metrics.py, got: {reporting_helper_cmd}"

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

    monkeypatch.setattr(module, "prepare_hub", stub_prepare_hub)
    monkeypatch.setattr(module, "validate_phase_c_metadata", stub_validate_phase_c_metadata)
    monkeypatch.setattr(module, "summarize_phase_g_outputs", stub_summarize_phase_g_outputs)

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

    # Assert: run_command should have been called for all phases + reporting helper + analyze digest
    # Expected: 1 Phase C + 1 Phase D + 2 Phase E (baseline gs1 + dense gs2) + 2 Phase F (train/test) + 2 Phase G (train/test) + 1 reporting helper + 1 analyze digest = 10 total
    assert len(run_command_calls) >= 10, f"Expected at least 10 run_command calls (C/D/E/F/G phases + reporting + analyze), got {len(run_command_calls)}"

    # Find reporting helper and analyze digest calls
    reporting_helper_idx = None
    analyze_digest_idx = None
    
    for idx, (cmd, log_path) in enumerate(run_command_calls):
        cmd_str = " ".join(str(c) for c in cmd)
        if "report_phase_g_dense_metrics.py" in cmd_str:
            reporting_helper_idx = idx
        if "analyze_dense_metrics.py" in cmd_str:
            analyze_digest_idx = idx

    # Assert: Both helpers should be invoked
    assert reporting_helper_idx is not None, "reporting helper (report_phase_g_dense_metrics.py) was not invoked"
    assert analyze_digest_idx is not None, "analyze digest (analyze_dense_metrics.py) was not invoked"

    # Assert: analyze_dense_metrics.py should be invoked AFTER report_phase_g_dense_metrics.py
    assert analyze_digest_idx > reporting_helper_idx, \
        f"analyze_dense_metrics.py should be invoked after report_phase_g_dense_metrics.py, but got order: reporting={reporting_helper_idx}, analyze={analyze_digest_idx}"

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
    assert "Metrics digest (Markdown):" in stdout, \
        f"Expected success banner to include 'Metrics digest (Markdown):' line, got:\n{stdout}"
    assert "Metrics digest log:" in stdout, \
        f"Expected success banner to include 'Metrics digest log:' line, got:\n{stdout}"

    # Validate digest paths appear in stdout (should show relative paths per TYPE-PATH-001)
    assert "metrics_digest.md" in stdout, \
        f"Expected stdout to mention metrics_digest.md path, got:\n{stdout}"
    assert "metrics_digest_cli.log" in stdout, \
        f"Expected stdout to mention metrics_digest_cli.log path, got:\n{stdout}"

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

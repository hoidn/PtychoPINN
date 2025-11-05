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

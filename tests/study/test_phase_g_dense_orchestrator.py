"""Tests for Phase G dense orchestrator summary helper."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _import_summarize_phase_g_outputs():
    """Import summarize_phase_g_outputs() from the orchestrator script using spec loader."""
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py"
    spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize_phase_g_outputs


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

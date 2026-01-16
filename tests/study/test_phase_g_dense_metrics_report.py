"""Tests for Phase G dense metrics reporting helper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _import_report_module():
    """Import the report helper module using spec loader."""
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py"
    spec = importlib.util.spec_from_file_location("report_phase_g_dense_metrics", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_phase_g_dense_metrics(tmp_path: Path) -> None:
    """
    Test reporting helper parses metrics_summary.json and emits delta tables.

    Acceptance:
    - Loads report_phase_g_dense_metrics.py via importlib
    - Creates fixture metrics_summary.json with PtychoPINN, Baseline, PtyChi aggregates
    - Invokes helper with --metrics pointing to fixture
    - Validates stdout contains:
      - Aggregate metrics tables for each model (deterministic sorted order)
      - Delta section showing PtychoPINN - Baseline and PtychoPINN - PtyChi
      - 3-decimal formatting for all floats
      - MS-SSIM amplitude/phase and MAE amplitude/phase comparisons
    - Optionally writes Markdown via --output flag
    - Optionally writes highlights via --highlights flag
    - Returns 0 on success

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Create fixture metrics_summary.json with realistic aggregate values
    metrics_file = tmp_path / "metrics_summary.json"
    fixture_data = {
        "n_jobs": 2,
        "n_success": 2,
        "n_failed": 0,
        "jobs": [],  # Not consumed by reporting helper
        "aggregate_metrics": {
            "PtychoPINN": {
                "ms_ssim": {
                    "mean_amplitude": 0.987,
                    "best_amplitude": 0.992,
                    "mean_phase": 0.945,
                    "best_phase": 0.956,
                },
                "mae": {
                    "mean_amplitude": 0.012,
                    "mean_phase": 0.034,
                },
            },
            "Baseline": {
                "ms_ssim": {
                    "mean_amplitude": 0.923,
                    "best_amplitude": 0.931,
                    "mean_phase": 0.889,
                    "best_phase": 0.902,
                },
                "mae": {
                    "mean_amplitude": 0.048,
                    "mean_phase": 0.067,
                },
            },
            "PtyChi": {
                "ms_ssim": {
                    "mean_amplitude": 0.912,
                    "best_amplitude": 0.919,
                    "mean_phase": 0.876,
                    "best_phase": 0.887,
                },
                "mae": {
                    "mean_amplitude": 0.055,
                    "mean_phase": 0.078,
                },
            },
        },
    }

    with metrics_file.open('w') as f:
        json.dump(fixture_data, f, indent=2)

    # Invoke helper script as subprocess (validates CLI interface)
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py"
    md_output = tmp_path / "report.md"
    highlights_output = tmp_path / "highlights.txt"

    result = subprocess.run(
        [sys.executable, str(script_path), "--metrics", str(metrics_file), "--output", str(md_output), "--highlights", str(highlights_output)],
        capture_output=True,
        text=True,
    )

    # Assert success
    assert result.returncode == 0, f"Helper failed: {result.stderr}"

    # Validate stdout contains expected sections
    stdout = result.stdout
    assert "MS-SSIM Sanity Check" in stdout, "Missing MS-SSIM sanity section"
    assert "_Threshold: mean amplitude/phase ≥ 0.80_" in stdout, "Missing threshold note"
    assert "| PtychoPINN | 0.987 | 0.945 | OK |" in stdout, "Missing absolute MS-SSIM row for PtychoPINN"
    assert "| Baseline | 0.923 | 0.889 | OK |" in stdout, "Missing absolute MS-SSIM row for Baseline"
    assert "| PtyChi | 0.912 | 0.876 | OK |" in stdout, "Missing absolute MS-SSIM row for PtyChi"
    assert "Aggregate Metrics" in stdout, "Missing aggregate metrics section in stdout"
    assert "PtychoPINN" in stdout, "Missing PtychoPINN in stdout"
    assert "Baseline" in stdout, "Missing Baseline in stdout"
    assert "PtyChi" in stdout, "Missing PtyChi in stdout"

    # Validate delta section present
    assert "Deltas" in stdout or "Delta" in stdout, "Missing delta section in stdout"

    # Validate 3-decimal formatting (spot check a few values)
    assert "0.987" in stdout, "Missing PtychoPINN MS-SSIM mean amplitude value"
    assert "0.923" in stdout, "Missing Baseline MS-SSIM mean amplitude value"

    # Validate delta computations present (PtychoPINN - Baseline)
    # Expected: 0.987 - 0.923 = 0.064 for MS-SSIM mean amplitude
    # Expected: 0.012 - 0.048 = -0.036 for MAE mean amplitude
    assert "0.064" in stdout, "Missing PtychoPINN vs Baseline MS-SSIM amplitude delta"
    assert "-0.036" in stdout, "Missing PtychoPINN vs Baseline MAE amplitude delta"

    # Validate Markdown file written
    assert md_output.exists(), "Markdown output file not created"
    md_content = md_output.read_text()
    assert "Aggregate Metrics" in md_content, "Missing aggregate metrics in Markdown"
    assert "Deltas" in md_content or "Delta" in md_content, "Missing delta section in Markdown"
    assert "0.987" in md_content, "Missing metric value in Markdown"

    # Validate highlights file written with top-line deltas and absolute block
    assert highlights_output.exists(), "Highlights output file not created"
    highlights_content = highlights_output.read_text()
    assert "MS-SSIM (mean) threshold" in highlights_content, "Highlights missing absolute MS-SSIM snapshot"
    assert "PtychoPINN amplitude/phase: 0.987 / 0.945" in highlights_content, "Highlights missing absolute values"
    # Should contain MS-SSIM and MAE deltas for both amplitude and phase
    assert "MS-SSIM" in highlights_content, "Missing MS-SSIM in highlights"
    assert "MAE" in highlights_content, "Missing MAE in highlights"
    assert "amplitude" in highlights_content.lower(), "Missing amplitude in highlights"
    assert "phase" in highlights_content.lower(), "Missing phase in highlights"
    # Should contain numeric deltas (spot check)
    assert "0.064" in highlights_content, "Missing MS-SSIM amplitude delta in highlights"
    assert "-0.036" in highlights_content, "Missing MAE amplitude delta in highlights"


def test_report_phase_g_dense_metrics_missing_model_fails(tmp_path: Path) -> None:
    """
    Test reporting helper fails gracefully when required model missing.

    Acceptance:
    - Creates metrics_summary.json with only PtychoPINN (missing Baseline/PtyChi)
    - Invokes helper
    - Asserts non-zero exit code
    - Stderr contains actionable error message mentioning missing model
    """
    metrics_file = tmp_path / "metrics_summary.json"
    incomplete_data = {
        "n_jobs": 1,
        "n_success": 1,
        "n_failed": 0,
        "jobs": [],
        "aggregate_metrics": {
            "PtychoPINN": {
                "ms_ssim": {
                    "mean_amplitude": 0.987,
                },
            },
        },
    }

    with metrics_file.open('w') as f:
        json.dump(incomplete_data, f, indent=2)

    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--metrics", str(metrics_file)],
        capture_output=True,
        text=True,
    )

    # Assert failure
    assert result.returncode != 0, "Helper should fail when required models missing"
    assert "Baseline" in result.stderr or "PtyChi" in result.stderr, "Error message should mention missing model"


def test_analyze_dense_metrics_flags_failures(tmp_path: Path) -> None:
    """
    Test analyze script exits with code 1 and emits failure banner when n_failed > 0.

    Acceptance:
    - Creates metrics_summary.json with n_failed > 0
    - Creates valid aggregate_highlights.txt
    - Invokes analyze_dense_metrics.py
    - Asserts exit code == 1 (not 0 or 2)
    - Stderr contains failure banner "⚠️ FAILURES PRESENT ⚠️"
    - Stdout/digest contains failure warning section
    - Digest file written successfully despite failures

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Create fixture with failures
    metrics_file = tmp_path / "metrics_summary.json"
    fixture_data = {
        "n_jobs": 3,
        "n_success": 2,
        "n_failed": 1,  # Critical: non-zero failure count
        "jobs": [],
        "aggregate_metrics": {
            "PtychoPINN": {
                "ms_ssim": {
                    "mean_amplitude": 0.987,
                    "best_amplitude": 0.992,
                    "mean_phase": 0.945,
                    "best_phase": 0.956,
                },
                "mae": {
                    "mean_amplitude": 0.012,
                    "mean_phase": 0.034,
                },
            },
            "Baseline": {
                "ms_ssim": {
                    "mean_amplitude": 0.923,
                    "best_amplitude": 0.931,
                    "mean_phase": 0.889,
                    "best_phase": 0.902,
                },
                "mae": {
                    "mean_amplitude": 0.048,
                    "mean_phase": 0.067,
                },
            },
            "PtyChi": {
                "ms_ssim": {
                    "mean_amplitude": 0.912,
                    "best_amplitude": 0.919,
                    "mean_phase": 0.876,
                    "best_phase": 0.887,
                },
                "mae": {
                    "mean_amplitude": 0.055,
                    "mean_phase": 0.078,
                },
            },
        },
    }

    with metrics_file.open('w') as f:
        json.dump(fixture_data, f, indent=2)

    # Create valid highlights fixture
    highlights_file = tmp_path / "aggregate_highlights.txt"
    highlights_content = """MS-SSIM Amplitude Delta (PtychoPINN - Baseline): +0.064
MS-SSIM Phase Delta (PtychoPINN - Baseline): +0.056
MAE Amplitude Delta (PtychoPINN - Baseline): -0.036
MAE Phase Delta (PtychoPINN - Baseline): -0.033
"""
    highlights_file.write_text(highlights_content)

    # Invoke analyze script
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py"
    output_file = tmp_path / "metrics_digest.md"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--metrics", str(metrics_file),
            "--highlights", str(highlights_file),
            "--output", str(output_file),
        ],
        capture_output=True,
        text=True,
    )

    # Assert exit code is exactly 1 (failures present, not invalid format)
    assert result.returncode == 1, f"Expected exit code 1 for n_failed > 0, got {result.returncode}"

    # Assert failure banner in stderr
    assert "⚠️ FAILURES PRESENT ⚠️" in result.stderr, "Missing failure banner in stderr"
    assert "1 comparison job(s) failed" in result.stderr, "Missing failure count in stderr"

    # Assert stdout contains digest with failure warning
    stdout = result.stdout
    assert "⚠️ FAILURES PRESENT ⚠️" in stdout, "Missing failure warning in stdout digest"
    assert "1 of 3 comparison job(s) failed" in stdout, "Missing job failure summary in digest"

    # Assert digest file written successfully
    assert output_file.exists(), "Digest file should be written even when failures present"
    digest_content = output_file.read_text()
    assert "⚠️ FAILURES PRESENT ⚠️" in digest_content, "Digest file should contain failure banner"
    assert "Failed: 1" in digest_content, "Digest should show failure count in summary"
    assert "MS-SSIM Sanity Check" in digest_content, "Digest should include MS-SSIM sanity table"
    assert "| Baseline | 0.923 | 0.889 | OK |" in digest_content, "Digest missing MS-SSIM row for Baseline"

    # Assert highlights embedded in digest
    assert "MS-SSIM Amplitude Delta" in digest_content, "Digest should embed highlights content"


def test_analyze_dense_metrics_success_digest(tmp_path: Path) -> None:
    """
    Test analyze script exits with code 0 and emits success banner when n_failed == 0.

    Acceptance:
    - Creates metrics_summary.json with n_failed == 0 (all jobs succeeded)
    - Creates valid aggregate_highlights.txt
    - Invokes analyze_dense_metrics.py
    - Asserts exit code == 0 (success)
    - Stdout/digest contains success banner and NO failure warning
    - Digest file written successfully
    - Stderr should NOT contain failure banner

    Follows TYPE-PATH-001 (Path normalization).
    """
    # Create fixture with all successes (n_failed == 0)
    metrics_file = tmp_path / "metrics_summary.json"
    fixture_data = {
        "n_jobs": 2,
        "n_success": 2,
        "n_failed": 0,  # Critical: zero failures
        "jobs": [],
        "aggregate_metrics": {
            "PtychoPINN": {
                "ms_ssim": {
                    "mean_amplitude": 0.987,
                    "best_amplitude": 0.992,
                    "mean_phase": 0.945,
                    "best_phase": 0.956,
                },
                "mae": {
                    "mean_amplitude": 0.012,
                    "mean_phase": 0.034,
                },
            },
            "Baseline": {
                "ms_ssim": {
                    "mean_amplitude": 0.923,
                    "best_amplitude": 0.931,
                    "mean_phase": 0.889,
                    "best_phase": 0.902,
                },
                "mae": {
                    "mean_amplitude": 0.048,
                    "mean_phase": 0.067,
                },
            },
            "PtyChi": {
                "ms_ssim": {
                    "mean_amplitude": 0.912,
                    "best_amplitude": 0.919,
                    "mean_phase": 0.876,
                    "best_phase": 0.887,
                },
                "mae": {
                    "mean_amplitude": 0.055,
                    "mean_phase": 0.078,
                },
            },
        },
    }

    with metrics_file.open('w') as f:
        json.dump(fixture_data, f, indent=2)

    # Create valid highlights fixture
    highlights_file = tmp_path / "aggregate_highlights.txt"
    highlights_content = """MS-SSIM Amplitude Delta (PtychoPINN - Baseline): +0.064
MS-SSIM Phase Delta (PtychoPINN - Baseline): +0.056
MAE Amplitude Delta (PtychoPINN - Baseline): -0.036
MAE Phase Delta (PtychoPINN - Baseline): -0.033
"""
    highlights_file.write_text(highlights_content)

    # Invoke analyze script
    script_path = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py"
    output_file = tmp_path / "metrics_digest.md"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--metrics", str(metrics_file),
            "--highlights", str(highlights_file),
            "--output", str(output_file),
        ],
        capture_output=True,
        text=True,
    )

    # Assert exit code is exactly 0 (success)
    assert result.returncode == 0, f"Expected exit code 0 for n_failed == 0, got {result.returncode}. Stderr: {result.stderr}"

    # Assert NO failure banner in stderr
    assert "⚠️ FAILURES PRESENT ⚠️" not in result.stderr, "Should not show failure banner in stderr when all jobs succeed"

    # Assert stdout contains digest WITHOUT failure warning
    stdout = result.stdout
    assert "⚠️ FAILURES PRESENT ⚠️" not in stdout, "Should not show failure warning in stdout digest when all jobs succeed"
    assert "MS-SSIM Sanity Check" in stdout, "Success digest should include MS-SSIM sanity section"
    assert "| Baseline | 0.923 | 0.889 | OK |" in stdout, "Success digest missing MS-SSIM row for Baseline"

    # Assert explicit success banner present (required per input.md)
    assert "✓ ALL COMPARISONS SUCCESSFUL ✓" in stdout, "Should show explicit success banner in stdout when all jobs succeed"

    # Assert success indicators present
    assert "Successful: 2" in stdout, "Should show successful count in digest"
    assert "Failed: 0" in stdout, "Should show zero failures in digest"

    # Assert digest file written successfully
    assert output_file.exists(), "Digest file should be written for successful runs"
    digest_content = output_file.read_text()

    # Assert NO failure banner in digest content
    assert "⚠️ FAILURES PRESENT ⚠️" not in digest_content, "Digest file should NOT contain failure banner when all jobs succeed"

    # Assert explicit success banner in digest content
    assert "✓ ALL COMPARISONS SUCCESSFUL ✓" in digest_content, "Digest file should contain success banner when all jobs succeed"
    assert "Failed: 0" in digest_content, "Digest should show zero failures in summary"
    assert "MS-SSIM Sanity Check" in digest_content, "Digest file should include MS-SSIM sanity section"
    assert "| Baseline | 0.923 | 0.889 | OK |" in digest_content, "Digest file missing MS-SSIM row for Baseline"

    # Assert highlights embedded in digest
    assert "MS-SSIM Amplitude Delta" in digest_content, "Digest should embed highlights content"

    # Assert key deltas section present
    assert "Key Deltas" in digest_content, "Digest should contain Key Deltas section"
    assert "PtychoPINN - Baseline" in digest_content, "Digest should show baseline comparison"

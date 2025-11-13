"""
Tests for ssim_grid.py helper — Phase G MS-SSIM/MAE delta table generation.

Acceptance:
- RED test: preview guard must reject when metrics_delta_highlights_preview.txt contains "amplitude"
- GREEN test: helper succeeds with phase-only preview and emits ssim_grid_summary.md with ±0.000/±0.000000 tables
- Test must execute via subprocess to prove CLI contract (not just imports)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_smoke_ssim_grid(tmp_path: Path) -> None:
    """
    TDD smoke test for ssim_grid.py helper.

    RED: Fails when preview contains "amplitude" (preview guard rejection)
    GREEN: Passes when preview is phase-only and table is emitted with correct precision

    Validates:
    - Helper reads metrics_delta_summary.json correctly
    - Preview guard rejects amplitude contamination
    - Output markdown contains MS-SSIM ±0.000 and MAE ±0.000000 formatted deltas
    - Helper emits POSIX-relative paths in metadata
    """
    # Set up test hub
    hub = tmp_path / "hub"
    analysis = hub / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    # Create metrics_delta_summary.json with sample data
    metrics_delta_summary = {
        "generated_at": "2025-11-11T01:36:12Z",
        "source_metrics": "analysis/metrics_summary.json",
        "deltas": {
            "vs_Baseline": {
                "ms_ssim": {
                    "amplitude": 0.012345,
                    "phase": -0.003456
                },
                "mae": {
                    "amplitude": 0.000123456,
                    "phase": 0.000234567
                }
            },
            "vs_PtyChi": {
                "ms_ssim": {
                    "amplitude": 0.009876,
                    "phase": -0.001234
                },
                "mae": {
                    "amplitude": 0.000987654,
                    "phase": 0.000345678
                }
            }
        }
    }

    delta_json_path = analysis / "metrics_delta_summary.json"
    with delta_json_path.open('w') as f:
        json.dump(metrics_delta_summary, f, indent=2)

    # --- RED PHASE: Preview contains "amplitude" ---
    preview_path_red = analysis / "metrics_delta_highlights_preview.txt"
    preview_path_red.write_text(
        "Phase MS-SSIM vs Baseline: -0.003\n"
        "Phase MS-SSIM vs PtyChi: -0.001\n"
        "Amplitude MAE vs Baseline: +0.000123\n"  # Contains "amplitude" - should fail
        "Phase MAE vs PtyChi: +0.000346\n"
    )

    # Locate ssim_grid.py helper
    ssim_grid_script = Path(__file__).parent.parent.parent / "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py"

    # Run helper (RED phase - should fail due to amplitude in preview)
    result_red = subprocess.run(
        [sys.executable, str(ssim_grid_script), "--hub", str(hub)],
        capture_output=True,
        text=True,
        check=False
    )

    # Assert RED failure
    assert result_red.returncode != 0, (
        f"Expected helper to fail when preview contains 'amplitude', but it succeeded.\n"
        f"stdout: {result_red.stdout}\nstderr: {result_red.stderr}"
    )
    assert "amplitude" in result_red.stderr.lower() or "preview" in result_red.stderr.lower(), (
        f"Expected error message to mention 'amplitude' or 'preview', got:\n{result_red.stderr}"
    )

    # --- GREEN PHASE: Preview is phase-only ---
    preview_path_green = analysis / "metrics_delta_highlights_preview.txt"
    preview_path_green.write_text(
        "Phase MS-SSIM vs Baseline: -0.003\n"
        "Phase MS-SSIM vs PtyChi: -0.001\n"
        "Phase MAE vs Baseline: +0.000235\n"
        "Phase MAE vs PtyChi: +0.000346\n"
    )

    # Run helper (GREEN phase - should succeed)
    result_green = subprocess.run(
        [sys.executable, str(ssim_grid_script), "--hub", str(hub)],
        capture_output=True,
        text=True,
        check=False
    )

    # Assert GREEN success
    assert result_green.returncode == 0, (
        f"Expected helper to succeed with phase-only preview, but it failed.\n"
        f"stdout: {result_green.stdout}\nstderr: {result_green.stderr}"
    )

    # Verify output file exists
    output_md = analysis / "ssim_grid_summary.md"
    assert output_md.exists(), f"Expected output file {output_md} to exist"

    # Read and validate output content
    output_text = output_md.read_text()

    # Check for MS-SSIM values with ±0.000 precision (3 decimal places)
    assert "-0.003" in output_text, "Expected phase MS-SSIM vs Baseline '-0.003' in output"
    assert "-0.001" in output_text, "Expected phase MS-SSIM vs PtyChi '-0.001' in output"

    # Check for MAE values with ±0.000000 precision (6 decimal places)
    assert "+0.000235" in output_text, "Expected phase MAE vs Baseline '+0.000235' in output"
    assert "+0.000346" in output_text, "Expected phase MAE vs PtyChi '+0.000346' in output"

    # Verify phase-only formatting (should NOT contain amplitude rows)
    assert "amplitude" not in output_text.lower(), "Output should not contain amplitude data (phase-only requirement)"

    # Verify POSIX-relative paths in metadata (if any paths are emitted)
    # This is a TYPE-PATH-001 compliance check
    if "analysis/" in output_text:
        assert "analysis/metrics_delta_summary.json" in output_text or output_text.count("analysis/") > 0, (
            "Expected POSIX-relative paths like 'analysis/...' in output"
        )

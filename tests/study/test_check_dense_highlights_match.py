#!/usr/bin/env python3
"""
Tests for check_dense_highlights_match.py highlights checker (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001)

Validates that the checker correctly enforces:
- PREVIEW-PHASE-001: Preview metadata contains phase-only indicator
- STUDY-001: MS-SSIM ±0.000 / MAE ±0.000000 precision in ssim_grid_summary.md table
- Highlights/preview text values match metrics_delta_summary.json

Coverage:
- test_summary_mismatch_fails: RED test with tampered ssim_grid table value
- test_summary_matches_json: GREEN test with aligned artifacts
"""

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


def create_minimal_hub(
    hub_dir: Path,
    ms_ssim_baseline: float = -0.003,
    ms_ssim_ptychi: float = -0.001,
    mae_baseline: float = 0.000235,
    mae_ptychi: float = 0.000346,
    tamper_ssim_grid: bool = False,
    remove_phase_only: bool = False
) -> None:
    """
    Create minimal hub with analysis artifacts for checker validation.

    Args:
        hub_dir: Hub root directory
        ms_ssim_baseline: MS-SSIM phase delta for vs_Baseline
        ms_ssim_ptychi: MS-SSIM phase delta for vs_PtyChi
        mae_baseline: MAE phase delta for vs_Baseline
        mae_ptychi: MAE phase delta for vs_PtyChi
        tamper_ssim_grid: If True, deliberately mismatch ssim_grid table values
        remove_phase_only: If True, remove phase-only metadata from ssim_grid
    """
    analysis_dir = hub_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. metrics_delta_summary.json
    summary = {
        "deltas": {
            "vs_Baseline": {
                "ms_ssim": {"phase": ms_ssim_baseline, "amplitude": -0.005},
                "mae": {"phase": mae_baseline, "amplitude": 0.000400}
            },
            "vs_PtyChi": {
                "ms_ssim": {"phase": ms_ssim_ptychi, "amplitude": -0.002},
                "mae": {"phase": mae_ptychi, "amplitude": 0.000500}
            }
        }
    }
    (analysis_dir / "metrics_delta_summary.json").write_text(json.dumps(summary, indent=2))

    # 2. metrics_delta_highlights.txt
    highlights = dedent(f"""\
        Phase MS-SSIM Δ vs_Baseline: {ms_ssim_baseline:+.3f}
        Phase MS-SSIM Δ vs_PtyChi: {ms_ssim_ptychi:+.3f}
        Phase MAE Δ vs_Baseline: {mae_baseline:+.6f}
        Phase MAE Δ vs_PtyChi: {mae_ptychi:+.6f}
    """)
    (analysis_dir / "metrics_delta_highlights.txt").write_text(highlights)

    # 3. metrics_delta_highlights_preview.txt
    preview = dedent(f"""\
        Preview (phase-only):
        MS-SSIM Δ vs_Baseline: {ms_ssim_baseline:+.3f}
        MS-SSIM Δ vs_PtyChi: {ms_ssim_ptychi:+.3f}
        MAE Δ vs_Baseline: {mae_baseline:+.6f}
        MAE Δ vs_PtyChi: {mae_ptychi:+.6f}
    """)
    (analysis_dir / "metrics_delta_highlights_preview.txt").write_text(preview)

    # 4. ssim_grid_summary.md
    # If tampered, use wrong values; otherwise use correct values
    if tamper_ssim_grid:
        grid_ms_ssim_baseline = ms_ssim_baseline + 0.010  # Deliberate mismatch
        grid_mae_ptychi = mae_ptychi + 0.000100  # Deliberate mismatch
    else:
        grid_ms_ssim_baseline = ms_ssim_baseline
        grid_mae_ptychi = mae_ptychi

    preview_metadata = "" if remove_phase_only else "**Preview validated:** `analysis/metrics_delta_highlights_preview.txt` (phase-only)\n\n"

    ssim_grid = dedent(f"""\
        # SSIM Grid Summary (Phase-Only)

        **Generated from:** `analysis/metrics_delta_summary.json`
        {preview_metadata}## MS-SSIM Deltas (Phase)

        | Comparison | Phase MS-SSIM Delta |
        |------------|---------------------|
        | vs_Baseline | {grid_ms_ssim_baseline:+.3f} |
        | vs_PtyChi   | {ms_ssim_ptychi:+.3f} |

        ## MAE Deltas (Phase)

        | Comparison | Phase MAE Delta |
        |------------|-----------------|
        | vs_Baseline | {mae_baseline:+.6f} |
        | vs_PtyChi   | {grid_mae_ptychi:+.6f} |

        ---

        _Note: This summary enforces PREVIEW-PHASE-001 (phase-only formatting)._
        _MS-SSIM formatted to ±0.000 (3 decimals), MAE to ±0.000000 (6 decimals) per STUDY-001._
    """)
    (analysis_dir / "ssim_grid_summary.md").write_text(ssim_grid)


@pytest.mark.study
@pytest.mark.mini
def test_summary_mismatch_fails(tmp_path: Path) -> None:
    """
    RED test: Checker fails when ssim_grid table values mismatch JSON.

    Validates STUDY-001 precision enforcement by tampering with ssim_grid_summary.md table.
    """
    hub = tmp_path / "hub_red"
    create_minimal_hub(hub, tamper_ssim_grid=True)

    checker_script = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py")
    assert checker_script.exists(), f"Checker script not found: {checker_script}"

    result = subprocess.run(
        [sys.executable, str(checker_script), "--hub", str(hub)],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "Checker should fail with tampered ssim_grid values"
    assert "mismatch" in result.stderr.lower(), f"Expected mismatch error, got: {result.stderr}"


@pytest.mark.study
@pytest.mark.mini
def test_summary_matches_json(tmp_path: Path) -> None:
    """
    GREEN test: Checker passes when all artifacts are aligned.

    Validates that checker correctly accepts well-formed artifacts matching STUDY-001 and PREVIEW-PHASE-001.
    """
    hub = tmp_path / "hub_green"
    create_minimal_hub(hub, tamper_ssim_grid=False)

    checker_script = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py")
    assert checker_script.exists(), f"Checker script not found: {checker_script}"

    result = subprocess.run(
        [sys.executable, str(checker_script), "--hub", str(hub)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Checker should pass with aligned artifacts, got stderr: {result.stderr}"
    assert "Highlights match" in result.stdout, f"Expected success message, got: {result.stdout}"
    assert "SSIM grid summary table values match" in result.stdout
    assert "Preview metadata confirms phase-only" in result.stdout


@pytest.mark.study
@pytest.mark.mini
def test_missing_phase_only_metadata_fails(tmp_path: Path) -> None:
    """
    RED test: Checker fails when ssim_grid_summary.md is missing phase-only metadata.

    Validates PREVIEW-PHASE-001 enforcement.
    """
    hub = tmp_path / "hub_no_phase_only"
    create_minimal_hub(hub, remove_phase_only=True)

    checker_script = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py")
    assert checker_script.exists(), f"Checker script not found: {checker_script}"

    result = subprocess.run(
        [sys.executable, str(checker_script), "--hub", str(hub)],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "Checker should fail when phase-only metadata is missing"
    assert "preview metadata validation failed" in result.stderr.lower(), f"Expected preview metadata error, got: {result.stderr}"

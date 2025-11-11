#!/usr/bin/env python3
"""
Verify dense Phase G highlights text matches metrics_delta_summary.json (initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001, owner: galph)
Inputs: --hub <report hub path>    Data deps: metrics_delta_summary.json, metrics_delta_highlights.txt, metrics_delta_highlights_preview.txt, ssim_grid_summary.md under <hub>/analysis/
Outputs: stdout summary, optional log via shell redirection under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/phase_g_dense_full_execution_real_run/analysis/
Repro: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub <path>

Acceptance:
- PREVIEW-PHASE-001: Enforces preview metadata contains "phase-only: ✓" indicator
- STUDY-001: Validates MS-SSIM ±0.000 / MAE ±0.000000 precision in ssim_grid_summary.md table
"""

import argparse
import json
import re
from pathlib import Path

# Expected models and metrics match the verifier schema
EXPECTED_MODELS = ("vs_Baseline", "vs_PtyChi")
EXPECTED_METRICS = ("ms_ssim", "mae")


def format_delta(value: float, metric_type: str) -> str:
    """Format delta value with metric-specific precision (MS-SSIM: 3, MAE: 6)."""
    precision = 3 if "ms_ssim" in metric_type.lower() else 6
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{precision}f}"


def parse_ssim_grid_table(summary_md: str, section_header: str, model_row_label: str) -> str | None:
    """
    Extract a delta value from the ssim_grid_summary.md Markdown table.

    Args:
        summary_md: Full ssim_grid_summary.md content
        section_header: Section heading (e.g., "## MS-SSIM Deltas (Phase)")
        model_row_label: Model label (e.g., "vs_Baseline")

    Returns:
        The formatted delta string (e.g., "-0.003" or "+0.000235") or None if not found
    """
    # Find the section (allow leading whitespace)
    section_match = re.search(rf'^\s*{re.escape(section_header)}\s*$', summary_md, re.MULTILINE)
    if not section_match:
        return None

    # Extract lines from section header to next section or end
    start = section_match.end()
    next_section = re.search(r'^\s*##\s', summary_md[start:], re.MULTILINE)
    section_text = summary_md[start:start + next_section.start()] if next_section else summary_md[start:]

    # Find the table row for the model (allow leading whitespace before |)
    # Expected format: | vs_Baseline | -0.003 |
    row_pattern = rf'^\s*\|\s*{re.escape(model_row_label)}\s*\|\s*([+-]?\d+\.\d+)\s*\|'
    row_match = re.search(row_pattern, section_text, re.MULTILINE)
    if not row_match:
        return None

    return row_match.group(1)


def validate_preview_metadata(summary_md: str) -> tuple[bool, str]:
    """
    Validate preview metadata indicates phase-only content (PREVIEW-PHASE-001).

    Args:
        summary_md: Full ssim_grid_summary.md content

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Look for preview metadata line with phase-only indicator
    # Expected format: **Preview validated:** `analysis/metrics_delta_highlights_preview.txt` (phase-only)
    if "phase-only" not in summary_md.lower():
        return False, "Missing 'phase-only' indicator in preview metadata"

    # Check for explicit phase-only marker (✓ or similar)
    preview_line_pattern = r'\*\*Preview validated:\*\*.*\(phase-only[^\)]*\)'
    if not re.search(preview_line_pattern, summary_md, re.IGNORECASE):
        return False, "Preview metadata does not contain proper '(phase-only)' marker"

    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dense Phase G highlight text against summary JSON")
    parser.add_argument("--hub", required=True, help="Report hub directory containing analysis artifacts")
    args = parser.parse_args()

    hub = Path(args.hub).resolve()
    analysis_dir = hub / "analysis"
    summary_path = analysis_dir / "metrics_delta_summary.json"
    highlights_path = analysis_dir / "metrics_delta_highlights.txt"
    preview_path = analysis_dir / "metrics_delta_highlights_preview.txt"
    ssim_grid_path = analysis_dir / "ssim_grid_summary.md"

    for path in (summary_path, highlights_path, preview_path, ssim_grid_path):
        if not path.exists():
            raise SystemExit(f"Missing artifact: {path.relative_to(hub)}")

    summary = json.loads(summary_path.read_text())
    highlights_text = highlights_path.read_text().strip()
    preview_text = preview_path.read_text().strip()
    ssim_grid_md = ssim_grid_path.read_text()

    # Extract deltas from JSON (matching run_phase_g_dense.py structure)
    deltas = summary.get('deltas', {})
    if not deltas:
        raise SystemExit("No deltas found in metrics_delta_summary.json")

    failures = []

    # 1. Validate preview metadata in ssim_grid_summary.md (PREVIEW-PHASE-001)
    preview_valid, preview_error = validate_preview_metadata(ssim_grid_md)
    if not preview_valid:
        failures.append(f"SSIM grid preview metadata validation failed: {preview_error}")

    # 2. Check highlights/preview text matches JSON
    for model in EXPECTED_MODELS:
        if model not in deltas:
            failures.append(f"Model '{model}' not present in summary JSON deltas")
            continue
        model_deltas = deltas[model]
        for metric in EXPECTED_METRICS:
            if metric not in model_deltas:
                failures.append(f"Metric '{metric}' missing for model '{model}'")
                continue
            # Use phase delta for highlights (per STUDY-001)
            phase_value = model_deltas[metric].get('phase')
            if phase_value is None:
                failures.append(f"Phase value missing for {model}.{metric}")
                continue
            expected = format_delta(phase_value, metric)
            for label, block in (("highlights", highlights_text), ("preview", preview_text)):
                if expected not in block:
                    failures.append(f"{label} text missing {model}.{metric} phase value {expected}")

    # 3. Validate ssim_grid_summary.md table values match JSON (STUDY-001)
    for model in EXPECTED_MODELS:
        if model not in deltas:
            continue
        model_deltas = deltas[model]

        # Check MS-SSIM table
        if "ms_ssim" in model_deltas:
            phase_value = model_deltas["ms_ssim"].get("phase")
            if phase_value is not None:
                expected = format_delta(phase_value, "ms_ssim")
                actual = parse_ssim_grid_table(ssim_grid_md, "## MS-SSIM Deltas (Phase)", model)
                if actual is None:
                    failures.append(f"SSIM grid missing MS-SSIM row for {model}")
                elif actual != expected:
                    failures.append(f"SSIM grid MS-SSIM mismatch for {model}: expected {expected}, got {actual}")

        # Check MAE table
        if "mae" in model_deltas:
            phase_value = model_deltas["mae"].get("phase")
            if phase_value is not None:
                expected = format_delta(phase_value, "mae")
                actual = parse_ssim_grid_table(ssim_grid_md, "## MAE Deltas (Phase)", model)
                if actual is None:
                    failures.append(f"SSIM grid missing MAE row for {model}")
                elif actual != expected:
                    failures.append(f"SSIM grid MAE mismatch for {model}: expected {expected}, got {actual}")

    if failures:
        raise SystemExit("\n".join(failures))

    print(f"✓ Highlights match metrics_delta_summary.json for all tracked fields")
    print(f"✓ SSIM grid summary table values match JSON (MS-SSIM ±0.000, MAE ±0.000000)")
    print(f"✓ Preview metadata confirms phase-only content (PREVIEW-PHASE-001)")


if __name__ == "__main__":
    main()

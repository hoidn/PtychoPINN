#!/usr/bin/env python3
"""
ssim_grid.py — Tier-2 CLI for Phase G MS-SSIM/MAE delta table generation

Loads metrics_delta_summary.json, enforces preview-phase-only content,
and writes ssim_grid_summary.md with MS-SSIM ±0.000 and MAE ±0.000000 tables.

Usage:
    python ssim_grid.py --hub <hub_dir> [--output <output_path>]

Arguments:
    --hub: Hub directory containing analysis/metrics_delta_summary.json
    --output: Optional output path (defaults to <hub>/analysis/ssim_grid_summary.md)

Acceptance:
- PREVIEW-PHASE-001: Reject preview files containing "amplitude" keyword
- STUDY-001: Format MS-SSIM with ±0.000 (3 decimals), MAE with ±0.000000 (6 decimals)
- TYPE-PATH-001: Emit POSIX-relative paths in metadata

Exit codes:
    0: Success
    1: Preview guard failure (amplitude contamination)
    2: Missing or invalid input files
    3: Other errors
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_preview_phase_only(preview_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that preview file contains only phase-related content.

    Args:
        preview_path: Path to metrics_delta_highlights_preview.txt

    Returns:
        Tuple of (is_valid, error_lines)
        - is_valid: True if no amplitude contamination
        - error_lines: List of lines containing "amplitude" (empty if valid)
    """
    if not preview_path.exists():
        return False, [f"Preview file not found: {preview_path}"]

    try:
        content = preview_path.read_text(encoding='utf-8')
    except Exception as e:
        return False, [f"Failed to read preview file: {e}"]

    error_lines = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        if "amplitude" in line.lower():
            error_lines.append(f"Line {line_num}: {line.strip()}")

    is_valid = len(error_lines) == 0
    return is_valid, error_lines


def format_delta(value: float | None, precision: int) -> str:
    """
    Format delta value with explicit sign and fixed precision.

    Args:
        value: Numeric delta value (can be None)
        precision: Number of decimal places

    Returns:
        Formatted string with ± sign (e.g., "+0.123" or "-0.456")
        Returns "N/A" if value is None
    """
    if value is None:
        return "N/A"

    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{precision}f}"


def generate_ssim_grid_summary(
    delta_json_path: Path,
    preview_path: Path,
    output_path: Path
) -> None:
    """
    Generate ssim_grid_summary.md from metrics_delta_summary.json.

    Args:
        delta_json_path: Path to metrics_delta_summary.json
        preview_path: Path to metrics_delta_highlights_preview.txt
        output_path: Path to write ssim_grid_summary.md

    Raises:
        SystemExit: On validation failure or missing files
    """
    # Validate preview is phase-only
    is_valid, error_lines = validate_preview_phase_only(preview_path)
    if not is_valid:
        print("ERROR: Preview guard failure (PREVIEW-PHASE-001)", file=sys.stderr)
        print("Preview file contains 'amplitude' keyword (violates phase-only requirement):", file=sys.stderr)
        for error_line in error_lines:
            print(f"  {error_line}", file=sys.stderr)
        sys.exit(1)

    # Load metrics_delta_summary.json
    if not delta_json_path.exists():
        print(f"ERROR: Metrics delta summary not found: {delta_json_path}", file=sys.stderr)
        sys.exit(2)

    try:
        with delta_json_path.open('r') as f:
            delta_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {delta_json_path}: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: Failed to read {delta_json_path}: {e}", file=sys.stderr)
        sys.exit(3)

    # Validate required structure
    if 'deltas' not in delta_data:
        print("ERROR: Missing 'deltas' field in metrics_delta_summary.json", file=sys.stderr)
        sys.exit(2)

    deltas = delta_data['deltas']
    required_comparisons = ['vs_Baseline', 'vs_PtyChi']
    for comp in required_comparisons:
        if comp not in deltas:
            print(f"ERROR: Missing comparison '{comp}' in deltas", file=sys.stderr)
            sys.exit(2)

    # Extract phase-only deltas
    baseline_msssim_phase = deltas['vs_Baseline'].get('ms_ssim', {}).get('phase')
    baseline_mae_phase = deltas['vs_Baseline'].get('mae', {}).get('phase')
    ptychi_msssim_phase = deltas['vs_PtyChi'].get('ms_ssim', {}).get('phase')
    ptychi_mae_phase = deltas['vs_PtyChi'].get('mae', {}).get('phase')

    # Generate markdown table
    markdown_lines = [
        "# SSIM Grid Summary (Phase-Only)",
        "",
        "**Generated from:** `analysis/metrics_delta_summary.json`",
        f"**Preview validated:** `analysis/metrics_delta_highlights_preview.txt` (phase-only)",
        "",
        "## MS-SSIM Deltas (Phase)",
        "",
        "| Comparison | Phase MS-SSIM Delta |",
        "|------------|---------------------|",
        f"| vs_Baseline | {format_delta(baseline_msssim_phase, 3)} |",
        f"| vs_PtyChi   | {format_delta(ptychi_msssim_phase, 3)} |",
        "",
        "## MAE Deltas (Phase)",
        "",
        "| Comparison | Phase MAE Delta |",
        "|------------|-----------------|",
        f"| vs_Baseline | {format_delta(baseline_mae_phase, 6)} |",
        f"| vs_PtyChi   | {format_delta(ptychi_mae_phase, 6)} |",
        "",
        "---",
        "",
        "_Note: This summary enforces PREVIEW-PHASE-001 (phase-only formatting)._",
        "_MS-SSIM formatted to ±0.000 (3 decimals), MAE to ±0.000000 (6 decimals) per STUDY-001._",
    ]

    # Write output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(markdown_lines) + "\n", encoding='utf-8')
        print(f"SUCCESS: Generated {output_path}")
        print(f"  Preview guard: PASSED (phase-only)")
        print(f"  MS-SSIM precision: ±0.000 (3 decimals)")
        print(f"  MAE precision: ±0.000000 (6 decimals)")
    except Exception as e:
        print(f"ERROR: Failed to write output to {output_path}: {e}", file=sys.stderr)
        sys.exit(3)


def main() -> None:
    """
    Main entry point for ssim_grid.py helper.
    """
    parser = argparse.ArgumentParser(
        description="Generate Phase G MS-SSIM/MAE delta summary table (phase-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--hub",
        type=Path,
        required=True,
        help="Hub directory containing analysis/metrics_delta_summary.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Output path for ssim_grid_summary.md (default: <hub>/analysis/ssim_grid_summary.md)"
    )

    args = parser.parse_args()

    # Resolve paths
    hub = args.hub.resolve()
    analysis = hub / "analysis"
    delta_json_path = analysis / "metrics_delta_summary.json"
    preview_path = analysis / "metrics_delta_highlights_preview.txt"
    output_path = args.output.resolve() if args.output else analysis / "ssim_grid_summary.md"

    # Generate summary
    generate_ssim_grid_summary(delta_json_path, preview_path, output_path)


if __name__ == "__main__":
    main()

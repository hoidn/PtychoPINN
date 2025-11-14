#!/usr/bin/env python3
"""Phase C2 PyTorch-only forward parity statistics comparison.

Ingests two hub stats JSON files (baseline vs candidate) and prints ratios/differences
for patch/canvas means, stds, and var_zero_mean to quantify variance collapse or other
scaling divergences.

POLICY-001 (PyTorch mandatory): This tool operates PyTorch-only during Phase C pending
TensorFlow translation layer fixes (see TF blocker files in the Reports Hub).

CONFIG-001 (shared config bridge): Both backends should produce equivalent stats via
the unified config/data contracts, but TensorFlow XLA/stitching blockers force a
PyTorch-only comparison for Phase C2.

Usage:
    export HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"
    python plans/active/FIX-PYTORCH-FORWARD-PARITY-001/bin/phase_c2_compare_stats.py \\
        --baseline-stats "$HUB/scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json" \\
        --candidate-stats "$HUB/scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1/stats.json" \\
        --label-baseline "Phase B3 (gridsize=2)" \\
        --label-candidate "Phase C1 GS1 fallback (PyTorch-only)" \\
        --out "$HUB/analysis/phase_c2_pytorch_only_metrics.txt"

Author: Ralph (2025-11-14)
Initiative: FIX-PYTORCH-FORWARD-PARITY-001
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_stats(path: Path) -> Dict[str, Any]:
    """Load stats JSON with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    with path.open('r') as f:
        return json.load(f)


def compute_metrics_delta(baseline: Dict[str, Any], candidate: Dict[str, Any],
                          baseline_label: str, candidate_label: str) -> Dict[str, Any]:
    """Compute ratios and differences between baseline and candidate stats.

    Schema (from docs/workflows/pytorch.md and specs/ptycho-workflow.md):
        {
          "patch_amplitude": {"mean": float, "std": float, "var_zero_mean": float},
          "canvas_amplitude": {"mean": float, "std": float, "var_zero_mean": float}
        }

    Returns:
        Dict with ratios, differences, and status flags.
    """
    results = {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "patch_amplitude": {},
        "canvas_amplitude": {},
        "status": "ok"
    }

    for section in ["patch_amplitude", "canvas_amplitude"]:
        if section not in baseline or section not in candidate:
            results["status"] = f"missing_{section}"
            continue

        baseline_data = baseline[section]
        candidate_data = candidate[section]

        for metric in ["mean", "std", "var_zero_mean"]:
            if metric not in baseline_data or metric not in candidate_data:
                results[section][metric] = {"status": f"missing_{metric}"}
                continue

            b_val = baseline_data[metric]
            c_val = candidate_data[metric]
            delta = c_val - b_val

            # Compute ratio with zero-baseline handling
            if b_val == 0.0:
                if c_val == 0.0:
                    ratio = 1.0
                    ratio_status = "both_zero"
                else:
                    ratio = float('nan')
                    ratio_status = "blocked_baseline_zero"
                    results["status"] = "blocked_baseline_zero"
            else:
                ratio = c_val / b_val
                ratio_status = "ok"

            results[section][metric] = {
                "baseline": b_val,
                "candidate": c_val,
                "delta": delta,
                "ratio": ratio,
                "status": ratio_status
            }

    return results


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics as human-readable text report."""
    lines = [
        "# Phase C2 PyTorch-only Forward Parity Metrics Comparison",
        "",
        "## Configuration",
        f"Baseline: {metrics['baseline_label']}",
        f"Candidate: {metrics['candidate_label']}",
        f"Overall Status: {metrics['status']}",
        "",
        "## Policy Citations",
        "POLICY-001: PyTorch (torch>=2.2) is mandatory; TensorFlow comparison blocked pending translation layer fixes.",
        "CONFIG-001: Both backends should bridge via update_legacy_dict(params.cfg, config) for equivalent stats.",
        "",
    ]

    for section in ["patch_amplitude", "canvas_amplitude"]:
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append("")

        if section not in metrics:
            lines.append(f"  Status: missing_{section}")
            lines.append("")
            continue

        section_data = metrics[section]

        for metric in ["mean", "std", "var_zero_mean"]:
            if metric not in section_data:
                continue

            data = section_data[metric]

            if "status" in data and data["status"] != "ok":
                lines.append(f"  {metric}: Status={data['status']}")
                continue

            lines.extend([
                f"  {metric}:",
                f"    Baseline:  {data['baseline']:.6e}",
                f"    Candidate: {data['candidate']:.6e}",
                f"    Delta:     {data['delta']:+.6e}",
                f"    Ratio:     {data['ratio']:.6f} ({data['status']})",
            ])

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch forward parity statistics between baseline and candidate runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--baseline-stats",
        type=Path,
        required=True,
        help="Path to baseline stats.json (e.g., Phase B3)"
    )

    parser.add_argument(
        "--candidate-stats",
        type=Path,
        required=True,
        help="Path to candidate stats.json (e.g., Phase C1 GS1)"
    )

    parser.add_argument(
        "--label-baseline",
        type=str,
        default="Baseline",
        help="Label for baseline run (default: 'Baseline')"
    )

    parser.add_argument(
        "--label-candidate",
        type=str,
        default="Candidate",
        help="Label for candidate run (default: 'Candidate')"
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for metrics text file"
    )

    args = parser.parse_args()

    # Load stats
    try:
        baseline_stats = load_stats(args.baseline_stats)
        candidate_stats = load_stats(args.candidate_stats)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Compute metrics
    metrics = compute_metrics_delta(
        baseline_stats,
        candidate_stats,
        args.label_baseline,
        args.label_candidate
    )

    # Format and write report
    report = format_metrics_report(metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)

    # Print to stdout for tee capture
    print(report)

    # Exit with status
    if metrics["status"] == "ok":
        print(f"\nSUCCESS: Metrics written to {args.out}", file=sys.stderr)
        return 0
    else:
        print(f"\nWARNING: Comparison completed with status={metrics['status']}", file=sys.stderr)
        print(f"Metrics written to {args.out}", file=sys.stderr)
        return 0  # Still write metrics, just warn about issues


if __name__ == "__main__":
    sys.exit(main())

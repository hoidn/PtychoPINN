#!/usr/bin/env python3
"""
Verify dense Phase G highlights text matches metrics_delta_summary.json (initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001, owner: galph)
Inputs: --hub <report hub path>    Data deps: metrics_delta_summary.json, metrics_delta_highlights.txt, metrics_delta_highlights_preview.txt under <hub>/analysis/
Outputs: stdout summary, optional log via shell redirection under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/phase_g_dense_full_execution_real_run/analysis/
Repro: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub <path>
"""

import argparse
import json
from pathlib import Path

EXPECTED_MODELS = ("baseline", "ptychi")
EXPECTED_FIELDS = ("ms_ssim_phase", "mae_phase")


def format_delta(value: float, precision: int) -> str:
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    return f"{sign}{magnitude:.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dense Phase G highlight text against summary JSON")
    parser.add_argument("--hub", required=True, help="Report hub directory containing analysis artifacts")
    args = parser.parse_args()

    hub = Path(args.hub).resolve()
    analysis_dir = hub / "analysis"
    summary_path = analysis_dir / "metrics_delta_summary.json"
    highlights_path = analysis_dir / "metrics_delta_highlights.txt"
    preview_path = analysis_dir / "metrics_delta_highlights_preview.txt"

    for path in (summary_path, highlights_path, preview_path):
        if not path.exists():
            raise SystemExit(f"Missing artifact: {path}")

    summary = json.loads(summary_path.read_text())
    highlights_text = highlights_path.read_text().strip()
    preview_text = preview_path.read_text().strip()

    failures = []
    for model in EXPECTED_MODELS:
        if model not in summary:
            failures.append(f"Model '{model}' not present in summary JSON")
            continue
        deltas = summary[model]
        for field in EXPECTED_FIELDS:
            if field not in deltas:
                failures.append(f"Field '{field}' missing for model '{model}'")
                continue
            precision = 3 if "ms_ssim" in field else 6
            expected = format_delta(deltas[field], precision)
            for label, block in (("highlights", highlights_text), ("preview", preview_text)):
                if expected not in block:
                    failures.append(f"{label} text missing {model}:{field} value {expected}")

    if failures:
        raise SystemExit("\n".join(failures))

    print("Highlights match metrics_delta_summary.json for all tracked fields.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.studies.sim_lines_4x.pipeline import (  # noqa: E402
    PREDICTION_SCALE_CHOICES,
    ScenarioSpec,
    run_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gs2 + custom probe scenario.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sim_lines_4x"),
        help="Root directory for scenario outputs.",
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--image-multiplier",
        type=int,
        default=1,
        help="Scale total image count by this factor.",
    )
    parser.add_argument(
        "--group-multiplier",
        type=int,
        default=1,
        help="Scale group count by this factor.",
    )
    parser.add_argument(
        "--prediction-scale-source",
        choices=PREDICTION_SCALE_CHOICES,
        default="none",
        help="Prediction scaling strategy (default: none).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = ScenarioSpec(
        name="gs2_custom",
        gridsize=2,
        probe_mode="custom",
        probe_scale=4.0,
    )
    run_scenario(
        scenario,
        output_root=args.output_root,
        nepochs=args.nepochs,
        image_multiplier=args.image_multiplier,
        group_multiplier=args.group_multiplier,
        prediction_scale_source=args.prediction_scale_source,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Runbook entrypoint for NERSC scan807 + cameraman orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.studies.invocation_logging import write_invocation_artifacts
from scripts.studies.nersc_orchestration import (
    DOWNSAMPLE_POLICY_CHOICES,
    run_nersc_scan807_cameraman_study,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NERSC orchestration: PtychoViT checkpoint inference on scan807/cameraman, "
            "hybrid_resnet training on cameraman half split, cross-dataset hybrid inference, "
            "and per-dataset metrics/visual aggregation."
        )
    )
    parser.add_argument("--scan807-dp", type=Path, required=True)
    parser.add_argument("--scan807-para", type=Path, required=True)
    parser.add_argument("--cameraman-dp", type=Path, required=True)
    parser.add_argument("--cameraman-para", type=Path, required=True)
    parser.add_argument("--ptychovit-checkpoint", type=Path, required=True)
    parser.add_argument("--ptychovit-repo", type=Path, default=Path("/home/ollie/Documents/ptycho-vit"))
    parser.add_argument("--half", type=str, choices=["top", "bottom"], default="top")
    parser.add_argument(
        "--downsample-policy",
        type=str,
        choices=list(DOWNSAMPLE_POLICY_CHOICES),
        default="bin-crop",
        help=(
            "Downsample policy for NERSC 256->128 preparation: "
            "'bin-crop' bins diffraction and crops real-space; "
            "'crop-bin' crops diffraction and bins real-space."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py",
        argv=(argv if argv is not None else sys.argv[1:]),
        parsed_args=vars(args),
    )

    manifest = run_nersc_scan807_cameraman_study(
        scan807_dp=args.scan807_dp,
        scan807_para=args.scan807_para,
        cameraman_dp=args.cameraman_dp,
        cameraman_para=args.cameraman_para,
        ptychovit_checkpoint=args.ptychovit_checkpoint,
        output_dir=args.output_dir,
        half=args.half,
        downsample_policy=args.downsample_policy,
        seed=args.seed,
        ptychovit_repo=args.ptychovit_repo,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

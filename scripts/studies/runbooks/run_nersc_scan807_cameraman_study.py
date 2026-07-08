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
    PROBE_MODE_POLICY_CHOICES,
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
        "--target-n",
        type=int,
        default=128,
        help="Target diffraction/object patch size for hybrid study path (default: 128).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Hybrid training epochs for this runbook (default: 40).",
    )
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
    parser.add_argument(
        "--probe-mode-policy",
        type=str,
        choices=list(PROBE_MODE_POLICY_CHOICES),
        default="incoherent_aggregate",
        help=(
            "Probe collapse policy for multimode para probes used in NERSC conversion: "
            "'incoherent_aggregate' (default) or 'first_mode'."
        ),
    )
    parser.add_argument(
        "--position-reassembly-backend",
        type=str,
        choices=["shift_sum"],
        default="shift_sum",
        help="Pinned external position reassembly backend for this study (must be shift_sum).",
    )
    parser.add_argument(
        "--position-crop-border",
        type=int,
        default=None,
        help=(
            "Optional center-crop border (pixels) applied during position reassembly. "
            "When unset, backend default is used."
        ),
    )
    parser.add_argument(
        "--probe-mask",
        dest="probe_mask",
        action="store_true",
        default=False,
        help="Enable probe mask in hybrid_resnet Torch path.",
    )
    parser.add_argument(
        "--no-probe-mask",
        dest="probe_mask",
        action="store_false",
        help="Disable probe mask in hybrid_resnet Torch path.",
    )
    parser.add_argument(
        "--probe-mask-sigma",
        type=float,
        default=1.0,
        help="Probe mask edge smoothing sigma (0.0 for hard edge).",
    )
    parser.add_argument(
        "--probe-mask-diameter",
        type=float,
        default=None,
        help="Optional probe mask diameter override.",
    )
    parser.add_argument(
        "--torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_true",
        default=False,
        help="Enable per-sample prediction-L2 matching to target in Torch MAE mode.",
    )
    parser.add_argument(
        "--no-torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_false",
        help="Disable prediction-L2 matching to target in Torch MAE mode.",
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
        target_n=args.target_n,
        epochs=args.epochs,
        downsample_policy=args.downsample_policy,
        probe_mode_policy=args.probe_mode_policy,
        position_reassembly_backend=args.position_reassembly_backend,
        position_crop_border=args.position_crop_border,
        probe_mask=args.probe_mask,
        probe_mask_sigma=args.probe_mask_sigma,
        probe_mask_diameter=args.probe_mask_diameter,
        torch_mae_pred_l2_match_target=args.torch_mae_pred_l2_match_target,
        seed=args.seed,
        ptychovit_repo=args.ptychovit_repo,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

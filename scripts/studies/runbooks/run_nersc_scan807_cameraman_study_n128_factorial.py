#!/usr/bin/env python3
"""Sequential N=128 factorial runbook for the NERSC scan807+cameraman study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.studies.invocation_logging import write_invocation_artifacts

PROBE_MASK_MODES = ("off", "on_soft", "on_hard")
MAE_NORM_MODES = (False, True)
DOWNSAMPLE_POLICIES = ("bin-crop", "crop-bin")


def _mae_label(enabled: bool) -> str:
    return "maenorm-on" if enabled else "maenorm-off"


def _probe_label(mode: str) -> str:
    labels = {
        "off": "pm-off",
        "on_soft": "pm-soft",
        "on_hard": "pm-hard",
    }
    return labels[mode]


def _downsample_label(policy: str) -> str:
    labels = {
        "bin-crop": "ds-bincrop",
        "crop-bin": "ds-cropbin",
    }
    return labels[policy]


def _resolve_probe_mask_settings(
    mode: str,
    *,
    soft_mask_sigma: float,
    probe_mask_diameter: float | None,
) -> dict[str, Any]:
    if mode == "off":
        return {
            "probe_mask": False,
            "probe_mask_sigma": float(soft_mask_sigma),
            "probe_mask_diameter": None,
        }
    if mode == "on_soft":
        return {
            "probe_mask": True,
            "probe_mask_sigma": float(soft_mask_sigma),
            "probe_mask_diameter": probe_mask_diameter,
        }
    if mode == "on_hard":
        return {
            "probe_mask": True,
            "probe_mask_sigma": 0.0,
            "probe_mask_diameter": probe_mask_diameter,
        }
    raise ValueError(f"Unknown probe mask mode: {mode}")


def build_factorial_matrix(
    *,
    soft_mask_sigma: float = 1.0,
    probe_mask_diameter: float | None = None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for probe_mode in PROBE_MASK_MODES:
        for mae_norm in MAE_NORM_MODES:
            for policy in DOWNSAMPLE_POLICIES:
                run_id = "__".join(
                    (
                        _probe_label(probe_mode),
                        _mae_label(mae_norm),
                        _downsample_label(policy),
                    )
                )
                probe_settings = _resolve_probe_mask_settings(
                    probe_mode,
                    soft_mask_sigma=soft_mask_sigma,
                    probe_mask_diameter=probe_mask_diameter,
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "probe_mode": probe_mode,
                        "downsample_policy": policy,
                        "torch_mae_pred_l2_match_target": bool(mae_norm),
                        **probe_settings,
                    }
                )
    return runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run N=128 NERSC factorial study over probe-mask mode, MAE L2-matching, "
            "and downsample policy."
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
        "--epochs",
        type=int,
        default=40,
        help="Fixed epoch count for every run in this sweep (not part of matrix).",
    )
    parser.add_argument(
        "--soft-mask-sigma",
        type=float,
        default=1.0,
        help="Probe mask sigma used by on_soft mode.",
    )
    parser.add_argument(
        "--probe-mask-diameter",
        type=float,
        default=None,
        help="Optional probe mask diameter for masked modes.",
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--prune-intermediates",
        dest="prune_intermediates",
        action="store_true",
        default=True,
        help="Remove large per-arm intermediate files/directories after each successful arm.",
    )
    parser.add_argument(
        "--no-prune-intermediates",
        dest="prune_intermediates",
        action="store_false",
        help="Keep all per-arm intermediate files/directories.",
    )
    return parser.parse_args(argv)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _prune_run_intermediates(run_output: Path) -> dict[str, Any]:
    """Remove large intermediate data directories generated by one arm."""
    candidates = [
        run_output / "scan807" / "working_pair",
        run_output / "cameraman256" / "working_pair",
        run_output / "scan807" / "hybrid_dataset",
        run_output / "cameraman256" / "hybrid_dataset",
        run_output / "scan807" / "hybrid_cached",
        run_output / "cameraman256" / "hybrid_cached",
        run_output / "hybrid_training" / "datasets",
        run_output / "scan807" / "runs" / "pinn_ptychovit" / "bridge_work",
        run_output / "cameraman256" / "runs" / "pinn_ptychovit" / "bridge_work",
    ]
    removed: list[str] = []
    bytes_removed = 0
    for path in candidates:
        if not path.exists():
            continue
        if path.is_dir():
            bytes_removed += _dir_size_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
        elif path.is_file():
            try:
                bytes_removed += path.stat().st_size
            except OSError:
                pass
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                pass
    return {"removed_paths": removed, "bytes_removed": int(bytes_removed)}


def _run_single_arm_subprocess(
    *,
    args: argparse.Namespace,
    run: dict[str, Any],
    run_output: Path,
) -> dict[str, Any]:
    logs_dir = run_output / "driver_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "arm_stdout.log"
    stderr_path = logs_dir / "arm_stderr.log"

    cmd = [
        sys.executable,
        "scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py",
        "--scan807-dp",
        str(args.scan807_dp),
        "--scan807-para",
        str(args.scan807_para),
        "--cameraman-dp",
        str(args.cameraman_dp),
        "--cameraman-para",
        str(args.cameraman_para),
        "--ptychovit-checkpoint",
        str(args.ptychovit_checkpoint),
        "--ptychovit-repo",
        str(args.ptychovit_repo),
        "--half",
        str(args.half),
        "--target-n",
        "128",
        "--epochs",
        str(int(args.epochs)),
        "--downsample-policy",
        str(run["downsample_policy"]),
        "--position-reassembly-backend",
        str(args.position_reassembly_backend),
        *(
            ["--position-crop-border", str(int(args.position_crop_border))]
            if args.position_crop_border is not None
            else []
        ),
        "--output-dir",
        str(run_output),
        "--seed",
        str(int(args.seed)),
    ]

    if bool(run["probe_mask"]):
        cmd.append("--probe-mask")
    else:
        cmd.append("--no-probe-mask")
    cmd.extend(["--probe-mask-sigma", str(float(run["probe_mask_sigma"]))])
    if run.get("probe_mask_diameter") is not None:
        cmd.extend(["--probe-mask-diameter", str(run["probe_mask_diameter"])])
    if bool(run["torch_mae_pred_l2_match_target"]):
        cmd.append("--torch-mae-pred-l2-match-target")
    else:
        cmd.append("--no-torch-mae-pred-l2-match-target")

    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Factorial arm failed for run_id={run['run_id']} exit={completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )

    run_manifest_path = run_output / "manifest.json"
    run_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text())

    return {
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "returncode": int(completed.returncode),
        "run_manifest_output_dir": run_manifest.get("output_dir"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.epochs) <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if float(args.soft_mask_sigma) < 0.0:
        raise ValueError(f"soft-mask-sigma must be >= 0, got {args.soft_mask_sigma}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_invocation_artifacts(
        output_dir=args.output_root,
        script_path="scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n128_factorial.py",
        argv=(argv if argv is not None else sys.argv[1:]),
        parsed_args=vars(args),
    )

    matrix = build_factorial_matrix(
        soft_mask_sigma=float(args.soft_mask_sigma),
        probe_mask_diameter=args.probe_mask_diameter,
    )
    runs: list[dict[str, Any]] = []

    for run in matrix:
        run_id = run["run_id"]
        run_output = args.output_root / "runs" / run_id
        run_output.mkdir(parents=True, exist_ok=True)
        arm_exec = _run_single_arm_subprocess(args=args, run=run, run_output=run_output)
        prune_info = {"removed_paths": [], "bytes_removed": 0}
        if bool(args.prune_intermediates):
            prune_info = _prune_run_intermediates(run_output)
        runs.append(
            {
                "run_id": run_id,
                "output_dir": str(run_output),
                "manifest_path": str(run_output / "manifest.json"),
                "factors": {
                    "probe_mode": run["probe_mode"],
                    "probe_mask": bool(run["probe_mask"]),
                    "probe_mask_sigma": float(run["probe_mask_sigma"]),
                    "probe_mask_diameter": run["probe_mask_diameter"],
                    "torch_mae_pred_l2_match_target": bool(run["torch_mae_pred_l2_match_target"]),
                    "downsample_policy": run["downsample_policy"],
                },
                "driver_stdout_log": arm_exec["stdout_log"],
                "driver_stderr_log": arm_exec["stderr_log"],
                "driver_returncode": arm_exec["returncode"],
                "run_manifest_output_dir": arm_exec["run_manifest_output_dir"],
                "prune_intermediates": bool(args.prune_intermediates),
                "prune_removed_paths": prune_info["removed_paths"],
                "prune_bytes_removed": prune_info["bytes_removed"],
            }
        )

    manifest = {
        "output_root": str(args.output_root),
        "target_n": 128,
        "epochs": int(args.epochs),
        "half": args.half,
        "position_reassembly_backend": args.position_reassembly_backend,
        "position_crop_border": (
            None if args.position_crop_border is None else int(args.position_crop_border)
        ),
        "soft_mask_sigma": float(args.soft_mask_sigma),
        "probe_mask_diameter": args.probe_mask_diameter,
        "seed": int(args.seed),
        "prune_intermediates": bool(args.prune_intermediates),
        "run_count": len(runs),
        "runs": runs,
    }
    (args.output_root / "factorial_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

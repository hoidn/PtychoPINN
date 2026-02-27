#!/usr/bin/env python3
"""Staged hybrid_resnet mode/skip/width sweep runbook with structural-axis hooks.

This runbook owns orchestration-layer matrix expansion, stage/substage guardrails,
promotion-source validation, seed-rerank aggregation, and artifact persistence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.studies.invocation_logging import write_invocation_artifacts


SUMMARY_SCHEMA_VERSION = "hybrid_resnet_mode_skip_sweep.v1"
ALLOWED_SUMMARY_SCHEMA_VERSIONS = {SUMMARY_SCHEMA_VERSION, "v1"}
STUDY_KEY = "hybrid-resnet-mode-skip-sweep"
SEED_SET = (3, 11, 17)


class StageValidationError(ValueError):
    """Raised when stage configuration is invalid."""


class PromotionSourceError(ValueError):
    """Raised when promotion source is invalid or missing."""


class MatrixExpansionError(ValueError):
    """Raised when matrix expansion violates stage constraints."""


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_optional_numeric_csv(raw: str) -> list[str]:
    values = _parse_csv(raw)
    return values if values else ["none"]


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any, key: str, *, default: float | None = None) -> float:
    if value is None or value == "":
        if default is None:
            raise PromotionSourceError(f"Missing required numeric value for '{key}'")
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PromotionSourceError(f"Invalid numeric value for '{key}': {value!r}") from exc


def _safe_int(value: Any, key: str, *, default: int | None = None) -> int:
    if value is None or value == "":
        if default is None:
            raise PromotionSourceError(f"Missing required integer value for '{key}'")
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise PromotionSourceError(f"Invalid integer value for '{key}': {value!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the sweep runbook."""
    parser = argparse.ArgumentParser(
        description="Run staged hybrid_resnet mode/skip/width sweep with structural-axis hooks"
    )

    # Stage A axes.
    parser.add_argument("--modes", type=_parse_int_csv, default=[12])
    parser.add_argument("--skip-values", type=_parse_csv, default=["off"])
    parser.add_argument("--widths", type=_parse_int_csv, default=[32])

    # Structural axes (Stages B-E).
    parser.add_argument("--fno-blocks-values", type=_parse_int_csv, default=[4])
    parser.add_argument("--downsample-schedule-values", type=_parse_int_csv, default=[2])
    parser.add_argument("--downsample-op-values", type=_parse_csv, default=["stride_conv"])
    parser.add_argument(
        "--encoder-conv-hidden-values",
        type=_parse_optional_numeric_csv,
        default=["none"],
    )
    parser.add_argument(
        "--encoder-spectral-hidden-values",
        type=_parse_optional_numeric_csv,
        default=["none"],
    )
    parser.add_argument("--max-hidden-values", type=_parse_optional_numeric_csv, default=["none"])
    parser.add_argument("--resnet-width-values", type=_parse_optional_numeric_csv, default=["none"])
    parser.add_argument("--resnet-blocks-values", type=_parse_int_csv, default=[6])
    parser.add_argument("--skip-style-values", type=_parse_csv, default=["add"])

    # Resolution and dataset profiles.
    parser.add_argument(
        "--ns",
        type=int,
        default=128,
        choices=[128, 256],
        help="Resolution N for this invocation. Defaults to 128.",
    )
    parser.add_argument(
        "--dataset-profiles-n128",
        type=_parse_csv,
        default=["integration_grid_lines_n128_v1"],
    )
    parser.add_argument(
        "--dataset-profiles-n256",
        type=_parse_csv,
        default=["cameraman256_halfsplit_v1"],
    )
    parser.add_argument("--cameraman-dp", type=Path)
    parser.add_argument("--cameraman-para", type=Path)
    parser.add_argument("--fly001-external-train-npz", type=Path)
    parser.add_argument("--fly001-external-test-npz", type=Path)
    parser.add_argument("--custom-n128-train-npz", type=Path)
    parser.add_argument("--custom-n128-test-npz", type=Path)
    parser.add_argument("--custom-n256-train-npz", type=Path)
    parser.add_argument("--custom-n256-test-npz", type=Path)

    # Stage and promotion controls.
    parser.add_argument("--stage-id", default="A", choices=["A", "B", "C", "D", "E"])
    parser.add_argument(
        "--substage-id",
        default="none",
        choices=["none", "C1", "C2", "D1", "D2", "D3", "D4"],
    )
    parser.add_argument("--promotion-source-summary", type=Path)
    parser.add_argument("--allow-n256-direct-diagnostic", action="store_true")
    parser.add_argument("--aggregate-seed-rerank-root", type=Path)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--emit-stage-anchor-summary", type=Path)
    parser.add_argument("--emit-robust-promotion-summary", type=Path)

    # Objectives and feasibility limits.
    parser.add_argument(
        "--promotion-objectives",
        type=_parse_csv,
        default=["amp_mae", "amp_mse", "train_wall_time_sec"],
    )
    parser.add_argument("--top-k-n256", type=int, default=6)
    parser.add_argument("--max-train-seconds-n128", type=int, default=2700)
    parser.add_argument("--max-train-seconds-n256", type=int, default=9000)
    parser.add_argument("--max-inference-seconds-n128", type=int, default=60)
    parser.add_argument("--max-inference-seconds-n256", type=int, default=240)
    parser.add_argument("--max-phase-ssim-drop", type=float, default=0.03)
    parser.add_argument("--max-model-params", type=int, default=300000000)

    # Training + confounders.
    parser.add_argument("--epochs-n128", type=int, default=20)
    parser.add_argument("--epochs-n256", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_false",
    )
    parser.add_argument(
        "--torch-no-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_false",
    )
    parser.add_argument("--probe-mask", dest="probe_mask", action="store_true", default=False)
    parser.add_argument("--no-probe-mask", dest="probe_mask", action="store_false")

    parser.add_argument(
        "--prune-heavy-artifacts",
        dest="prune_heavy_artifacts",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-prune-heavy-artifacts",
        dest="prune_heavy_artifacts",
        action="store_false",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root for sweep artifacts (required outside aggregation mode)",
    )

    return parser.parse_args(argv)


def _active_profiles(args: argparse.Namespace) -> list[str]:
    return list(args.dataset_profiles_n256 if args.ns == 256 else args.dataset_profiles_n128)


def validate_stage_configuration(args: argparse.Namespace) -> None:
    """Validate stage/substage/profile constraints."""
    if args.stage_id == "C" and args.substage_id not in {"C1", "C2"}:
        raise StageValidationError(
            f"Stage C requires substage_id to be C1 or C2, got '{args.substage_id}'"
        )
    if args.stage_id == "D" and args.substage_id not in {"D1", "D2", "D3", "D4"}:
        raise StageValidationError(
            f"Stage D requires substage_id to be D1, D2, D3, or D4, got '{args.substage_id}'"
        )
    if args.stage_id in {"A", "B", "E"} and args.substage_id != "none":
        raise StageValidationError(
            f"Stage {args.stage_id} requires substage_id to be 'none', got '{args.substage_id}'"
        )

    if args.allow_n256_direct_diagnostic and (args.ns != 256 or args.top_k_n256 != 0):
        raise StageValidationError("--allow-n256-direct-diagnostic requires --ns 256 --top-k-n256 0")

    if args.aggregate_seed_rerank_root:
        if not args.source_summary:
            raise StageValidationError("--aggregate-seed-rerank-root requires --source-summary")
        if not args.emit_robust_promotion_summary:
            raise StageValidationError(
                "--aggregate-seed-rerank-root requires --emit-robust-promotion-summary"
            )
        if not args.emit_stage_anchor_summary:
            raise StageValidationError(
                "--aggregate-seed-rerank-root requires --emit-stage-anchor-summary"
            )
    elif args.output_root is None:
        raise StageValidationError("--output-root is required when not aggregating seed rerank")

    if (
        not args.aggregate_seed_rerank_root
        and args.stage_id in {"B", "C", "D", "E"}
        and not args.promotion_source_summary
    ):
        raise PromotionSourceError(
            f"Stage {args.stage_id} requires --promotion-source-summary to inherit upstream anchors"
        )

    if (
        not args.aggregate_seed_rerank_root
        and args.stage_id in {"B", "C", "D", "E"}
        and args.ns == 256
        and not args.promotion_source_summary
    ):
        raise PromotionSourceError(
            f"Stage {args.stage_id} with N=256 requires --promotion-source-summary"
        )

    if args.ns == 256 and args.top_k_n256 > 0 and not args.allow_n256_direct_diagnostic:
        if not args.promotion_source_summary:
            raise PromotionSourceError(
                "Promotion-enabled N=256 runs require --promotion-source-summary"
            )

    skip_values = {value.lower() for value in args.skip_values}
    if not skip_values.issubset({"off", "on"}):
        raise StageValidationError("--skip-values must be a comma list from: off,on")

    downsample_ops = {value.lower() for value in args.downsample_op_values}
    allowed_ops = {"stride_conv", "avgpool_conv", "blurpool_conv"}
    if not downsample_ops.issubset(allowed_ops):
        raise StageValidationError(
            "--downsample-op-values must use: stride_conv|avgpool_conv|blurpool_conv"
        )

    skip_styles = {value.lower() for value in args.skip_style_values}
    allowed_skip_styles = {"add", "concat", "gated_add"}
    if not skip_styles.issubset(allowed_skip_styles):
        raise StageValidationError("--skip-style-values must use: add|concat|gated_add")

    profiles = _active_profiles(args)
    if args.ns == 256:
        if "cameraman256_halfsplit_v1" in profiles and (not args.cameraman_dp or not args.cameraman_para):
            raise StageValidationError(
                "cameraman256_halfsplit_v1 profile requires --cameraman-dp and --cameraman-para"
            )
        if "custom_npz_pair_n256" in profiles and (
            not args.custom_n256_train_npz or not args.custom_n256_test_npz
        ):
            raise StageValidationError(
                "custom_npz_pair_n256 requires --custom-n256-train-npz and --custom-n256-test-npz"
            )
    if args.ns == 128:
        if "fly001_external_n128_top_bottom_v1" in profiles and (
            not args.fly001_external_train_npz or not args.fly001_external_test_npz
        ):
            raise StageValidationError(
                "fly001_external_n128_top_bottom_v1 requires --fly001-external-train-npz and "
                "--fly001-external-test-npz"
            )
        if "custom_npz_pair_n128" in profiles and (
            not args.custom_n128_train_npz or not args.custom_n128_test_npz
        ):
            raise StageValidationError(
                "custom_npz_pair_n128 requires --custom-n128-train-npz and --custom-n128-test-npz"
            )


def validate_matrix_constraints(args: argparse.Namespace) -> None:
    """Validate stage-specific matrix-expansion rules."""
    structural_axis_lengths = {
        "fno_blocks": len(args.fno_blocks_values),
        "downsample_schedule": len(args.downsample_schedule_values),
        "downsample_op": len(args.downsample_op_values),
        "encoder_conv_hidden": len(args.encoder_conv_hidden_values),
        "encoder_spectral_hidden": len(args.encoder_spectral_hidden_values),
        "max_hidden": len(args.max_hidden_values),
        "resnet_width": len(args.resnet_width_values),
        "resnet_blocks": len(args.resnet_blocks_values),
        "skip_style": len(args.skip_style_values),
    }
    structural_axis_values = {
        "fno_blocks": list(args.fno_blocks_values),
        "downsample_schedule": list(args.downsample_schedule_values),
        "downsample_op": list(args.downsample_op_values),
        "encoder_conv_hidden": list(args.encoder_conv_hidden_values),
        "encoder_spectral_hidden": list(args.encoder_spectral_hidden_values),
        "max_hidden": list(args.max_hidden_values),
        "resnet_width": list(args.resnet_width_values),
        "resnet_blocks": list(args.resnet_blocks_values),
        "skip_style": list(args.skip_style_values),
    }

    if args.stage_id == "A":
        stage_a_defaults = {
            "fno_blocks": "4",
            "downsample_schedule": "2",
            "downsample_op": "stride_conv",
            "encoder_conv_hidden": "none",
            "encoder_spectral_hidden": "none",
            "max_hidden": "none",
            "resnet_width": "none",
            "resnet_blocks": "6",
            "skip_style": "add",
        }
        violations: list[str] = []
        for axis_name, values in structural_axis_values.items():
            if len(values) != 1:
                violations.append(f"{axis_name}={values}")
                continue
            observed = str(values[0]).strip().lower()
            expected = stage_a_defaults[axis_name]
            if observed != expected:
                violations.append(f"{axis_name}={values[0]}")

        if violations:
            raise MatrixExpansionError(
                "Stage A can only vary {modes, skip-values, widths}; "
                "all structural axes must remain at defaults "
                "(fno_blocks=4, downsample_schedule=2, downsample_op=stride_conv, "
                "encoder_conv_hidden=none, encoder_spectral_hidden=none, max_hidden=none, "
                "resnet_width=none, resnet_blocks=6, skip_style=add); "
                f"got: {violations}"
            )
    else:
        if len(args.modes) > 1 or len(args.skip_values) > 1 or len(args.widths) > 1:
            raise MatrixExpansionError(
                "Stages B-E require non-active axes to stay scalar; "
                "modes/skip-values/widths must each have one value because they are inherited"
            )

        active_axes: set[str] = set()
        if args.stage_id == "B":
            active_axes = {"fno_blocks"}
        elif args.stage_id == "C":
            active_axes = {"downsample_schedule"} if args.substage_id == "C1" else {"downsample_op"}
        elif args.stage_id == "D":
            if args.substage_id == "D1":
                active_axes = {"encoder_conv_hidden"}
            elif args.substage_id == "D2":
                active_axes = {"encoder_spectral_hidden"}
            elif args.substage_id == "D3":
                active_axes = {"max_hidden", "resnet_width"}
            else:
                active_axes = {"resnet_blocks"}
        elif args.stage_id == "E":
            active_axes = {"skip_style"}

        non_active_multivalue = [
            axis for axis, size in structural_axis_lengths.items() if axis not in active_axes and size > 1
        ]
        if non_active_multivalue:
            raise MatrixExpansionError(
                "Stages B-E reject multi-value lists on non-active axes; "
                f"found non-active multivalue axes: {non_active_multivalue}"
            )

        active_multivalue = [axis for axis in active_axes if structural_axis_lengths.get(axis, 1) > 1]
        if len(active_multivalue) > 1:
            raise MatrixExpansionError(
                f"Stage {args.stage_id}/{args.substage_id} can vary at most one structural axis per invocation; "
                f"got {active_multivalue}"
            )

    for width_raw in args.resnet_width_values:
        if str(width_raw).lower() == "none":
            continue
        width = _safe_int(width_raw, "resnet_width")
        if width <= 0 or width % 4 != 0:
            raise StageValidationError(
                f"resnet_width must be positive and divisible by 4, got {width}"
            )



def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_resolution(args: argparse.Namespace, profile: str) -> dict[str, Any]:
    """Resolve profile metadata and input provenance payload."""
    if args.ns == 128:
        if profile == "integration_grid_lines_n128_v1":
            return {
                "profile": profile,
                "emitted_dataset_args": ["--train-npz", "--test-npz"],
                "resolved_paths": {
                    "train_npz": str((args.output_root or Path("outputs")) / "datasets" / profile / "train.npz"),
                    "test_npz": str((args.output_root or Path("outputs")) / "datasets" / profile / "test.npz"),
                },
            }
        if profile == "fly001_external_n128_top_bottom_v1":
            return {
                "profile": profile,
                "emitted_dataset_args": ["--train-npz", "--test-npz"],
                "resolved_paths": {
                    "train_npz": str(args.fly001_external_train_npz),
                    "test_npz": str(args.fly001_external_test_npz),
                },
            }
        if profile == "custom_npz_pair_n128":
            return {
                "profile": profile,
                "emitted_dataset_args": ["--train-npz", "--test-npz"],
                "resolved_paths": {
                    "train_npz": str(args.custom_n128_train_npz),
                    "test_npz": str(args.custom_n128_test_npz),
                },
            }
    if args.ns == 256:
        if profile == "cameraman256_halfsplit_v1":
            return {
                "profile": profile,
                "emitted_dataset_args": ["--train-data", "--test-data"],
                "resolved_paths": {
                    "source_dp_hdf5": str(args.cameraman_dp),
                    "source_para_hdf5": str(args.cameraman_para),
                },
            }
        if profile == "custom_npz_pair_n256":
            return {
                "profile": profile,
                "emitted_dataset_args": ["--train-npz", "--test-npz"],
                "resolved_paths": {
                    "train_npz": str(args.custom_n256_train_npz),
                    "test_npz": str(args.custom_n256_test_npz),
                },
            }
    raise StageValidationError(f"Unsupported dataset profile for N={args.ns}: {profile}")


def build_sweep_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build run-level manifest payload."""
    profiles = _active_profiles(args)
    profile_payload = [_profile_resolution(args, profile) for profile in profiles]

    # Attach optional hashes when files are present.
    for payload in profile_payload:
        hashes: dict[str, str] = {}
        for key, raw_path in payload["resolved_paths"].items():
            path = Path(raw_path)
            if path.exists() and path.is_file():
                hashes[key] = _compute_file_sha256(path)
        payload["sha256"] = hashes

    manifest: dict[str, Any] = {
        "study_key": STUDY_KEY,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "script": str(Path(__file__)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage_id": args.stage_id,
        "substage_id": args.substage_id,
        "resolution": int(args.ns),
        "seed": int(args.seed),
        "probe_mask_enabled": bool(args.probe_mask),
        "torch_mae_pred_l2_match_target": bool(args.torch_mae_pred_l2_match_target),
        "dataset_profiles": {
            "n128": list(args.dataset_profiles_n128),
            "n256": list(args.dataset_profiles_n256),
            "active": profiles,
            "resolved": profile_payload,
        },
        "sweep_axes": {
            "modes": list(args.modes),
            "skip_values": list(args.skip_values),
            "widths": list(args.widths),
            "fno_blocks": list(args.fno_blocks_values),
            "downsample_schedule": list(args.downsample_schedule_values),
            "downsample_op": list(args.downsample_op_values),
            "encoder_conv_hidden": list(args.encoder_conv_hidden_values),
            "encoder_spectral_hidden": list(args.encoder_spectral_hidden_values),
            "max_hidden": list(args.max_hidden_values),
            "resnet_width": list(args.resnet_width_values),
            "resnet_blocks": list(args.resnet_blocks_values),
            "skip_style": list(args.skip_style_values),
        },
        "promotion": {
            "objectives": list(args.promotion_objectives),
            "top_k_n256": int(args.top_k_n256),
            "promotion_source_summary": str(args.promotion_source_summary)
            if args.promotion_source_summary
            else "",
            "allow_n256_direct_diagnostic": bool(args.allow_n256_direct_diagnostic),
        },
        "guardrails": {
            "max_train_seconds_n128": int(args.max_train_seconds_n128),
            "max_train_seconds_n256": int(args.max_train_seconds_n256),
            "max_inference_seconds_n128": int(args.max_inference_seconds_n128),
            "max_inference_seconds_n256": int(args.max_inference_seconds_n256),
            "max_phase_ssim_drop": float(args.max_phase_ssim_drop),
            "max_model_params": int(args.max_model_params),
        },
    }
    if args.aggregate_seed_rerank_root:
        manifest["aggregation_mode"] = {
            "aggregate_seed_rerank_root": str(args.aggregate_seed_rerank_root),
            "source_summary": str(args.source_summary),
            "emit_stage_anchor_summary": str(args.emit_stage_anchor_summary),
            "emit_robust_promotion_summary": str(args.emit_robust_promotion_summary),
        }
    return manifest


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise PromotionSourceError(f"CSV file does not exist: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _require_columns(path: Path, fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [column for column in required if column not in set(fieldnames)]
    if missing:
        raise PromotionSourceError(f"{path} is missing required columns: {missing}")


def _require_schema_version(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if "summary_schema_version" not in set(fieldnames):
        raise PromotionSourceError(
            f"{path} is missing required column 'summary_schema_version'"
        )
    for row in rows:
        version = str(row.get("summary_schema_version", "")).strip()
        if version == "":
            raise PromotionSourceError(
                f"{path} contains rows with empty summary_schema_version"
            )
        if version not in ALLOWED_SUMMARY_SCHEMA_VERSIONS:
            allowed = ", ".join(sorted(ALLOWED_SUMMARY_SCHEMA_VERSIONS))
            raise PromotionSourceError(
                f"{path} has unsupported summary_schema_version '{version}'. "
                f"Expected one of: {allowed}"
            )


def _feasibility_limits(args: argparse.Namespace, resolution: int) -> dict[str, float]:
    return {
        "max_train_seconds": float(
            args.max_train_seconds_n256 if resolution == 256 else args.max_train_seconds_n128
        ),
        "max_inference_seconds": float(
            args.max_inference_seconds_n256 if resolution == 256 else args.max_inference_seconds_n128
        ),
        "max_phase_ssim_drop": float(args.max_phase_ssim_drop),
        "max_model_params": float(args.max_model_params),
    }


def _row_feasibility(row: Mapping[str, Any], args: argparse.Namespace, resolution: int) -> tuple[bool, list[str]]:
    limits = _feasibility_limits(args, resolution)
    phase_drop = _safe_float(row.get("phase_ssim_drop_vs_baseline", 0.0), "phase_ssim_drop_vs_baseline")
    model_params = _safe_float(row.get("model_params", 0.0), "model_params")
    train_sec = _safe_float(row.get("train_wall_time_sec", 0.0), "train_wall_time_sec")
    inference_sec = _safe_float(row.get("inference_time_s", 0.0), "inference_time_s")

    violations: list[str] = []
    if phase_drop > limits["max_phase_ssim_drop"]:
        violations.append("phase_ssim_drop")
    if model_params > limits["max_model_params"]:
        violations.append("model_params")
    if train_sec > limits["max_train_seconds"]:
        violations.append("train_wall_time_sec")
    if inference_sec > limits["max_inference_seconds"]:
        violations.append("inference_time_s")
    return (len(violations) == 0), violations


def _load_promotion_rows(
    path: Path,
    args: argparse.Namespace,
    *,
    require_robust_fields: bool,
) -> list[dict[str, Any]]:
    fieldnames, rows = _read_csv(path)
    if not rows:
        raise PromotionSourceError(f"Promotion summary is empty: {path}")
    _require_schema_version(path, fieldnames, rows)

    required = [
        "run_id",
        "modes",
        "skip",
        "width",
        "amp_mae",
        "amp_mse",
        "train_wall_time_sec",
        "inference_time_s",
        "model_params",
        "phase_ssim_drop_vs_baseline",
    ]
    if require_robust_fields:
        required.extend(["pareto_rank_seed3", "pareto_rank_seed11", "pareto_rank_seed17", "pareto_rank_median"])
    _require_columns(path, fieldnames, required)

    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed_row = dict(row)
        parsed_row["run_id"] = str(row["run_id"])
        parsed_row["modes"] = _safe_int(row.get("modes"), "modes")
        parsed_row["skip"] = str(row.get("skip", "off")).strip().lower()
        parsed_row["width"] = _safe_int(row.get("width"), "width")
        parsed_row["amp_mae"] = _safe_float(row.get("amp_mae"), "amp_mae")
        parsed_row["amp_mse"] = _safe_float(row.get("amp_mse"), "amp_mse")
        parsed_row["train_wall_time_sec"] = _safe_float(row.get("train_wall_time_sec"), "train_wall_time_sec")
        parsed_row["inference_time_s"] = _safe_float(row.get("inference_time_s"), "inference_time_s")
        parsed_row["model_params"] = _safe_float(row.get("model_params"), "model_params")
        parsed_row["phase_ssim_drop_vs_baseline"] = _safe_float(
            row.get("phase_ssim_drop_vs_baseline"),
            "phase_ssim_drop_vs_baseline",
            default=0.0,
        )
        parsed_row["stage_id"] = str(row.get("stage_id", ""))
        parsed_row["substage_id"] = str(row.get("substage_id", ""))
        parsed_row["is_stage_anchor"] = _bool_from_value(row.get("is_stage_anchor", False))

        parsed_row["fno_blocks"] = _safe_int(row.get("fno_blocks"), "fno_blocks", default=4)
        parsed_row["downsample_schedule"] = _safe_int(
            row.get("downsample_schedule"), "downsample_schedule", default=2
        )
        parsed_row["downsample_op"] = str(row.get("downsample_op", "stride_conv"))
        parsed_row["encoder_conv_hidden"] = str(row.get("encoder_conv_hidden", "none"))
        parsed_row["encoder_spectral_hidden"] = str(row.get("encoder_spectral_hidden", "none"))
        parsed_row["max_hidden"] = str(row.get("max_hidden", "none"))
        parsed_row["resnet_width"] = str(row.get("resnet_width", "none"))
        parsed_row["resnet_blocks"] = _safe_int(row.get("resnet_blocks"), "resnet_blocks", default=6)
        parsed_row["skip_style"] = str(row.get("skip_style", "add"))

        parsed_row["pareto_rank_macro"] = _safe_float(
            row.get("pareto_rank_macro"),
            "pareto_rank_macro",
            default=9999.0,
        )
        if require_robust_fields:
            parsed_row["pareto_rank_seed3"] = _safe_float(row.get("pareto_rank_seed3"), "pareto_rank_seed3")
            parsed_row["pareto_rank_seed11"] = _safe_float(row.get("pareto_rank_seed11"), "pareto_rank_seed11")
            parsed_row["pareto_rank_seed17"] = _safe_float(row.get("pareto_rank_seed17"), "pareto_rank_seed17")
            parsed_row["pareto_rank_median"] = _safe_float(row.get("pareto_rank_median"), "pareto_rank_median")

        feasible, violations = _row_feasibility(parsed_row, args, args.ns)
        parsed_row["is_feasible"] = _bool_from_value(row.get("is_feasible", feasible)) and feasible
        parsed_row["violated_constraints"] = ";".join(violations)
        parsed.append(parsed_row)

    return parsed


def _resolve_single_anchor(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors = [dict(row) for row in rows if _bool_from_value(row.get("is_stage_anchor", False))]
    if len(anchors) == 1:
        return anchors[0]
    if len(anchors) > 1:
        raise PromotionSourceError(
            f"{path} resolves to multiple stage anchors ({len(anchors)}); expected exactly one"
        )
    if len(rows) == 1:
        return dict(rows[0])
    raise PromotionSourceError(
        f"{path} must resolve exactly one anchor row for stage-isolated N=128 runs"
    )


def _objective_tuple(row: Mapping[str, Any], objectives: Sequence[str]) -> tuple[float, ...]:
    values: list[float] = []
    for objective in objectives:
        values.append(_safe_float(row.get(objective), objective))
    return tuple(values)


def _dominates(a: Mapping[str, Any], b: Mapping[str, Any], objectives: Sequence[str]) -> bool:
    a_vals = _objective_tuple(a, objectives)
    b_vals = _objective_tuple(b, objectives)
    return all(x <= y for x, y in zip(a_vals, b_vals)) and any(x < y for x, y in zip(a_vals, b_vals))


def pareto_ranks(rows: Sequence[Mapping[str, Any]], objectives: Sequence[str]) -> dict[str, int]:
    """Compute non-dominated sorting rank for each row by run_id."""
    working = [dict(row) for row in rows]
    remaining = set(range(len(working)))
    rank = 1
    ranks: dict[str, int] = {}

    while remaining:
        front: list[int] = []
        for i in sorted(remaining):
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if _dominates(working[j], working[i], objectives):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for idx in front:
            ranks[str(working[idx]["run_id"])] = rank
        remaining.difference_update(front)
        rank += 1

    return ranks


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    rank_value = row.get("pareto_rank_median")
    if rank_value in (None, ""):
        rank_value = row.get("pareto_rank_macro", 9999.0)
    return (
        _safe_float(rank_value, "pareto_rank"),
        _safe_float(row.get("amp_mae"), "amp_mae"),
        _safe_float(row.get("model_params"), "model_params"),
        str(row.get("run_id", "")),
    )


def _candidate_config_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _safe_int(row.get("modes"), "modes", default=12),
        str(row.get("skip", "off")).lower(),
        _safe_int(row.get("width"), "width", default=32),
        _safe_int(row.get("fno_blocks"), "fno_blocks", default=4),
        _safe_int(row.get("downsample_schedule"), "downsample_schedule", default=2),
        _resolve_axis_value(row.get("downsample_op"), "stride_conv"),
        _resolve_axis_value(row.get("encoder_conv_hidden"), "none"),
        _resolve_axis_value(row.get("encoder_spectral_hidden"), "none"),
        _resolve_axis_value(row.get("max_hidden"), "none"),
        _resolve_axis_value(row.get("resnet_width"), "none"),
        _safe_int(row.get("resnet_blocks"), "resnet_blocks", default=6),
        _resolve_axis_value(row.get("skip_style"), "add"),
    )


def _dedupe_rows_by_config(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = _candidate_config_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _is_stage_a_control_anchor_row(row: Mapping[str, Any]) -> bool:
    try:
        return (
            _safe_int(row.get("modes"), "modes", default=12) == 12
            and str(row.get("skip", "off")).strip().lower() == "off"
            and _safe_int(row.get("width"), "width", default=32) == 32
            and _safe_int(row.get("fno_blocks"), "fno_blocks", default=4) == 4
            and _safe_int(row.get("downsample_schedule"), "downsample_schedule", default=2) == 2
            and _resolve_axis_value(row.get("downsample_op"), "stride_conv") == "stride_conv"
            and _resolve_axis_value(row.get("encoder_conv_hidden"), "none") == "none"
            and _resolve_axis_value(row.get("encoder_spectral_hidden"), "none") == "none"
            and _resolve_axis_value(row.get("max_hidden"), "none") == "none"
            and _resolve_axis_value(row.get("resnet_width"), "none") == "none"
            and _safe_int(row.get("resnet_blocks"), "resnet_blocks", default=6) == 6
            and _resolve_axis_value(row.get("skip_style"), "add") == "add"
        )
    except PromotionSourceError:
        return False


def _build_anchor_row_from_source(
    source_row: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": str(source_row.get("run_id", "stage_a_control_anchor")),
        "stage_id": str(source_row.get("stage_id", args.stage_id) or args.stage_id),
        "substage_id": str(source_row.get("substage_id", args.substage_id) or args.substage_id),
        "modes": _safe_int(source_row.get("modes"), "modes", default=12),
        "skip": str(source_row.get("skip", "off")).strip().lower(),
        "width": _safe_int(source_row.get("width"), "width", default=32),
        "fno_blocks": _safe_int(source_row.get("fno_blocks"), "fno_blocks", default=4),
        "downsample_schedule": _safe_int(source_row.get("downsample_schedule"), "downsample_schedule", default=2),
        "downsample_op": _resolve_axis_value(source_row.get("downsample_op"), "stride_conv"),
        "encoder_conv_hidden": _resolve_axis_value(source_row.get("encoder_conv_hidden"), "none"),
        "encoder_spectral_hidden": _resolve_axis_value(source_row.get("encoder_spectral_hidden"), "none"),
        "max_hidden": _resolve_axis_value(source_row.get("max_hidden"), "none"),
        "resnet_width": _resolve_axis_value(source_row.get("resnet_width"), "none"),
        "resnet_blocks": _safe_int(source_row.get("resnet_blocks"), "resnet_blocks", default=6),
        "skip_style": _resolve_axis_value(source_row.get("skip_style"), "add"),
        "amp_mae": _safe_float(source_row.get("amp_mae"), "amp_mae"),
        "amp_mse": _safe_float(source_row.get("amp_mse"), "amp_mse"),
        "train_wall_time_sec": _safe_float(source_row.get("train_wall_time_sec"), "train_wall_time_sec"),
        "inference_time_s": _safe_float(source_row.get("inference_time_s"), "inference_time_s"),
        "model_params": int(_safe_float(source_row.get("model_params"), "model_params")),
        "phase_ssim_drop_vs_baseline": _safe_float(
            source_row.get("phase_ssim_drop_vs_baseline"),
            "phase_ssim_drop_vs_baseline",
            default=0.0,
        ),
        "pareto_rank_seed3": "",
        "pareto_rank_seed11": "",
        "pareto_rank_seed17": "",
        "pareto_rank_median": "",
        "pareto_rank_macro": _safe_float(source_row.get("pareto_rank_macro"), "pareto_rank_macro", default=9999.0),
        "is_feasible": _bool_from_value(source_row.get("is_feasible", True)),
        "violated_constraints": str(source_row.get("violated_constraints", "")),
        "is_stage_anchor": True,
        "anchor_source_summary": str(args.source_summary),
    }


def _collect_seed_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.csv")):
        fieldnames, csv_rows = _read_csv(summary_path)
        _require_schema_version(summary_path, fieldnames, csv_rows)
        _require_columns(
            summary_path,
            fieldnames,
            [
                "run_id",
                "amp_mae",
                "amp_mse",
                "train_wall_time_sec",
                "inference_time_s",
                "model_params",
                "phase_ssim_drop_vs_baseline",
            ],
        )
        for raw_row in csv_rows:
            row = dict(raw_row)
            row["run_id"] = str(raw_row["run_id"])
            seed_value = raw_row.get("seed")
            if seed_value is None or str(seed_value).strip() == "":
                match = re.search(r"seed(\d+)", summary_path.as_posix())
                if not match:
                    raise PromotionSourceError(
                        f"Could not resolve seed for rerank row {summary_path}:{raw_row}"
                    )
                seed_value = match.group(1)
            row["seed"] = _safe_int(seed_value, "seed")
            row["modes"] = _safe_int(raw_row.get("modes"), "modes", default=12)
            row["skip"] = str(raw_row.get("skip", "off")).lower()
            row["width"] = _safe_int(raw_row.get("width"), "width", default=32)
            row["amp_mae"] = _safe_float(raw_row.get("amp_mae"), "amp_mae")
            row["amp_mse"] = _safe_float(raw_row.get("amp_mse"), "amp_mse")
            row["train_wall_time_sec"] = _safe_float(raw_row.get("train_wall_time_sec"), "train_wall_time_sec")
            row["inference_time_s"] = _safe_float(raw_row.get("inference_time_s"), "inference_time_s")
            row["model_params"] = _safe_float(raw_row.get("model_params"), "model_params")
            row["phase_ssim_drop_vs_baseline"] = _safe_float(
                raw_row.get("phase_ssim_drop_vs_baseline"),
                "phase_ssim_drop_vs_baseline",
                default=0.0,
            )
            rows.append(row)
    return rows


def run_seed_rerank_aggregation(args: argparse.Namespace) -> None:
    """Aggregate boundary seed reranks and emit robust promotion summaries."""
    source_rows = _load_promotion_rows(args.source_summary, args, require_robust_fields=False)

    feasible_source = [row for row in source_rows if _bool_from_value(row.get("is_feasible", False))]
    if not feasible_source:
        raise PromotionSourceError("No feasible candidates in source summary for seed rerank aggregation")

    ranked_source = sorted(feasible_source, key=_candidate_sort_key)
    ranked_unique = _dedupe_rows_by_config(ranked_source)
    boundary_count = len(ranked_unique)
    if args.top_k_n256 > 0:
        boundary_count = min(len(ranked_unique), args.top_k_n256 + 2)
    boundary = ranked_unique[:boundary_count]
    boundary_ids = {str(row["run_id"]) for row in boundary}

    rerank_rows = _collect_seed_rows(args.aggregate_seed_rerank_root)
    if not rerank_rows:
        raise PromotionSourceError(
            f"No seed rerank summary.csv rows found under {args.aggregate_seed_rerank_root}"
        )

    # Seed=3 comes from source summary by contract; boundary reruns provide seeds 11 and 17.
    rows_by_candidate: dict[str, dict[int, dict[str, Any]]] = {}
    for source_row in boundary:
        run_id = str(source_row["run_id"])
        rows_by_candidate[run_id] = {
            3: {
                "run_id": run_id,
                "seed": 3,
                "modes": int(source_row["modes"]),
                "skip": str(source_row["skip"]),
                "width": int(source_row["width"]),
                "amp_mae": float(source_row["amp_mae"]),
                "amp_mse": float(source_row["amp_mse"]),
                "train_wall_time_sec": float(source_row["train_wall_time_sec"]),
                "inference_time_s": float(source_row["inference_time_s"]),
                "model_params": float(source_row["model_params"]),
                "phase_ssim_drop_vs_baseline": float(source_row["phase_ssim_drop_vs_baseline"]),
            }
        }

    for row in rerank_rows:
        run_id = str(row["run_id"])
        if run_id not in boundary_ids:
            continue
        rows_by_candidate.setdefault(run_id, {})[int(row["seed"])] = row

    missing: list[str] = []
    for run_id in sorted(boundary_ids):
        seed_rows = rows_by_candidate.get(run_id, {})
        missing_seeds = [seed for seed in SEED_SET if seed not in seed_rows]
        if missing_seeds:
            missing.append(f"{run_id}:{missing_seeds}")
    if missing:
        raise PromotionSourceError(
            "Seed rerank aggregation missing required seeds {3,11,17} for boundary candidates: "
            + ", ".join(missing)
        )

    seed_ranks: dict[int, dict[str, int]] = {}
    for seed in SEED_SET:
        seed_rows = [rows_by_candidate[run_id][seed] for run_id in sorted(boundary_ids)]
        seed_ranks[seed] = pareto_ranks(seed_rows, args.promotion_objectives)

    robust_rows: list[dict[str, Any]] = []
    for source_row in boundary:
        run_id = str(source_row["run_id"])
        per_seed = rows_by_candidate[run_id]
        ranks = [seed_ranks[seed][run_id] for seed in SEED_SET]
        median_rank = sorted(ranks)[1]

        seed3_row = per_seed[3]
        robust_rows.append(
            {
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                "run_id": run_id,
                "stage_id": str(source_row.get("stage_id", args.stage_id) or args.stage_id),
                "substage_id": str(source_row.get("substage_id", args.substage_id) or args.substage_id),
                "modes": int(source_row["modes"]),
                "skip": str(source_row["skip"]),
                "width": int(source_row["width"]),
                "fno_blocks": _safe_int(source_row.get("fno_blocks"), "fno_blocks", default=4),
                "downsample_schedule": _safe_int(
                    source_row.get("downsample_schedule"), "downsample_schedule", default=2
                ),
                "downsample_op": str(source_row.get("downsample_op", "stride_conv")),
                "encoder_conv_hidden": str(source_row.get("encoder_conv_hidden", "none")),
                "encoder_spectral_hidden": str(source_row.get("encoder_spectral_hidden", "none")),
                "max_hidden": str(source_row.get("max_hidden", "none")),
                "resnet_width": str(source_row.get("resnet_width", "none")),
                "resnet_blocks": _safe_int(source_row.get("resnet_blocks"), "resnet_blocks", default=6),
                "skip_style": str(source_row.get("skip_style", "add")),
                "amp_mae": float(seed3_row["amp_mae"]),
                "amp_mse": float(seed3_row["amp_mse"]),
                "train_wall_time_sec": float(seed3_row["train_wall_time_sec"]),
                "inference_time_s": float(seed3_row["inference_time_s"]),
                "model_params": int(float(seed3_row["model_params"])),
                "phase_ssim_drop_vs_baseline": float(seed3_row["phase_ssim_drop_vs_baseline"]),
                "pareto_rank_seed3": int(seed_ranks[3][run_id]),
                "pareto_rank_seed11": int(seed_ranks[11][run_id]),
                "pareto_rank_seed17": int(seed_ranks[17][run_id]),
                "pareto_rank_median": int(median_rank),
                "pareto_rank_macro": int(median_rank),
                "is_feasible": True,
                "violated_constraints": "",
                "is_stage_anchor": False,
                "anchor_source_summary": str(args.source_summary),
            }
        )

    robust_rows.sort(key=_candidate_sort_key)

    if not robust_rows:
        raise PromotionSourceError("Seed rerank aggregation produced no robust rows")

    args.emit_robust_promotion_summary.parent.mkdir(parents=True, exist_ok=True)
    robust_fieldnames = [
        "summary_schema_version",
        "run_id",
        "stage_id",
        "substage_id",
        "modes",
        "skip",
        "width",
        "fno_blocks",
        "downsample_schedule",
        "downsample_op",
        "encoder_conv_hidden",
        "encoder_spectral_hidden",
        "max_hidden",
        "resnet_width",
        "resnet_blocks",
        "skip_style",
        "amp_mae",
        "amp_mse",
        "train_wall_time_sec",
        "inference_time_s",
        "model_params",
        "phase_ssim_drop_vs_baseline",
        "pareto_rank_seed3",
        "pareto_rank_seed11",
        "pareto_rank_seed17",
        "pareto_rank_median",
        "pareto_rank_macro",
        "is_feasible",
        "violated_constraints",
        "is_stage_anchor",
        "anchor_source_summary",
    ]
    with args.emit_robust_promotion_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=robust_fieldnames)
        writer.writeheader()
        writer.writerows(robust_rows)

    if args.stage_id == "A":
        control_matches = [dict(row) for row in source_rows if _is_stage_a_control_anchor_row(row)]
        if not control_matches:
            raise PromotionSourceError(
                "Stage A seed rerank aggregation requires a true-default control anchor "
                "(modes=12, skip=off, width=32, fno_blocks=4, downsample_schedule=2, "
                "downsample_op=stride_conv, encoder_conv_hidden=none, "
                "encoder_spectral_hidden=none, max_hidden=none, resnet_width=none, "
                "resnet_blocks=6, skip_style=add) in --source-summary"
            )
        unique_control_configs = _dedupe_rows_by_config(control_matches)
        if len(unique_control_configs) != 1:
            raise PromotionSourceError(
                "Stage A source summary contains multiple distinct control-anchor tuples; "
                "expected exactly one true-default tuple"
            )
        control_matches.sort(
            key=lambda row: (
                _safe_float(row.get("amp_mae"), "amp_mae", default=1e12),
                str(row.get("run_id", "")),
            )
        )
        anchor_row = _build_anchor_row_from_source(control_matches[0], args=args)
        for robust_row in robust_rows:
            if _candidate_config_key(robust_row) == _candidate_config_key(anchor_row):
                for key in (
                    "pareto_rank_seed3",
                    "pareto_rank_seed11",
                    "pareto_rank_seed17",
                    "pareto_rank_median",
                    "pareto_rank_macro",
                ):
                    anchor_row[key] = robust_row.get(key, anchor_row.get(key, ""))
                break
    else:
        anchor_row = dict(robust_rows[0])
    anchor_row["is_stage_anchor"] = True
    args.emit_stage_anchor_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.emit_stage_anchor_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=robust_fieldnames)
        writer.writeheader()
        writer.writerow(anchor_row)


def _resolve_axis_value(raw: str | int | float | None, default: str) -> str:
    if raw is None:
        return default
    text = str(raw).strip()
    return text if text else default


def _select_scalar(value_list: Sequence[Any]) -> Any:
    return value_list[0] if value_list else None


def _build_stage_a_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    profiles = _active_profiles(args)
    candidates: list[dict[str, Any]] = []
    for mode, skip, width, profile in itertools.product(args.modes, args.skip_values, args.widths, profiles):
        skip_flag = str(skip).lower()
        run_key = f"stage{args.stage_id}_n{args.ns}_profile-{profile}_m{mode}_s{skip_flag}_w{width}"
        candidates.append(
            {
                "run_key": run_key,
                "modes": int(mode),
                "skip": skip_flag,
                "width": int(width),
                "fno_blocks": int(_select_scalar(args.fno_blocks_values) or 4),
                "downsample_schedule": int(_select_scalar(args.downsample_schedule_values) or 2),
                "downsample_op": str(_select_scalar(args.downsample_op_values) or "stride_conv"),
                "encoder_conv_hidden": str(_select_scalar(args.encoder_conv_hidden_values) or "none"),
                "encoder_spectral_hidden": str(
                    _select_scalar(args.encoder_spectral_hidden_values) or "none"
                ),
                "max_hidden": str(_select_scalar(args.max_hidden_values) or "none"),
                "resnet_width": str(_select_scalar(args.resnet_width_values) or "none"),
                "resnet_blocks": int(_select_scalar(args.resnet_blocks_values) or 6),
                "skip_style": str(_select_scalar(args.skip_style_values) or "add"),
                "dataset_profile": profile,
            }
        )
    return candidates


def _active_axis_values(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.stage_id == "B":
        return [{"fno_blocks": int(v)} for v in args.fno_blocks_values]
    if args.stage_id == "C" and args.substage_id == "C1":
        return [{"downsample_schedule": int(v)} for v in args.downsample_schedule_values]
    if args.stage_id == "C" and args.substage_id == "C2":
        return [{"downsample_op": str(v)} for v in args.downsample_op_values]
    if args.stage_id == "D" and args.substage_id == "D1":
        return [{"encoder_conv_hidden": str(v)} for v in args.encoder_conv_hidden_values]
    if args.stage_id == "D" and args.substage_id == "D2":
        return [{"encoder_spectral_hidden": str(v)} for v in args.encoder_spectral_hidden_values]
    if args.stage_id == "D" and args.substage_id == "D3":
        values: list[dict[str, Any]] = []
        max_hidden_multi = len(args.max_hidden_values) > 1
        resnet_width_multi = len(args.resnet_width_values) > 1
        if max_hidden_multi and resnet_width_multi:
            raise MatrixExpansionError(
                "Stage D3 cannot vary max_hidden and resnet_width together in one invocation"
            )
        if max_hidden_multi:
            values = [{"max_hidden": str(v)} for v in args.max_hidden_values]
        elif resnet_width_multi:
            values = [{"resnet_width": str(v)} for v in args.resnet_width_values]
        else:
            values = [{
                "max_hidden": str(_select_scalar(args.max_hidden_values) or "none"),
                "resnet_width": str(_select_scalar(args.resnet_width_values) or "none"),
            }]
        return values
    if args.stage_id == "D" and args.substage_id == "D4":
        return [{"resnet_blocks": int(v)} for v in args.resnet_blocks_values]
    if args.stage_id == "E":
        return [{"skip_style": str(v)} for v in args.skip_style_values]
    return [{}]


def _build_stage_b_to_e_candidates(
    args: argparse.Namespace,
    *,
    anchor_row: Mapping[str, Any] | None,
    promoted_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    profiles = _active_profiles(args)
    candidates: list[dict[str, Any]] = []

    if args.ns == 128:
        if anchor_row is None:
            raise PromotionSourceError("Stage B-E N=128 sweeps require a single promotion anchor row")
        base = {
            "modes": _safe_int(anchor_row.get("modes"), "modes"),
            "skip": str(anchor_row.get("skip", "off")).lower(),
            "width": _safe_int(anchor_row.get("width"), "width"),
            "fno_blocks": _safe_int(anchor_row.get("fno_blocks"), "fno_blocks", default=4),
            "downsample_schedule": _safe_int(
                anchor_row.get("downsample_schedule"), "downsample_schedule", default=2
            ),
            "downsample_op": _resolve_axis_value(anchor_row.get("downsample_op"), "stride_conv"),
            "encoder_conv_hidden": _resolve_axis_value(anchor_row.get("encoder_conv_hidden"), "none"),
            "encoder_spectral_hidden": _resolve_axis_value(
                anchor_row.get("encoder_spectral_hidden"), "none"
            ),
            "max_hidden": _resolve_axis_value(anchor_row.get("max_hidden"), "none"),
            "resnet_width": _resolve_axis_value(anchor_row.get("resnet_width"), "none"),
            "resnet_blocks": _safe_int(anchor_row.get("resnet_blocks"), "resnet_blocks", default=6),
            "skip_style": _resolve_axis_value(anchor_row.get("skip_style"), "add"),
        }

        for profile, axis_values in itertools.product(profiles, _active_axis_values(args)):
            row = dict(base)
            row.update(axis_values)
            row["dataset_profile"] = profile
            row["run_key"] = (
                f"stage{args.stage_id}{args.substage_id}_n{args.ns}_profile-{profile}_"
                f"m{row['modes']}_s{row['skip']}_w{row['width']}_"
                f"fb{row['fno_blocks']}_ds{row['downsample_schedule']}_{row['downsample_op']}_"
                f"ec{row['encoder_conv_hidden']}_es{row['encoder_spectral_hidden']}_"
                f"mh{row['max_hidden']}_rw{row['resnet_width']}_rb{row['resnet_blocks']}_"
                f"ss{row['skip_style']}"
            )
            candidates.append(row)
        return candidates

    # N=256: consume promoted rows from source summary.
    if args.top_k_n256 == 0 and args.allow_n256_direct_diagnostic:
        return _build_stage_a_candidates(args)

    if not promoted_rows:
        raise PromotionSourceError("Promotion-enabled N=256 invocation has no promoted rows")

    for source in promoted_rows:
        for profile in profiles:
            row = {
                "run_key": f"{source['run_id']}_n256_profile-{profile}",
                "modes": _safe_int(source.get("modes"), "modes"),
                "skip": str(source.get("skip", "off")).lower(),
                "width": _safe_int(source.get("width"), "width"),
                "fno_blocks": _safe_int(source.get("fno_blocks"), "fno_blocks", default=4),
                "downsample_schedule": _safe_int(
                    source.get("downsample_schedule"), "downsample_schedule", default=2
                ),
                "downsample_op": _resolve_axis_value(source.get("downsample_op"), "stride_conv"),
                "encoder_conv_hidden": _resolve_axis_value(source.get("encoder_conv_hidden"), "none"),
                "encoder_spectral_hidden": _resolve_axis_value(
                    source.get("encoder_spectral_hidden"), "none"
                ),
                "max_hidden": _resolve_axis_value(source.get("max_hidden"), "none"),
                "resnet_width": _resolve_axis_value(source.get("resnet_width"), "none"),
                "resnet_blocks": _safe_int(source.get("resnet_blocks"), "resnet_blocks", default=6),
                "skip_style": _resolve_axis_value(source.get("skip_style"), "add"),
                "dataset_profile": profile,
            }
            candidates.append(row)
    return candidates


def _build_candidates(
    args: argparse.Namespace,
    *,
    anchor_row: Mapping[str, Any] | None,
    promoted_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if args.stage_id == "A":
        if args.ns == 256 and args.top_k_n256 > 0:
            # Promotion-driven N=256 Stage A pass.
            if not promoted_rows:
                raise PromotionSourceError("Stage A N=256 promotion run requires promoted source rows")
            return _build_stage_b_to_e_candidates(args, anchor_row=None, promoted_rows=promoted_rows)
        return _build_stage_a_candidates(args)
    return _build_stage_b_to_e_candidates(args, anchor_row=anchor_row, promoted_rows=promoted_rows)


def _resolve_profile_npz_inputs(
    args: argparse.Namespace,
    profile: str,
    cache: dict[str, tuple[Path, Path]],
) -> tuple[Path, Path]:
    cached = cache.get(profile)
    if cached:
        return cached

    if profile == "integration_grid_lines_n128_v1":
        from ptycho.workflows.grid_lines_workflow import GridLinesConfig
        from scripts.studies.grid_study_dataset_builder import build_datasets

        profile_dir = args.output_root / "datasets" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        train_npz = profile_dir / "train.npz"
        test_npz = profile_dir / "test.npz"
        if not train_npz.exists() or not test_npz.exists():
            workspace = profile_dir / "builder_workspace"
            cfg = GridLinesConfig(
                N=int(args.ns),
                gridsize=1,
                output_dir=workspace,
                probe_npz=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
                nimgs_train=2,
                nimgs_test=2,
                nphotons=1e9,
                nepochs=1,
                batch_size=8,
                probe_source="custom",
                probe_scale_mode="pad_extrapolate",
                set_phi=True,
            )
            bundles = build_datasets(
                dataset_source="synthetic_lines",
                cfg=cfg,
                required_ns=[int(args.ns)],
            )
            built_train = Path(bundles[int(args.ns)]["train_npz"])
            built_test = Path(bundles[int(args.ns)]["test_npz"])
            shutil.copy2(built_train, train_npz)
            shutil.copy2(built_test, test_npz)
        cache[profile] = (train_npz, test_npz)
        return cache[profile]

    if profile == "fly001_external_n128_top_bottom_v1":
        cache[profile] = (Path(args.fly001_external_train_npz), Path(args.fly001_external_test_npz))
        return cache[profile]

    if profile == "custom_npz_pair_n128":
        cache[profile] = (Path(args.custom_n128_train_npz), Path(args.custom_n128_test_npz))
        return cache[profile]

    if profile == "cameraman256_halfsplit_v1":
        from ptycho.workflows.grid_lines_workflow import GridLinesConfig
        from scripts.studies.grid_study_dataset_builder import build_datasets
        from scripts.studies.prepare_nersc_hybrid_dataset import prepare_hybrid_dataset

        profile_dir = args.output_root / "datasets" / profile
        prepared = prepare_hybrid_dataset(
            dp_h5=Path(args.cameraman_dp),
            para_h5=Path(args.cameraman_para),
            output_dir=profile_dir,
            half="top",
            target_n=int(args.ns),
            downsample_policy="bin-crop",
        )

        cfg = GridLinesConfig(
            N=int(args.ns),
            gridsize=1,
            output_dir=profile_dir / "canonical_cache",
            probe_npz=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
            nimgs_train=2,
            nimgs_test=2,
            nphotons=1e9,
            nepochs=1,
            batch_size=8,
            probe_source="custom",
            probe_scale_mode="pad_extrapolate",
            set_phi=True,
        )
        bundles = build_datasets(
            dataset_source="external_raw_npz",
            cfg=cfg,
            required_ns=[int(args.ns)],
            train_data=Path(prepared["train_npz"]),
            test_data=Path(prepared["test_npz"]),
            n_groups=512,
            neighbor_count=7,
            subsample_seed=int(args.seed),
        )
        bundle = bundles[int(args.ns)]
        cache[profile] = (Path(bundle["train_npz"]), Path(bundle["test_npz"]))
        return cache[profile]

    if profile == "custom_npz_pair_n256":
        cache[profile] = (Path(args.custom_n256_train_npz), Path(args.custom_n256_test_npz))
        return cache[profile]

    raise StageValidationError(f"Unsupported dataset profile for runner execution: {profile}")


def _extract_metric_component(raw: Any, *, index: int, key: str) -> float:
    value = raw
    if isinstance(raw, (list, tuple)):
        if index >= len(raw):
            raise StageValidationError(
                f"metrics[{key}] missing component index {index}; got {raw!r}"
            )
        value = raw[index]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise StageValidationError(f"metrics[{key}] component {index} is non-numeric: {value!r}") from exc


def _extract_scalar_from_log(log_path: Path, key: str) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    match = re.search(rf'"{re.escape(key)}"\s*:\s*([-+0-9.eE]+)', text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _count_model_params_from_state_dict(model_path: Path) -> int:
    if not model_path.exists():
        return 0
    try:
        import torch
    except Exception:
        return 0
    try:
        state = torch.load(model_path, map_location="cpu")
    except Exception:
        return 0
    if not isinstance(state, dict):
        return 0
    total = 0
    for tensor in state.values():
        if hasattr(tensor, "numel"):
            total += int(tensor.numel())
    return int(total)


def _run_candidate_with_runner(
    *,
    args: argparse.Namespace,
    candidate: Mapping[str, Any],
    run_dir: Path,
    train_npz: Path,
    test_npz: Path,
) -> dict[str, float]:
    logs_dir = run_dir / "driver_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / "runner_stdout.log"
    stderr_log = logs_dir / "runner_stderr.log"

    epochs = int(args.epochs_n256 if args.ns == 256 else args.epochs_n128)
    cmd = [
        "python",
        "scripts/studies/grid_lines_torch_runner.py",
        "--train-npz",
        str(train_npz),
        "--test-npz",
        str(test_npz),
        "--output-dir",
        str(run_dir),
        "--architecture",
        "hybrid_resnet",
        "--seed",
        str(int(args.seed)),
        "--epochs",
        str(epochs),
        "--N",
        str(int(args.ns)),
        "--gridsize",
        "1",
        "--fno-modes",
        str(int(candidate["modes"])),
        "--fno-width",
        str(int(candidate["width"])),
        "--fno-blocks",
        str(int(candidate["fno_blocks"])),
    ]

    if _bool_from_value(candidate.get("skip", "off")):
        cmd.append("--hybrid-skip-connections")
    else:
        cmd.append("--no-hybrid-skip-connections")
    cmd.extend(
        [
            "--hybrid-downsample-steps",
            str(_safe_int(candidate.get("downsample_schedule"), "downsample_schedule", default=2)),
            "--hybrid-downsample-op",
            _resolve_axis_value(candidate.get("downsample_op"), "stride_conv"),
        ]
    )
    encoder_conv_hidden_raw = _resolve_axis_value(candidate.get("encoder_conv_hidden"), "none")
    if encoder_conv_hidden_raw.lower() != "none":
        cmd.extend(
            [
                "--hybrid-encoder-conv-hidden",
                str(_safe_int(encoder_conv_hidden_raw, "encoder_conv_hidden")),
            ]
        )
    encoder_spectral_hidden_raw = _resolve_axis_value(candidate.get("encoder_spectral_hidden"), "none")
    if encoder_spectral_hidden_raw.lower() != "none":
        cmd.extend(
            [
                "--hybrid-encoder-spectral-hidden",
                str(_safe_int(encoder_spectral_hidden_raw, "encoder_spectral_hidden")),
            ]
        )
    cmd.extend(
        [
            "--hybrid-resnet-blocks",
            str(_safe_int(candidate.get("resnet_blocks"), "resnet_blocks", default=6)),
            "--hybrid-skip-style",
            _resolve_axis_value(candidate.get("skip_style"), "add"),
        ]
    )

    resnet_width_raw = str(candidate.get("resnet_width", "none")).strip().lower()
    if resnet_width_raw != "none":
        cmd.extend(
            [
                "--torch-resnet-width",
                str(_safe_int(resnet_width_raw, "resnet_width")),
            ]
        )

    if bool(args.probe_mask):
        cmd.append("--probe-mask")
    else:
        cmd.append("--no-probe-mask")

    if bool(args.torch_mae_pred_l2_match_target):
        cmd.append("--torch-mae-pred-l2-match-target")
    else:
        cmd.append("--no-torch-mae-pred-l2-match-target")

    started = time.perf_counter()
    with stdout_log.open("w") as stdout_handle, stderr_log.open("w") as stderr_handle:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    elapsed = float(time.perf_counter() - started)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Runner failed for {candidate['run_key']} with exit={completed.returncode}. "
            f"See {stdout_log} and {stderr_log}."
        )

    run_metrics_path = run_dir / "runs" / "pinn_hybrid_resnet" / "metrics.json"
    if not run_metrics_path.exists():
        raise RuntimeError(
            f"Expected runner metrics file missing for {candidate['run_key']}: {run_metrics_path}"
        )
    run_metrics = json.loads(run_metrics_path.read_text())
    amp_mae = _extract_metric_component(run_metrics.get("mae"), index=0, key="mae")
    amp_mse = _extract_metric_component(run_metrics.get("mse"), index=0, key="mse")
    phase_ssim = _extract_metric_component(run_metrics.get("ssim"), index=1, key="ssim")

    inference_time = _extract_scalar_from_log(stdout_log, "inference_time_s")
    if inference_time is None:
        inference_time = 0.0
    model_params = _extract_scalar_from_log(stdout_log, "model_params")
    if model_params is None:
        model_params = float(
            _count_model_params_from_state_dict(run_dir / "runs" / "pinn_hybrid_resnet" / "model.pt")
        )

    # Baseline metrics are not emitted by this runner path; keep an explicit zero delta.
    phase_drop = 0.0

    return {
        "amp_mae": float(amp_mae),
        "amp_mse": float(amp_mse),
        "phase_ssim": float(phase_ssim),
        "phase_ssim_drop_vs_baseline": float(phase_drop),
        "model_params": int(model_params),
        "train_wall_time_sec": float(elapsed),
        "inference_time_s": float(inference_time),
    }


def _write_cleanup_report(
    run_dir: Path,
    *,
    retention_tier: str,
    deleted_paths: list[str],
    bytes_reclaimed: int,
) -> None:
    payload = {
        "retention_tier": retention_tier,
        "deleted_paths": deleted_paths,
        "bytes_reclaimed": int(bytes_reclaimed),
    }
    (run_dir / "cleanup_report.json").write_text(json.dumps(payload, indent=2) + "\n")


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return int(total)


def _prune_heavy_runtime_artifacts(run_dir: Path) -> tuple[list[str], int]:
    runtime_root = run_dir / "runs" / "pinn_hybrid_resnet"
    if not runtime_root.exists():
        return [], 0

    heavy_dirs = (
        "checkpoints",
        "lightning_logs",
        "mlruns",
        "tb_logs",
        "tensorboard",
        "profiler",
        "cache",
    )
    heavy_file_patterns = (
        "*.ckpt",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.npy",
        "*.npz",
        "*.h5",
        "*.hdf5",
        "events.out.tfevents*",
    )

    dir_targets: list[Path] = []
    for dir_name in heavy_dirs:
        candidate = runtime_root / dir_name
        if candidate.exists() and candidate.is_dir():
            dir_targets.append(candidate)
    dir_target_set = {path.resolve() for path in dir_targets}

    file_targets: list[Path] = []
    for pattern in heavy_file_patterns:
        for candidate in runtime_root.rglob(pattern):
            if not candidate.is_file():
                continue
            candidate_resolved = candidate.resolve()
            if any(parent in dir_target_set for parent in candidate_resolved.parents):
                continue
            file_targets.append(candidate)

    targets = sorted(
        {*dir_targets, *file_targets},
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    )

    deleted_paths: list[str] = []
    bytes_reclaimed = 0
    for target in targets:
        if not target.exists():
            continue
        bytes_reclaimed += _path_size_bytes(target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        try:
            deleted_paths.append(str(target.relative_to(run_dir)))
        except ValueError:
            deleted_paths.append(str(target))

    return deleted_paths, int(bytes_reclaimed)


def _candidate_tuple_key(args: argparse.Namespace, row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(args.stage_id),
        str(args.substage_id),
        int(args.ns),
        str(row["dataset_profile"]),
    )


def run_sweep(args: argparse.Namespace, *, argv_payload: Sequence[str]) -> int:
    """Execute matrix expansion and emit summary artifacts."""
    if args.output_root is None:
        raise StageValidationError("--output-root is required for sweep execution")

    args.output_root.mkdir(parents=True, exist_ok=True)

    write_invocation_artifacts(
        output_dir=args.output_root,
        script_path=__file__,
        argv=argv_payload,
        parsed_args=vars(args),
    )

    source_rows: list[dict[str, Any]] = []
    anchor_row: dict[str, Any] | None = None
    promoted_rows: list[dict[str, Any]] = []

    if args.promotion_source_summary:
        require_robust = bool(args.ns == 256 and args.top_k_n256 > 0)
        source_rows = _load_promotion_rows(
            args.promotion_source_summary,
            args,
            require_robust_fields=require_robust,
        )

        feasible_source = [row for row in source_rows if _bool_from_value(row.get("is_feasible", False))]
        if not feasible_source:
            raise PromotionSourceError(
                f"Promotion source {args.promotion_source_summary} has no feasible candidates"
            )

        if args.stage_id in {"B", "C", "D", "E"} and args.ns == 128:
            anchor_row = _resolve_single_anchor(args.promotion_source_summary, feasible_source)

        if args.ns == 256 and args.top_k_n256 > 0:
            ranked = sorted(feasible_source, key=_candidate_sort_key)
            promoted_rows = _dedupe_rows_by_config(ranked)[: args.top_k_n256]

    candidates = _build_candidates(args, anchor_row=anchor_row, promoted_rows=promoted_rows)
    if not candidates:
        raise MatrixExpansionError("Sweep matrix expansion produced zero candidates")

    rows: list[dict[str, Any]] = []
    seen_retention_keys: set[tuple[str, str, int, str]] = set()
    dataset_cache: dict[str, tuple[Path, Path]] = {}
    for profile in sorted({str(candidate["dataset_profile"]) for candidate in candidates}):
        _resolve_profile_npz_inputs(args, profile, dataset_cache)

    manifest = build_sweep_manifest(args)
    (args.output_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for index, candidate in enumerate(candidates):
        run_id = str(candidate["run_key"])
        run_dir = args.output_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_npz, test_npz = _resolve_profile_npz_inputs(
            args,
            str(candidate["dataset_profile"]),
            dataset_cache,
        )
        metrics = _run_candidate_with_runner(
            args=args,
            candidate=candidate,
            run_dir=run_dir,
            train_npz=train_npz,
            test_npz=test_npz,
        )
        feasible, violations = _row_feasibility(metrics, args, args.ns)

        # Create per-run payload.
        metrics_payload = {
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": run_id,
            "stage_id": args.stage_id,
            "substage_id": args.substage_id,
            "dataset_profile": candidate["dataset_profile"],
            **metrics,
            "is_feasible": feasible,
            "violated_constraints": violations,
            "probe_mask_enabled": bool(args.probe_mask),
            "torch_mae_pred_l2_match_target": bool(args.torch_mae_pred_l2_match_target),
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n")

        tuple_key = _candidate_tuple_key(args, candidate)
        if tuple_key not in seen_retention_keys:
            retention_tier = "full_anchor"
            seen_retention_keys.add(tuple_key)
            deleted: list[str] = []
            reclaimed = 0
        else:
            retention_tier = "pruned"
            deleted, reclaimed = [], 0
            if args.prune_heavy_artifacts:
                deleted, reclaimed = _prune_heavy_runtime_artifacts(run_dir)

        _write_cleanup_report(
            run_dir,
            retention_tier=retention_tier,
            deleted_paths=deleted,
            bytes_reclaimed=reclaimed,
        )

        row = {
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": run_id,
            "stage_id": args.stage_id,
            "substage_id": args.substage_id,
            "N": int(args.ns),
            "dataset_profile": str(candidate["dataset_profile"]),
            "modes": int(candidate["modes"]),
            "skip": str(candidate["skip"]),
            "width": int(candidate["width"]),
            "fno_blocks": int(candidate["fno_blocks"]),
            "downsample_schedule": int(candidate["downsample_schedule"]),
            "downsample_op": str(candidate["downsample_op"]),
            "encoder_conv_hidden": str(candidate["encoder_conv_hidden"]),
            "encoder_spectral_hidden": str(candidate["encoder_spectral_hidden"]),
            "max_hidden": str(candidate["max_hidden"]),
            "resnet_width": str(candidate["resnet_width"]),
            "resnet_blocks": int(candidate["resnet_blocks"]),
            "skip_style": str(candidate["skip_style"]),
            "amp_mae": float(metrics["amp_mae"]),
            "amp_mse": float(metrics["amp_mse"]),
            "phase_ssim_drop_vs_baseline": float(metrics["phase_ssim_drop_vs_baseline"]),
            "max_phase_ssim_drop": float(args.max_phase_ssim_drop),
            "phase_guardrail_pass": bool(metrics["phase_ssim_drop_vs_baseline"] <= args.max_phase_ssim_drop),
            "model_params": int(metrics["model_params"]),
            "train_wall_time_sec": float(metrics["train_wall_time_sec"]),
            "inference_time_s": float(metrics["inference_time_s"]),
            "is_feasible": bool(feasible),
            "violated_constraints": ";".join(violations),
            "probe_mask_enabled": bool(args.probe_mask),
            "torch_mae_pred_l2_match_target": bool(args.torch_mae_pred_l2_match_target),
            "pareto_rank_profile": "",
            "pareto_rank_macro": "",
            "pareto_rank_seed3": "",
            "pareto_rank_seed11": "",
            "pareto_rank_seed17": "",
            "pareto_rank_median": "",
            "promote_to_n256": False,
            "is_stage_anchor": False,
            "anchor_source_summary": str(args.promotion_source_summary or ""),
            "retention_tier": retention_tier,
            "run_index": index,
        }
        rows.append(row)

    # Per-profile Pareto ranks.
    profiles = sorted({str(row["dataset_profile"]) for row in rows})
    for profile in profiles:
        profile_rows = [row for row in rows if row["dataset_profile"] == profile]
        profile_ranks = pareto_ranks(profile_rows, args.promotion_objectives)
        for row in profile_rows:
            row["pareto_rank_profile"] = int(profile_ranks[row["run_id"]])

    # Macro rank (median profile rank per candidate key).
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        candidate_key = f"{row['modes']}|{row['skip']}|{row['width']}|{row['fno_blocks']}|{row['downsample_schedule']}|{row['downsample_op']}|{row['encoder_conv_hidden']}|{row['encoder_spectral_hidden']}|{row['max_hidden']}|{row['resnet_width']}|{row['resnet_blocks']}|{row['skip_style']}"
        by_candidate.setdefault(candidate_key, []).append(row)

    macro_items: list[tuple[str, float, float]] = []
    for key, candidate_rows in by_candidate.items():
        profile_ranks = sorted(int(r["pareto_rank_profile"]) for r in candidate_rows)
        median_rank = profile_ranks[len(profile_ranks) // 2]
        mean_amp = sum(float(r["amp_mae"]) for r in candidate_rows) / len(candidate_rows)
        macro_items.append((key, float(median_rank), float(mean_amp)))

    macro_items.sort(key=lambda item: (item[1], item[2], item[0]))
    macro_rank_map = {key: idx + 1 for idx, (key, _, _) in enumerate(macro_items)}

    for key, candidate_rows in by_candidate.items():
        rank = macro_rank_map[key]
        for row in candidate_rows:
            row["pareto_rank_macro"] = int(rank)

    feasible_rows = [row for row in rows if _bool_from_value(row["is_feasible"])]
    feasible_rows.sort(key=lambda row: (float(row["pareto_rank_macro"]), float(row["amp_mae"]), row["run_id"]))

    # Promotion and stage-anchor flags.
    if feasible_rows:
        feasible_rows[0]["is_stage_anchor"] = True
        if args.ns == 128 and args.top_k_n256 > 0:
            for row in feasible_rows[: args.top_k_n256]:
                row["promote_to_n256"] = True

    summary_fieldnames = [
        "summary_schema_version",
        "run_id",
        "stage_id",
        "substage_id",
        "N",
        "dataset_profile",
        "modes",
        "skip",
        "width",
        "fno_blocks",
        "downsample_schedule",
        "downsample_op",
        "encoder_conv_hidden",
        "encoder_spectral_hidden",
        "max_hidden",
        "resnet_width",
        "resnet_blocks",
        "skip_style",
        "amp_mae",
        "amp_mse",
        "phase_ssim_drop_vs_baseline",
        "max_phase_ssim_drop",
        "phase_guardrail_pass",
        "model_params",
        "train_wall_time_sec",
        "inference_time_s",
        "is_feasible",
        "violated_constraints",
        "pareto_rank_profile",
        "pareto_rank_macro",
        "pareto_rank_seed3",
        "pareto_rank_seed11",
        "pareto_rank_seed17",
        "pareto_rank_median",
        "promote_to_n256",
        "is_stage_anchor",
        "anchor_source_summary",
        "probe_mask_enabled",
        "torch_mae_pred_l2_match_target",
        "retention_tier",
    ]

    rows.sort(key=lambda row: int(row["run_index"]))

    with (args.output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = {key: row.get(key, "") for key in summary_fieldnames}
            writer.writerow(row_copy)

    feasible_count = sum(1 for row in rows if _bool_from_value(row["is_feasible"]))
    with (args.output_root / "summary.md").open("w") as f:
        f.write("# Hybrid ResNet Mode/Skip Sweep Summary\n\n")
        f.write(f"- Stage: {args.stage_id}\n")
        f.write(f"- Substage: {args.substage_id}\n")
        f.write(f"- stage_id: {args.stage_id}\n")
        f.write(f"- substage_id: {args.substage_id}\n")
        f.write(f"- Resolution: N={args.ns}\n")
        f.write(f"- Candidates: {len(rows)}\n")
        f.write(f"- Feasible: {feasible_count}\n")
        f.write(f"- Summary Schema: {SUMMARY_SCHEMA_VERSION}\n")
        f.write(f"- summary_schema_version: {SUMMARY_SCHEMA_VERSION}\n")

        f.write("\n## Candidates\n\n")
        f.write("| run_id | dataset_profile | modes | skip | width | fno_blocks | amp_mae | is_feasible |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['run_id']} | {row['dataset_profile']} | {row['modes']} | {row['skip']} | "
                f"{row['width']} | {row['fno_blocks']} | {float(row['amp_mae']):.6f} | {row['is_feasible']} |\n"
            )

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
        validate_stage_configuration(args)
        validate_matrix_constraints(args)

        if args.aggregate_seed_rerank_root:
            run_seed_rerank_aggregation(args)
            return 0
        return run_sweep(args, argv_payload=list(argv) if argv is not None else sys.argv[1:])

    except (StageValidationError, PromotionSourceError, MatrixExpansionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

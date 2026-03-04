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
import math
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
MIN_STAGE_EPOCHS = 10
DEFAULT_PROMOTION_OBJECTIVES = ("amp_ssim", "train_wall_time_sec")
REQUIRED_PROMOTION_OBJECTIVES_BY_STAGE = {
    stage_id: DEFAULT_PROMOTION_OBJECTIVES for stage_id in ("A", "B", "C", "D", "E")
}
CANONICAL_N256_DATASET_PROFILES = (
    "cameraman256_halfsplit_v1",
    "custom_npz_pair_n256",
)
OBJECTIVE_DIRECTIONS = {
    "amp_ssim": "max",
    "phase_ssim": "max",
    "amp_mae": "min",
    "amp_mse": "min",
    "train_wall_time_sec": "min",
    "inference_time_s": "min",
    "model_params": "min",
    "phase_ssim_drop_vs_baseline": "min",
}


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


def _parse_float_csv(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


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


def _compute_phase_drop(*, phase_ssim: float, baseline_phase_ssim: float) -> float:
    """Compute non-negative phase-SSIM drop relative to baseline provenance."""
    return max(0.0, float(baseline_phase_ssim) - float(phase_ssim))


def _is_default_scale_list(values: Sequence[float]) -> bool:
    return len(values) == 1 and math.isclose(float(values[0]), 1.0, rel_tol=0.0, abs_tol=1e-12)


def _uses_legacy_conv_hidden_axis(args: argparse.Namespace) -> bool:
    values = [str(value).strip().lower() for value in args.encoder_conv_hidden_values]
    return _is_default_scale_list(args.encoder_conv_hidden_scale_values) and (
        len(values) > 1 or (len(values) == 1 and values[0] != "none")
    )


def _uses_legacy_spectral_hidden_axis(args: argparse.Namespace) -> bool:
    values = [str(value).strip().lower() for value in args.encoder_spectral_hidden_values]
    return _is_default_scale_list(args.encoder_spectral_hidden_scale_values) and (
        len(values) > 1 or (len(values) == 1 and values[0] != "none")
    )


def _has_non_default_legacy_hidden_values(values: Sequence[Any]) -> bool:
    normalized = [str(value).strip().lower() for value in values]
    return len(normalized) > 1 or (len(normalized) == 1 and normalized[0] != "none")


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
        "--encoder-conv-hidden-scale-values",
        type=_parse_float_csv,
        default=[1.0],
    )
    parser.add_argument(
        "--encoder-spectral-hidden-scale-values",
        type=_parse_float_csv,
        default=[1.0],
    )
    parser.add_argument(
        "--encoder-conv-hidden-values",
        type=_parse_optional_numeric_csv,
        default=["none"],
        help=(
            "Diagnostic legacy alias for explicit branch hidden width values. "
            "Canonical staged runs should use --encoder-conv-hidden-scale-values."
        ),
    )
    parser.add_argument(
        "--encoder-spectral-hidden-values",
        type=_parse_optional_numeric_csv,
        default=["none"],
        help=(
            "Diagnostic legacy alias for explicit branch hidden width values. "
            "Canonical staged runs should use --encoder-spectral-hidden-scale-values."
        ),
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
        default=list(DEFAULT_PROMOTION_OBJECTIVES),
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
    parser.add_argument(
        "--reuse-existing-run-metrics",
        action="store_true",
        help=(
            "Reuse per-run metrics from runs/<run_id>/metrics.json when present "
            "instead of re-running the runner for that candidate."
        ),
    )
    parser.add_argument(
        "--validate-phase-guardrail",
        action="store_true",
        help=(
            "Validate a generated summary.csv by recomputing phase_ssim_drop_vs_baseline "
            "against a baseline summary and fail closed on mismatch or guardrail breach."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Summary CSV to validate when --validate-phase-guardrail is set.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help="Baseline/promotion source summary CSV used for phase-drop recomputation.",
    )
    parser.add_argument(
        "--write-validation-report",
        type=Path,
        help="Path to write JSON validation report for --validate-phase-guardrail.",
    )

    return parser.parse_args(argv)


def _active_profiles(args: argparse.Namespace) -> list[str]:
    return list(args.dataset_profiles_n256 if args.ns == 256 else args.dataset_profiles_n128)


def _validate_required_objective_tuple(args: argparse.Namespace) -> None:
    required = REQUIRED_PROMOTION_OBJECTIVES_BY_STAGE.get(str(args.stage_id))
    if required is None:
        return
    observed = tuple(str(value).strip() for value in args.promotion_objectives if str(value).strip())
    if observed != required:
        required_text = ",".join(required)
        observed_text = ",".join(observed)
        raise StageValidationError(
            f"Stage {args.stage_id} requires --promotion-objectives {required_text}; "
            f"got {observed_text}"
        )


def validate_stage_configuration(args: argparse.Namespace) -> None:
    """Validate stage/substage/profile constraints."""
    _validate_required_objective_tuple(args)

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

    # Structural-stage governance requires epoch floor compliance for every run invocation.
    if not args.aggregate_seed_rerank_root:
        active_epochs = int(args.epochs_n256 if args.ns == 256 else args.epochs_n128)
        if active_epochs < MIN_STAGE_EPOCHS:
            raise StageValidationError(
                f"Stage {args.stage_id}/{args.substage_id} requires epoch budget >= {MIN_STAGE_EPOCHS} "
                f"for N={args.ns}; got {active_epochs}"
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

    for scale in list(args.encoder_conv_hidden_scale_values) + list(args.encoder_spectral_hidden_scale_values):
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise StageValidationError(
                "Encoder branch scale values must be finite and > 0 "
                f"(got {scale!r})"
            )

    for hidden_raw in list(args.encoder_conv_hidden_values) + list(args.encoder_spectral_hidden_values):
        text = str(hidden_raw).strip().lower()
        if text == "none":
            continue
        try:
            hidden = int(float(text))
        except ValueError as exc:
            raise StageValidationError(
                f"Legacy encoder hidden value must be an integer or 'none' (got {hidden_raw!r})"
            ) from exc
        if hidden <= 0:
            raise StageValidationError(
                f"Legacy encoder hidden value must be > 0 when set (got {hidden_raw!r})"
            )

    if _has_non_default_legacy_hidden_values(args.encoder_conv_hidden_values) and not _is_default_scale_list(
        args.encoder_conv_hidden_scale_values
    ):
        raise StageValidationError(
            "Cannot combine legacy --encoder-conv-hidden-values sweeps with non-default "
            "--encoder-conv-hidden-scale-values. Use one axis at a time."
        )
    if _has_non_default_legacy_hidden_values(args.encoder_spectral_hidden_values) and not _is_default_scale_list(
        args.encoder_spectral_hidden_scale_values
    ):
        raise StageValidationError(
            "Cannot combine legacy --encoder-spectral-hidden-values sweeps with non-default "
            "--encoder-spectral-hidden-scale-values. Use one axis at a time."
        )

    profiles = _active_profiles(args)
    if args.ns == 256:
        if args.top_k_n256 > 0 and not args.allow_n256_direct_diagnostic:
            required_profiles = set(CANONICAL_N256_DATASET_PROFILES)
            observed_profiles = set(profiles)
            missing_profiles = sorted(required_profiles - observed_profiles)
            if missing_profiles:
                required_text = ",".join(CANONICAL_N256_DATASET_PROFILES)
                observed_text = ",".join(profiles) if profiles else "<none>"
                missing_text = ",".join(missing_profiles)
                raise StageValidationError(
                    "Canonical non-diagnostic N=256 runs require both dataset profiles "
                    f"in --dataset-profiles-n256 ({required_text}); missing {missing_text}. "
                    f"Observed: {observed_text}"
                )
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
    legacy_conv_axis = _uses_legacy_conv_hidden_axis(args)
    legacy_spectral_axis = _uses_legacy_spectral_hidden_axis(args)

    structural_axis_lengths = {
        "fno_blocks": len(args.fno_blocks_values),
        "downsample_schedule": len(args.downsample_schedule_values),
        "downsample_op": len(args.downsample_op_values),
        "encoder_conv_hidden_scale": len(args.encoder_conv_hidden_scale_values),
        "encoder_spectral_hidden_scale": len(args.encoder_spectral_hidden_scale_values),
        "encoder_conv_hidden_legacy": len(args.encoder_conv_hidden_values),
        "encoder_spectral_hidden_legacy": len(args.encoder_spectral_hidden_values),
        "max_hidden": len(args.max_hidden_values),
        "resnet_width": len(args.resnet_width_values),
        "resnet_blocks": len(args.resnet_blocks_values),
        "skip_style": len(args.skip_style_values),
    }
    structural_axis_values = {
        "fno_blocks": list(args.fno_blocks_values),
        "downsample_schedule": list(args.downsample_schedule_values),
        "downsample_op": list(args.downsample_op_values),
        "encoder_conv_hidden_scale": list(args.encoder_conv_hidden_scale_values),
        "encoder_spectral_hidden_scale": list(args.encoder_spectral_hidden_scale_values),
        "encoder_conv_hidden_legacy": list(args.encoder_conv_hidden_values),
        "encoder_spectral_hidden_legacy": list(args.encoder_spectral_hidden_values),
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
            "encoder_conv_hidden_scale": "1.0",
            "encoder_spectral_hidden_scale": "1.0",
            "encoder_conv_hidden_legacy": "none",
            "encoder_spectral_hidden_legacy": "none",
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
            expected = stage_a_defaults[axis_name]
            if axis_name in {"encoder_conv_hidden_scale", "encoder_spectral_hidden_scale"}:
                if not math.isclose(float(values[0]), float(expected), rel_tol=0.0, abs_tol=1e-12):
                    violations.append(f"{axis_name}={values[0]}")
                continue
            observed = str(values[0]).strip().lower()
            if observed != expected:
                violations.append(f"{axis_name}={values[0]}")

        if violations:
            raise MatrixExpansionError(
                "Stage A can only vary {modes, skip-values, widths}; "
                "all structural axes must remain at defaults "
                "(fno_blocks=4, downsample_schedule=2, downsample_op=stride_conv, "
                "encoder_conv_hidden_scale=1, encoder_spectral_hidden_scale=1, "
                "legacy encoder hidden values=none, max_hidden=none, resnet_width=none, "
                "resnet_blocks=6, skip_style=add); "
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
                active_axes = {"encoder_conv_hidden_legacy"} if legacy_conv_axis else {
                    "encoder_conv_hidden_scale"
                }
            elif args.substage_id == "D2":
                active_axes = {"encoder_spectral_hidden_legacy"} if legacy_spectral_axis else {
                    "encoder_spectral_hidden_scale"
                }
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
        "reuse_existing_run_metrics": bool(args.reuse_existing_run_metrics),
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
            "encoder_conv_hidden_scale": list(args.encoder_conv_hidden_scale_values),
            "encoder_spectral_hidden_scale": list(args.encoder_spectral_hidden_scale_values),
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
        "amp_ssim",
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
        parsed_row["amp_ssim"] = _safe_float(row.get("amp_ssim"), "amp_ssim")
        parsed_row["amp_mae"] = _safe_float(row.get("amp_mae"), "amp_mae")
        parsed_row["amp_mse"] = _safe_float(row.get("amp_mse"), "amp_mse")
        parsed_row["phase_ssim"] = _safe_float(
            row.get("phase_ssim"),
            "phase_ssim",
            default=parsed_row["amp_ssim"],
        )
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
        parsed_row["encoder_conv_hidden_scale"] = _safe_positive_scale(
            row.get("encoder_conv_hidden_scale", 1.0),
            "encoder_conv_hidden_scale",
            default=1.0,
        )
        parsed_row["encoder_spectral_hidden_scale"] = _safe_positive_scale(
            row.get("encoder_spectral_hidden_scale", 1.0),
            "encoder_spectral_hidden_scale",
            default=1.0,
        )
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
        parsed.append(_enrich_candidate_with_branch_metadata(parsed_row))

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
        value = _safe_float(row.get(objective), objective)
        direction = OBJECTIVE_DIRECTIONS.get(objective, "min")
        if direction == "max":
            value = -value
        values.append(value)
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


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    rank_value = row.get("pareto_rank_median")
    if rank_value in (None, ""):
        rank_value = row.get("pareto_rank_macro", 9999.0)
    amp_ssim_tiebreak = row.get("amp_ssim_mean_seed", row.get("amp_ssim"))
    return (
        _safe_float(rank_value, "pareto_rank"),
        -_safe_float(amp_ssim_tiebreak, "amp_ssim", default=-1.0),
        _safe_float(row.get("model_params"), "model_params"),
        _safe_float(row.get("inference_time_s"), "inference_time_s", default=1e12),
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
        round(_safe_float(row.get("encoder_conv_hidden_scale"), "encoder_conv_hidden_scale", default=1.0), 8),
        round(
            _safe_float(
                row.get("encoder_spectral_hidden_scale"),
                "encoder_spectral_hidden_scale",
                default=1.0,
            ),
            8,
        ),
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
            and math.isclose(
                _safe_float(row.get("encoder_conv_hidden_scale"), "encoder_conv_hidden_scale", default=1.0),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _safe_float(
                    row.get("encoder_spectral_hidden_scale"),
                    "encoder_spectral_hidden_scale",
                    default=1.0,
                ),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
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
    anchor = {
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
        "encoder_conv_hidden_scale": _safe_positive_scale(
            source_row.get("encoder_conv_hidden_scale", 1.0),
            "encoder_conv_hidden_scale",
            default=1.0,
        ),
        "encoder_spectral_hidden_scale": _safe_positive_scale(
            source_row.get("encoder_spectral_hidden_scale", 1.0),
            "encoder_spectral_hidden_scale",
            default=1.0,
        ),
        "encoder_conv_hidden": _resolve_axis_value(source_row.get("encoder_conv_hidden"), "none"),
        "encoder_spectral_hidden": _resolve_axis_value(source_row.get("encoder_spectral_hidden"), "none"),
        "encoder_stage_channels": _resolve_axis_value(source_row.get("encoder_stage_channels"), ""),
        "encoder_conv_hidden_resolved_width": _resolve_axis_value(
            source_row.get("encoder_conv_hidden_resolved_width"),
            "none",
        ),
        "encoder_conv_hidden_resolved_per_block": _resolve_axis_value(
            source_row.get("encoder_conv_hidden_resolved_per_block"),
            "",
        ),
        "encoder_spectral_hidden_resolved_width": _resolve_axis_value(
            source_row.get("encoder_spectral_hidden_resolved_width"),
            "none",
        ),
        "encoder_spectral_hidden_resolved_per_block": _resolve_axis_value(
            source_row.get("encoder_spectral_hidden_resolved_per_block"),
            "",
        ),
        "max_hidden": _resolve_axis_value(source_row.get("max_hidden"), "none"),
        "resnet_width": _resolve_axis_value(source_row.get("resnet_width"), "none"),
        "resnet_blocks": _safe_int(source_row.get("resnet_blocks"), "resnet_blocks", default=6),
        "skip_style": _resolve_axis_value(source_row.get("skip_style"), "add"),
        "amp_ssim": _safe_float(source_row.get("amp_ssim"), "amp_ssim"),
        "amp_mae": _safe_float(source_row.get("amp_mae"), "amp_mae"),
        "amp_mse": _safe_float(source_row.get("amp_mse"), "amp_mse"),
        "phase_ssim": _safe_float(source_row.get("phase_ssim"), "phase_ssim", default=0.0),
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
    return _enrich_candidate_with_branch_metadata(anchor)


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
                "amp_ssim",
                "amp_mae",
                "amp_mse",
                "phase_ssim",
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
            row["amp_ssim"] = _safe_float(raw_row.get("amp_ssim"), "amp_ssim")
            row["amp_mae"] = _safe_float(raw_row.get("amp_mae"), "amp_mae")
            row["amp_mse"] = _safe_float(raw_row.get("amp_mse"), "amp_mse")
            row["phase_ssim"] = _safe_float(raw_row.get("phase_ssim"), "phase_ssim")
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
                "amp_ssim": float(source_row["amp_ssim"]),
                "amp_mae": float(source_row["amp_mae"]),
                "amp_mse": float(source_row["amp_mse"]),
                "phase_ssim": float(source_row.get("phase_ssim", 0.0)),
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
        mean_amp_ssim = sum(float(per_seed[seed]["amp_ssim"]) for seed in SEED_SET) / float(len(SEED_SET))

        seed3_row = per_seed[3]
        robust_row = _enrich_candidate_with_branch_metadata(
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
                "encoder_conv_hidden_scale": _safe_positive_scale(
                    source_row.get("encoder_conv_hidden_scale", 1.0),
                    "encoder_conv_hidden_scale",
                    default=1.0,
                ),
                "encoder_spectral_hidden_scale": _safe_positive_scale(
                    source_row.get("encoder_spectral_hidden_scale", 1.0),
                    "encoder_spectral_hidden_scale",
                    default=1.0,
                ),
                "encoder_conv_hidden": str(source_row.get("encoder_conv_hidden", "none")),
                "encoder_spectral_hidden": str(source_row.get("encoder_spectral_hidden", "none")),
                "max_hidden": str(source_row.get("max_hidden", "none")),
                "resnet_width": str(source_row.get("resnet_width", "none")),
                "resnet_blocks": _safe_int(source_row.get("resnet_blocks"), "resnet_blocks", default=6),
                "skip_style": str(source_row.get("skip_style", "add")),
                "amp_ssim": float(seed3_row["amp_ssim"]),
                "amp_ssim_mean_seed": float(mean_amp_ssim),
                "amp_mae": float(seed3_row["amp_mae"]),
                "amp_mse": float(seed3_row["amp_mse"]),
                "phase_ssim": float(seed3_row["phase_ssim"]),
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
        robust_rows.append(robust_row)

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
        "encoder_conv_hidden_scale",
        "encoder_spectral_hidden_scale",
        "encoder_conv_hidden",
        "encoder_spectral_hidden",
        "encoder_stage_channels",
        "encoder_conv_hidden_resolved_width",
        "encoder_conv_hidden_resolved_per_block",
        "encoder_spectral_hidden_resolved_width",
        "encoder_spectral_hidden_resolved_per_block",
        "max_hidden",
        "resnet_width",
        "resnet_blocks",
        "skip_style",
        "amp_ssim",
        "amp_ssim_mean_seed",
        "amp_mae",
        "amp_mse",
        "phase_ssim",
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
                "downsample_op=stride_conv, encoder_conv_hidden_scale=1, "
                "encoder_spectral_hidden_scale=1, encoder_conv_hidden=none, "
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
                -_safe_float(row.get("amp_ssim"), "amp_ssim", default=-1.0),
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


def _safe_positive_scale(raw: Any, key: str, *, default: float = 1.0) -> float:
    value = _safe_float(raw, key, default=default)
    if not math.isfinite(value) or value <= 0.0:
        raise StageValidationError(f"{key} must be finite and > 0 (got {raw!r})")
    return float(value)


def _optional_positive_int_or_none(raw: Any, key: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"", "none"}:
        return None
    value = _safe_int(text, key)
    if value <= 0:
        raise StageValidationError(f"{key} must be > 0 when set (got {raw!r})")
    return value


def _derive_encoder_stage_channels(
    *,
    width: int,
    fno_blocks: int,
    downsample_schedule: int,
    max_hidden_raw: Any,
) -> list[int]:
    max_hidden = _optional_positive_int_or_none(max_hidden_raw, "max_hidden")
    channels: list[int] = []
    current = int(width)
    for block_idx in range(int(fno_blocks)):
        channels.append(current)
        if block_idx < int(downsample_schedule):
            next_channels = current * 2
            if max_hidden is not None:
                next_channels = min(next_channels, max_hidden)
            current = next_channels
    return channels


def _resolve_branch_hidden_metadata(
    *,
    stage_channels: Sequence[int],
    scale: float,
    explicit_hidden_raw: Any,
    key_name: str,
) -> dict[str, Any]:
    explicit_hidden = _optional_positive_int_or_none(explicit_hidden_raw, key_name)
    if explicit_hidden is not None:
        per_block = [int(explicit_hidden)] * len(stage_channels)
        explicit_hidden_value = str(explicit_hidden)
    else:
        per_block = [max(1, int(round(int(channel) * float(scale)))) for channel in stage_channels]
        explicit_hidden_value = "none"

    resolved_width = "none"
    if per_block:
        resolved_width = str(per_block[0])

    return {
        "explicit_hidden": explicit_hidden_value,
        "resolved_width": resolved_width,
        "resolved_per_block": "|".join(str(value) for value in per_block),
    }


def _format_scale_token(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.6g}".replace(".", "p").replace("-", "m")


def _enrich_candidate_with_branch_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    stage_channels = _derive_encoder_stage_channels(
        width=_safe_int(row.get("width"), "width", default=32),
        fno_blocks=_safe_int(row.get("fno_blocks"), "fno_blocks", default=4),
        downsample_schedule=_safe_int(row.get("downsample_schedule"), "downsample_schedule", default=2),
        max_hidden_raw=row.get("max_hidden", "none"),
    )

    conv_scale = _safe_positive_scale(
        row.get("encoder_conv_hidden_scale", 1.0),
        "encoder_conv_hidden_scale",
        default=1.0,
    )
    spectral_scale = _safe_positive_scale(
        row.get("encoder_spectral_hidden_scale", 1.0),
        "encoder_spectral_hidden_scale",
        default=1.0,
    )
    row["encoder_conv_hidden_scale"] = float(conv_scale)
    row["encoder_spectral_hidden_scale"] = float(spectral_scale)
    row["encoder_stage_channels"] = "|".join(str(value) for value in stage_channels)
    row["encoder_conv_hidden"] = _resolve_axis_value(row.get("encoder_conv_hidden"), "none")
    row["encoder_spectral_hidden"] = _resolve_axis_value(row.get("encoder_spectral_hidden"), "none")

    conv_meta = _resolve_branch_hidden_metadata(
        stage_channels=stage_channels,
        scale=conv_scale,
        explicit_hidden_raw=row["encoder_conv_hidden"],
        key_name="encoder_conv_hidden",
    )
    spectral_meta = _resolve_branch_hidden_metadata(
        stage_channels=stage_channels,
        scale=spectral_scale,
        explicit_hidden_raw=row["encoder_spectral_hidden"],
        key_name="encoder_spectral_hidden",
    )

    row["encoder_conv_hidden"] = str(conv_meta["explicit_hidden"])
    row["encoder_spectral_hidden"] = str(spectral_meta["explicit_hidden"])
    row["encoder_conv_hidden_resolved_width"] = str(conv_meta["resolved_width"])
    row["encoder_spectral_hidden_resolved_width"] = str(spectral_meta["resolved_width"])
    row["encoder_conv_hidden_resolved_per_block"] = str(conv_meta["resolved_per_block"])
    row["encoder_spectral_hidden_resolved_per_block"] = str(spectral_meta["resolved_per_block"])
    return row


def _build_stage_a_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    profiles = _active_profiles(args)
    candidates: list[dict[str, Any]] = []
    for mode, skip, width, profile in itertools.product(args.modes, args.skip_values, args.widths, profiles):
        skip_flag = str(skip).lower()
        candidate = _enrich_candidate_with_branch_metadata(
            {
                "modes": int(mode),
                "skip": skip_flag,
                "width": int(width),
                "fno_blocks": int(_select_scalar(args.fno_blocks_values) or 4),
                "downsample_schedule": int(_select_scalar(args.downsample_schedule_values) or 2),
                "downsample_op": str(_select_scalar(args.downsample_op_values) or "stride_conv"),
                "encoder_conv_hidden_scale": float(_select_scalar(args.encoder_conv_hidden_scale_values) or 1.0),
                "encoder_spectral_hidden_scale": float(
                    _select_scalar(args.encoder_spectral_hidden_scale_values) or 1.0
                ),
                "encoder_conv_hidden": str(_select_scalar(args.encoder_conv_hidden_values) or "none"),
                "encoder_spectral_hidden": str(_select_scalar(args.encoder_spectral_hidden_values) or "none"),
                "max_hidden": str(_select_scalar(args.max_hidden_values) or "none"),
                "resnet_width": str(_select_scalar(args.resnet_width_values) or "none"),
                "resnet_blocks": int(_select_scalar(args.resnet_blocks_values) or 6),
                "skip_style": str(_select_scalar(args.skip_style_values) or "add"),
                "source_run_id": "",
                "dataset_profile": profile,
            }
        )
        candidate["run_key"] = (
            f"stage{args.stage_id}_n{args.ns}_profile-{profile}_m{mode}_s{skip_flag}_w{width}_"
            f"ecs{_format_scale_token(candidate['encoder_conv_hidden_scale'])}_"
            f"ess{_format_scale_token(candidate['encoder_spectral_hidden_scale'])}_"
            f"ec{candidate['encoder_conv_hidden']}_es{candidate['encoder_spectral_hidden']}"
        )
        candidates.append(candidate)
    return candidates


def _active_axis_values(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.stage_id == "B":
        return [{"fno_blocks": int(v)} for v in args.fno_blocks_values]
    if args.stage_id == "C" and args.substage_id == "C1":
        return [{"downsample_schedule": int(v)} for v in args.downsample_schedule_values]
    if args.stage_id == "C" and args.substage_id == "C2":
        return [{"downsample_op": str(v)} for v in args.downsample_op_values]
    if args.stage_id == "D" and args.substage_id == "D1":
        if _uses_legacy_conv_hidden_axis(args):
            return [{"encoder_conv_hidden": str(v)} for v in args.encoder_conv_hidden_values]
        return [{"encoder_conv_hidden_scale": float(v)} for v in args.encoder_conv_hidden_scale_values]
    if args.stage_id == "D" and args.substage_id == "D2":
        if _uses_legacy_spectral_hidden_axis(args):
            return [{"encoder_spectral_hidden": str(v)} for v in args.encoder_spectral_hidden_values]
        return [
            {"encoder_spectral_hidden_scale": float(v)}
            for v in args.encoder_spectral_hidden_scale_values
        ]
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
            "skip": "on"
            if args.stage_id == "E"
            else str(anchor_row.get("skip", "off")).lower(),
            "width": _safe_int(anchor_row.get("width"), "width"),
            "fno_blocks": _safe_int(anchor_row.get("fno_blocks"), "fno_blocks", default=4),
            "downsample_schedule": _safe_int(
                anchor_row.get("downsample_schedule"), "downsample_schedule", default=2
            ),
            "downsample_op": _resolve_axis_value(anchor_row.get("downsample_op"), "stride_conv"),
            "encoder_conv_hidden_scale": _safe_positive_scale(
                anchor_row.get("encoder_conv_hidden_scale", 1.0),
                "encoder_conv_hidden_scale",
                default=1.0,
            ),
            "encoder_spectral_hidden_scale": _safe_positive_scale(
                anchor_row.get("encoder_spectral_hidden_scale", 1.0),
                "encoder_spectral_hidden_scale",
                default=1.0,
            ),
            "encoder_conv_hidden": _resolve_axis_value(anchor_row.get("encoder_conv_hidden"), "none"),
            "encoder_spectral_hidden": _resolve_axis_value(
                anchor_row.get("encoder_spectral_hidden"), "none"
            ),
            "max_hidden": _resolve_axis_value(anchor_row.get("max_hidden"), "none"),
            "resnet_width": _resolve_axis_value(anchor_row.get("resnet_width"), "none"),
            "resnet_blocks": _safe_int(anchor_row.get("resnet_blocks"), "resnet_blocks", default=6),
            "skip_style": _resolve_axis_value(anchor_row.get("skip_style"), "add"),
            "source_run_id": str(anchor_row.get("run_id", "")),
        }

        for profile, axis_values in itertools.product(profiles, _active_axis_values(args)):
            row = dict(base)
            row.update(axis_values)
            row["dataset_profile"] = profile
            row = _enrich_candidate_with_branch_metadata(row)
            row["run_key"] = (
                f"stage{args.stage_id}{args.substage_id}_n{args.ns}_profile-{profile}_"
                f"m{row['modes']}_s{row['skip']}_w{row['width']}_"
                f"fb{row['fno_blocks']}_ds{row['downsample_schedule']}_{row['downsample_op']}_"
                f"ecs{_format_scale_token(row['encoder_conv_hidden_scale'])}_"
                f"ess{_format_scale_token(row['encoder_spectral_hidden_scale'])}_"
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
                "skip": "on" if args.stage_id == "E" else str(source.get("skip", "off")).lower(),
                "width": _safe_int(source.get("width"), "width"),
                "fno_blocks": _safe_int(source.get("fno_blocks"), "fno_blocks", default=4),
                "downsample_schedule": _safe_int(
                    source.get("downsample_schedule"), "downsample_schedule", default=2
                ),
                "downsample_op": _resolve_axis_value(source.get("downsample_op"), "stride_conv"),
                "encoder_conv_hidden_scale": _safe_positive_scale(
                    source.get("encoder_conv_hidden_scale", 1.0),
                    "encoder_conv_hidden_scale",
                    default=1.0,
                ),
                "encoder_spectral_hidden_scale": _safe_positive_scale(
                    source.get("encoder_spectral_hidden_scale", 1.0),
                    "encoder_spectral_hidden_scale",
                    default=1.0,
                ),
                "encoder_conv_hidden": _resolve_axis_value(source.get("encoder_conv_hidden"), "none"),
                "encoder_spectral_hidden": _resolve_axis_value(
                    source.get("encoder_spectral_hidden"), "none"
                ),
                "max_hidden": _resolve_axis_value(source.get("max_hidden"), "none"),
                "resnet_width": _resolve_axis_value(source.get("resnet_width"), "none"),
                "resnet_blocks": _safe_int(source.get("resnet_blocks"), "resnet_blocks", default=6),
                "skip_style": _resolve_axis_value(source.get("skip_style"), "add"),
                "source_run_id": str(source.get("run_id", "")),
                "dataset_profile": profile,
            }
            candidates.append(_enrich_candidate_with_branch_metadata(row))
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


def _enforce_stage_e_skip_enabled(
    args: argparse.Namespace,
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    if args.stage_id != "E":
        return
    invalid = [
        str(candidate.get("run_key", "<unknown>"))
        for candidate in candidates
        if str(candidate.get("skip", "off")).strip().lower() != "on"
    ]
    if invalid:
        preview = ", ".join(invalid[:3])
        suffix = "..." if len(invalid) > 3 else ""
        raise StageValidationError(
            "Stage E requires skip=on for all candidates; "
            f"found skip!=on for: {preview}{suffix}"
        )


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
            "--hybrid-encoder-conv-hidden-scale",
            str(
                _safe_positive_scale(
                    candidate.get("encoder_conv_hidden_scale", 1.0),
                    "encoder_conv_hidden_scale",
                    default=1.0,
                )
            ),
            "--hybrid-encoder-spectral-hidden-scale",
            str(
                _safe_positive_scale(
                    candidate.get("encoder_spectral_hidden_scale", 1.0),
                    "encoder_spectral_hidden_scale",
                    default=1.0,
                )
            ),
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
    amp_ssim = _extract_metric_component(run_metrics.get("ssim"), index=0, key="ssim")
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

    # Baseline phase provenance is resolved in run_sweep after row assembly.
    phase_drop = 0.0

    return {
        "amp_ssim": float(amp_ssim),
        "amp_mae": float(amp_mae),
        "amp_mse": float(amp_mse),
        "phase_ssim": float(phase_ssim),
        "phase_ssim_drop_vs_baseline": float(phase_drop),
        "model_params": int(model_params),
        "train_wall_time_sec": float(elapsed),
        "inference_time_s": float(inference_time),
    }


def _load_existing_run_metrics(run_dir: Path) -> dict[str, float] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None

    required = (
        "amp_ssim",
        "amp_mae",
        "amp_mse",
        "phase_ssim",
        "phase_ssim_drop_vs_baseline",
        "model_params",
        "train_wall_time_sec",
        "inference_time_s",
    )
    if any(key not in payload for key in required):
        return None

    try:
        return {
            "amp_ssim": float(payload["amp_ssim"]),
            "amp_mae": float(payload["amp_mae"]),
            "amp_mse": float(payload["amp_mse"]),
            "phase_ssim": float(payload["phase_ssim"]),
            "phase_ssim_drop_vs_baseline": float(payload["phase_ssim_drop_vs_baseline"]),
            "model_params": int(float(payload["model_params"])),
            "train_wall_time_sec": float(payload["train_wall_time_sec"]),
            "inference_time_s": float(payload["inference_time_s"]),
        }
    except (TypeError, ValueError):
        return None


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
    prune_roots = [run_dir]
    if runtime_root.exists() and runtime_root.is_dir():
        prune_roots.append(runtime_root)
    if not any(root.exists() for root in prune_roots):
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

    dir_target_map: dict[Path, Path] = {}
    for root in prune_roots:
        for dir_name in heavy_dirs:
            candidate = root / dir_name
            if candidate.exists() and candidate.is_dir():
                dir_target_map[candidate.resolve()] = candidate
    dir_target_set = set(dir_target_map.keys())

    file_target_map: dict[Path, Path] = {}
    for root in prune_roots:
        for pattern in heavy_file_patterns:
            for candidate in root.rglob(pattern):
                if not candidate.is_file():
                    continue
                candidate_resolved = candidate.resolve()
                if any(parent in dir_target_set for parent in candidate_resolved.parents):
                    continue
                file_target_map[candidate_resolved] = candidate

    targets = sorted(
        {*dir_target_map.values(), *file_target_map.values()},
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


def _reconcile_retention_tiers(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
    prune_heavy_artifacts: bool,
) -> None:
    tuple_anchor_run_ids: set[str] = set()
    grouped_rows: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("stage_id", "")),
            str(row.get("substage_id", "")),
            int(_safe_int(row.get("N"), "N", default=0)),
            str(row.get("dataset_profile", "")),
        )
        grouped_rows.setdefault(key, []).append(row)

    for group_rows in grouped_rows.values():
        if not group_rows:
            continue
        stage_anchor_rows = [row for row in group_rows if _bool_from_value(row.get("is_stage_anchor", False))]
        if stage_anchor_rows:
            selected = sorted(stage_anchor_rows, key=lambda payload: int(payload.get("run_index", 0)))[0]
            tuple_anchor_run_ids.add(str(selected["run_id"]))
            continue

        feasible_rows = [row for row in group_rows if _bool_from_value(row.get("is_feasible", False))]
        if feasible_rows:
            selected = sorted(feasible_rows, key=_candidate_sort_key)[0]
            tuple_anchor_run_ids.add(str(selected["run_id"]))
            continue

        selected = sorted(group_rows, key=lambda payload: int(payload.get("run_index", 0)))[0]
        tuple_anchor_run_ids.add(str(selected["run_id"]))

    for row in rows:
        run_id = str(row["run_id"])
        run_dir = output_root / "runs" / run_id
        if run_id in tuple_anchor_run_ids:
            retention_tier = "full_anchor"
            deleted_paths: list[str] = []
            bytes_reclaimed = 0
        else:
            retention_tier = "pruned"
            deleted_paths, bytes_reclaimed = [], 0
            if prune_heavy_artifacts:
                deleted_paths, bytes_reclaimed = _prune_heavy_runtime_artifacts(run_dir)

        row["retention_tier"] = retention_tier
        _write_cleanup_report(
            run_dir,
            retention_tier=retention_tier,
            deleted_paths=deleted_paths,
            bytes_reclaimed=bytes_reclaimed,
        )


def _prune_orphan_run_dirs(
    output_root: Path,
    *,
    expected_run_ids: set[str],
) -> None:
    runs_root = output_root / "runs"
    if not runs_root.exists() or not runs_root.is_dir():
        return

    orphan_dirs: list[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in expected_run_ids:
            continue
        orphan_dirs.append(child)

    if not orphan_dirs:
        return

    bytes_reclaimed = 0
    orphan_ids: list[str] = []
    for orphan_dir in orphan_dirs:
        bytes_reclaimed += _path_size_bytes(orphan_dir)
        shutil.rmtree(orphan_dir)
        orphan_ids.append(orphan_dir.name)

    payload = {
        "orphan_count": len(orphan_ids),
        "orphan_run_ids": orphan_ids,
        "bytes_reclaimed": int(bytes_reclaimed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "orphan_run_cleanup.json").write_text(json.dumps(payload, indent=2) + "\n")


def _validate_manifest_objective_contract(manifest: Mapping[str, Any]) -> None:
    stage_id = str(manifest.get("stage_id", "")).strip()
    required = REQUIRED_PROMOTION_OBJECTIVES_BY_STAGE.get(stage_id)
    if required is None:
        return
    observed = tuple(
        str(value).strip()
        for value in manifest.get("promotion", {}).get("objectives", [])
        if str(value).strip()
    )
    if observed != required:
        required_text = ",".join(required)
        observed_text = ",".join(observed)
        raise StageValidationError(
            f"Manifest promotion objectives mismatch for stage {stage_id}: "
            f"expected {required_text}, got {observed_text}"
        )


def _build_stage_a_profile_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, str]]:
    """Resolve Stage-A default-control baselines per dataset profile."""
    by_profile: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        profile = str(row.get("dataset_profile", ""))
        by_profile.setdefault(profile, []).append(row)

    baselines: dict[str, tuple[float, str]] = {}
    for profile, profile_rows in by_profile.items():
        matches = [row for row in profile_rows if _is_stage_a_control_anchor_row(row)]
        if not matches:
            continue
        if len(matches) != 1:
            raise PromotionSourceError(
                f"Stage-A baseline provenance is ambiguous for profile '{profile}': "
                f"found {len(matches)} true-default rows"
            )
        baseline_row = matches[0]
        baselines[profile] = (
            _safe_float(baseline_row.get("phase_ssim"), "phase_ssim", default=0.0),
            str(baseline_row.get("run_id", "")),
        )
    return baselines


def _resolve_phase_baseline_for_row(
    *,
    row: Mapping[str, Any],
    source_phase_by_run: Mapping[str, float],
    anchor_row: Mapping[str, Any] | None,
    stage_a_baselines: Mapping[str, tuple[float, str]],
) -> tuple[float, str, str]:
    """Resolve baseline phase SSIM and provenance for one summary row."""
    source_run_id = str(row.get("source_run_id", "")).strip()
    if source_run_id:
        if source_run_id in source_phase_by_run:
            return (
                float(source_phase_by_run[source_run_id]),
                "promotion_source_summary",
                source_run_id,
            )
        raise PromotionSourceError(
            f"Could not resolve source_run_id '{source_run_id}' in baseline summary"
        )

    if anchor_row is not None:
        anchor_run_id = str(anchor_row.get("run_id", "")).strip()
        return (
            _safe_float(anchor_row.get("phase_ssim"), "phase_ssim", default=0.0),
            "promotion_source_anchor",
            anchor_run_id,
        )

    profile = str(row.get("dataset_profile", "")).strip()
    stage_a_entry = stage_a_baselines.get(profile)
    if stage_a_entry is not None:
        baseline_phase, baseline_run_id = stage_a_entry
        return (
            float(baseline_phase),
            "stage_a_default_control",
            str(baseline_run_id),
        )

    if len(source_phase_by_run) == 1:
        only_run_id, only_phase = next(iter(source_phase_by_run.items()))
        return (
            float(only_phase),
            "promotion_source_singleton",
            str(only_run_id),
        )

    raise PromotionSourceError(
        f"Could not resolve phase baseline provenance for row '{row.get('run_id', '<unknown>')}'"
    )


def _apply_phase_guardrail_and_feasibility(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Mapping[str, Any]],
    anchor_row: Mapping[str, Any] | None,
) -> None:
    """Recompute phase drop from provenance, then enforce feasibility gates."""
    source_phase_by_run: dict[str, float] = {}
    for source_row in source_rows:
        run_id = str(source_row.get("run_id", "")).strip()
        if not run_id:
            continue
        source_phase_by_run[run_id] = _safe_float(source_row.get("phase_ssim"), "phase_ssim", default=0.0)

    stage_a_baselines = _build_stage_a_profile_baselines(rows)
    for row in rows:
        baseline_phase, baseline_source, baseline_run_id = _resolve_phase_baseline_for_row(
            row=row,
            source_phase_by_run=source_phase_by_run,
            anchor_row=anchor_row,
            stage_a_baselines=stage_a_baselines,
        )
        phase_ssim = _safe_float(row.get("phase_ssim"), "phase_ssim", default=0.0)
        phase_drop = _compute_phase_drop(phase_ssim=phase_ssim, baseline_phase_ssim=baseline_phase)
        row["phase_ssim_baseline"] = float(baseline_phase)
        row["phase_ssim_baseline_source"] = str(baseline_source)
        row["phase_ssim_baseline_run_id"] = str(baseline_run_id)
        row["phase_ssim_drop_vs_baseline"] = float(phase_drop)
        row["max_phase_ssim_drop"] = float(args.max_phase_ssim_drop)
        row["phase_guardrail_pass"] = bool(phase_drop <= args.max_phase_ssim_drop)

        feasible, violations = _row_feasibility(row, args, args.ns)
        row["is_feasible"] = bool(feasible)
        row["violated_constraints"] = ";".join(violations)


def _write_per_run_metrics_payloads(
    rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
) -> None:
    """Persist per-run payloads after guardrail recomputation."""
    for row in rows:
        run_dir_raw = row.get("_run_dir")
        if run_dir_raw is None:
            raise RuntimeError(f"Missing _run_dir for row '{row.get('run_id', '<unknown>')}'")
        run_dir = Path(str(run_dir_raw))
        payload = {
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": str(row.get("run_id", "")),
            "stage_id": args.stage_id,
            "substage_id": args.substage_id,
            "dataset_profile": str(row.get("dataset_profile", "")),
            "source_run_id": str(row.get("source_run_id", "")),
            "encoder_conv_hidden_scale": float(row.get("encoder_conv_hidden_scale", 1.0)),
            "encoder_spectral_hidden_scale": float(row.get("encoder_spectral_hidden_scale", 1.0)),
            "encoder_stage_channels": str(row.get("encoder_stage_channels", "")),
            "encoder_conv_hidden_resolved_width": str(row.get("encoder_conv_hidden_resolved_width", "")),
            "encoder_conv_hidden_resolved_per_block": str(
                row.get("encoder_conv_hidden_resolved_per_block", "")
            ),
            "encoder_spectral_hidden_resolved_width": str(
                row.get("encoder_spectral_hidden_resolved_width", "")
            ),
            "encoder_spectral_hidden_resolved_per_block": str(
                row.get("encoder_spectral_hidden_resolved_per_block", "")
            ),
            "amp_ssim": float(row.get("amp_ssim", 0.0)),
            "amp_mae": float(row.get("amp_mae", 0.0)),
            "amp_mse": float(row.get("amp_mse", 0.0)),
            "phase_ssim": float(row.get("phase_ssim", 0.0)),
            "phase_ssim_baseline": float(row.get("phase_ssim_baseline", 0.0)),
            "phase_ssim_baseline_source": str(row.get("phase_ssim_baseline_source", "")),
            "phase_ssim_baseline_run_id": str(row.get("phase_ssim_baseline_run_id", "")),
            "phase_ssim_drop_vs_baseline": float(row.get("phase_ssim_drop_vs_baseline", 0.0)),
            "max_phase_ssim_drop": float(args.max_phase_ssim_drop),
            "phase_guardrail_pass": bool(row.get("phase_guardrail_pass", False)),
            "model_params": int(_safe_float(row.get("model_params"), "model_params", default=0.0)),
            "train_wall_time_sec": float(row.get("train_wall_time_sec", 0.0)),
            "inference_time_s": float(row.get("inference_time_s", 0.0)),
            "is_feasible": bool(row.get("is_feasible", False)),
            "violated_constraints": str(row.get("violated_constraints", "")),
            "probe_mask_enabled": bool(row.get("probe_mask_enabled", False)),
            "torch_mae_pred_l2_match_target": bool(
                row.get("torch_mae_pred_l2_match_target", False)
            ),
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


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
    _enforce_stage_e_skip_enabled(args, candidates)
    _prune_orphan_run_dirs(
        args.output_root,
        expected_run_ids={str(candidate["run_key"]) for candidate in candidates},
    )

    rows: list[dict[str, Any]] = []
    dataset_cache: dict[str, tuple[Path, Path]] = {}
    for profile in sorted({str(candidate["dataset_profile"]) for candidate in candidates}):
        _resolve_profile_npz_inputs(args, profile, dataset_cache)

    manifest = build_sweep_manifest(args)
    _validate_manifest_objective_contract(manifest)
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
        metrics = None
        if args.reuse_existing_run_metrics:
            metrics = _load_existing_run_metrics(run_dir)
        if metrics is None:
            metrics = _run_candidate_with_runner(
                args=args,
                candidate=candidate,
                run_dir=run_dir,
                train_npz=train_npz,
                test_npz=test_npz,
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
            "encoder_conv_hidden_scale": float(candidate["encoder_conv_hidden_scale"]),
            "encoder_spectral_hidden_scale": float(candidate["encoder_spectral_hidden_scale"]),
            "encoder_conv_hidden": str(candidate["encoder_conv_hidden"]),
            "encoder_spectral_hidden": str(candidate["encoder_spectral_hidden"]),
            "encoder_stage_channels": str(candidate["encoder_stage_channels"]),
            "encoder_conv_hidden_resolved_width": str(candidate["encoder_conv_hidden_resolved_width"]),
            "encoder_conv_hidden_resolved_per_block": str(
                candidate["encoder_conv_hidden_resolved_per_block"]
            ),
            "encoder_spectral_hidden_resolved_width": str(
                candidate["encoder_spectral_hidden_resolved_width"]
            ),
            "encoder_spectral_hidden_resolved_per_block": str(
                candidate["encoder_spectral_hidden_resolved_per_block"]
            ),
            "max_hidden": str(candidate["max_hidden"]),
            "resnet_width": str(candidate["resnet_width"]),
            "resnet_blocks": int(candidate["resnet_blocks"]),
            "skip_style": str(candidate["skip_style"]),
            "source_run_id": str(candidate.get("source_run_id", "")),
            "amp_ssim": float(metrics["amp_ssim"]),
            "amp_mae": float(metrics["amp_mae"]),
            "amp_mse": float(metrics["amp_mse"]),
            "phase_ssim": float(metrics["phase_ssim"]),
            "phase_ssim_baseline": float("nan"),
            "phase_ssim_baseline_source": "",
            "phase_ssim_baseline_run_id": "",
            "phase_ssim_drop_vs_baseline": float(
                metrics.get("phase_ssim_drop_vs_baseline", 0.0)
            ),
            "max_phase_ssim_drop": float(args.max_phase_ssim_drop),
            "phase_guardrail_pass": False,
            "model_params": int(metrics["model_params"]),
            "train_wall_time_sec": float(metrics["train_wall_time_sec"]),
            "inference_time_s": float(metrics["inference_time_s"]),
            "is_feasible": False,
            "violated_constraints": "",
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
            "retention_tier": "pending_anchor_resolution",
            "run_index": index,
            "_run_dir": str(run_dir),
        }
        rows.append(row)

    _apply_phase_guardrail_and_feasibility(
        rows,
        args=args,
        source_rows=source_rows,
        anchor_row=anchor_row,
    )
    _write_per_run_metrics_payloads(rows, args=args)

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
        candidate_key = (
            f"{row['modes']}|{row['skip']}|{row['width']}|{row['fno_blocks']}|"
            f"{row['downsample_schedule']}|{row['downsample_op']}|"
            f"{row['encoder_conv_hidden_scale']}|{row['encoder_spectral_hidden_scale']}|"
            f"{row['encoder_conv_hidden']}|{row['encoder_spectral_hidden']}|"
            f"{row['max_hidden']}|{row['resnet_width']}|{row['resnet_blocks']}|{row['skip_style']}"
        )
        by_candidate.setdefault(candidate_key, []).append(row)

    macro_items: list[tuple[str, float, float]] = []
    mean_params_by_candidate: dict[str, float] = {}
    for key, candidate_rows in by_candidate.items():
        profile_ranks = sorted(int(r["pareto_rank_profile"]) for r in candidate_rows)
        median_rank = profile_ranks[len(profile_ranks) // 2]
        mean_amp_ssim = sum(float(r["amp_ssim"]) for r in candidate_rows) / len(candidate_rows)
        mean_params = sum(float(r["model_params"]) for r in candidate_rows) / len(candidate_rows)
        mean_params_by_candidate[key] = float(mean_params)
        macro_items.append((key, float(median_rank), float(mean_amp_ssim)))

    macro_items.sort(
        key=lambda item: (item[1], -item[2], mean_params_by_candidate[item[0]], item[0])
    )
    macro_rank_map = {key: idx + 1 for idx, (key, _, _) in enumerate(macro_items)}

    for key, candidate_rows in by_candidate.items():
        rank = macro_rank_map[key]
        for row in candidate_rows:
            row["pareto_rank_macro"] = int(rank)

    feasible_rows = [row for row in rows if _bool_from_value(row["is_feasible"])]
    feasible_rows.sort(key=_candidate_sort_key)

    # Promotion and stage-anchor flags.
    if feasible_rows:
        feasible_rows[0]["is_stage_anchor"] = True
        if args.ns == 128 and args.top_k_n256 > 0:
            for row in feasible_rows[: args.top_k_n256]:
                row["promote_to_n256"] = True
    elif rows:
        # Seed-rerank singletons can be infeasible under strict caps; still emit a
        # deterministic anchor row so summary/cleanup artifacts can be written.
        sorted_rows = sorted(rows, key=_candidate_sort_key)
        sorted_rows[0]["is_stage_anchor"] = True

    _reconcile_retention_tiers(
        rows,
        output_root=args.output_root,
        prune_heavy_artifacts=bool(args.prune_heavy_artifacts),
    )

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
        "encoder_conv_hidden_scale",
        "encoder_spectral_hidden_scale",
        "encoder_conv_hidden",
        "encoder_spectral_hidden",
        "encoder_stage_channels",
        "encoder_conv_hidden_resolved_width",
        "encoder_conv_hidden_resolved_per_block",
        "encoder_spectral_hidden_resolved_width",
        "encoder_spectral_hidden_resolved_per_block",
        "max_hidden",
        "resnet_width",
        "resnet_blocks",
        "skip_style",
        "source_run_id",
        "amp_ssim",
        "amp_mae",
        "amp_mse",
        "phase_ssim",
        "phase_ssim_baseline",
        "phase_ssim_baseline_source",
        "phase_ssim_baseline_run_id",
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

    anchor_summary_path: Path | None = args.emit_stage_anchor_summary
    if anchor_summary_path is None:
        anchor_summary_path = args.output_root / "promotion" / "stage_anchor_summary.csv"
    anchor_rows = [row for row in rows if _bool_from_value(row.get("is_stage_anchor", False))]
    if len(anchor_rows) != 1:
        raise PromotionSourceError(
            "Sweep execution requires exactly one stage anchor row before emitting "
            f"{anchor_summary_path} (found {len(anchor_rows)})."
        )
    anchor_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with anchor_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow({key: anchor_rows[0].get(key, "") for key in summary_fieldnames})

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
        f.write("| run_id | dataset_profile | modes | skip | width | fno_blocks | amp_ssim | amp_mae | is_feasible |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['run_id']} | {row['dataset_profile']} | {row['modes']} | {row['skip']} | "
                f"{row['width']} | {row['fno_blocks']} | {float(row['amp_ssim']):.6f} | "
                f"{float(row['amp_mae']):.6f} | {row['is_feasible']} |\n"
            )

    return 0


def run_phase_guardrail_validation(args: argparse.Namespace) -> int:
    """Validate persisted phase-drop fields against baseline provenance."""
    if args.summary_csv is None:
        raise StageValidationError("--validate-phase-guardrail requires --summary-csv")
    if args.baseline_summary is None:
        raise StageValidationError("--validate-phase-guardrail requires --baseline-summary")
    if args.write_validation_report is None:
        raise StageValidationError("--validate-phase-guardrail requires --write-validation-report")

    summary_fieldnames, summary_rows = _read_csv(args.summary_csv)
    if not summary_rows:
        raise StageValidationError(f"Summary CSV is empty: {args.summary_csv}")
    _require_columns(
        args.summary_csv,
        summary_fieldnames,
        [
            "run_id",
            "phase_ssim",
            "phase_ssim_drop_vs_baseline",
            "phase_guardrail_pass",
            "is_feasible",
        ],
    )

    _, baseline_rows = _read_csv(args.baseline_summary)
    if not baseline_rows:
        raise StageValidationError(f"Baseline summary CSV is empty: {args.baseline_summary}")

    baseline_phase_by_run: dict[str, float] = {}
    for source_row in baseline_rows:
        source_run_id = str(source_row.get("run_id", "")).strip()
        if not source_run_id:
            continue
        amp_ssim = _safe_float(source_row.get("amp_ssim"), "amp_ssim", default=0.0)
        baseline_phase_by_run[source_run_id] = _safe_float(
            source_row.get("phase_ssim"),
            "phase_ssim",
            default=amp_ssim,
        )
    if not baseline_phase_by_run:
        raise StageValidationError(
            f"Could not resolve baseline run_id -> phase_ssim map from {args.baseline_summary}"
        )

    singleton_baseline: tuple[str, float] | None = None
    if len(baseline_phase_by_run) == 1:
        singleton_baseline = next(iter(baseline_phase_by_run.items()))

    failures: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    requested_threshold = float(args.max_phase_ssim_drop)
    for raw_row in summary_rows:
        run_id = str(raw_row.get("run_id", "")).strip()
        source_run_id = str(raw_row.get("source_run_id", "")).strip()
        phase_ssim = _safe_float(raw_row.get("phase_ssim"), "phase_ssim", default=0.0)
        persisted_drop = _safe_float(
            raw_row.get("phase_ssim_drop_vs_baseline"),
            "phase_ssim_drop_vs_baseline",
            default=0.0,
        )
        row_threshold = _safe_float(
            raw_row.get("max_phase_ssim_drop"),
            "max_phase_ssim_drop",
            default=requested_threshold,
        )
        effective_threshold = min(float(row_threshold), requested_threshold)
        persisted_pass = _bool_from_value(raw_row.get("phase_guardrail_pass", False))
        row_is_feasible = _bool_from_value(raw_row.get("is_feasible", False))

        baseline_phase: float | None = None
        baseline_run_id: str = ""
        baseline_source = ""
        if source_run_id:
            baseline_phase = baseline_phase_by_run.get(source_run_id)
            baseline_run_id = source_run_id
            baseline_source = "source_run_id"
        if baseline_phase is None and run_id in baseline_phase_by_run:
            baseline_phase = baseline_phase_by_run[run_id]
            baseline_run_id = run_id
            baseline_source = "run_id"
        if baseline_phase is None and singleton_baseline is not None:
            baseline_run_id, baseline_phase = singleton_baseline
            baseline_source = "singleton_baseline"

        reasons: list[str] = []
        recomputed_drop: float | None = None
        recomputed_pass: bool | None = None
        if baseline_phase is None:
            reasons.append("missing_baseline_provenance")
        else:
            recomputed_drop = _compute_phase_drop(
                phase_ssim=phase_ssim,
                baseline_phase_ssim=float(baseline_phase),
            )
            recomputed_pass = bool(recomputed_drop <= effective_threshold)
            if not math.isclose(
                float(persisted_drop),
                float(recomputed_drop),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                reasons.append("drop_mismatch")
            if persisted_pass != recomputed_pass:
                reasons.append("guardrail_pass_mismatch")
            if (not recomputed_pass) and row_is_feasible:
                reasons.append("feasible_guardrail_breach")

        validation_row = {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "phase_ssim": float(phase_ssim),
            "persisted_phase_drop": float(persisted_drop),
            "recomputed_phase_drop": None if recomputed_drop is None else float(recomputed_drop),
            "persisted_guardrail_pass": bool(persisted_pass),
            "recomputed_guardrail_pass": None if recomputed_pass is None else bool(recomputed_pass),
            "row_is_feasible": bool(row_is_feasible),
            "requested_max_phase_ssim_drop": float(requested_threshold),
            "row_max_phase_ssim_drop": float(row_threshold),
            "effective_max_phase_ssim_drop": float(effective_threshold),
            "baseline_phase_ssim": None if baseline_phase is None else float(baseline_phase),
            "baseline_run_id": baseline_run_id,
            "baseline_source": baseline_source,
            "valid": len(reasons) == 0,
            "failure_reasons": reasons,
        }
        validation_rows.append(validation_row)
        if reasons:
            failures.append(validation_row)

    report_payload = {
        "summary_csv": str(args.summary_csv),
        "baseline_summary": str(args.baseline_summary),
        "max_phase_ssim_drop": float(requested_threshold),
        "checked_rows": len(validation_rows),
        "failed_rows": len(failures),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": validation_rows,
    }
    args.write_validation_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_validation_report.write_text(json.dumps(report_payload, indent=2) + "\n")

    if failures:
        raise StageValidationError(
            "Phase guardrail semantic validation failed for "
            f"{len(failures)} row(s); see {args.write_validation_report}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
        if args.validate_phase_guardrail:
            return run_phase_guardrail_validation(args)
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

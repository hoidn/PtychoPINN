#!/usr/bin/env python3
"""Verify a fresh checkpoint-restored PtychoViT initial baseline output bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


REQUIRED_FILES = (
    "metrics_by_model.json",
    "recons/pinn_ptychovit/recon.npz",
    "recons/gt/recon.npz",
    "visuals/amp_phase_pinn_ptychovit.png",
    "visuals/compare_amp_phase.png",
    "runs/pinn_ptychovit/manifest.json",
    "runs/pinn_ptychovit/stdout.log",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify fresh ptychovit initial baseline artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-external-checkpoint",
        action="store_true",
        help="Allow manifest checkpoint path to point outside output-dir/runs/pinn_ptychovit/best_model.pth",
    )
    parser.add_argument(
        "--covered-region-amp-std-min",
        type=float,
        default=1.0e-6,
        help="Minimum amplitude standard deviation required on the scan-covered region",
    )
    return parser.parse_args(argv)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _iter_metric_scalars(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            yield from _iter_metric_scalars(item)
        return
    yield value


def _validate_metrics_payload(metrics_by_model: dict, errors: list[str]) -> None:
    if "pinn_ptychovit" not in metrics_by_model:
        errors.append("metrics_by_model.json missing 'pinn_ptychovit' entry")
        return

    model_payload = metrics_by_model["pinn_ptychovit"]
    metrics = model_payload.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append("pinn_ptychovit.metrics must be an object")
        return

    for metric_name, values in metrics.items():
        for value in _iter_metric_scalars(values):
            if value is None:
                continue
            if not _is_finite_number(value):
                errors.append(f"metric {metric_name} has non-finite/non-numeric value: {value!r}")


def _resolve_test_para_path(output_dir: Path, manifest: dict) -> Path | None:
    candidates: list[Path] = []

    manifest_path = manifest.get("test_para")
    if manifest_path:
        candidates.append(Path(manifest_path))

    candidates.append(output_dir / "runs" / "pinn_ptychovit" / "bridge_work" / "data" / "test_para.hdf5")
    candidates.append(output_dir / "interop" / "test_para.hdf5")

    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_test_dp_path(output_dir: Path, manifest: dict) -> Path | None:
    candidates: list[Path] = []

    manifest_path = manifest.get("test_dp")
    if manifest_path:
        candidates.append(Path(manifest_path))

    candidates.append(output_dir / "runs" / "pinn_ptychovit" / "bridge_work" / "data" / "test_dp.hdf5")
    candidates.append(output_dir / "interop" / "test_dp.hdf5")

    for path in candidates:
        if path.exists():
            return path
    return None


def _validate_scan_position_contract(output_dir: Path, manifest: dict, errors: list[str]) -> None:
    para_path = _resolve_test_para_path(output_dir, manifest)
    if para_path is None:
        errors.append("unable to locate interop test_para.hdf5 for scan-position validation")
        return

    with h5py.File(para_path, "r") as para_file:
        if "probe_position_x_m" not in para_file or "probe_position_y_m" not in para_file:
            errors.append(f"{para_path} missing probe position datasets")
            return
        x = np.asarray(para_file["probe_position_x_m"])
        y = np.asarray(para_file["probe_position_y_m"])

    if x.ndim != 1 or y.ndim != 1:
        errors.append("probe_position_x_m/probe_position_y_m must be rank-1 vectors")
        return
    if x.shape[0] == 0 or y.shape[0] == 0:
        errors.append("probe position vectors must be non-empty")
        return
    if x.shape[0] != y.shape[0]:
        errors.append("probe position vectors must have identical lengths")
        return
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        errors.append("probe position vectors must contain only finite values")
    if np.unique(x).size <= 1:
        errors.append("probe_position_x_m must be non-constant")
    if np.unique(y).size <= 1:
        errors.append("probe_position_y_m must be non-constant")


def _load_reconstruction_amplitude(recon_path: Path) -> np.ndarray:
    with np.load(recon_path, allow_pickle=True) as npz:
        if "amp" in npz:
            amp = np.asarray(npz["amp"])
        elif "YY_pred" in npz:
            amp = np.abs(np.asarray(npz["YY_pred"]))
        else:
            raise ValueError(f"{recon_path} missing both 'amp' and 'YY_pred' arrays")
    amp = np.asarray(amp)
    if amp.ndim != 2:
        raise ValueError(f"Expected reconstruction amplitude rank-2, got {amp.shape}")
    if not np.all(np.isfinite(amp)):
        raise ValueError("Reconstruction amplitude contains non-finite values")
    return amp.astype(np.float32)


def _build_coverage_mask(
    *,
    object_shape_hw: tuple[int, int],
    positions_px: np.ndarray,
    patch_shape_hw: tuple[int, int],
) -> np.ndarray:
    h_obj, w_obj = int(object_shape_hw[0]), int(object_shape_hw[1])
    h_patch, w_patch = int(patch_shape_hw[0]), int(patch_shape_hw[1])
    mask = np.zeros((h_obj, w_obj), dtype=bool)
    half_h = (h_patch - 1.0) / 2.0
    half_w = (w_patch - 1.0) / 2.0
    for cy, cx in positions_px:
        top = int(np.floor(float(cy) - half_h))
        left = int(np.floor(float(cx) - half_w))
        bottom = top + h_patch
        right = left + w_patch

        y0 = max(0, top)
        x0 = max(0, left)
        y1 = min(h_obj, bottom)
        x1 = min(w_obj, right)
        if y0 >= y1 or x0 >= x1:
            continue
        mask[y0:y1, x0:x1] = True
    return mask


def _validate_recon_covered_region_contract(
    output_dir: Path,
    manifest: dict,
    errors: list[str],
    *,
    covered_region_amp_std_min: float,
) -> None:
    para_path = _resolve_test_para_path(output_dir, manifest)
    dp_path = _resolve_test_dp_path(output_dir, manifest)
    if para_path is None or dp_path is None:
        errors.append("unable to locate interop test_dp/test_para files for covered-region reconstruction validation")
        return

    recon_path = output_dir / "recons" / "pinn_ptychovit" / "recon.npz"
    try:
        recon_amp = _load_reconstruction_amplitude(recon_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        errors.append(f"failed to load reconstruction amplitude for covered-region validation: {exc}")
        return

    with h5py.File(para_path, "r") as para_file, h5py.File(dp_path, "r") as dp_file:
        if "object" not in para_file:
            errors.append(f"{para_path} missing object dataset required for covered-region validation")
            return
        if "probe_position_x_m" not in para_file or "probe_position_y_m" not in para_file:
            errors.append(f"{para_path} missing probe position datasets required for covered-region validation")
            return
        if "dp" not in dp_file:
            errors.append(f"{dp_path} missing dp dataset required for covered-region validation")
            return

        obj_dset = para_file["object"]
        pixel_h = float(obj_dset.attrs.get("pixel_height_m", np.nan))
        pixel_w = float(obj_dset.attrs.get("pixel_width_m", np.nan))
        if not np.isfinite(pixel_h) or not np.isfinite(pixel_w) or pixel_h <= 0.0 or pixel_w <= 0.0:
            errors.append("object pixel size attrs must be finite and > 0 for covered-region validation")
            return

        x_m = np.asarray(para_file["probe_position_x_m"], dtype=np.float64)
        y_m = np.asarray(para_file["probe_position_y_m"], dtype=np.float64)
        dp = np.asarray(dp_file["dp"])
        if dp.ndim != 3:
            errors.append("dp must be rank-3 for covered-region validation")
            return
        if x_m.ndim != 1 or y_m.ndim != 1 or x_m.shape[0] != y_m.shape[0]:
            errors.append("probe position vectors must be rank-1 and equal-length for covered-region validation")
            return
        if x_m.shape[0] != dp.shape[0]:
            errors.append("dp/position scan-count mismatch in covered-region validation")
            return

        object_arr = np.squeeze(np.asarray(obj_dset))
        if object_arr.ndim == 3:
            object_arr = object_arr[0]
        if object_arr.ndim != 2:
            errors.append(f"object dataset must be 2D (or [1,H,W]) for covered-region validation, got {object_arr.shape}")
            return
        obj_h, obj_w = int(object_arr.shape[0]), int(object_arr.shape[1])
        origin_y = float(np.round(obj_h / 2.0) + 0.5)
        origin_x = float(np.round(obj_w / 2.0) + 0.5)
        positions_px = np.column_stack([y_m / pixel_h + origin_y, x_m / pixel_w + origin_x]).astype(np.float32)
        coverage = _build_coverage_mask(
            object_shape_hw=(obj_h, obj_w),
            positions_px=positions_px,
            patch_shape_hw=(int(dp.shape[1]), int(dp.shape[2])),
        )

    if recon_amp.shape != coverage.shape:
        errors.append(
            f"reconstruction amplitude shape {recon_amp.shape} must match para object shape {coverage.shape} for covered-region validation"
        )
        return
    covered_values = recon_amp[coverage]
    if covered_values.size == 0:
        errors.append("covered-region mask is empty; cannot validate reconstruction collapse")
        return
    amp_std = float(np.std(covered_values))
    if amp_std < float(covered_region_amp_std_min):
        errors.append(
            f"covered-region amplitude std {amp_std:.3e} below threshold {covered_region_amp_std_min:.3e}; "
            "likely position-aware stitching or collapse regression"
        )


def verify_output(
    output_dir: Path,
    *,
    allow_external_checkpoint: bool,
    covered_region_amp_std_min: float = 1.0e-6,
) -> list[str]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        path = output_dir / rel
        if not path.exists():
            errors.append(f"missing required file: {path}")

    if errors:
        return errors

    stdout_text = (output_dir / "runs" / "pinn_ptychovit" / "stdout.log").read_text()
    if "Skipped backend execution" in stdout_text:
        errors.append("stdout.log indicates skipped backend execution (stale artifact reuse)")
    if "Normalization file not found" in stdout_text:
        errors.append("stdout.log indicates normalization fallback (missing normalization dictionary)")

    manifest_path = output_dir / "runs" / "pinn_ptychovit" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mode") != "inference":
        errors.append(f"manifest.mode must be 'inference' (found {manifest.get('mode')!r})")
    if manifest.get("training_returncode") is not None:
        errors.append(
            "manifest.training_returncode must be null for checkpoint-restored inference baseline"
        )

    checkpoint_value = manifest.get("checkpoint")
    if not checkpoint_value:
        errors.append("manifest.checkpoint is missing")
    else:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.exists():
            errors.append(f"manifest checkpoint path does not exist: {checkpoint_path}")
        expected_checkpoint = output_dir / "runs" / "pinn_ptychovit" / "best_model.pth"
        if not allow_external_checkpoint and checkpoint_path.resolve() != expected_checkpoint.resolve():
            errors.append(
                "manifest checkpoint path must equal "
                f"{expected_checkpoint} (found {checkpoint_path})"
            )

    metrics_by_model = json.loads((output_dir / "metrics_by_model.json").read_text())
    _validate_metrics_payload(metrics_by_model, errors)
    _validate_scan_position_contract(output_dir, manifest, errors)
    _validate_recon_covered_region_contract(
        output_dir,
        manifest,
        errors,
        covered_region_amp_std_min=covered_region_amp_std_min,
    )

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = verify_output(
        args.output_dir,
        allow_external_checkpoint=args.allow_external_checkpoint,
        covered_region_amp_std_min=float(args.covered_region_amp_std_min),
    )
    if errors:
        print("Fresh baseline verification FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Fresh baseline verification PASSED")
    print(f"- output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

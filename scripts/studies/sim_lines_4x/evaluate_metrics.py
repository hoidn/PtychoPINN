#!/usr/bin/env python3
"""Evaluate SIM-LINES-4X reconstructions against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ptycho.config.config import InferenceConfig, ModelConfig
from ptycho.evaluation import eval_reconstruction
from ptycho.image.cropping import align_for_evaluation_with_registration
from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
from ptycho import loader
from ptycho import nbutils
from ptycho import tf_helper
from scripts.simulation.synthetic_helpers import (
    make_lines_object,
    make_probe,
    simulate_nongrid_raw_data,
    split_raw_data_by_axis,
)
from scripts.studies.sim_lines_4x.pipeline import CUSTOM_PROBE_PATH, RunParams


def _load_metadata(metadata_path: Path) -> Dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    defaults = RunParams()
    metadata.setdefault("object_seed", defaults.object_seed)
    metadata.setdefault("buffer", defaults.buffer)
    metadata.setdefault("reassemble_M", defaults.reassemble_M)
    return metadata


def _make_test_split(
    metadata: Dict[str, object],
    image_multiplier: int,
) -> Tuple[object, np.ndarray]:
    object_guess = make_lines_object(
        int(metadata["object_size"]),
        seed=int(metadata["object_seed"]),
    )
    probe_guess = make_probe(
        int(metadata["N"]),
        mode=str(metadata["probe_mode"]),
        path=CUSTOM_PROBE_PATH,
    )
    raw_data = simulate_nongrid_raw_data(
        object_guess,
        probe_guess,
        N=int(metadata["N"]),
        n_images=int(metadata["total_images"]) * image_multiplier,
        nphotons=float(metadata["nphotons"]),
        seed=42,
        buffer=float(metadata["buffer"]),
    )
    _, test_raw = split_raw_data_by_axis(
        raw_data,
        split_fraction=float(metadata["split_fraction"]),
        axis="y",
    )
    return test_raw, object_guess


def _run_inference(
    model_dir: Path,
    test_raw,
    metadata: Dict[str, object],
    nsamples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    infer_config = InferenceConfig(
        model=ModelConfig(N=int(metadata["N"]), gridsize=int(metadata["gridsize"])),
        model_path=model_dir,
        test_data_file=CUSTOM_PROBE_PATH,
        n_groups=nsamples,
        neighbor_count=int(metadata["neighbor_count"]),
        backend="tensorflow",
    )
    model, params_dict = load_inference_bundle_with_backend(model_dir, infer_config)
    grouped = test_raw.generate_grouped_data(
        int(metadata["N"]),
        K=int(metadata["neighbor_count"]),
        nsamples=nsamples,
        seed=seed,
        gridsize=int(params_dict.get("gridsize", metadata["gridsize"])),
    )
    container = loader.load(lambda: grouped, test_raw.probeGuess, which=None, create_split=False)
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(container, diffraction_to_obj=model)
    obj_image = tf_helper.reassemble_position(
        obj_tensor_full,
        global_offsets,
        M=int(metadata["reassemble_M"]),
    )
    return obj_image, global_offsets


def _evaluate_case(
    model_dir: Path,
    metadata: Dict[str, object],
    nsamples: int,
    seed: int,
    image_multiplier: int,
    save_images_dir: Path | None,
    skip_registration: bool,
    registration_upsample: int,
    registration_border_crop: int,
) -> Dict[str, Tuple[float, float]]:
    test_raw, object_guess = _make_test_split(metadata, image_multiplier)
    obj_image, global_offsets = _run_inference(model_dir, test_raw, metadata, nsamples, seed)

    scan_coords_xy = np.squeeze(global_offsets)
    if scan_coords_xy.ndim != 2 or scan_coords_xy.shape[1] != 2:
        raise ValueError(f"Unexpected global_offsets shape: {global_offsets.shape}")
    scan_coords_yx = scan_coords_xy[:, [1, 0]]

    recon_aligned, gt_aligned, registration_offset = align_for_evaluation_with_registration(
        reconstruction_image=obj_image,
        ground_truth_image=object_guess,
        scan_coords_yx=scan_coords_yx,
        stitch_patch_size=int(metadata["reassemble_M"]),
        upsample_factor=registration_upsample,
        border_crop=registration_border_crop,
        skip_registration=skip_registration,
    )

    if save_images_dir is not None:
        from ptycho.workflows.components import save_outputs

        scenario_dir = save_images_dir / metadata["scenario"]
        eval_dir = scenario_dir / f"eval_outputs_nsamples{nsamples}_seed{seed}"
        recon_dir = eval_dir / "reconstruction"
        gt_dir = eval_dir / "ground_truth"

        save_outputs(
            np.abs(recon_aligned),
            np.angle(recon_aligned),
            {},
            str(recon_dir),
            crop_mode="none",
        )
        save_outputs(
            np.abs(gt_aligned),
            np.angle(gt_aligned),
            {},
            str(gt_dir),
            crop_mode="none",
        )

    recon_aligned = recon_aligned[..., None]
    gt_aligned = gt_aligned[..., None]

    metrics = eval_reconstruction(recon_aligned, gt_aligned, label=metadata["scenario"])
    metrics["registration_offset"] = registration_offset
    return metrics


def evaluate_scenario(
    scenario_dir: Path,
    *,
    nsamples: int,
    seed: int,
    gs2_image_multiplier: int = 1,
    save_images_dir: Path | None = None,
    skip_registration: bool = False,
    registration_upsample: int = 50,
    registration_border_crop: int = 2,
) -> Dict[str, Tuple[float, float]]:
    """Evaluate a SIM-LINES-4X scenario directory and return metrics."""
    metadata_path = scenario_dir / "run_metadata.json"
    metadata = _load_metadata(metadata_path)
    metadata["scenario"] = scenario_dir.name
    model_dir = scenario_dir / "train_outputs"

    image_multiplier = 1
    if int(metadata["gridsize"]) == 2:
        image_multiplier = int(gs2_image_multiplier)

    return _evaluate_case(
        model_dir,
        metadata,
        nsamples,
        seed,
        image_multiplier,
        save_images_dir,
        skip_registration,
        registration_upsample,
        registration_border_crop,
    )


def _sanitize_metric(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _serialize_metrics(metrics: Dict[str, Tuple[float, float]]) -> Dict[str, List[float | None]]:
    output: Dict[str, List[float | None]] = {}
    for key in ("mae", "mse", "psnr", "ssim", "ms_ssim", "frc50"):
        amp_val, phase_val = metrics[key]
        output[key] = [
            _sanitize_metric(amp_val),
            _sanitize_metric(phase_val),
        ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--gs2-image-multiplier",
        type=int,
        default=1,
        help="Multiplier for gs2 test image pool size (keeps group count fixed).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario list (default: all).",
    )
    parser.add_argument(
        "--save-images-dir",
        type=Path,
        default=None,
        help="Optional base directory for evaluation PNG outputs.",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Skip fine-scale registration before evaluation.",
    )
    parser.add_argument(
        "--registration-upsample",
        type=int,
        default=50,
        help="Upsample factor for registration.",
    )
    parser.add_argument(
        "--registration-border-crop",
        type=int,
        default=2,
        help="Border crop applied after registration.",
    )
    args = parser.parse_args()

    scenarios = [
        "gs1_ideal",
        "gs1_custom",
        "gs2_ideal",
        "gs2_custom",
    ]
    if args.scenarios:
        scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]

    cases = []
    for scenario in scenarios:
        scenario_dir = args.output_root / scenario
        metadata_path = scenario_dir / "run_metadata.json"
        metadata = _load_metadata(metadata_path)
        metadata["scenario"] = scenario
        model_dir = scenario_dir / "train_outputs"
        image_multiplier = 1
        if int(metadata["gridsize"]) == 2:
            image_multiplier = int(args.gs2_image_multiplier)

        metrics = _evaluate_case(
            model_dir,
            metadata,
            args.nsamples,
            args.seed,
            image_multiplier,
            args.save_images_dir,
            args.skip_registration,
            args.registration_upsample,
            args.registration_border_crop,
        )
        registration_offset = metrics.pop("registration_offset")
        cases.append(
            {
                "id": scenario,
                "label": f"gs{metadata['gridsize']} + {metadata['probe_mode']} probe",
                "metrics": _serialize_metrics(metrics),
                "registration_offset": (
                    [float(registration_offset[0]), float(registration_offset[1])]
                    if registration_offset is not None
                    else None
                ),
            }
        )

    payload = {
        "title": "SIM-LINES-4X reconstruction metrics",
        "notes": {
            "metric_source": "ptycho.evaluation.eval_reconstruction",
            "test_split": "bottom-half spatial split (y-axis)",
            "test_sampling": (
                "random subsample of test split "
                f"(nsamples={args.nsamples}, seed={args.seed}, "
                f"gs2_image_multiplier={args.gs2_image_multiplier})"
            ),
            "registration": (
                "enabled" if not args.skip_registration else "skipped"
            ),
            "registration_upsample": args.registration_upsample,
            "registration_border_crop": args.registration_border_crop,
        },
        "cases": cases,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        existing_cases = {case["id"]: case for case in existing.get("cases", [])}
        for case in cases:
            existing_cases[case["id"]] = case
        payload["cases"] = list(existing_cases.values())
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

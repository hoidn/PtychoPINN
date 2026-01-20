#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.studies.sim_lines_4x.pipeline import (  # noqa: E402
    PREDICTION_SCALE_CHOICES,
    ScenarioSpec,
    run_scenario,
)
from scripts.studies.sim_lines_4x import evaluate_metrics  # noqa: E402


def _parse_probe_scales(value: str) -> List[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("probe_scales must be a non-empty, comma-separated list")
    return [float(item) for item in items]


def _format_scale_label(scale: float) -> str:
    text = f"{scale:.4g}"
    return text.replace(".", "p")


def _serialize_metrics(metrics: Dict[str, Tuple[float, float]]) -> Dict[str, List[float | None]]:
    output: Dict[str, List[float | None]] = {}
    for key in ("mae", "mse", "psnr", "ssim", "ms_ssim", "frc50"):
        amp_val, phase_val = metrics[key]
        output[key] = [
            _sanitize_metric(amp_val),
            _sanitize_metric(phase_val),
        ]
    return output


def _pick_best(results: List[Dict[str, object]]) -> Dict[str, object] | None:
    valid = [item for item in results if item.get("ssim_amplitude") is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: float(item["ssim_amplitude"]))


def _sanitize_metric(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid sweep of probe_scale for gs2 + idealized probe using SSIM scoring.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".artifacts/sim_lines_4x_probe_scale_sweep"),
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--probe-scales",
        type=str,
        default="2,4,6,8,10",
        help="Comma-separated probe_scale values to test.",
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        default=5,
        help="Number of training epochs per scale.",
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
        "--object-seed",
        type=int,
        default=42,
        help="Random seed for object generation.",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=42,
        help="Random seed for simulation data generation.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1000,
        help="Number of groups to evaluate for SSIM.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=7,
        help="Random seed for evaluation grouping.",
    )
    parser.add_argument(
        "--gs2-image-multiplier",
        type=int,
        default=1,
        help="Multiplier for gs2 test image pool size during evaluation.",
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write JSON summary (defaults to output-root).",
    )
    parser.add_argument(
        "--prediction-scale-source",
        choices=PREDICTION_SCALE_CHOICES,
        default="none",
        help="Prediction scaling strategy applied during each scenario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_scales = _parse_probe_scales(args.probe_scales)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (output_root / "probe_scale_sweep.json")

    results: List[Dict[str, object]] = []

    for scale in probe_scales:
        scale_label = _format_scale_label(scale)
        scenario_name = f"gs2_ideal_scale_{scale_label}"
        scenario = ScenarioSpec(
            name=scenario_name,
            gridsize=2,
            probe_mode="idealized",
            probe_scale=scale,
        )
        run_scenario(
            scenario,
            output_root=output_root,
            nepochs=args.nepochs,
            image_multiplier=args.image_multiplier,
            group_multiplier=args.group_multiplier,
            object_seed=args.object_seed,
            sim_seed=args.sim_seed,
            prediction_scale_source=args.prediction_scale_source,
        )
        scenario_dir = output_root / scenario_name
        metrics = None
        error = None
        registration_fallback = False
        try:
            metrics = evaluate_metrics.evaluate_scenario(
                scenario_dir,
                nsamples=args.nsamples,
                seed=args.eval_seed,
                gs2_image_multiplier=args.gs2_image_multiplier,
                skip_registration=args.skip_registration,
                registration_upsample=args.registration_upsample,
                registration_border_crop=args.registration_border_crop,
            )
        except ValueError as exc:
            error = str(exc)
            if not args.skip_registration and "NaN values" in error:
                try:
                    metrics = evaluate_metrics.evaluate_scenario(
                        scenario_dir,
                        nsamples=args.nsamples,
                        seed=args.eval_seed,
                        gs2_image_multiplier=args.gs2_image_multiplier,
                        skip_registration=True,
                        registration_upsample=args.registration_upsample,
                        registration_border_crop=args.registration_border_crop,
                    )
                    registration_fallback = True
                except ValueError as fallback_exc:
                    error = str(fallback_exc)

        ssim_amp = None
        ssim_phase = None
        serialized_metrics: Dict[str, List[float | None]] = {}
        if metrics is not None:
            ssim_amp, ssim_phase = metrics["ssim"]
            ssim_amp = _sanitize_metric(ssim_amp)
            ssim_phase = _sanitize_metric(ssim_phase)
            serialized_metrics = _serialize_metrics(metrics)
        results.append(
            {
                "scenario": scenario_name,
                "probe_scale": scale,
                "ssim_amplitude": ssim_amp,
                "ssim_phase": ssim_phase,
                "metrics": serialized_metrics,
                "registration_fallback": registration_fallback,
                "error": error,
            }
        )

    best = _pick_best(results)
    payload = {
        "title": "SIM-LINES-4X gs2_ideal probe_scale sweep",
        "metric": "ssim_amplitude",
        "probe_scales": probe_scales,
        "results": results,
        "best": best,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if best is not None:
        print(
            "Best scale:",
            best["probe_scale"],
            "ssim_amp=",
            best["ssim_amplitude"],
        )
    else:
        print("No valid SSIM scores found.")


if __name__ == "__main__":
    main()

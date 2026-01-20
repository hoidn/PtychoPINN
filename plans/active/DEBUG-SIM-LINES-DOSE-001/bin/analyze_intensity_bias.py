"""Aggregate intensity/bias telemetry across plan-local scenarios.

This CLI ingests one or more scenario directories (each created by
``run_phase_c2_scenario.py``) and generates JSON/Markdown summaries that track:

* amplitude / phase bias statistics from ``comparison_metrics.json``
* intensity-scale parity between bundle + legacy params
* normalization stage statistics from ``intensity_stats.json``
* inference canvas / NaN telemetry from ``inference_outputs/stats.json``
* training NaN indicators from ``train_outputs/history_summary.json``

The JSON payload is meant for further automation while the Markdown write-up
provides a quick, human-readable overview for the fix plan.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


REQUIRED_FILES = {
    "run_metadata": "run_metadata.json",
    "intensity_stats": "intensity_stats.json",
    "inference_stats": "inference_outputs/stats.json",
    "comparison_metrics": "comparison_metrics.json",
    "training_summary": "train_outputs/history_summary.json",
    "prediction_amplitude": "inference_outputs/amplitude.npy",
    "ground_truth_amplitude": "ground_truth_amp.npy",
}

STAGE_LABELS = [
    ("raw_diffraction", "Raw diffraction"),
    ("grouped_diffraction", "Grouped diffraction"),
    ("grouped_X_full", "Grouped X (normalized)"),
    ("container_X", "Container X"),
]

RATIO_LABELS = [
    ("raw_to_grouped", "Raw → grouped"),
    ("grouped_to_normalized", "Grouped → normalized"),
    ("normalized_to_prediction", "Normalized → prediction"),
    ("prediction_to_truth", "Prediction → truth"),
]

# Normalization invariant tolerance per specs/spec-ptycho-core.md §Normalization Invariants
NORMALIZATION_INVARIANT_TOLERANCE = 0.05  # 5%


@dataclass(frozen=True)
class ScenarioInput:
    """Directory bundle required for analysis."""

    name: str
    base_dir: Path
    run_metadata: Path
    intensity_stats: Path
    inference_stats: Path
    comparison_metrics: Path
    training_summary: Path
    prediction_amplitude: Path
    ground_truth_amplitude: Path
    prediction_amplitude_unscaled: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize intensity/bias telemetry for one or more scenarios."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help=(
            "Scenario spec in the form name=/path/to/scenario "
            "(may be provided multiple times)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where bias_summary.json and .md will be written.",
    )
    return parser.parse_args()


def parse_scenario_arg(raw_value: str) -> Tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError(
            f"Scenario argument '{raw_value}' must use the form name=/path"
        )
    name, raw_path = raw_value.split("=", maxsplit=1)
    name = name.strip()
    if not name:
        raise ValueError(
            f"Scenario argument '{raw_value}' is missing the scenario name prefix"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scenario directory not found: {path}")
    return name, path


def build_scenario_input(name: str, base_dir: Path) -> ScenarioInput:
    resolved: Dict[str, Path] = {}
    for key, relative in REQUIRED_FILES.items():
        candidate = base_dir / relative
        if not candidate.exists():
            raise FileNotFoundError(
                f"Scenario '{name}' is missing required file: {candidate}"
            )
        resolved[key] = candidate

    unscaled_path = base_dir / "inference_outputs/amplitude_unscaled.npy"
    return ScenarioInput(
        name=name,
        base_dir=base_dir,
        run_metadata=resolved["run_metadata"],
        intensity_stats=resolved["intensity_stats"],
        inference_stats=resolved["inference_stats"],
        comparison_metrics=resolved["comparison_metrics"],
        training_summary=resolved["training_summary"],
        prediction_amplitude=resolved["prediction_amplitude"],
        ground_truth_amplitude=resolved["ground_truth_amplitude"],
        prediction_amplitude_unscaled=unscaled_path if unscaled_path.exists() else None,
    )


def load_json(path: Path) -> MutableMapping[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metric_summary(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bias_summary": data.get("bias_summary"),
        "mae": data.get("mae"),
        "rmse": data.get("rmse"),
        "max_abs": data.get("max_abs"),
        "pearson_r": data.get("pearson_r"),
        "count": data.get("count"),
        "pred_stats": data.get("pred_stats"),
        "truth_stats": data.get("truth_stats"),
    }


def build_normalization_summary(
    stages: List[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for stage in stages:
        stats = stage.get("stats") or {}
        metadata = stage.get("metadata") or {}
        summaries.append(
            {
                "name": stage.get("name"),
                "source": metadata.get("source"),
                "count": metadata.get("count"),
                "gridsize": metadata.get("gridsize"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }
        )
    return summaries


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _format_stage_label(name: Optional[str]) -> str:
    if not name:
        return "n/a"
    for key, label in STAGE_LABELS:
        if key == name:
            return label
    lower = name.lower()
    if lower == "prediction":
        return "Prediction"
    if lower == "truth":
        return "Truth"
    return name


def extract_stage_means(stages: List[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    stage_means: Dict[str, Optional[float]] = {}
    for stage in stages:
        name = stage.get("name")
        stats = stage.get("stats") or {}
        mean = _as_float(stats.get("mean"))
        if name:
            stage_means[name] = mean
    return stage_means


def compute_normalization_invariant_check(
    ratios: Mapping[str, Optional[float]],
    tolerance: float = NORMALIZATION_INVARIANT_TOLERANCE,
) -> Dict[str, Any]:
    """Compute the product of stage ratios and check normalization invariants.

    Per `specs/spec-ptycho-core.md §Normalization Invariants`:
    - Symmetry SHALL hold: X_scaled = s · X
    - Training inputs, labels, and model outputs must maintain consistent scaling

    This function multiplies the chain of ratios (raw→grouped→normalized→prediction→truth)
    to determine the cumulative scaling factor and flags deviations vs tolerance.
    """
    ratio_keys = [
        "raw_to_grouped",
        "grouped_to_normalized",
        "normalized_to_prediction",
        "prediction_to_truth",
    ]

    # Gather individual ratios
    ratio_values: Dict[str, Optional[float]] = {}
    for key in ratio_keys:
        ratio_values[key] = ratios.get(key)

    # Compute cumulative products at each stage
    cumulative_products: Dict[str, Optional[float]] = {}
    running_product: Optional[float] = 1.0

    for key in ratio_keys:
        val = ratio_values.get(key)
        if running_product is not None and val is not None and np.isfinite(val):
            running_product = running_product * val
            cumulative_products[key] = running_product
        else:
            running_product = None
            cumulative_products[key] = None

    # The full chain product (raw→truth)
    full_chain_product = cumulative_products.get("prediction_to_truth")

    # For normalization invariance, the product should ideally be 1.0 (symmetric scaling)
    # Deviation from 1.0 indicates asymmetric normalization
    if full_chain_product is not None and np.isfinite(full_chain_product):
        deviation_from_unity = abs(full_chain_product - 1.0)
        relative_deviation = deviation_from_unity  # Since ideal is 1.0, relative deviation = absolute
        passes_tolerance = relative_deviation <= tolerance
    else:
        deviation_from_unity = None
        relative_deviation = None
        passes_tolerance = None

    # Identify which stages contribute most to deviation
    stage_deviations: List[Dict[str, Any]] = []
    for key in ratio_keys:
        val = ratio_values.get(key)
        if val is not None and np.isfinite(val):
            stage_dev = abs(val - 1.0)
            stage_deviations.append({
                "stage": key,
                "ratio": val,
                "deviation_from_unity": stage_dev,
                "contributes_loss": val < 1.0,  # <1.0 means amplitude reduction
                "contributes_gain": val > 1.0,  # >1.0 means amplitude amplification
            })

    # Sort by deviation magnitude (largest first)
    stage_deviations.sort(key=lambda x: x["deviation_from_unity"], reverse=True)

    # Identify the primary deviation source
    primary_deviation_stage = stage_deviations[0] if stage_deviations else None

    return {
        "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants",
        "tolerance": tolerance,
        "individual_ratios": ratio_values,
        "cumulative_products": cumulative_products,
        "full_chain_product": full_chain_product,
        "deviation_from_unity": deviation_from_unity,
        "relative_deviation": relative_deviation,
        "passes_tolerance": passes_tolerance,
        "stage_deviations": stage_deviations,
        "primary_deviation_stage": primary_deviation_stage,
        "symmetry_violated": (
            passes_tolerance is False
            if passes_tolerance is not None
            else None
        ),
    }


def build_stage_ratio_summary(
    stage_means: Mapping[str, Optional[float]],
    amplitude_metrics: Mapping[str, Any],
) -> Tuple[Dict[str, Optional[float]], Optional[Dict[str, Any]]]:
    ratios: Dict[str, Optional[float]] = {}
    transitions: List[Dict[str, Any]] = []

    def add_transition(
        key: str,
        from_label: str,
        to_label: str,
        from_value: Optional[float],
        to_value: Optional[float],
    ) -> None:
        ratio = _safe_ratio(to_value, from_value)
        ratios[key] = ratio
        transitions.append(
            {
                "key": key,
                "from_stage": from_label,
                "to_stage": to_label,
                "ratio": ratio,
            }
        )

    raw_mean = stage_means.get("raw_diffraction")
    grouped_mean = stage_means.get("grouped_diffraction")
    normalized_mean = stage_means.get("container_X")
    if normalized_mean is None:
        normalized_mean = stage_means.get("grouped_X_full")

    add_transition("raw_to_grouped", "raw_diffraction", "grouped_diffraction", raw_mean, grouped_mean)
    add_transition(
        "grouped_to_normalized",
        "grouped_diffraction",
        "container_X",
        grouped_mean,
        normalized_mean,
    )

    pred_stats = amplitude_metrics.get("pred_stats") or {}
    truth_stats = amplitude_metrics.get("truth_stats") or {}
    prediction_mean = _as_float(pred_stats.get("mean"))
    truth_mean = _as_float(truth_stats.get("mean"))

    add_transition(
        "normalized_to_prediction",
        "container_X",
        "prediction",
        normalized_mean,
        prediction_mean,
    )
    add_transition(
        "prediction_to_truth",
        "prediction",
        "truth",
        prediction_mean,
        truth_mean,
    )

    enriched: List[Dict[str, Any]] = []
    for entry in transitions:
        ratio = entry.get("ratio")
        if not isinstance(ratio, (int, float)):
            continue
        delta = ratio - 1.0
        updated = entry.copy()
        updated["delta"] = delta
        updated["abs_delta"] = abs(delta)
        enriched.append(updated)

    drops = [entry for entry in enriched if entry["delta"] < 0]
    if drops:
        largest_drop = min(drops, key=lambda entry: entry["delta"])
    elif enriched:
        largest_drop = max(enriched, key=lambda entry: entry["abs_delta"])
    else:
        largest_drop = None

    return ratios, largest_drop


def _load_numpy_array(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data, dtype=np.float64)


def _compute_ratio_stats(ratios: np.ndarray) -> Dict[str, Optional[float]]:
    if ratios.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(ratios.size),
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "p05": float(np.percentile(ratios, 5)),
        "p95": float(np.percentile(ratios, 95)),
    }


def _evaluate_scalar_errors(
    truth: np.ndarray,
    prediction: np.ndarray,
    scalar: float,
) -> Dict[str, float]:
    scaled_pred = prediction * scalar
    residual = truth - scaled_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {"mae": mae, "rmse": rmse}


def compute_scaling_analysis(
    prediction_path: Path,
    truth_path: Path,
    amplitude_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    prediction = _load_numpy_array(prediction_path).ravel()
    truth = _load_numpy_array(truth_path).ravel()
    if prediction.shape != truth.shape:
        raise ValueError(
            f"Amplitude array shape mismatch: pred={prediction.shape}, truth={truth.shape}"
        )
    finite_mask = np.isfinite(prediction) & np.isfinite(truth)
    finite_pred = prediction[finite_mask]
    finite_truth = truth[finite_mask]

    ratio_mask = np.abs(finite_pred) > 1e-12
    ratios = finite_truth[ratio_mask] / finite_pred[ratio_mask]
    ratio_stats = _compute_ratio_stats(ratios)

    if finite_pred.size:
        denom = float(np.sum(finite_pred**2))
        ls_scalar = float(np.sum(finite_pred * finite_truth) / denom) if denom else None
    else:
        ls_scalar = None

    candidate_scalars: Dict[str, Optional[float]] = {
        "ratio_mean": ratio_stats.get("mean"),
        "ratio_median": ratio_stats.get("median"),
        "ratio_p05": ratio_stats.get("p05"),
        "ratio_p95": ratio_stats.get("p95"),
        "least_squares": ls_scalar,
    }

    per_scalar_metrics: Dict[str, Dict[str, float]] = {}
    best_name: Optional[str] = None
    best_metrics: Optional[Dict[str, float]] = None
    for name, value in candidate_scalars.items():
        if value is None:
            continue
        metrics = _evaluate_scalar_errors(finite_truth, finite_pred, float(value))
        per_scalar_metrics[name] = metrics
        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_name = name
            best_metrics = metrics

    baseline_mae = _as_float(amplitude_metrics.get("mae"))
    baseline_rmse = _as_float(amplitude_metrics.get("rmse"))

    best_scalar_value: Optional[float]
    if best_name and best_metrics:
        best_scalar_value = candidate_scalars[best_name]
        best_scalar_payload: Optional[Dict[str, Any]] = {
            "name": best_name,
            "value": best_scalar_value,
            "mae": best_metrics["mae"],
            "rmse": best_metrics["rmse"],
        }
    else:
        best_scalar_value = None
        best_scalar_payload = None

    return {
        "count": int(finite_pred.size),
        "ratio_stats": ratio_stats,
        "candidate_scalars": candidate_scalars,
        "per_scalar_metrics": per_scalar_metrics,
        "best_scalar": best_scalar_payload,
        "baseline_errors": {
            "mae": baseline_mae,
            "rmse": baseline_rmse,
        },
        "mae_scaled": best_metrics["mae"] if best_metrics else None,
        "rmse_scaled": best_metrics["rmse"] if best_metrics else None,
        "best_scalar_name": best_name,
        "best_scalar_value": best_scalar_value,
    }


def gather_scenario_data(scenario: ScenarioInput) -> Dict[str, Any]:
    run_metadata = load_json(scenario.run_metadata)
    intensity_stats = load_json(scenario.intensity_stats)
    inference_stats = load_json(scenario.inference_stats)
    comparison_metrics = load_json(scenario.comparison_metrics)
    training_summary = load_json(scenario.training_summary)

    intensity_info = {
        "bundle": intensity_stats.get("bundle_intensity_scale"),
        "legacy_params": intensity_stats.get("legacy_params_intensity_scale"),
        "scale_delta": intensity_stats.get("scale_delta"),
        "normalize_gain": intensity_stats.get("normalize_gain"),
        "run_metadata": run_metadata.get("intensity_stats"),
    }

    bundle_val = intensity_info.get("bundle")
    legacy_val = intensity_info.get("legacy_params")
    if isinstance(bundle_val, (int, float)) and isinstance(
        legacy_val, (int, float)
    ):
        intensity_info["matches_within_1e-6"] = abs(bundle_val - legacy_val) <= 1e-6
        intensity_info["absolute_delta"] = abs(bundle_val - legacy_val)
    else:
        intensity_info["matches_within_1e-6"] = None
        intensity_info["absolute_delta"] = None

    amplitude_metrics = build_metric_summary(comparison_metrics.get("amplitude", {}))
    phase_metrics = build_metric_summary(comparison_metrics.get("phase", {}))

    raw_stage_payloads = intensity_stats.get("stages", [])
    normalization = build_normalization_summary(raw_stage_payloads)
    stage_means = extract_stage_means(raw_stage_payloads)
    stage_ratios, largest_drop = build_stage_ratio_summary(stage_means, amplitude_metrics)
    scaling_analysis = compute_scaling_analysis(
        scenario.prediction_amplitude_unscaled or scenario.prediction_amplitude,
        scenario.ground_truth_amplitude,
        amplitude_metrics,
    )
    # Compute normalization invariant check per specs/spec-ptycho-core.md §Normalization Invariants
    normalization_invariant = compute_normalization_invariant_check(stage_ratios)
    prediction_scale_meta = run_metadata.get("prediction_scale")

    offsets = (inference_stats.get("offsets") or {}).copy()
    inference_info = {
        "fits_canvas": inference_stats.get("fits_canvas"),
        "padded_size": inference_stats.get("padded_size"),
        "required_canvas": inference_stats.get("required_canvas"),
        "amplitude": inference_stats.get("amplitude"),
        "phase": inference_stats.get("phase"),
        "offsets": offsets,
    }

    nan_overview = training_summary.get("nan_overview") or {}
    raw_metrics = training_summary.get("metrics") or {}
    metrics_with_nan = sorted(
        [
            name
            for name, payload in raw_metrics.items()
            if isinstance(payload, Mapping) and payload.get("has_nan")
        ]
    )
    training_info = {
        "has_nan": bool(nan_overview.get("has_nan")),
        "nan_metrics": nan_overview.get("metrics") or {},
        "metrics_with_nan": metrics_with_nan,
    }

    return {
        "name": scenario.name,
        "paths": {
            "base_dir": str(scenario.base_dir),
            "run_metadata": str(scenario.run_metadata),
            "intensity_stats": str(scenario.intensity_stats),
            "inference_stats": str(scenario.inference_stats),
            "comparison_metrics": str(scenario.comparison_metrics),
            "training_summary": str(scenario.training_summary),
            "prediction_amplitude": str(scenario.prediction_amplitude),
            "ground_truth_amplitude": str(scenario.ground_truth_amplitude),
            "prediction_amplitude_unscaled": (
                str(scenario.prediction_amplitude_unscaled)
                if scenario.prediction_amplitude_unscaled
                else None
            ),
        },
        "intensity": intensity_info,
        "comparison": {
            "amplitude": amplitude_metrics,
            "phase": phase_metrics,
        },
        "normalization": normalization,
        "inference": inference_info,
        "prediction_scale": prediction_scale_meta,
        "prediction_scale_note": run_metadata.get("prediction_scale_note"),
        "training": training_info,
        "derived": {
            "stage_means": stage_means,
            "ratios": stage_ratios,
            "largest_drop": largest_drop,
            "scaling_analysis": scaling_analysis,
            "normalization_invariant": normalization_invariant,
        },
    }


def build_overview(reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    amp_bias_mean = {}
    phase_bias_mean = {}
    scale_deltas = {}
    training_flags = {}

    for name, payload in reports.items():
        amp_bias = (payload.get("comparison", {})
                    .get("amplitude", {})
                    .get("bias_summary", {}))
        phase_bias = (payload.get("comparison", {})
                      .get("phase", {})
                      .get("bias_summary", {}))
        amp_bias_mean[name] = amp_bias.get("mean") if isinstance(amp_bias, Mapping) else None
        phase_bias_mean[name] = phase_bias.get("mean") if isinstance(phase_bias, Mapping) else None

        intensity = payload.get("intensity", {})
        scale_deltas[name] = intensity.get("absolute_delta")
        training_flags[name] = bool(payload.get("training", {}).get("has_nan"))

    amp_best = _best_abs_value(amp_bias_mean)
    phase_best = _best_abs_value(phase_bias_mean)

    return {
        "amplitude_bias_mean": amp_bias_mean,
        "phase_bias_mean": phase_bias_mean,
        "amplitude_abs_delta_vs_best": _delta_vs_best(amp_bias_mean, amp_best),
        "phase_abs_delta_vs_best": _delta_vs_best(phase_bias_mean, phase_best),
        "intensity_scale_absolute_delta": scale_deltas,
        "training_nan_flags": training_flags,
    }


def _best_abs_value(values: Mapping[str, Optional[float]]) -> Optional[float]:
    finite = [
        abs(val)
        for val in values.values()
        if isinstance(val, (int, float))
    ]
    if not finite:
        return None
    return min(finite)


def _delta_vs_best(
    values: Mapping[str, Optional[float]],
    best_abs: Optional[float],
) -> Dict[str, Optional[float]]:
    if best_abs is None:
        return {name: None for name in values}
    output: Dict[str, Optional[float]] = {}
    for name, value in values.items():
        if isinstance(value, (int, float)):
            output[name] = abs(value) - best_abs
        else:
            output[name] = None
    return output


def analyze_scenarios(scenarios: List[ScenarioInput]) -> Dict[str, Any]:
    reports: Dict[str, Dict[str, Any]] = {}
    for scenario in scenarios:
        reports[scenario.name] = gather_scenario_data(scenario)
    generated_at = datetime.now(timezone.utc).isoformat()
    overview = build_overview(reports)
    return {
        "generated_at": generated_at,
        "scenario_count": len(scenarios),
        "scenarios": reports,
        "overview": overview,
    }


def fmt_value(value: Any, precision: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    if isinstance(value, str):
        return value
    if value is None:
        return "n/a"
    return str(value)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Intensity Bias Summary")
    lines.append("")
    lines.append("**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`")
    lines.append("")
    lines.append(f"- Generated at: {summary.get('generated_at')}")
    lines.append(f"- Scenario count: {summary.get('scenario_count')}")
    lines.append("")

    scenarios = summary.get("scenarios", {})
    if scenarios:
        lines.append(
            "| Scenario | Amp bias mean | Amp median | Phase bias mean | "
            "Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        for name, payload in scenarios.items():
            amp_bias = (payload.get("comparison", {})
                        .get("amplitude", {})
                        .get("bias_summary", {}) or {})
            phase_bias = (payload.get("comparison", {})
                          .get("phase", {})
                          .get("bias_summary", {}) or {})
            intensity = payload.get("intensity", {})
            scale_meta = payload.get("prediction_scale") or {}
            training = payload.get("training", {})
            lines.append(
                "| {name} | {amp_mean} | {amp_med} | {phase_mean} | {phase_med} | "
                "{bundle} | {legacy} | {delta} | {scale_mode} | {nan_flag} |".format(
                    name=name,
                    amp_mean=fmt_value(amp_bias.get("mean")),
                    amp_med=fmt_value(amp_bias.get("median")),
                    phase_mean=fmt_value(phase_bias.get("mean")),
                    phase_med=fmt_value(phase_bias.get("median")),
                    bundle=fmt_value(intensity.get("bundle")),
                    legacy=fmt_value(intensity.get("legacy_params")),
                    delta=fmt_value(intensity.get("absolute_delta")),
                    scale_mode=scale_meta.get("mode") or "n/a",
                    nan_flag="Yes" if training.get("has_nan") else "No",
                )
            )
        lines.append("")

    for name, payload in scenarios.items():
        lines.append(f"## Scenario: {name}")
        lines.append(f"- Base directory: `{payload['paths']['base_dir']}`")
        intensity = payload.get("intensity", {})
        lines.append(
            "- Intensity scale: bundle "
            f"{fmt_value(intensity.get('bundle'))} vs legacy "
            f"{fmt_value(intensity.get('legacy_params'))} "
            f"(Δ={fmt_value(intensity.get('absolute_delta'))})"
        )
        normalize_gain = intensity.get("normalize_gain")
        if normalize_gain is not None:
            lines.append(f"- normalize_data gain: {fmt_value(normalize_gain)}")
        scale_meta = payload.get("prediction_scale") or {}
        if scale_meta:
            lines.append(
                "- Prediction scale: "
                f"mode={scale_meta.get('mode')} "
                f"value={fmt_value(scale_meta.get('value'))} "
                f"source={scale_meta.get('source')}"
            )
        scale_note = payload.get("prediction_scale_note")
        if scale_note:
            lines.append(f"  * Note: {scale_note}")
        training = payload.get("training", {})
        nan_metrics = training.get("metrics_with_nan") or []
        lines.append(
            f"- Training NaNs: {'YES' if training.get('has_nan') else 'no'}"
            + (f" (metrics: {', '.join(nan_metrics)})" if nan_metrics else "")
        )
        inference = payload.get("inference", {})
        lines.append(
            "- Inference canvas: "
            f"padded={inference.get('padded_size')} "
            f"required={inference.get('required_canvas')} "
            f"fits_canvas={inference.get('fits_canvas')}"
        )
        derived = payload.get("derived") or {}
        largest_drop = derived.get("largest_drop")
        if largest_drop:
            lines.append(
                f"- **Largest drop:** "
                f"{_format_stage_label(largest_drop.get('from_stage'))} → "
                f"{_format_stage_label(largest_drop.get('to_stage'))} "
                f"(ratio={fmt_value(largest_drop.get('ratio'))}, "
                f"Δ={fmt_value(largest_drop.get('delta'))})"
            )
            lines.append(
                "  - Per `specs/spec-ptycho-core.md §Normalization Invariants`: "
                "symmetry SHALL hold for X_scaled = s · X"
            )
        lines.append("")
        stage_means_payload = derived.get("stage_means") or {}
        if stage_means_payload:
            lines.append("### Stage Means")
            lines.append("| Stage | Mean |")
            lines.append("| --- | ---: |")
            seen = set()
            for key, label in STAGE_LABELS:
                lines.append(f"| {label} | {fmt_value(stage_means_payload.get(key))} |")
                seen.add(key)
            for key, value in stage_means_payload.items():
                if key in seen:
                    continue
                lines.append(f"| {_format_stage_label(key)} | {fmt_value(value)} |")
            lines.append("")
        ratio_payload = derived.get("ratios") or {}
        if ratio_payload:
            lines.append("### Stage Ratios")
            lines.append("| Transition | Ratio |")
            lines.append("| --- | ---: |")
            seen = set()
            for key, label in RATIO_LABELS:
                lines.append(f"| {label} | {fmt_value(ratio_payload.get(key))} |")
                seen.add(key)
            for key, value in ratio_payload.items():
                if key in seen:
                    continue
                lines.append(f"| {key} | {fmt_value(value)} |")
            lines.append("")

        # Normalization Invariant section per specs/spec-ptycho-core.md §Normalization Invariants
        norm_inv = derived.get("normalization_invariant") or {}
        if norm_inv:
            lines.append("### Normalization Invariant Check")
            lines.append("")
            lines.append(f"**Spec Reference:** `{norm_inv.get('spec_reference', 'specs/spec-ptycho-core.md §Normalization Invariants')}`")
            lines.append("")
            lines.append(
                "Per the spec, symmetry SHALL hold: `X_scaled = s · X`. "
                "The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization."
            )
            lines.append("")

            # Show cumulative products table
            cumulative = norm_inv.get("cumulative_products") or {}
            individual = norm_inv.get("individual_ratios") or {}
            lines.append("| Stage Transition | Individual Ratio | Cumulative Product |")
            lines.append("| --- | ---: | ---: |")
            for key, label in RATIO_LABELS:
                ind_val = individual.get(key)
                cum_val = cumulative.get(key)
                lines.append(f"| {label} | {fmt_value(ind_val)} | {fmt_value(cum_val)} |")
            lines.append("")

            # Show invariant summary
            full_product = norm_inv.get("full_chain_product")
            deviation = norm_inv.get("deviation_from_unity")
            tolerance = norm_inv.get("tolerance", NORMALIZATION_INVARIANT_TOLERANCE)
            passes = norm_inv.get("passes_tolerance")
            symmetry_violated = norm_inv.get("symmetry_violated")

            status_emoji = "✅" if passes else ("❌" if passes is False else "⚠️")
            lines.append(f"**Full chain product (raw→truth):** {fmt_value(full_product)}")
            lines.append(f"**Deviation from unity:** {fmt_value(deviation)}")
            lines.append(f"**Tolerance:** {fmt_value(tolerance, precision=2)} ({tolerance*100:.0f}%)")
            lines.append(f"**Passes tolerance:** {status_emoji} {'Yes' if passes else ('No' if passes is False else 'N/A')}")
            if symmetry_violated:
                lines.append("")
                lines.append(
                    "⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude "
                    "as required by `specs/spec-ptycho-core.md §Normalization Invariants`."
                )
            lines.append("")

            # Show stage deviations breakdown
            stage_devs = norm_inv.get("stage_deviations") or []
            if stage_devs:
                lines.append("**Stage Deviation Breakdown** (sorted by impact):")
                lines.append("")
                lines.append("| Stage | Ratio | Deviation | Effect |")
                lines.append("| --- | ---: | ---: | --- |")
                for sd in stage_devs:
                    effect = "loss" if sd.get("contributes_loss") else "gain"
                    lines.append(
                        f"| {sd.get('stage')} | {fmt_value(sd.get('ratio'))} | "
                        f"{fmt_value(sd.get('deviation_from_unity'))} | {effect} |"
                    )
                lines.append("")

            primary = norm_inv.get("primary_deviation_stage")
            if primary:
                lines.append(
                    f"**Primary deviation source:** `{primary.get('stage')}` "
                    f"(ratio={fmt_value(primary.get('ratio'))}, "
                    f"deviation={fmt_value(primary.get('deviation_from_unity'))})"
                )
                lines.append("")

        lines.append("")
        scaling = derived.get("scaling_analysis") or {}
        if scaling:
            ratio_stats = scaling.get("ratio_stats") or {}
            lines.append("### Prediction ↔ Truth Scaling")
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            lines.append(
                f"| Ratio mean | {fmt_value(ratio_stats.get('mean'))} |"
            )
            lines.append(
                f"| Ratio median | {fmt_value(ratio_stats.get('median'))} |"
            )
            lines.append(
                f"| Ratio p05 | {fmt_value(ratio_stats.get('p05'))} |"
            )
            lines.append(
                f"| Ratio p95 | {fmt_value(ratio_stats.get('p95'))} |"
            )
            lines.append(
                f"| Ratio count | {fmt_value(ratio_stats.get('count'), precision=0)} |"
            )
            lines.append("")
            best_scalar = scaling.get("best_scalar") or {}
            baseline_errors = scaling.get("baseline_errors") or {}
            lines.append(
                f"* Best scalar: {(best_scalar.get('name') or 'n/a')} "
                f"({fmt_value(best_scalar.get('value'))})"
            )
            lines.append(
                f"* MAE baseline {fmt_value(baseline_errors.get('mae'))} → "
                f"{fmt_value(scaling.get('mae_scaled'))}; "
                f"RMSE baseline {fmt_value(baseline_errors.get('rmse'))} → "
                f"{fmt_value(scaling.get('rmse_scaled'))}"
            )
            candidates = scaling.get("candidate_scalars") or {}
            per_scalar_metrics = scaling.get("per_scalar_metrics") or {}
            lines.append("| Scalar | Value | Scaled MAE | Scaled RMSE |")
            lines.append("| --- | ---: | ---: | ---: |")
            for key, label in [
                ("ratio_mean", "Ratio mean"),
                ("ratio_median", "Ratio median"),
                ("ratio_p05", "Ratio p05"),
                ("ratio_p95", "Ratio p95"),
                ("least_squares", "Least squares"),
            ]:
                value = candidates.get(key)
                metrics = per_scalar_metrics.get(key) or {}
                lines.append(
                    "| {label} | {value} | {mae} | {rmse} |".format(
                        label=label,
                        value=fmt_value(value),
                        mae=fmt_value(metrics.get("mae")),
                        rmse=fmt_value(metrics.get("rmse")),
                    )
                )
            lines.append("")
        lines.append("### Amplitude Bias")
        amp_metrics = payload.get("comparison", {}).get("amplitude", {})
        lines.append(
            f"* mean={fmt_value((amp_metrics.get('bias_summary') or {}).get('mean'))}, "
            f"median={fmt_value((amp_metrics.get('bias_summary') or {}).get('median'))}, "
            f"p05={fmt_value((amp_metrics.get('bias_summary') or {}).get('p05'))}, "
            f"p95={fmt_value((amp_metrics.get('bias_summary') or {}).get('p95'))}"
        )
        lines.append(
            f"* MAE={fmt_value(amp_metrics.get('mae'))}, "
            f"RMSE={fmt_value(amp_metrics.get('rmse'))}, "
            f"max_abs={fmt_value(amp_metrics.get('max_abs'))}, "
            f"pearson_r={fmt_value(amp_metrics.get('pearson_r'))}"
        )
        lines.append("")
        lines.append("### Phase Bias")
        phase_metrics = payload.get("comparison", {}).get("phase", {})
        lines.append(
            f"* mean={fmt_value((phase_metrics.get('bias_summary') or {}).get('mean'))}, "
            f"median={fmt_value((phase_metrics.get('bias_summary') or {}).get('median'))}, "
            f"p05={fmt_value((phase_metrics.get('bias_summary') or {}).get('p05'))}, "
            f"p95={fmt_value((phase_metrics.get('bias_summary') or {}).get('p95'))}"
        )
        lines.append(
            f"* MAE={fmt_value(phase_metrics.get('mae'))}, "
            f"RMSE={fmt_value(phase_metrics.get('rmse'))}, "
            f"max_abs={fmt_value(phase_metrics.get('max_abs'))}"
        )
        lines.append("")
        normalization = payload.get("normalization") or []
        if normalization:
            lines.append("### Normalization Stage Stats")
            lines.append(
                "| Stage | Source | Count | Mean | Std | Min | Max |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
            for stage in normalization:
                lines.append(
                    "| {name} | {source} | {count} | {mean} | {std} | {min} | {max} |".format(
                        name=stage.get("name") or "n/a",
                        source=stage.get("source") or "n/a",
                        count=fmt_value(stage.get("count"), precision=0),
                        mean=fmt_value(stage.get("mean")),
                        std=fmt_value(stage.get("std")),
                        min=fmt_value(stage.get("min")),
                        max=fmt_value(stage.get("max")),
                    )
                )
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_outputs(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bias_summary.json"
    md_path = output_dir / "bias_summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(payload))
    print(f"Wrote bias summary JSON to {json_path}")
    print(f"Wrote bias summary Markdown to {md_path}")


def main() -> None:
    args = parse_args()
    scenarios: List[ScenarioInput] = []
    for raw_value in args.scenario:
        name, path = parse_scenario_arg(raw_value)
        scenario_input = build_scenario_input(name, path)
        scenarios.append(scenario_input)
    summary = analyze_scenarios(scenarios)
    write_outputs(args.output_dir, summary)


if __name__ == "__main__":
    main()

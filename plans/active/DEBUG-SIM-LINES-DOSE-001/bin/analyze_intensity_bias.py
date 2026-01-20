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


REQUIRED_FILES = {
    "run_metadata": "run_metadata.json",
    "intensity_stats": "intensity_stats.json",
    "inference_stats": "inference_outputs/stats.json",
    "comparison_metrics": "comparison_metrics.json",
    "training_summary": "train_outputs/history_summary.json",
}


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

    return ScenarioInput(
        name=name,
        base_dir=base_dir,
        run_metadata=resolved["run_metadata"],
        intensity_stats=resolved["intensity_stats"],
        inference_stats=resolved["inference_stats"],
        comparison_metrics=resolved["comparison_metrics"],
        training_summary=resolved["training_summary"],
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

    normalization = build_normalization_summary(intensity_stats.get("stages", []))

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
        },
        "intensity": intensity_info,
        "comparison": {
            "amplitude": amplitude_metrics,
            "phase": phase_metrics,
        },
        "normalization": normalization,
        "inference": inference_info,
        "training": training_info,
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
    lines.append(f"- Generated at: {summary.get('generated_at')}")
    lines.append(f"- Scenario count: {summary.get('scenario_count')}")
    lines.append("")

    scenarios = summary.get("scenarios", {})
    if scenarios:
        lines.append(
            "| Scenario | Amp bias mean | Amp median | Phase bias mean | "
            "Phase median | Bundle scale | Legacy scale | Δscale | Training NaN |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        for name, payload in scenarios.items():
            amp_bias = (payload.get("comparison", {})
                        .get("amplitude", {})
                        .get("bias_summary", {}) or {})
            phase_bias = (payload.get("comparison", {})
                          .get("phase", {})
                          .get("bias_summary", {}) or {})
            intensity = payload.get("intensity", {})
            training = payload.get("training", {})
            lines.append(
                "| {name} | {amp_mean} | {amp_med} | {phase_mean} | {phase_med} | "
                "{bundle} | {legacy} | {delta} | {nan_flag} |".format(
                    name=name,
                    amp_mean=fmt_value(amp_bias.get("mean")),
                    amp_med=fmt_value(amp_bias.get("median")),
                    phase_mean=fmt_value(phase_bias.get("mean")),
                    phase_med=fmt_value(phase_bias.get("median")),
                    bundle=fmt_value(intensity.get("bundle")),
                    legacy=fmt_value(intensity.get("legacy_params")),
                    delta=fmt_value(intensity.get("absolute_delta")),
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

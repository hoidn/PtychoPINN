"""Grid-lines metrics table generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


MODEL_LABELS = {
    "pinn": "PINN (CNN)",
    "baseline": "Baseline",
    "pinn_fno_vanilla": "FNO Vanilla",
    "pinn_fno": "FNO",
    "pinn_hybrid": "Hybrid",
    "pinn_stable_hybrid": "Stable Hybrid",
    "pinn_hybrid_resnet": "Hybrid ResNet",
    "pinn_ptychovit": "PtychoViT",
}

MODEL_ORDER = (
    "pinn",
    "baseline",
    "pinn_fno_vanilla",
    "pinn_fno",
    "pinn_hybrid",
    "pinn_stable_hybrid",
    "pinn_hybrid_resnet",
    "pinn_ptychovit",
)

METRICS = (
    ("mae", "MAE"),
    ("mse", "MSE"),
    ("psnr", "PSNR"),
    ("ssim", "SSIM"),
    ("frc50", "FRC50"),
)

LOWER_BETTER = {"mae", "mse"}


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid metric value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported metric value type: {type(value)!r}")


def _extract_pair(model_metrics: dict, metric_key: str) -> Optional[Tuple[float, float]]:
    raw = model_metrics.get(metric_key)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return (_to_float(raw[0]), _to_float(raw[1]))
    except Exception:
        return None


def _ordered_models(metrics: Mapping[str, dict]) -> List[str]:
    present = [model for model in MODEL_ORDER if model in metrics]
    extra = sorted(model for model in metrics.keys() if model not in MODEL_ORDER)
    return [*present, *extra]


def _format_value(metric_key: str, value: float, bold: bool = False) -> str:
    if metric_key == "frc50":
        text = str(int(round(value)))
    else:
        text = f"{value:.6f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def _compute_best(metrics: Mapping[str, dict], models: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    best: Dict[str, Tuple[float, float]] = {}
    for metric_key, _ in METRICS:
        values: List[Tuple[float, float]] = []
        for model in models:
            pair = _extract_pair(metrics.get(model, {}), metric_key)
            if pair is not None:
                values.append(pair)
        if not values:
            continue
        if metric_key in LOWER_BETTER:
            best_amp = min(v[0] for v in values)
            best_phase = min(v[1] for v in values)
        else:
            best_amp = max(v[0] for v in values)
            best_phase = max(v[1] for v in values)
        best[metric_key] = (best_amp, best_phase)
    return best


def _same_value(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12


def _resolve_model_ns(metrics: Mapping[str, dict], model_ns: Optional[Mapping[str, int]]) -> Dict[str, Optional[int]]:
    resolved: Dict[str, Optional[int]] = {}
    for model in metrics.keys():
        if not model_ns or model not in model_ns or model_ns[model] is None:
            resolved[model] = None
            continue
        resolved[model] = int(model_ns[model])
    return resolved


def _group_models_by_n(
    metrics: Mapping[str, dict],
    model_ns: Optional[Mapping[str, int]],
) -> List[Tuple[Optional[int], List[str]]]:
    resolved_ns = _resolve_model_ns(metrics, model_ns)
    grouped: Dict[Optional[int], List[str]] = {}
    for model in _ordered_models(metrics):
        n_value = resolved_ns.get(model)
        grouped.setdefault(n_value, []).append(model)
    int_ns = sorted(n for n in grouped.keys() if n is not None)
    ordered_ns: List[Optional[int]] = [*int_ns]
    if None in grouped:
        ordered_ns.append(None)
    return [(n_value, grouped[n_value]) for n_value in ordered_ns]


def _build_main_table(metrics: Mapping[str, dict], model_ns: Optional[Mapping[str, int]]) -> str:
    grouped_models = _group_models_by_n(metrics, model_ns)
    best_by_n: Dict[Optional[int], Dict[str, Tuple[float, float]]] = {
        n_value: _compute_best(metrics, models)
        for n_value, models in grouped_models
    }

    colspec = "ll" + ("r" * len(METRICS))
    row_end = r" \\"
    lines = []
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header = "N & Model" + "".join([f" & {label} (A/P)" for _, label in METRICS]) + row_end
    lines.append(header)
    lines.append(r"\midrule")

    for group_idx, (n_value, models) in enumerate(grouped_models):
        best = best_by_n.get(n_value, {})
        for model_idx, model in enumerate(models):
            label = MODEL_LABELS.get(model, model)
            n_text = str(n_value) if (model_idx == 0 and n_value is not None) else ""
            if model_idx == 0 and n_value is None:
                n_text = "-"
            row = [n_text, label]
            model_metrics = metrics.get(model, {})
            for metric_key, _ in METRICS:
                pair = _extract_pair(model_metrics, metric_key)
                if pair is None:
                    row.extend(["-", "-"])
                    continue
                amp, phase = pair
                best_pair = best.get(metric_key)
                bold_amp = best_pair is not None and _same_value(amp, best_pair[0])
                bold_phase = best_pair is not None and _same_value(phase, best_pair[1])
                row.append(
                    f"{_format_value(metric_key, amp, bold_amp)} / "
                    f"{_format_value(metric_key, phase, bold_phase)}"
                )
            lines.append(" & ".join(row) + row_end)
        if group_idx < len(grouped_models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _build_best_table(metrics: Mapping[str, dict], model_ns: Optional[Mapping[str, int]]) -> str:
    grouped_models = _group_models_by_n(metrics, model_ns)
    lines = []
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(r"N & Metric & Best Amp & Best Phase \\")
    lines.append(r"\midrule")

    for group_idx, (n_value, models) in enumerate(grouped_models):
        n_text = str(n_value) if n_value is not None else "-"
        emitted_for_group = 0
        for metric_key, metric_label in METRICS:
            best_amp: Optional[Tuple[float, str]] = None
            best_phase: Optional[Tuple[float, str]] = None
            for model in models:
                pair = _extract_pair(metrics.get(model, {}), metric_key)
                if pair is None:
                    continue
                amp, phase = pair
                model_label = MODEL_LABELS.get(model, model)
                if best_amp is None:
                    best_amp = (amp, model_label)
                else:
                    if metric_key in LOWER_BETTER and amp < best_amp[0]:
                        best_amp = (amp, model_label)
                    elif metric_key not in LOWER_BETTER and amp > best_amp[0]:
                        best_amp = (amp, model_label)
                if best_phase is None:
                    best_phase = (phase, model_label)
                else:
                    if metric_key in LOWER_BETTER and phase < best_phase[0]:
                        best_phase = (phase, model_label)
                    elif metric_key not in LOWER_BETTER and phase > best_phase[0]:
                        best_phase = (phase, model_label)

            if best_amp is None or best_phase is None:
                continue
            n_cell = n_text if emitted_for_group == 0 else ""
            amp_text = f"{_format_value(metric_key, best_amp[0])} ({best_amp[1]})"
            phase_text = f"{_format_value(metric_key, best_phase[0])} ({best_phase[1]})"
            lines.append(f"{n_cell} & {metric_label} & {amp_text} & {phase_text} \\\\")
            emitted_for_group += 1
        if group_idx < len(grouped_models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def write_metrics_tables(
    output_dir: Path,
    metrics: Dict[str, dict],
    model_ns: Optional[Mapping[str, int]] = None,
) -> Dict[str, str]:
    """Write metrics comparison tables as LaTeX artifacts.

    Returns:
        Mapping with `metrics_table_tex` and `metrics_table_best_tex` paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_tex = output_dir / "metrics_table.tex"
    best_tex = output_dir / "metrics_table_best.tex"
    main_tex.write_text(_build_main_table(metrics, model_ns))
    best_tex.write_text(_build_best_table(metrics, model_ns))
    return {
        "metrics_table_tex": str(main_tex),
        "metrics_table_best_tex": str(best_tex),
    }

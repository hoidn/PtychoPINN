"""Grid-lines metrics table generation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


MODEL_LABELS = {
    "pinn": "PtychoPINN (CNN)",
    "baseline": "Baseline",
    "pinn_ffno": "FFNO",
    "pinn_fno_vanilla": "FNO Vanilla",
    "pinn_fno": "FNO",
    "pinn_hybrid": "Hybrid",
    "pinn_stable_hybrid": "Stable Hybrid",
    "pinn_hybrid_resnet": "Hybrid ResNet",
    "pinn_spectral_resnet_bottleneck_net": "Spectral ResNet Bottleneck",
    "pinn_ptychovit": "PtychoViT",
}

MODEL_ORDER = (
    "pinn",
    "baseline",
    "pinn_ffno",
    "pinn_fno_vanilla",
    "pinn_fno",
    "pinn_hybrid",
    "pinn_stable_hybrid",
    "pinn_hybrid_resnet",
    "pinn_spectral_resnet_bottleneck_net",
    "pinn_ptychovit",
)

METRICS = (
    ("mae", "MAE"),
    ("mse", "MSE"),
    ("psnr", "PSNR"),
    ("ssim", "SSIM"),
    ("ms_ssim", "MS-SSIM"),
    ("frc50", "FRC50"),
)

LOWER_BETTER = {"mae", "mse"}
PAPER_BENCHMARK_STATUS_VALUES = ("paper_complete", "benchmark_incomplete")
PAPER_REQUIRED_FIELDS = (
    "model_label",
    "parameter_count",
    "epoch_budget",
    "final_completed_epoch",
    "final_train_loss",
    "validation_loss",
    "runtime_hardware_summary",
    "caveats",
)


def _latex_escape(text: str) -> str:
    escaped = (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )
    return escaped


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
        text = f"{value:.2f}"
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
            label = _latex_escape(label)
            n_text = str(n_value) if (model_idx == 0 and n_value is not None) else ""
            if model_idx == 0 and n_value is None:
                n_text = "-"
            row = [n_text, label]
            model_metrics = metrics.get(model, {})
            for metric_key, _ in METRICS:
                pair = _extract_pair(model_metrics, metric_key)
                if pair is None:
                    row.append("-")
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
                model_label = _latex_escape(MODEL_LABELS.get(model, model))
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


def _write_csv_table(
    output_path: Path,
    metrics: Mapping[str, dict],
    model_ns: Optional[Mapping[str, int]],
) -> Path:
    grouped_models = _group_models_by_n(metrics, model_ns)
    fieldnames = [
        "model_id",
        "model_label",
        "N",
    ]
    for metric_key, _metric_label in METRICS:
        fieldnames.extend([f"{metric_key}_amp", f"{metric_key}_phase"])

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for n_value, models in grouped_models:
            for model in models:
                row = {
                    "model_id": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "N": "" if n_value is None else int(n_value),
                }
                model_metrics = metrics.get(model, {})
                for metric_key, _metric_label in METRICS:
                    pair = _extract_pair(model_metrics, metric_key)
                    if pair is None:
                        row[f"{metric_key}_amp"] = ""
                        row[f"{metric_key}_phase"] = ""
                        continue
                    row[f"{metric_key}_amp"] = pair[0]
                    row[f"{metric_key}_phase"] = pair[1]
                writer.writerow(row)
    return output_path


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
    csv_path = output_dir / "metrics_table.csv"
    main_tex.write_text(_build_main_table(metrics, model_ns))
    best_tex.write_text(_build_best_table(metrics, model_ns))
    _write_csv_table(csv_path, metrics, model_ns)
    return {
        "metrics_table_tex": str(main_tex),
        "metrics_table_best_tex": str(best_tex),
        "metrics_table_csv": str(csv_path),
    }


def _validate_validation_loss(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        return False
    if status == "no_validation_series":
        return "value" in payload
    value = payload.get("value")
    return value is None or isinstance(value, (int, float))


def _missing_paper_fields(row_payload: Mapping[str, object]) -> List[str]:
    missing: List[str] = []
    for field in PAPER_REQUIRED_FIELDS:
        value = row_payload.get(field)
        if field == "validation_loss":
            if not _validate_validation_loss(value):
                missing.append(field)
            continue
        if field == "caveats":
            if not isinstance(value, list):
                missing.append(field)
            continue
        if value is None:
            missing.append(field)
            continue
    metrics = row_payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return [*missing, *(f"metrics.{metric_key}" for metric_key, _ in METRICS)]
    for metric_key, _metric_label in METRICS:
        if _extract_pair(metrics, metric_key) is None:
            missing.append(f"metrics.{metric_key}")
    return missing


def _build_metric_schema() -> Dict[str, object]:
    metric_fields = [
        {
            "name": metric_key,
            "label": metric_label,
            "pair_axes": ["amplitude", "phase"],
            "required": True,
        }
        for metric_key, metric_label in METRICS
    ]
    return {
        "status_values": list(PAPER_BENCHMARK_STATUS_VALUES),
        "required_fields": list(PAPER_REQUIRED_FIELDS),
        "metric_fields": metric_fields,
        "downgrade_rule": "Any missing required field or required metric downgrades the merged result to benchmark_incomplete.",
    }


def write_paper_benchmark_bundle(
    *,
    output_dir: Path,
    row_payloads: Mapping[str, Mapping[str, object]],
    required_rows: Iterable[str],
    fixed_sample_ids: Iterable[int],
    shared_visual_scales: Mapping[str, object],
    selected_fno_comparator: Optional[str] = None,
    row_statuses: Optional[Mapping[str, Mapping[str, object]]] = None,
    evidence_scope: str = "readiness_only_not_benchmark_performance",
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_rows = tuple(required_rows)
    missing_fields_by_row: Dict[str, List[str]] = {}
    model_ns: Dict[str, int] = {}
    metrics_only: Dict[str, dict] = {}

    incomplete = False
    for model_id, payload in row_payloads.items():
        missing = _missing_paper_fields(payload)
        missing_fields_by_row[model_id] = missing
        if missing:
            incomplete = True
        metrics_only[model_id] = dict(payload.get("metrics", {}))
        model_ns[model_id] = int(payload.get("N", 128))

    for model_id in required_rows:
        if model_id not in row_payloads:
            incomplete = True
            missing_fields_by_row[model_id] = ["row_missing"]

    benchmark_status = "benchmark_incomplete" if incomplete else "paper_complete"
    bundle_payload = {
        "benchmark_status": benchmark_status,
        "required_rows": list(required_rows),
        "selected_fno_comparator": selected_fno_comparator,
        "evidence_scope": evidence_scope,
        "missing_fields_by_row": missing_fields_by_row,
        "row_statuses": dict(row_statuses or {}),
        "visual_collation": {
            "fixed_sample_ids": list(fixed_sample_ids),
            "shared_visual_scales": dict(shared_visual_scales),
        },
        "rows": row_payloads,
    }

    metrics_json = output_dir / "metrics.json"
    metric_schema_json = output_dir / "metric_schema.json"
    metrics_json.write_text(json.dumps(bundle_payload, indent=2), encoding="utf-8")
    metric_schema_json.write_text(json.dumps(_build_metric_schema(), indent=2), encoding="utf-8")
    table_paths = write_metrics_tables(output_dir, metrics_only, model_ns=model_ns)
    return {
        "metrics_json": str(metrics_json),
        "metric_schema_json": str(metric_schema_json),
        **table_paths,
    }

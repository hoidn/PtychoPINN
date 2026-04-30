"""Grid-lines metrics table generation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


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
PAPER_ROW_STATUS_VALUES = (
    "paper_grade",
    "decision_support",
    "blocked",
    "not_protocol_compatible",
)
PAPER_REQUIRED_FIELDS = (
    "model_label",
    "architecture_id",
    "training_procedure",
    "parameter_count",
    "epoch_budget",
    "final_completed_epoch",
    "final_train_loss",
    "validation_loss",
    "runtime_summary",
    "hardware_summary",
    "row_status",
    "caveats",
)
PAPER_PROVENANCE_FIELDS = (
    "invocation",
    "config",
    "git",
    "environment",
    "dataset",
    "splits",
    "randomness",
    "outputs",
    "visuals",
)
PAPER_REQUIRED_FIELD_DEFINITIONS = (
    {"name": "model_key", "units": None, "nullable": False, "source": "row_map_key"},
    {"name": "model_label", "units": None, "nullable": False},
    {"name": "architecture_id", "units": None, "nullable": False},
    {"name": "training_procedure", "units": None, "nullable": False},
    {"name": "parameter_count", "units": "parameters", "nullable": False},
    {"name": "epoch_budget", "units": "epochs", "nullable": False},
    {"name": "final_completed_epoch", "units": "epochs", "nullable": False},
    {"name": "final_train_loss", "units": "training_loss_units", "nullable": False},
    {"name": "validation_loss", "units": "training_loss_units", "nullable": True},
    {"name": "runtime_summary", "units": None, "nullable": False},
    {"name": "hardware_summary", "units": None, "nullable": False},
    {"name": "row_status", "units": None, "nullable": False},
    {"name": "caveats", "units": None, "nullable": False},
    {"name": "invocation", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "config", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "git", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "environment", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "dataset", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "splits", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "randomness", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "outputs", "units": None, "nullable": False, "paper_grade_only": True},
    {"name": "visuals", "units": None, "nullable": False, "paper_grade_only": True},
)
PAPER_METRIC_FIELD_DEFINITIONS = (
    {
        "name": "mae",
        "label": "MAE",
        "pair_axes": ["amplitude", "phase"],
        "units": {"amplitude": "normalized_amplitude", "phase": "radians"},
        "nullable": False,
        "required": True,
    },
    {
        "name": "mse",
        "label": "MSE",
        "pair_axes": ["amplitude", "phase"],
        "units": {
            "amplitude": "normalized_amplitude_squared",
            "phase": "radians_squared",
        },
        "nullable": False,
        "required": True,
    },
    {
        "name": "psnr",
        "label": "PSNR",
        "pair_axes": ["amplitude", "phase"],
        "units": {"amplitude": "dB", "phase": "dB"},
        "nullable": False,
        "required": True,
    },
    {
        "name": "ssim",
        "label": "SSIM",
        "pair_axes": ["amplitude", "phase"],
        "units": {"amplitude": "unitless", "phase": "unitless"},
        "nullable": False,
        "required": True,
    },
    {
        "name": "ms_ssim",
        "label": "MS-SSIM",
        "pair_axes": ["amplitude", "phase"],
        "units": {"amplitude": "unitless", "phase": "unitless"},
        "nullable": False,
        "required": True,
    },
    {
        "name": "frc50",
        "label": "FRC50",
        "pair_axes": ["amplitude", "phase"],
        "units": {"amplitude": "frequency_bin", "phase": "frequency_bin"},
        "nullable": False,
        "required": True,
    },
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


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid metric value")
    if isinstance(value, np.integer):
        return float(int(value))
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported metric value type: {type(value)!r}")


def _extract_pair(model_metrics: dict, metric_key: str) -> Optional[Tuple[float, float]]:
    raw = model_metrics.get(metric_key)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
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
    if status in {"no_validation_series", "not_emitted"}:
        return "value" in payload
    value = payload.get("value")
    return value is None or isinstance(value, (int, float))


def _validate_string_fields(payload: object, required_keys: Tuple[str, ...]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for key in required_keys:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _validate_git_payload(payload: object) -> bool:
    return _validate_string_fields(payload, ("commit",))


def _validate_environment_payload(payload: object) -> bool:
    return _validate_string_fields(payload, ("python_executable",))


def _validate_dataset_payload(payload: object) -> bool:
    return _validate_string_fields(payload, ("train_npz", "test_npz"))


def _validate_splits_payload(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return payload.get("nimgs_train") is not None and payload.get("nimgs_test") is not None


def _validate_randomness_payload(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return any(payload.get(key) is not None for key in ("requested_seed", "seed", "seed_policy"))


def _validate_outputs_payload(payload: object) -> bool:
    return _validate_string_fields(payload, ("metrics_json", "history_json", "recon_npz"))


def _validate_visuals_payload(payload: object) -> bool:
    return _validate_string_fields(payload, ("amp_phase_png", "amp_phase_error_png"))


def _missing_paper_fields(
    row_payload: Mapping[str, object],
    *,
    require_row_provenance: bool = False,
) -> List[str]:
    missing: List[str] = []
    for field in PAPER_REQUIRED_FIELDS:
        value = row_payload.get(field)
        if field == "validation_loss":
            if not _validate_validation_loss(value):
                missing.append(field)
            continue
        if field in {"architecture_id", "training_procedure"}:
            if not isinstance(value, str) or not value.strip():
                missing.append(field)
            continue
        if field == "row_status":
            if not isinstance(value, str) or value not in PAPER_ROW_STATUS_VALUES:
                missing.append(field)
            continue
        if field in {"runtime_summary", "hardware_summary"}:
            if not isinstance(value, Mapping):
                missing.append(field)
            continue
        if field == "caveats":
            if not isinstance(value, list):
                missing.append(field)
            continue
        if value is None:
            missing.append(field)
            continue
    if require_row_provenance:
        provenance_validators = {
            "invocation": lambda value: _validate_string_fields(value, ("json", "shell")),
            "config": lambda value: _validate_string_fields(value, ("json",)),
            "git": _validate_git_payload,
            "environment": _validate_environment_payload,
            "dataset": _validate_dataset_payload,
            "splits": _validate_splits_payload,
            "randomness": _validate_randomness_payload,
            "outputs": _validate_outputs_payload,
            "visuals": _validate_visuals_payload,
        }
        for field in PAPER_PROVENANCE_FIELDS:
            validator = provenance_validators[field]
            if not validator(row_payload.get(field)):
                missing.append(field)
    metrics = row_payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return [*missing, *(f"metrics.{metric_key}" for metric_key, _ in METRICS)]
    for metric_key, _metric_label in METRICS:
        if _extract_pair(metrics, metric_key) is None:
            missing.append(f"metrics.{metric_key}")
    return missing


def _build_metric_schema() -> Dict[str, object]:
    return {
        "status_values": list(PAPER_BENCHMARK_STATUS_VALUES),
        "row_status_values": list(PAPER_ROW_STATUS_VALUES),
        "required_fields": list(PAPER_REQUIRED_FIELDS),
        "paper_grade_provenance_fields": list(PAPER_PROVENANCE_FIELDS),
        "field_definitions": list(PAPER_REQUIRED_FIELD_DEFINITIONS),
        "metric_fields": list(PAPER_METRIC_FIELD_DEFINITIONS),
        "downgrade_rule": "Any missing required field or required metric downgrades the merged result to benchmark_incomplete.",
    }


def _build_model_manifest(
    *,
    row_payloads: Mapping[str, Mapping[str, object]],
    missing_fields_by_row: Mapping[str, List[str]],
    benchmark_status: str,
    claim_boundary: str,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for model_id in row_payloads.keys():
        payload = row_payloads[model_id]
        rows.append(
            {
                "model_id": model_id,
                "model_label": payload.get("model_label"),
                "architecture_id": payload.get("architecture_id"),
                "training_procedure": payload.get("training_procedure"),
                "N": payload.get("N"),
                "parameter_count": payload.get("parameter_count"),
                "epoch_budget": payload.get("epoch_budget"),
                "final_completed_epoch": payload.get("final_completed_epoch"),
                "final_train_loss": payload.get("final_train_loss"),
                "validation_loss": payload.get("validation_loss"),
                "runtime_summary": payload.get("runtime_summary"),
                "hardware_summary": payload.get("hardware_summary"),
                "row_status": payload.get("row_status"),
                "missing_fields": list(missing_fields_by_row.get(model_id, [])),
            }
        )
    return {
        "benchmark_status": benchmark_status,
        "claim_boundary": claim_boundary,
        "rows": rows,
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
    claim_boundary: str = "minimum_draftable_cdi_subset",
    require_row_provenance: bool = False,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_rows = tuple(required_rows)
    row_statuses = dict(row_statuses or {})
    has_row_statuses = bool(row_statuses)
    missing_fields_by_row: Dict[str, List[str]] = {}
    model_ns: Dict[str, int] = {}
    metrics_only: Dict[str, dict] = {}

    incomplete = False
    for model_id, payload in row_payloads.items():
        missing = _missing_paper_fields(payload, require_row_provenance=require_row_provenance)
        missing_fields_by_row[model_id] = missing
        if missing:
            incomplete = True
        metrics_only[model_id] = dict(payload.get("metrics", {}))
        model_ns[model_id] = int(payload.get("N", 128))

    for model_id in required_rows:
        if model_id not in row_payloads:
            incomplete = True
            missing_fields_by_row[model_id] = ["row_missing"]
            continue
        row_status = row_payloads[model_id].get("row_status")
        if row_status != "paper_grade":
            incomplete = True
            row_missing = missing_fields_by_row.setdefault(model_id, [])
            if "row_status" not in row_missing:
                row_missing.append("row_status")

        if has_row_statuses:
            status = None
            status_payload = row_statuses.get(model_id)
            if isinstance(status_payload, Mapping):
                status = status_payload.get("status")
            if status != "supported_for_harness":
                incomplete = True

    benchmark_status = "benchmark_incomplete" if incomplete else "paper_complete"
    bundle_payload = {
        "benchmark_status": benchmark_status,
        "claim_boundary": claim_boundary,
        "required_rows": list(required_rows),
        "selected_fno_comparator": selected_fno_comparator,
        "evidence_scope": evidence_scope,
        "missing_fields_by_row": missing_fields_by_row,
        "row_statuses": row_statuses,
        "visual_collation": {
            "fixed_sample_ids": list(fixed_sample_ids),
            "shared_visual_scales": dict(shared_visual_scales),
        },
        "rows": row_payloads,
    }

    metrics_json = output_dir / "metrics.json"
    metric_schema_json = output_dir / "metric_schema.json"
    model_manifest_json = output_dir / "model_manifest.json"
    metrics_json.write_text(json.dumps(bundle_payload, indent=2, default=_json_default), encoding="utf-8")
    metric_schema_json.write_text(json.dumps(_build_metric_schema(), indent=2), encoding="utf-8")
    model_manifest_json.write_text(
        json.dumps(
            _build_model_manifest(
                row_payloads=row_payloads,
                missing_fields_by_row=missing_fields_by_row,
                benchmark_status=benchmark_status,
                claim_boundary=claim_boundary,
            ),
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    table_paths = write_metrics_tables(output_dir, metrics_only, model_ns=model_ns)
    return {
        "metrics_json": str(metrics_json),
        "metric_schema_json": str(metric_schema_json),
        "model_manifest_json": str(model_manifest_json),
        **table_paths,
    }

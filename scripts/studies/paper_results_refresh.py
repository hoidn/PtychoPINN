"""Generate paper-local result tables from completed NeurIPS artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from math import sqrt
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
NEURIPS_DIR = REPO_ROOT / "docs" / "plans" / "NEURIPS-HYBRID-RESNET-2026"
TABLES_DIR = NEURIPS_DIR / "tables"
FIGURES_DIR = NEURIPS_DIR / "figures"

BRDT_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-brdt-four-row-preflight"
)
BRDT_METRICS_JSON = BRDT_ROOT / "metrics.json"
BRDT_SOURCE_FIGURE = BRDT_ROOT / "visuals" / "brdt_compare_q.png"

CDI_UNO_METRICS_JSON = (
    REPO_ROOT
    / ".artifacts"
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-30-cdi-lines128-uno-table-extension"
    / "runs"
    / "complete_table_plus_uno_20260504T100347Z"
    / "metrics.json"
)
CDI_SUPERVISED_FFNO_METRICS_JSON = (
    REPO_ROOT
    / ".artifacts"
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-cdi-lines128-supervised-equivalent-rows"
    / "runs"
    / "supervised_ffno_extension_20260430T180217Z"
    / "runs"
    / "supervised_ffno"
    / "metrics.json"
)

REQUIRED_CNS_HISTORY5_ROWS = [
    "author_ffno_cns_base",
    "spectral_resnet_bottleneck_base",
    "fno_base",
    "unet_strong",
]

KNOWN_CNS_HISTORY5_ROWS = {
    "author_ffno_cns_base": {"history_len": 5, "epochs": 40},
    "spectral_resnet_bottleneck_base": {"history_len": 5, "epochs": 40},
    "fno_base": {"history_len": 5, "epochs": 40},
    "unet_strong": {"history_len": 5, "epochs": 40},
}

BRDT_LABELS = {
    "classical_born_backprop": "Classical Born backprop",
    "unet": "U-Net",
    "fno_vanilla": "FNO",
    "hybrid_resnet": "Hybrid ResNet",
}

CDI_ROW_ORDER = [
    "baseline",
    "pinn",
    "pinn_fno_vanilla",
    "pinn_ffno",
    "supervised_ffno",
    "pinn_hybrid_resnet",
    "pinn_spectral_resnet_bottleneck_net",
    "pinn_neuralop_uno",
    "supervised_neuralop_uno",
]

CDI_LABELS = {
    "baseline": ("CNN", "supervised"),
    "pinn": ("CNN", "PINN"),
    "pinn_fno_vanilla": ("FNO", "PINN"),
    "pinn_ffno": ("FFNO", "PINN"),
    "supervised_ffno": ("FFNO", "supervised"),
    "pinn_hybrid_resnet": ("SRU-Net", "PINN"),
    "pinn_spectral_resnet_bottleneck_net": ("Spectral SRU-Net", "PINN"),
    "pinn_neuralop_uno": ("U-NO", "PINN"),
    "supervised_neuralop_uno": ("U-NO", "supervised"),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def detect_cns_history5_gaps(
    available_rows: Mapping[str, Mapping[str, object]],
    *,
    required_rows: Sequence[str],
    history_len: int = 5,
    epochs: int = 40,
) -> list[str]:
    gaps: list[str] = []
    for row_id in required_rows:
        row = dict(available_rows.get(row_id, {}))
        if int(row.get("history_len", -1)) != history_len or int(row.get("epochs", -1)) != epochs:
            gaps.append(row_id)
    return gaps


def audit_cns_history5_availability() -> dict[str, object]:
    gaps = detect_cns_history5_gaps(
        KNOWN_CNS_HISTORY5_ROWS,
        required_rows=REQUIRED_CNS_HISTORY5_ROWS,
    )
    return {
        "required_rows": REQUIRED_CNS_HISTORY5_ROWS,
        "available_rows": KNOWN_CNS_HISTORY5_ROWS,
        "missing_rows": gaps,
    }


def _brdt_value(row: Mapping[str, object], section: str, key: str) -> float:
    section_payload = row.get(section, {})
    if not isinstance(section_payload, Mapping):
        raise KeyError(f"Expected mapping at {section}")
    return float(section_payload[key])


def normalized_brdt_rows(metrics_payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_row in metrics_payload["rows"]:  # type: ignore[index]
        row = dict(raw_row)
        row_id = str(row["row_id"])
        status = str(row.get("row_status", row.get("status", "")))
        label = str(row.get("paper_label", BRDT_LABELS.get(row_id, row_id.replace("_", " "))))
        display = {
            "row_id": row_id,
            "label": label,
            "status": status,
        }
        if status != "blocked":
            display.update(
                {
                    "image_relative_l2_phys": _brdt_value(row, "image", "image_relative_l2_phys"),
                    "meas_relative_l2": _brdt_value(row, "measurement", "meas_relative_l2"),
                    "psnr_phys": _brdt_value(row, "supporting", "psnr_phys"),
                    "ssim_phys": _brdt_value(row, "supporting", "ssim_phys"),
                }
            )
        rows.append(display)
    return rows


def render_brdt_metrics_table(metrics_payload: Mapping[str, object]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Row & Image rel. $L_2$ $\downarrow$ & Meas. rel. $L_2$ $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & Status \\",
        r"\midrule",
    ]
    for row in normalized_brdt_rows(metrics_payload):
        label = _latex_escape(row["label"])
        status = str(row["status"])
        if status == "blocked":
            lines.append(f"{label} & -- & -- & -- & -- & blocked \\\\")
            continue
        lines.append(
            f"{label} & {float(row['image_relative_l2_phys']):.3f} & "
            f"{float(row['meas_relative_l2']):.3f} & "
            f"{float(row['psnr_phys']):.2f} & "
            f"{float(row['ssim_phys']):.3f} & {status} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_brdt_assets() -> dict[str, str]:
    payload = _read_json(BRDT_METRICS_JSON)
    rows = normalized_brdt_rows(payload)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLES_DIR / "brdt_decision_support_metrics.tex"
    csv_path = TABLES_DIR / "brdt_decision_support_metrics.csv"
    json_path = TABLES_DIR / "brdt_decision_support_metrics.json"
    fig_path = FIGURES_DIR / "brdt_decision_support_recon.png"

    tex_path.write_text(render_brdt_metrics_table(payload), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "label",
                "status",
                "image_relative_l2_phys",
                "meas_relative_l2",
                "psnr_phys",
                "ssim_phys",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        json_path,
        {
            "claim_boundary": "decision_support_preflight_only",
            "source_metrics_json": str(BRDT_METRICS_JSON.relative_to(REPO_ROOT)),
            "source_figure": str(BRDT_SOURCE_FIGURE.relative_to(REPO_ROOT)),
            "paper_figure": str(fig_path.relative_to(NEURIPS_DIR)),
            "rows": rows,
        },
    )
    shutil.copyfile(BRDT_SOURCE_FIGURE, fig_path)
    return {
        "tex": str(tex_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "figure": str(fig_path),
    }


def _pair(payload: Mapping[str, object], key: str) -> tuple[float, float]:
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise KeyError("metrics")
    raw = metrics[key]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"Metric {key!r} must be a two-value sequence")
    return float(raw[0]), float(raw[1])


def cdi_display_metrics(metrics_payload: Mapping[str, object]) -> list[dict[str, object]]:
    source_rows = metrics_payload.get("rows", metrics_payload)
    if not isinstance(source_rows, Mapping):
        raise ValueError("CDI metrics payload must be a row mapping or contain a 'rows' mapping")

    rows: list[dict[str, object]] = []
    for row_id in CDI_ROW_ORDER:
        if row_id not in source_rows:
            continue
        payload = dict(source_rows[row_id])
        amp_mae, phase_mae = _pair(payload, "mae")
        amp_mse, phase_mse = _pair(payload, "mse")
        amp_ssim, phase_ssim = _pair(payload, "ssim")
        model, training = CDI_LABELS[row_id]
        rows.append(
            {
                "row_id": row_id,
                "model": model,
                "training": training,
                "amp_mae": amp_mae,
                "phase_mae": phase_mae,
                "amp_mse": amp_mse,
                "phase_mse": phase_mse,
                "amp_rmse": sqrt(amp_mse),
                "phase_rmse": sqrt(phase_mse),
                "amp_ssim": amp_ssim,
                "phase_ssim": phase_ssim,
            }
        )
    return rows


def render_cdi_metrics_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Training & Amp MAE & Phase MAE & Amp MSE & Phase MSE & Amp RMSE & Phase RMSE & Amp SSIM & Phase SSIM \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['model'])} & {_latex_escape(row['training'])} & "
            f"{float(row['amp_mae']):.4f} & {float(row['phase_mae']):.4f} & "
            f"{float(row['amp_mse']):.4f} & {float(row['phase_mse']):.4f} & "
            f"{float(row['amp_rmse']):.4f} & {float(row['phase_rmse']):.4f} & "
            f"{float(row['amp_ssim']):.4f} & {float(row['phase_ssim']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_cdi_extended_assets() -> dict[str, str]:
    payload = _read_json(CDI_UNO_METRICS_JSON)
    source_rows = dict(payload["rows"])
    source_rows["supervised_ffno"] = {"metrics": _read_json(CDI_SUPERVISED_FFNO_METRICS_JSON)}
    payload = {**payload, "rows": source_rows}
    rows = cdi_display_metrics(payload)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLES_DIR / "cdi_lines128_metrics_extended.tex"
    csv_path = TABLES_DIR / "cdi_lines128_metrics_extended.csv"
    json_path = TABLES_DIR / "cdi_lines128_metrics_extended.json"

    tex_path.write_text(render_cdi_metrics_table(rows), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        json_path,
        {
            "claim_boundary": "complete_lines128_cdi_benchmark_plus_uno_extension",
            "source_metrics_json": str(CDI_UNO_METRICS_JSON.relative_to(REPO_ROOT)),
            "supervised_ffno_source_metrics_json": str(
                CDI_SUPERVISED_FFNO_METRICS_JSON.relative_to(REPO_ROOT)
            ),
            "rows": rows,
        },
    )
    return {"tex": str(tex_path), "csv": str(csv_path), "json": str(json_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-cns-history5", action="store_true")
    parser.add_argument("--write-brdt-assets", action="store_true")
    parser.add_argument("--write-cdi-extended-assets", action="store_true")
    args = parser.parse_args(argv)

    outputs: dict[str, object] = {}
    if args.audit_cns_history5:
        outputs["cns_history5_audit"] = audit_cns_history5_availability()
    if args.write_brdt_assets:
        outputs["brdt_assets"] = write_brdt_assets()
    if args.write_cdi_extended_assets:
        outputs["cdi_extended_assets"] = write_cdi_extended_assets()
    if outputs:
        print(json.dumps(outputs, indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append-only BRDT SRU-Net coordgrid diagnostic runner.

Runs exactly one fresh BRDT sinogram-input SRU-Net row with deterministic
object-grid ``x/y`` coordinate channels appended after bilinear resize, then
assembles an append-only bundle by loading the completed unconditioned
sinogram-input authority rows strictly by lineage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from scripts.studies.born_rytov_dt import convergence as conv_mod
from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import preflight_visuals as visuals_mod
from scripts.studies.born_rytov_dt import reporting as reporting_mod
from scripts.studies.born_rytov_dt import run_brdt_40ep_paper_evidence as paper_mod
from scripts.studies.born_rytov_dt import run_preflight as preflight_mod
from scripts.studies.born_rytov_dt.data import DatasetAuthority, load_dataset_authority
from scripts.studies.born_rytov_dt.run_config import (
    LossWeights,
    RowConfig,
    sinogram_coordgrid_row,
)
from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_srunet_coordgrid_extension.py"
BACKLOG_ITEM = "2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation"
CLAIM_BOUNDARY = "decision_support_append_only_coordgrid_diagnostic"
EXPECTED_EPOCHS = 40
ROW_ID = "sru_net_coordgrid"
BASELINE_ROW_IDS = ("classical_born_backprop", "ffno", "sru_net")
EXPECTED_INPUT_CONTRACT = {
    "input_mode": "sinogram",
    "in_channels": 4,
    "coordinate_channels": "object_xy",
}
DEFAULT_MANIFEST = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-brdt-four-row-preflight/decision_support_dataset/"
    "dataset_manifest.json"
)
DEFAULT_BASELINE_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-05-07-brdt-sinogram-input-40ep-paper-evidence"
)
DEFAULT_OUTPUT_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog"
) / BACKLOG_ITEM


class BaselineLineageError(ValueError):
    """Raised when the immutable lineage bundle is missing required surfaces."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _default_contract(
    *,
    epochs: int = EXPECTED_EPOCHS,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
) -> preflight_mod.TrainingContract:
    return preflight_mod.TrainingContract(
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        optimizer="Adam",
        loss_weights=LossWeights(),
        seed=42,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )


def _resolve_log_path(output_root: Path) -> Optional[Path]:
    top_level_log = output_root / "run.log"
    if top_level_log.exists():
        return top_level_log
    return paper_mod._resolve_log_path(output_root)


def _validate_baseline_bundle(
    baseline_root: Path,
    *,
    required_sample_id: int,
) -> Dict[str, Any]:
    root = Path(baseline_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"missing baseline lineage root: {root}")
    manifest_path = root / "preflight_manifest.json"
    combined_metrics_path = root / "combined_metrics.json"
    required = [manifest_path, combined_metrics_path]
    for row_id in BASELINE_ROW_IDS:
        required.extend(
            [
                root / "rows" / row_id / "row_summary.json",
                root / "rows" / row_id / "model_profile.json",
            ]
        )
    sample_paths = [
        root / "figures" / "source_arrays" / f"sample_{required_sample_id:04d}_q_target.npy",
        root / "figures" / "source_arrays" / f"sample_{required_sample_id:04d}_sino_obs.npy",
    ]
    for row_id in BASELINE_ROW_IDS:
        sample_paths.extend(
            [
                root
                / "figures"
                / "source_arrays"
                / f"sample_{required_sample_id:04d}_{row_id}_q_pred.npy",
                root
                / "figures"
                / "source_arrays"
                / f"sample_{required_sample_id:04d}_{row_id}_sino_pred.npy",
            ]
        )
    missing = [str(path) for path in (*required, *sample_paths) if not path.exists()]
    if missing:
        raise BaselineLineageError(
            "missing baseline lineage inputs: " + ", ".join(missing)
        )
    manifest = json.loads(manifest_path.read_text())
    input_contract = dict(manifest.get("input_contract") or {})
    if input_contract.get("input_mode") != "sinogram":
        raise BaselineLineageError(
            "baseline lineage input contract must stay on input_mode='sinogram'"
        )
    if int(input_contract.get("in_channels", 0)) != 2:
        raise BaselineLineageError(
            "baseline lineage input contract must keep in_channels=2"
        )
    combined = json.loads(combined_metrics_path.read_text())
    row_map = {
        str(row.get("row_id")): row
        for row in (combined.get("rows") or [])
    }
    for row_id in BASELINE_ROW_IDS:
        if row_id not in row_map:
            raise BaselineLineageError(
                f"baseline combined_metrics.json missing row_id={row_id!r}"
            )
    return {
        "root": root,
        "manifest": manifest,
        "combined_metrics": combined,
        "rows": row_map,
    }


def _manifest_payload(
    *,
    output_root: Path,
    baseline_root: Path,
    manifest_path: Path,
    row: RowConfig,
    authority: DatasetAuthority,
    contract: preflight_mod.TrainingContract,
    fixed_sample_ids: List[int],
    required_paper_sample: int,
) -> Dict[str, Any]:
    counts = dict((authority.raw_manifest.get("split") or {}).get("counts") or {})
    return {
        "schema_version": "brdt_srunet_coordgrid_extension_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "promotion_status": "completed",
        "output_root": str(output_root),
        "dataset_manifest_path": str(manifest_path),
        "baseline_lineage": {
            "baseline_root": str(Path(baseline_root).resolve()),
            "baseline_backlog_item": "2026-05-07-brdt-sinogram-input-40ep-paper-evidence",
            "baseline_preflight_manifest": str(
                (Path(baseline_root) / "preflight_manifest.json").resolve()
            ),
            "baseline_combined_metrics": str(
                (Path(baseline_root) / "combined_metrics.json").resolve()
            ),
        },
        "dataset": {
            "dataset_id": str(authority.dataset_id),
            "tier": authority.raw_manifest.get("dataset_identity", {}).get("tier"),
            "split_counts": counts,
            "normalization": authority.normalization.as_dict(),
        },
        "input_contract": {
            "input_mode": "sinogram",
            "in_channels": 4,
            "tensor_shape": ["B", 2, "angle_count", "detector_size"],
            "model_input_source": (
                "measured complex sinogram real/imag channels with deterministic "
                "normalized object-grid x/y channels appended after bilinear resize"
            ),
            "sinogram_to_grid": "bilinear_resize",
            "coordinate_channels": "object_xy",
            "baseline_input_mode": "sinogram",
            "baseline_in_channels": 2,
        },
        "training_contract": contract.as_dict(),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "required_paper_sample": int(required_paper_sample),
        "rows": [row.to_dict()],
    }


def _build_row_runtime(
    *,
    device: Any,
    contract: preflight_mod.TrainingContract,
    runtime_meta: Mapping[str, Any],
    train_seconds: float,
    eval_seconds: float,
    eval_samples: int,
) -> Dict[str, Any]:
    eval_samples_per_second = (
        float(eval_samples) / float(eval_seconds) if eval_seconds > 0.0 else None
    )
    payload = metrics_mod.collect_runtime_metadata(
        device=str(device),
        device_name=preflight_mod._device_name(device),
        epochs=int(contract.epochs),
        batch_size=int(contract.batch_size),
        learning_rate=float(contract.learning_rate),
        parameter_count=int(runtime_meta.get("parameter_count", 0)),
        wall_time_train_s=float(train_seconds),
        wall_time_eval_s=float(eval_seconds),
        row_status="completed",
        extras={
            "eval_samples_per_second": eval_samples_per_second,
            "arch_kwargs": dict(runtime_meta.get("arch_kwargs") or {}),
        },
    )
    payload["eval_samples_per_second"] = eval_samples_per_second
    return payload


def _row_summary_payload(
    *,
    row: RowConfig,
    row_dir: Path,
    image_metrics: Mapping[str, float],
    meas_metrics: Mapping[str, float],
    supporting_metrics: Mapping[str, float],
    runtime_block: Mapping[str, Any],
    runtime_meta: Mapping[str, Any],
    output_dynamic_range: Mapping[str, float],
    final_loss_total: float,
    best_train_total_loss: float,
    materially_improving_at_stop: bool,
) -> Dict[str, Any]:
    summary = {
        "row_id": row.row_id,
        "row_status": "completed",
        "paper_label": row.visible_label,
        "architecture": row.model,
        "input_mode": row.input_mode,
        "coordinate_channels": "object_xy",
        "execution_path": "sinogram_input_coordgrid_train_eval",
        "image_metrics": dict(image_metrics),
        "measurement_metrics": dict(meas_metrics),
        "supporting": dict(supporting_metrics),
        "runtime": dict(runtime_block),
        "model_profile_path": str(row_dir / "model_profile.json"),
        "model_state_path": runtime_meta.get("model_state_path"),
        "history_json_path": runtime_meta.get("history_json_path"),
        "history_csv_path": runtime_meta.get("history_csv_path"),
        "history_length": runtime_meta.get("history_length"),
        "scheduler": runtime_meta.get("scheduler"),
        "output_dynamic_range": dict(output_dynamic_range),
        "final_loss_total": float(final_loss_total),
        "best_train_total_loss": float(best_train_total_loss),
        "materially_improving_at_stop": bool(materially_improving_at_stop),
    }
    return summary


def _write_adapter_contract(
    *,
    output_root: Path,
    row: RowConfig,
    authority: DatasetAuthority,
    backend: Any,
    summary: Mapping[str, Any],
) -> Path:
    payload = reporting_mod.build_adapter_contract(
        dataset_id=authority.dataset_id,
        operator_version=row.operator_version,
        rows=[{**row.to_dict(), "row_status": "completed", "summary": dict(summary)}],
        classical_backend={
            "name": backend.name,
            "reason": backend.reason,
            "claim_boundary": backend.claim_boundary,
        },
        loss_contract={
            "training_label": row.training,
            "coordinate_channels": "object_xy",
        },
        extra={
            "input_mode": "sinogram",
            "sinogram_to_grid": "bilinear_resize",
            "coordinate_channels": "object_xy",
            "in_channels": 4,
        },
    )
    path = output_root / "adapter_contract.json"
    reporting_mod.write_json(path, payload)
    return path


def _write_visual_bundle(
    *,
    output_root: Path,
    baseline_root: Path,
    sample_id: int,
    current_sample: Mapping[str, np.ndarray],
) -> Path:
    visuals_dir = output_root / "visuals"
    arrays_dir = output_root / "figures" / "source_arrays"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    source_root = Path(baseline_root) / "figures" / "source_arrays"
    q_target = np.load(source_root / f"sample_{sample_id:04d}_q_target.npy")
    sino_obs = np.load(source_root / f"sample_{sample_id:04d}_sino_obs.npy")
    np.save(arrays_dir / f"sample_{sample_id:04d}_q_target.npy", q_target)
    np.save(arrays_dir / f"sample_{sample_id:04d}_sino_obs.npy", sino_obs)

    preds_by_row: Dict[str, np.ndarray] = {}
    sino_preds_by_row: Dict[str, np.ndarray] = {}
    for row_id in BASELINE_ROW_IDS:
        q_pred = np.load(source_root / f"sample_{sample_id:04d}_{row_id}_q_pred.npy")
        sino_pred = np.load(
            source_root / f"sample_{sample_id:04d}_{row_id}_sino_pred.npy"
        )
        preds_by_row[row_id] = q_pred
        sino_preds_by_row[row_id] = sino_pred
        np.save(arrays_dir / f"sample_{sample_id:04d}_{row_id}_q_pred.npy", q_pred)
        np.save(
            arrays_dir / f"sample_{sample_id:04d}_{row_id}_sino_pred.npy", sino_pred
        )

    preds_by_row[ROW_ID] = np.asarray(current_sample["q_pred"])
    sino_preds_by_row[ROW_ID] = np.asarray(current_sample["sino_pred"])
    np.save(
        arrays_dir / f"sample_{sample_id:04d}_{ROW_ID}_q_pred.npy",
        preds_by_row[ROW_ID],
    )
    np.save(
        arrays_dir / f"sample_{sample_id:04d}_{ROW_ID}_sino_pred.npy",
        sino_preds_by_row[ROW_ID],
    )

    compare_path = visuals_dir / f"sample_{sample_id:04d}_compare_q.png"
    error_path = visuals_dir / f"sample_{sample_id:04d}_error_q.png"
    residual_path = visuals_dir / f"sample_{sample_id:04d}_sinogram_residual.png"
    visuals_mod.render_compare_q(
        preds_by_row=preds_by_row,
        target=q_target,
        out_path=compare_path,
        sample_id=sample_id,
    )
    visuals_mod.render_error_q(
        preds_by_row=preds_by_row,
        target=q_target,
        out_path=error_path,
        sample_id=sample_id,
    )
    visuals_mod.render_sinogram_residual(
        sino_obs=sino_obs,
        sino_preds_by_row=sino_preds_by_row,
        out_path=residual_path,
        sample_id=sample_id,
    )
    manifest = {
        "schema_version": "brdt_coordgrid_visuals_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "required_sample_id": int(sample_id),
        "rows_present": [
            "classical_born_backprop",
            "ffno",
            "sru_net",
            ROW_ID,
        ],
        "baseline_lineage_root": str(Path(baseline_root).resolve()),
        "figures": [
            str(compare_path.relative_to(output_root)),
            str(error_path.relative_to(output_root)),
            str(residual_path.relative_to(output_root)),
        ],
    }
    manifest_path = output_root / "visual_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _row_payload_to_metrics(row_payload: Mapping[str, Any]) -> metrics_mod.RowMetrics:
    return metrics_mod.RowMetrics(
        row_id=str(row_payload.get("row_id")),
        paper_label=str(row_payload.get("paper_label")),
        architecture=str(row_payload.get("architecture")),
        row_status=str(row_payload.get("row_status")),
        image=dict(row_payload.get("image") or {}),
        measurement=dict(row_payload.get("measurement") or {}),
        supporting=dict(row_payload.get("supporting") or {}),
        runtime=dict(row_payload.get("runtime") or {}),
        extra={"source": row_payload.get("source")},
    )


def _build_combined_rows(
    *,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    coordgrid_row: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row_id in BASELINE_ROW_IDS:
        rows.append({**dict(baseline_rows[row_id]), "source": "baseline_lineage"})
    rows.append({**dict(coordgrid_row), "source": "extension"})
    return rows


def _write_combined_surfaces(
    *,
    output_root: Path,
    baseline_root: Path,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    coordgrid_row: Mapping[str, Any],
) -> Dict[str, Path]:
    combined_rows = _build_combined_rows(
        baseline_rows=baseline_rows,
        coordgrid_row=coordgrid_row,
    )
    combined_json_path = output_root / "combined_metrics.json"
    combined_csv_path = output_root / "combined_metrics.csv"
    combined_payload = {
        "schema_version": "brdt_coordgrid_combined_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "baseline_lineage_root": str(Path(baseline_root).resolve()),
        "rows": combined_rows,
    }
    _write_json(combined_json_path, combined_payload)
    metrics_mod.write_metrics_csv(
        combined_csv_path,
        [_row_payload_to_metrics(row) for row in combined_rows],
    )
    combined_manifest = {
        "schema_version": "brdt_coordgrid_combined_manifest_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "baseline_lineage_root": str(Path(baseline_root).resolve()),
        "preflight_manifest_path": str(output_root / "preflight_manifest.json"),
        "combined_metrics_json": str(combined_json_path),
        "combined_metrics_csv": str(combined_csv_path),
        "rows": [
            {
                "row_id": row.get("row_id"),
                "paper_label": row.get("paper_label"),
                "architecture": row.get("architecture"),
                "row_status": row.get("row_status"),
                "source": row.get("source"),
            }
            for row in combined_rows
        ],
    }
    combined_manifest_path = output_root / "combined_manifest.json"
    _write_json(combined_manifest_path, combined_manifest)
    return {
        "combined_metrics_json": combined_json_path,
        "combined_metrics_csv": combined_csv_path,
        "combined_manifest_json": combined_manifest_path,
    }


def _write_comparison_summary(
    *,
    output_root: Path,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    coordgrid_row: Mapping[str, Any],
    materially_improving_at_stop: bool,
    final_loss_total: float,
    best_train_total_loss: float,
) -> Path:
    baseline_sru = dict(baseline_rows["sru_net"])
    ffno = dict(baseline_rows["ffno"])
    classical = dict(baseline_rows["classical_born_backprop"])
    coord_image = dict(coordgrid_row.get("image") or {})
    coord_meas = dict(coordgrid_row.get("measurement") or {})
    coord_support = dict(coordgrid_row.get("supporting") or {})
    base_image = dict(baseline_sru.get("image") or {})
    base_meas = dict(baseline_sru.get("measurement") or {})
    base_support = dict(baseline_sru.get("supporting") or {})
    coord_runtime = dict(coordgrid_row.get("runtime") or {})
    eval_samples_per_second = coord_runtime.get("eval_samples_per_second")
    if eval_samples_per_second is None:
        eval_samples_per_second = (
            dict(coord_runtime.get("extras") or {}).get("eval_samples_per_second")
        )

    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "# BRDT SRU-Net Coordgrid Diagnostic",
        "",
        f"- Backlog item: `{BACKLOG_ITEM}`",
        f"- Claim boundary: `{CLAIM_BOUNDARY}`",
        "- Read this as append-only representational diagnostic evidence only.",
        "- The coordgrid row is not a physically principled inverse operator and does not replace the baseline BRDT authority.",
        "",
        "## Coordgrid vs unconditioned SRU-Net",
        "",
        "| Metric | `sru_net` baseline | `sru_net_coordgrid` | Delta |",
        "|---|---:|---:|---:|",
    ]
    metric_pairs = [
        ("image_relative_l2_phys", "Image relative L2", base_image, coord_image),
        ("image_rmse_phys", "Image RMSE", base_image, coord_image),
        ("image_mae_phys", "Image MAE", base_image, coord_image),
        ("psnr_phys", "PSNR proxy", base_support, coord_support),
        ("ssim_phys", "SSIM phys", base_support, coord_support),
        ("meas_relative_l2", "Measurement relative L2", base_meas, coord_meas),
        ("meas_rmse", "Measurement RMSE", base_meas, coord_meas),
        ("meas_mae", "Measurement MAE", base_meas, coord_meas),
    ]
    for key, label, baseline_bucket, coord_bucket in metric_pairs:
        baseline_value = float(baseline_bucket.get(key, float("nan")))
        coord_value = float(coord_bucket.get(key, float("nan")))
        delta_value = coord_value - baseline_value
        lines.append(
            f"| {label} | {_fmt(baseline_value)} | {_fmt(coord_value)} | {_fmt(delta_value)} |"
        )
    lines.extend(
        [
            "",
            "## Runtime And Convergence",
            "",
            f"- Parameter count: `{coord_runtime.get('parameter_count')}`",
            f"- Eval samples/s: `{_fmt(eval_samples_per_second)}`",
            f"- Final loss: `{_fmt(final_loss_total)}`",
            f"- Best observed loss: `{_fmt(best_train_total_loss)}`",
            f"- `materially_improving_at_stop`: `{materially_improving_at_stop}`",
            "",
            "## Lineage Context",
            "",
            f"- Baseline `ffno` image relative L2: `{_fmt((ffno.get('image') or {}).get('image_relative_l2_phys'))}`",
            f"- Baseline `classical_born_backprop` image relative L2: `{_fmt((classical.get('image') or {}).get('image_relative_l2_phys'))}`",
        ]
    )
    path = output_root / "comparison_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _sample_bundle_ok(output_root: Path, sample_id: int) -> bool:
    root = Path(output_root)
    required = [
        root / "combined_metrics.json",
        root / "combined_metrics.csv",
        root / "comparison_summary.md",
        root / "convergence_audit.json",
        root / "visuals" / f"sample_{sample_id:04d}_compare_q.png",
        root / "visuals" / f"sample_{sample_id:04d}_error_q.png",
        root / "visuals" / f"sample_{sample_id:04d}_sinogram_residual.png",
        root / "figures" / "source_arrays" / f"sample_{sample_id:04d}_{ROW_ID}_q_pred.npy",
    ]
    return all(path.exists() for path in required)


def _run_inner(
    *,
    manifest_path: Path,
    baseline_root: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_choice: str,
    dry_run: bool,
    fixed_sample_ids: List[int],
    required_paper_sample: int,
    parent_argv: List[str],
) -> Dict[str, Any]:
    baseline = _validate_baseline_bundle(
        baseline_root, required_sample_id=int(required_paper_sample)
    )
    authority = load_dataset_authority(manifest_path)
    preflight_mod.assert_decision_support_manifest(authority.raw_manifest)
    operator_pointer = (
        authority.raw_manifest.get("operator", {}).get("validation_artifact")
        or authority.raw_manifest.get("operator", {}).get("validation_report")
        or "unspecified"
    )
    row = sinogram_coordgrid_row(
        dataset_id=str(authority.dataset_id),
        operator_version=str(operator_pointer),
    )
    contract = _default_contract(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    manifest_payload = _manifest_payload(
        output_root=output_root,
        baseline_root=baseline_root,
        manifest_path=manifest_path,
        row=row,
        authority=authority,
        contract=contract,
        fixed_sample_ids=fixed_sample_ids,
        required_paper_sample=required_paper_sample,
    )
    preflight_manifest_path = output_root / "preflight_manifest.json"
    _write_json(preflight_manifest_path, manifest_payload)
    write_invocation_artifacts(
        output_root,
        SCRIPT_PATH,
        parent_argv,
        {
            "manifest_path": str(manifest_path),
            "baseline_root": str(Path(baseline_root).resolve()),
            "output_root": str(output_root),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "device": str(device_choice),
            "dry_run": bool(dry_run),
            "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
            "required_paper_sample": int(required_paper_sample),
            "training_contract": contract.as_dict(),
        },
        extra={
            "backlog_item": BACKLOG_ITEM,
            "claim_boundary": CLAIM_BOUNDARY,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )
    metrics_mod.write_metric_schema(
        output_root / "metric_schema.json",
        claim_boundary=CLAIM_BOUNDARY,
    )
    if dry_run:
        return {
            "dry_run": True,
            "preflight_manifest_path": str(preflight_manifest_path),
            "metric_schema_path": str(output_root / "metric_schema.json"),
        }

    log_path = _resolve_log_path(output_root)
    provenance_paths = paper_mod._write_top_level_provenance(
        output_root=output_root,
        authority=authority,
        fixed_sample_ids=fixed_sample_ids,
        log_path=log_path,
    )
    device = preflight_mod._select_device(device_choice)
    operator = preflight_mod._build_operator(device)
    backend = preflight_mod._born_init_backend()
    row_dir = output_root / "rows" / row.row_id
    row_dir.mkdir(parents=True, exist_ok=True)

    module, runtime_meta, train_seconds = preflight_mod._train_neural_row(
        row=row,
        authority=authority,
        operator=operator,
        backend=backend,
        device=device,
        contract=contract,
        in_channels=2,
        output_dir=row_dir,
    )
    if module is None:
        raise RuntimeError(f"failed to build coordgrid row: {runtime_meta}")

    eval_started = time.perf_counter()
    image_metrics, meas_metrics, sample_arrays, output_dynamic_range = (
        preflight_mod._evaluate_split(
            module=module,
            authority=authority,
            operator=operator,
            backend=backend,
            device=device,
            in_channels=2,
            classical_only=False,
            fixed_sample_ids=fixed_sample_ids,
            out_dir=row_dir,
            input_mode=row.input_mode,
        )
    )
    eval_seconds = time.perf_counter() - eval_started
    preflight_mod._save_fixed_sample_arrays(
        source_arrays_dir=output_root / "figures" / "source_arrays",
        sample_arrays=sample_arrays,
        row_id=row.row_id,
    )

    history_payload = conv_mod.load_history(Path(str(runtime_meta["history_json_path"])))
    history_summary = conv_mod.summarize_history(
        row_id=row.row_id,
        history_payload=history_payload,
    )
    losses = [
        float(epoch.get("train_total_loss", 0.0))
        for epoch in (history_payload.get("epochs") or [])
    ]
    final_loss_total = float(losses[-1]) if losses else float("nan")
    best_train_total_loss = float(min(losses)) if losses else float("nan")
    supporting_metrics = {
        key: value
        for key, value in image_metrics.items()
        if key in metrics_mod.SUPPORTING_METRICS
    }
    runtime_block = _build_row_runtime(
        device=device,
        contract=contract,
        runtime_meta=runtime_meta,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        eval_samples=int((authority.raw_manifest.get("split") or {}).get("counts", {}).get("test", 0)),
    )
    model_profile = {
        "row_id": row.row_id,
        "architecture": row.model,
        "parameter_count": int(runtime_meta.get("parameter_count", 0)),
        "arch_kwargs": dict(runtime_meta.get("arch_kwargs") or {}),
        "input_mode": row.input_mode,
        "coordinate_channels": "object_xy",
        "in_channels": 4,
    }
    _write_json(row_dir / "model_profile.json", model_profile)
    row_summary = _row_summary_payload(
        row=row,
        row_dir=row_dir,
        image_metrics={
            key: value
            for key, value in image_metrics.items()
            if key in metrics_mod.IMAGE_METRICS
        },
        meas_metrics=meas_metrics,
        supporting_metrics=supporting_metrics,
        runtime_block=runtime_block,
        runtime_meta=runtime_meta,
        output_dynamic_range=output_dynamic_range,
        final_loss_total=final_loss_total,
        best_train_total_loss=best_train_total_loss,
        materially_improving_at_stop=bool(
            history_summary.get("materially_improving_at_stop")
        ),
    )
    _write_json(row_dir / "row_summary.json", row_summary)
    adapter_contract_path = _write_adapter_contract(
        output_root=row_dir,
        row=row,
        authority=authority,
        backend=backend,
        summary=row_summary,
    )

    coordgrid_metrics = metrics_mod.RowMetrics(
        row_id=row.row_id,
        paper_label=row.visible_label,
        architecture=row.model,
        row_status="completed",
        image={
            key: value
            for key, value in image_metrics.items()
            if key in metrics_mod.IMAGE_METRICS
        },
        measurement=dict(meas_metrics),
        supporting=dict(supporting_metrics),
        runtime=dict(runtime_block),
        extra={
            "input_mode": row.input_mode,
            "coordinate_channels": "object_xy",
            "final_loss_total": final_loss_total,
            "best_train_total_loss": best_train_total_loss,
        },
    )
    metrics_mod.write_metrics_json(
        output_root / "metrics.json",
        [coordgrid_metrics],
        claim_boundary=CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_csv(output_root / "metrics.csv", [coordgrid_metrics])

    sample_id = int(required_paper_sample)
    if sample_id not in sample_arrays:
        raise RuntimeError(
            f"required_paper_sample={sample_id} missing from coordgrid sample arrays"
        )
    visual_manifest_path = _write_visual_bundle(
        output_root=output_root,
        baseline_root=baseline["root"],
        sample_id=sample_id,
        current_sample=sample_arrays[sample_id],
    )

    coordgrid_row_payload = {
        **coordgrid_metrics.to_dict(),
        "runtime": {
            **dict(coordgrid_metrics.runtime),
            "final_loss_total": final_loss_total,
            "best_train_total_loss": best_train_total_loss,
            "materially_improving_at_stop": bool(
                history_summary.get("materially_improving_at_stop")
            ),
        },
    }
    combined_paths = _write_combined_surfaces(
        output_root=output_root,
        baseline_root=baseline["root"],
        baseline_rows=baseline["rows"],
        coordgrid_row=coordgrid_row_payload,
    )

    current_rows = {
        row.row_id: {
            "row_summary": row_summary,
            "history_summary": history_summary,
        }
    }
    baseline_rows = {
        row.row_id: {
            "image_metrics": dict((baseline["rows"]["sru_net"].get("image") or {})),
            "measurement_metrics": dict(
                (baseline["rows"]["sru_net"].get("measurement") or {})
            ),
            "supporting": dict((baseline["rows"]["sru_net"].get("supporting") or {})),
            "runtime": dict((baseline["rows"]["sru_net"].get("runtime") or {})),
        }
    }
    audit_payload = conv_mod.build_convergence_audit(
        backlog_item=BACKLOG_ITEM,
        baseline_rows=baseline_rows,
        current_rows=current_rows,
    )
    conv_mod.write_convergence_audit_json(
        output_root / "convergence_audit.json",
        audit_payload,
    )
    conv_mod.write_convergence_audit_csv(
        output_root / "convergence_audit.csv",
        audit_payload,
    )
    comparison_summary_path = _write_comparison_summary(
        output_root=output_root,
        baseline_rows=baseline["rows"],
        coordgrid_row=coordgrid_row_payload,
        materially_improving_at_stop=bool(
            history_summary.get("materially_improving_at_stop")
        ),
        final_loss_total=final_loss_total,
        best_train_total_loss=best_train_total_loss,
    )
    log_path = _resolve_log_path(output_root)
    paper_mod._write_run_exit_status(
        output_root,
        pid=os.getpid(),
        exit_code=0,
        status="completed",
        log_path=log_path,
    )

    if not _sample_bundle_ok(output_root, sample_id):
        raise RuntimeError("coordgrid bundle incomplete after assembly")

    return {
        "dry_run": False,
        "preflight_manifest_path": str(preflight_manifest_path),
        "metrics_json_path": str(output_root / "metrics.json"),
        "combined_metrics_json_path": str(combined_paths["combined_metrics_json"]),
        "combined_manifest_json_path": str(combined_paths["combined_manifest_json"]),
        "convergence_audit_json_path": str(output_root / "convergence_audit.json"),
        "visual_manifest_path": str(visual_manifest_path),
        "comparison_summary_path": str(comparison_summary_path),
        "adapter_contract_path": str(adapter_contract_path),
        "source_arrays_dir": str(output_root / "figures" / "source_arrays"),
        **provenance_paths,
    }


def run_srunet_coordgrid_extension(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: int = EXPECTED_EPOCHS,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    device_choice: str = "auto",
    dry_run: bool = False,
    fixed_sample_ids: Optional[List[int]] = None,
    required_paper_sample: int = 255,
    parent_argv: Optional[List[str]] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    baseline_root = Path(baseline_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_sample_ids = [int(i) for i in (fixed_sample_ids or [255])]
    if int(required_paper_sample) not in set(fixed_sample_ids):
        raise ValueError("required_paper_sample must be in fixed_sample_ids")
    parent_argv = list(parent_argv) if parent_argv is not None else list(sys.argv[1:])

    completed = (output_root / "run_exit_status.json").exists() and (
        output_root / "combined_metrics.json"
    ).exists()
    if dry_run:
        dry_run_root = output_root / "dry_run" if completed else output_root
        dry_run_root.mkdir(parents=True, exist_ok=True)
        return _run_inner(
            manifest_path=manifest_path,
            baseline_root=baseline_root,
            output_root=dry_run_root,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device_choice=device_choice,
            dry_run=True,
            fixed_sample_ids=fixed_sample_ids,
            required_paper_sample=int(required_paper_sample),
            parent_argv=parent_argv,
        )

    if completed and not force_overwrite:
        raise paper_mod.WriterConflictError(
            "output root already contains a completed coordgrid bundle; "
            "rerun with --force-overwrite to replace it"
        )
    lock_path = paper_mod._acquire_writer_lock(
        output_root, allow_force=force_overwrite
    )
    try:
        return _run_inner(
            manifest_path=manifest_path,
            baseline_root=baseline_root,
            output_root=output_root,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device_choice=device_choice,
            dry_run=False,
            fixed_sample_ids=fixed_sample_ids,
            required_paper_sample=int(required_paper_sample),
            parent_argv=parent_argv,
        )
    finally:
        paper_mod._release_writer_lock(lock_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brdt_srunet_coordgrid_extension")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixed-sample-id", type=int, action="append", dest="fixed_ids")
    parser.add_argument("--required-paper-sample", type=int, default=255)
    parser.add_argument("--force-overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parent_argv = sys.argv[1:] if argv is None else list(argv)
    write_invocation_artifacts(
        output_root,
        SCRIPT_PATH,
        parent_argv,
        vars(args),
        extra={
            "backlog_item": BACKLOG_ITEM,
            "claim_boundary": CLAIM_BOUNDARY,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )
    try:
        result = run_srunet_coordgrid_extension(
            manifest_path=Path(args.manifest),
            baseline_root=Path(args.baseline_root),
            output_root=output_root,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            device_choice=str(args.device),
            dry_run=bool(args.dry_run),
            fixed_sample_ids=args.fixed_ids,
            required_paper_sample=int(args.required_paper_sample),
            parent_argv=parent_argv,
            force_overwrite=bool(args.force_overwrite),
        )
    except (BaselineLineageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

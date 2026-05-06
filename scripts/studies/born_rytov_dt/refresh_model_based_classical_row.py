"""Refresh only the BRDT model-based Born inverse row in an existing bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import preflight_visuals as visuals_mod
from scripts.studies.born_rytov_dt.data import load_dataset_authority
from scripts.studies.born_rytov_dt.run_config import LossWeights
from scripts.studies.born_rytov_dt.run_preflight import (
    BACKLOG_ITEM,
    MODEL_BASED_INVERSE_EXECUTION_PATH,
    MODEL_BASED_INVERSE_VERSION,
    PREFLIGHT_MANIFEST_NAME,
    ROW_SUMMARY_NAME,
    TrainingContract,
    _build_operator,
    _device_name,
    _evaluate_model_based_inverse_split,
    _load_saved_row_arrays,
    _model_based_inverse_config,
    _neuralop_available,
    _row_metrics_from_summary,
    _save_fixed_sample_arrays,
    _select_device,
    _write_manifest,
    _write_row_invocation_artifacts,
    attempt_classical_backend_with_fix,
    classical_inverse_authorization,
    resolve_row_roster,
    row_contract_fingerprint,
    writer_lock,
)


def _contract_from_payload(payload: Mapping[str, Any]) -> TrainingContract:
    weights = payload.get("loss_weights") or {}
    return TrainingContract(
        epochs=int(payload.get("epochs", 20)),
        batch_size=int(payload.get("batch_size", 8)),
        learning_rate=float(payload.get("learning_rate", 2e-4)),
        optimizer=str(payload.get("optimizer", "Adam")),
        loss_weights=LossWeights(
            image=float(weights.get("image", 1.0)),
            physics=float(weights.get("physics", 0.1)),
            relative_physics=float(weights.get("relative_physics", 0.1)),
            tv=float(weights.get("tv", 1e-5)),
            positivity=float(weights.get("positivity", 1e-4)),
        ),
        seed=int(payload.get("seed", 42)),
    )


def _record_completed_arrays(
    *,
    row_id: str,
    sample_arrays: Mapping[int, Mapping[str, np.ndarray]],
    fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]],
    fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]],
    fixed_targets: Dict[int, Dict[str, np.ndarray]],
) -> None:
    for sid, arrays in sample_arrays.items():
        fixed_q_pred_by_row[sid][row_id] = arrays["q_pred"]
        fixed_sino_pred_by_row[sid][row_id] = arrays["sino_pred"]
        fixed_targets.setdefault(
            sid,
            {
                "q_target": arrays["q_target"],
                "sino_obs": arrays["sino_obs"],
            },
        )


def _write_aggregate_outputs(
    *,
    output_root: Path,
    fixed_sample_ids: List[int],
    row_metrics: List[metrics_mod.RowMetrics],
    fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]],
    fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]],
    fixed_targets: Dict[int, Dict[str, np.ndarray]],
) -> None:
    visuals_dir = output_root / "visuals"
    source_arrays_dir = output_root / "figures" / "source_arrays"
    metrics_mod.write_metric_schema(output_root / "metric_schema.json")
    metrics_mod.write_metrics_json(output_root / "metrics.json", row_metrics)
    metrics_mod.write_metrics_csv(output_root / "metrics.csv", row_metrics)

    visual_entries: List[visuals_mod.VisualBundleEntry] = []
    figure_paths: List[str] = []
    source_arrays_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    for sid in fixed_sample_ids:
        per_row_q = fixed_q_pred_by_row.get(sid, {})
        per_row_sino = fixed_sino_pred_by_row.get(sid, {})
        targets = fixed_targets.get(sid)
        if not per_row_q or targets is None:
            continue
        target_array_path = source_arrays_dir / f"sample_{sid:04d}_q_target.npy"
        sino_obs_array_path = source_arrays_dir / f"sample_{sid:04d}_sino_obs.npy"
        np.save(target_array_path, targets["q_target"])
        np.save(sino_obs_array_path, targets["sino_obs"])
        for row_id, q_pred in per_row_q.items():
            pred_path = source_arrays_dir / f"sample_{sid:04d}_{row_id}_q_pred.npy"
            sino_pred_path = (
                source_arrays_dir / f"sample_{sid:04d}_{row_id}_sino_pred.npy"
            )
            np.save(pred_path, q_pred)
            np.save(
                sino_pred_path,
                per_row_sino.get(row_id, np.zeros_like(targets["sino_obs"])),
            )
            visual_entries.append(
                visuals_mod.VisualBundleEntry(
                    sample_id=int(sid),
                    row_id=row_id,
                    pred_array=str(pred_path.relative_to(output_root)),
                    target_array=str(target_array_path.relative_to(output_root)),
                    sino_pred_array=str(sino_pred_path.relative_to(output_root)),
                    sino_obs_array=str(sino_obs_array_path.relative_to(output_root)),
                )
            )

    if fixed_sample_ids:
        first_id = fixed_sample_ids[0]
        per_row_q = fixed_q_pred_by_row.get(first_id, {})
        per_row_sino = fixed_sino_pred_by_row.get(first_id, {})
        targets = fixed_targets.get(first_id)
        if per_row_q and targets is not None:
            compare_path = visuals_dir / "brdt_compare_q.png"
            error_path = visuals_dir / "brdt_error_q.png"
            sino_path = visuals_dir / "brdt_sinogram_residual.png"
            visuals_mod.render_compare_q(
                preds_by_row=per_row_q,
                target=targets["q_target"],
                out_path=compare_path,
                sample_id=int(first_id),
            )
            visuals_mod.render_error_q(
                preds_by_row=per_row_q,
                target=targets["q_target"],
                out_path=error_path,
                sample_id=int(first_id),
            )
            visuals_mod.render_sinogram_residual(
                sino_obs=targets["sino_obs"],
                sino_preds_by_row=per_row_sino,
                out_path=sino_path,
                sample_id=int(first_id),
            )
            figure_paths = [
                str(compare_path.relative_to(output_root)),
                str(error_path.relative_to(output_root)),
                str(sino_path.relative_to(output_root)),
            ]

    visuals_mod.write_visual_manifest(
        output_root / "visual_manifest.json",
        figures=figure_paths,
        entries=visual_entries,
    )


def refresh_model_based_classical_row(
    *,
    manifest_path: Path,
    preflight_root: Path,
    device_choice: str = "auto",
    parent_argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rewrite only the classical row and rebuild aggregate bundle outputs."""
    preflight_root = Path(preflight_root).resolve()
    parent_argv = list(parent_argv) if parent_argv is not None else []
    with writer_lock(preflight_root):
        preflight_manifest_path = preflight_root / PREFLIGHT_MANIFEST_NAME
        preflight_manifest = json.loads(preflight_manifest_path.read_text())
        authority = load_dataset_authority(manifest_path)
        rows = resolve_row_roster(manifest_path=manifest_path, hybrid_label="hybrid_resnet")
        row_by_id = {row.row_id: row for row in rows}
        classical_row = row_by_id["classical_born_backprop"]
        fixed_sample_ids = [int(i) for i in preflight_manifest["fixed_sample_ids"]]
        contract = _contract_from_payload(preflight_manifest["training_contract"])
        device = _select_device(device_choice)
        operator = _build_operator(device)
        inverse_config = _model_based_inverse_config(authority)
        backend, narrow_fix_attempts = attempt_classical_backend_with_fix()
        classical_inverse_auth = classical_inverse_authorization(
            authority.raw_manifest,
            manifest_path=authority.manifest_path,
        )
        operator_pointer = (
            authority.raw_manifest.get("operator", {}).get("validation_artifact")
            or authority.raw_manifest.get("operator", {}).get("validation_report")
            or "unspecified"
        )
        row_fp = row_contract_fingerprint(
            row_id=classical_row.row_id,
            model=classical_row.model,
            training=classical_row.training,
            input_mode=classical_row.input_mode,
            dataset_id=str(authority.dataset_id),
            operator_pointer=str(operator_pointer),
            training_contract=contract.as_dict(),
            fixed_sample_ids=fixed_sample_ids,
            in_channels=int(preflight_manifest["input_contract"]["in_channels"]),
            classical_backend_name=MODEL_BASED_INVERSE_EXECUTION_PATH,
            execution_path=MODEL_BASED_INVERSE_EXECUTION_PATH,
            solver_version=MODEL_BASED_INVERSE_VERSION,
            solver_config=inverse_config.as_dict(),
        )

        rows_dir = preflight_root / "rows"
        source_arrays_dir = preflight_root / "figures" / "source_arrays"
        classical_dir = rows_dir / classical_row.row_id
        classical_dir.mkdir(parents=True, exist_ok=True)
        _write_row_invocation_artifacts(
            row_dir=classical_dir,
            row=classical_row,
            contract=contract,
            fixed_sample_ids=fixed_sample_ids,
            in_channels=int(preflight_manifest["input_contract"]["in_channels"]),
            device=device,
            classical_backend=backend,
            contract_fingerprint=row_fp,
            parent_argv=parent_argv,
            extra_attempts=narrow_fix_attempts,
        )

        started = time.perf_counter()
        image_metrics, meas_metrics, sample_arrays, solver_summary = (
            _evaluate_model_based_inverse_split(
                authority=authority,
                operator=operator,
                device=device,
                batch_size=int(contract.batch_size),
                fixed_sample_ids=fixed_sample_ids,
                config=inverse_config,
            )
        )
        eval_seconds = time.perf_counter() - started
        _save_fixed_sample_arrays(
            source_arrays_dir=source_arrays_dir,
            sample_arrays=sample_arrays,
            row_id=classical_row.row_id,
        )
        runtime_block = metrics_mod.collect_runtime_metadata(
            device=str(device),
            device_name=_device_name(device),
            epochs=0,
            batch_size=int(contract.batch_size),
            learning_rate=float(contract.learning_rate),
            parameter_count=0,
            wall_time_train_s=0.0,
            wall_time_eval_s=eval_seconds,
            row_status="completed",
            extras={
                "solver": solver_summary,
                "odtbrain_diagnostic": {
                    "backend_detection": {
                        "name": backend.name,
                        "reason": backend.reason,
                        "claim_boundary": backend.claim_boundary,
                    },
                    "narrow_fix_attempts": list(narrow_fix_attempts),
                    "inverse_authorization": dict(classical_inverse_auth),
                },
            },
        )
        classical_payload: Dict[str, Any] = {
            "row_id": classical_row.row_id,
            "row_status": "completed",
            "paper_label": classical_row.visible_label,
            "architecture": classical_row.model,
            "execution_path": MODEL_BASED_INVERSE_EXECUTION_PATH,
            "solver_version": MODEL_BASED_INVERSE_VERSION,
            "solver_config": inverse_config.as_dict(),
            "solver_summary": solver_summary,
            "image_metrics": image_metrics,
            "measurement_metrics": meas_metrics,
            "runtime": runtime_block,
            "contract_fingerprint": row_fp,
            "narrow_fix_attempts": list(narrow_fix_attempts),
            "classical_inverse_authorization": dict(classical_inverse_auth),
        }
        (classical_dir / ROW_SUMMARY_NAME).write_text(
            json.dumps(classical_payload, indent=2, sort_keys=True) + "\n"
        )

        row_metrics: List[metrics_mod.RowMetrics] = []
        row_status_updates: Dict[str, Dict[str, Any]] = {}
        fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
            sid: {} for sid in fixed_sample_ids
        }
        fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
            sid: {} for sid in fixed_sample_ids
        }
        fixed_targets: Dict[int, Dict[str, np.ndarray]] = {}
        for row in rows:
            if row.row_id == classical_row.row_id:
                summary = classical_payload
                arrays = sample_arrays
            else:
                summary_path = rows_dir / row.row_id / ROW_SUMMARY_NAME
                if not summary_path.exists():
                    raise FileNotFoundError(f"missing cached neural row: {summary_path}")
                summary = json.loads(summary_path.read_text())
                arrays = {}
                if summary.get("row_status") == "completed":
                    loaded = _load_saved_row_arrays(
                        source_arrays_dir=source_arrays_dir,
                        row_id=row.row_id,
                        fixed_sample_ids=fixed_sample_ids,
                    )
                    if loaded is None:
                        raise FileNotFoundError(
                            f"missing fixed-sample arrays for cached row {row.row_id}"
                        )
                    arrays = loaded
            row_metrics.append(
                _row_metrics_from_summary(
                    summary,
                    default_paper_label=row.visible_label,
                    default_arch=row.model,
                )
            )
            row_status_updates[row.row_id] = dict(summary)
            if summary.get("row_status") == "completed":
                _record_completed_arrays(
                    row_id=row.row_id,
                    sample_arrays=arrays,
                    fixed_q_pred_by_row=fixed_q_pred_by_row,
                    fixed_sino_pred_by_row=fixed_sino_pred_by_row,
                    fixed_targets=fixed_targets,
                )

        _write_aggregate_outputs(
            output_root=preflight_root,
            fixed_sample_ids=fixed_sample_ids,
            row_metrics=row_metrics,
            fixed_q_pred_by_row=fixed_q_pred_by_row,
            fixed_sino_pred_by_row=fixed_sino_pred_by_row,
            fixed_targets=fixed_targets,
        )
        preflight_manifest["rows"] = [
            {**row_by_id[row_id].to_dict(), **payload}
            for row_id, payload in row_status_updates.items()
        ]
        preflight_manifest["row_metrics"] = [row.to_dict() for row in row_metrics]
        preflight_manifest["bundle_artifacts"] = {
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "metric_schema": "metric_schema.json",
            "visual_manifest": "visual_manifest.json",
            "visuals_dir": "visuals/",
            "source_arrays_dir": "figures/source_arrays/",
            "rows_dir": "rows/",
        }
        preflight_manifest["dependency_status"] = {
            "odtbrain": backend.name == "odtbrain",
            "neuralop": _neuralop_available(),
        }
        notes = dict(preflight_manifest.get("notes") or {})
        notes["model_based_inverse"] = {
            "execution_path": MODEL_BASED_INVERSE_EXECUTION_PATH,
            "version": MODEL_BASED_INVERSE_VERSION,
            "config": inverse_config.as_dict(),
        }
        notes["classical_inverse_authorization"] = dict(classical_inverse_auth)
        notes["classical_narrow_fix_attempts"] = list(narrow_fix_attempts)
        notes.setdefault("row_contract_fingerprints", {})[
            classical_row.row_id
        ] = row_fp
        preflight_manifest["notes"] = notes
        preflight_manifest["resumed_rows"] = [
            row.row_id for row in rows if row.row_id != classical_row.row_id
        ]
        _write_manifest(preflight_manifest, preflight_manifest_path)
        return {
            "preflight_manifest_path": str(preflight_manifest_path),
            "classical_row_summary_path": str(classical_dir / ROW_SUMMARY_NAME),
            "metrics_json_path": str(preflight_root / "metrics.json"),
            "visual_manifest_path": str(preflight_root / "visual_manifest.json"),
            "resumed_rows": preflight_manifest["resumed_rows"],
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_refresh_model_based_classical_row",
        description="Refresh only the BRDT model-based Born inverse row.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preflight-root", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = refresh_model_based_classical_row(
        manifest_path=args.manifest,
        preflight_root=args.preflight_root,
        device_choice=args.device,
        parent_argv=["--manifest", str(args.manifest), "--preflight-root", str(args.preflight_root)],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

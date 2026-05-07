"""BRDT sinogram-input paper-evidence runner.

Runs the current BRDT manuscript contract in which learned models consume the
measured complex sinogram directly. The model-based Born inverse is retained as
the non-learned reference row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from scripts.studies.born_rytov_dt import convergence as conv_mod
from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import run_brdt_40ep_paper_evidence as paper_mod
from scripts.studies.born_rytov_dt import run_preflight as preflight_mod
from scripts.studies.born_rytov_dt.data import load_dataset_authority
from scripts.studies.born_rytov_dt.run_config import (
    LossWeights,
    RowConfig,
    sinogram_input_row_roster,
)
from scripts.studies.invocation_logging import capture_runtime_provenance, write_invocation_artifacts


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py"
BACKLOG_ITEM = "2026-05-07-brdt-sinogram-input-40ep-paper-evidence"
CLAIM_BOUNDARY = "paper_evidence_brdt_sinogram_input"
EXPECTED_EPOCHS = 40
EXPECTED_INPUT_CONTRACT = {
    "input_mode": "sinogram",
    "in_channels": 2,
}
DEFAULT_MANIFEST = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-brdt-four-row-preflight/decision_support_dataset/"
    "dataset_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog"
) / BACKLOG_ITEM
HISTORICAL_CONTEXT_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-05-06-brdt-corrected-ffno-40ep-rerun"
)


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


def _manifest_payload(
    *,
    output_root: Path,
    manifest_path: Path,
    rows: List[RowConfig],
    authority: Any,
    contract: preflight_mod.TrainingContract,
    fixed_sample_ids: List[int],
    required_paper_sample: int,
) -> Dict[str, Any]:
    counts = dict((authority.raw_manifest.get("split") or {}).get("counts") or {})
    return {
        "schema_version": "brdt_sinogram_input_40ep_v2",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "promotion_status": "pending",
        "output_root": str(output_root),
        "dataset_manifest_path": str(manifest_path),
        "historical_context_root": str(HISTORICAL_CONTEXT_ROOT.resolve()),
        "dataset": {
            "dataset_id": str(authority.dataset_id),
            "tier": authority.raw_manifest.get("dataset_identity", {}).get("tier"),
            "split_counts": counts,
            "normalization": authority.normalization.as_dict()
            if hasattr(authority.normalization, "as_dict")
            else {
                "mean": authority.normalization.mean,
                "std": authority.normalization.std,
                "qmin": authority.normalization.qmin,
                "qmax": authority.normalization.qmax,
            },
        },
        "input_contract": {
            "input_mode": "sinogram",
            "in_channels": 2,
            "tensor_shape": ["B", 2, "angle_count", "detector_size"],
            "model_input_source": "measured complex sinogram real/imag channels",
            "born_inverse_role": "non_learned_reference_only",
        },
        "training_contract": contract.as_dict(),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "required_paper_sample": int(required_paper_sample),
        "rows": [row.to_dict() for row in rows],
    }


def _runtime_block(
    *,
    device: Any,
    contract: preflight_mod.TrainingContract,
    parameter_count: int,
    train_seconds: float,
    eval_seconds: float,
    runtime_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    return metrics_mod.collect_runtime_metadata(
        device=str(device),
        device_name=preflight_mod._device_name(device),
        epochs=int(contract.epochs),
        batch_size=int(contract.batch_size),
        learning_rate=float(contract.learning_rate),
        parameter_count=int(parameter_count),
        wall_time_train_s=float(train_seconds),
        wall_time_eval_s=float(eval_seconds),
        row_status="completed",
        extras=dict(runtime_meta),
    )


def _write_row_outputs(
    *,
    row: RowConfig,
    row_dir: Path,
    image_metrics: Mapping[str, float],
    meas_metrics: Mapping[str, float],
    runtime_block: Mapping[str, Any],
    runtime_meta: Mapping[str, Any],
    output_dynamic_range: Mapping[str, float],
    execution_path: str,
) -> metrics_mod.RowMetrics:
    model_profile_path = row_dir / "model_profile.json"
    _write_json(
        model_profile_path,
        {
            "row_id": row.row_id,
            "architecture": row.model,
            "parameter_count": int(runtime_block.get("parameter_count", 0)),
            "arch_kwargs": dict(runtime_meta.get("arch_kwargs") or {}),
            "input_mode": row.input_mode,
        },
    )
    row_summary = {
        "row_id": row.row_id,
        "row_status": "completed",
        "paper_label": row.visible_label,
        "architecture": row.model,
        "input_mode": row.input_mode,
        "execution_path": execution_path,
        "image_metrics": {
            k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS
        },
        "measurement_metrics": dict(meas_metrics),
        "supporting": {
            k: v for k, v in image_metrics.items() if k in metrics_mod.SUPPORTING_METRICS
        },
        "runtime": dict(runtime_block),
        "model_profile_path": str(model_profile_path),
        "model_state_path": runtime_meta.get("model_state_path"),
        "history_json_path": runtime_meta.get("history_json_path"),
        "history_csv_path": runtime_meta.get("history_csv_path"),
        "history_length": runtime_meta.get("history_length"),
        "scheduler": runtime_meta.get("scheduler"),
        "output_dynamic_range": dict(output_dynamic_range),
    }
    _write_json(row_dir / "row_summary.json", row_summary)
    return metrics_mod.RowMetrics(
        row_id=row.row_id,
        paper_label=row.visible_label,
        architecture=row.model,
        row_status="completed",
        image={k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS},
        measurement=dict(meas_metrics),
        supporting={
            k: v for k, v in image_metrics.items() if k in metrics_mod.SUPPORTING_METRICS
        },
        runtime=dict(runtime_block),
        extra={"input_mode": row.input_mode},
    )


def _historical_baseline_rows() -> Dict[str, Mapping[str, Any]]:
    metrics_path = HISTORICAL_CONTEXT_ROOT / "combined_metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text())
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in payload.get("rows") or []:
        row_id = str(row.get("row_id") or "")
        if row_id == "hybrid_resnet":
            row_id = "sru_net"
        if row_id in {"ffno", "sru_net"}:
            indexed[row_id] = row
    return indexed


def _sample_bundle_ok(output_root: Path, sample_id: int) -> bool:
    visuals_dir = output_root / "visuals"
    arrays_dir = output_root / "figures" / "source_arrays"
    required_paths = [
        visuals_dir / f"sample_{sample_id:04d}_compare_q.png",
        visuals_dir / f"sample_{sample_id:04d}_error_q.png",
        visuals_dir / f"sample_{sample_id:04d}_sinogram_residual.png",
        arrays_dir / f"sample_{sample_id:04d}_q_target.npy",
        arrays_dir / f"sample_{sample_id:04d}_sino_obs.npy",
        arrays_dir / f"sample_{sample_id:04d}_classical_born_backprop_q_pred.npy",
        arrays_dir / f"sample_{sample_id:04d}_ffno_q_pred.npy",
        arrays_dir / f"sample_{sample_id:04d}_sru_net_q_pred.npy",
    ]
    return all(path.exists() for path in required_paths)


def _build_provenance_checks(
    *,
    output_root: Path,
    provenance_paths: Mapping[str, str],
    gate_rows: Mapping[str, Mapping[str, Any]],
    sample_id: int,
) -> Dict[str, bool]:
    runtime_path = Path(provenance_paths["runtime_provenance_path"])
    exit_path = output_root / "run_exit_status.json"
    runtime_payload = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    exit_payload = json.loads(exit_path.read_text()) if exit_path.exists() else {}
    log_path = _resolve_log_path(output_root)
    tracked_pid = runtime_payload.get("tracked_pid")
    exit_code_proof = (
        runtime_payload.get("tracked_pid") is not None
        and runtime_payload.get("tracked_pid") == exit_payload.get("tracked_pid")
        and int(exit_payload.get("exit_code", 1)) == 0
        and str(exit_payload.get("status")) == "completed"
    )
    model_profiles_present = all(
        (output_root / "rows" / row_id / "model_profile.json").exists()
        for row_id in gate_rows
    )
    torch_block = runtime_payload.get("torch") or {}
    return {
        "runtime_provenance": runtime_path.exists(),
        "git_provenance": runtime_payload.get("git_sha") is not None
        and runtime_payload.get("git_dirty") is not None,
        "host_provenance": runtime_payload.get("hostname") is not None
        and runtime_payload.get("gpu_count") is not None,
        "python_provenance": runtime_payload.get("python_executable") is not None
        and runtime_payload.get("python_version") is not None,
        "torch_provenance": bool(torch_block.get("version")),
        "dataset_identity": Path(
            provenance_paths["dataset_identity_manifest_path"]
        ).exists(),
        "split_manifest": Path(provenance_paths["split_manifest_path"]).exists(),
        "model_profiles": model_profiles_present,
        "run_log_present": log_path is not None and log_path.exists(),
        "sample_255_visual_bundle": _sample_bundle_ok(output_root, sample_id),
        "exit_code_proof": bool(exit_code_proof),
    }


def _write_visual_manifest(output_root: Path, claim_boundary: str) -> Path:
    path = output_root / "visual_manifest.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.update(
        {
            "backlog_item": BACKLOG_ITEM,
            "claim_boundary": claim_boundary,
            "output_root": str(output_root),
        }
    )
    _write_json(path, payload)
    return path


def _write_combined_manifest(output_root: Path, claim_boundary: str) -> Path:
    metrics_path = output_root / "combined_metrics.json"
    rows = []
    if metrics_path.exists():
        try:
            rows = [
                {
                    "row_id": row.get("row_id"),
                    "row_status": row.get("row_status"),
                    "paper_label": row.get("paper_label"),
                    "architecture": row.get("architecture"),
                }
                for row in (json.loads(metrics_path.read_text()).get("rows") or [])
            ]
        except Exception:
            rows = []
    payload = {
        "schema_version": "brdt_sinogram_input_40ep_combined_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": claim_boundary,
        "output_root": str(output_root),
        "historical_context_root": str(HISTORICAL_CONTEXT_ROOT.resolve()),
        "paper_evidence_gate_path": str(output_root / "paper_evidence_gate.json"),
        "preflight_manifest_path": str(output_root / "preflight_manifest.json"),
        "rows": rows,
    }
    out_path = output_root / "combined_manifest.json"
    _write_json(out_path, payload)
    return out_path


def _run_sinogram_input_40ep_inner(
    *,
    manifest_path: Path,
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
    authority = load_dataset_authority(manifest_path)
    preflight_mod.assert_decision_support_manifest(authority.raw_manifest)
    operator_pointer = (
        authority.raw_manifest.get("operator", {}).get("validation_artifact")
        or authority.raw_manifest.get("operator", {}).get("validation_report")
        or "unspecified"
    )
    rows = sinogram_input_row_roster(
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
        manifest_path=manifest_path,
        rows=rows,
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
    source_arrays_dir = output_root / "figures" / "source_arrays"
    row_metrics: List[metrics_mod.RowMetrics] = []
    fixed_targets: Dict[int, Dict[str, Any]] = {}
    fixed_q_pred_by_row: Dict[int, Dict[str, Any]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    fixed_sino_pred_by_row: Dict[int, Dict[str, Any]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    current_rows: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        row_dir = output_root / "rows" / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        if row.row_id == "classical_born_backprop":
            started = time.perf_counter()
            image_metrics, meas_metrics, sample_arrays, output_dynamic_range = (
                preflight_mod._evaluate_split(
                    module=None,
                    authority=authority,
                    operator=operator,
                    backend=backend,
                    device=device,
                    in_channels=1,
                    classical_only=True,
                    fixed_sample_ids=fixed_sample_ids,
                    out_dir=row_dir,
                )
            )
            eval_seconds = time.perf_counter() - started
            for sid, arrays in sample_arrays.items():
                fixed_targets[int(sid)] = {
                    "q_target": arrays["q_target"],
                    "sino_obs": arrays["sino_obs"],
                }
                fixed_q_pred_by_row[int(sid)][row.row_id] = arrays["q_pred"]
                fixed_sino_pred_by_row[int(sid)][row.row_id] = arrays["sino_pred"]
            preflight_mod._save_fixed_sample_arrays(
                source_arrays_dir=source_arrays_dir,
                sample_arrays=sample_arrays,
                row_id=row.row_id,
            )
            runtime_meta: Dict[str, Any] = {
                "input_mode": row.input_mode,
                "model_state_path": None,
                "history_json_path": None,
                "history_csv_path": None,
                "history_length": 0,
            }
            runtime = _runtime_block(
                device=device,
                contract=contract,
                parameter_count=0,
                train_seconds=0.0,
                eval_seconds=eval_seconds,
                runtime_meta=runtime_meta,
            )
            row_metrics.append(
                _write_row_outputs(
                    row=row,
                    row_dir=row_dir,
                    image_metrics=image_metrics,
                    meas_metrics=meas_metrics,
                    runtime_block=runtime,
                    runtime_meta=runtime_meta,
                    output_dynamic_range=output_dynamic_range,
                    execution_path="sinogram_input_classical_reference",
                )
            )
            continue

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
            raise RuntimeError(f"failed to build row {row.row_id}: {runtime_meta}")
        started = time.perf_counter()
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
        eval_seconds = time.perf_counter() - started
        for sid, arrays in sample_arrays.items():
            fixed_targets.setdefault(
                int(sid),
                {
                    "q_target": arrays["q_target"],
                    "sino_obs": arrays["sino_obs"],
                },
            )
            fixed_q_pred_by_row[int(sid)][row.row_id] = arrays["q_pred"]
            fixed_sino_pred_by_row[int(sid)][row.row_id] = arrays["sino_pred"]
        preflight_mod._save_fixed_sample_arrays(
            source_arrays_dir=source_arrays_dir,
            sample_arrays=sample_arrays,
            row_id=row.row_id,
        )
        runtime_meta = {**dict(runtime_meta), "input_mode": row.input_mode}
        runtime = _runtime_block(
            device=device,
            contract=contract,
            parameter_count=int(runtime_meta.get("parameter_count", 0)),
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
            runtime_meta=runtime_meta,
        )
        row_metrics.append(
            _write_row_outputs(
                row=row,
                row_dir=row_dir,
                image_metrics=image_metrics,
                meas_metrics=meas_metrics,
                runtime_block=runtime,
                runtime_meta=runtime_meta,
                output_dynamic_range=output_dynamic_range,
                execution_path="sinogram_input_train_eval",
            )
        )
        history_payload = conv_mod.load_history(
            Path(str(runtime_meta["history_json_path"]))
        )
        current_rows[row.row_id] = {
            "row_summary": json.loads((row_dir / "row_summary.json").read_text()),
            "history_summary": conv_mod.summarize_history(
                row_id=row.row_id,
                history_payload=history_payload,
            ),
        }

    metrics_mod.write_metrics_json(
        output_root / "metrics.json",
        row_metrics,
        claim_boundary=CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_csv(output_root / "metrics.csv", row_metrics)
    metrics_payload = json.loads((output_root / "metrics.json").read_text())
    _write_json(output_root / "combined_metrics.json", metrics_payload)
    metrics_mod.write_metrics_csv(output_root / "combined_metrics.csv", row_metrics)

    visual_status = paper_mod._write_bundle_visuals(
        output_root=output_root,
        sample_id=int(required_paper_sample),
        fixed_targets=fixed_targets,
        fixed_q_pred_by_row=fixed_q_pred_by_row,
        fixed_sino_pred_by_row=fixed_sino_pred_by_row,
        baseline_root=output_root,
    )
    baseline_rows = _historical_baseline_rows()
    audit_payload = conv_mod.build_convergence_audit(
        backlog_item=BACKLOG_ITEM,
        baseline_rows={
            row_id: {
                "image_metrics": row.get("image"),
                "measurement_metrics": row.get("measurement"),
                "supporting": row.get("supporting"),
                "runtime": row.get("runtime"),
            }
            for row_id, row in baseline_rows.items()
            if row_id in current_rows
        },
        current_rows=current_rows,
    )
    conv_mod.write_convergence_audit_json(
        output_root / "convergence_audit.json", audit_payload
    )
    conv_mod.write_convergence_audit_csv(
        output_root / "convergence_audit.csv", audit_payload
    )

    contract_dict = contract.as_dict()
    gate_rows = {
        row_id: {
            "row_status": data["row_summary"]["row_status"],
            "history_records": data["history_summary"]["history_records"],
            "scheduler_matches_contract": paper_mod._scheduler_matches_contract(
                row_summary=data["row_summary"],
                contract_dict=contract_dict,
            ),
        }
        for row_id, data in current_rows.items()
    }
    paper_evidence_gate_path = output_root / "paper_evidence_gate.json"
    log_path = _resolve_log_path(output_root)
    try:
        paper_mod._write_run_exit_status(
            output_root,
            pid=os.getpid(),
            exit_code=0,
            status="completed",
            log_path=log_path,
        )
        provenance_checks = _build_provenance_checks(
            output_root=output_root,
            provenance_paths=provenance_paths,
            gate_rows=gate_rows,
            sample_id=int(required_paper_sample),
        )
        manifest_input_contract = json.loads(
            preflight_manifest_path.read_text()
        ).get("input_contract")
        gate_payload = conv_mod.build_paper_evidence_gate(
            backlog_item=BACKLOG_ITEM,
            expected_epochs=EXPECTED_EPOCHS,
            rows=gate_rows,
            provenance_checks=provenance_checks,
            input_contract=manifest_input_contract,
            expected_input_contract=EXPECTED_INPUT_CONTRACT,
        )
        conv_mod.write_paper_evidence_gate(paper_evidence_gate_path, gate_payload)
        paper_mod._reseed_top_level_manifest_with_gate(
            preflight_manifest_path,
            gate_payload=gate_payload,
            paper_evidence_gate_path=paper_evidence_gate_path,
        )
        paper_mod._reseed_metrics_with_gate(
            output_root=output_root,
            gate_payload=gate_payload,
        )
        visual_manifest_path = _write_visual_manifest(
            output_root, str(gate_payload["claim_boundary"])
        )
        combined_manifest_path = _write_combined_manifest(
            output_root, str(gate_payload["claim_boundary"])
        )
    except BaseException:
        try:
            paper_mod._write_run_exit_status(
                output_root,
                pid=os.getpid(),
                exit_code=1,
                status="failed",
                log_path=log_path,
            )
        except Exception:
            pass
        raise

    return {
        "dry_run": False,
        "preflight_manifest_path": str(preflight_manifest_path),
        "metrics_json_path": str(output_root / "metrics.json"),
        "combined_metrics_json_path": str(output_root / "combined_metrics.json"),
        "combined_manifest_json_path": str(combined_manifest_path),
        "convergence_audit_json_path": str(output_root / "convergence_audit.json"),
        "paper_evidence_gate_json_path": str(paper_evidence_gate_path),
        "visual_manifest_path": str(visual_manifest_path),
        "source_arrays_dir": str(output_root / "figures" / "source_arrays"),
        **provenance_paths,
    }


def run_sinogram_input_40ep(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
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
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_sample_ids = [int(i) for i in (fixed_sample_ids or [255])]
    if int(required_paper_sample) not in set(fixed_sample_ids):
        raise ValueError("required_paper_sample must be in fixed_sample_ids")
    parent_argv = list(parent_argv) if parent_argv is not None else list(sys.argv[1:])
    if dry_run:
        return _run_sinogram_input_40ep_inner(
            manifest_path=manifest_path,
            output_root=output_root,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device_choice=device_choice,
            dry_run=True,
            fixed_sample_ids=fixed_sample_ids,
            required_paper_sample=int(required_paper_sample),
            parent_argv=parent_argv,
        )

    paper_mod._refuse_overwrite_when_completed(
        output_root, allow_force=force_overwrite
    )
    lock_path = paper_mod._acquire_writer_lock(
        output_root, allow_force=force_overwrite
    )
    try:
        return _run_sinogram_input_40ep_inner(
            manifest_path=manifest_path,
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
    parser = argparse.ArgumentParser(prog="brdt_sinogram_input_40ep")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
    result = run_sinogram_input_40ep(
        manifest_path=args.manifest,
        output_root=args.output_root,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device_choice=str(args.device),
        dry_run=bool(args.dry_run),
        fixed_sample_ids=args.fixed_ids,
        required_paper_sample=int(args.required_paper_sample),
        parent_argv=list(argv) if argv is not None else list(sys.argv[1:]),
        force_overwrite=bool(args.force_overwrite),
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

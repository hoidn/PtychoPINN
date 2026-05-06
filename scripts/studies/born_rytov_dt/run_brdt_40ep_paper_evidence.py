"""Immutable BRDT 40-epoch paper-evidence runner."""

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
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle
from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import preflight_visuals as visuals_mod
from scripts.studies.born_rytov_dt import run_preflight as preflight_mod
from scripts.studies.born_rytov_dt.classical import ClassicalBackendInfo
from scripts.studies.born_rytov_dt.data import (
    DatasetAuthority,
    load_dataset_authority,
)
from scripts.studies.born_rytov_dt.run_config import RowConfig
from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py"
BACKLOG_ITEM = "2026-05-05-brdt-supervised-born-40ep-paper-evidence"
CLAIM_BOUNDARY = "decision_support_convergence_followup"
EXPECTED_EPOCHS = 40
ROW_SUMMARY_NAME = preflight_mod.ROW_SUMMARY_NAME


def _default_contract() -> preflight_mod.TrainingContract:
    return preflight_mod.TrainingContract(
        epochs=EXPECTED_EPOCHS,
        batch_size=16,
        learning_rate=2e-4,
        optimizer="Adam",
        seed=42,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _make_rows(*, dataset_id: str, operator_version: str) -> List[RowConfig]:
    return [
        RowConfig(
            row_id="hybrid_resnet",
            model="hybrid_resnet",
            training=preflight_mod.DEFAULT_TRAINING_LABEL,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="Hybrid ResNet",
        ),
        RowConfig(
            row_id="ffno",
            model="ffno",
            training=preflight_mod.DEFAULT_TRAINING_LABEL,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="FFNO",
        ),
    ]


def _validate_ffno_extension_bundle(
    ffno_extension_root: Path,
    *,
    baseline_root: Path,
) -> Dict[str, Any]:
    root = Path(ffno_extension_root).resolve()
    manifest_path = root / "preflight_manifest.json"
    metrics_path = root / "metrics.json"
    combined_path = root / "combined_metrics.json"
    if not manifest_path.exists():
        raise ext_bundle.BaselineContractMismatchError(
            f"ffno extension bundle missing preflight_manifest.json at {manifest_path}"
        )
    if not metrics_path.exists():
        raise ext_bundle.BaselineContractMismatchError(
            f"ffno extension bundle missing metrics.json at {metrics_path}"
        )
    if not combined_path.exists():
        raise ext_bundle.BaselineContractMismatchError(
            f"ffno extension bundle missing combined_metrics.json at {combined_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("backlog_item") != ext_bundle.EXTENSION_BACKLOG_ITEM:
        raise ext_bundle.BaselineContractMismatchError(
            "ffno extension backlog_item mismatch"
        )
    lineage = manifest.get("baseline_lineage") or {}
    declared_baseline = lineage.get("baseline_root")
    if declared_baseline is not None and Path(declared_baseline).resolve() != Path(
        baseline_root
    ).resolve():
        raise ext_bundle.BaselineContractMismatchError(
            "ffno extension baseline_root does not match the requested baseline root"
        )
    return manifest


def _baseline_rows(
    *, baseline_root: Path, ffno_extension_root: Path
) -> Dict[str, Mapping[str, Any]]:
    baseline_metrics = _read_json(Path(baseline_root) / "metrics.json")
    ffno_metrics = _read_json(Path(ffno_extension_root) / "metrics.json")
    indexed = {str(row["row_id"]): row for row in baseline_metrics.get("rows") or []}
    for row in ffno_metrics.get("rows") or []:
        indexed[str(row["row_id"])] = row
    return indexed


def _write_bundle_visuals(
    *,
    output_root: Path,
    sample_id: int,
    fixed_targets: Mapping[int, Mapping[str, np.ndarray]],
    fixed_q_pred_by_row: Mapping[int, Mapping[str, np.ndarray]],
    fixed_sino_pred_by_row: Mapping[int, Mapping[str, np.ndarray]],
    baseline_root: Path,
) -> Dict[str, Any]:
    visuals_dir = output_root / "visuals"
    arrays_dir = output_root / "figures" / "source_arrays"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    targets = fixed_targets.get(int(sample_id))
    preds_by_row = dict(fixed_q_pred_by_row.get(int(sample_id)) or {})
    sino_preds_by_row = dict(fixed_sino_pred_by_row.get(int(sample_id)) or {})
    classical_q_path = (
        Path(baseline_root)
        / "figures"
        / "source_arrays"
        / f"sample_{int(sample_id):04d}_classical_born_backprop_q_pred.npy"
    )
    classical_sino_path = (
        Path(baseline_root)
        / "figures"
        / "source_arrays"
        / f"sample_{int(sample_id):04d}_classical_born_backprop_sino_pred.npy"
    )
    classical_present = False
    if classical_q_path.exists() and classical_sino_path.exists():
        preds_by_row["classical_born_backprop"] = np.load(classical_q_path)
        sino_preds_by_row["classical_born_backprop"] = np.load(classical_sino_path)
        np.save(
            arrays_dir / f"sample_{int(sample_id):04d}_classical_born_backprop_q_pred.npy",
            preds_by_row["classical_born_backprop"],
        )
        np.save(
            arrays_dir
            / f"sample_{int(sample_id):04d}_classical_born_backprop_sino_pred.npy",
            sino_preds_by_row["classical_born_backprop"],
        )
        classical_present = True
    figure_paths: List[str] = []
    if targets is not None and preds_by_row:
        compare_path = visuals_dir / f"sample_{int(sample_id):04d}_compare_q.png"
        error_path = visuals_dir / f"sample_{int(sample_id):04d}_error_q.png"
        sino_path = visuals_dir / f"sample_{int(sample_id):04d}_sinogram_residual.png"
        visuals_mod.render_compare_q(
            preds_by_row=preds_by_row,
            target=targets["q_target"],
            out_path=compare_path,
            sample_id=int(sample_id),
        )
        visuals_mod.render_error_q(
            preds_by_row=preds_by_row,
            target=targets["q_target"],
            out_path=error_path,
            sample_id=int(sample_id),
        )
        visuals_mod.render_sinogram_residual(
            sino_obs=targets["sino_obs"],
            sino_preds_by_row=sino_preds_by_row,
            out_path=sino_path,
            sample_id=int(sample_id),
        )
        figure_paths = [
            str(compare_path.relative_to(output_root)),
            str(error_path.relative_to(output_root)),
            str(sino_path.relative_to(output_root)),
        ]
    visual_manifest = {
        "schema_version": "brdt_paper_evidence_visuals_v1",
        "required_sample_id": int(sample_id),
        "classical_present": bool(classical_present),
        "rows_present": sorted(preds_by_row.keys()),
        "figures": figure_paths,
    }
    _write_json(output_root / "visual_manifest.json", visual_manifest)
    return {
        "classical_present": bool(classical_present),
        "figures": figure_paths,
        "rows_present": sorted(preds_by_row.keys()),
        "visual_manifest_path": str(output_root / "visual_manifest.json"),
    }


def _write_top_level_provenance(
    *,
    output_root: Path,
    authority: DatasetAuthority,
    fixed_sample_ids: List[int],
) -> Dict[str, str]:
    runtime_provenance_path = output_root / "runtime_provenance.json"
    dataset_identity_path = output_root / "dataset_identity_manifest.json"
    split_manifest_path = output_root / "split_manifest.json"
    _write_json(
        runtime_provenance_path,
        {
            **capture_runtime_provenance(),
            "pid": int(os.getpid()),
        },
    )
    _write_json(
        dataset_identity_path,
        {
            "dataset_id": authority.dataset_id,
            "manifest_path": str(authority.manifest_path),
            "dataset_identity": authority.raw_manifest.get("dataset_identity"),
        },
    )
    _write_json(
        split_manifest_path,
        {
            "split_counts": authority.raw_manifest.get("split", {}).get("counts"),
            "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        },
    )
    return {
        "runtime_provenance_path": str(runtime_provenance_path),
        "dataset_identity_manifest_path": str(dataset_identity_path),
        "split_manifest_path": str(split_manifest_path),
    }


def _build_manifest(
    *,
    output_root: Path,
    baseline_root: Path,
    ffno_extension_root: Path,
    authority: DatasetAuthority,
    rows: List[RowConfig],
    fixed_sample_ids: List[int],
    required_paper_sample: int,
    contract: preflight_mod.TrainingContract,
) -> Dict[str, Any]:
    return {
        "schema_version": "brdt_paper_evidence_runner_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "promotion_status": "pending",
        "output_root": str(output_root),
        "baseline_lineage": {
            "baseline_root": str(Path(baseline_root).resolve()),
            "ffno_extension_root": str(Path(ffno_extension_root).resolve()),
        },
        "dataset": {
            "dataset_id": authority.dataset_id,
            "tier": authority.raw_manifest["dataset_identity"].get("tier"),
            "manifest_path": str(authority.manifest_path),
            "split_counts": authority.raw_manifest["split"]["counts"],
            "normalization": authority.normalization.as_dict(),
        },
        "operator": {
            "validation_artifact": authority.raw_manifest.get("operator", {}).get(
                "validation_artifact"
            ),
            "geometry": {
                "grid_size": dc.LOCKED_GRID_SIZE,
                "detector_size": dc.LOCKED_DETECTOR_SIZE,
                "angle_count": dc.LOCKED_ANGLE_COUNT,
                "wavelength_px": dc.LOCKED_WAVELENGTH_PX,
                "medium_ri": dc.LOCKED_MEDIUM_RI,
                "mode": dc.LOCKED_OPERATOR_MODE,
                "normalize": dc.LOCKED_NORMALIZE,
            },
        },
        "training_contract": contract.as_dict(),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "required_paper_sample": int(required_paper_sample),
        "rows": [row.to_dict() for row in rows],
    }


def run_paper_evidence(
    *,
    baseline_root: Path,
    ffno_extension_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[preflight_mod.TrainingContract] = None,
    device_choice: str = "auto",
    dry_run: bool = False,
    fixed_sample_ids: Optional[List[int]] = None,
    required_paper_sample: int = 255,
    parent_argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(baseline_root).resolve()
    ffno_extension_root = Path(ffno_extension_root).resolve()
    parent_argv = list(parent_argv) if parent_argv is not None else []

    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    _validate_ffno_extension_bundle(ffno_extension_root, baseline_root=baseline_root)
    authority = load_dataset_authority(manifest_path)
    preflight_mod.assert_decision_support_manifest(authority.raw_manifest)
    contract = contract or _default_contract()
    fixed_sample_ids = (
        [int(i) for i in fixed_sample_ids]
        if fixed_sample_ids is not None
        else [int(i) for i in baseline_manifest.get("fixed_sample_ids") or []]
    )
    if int(required_paper_sample) not in set(int(i) for i in fixed_sample_ids):
        raise ValueError(
            "required_paper_sample must be included in fixed_sample_ids for the paper-facing bundle"
        )
    operator_pointer = (
        authority.raw_manifest.get("operator", {}).get("validation_artifact")
        or authority.raw_manifest.get("operator", {}).get("validation_report")
        or "unspecified"
    )
    rows = _make_rows(dataset_id=str(authority.dataset_id), operator_version=str(operator_pointer))
    manifest_payload = _build_manifest(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        authority=authority,
        rows=rows,
        fixed_sample_ids=fixed_sample_ids,
        required_paper_sample=int(required_paper_sample),
        contract=contract,
    )
    preflight_manifest_path = output_root / "preflight_manifest.json"
    _write_json(preflight_manifest_path, manifest_payload)
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=parent_argv,
        parsed_args={
            "baseline_root": str(baseline_root),
            "ffno_extension_root": str(ffno_extension_root),
            "manifest_path": str(manifest_path),
            "output_root": str(output_root),
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
    provenance_paths = _write_top_level_provenance(
        output_root=output_root,
        authority=authority,
        fixed_sample_ids=fixed_sample_ids,
    )
    if dry_run:
        metrics_mod.write_metric_schema(
            output_root / "metric_schema.json",
            claim_boundary=CLAIM_BOUNDARY,
        )
        return {
            "dry_run": True,
            "preflight_manifest_path": str(preflight_manifest_path),
            "metric_schema_path": str(output_root / "metric_schema.json"),
            **provenance_paths,
        }

    device = preflight_mod._select_device(device_choice)
    operator = preflight_mod._build_operator(device)
    input_backend = preflight_mod._born_init_backend()
    source_arrays_dir = output_root / "figures" / "source_arrays"
    fixed_targets: Dict[int, Dict[str, np.ndarray]] = {}
    fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
        int(sid): {} for sid in fixed_sample_ids
    }
    row_metrics: List[metrics_mod.RowMetrics] = []
    current_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_dir = output_root / "rows" / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        row_fp = preflight_mod.row_contract_fingerprint(
            row_id=row.row_id,
            model=row.model,
            training=row.training,
            input_mode=row.input_mode,
            dataset_id=str(authority.dataset_id),
            operator_pointer=str(operator_pointer),
            training_contract=contract.as_dict(),
            fixed_sample_ids=fixed_sample_ids,
            in_channels=1,
            classical_backend_name=f"{input_backend.name}:born_init_image",
        )
        preflight_mod._write_row_invocation_artifacts(
            row_dir=row_dir,
            row=row,
            contract=contract,
            fixed_sample_ids=fixed_sample_ids,
            in_channels=1,
            device=device,
            classical_backend=input_backend,
            contract_fingerprint=row_fp,
            parent_argv=parent_argv,
            backlog_item=BACKLOG_ITEM,
        )
        module, runtime_meta, train_seconds = preflight_mod._train_neural_row(
            row=row,
            authority=authority,
            operator=operator,
            backend=input_backend,
            device=device,
            contract=contract,
            in_channels=1,
            output_dir=row_dir,
        )
        if module is None:
            raise ValueError(f"failed to build adapter for row {row.row_id}")
        eval_started = time.perf_counter()
        image_metrics, meas_metrics, sample_arrays, output_dynamic_range = (
            preflight_mod._evaluate_split(
                module=module,
                authority=authority,
                operator=operator,
                backend=input_backend,
                device=device,
                in_channels=1,
                classical_only=False,
                fixed_sample_ids=fixed_sample_ids,
                out_dir=row_dir,
            )
        )
        eval_seconds = time.perf_counter() - eval_started
        preflight_mod._save_fixed_sample_arrays(
            source_arrays_dir=source_arrays_dir,
            sample_arrays=sample_arrays,
            row_id=row.row_id,
        )
        for sid, arrays in sample_arrays.items():
            fixed_q_pred_by_row[int(sid)][row.row_id] = arrays["q_pred"]
            fixed_sino_pred_by_row[int(sid)][row.row_id] = arrays["sino_pred"]
            fixed_targets.setdefault(
                int(sid),
                {
                    "q_target": arrays["q_target"],
                    "sino_obs": arrays["sino_obs"],
                },
            )
        runtime_block = metrics_mod.collect_runtime_metadata(
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
                "train_steps": runtime_meta.get("train_steps"),
                "final_loss_breakdown": runtime_meta.get("final_loss_breakdown"),
                "model_state_path": runtime_meta.get("model_state_path"),
                "history_json_path": runtime_meta.get("history_json_path"),
                "history_csv_path": runtime_meta.get("history_csv_path"),
                "history_length": runtime_meta.get("history_length"),
                "scheduler": runtime_meta.get("scheduler"),
                "output_dynamic_range": output_dynamic_range,
            },
        )
        row_metric = metrics_mod.RowMetrics(
            row_id=row.row_id,
            paper_label=row.visible_label,
            architecture=row.model,
            row_status="completed",
            image={k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS},
            measurement=meas_metrics,
            supporting={
                k: v for k, v in image_metrics.items() if k in metrics_mod.SUPPORTING_METRICS
            },
            runtime=runtime_block,
        )
        row_metrics.append(row_metric)
        model_profile_path = row_dir / "model_profile.json"
        _write_json(
            model_profile_path,
            {
                "row_id": row.row_id,
                "architecture": row.model,
                "parameter_count": int(runtime_meta.get("parameter_count", 0)),
                "arch_kwargs": runtime_meta.get("arch_kwargs") or {},
            },
        )
        row_summary = {
            "row_id": row.row_id,
            "row_status": "completed",
            "paper_label": row.visible_label,
            "architecture": row.model,
            "execution_path": "paper_evidence_train_eval",
            "image_metrics": {
                k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS
            },
            "measurement_metrics": meas_metrics,
            "supporting": {
                k: v for k, v in image_metrics.items() if k in metrics_mod.SUPPORTING_METRICS
            },
            "model_state_path": runtime_meta.get("model_state_path"),
            "model_profile_path": str(model_profile_path),
            "history_json_path": runtime_meta.get("history_json_path"),
            "history_csv_path": runtime_meta.get("history_csv_path"),
            "history_length": runtime_meta.get("history_length"),
            "scheduler": runtime_meta.get("scheduler"),
            "runtime": runtime_block,
            "contract_fingerprint": row_fp,
            "output_dynamic_range": output_dynamic_range,
        }
        _write_json(row_dir / ROW_SUMMARY_NAME, row_summary)
        history_payload = conv_mod.load_history(Path(runtime_meta["history_json_path"]))
        current_rows[row.row_id] = {
            "row_summary": row_summary,
            "history_summary": conv_mod.summarize_history(
                row_id=row.row_id,
                history_payload=history_payload,
            ),
        }

    metrics_mod.write_metric_schema(
        output_root / "metric_schema.json",
        claim_boundary=CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_json(
        output_root / "metrics.json",
        row_metrics,
        claim_boundary=CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_csv(output_root / "metrics.csv", row_metrics)
    metrics_json_payload = _read_json(output_root / "metrics.json")
    _write_json(output_root / "combined_metrics.json", metrics_json_payload)
    metrics_mod.write_metrics_csv(output_root / "combined_metrics.csv", row_metrics)

    visual_status = _write_bundle_visuals(
        output_root=output_root,
        sample_id=int(required_paper_sample),
        fixed_targets=fixed_targets,
        fixed_q_pred_by_row=fixed_q_pred_by_row,
        fixed_sino_pred_by_row=fixed_sino_pred_by_row,
        baseline_root=baseline_root,
    )
    baseline_rows = _baseline_rows(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )
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

    gate_rows = {
        row_id: {
            "row_status": data["row_summary"]["row_status"],
            "history_records": data["history_summary"]["history_records"],
            "scheduler_matches_contract": (
                data["row_summary"].get("scheduler", {}).get("name")
                == contract.as_dict().get("scheduler")
            ),
        }
        for row_id, data in current_rows.items()
    }
    _write_json(
        output_root / "run_exit_status.json",
        {
            "pid": int(os.getpid()),
            "exit_code": 0,
            "status": "completed",
        },
    )
    provenance_checks = {
        "runtime_provenance": Path(provenance_paths["runtime_provenance_path"]).exists(),
        "dataset_identity": Path(
            provenance_paths["dataset_identity_manifest_path"]
        ).exists(),
        "split_manifest": Path(provenance_paths["split_manifest_path"]).exists(),
        "sample_255_visual_bundle": bool(visual_status["classical_present"])
        and bool(visual_status["figures"]),
        "exit_code_proof": (output_root / "run_exit_status.json").exists(),
        "evidence_surfaces_prepared": False,
        "same_contract_lineage": True,
    }
    gate_payload = conv_mod.build_paper_evidence_gate(
        backlog_item=BACKLOG_ITEM,
        expected_epochs=EXPECTED_EPOCHS,
        rows=gate_rows,
        provenance_checks=provenance_checks,
    )
    conv_mod.write_paper_evidence_gate(
        output_root / "paper_evidence_gate.json", gate_payload
    )

    return {
        "preflight_manifest_path": str(preflight_manifest_path),
        "metrics_json_path": str(output_root / "metrics.json"),
        "combined_metrics_json_path": str(output_root / "combined_metrics.json"),
        "convergence_audit_json_path": str(output_root / "convergence_audit.json"),
        "paper_evidence_gate_json_path": str(output_root / "paper_evidence_gate.json"),
        "visual_manifest_path": visual_status["visual_manifest_path"],
        **provenance_paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_run_40ep_paper_evidence",
        description="Run the immutable BRDT 40-epoch Hybrid ResNet + FFNO paper-evidence bundle.",
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--ffno-extension-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--required-paper-sample", type=int, default=255)
    parser.add_argument("--fixed-sample-ids", nargs="+", type=int, default=[145, 83, 255, 126])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_paper_evidence(
        baseline_root=args.baseline_root,
        ffno_extension_root=args.ffno_extension_root,
        manifest_path=args.manifest,
        output_root=args.output_root,
        device_choice=str(args.device),
        dry_run=bool(args.dry_run),
        fixed_sample_ids=[int(i) for i in args.fixed_sample_ids],
        required_paper_sample=int(args.required_paper_sample),
        parent_argv=sys.argv[1:] if argv is None else list(argv),
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

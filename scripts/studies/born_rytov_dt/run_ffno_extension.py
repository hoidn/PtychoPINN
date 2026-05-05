"""Append-only BRDT FFNO row extension runner.

This script is the authoritative entrypoint for the
``2026-05-04-brdt-ffno-row-extension`` backlog item. It runs ONLY the
new FFNO row under the same dataset/operator/input/split/normalization/
fixed-sample/training contract used by the completed four-row preflight,
and then assembles a read-only-lineage five-row combined bundle
(baseline four rows + FFNO) without rerunning or overwriting any
baseline artifact.

Authority surface:

- The locked baseline four-row preflight bundle is consumed read-only.
  Its dataset id, operator pointer, input contract, fixed sample IDs
  and training contract are inherited byte-for-byte; the FFNO extension
  refuses to launch if those fields cannot be reproduced.
- The extension root is published under the FFNO-extension backlog item
  with ``claim_boundary="decision_support_append_only"``. The
  ``preflight_manifest.json`` here is the FFNO-only view; the cross-row
  combined view lives in ``combined_*`` files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
from scripts.studies.born_rytov_dt.run_config import (
    DEFAULT_TRAINING_LABEL,
    LossWeights,
    RowConfig,
)
from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_ffno_extension.py"
BACKLOG_ITEM = ext_bundle.EXTENSION_BACKLOG_ITEM
CLAIM_BOUNDARY = ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY
FFNO_ROW_ID = ext_bundle.FFNO_ROW_ID
FFNO_PAPER_LABEL = ext_bundle.FFNO_PAPER_LABEL
ROW_SUMMARY_NAME = preflight_mod.ROW_SUMMARY_NAME


def make_ffno_row(
    *,
    dataset_id: str,
    operator_version: str,
    training_label: str = DEFAULT_TRAINING_LABEL,
) -> RowConfig:
    """Build the bounded BRDT FFNO row under the locked supervised+Born contract."""
    return RowConfig(
        row_id=FFNO_ROW_ID,
        model="ffno",
        training=training_label,
        input_mode="born_init_image",
        dataset_id=dataset_id,
        operator_version=operator_version,
        paper_label=FFNO_PAPER_LABEL,
    )


def _resolve_extension_training_contract(
    baseline_manifest: Mapping[str, Any],
) -> preflight_mod.TrainingContract:
    """Inherit the baseline training contract verbatim.

    The plan locks 20 epochs / batch 16 / lr 2e-4 / seed 42 and the
    completed loss weights. We read those values straight from the
    baseline manifest so the FFNO row cannot drift via local defaults.
    """
    base = baseline_manifest.get("training_contract", {}) or {}
    weights_block = base.get("loss_weights", {}) or {}
    weights = LossWeights(
        image=float(weights_block.get("image", 1.0)),
        physics=float(weights_block.get("physics", 0.1)),
        relative_physics=float(weights_block.get("relative_physics", 0.1)),
        tv=float(weights_block.get("tv", 1e-5)),
        positivity=float(weights_block.get("positivity", 1e-4)),
    )
    return preflight_mod.TrainingContract(
        epochs=int(base.get("epochs", 20)),
        batch_size=int(base.get("batch_size", 16)),
        learning_rate=float(base.get("learning_rate", 2e-4)),
        optimizer=str(base.get("optimizer", "Adam")),
        loss_weights=weights,
        seed=int(base.get("seed", 42)),
    )


def _build_extension_manifest(
    *,
    output_root: Path,
    baseline_root: Path,
    authority: DatasetAuthority,
    ffno_row: RowConfig,
    fixed_sample_ids: List[int],
    fixed_sample_seed: int,
    contract: preflight_mod.TrainingContract,
    in_channels: int,
    classical_backend: ClassicalBackendInfo,
    contract_fingerprint: str,
    baseline_fingerprint: str,
    operator_pointer: str,
) -> Dict[str, Any]:
    """Build the extension-root preflight_manifest for the FFNO-only view."""
    payload: Dict[str, Any] = {
        "schema_version": "brdt_preflight_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "next_backlog_item": "n/a",
        "output_root": str(output_root),
        "baseline_lineage": {
            "baseline_root": str(Path(baseline_root).resolve()),
            "baseline_backlog_item": ext_bundle.BASELINE_BACKLOG_ITEM,
            "baseline_preflight_manifest": str(
                (Path(baseline_root) / "preflight_manifest.json").resolve()
            ),
            "baseline_metrics_json": str(
                (Path(baseline_root) / "metrics.json").resolve()
            ),
            "baseline_contract_fingerprint": baseline_fingerprint,
        },
        "dataset": {
            "dataset_id": authority.dataset_id,
            "tier": authority.raw_manifest["dataset_identity"].get("tier"),
            "manifest_path": str(authority.manifest_path),
            "split_counts": authority.raw_manifest["split"]["counts"],
            "normalization": authority.normalization.as_dict(),
            "claim_boundary": authority.raw_manifest.get("claim_boundary"),
        },
        "operator": {
            "validation_artifact": authority.raw_manifest.get("operator", {}).get(
                "validation_artifact"
            ),
            "validation_report": authority.raw_manifest.get("operator", {}).get(
                "validation_report"
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
            "operator_pointer": operator_pointer,
        },
        "input_contract": {
            "input_mode": "born_init_image",
            "in_channels": int(in_channels),
            "classical_backend": {
                "name": classical_backend.name,
                "reason": classical_backend.reason,
                "claim_boundary": classical_backend.claim_boundary,
            },
        },
        "training_contract": contract.as_dict(),
        "metric_schema": {
            "version": metrics_mod.METRIC_SCHEMA_VERSION,
            "blocking": {
                "image_space_physical_q": list(metrics_mod.IMAGE_METRICS),
                "measurement_space": list(metrics_mod.MEASUREMENT_METRICS),
            },
            "supporting": list(metrics_mod.SUPPORTING_METRICS),
            "runtime_fields": list(metrics_mod.RUNTIME_FIELDS),
        },
        "fixed_sample_seed": int(fixed_sample_seed),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "rows": [ffno_row.to_dict()],
        "row_contract_fingerprint": contract_fingerprint,
        "environment": preflight_mod._runtime_environment_block(),
    }
    return payload


def _write_manifest(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def run_ffno_extension(
    *,
    baseline_root: Path,
    manifest_path: Path,
    output_root: Path,
    contract: Optional[preflight_mod.TrainingContract] = None,
    fixed_sample_seed: int = 17,
    fixed_sample_count: int = 4,
    in_channels: int = 1,
    device_choice: str = "auto",
    dry_run: bool = False,
    parent_argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute the append-only FFNO row and emit the combined-bundle artifacts."""
    parent_argv = list(parent_argv) if parent_argv is not None else []
    output_root = Path(output_root).resolve()
    baseline_root = Path(baseline_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Validate baseline bundle BEFORE acquiring writer lock so a bad
    # baseline never produces a half-written extension root.
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    baseline_fingerprint = ext_bundle.baseline_contract_fingerprint(baseline_manifest)

    with preflight_mod.writer_lock(output_root):
        authority = load_dataset_authority(manifest_path)
        preflight_mod.assert_decision_support_manifest(authority.raw_manifest)

        operator_pointer = (
            authority.raw_manifest.get("operator", {}).get("validation_artifact")
            or authority.raw_manifest.get("operator", {}).get("validation_report")
            or "unspecified"
        )
        ffno_row = make_ffno_row(
            dataset_id=str(authority.dataset_id),
            operator_version=str(operator_pointer),
            training_label=DEFAULT_TRAINING_LABEL,
        )

        # Inherit baseline training contract unless the caller already
        # supplied one (e.g. tests injecting a fast contract).
        if contract is None:
            contract = _resolve_extension_training_contract(baseline_manifest)

        fixed_sample_ids = [
            int(i) for i in baseline_manifest.get("fixed_sample_ids") or []
        ]
        if not fixed_sample_ids:
            # Defensive fallback; the validator already checked this.
            fixed_sample_ids = preflight_mod.choose_fixed_sample_ids(
                manifest_path,
                count=int(fixed_sample_count),
                seed=int(fixed_sample_seed),
            )

        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id=str(authority.dataset_id),
            extension_input_mode="born_init_image",
            extension_in_channels=int(in_channels),
            extension_training_contract=contract.as_dict(),
            extension_fixed_sample_ids=fixed_sample_ids,
            extension_operator_pointer=str(operator_pointer),
            extension_claim_boundary=CLAIM_BOUNDARY,
        )

        input_backend = preflight_mod._born_init_backend()
        device = preflight_mod._select_device(device_choice)

        contract_fingerprint = preflight_mod.row_contract_fingerprint(
            row_id=ffno_row.row_id,
            model=ffno_row.model,
            training=ffno_row.training,
            input_mode=ffno_row.input_mode,
            dataset_id=str(authority.dataset_id),
            operator_pointer=str(operator_pointer),
            training_contract=contract.as_dict(),
            fixed_sample_ids=fixed_sample_ids,
            in_channels=int(in_channels),
            classical_backend_name=f"{input_backend.name}:born_init_image",
        )

        extension_manifest_payload = _build_extension_manifest(
            output_root=output_root,
            baseline_root=baseline_root,
            authority=authority,
            ffno_row=ffno_row,
            fixed_sample_ids=fixed_sample_ids,
            fixed_sample_seed=int(fixed_sample_seed),
            contract=contract,
            in_channels=int(in_channels),
            classical_backend=input_backend,
            contract_fingerprint=contract_fingerprint,
            baseline_fingerprint=baseline_fingerprint,
            operator_pointer=str(operator_pointer),
        )
        manifest_path_out = output_root / preflight_mod.PREFLIGHT_MANIFEST_NAME
        _write_manifest(extension_manifest_payload, manifest_path_out)

        if dry_run:
            metrics_mod.write_metric_schema(
                output_root / "metric_schema.json",
                claim_boundary=CLAIM_BOUNDARY,
            )
            return {
                "dry_run": True,
                "preflight_manifest_path": str(manifest_path_out),
                "metric_schema_path": str(output_root / "metric_schema.json"),
                "baseline_contract_fingerprint": baseline_fingerprint,
            }

        # ---- Live FFNO row execution ----
        operator = preflight_mod._build_operator(device)
        rows_dir = output_root / "rows"
        ffno_dir = rows_dir / FFNO_ROW_ID
        visuals_dir = output_root / "visuals"
        source_arrays_dir = output_root / "figures" / "source_arrays"
        ffno_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        source_arrays_dir.mkdir(parents=True, exist_ok=True)

        preflight_mod._write_row_invocation_artifacts(
            row_dir=ffno_dir,
            row=ffno_row,
            contract=contract,
            fixed_sample_ids=fixed_sample_ids,
            in_channels=int(in_channels),
            device=device,
            classical_backend=input_backend,
            contract_fingerprint=contract_fingerprint,
            parent_argv=parent_argv,
            backlog_item=BACKLOG_ITEM,
        )

        module, runtime_meta, train_seconds = preflight_mod._train_neural_row(
            row=ffno_row,
            authority=authority,
            operator=operator,
            backend=input_backend,
            device=device,
            contract=contract,
            in_channels=int(in_channels),
            output_dir=ffno_dir,
        )
        if module is None:
            err = runtime_meta.get("adapter_build_error", {}) or {}
            blocked_runtime = metrics_mod.collect_runtime_metadata(
                device=str(device),
                device_name=preflight_mod._device_name(device),
                epochs=0,
                batch_size=int(contract.batch_size),
                learning_rate=float(contract.learning_rate),
                parameter_count=0,
                wall_time_train_s=0.0,
                wall_time_eval_s=0.0,
                row_status="blocked",
            )
            blocked_metrics = metrics_mod.RowMetrics(
                row_id=ffno_row.row_id,
                paper_label=ffno_row.visible_label,
                architecture=ffno_row.model,
                row_status="blocked",
                blocker_reason=str(err.get("reason", "adapter_build_error")),
                blocker_message=str(err.get("message", "adapter_build_error")),
                runtime=blocked_runtime,
            )
            row_payload = {
                "row_id": ffno_row.row_id,
                "row_status": "blocked",
                "paper_label": ffno_row.visible_label,
                "architecture": ffno_row.model,
                "execution_path": "neural_blocked",
                "blocker_reason": blocked_metrics.blocker_reason,
                "blocker_message": blocked_metrics.blocker_message,
                "runtime": blocked_runtime,
                "contract_fingerprint": contract_fingerprint,
            }
            (ffno_dir / ROW_SUMMARY_NAME).write_text(
                json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
            )
            row_metrics: List[metrics_mod.RowMetrics] = [blocked_metrics]
            visual_entries: List[visuals_mod.VisualBundleEntry] = []
            figure_paths: List[str] = []
        else:
            eval_started = time.perf_counter()
            (
                image_metrics,
                meas_metrics,
                sample_arrays,
                output_dynamic_range,
            ) = preflight_mod._evaluate_split(
                module=module,
                authority=authority,
                operator=operator,
                backend=input_backend,
                device=device,
                in_channels=int(in_channels),
                classical_only=False,
                fixed_sample_ids=fixed_sample_ids,
                out_dir=ffno_dir,
            )
            eval_seconds = time.perf_counter() - eval_started
            preflight_mod._save_fixed_sample_arrays(
                source_arrays_dir=source_arrays_dir,
                sample_arrays=sample_arrays,
                row_id=ffno_row.row_id,
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
                    "final_loss_breakdown": runtime_meta.get(
                        "final_loss_breakdown"
                    ),
                    "model_state_path": runtime_meta.get("model_state_path"),
                    "output_dynamic_range": output_dynamic_range,
                },
            )
            row_metrics = [
                metrics_mod.RowMetrics(
                    row_id=ffno_row.row_id,
                    paper_label=ffno_row.visible_label,
                    architecture=ffno_row.model,
                    row_status="completed",
                    image={
                        k: v
                        for k, v in image_metrics.items()
                        if k in metrics_mod.IMAGE_METRICS
                    },
                    measurement=meas_metrics,
                    supporting={
                        k: v
                        for k, v in image_metrics.items()
                        if k in metrics_mod.SUPPORTING_METRICS
                    },
                    runtime=runtime_block,
                )
            ]
            row_payload = {
                "row_id": ffno_row.row_id,
                "row_status": "completed",
                "paper_label": ffno_row.visible_label,
                "architecture": ffno_row.model,
                "execution_path": "neural_train_eval",
                "parameter_count": int(runtime_meta.get("parameter_count", 0)),
                "image_metrics": image_metrics,
                "measurement_metrics": meas_metrics,
                "wall_time_train_s": float(train_seconds),
                "wall_time_eval_s": float(eval_seconds),
                "model_state_path": runtime_meta.get("model_state_path"),
                "runtime": runtime_block,
                "contract_fingerprint": contract_fingerprint,
                "output_dynamic_range": output_dynamic_range,
            }
            (ffno_dir / ROW_SUMMARY_NAME).write_text(
                json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
            )

            visual_entries = []
            figure_paths = []
            for sid, arrays in sample_arrays.items():
                target_array_path = source_arrays_dir / f"sample_{sid:04d}_q_target.npy"
                sino_obs_array_path = source_arrays_dir / f"sample_{sid:04d}_sino_obs.npy"
                pred_path = (
                    source_arrays_dir
                    / f"sample_{sid:04d}_{ffno_row.row_id}_q_pred.npy"
                )
                sino_pred_path = (
                    source_arrays_dir
                    / f"sample_{sid:04d}_{ffno_row.row_id}_sino_pred.npy"
                )
                visual_entries.append(
                    visuals_mod.VisualBundleEntry(
                        sample_id=int(sid),
                        row_id=ffno_row.row_id,
                        pred_array=str(pred_path.relative_to(output_root)),
                        target_array=str(target_array_path.relative_to(output_root)),
                        sino_pred_array=str(sino_pred_path.relative_to(output_root)),
                        sino_obs_array=str(
                            sino_obs_array_path.relative_to(output_root)
                        ),
                    )
                )

            if fixed_sample_ids:
                first_id = int(fixed_sample_ids[0])
                if first_id in sample_arrays:
                    targets = sample_arrays[first_id]
                    preds_by_row = {ffno_row.row_id: targets["q_pred"]}
                    sino_preds_by_row = {ffno_row.row_id: targets["sino_pred"]}
                    compare_path = visuals_dir / "brdt_compare_q.png"
                    error_path = visuals_dir / "brdt_error_q.png"
                    sino_path = visuals_dir / "brdt_sinogram_residual.png"
                    visuals_mod.render_compare_q(
                        preds_by_row=preds_by_row,
                        target=targets["q_target"],
                        out_path=compare_path,
                        sample_id=first_id,
                    )
                    visuals_mod.render_error_q(
                        preds_by_row=preds_by_row,
                        target=targets["q_target"],
                        out_path=error_path,
                        sample_id=first_id,
                    )
                    visuals_mod.render_sinogram_residual(
                        sino_obs=targets["sino_obs"],
                        sino_preds_by_row=sino_preds_by_row,
                        out_path=sino_path,
                        sample_id=first_id,
                    )
                    figure_paths = [
                        str(compare_path.relative_to(output_root)),
                        str(error_path.relative_to(output_root)),
                        str(sino_path.relative_to(output_root)),
                    ]

        # Aggregated metric / schema / visual outputs (FFNO-only view).
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
        visuals_mod.write_visual_manifest(
            output_root / "visual_manifest.json",
            figures=figure_paths,
            entries=visual_entries,
        )

        # Final manifest update with the row outcome attached.
        extension_manifest_payload["rows"] = [
            {**ffno_row.to_dict(), **row_payload},
        ]
        extension_manifest_payload["row_metrics"] = [
            r.to_dict() for r in row_metrics
        ]
        extension_manifest_payload["bundle_artifacts"] = {
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "metric_schema": "metric_schema.json",
            "visual_manifest": "visual_manifest.json",
            "visuals_dir": "visuals/",
            "source_arrays_dir": "figures/source_arrays/",
            "rows_dir": "rows/",
        }
        _write_manifest(extension_manifest_payload, manifest_path_out)

        # Combined bundle assembly (read-only-lineage five-row view).
        combined = ext_bundle.emit_combined_bundle(
            baseline_root=baseline_root,
            extension_root=output_root,
        )

        return {
            "preflight_manifest_path": str(manifest_path_out),
            "metrics_json_path": str(output_root / "metrics.json"),
            "metrics_csv_path": str(output_root / "metrics.csv"),
            "metric_schema_path": str(output_root / "metric_schema.json"),
            "visual_manifest_path": str(output_root / "visual_manifest.json"),
            "combined_metrics_json_path": str(combined["combined_metrics_json"]),
            "combined_metrics_csv_path": str(combined["combined_metrics_csv"]),
            "combined_manifest_json_path": str(combined["combined_manifest_json"]),
            "rows": [r.to_dict() for r in row_metrics],
            "baseline_contract_fingerprint": baseline_fingerprint,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_run_ffno_extension",
        description=(
            "Run the append-only BRDT FFNO row under the locked four-row "
            "preflight contract and emit a five-row combined bundle that "
            "references the baseline by lineage."
        ),
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fixed-sample-seed", type=int, default=17)
    parser.add_argument("--fixed-sample-count", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _maybe_override_contract(
    args: argparse.Namespace,
    baseline_manifest: Mapping[str, Any],
) -> Optional[preflight_mod.TrainingContract]:
    """Allow narrow CLI overrides while still inheriting baseline weights/seed.

    Used by tests that need a tiny epoch budget. Production runs leave
    these flags unset so the baseline contract is inherited byte-for-byte.
    """
    if all(
        getattr(args, k) is None
        for k in ("epochs", "batch_size", "learning_rate", "seed")
    ):
        return None
    base_contract = _resolve_extension_training_contract(baseline_manifest)
    return preflight_mod.TrainingContract(
        epochs=int(args.epochs) if args.epochs is not None else base_contract.epochs,
        batch_size=(
            int(args.batch_size)
            if args.batch_size is not None
            else base_contract.batch_size
        ),
        learning_rate=(
            float(args.learning_rate)
            if args.learning_rate is not None
            else base_contract.learning_rate
        ),
        optimizer=base_contract.optimizer,
        loss_weights=base_contract.loss_weights,
        seed=int(args.seed) if args.seed is not None else base_contract.seed,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parent_argv = sys.argv[1:] if argv is None else list(argv)

    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=parent_argv,
        parsed_args=vars(args),
        extra={
            "backlog_item": BACKLOG_ITEM,
            "claim_boundary": CLAIM_BOUNDARY,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )

    try:
        baseline_manifest = ext_bundle.validate_baseline_bundle(
            Path(args.baseline_root)
        )
    except ext_bundle.BaselineContractMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    contract = _maybe_override_contract(args, baseline_manifest)

    try:
        result = run_ffno_extension(
            baseline_root=Path(args.baseline_root),
            manifest_path=Path(args.manifest),
            output_root=output_root,
            contract=contract,
            fixed_sample_seed=int(args.fixed_sample_seed),
            fixed_sample_count=int(args.fixed_sample_count),
            in_channels=int(args.in_channels),
            device_choice=str(args.device),
            dry_run=bool(args.dry_run),
            parent_argv=parent_argv,
        )
    except (ValueError, ext_bundle.BaselineContractMismatchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

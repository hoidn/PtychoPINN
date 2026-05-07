"""BRDT sinogram-input paper-evidence runner.

Runs the current BRDT manuscript contract in which learned models consume the
measured complex sinogram directly. The model-based Born inverse is retained as
a non-learned reference only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import run_preflight as preflight_mod
from scripts.studies.born_rytov_dt.data import load_dataset_authority
from scripts.studies.born_rytov_dt.run_config import (
    LossWeights,
    RowConfig,
    sinogram_input_row_roster,
)
from scripts.studies.invocation_logging import write_invocation_artifacts


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py"
BACKLOG_ITEM = "2026-05-07-brdt-sinogram-input-40ep-paper-evidence"
CLAIM_BOUNDARY = "paper_evidence_brdt_sinogram_input"
DEFAULT_MANIFEST = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-brdt-four-row-preflight/decision_support_dataset/"
    "dataset_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog"
) / BACKLOG_ITEM


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _default_contract(
    *,
    epochs: int = 40,
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
        "schema_version": "brdt_sinogram_input_40ep_v1",
        "backlog_item": BACKLOG_ITEM,
        "claim_boundary": CLAIM_BOUNDARY,
        "output_root": str(output_root),
        "dataset_manifest_path": str(manifest_path),
        "dataset": {
            "dataset_id": str(authority.dataset_id),
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
        "execution_path": "sinogram_input_train_eval",
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


def run_sinogram_input_40ep(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    device_choice: str = "auto",
    dry_run: bool = False,
    fixed_sample_ids: Optional[List[int]] = None,
    required_paper_sample: int = 255,
) -> Dict[str, Any]:
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
    fixed_sample_ids = [int(i) for i in (fixed_sample_ids or [255])]
    if int(required_paper_sample) not in set(fixed_sample_ids):
        raise ValueError("required_paper_sample must be in fixed_sample_ids")
    contract = _default_contract(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    manifest = _manifest_payload(
        output_root=output_root,
        manifest_path=manifest_path,
        rows=rows,
        authority=authority,
        contract=contract,
        fixed_sample_ids=fixed_sample_ids,
        required_paper_sample=required_paper_sample,
    )
    preflight_manifest_path = output_root / "preflight_manifest.json"
    _write_json(preflight_manifest_path, manifest)
    write_invocation_artifacts(
        output_root,
        SCRIPT_PATH,
        sys.argv[1:],
        {
            "manifest_path": str(manifest_path),
            "output_root": str(output_root),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "device": str(device_choice),
            "dry_run": bool(dry_run),
            "fixed_sample_ids": fixed_sample_ids,
            "required_paper_sample": int(required_paper_sample),
        },
        extra={"backlog_item": BACKLOG_ITEM, "claim_boundary": CLAIM_BOUNDARY},
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

    device = preflight_mod._select_device(device_choice)
    operator = preflight_mod._build_operator(device)
    backend = preflight_mod._born_init_backend()
    source_arrays_dir = output_root / "figures" / "source_arrays"
    row_metrics: List[metrics_mod.RowMetrics] = []

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
            )
        )

    metrics_mod.write_metrics_json(
        output_root / "metrics.json",
        row_metrics,
        claim_boundary=CLAIM_BOUNDARY,
    )
    metrics_mod.write_metrics_csv(output_root / "metrics.csv", row_metrics)
    payload = json.loads((output_root / "metrics.json").read_text())
    _write_json(output_root / "combined_metrics.json", payload)
    metrics_mod.write_metrics_csv(output_root / "combined_metrics.csv", row_metrics)
    return {
        "dry_run": False,
        "preflight_manifest_path": str(preflight_manifest_path),
        "metrics_json_path": str(output_root / "metrics.json"),
        "combined_metrics_json_path": str(output_root / "combined_metrics.json"),
        "source_arrays_dir": str(source_arrays_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brdt_sinogram_input_40ep")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixed-sample-id", type=int, action="append", dest="fixed_ids")
    parser.add_argument("--required-paper-sample", type=int, default=255)
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
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

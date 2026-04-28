"""Reporting payloads for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.studies.pdebench_image128.run_config import required_primary_profiles_for_task
from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec


COMPARISON_METRIC_FIELDS = [
    "err_RMSE",
    "err_nRMSE",
    "relative_l2",
    "fRMSE_low",
    "fRMSE_mid",
    "fRMSE_high",
]
REQUIRED_RUN_ARTIFACTS = [
    "invocation.json",
    "dataset_manifest.json",
    "split_manifest.json",
    "comparison_summary.json",
]
FIXED_CONTRACT_FIELDS = [
    "dataset_file",
    "split_counts",
    "max_windows_per_trajectory",
    "epochs",
    "batch_size",
    "training_loss",
    "metric_family",
]
STRICT_CONTRACT_FIELDS = [
    *FIXED_CONTRACT_FIELDS,
    "history_len",
]


def darcy_literature_context() -> dict[str, Any]:
    caveat = (
        "Published values are calibration context, not exact native-128 local pass/fail thresholds; "
        "PDEBench/HAMLET commonly use x2 spatial subsampling and different training protocols."
    )
    return {
        "schema_version": "pdebench_darcy_literature_context_v1",
        "task_id": "darcy",
        "access_date": "2026-04-20",
        "sources": {
            "pdebench_repository": "https://github.com/pdebench/PDEBench",
            "pdebench_supplement_table_8": "https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Supplemental-Datasets_and_Benchmarks.pdf",
            "fno_jmlr": "https://jmlr.org/papers/volume24/21-1524/21-1524.pdf",
            "hamlet_openreview": "https://openreview.net/pdf/2a77316b975457831c5d7c21021f5bb87c99eff1.pdf",
            "pdebench_darcy_config": "https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/models/config/args/config_Darcy.yaml",
        },
        "calibration_targets": {
            "pdebench_unet": {"model": "U-Net", "RMSE": 6.4e-3, "nRMSE": 3.3e-2, "protocol_caveat": caveat},
            "pdebench_fno": {"model": "FNO", "RMSE": 1.2e-2, "nRMSE": 6.4e-2, "protocol_caveat": caveat},
            "oformer": {"model": "OFormer", "nRMSE": 2.05e-2, "protocol_caveat": caveat},
            "hamlet": {"model": "HAMLET", "nRMSE": 1.40e-2, "protocol_caveat": caveat},
            "fno_jmlr": {
                "model": "FNO",
                "relative_error_range": [1.08e-2, 9.8e-3],
                "protocol_caveat": "Canonical Darcy neural-operator context, not the exact PDEBench beta-file protocol.",
            },
        },
    }


def write_literature_context(output_root: Path, *, task_id: str = "darcy") -> Path:
    if task_id != "darcy":
        raise ValueError("only Darcy literature context is implemented")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "literature_context.json"
    path.write_text(json.dumps(darcy_literature_context(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_comparison_summary(
    *,
    task_id: str,
    mode: str,
    output_root: Path,
    profile_results: list[dict[str, Any]],
    run_id: str | None = None,
    blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    profile_ids = [str(item.get("profile_id")) for item in profile_results]
    if mode == "benchmark" and "unet_tiny_smoke" in profile_ids:
        raise ValueError("unet_tiny_smoke is readiness-only and cannot be included as a strong benchmark baseline")
    required = set(required_primary_profiles_for_task(task_id))
    completed = {str(item.get("profile_id")) for item in profile_results if item.get("status") == "completed"}
    performance_complete = mode == "benchmark" and required.issubset(completed)
    if mode == "pilot":
        evidence_scope = "capped_decision_support_only"
        metric_interpretation = "decision_support_not_benchmark_performance"
    elif mode == "benchmark" and performance_complete:
        evidence_scope = "benchmark_performance"
        metric_interpretation = "benchmark_performance"
    elif mode == "benchmark":
        evidence_scope = "blocked_or_incomplete_benchmark"
        metric_interpretation = "sanity_only_not_benchmark_performance"
    else:
        evidence_scope = "smoke_feasibility_only"
        metric_interpretation = "sanity_only_not_benchmark_performance"
    return {
        "schema_version": "pdebench_image128_comparison_summary_v1",
        "task_id": task_id,
        "mode": mode,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(Path(output_root)),
        "profile_results": profile_results,
        "required_primary_profiles": sorted(required),
        "performance_assessment_complete": bool(performance_complete),
        "evidence_scope": evidence_scope,
        "metric_interpretation": metric_interpretation,
        "literature_context": darcy_literature_context() if task_id == "darcy" else None,
        "blockers": blockers or [],
    }


def write_comparison_summary(payload: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "comparison_summary.json"
    csv_path = output_root / "comparison_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = payload.get("profile_results", [])
    fields = [
        "profile_id",
        "status",
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
        "parameter_count",
        "blocker_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return json_path, csv_path


def _expand_reference_rows(
    rows_by_budget: dict[str, list[dict[str, Any]]] | None,
    *,
    dataset_file: str,
    split_counts: dict[str, int],
    max_windows_per_trajectory: int,
    history_len: int,
    training_loss: str,
    batch_size: int,
    metric_family: list[str],
) -> dict[str, list[dict[str, Any]]]:
    expanded: dict[str, list[dict[str, Any]]] = {}
    for budget_label, rows in (rows_by_budget or {}).items():
        expanded[str(budget_label)] = []
        for row in rows:
            expanded[str(budget_label)].append(
                {
                    "run_root": str(row["run_root"]),
                    "profile_id": str(row["profile_id"]),
                    "epochs": int(row["epochs"]),
                    "dataset_file": str(row.get("dataset_file", dataset_file)),
                    "split_counts": dict(row.get("split_counts", split_counts)),
                    "max_windows_per_trajectory": int(
                        row.get("max_windows_per_trajectory", max_windows_per_trajectory)
                    ),
                    "history_len": int(row.get("history_len", history_len)),
                    "training_loss": str(row.get("training_loss", training_loss)),
                    "batch_size": int(row.get("batch_size", batch_size)),
                    "metric_family": list(row.get("metric_family", metric_family)),
                    "source_document": str(row["source_document"]),
                }
            )
    return expanded


def build_reference_run_manifest(
    *,
    task_id: str,
    dataset_file: str,
    split_counts: dict[str, int],
    max_windows_per_trajectory: int,
    history_len: int,
    training_loss: str,
    batch_size: int,
    metric_family: list[str],
    required_rows: dict[str, list[dict[str, Any]]],
    optional_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "pdebench_image128_reference_runs_v2",
        "task_id": str(task_id),
        "dataset_file": str(dataset_file),
        "split_counts": dict(split_counts),
        "max_windows_per_trajectory": int(max_windows_per_trajectory),
        "history_len": int(history_len),
        "training_loss": str(training_loss),
        "batch_size": int(batch_size),
        "metric_family": list(metric_family),
        "required_rows": _expand_reference_rows(
            required_rows,
            dataset_file=dataset_file,
            split_counts=split_counts,
            max_windows_per_trajectory=max_windows_per_trajectory,
            history_len=history_len,
            training_loss=training_loss,
            batch_size=batch_size,
            metric_family=metric_family,
        ),
        "optional_rows": _expand_reference_rows(
            optional_rows,
            dataset_file=dataset_file,
            split_counts=split_counts,
            max_windows_per_trajectory=max_windows_per_trajectory,
            history_len=history_len,
            training_loss=training_loss,
            batch_size=batch_size,
            metric_family=metric_family,
        ),
    }


def write_reference_run_manifest(payload: dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sample_contract(history_len: int) -> str:
    return f"concat u[t-{int(history_len)}:t] -> u[t]"


def _fixed_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {field: contract.get(field) for field in FIXED_CONTRACT_FIELDS}


def _load_run_record(run_root: Path, *, profile_id: str, source_document: str) -> dict[str, Any]:
    required_paths = {name: run_root / name for name in REQUIRED_RUN_ARTIFACTS}
    required_paths["metrics"] = run_root / f"metrics_{profile_id}.json"
    required_paths["model_profile"] = run_root / f"model_profile_{profile_id}.json"
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing required artifact for {profile_id}: {path}")

    invocation = _load_json(required_paths["invocation.json"])
    dataset_manifest = _load_json(required_paths["dataset_manifest.json"])
    split_manifest = _load_json(required_paths["split_manifest.json"])
    comparison_summary = _load_json(required_paths["comparison_summary.json"])
    metrics = _load_json(required_paths["metrics"])
    model_profile = _load_json(required_paths["model_profile"])

    history_len = int(split_manifest.get("history_len", dataset_manifest.get("history_len", 0)))
    metric_family = [field for field in COMPARISON_METRIC_FIELDS if field in metrics]
    field_order = [str(item) for item in dataset_manifest.get("field_order", [])]
    target_channels = int(
        metrics.get("task_metadata", {}).get("target_channels")
        or dataset_manifest.get("target_channels")
        or len(field_order)
    )
    input_channels = int(
        metrics.get("task_metadata", {}).get("input_channels")
        or dataset_manifest.get("input_channels")
        or (len(field_order) * history_len if field_order else 0)
    )
    sample_contract = str(
        dataset_manifest.get("sample_contract")
        or metrics.get("task_metadata", {}).get("sample_contract")
        or _canonical_sample_contract(history_len)
    )
    contract = {
        "dataset_file": str(dataset_manifest.get("data_file") or split_manifest.get("source_file", {}).get("path", "")),
        "split_counts": dict(split_manifest.get("split_counts", {})),
        "max_windows_per_trajectory": int(split_manifest.get("max_windows_per_trajectory")),
        "history_len": history_len,
        "epochs": int(invocation.get("parsed_args", {}).get("epochs")),
        "batch_size": int(invocation.get("parsed_args", {}).get("batch_size")),
        "training_loss": str(metrics.get("training_loss")),
        "metric_family": metric_family,
        "sample_contract": sample_contract,
        "field_order": field_order,
        "input_channels": input_channels,
        "target_channels": target_channels,
    }
    row = {
        "profile_id": profile_id,
        "status": str(metrics.get("status", "completed")),
        "err_RMSE": metrics.get("err_RMSE"),
        "err_nRMSE": metrics.get("err_nRMSE"),
        "relative_l2": metrics.get("relative_l2"),
        "fRMSE_low": metrics.get("fRMSE_low"),
        "fRMSE_mid": metrics.get("fRMSE_mid"),
        "fRMSE_high": metrics.get("fRMSE_high"),
        "parameter_count": model_profile.get(
            "parameter_count",
            metrics.get("parameter_count", metrics.get("model_profile", {}).get("parameter_count")),
        ),
        "run_root": str(run_root),
        "source_document": str(source_document),
        "epochs": contract["epochs"],
        "dataset_file": contract["dataset_file"],
        "split_counts": contract["split_counts"],
        "max_windows_per_trajectory": contract["max_windows_per_trajectory"],
        "history_len": contract["history_len"],
        "training_loss": contract["training_loss"],
        "batch_size": contract["batch_size"],
        "metric_family": contract["metric_family"],
        "sample_contract": contract["sample_contract"],
        "field_order": contract["field_order"],
        "input_channels": contract["input_channels"],
        "target_channels": contract["target_channels"],
        "task_id": str(dataset_manifest.get("task_id", comparison_summary.get("task_id", ""))),
        "mode": str(comparison_summary.get("mode", "")),
        "evidence_scope": str(comparison_summary.get("evidence_scope", "")),
        "metric_interpretation": str(comparison_summary.get("metric_interpretation", "")),
    }
    comparison_npz = run_root / f"comparison_{profile_id}_sample0.npz"
    return {
        "row": row,
        "contract": contract,
        "npz_path": comparison_npz if comparison_npz.exists() else None,
    }


def _assert_contract_matches(
    *,
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key in STRICT_CONTRACT_FIELDS:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{label} contract mismatch for {key}: {actual.get(key)!r} != {expected.get(key)!r}")


def _assert_fixed_contract_matches(
    *,
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key in FIXED_CONTRACT_FIELDS:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{label} contract mismatch for {key}: {actual.get(key)!r} != {expected.get(key)!r}")


def _load_npz_record(record: dict[str, Any]) -> dict[str, Any]:
    path = record.get("npz_path")
    if path is None:
        raise FileNotFoundError(f"missing sample artifact for {record['row']['profile_id']}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "profile_id": record["row"]["profile_id"],
            "prediction": np.asarray(data["prediction"]),
            "target": np.asarray(data["target"]),
            "abs_error": np.asarray(data["abs_error"]),
            "field_order": [str(item) for item in data["field_order"].tolist()],
        }


def _render_gallery(
    *,
    target: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    field_order: list[str],
    output_path: Path,
    is_error: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = int(target.shape[0])
    fig, axes = plt.subplots(
        channels,
        len(panels),
        figsize=(4.2 * len(panels), 3.0 * channels),
        squeeze=False,
        constrained_layout=True,
    )
    for channel in range(channels):
        label = field_order[channel] if channel < len(field_order) else f"c{channel}"
        images = [image[channel] for _, image in panels]
        if is_error:
            spec = cfd_cns_field_visual_spec(label, images, is_error=True)
        else:
            spec = cfd_cns_field_visual_spec(label, [target[channel], *images], is_error=False)
        for col, (title, image) in enumerate(panels):
            axis = axes[channel][col]
            handle = axis.imshow(image[channel], cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
            if channel == 0:
                axis.set_title(title)
            if col == 0:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _maybe_render_cross_run_gallery(
    *,
    records: list[dict[str, Any]],
    output_root: Path,
    epoch_label: str,
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    try:
        npz_records = [_load_npz_record(record) for record in records]
    except FileNotFoundError as exc:
        return None, {"reason": "missing_sample_artifact", "message": str(exc)}

    first = npz_records[0]
    for current in npz_records[1:]:
        if current["field_order"] != first["field_order"]:
            return None, {
                "reason": "field_order_mismatch",
                "message": (
                    f"field_order mismatch between {first['profile_id']} and {current['profile_id']}: "
                    f"{first['field_order']} != {current['field_order']}"
                ),
            }
        if not np.allclose(current["target"], first["target"], atol=1e-6, rtol=1e-6):
            return None, {
                "reason": "target_mismatch",
                "message": f"target mismatch between {first['profile_id']} and {current['profile_id']}",
            }

    prediction_panels = [("Ground truth", first["target"])] + [
        (record["profile_id"], record["prediction"]) for record in npz_records
    ]
    error_panels = [(record["profile_id"], record["abs_error"]) for record in npz_records]
    prediction_path = _render_gallery(
        target=first["target"],
        panels=prediction_panels,
        field_order=first["field_order"],
        output_path=output_root / f"compare_{epoch_label}_sample0.png",
        is_error=False,
    )
    error_path = _render_gallery(
        target=first["target"],
        panels=error_panels,
        field_order=first["field_order"],
        output_path=output_root / f"compare_{epoch_label}_sample0_error.png",
        is_error=True,
    )
    return {
        "prediction_gallery": str(prediction_path),
        "error_gallery": str(error_path),
    }, None


def write_cross_run_compare(
    *,
    output_root: Path,
    epoch_label: str,
    expected_epochs: int,
    fresh_run_root: Path | None = None,
    fresh_profile_id: str | None = None,
    required_reference_rows: list[dict[str, Any]],
    optional_reference_rows: list[dict[str, Any]] | None = None,
    author_run_root: Path | None = None,
    author_profile_id: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if fresh_run_root is None:
        fresh_run_root = author_run_root
    elif author_run_root is not None and Path(author_run_root) != Path(fresh_run_root):
        raise ValueError("fresh_run_root and author_run_root must match when both are provided")
    if fresh_profile_id is None:
        fresh_profile_id = author_profile_id
    elif author_profile_id is not None and str(author_profile_id) != str(fresh_profile_id):
        raise ValueError("fresh_profile_id and author_profile_id must match when both are provided")
    if fresh_run_root is None or fresh_profile_id is None:
        raise ValueError("write_cross_run_compare requires fresh_run_root and fresh_profile_id")

    required_reference_rows = list(required_reference_rows)
    if not required_reference_rows:
        raise ValueError("required_reference_rows must not be empty")
    expected_contract = {
        "dataset_file": str(required_reference_rows[0]["dataset_file"]),
        "split_counts": dict(required_reference_rows[0]["split_counts"]),
        "max_windows_per_trajectory": int(required_reference_rows[0]["max_windows_per_trajectory"]),
        "history_len": int(required_reference_rows[0]["history_len"]),
        "epochs": int(expected_epochs),
        "batch_size": int(required_reference_rows[0]["batch_size"]),
        "training_loss": str(required_reference_rows[0]["training_loss"]),
        "metric_family": list(required_reference_rows[0]["metric_family"]),
    }

    fresh_record = _load_run_record(
        Path(fresh_run_root),
        profile_id=str(fresh_profile_id),
        source_document="fresh_run",
    )
    _assert_contract_matches(
        label=f"fresh row {fresh_profile_id}",
        actual=fresh_record["contract"],
        expected=expected_contract,
    )

    selected_records = [fresh_record]
    included_required_rows: list[dict[str, Any]] = []
    for row in required_reference_rows:
        record = _load_run_record(Path(row["run_root"]), profile_id=str(row["profile_id"]), source_document=str(row["source_document"]))
        _assert_contract_matches(
            label=f"reference row {row['profile_id']}",
            actual=record["contract"],
            expected={
                "dataset_file": str(row["dataset_file"]),
                "split_counts": dict(row["split_counts"]),
                "max_windows_per_trajectory": int(row["max_windows_per_trajectory"]),
                "history_len": int(row["history_len"]),
                "epochs": int(row["epochs"]),
                "batch_size": int(row["batch_size"]),
                "training_loss": str(row["training_loss"]),
                "metric_family": list(row["metric_family"]),
            },
        )
        _assert_contract_matches(
            label=f"reference row {row['profile_id']} vs fresh row",
            actual=record["contract"],
            expected=expected_contract,
        )
        selected_records.append(record)
        included_required_rows.append(dict(row))

    skipped_optional_rows: list[dict[str, Any]] = []
    included_optional_rows: list[dict[str, Any]] = []
    for row in optional_reference_rows or []:
        try:
            record = _load_run_record(
                Path(row["run_root"]),
                profile_id=str(row["profile_id"]),
                source_document=str(row["source_document"]),
            )
            _assert_contract_matches(
                label=f"optional row {row['profile_id']}",
                actual=record["contract"],
                expected={
                    "dataset_file": str(row["dataset_file"]),
                    "split_counts": dict(row["split_counts"]),
                    "max_windows_per_trajectory": int(row["max_windows_per_trajectory"]),
                    "history_len": int(row["history_len"]),
                    "epochs": int(row["epochs"]),
                    "batch_size": int(row["batch_size"]),
                    "training_loss": str(row["training_loss"]),
                    "metric_family": list(row["metric_family"]),
                },
            )
            _assert_contract_matches(
                label=f"optional row {row['profile_id']} vs fresh row",
                actual=record["contract"],
                expected=expected_contract,
            )
        except (FileNotFoundError, ValueError) as exc:
            skipped_optional_rows.append(
                {
                    "profile_id": str(row["profile_id"]),
                    "run_root": str(row["run_root"]),
                    "reason": "contract_or_artifact_mismatch",
                    "message": str(exc),
                }
            )
            continue
        selected_records.append(record)
        included_optional_rows.append(dict(row))

    gallery_artifacts, gallery_blocker = _maybe_render_cross_run_gallery(
        records=selected_records,
        output_root=output_root,
        epoch_label=epoch_label,
    )

    profile_results = [record["row"] for record in selected_records]
    payload = {
        "schema_version": "pdebench_image128_cross_run_compare_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "epoch_label": str(epoch_label),
        "expected_epochs": int(expected_epochs),
        "fresh_profile_id": str(fresh_profile_id),
        "fresh_run_root": str(fresh_run_root),
        "contract": expected_contract,
        "profile_results": profile_results,
        "included_required_reference_rows": included_required_rows,
        "included_optional_reference_rows": included_optional_rows,
        "skipped_optional_reference_rows": skipped_optional_rows,
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }

    json_path = output_root / f"compare_{epoch_label}_against_existing.json"
    csv_path = output_root / f"compare_{epoch_label}_against_existing.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "profile_id",
        "status",
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
        "parameter_count",
        "run_root",
        "source_document",
        "epochs",
        "training_loss",
        "batch_size",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in profile_results:
            writer.writerow({field: row.get(field, "") for field in fields})
    return json_path, csv_path, payload


def write_history_delta_compare(
    *,
    output_root: Path,
    epoch_label: str,
    fresh_run_root: Path,
    fresh_profile_ids: list[str],
    reference_rows: list[dict[str, Any]],
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fresh_profile_ids = [str(profile_id) for profile_id in fresh_profile_ids]
    if not fresh_profile_ids:
        raise ValueError("fresh_profile_ids must not be empty")

    reference_rows = list(reference_rows)
    if not reference_rows:
        raise ValueError("reference_rows must not be empty")

    reference_rows_by_profile: dict[str, dict[str, Any]] = {}
    for row in reference_rows:
        profile_id = str(row["profile_id"])
        if profile_id in reference_rows_by_profile:
            raise ValueError(f"duplicate reference profile_id {profile_id!r}")
        reference_rows_by_profile[profile_id] = row

    fresh_profile_set = set(fresh_profile_ids)
    reference_profile_set = set(reference_rows_by_profile)
    if fresh_profile_set != reference_profile_set:
        raise ValueError(
            "fresh/reference profile_id set mismatch: "
            f"fresh={sorted(fresh_profile_set)} reference={sorted(reference_profile_set)}"
        )

    first_reference_row = reference_rows_by_profile[fresh_profile_ids[0]]
    fixed_contract = _fixed_contract(first_reference_row)
    reference_history_len = int(first_reference_row["history_len"])

    fresh_records: list[dict[str, Any]] = []
    frozen_records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for profile_id in fresh_profile_ids:
        reference_row = reference_rows_by_profile[profile_id]
        reference_expected_contract = {
            "dataset_file": str(reference_row["dataset_file"]),
            "split_counts": dict(reference_row["split_counts"]),
            "max_windows_per_trajectory": int(reference_row["max_windows_per_trajectory"]),
            "history_len": int(reference_row["history_len"]),
            "epochs": int(reference_row["epochs"]),
            "batch_size": int(reference_row["batch_size"]),
            "training_loss": str(reference_row["training_loss"]),
            "metric_family": list(reference_row["metric_family"]),
        }
        frozen_record = _load_run_record(
            Path(reference_row["run_root"]),
            profile_id=profile_id,
            source_document=str(reference_row["source_document"]),
        )
        _assert_contract_matches(
            label=f"reference row {profile_id}",
            actual=frozen_record["contract"],
            expected=reference_expected_contract,
        )
        _assert_fixed_contract_matches(
            label=f"reference row {profile_id}",
            actual=frozen_record["contract"],
            expected=fixed_contract,
        )

        fresh_record = _load_run_record(
            Path(fresh_run_root),
            profile_id=profile_id,
            source_document="fresh_history1_run",
        )
        _assert_fixed_contract_matches(
            label=f"fresh row {profile_id}",
            actual=fresh_record["contract"],
            expected=fixed_contract,
        )
        if int(fresh_record["contract"]["history_len"]) == int(frozen_record["contract"]["history_len"]):
            raise ValueError(f"fresh row {profile_id} must differ from the frozen row on history_len")
        if fresh_record["contract"]["field_order"] != frozen_record["contract"]["field_order"]:
            raise ValueError(
                f"history compare requires matching field_order for {profile_id}: "
                f"{fresh_record['contract']['field_order']!r} != {frozen_record['contract']['field_order']!r}"
            )
        if int(fresh_record["contract"]["target_channels"]) != int(frozen_record["contract"]["target_channels"]):
            raise ValueError(
                f"history compare requires matching target_channels for {profile_id}: "
                f"{fresh_record['contract']['target_channels']!r} != {frozen_record['contract']['target_channels']!r}"
            )

        fresh_records.append(fresh_record)
        frozen_records.append(frozen_record)
        comparisons.append(
            {
                "profile_id": profile_id,
                "fresh_history_len": int(fresh_record["contract"]["history_len"]),
                "reference_history_len": int(frozen_record["contract"]["history_len"]),
                "fresh_sample_contract": str(fresh_record["contract"]["sample_contract"]),
                "reference_sample_contract": str(frozen_record["contract"]["sample_contract"]),
                "fresh_input_channels": int(fresh_record["contract"]["input_channels"]),
                "reference_input_channels": int(frozen_record["contract"]["input_channels"]),
                "target_channels": int(fresh_record["contract"]["target_channels"]),
                "fresh_run_root": str(fresh_record["row"]["run_root"]),
                "reference_run_root": str(frozen_record["row"]["run_root"]),
            }
        )

    fresh_history_lengths = {item["fresh_history_len"] for item in comparisons}
    reference_history_lengths = {item["reference_history_len"] for item in comparisons}
    if len(fresh_history_lengths) != 1 or len(reference_history_lengths) != 1:
        raise ValueError(
            "history compare requires one fresh history_len and one reference history_len: "
            f"fresh={sorted(fresh_history_lengths)} reference={sorted(reference_history_lengths)}"
        )

    fresh_history_len = next(iter(fresh_history_lengths))
    reference_history_len = next(iter(reference_history_lengths))
    target_channels = comparisons[0]["target_channels"]
    fresh_sample_contracts = {item["fresh_sample_contract"] for item in comparisons}
    reference_sample_contracts = {item["reference_sample_contract"] for item in comparisons}
    fresh_input_channels = {item["fresh_input_channels"] for item in comparisons}
    reference_input_channels = {item["reference_input_channels"] for item in comparisons}
    target_channel_counts = {item["target_channels"] for item in comparisons}
    if len(fresh_sample_contracts) != 1 or len(reference_sample_contracts) != 1:
        raise ValueError(
            "history compare requires one fresh/reference sample contract: "
            f"fresh={sorted(fresh_sample_contracts)} reference={sorted(reference_sample_contracts)}"
        )
    if len(fresh_input_channels) != 1 or len(reference_input_channels) != 1:
        raise ValueError(
            "history compare requires one fresh/reference input-channel contract: "
            f"fresh={sorted(fresh_input_channels)} reference={sorted(reference_input_channels)}"
        )
    if len(target_channel_counts) != 1:
        raise ValueError(f"history compare requires one target-channel contract: {sorted(target_channel_counts)}")
    allowed_contract_delta = {
        "delta_kind": "history_len_only",
        "reference_history_len": int(reference_history_len),
        "fresh_history_len": int(fresh_history_len),
        "reference_sample_contract": next(iter(reference_sample_contracts)),
        "fresh_sample_contract": next(iter(fresh_sample_contracts)),
        "reference_input_channels": int(target_channels * reference_history_len),
        "fresh_input_channels": int(target_channels * fresh_history_len),
        "target_channels": int(target_channels),
    }
    if allowed_contract_delta["reference_input_channels"] != int(next(iter(reference_input_channels))):
        raise ValueError(
            "reference input-channel contract mismatch: "
            f"{next(iter(reference_input_channels))!r} != "
            f"{allowed_contract_delta['reference_input_channels']!r}"
        )
    if allowed_contract_delta["fresh_input_channels"] != int(next(iter(fresh_input_channels))):
        raise ValueError(
            "fresh input-channel contract mismatch: "
            f"{next(iter(fresh_input_channels))!r} != "
            f"{allowed_contract_delta['fresh_input_channels']!r}"
        )
    if int(reference_history_len) != int(first_reference_row["history_len"]):
        raise ValueError(
            f"reference history_len mismatch for history compare: {reference_history_len!r} != {first_reference_row['history_len']!r}"
        )
    if fresh_history_len >= reference_history_len:
        raise ValueError(
            f"history compare expects the fresh run to reduce temporal context: {fresh_history_len!r} !< {reference_history_len!r}"
        )

    gallery_records = [*fresh_records, *frozen_records]
    gallery_artifacts, gallery_blocker = _maybe_render_cross_run_gallery(
        records=gallery_records,
        output_root=output_root,
        epoch_label=epoch_label,
    )

    payload = {
        "schema_version": "pdebench_image128_history_delta_compare_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "epoch_label": str(epoch_label),
        "fresh_run_root": str(fresh_run_root),
        "fixed_contract": fixed_contract,
        "allowed_contract_delta": allowed_contract_delta,
        "comparison_standard": "Only history_len and its derived sample/input-channel contract may differ.",
        "evidence_scope": "capped_decision_support_only",
        "metric_interpretation": "decision_support_not_benchmark_performance",
        "fresh_profile_results": [record["row"] for record in fresh_records],
        "reference_profile_results": [record["row"] for record in frozen_records],
        "profile_comparisons": comparisons,
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }

    json_path = output_root / f"compare_{epoch_label}_against_history2.json"
    csv_path = output_root / f"compare_{epoch_label}_against_history2.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "row_family",
        "profile_id",
        "status",
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
        "parameter_count",
        "run_root",
        "source_document",
        "epochs",
        "history_len",
        "sample_contract",
        "input_channels",
        "target_channels",
        "training_loss",
        "batch_size",
        "mode",
        "evidence_scope",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["fresh_profile_results"]:
            writer.writerow({"row_family": "fresh_history1", **{field: row.get(field, "") for field in fields if field != "row_family"}})
        for row in payload["reference_profile_results"]:
            writer.writerow(
                {"row_family": "reference_history2", **{field: row.get(field, "") for field in fields if field != "row_family"}}
            )
    return json_path, csv_path, payload

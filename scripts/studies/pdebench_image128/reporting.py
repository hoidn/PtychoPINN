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
SCALING_INVARIANT_CONTRACT_FIELDS = [
    "dataset_file",
    "max_windows_per_trajectory",
    "epochs",
    "batch_size",
    "training_loss",
    "metric_family",
    "history_len",
    "sample_contract",
    "field_order",
    "input_channels",
    "target_channels",
]
EPOCH_BUDGET_INVARIANT_CONTRACT_FIELDS = [
    "dataset_file",
    "split_counts",
    "max_windows_per_trajectory",
    "history_len",
    "batch_size",
    "training_loss",
    "metric_family",
    "sample_contract",
    "field_order",
    "input_channels",
    "target_channels",
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
    split_dimensions = dict(split_manifest.get("dimensions", {}))
    window_counts = dict(split_manifest.get("window_counts", {}))
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
    raw_windows_per_trajectory = None
    raw_available_windows = None
    if split_dimensions:
        time_steps = split_dimensions.get("time_steps")
        num_trajectories = split_dimensions.get("num_trajectories")
        if time_steps is not None:
            raw_windows_per_trajectory = int(time_steps) - history_len
        if raw_windows_per_trajectory is not None and num_trajectories is not None:
            raw_available_windows = int(num_trajectories) * int(raw_windows_per_trajectory)
    contract = {
        "dataset_file": str(dataset_manifest.get("data_file") or split_manifest.get("source_file", {}).get("path", "")),
        "split_counts": dict(split_manifest.get("split_counts", {})),
        "window_counts": window_counts,
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
        "runtime_sec": metrics.get("runtime_sec"),
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
        "window_counts": contract["window_counts"],
        "max_windows_per_trajectory": contract["max_windows_per_trajectory"],
        "history_len": contract["history_len"],
        "training_loss": contract["training_loss"],
        "batch_size": contract["batch_size"],
        "metric_family": contract["metric_family"],
        "sample_contract": contract["sample_contract"],
        "field_order": contract["field_order"],
        "input_channels": contract["input_channels"],
        "target_channels": contract["target_channels"],
        "raw_windows_per_trajectory": raw_windows_per_trajectory,
        "raw_available_windows": raw_available_windows,
        "task_id": str(dataset_manifest.get("task_id", comparison_summary.get("task_id", ""))),
        "mode": str(comparison_summary.get("mode", "")),
        "evidence_scope": str(comparison_summary.get("evidence_scope", "")),
        "metric_interpretation": str(comparison_summary.get("metric_interpretation", "")),
    }
    comparison_npz = run_root / f"comparison_{profile_id}_sample0.npz"
    return {
        "row": row,
        "contract": contract,
        "metrics": metrics,
        "model_profile": model_profile,
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


def _assert_contract_fields_match(
    *,
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    fields: list[str],
) -> None:
    for key in fields:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{label} contract mismatch for {key}: {actual.get(key)!r} != {expected.get(key)!r}")


def _assert_shell_contract_matches(
    *,
    label: str,
    actual_model_profile: dict[str, Any],
    shell_contract: dict[str, Any],
) -> None:
    for key in ["profile_id", "base_model", "parameter_count"]:
        if actual_model_profile.get(key) != shell_contract.get(key):
            raise ValueError(
                f"{label} shell contract mismatch for {key}: "
                f"{actual_model_profile.get(key)!r} != {shell_contract.get(key)!r}"
            )
    if actual_model_profile.get("profile_config") != shell_contract.get("profile_config"):
        raise ValueError(f"{label} shell contract mismatch for profile_config")


def _assert_scaling_invariant_contract_matches(
    *,
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key in SCALING_INVARIANT_CONTRACT_FIELDS:
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
    compare_stem: str | None = None,
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
    compare_stem = str(compare_stem or f"compare_{epoch_label}")
    prediction_path = _render_gallery(
        target=first["target"],
        panels=prediction_panels,
        field_order=first["field_order"],
        output_path=output_root / f"{compare_stem}_sample0.png",
        is_error=False,
    )
    error_path = _render_gallery(
        target=first["target"],
        panels=error_panels,
        field_order=first["field_order"],
        output_path=output_root / f"{compare_stem}_sample0_error.png",
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
    fresh_profile_ids: list[str] | None = None,
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
        if not fresh_profile_ids:
            raise ValueError("write_cross_run_compare requires fresh_run_root and a fresh profile selection")

    normalized_fresh_profile_ids: list[str] = []
    if fresh_profile_ids is not None:
        normalized_fresh_profile_ids = [str(profile_id) for profile_id in fresh_profile_ids]
        if not normalized_fresh_profile_ids:
            raise ValueError("fresh_profile_ids must not be empty when provided")
        if fresh_profile_id is not None:
            raise ValueError("fresh_profile_id and fresh_profile_ids are mutually exclusive")
    elif fresh_profile_id is not None:
        normalized_fresh_profile_ids = [str(fresh_profile_id)]
    else:
        raise ValueError("write_cross_run_compare requires fresh_run_root and a fresh profile selection")

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

    fresh_records: list[dict[str, Any]] = []
    for profile_id in normalized_fresh_profile_ids:
        fresh_record = _load_run_record(
            Path(fresh_run_root),
            profile_id=profile_id,
            source_document="fresh_run",
        )
        _assert_contract_matches(
            label=f"fresh row {profile_id}",
            actual=fresh_record["contract"],
            expected=expected_contract,
        )
        fresh_records.append(fresh_record)

    selected_records = list(fresh_records)
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
        "fresh_run_root": str(fresh_run_root),
        "contract": expected_contract,
        "profile_results": profile_results,
        "included_required_reference_rows": included_required_rows,
        "included_optional_reference_rows": included_optional_rows,
        "skipped_optional_reference_rows": skipped_optional_rows,
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }
    if len(normalized_fresh_profile_ids) == 1:
        payload["fresh_profile_id"] = normalized_fresh_profile_ids[0]
    else:
        payload["fresh_profile_ids"] = normalized_fresh_profile_ids

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
            source_document="fresh_history_run",
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
    if fresh_history_len > reference_history_len:
        delta_direction = "increase_temporal_context"
    else:
        delta_direction = "reduce_temporal_context"
    fresh_row_family = f"fresh_history{fresh_history_len}"
    reference_row_family = f"reference_history{reference_history_len}"
    compare_stem = f"compare_{epoch_label}_history{fresh_history_len}_against_history{reference_history_len}"

    gallery_records = [*fresh_records, *frozen_records]
    gallery_artifacts, gallery_blocker = _maybe_render_cross_run_gallery(
        records=gallery_records,
        output_root=output_root,
        epoch_label=epoch_label,
        compare_stem=compare_stem,
    )

    payload = {
        "schema_version": "pdebench_image128_history_delta_compare_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "epoch_label": str(epoch_label),
        "fresh_run_root": str(fresh_run_root),
        "fixed_contract": fixed_contract,
        "allowed_contract_delta": {
            **allowed_contract_delta,
            "delta_direction": delta_direction,
        },
        "comparison_standard": "Only history_len and its derived sample/input-channel contract may differ.",
        "evidence_scope": "capped_decision_support_only",
        "metric_interpretation": "decision_support_not_benchmark_performance",
        "row_family_labels": {
            "fresh": fresh_row_family,
            "reference": reference_row_family,
        },
        "fresh_profile_results": [record["row"] for record in fresh_records],
        "reference_profile_results": [record["row"] for record in frozen_records],
        "profile_comparisons": comparisons,
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }

    json_path = output_root / f"{compare_stem}.json"
    csv_path = output_root / f"{compare_stem}.csv"
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
            writer.writerow(
                {"row_family": fresh_row_family, **{field: row.get(field, "") for field in fields if field != "row_family"}}
            )
        for row in payload["reference_profile_results"]:
            writer.writerow(
                {
                    "row_family": reference_row_family,
                    **{field: row.get(field, "") for field in fields if field != "row_family"},
                }
            )
    return json_path, csv_path, payload


def _first_regressed_metric(metric_deltas: dict[str, float]) -> str | None:
    for field in COMPARISON_METRIC_FIELDS:
        if float(metric_deltas[field]) > 0.0:
            return field
    return None


def write_same_profile_multi_reference_history_compare(
    *,
    output_root: Path,
    epoch_label: str,
    fresh_run_root: Path,
    profile_id: str,
    manuscript_label: str,
    reference_rows: list[dict[str, Any]],
    claim_scope: str = "adjacent_capped_context_only",
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile_id = str(profile_id)
    manuscript_label = str(manuscript_label)
    reference_rows = list(reference_rows)
    if not reference_rows:
        raise ValueError("reference_rows must not be empty")

    fixed_contract = _fixed_contract(reference_rows[0])
    fresh_record = _load_run_record(
        Path(fresh_run_root),
        profile_id=profile_id,
        source_document="fresh_history_run",
    )
    _assert_fixed_contract_matches(
        label=f"fresh row {profile_id}",
        actual=fresh_record["contract"],
        expected=fixed_contract,
    )

    reference_records: list[dict[str, Any]] = []
    reference_comparisons: list[dict[str, Any]] = []
    reference_history_lens: list[int] = []
    reference_profile_results: list[dict[str, Any]] = []

    for row in sorted(reference_rows, key=lambda item: int(item["history_len"])):
        if str(row["profile_id"]) != profile_id:
            raise ValueError(
                f"reference row profile_id mismatch for multi-reference history compare: "
                f"{row['profile_id']!r} != {profile_id!r}"
            )
        expected_contract = {
            "dataset_file": str(row["dataset_file"]),
            "split_counts": dict(row["split_counts"]),
            "max_windows_per_trajectory": int(row["max_windows_per_trajectory"]),
            "history_len": int(row["history_len"]),
            "epochs": int(row["epochs"]),
            "batch_size": int(row["batch_size"]),
            "training_loss": str(row["training_loss"]),
            "metric_family": list(row["metric_family"]),
        }
        record = _load_run_record(
            Path(row["run_root"]),
            profile_id=profile_id,
            source_document=str(row["source_document"]),
        )
        _assert_contract_matches(
            label=f"reference row {profile_id} history{row['history_len']}",
            actual=record["contract"],
            expected=expected_contract,
        )
        _assert_fixed_contract_matches(
            label=f"reference row {profile_id} history{row['history_len']}",
            actual=record["contract"],
            expected=fixed_contract,
        )
        if int(record["contract"]["history_len"]) == int(fresh_record["contract"]["history_len"]):
            raise ValueError(f"reference row {profile_id} must differ from the fresh row on history_len")
        if record["contract"]["field_order"] != fresh_record["contract"]["field_order"]:
            raise ValueError(
                f"multi-reference history compare requires matching field_order for {profile_id}: "
                f"{record['contract']['field_order']!r} != {fresh_record['contract']['field_order']!r}"
            )
        if int(record["contract"]["target_channels"]) != int(fresh_record["contract"]["target_channels"]):
            raise ValueError(
                f"multi-reference history compare requires matching target_channels for {profile_id}: "
                f"{record['contract']['target_channels']!r} != {fresh_record['contract']['target_channels']!r}"
            )

        reference_history_len = int(record["contract"]["history_len"])
        reference_label = f"history{reference_history_len}"
        reference_history_lens.append(reference_history_len)
        reference_records.append(record)
        reference_profile_results.append({**record["row"], "manuscript_label": manuscript_label, "claim_scope": claim_scope})

        metric_deltas = {
            field: _rounded(float(fresh_record["row"][field]) - float(record["row"][field]))
            for field in COMPARISON_METRIC_FIELDS
        }
        metric_deltas["runtime_sec"] = _rounded(
            float(fresh_record["row"]["runtime_sec"]) - float(record["row"]["runtime_sec"])
        )
        reference_comparisons.append(
            {
                "reference_history_label": reference_label,
                "reference_history_len": reference_history_len,
                "metric_deltas": metric_deltas,
                "first_regressed_metric": _first_regressed_metric(metric_deltas),
                "reference_profile_result": {
                    **record["row"],
                    "manuscript_label": manuscript_label,
                    "claim_scope": claim_scope,
                },
            }
        )

    if len(set(reference_history_lens)) != len(reference_history_lens):
        raise ValueError(f"duplicate reference history_len values are not allowed: {reference_history_lens!r}")

    fresh_history_len = int(fresh_record["contract"]["history_len"])
    target_channels = int(fresh_record["contract"]["target_channels"])
    reference_history_labels = [f"history{value}" for value in reference_history_lens]
    compare_stem = (
        f"compare_{epoch_label}_history{fresh_history_len}_against_"
        + "_".join(reference_history_labels)
    )
    gallery_artifacts, gallery_blocker = _maybe_render_cross_run_gallery(
        records=[fresh_record, *reference_records],
        output_root=output_root,
        epoch_label=epoch_label,
        compare_stem=compare_stem,
    )

    payload = {
        "schema_version": "pdebench_image128_same_profile_multi_reference_history_compare_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "epoch_label": str(epoch_label),
        "profile_id": profile_id,
        "manuscript_label": manuscript_label,
        "claim_scope": claim_scope,
        "fresh_run_root": str(fresh_run_root),
        "fixed_contract": fixed_contract,
        "allowed_contract_delta": {
            "delta_kind": "history_len_only",
            "fresh_history_len": fresh_history_len,
            "reference_history_lens": reference_history_lens,
            "fresh_sample_contract": str(fresh_record["contract"]["sample_contract"]),
            "reference_sample_contracts": {
                f"history{int(record['contract']['history_len'])}": str(record["contract"]["sample_contract"])
                for record in reference_records
            },
            "fresh_input_channels": int(fresh_record["contract"]["input_channels"]),
            "reference_input_channels": {
                f"history{int(record['contract']['history_len'])}": int(record["contract"]["input_channels"])
                for record in reference_records
            },
            "target_channels": target_channels,
        },
        "comparison_standard": "Only history_len and its derived sample/input-channel contract may differ.",
        "evidence_scope": str(fresh_record["row"]["evidence_scope"]),
        "metric_interpretation": str(fresh_record["row"]["metric_interpretation"]),
        "fresh_profile_result": {
            **fresh_record["row"],
            "manuscript_label": manuscript_label,
            "claim_scope": claim_scope,
        },
        "reference_profile_results": reference_profile_results,
        "reference_comparisons": reference_comparisons,
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }

    json_path = output_root / f"{compare_stem}.json"
    csv_path = output_root / f"{compare_stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "row_family",
        "profile_id",
        "manuscript_label",
        "claim_scope",
        "status",
        "history_len",
        "sample_contract",
        "input_channels",
        "target_channels",
        "split_counts",
        "window_counts",
        "raw_windows_per_trajectory",
        "raw_available_windows",
        "epochs",
        "runtime_sec",
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
        "parameter_count",
        "run_root",
        "source_document",
        "training_loss",
        "batch_size",
        "evidence_scope",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "row_family": f"fresh_history{fresh_history_len}",
                **{field: payload["fresh_profile_result"].get(field, "") for field in fields if field != "row_family"},
            }
        )
        for record in payload["reference_profile_results"]:
            writer.writerow(
                {
                    "row_family": f"reference_history{int(record['history_len'])}",
                    **{field: record.get(field, "") for field in fields if field != "row_family"},
                }
            )
        for comparison in reference_comparisons:
            writer.writerow(
                {
                    "row_family": f"delta_vs_{comparison['reference_history_label']}",
                    "profile_id": profile_id,
                    "manuscript_label": manuscript_label,
                    "claim_scope": claim_scope,
                    **comparison["metric_deltas"],
                }
            )
    return json_path, csv_path, payload


def _cap_label(split_counts: dict[str, Any]) -> str:
    train_count = int(split_counts.get("train"))
    return f"{train_count}cap"


def _metrics_with_runtime(row: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for field in COMPARISON_METRIC_FIELDS:
        payload[field] = float(row[field])
    runtime_sec = row.get("runtime_sec")
    if runtime_sec is None:
        raise ValueError(f"missing runtime_sec for {row.get('profile_id')}")
    payload["runtime_sec"] = float(runtime_sec)
    return payload


def _rounded_convergence(value: float) -> float:
    return round(float(value), 6)


def write_cfd_cns_convergence_audit(
    *,
    output_root: Path,
    run_root: Path,
    profile_ids: list[str],
    expected_loss_count: int = 80,
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(run_root)

    normalized_profile_ids = [str(profile_id) for profile_id in profile_ids]
    if not normalized_profile_ids:
        raise ValueError("profile_ids must not be empty")

    prev_window = slice(expected_loss_count - 20, expected_loss_count - 10)
    final_window = slice(expected_loss_count - 10, expected_loss_count)
    convergence_rule = {
        "late_window_prev_epoch_range": [prev_window.start + 1, prev_window.stop],
        "late_window_final_epoch_range": [final_window.start + 1, final_window.stop],
        "late_window_ratio_lt": 0.95,
        "last5_delta_lte": -0.001,
        "expected_loss_count": int(expected_loss_count),
    }

    records = [
        _load_run_record(run_root, profile_id=profile_id, source_document="fresh_run")
        for profile_id in normalized_profile_ids
    ]
    first_record = records[0]
    if first_record["row"]["task_id"] != "2d_cfd_cns":
        raise ValueError(f"convergence audit requires 2d_cfd_cns task_id, got {first_record['row']['task_id']!r}")
    fixed_contract = dict(first_record["contract"])
    for record in records[1:]:
        _assert_contract_matches(
            label=f"convergence audit row {record['row']['profile_id']}",
            actual=record["contract"],
            expected=fixed_contract,
        )

    profile_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        metrics = dict(record["metrics"])
        losses = metrics.get("train_epoch_losses")
        if not isinstance(losses, list) or len(losses) != expected_loss_count:
            raise ValueError(
                f"profile {record['row']['profile_id']} expected {expected_loss_count} train_epoch_losses, "
                f"found {0 if not isinstance(losses, list) else len(losses)}"
            )
        loss_values = [float(value) for value in losses]
        late_window_mean_prev = float(np.mean(loss_values[prev_window]))
        if abs(late_window_mean_prev) < 1e-12:
            raise ValueError(f"profile {record['row']['profile_id']} has zero late_window_mean_prev")
        late_window_mean_final = float(np.mean(loss_values[final_window]))
        late_window_ratio = late_window_mean_final / late_window_mean_prev
        last5_delta = loss_values[-1] - loss_values[-6]
        still_materially_improving = bool(
            late_window_ratio < convergence_rule["late_window_ratio_lt"]
            or last5_delta <= convergence_rule["last5_delta_lte"]
        )
        payload_row = {
            "profile_id": record["row"]["profile_id"],
            "loss_count": len(loss_values),
            "late_window_mean_prev": _rounded_convergence(late_window_mean_prev),
            "late_window_mean_final": _rounded_convergence(late_window_mean_final),
            "late_window_ratio": _rounded_convergence(late_window_ratio),
            "last5_delta": _rounded_convergence(last5_delta),
            "still_materially_improving": still_materially_improving,
            "final_eval_metrics": {
                field: metrics[field]
                for field in COMPARISON_METRIC_FIELDS
                if field in metrics
            },
        }
        profile_rows.append(payload_row)
        csv_rows.append(
            {
                "profile_id": payload_row["profile_id"],
                "loss_count": payload_row["loss_count"],
                "late_window_mean_prev": payload_row["late_window_mean_prev"],
                "late_window_mean_final": payload_row["late_window_mean_final"],
                "late_window_ratio": payload_row["late_window_ratio"],
                "last5_delta": payload_row["last5_delta"],
                "still_materially_improving": payload_row["still_materially_improving"],
                **payload_row["final_eval_metrics"],
            }
        )

    payload = {
        "schema_version": "pdebench_image128_cfd_cns_convergence_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "run_root": str(run_root),
        "profile_ids": normalized_profile_ids,
        "fixed_contract": fixed_contract,
        "convergence_rule": convergence_rule,
        "evidence_scope": str(first_record["row"]["evidence_scope"]),
        "metric_interpretation": str(first_record["row"]["metric_interpretation"]),
        "profiles": profile_rows,
    }

    json_path = output_root / "convergence_audit.json"
    csv_path = output_root / "convergence_audit.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "profile_id",
        "loss_count",
        "late_window_mean_prev",
        "late_window_mean_final",
        "late_window_ratio",
        "last5_delta",
        "still_materially_improving",
        *COMPARISON_METRIC_FIELDS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return json_path, csv_path, payload


def _profile_output_label(profile_id: str) -> str:
    for prefix in ["spectral_resnet_bottleneck_", "hybrid_resnet_"]:
        if profile_id.startswith(prefix):
            return profile_id[len(prefix) :]
    return profile_id


def write_same_profile_epoch_budget_delta(
    *,
    output_root: Path,
    reference_manifest_path: Path,
    fresh_run_root: Path,
    profile_id: str,
    fresh_source_document: str,
    shell_contract_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile_id = str(profile_id)
    manifest_payload, rows_by_profile, cap_label = _load_reference_manifest_payload(
        Path(reference_manifest_path),
        profile_ids=[profile_id],
    )
    reference_row = rows_by_profile[profile_id]
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
    reference_record = _load_run_record(
        Path(reference_row["run_root"]),
        profile_id=profile_id,
        source_document=str(reference_row["source_document"]),
    )
    _assert_contract_matches(
        label=f"reference row {profile_id}",
        actual=reference_record["contract"],
        expected=reference_expected_contract,
    )

    shell_contract = _load_json(Path(shell_contract_path))
    _assert_shell_contract_matches(
        label=f"reference row {profile_id}",
        actual_model_profile=reference_record["model_profile"],
        shell_contract=shell_contract,
    )

    fresh_record = _load_run_record(
        Path(fresh_run_root),
        profile_id=profile_id,
        source_document=str(fresh_source_document),
    )
    invariant_contract = {
        key: reference_record["contract"].get(key) for key in EPOCH_BUDGET_INVARIANT_CONTRACT_FIELDS
    }
    _assert_contract_fields_match(
        label=f"fresh row {profile_id}",
        actual=fresh_record["contract"],
        expected=invariant_contract,
        fields=EPOCH_BUDGET_INVARIANT_CONTRACT_FIELDS,
    )

    reference_epochs = int(reference_record["contract"]["epochs"])
    fresh_epochs = int(fresh_record["contract"]["epochs"])
    if fresh_epochs <= reference_epochs:
        raise ValueError(
            f"fresh row {profile_id} must use a longer epoch budget than the reference row: "
            f"{fresh_epochs} !> {reference_epochs}"
        )

    _assert_shell_contract_matches(
        label=f"fresh row {profile_id}",
        actual_model_profile=fresh_record["model_profile"],
        shell_contract=shell_contract,
    )

    metric_deltas = {
        field: _rounded(float(fresh_record["row"][field]) - float(reference_record["row"][field]))
        for field in COMPARISON_METRIC_FIELDS
    }
    metric_deltas["runtime_sec"] = _rounded(
        float(fresh_record["row"]["runtime_sec"]) - float(reference_record["row"]["runtime_sec"])
    )

    gallery_artifacts, gallery_blocker = _maybe_render_cross_run_gallery(
        records=[fresh_record, reference_record],
        output_root=output_root,
        epoch_label=f"{reference_epochs}ep_vs_{fresh_epochs}ep",
    )

    payload = {
        "schema_version": "pdebench_image128_same_profile_epoch_budget_delta_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "profile_id": profile_id,
        "cap_label": cap_label,
        "fixed_contract": invariant_contract,
        "allowed_contract_delta": {
            "delta_kind": "epochs_only",
            "reference_epochs": reference_epochs,
            "fresh_epochs": fresh_epochs,
        },
        "evidence_scope": str(fresh_record["row"]["evidence_scope"]),
        "metric_interpretation": str(fresh_record["row"]["metric_interpretation"]),
        "reference_manifest_path": str(reference_manifest_path),
        "reference_profile_result": reference_record["row"],
        "fresh_profile_result": fresh_record["row"],
        "metric_deltas": metric_deltas,
        "shell_contract": {
            "path": str(shell_contract_path),
            "profile_id": str(shell_contract.get("profile_id")),
            "base_model": str(shell_contract.get("base_model")),
            "parameter_count": shell_contract.get("parameter_count"),
            "source_run_root": str(shell_contract.get("source_run_root")),
            "source_document": str(shell_contract.get("source_document")),
            "profile_config": shell_contract.get("profile_config"),
        },
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }

    stem = f"{_profile_output_label(profile_id)}_{cap_label}_{reference_epochs}ep_vs_{fresh_epochs}ep"
    json_path = output_root / f"{stem}.json"
    csv_path = output_root / f"{stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "row_family",
        "profile_id",
        "status",
        "epochs",
        "runtime_sec",
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
        "parameter_count",
        "run_root",
        "source_document",
        "training_loss",
        "batch_size",
        "evidence_scope",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {"row_family": "reference", **{field: payload["reference_profile_result"].get(field, "") for field in fields if field != "row_family"}}
        )
        writer.writerow(
            {"row_family": "fresh", **{field: payload["fresh_profile_result"].get(field, "") for field in fields if field != "row_family"}}
        )
        writer.writerow({"row_family": "delta", **{field: payload["metric_deltas"].get(field, "") for field in fields if field != "row_family"}})
    return json_path, csv_path, payload


def _rounded(value: float) -> float:
    return round(float(value), 15)


def _load_reference_manifest_payload(
    manifest_path: Path,
    *,
    profile_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    payload = _load_json(Path(manifest_path))
    required_rows = payload.get("required_rows", {})
    if not required_rows:
        raise ValueError(f"reference manifest has no required_rows: {manifest_path}")
    rows: list[dict[str, Any]] = []
    for items in required_rows.values():
        rows.extend(items)
    rows_by_profile = {str(row["profile_id"]): row for row in rows}
    expected_profile_ids = [str(profile_id) for profile_id in profile_ids]
    if sorted(rows_by_profile) != sorted(expected_profile_ids):
        raise ValueError(
            f"reference manifest profile_ids mismatch for {manifest_path}: "
            f"{sorted(rows_by_profile)} != {sorted(expected_profile_ids)}"
        )
    split_counts = dict(payload.get("split_counts", {}))
    if not split_counts:
        raise ValueError(f"reference manifest missing split_counts: {manifest_path}")
    return payload, rows_by_profile, _cap_label(split_counts)


def write_split_cap_scaling_trend(
    *,
    output_root: Path,
    profile_ids: list[str],
    reference_manifest_paths: list[Path],
    fresh_run_root: Path,
    fresh_profile_ids: list[str],
    fresh_source_document: str,
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile_ids = [str(profile_id) for profile_id in profile_ids]
    fresh_profile_ids = [str(profile_id) for profile_id in fresh_profile_ids]
    if not profile_ids or not reference_manifest_paths:
        raise ValueError("write_split_cap_scaling_trend requires profile_ids and reference_manifest_paths")
    if sorted(profile_ids) != sorted(fresh_profile_ids):
        raise ValueError(
            f"fresh_profile_ids mismatch: {sorted(fresh_profile_ids)} != {sorted(profile_ids)}"
        )

    cap_rows: dict[str, dict[str, dict[str, Any]]] = {}
    cap_contracts: dict[str, dict[str, Any]] = {}
    manifest_records: list[str] = []
    invariant_contract: dict[str, Any] | None = None

    for manifest_path in reference_manifest_paths:
        manifest_payload, rows_by_profile, cap_label = _load_reference_manifest_payload(
            Path(manifest_path),
            profile_ids=profile_ids,
        )
        manifest_records.append(str(manifest_path))
        profile_records: dict[str, dict[str, Any]] = {}
        for profile_id in profile_ids:
            row = rows_by_profile[profile_id]
            record = _load_run_record(
                Path(row["run_root"]),
                profile_id=profile_id,
                source_document=str(row["source_document"]),
            )
            expected_contract = {
                "dataset_file": str(row["dataset_file"]),
                "split_counts": dict(row["split_counts"]),
                "max_windows_per_trajectory": int(row["max_windows_per_trajectory"]),
                "history_len": int(row["history_len"]),
                "epochs": int(row["epochs"]),
                "batch_size": int(row["batch_size"]),
                "training_loss": str(row["training_loss"]),
                "metric_family": list(row["metric_family"]),
            }
            _assert_contract_matches(
                label=f"reference row {profile_id} ({cap_label})",
                actual=record["contract"],
                expected=expected_contract,
            )
            profile_records[profile_id] = record
            if invariant_contract is None:
                invariant_contract = {
                    key: record["contract"].get(key) for key in SCALING_INVARIANT_CONTRACT_FIELDS
                }
            else:
                _assert_scaling_invariant_contract_matches(
                    label=f"reference row {profile_id} ({cap_label})",
                    actual=record["contract"],
                    expected=invariant_contract,
                )
        cap_rows[cap_label] = profile_records
        cap_contracts[cap_label] = {
            "split_counts": dict(manifest_payload["split_counts"]),
            "epochs": int(manifest_payload["required_rows"][next(iter(manifest_payload["required_rows"]))][0]["epochs"]),
        }

    fresh_run_root = Path(fresh_run_root)
    fresh_records: dict[str, dict[str, Any]] = {}
    fresh_cap_label: str | None = None
    for profile_id in profile_ids:
        record = _load_run_record(
            fresh_run_root,
            profile_id=profile_id,
            source_document=str(fresh_source_document),
        )
        if invariant_contract is None:
            invariant_contract = {
                key: record["contract"].get(key) for key in SCALING_INVARIANT_CONTRACT_FIELDS
            }
        else:
            _assert_scaling_invariant_contract_matches(
                label=f"fresh row {profile_id}",
                actual=record["contract"],
                expected=invariant_contract,
            )
        current_cap_label = _cap_label(record["contract"]["split_counts"])
        if fresh_cap_label is None:
            fresh_cap_label = current_cap_label
        elif fresh_cap_label != current_cap_label:
            raise ValueError(
                f"fresh run split-count mismatch across profiles: {fresh_cap_label!r} != {current_cap_label!r}"
            )
        fresh_records[profile_id] = record

    if fresh_cap_label is None:
        raise ValueError("unable to determine fresh cap label")
    cap_rows[fresh_cap_label] = fresh_records
    cap_contracts[fresh_cap_label] = {
        "split_counts": dict(next(iter(fresh_records.values()))["contract"]["split_counts"]),
        "epochs": int(next(iter(fresh_records.values()))["contract"]["epochs"]),
    }

    cap_sequence = sorted(cap_rows, key=lambda label: int(cap_contracts[label]["split_counts"]["train"]))
    if len(cap_sequence) < 2:
        raise ValueError("split-cap scaling trend requires at least two cap checkpoints")

    allowed_contract_delta = {
        "delta_kind": "split_counts_only",
        "split_counts_by_cap": {
            label: dict(cap_contracts[label]["split_counts"]) for label in cap_sequence
        },
    }

    profiles_payload: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    gallery_records: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        metrics_by_cap: dict[str, dict[str, float]] = {}
        deltas: dict[str, dict[str, float]] = {}
        runtime_deltas: dict[str, float] = {}
        improvement_per_added: dict[str, dict[str, float]] = {}
        for label in cap_sequence:
            record = cap_rows[label][profile_id]
            metrics_by_cap[label] = _metrics_with_runtime(record["row"])
            gallery_record = {
                **record,
                "row": {
                    **record["row"],
                    "profile_id": f"{profile_id}@{label}",
                },
            }
            gallery_records.append(gallery_record)
            csv_rows.append(
                {
                    "profile_id": profile_id,
                    "cap_label": label,
                    "train_trajectories": int(record["row"]["split_counts"]["train"]),
                    "val_trajectories": int(record["row"]["split_counts"]["val"]),
                    "test_trajectories": int(record["row"]["split_counts"]["test"]),
                    **metrics_by_cap[label],
                }
            )

        for previous, current in zip(cap_sequence, cap_sequence[1:]):
            previous_train = int(cap_contracts[previous]["split_counts"]["train"])
            current_train = int(cap_contracts[current]["split_counts"]["train"])
            delta_key = f"{current_train}_minus_{previous_train}"
            previous_metrics = metrics_by_cap[previous]
            current_metrics = metrics_by_cap[current]
            deltas[delta_key] = {
                field: _rounded(current_metrics[field] - previous_metrics[field])
                for field in COMPARISON_METRIC_FIELDS
            }
            runtime_deltas[delta_key] = _rounded(current_metrics["runtime_sec"] - previous_metrics["runtime_sec"])
            added_training = current_train - previous_train
            if added_training <= 0:
                raise ValueError(f"non-increasing train split between {previous} and {current}")
            improvement_per_added[delta_key] = {
                "err_nRMSE": _rounded(
                    float(previous_metrics["err_nRMSE"] - current_metrics["err_nRMSE"]) / float(added_training)
                ),
                "relative_l2": _rounded(
                    float(previous_metrics["relative_l2"] - current_metrics["relative_l2"]) / float(added_training)
                ),
                "fRMSE_high": _rounded(
                    float(previous_metrics["fRMSE_high"] - current_metrics["fRMSE_high"]) / float(added_training)
                ),
            }

        profile_payload = {
            "profile_id": profile_id,
            "metrics_by_cap": metrics_by_cap,
            "improvement_per_added_training_trajectory": improvement_per_added,
        }
        for delta_key, delta_payload in deltas.items():
            profile_payload[f"delta_{delta_key}"] = delta_payload
            profile_payload[f"runtime_delta_{delta_key}"] = runtime_deltas[delta_key]
        profiles_payload.append(profile_payload)

    gallery_artifacts, gallery_blocker = _maybe_render_cross_run_gallery(
        records=gallery_records,
        output_root=output_root,
        epoch_label="scaling_512_1024_2048",
    )

    payload = {
        "schema_version": "pdebench_image128_split_cap_scaling_trend_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "2d_cfd_cns",
        "profile_ids": profile_ids,
        "cap_sequence": cap_sequence,
        "fixed_contract": invariant_contract,
        "allowed_contract_delta": allowed_contract_delta,
        "evidence_scope": "capped_decision_support_only",
        "metric_interpretation": "decision_support_not_benchmark_performance",
        "reference_manifest_paths": manifest_records,
        "fresh_run_root": str(fresh_run_root),
        "profiles": profiles_payload,
        "gallery_artifacts": gallery_artifacts,
        "cross_run_gallery_blocked": gallery_blocker,
    }

    json_path = output_root / "finalist_scaling_trend_512_1024_2048.json"
    csv_path = output_root / "finalist_scaling_trend_512_1024_2048.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "profile_id",
        "cap_label",
        "train_trajectories",
        "val_trajectories",
        "test_trajectories",
        "runtime_sec",
        *COMPARISON_METRIC_FIELDS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return json_path, csv_path, payload


_CNS_PAPER_TABLE_FIELDS = [
    "row_id",
    "row_role",
    "row_status",
    "claim_scope",
    "split_label",
    "cap_label",
    "training_scope",
    "err_nRMSE",
    "err_RMSE",
    "relative_l2",
    "fRMSE_low",
    "fRMSE_mid",
    "fRMSE_high",
    "parameter_count",
    "runtime_sec",
    "hardware_label",
    "hardware_runtime_note",
    "source_run_root",
]


def _row_split_label(split_counts: dict[str, Any]) -> str:
    return f"{int(split_counts['train'])} / {int(split_counts['val'])} / {int(split_counts['test'])}"


def _row_cap_label(split_counts: dict[str, Any]) -> str:
    return f"{int(split_counts['train'])}_{int(split_counts['val'])}_{int(split_counts['test'])}"


def _hardware_fields_for_locked_row(row: dict[str, Any]) -> tuple[str, str]:
    runtime_provenance = row.get("runtime_provenance", {}) or {}
    accelerator = runtime_provenance.get("accelerator")
    python_executable = runtime_provenance.get("python_executable")
    if isinstance(accelerator, str) and accelerator.strip():
        note_bits = [f"accelerator={accelerator.strip()}"]
        if isinstance(python_executable, str) and python_executable.strip():
            note_bits.append(f"python={python_executable.strip()}")
        return accelerator.strip(), "; ".join(note_bits)
    missing_note = "artifact_missing_precise_accelerator"
    if isinstance(python_executable, str) and python_executable.strip():
        return missing_note, f"{missing_note}; python={python_executable.strip()}"
    return missing_note, missing_note


def build_cns_paper_table_bundle(
    locked_rows_payload: dict[str, Any],
    *,
    authoritative_manifest_path: str,
) -> dict[str, Any]:
    rows = list(locked_rows_payload.get("rows", []))
    if not rows:
        raise ValueError("locked rows payload must contain rows")

    rows_by_id = {str(row["row_id"]): dict(row) for row in rows}
    headline_row_ids = [str(row_id) for row_id in locked_rows_payload.get("headline_row_ids", [])]
    continuity_row_ids = [str(row_id) for row_id in locked_rows_payload.get("continuity_row_ids", [])]
    if not headline_row_ids:
        raise ValueError("locked rows payload must contain headline_row_ids")

    missing_rows = [row_id for row_id in [*headline_row_ids, *continuity_row_ids] if row_id not in rows_by_id]
    if missing_rows:
        raise ValueError(f"locked rows payload missing row entries for: {', '.join(missing_rows)}")

    headline_split_labels = {_row_split_label(rows_by_id[row_id]["split_counts"]) for row_id in headline_row_ids}
    if len(headline_split_labels) != 1:
        raise ValueError(f"mixed-cap headline rows are not allowed: {sorted(headline_split_labels)}")

    normalized_rows: list[dict[str, Any]] = []
    benchmark_incomplete = False
    for row_id in [*headline_row_ids, *continuity_row_ids]:
        row = rows_by_id[row_id]
        metrics = dict(row.get("metrics", {}))
        hardware_label, hardware_runtime_note = _hardware_fields_for_locked_row(row)
        normalized = {
            "row_id": row_id,
            "row_role": str(row.get("row_role", "")),
            "row_status": str(row.get("row_status", "")),
            "claim_scope": str(row.get("claim_scope", "")),
            "split_label": _row_split_label(row["split_counts"]),
            "cap_label": _row_cap_label(row["split_counts"]),
            "training_scope": "capped_decision_support",
            "err_nRMSE": metrics.get("err_nRMSE"),
            "err_RMSE": metrics.get("err_RMSE"),
            "relative_l2": metrics.get("relative_l2"),
            "fRMSE_low": metrics.get("fRMSE_low"),
            "fRMSE_mid": metrics.get("fRMSE_mid"),
            "fRMSE_high": metrics.get("fRMSE_high"),
            "parameter_count": row.get("parameter_count"),
            "runtime_sec": row.get("runtime_sec"),
            "hardware_label": hardware_label,
            "hardware_runtime_note": hardware_runtime_note,
            "source_run_root": str(row.get("run_root", "")),
            "missing_fields": [],
        }
        for field in [
            "err_nRMSE",
            "err_RMSE",
            "relative_l2",
            "fRMSE_low",
            "fRMSE_mid",
            "fRMSE_high",
            "parameter_count",
            "runtime_sec",
            "source_run_root",
        ]:
            value = normalized.get(field)
            if value in (None, ""):
                normalized["missing_fields"].append(field)
        if normalized["row_status"] != "capped_decision_support":
            normalized["missing_fields"].append("row_status_not_capped_decision_support")
        if "paper_grade" in normalized["claim_scope"] or "full_training" in normalized["claim_scope"]:
            normalized["missing_fields"].append("claim_scope_out_of_bounds")
        if normalized["missing_fields"]:
            benchmark_incomplete = True
        normalized_rows.append(normalized)

    return {
        "schema_version": "pdebench_cns_paper_table_bundle_v1",
        "authoritative_locked_rows_path": str(authoritative_manifest_path),
        "contract_authority": str(locked_rows_payload.get("contract_authority", "")),
        "headline_row_ids": headline_row_ids,
        "continuity_row_ids": continuity_row_ids,
        "headline_rows": [row for row in normalized_rows if row["row_role"] == "headline"],
        "continuity_rows": [row for row in normalized_rows if row["row_role"] != "headline"],
        "rows": normalized_rows,
        "benchmark_status": "benchmark_incomplete" if benchmark_incomplete else "paper_complete",
        "claim_boundary": "capped_decision_support_only",
    }


def validate_cns_paper_table_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    table_row_ids = [str(row.get("row_id", "")) for row in rows]
    headline_row_ids = [str(row_id) for row_id in payload.get("headline_row_ids", [])]
    continuity_row_ids = [str(row_id) for row_id in payload.get("continuity_row_ids", [])]
    headline_rows = [row for row in rows if row.get("row_role") == "headline"]
    split_labels = {row.get("split_label") for row in headline_rows}
    mixed_cap = len(split_labels) > 1
    all_rows_capped = all(str(row.get("row_status")) == "capped_decision_support" for row in rows)
    no_paper_grade = all(
        "paper_grade" not in str(row.get("claim_scope", "")) and "full_training" not in str(row.get("claim_scope", ""))
        for row in rows
    )
    return {
        "schema_version": "pdebench_cns_paper_table_validation_v1",
        "headline_contract_consistent": not mixed_cap,
        "mixed_cap_headline_table": mixed_cap,
        "all_rows_capped_decision_support": all_rows_capped,
        "no_paper_grade_or_full_training_labels": no_paper_grade,
        "benchmark_status": str(payload.get("benchmark_status", "")),
        "table_row_ids": table_row_ids,
        "table_headline_row_ids": headline_row_ids,
        "table_continuity_row_ids": continuity_row_ids,
    }


def write_cns_paper_table_bundle(payload: dict[str, Any], output_root: Path) -> tuple[Path, Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "cns_paper_table_rows.json"
    csv_path = output_root / "cns_paper_table_rows.csv"
    tex_path = output_root / "cns_paper_table_rows.tex"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*_CNS_PAPER_TABLE_FIELDS, "missing_fields"])
        writer.writeheader()
        for row in payload.get("rows", []):
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in _CNS_PAPER_TABLE_FIELDS},
                    "missing_fields": ",".join(row.get("missing_fields", [])),
                }
            )

    lines = [
        "% Auto-generated CNS paper table bundle",
        "\\begin{tabular}{lllrrrrrrrl}",
        "Row & Role & Split & nRMSE & RMSE & relL2 & fLow & fMid & fHigh & Params & Runtime \\\\",
        "\\hline",
    ]
    for row in payload.get("rows", []):
        lines.append(
            " & ".join(
                [
                    str(row.get("row_id", "")),
                    str(row.get("row_role", "")),
                    str(row.get("split_label", "")),
                    _format_tex_value(row.get("err_nRMSE")),
                    _format_tex_value(row.get("err_RMSE")),
                    _format_tex_value(row.get("relative_l2")),
                    _format_tex_value(row.get("fRMSE_low")),
                    _format_tex_value(row.get("fRMSE_mid")),
                    _format_tex_value(row.get("fRMSE_high")),
                    _format_tex_value(row.get("parameter_count")),
                    _format_tex_value(row.get("runtime_sec")),
                ]
            )
            + " \\\\"
        )
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, tex_path


def _format_tex_value(value: Any) -> str:
    if value in (None, ""):
        return "\\textit{missing}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)

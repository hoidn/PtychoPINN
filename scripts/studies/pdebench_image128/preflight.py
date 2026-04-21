"""Data/schema preflight for the PDEBench 128x128 image-suite plan."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.studies.pdebench_image128.task_specs import TASK_IDS, TaskSpec, get_task_spec
from scripts.studies.pdebench_swe.manifest import (
    ManifestBlocker,
    file_identity,
    infer_axis_order,
    inspect_hdf5,
    select_state_dataset,
)


SCHEMA_VERSION = "pdebench_image128_suite_preflight_v1"
CFD_CNS_FIELD_ORDER = ("density", "Vx", "Vy", "pressure")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _blocker(task_id: str, reason: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "status": reason,
        "blocker": {
            "reason": reason,
            "message": message,
        },
        **extra,
    }


def _axis_index(axis_order: str, axis: str) -> int | None:
    try:
        return axis_order.index(axis)
    except ValueError:
        return None


def _spatial_shape(shape: Sequence[int], axis_order: str | None) -> list[int] | None:
    if axis_order is None:
        return None
    h_index = _axis_index(axis_order, "H")
    w_index = _axis_index(axis_order, "W")
    if h_index is None or w_index is None:
        return None
    return [int(shape[h_index]), int(shape[w_index])]


def _time_steps(shape: Sequence[int], axis_order: str | None) -> int | None:
    if axis_order is None:
        return None
    t_index = _axis_index(axis_order, "T")
    if t_index is None:
        return None
    return int(shape[t_index])


def _sample_count(shape: Sequence[int], axis_order: str | None) -> int | None:
    if axis_order is None:
        return None
    n_index = _axis_index(axis_order, "N")
    if n_index is None:
        return None
    return int(shape[n_index])


def _infer_static_axis_order(shape: Sequence[int]) -> str | None:
    if len(shape) == 3:
        return "NHW"
    if len(shape) == 4:
        if int(shape[-1]) <= 16:
            return "NHWC"
        if int(shape[1]) <= 16:
            return "NCHW"
    return infer_axis_order([int(dim) for dim in shape])


def _is_numeric_dataset(record: dict[str, Any]) -> bool:
    try:
        dtype = np.dtype(record["dtype"])
    except TypeError:
        return False
    return bool(np.issubdtype(dtype, np.number))


def _name_score(record: dict[str, Any], names: tuple[str, ...]) -> int:
    parts = str(record.get("path", "")).lower().replace("-", "_").split("/")
    basename = parts[-1] if parts else ""
    score = 0
    for index, name in enumerate(names):
        if basename == name:
            score += 100 - index
        elif name in basename:
            score += 50 - index
    return score


def _select_static_operator_pair(metadata: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for record in metadata.get("datasets", []):
        if not _is_numeric_dataset(record):
            continue
        axis_order = _infer_static_axis_order(record["shape"])
        spatial_shape = _spatial_shape(record["shape"], axis_order)
        if spatial_shape is None:
            continue
        enriched = dict(record)
        enriched["axis_order"] = axis_order
        enriched["spatial_shape"] = spatial_shape
        candidates.append(enriched)

    if len(candidates) < 2:
        raise ManifestBlocker(
            "missing_static_operator_pair",
            "static operator task requires at least two numeric image datasets",
            candidate_datasets=[str(item["path"]) for item in candidates],
        )

    input_names = ("a", "input", "inputs", "coeff", "coefficient", "nu", "permeability", "x")
    target_names = ("u", "tensor", "target", "targets", "solution", "sol", "y")
    input_dataset = max(candidates, key=lambda item: (_name_score(item, input_names), -len(str(item["path"]))))
    target_candidates = [item for item in candidates if item["path"] != input_dataset["path"]]
    target_dataset = max(
        target_candidates,
        key=lambda item: (_name_score(item, target_names), -len(str(item["path"]))),
    )

    if _name_score(input_dataset, input_names) <= 0 or _name_score(target_dataset, target_names) <= 0:
        raise ManifestBlocker(
            "ambiguous_static_operator_pair",
            "could not select input and target datasets by name; task needs explicit schema mapping",
            candidate_datasets=[str(item["path"]) for item in candidates],
        )

    return {
        "kind": "static_operator",
        "input_dataset": input_dataset,
        "target_dataset": target_dataset,
    }


def _dynamic_layout(metadata: dict[str, Any]) -> dict[str, Any]:
    selected = select_state_dataset(metadata)
    return {
        "kind": "dynamic_state",
        "state_dataset": selected,
    }


def _select_cfd_cns_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    records_by_name = {str(record.get("path", "")).split("/")[-1].lower(): record for record in metadata.get("datasets", [])}
    fields: list[dict[str, Any]] = []
    missing: list[str] = []
    for field_name in CFD_CNS_FIELD_ORDER:
        record = records_by_name.get(field_name.lower())
        if record is None or not _is_numeric_dataset(record):
            missing.append(field_name)
            continue
        axis_order = "NTHW" if len(record["shape"]) == 4 else infer_axis_order([int(dim) for dim in record["shape"]])
        spatial_shape = _spatial_shape(record["shape"], axis_order)
        enriched = dict(record)
        enriched["field_name"] = field_name
        enriched["axis_order"] = axis_order
        enriched["spatial_shape"] = spatial_shape
        fields.append(enriched)
    if missing:
        raise ManifestBlocker(
            "missing_cfd_cns_fields",
            f"2D CFD CNS file is missing required fields: {missing}",
            candidate_datasets=[str(item["path"]) for item in metadata.get("datasets", [])],
        )
    shapes = {tuple(field["shape"]) for field in fields}
    axis_orders = {field.get("axis_order") for field in fields}
    if len(shapes) != 1 or len(axis_orders) != 1:
        raise ManifestBlocker(
            "inconsistent_cfd_cns_field_shapes",
            "2D CFD CNS fields must share one shape and axis order",
            candidate_datasets=[str(field["path"]) for field in fields],
        )
    return {
        "kind": "dynamic_multifield",
        "field_order": list(CFD_CNS_FIELD_ORDER),
        "fields": fields,
    }


def _layout_for_task(spec: TaskSpec, metadata: dict[str, Any]) -> dict[str, Any]:
    if spec.task_type == "static_operator":
        return _select_static_operator_pair(metadata)
    if spec.task_id == "2d_cfd_cns":
        return _select_cfd_cns_fields(metadata)
    return _dynamic_layout(metadata)


def _layout_spatial_shape(layout: dict[str, Any]) -> list[int] | None:
    if layout["kind"] == "static_operator":
        input_shape = layout["input_dataset"].get("spatial_shape")
        target_shape = layout["target_dataset"].get("spatial_shape")
        return input_shape if input_shape == target_shape else None
    if layout["kind"] == "dynamic_multifield":
        shapes = {tuple(field.get("spatial_shape") or []) for field in layout["fields"]}
        if len(shapes) == 1:
            shape = next(iter(shapes))
            return list(shape) if shape else None
        return None
    state = layout["state_dataset"]
    return _spatial_shape(state["shape"], state.get("axis_order"))


def _available_supervision_units(layout: dict[str, Any], *, dynamic_history_len: int = 1) -> int | None:
    if layout["kind"] == "static_operator":
        dataset = layout["target_dataset"]
        return _sample_count(dataset["shape"], dataset.get("axis_order"))
    if layout["kind"] == "dynamic_multifield":
        first = layout["fields"][0]
        sample_count = _sample_count(first["shape"], first.get("axis_order"))
        time_steps = _time_steps(first["shape"], first.get("axis_order"))
        if sample_count is None or time_steps is None:
            return None
        return int(sample_count * max(0, time_steps - int(dynamic_history_len)))
    state = layout["state_dataset"]
    sample_count = _sample_count(state["shape"], state.get("axis_order"))
    time_steps = _time_steps(state["shape"], state.get("axis_order"))
    if sample_count is None or time_steps is None:
        return None
    return int(sample_count * max(0, time_steps - int(dynamic_history_len)))


def inspect_task_file(
    *,
    task_id: str,
    data_file: Path,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Inspect one task file and return a preflight payload."""
    spec = get_task_spec(task_id)
    data_file = Path(data_file).expanduser()
    if not data_file.exists():
        return _blocker(
            task_id,
            "missing_file",
            f"expected PDEBench file is missing: {data_file}",
            expected_filename=spec.expected_filename,
            data_file=str(data_file),
        )

    try:
        identity = file_identity(data_file, sha256=compute_sha256)
        metadata = inspect_hdf5(data_file)
        layout = _layout_for_task(spec, metadata)
    except (OSError, KeyError, ManifestBlocker, ValueError) as exc:
        candidate_datasets = []
        if isinstance(exc, ManifestBlocker):
            candidate_datasets = exc.candidate_datasets
        return _blocker(
            task_id,
            "schema_blocker",
            str(exc),
            expected_filename=spec.expected_filename,
            data_file=str(data_file),
            candidate_datasets=candidate_datasets,
        )

    spatial_shape = _layout_spatial_shape(layout)
    expected_shape = [int(item) for item in spec.expected_spatial_shape]
    is_native_128 = spatial_shape == expected_shape
    status = "ready" if is_native_128 else "non_native_spatial_shape"
    dynamic_history_len = 2 if spec.task_id == "2d_cfd_cns" else 1
    payload: dict[str, Any] = {
        "task_id": task_id,
        "pde_name": spec.pde_name,
        "title": spec.title,
        "task_type": spec.task_type,
        "status": status,
        "expected_filename": spec.expected_filename,
        "expected_darus_id": spec.expected_darus_id,
        "data_file": str(data_file),
        "file_identity": identity,
        "hdf5_dataset_count": len(metadata.get("datasets", [])),
        "layout": layout,
        "native_spatial_shape": spatial_shape,
        "expected_spatial_shape": expected_shape,
        "is_native_128": is_native_128,
        "available_supervision_units": _available_supervision_units(
            layout,
            dynamic_history_len=dynamic_history_len,
        ),
    }
    if spec.task_id == "2d_cfd_cns":
        payload["history_len"] = dynamic_history_len
    if not is_native_128:
        payload["blocker"] = {
            "reason": "non_native_spatial_shape",
            "message": f"expected native spatial shape {expected_shape}, observed {spatial_shape}",
        }
    return _jsonable(payload)


def _default_task_path(data_root: Path, spec: TaskSpec) -> Path:
    return data_root / spec.task_id / spec.expected_filename


def build_suite_preflight(
    *,
    data_root: Path,
    compute_sha256: bool = True,
    task_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Build the three-task suite preflight payload."""
    data_root = Path(data_root).expanduser()
    task_paths = task_paths or {}
    tasks = []
    for task_id in TASK_IDS:
        spec = get_task_spec(task_id)
        path = task_paths.get(task_id, _default_task_path(data_root, spec))
        tasks.append(inspect_task_file(task_id=task_id, data_file=path, compute_sha256=compute_sha256))

    ready_count = sum(1 for task in tasks if task["status"] == "ready")
    missing_listed_size_gb = 0.0
    for task in tasks:
        if task["status"] == "missing_file":
            listed = get_task_spec(task["task_id"]).listed_size_gb
            if listed is not None:
                missing_listed_size_gb += float(listed)
    storage_probe_root = data_root if data_root.exists() else data_root.parent
    usage = shutil.disk_usage(storage_probe_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "tasks": tasks,
        "ready_task_count": int(ready_count),
        "all_tasks_ready": ready_count == len(TASK_IDS),
        "missing_listed_size_gb": round(missing_listed_size_gb, 3),
        "storage": {
            "probe_root": str(storage_probe_root),
            "data_root_total_bytes": int(usage.total),
            "data_root_used_bytes": int(usage.used),
            "data_root_available_bytes": int(usage.free),
        },
        "meaningful_benchmark_training_rule": (
            "Benchmark-performance rows must train on the full available training split "
            "after validation/test holdout; capped, subsampled, smoke, and pilot runs are decision-support only."
        ),
    }


def _task_markdown_row(task: dict[str, Any]) -> str:
    blocker = task.get("blocker", {})
    units = task.get("available_supervision_units")
    units_text = "" if units is None else str(units)
    shape = task.get("native_spatial_shape")
    shape_text = "" if shape is None else "x".join(str(item) for item in shape)
    return (
        f"| `{task['task_id']}` | `{task['status']}` | `{shape_text}` | "
        f"{units_text} | {blocker.get('reason', '')} | {blocker.get('message', '')} |"
    )


def _shape_markdown(record: dict[str, Any]) -> str:
    return "[" + ", ".join(str(dim) for dim in record.get("shape", [])) + "]"


def _layout_markdown_line(task: dict[str, Any]) -> str | None:
    layout = task.get("layout")
    if not layout:
        return None
    if layout.get("kind") == "static_operator":
        input_dataset = layout["input_dataset"]
        target_dataset = layout["target_dataset"]
        return (
            f"- `{task['task_id']}`: static input `{input_dataset['path']}`, "
            f"shape `{_shape_markdown(input_dataset)}`, axis `{input_dataset.get('axis_order', '')}`; "
            f"target `{target_dataset['path']}`, shape `{_shape_markdown(target_dataset)}`, "
            f"axis `{target_dataset.get('axis_order', '')}`"
        )
    if layout.get("kind") == "dynamic_state":
        state_dataset = layout["state_dataset"]
        return (
            f"- `{task['task_id']}`: dynamic state `{state_dataset['path']}`, "
            f"shape `{_shape_markdown(state_dataset)}`, axis `{state_dataset.get('axis_order', '')}`"
        )
    if layout.get("kind") == "dynamic_multifield":
        first = layout["fields"][0]
        fields = ", ".join(f"`{field}`" for field in layout["field_order"])
        return (
            f"- `{task['task_id']}`: dynamic multi-field state fields {fields}, "
            f"shape `{_shape_markdown(first)}`, axis `{first.get('axis_order', '')}`"
        )
    return None


def render_preflight_markdown(payload: dict[str, Any]) -> str:
    """Render a durable markdown summary for the suite preflight."""
    lines = [
        "# PDEBench 128x128 Image-Suite Preflight",
        "",
        f"- Created: `{payload['created_at_utc']}`",
        f"- Data root: `{payload['data_root']}`",
        f"- Ready tasks: `{payload['ready_task_count']}` / `{len(payload['tasks'])}`",
        f"- All tasks ready: `{payload['all_tasks_ready']}`",
        f"- Missing listed data volume: `{payload['missing_listed_size_gb']}` GB",
        f"- Data-root available bytes: `{payload['storage']['data_root_available_bytes']}`",
        "",
        "Meaningful benchmark rows must train on the full available training split after validation/test holdout. Capped, subsampled, smoke, and pilot runs are decision-support only.",
        "",
        "| Task | Status | Native shape | Available supervision units | Blocker | Message |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    lines.extend(_task_markdown_row(task) for task in payload["tasks"])
    layout_lines = [_layout_markdown_line(task) for task in payload["tasks"]]
    layout_lines = [line for line in layout_lines if line is not None]
    if layout_lines:
        lines.extend(["", "## Dataset Schema Details", "", *layout_lines])
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "If any task is `missing_file`, stage the official PDEBench file outside git or on an approved external data root before launching benchmark training.",
            "",
        ]
    )
    return "\n".join(lines)


def write_preflight_artifacts(
    payload: dict[str, Any],
    *,
    output_root: Path,
    markdown_path: Path,
) -> tuple[Path, Path]:
    """Write machine-readable and durable markdown preflight artifacts."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "pdebench_image128_suite_preflight.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_preflight_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--markdown-path", type=Path, required=True)
    parser.add_argument("--no-sha256", action="store_true", help="Skip expensive SHA256 hashing during preflight.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_suite_preflight(data_root=args.data_root, compute_sha256=not args.no_sha256)
    json_path, markdown_path = write_preflight_artifacts(
        payload,
        output_root=args.output_root,
        markdown_path=args.markdown_path,
    )
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

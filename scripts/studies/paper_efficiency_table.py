"""Generate the NeurIPS paper efficiency table from existing artifacts."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
NEURIPS_DIR = Path("docs") / "plans" / "NEURIPS-HYBRID-RESNET-2026"
TABLES_DIR = NEURIPS_DIR / "tables"

CDI_TABLE_JSON = TABLES_DIR / "cdi_lines128_metrics_extended.json"
CNS_TABLE_JSON = TABLES_DIR / "pdebench_cns_matched_condition_metrics.json"
BRDT_TABLE_JSON = TABLES_DIR / "brdt_decision_support_metrics.json"
MODEL_CONFIG_JSON = TABLES_DIR / "model_config_by_benchmark.json"
PAPER_EVIDENCE_INDEX = NEURIPS_DIR / "paper_evidence_index.md"
SUMMARY_PATH = NEURIPS_DIR / "paper_efficiency_table_summary.md"

CDI_UNO_ROOT = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-30-cdi-lines128-uno-table-extension"
    / "runs"
    / "complete_table_plus_uno_20260504T100347Z"
)
CDI_UNO_MODEL_MANIFEST = CDI_UNO_ROOT / "model_manifest.json"
CDI_FFNO_NO_REFINER_MODEL_MANIFEST = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun"
    / "runs"
    / "ffno_no_refiner_20260506T223454Z"
    / "model_manifest.json"
)
CDI_SUPERVISED_FFNO_MODEL_MANIFEST = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun"
    / "runs"
    / "supervised_ffno_no_refiner_20260506T232535Z"
    / "model_manifest.json"
)
CNS_REFRESH_TABLE_JSON = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-04-cns-matched-condition-table-refresh"
    / "cns_paper_table_rows.json"
)
BRDT_40EP_ROOT = (
    Path(".artifacts")
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-07-brdt-sinogram-input-40ep-paper-evidence"
)
BRDT_40EP_METRICS_JSON = BRDT_40EP_ROOT / "combined_metrics.json"
BRDT_40EP_SPLIT_MANIFEST = BRDT_40EP_ROOT / "split_manifest.json"
BRDT_40EP_GATE_JSON = BRDT_40EP_ROOT / "paper_evidence_gate.json"

RUNTIME_FIELDS = (
    "train_wall_time_sec",
    "training_wall_time_sec",
    "wall_time_train_s",
    "command_wall_time_sec",
    "runtime_sec",
    "evaluation_wall_time_sec",
    "wall_time_eval_s",
    "eval_wall_time_sec",
)
THROUGHPUT_FIELDS = (
    "samples_per_second",
    "throughput_samples_per_second",
    "inference_samples_per_second",
)
LATENCY_FIELDS = (
    "latency_ms",
    "inference_latency_ms",
    "forward_latency_ms",
)
PARAMETER_FIELDS = ("unique_trainable_params", "parameter_count", "params", "num_params")
HARDWARE_FIELDS = (
    "hardware_label",
    "hardware_runtime_note",
    "device",
    "device_name",
    "accelerator",
    "gpu",
)


@dataclass(frozen=True)
class ThroughputEvidence:
    status: str
    samples_per_second: float | None
    latency_ms: float | None
    source_field: str | None
    source_path: str | None


@dataclass(frozen=True)
class EfficiencyRow:
    benchmark: str
    row_id: str
    model_label: str
    parameter_count: int | None
    parameter_count_source_field: str | None
    training_runtime_seconds: float | None
    training_runtime_source_field: str | None
    training_runtime_status: str
    hardware_label: str | None
    inference_throughput_status: str
    inference_samples_per_second: float | None
    inference_latency_ms: float | None
    throughput_source_field: str | None
    source_path: str
    claim_boundary: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def first_present(
    payload: Mapping[str, object],
    field_names: Sequence[str],
) -> tuple[str | None, object | None]:
    for field_name in field_names:
        if field_name in payload and payload[field_name] is not None:
            return field_name, payload[field_name]
    return None, None


def format_parameter_count(value: object) -> str:
    parsed = _as_int(value)
    if parsed is None:
        return "--"
    if parsed >= 1_000_000:
        return f"{parsed / 1_000_000:.2f}M"
    if parsed >= 1_000:
        return f"{parsed / 1_000:.1f}k"
    return str(parsed)


def escape_latex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def classify_inference_throughput(
    payload: Mapping[str, object],
    *,
    source_path: str,
) -> ThroughputEvidence:
    throughput_field, throughput_value = first_present(payload, THROUGHPUT_FIELDS)
    if throughput_field is not None:
        samples_per_second = _as_float(throughput_value)
        if samples_per_second is not None:
            return ThroughputEvidence(
                status="measured",
                samples_per_second=samples_per_second,
                latency_ms=None,
                source_field=throughput_field,
                source_path=source_path,
            )

    latency_field, latency_value = first_present(payload, LATENCY_FIELDS)
    if latency_field is not None:
        latency_ms = _as_float(latency_value)
        if latency_ms is not None:
            return ThroughputEvidence(
                status="measured",
                samples_per_second=None,
                latency_ms=latency_ms,
                source_field=latency_field,
                source_path=source_path,
            )

    return ThroughputEvidence(
        status="missing",
        samples_per_second=None,
        latency_ms=None,
        source_field=None,
        source_path=None,
    )


def _hardware_label(payload: Mapping[str, object]) -> str | None:
    hardware_summary = payload.get("hardware_summary")
    if isinstance(hardware_summary, Mapping):
        parts = [
            str(hardware_summary[key])
            for key in ("backend", "accelerator", "gpu", "device")
            if hardware_summary.get(key)
        ]
        if parts:
            return " / ".join(parts)

    environment = payload.get("environment")
    if isinstance(environment, Mapping):
        parts = [
            str(environment[key])
            for key in ("gpu", "torch_version", "cuda_version")
            if environment.get(key)
        ]
        if parts:
            return " / ".join(parts)

    field, value = first_present(payload, HARDWARE_FIELDS)
    if field is not None and value not in (None, ""):
        return str(value)
    return None


def _flatten_runtime_payload(payload: Mapping[str, object]) -> dict[str, object]:
    flat = dict(payload)
    for nested_key in ("runtime_summary", "runtime", "supporting"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                flat.setdefault(str(key), value)
    return flat


def normalize_efficiency_row(
    *,
    benchmark: str,
    row_id: str,
    model_label: str,
    source_path: str,
    payload: Mapping[str, object],
    claim_boundary: str,
) -> EfficiencyRow:
    flat = _flatten_runtime_payload(payload)
    parameter_field, parameter_value = first_present(flat, PARAMETER_FIELDS)
    runtime_field, runtime_value = first_present(flat, RUNTIME_FIELDS)
    runtime_seconds = _as_float(runtime_value)
    throughput = classify_inference_throughput(flat, source_path=source_path)
    runtime_status = "provenance_context" if runtime_seconds is not None else "missing"

    return EfficiencyRow(
        benchmark=benchmark,
        row_id=row_id,
        model_label=model_label,
        parameter_count=_as_int(parameter_value),
        parameter_count_source_field=parameter_field,
        training_runtime_seconds=runtime_seconds,
        training_runtime_source_field=runtime_field,
        training_runtime_status=runtime_status,
        hardware_label=_hardware_label(flat),
        inference_throughput_status=throughput.status,
        inference_samples_per_second=throughput.samples_per_second,
        inference_latency_ms=throughput.latency_ms,
        throughput_source_field=throughput.source_field,
        source_path=source_path,
        claim_boundary=claim_boundary,
    )


def _row_to_dict(row: EfficiencyRow | Mapping[str, object]) -> dict[str, object]:
    if isinstance(row, EfficiencyRow):
        return asdict(row)
    return dict(row)


def group_rows_by_benchmark(
    rows: Iterable[EfficiencyRow | Mapping[str, object]],
) -> OrderedDict[str, list[dict[str, object]]]:
    grouped: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for row in rows:
        payload = _row_to_dict(row)
        benchmark = str(payload["benchmark"])
        grouped.setdefault(benchmark, []).append(payload)
    return grouped


def _format_seconds(value: object) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "--"
    if parsed == 0:
        return "0"
    if parsed < 1:
        return f"{parsed:.3f}"
    return f"{parsed:.1f}"


def _format_throughput(row: Mapping[str, object]) -> str:
    status = str(row.get("inference_throughput_status") or "missing")
    samples = _as_float(row.get("inference_samples_per_second"))
    latency = _as_float(row.get("inference_latency_ms"))
    if status == "measured" and samples is not None:
        return f"{samples:.1f} samples/s"
    if status == "measured" and latency is not None:
        return f"{latency:.2f} ms"
    return status


def render_efficiency_table_tex(
    rows: Sequence[EfficiencyRow | Mapping[str, object]],
) -> str:
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Benchmark & Row & Model & Params & Runtime (s) & Runtime field & Inference \\",
        r"\midrule",
    ]
    grouped = group_rows_by_benchmark(rows)
    first_group = True
    for benchmark, group_rows in grouped.items():
        if not first_group:
            lines.append(r"\addlinespace")
        first_group = False
        lines.append(
            r"\multicolumn{7}{l}{\textbf{"
            + escape_latex(benchmark)
            + r"}} \\"
        )
        for row in group_rows:
            lines.append(
                " & ".join(
                    [
                        "",
                        escape_latex(row.get("row_id", "")),
                        escape_latex(row.get("model_label", "")),
                        escape_latex(format_parameter_count(row.get("parameter_count"))),
                        escape_latex(_format_seconds(row.get("training_runtime_seconds"))),
                        escape_latex(row.get("training_runtime_source_field") or "--"),
                        escape_latex(_format_throughput(row)),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _load_manifest_rows(path: Path) -> dict[str, tuple[dict[str, object], Path]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    result: dict[str, tuple[dict[str, object], Path]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            row_id = str(row.get("model_id") or row.get("row_id") or "")
            if row_id:
                result[row_id] = (dict(row), path)
    return result


def _load_model_config_param_counts(repo_root: Path) -> dict[tuple[str, str], tuple[int, Path]]:
    path = _repo_path(repo_root, MODEL_CONFIG_JSON)
    if not path.exists():
        return {}
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    counts: dict[tuple[str, str], tuple[int, Path]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        benchmark = str(row.get("benchmark") or "")
        row_id = str(row.get("row_id") or "")
        unique = _as_int(row.get("unique_trainable_params"))
        if benchmark and row_id and unique is not None:
            counts[(benchmark, row_id)] = (unique, path)
    return counts


def _collect_cdi_rows(repo_root: Path) -> list[EfficiencyRow]:
    table_path = _repo_path(repo_root, CDI_TABLE_JSON)
    if not table_path.exists():
        return []
    table_payload = _read_json(table_path)
    table_rows = table_payload.get("rows")
    if not isinstance(table_rows, list):
        return []

    manifest_rows: dict[str, tuple[dict[str, object], Path]] = {}
    for manifest in (
        CDI_UNO_MODEL_MANIFEST,
        CDI_FFNO_NO_REFINER_MODEL_MANIFEST,
        CDI_SUPERVISED_FFNO_MODEL_MANIFEST,
    ):
        manifest_rows.update(_load_manifest_rows(_repo_path(repo_root, manifest)))

    param_counts = _load_model_config_param_counts(repo_root)
    benchmark_label = str(table_payload.get("benchmark") or "CDI")
    rows: list[EfficiencyRow] = []
    for table_row in table_rows:
        if not isinstance(table_row, Mapping):
            continue
        row_id = str(table_row.get("row_id") or "")
        if not row_id:
            continue
        manifest_row, manifest_path = manifest_rows.get(row_id, ({}, table_path))
        payload = {**dict(table_row), **manifest_row}
        model_config_count = param_counts.get((benchmark_label, row_id))
        if model_config_count is not None:
            payload["unique_trainable_params"] = model_config_count[0]
            manifest_path = model_config_count[1]
        model = str(table_row.get("model") or payload.get("model_label") or row_id)
        training = str(table_row.get("training") or payload.get("training_procedure") or "")
        model_label = f"{model} + {training}" if training else model
        rows.append(
            normalize_efficiency_row(
                benchmark=benchmark_label,
                row_id=row_id,
                model_label=model_label,
                source_path=_repo_rel(repo_root, manifest_path),
                payload=payload,
                claim_boundary=str(
                    payload.get("claim_boundary")
                    or table_payload.get("claim_boundary")
                    or "complete_lines128_cdi_benchmark_plus_uno_extension"
                ),
            )
        )
    return rows


def _collect_cns_rows(repo_root: Path) -> list[EfficiencyRow]:
    preferred = _repo_path(repo_root, CNS_REFRESH_TABLE_JSON)
    table_path = preferred if preferred.exists() else _repo_path(repo_root, CNS_TABLE_JSON)
    if not table_path.exists():
        return []
    table_payload = _read_json(table_path)
    table_rows = table_payload.get("rows")
    if not isinstance(table_rows, list):
        return []
    rows: list[EfficiencyRow] = []
    for row in table_rows:
        if not isinstance(row, Mapping):
            continue
        row_id = str(row.get("row_id") or "")
        if not row_id:
            continue
        rows.append(
            normalize_efficiency_row(
                benchmark="PDEBench CNS",
                row_id=row_id,
                model_label=str(row.get("manuscript_label") or row_id),
                source_path=_repo_rel(repo_root, table_path),
                payload=row,
                claim_boundary=str(
                    row.get("claim_scope")
                    or table_payload.get("claim_boundary")
                    or "bounded_capped_decision_support_only"
                ),
            )
        )
    return rows


def _candidate_context(repo_root: Path) -> list[dict[str, object]]:
    return []


def _brdt_claim_boundary(repo_root: Path) -> str:
    gate_path = _repo_path(repo_root, BRDT_40EP_GATE_JSON)
    if gate_path.exists():
        payload = _read_json(gate_path)
        boundary = str(payload.get("claim_boundary") or "")
        if boundary:
            return boundary
    metrics_path = _repo_path(repo_root, BRDT_40EP_METRICS_JSON)
    if metrics_path.exists():
        payload = _read_json(metrics_path)
        boundary = str(payload.get("claim_boundary") or "")
        if boundary:
            return boundary
    return "decision_support_convergence_followup"


def _brdt_test_count(repo_root: Path) -> int | None:
    split_path = _repo_path(repo_root, BRDT_40EP_SPLIT_MANIFEST)
    payload: Mapping[str, Any] | None = None
    if split_path.exists():
        payload = _read_json(split_path)
    else:
        preflight_path = _repo_path(repo_root, BRDT_40EP_ROOT / "preflight_manifest.json")
        if preflight_path.exists():
            payload = _read_json(preflight_path)
    if payload is None:
        return None
    split_counts = payload.get("split_counts")
    if not isinstance(split_counts, Mapping):
        dataset_block = payload.get("dataset")
        if isinstance(dataset_block, Mapping):
            split_counts = dataset_block.get("split_counts")
    if isinstance(split_counts, Mapping):
        return _as_int(split_counts.get("test"))
    return None


def _collect_brdt_rows(repo_root: Path) -> list[EfficiencyRow]:
    metrics_path = _repo_path(repo_root, BRDT_40EP_METRICS_JSON)
    if not metrics_path.exists():
        return []
    payload = _read_json(metrics_path)
    table_rows = payload.get("rows")
    if not isinstance(table_rows, list):
        return []

    test_count = _brdt_test_count(repo_root)
    claim_boundary = _brdt_claim_boundary(repo_root)
    rows: list[EfficiencyRow] = []
    for table_row in table_rows:
        if not isinstance(table_row, Mapping):
            continue
        row_id = str(table_row.get("row_id") or "")
        if not row_id:
            continue
        if row_id == "classical_born_backprop":
            continue
        normalized_payload = dict(table_row)
        runtime = table_row.get("runtime")
        if isinstance(runtime, Mapping):
            eval_seconds = _as_float(runtime.get("wall_time_eval_s"))
            if test_count is not None and eval_seconds is not None and eval_seconds > 0:
                normalized_payload["samples_per_second"] = test_count / eval_seconds
        rows.append(
            normalize_efficiency_row(
                benchmark="BRDT",
                row_id=row_id,
                model_label="SRU-Net"
                if row_id == "sru_net"
                else str(table_row.get("paper_label") or row_id),
                source_path=_repo_rel(repo_root, metrics_path),
                payload=normalized_payload,
                claim_boundary=claim_boundary,
            )
        )
    return rows


def collect_efficiency_rows(repo_root: Path = REPO_ROOT) -> list[EfficiencyRow]:
    repo_root = Path(repo_root)
    return [
        *_collect_cdi_rows(repo_root),
        *_collect_cns_rows(repo_root),
        *_collect_brdt_rows(repo_root),
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "row_id",
        "model_label",
        "parameter_count",
        "parameter_count_source_field",
        "training_runtime_seconds",
        "training_runtime_source_field",
        "training_runtime_status",
        "hardware_label",
        "inference_throughput_status",
        "inference_samples_per_second",
        "inference_latency_ms",
        "throughput_source_field",
        "source_path",
        "claim_boundary",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def render_summary(
    rows: Sequence[dict[str, object]],
    *,
    excluded_candidate_context: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Paper Efficiency Table Summary",
        "",
        "This summary records the repo-local efficiency table generated for the NeurIPS SRU-Net evidence package.",
        "The table groups rows by benchmark and keeps runtime fields as provenance context unless an explicit throughput field exists.",
        "",
        "## Outputs",
        "",
        "- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`",
        "- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`",
        "- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`",
        "",
        "## Row Counts",
        "",
    ]
    grouped = group_rows_by_benchmark(rows)
    for benchmark, group_rows in grouped.items():
        lines.append(f"- {benchmark}: {len(group_rows)} rows")
    lines.extend(
        [
            "",
            "## Runtime And Throughput Policy",
            "",
            "- Parameter counts use `unique_trainable_params` from the model-configuration table when available; otherwise they fall back to existing row artifacts.",
            "- Training/runtime fields keep their original source field names.",
            "- Missing inference throughput is labeled `missing`; training runtime is not converted into throughput.",
            "- Rows from different benchmarks are not ranked against each other.",
            "",
            "## Superseded Context",
            "",
        ]
    )
    if excluded_candidate_context:
        for item in excluded_candidate_context:
            lines.append(
                "- "
                + str(item.get("benchmark"))
                + ": "
                + str(item.get("reason"))
                + f" (`{item.get('source_path')}`)"
            )
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def write_paper_efficiency_table(
    repo_root: Path = REPO_ROOT,
    output_dir: Path | None = None,
) -> dict[str, str]:
    repo_root = Path(repo_root)
    if output_dir is None:
        output_dir = _repo_path(repo_root, TABLES_DIR)
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir

    rows = [asdict(row) for row in collect_efficiency_rows(repo_root)]
    excluded = _candidate_context(repo_root)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "paper_efficiency_table.v1",
        "generated_at_utc": generated_at,
        "rows": rows,
        "excluded_candidate_context": excluded,
        "notes": [
            "Training runtime is provenance context unless a common measurement protocol is recorded.",
            "Inference throughput is reported only from explicit throughput or latency fields.",
        ],
    }

    json_path = output_dir / "paper_efficiency_table.json"
    csv_path = output_dir / "paper_efficiency_table.csv"
    tex_path = output_dir / "paper_efficiency_table.tex"
    summary_path = _repo_path(repo_root, SUMMARY_PATH)

    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    tex_path.write_text(render_efficiency_table_tex(rows), encoding="utf-8")
    summary_path.write_text(
        render_summary(rows, excluded_candidate_context=excluded),
        encoding="utf-8",
    )

    return {
        "json": _repo_rel(repo_root, json_path),
        "csv": _repo_rel(repo_root, csv_path),
        "tex": _repo_rel(repo_root, tex_path),
        "summary": _repo_rel(repo_root, summary_path),
    }


def main() -> int:
    outputs = write_paper_efficiency_table()
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

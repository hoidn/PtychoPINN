#!/usr/bin/env python3
"""Build the bounded CNS paper table and figure bundle from locked rows."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.studies.pdebench_image128.reporting import (
    _load_run_record,
    build_cns_paper_table_bundle,
    validate_cns_paper_table_bundle,
    write_cns_paper_table_bundle,
)
from scripts.studies.pdebench_image128.visualization import cfd_cns_shared_scale_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCKED_ROWS_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle"
)
DEFAULT_SEARCH_ROOTS = [REPO_ROOT / ".artifacts/NEURIPS-HYBRID-RESNET-2026"]
REQUIRED_1024_HEADLINE_ROWS = [
    "spectral_resnet_bottleneck_base",
    "fno_base",
    "unet_strong",
    "author_ffno_cns_base",
]
PREFERRED_1024_RUN_ROOTS = {
    "spectral_resnet_bottleneck_base": REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z",
}
CONTRACT_KEYS = [
    "dataset_file",
    "split_counts",
    "max_windows_per_trajectory",
    "history_len",
    "epochs",
    "batch_size",
    "training_loss",
    "metric_family",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _expected_1024_contract(locked_rows_payload: dict[str, Any]) -> dict[str, Any]:
    selected = dict(locked_rows_payload.get("selected_contract", {}))
    split_counts = {"train": 1024, "val": 128, "test": 128}
    selected["split_counts"] = split_counts
    selected["max_windows_per_trajectory"] = int(selected.get("max_windows_per_trajectory", 8))
    selected["history_len"] = int(selected.get("history_len", locked_rows_payload.get("history_len", 2)))
    selected["epochs"] = int(selected.get("epochs", locked_rows_payload.get("epochs", 40)))
    selected["batch_size"] = int(locked_rows_payload.get("batch_size", 4))
    selected["training_loss"] = str(locked_rows_payload.get("training_loss", selected.get("training_loss", "mse")))
    selected["metric_family"] = list(locked_rows_payload.get("metric_family", selected.get("metric_family", [])))
    selected["dataset_file"] = str(locked_rows_payload.get("dataset_file", selected.get("dataset_file", "")))
    return {key: selected.get(key) for key in CONTRACT_KEYS}


def _contract_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == expected.get(key) for key in CONTRACT_KEYS)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _discover_compatible_run(
    profile_id: str,
    *,
    expected_contract: dict[str, Any],
    search_roots: list[Path],
    preferred_run_root: Path | None = None,
) -> dict[str, Any] | None:
    candidate_roots: list[Path] = []
    if preferred_run_root is not None and preferred_run_root.exists():
        candidate_roots.append(Path(preferred_run_root))
    for search_root in search_roots:
        if not Path(search_root).exists():
            continue
        for metrics_path in Path(search_root).rglob(f"metrics_{profile_id}.json"):
            candidate_roots.append(metrics_path.parent)
    for run_root in _dedupe_paths(candidate_roots):
        try:
            record = _load_run_record(run_root, profile_id=profile_id, source_document="artifact_scan")
        except (FileNotFoundError, ValueError):
            continue
        if _contract_matches(record["contract"], expected_contract):
            return {
                "row_id": profile_id,
                "run_root": str(run_root),
                "contract": record["contract"],
                "metrics": record["row"],
            }
    return None


def _rerun_output_root(profile_id: str, output_root: Path) -> Path:
    stem = profile_id.replace("_base", "")
    return output_root / "rerun_candidates" / f"{stem}-1024cap-40ep"


def _rerun_command(profile_id: str, *, expected_contract: dict[str, Any], output_root: Path) -> str:
    dataset_file = Path(str(expected_contract["dataset_file"]))
    data_root = dataset_file.parent.parent
    split_counts = dict(expected_contract["split_counts"])
    return (
        "python scripts/studies/run_pdebench_image128_suite.py "
        "--task 2d_cfd_cns "
        "--mode readiness "
        f"--data-root {data_root} "
        f"--output-root {_rerun_output_root(profile_id, output_root)} "
        f"--profiles {profile_id} "
        f"--history-len {int(expected_contract['history_len'])} "
        f"--epochs {int(expected_contract['epochs'])} "
        f"--batch-size {int(expected_contract['batch_size'])} "
        f"--max-train-trajectories {int(split_counts['train'])} "
        f"--max-val-trajectories {int(split_counts['val'])} "
        f"--max-test-trajectories {int(split_counts['test'])} "
        f"--max-windows-per-trajectory {int(expected_contract['max_windows_per_trajectory'])} "
        "--device cuda "
        "--num-workers 0"
    )


def audit_cns_paper_bundle_upgrade(
    *,
    locked_rows_path: Path,
    output_root: Path,
    search_roots: list[Path] | None = None,
    preferred_run_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    locked_rows_payload = _load_json(Path(locked_rows_path))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    search_roots = [Path(path) for path in (search_roots or DEFAULT_SEARCH_ROOTS)]
    preferred_run_roots = {**PREFERRED_1024_RUN_ROOTS, **(preferred_run_roots or {})}
    expected_contract = _expected_1024_contract(locked_rows_payload)

    compatible_rows: dict[str, Any] = {}
    missing_rows: list[dict[str, Any]] = []
    for row_id in REQUIRED_1024_HEADLINE_ROWS:
        match = _discover_compatible_run(
            row_id,
            expected_contract=expected_contract,
            search_roots=search_roots,
            preferred_run_root=preferred_run_roots.get(row_id),
        )
        if match is None:
            missing_rows.append(
                {
                    "row_id": row_id,
                    "reason": "missing_same_contract_1024_row",
                    "rerun_command": _rerun_command(row_id, expected_contract=expected_contract, output_root=output_root),
                }
            )
            continue
        compatible_rows[row_id] = match

    audit_outcome = "upgrade_ready" if not missing_rows else "fallback_to_512_required"
    payload = {
        "schema_version": "pdebench_cns_paper_bundle_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_rows_path": str(locked_rows_path),
        "fallback_locked_rows_path": str(locked_rows_path),
        "audit_outcome": audit_outcome,
        "expected_1024_contract": expected_contract,
        "compatible_1024_rows": compatible_rows,
        "missing_or_incompatible_rows": missing_rows,
        "comparison_standard": "Exact match on dataset file, split counts, max_windows_per_trajectory, history_len, epochs, batch_size, training_loss, and metric_family.",
    }
    _write_json(output_root / "1024_same_cap_audit.json", payload)
    _write_audit_markdown(output_root / "1024_same_cap_audit.md", payload)
    return payload


def _write_audit_markdown(path: Path, payload: dict[str, Any]) -> Path:
    lines = [
        "# 1024 Same-Cap Audit",
        "",
        f"- Outcome: `{payload['audit_outcome']}`",
        f"- Locked fallback manifest: `{payload['fallback_locked_rows_path']}`",
        "",
        "## Compatible 1024 Rows",
    ]
    if payload["compatible_1024_rows"]:
        for row_id, row in payload["compatible_1024_rows"].items():
            lines.append(f"- `{row_id}`: `{row['run_root']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Missing Or Incompatible Rows"])
    if payload["missing_or_incompatible_rows"]:
        for row in payload["missing_or_incompatible_rows"]:
            lines.append(f"- `{row['row_id']}`: `{row['reason']}`")
            lines.append(f"  rerun: `{row['rerun_command']}`")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_sample_id(path: Path) -> int:
    stem = path.stem
    marker = "sample"
    if marker not in stem:
        raise ValueError(f"cannot parse sample id from {path}")
    return int(stem.split(marker)[-1])


def _row_npz_candidates(row: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    asset_pointers = row.get("asset_pointers", {}) or {}
    sample_npz = asset_pointers.get("sample_npz")
    if isinstance(sample_npz, str) and sample_npz.strip():
        candidates.append(Path(sample_npz))
    run_root = Path(str(row.get("run_root", "")))
    row_id = str(row.get("row_id"))
    if run_root.exists():
        candidates.extend(sorted(run_root.glob(f"comparison_{row_id}_sample*.npz")))
    return _dedupe_paths(candidates)


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "prediction": np.asarray(data["prediction"], dtype=np.float32),
            "target": np.asarray(data["target"], dtype=np.float32),
            "abs_error": np.asarray(data["abs_error"], dtype=np.float32),
            "field_order": [str(item) for item in data["field_order"].tolist()],
        }


def _visual_rows_from_locked_payload(locked_rows_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_id = {str(row["row_id"]): dict(row) for row in locked_rows_payload["rows"]}
    headline_ids = [str(row_id) for row_id in locked_rows_payload.get("headline_row_ids", [])]
    continuity_ids = [str(row_id) for row_id in locked_rows_payload.get("continuity_row_ids", [])]
    headline_split = rows_by_id[headline_ids[0]]["split_counts"]
    selected_rows = [rows_by_id[row_id] for row_id in headline_ids]
    for row_id in continuity_ids:
        if rows_by_id[row_id]["split_counts"] == headline_split:
            selected_rows.append(rows_by_id[row_id])
    return selected_rows


def _resolve_shared_sample_ids(rows: list[dict[str, Any]]) -> tuple[list[int], dict[str, dict[int, Path]]]:
    sample_map: dict[str, dict[int, Path]] = {}
    intersection: set[int] | None = None
    for row in rows:
        row_id = str(row["row_id"])
        row_samples = { _parse_sample_id(path): path for path in _row_npz_candidates(row) if path.exists() }
        if not row_samples:
            raise FileNotFoundError(f"no sample npz artifacts available for {row_id}")
        sample_map[row_id] = row_samples
        row_ids = set(row_samples)
        intersection = row_ids if intersection is None else intersection & row_ids
    if not intersection:
        raise ValueError("no compatible sample ids exist across the selected visual rows")
    return sorted(intersection), sample_map


def _copy_source_npz(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _save_panel(path: Path, image: np.ndarray, spec: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image, cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
    return path


def _build_figure_bundle(locked_rows_payload: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    rows = _visual_rows_from_locked_payload(locked_rows_payload)
    sample_ids, sample_paths = _resolve_shared_sample_ids(rows)
    field_order: list[str] | None = None
    shared_field_scales: dict[str, Any] = {}
    shared_error_scales: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample_label = f"sample{sample_id:03d}"
        loaded_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = str(row["row_id"])
            source_path = sample_paths[row_id][sample_id]
            copied_path = _copy_source_npz(
                source_path,
                output_root / "figure_sources" / sample_label / f"{row_id}.npz",
            )
            loaded_rows[row_id] = {
                **_load_npz_payload(copied_path),
                "copied_npz_path": str(copied_path),
            }

        first_row_id = str(rows[0]["row_id"])
        reference_target = loaded_rows[first_row_id]["target"]
        reference_field_order = loaded_rows[first_row_id]["field_order"]
        for row_id, payload in loaded_rows.items():
            if payload["field_order"] != reference_field_order:
                raise ValueError(f"field-order mismatch for sample {sample_id}: {row_id}")
            if not np.allclose(payload["target"], reference_target, atol=1e-6, rtol=1e-6):
                raise ValueError(f"target mismatch for sample {sample_id}: {row_id}")
        field_order = list(reference_field_order)

        for channel, field_name in enumerate(field_order):
            bundle = cfd_cns_shared_scale_bundle(
                field_name,
                value_arrays=[reference_target[channel], *[payload["prediction"][channel] for payload in loaded_rows.values()]],
                error_arrays=[payload["abs_error"][channel] for payload in loaded_rows.values()],
            )
            shared_field_scales[field_name] = bundle["value_scale"]
            shared_error_scales[field_name] = bundle["error_scale"]

            target_path = _save_panel(
                output_root / "figures" / sample_label / f"{field_name}__target.png",
                reference_target[channel],
                bundle["value_scale"],
            )
            entries.append(
                {
                    "sample_id": sample_id,
                    "field_name": field_name,
                    "row_id": "ground_truth",
                    "panel_kind": "target",
                    "png_path": str(target_path),
                    "scale_kind": "value",
                }
            )
            for row in rows:
                row_id = str(row["row_id"])
                prediction_path = _save_panel(
                    output_root / "figures" / sample_label / f"{field_name}__{row_id}__prediction.png",
                    loaded_rows[row_id]["prediction"][channel],
                    bundle["value_scale"],
                )
                error_path = _save_panel(
                    output_root / "figures" / sample_label / f"{field_name}__{row_id}__abs_error.png",
                    loaded_rows[row_id]["abs_error"][channel],
                    bundle["error_scale"],
                )
                entries.extend(
                    [
                        {
                            "sample_id": sample_id,
                            "field_name": field_name,
                            "row_id": row_id,
                            "panel_kind": "prediction",
                            "png_path": str(prediction_path),
                            "scale_kind": "value",
                            "source_npz_path": loaded_rows[row_id]["copied_npz_path"],
                        },
                        {
                            "sample_id": sample_id,
                            "field_name": field_name,
                            "row_id": row_id,
                            "panel_kind": "abs_error",
                            "png_path": str(error_path),
                            "scale_kind": "error",
                            "source_npz_path": loaded_rows[row_id]["copied_npz_path"],
                        },
                    ]
                )

    field_order = field_order or []
    field_scale_path = _write_json(output_root / "shared_field_scales.json", shared_field_scales)
    error_scale_path = _write_json(output_root / "shared_error_scales.json", shared_error_scales)
    sample_manifest = {
        "schema_version": "pdebench_cns_fixed_sample_manifest_v1",
        "sample_ids": sample_ids,
        "rows_in_visual_bundle": [str(row["row_id"]) for row in rows],
        "field_order": field_order,
    }
    figure_manifest = {
        "schema_version": "pdebench_cns_figure_manifest_v1",
        "sample_ids": sample_ids,
        "rows_in_visual_bundle": [str(row["row_id"]) for row in rows],
        "field_order": field_order,
        "entries": entries,
    }
    _write_json(output_root / "fixed_sample_manifest.json", sample_manifest)
    _write_json(output_root / "figure_manifest.json", figure_manifest)
    return {
        "sample_manifest_path": str(output_root / "fixed_sample_manifest.json"),
        "figure_manifest_path": str(output_root / "figure_manifest.json"),
        "field_scale_path": str(field_scale_path),
        "error_scale_path": str(error_scale_path),
        "sample_ids": sample_ids,
        "rows_in_visual_bundle": [str(row["row_id"]) for row in rows],
    }


def run_cns_paper_bundle(
    *,
    locked_rows_path: Path = DEFAULT_LOCKED_ROWS_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    locked_rows_path = Path(locked_rows_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    audit_payload = audit_cns_paper_bundle_upgrade(
        locked_rows_path=locked_rows_path,
        output_root=output_root,
        search_roots=search_roots,
    )
    locked_rows_payload = _load_json(locked_rows_path)
    bundle_kind = (
        "same_contract_1024_bundle_complete"
        if int(locked_rows_payload["split_counts"]["train"]) == 1024
        else "fallback_512_bundle_used"
    )
    _write_json(
        output_root / "bundle_input_manifest.json",
        {
            "schema_version": "pdebench_cns_bundle_input_manifest_v1",
            "authoritative_locked_rows_path": str(locked_rows_path),
            "contract_authority": str(locked_rows_payload.get("contract_authority", "")),
            "bundle_kind": bundle_kind,
            "audit_outcome": audit_payload["audit_outcome"],
        },
    )

    table_payload = build_cns_paper_table_bundle(
        locked_rows_payload,
        authoritative_manifest_path=str(locked_rows_path),
    )
    table_json, table_csv, table_tex = write_cns_paper_table_bundle(table_payload, output_root)
    figure_payload = _build_figure_bundle(locked_rows_payload, output_root=output_root)

    validation_payload = {
        **validate_cns_paper_table_bundle(table_payload),
        "table_json_path": str(table_json),
        "table_csv_path": str(table_csv),
        "table_tex_path": str(table_tex),
        "figure_manifest_path": figure_payload["figure_manifest_path"],
        "sample_manifest_path": figure_payload["sample_manifest_path"],
    }
    _write_json(output_root / "bundle_validation.json", validation_payload)
    return {
        "bundle_kind": bundle_kind,
        "audit_outcome": audit_payload["audit_outcome"],
        "table_json_path": str(table_json),
        "table_csv_path": str(table_csv),
        "table_tex_path": str(table_tex),
        "figure_manifest_path": figure_payload["figure_manifest_path"],
        "validation_path": str(output_root / "bundle_validation.json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-rows-path", type=Path, default=DEFAULT_LOCKED_ROWS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--search-root", dest="search_roots", action="append", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_cns_paper_bundle(
        locked_rows_path=args.locked_rows_path,
        output_root=args.output_root,
        search_roots=args.search_roots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

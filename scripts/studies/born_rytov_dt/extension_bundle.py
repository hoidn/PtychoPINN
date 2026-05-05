"""Append-only baseline + FFNO extension bundle assembly.

Produces three combined artifacts under the FFNO extension root:

- ``combined_metrics.json``  — five-row metrics view (baseline four rows
  by lineage + FFNO appended).
- ``combined_metrics.csv``   — flat CSV mirror of the combined metrics.
- ``combined_manifest.json`` — top-level lineage manifest pointing at
  the baseline bundle by absolute path and at the FFNO extension
  artifacts by relative path.

The helper is task-local and intentionally narrow:

- it never re-runs anything,
- it never overwrites the baseline four-row bundle,
- it preserves the baseline rows' visible identity (``paper_label`` and
  ``architecture``) byte-for-byte so the combined view does not silently
  relabel the existing classical/U-Net/FNO/Hybrid-family rows.

The FFNO row identity is required to be ``ffno`` and is appended as the
fifth row. The combined manifest's ``claim_boundary`` is always
``decision_support_append_only``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod


COMBINED_BUNDLE_SCHEMA_VERSION: str = "brdt_ffno_extension_combined_v1"
COMBINED_METRICS_JSON: str = "combined_metrics.json"
COMBINED_METRICS_CSV: str = "combined_metrics.csv"
COMBINED_MANIFEST_JSON: str = "combined_manifest.json"
APPEND_ONLY_CLAIM_BOUNDARY: str = "decision_support_append_only"
EXTENSION_BACKLOG_ITEM: str = "2026-05-04-brdt-ffno-row-extension"
BASELINE_BACKLOG_ITEM: str = "2026-04-29-brdt-four-row-preflight"
FFNO_ROW_ID: str = "ffno"
FFNO_PAPER_LABEL: str = "FFNO"

# Baseline-bundle contract fields the FFNO extension MUST inherit unchanged.
# Only fields the new FFNO row could plausibly drift from are checked here;
# the operator pointer + dataset id + claim boundary lock down the rest.
_BASELINE_REQUIRED_DATASET_KEYS: Tuple[str, ...] = (
    "dataset_id",
    "split_counts",
    "normalization",
    "manifest_path",
)
_BASELINE_REQUIRED_INPUT_KEYS: Tuple[str, ...] = ("input_mode", "in_channels")
_BASELINE_REQUIRED_TRAINING_KEYS: Tuple[str, ...] = (
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "loss_weights",
    "seed",
)
_BASELINE_REQUIRED_OPERATOR_KEYS: Tuple[str, ...] = (
    "geometry",
    "validation_artifact",
)


class BaselineContractMismatchError(ValueError):
    """Raised when the FFNO extension cannot inherit the baseline contract."""


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def baseline_contract_fingerprint(baseline_manifest: Mapping[str, Any]) -> str:
    """Stable fingerprint of the baseline contract the FFNO row must inherit.

    Covers the dataset identity, split counts, normalization, operator
    pointer/geometry, input contract, fixed-sample ids/seed, and training
    contract — i.e. exactly the fields the FFNO extension must reproduce
    byte-for-byte. Excludes per-row metric values, runtime, and rendered
    visuals so the fingerprint is invariant under harmless visual rebuilds.
    """
    payload = {
        "dataset": {
            k: baseline_manifest.get("dataset", {}).get(k)
            for k in _BASELINE_REQUIRED_DATASET_KEYS
        },
        "operator": {
            k: baseline_manifest.get("operator", {}).get(k)
            for k in _BASELINE_REQUIRED_OPERATOR_KEYS
        },
        "input_contract": {
            k: baseline_manifest.get("input_contract", {}).get(k)
            for k in _BASELINE_REQUIRED_INPUT_KEYS
        },
        "training_contract": {
            k: baseline_manifest.get("training_contract", {}).get(k)
            for k in _BASELINE_REQUIRED_TRAINING_KEYS
        },
        "fixed_sample_ids": baseline_manifest.get("fixed_sample_ids", []),
        "fixed_sample_seed": baseline_manifest.get("fixed_sample_seed"),
        "claim_boundary": baseline_manifest.get("claim_boundary"),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_baseline_bundle(baseline_root: Path) -> Dict[str, Any]:
    """Read and structurally check a baseline four-row preflight bundle.

    Returns the parsed baseline ``preflight_manifest.json`` payload.
    Raises ``BaselineContractMismatchError`` if any required field is
    missing or if the bundle's claim boundary / backlog item is not the
    completed four-row preflight.
    """
    base = Path(baseline_root)
    manifest_path = base / "preflight_manifest.json"
    metrics_path = base / "metrics.json"
    if not manifest_path.exists():
        raise BaselineContractMismatchError(
            f"baseline bundle missing preflight_manifest.json at {manifest_path}"
        )
    if not metrics_path.exists():
        raise BaselineContractMismatchError(
            f"baseline bundle missing metrics.json at {metrics_path}"
        )
    manifest = _read_json(manifest_path)
    backlog = manifest.get("backlog_item")
    if backlog != BASELINE_BACKLOG_ITEM:
        raise BaselineContractMismatchError(
            f"baseline manifest backlog_item={backlog!r}; expected "
            f"{BASELINE_BACKLOG_ITEM!r} (the completed four-row preflight)"
        )
    claim = manifest.get("claim_boundary")
    if claim != "decision_support_preflight_only":
        raise BaselineContractMismatchError(
            f"baseline manifest claim_boundary={claim!r}; expected "
            "'decision_support_preflight_only'"
        )
    for section, keys in (
        ("dataset", _BASELINE_REQUIRED_DATASET_KEYS),
        ("operator", _BASELINE_REQUIRED_OPERATOR_KEYS),
        ("input_contract", _BASELINE_REQUIRED_INPUT_KEYS),
        ("training_contract", _BASELINE_REQUIRED_TRAINING_KEYS),
    ):
        block = manifest.get(section)
        if not isinstance(block, Mapping):
            raise BaselineContractMismatchError(
                f"baseline manifest missing or non-mapping section {section!r}"
            )
        missing = [k for k in keys if k not in block]
        if missing:
            raise BaselineContractMismatchError(
                f"baseline manifest section {section!r} missing keys: {missing}"
            )
    if not manifest.get("fixed_sample_ids"):
        raise BaselineContractMismatchError(
            "baseline manifest missing fixed_sample_ids"
        )
    return manifest


def assert_extension_inherits_baseline(
    *,
    baseline_manifest: Mapping[str, Any],
    extension_dataset_id: str,
    extension_input_mode: str,
    extension_in_channels: int,
    extension_training_contract: Mapping[str, Any],
    extension_fixed_sample_ids: List[int],
    extension_operator_pointer: str,
    extension_claim_boundary: str,
) -> None:
    """Refuse an FFNO extension that does not inherit the baseline contract.

    Mismatches are reported as a single ``BaselineContractMismatchError``
    naming every divergent field so the runner produces a row-level
    blocker instead of silently relaxing the locked contract.
    """
    mismatches: List[str] = []
    base_dataset = baseline_manifest.get("dataset", {}) or {}
    base_input = baseline_manifest.get("input_contract", {}) or {}
    base_training = baseline_manifest.get("training_contract", {}) or {}
    base_operator = baseline_manifest.get("operator", {}) or {}

    if base_dataset.get("dataset_id") != extension_dataset_id:
        mismatches.append(
            f"dataset_id: baseline={base_dataset.get('dataset_id')!r}, "
            f"extension={extension_dataset_id!r}"
        )
    if base_input.get("input_mode") != extension_input_mode:
        mismatches.append(
            f"input_mode: baseline={base_input.get('input_mode')!r}, "
            f"extension={extension_input_mode!r}"
        )
    if int(base_input.get("in_channels", -1)) != int(extension_in_channels):
        mismatches.append(
            f"in_channels: baseline={base_input.get('in_channels')!r}, "
            f"extension={extension_in_channels!r}"
        )
    base_pointer = (
        base_operator.get("validation_artifact")
        or base_operator.get("validation_report")
    )
    if str(base_pointer) != str(extension_operator_pointer):
        mismatches.append(
            f"operator_pointer: baseline={base_pointer!r}, "
            f"extension={extension_operator_pointer!r}"
        )
    for key in _BASELINE_REQUIRED_TRAINING_KEYS:
        if base_training.get(key) != extension_training_contract.get(key):
            mismatches.append(
                f"training_contract.{key}: baseline={base_training.get(key)!r}, "
                f"extension={extension_training_contract.get(key)!r}"
            )
    base_ids = [int(i) for i in (baseline_manifest.get("fixed_sample_ids") or [])]
    ext_ids = [int(i) for i in extension_fixed_sample_ids]
    if base_ids != ext_ids:
        mismatches.append(
            f"fixed_sample_ids: baseline={base_ids!r}, extension={ext_ids!r}"
        )
    if extension_claim_boundary != APPEND_ONLY_CLAIM_BOUNDARY:
        mismatches.append(
            f"claim_boundary: extension={extension_claim_boundary!r}; "
            f"FFNO extension must use {APPEND_ONLY_CLAIM_BOUNDARY!r}"
        )
    if mismatches:
        raise BaselineContractMismatchError(
            "FFNO extension does not inherit the baseline contract: "
            + "; ".join(mismatches)
        )


def _baseline_row_lineage(
    *,
    baseline_root: Path,
    baseline_metrics: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build the read-only baseline-row lineage block for combined views.

    Each entry carries the visible row identity, the per-row metric
    summary, and an absolute ``source_metrics_json`` pointer so the
    combined view never paraphrases the baseline numbers — they are
    referenced verbatim from the baseline bundle.
    """
    rows = list(baseline_metrics.get("rows") or [])
    manifest_rows = list(baseline_manifest.get("rows") or [])
    manifest_by_id = {
        str(r.get("row_id")): r for r in manifest_rows if r.get("row_id")
    }
    lineage: List[Dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        manifest_row = manifest_by_id.get(row_id) or {}
        lineage_entry: Dict[str, Any] = {
            "source": "baseline_lineage",
            "baseline_root": str(baseline_root.resolve()),
            "row_id": row_id,
            "paper_label": row.get("paper_label") or manifest_row.get("paper_label"),
            "architecture": row.get("architecture")
            or manifest_row.get("model"),
            "row_status": row.get("row_status"),
            "image": dict(row.get("image") or {}),
            "measurement": dict(row.get("measurement") or {}),
            "supporting": dict(row.get("supporting") or {}),
            "runtime": dict(row.get("runtime") or {}),
        }
        for opt_key in ("blocker_reason", "blocker_message"):
            if row.get(opt_key):
                lineage_entry[opt_key] = row[opt_key]
        lineage.append(lineage_entry)
    return lineage


def _ffno_combined_row(
    *,
    extension_root: Path,
    ffno_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project the extension-bundle FFNO row into the combined view."""
    rows = list(ffno_metrics.get("rows") or [])
    ffno = next(
        (row for row in rows if str(row.get("row_id")) == FFNO_ROW_ID),
        None,
    )
    if ffno is None:
        raise BaselineContractMismatchError(
            "extension metrics.json does not contain an 'ffno' row"
        )
    payload: Dict[str, Any] = {
        "source": "extension",
        "extension_root": str(extension_root.resolve()),
        "row_id": FFNO_ROW_ID,
        "paper_label": ffno.get("paper_label") or FFNO_PAPER_LABEL,
        "architecture": ffno.get("architecture") or "ffno",
        "row_status": ffno.get("row_status"),
        "image": dict(ffno.get("image") or {}),
        "measurement": dict(ffno.get("measurement") or {}),
        "supporting": dict(ffno.get("supporting") or {}),
        "runtime": dict(ffno.get("runtime") or {}),
    }
    for opt_key in ("blocker_reason", "blocker_message"):
        if ffno.get(opt_key):
            payload[opt_key] = ffno[opt_key]
    return payload


def build_combined_metrics_payload(
    *,
    baseline_root: Path,
    extension_root: Path,
    baseline_metrics: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    ffno_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the read-only-lineage five-row combined metrics payload."""
    baseline_rows = _baseline_row_lineage(
        baseline_root=baseline_root,
        baseline_metrics=baseline_metrics,
        baseline_manifest=baseline_manifest,
    )
    ffno_row = _ffno_combined_row(
        extension_root=extension_root,
        ffno_metrics=ffno_metrics,
    )
    return {
        "schema_version": COMBINED_BUNDLE_SCHEMA_VERSION,
        "claim_boundary": APPEND_ONLY_CLAIM_BOUNDARY,
        "metric_schema_version": metrics_mod.METRIC_SCHEMA_VERSION,
        "baseline": {
            "backlog_item": BASELINE_BACKLOG_ITEM,
            "root": str(baseline_root.resolve()),
            "metrics_json": str((baseline_root / "metrics.json").resolve()),
            "preflight_manifest": str(
                (baseline_root / "preflight_manifest.json").resolve()
            ),
            "claim_boundary": baseline_manifest.get("claim_boundary"),
        },
        "extension": {
            "backlog_item": EXTENSION_BACKLOG_ITEM,
            "root": str(extension_root.resolve()),
            "metrics_json": str((extension_root / "metrics.json").resolve()),
            "preflight_manifest": str(
                (extension_root / "preflight_manifest.json").resolve()
            ),
            "claim_boundary": APPEND_ONLY_CLAIM_BOUNDARY,
        },
        "rows": baseline_rows + [ffno_row],
    }


def write_combined_metrics_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_combined_metrics_csv(payload: Mapping[str, Any], path: Path) -> None:
    """Flat CSV mirroring the combined-metrics row schema.

    Mirrors ``preflight_metrics.write_metrics_csv`` so existing tooling
    can ingest the combined view with the same column expectations,
    plus a leading ``source`` column distinguishing baseline lineage
    rows from the appended FFNO row.
    """
    fieldnames = [
        "source",
        "row_id",
        "paper_label",
        "architecture",
        "row_status",
        *metrics_mod.IMAGE_METRICS,
        *metrics_mod.MEASUREMENT_METRICS,
        *metrics_mod.SUPPORTING_METRICS,
        "parameter_count",
        "wall_time_train_s",
        "wall_time_eval_s",
        "epochs",
        "batch_size",
        "learning_rate",
        "device",
        "device_name",
        "blocker_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("rows") or []:
            runtime = row.get("runtime") or {}
            flat: Dict[str, Any] = {
                "source": row.get("source", ""),
                "row_id": row.get("row_id", ""),
                "paper_label": row.get("paper_label", ""),
                "architecture": row.get("architecture", ""),
                "row_status": row.get("row_status", ""),
                "blocker_reason": row.get("blocker_reason", ""),
            }
            for k in metrics_mod.IMAGE_METRICS:
                flat[k] = (row.get("image") or {}).get(k, "")
            for k in metrics_mod.MEASUREMENT_METRICS:
                flat[k] = (row.get("measurement") or {}).get(k, "")
            for k in metrics_mod.SUPPORTING_METRICS:
                flat[k] = (row.get("supporting") or {}).get(k, "")
            for k in (
                "parameter_count",
                "wall_time_train_s",
                "wall_time_eval_s",
                "epochs",
                "batch_size",
                "learning_rate",
                "device",
                "device_name",
            ):
                flat[k] = runtime.get(k, "")
            writer.writerow(flat)


def build_combined_manifest_payload(
    *,
    baseline_root: Path,
    extension_root: Path,
    baseline_manifest: Mapping[str, Any],
    extension_manifest: Mapping[str, Any],
    combined_metrics_path: Path,
    combined_metrics_csv_path: Path,
) -> Dict[str, Any]:
    """Build the top-level lineage manifest for the combined bundle."""
    return {
        "schema_version": COMBINED_BUNDLE_SCHEMA_VERSION,
        "backlog_item": EXTENSION_BACKLOG_ITEM,
        "claim_boundary": APPEND_ONLY_CLAIM_BOUNDARY,
        "baseline": {
            "backlog_item": BASELINE_BACKLOG_ITEM,
            "root": str(baseline_root.resolve()),
            "preflight_manifest": str(
                (baseline_root / "preflight_manifest.json").resolve()
            ),
            "metrics_json": str((baseline_root / "metrics.json").resolve()),
            "fixed_sample_ids": list(baseline_manifest.get("fixed_sample_ids") or []),
            "rows": [
                {
                    "row_id": row.get("row_id"),
                    "paper_label": row.get("paper_label") or row.get("model"),
                    "architecture": row.get("model"),
                    "row_status": row.get("row_status"),
                }
                for row in (baseline_manifest.get("rows") or [])
            ],
            "contract_fingerprint": baseline_contract_fingerprint(baseline_manifest),
        },
        "extension": {
            "backlog_item": EXTENSION_BACKLOG_ITEM,
            "root": str(extension_root.resolve()),
            "preflight_manifest": str(
                (extension_root / "preflight_manifest.json").resolve()
            ),
            "metrics_json": str((extension_root / "metrics.json").resolve()),
            "rows": [
                {
                    "row_id": row.get("row_id"),
                    "paper_label": row.get("paper_label") or row.get("model"),
                    "architecture": row.get("model"),
                    "row_status": row.get("row_status"),
                }
                for row in (extension_manifest.get("rows") or [])
            ],
        },
        "combined_metrics_json": str(combined_metrics_path.resolve()),
        "combined_metrics_csv": str(combined_metrics_csv_path.resolve()),
    }


def write_combined_manifest(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def emit_combined_bundle(
    *,
    baseline_root: Path,
    extension_root: Path,
) -> Dict[str, Path]:
    """Read both bundles and write combined_metrics.{json,csv} + combined_manifest.json.

    The baseline bundle is treated as read-only authority: the helper
    refuses to start if any expected baseline file is missing. The
    extension bundle is read fresh from disk so the combined view always
    matches what the FFNO row run actually wrote.
    """
    baseline_root = Path(baseline_root)
    extension_root = Path(extension_root)
    baseline_manifest = validate_baseline_bundle(baseline_root)
    baseline_metrics = _read_json(baseline_root / "metrics.json")
    extension_manifest_path = extension_root / "preflight_manifest.json"
    extension_metrics_path = extension_root / "metrics.json"
    if not extension_manifest_path.exists():
        raise BaselineContractMismatchError(
            f"extension bundle missing preflight_manifest.json at {extension_manifest_path}"
        )
    if not extension_metrics_path.exists():
        raise BaselineContractMismatchError(
            f"extension bundle missing metrics.json at {extension_metrics_path}"
        )
    extension_manifest = _read_json(extension_manifest_path)
    extension_metrics = _read_json(extension_metrics_path)

    combined_payload = build_combined_metrics_payload(
        baseline_root=baseline_root,
        extension_root=extension_root,
        baseline_metrics=baseline_metrics,
        baseline_manifest=baseline_manifest,
        ffno_metrics=extension_metrics,
    )
    combined_json = extension_root / COMBINED_METRICS_JSON
    combined_csv = extension_root / COMBINED_METRICS_CSV
    write_combined_metrics_json(combined_payload, combined_json)
    write_combined_metrics_csv(combined_payload, combined_csv)

    manifest_payload = build_combined_manifest_payload(
        baseline_root=baseline_root,
        extension_root=extension_root,
        baseline_manifest=baseline_manifest,
        extension_manifest=extension_manifest,
        combined_metrics_path=combined_json,
        combined_metrics_csv_path=combined_csv,
    )
    combined_manifest = extension_root / COMBINED_MANIFEST_JSON
    write_combined_manifest(manifest_payload, combined_manifest)
    return {
        "combined_metrics_json": combined_json,
        "combined_metrics_csv": combined_csv,
        "combined_manifest_json": combined_manifest,
    }
